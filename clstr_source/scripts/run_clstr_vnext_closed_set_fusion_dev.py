#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_data import runtime_visible_mask
from clstr.vnext_eval import (
    _equal_reciprocal_rank_fusion_logits,
    load_vnext_stage2_for_evaluation,
)
from clstr.vnext_stage0_train import (
    _filter_rows,
    _partition_stage0_eligible_rows,
    _require_legal_stage0_positives,
    _stage0_source_family,
)
from clstr.vnext_training import (
    file_sha256,
    load_inventory_catalogs,
    load_or_build_frozen_text_cache,
    read_jsonl,
)


THRESHOLDS = (4, 8, 16, 32, 64, 128, 256)


def _positive_rank(
    ordered_indices: list[int],
    positive_indices: set[int],
) -> int:
    for rank, skill_index in enumerate(ordered_indices, start=1):
        if skill_index in positive_indices:
            return rank
    raise ValueError("closed-set dev row lacks a ranked legal positive")


def _row_ranks(
    static_scores: torch.Tensor,
    semantic_scores: torch.Tensor,
    legal_indices: torch.Tensor,
    positive_indices: set[int],
) -> tuple[int, int]:
    indices = [int(value) for value in legal_indices.cpu().tolist()]
    static_values = [float(value) for value in static_scores.float().cpu().tolist()]
    static_order = sorted(
        range(len(indices)),
        key=lambda position: (-static_values[position], indices[position]),
    )
    fused_values = _equal_reciprocal_rank_fusion_logits(
        static_scores.unsqueeze(0),
        semantic_scores.unsqueeze(0),
        torch.ones((1, len(indices)), dtype=torch.bool, device=static_scores.device),
        torch.ones(1, dtype=torch.bool, device=static_scores.device),
    )[0].cpu().tolist()
    fused_order = sorted(
        range(len(indices)),
        key=lambda position: (-float(fused_values[position]), indices[position]),
    )
    return (
        _positive_rank([indices[position] for position in static_order], positive_indices),
        _positive_rank([indices[position] for position in fused_order], positive_indices),
    )


def _metrics(ranks: list[int]) -> dict[str, float | int]:
    count = len(ranks)
    if count <= 0:
        return {"count": 0, "mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0}
    return {
        "count": count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "recall@1": sum(int(rank <= 1) for rank in ranks) / count,
        "recall@5": sum(int(rank <= 5) for rank in ranks) / count,
    }


def _summarize(records: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    ranks: list[int] = []
    by_family: dict[str, list[int]] = defaultdict(list)
    fused_count = 0
    threshold = None
    if selector.startswith("static_le_"):
        threshold = int(selector.split("_")[2])
    for record in records:
        use_fused = selector == "rrf_all" or (
            threshold is not None and int(record["pool_size"]) > threshold
        )
        rank = int(record["rrf_rank"] if use_fused else record["static_rank"])
        ranks.append(rank)
        by_family[str(record["family"])].append(rank)
        fused_count += int(use_fused)
    return {
        **_metrics(ranks),
        "fused_row_count": fused_count,
        "per_family": {
            family: _metrics(values) for family, values in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--training_skills_path", required=True)
    parser.add_argument("--static_route_dev_rows_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--frozen_cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--belief_top_k", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("closed-set fusion dev diagnostic requires CUDA")
    device = torch.device("cuda")
    model, skills, skill_id_to_idx, checkpoint_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=args.checkpoint_path,
            training_skills_path=args.training_skills_path,
            benchmark_skills=[],
            device=device,
            allow_stage0_static_diagnostic=True,
        )
    )
    catalogs = load_inventory_catalogs(args.inventory_catalogs_path)
    rows = _filter_rows(
        read_jsonl(args.static_route_dev_rows_path),
        set(skill_id_to_idx),
        kind="static_route",
    )
    _require_legal_stage0_positives(rows, catalogs)
    rows, eligibility = _partition_stage0_eligible_rows(
        rows,
        catalogs,
        kind="closed_set_fusion_dev",
    )
    rows = [row for row in rows if int(row.get("inventory_pool_size") or 0) <= 500]
    if not rows:
        raise ValueError("closed-set fusion dev diagnostic has no closed legal pools")
    cache = load_or_build_frozen_text_cache(
        model,
        (str(row["_vnext_query"]) for row in rows),
        role="state",
        batch_size=int(args.batch_size),
        cache_root=args.frozen_cache_dir,
    )

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), int(args.batch_size)):
            batch = rows[start : start + int(args.batch_size)]
            states = cache.batch(
                [str(row["_vnext_query"]) for row in batch],
                device=device,
            )
            legal = runtime_visible_mask(
                batch,
                skill_id_to_idx,
                device=device,
                inventory_catalogs=catalogs,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                belief = model.vnext_initial_belief(
                    states,
                    legal,
                    top_k=int(args.belief_top_k),
                )
                _recall_query, route_query = model.vnext.static_queries(states, belief)
                static_logits = model.vnext_full_pool_logits(route_query, head="route")
                semantic_logits = model.vnext_full_pool_logits(
                    F.normalize(states.float(), p=2, dim=-1).to(states.dtype),
                    head="route",
                )
            for row_index, row in enumerate(batch):
                legal_indices = legal[row_index].nonzero(as_tuple=False).view(-1)
                positives = {
                    skill_id_to_idx[skill_id]
                    for skill_id in row["_vnext_positive_skill_ids"]
                }
                static_rank, rrf_rank = _row_ranks(
                    static_logits[row_index].index_select(0, legal_indices),
                    semantic_logits[row_index].index_select(0, legal_indices),
                    legal_indices,
                    positives,
                )
                records.append(
                    {
                        "family": _stage0_source_family(row),
                        "pool_size": int(legal_indices.numel()),
                        "static_rank": static_rank,
                        "rrf_rank": rrf_rank,
                    }
                )

    alternatives = {"static": _summarize(records, "static")}
    alternatives["rrf_all"] = _summarize(records, "rrf_all")
    for threshold in THRESHOLDS:
        name = f"static_le_{threshold}_else_rrf"
        alternatives[name] = _summarize(records, name)
    report = {
        "schema_version": "clstr_vnext_closed_set_fusion_dev_v1",
        "status": "ok",
        "checkpoint": checkpoint_report,
        "training_skills_path": str(Path(args.training_skills_path).resolve()),
        "static_route_dev_rows_path": str(
            Path(args.static_route_dev_rows_path).resolve()
        ),
        "static_route_dev_rows_sha256": file_sha256(
            args.static_route_dev_rows_path
        ),
        "inventory_catalogs_path": str(Path(args.inventory_catalogs_path).resolve()),
        "inventory_catalogs_sha256": file_sha256(args.inventory_catalogs_path),
        "eligibility": eligibility,
        "closed_row_count": len(records),
        "threshold_candidates": list(THRESHOLDS),
        "fusion": "equal_reciprocal_rank_fusion_k0",
        "alternatives": alternatives,
        "cache": cache.report(),
    }
    target = Path(args.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
