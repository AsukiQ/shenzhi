from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from clstr.full_base_train import (
    STAGE0_HANDOFF_QUERY_MODES,
    _encode_stage0_handoff_queries,
    _read_jsonl,
    _skill_id,
    _stage0_unified_static_logits,
    _split_name,
    _stage0_topk_with_explicit_inventory,
    _stage0_handoff_query_text,
)
from clstr.stage0_audit_sampling import select_audit_rows
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint
from clstr.trajectory_inventory import backfill_tool_inventory_from_trajectory_rows


TRAIN_SPLIT_NAMES = {"train", "training", "train_or_released_g3", "released_train"}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _filter_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _split_name(row) in TRAIN_SPLIT_NAMES]


def _empty_counts(k_values: tuple[int, ...]) -> dict[str, Any]:
    return {
        "current_total": 0,
        "next_total": 0,
        "current_hits": {int(k): 0 for k in k_values},
        "next_hits": {int(k): 0 for k in k_values},
    }


def _finalize_counts(counts: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    output = {
        "current_total": int(counts["current_total"]),
        "next_total": int(counts["next_total"]),
    }
    for k in k_values:
        current_total = max(int(counts["current_total"]), 1)
        next_total = max(int(counts["next_total"]), 1)
        output[f"current_recall@{k}"] = float(counts["current_hits"][int(k)] / current_total)
        output[f"next_recall@{k}"] = float(counts["next_hits"][int(k)] / next_total)
    return output


def _update_hits(
    counts: dict[str, Any],
    *,
    kind: str,
    positive: int,
    ranked: list[int],
    k_values: tuple[int, ...],
) -> None:
    total_key = f"{kind}_total"
    hits_key = f"{kind}_hits"
    counts[total_key] += 1
    for k in k_values:
        if int(positive) in ranked[: int(k)]:
            counts[hits_key][int(k)] += 1


def _rank_stage0_candidates(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    skill_count: int,
    skill_id_to_idx: dict[str, int],
    query_mode: str,
    max_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[int]], list[list[int]]]:
    current_ranked: list[list[int]] = []
    next_ranked: list[list[int]] = []
    batch_size = max(1, int(batch_size))
    k = min(max(1, int(max_k)), int(skill_count))
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            current_texts = [_stage0_handoff_query_text(row, target="current", mode=query_mode) for row in batch]
            next_texts = [_stage0_handoff_query_text(row, target="next", mode=query_mode) for row in batch]
            current_h = _encode_stage0_handoff_queries(
                model,
                current_texts,
                query_mode=query_mode,
            ).to(device)
            current_logits = _stage0_unified_static_logits(model, current_h, skill_count)
            next_h = _encode_stage0_handoff_queries(
                model,
                next_texts,
                query_mode=query_mode,
            ).to(device)
            next_logits = _stage0_unified_static_logits(model, next_h, skill_count)
            current_top, _current_scores, _current_inventory_audit = _stage0_topk_with_explicit_inventory(
                current_logits[:, :skill_count],
                batch,
                skill_id_to_idx=skill_id_to_idx,
                skill_count=skill_count,
                k=k,
            )
            next_top, _next_scores, _next_inventory_audit = _stage0_topk_with_explicit_inventory(
                next_logits[:, :skill_count],
                batch,
                skill_id_to_idx=skill_id_to_idx,
                skill_count=skill_count,
                k=k,
            )
            current_ranked.extend(current_top)
            next_ranked.extend(next_top)
    if was_training and hasattr(model, "train"):
        model.train()
    return current_ranked, next_ranked


def audit_stage0_handoff_coverage(
    *,
    model: Any,
    train_path: str | Path,
    skills_path: str | Path,
    output_path: str | Path | None = None,
    top_k_values: tuple[int, ...] | list[int] = (20, 50, 100, 200, 500),
    query_modes: tuple[str, ...] | list[str] = ("raw_state", "skillrouter_state"),
    max_rows: int | None = None,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    available_rows = _filter_train_rows(_read_jsonl(train_path))
    available_rows, tool_inventory_backfill_report = backfill_tool_inventory_from_trajectory_rows(available_rows)
    rows = select_audit_rows(available_rows, max_rows)
    skills = _read_jsonl(skills_path)
    skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    if not rows:
        raise ValueError(f"no train rows available for Stage0 handoff audit: {train_path}")
    if not skill_ids:
        raise ValueError(f"no skills available for Stage0 handoff audit: {skills_path}")
    k_values = tuple(sorted({max(1, int(k)) for k in top_k_values}))
    max_k = min(max(k_values), len(skill_ids))
    modes = tuple(str(mode) for mode in query_modes)
    unsupported = [mode for mode in modes if mode not in STAGE0_HANDOFF_QUERY_MODES]
    if unsupported:
        raise ValueError(f"unsupported Stage0 handoff query modes: {unsupported}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)

    report_modes: dict[str, Any] = {}
    for mode in modes:
        global_counts = _empty_counts(k_values)
        benchmark_counts: dict[str, dict[str, Any]] = defaultdict(lambda: _empty_counts(k_values))
        current_ranked, next_ranked = _rank_stage0_candidates(
            model,
            rows,
            skill_count=len(skill_ids),
            skill_id_to_idx=skill_id_to_idx,
            query_mode=mode,
            max_k=max_k,
            batch_size=batch_size,
            device=device,
        )
        for row, current_top, next_top in zip(rows, current_ranked, next_ranked):
            benchmark = str(row.get("benchmark") or "<missing>")
            loss_mask = row.get("loss_mask") or {}
            skill_id = str(row.get("skill_id") or "")
            next_skill_id = str(row.get("next_skill_id") or "")
            if loss_mask.get("routing") and skill_id in skill_id_to_idx:
                positive = int(skill_id_to_idx[skill_id])
                _update_hits(global_counts, kind="current", positive=positive, ranked=current_top, k_values=k_values)
                _update_hits(
                    benchmark_counts[benchmark],
                    kind="current",
                    positive=positive,
                    ranked=current_top,
                    k_values=k_values,
                )
            if loss_mask.get("L_trans_skill_ce") and next_skill_id in skill_id_to_idx:
                positive = int(skill_id_to_idx[next_skill_id])
                _update_hits(global_counts, kind="next", positive=positive, ranked=next_top, k_values=k_values)
                _update_hits(
                    benchmark_counts[benchmark],
                    kind="next",
                    positive=positive,
                    ranked=next_top,
                    k_values=k_values,
                )
        report_modes[mode] = {
            "global": _finalize_counts(global_counts, k_values),
            "benchmarks": {
                benchmark: _finalize_counts(counts, k_values)
                for benchmark, counts in sorted(benchmark_counts.items())
            },
        }

    report = {
        "status": "ok",
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "available_row_count": len(available_rows),
        "row_count": len(rows),
        "max_rows": None if max_rows is None else int(max_rows),
        "row_selection": "benchmark_label_round_robin" if max_rows is not None else "all_train_rows",
        "tool_inventory_backfill": tool_inventory_backfill_report,
        "skill_count": len(skill_ids),
        "top_k_values": list(k_values),
        "query_modes": report_modes,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def audit_stage0_handoff_coverage_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_path: str | Path | None = None,
    top_k_values: tuple[int, ...] | list[int] = (20, 50, 100, 200, 500),
    query_modes: tuple[str, ...] | list[str] = ("raw_state", "skillrouter_state"),
    max_rows: int | None = None,
    batch_size: int = 16,
    model_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    model, _model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=model_cache_dir,
    )
    report = audit_stage0_handoff_coverage(
        model=model,
        train_path=train_path,
        skills_path=skills_path,
        output_path=output_path,
        top_k_values=top_k_values,
        query_modes=query_modes,
        max_rows=max_rows,
        batch_size=batch_size,
    )
    report["checkpoint_path"] = str(checkpoint_path)
    report["routing_init"] = routing_report
    if output_path is not None:
        _write_json(output_path, report)
    return report
