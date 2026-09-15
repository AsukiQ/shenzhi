#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_data import runtime_visible_mask
from clstr.vnext_eval import (
    _preserved_foundation_candidate_logits,
    _stage2_release_selection_contract,
    _unified_router_release_wrapper,
    load_vnext_stage2_for_evaluation,
)
from clstr.vnext_stage0_train import (
    _filter_rows,
    _partition_stage0_eligible_rows,
    _require_legal_stage0_positives,
    _stage0_source_family,
)
from clstr.vnext_stage2_train import (
    _batch_memory_at_start,
    _build_ordinary_dev_anchors,
    _causal_route_state_text,
    _correction_result_text,
    _memory_conditioned_natural_support,
    _positive_mask,
    _prepare_trajectories,
)
from clstr.vnext_training import (
    file_sha256,
    load_inventory_catalogs,
    load_or_build_frozen_text_cache,
    read_jsonl,
    semantic_source_id,
    stable_stratified_cap_rows,
)
from clstr.vnext_unified_router import (
    UNIFIED_EXPERT_NAMES,
    UNIFIED_ROUTER_FEATURE_NAMES,
    UNIFIED_ROUTER_SCHEMA,
    UNIFIED_ROUTING_MODES,
    UNIFIED_ROUTING_MODE_SPARSE,
    UnifiedThreeExpertRouter,
    unified_router_features,
)


def _candidate_logits(
    model: Any,
    states: torch.Tensor,
    legal: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    belief_top_k: int,
) -> torch.Tensor:
    belief = model.vnext_initial_belief(states, legal, top_k=int(belief_top_k))
    _recall, route = model.vnext.static_queries(states, belief)
    return model.vnext_candidate_logits(
        route,
        candidate_ids,
        candidate_valid,
        head="route",
    )


def _padded_legal_candidates(legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    widths = legal.sum(dim=-1)
    width = int(widths.max().item())
    if width <= 0:
        raise ValueError("unified-router rows require nonempty legal pools")
    candidate_ids = torch.zeros(
        (int(legal.size(0)), width), dtype=torch.long, device=legal.device
    )
    valid = torch.zeros_like(candidate_ids, dtype=torch.bool)
    for index in range(int(legal.size(0))):
        ids = legal[index].nonzero(as_tuple=False).view(-1)
        candidate_ids[index, : int(ids.numel())] = ids
        valid[index, : int(ids.numel())] = True
    return candidate_ids, valid


def _positive_on_support(
    positive: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    selected = positive.gather(1, candidate_ids)
    return selected & candidate_valid


STATIC_MEMORY_EVIDENCE_MODE = "static_no_history"
FACTUAL_MEMORY_EVIDENCE_MODE = "factual_replay"
ACTION_ONLY_MEMORY_EVIDENCE_MODE = "action_only_counterfactual"
MEMORY_EVIDENCE_MODES = frozenset(
    {
        STATIC_MEMORY_EVIDENCE_MODE,
        FACTUAL_MEMORY_EVIDENCE_MODE,
        ACTION_ONLY_MEMORY_EVIDENCE_MODE,
    }
)
SPARSE_EXPERT_SUPERVISION_PROTOCOL = (
    "class_balanced_sparse_minimum_regret_top1_v1"
)
MIN_ORACLE_EXPERT_ROWS = 20
MIN_ORACLE_EXPERT_RECALL = 0.20
GLOBAL_MRR_NO_REGRET_TOLERANCE = 0.005
GLOBAL_R5_NO_REGRET_TOLERANCE = 0.01
FAMILY_MRR_NO_REGRET_TOLERANCE = 0.02
FAMILY_R5_NO_REGRET_TOLERANCE = 0.03
MEMORY_EVIDENCE_MRR_NO_REGRET_TOLERANCE = 0.005
MEMORY_EVIDENCE_R5_NO_REGRET_TOLERANCE = 0.01


def _prefix_observation_correction_count(
    rows: list[dict[str, Any]],
    prefix_length: int,
) -> int:
    if int(prefix_length) < 0 or int(prefix_length) > len(rows):
        raise ValueError("calibration prefix length is outside its trajectory")
    return sum(
        int(bool(_correction_result_text(row)))
        for row in rows[: int(prefix_length)]
    )


def _suppress_prefix_observation_corrections(
    rows: list[dict[str, Any]],
    prefix_length: int,
) -> list[dict[str, Any]]:
    """Return an action-identical replay view without result correction."""

    if int(prefix_length) < 0 or int(prefix_length) > len(rows):
        raise ValueError("calibration prefix length is outside its trajectory")
    copied_rows = [dict(row) for row in rows]
    for row in copied_rows[: int(prefix_length)]:
        row["actual_result_text"] = ""
        row["actual_result_executed"] = False
        row["_vnext_result_correction_eligible"] = False
        row["_vnext_result_correction_suppressed_for_calibration"] = True
    return copied_rows


def _paired_dynamic_anchor_views(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for anchor in anchors:
        rows = anchor["rows"]
        target_index = int(anchor["target_index"])
        pair_id = str(anchor.get("identity") or "").strip()
        if not pair_id:
            raise ValueError("dynamic calibration anchor lacks a stable pair identity")
        correction_count = _prefix_observation_correction_count(
            rows,
            target_index,
        )
        views.append(
            {
                **anchor,
                "rows": rows,
                "calibration_pair_id": pair_id,
                "memory_evidence_mode": FACTUAL_MEMORY_EVIDENCE_MODE,
                "observation_correction_count": correction_count,
            }
        )
        views.append(
            {
                **anchor,
                "rows": _suppress_prefix_observation_corrections(
                    rows,
                    target_index,
                ),
                "calibration_pair_id": pair_id,
                "memory_evidence_mode": ACTION_ONLY_MEMORY_EVIDENCE_MODE,
                "observation_correction_count": 0,
            }
        )
    return views


def _record(
    *,
    family: str,
    source: str,
    cluster_id: str,
    foundation_logits: torch.Tensor,
    static_logits: torch.Tensor,
    recurrent_logits: torch.Tensor,
    valid: torch.Tensor,
    static_valid: torch.Tensor,
    positive: torch.Tensor,
    history_depth: float,
    observation_correction_count: int,
    legal_pool_size: int,
    memory_evidence_mode: str,
    calibration_pair_id: str = "",
) -> dict[str, Any] | None:
    if memory_evidence_mode not in MEMORY_EVIDENCE_MODES:
        raise ValueError(
            f"unsupported unified-router memory evidence mode: {memory_evidence_mode}"
        )
    if int(observation_correction_count) < 0 or int(
        observation_correction_count
    ) > float(history_depth):
        raise ValueError("invalid observation correction count for calibration row")
    valid = valid.bool()
    positive = positive.bool() & valid
    if not bool(positive.any().item()) or not bool((valid & ~positive).any().item()):
        return None
    depth = torch.tensor([float(history_depth)], device=foundation_logits.device)
    correction_count = torch.tensor(
        [float(observation_correction_count)],
        device=foundation_logits.device,
    )
    pool = torch.tensor([float(legal_pool_size)], device=foundation_logits.device)
    features = unified_router_features(
        foundation_logits.unsqueeze(0),
        static_logits.unsqueeze(0),
        recurrent_logits.unsqueeze(0),
        valid.unsqueeze(0),
        static_support_mask=static_valid.unsqueeze(0),
        history_depth=depth,
        observation_correction_count=correction_count,
        legal_pool_size=pool,
    )[0]
    return {
        "family": str(family),
        "source": str(source),
        "cluster_id": str(cluster_id),
        "features": features.detach().cpu(),
        "expert_logits": torch.stack(
            (foundation_logits, static_logits, recurrent_logits), dim=0
        ).detach().float().cpu(),
        "valid": valid.detach().cpu(),
        "positive": positive.detach().cpu(),
        "history_depth": float(history_depth),
        "observation_correction_count": int(observation_correction_count),
        "legal_pool_size": int(legal_pool_size),
        "memory_evidence_mode": str(memory_evidence_mode),
        "calibration_pair_id": str(calibration_pair_id),
    }


def _collect_static_records(
    stage2_model: Any,
    foundation_model: Any,
    rows: list[dict[str, Any]],
    *,
    state_cache: Any,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        texts = [str(row["_vnext_query"]) for row in batch]
        stage2_states = state_cache.batch(texts, device=device)
        foundation_states = state_cache.batch_with_projection(
            texts,
            device=device,
            projection_fn=foundation_model.encoder.project_pooled,
        )
        legal = runtime_visible_mask(
            batch,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        candidate_ids, candidate_valid = _padded_legal_candidates(legal)
        positive_full = _positive_mask(batch, skill_id_to_idx, device=device)
        positive = _positive_on_support(positive_full, candidate_ids, candidate_valid)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            static_logits = _candidate_logits(
                stage2_model,
                stage2_states,
                legal,
                candidate_ids,
                candidate_valid,
                belief_top_k=belief_top_k,
            )
            foundation_logits, _foundation_fusion_applied = (
                _preserved_foundation_candidate_logits(
                    foundation_model,
                    foundation_states,
                    legal,
                    candidate_ids,
                    candidate_valid,
                    belief_top_k=belief_top_k,
                )
            )
        for index, row in enumerate(batch):
            width = int(candidate_valid[index].sum().item())
            source = semantic_source_id(row)
            trajectory = str(
                row.get("trajectory_id")
                or row.get("task_id")
                or row.get("inventory_catalog_digest")
                or f"static-{start + index}"
            )
            record = _record(
                family=_stage0_source_family(row),
                source=source,
                cluster_id=f"{source}:{trajectory}",
                foundation_logits=foundation_logits[index, :width],
                static_logits=static_logits[index, :width],
                recurrent_logits=static_logits[index, :width],
                valid=candidate_valid[index, :width],
                static_valid=candidate_valid[index, :width],
                positive=positive[index, :width],
                history_depth=0.0,
                observation_correction_count=0,
                legal_pool_size=width,
                memory_evidence_mode=STATIC_MEMORY_EVIDENCE_MODE,
            )
            if record is not None:
                records.append(record)
    return records


def _collect_dynamic_records(
    stage2_model: Any,
    foundation_model: Any,
    anchors: list[dict[str, Any]],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    anchor_batch_size = max(1, int(batch_size) // 2)
    for start in range(0, len(anchors), anchor_batch_size):
        batch = _paired_dynamic_anchor_views(
            anchors[start : start + anchor_batch_size]
        )
        batch_pair_ids = list(
            dict.fromkeys(str(anchor["calibration_pair_id"]) for anchor in batch)
        )
        batch_pair_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        segments = [
            (
                anchor["rows"],
                int(anchor["target_index"]),
                int(anchor["target_index"]) + 1,
                False,
            )
            for anchor in batch
        ]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            memories = _batch_memory_at_start(
                stage2_model,
                segments,
                state_cache=state_cache,
                action_cache=action_cache,
                result_cache=result_cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
            )
        rows = [anchor["rows"][int(anchor["target_index"])] for anchor in batch]
        route_texts = [_causal_route_state_text(row) for row in rows]
        current_texts = [str(row["state_text_current"]) for row in rows]
        states = state_cache.batch(route_texts, device=device)
        current_states = state_cache.batch(current_texts, device=device)
        foundation_states = state_cache.batch_with_projection(
            route_texts,
            device=device,
            projection_fn=foundation_model.encoder.project_pooled,
        )
        legal = runtime_visible_mask(
            rows,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        positive_full = _positive_mask(rows, skill_id_to_idx, device=device)
        history = torch.ones(len(rows), dtype=torch.bool, device=device)
        history_depth = torch.tensor(
            [float(anchor["target_index"]) for anchor in batch],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            static_memory = stage2_model.vnext_initial_belief(
                states, legal, top_k=belief_top_k
            )
            (_queries, path, _coarse_labels, support_labels) = (
                _memory_conditioned_natural_support(
                    stage2_model,
                    states,
                    memories,
                    static_memory,
                    history,
                    legal,
                    positive_full,
                    current_states=current_states,
                    coarse_k=coarse_k,
                    compressed_m=compressed_m,
                )
            )
            route_scores = stage2_model.vnext_safe_candidate_route_scores(
                states,
                memories,
                static_memory,
                history,
                path.support.candidate_ids,
                path.support.valid_mask,
                static_candidate_ids=path.coarse_candidate_ids,
                static_valid_mask=path.coarse_valid_mask,
                history_depth=history_depth,
                memory_state=current_states,
                hard_fallback=False,
            )
            foundation_logits, _foundation_fusion_applied = (
                _preserved_foundation_candidate_logits(
                    foundation_model,
                    foundation_states,
                    legal,
                    path.support.candidate_ids,
                    path.support.valid_mask,
                    belief_top_k=belief_top_k,
                )
            )
        for index, anchor in enumerate(batch):
            width = int(path.support.valid_mask[index].sum().item())
            record = _record(
                family=str(anchor["family"]),
                source=str(anchor["source"]),
                cluster_id=str(anchor["trajectory_cluster_id"]),
                foundation_logits=foundation_logits[index, :width],
                static_logits=route_scores.static_logits[index, :width],
                recurrent_logits=route_scores.raw_dynamic_logits[index, :width],
                valid=path.support.valid_mask[index, :width],
                static_valid=route_scores.static_support_mask[index, :width],
                positive=support_labels.positive_mask[index, :width],
                history_depth=float(history_depth[index].item()),
                observation_correction_count=int(
                    anchor["observation_correction_count"]
                ),
                legal_pool_size=int(legal[index].sum().item()),
                memory_evidence_mode=str(anchor["memory_evidence_mode"]),
                calibration_pair_id=str(anchor["calibration_pair_id"]),
            )
            if record is not None:
                batch_pair_records[str(anchor["calibration_pair_id"])].append(
                    record
                )
        for pair_id in batch_pair_ids:
            pair_records = batch_pair_records.get(pair_id, [])
            pair_modes = {
                str(record["memory_evidence_mode"]) for record in pair_records
            }
            if len(pair_records) != 2 or pair_modes != {
                FACTUAL_MEMORY_EVIDENCE_MODE,
                ACTION_ONLY_MEMORY_EVIDENCE_MODE,
            }:
                continue
            records.extend(
                sorted(
                    pair_records,
                    key=lambda record: (
                        str(record["memory_evidence_mode"])
                        != FACTUAL_MEMORY_EVIDENCE_MODE
                    ),
                )
            )
    return records


def _split_records(records: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    train: list[int] = []
    dev: list[int] = []
    for index, record in enumerate(records):
        digest = hashlib.sha256(
            f"{seed}|{record['source']}|{record['cluster_id']}".encode("utf-8")
        ).digest()
        (dev if digest[0] < 51 else train).append(index)
    if not train or not dev:
        raise ValueError("unified-router split requires nonempty train and dev rows")
    train_clusters = {
        (str(records[index]["source"]), str(records[index]["cluster_id"]))
        for index in train
    }
    dev_clusters = {
        (str(records[index]["source"]), str(records[index]["cluster_id"]))
        for index in dev
    }
    if train_clusters & dev_clusters:
        raise RuntimeError("unified-router trajectory split leaked across train/dev")
    return train, dev


def _pad_records(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(int(record["valid"].numel()) for record in records)
    count = len(records)
    expert_logits = torch.full((count, 3, width), -1.0e4, dtype=torch.float32)
    valid = torch.zeros((count, width), dtype=torch.bool)
    positive = torch.zeros((count, width), dtype=torch.bool)
    features = torch.stack([record["features"].float() for record in records])
    history_depth = torch.tensor(
        [float(record["history_depth"]) for record in records], dtype=torch.float32
    )
    legal_pool_size = torch.tensor(
        [float(record["legal_pool_size"]) for record in records], dtype=torch.float32
    )
    for index, record in enumerate(records):
        row_width = int(record["valid"].numel())
        expert_logits[index, :, :row_width] = record["expert_logits"]
        valid[index, :row_width] = record["valid"]
        positive[index, :row_width] = record["positive"]
    return {
        "expert_logits": expert_logits,
        "valid": valid,
        "positive": positive,
        "features": features,
        "history_depth": history_depth,
        "legal_pool_size": legal_pool_size,
    }


def _ranks(logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.float().masked_fill(~valid, float("-inf"))
    best_positive = masked.masked_fill(~positive, float("-inf")).max(dim=-1).values
    return (masked > best_positive.unsqueeze(-1)).sum(dim=-1) + 1


def _minimum_regret_expert_targets(
    expert_ranks: torch.Tensor,
    history_depth: torch.Tensor,
) -> torch.Tensor:
    """Choose the best deployable expert, preferring simpler experts on ties."""

    if expert_ranks.ndim != 2 or int(expert_ranks.size(-1)) != len(
        UNIFIED_EXPERT_NAMES
    ):
        raise ValueError("expert ranks must match the unified expert count")
    depth = history_depth.to(device=expert_ranks.device, dtype=torch.float32)
    if depth.ndim != 1 or int(depth.numel()) != int(expert_ranks.size(0)):
        raise ValueError("expert targets require one history depth per row")
    utility = expert_ranks.float().reciprocal() + 0.5 * expert_ranks.le(5).float()
    utility[:, 2] = utility[:, 2].masked_fill(depth.le(0), -1.0e4)
    # Expert order is foundation -> adapted-static -> recurrent-memory, so
    # argmax implements the predeclared minimum-complexity tie break.
    return utility.argmax(dim=-1)


def _metrics(ranks: torch.Tensor) -> dict[str, float]:
    values = ranks.float()
    return {
        "mrr": float(values.reciprocal().mean().item()),
        "recall@1": float(values.le(1).float().mean().item()),
        "recall@5": float(values.le(5).float().mean().item()),
    }


def _evaluate(
    router: UnifiedThreeExpertRouter,
    tensors: dict[str, torch.Tensor],
    indices: list[int],
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    routing_mode: str,
) -> dict[str, Any]:
    if routing_mode not in UNIFIED_ROUTING_MODES:
        raise ValueError(f"unsupported unified routing mode: {routing_mode}")
    all_ranks: list[torch.Tensor] = []
    all_weights: list[torch.Tensor] = []
    expert_ranks: list[list[torch.Tensor]] = [[], [], []]
    for start in range(0, len(indices), int(batch_size)):
        selected = torch.tensor(indices[start : start + int(batch_size)], dtype=torch.long)
        logits = tensors["expert_logits"].index_select(0, selected).to(device)
        valid = tensors["valid"].index_select(0, selected).to(device)
        positive = tensors["positive"].index_select(0, selected).to(device)
        features = tensors["features"].index_select(0, selected).to(device)
        depth = tensors["history_depth"].index_select(0, selected).to(device)
        with torch.no_grad():
            weights = router(features, history_mask=depth.gt(0)).float()
            distributions = torch.softmax(
                logits.masked_fill(~valid.unsqueeze(1), float("-inf")), dim=-1
            ).masked_fill(~valid.unsqueeze(1), 0.0)
            mixture = (weights.unsqueeze(-1) * distributions).sum(dim=1)
            mixed_logits = torch.log(mixture.clamp_min(1.0e-12))
            all_ranks.append(_ranks(mixed_logits, positive, valid).cpu())
            all_weights.append(weights.cpu())
            for expert_index in range(3):
                expert_ranks[expert_index].append(
                    _ranks(logits[:, expert_index], positive, valid).cpu()
                )
    soft_ranks = torch.cat(all_ranks)
    weights = torch.cat(all_weights)
    expert_rank_tensors = [torch.cat(values) for values in expert_ranks]
    expert_rank_matrix = torch.stack(expert_rank_tensors, dim=-1)
    selected_expert = weights.argmax(dim=-1)
    hard_ranks = expert_rank_matrix.gather(
        1, selected_expert.unsqueeze(-1)
    ).squeeze(-1)
    deployed_ranks = (
        hard_ranks
        if routing_mode == UNIFIED_ROUTING_MODE_SPARSE
        else soft_ranks
    )
    evaluated_depth = tensors["history_depth"].index_select(
        0, torch.tensor(indices, dtype=torch.long)
    )
    oracle_expert = _minimum_regret_expert_targets(
        expert_rank_matrix,
        evaluated_depth,
    )
    oracle_ranks = expert_rank_matrix.gather(
        1, oracle_expert.unsqueeze(-1)
    ).squeeze(-1)

    def route_diagnostics(index: torch.Tensor | None = None) -> dict[str, Any]:
        selected = selected_expert if index is None else selected_expert.index_select(0, index)
        oracle = oracle_expert if index is None else oracle_expert.index_select(0, index)
        local_weights = weights if index is None else weights.index_select(0, index)
        local_hard_ranks = hard_ranks if index is None else hard_ranks.index_select(0, index)
        local_oracle_ranks = oracle_ranks if index is None else oracle_ranks.index_select(0, index)
        oracle_expert_recall: dict[str, float | None] = {}
        for expert_index, expert_name in enumerate(UNIFIED_EXPERT_NAMES):
            expert_mask = oracle.eq(expert_index)
            oracle_expert_recall[expert_name] = (
                float(selected[expert_mask].eq(expert_index).float().mean().item())
                if bool(expert_mask.any().item())
                else None
            )
        return {
            "hard_routed": _metrics(local_hard_ranks),
            "oracle": _metrics(local_oracle_ranks),
            "mean_expert_weights": {
                name: float(local_weights[:, expert].mean().item())
                for expert, name in enumerate(UNIFIED_EXPERT_NAMES)
            },
            "winner_counts": dict(
                sorted(
                    Counter(
                        UNIFIED_EXPERT_NAMES[int(value)]
                        for value in selected.tolist()
                    ).items()
                )
            ),
            "oracle_winner_counts": dict(
                sorted(
                    Counter(
                        UNIFIED_EXPERT_NAMES[int(value)]
                        for value in oracle.tolist()
                    ).items()
                )
            ),
            "router_matches_oracle_fraction": float(
                selected.eq(oracle).float().mean().item()
            ),
            "oracle_expert_recall": oracle_expert_recall,
        }

    overall_route = route_diagnostics()
    output: dict[str, Any] = {
        "row_count": len(indices),
        "routing_mode": routing_mode,
        "unified": _metrics(deployed_ranks),
        "soft_mixture": _metrics(soft_ranks),
        "experts": {
            name: _metrics(expert_rank_tensors[index])
            for index, name in enumerate(UNIFIED_EXPERT_NAMES)
        },
        **overall_route,
    }
    def grouped_reports(field: str) -> dict[str, Any]:
        groups: dict[str, list[int]] = defaultdict(list)
        for local_index, record_index in enumerate(indices):
            groups[str(records[record_index][field])].append(local_index)
        reports: dict[str, Any] = {}
        for value, local_indices in sorted(groups.items()):
            index = torch.tensor(local_indices, dtype=torch.long)
            reports[value] = {
                "row_count": len(local_indices),
                "routing_mode": routing_mode,
                "unified": _metrics(deployed_ranks.index_select(0, index)),
                "soft_mixture": _metrics(soft_ranks.index_select(0, index)),
                "experts": {
                    name: _metrics(
                        expert_rank_tensors[expert].index_select(0, index)
                    )
                    for expert, name in enumerate(UNIFIED_EXPERT_NAMES)
                },
                **route_diagnostics(index),
            }
        return reports

    output["per_family"] = grouped_reports("family")
    output["per_memory_evidence_mode"] = grouped_reports(
        "memory_evidence_mode"
    )
    return output


def _calibration_blockers(dev: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    best_expert_r5 = max(
        float(metrics["recall@5"]) for metrics in dev["experts"].values()
    )
    best_expert_mrr = max(
        float(metrics["mrr"]) for metrics in dev["experts"].values()
    )
    if (
        float(dev["unified"]["recall@5"])
        + GLOBAL_R5_NO_REGRET_TOLERANCE
        < best_expert_r5
    ):
        blockers.append("dev_r5_regresses_best_constant_expert")
    if (
        float(dev["unified"]["mrr"])
        + GLOBAL_MRR_NO_REGRET_TOLERANCE
        < best_expert_mrr
    ):
        blockers.append("dev_mrr_regresses_best_constant_expert")
    if len(dev["winner_counts"]) < 2:
        blockers.append("dev_router_collapsed_to_one_expert")
    for expert_name in UNIFIED_EXPERT_NAMES:
        oracle_count = int(dev["oracle_winner_counts"].get(expert_name, 0))
        expert_recall = dev["oracle_expert_recall"].get(expert_name)
        if (
            oracle_count >= MIN_ORACLE_EXPERT_ROWS
            and (
                expert_recall is None
                or float(expert_recall) < MIN_ORACLE_EXPERT_RECALL
            )
        ):
            blockers.append(
                f"dev_oracle_expert_recall_below_floor:{expert_name}"
            )
    for family, family_report in sorted(dev["per_family"].items()):
        if int(family_report["row_count"]) < 20:
            continue
        unified_family = family_report["unified"]
        best_family_mrr = max(
            float(metrics["mrr"])
            for metrics in family_report["experts"].values()
        )
        best_family_r5 = max(
            float(metrics["recall@5"])
            for metrics in family_report["experts"].values()
        )
        if (
            float(unified_family["mrr"])
            + FAMILY_MRR_NO_REGRET_TOLERANCE
            < best_family_mrr
        ):
            blockers.append(f"dev_family_mrr_regression:{family}")
        if (
            float(unified_family["recall@5"])
            + FAMILY_R5_NO_REGRET_TOLERANCE
            < best_family_r5
        ):
            blockers.append(f"dev_family_r5_regression:{family}")
    for evidence_mode in (
        FACTUAL_MEMORY_EVIDENCE_MODE,
        ACTION_ONLY_MEMORY_EVIDENCE_MODE,
    ):
        evidence_report = dev["per_memory_evidence_mode"].get(evidence_mode)
        if evidence_report is None or int(evidence_report["row_count"]) < 20:
            blockers.append(f"dev_memory_evidence_mode_insufficient:{evidence_mode}")
            continue
        unified_evidence = evidence_report["unified"]
        best_evidence_mrr = max(
            float(metrics["mrr"])
            for metrics in evidence_report["experts"].values()
        )
        best_evidence_r5 = max(
            float(metrics["recall@5"])
            for metrics in evidence_report["experts"].values()
        )
        if (
            float(unified_evidence["mrr"])
            + MEMORY_EVIDENCE_MRR_NO_REGRET_TOLERANCE
            < best_evidence_mrr
        ):
            blockers.append(
                f"dev_memory_evidence_mrr_regression:{evidence_mode}"
            )
        if (
            float(unified_evidence["recall@5"])
            + MEMORY_EVIDENCE_R5_NO_REGRET_TOLERANCE
            < best_evidence_r5
        ):
            blockers.append(
                f"dev_memory_evidence_r5_regression:{evidence_mode}"
            )
    return blockers


def _train_router(
    records: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    expert_utility_loss_weight: float,
    batch_size: int,
    device: torch.device,
    routing_mode: str,
) -> tuple[UnifiedThreeExpertRouter, dict[str, Any]]:
    if float(expert_utility_loss_weight) <= 0.0:
        raise ValueError("expert utility loss weight must be positive")
    tensors = _pad_records(records)
    train_indices, dev_indices = _split_records(records, seed)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    train_logits = tensors["expert_logits"].index_select(0, train_index_tensor)
    train_valid = tensors["valid"].index_select(0, train_index_tensor)
    train_positive = tensors["positive"].index_select(0, train_index_tensor)
    train_depth = tensors["history_depth"].index_select(0, train_index_tensor)
    train_ranks = torch.stack(
        [
            _ranks(train_logits[:, expert], train_positive, train_valid).float()
            for expert in range(3)
        ],
        dim=-1,
    )
    train_winners = _minimum_regret_expert_targets(
        train_ranks,
        train_depth,
    ).tolist()
    family_counts = Counter(str(records[index]["family"]) for index in train_indices)
    winner_counts = Counter(int(value) for value in train_winners)
    row_weight = torch.ones(len(records), dtype=torch.float32)
    for local_index, record_index in enumerate(train_indices):
        family = str(records[record_index]["family"])
        winner = int(train_winners[local_index])
        row_weight[record_index] = 1.0 / (
            max(1, family_counts[family]) * max(1, winner_counts[winner])
        ) ** 0.5
    row_weight = row_weight / row_weight.index_select(0, train_index_tensor).mean()
    router = UnifiedThreeExpertRouter().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    rng = random.Random(int(seed))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        shuffled = list(train_indices)
        rng.shuffle(shuffled)
        router.train()
        losses: list[float] = []
        for start in range(0, len(shuffled), int(batch_size)):
            selected = torch.tensor(shuffled[start : start + int(batch_size)], dtype=torch.long)
            logits = tensors["expert_logits"].index_select(0, selected).to(device)
            valid = tensors["valid"].index_select(0, selected).to(device)
            positive = tensors["positive"].index_select(0, selected).to(device)
            features = tensors["features"].index_select(0, selected).to(device)
            depth = tensors["history_depth"].index_select(0, selected).to(device)
            batch_weight = row_weight.index_select(0, selected).to(device)
            weights = router(features, history_mask=depth.gt(0)).float()
            distributions = torch.softmax(
                logits.masked_fill(~valid.unsqueeze(1), float("-inf")), dim=-1
            ).masked_fill(~valid.unsqueeze(1), 0.0)
            positive_mass = (
                distributions * positive.unsqueeze(1).float()
            ).sum(dim=-1).clamp_min(1.0e-9)
            mixture_positive_mass = (weights * positive_mass).sum(dim=-1).clamp_min(1.0e-9)
            route_row_loss = -torch.log(mixture_positive_mass)
            route_loss = (batch_weight * route_row_loss).sum() / batch_weight.sum()
            expert_rank = torch.stack(
                [
                    _ranks(logits[:, expert], positive, valid).float()
                    for expert in range(3)
                ],
                dim=-1,
            )
            target = _minimum_regret_expert_targets(
                expert_rank,
                depth,
            ).detach()
            calibration_row_loss = -torch.log(
                weights.gather(1, target.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1.0e-8)
            )
            calibration_loss = (
                batch_weight * calibration_row_loss
            ).sum() / batch_weight.sum()
            loss = route_loss + float(expert_utility_loss_weight) * calibration_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 1 or epoch % 5 == 0 or epoch == int(epochs):
            router.eval()
            dev = _evaluate(
                router,
                tensors,
                dev_indices,
                records,
                device=device,
                batch_size=batch_size,
                routing_mode=routing_mode,
            )
            metric = dev["unified"]
            score = 0.35 * metric["mrr"] + 0.25 * metric["recall@1"] + 0.40 * metric["recall@5"]
            epoch_blockers = _calibration_blockers(dev)
            history.append(
                {
                    "epoch": epoch,
                    "loss": sum(losses) / max(1, len(losses)),
                    "selection_score": score,
                    "dev": dev,
                    "blockers": epoch_blockers,
                }
            )
            eligible_score = score if not epoch_blockers else score - 10.0
            if eligible_score > best_score:
                best_score = eligible_score
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in router.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("unified-router training did not produce a checkpoint")
    router.load_state_dict(best_state)
    router.eval()
    train_report = _evaluate(
        router,
        tensors,
        train_indices,
        records,
        device=device,
        batch_size=batch_size,
        routing_mode=routing_mode,
    )
    dev_report = _evaluate(
        router,
        tensors,
        dev_indices,
        records,
        device=device,
        batch_size=batch_size,
        routing_mode=routing_mode,
    )
    return router, {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "expert_utility_loss_weight": float(expert_utility_loss_weight),
        "routing_mode": routing_mode,
        "train_row_count": len(train_indices),
        "dev_row_count": len(dev_indices),
        "train_cluster_sha256": hashlib.sha256(
            "\n".join(
                sorted(
                    f"{records[index]['source']}::{records[index]['cluster_id']}"
                    for index in train_indices
                )
            ).encode("utf-8")
        ).hexdigest(),
        "dev_cluster_sha256": hashlib.sha256(
            "\n".join(
                sorted(
                    f"{records[index]['source']}::{records[index]['cluster_id']}"
                    for index in dev_indices
                )
            ).encode("utf-8")
        ).hexdigest(),
        "train": train_report,
        "dev": dev_report,
        "history": history,
        "train_family_counts": dict(sorted(family_counts.items())),
        "train_oracle_winner_counts": {
            UNIFIED_EXPERT_NAMES[index]: int(winner_counts.get(index, 0))
            for index in range(3)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_checkpoint_path", required=True)
    parser.add_argument("--stage2_release_selection_path", required=True)
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--training_skills_path", required=True)
    parser.add_argument("--trajectory_dev_rows_path", required=True)
    parser.add_argument("--static_route_dev_rows_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--frozen_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_dynamic_rows", type=int, default=2048)
    parser.add_argument("--max_static_rows", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--compressed_m", type=int, default=64)
    parser.add_argument("--max_horizon", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=3.0e-3)
    parser.add_argument("--expert_utility_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--routing_mode",
        choices=UNIFIED_ROUTING_MODES,
        default=UNIFIED_ROUTING_MODE_SPARSE,
    )
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("unified-router calibration requires a CUDA compute node")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_status:
        raise ValueError("unified-router calibration requires a clean immutable source")
    stage2_release_contract = _stage2_release_selection_contract(
        args.stage2_release_selection_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
    )
    stage2_checkpoint_sha256 = file_sha256(args.stage2_checkpoint_path)
    foundation_checkpoint_sha256 = file_sha256(args.foundation_checkpoint_path)
    training_skills_sha256 = file_sha256(args.training_skills_path)
    stage2_release_wrapper = _unified_router_release_wrapper(
        stage2_release_contract,
        router_source_commit=source_commit,
        foundation_checkpoint_path=args.foundation_checkpoint_path,
        foundation_checkpoint_sha256=foundation_checkpoint_sha256,
        training_skills_path=args.training_skills_path,
        training_skills_sha256=training_skills_sha256,
    )
    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    stage2_model, skills, skill_id_to_idx, stage2_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=args.stage2_checkpoint_path,
            training_skills_path=args.training_skills_path,
            benchmark_skills=[],
            device=device,
        )
    )
    foundation_model, foundation_skills, foundation_id_to_idx, foundation_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=args.foundation_checkpoint_path,
            training_skills_path=args.training_skills_path,
            benchmark_skills=[],
            device=device,
            allow_stage0_static_diagnostic=True,
        )
    )
    if foundation_id_to_idx != skill_id_to_idx or len(foundation_skills) != len(skills):
        raise ValueError("unified experts do not share an exact skill index")
    stage2_model.eval()
    foundation_model.eval()
    # Both experts use the identical frozen backbone. All requested text is
    # served by the pooled cache, so retain only the foundation projection and
    # heads instead of keeping a duplicate 0.6B backbone on the GPU.
    foundation_model.encoder.backbone = torch.nn.Identity()
    torch.cuda.empty_cache()
    catalogs = load_inventory_catalogs(args.inventory_catalogs_path)
    skill_ids = set(skill_id_to_idx)
    trajectories, trajectory_report = _prepare_trajectories(
        read_jsonl(args.trajectory_dev_rows_path),
        skill_ids,
        catalogs,
        sequence_role="ordinary",
    )
    anchors, anchor_sampling = _build_ordinary_dev_anchors(
        trajectories,
        max_rows=int(args.max_dynamic_rows),
        max_horizon=int(args.max_horizon),
    )
    static_rows = _filter_rows(
        read_jsonl(args.static_route_dev_rows_path), skill_ids, kind="static_route"
    )
    _require_legal_stage0_positives(static_rows, catalogs)
    static_rows, static_eligibility = _partition_stage0_eligible_rows(
        static_rows, catalogs, kind="unified_router_static"
    )
    static_rows = [
        row for row in static_rows if int(row.get("inventory_pool_size") or 0) <= 500
    ]
    static_rows, static_sampling = stable_stratified_cap_rows(
        static_rows,
        int(args.max_static_rows),
        extra_stratum=lambda row: (
            "tiny"
            if int(row.get("inventory_pool_size") or 0) <= 8
            else "moderate"
        ),
    )
    state_texts = [
        text
        for rows in trajectories
        for row in rows
        for text in (str(row["state_text_current"]), _causal_route_state_text(row))
    ]
    state_texts.extend(str(row["_vnext_query"]) for row in static_rows)
    state_cache = load_or_build_frozen_text_cache(
        stage2_model,
        state_texts,
        role="state",
        batch_size=int(args.cache_batch_size),
        cache_root=args.frozen_cache_dir,
    )
    action_cache = load_or_build_frozen_text_cache(
        stage2_model,
        (str(row.get("action_text") or "") for rows in trajectories for row in rows),
        role="action",
        batch_size=int(args.cache_batch_size),
        cache_root=args.frozen_cache_dir,
    )
    result_texts = [
        _correction_result_text(row)
        for rows in trajectories
        for row in rows
        if _correction_result_text(row)
    ]
    result_cache = (
        load_or_build_frozen_text_cache(
            stage2_model,
            result_texts,
            role="result",
            batch_size=int(args.cache_batch_size),
            cache_root=args.frozen_cache_dir,
        )
        if result_texts
        else None
    )
    static_records = _collect_static_records(
        stage2_model,
        foundation_model,
        static_rows,
        state_cache=state_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=int(args.belief_top_k),
        batch_size=int(args.batch_size),
    )
    dynamic_records = _collect_dynamic_records(
        stage2_model,
        foundation_model,
        anchors,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=int(args.belief_top_k),
        coarse_k=int(args.coarse_k),
        compressed_m=int(args.compressed_m),
        batch_size=int(args.batch_size),
    )
    dynamic_mode_counts = Counter(
        str(record["memory_evidence_mode"]) for record in dynamic_records
    )
    retained_pair_ids = {
        str(record["calibration_pair_id"]) for record in dynamic_records
    }
    if dynamic_mode_counts != Counter(
        {
            FACTUAL_MEMORY_EVIDENCE_MODE: len(retained_pair_ids),
            ACTION_ONLY_MEMORY_EVIDENCE_MODE: len(retained_pair_ids),
        }
    ):
        raise RuntimeError(
            "unified-router dynamic calibration views are not exactly paired"
        )
    if any(
        int(record["observation_correction_count"]) != 0
        for record in dynamic_records
        if record["memory_evidence_mode"] == ACTION_ONLY_MEMORY_EVIDENCE_MODE
    ):
        raise RuntimeError("action-only calibration view retained a result correction")
    records = [*static_records, *dynamic_records]
    if not static_records or not dynamic_records:
        raise ValueError("unified-router calibration requires static and dynamic records")
    router, training = _train_router(
        records,
        seed=int(args.seed),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        expert_utility_loss_weight=float(args.expert_utility_loss_weight),
        batch_size=int(args.batch_size),
        device=device,
        routing_mode=str(args.routing_mode),
    )
    dev = training["dev"]
    blockers = _calibration_blockers(dev)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "unified_router.pt"
    torch.save(
        {
            "schema_version": UNIFIED_ROUTER_SCHEMA,
            "status": "ok" if not blockers else "action_required",
            "blockers": blockers,
            "state_dict": {
                name: value.detach().cpu() for name, value in router.state_dict().items()
            },
            "feature_names": list(UNIFIED_ROUTER_FEATURE_NAMES),
            "expert_names": list(UNIFIED_EXPERT_NAMES),
            "routing_mode": str(args.routing_mode),
            "stage2_checkpoint_path": str(Path(args.stage2_checkpoint_path).resolve()),
            "stage2_checkpoint_sha256": stage2_checkpoint_sha256,
            "foundation_checkpoint_path": str(Path(args.foundation_checkpoint_path).resolve()),
            "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
            "training_skills_sha256": training_skills_sha256,
            "trajectory_dev_rows_sha256": file_sha256(args.trajectory_dev_rows_path),
            "static_route_dev_rows_sha256": file_sha256(args.static_route_dev_rows_path),
            "inventory_catalogs_sha256": file_sha256(args.inventory_catalogs_path),
            "selection": {
                "uses_benchmark_eval_rows": False,
                "uses_benchmark_identity_feature": False,
                "best_epoch": training["best_epoch"],
            },
            "training_contract": {
                "positive_mass_loss_weight": 1.0,
                "expert_utility_loss_weight": float(
                    args.expert_utility_loss_weight
                ),
                "expert_supervision_protocol": SPARSE_EXPERT_SUPERVISION_PROTOCOL,
                "expert_tie_break_order": list(UNIFIED_EXPERT_NAMES),
                "expert_utility": "reciprocal_rank_plus_0.5_top5",
                "oracle_expert_recall_min_rows": MIN_ORACLE_EXPERT_ROWS,
                "oracle_expert_recall_floor": MIN_ORACLE_EXPERT_RECALL,
                "calibration_no_regret_tolerances": {
                    "global_mrr": GLOBAL_MRR_NO_REGRET_TOLERANCE,
                    "global_recall_at_5": GLOBAL_R5_NO_REGRET_TOLERANCE,
                    "family_mrr": FAMILY_MRR_NO_REGRET_TOLERANCE,
                    "family_recall_at_5": FAMILY_R5_NO_REGRET_TOLERANCE,
                    "memory_evidence_mrr": (
                        MEMORY_EVIDENCE_MRR_NO_REGRET_TOLERANCE
                    ),
                    "memory_evidence_recall_at_5": (
                        MEMORY_EVIDENCE_R5_NO_REGRET_TOLERANCE
                    ),
                },
                "routing_mode": str(args.routing_mode),
                "default_expert_prior": "foundation",
                "observation_grounding_feature": (
                    "observation_correction_fraction"
                ),
                "paired_calibration_protocol": (
                    "same_prefix_factual_and_action_only_result_suppressed_v1"
                ),
                "paired_views_share_trajectory_split": True,
                "foundation_expert_protocol": (
                    "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
                ),
            },
            "source_commit": source_commit,
            "stage2_release_wrapper": stage2_release_wrapper,
        },
        artifact_path,
    )
    report = {
        "schema_version": "clstr_vnext_unified_router_calibration_v6",
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "source_commit": source_commit,
        "stage2_release_wrapper": stage2_release_wrapper,
        "stage2_checkpoint": stage2_report,
        "foundation_checkpoint": foundation_report,
        "data": {
            "static_record_count": len(static_records),
            "dynamic_record_count": len(dynamic_records),
            "sampled_dynamic_anchor_count": len(anchors),
            "retained_dynamic_pair_count": len(retained_pair_ids),
            "dropped_dynamic_pair_count": len(anchors) - len(retained_pair_ids),
            "dynamic_memory_evidence_mode_counts": dict(
                sorted(dynamic_mode_counts.items())
            ),
            "factual_observation_supported_record_count": sum(
                int(int(record["observation_correction_count"]) > 0)
                for record in dynamic_records
                if record["memory_evidence_mode"]
                == FACTUAL_MEMORY_EVIDENCE_MODE
            ),
            "action_only_observation_supported_record_count": sum(
                int(int(record["observation_correction_count"]) > 0)
                for record in dynamic_records
                if record["memory_evidence_mode"]
                == ACTION_ONLY_MEMORY_EVIDENCE_MODE
            ),
            "static_eligibility": static_eligibility,
            "static_sampling": static_sampling,
            "trajectory": trajectory_report,
            "ordinary_anchor": anchor_sampling,
        },
        "training": training,
        "cache": {
            "state": state_cache.report(),
            "action": action_cache.report(),
            "result": None if result_cache is None else result_cache.report(),
        },
        "method_contract": {
            "expert_names": list(UNIFIED_EXPERT_NAMES),
            "routing_mode": str(args.routing_mode),
            "feature_names": list(UNIFIED_ROUTER_FEATURE_NAMES),
            "benchmark_identity_feature": False,
            "source_identity_feature": False,
            "positive_injection_count": 0,
            "candidate_membership_changed": False,
            "frozen_qwen_backbone_reused": True,
            "foundation_and_adapted_share_pooled_cache": True,
            "positive_mass_loss_weight": 1.0,
            "expert_utility_loss_weight": float(args.expert_utility_loss_weight),
            "expert_supervision_protocol": SPARSE_EXPERT_SUPERVISION_PROTOCOL,
            "expert_tie_break_order": list(UNIFIED_EXPERT_NAMES),
            "expert_utility": "reciprocal_rank_plus_0.5_top5",
            "oracle_expert_recall_min_rows": MIN_ORACLE_EXPERT_ROWS,
            "oracle_expert_recall_floor": MIN_ORACLE_EXPERT_RECALL,
            "calibration_no_regret_tolerances": {
                "global_mrr": GLOBAL_MRR_NO_REGRET_TOLERANCE,
                "global_recall_at_5": GLOBAL_R5_NO_REGRET_TOLERANCE,
                "family_mrr": FAMILY_MRR_NO_REGRET_TOLERANCE,
                "family_recall_at_5": FAMILY_R5_NO_REGRET_TOLERANCE,
                "memory_evidence_mrr": (
                    MEMORY_EVIDENCE_MRR_NO_REGRET_TOLERANCE
                ),
                "memory_evidence_recall_at_5": (
                    MEMORY_EVIDENCE_R5_NO_REGRET_TOLERANCE
                ),
            },
            "default_expert_prior": "foundation",
            "observation_grounding_feature": (
                "observation_correction_fraction"
            ),
            "paired_calibration_protocol": (
                "same_prefix_factual_and_action_only_result_suppressed_v1"
            ),
            "paired_views_share_trajectory_split": True,
            "foundation_expert_protocol": (
                "preserved_stage0_static_semantic_rrf_above_heldout_pool8_v1"
            ),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
