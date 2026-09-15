from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from clstr.counterfactual_memory_calibration import (
    CMC_FEATURE_CANDIDATE_COUNT_CAP,
    CMC_FEATURE_UPDATE_COUNT_CAP,
)
from clstr.memory_candidate_recall import (
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_UNION_VERSION,
    CandidateUnion,
    build_static_dynamic_union,
    candidate_recall_report,
)
from clstr.memory_utility_gate import (
    HEURISTIC_ALPHA_VERSION,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    effective_memory_alpha,
    fuse_route_scores,
    memory_utility_features,
    positive_rank_and_utility,
)
from clstr.memory_utility_records import canonical_digest
from clstr.memory_utility_records import (
    ROUTE_RECORD_SCHEMA_VERSION,
    persist_memory_utility_route_records,
)
from clstr.qwen_clstr_lineage import sha256_path


STAGE4_VALIDATION_REPORT_SCHEMA = "stage4_validation_report_v1"
STAGE4_DYNAMIC_SELECTION_SCHEMA = "stage4_dynamic_selection_v1"
FIXED_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_STAGE4_BENCHMARKS = {
    "toolbench_g3",
    "traject_bench",
    "alfworld",
    "webshop",
}


def _candidate_union_tensors(
    *,
    static_full_logits: torch.Tensor,
    dynamic_full_logits: torch.Tensor,
    positive_full_mask: torch.Tensor,
    valid_full_mask: torch.Tensor,
    static_k: int,
    dynamic_extra_k: int,
) -> tuple[CandidateUnion, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    union = build_static_dynamic_union(
        static_full_logits,
        dynamic_full_logits,
        valid_full_mask,
        static_k=static_k,
        dynamic_extra_k=dynamic_extra_k,
    )
    width = max((len(row) for row in union.candidate_rows), default=0)
    width = max(1, width)
    floor = torch.finfo(static_full_logits.dtype).min
    static = torch.full(
        (len(union.candidate_rows), width),
        floor,
        dtype=static_full_logits.dtype,
        device=static_full_logits.device,
    )
    dynamic = torch.full_like(static, floor)
    valid = torch.zeros_like(static, dtype=torch.bool)
    positive = torch.zeros_like(static, dtype=torch.bool)
    for row_index, candidate_indices in enumerate(union.candidate_rows):
        if not candidate_indices:
            continue
        indices = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=static_full_logits.device,
        )
        row_width = len(candidate_indices)
        static[row_index, :row_width] = static_full_logits[row_index].index_select(
            0,
            indices,
        )
        dynamic[row_index, :row_width] = dynamic_full_logits[row_index].index_select(
            0,
            indices,
        )
        valid[row_index, :row_width] = True
        positive[row_index, :row_width] = positive_full_mask[
            row_index
        ].index_select(0, indices)
    return union, static, dynamic, positive, valid


def _gather_union_logits(
    full_logits: torch.Tensor,
    candidate_rows: list[list[int]],
    *,
    width: int,
) -> torch.Tensor:
    floor = torch.finfo(full_logits.dtype).min
    gathered = torch.full(
        (len(candidate_rows), int(width)),
        floor,
        dtype=full_logits.dtype,
        device=full_logits.device,
    )
    for row_index, candidate_indices in enumerate(candidate_rows):
        if not candidate_indices:
            continue
        indices = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=full_logits.device,
        )
        gathered[row_index, : len(candidate_indices)] = full_logits[
            row_index
        ].index_select(0, indices)
    return gathered


def _route_summary(
    *,
    rows: list[dict[str, Any]],
    ranks: torch.Tensor,
) -> dict[str, Any]:
    reciprocal_by_benchmark: dict[str, list[float]] = {}
    recall5_by_benchmark: dict[str, list[float]] = {}
    for row, raw_rank in zip(rows, ranks.detach().cpu().tolist()):
        benchmark = str(
            row.get("source_benchmark") or row.get("benchmark") or "unknown"
        )
        rank = int(raw_rank)
        reciprocal_by_benchmark.setdefault(benchmark, []).append(
            1.0 / rank if rank > 0 else 0.0
        )
        recall5_by_benchmark.setdefault(benchmark, []).append(
            1.0 if 0 < rank <= 5 else 0.0
        )
    by_benchmark = {
        benchmark: {
            "mrr": sum(reciprocals) / len(reciprocals),
            "recall@5": sum(recall5_by_benchmark[benchmark])
            / len(recall5_by_benchmark[benchmark]),
            "row_count": len(reciprocals),
        }
        for benchmark, reciprocals in sorted(reciprocal_by_benchmark.items())
    }
    macro_mrr = (
        sum(item["mrr"] for item in by_benchmark.values()) / len(by_benchmark)
        if by_benchmark
        else 0.0
    )
    macro_recall5 = (
        sum(item["recall@5"] for item in by_benchmark.values())
        / len(by_benchmark)
        if by_benchmark
        else 0.0
    )
    return {
        "balanced_macro_mrr": float(macro_mrr),
        "balanced_macro_recall@5": float(macro_recall5),
        "by_benchmark": by_benchmark,
    }


def _build_route_record(
    *,
    row: dict[str, Any],
    source_row_index: int,
    candidate_indices: list[int],
    ordered_skill_ids: list[str],
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    features: torch.Tensor,
    causal_update_count: torch.Tensor,
    raw_alpha: torch.Tensor,
    effective_alpha: torch.Tensor,
) -> dict[str, Any]:
    width = len(candidate_indices)
    if width == 0:
        candidate_indices = [-1]
        candidate_skill_ids = ["__no_legal_candidate__"]
        static_values = [0.0]
        dynamic_values = [0.0]
        valid_values = [False]
        positive_values = [False]
    else:
        candidate_skill_ids = [ordered_skill_ids[index] for index in candidate_indices]
        static_values = [
            float(value)
            for value in static_logits[:width].detach().cpu().tolist()
        ]
        dynamic_values = [
            float(value)
            for value in dynamic_logits[:width].detach().cpu().tolist()
        ]
        valid_values = [
            bool(value) for value in valid_mask[:width].detach().cpu().tolist()
        ]
        positive_values = [
            bool(value)
            for value in positive_mask[:width].detach().cpu().tolist()
        ]
    static_rank, static_utility, _eligible = positive_rank_and_utility(
        torch.tensor(static_values, dtype=torch.float32).view(1, -1),
        torch.tensor(positive_values, dtype=torch.bool).view(1, -1),
        torch.tensor(valid_values, dtype=torch.bool).view(1, -1),
    )
    dynamic_rank, dynamic_utility, _eligible = positive_rank_and_utility(
        torch.tensor(dynamic_values, dtype=torch.float32).view(1, -1),
        torch.tensor(positive_values, dtype=torch.bool).view(1, -1),
        torch.tensor(valid_values, dtype=torch.bool).view(1, -1),
    )
    benchmark = str(
        row.get("source_benchmark") or row.get("benchmark") or "unknown"
    )
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not trajectory_id:
        raise ValueError("Stage4 validation route records require trajectory_id")
    row_digest = canonical_digest(
        {
            key: value
            for key, value in row.items()
            if not str(key).startswith("_")
        }
    )
    return {
        "schema_version": ROUTE_RECORD_SCHEMA_VERSION,
        "row_digest": row_digest,
        "source_row_index": int(source_row_index),
        "trajectory_id": trajectory_id,
        "task_id": str(row.get("task_id") or row.get("row_id") or trajectory_id),
        "benchmark": benchmark,
        "sequential_benchmark": True,
        "causal_update_count": float(causal_update_count.detach().cpu().item()),
        "features": [float(value) for value in features.detach().cpu().tolist()],
        "candidate_indices": [int(index) for index in candidate_indices],
        "candidate_skill_ids": candidate_skill_ids,
        "static_logits": static_values,
        "dynamic_logits": dynamic_values,
        "valid_mask": valid_values,
        "positive_mask": positive_values,
        "static_rank": int(static_rank.item()),
        "dynamic_rank": int(dynamic_rank.item()),
        "static_utility": float(static_utility.item()),
        "dynamic_utility": float(dynamic_utility.item()),
        "raw_alpha": float(raw_alpha.detach().cpu().item()),
        "effective_alpha": float(effective_alpha.detach().cpu().item()),
    }


def _evaluate_stage4_partition(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    batch_size: int,
    static_k: int,
    dynamic_extra_k: int,
    final_k: int,
    counterfactual_utility_weight: float,
    counterfactual_gain_margin: float,
    counterfactual_safety_tolerance: float,
    counterfactual_gain_weight: float,
    counterfactual_safety_weight: float,
    safe_memory_residual_bound: float,
    stage4_method: str = "stage4_safe_memory_v1",
    validation_step: int | None = None,
) -> dict[str, Any]:
    if final_k > static_k:
        raise ValueError("Stage4 final_k must not exceed static_k")
    if not rows:
        raise ValueError("Stage4 validation partition requires rows")
    from clstr.stage4_act_train import (
        CANDIDATE_ADMISSION_RESIDUAL_V1,
        COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
        _build_stage4_candidate_admission_route_batch,
        _build_stage4_cmc_route_batch,
        _build_stage4_safe_route_batch,
    )

    cmc_enabled = str(stage4_method) == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
    candidate_admission_enabled = (
        str(stage4_method) == CANDIDATE_ADMISSION_RESIDUAL_V1
    )

    ordered_skill_ids = [
        skill_id
        for skill_id, _index in sorted(
            skill_id_to_idx.items(),
            key=lambda item: int(item[1]),
        )
    ]
    all_rows: list[dict[str, Any]] = []
    static_candidate_batches: list[torch.Tensor] = []
    dynamic_candidate_batches: list[torch.Tensor] = []
    safe_fused_candidate_batches: list[torch.Tensor] = []
    positive_batches: list[torch.Tensor] = []
    valid_batches: list[torch.Tensor] = []
    feature_batches: list[torch.Tensor] = []
    count_batches: list[torch.Tensor] = []
    alpha_batches: list[torch.Tensor] = []
    admission_logit_batches: list[torch.Tensor] = []
    dynamic_extra_batches: list[torch.Tensor] = []
    static_full_batches: list[torch.Tensor] = []
    positive_full_batches: list[torch.Tensor] = []
    valid_full_batches: list[torch.Tensor] = []
    unions: list[CandidateUnion] = []
    records: list[dict[str, Any]] = []
    alpha_zero_exact = True
    alpha_one_exact = True
    for start in range(0, len(rows), max(1, int(batch_size))):
        batch_rows = rows[start : start + max(1, int(batch_size))]
        with torch.no_grad():
            if candidate_admission_enabled:
                output = _build_stage4_candidate_admission_route_batch(
                    model,
                    batch_rows,
                    device,
                    skill_id_to_idx=skill_id_to_idx,
                    equivalent_skill_ids_by_skill_id=(
                        equivalent_skill_ids_by_skill_id
                    ),
                    static_k=static_k,
                    dynamic_extra_k=dynamic_extra_k,
                    gain_margin=counterfactual_gain_margin,
                )
                static_full_logits = output.static_full_logits
                dynamic_full_logits = output.dynamic_full_logits
                fused_full_logits = None
                selected_alpha = (
                    output.scoring.admission_probability
                    .masked_fill(~output.candidate_valid_mask, 0.0)
                    .sum(dim=-1)
                    / output.candidate_valid_mask.sum(dim=-1).clamp_min(1)
                )
                record_raw_alpha = selected_alpha
            elif cmc_enabled:
                output = _build_stage4_cmc_route_batch(
                    model,
                    batch_rows,
                    device,
                    skill_id_to_idx=skill_id_to_idx,
                    equivalent_skill_ids_by_skill_id=(
                        equivalent_skill_ids_by_skill_id
                    ),
                )
                static_full_logits = output.static_full_logits
                dynamic_full_logits = output.dynamic_full_logits
                fused_full_logits = output.fused_full_logits
                selected_alpha = output.effective_alpha
                alpha_zero_exact = alpha_zero_exact and torch.equal(
                    output.objective.alpha_zero_logits,
                    output.static_full_logits.masked_fill(
                        ~output.valid_mask,
                        torch.finfo(output.static_full_logits.dtype).min,
                    ),
                )
                alpha_one_exact = alpha_one_exact and torch.equal(
                    output.objective.alpha_one_logits,
                    output.dynamic_full_logits.masked_fill(
                        ~output.valid_mask,
                        torch.finfo(output.dynamic_full_logits.dtype).min,
                    ),
                )
                if int(validation_step or 0) == 0:
                    fused_full_logits = static_full_logits
                    selected_alpha = torch.zeros_like(selected_alpha)
                record_raw_alpha = output.raw_alpha
            else:
                output = _build_stage4_safe_route_batch(
                    model,
                    batch_rows,
                    device,
                    skill_id_to_idx=skill_id_to_idx,
                    equivalent_skill_ids_by_skill_id=(
                        equivalent_skill_ids_by_skill_id
                    ),
                    trainable_replay_prefix=False,
                    counterfactual_utility_weight=counterfactual_utility_weight,
                    counterfactual_gain_margin=counterfactual_gain_margin,
                    counterfactual_safety_tolerance=counterfactual_safety_tolerance,
                    counterfactual_gain_weight=counterfactual_gain_weight,
                    counterfactual_safety_weight=counterfactual_safety_weight,
                    counterfactual_scale=1.0,
                    safe_memory_residual_bound=safe_memory_residual_bound,
                )
                static_full_logits = output.static_full_logits
                dynamic_full_logits = output.dynamic_full_logits
                fused_full_logits = output.safe_fused_full_logits
                selected_alpha = output.safe_alpha
                record_raw_alpha = selected_alpha
        if candidate_admission_enabled:
            union = output.candidate_union
            static = output.scoring.static_logits.masked_fill(
                ~output.candidate_valid_mask,
                torch.finfo(output.scoring.static_logits.dtype).min,
            )
            dynamic = output.scoring.dynamic_logits.masked_fill(
                ~output.candidate_valid_mask,
                torch.finfo(output.scoring.dynamic_logits.dtype).min,
            )
            positive = output.candidate_positive_mask
            valid = output.candidate_valid_mask
            safe_fused = output.scoring.final_logits
            positive_full_mask = output.positive_full_mask
            valid_full_mask = output.valid_full_mask
            admission_logit_batches.append(
                output.scoring.admission_logits.detach().cpu()
            )
            dynamic_extra_batches.append(output.dynamic_extra_mask.detach().cpu())
        else:
            union, static, dynamic, positive, valid = _candidate_union_tensors(
                static_full_logits=static_full_logits,
                dynamic_full_logits=dynamic_full_logits,
                positive_full_mask=output.positive_mask,
                valid_full_mask=output.valid_mask,
                static_k=static_k,
                dynamic_extra_k=dynamic_extra_k,
            )
            safe_fused = _gather_union_logits(
                fused_full_logits,
                union.candidate_rows,
                width=int(static.size(1)),
            )
            positive_full_mask = output.positive_mask
            valid_full_mask = output.valid_mask
        update_cap = max(
            1.0,
            float(output.causal_update_count.max().detach().cpu().item()),
        )
        candidate_cap = max(1.0, float(static_k + dynamic_extra_k))
        features = memory_utility_features(
            static,
            dynamic,
            valid,
            output.static_memory,
            output.dynamic_memory,
            output.causal_update_count,
            update_count_cap=update_cap,
            candidate_count_cap=candidate_cap,
        )
        for row_index, (row, candidate_indices) in enumerate(
            zip(batch_rows, union.candidate_rows)
        ):
            records.append(
                _build_route_record(
                    row=row,
                    source_row_index=start + row_index,
                    candidate_indices=candidate_indices,
                    ordered_skill_ids=ordered_skill_ids,
                    static_logits=static[row_index],
                    dynamic_logits=dynamic[row_index],
                    valid_mask=valid[row_index],
                    positive_mask=positive[row_index],
                    features=features[row_index],
                    causal_update_count=output.causal_update_count[row_index],
                    raw_alpha=record_raw_alpha[row_index],
                    effective_alpha=selected_alpha[row_index],
                )
            )
        all_rows.extend(batch_rows)
        static_candidate_batches.append(static.detach().cpu())
        dynamic_candidate_batches.append(dynamic.detach().cpu())
        safe_fused_candidate_batches.append(safe_fused.detach().cpu())
        positive_batches.append(positive.detach().cpu())
        valid_batches.append(valid.detach().cpu())
        feature_batches.append(features.detach().cpu())
        count_batches.append(output.causal_update_count.detach().cpu())
        alpha_batches.append(selected_alpha.detach().cpu())
        static_full_batches.append(static_full_logits.detach().cpu())
        positive_full_batches.append(positive_full_mask.detach().cpu())
        valid_full_batches.append(valid_full_mask.detach().cpu())
        unions.append(union)

    maximum_width = max(tensor.size(1) for tensor in static_candidate_batches)

    def padded(tensor: torch.Tensor, *, value: float | bool) -> torch.Tensor:
        if int(tensor.size(1)) == maximum_width:
            return tensor
        shape = (int(tensor.size(0)), maximum_width - int(tensor.size(1)))
        suffix = torch.full(shape, value, dtype=tensor.dtype)
        return torch.cat((tensor, suffix), dim=1)

    floor = torch.finfo(static_candidate_batches[0].dtype).min
    static = torch.cat(
        [padded(tensor, value=floor) for tensor in static_candidate_batches],
        dim=0,
    ).to(device)
    dynamic = torch.cat(
        [padded(tensor, value=floor) for tensor in dynamic_candidate_batches],
        dim=0,
    ).to(device)
    safe_fused = torch.cat(
        [padded(tensor, value=floor) for tensor in safe_fused_candidate_batches],
        dim=0,
    ).to(device)
    positive = torch.cat(
        [padded(tensor, value=False) for tensor in positive_batches],
        dim=0,
    ).to(device)
    valid = torch.cat(
        [padded(tensor, value=False) for tensor in valid_batches],
        dim=0,
    ).to(device)
    causal_update_count = torch.cat(count_batches, dim=0).to(device)
    static_ranks, static_utility, static_eligible = positive_rank_and_utility(
        static,
        positive,
        valid,
    )
    dynamic_ranks, dynamic_utility, dynamic_eligible = positive_rank_and_utility(
        dynamic,
        positive,
        valid,
    )
    if not torch.equal(static_eligible, dynamic_eligible):
        raise ValueError("Stage4 branch eligibility drifted")
    static_summary = _route_summary(rows=all_rows, ranks=static_ranks)
    dynamic_summary = _route_summary(rows=all_rows, ranks=dynamic_ranks)
    safe_fused_ranks, fused_utility, safe_eligible = positive_rank_and_utility(
        safe_fused,
        positive,
        valid,
    )
    if not torch.equal(static_eligible, safe_eligible):
        raise ValueError("Stage4 safe-fused branch eligibility drifted")
    safe_fused_summary = _route_summary(rows=all_rows, ranks=safe_fused_ranks)
    candidate_admission_metrics: dict[str, Any] | None = None
    if candidate_admission_enabled:
        admission_logits = torch.cat(
            [padded(tensor, value=0.0) for tensor in admission_logit_batches],
            dim=0,
        ).to(device)
        dynamic_extra = torch.cat(
            [padded(tensor, value=False) for tensor in dynamic_extra_batches],
            dim=0,
        ).to(device)
        extra_eligible = dynamic_extra & valid
        extra_positive = extra_eligible & positive
        positive_count = int(extra_positive.sum().item())
        extra_count = int(extra_eligible.sum().item())
        prevalence = (
            float(positive_count) / float(extra_count)
            if extra_count > 0
            else 0.0
        )
        if positive_count > 0:
            labels = extra_positive[extra_eligible]
            scores = admission_logits[extra_eligible]
            order = torch.argsort(scores, descending=True, stable=True)
            ordered = labels.index_select(0, order).to(torch.float32)
            precision = ordered.cumsum(dim=0) / torch.arange(
                1,
                int(ordered.numel()) + 1,
                dtype=torch.float32,
                device=device,
            )
            admission_auprc = float(
                (precision * ordered).sum().div(float(positive_count)).item()
            )
        else:
            admission_auprc = 0.0
        negative = valid & ~positive
        margin_eligible = positive.any(dim=-1) & negative.any(dim=-1)
        floor = torch.finfo(static.dtype).min
        static_margin = (
            static.masked_fill(~positive, floor).max(dim=-1).values
            - static.masked_fill(~negative, floor).max(dim=-1).values
        )
        final_margin = (
            safe_fused.masked_fill(~positive, floor).max(dim=-1).values
            - safe_fused.masked_fill(~negative, floor).max(dim=-1).values
        )
        violation = margin_eligible & (
            final_margin + float(counterfactual_safety_tolerance)
            < static_margin
        )
        eligible_count = int(margin_eligible.sum().item())
        violation_rate = (
            float(violation.sum().item()) / float(eligible_count)
            if eligible_count > 0
            else 0.0
        )
        candidate_admission_metrics = {
            **safe_fused_summary,
            "static_balanced_macro_mrr": float(
                static_summary["balanced_macro_mrr"]
            ),
            "admission_auprc": admission_auprc,
            "admission_prevalence": prevalence,
            "dynamic_extra_positive_count": positive_count,
            "dynamic_extra_negative_count": max(0, extra_count - positive_count),
            "no_regret_margin_violation_rate": violation_rate,
            "no_regret_tolerance": float(counterfactual_safety_tolerance),
            "step0_exact_static": bool(
                int(validation_step or 0) == 0
                and torch.equal(safe_fused, static)
            ),
        }
    raw_dynamic_regret = float(
        torch.relu(static_utility.detach() - dynamic_utility).mean().item()
    )
    fused_regret = float(
        torch.relu(static_utility.detach() - fused_utility).mean().item()
    )
    safe_alpha = torch.cat(alpha_batches, dim=0).to(device)
    zero_history_exact = bool(
        torch.equal(
            safe_fused[causal_update_count <= 0],
            static[causal_update_count <= 0],
        )
    )
    fixed_alpha: dict[str, Any] = {}
    for alpha in FIXED_ALPHA_GRID:
        raw_alpha = torch.full(
            (len(all_rows),),
            float(alpha),
            dtype=static.dtype,
            device=device,
        )
        effective_alpha = effective_memory_alpha(
            raw_alpha,
            causal_update_count,
        )
        fused = fuse_route_scores(static, dynamic, effective_alpha, valid)
        fused_ranks, _utility, _eligible = positive_rank_and_utility(
            fused,
            positive,
            valid,
        )
        summary = _route_summary(rows=all_rows, ranks=fused_ranks)
        fixed_alpha[str(alpha)] = {
            "balanced_macro_mrr": summary["balanced_macro_mrr"],
            "balanced_macro_recall@5": summary[
                "balanced_macro_recall@5"
            ],
            "zero_history_exact": bool(
                torch.equal(
                    fused[causal_update_count <= 0],
                    static[causal_update_count <= 0],
                )
            ),
            "by_benchmark": summary["by_benchmark"],
        }
    combined_union = CandidateUnion(
        candidate_rows=[row for union in unions for row in union.candidate_rows],
        static_rows=[row for union in unions for row in union.static_rows],
        dynamic_top_rows=[row for union in unions for row in union.dynamic_top_rows],
        dynamic_extra_rows=[row for union in unions for row in union.dynamic_extra_rows],
        static_equal_budget_rows=[
            row for union in unions for row in union.static_equal_budget_rows
        ],
    )
    candidate_recall = candidate_recall_report(
        combined_union,
        torch.cat(positive_full_batches, dim=0),
        torch.cat(valid_full_batches, dim=0),
        causal_update_count.detach().cpu(),
    )
    by_benchmark = {}
    for benchmark in sorted(static_summary["by_benchmark"]):
        static_metrics = static_summary["by_benchmark"][benchmark]
        dynamic_metrics = dynamic_summary["by_benchmark"][benchmark]
        safe_metrics = safe_fused_summary["by_benchmark"][benchmark]
        by_benchmark[benchmark] = {
            "static_mrr": float(static_metrics["mrr"]),
            "dynamic_mrr": float(dynamic_metrics["mrr"]),
            "raw_dynamic_mrr": float(dynamic_metrics["mrr"]),
            "dynamic_minus_static_mrr": float(
                dynamic_metrics["mrr"] - static_metrics["mrr"]
            ),
            "dynamic_recall@5": float(dynamic_metrics["recall@5"]),
            "safe_fused_mrr": float(safe_metrics["mrr"]),
            "fused_mrr": float(safe_metrics["mrr"]),
            "safe_fused_minus_static_mrr": float(
                safe_metrics["mrr"] - static_metrics["mrr"]
            ),
            "fused_minus_static_mrr": float(
                safe_metrics["mrr"] - static_metrics["mrr"]
            ),
            "candidate_admission_final_mrr": float(safe_metrics["mrr"]),
            "candidate_admission_final_minus_static_mrr": float(
                safe_metrics["mrr"] - static_metrics["mrr"]
            ),
            "safe_fused_recall@5": float(safe_metrics["recall@5"]),
            "row_count": int(dynamic_metrics["row_count"]),
        }
    return {
        "stage4_method": str(stage4_method),
        "rows": all_rows,
        "records": records,
        "static_logits_batches": static_full_batches,
        "features": torch.cat(feature_batches, dim=0),
        "causal_update_count": causal_update_count.detach().cpu(),
        "static_summary": static_summary,
        "dynamic_summary": dynamic_summary,
        "safe_fused_summary": safe_fused_summary,
        "by_benchmark": by_benchmark,
        "fixed_alpha": fixed_alpha,
        "safe_fused": {
            **safe_fused_summary,
            "zero_history_exact": zero_history_exact,
            "residual_bound": float(safe_memory_residual_bound),
            "alpha_mean": float(safe_alpha.mean().item()),
            "alpha_min": float(safe_alpha.min().item()),
            "alpha_max": float(safe_alpha.max().item()),
        },
        "cmc_fused": {
            **safe_fused_summary,
            "regret": fused_regret,
            "raw_dynamic_regret": raw_dynamic_regret,
            "alpha_zero_exact": bool(alpha_zero_exact),
            "alpha_one_exact": bool(alpha_one_exact),
            "zero_history_exact": zero_history_exact,
            "alpha_mean": float(safe_alpha.mean().item()),
            "alpha_min": float(safe_alpha.min().item()),
            "alpha_max": float(safe_alpha.max().item()),
        },
        "candidate_admission": candidate_admission_metrics,
        "candidate_recall": candidate_recall,
        "balanced_macro": {
            "static_mrr": static_summary["balanced_macro_mrr"],
            "dynamic_mrr": dynamic_summary["balanced_macro_mrr"],
            "dynamic_minus_static_mrr": (
                dynamic_summary["balanced_macro_mrr"]
                - static_summary["balanced_macro_mrr"]
            ),
            "dynamic_recall@5": dynamic_summary[
                "balanced_macro_recall@5"
            ],
            "safe_fused_mrr": safe_fused_summary["balanced_macro_mrr"],
            "safe_fused_minus_static_mrr": (
                safe_fused_summary["balanced_macro_mrr"]
                - static_summary["balanced_macro_mrr"]
            ),
            "safe_fused_recall@5": safe_fused_summary[
                "balanced_macro_recall@5"
            ],
            "raw_dynamic_mrr": dynamic_summary["balanced_macro_mrr"],
            "fused_mrr": safe_fused_summary["balanced_macro_mrr"],
            "fused_minus_static_mrr": (
                safe_fused_summary["balanced_macro_mrr"]
                - static_summary["balanced_macro_mrr"]
            ),
            "fused_regret": fused_regret,
            "raw_dynamic_regret": raw_dynamic_regret,
            "candidate_admission_final_mrr": (
                safe_fused_summary["balanced_macro_mrr"]
            ),
            "candidate_admission_final_minus_static_mrr": (
                safe_fused_summary["balanced_macro_mrr"]
                - static_summary["balanced_macro_mrr"]
            ),
        },
        "exclusion_counts": {
            "missing_positive": int((~positive.any(dim=-1)).sum().item()),
            "no_legal_candidate": int((~valid.any(dim=-1)).sum().item()),
        },
        "nonfinite_counts": {
            "loss": 0,
            "logits": int(
                (~torch.isfinite(static[valid])).sum().item()
                + (~torch.isfinite(dynamic[valid])).sum().item()
                + (~torch.isfinite(safe_fused[valid])).sum().item()
            ),
            "metrics": 0,
            "gradients": 0,
        },
        "candidate_union": {
            "static_k": int(static_k),
            "dynamic_extra_k": int(dynamic_extra_k),
            "final_k": int(final_k),
            "candidate_union_version": CANDIDATE_UNION_VERSION,
            "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        },
    }


def evaluate_stage4_validation(
    *,
    model: Any,
    validation_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    device: torch.device,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    batch_size: int,
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    final_k: int = 64,
    counterfactual_utility_weight: float = 0.05,
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    safe_memory_residual_bound: float = 2.0,
    stage4_method: str = "stage4_safe_memory_v1",
    validation_step: int | None = None,
) -> dict[str, Any]:
    common = {
        "model": model,
        "device": device,
        "skill_id_to_idx": skill_id_to_idx,
        "equivalent_skill_ids_by_skill_id": equivalent_skill_ids_by_skill_id,
        "batch_size": batch_size,
        "static_k": static_k,
        "dynamic_extra_k": dynamic_extra_k,
        "final_k": final_k,
        "counterfactual_utility_weight": counterfactual_utility_weight,
        "counterfactual_gain_margin": counterfactual_gain_margin,
        "counterfactual_safety_tolerance": counterfactual_safety_tolerance,
        "counterfactual_gain_weight": counterfactual_gain_weight,
        "counterfactual_safety_weight": counterfactual_safety_weight,
        "safe_memory_residual_bound": safe_memory_residual_bound,
        "stage4_method": stage4_method,
        "validation_step": validation_step,
    }
    return {
        "validation": _evaluate_stage4_partition(
            rows=validation_rows,
            **common,
        ),
        "gate": _evaluate_stage4_partition(rows=gate_rows, **common),
    }


def stage2_baseline_from_evaluation(
    evaluation: dict[str, Any],
    *,
    full_router_digest: str,
) -> dict[str, Any]:
    validation = dict(evaluation["validation"])
    if validation.get("stage4_method") == "candidate_admission_residual_v1":
        candidate = dict(validation.get("candidate_admission") or {})
        return {
            "full_router_digest": str(full_router_digest),
            "static_macro_mrr": float(
                validation["static_summary"]["balanced_macro_mrr"]
            ),
            "candidate_admission_step0_exact_static": (
                candidate.get("step0_exact_static") is True
            ),
            "by_benchmark": {
                benchmark: {"static_mrr": float(metrics["static_mrr"])}
                for benchmark, metrics in validation["by_benchmark"].items()
            },
        }
    if (
        validation.get("stage4_method")
        == "counterfactual_memory_calibration_v1"
    ):
        return {
            "full_router_digest": str(full_router_digest),
            "static_macro_mrr": float(
                validation["static_summary"]["balanced_macro_mrr"]
            ),
            "raw_dynamic_macro_mrr": float(
                validation["dynamic_summary"]["balanced_macro_mrr"]
            ),
            "raw_dynamic_regret": float(
                validation["balanced_macro"]["raw_dynamic_regret"]
            ),
            "by_benchmark": {
                benchmark: {
                    "static_mrr": float(metrics["static_mrr"]),
                    "raw_dynamic_mrr": float(metrics["raw_dynamic_mrr"]),
                }
                for benchmark, metrics in validation["by_benchmark"].items()
            },
        }
    return {
        "full_router_digest": str(full_router_digest),
        "static_macro_mrr": float(
            validation["static_summary"]["balanced_macro_mrr"]
        ),
        "dynamic_macro_mrr": float(
            validation["dynamic_summary"]["balanced_macro_mrr"]
        ),
        "safe_fused_macro_mrr": float(
            validation["safe_fused_summary"]["balanced_macro_mrr"]
        ),
        "by_benchmark": {
            benchmark: {
                "static_mrr": float(metrics["static_mrr"]),
                "dynamic_mrr": float(metrics["dynamic_mrr"]),
                "safe_fused_mrr": float(metrics["safe_fused_mrr"]),
            }
            for benchmark, metrics in validation["by_benchmark"].items()
        },
    }


def build_stage4_validation_report(
    evaluation: dict[str, Any],
    *,
    step: int,
    stage2_baseline: dict[str, Any],
    router_integrity: dict[str, Any],
    delta_state_validation: dict[str, Any],
    stage4_method: str = "stage4_safe_memory_v1",
    gradient_health: dict[str, Any] | None = None,
    validation_interval_steps: int = 400,
) -> dict[str, Any]:
    validation = dict(evaluation["validation"])
    by_benchmark = dict(validation["by_benchmark"])
    baseline_by_benchmark = dict(stage2_baseline["by_benchmark"])
    if stage4_method == "candidate_admission_residual_v1":
        candidate = dict(validation.get("candidate_admission") or {})
        gradient_health = dict(gradient_health or {})
        final_mrr = float(candidate.get("balanced_macro_mrr", float("nan")))
        static_mrr = float(
            candidate.get("static_balanced_macro_mrr", float("nan"))
        )
        auprc = float(candidate.get("admission_auprc", float("nan")))
        prevalence = float(
            candidate.get("admission_prevalence", float("nan"))
        )
        violation_rate = float(
            candidate.get("no_regret_margin_violation_rate", float("nan"))
        )
        tolerance = float(candidate.get("no_regret_tolerance", float("nan")))
        release_checks = {
            "nonzero_validation_step": int(step) > 0,
            "dynamic_extra_positive_support": int(
                candidate.get("dynamic_extra_positive_count", 0)
            )
            > 0,
            "admission_signal_above_prevalence": auprc > prevalence,
            "final_static_nonregression": final_mrr >= static_mrr,
            "no_regret_margin_constraint": violation_rate <= tolerance,
            "finite_nonzero_gradients": (
                gradient_health.get("finite") is True
                and gradient_health.get("nonzero") is True
            ),
            "step0_static_control_verified": (
                stage2_baseline.get("candidate_admission_step0_exact_static")
                is True
            ),
            "stage2_router_unchanged": (
                router_integrity.get("full_router_digest_unchanged") is True
                and router_integrity.get("fast_router_digest_unchanged") is True
            ),
        }
        nonfinite_counts = dict(validation["nonfinite_counts"])
        finite_metrics = all(
            math.isfinite(value)
            for value in (final_mrr, static_mrr, auprc, prevalence, violation_rate, tolerance)
        )
        mechanically_eligible = (
            set(by_benchmark) == EXPECTED_STAGE4_BENCHMARKS
            and router_integrity.get("static_logits_exact") is True
            and float(
                router_integrity.get(
                    "max_abs_static_logit_difference",
                    float("inf"),
                )
            )
            == 0.0
            and sum(int(value) for value in nonfinite_counts.values()) == 0
            and dict(delta_state_validation).get("status") == "ok"
            and gradient_health.get("finite") is True
            and finite_metrics
        )
        return {
            "schema_version": STAGE4_VALIDATION_REPORT_SCHEMA,
            "status": "ok",
            "stage4_method": stage4_method,
            "release_status": (
                "ok" if all(release_checks.values()) else "action_required"
            ),
            "release_checks": release_checks,
            "step": int(step),
            "validation_interval_steps": int(validation_interval_steps),
            "mechanically_eligible": bool(mechanically_eligible),
            "router_integrity": dict(router_integrity),
            "stage2_baseline": dict(stage2_baseline),
            "candidate_union": dict(validation["candidate_union"]),
            "candidate_recall": dict(validation["candidate_recall"]),
            "nonfinite_counts": nonfinite_counts,
            "exclusion_counts": dict(validation["exclusion_counts"]),
            "delta_state_validation": dict(delta_state_validation),
            "balanced_macro": dict(validation["balanced_macro"]),
            "by_benchmark": by_benchmark,
            "candidate_admission": candidate,
            "gradient_health": gradient_health,
            "validation_row_count": len(validation["rows"]),
            "gate_row_count": len(evaluation["gate"]["rows"]),
        }
    if stage4_method == "counterfactual_memory_calibration_v1":
        cmc_fused = dict(validation["cmc_fused"])
        gradient_health = dict(gradient_health or {})
        fused_macro = float(cmc_fused["balanced_macro_mrr"])
        fused_regret = float(cmc_fused["regret"])
        release_checks = {
            "nonzero_validation_step": int(step) > 0,
            "fused_macro_improvement": (
                fused_macro >= float(stage2_baseline["static_macro_mrr"]) + 0.005
            ),
            "per_regime_nonregression": all(
                float(metrics["fused_mrr"])
                >= float(baseline_by_benchmark[benchmark]["static_mrr"]) - 0.01
                for benchmark, metrics in by_benchmark.items()
            ),
            "regret_reduced": (
                fused_regret < float(stage2_baseline["raw_dynamic_regret"])
            ),
            "exact_endpoints": (
                cmc_fused.get("alpha_zero_exact") is True
                and cmc_fused.get("alpha_one_exact") is True
                and cmc_fused.get("zero_history_exact") is True
            ),
            "finite_nonzero_gradients": (
                gradient_health.get("finite") is True
                and gradient_health.get("nonzero") is True
            ),
            "stage2_router_unchanged": (
                router_integrity.get("full_router_digest_unchanged") is True
                and router_integrity.get("fast_router_digest_unchanged") is True
            ),
        }
        nonfinite_counts = dict(validation["nonfinite_counts"])
        mechanically_eligible = (
            set(by_benchmark) == EXPECTED_STAGE4_BENCHMARKS
            and router_integrity.get("static_logits_exact") is True
            and float(
                router_integrity.get(
                    "max_abs_static_logit_difference",
                    float("inf"),
                )
            )
            == 0.0
            and sum(int(value) for value in nonfinite_counts.values()) == 0
            and dict(delta_state_validation).get("status") == "ok"
            and release_checks["exact_endpoints"]
            and gradient_health.get("finite") is True
        )
        return {
            "schema_version": STAGE4_VALIDATION_REPORT_SCHEMA,
            "status": "ok",
            "stage4_method": stage4_method,
            "release_status": (
                "ok" if all(release_checks.values()) else "action_required"
            ),
            "release_checks": release_checks,
            "step": int(step),
            "validation_interval_steps": int(validation_interval_steps),
            "mechanically_eligible": bool(mechanically_eligible),
            "router_integrity": dict(router_integrity),
            "stage2_baseline": dict(stage2_baseline),
            "candidate_union": dict(validation["candidate_union"]),
            "candidate_recall": dict(validation["candidate_recall"]),
            "nonfinite_counts": nonfinite_counts,
            "exclusion_counts": dict(validation["exclusion_counts"]),
            "delta_state_validation": dict(delta_state_validation),
            "balanced_macro": dict(validation["balanced_macro"]),
            "by_benchmark": by_benchmark,
            "cmc_fused": cmc_fused,
            "gradient_health": gradient_health,
            "validation_row_count": len(validation["rows"]),
            "gate_row_count": len(evaluation["gate"]["rows"]),
        }
    safe_fused_macro = float(validation["balanced_macro"]["safe_fused_mrr"])
    safe_delta_macro = float(
        validation["balanced_macro"]["safe_fused_minus_static_mrr"]
    )
    nonnegative_benchmarks = sum(
        float(metrics["safe_fused_minus_static_mrr"]) >= -0.01
        for metrics in by_benchmark.values()
    )
    maximum_regression = max(
        (
            float(baseline_by_benchmark[benchmark]["safe_fused_mrr"])
            - float(metrics["safe_fused_mrr"])
            for benchmark, metrics in by_benchmark.items()
        ),
        default=0.0,
    )
    release_checks = {
        "safe_fused_stage2_nonregression": (
            safe_fused_macro
            >= float(stage2_baseline["safe_fused_macro_mrr"]) - 0.01
        ),
        "safe_fused_static_nonregression": safe_delta_macro >= -0.005,
        "three_safe_benchmarks": nonnegative_benchmarks >= 3,
        "safe_fused_per_benchmark_regression": maximum_regression <= 0.03,
        "zero_history_exact": validation["safe_fused"]["zero_history_exact"] is True,
    }
    nonfinite_counts = dict(validation["nonfinite_counts"])
    mechanically_eligible = (
        set(by_benchmark) == EXPECTED_STAGE4_BENCHMARKS
        and router_integrity.get("static_logits_exact") is True
        and float(
            router_integrity.get(
                "max_abs_static_logit_difference",
                float("inf"),
            )
        )
        == 0.0
        and sum(int(value) for value in nonfinite_counts.values()) == 0
        and dict(delta_state_validation).get("status") == "ok"
    )
    return {
        "schema_version": STAGE4_VALIDATION_REPORT_SCHEMA,
        "status": "ok",
        "release_status": (
            "ok" if all(release_checks.values()) else "action_required"
        ),
        "release_checks": release_checks,
        "step": int(step),
        "mechanically_eligible": bool(mechanically_eligible),
        "router_integrity": dict(router_integrity),
        "stage2_baseline": dict(stage2_baseline),
        "candidate_union": dict(validation["candidate_union"]),
        "candidate_recall": dict(validation["candidate_recall"]),
        "nonfinite_counts": nonfinite_counts,
        "exclusion_counts": dict(validation["exclusion_counts"]),
        "delta_state_validation": dict(delta_state_validation),
        "balanced_macro": dict(validation["balanced_macro"]),
        "by_benchmark": by_benchmark,
        "safe_fused": dict(validation["safe_fused"]),
        "fixed_alpha": dict(validation["fixed_alpha"]),
        "validation_row_count": len(validation["rows"]),
        "gate_row_count": len(evaluation["gate"]["rows"]),
    }


def persist_stage4_validation_artifacts(
    *,
    output_dir: str | Path,
    step: int,
    report: dict[str, Any],
    evaluation: dict[str, Any],
    compact_checkpoint_path: str | Path,
    skill_id_to_idx: dict[str, int],
    resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    validation_dir = Path(output_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = sha256_path(compact_checkpoint_path)
    candidate_union = dict(report["candidate_union"])
    all_records = [
        *evaluation["validation"]["records"],
        *evaluation["gate"]["records"],
    ]
    update_count_cap = max(
        1.0,
        max(
            (float(row["causal_update_count"]) for row in all_records),
            default=0.0,
        ),
    )
    candidate_count_cap = max(
        1.0,
        max(
            (
                float(sum(bool(value) for value in row["valid_mask"]))
                for row in all_records
            ),
            default=0.0,
        ),
    )
    manifest_identity = {
        "candidate_union_version": candidate_union[
            "candidate_union_version"
        ],
        "candidate_selection_version": candidate_union[
            "candidate_selection_version"
        ],
        "pool_protocol": "static_plus_dynamic_extra",
        "static_k": int(candidate_union["static_k"]),
        "dynamic_extra_k": int(candidate_union["dynamic_extra_k"]),
        "final_k": int(candidate_union["final_k"]),
        "feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
        "heuristic_alpha_version": HEURISTIC_ALPHA_VERSION,
        "model_checkpoint_chain_digest": canonical_digest(
            {"stage4_delta_sha256": checkpoint_identity["sha256"]}
        ),
        "skill_mapping_digest": canonical_digest(
            [
                skill_id
                for skill_id, _index in sorted(
                    skill_id_to_idx.items(),
                    key=lambda item: int(item[1]),
                )
            ]
        ),
        "feature_update_count_cap": float(update_count_cap),
        "feature_candidate_count_cap": float(candidate_count_cap),
        "sequential_benchmarks": sorted(EXPECTED_STAGE4_BENCHMARKS),
    }
    validation_records_path = validation_dir / f"step{step}.route_records.jsonl"
    validation_manifest_path = validation_dir / f"step{step}.route_manifest.json"
    persist_memory_utility_route_records(
        validation_records_path,
        list(evaluation["validation"]["records"]),
        manifest_identity=manifest_identity,
        manifest_path=validation_manifest_path,
    )
    gate_records_path = validation_dir / f"step{step}.gate_route_records.jsonl"
    gate_manifest_path = validation_dir / f"step{step}.gate_route_manifest.json"
    persist_memory_utility_route_records(
        gate_records_path,
        list(evaluation["gate"]["records"]),
        manifest_identity=manifest_identity,
        manifest_path=gate_manifest_path,
    )
    payload = {
        **report,
        "stage4_checkpoint": checkpoint_identity,
        "validation_route_records": {
            **sha256_path(validation_records_path),
            "record_count": len(evaluation["validation"]["records"]),
            "row_digest_sha256": canonical_digest(
                [
                    row["row_digest"]
                    for row in evaluation["validation"]["records"]
                ]
            ),
        },
        "validation_route_manifest": sha256_path(validation_manifest_path),
        "gate_route_records": {
            **sha256_path(gate_records_path),
            "record_count": len(evaluation["gate"]["records"]),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in evaluation["gate"]["records"]]
            ),
        },
        "gate_route_manifest": sha256_path(gate_manifest_path),
    }
    if resume_checkpoint is not None:
        resume_identity = dict(resume_checkpoint)
        actual_resume = sha256_path(resume_identity["path"])
        if actual_resume["sha256"] != str(resume_identity.get("sha256") or ""):
            raise ValueError("Stage4 resume checkpoint SHA-256 mismatch")
        payload["resume_checkpoint"] = actual_resume
    payload["manifest_sha256"] = canonical_digest(payload)
    report_path = validation_dir / f"step{step}.json"
    _atomic_write_json(report_path, payload)
    return payload, report_path


def validate_static_baseline_batches(
    baseline_batches: list[torch.Tensor],
    current_batches: list[torch.Tensor],
) -> dict[str, Any]:
    if len(baseline_batches) != len(current_batches):
        raise ValueError("static validation batch count drifted")
    maximum = 0.0
    for baseline, current in zip(baseline_batches, current_batches):
        if baseline.shape != current.shape or baseline.dtype != current.dtype:
            raise ValueError("static validation tensor contract drifted")
        difference = (
            float((baseline - current).abs().max().item())
            if baseline.numel()
            else 0.0
        )
        maximum = max(maximum, difference)
        if not torch.equal(baseline, current):
            raise ValueError("static validation logits drifted")
    return {
        "status": "ok",
        "static_logits_exact": True,
        "max_abs_static_logit_difference": maximum,
    }


def choose_fixed_alpha(report: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for raw_alpha, metrics in dict(report.get("fixed_alpha") or {}).items():
        alpha = float(raw_alpha)
        macro_mrr = float(metrics["balanced_macro_mrr"])
        if (
            not math.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
            or not math.isfinite(macro_mrr)
        ):
            raise ValueError("fixed-alpha validation metrics must be finite")
        candidates.append((macro_mrr, -alpha, alpha, dict(metrics)))
    if not candidates:
        raise ValueError("validation report has no fixed-alpha results")
    macro_mrr, _negative_alpha, alpha, metrics = max(
        candidates,
        key=lambda item: item[:2],
    )
    return {
        "fixed_alpha": alpha,
        **metrics,
        "balanced_macro_mrr": macro_mrr,
    }


def _validate_route_record_identity(
    recorded: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    identity = dict(recorded or {})
    actual = sha256_path(identity["path"])
    if actual["sha256"] != str(identity.get("sha256") or ""):
        raise ValueError(f"{label} SHA-256 mismatch")
    rows = [
        json.loads(line)
        for line in Path(identity["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != int(identity.get("record_count", -1)):
        raise ValueError(f"{label} count mismatch")
    if canonical_digest([row["row_digest"] for row in rows]) != str(
        identity.get("row_digest_sha256") or ""
    ):
        raise ValueError(f"{label} row digest mismatch")
    return identity


def select_stage4_dynamic_checkpoint(
    reports: list[dict[str, Any]],
    *,
    checkpoint_paths: dict[int, Path],
    validation_report_paths: dict[int, Path],
) -> dict[str, Any]:
    eligible = []
    for report in reports:
        if (
            report.get("status") != "ok"
            or report.get("mechanically_eligible") is not True
        ):
            continue
        by_benchmark = dict(report.get("by_benchmark") or {})
        if set(by_benchmark) != EXPECTED_STAGE4_BENCHMARKS:
            continue
        router_integrity = dict(report.get("router_integrity") or {})
        stage2_baseline = dict(report.get("stage2_baseline") or {})
        if (
            router_integrity.get("static_logits_exact") is not True
            or float(
                router_integrity.get(
                    "max_abs_static_logit_difference",
                    float("inf"),
                )
            )
            != 0.0
            or str(router_integrity.get("full_router_digest") or "")
            != str(stage2_baseline.get("full_router_digest") or "")
        ):
            raise ValueError(
                "Stage4 validation report violates exact static parity"
            )
        if dict(report.get("delta_state_validation") or {}).get("status") != "ok":
            raise ValueError(
                "Stage4 validation report has an invalid delta checkpoint"
            )
        candidate_union = dict(report.get("candidate_union") or {})
        if int(candidate_union["final_k"]) > int(candidate_union["static_k"]):
            raise ValueError("Stage4 final_k must not exceed static_k")
        if report.get("release_status") not in {"ok", "action_required"}:
            raise ValueError(
                "Stage4 validation report has invalid release status"
            )
        if (
            sum(
                int(value)
                for value in dict(report.get("nonfinite_counts") or {}).values()
            )
            != 0
        ):
            raise ValueError(
                "Stage4 validation report contains nonfinite values"
            )
        macro = dict(report.get("balanced_macro") or {})
        metric_values = [
            float(macro["dynamic_mrr"]),
            float(macro["dynamic_minus_static_mrr"]),
            float(macro["dynamic_recall@5"]),
            float(macro["safe_fused_mrr"]),
            float(macro["safe_fused_minus_static_mrr"]),
            *[
                float(metrics[name])
                for metrics in by_benchmark.values()
                for name in (
                    "dynamic_mrr",
                    "static_mrr",
                    "dynamic_minus_static_mrr",
                    "safe_fused_mrr",
                    "safe_fused_minus_static_mrr",
                )
            ],
        ]
        if not all(math.isfinite(value) for value in metric_values):
            raise ValueError(
                "Stage4 validation selection metrics must be finite"
            )
        step = int(report["step"])
        checkpoint = Path(checkpoint_paths[step]).resolve()
        if not checkpoint.is_file():
            raise ValueError(
                f"missing Stage4 validation checkpoint: {checkpoint}"
            )
        report_path = Path(validation_report_paths[step]).resolve()
        if not report_path.is_file():
            raise ValueError(
                f"missing Stage4 validation report: {report_path}"
            )
        persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
        if persisted_report != report:
            raise ValueError(
                "Stage4 validation report path does not match selected report"
            )
        report_without_hash = dict(report)
        recorded_report_sha = str(
            report_without_hash.pop("manifest_sha256", "")
        )
        if recorded_report_sha != canonical_digest(report_without_hash):
            raise ValueError("Stage4 validation report self-hash mismatch")
        validation_route_records = _validate_route_record_identity(
            report.get("validation_route_records") or {},
            label="Stage4 validation route records",
        )
        gate_route_records = _validate_route_record_identity(
            report.get("gate_route_records") or {},
            label="Stage4 gate route records",
        )
        gate_route_manifest = dict(report.get("gate_route_manifest") or {})
        actual_gate_manifest = sha256_path(gate_route_manifest["path"])
        if actual_gate_manifest["sha256"] != str(
            gate_route_manifest.get("sha256") or ""
        ):
            raise ValueError(
                "Stage4 gate route-manifest SHA-256 mismatch"
            )
        safe_fused = dict(report.get("safe_fused") or {})
        safe_fused_mrr = float(safe_fused.get("balanced_macro_mrr", float("nan")))
        if not math.isfinite(safe_fused_mrr):
            raise ValueError("Stage4 safe-fused validation MRR must be finite")
        key = (
            report.get("release_status") == "ok",
            safe_fused_mrr,
            float(macro.get("safe_fused_minus_static_mrr", float("-inf"))),
            float(macro["dynamic_recall@5"]),
            -step,
        )
        eligible.append(
            (
                key,
                report,
                checkpoint,
                report_path,
                validation_route_records,
                gate_route_records,
                gate_route_manifest,
            )
        )
    if not eligible:
        raise ValueError(
            "no mechanically eligible Stage4 validation checkpoint"
        )
    (
        _key,
        report,
        checkpoint,
        report_path,
        validation_route_records,
        gate_route_records,
        gate_route_manifest,
    ) = max(eligible, key=lambda item: item[0])
    return {
        "status": "ok",
        "release_status": str(report["release_status"]),
        "selected_step": int(report["step"]),
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(report_path),
        "selected_validation_report_sha256": sha256_path(report_path)["sha256"],
        "validation_route_records": validation_route_records,
        "gate_route_records": gate_route_records,
        "gate_route_manifest": gate_route_manifest,
        "router_integrity": dict(report["router_integrity"]),
        "stage2_baseline": dict(report["stage2_baseline"]),
        "candidate_union": dict(report["candidate_union"]),
        "safe_fused_selection": dict(report["safe_fused"]),
        "fixed_alpha_selection": choose_fixed_alpha(report),
    }


def select_stage4_cmc_checkpoint(
    reports: list[dict[str, Any]],
    *,
    checkpoint_paths: dict[int, Path],
    validation_report_paths: dict[int, Path],
    terminal_step: int | None = None,
) -> dict[str, Any]:
    """Select by deployed CMC fusion, then apply the no-regret promotion gate."""

    resolved_terminal_step = (
        None if terminal_step is None else int(terminal_step)
    )
    if resolved_terminal_step is not None and resolved_terminal_step <= 0:
        raise ValueError("CMC terminal validation step must be positive")

    eligible: list[
        tuple[
            tuple[float, float, int],
            dict[str, Any],
            Path,
            Path,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    baseline_identity: str | None = None
    for report in reports:
        if (
            report.get("status") != "ok"
            or report.get("mechanically_eligible") is not True
            or report.get("stage4_method")
            != "counterfactual_memory_calibration_v1"
        ):
            continue
        step = int(report.get("step", -1))
        interval = int(report.get("validation_interval_steps", 400))
        if (
            interval <= 0
            or step < 0
            or (
                step != 0
                and step % interval != 0
                and step != resolved_terminal_step
            )
            or (
                resolved_terminal_step is not None
                and step > resolved_terminal_step
            )
        ):
            raise ValueError(
                "CMC validation steps must be zero, match the validation interval, "
                "or equal the explicit terminal step"
            )
        by_benchmark = dict(report.get("by_benchmark") or {})
        if set(by_benchmark) != EXPECTED_STAGE4_BENCHMARKS:
            continue
        router_integrity = dict(report.get("router_integrity") or {})
        stage2_baseline = dict(report.get("stage2_baseline") or {})
        if (
            router_integrity.get("static_logits_exact") is not True
            or float(
                router_integrity.get(
                    "max_abs_static_logit_difference",
                    float("inf"),
                )
            )
            != 0.0
            or str(router_integrity.get("full_router_digest") or "")
            != str(stage2_baseline.get("full_router_digest") or "")
            or router_integrity.get("full_router_digest_unchanged") is not True
            or router_integrity.get("fast_router_digest_unchanged") is not True
        ):
            raise ValueError("CMC validation report violates exact Stage2 router parity")
        current_baseline_identity = canonical_digest(stage2_baseline)
        if baseline_identity is None:
            baseline_identity = current_baseline_identity
        elif current_baseline_identity != baseline_identity:
            raise ValueError("CMC validation reports disagree on the Stage2 baseline")
        if dict(report.get("delta_state_validation") or {}).get("status") != "ok":
            raise ValueError("CMC validation report has an invalid delta checkpoint")
        candidate_union = dict(report.get("candidate_union") or {})
        if int(candidate_union["final_k"]) > int(candidate_union["static_k"]):
            raise ValueError("Stage4 final_k must not exceed static_k")
        if (
            sum(
                int(value)
                for value in dict(report.get("nonfinite_counts") or {}).values()
            )
            != 0
        ):
            raise ValueError("CMC validation report contains nonfinite values")

        macro = dict(report.get("balanced_macro") or {})
        cmc_fused = dict(report.get("cmc_fused") or {})
        gradient_health = dict(report.get("gradient_health") or {})
        fused_mrr = float(cmc_fused.get("balanced_macro_mrr", float("nan")))
        fused_regret = float(cmc_fused.get("regret", float("nan")))
        metric_values = [
            float(macro.get("raw_dynamic_mrr", float("nan"))),
            float(macro.get("fused_mrr", float("nan"))),
            float(macro.get("fused_minus_static_mrr", float("nan"))),
            float(macro.get("fused_regret", float("nan"))),
            fused_mrr,
            fused_regret,
            *[
                float(metrics[name])
                for metrics in by_benchmark.values()
                for name in (
                    "static_mrr",
                    "raw_dynamic_mrr",
                    "fused_mrr",
                    "fused_minus_static_mrr",
                )
            ],
        ]
        if not all(math.isfinite(value) for value in metric_values):
            raise ValueError("CMC validation selection metrics must be finite")
        endpoint_exact = (
            cmc_fused.get("alpha_zero_exact") is True
            and cmc_fused.get("alpha_one_exact") is True
            and cmc_fused.get("zero_history_exact") is True
        )
        if not endpoint_exact:
            raise ValueError("CMC validation endpoints are not exact")
        if gradient_health.get("finite") is not True:
            raise ValueError("CMC validation gradients are not finite")
        if step > 0 and gradient_health.get("nonzero") is not True:
            raise ValueError("CMC validation gradients are zero")

        static_macro = float(stage2_baseline["static_macro_mrr"])
        if step == 0:
            if fused_mrr != static_macro or any(
                float(metrics["fused_mrr"])
                != float(stage2_baseline["by_benchmark"][benchmark]["static_mrr"])
                for benchmark, metrics in by_benchmark.items()
            ):
                raise ValueError("CMC step zero must be the exact static control")

        checkpoint = Path(checkpoint_paths[step]).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"missing CMC validation checkpoint: {checkpoint}")
        report_path = Path(validation_report_paths[step]).resolve()
        if not report_path.is_file():
            raise ValueError(f"missing CMC validation report: {report_path}")
        persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
        if persisted_report != report:
            raise ValueError("CMC validation report path does not match report")
        report_without_hash = dict(report)
        recorded_report_sha = str(report_without_hash.pop("manifest_sha256", ""))
        if recorded_report_sha != canonical_digest(report_without_hash):
            raise ValueError("CMC validation report self-hash mismatch")
        validation_route_records = _validate_route_record_identity(
            report.get("validation_route_records") or {},
            label="CMC validation route records",
        )
        gate_route_records = _validate_route_record_identity(
            report.get("gate_route_records") or {},
            label="CMC gate route records",
        )
        gate_route_manifest = dict(report.get("gate_route_manifest") or {})
        actual_gate_manifest = sha256_path(gate_route_manifest["path"])
        if actual_gate_manifest["sha256"] != str(
            gate_route_manifest.get("sha256") or ""
        ):
            raise ValueError("CMC gate route-manifest SHA-256 mismatch")
        eligible.append(
            (
                (fused_mrr, -fused_regret, -step),
                report,
                checkpoint,
                report_path,
                validation_route_records,
                gate_route_records,
                gate_route_manifest,
            )
        )
    if not eligible:
        raise ValueError("no mechanically eligible CMC validation checkpoint")

    (
        _selection_key,
        selected,
        checkpoint,
        report_path,
        validation_route_records,
        gate_route_records,
        gate_route_manifest,
    ) = max(eligible, key=lambda item: item[0])
    step = int(selected["step"])
    baseline = dict(selected["stage2_baseline"])
    by_benchmark = dict(selected["by_benchmark"])
    cmc_fused = dict(selected["cmc_fused"])
    fused_mrr = float(cmc_fused["balanced_macro_mrr"])
    fused_regret = float(cmc_fused["regret"])
    release_checks = {
        "nonzero_selected_step": step > 0,
        "fused_macro_improvement": (
            fused_mrr >= float(baseline["static_macro_mrr"]) + 0.005
        ),
        "per_regime_nonregression": all(
            float(metrics["fused_mrr"])
            >= float(baseline["by_benchmark"][benchmark]["static_mrr"]) - 0.01
            for benchmark, metrics in by_benchmark.items()
        ),
        "regret_reduced": fused_regret < float(baseline["raw_dynamic_regret"]),
        "exact_endpoints": (
            cmc_fused.get("alpha_zero_exact") is True
            and cmc_fused.get("alpha_one_exact") is True
            and cmc_fused.get("zero_history_exact") is True
        ),
        "finite_nonzero_gradients": (
            dict(selected.get("gradient_health") or {}).get("finite") is True
            and dict(selected.get("gradient_health") or {}).get("nonzero") is True
        ),
        "stage2_router_unchanged": (
            dict(selected["router_integrity"]).get(
                "full_router_digest_unchanged"
            )
            is True
            and dict(selected["router_integrity"]).get(
                "fast_router_digest_unchanged"
            )
            is True
        ),
    }
    promoted = all(release_checks.values())
    return {
        "status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "release_status": "ok" if promoted else "action_required",
        "promotion_status": "promoted" if promoted else "stage4_not_promoted",
        "release_checks": release_checks,
        "selected_step": step,
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(report_path),
        "selected_validation_report_sha256": sha256_path(report_path)["sha256"],
        "validation_route_records": validation_route_records,
        "gate_route_records": gate_route_records,
        "gate_route_manifest": gate_route_manifest,
        "router_integrity": dict(selected["router_integrity"]),
        "stage2_baseline": baseline,
        "candidate_union": dict(selected["candidate_union"]),
        "cmc_fused_selection": cmc_fused,
        "gradient_health": dict(selected["gradient_health"]),
    }


def select_stage4_candidate_admission_checkpoint(
    reports: list[dict[str, Any]],
    *,
    checkpoint_paths: dict[int, Path],
    validation_report_paths: dict[int, Path],
    terminal_step: int | None = None,
) -> dict[str, Any]:
    """Select final MRR subject to the candidate-admission promotion contract."""

    terminal = None if terminal_step is None else int(terminal_step)
    eligible: list[
        tuple[
            tuple[bool, float, float, int],
            dict[str, Any],
            Path,
            Path,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, bool],
        ]
    ] = []
    baseline_identity: str | None = None
    for report in reports:
        if (
            report.get("status") != "ok"
            or report.get("mechanically_eligible") is not True
            or report.get("stage4_method")
            != "candidate_admission_residual_v1"
        ):
            continue
        step = int(report.get("step", -1))
        interval = int(report.get("validation_interval_steps", 300))
        if (
            interval <= 0
            or step < 0
            or (
                step != 0
                and step % interval != 0
                and step != terminal
            )
            or (terminal is not None and step > terminal)
        ):
            raise ValueError("candidate-admission validation step is off schedule")
        by_benchmark = dict(report.get("by_benchmark") or {})
        if set(by_benchmark) != EXPECTED_STAGE4_BENCHMARKS:
            continue
        router = dict(report.get("router_integrity") or {})
        baseline = dict(report.get("stage2_baseline") or {})
        if (
            router.get("static_logits_exact") is not True
            or float(router.get("max_abs_static_logit_difference", float("inf")))
            != 0.0
            or str(router.get("full_router_digest") or "")
            != str(baseline.get("full_router_digest") or "")
            or router.get("full_router_digest_unchanged") is not True
            or router.get("fast_router_digest_unchanged") is not True
        ):
            raise ValueError(
                "candidate-admission validation violates Stage2 router parity"
            )
        current_baseline_identity = canonical_digest(baseline)
        if baseline_identity is None:
            baseline_identity = current_baseline_identity
        elif current_baseline_identity != baseline_identity:
            raise ValueError(
                "candidate-admission validation reports disagree on baseline"
            )
        if dict(report.get("delta_state_validation") or {}).get("status") != "ok":
            raise ValueError("candidate-admission delta validation failed")
        if sum(
            int(value)
            for value in dict(report.get("nonfinite_counts") or {}).values()
        ) != 0:
            raise ValueError("candidate-admission validation contains nonfinite values")
        candidate = dict(report.get("candidate_admission") or {})
        gradient = dict(report.get("gradient_health") or {})
        final_mrr = float(candidate.get("balanced_macro_mrr", float("nan")))
        static_mrr = float(
            candidate.get("static_balanced_macro_mrr", float("nan"))
        )
        auprc = float(candidate.get("admission_auprc", float("nan")))
        prevalence = float(
            candidate.get("admission_prevalence", float("nan"))
        )
        violation_rate = float(
            candidate.get("no_regret_margin_violation_rate", float("nan"))
        )
        tolerance = float(candidate.get("no_regret_tolerance", float("nan")))
        if not all(
            math.isfinite(value)
            for value in (
                final_mrr,
                static_mrr,
                auprc,
                prevalence,
                violation_rate,
                tolerance,
            )
        ):
            raise ValueError("candidate-admission selection metrics must be finite")
        release_checks = {
            "nonzero_selected_step": step > 0,
            "dynamic_extra_positive_support": int(
                candidate.get("dynamic_extra_positive_count", 0)
            )
            > 0,
            "admission_signal_above_prevalence": auprc > prevalence,
            "final_static_nonregression": final_mrr >= static_mrr,
            "no_regret_margin_constraint": violation_rate <= tolerance,
            "finite_nonzero_gradients": (
                gradient.get("finite") is True
                and gradient.get("nonzero") is True
            ),
            "step0_static_control_verified": (
                baseline.get("candidate_admission_step0_exact_static") is True
            ),
            "stage2_router_unchanged": True,
        }
        promoted = all(release_checks.values())
        checkpoint = Path(checkpoint_paths[step]).resolve()
        report_path = Path(validation_report_paths[step]).resolve()
        if not checkpoint.is_file() or not report_path.is_file():
            raise ValueError("candidate-admission selection artifact is missing")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        if persisted != report:
            raise ValueError(
                "candidate-admission validation path does not match report"
            )
        unhashed = dict(report)
        recorded_hash = str(unhashed.pop("manifest_sha256", ""))
        if recorded_hash != canonical_digest(unhashed):
            raise ValueError("candidate-admission validation self-hash mismatch")
        validation_records = _validate_route_record_identity(
            report.get("validation_route_records") or {},
            label="candidate-admission validation route records",
        )
        gate_records = _validate_route_record_identity(
            report.get("gate_route_records") or {},
            label="candidate-admission gate route records",
        )
        gate_manifest = dict(report.get("gate_route_manifest") or {})
        if sha256_path(gate_manifest["path"])["sha256"] != str(
            gate_manifest.get("sha256") or ""
        ):
            raise ValueError("candidate-admission gate manifest SHA-256 mismatch")
        eligible.append(
            (
                (promoted, final_mrr, -violation_rate, -step),
                report,
                checkpoint,
                report_path,
                validation_records,
                gate_records,
                gate_manifest,
                release_checks,
            )
        )
    if not eligible:
        raise ValueError(
            "no mechanically eligible candidate-admission validation checkpoint"
        )
    (
        key,
        selected,
        checkpoint,
        report_path,
        validation_records,
        gate_records,
        gate_manifest,
        release_checks,
    ) = max(eligible, key=lambda item: item[0])
    promoted = bool(key[0])
    return {
        "status": "ok",
        "stage4_method": "candidate_admission_residual_v1",
        "release_status": "ok" if promoted else "action_required",
        "promotion_status": "promoted" if promoted else "stage4_not_promoted",
        "release_checks": release_checks,
        "selected_step": int(selected["step"]),
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(report_path),
        "selected_validation_report_sha256": sha256_path(report_path)["sha256"],
        "validation_route_records": validation_records,
        "gate_route_records": gate_records,
        "gate_route_manifest": gate_manifest,
        "router_integrity": dict(selected["router_integrity"]),
        "stage2_baseline": dict(selected["stage2_baseline"]),
        "candidate_union": dict(selected["candidate_union"]),
        "candidate_admission_selection": dict(selected["candidate_admission"]),
        "gradient_health": dict(selected["gradient_health"]),
        "base_cmc_checkpoint_sha256": str(
            selected.get("base_cmc_checkpoint_sha256") or ""
        ),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_file_sha(
    path: str | Path,
    expected: str,
    *,
    label: str,
) -> None:
    if sha256_path(path)["sha256"] != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")


def _read_self_hashed_manifest(
    path: str | Path,
    expected_schema: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != expected_schema
    ):
        raise ValueError(f"unsupported manifest schema: {path}")
    digest_payload = dict(payload)
    recorded = str(digest_payload.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(digest_payload):
        raise ValueError(f"manifest self-hash mismatch: {path}")
    return payload


def write_stage4_dynamic_selection(
    output_path: str | Path,
    *,
    selection: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        **selection,
        "schema_version": STAGE4_DYNAMIC_SELECTION_SCHEMA,
    }
    required = {
        "status",
        "release_status",
        "selected_step",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "selected_validation_report_path",
        "selected_validation_report_sha256",
        "validation_route_records",
        "gate_route_records",
        "gate_route_manifest",
        "router_integrity",
        "stage2_baseline",
        "candidate_union",
    }
    if payload.get("stage4_method") in {
        "counterfactual_memory_calibration_v1",
        "candidate_admission_residual_v1",
    }:
        required.update(
            {
                "stage4_method",
                "promotion_status",
                "release_checks",
                "gradient_health",
            }
        )
        required.add(
            "candidate_admission_selection"
            if payload.get("stage4_method")
            == "candidate_admission_residual_v1"
            else "cmc_fused_selection"
        )
        if payload.get("stage4_method") == "candidate_admission_residual_v1":
            required.add("base_cmc_checkpoint_sha256")
    else:
        required.update({"safe_fused_selection", "fixed_alpha_selection"})
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"Stage4 dynamic selection missing field: {missing[0]}"
        )
    if payload["status"] != "ok" or payload["release_status"] not in {
        "ok",
        "action_required",
    }:
        raise ValueError("Stage4 dynamic selection status is invalid")
    _require_file_sha(
        payload["selected_checkpoint_path"],
        payload["selected_checkpoint_sha256"],
        label="selected Stage4 checkpoint",
    )
    _require_file_sha(
        payload["selected_validation_report_path"],
        payload["selected_validation_report_sha256"],
        label="selected Stage4 validation report",
    )
    _validate_route_record_identity(
        payload["validation_route_records"],
        label="Stage4 validation route records",
    )
    _validate_route_record_identity(
        payload["gate_route_records"],
        label="Stage4 gate route records",
    )
    _require_file_sha(
        payload["gate_route_manifest"]["path"],
        payload["gate_route_manifest"]["sha256"],
        label="Stage4 gate route manifest",
    )
    if payload["router_integrity"].get("static_logits_exact") is not True:
        raise ValueError(
            "Stage4 dynamic selection lacks exact static parity"
        )
    if int(payload["candidate_union"]["final_k"]) > int(
        payload["candidate_union"]["static_k"]
    ):
        raise ValueError("Stage4 final_k must not exceed static_k")
    payload["manifest_sha256"] = canonical_digest(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload


def _require_identity(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    actual = sha256_path(Path(path).resolve())["sha256"]
    if actual != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 mismatch")


def _final_reliability_release_checks(
    dynamic: dict[str, Any],
    fused_summary: dict[str, Any],
) -> dict[str, bool]:
    baseline = dict(dynamic["stage2_baseline"])
    reference = float(baseline["static_macro_mrr"])
    baseline_by_benchmark = dict(baseline["by_benchmark"])
    fused_by_benchmark = dict(fused_summary["by_benchmark"])
    if set(fused_by_benchmark) != set(baseline_by_benchmark):
        raise ValueError("Stage4 reliability benchmark coverage mismatch")
    return {
        "safe_fused_static_nonregression": (
            float(fused_summary["balanced_macro_mrr"]) >= reference - 0.005
        ),
        "fused_per_benchmark_nonregression": all(
            float(fused_by_benchmark[benchmark]["mrr"])
            >= float(baseline_by_benchmark[benchmark]["static_mrr"]) - 0.01
            for benchmark in baseline_by_benchmark
        ),
        "zero_history_exact": fused_summary.get("zero_history_exact") is True,
    }


def finalize_stage4_selection(
    *,
    dynamic_selection_path: str | Path,
    output_path: str | Path,
    gate_report_path: str | Path | None = None,
) -> dict[str, Any]:
    dynamic = _read_self_hashed_manifest(
        dynamic_selection_path,
        STAGE4_DYNAMIC_SELECTION_SCHEMA,
    )
    if dynamic.get("status") != "ok":
        raise ValueError("dynamic Stage4 selection is not complete")
    if dynamic.get("release_status") not in {"ok", "action_required"}:
        raise ValueError("dynamic Stage4 selection has invalid release status")
    _require_identity(
        dynamic["selected_checkpoint_path"],
        dynamic["selected_checkpoint_sha256"],
        label="selected Stage4 checkpoint",
    )
    _require_identity(
        dynamic["selected_validation_report_path"],
        dynamic["selected_validation_report_sha256"],
        label="selected Stage4 validation report",
    )
    if dynamic.get("stage4_method") == "candidate_admission_residual_v1":
        if gate_report_path is not None:
            raise ValueError(
                "candidate admission is checkpoint-native and rejects a post-hoc gate"
            )
        base_cmc_sha = str(dynamic.get("base_cmc_checkpoint_sha256") or "")
        if len(base_cmc_sha) != 64:
            raise ValueError("candidate admission base CMC identity is missing")
        reliability = {
            "mode": "candidate_admission_residual",
            "fixed_alpha": None,
            "gate_checkpoint": None,
            "safe_memory_residual_bound": 2.0,
            "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
            "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
            "base_cmc_checkpoint_sha256": base_cmc_sha,
            "deployed_reliability_source": (
                "candidate_admission_constrained_residual"
            ),
        }
        reliability["reliability_sha256"] = canonical_digest(reliability)
        release_checks = dict(dynamic.get("release_checks") or {})
        promoted = (
            dynamic.get("release_status") == "ok"
            and dynamic.get("promotion_status") == "promoted"
            and bool(release_checks)
            and all(value is True for value in release_checks.values())
        )
        candidate_union = dict(dynamic["candidate_union"])
        payload = {
            "schema_version": "stage4_selection_v1",
            "status": "ok",
            "stage4_method": "candidate_admission_residual_v1",
            "release_status": "ok" if promoted else "action_required",
            "promotion_status": str(dynamic["promotion_status"]),
            "dynamic_release_status": str(dynamic["release_status"]),
            "release_checks": release_checks,
            "selected_step": int(dynamic["selected_step"]),
            "selected_checkpoint_path": str(
                Path(dynamic["selected_checkpoint_path"]).resolve()
            ),
            "selected_checkpoint_sha256": str(
                dynamic["selected_checkpoint_sha256"]
            ),
            "selected_validation_report_path": str(
                Path(dynamic["selected_validation_report_path"]).resolve()
            ),
            "selected_validation_report_sha256": str(
                dynamic["selected_validation_report_sha256"]
            ),
            "router_integrity": dict(dynamic["router_integrity"]),
            "stage2_baseline": dict(dynamic["stage2_baseline"]),
            "candidate_union": candidate_union,
            "candidate_admission_selection": dict(
                dynamic["candidate_admission_selection"]
            ),
            "gradient_health": dict(dynamic["gradient_health"]),
            "base_cmc_checkpoint_sha256": base_cmc_sha,
            "reliability": reliability,
            "reliability_validation": dict(
                dynamic["candidate_admission_selection"]
            ),
        }
        payload["manifest_sha256"] = canonical_digest(payload)
        _atomic_write_json(Path(output_path), payload)
        return payload
    if (
        dynamic.get("stage4_method")
        == "counterfactual_memory_calibration_v1"
    ):
        if gate_report_path is not None:
            raise ValueError(
                "CMC reliability is checkpoint-native and rejects a post-hoc gate"
            )
        reliability = {
            "mode": "cmc_candidate_gate",
            "fixed_alpha": None,
            "gate_checkpoint": None,
            "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
            "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
        }
        reliability["reliability_sha256"] = canonical_digest(reliability)
        release_checks = dict(dynamic.get("release_checks") or {})
        promoted = (
            dynamic.get("release_status") == "ok"
            and dynamic.get("promotion_status") == "promoted"
            and bool(release_checks)
            and all(value is True for value in release_checks.values())
        )
        candidate_union = dict(dynamic["candidate_union"])
        if int(candidate_union["final_k"]) > int(candidate_union["static_k"]):
            raise ValueError("Stage4 final_k must not exceed static_k")
        payload = {
            "schema_version": "stage4_selection_v1",
            "status": "ok",
            "stage4_method": "counterfactual_memory_calibration_v1",
            "release_status": "ok" if promoted else "action_required",
            "promotion_status": str(dynamic["promotion_status"]),
            "dynamic_release_status": str(dynamic["release_status"]),
            "release_checks": release_checks,
            "selected_step": int(dynamic["selected_step"]),
            "selected_checkpoint_path": str(
                Path(dynamic["selected_checkpoint_path"]).resolve()
            ),
            "selected_checkpoint_sha256": str(
                dynamic["selected_checkpoint_sha256"]
            ),
            "selected_validation_report_path": str(
                Path(dynamic["selected_validation_report_path"]).resolve()
            ),
            "selected_validation_report_sha256": str(
                dynamic["selected_validation_report_sha256"]
            ),
            "router_integrity": dict(dynamic["router_integrity"]),
            "stage2_baseline": dict(dynamic["stage2_baseline"]),
            "candidate_union": candidate_union,
            "cmc_fused_selection": dict(dynamic["cmc_fused_selection"]),
            "gradient_health": dict(dynamic["gradient_health"]),
            "reliability": reliability,
            "reliability_validation": dict(dynamic["cmc_fused_selection"]),
        }
        payload["manifest_sha256"] = canonical_digest(payload)
        _atomic_write_json(Path(output_path), payload)
        return payload
    reliability: dict[str, Any] = {
        "mode": "causal_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "safe_memory_residual_bound": float(
            dynamic["safe_fused_selection"]["residual_bound"]
        ),
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    reliability_validation = dict(dynamic["safe_fused_selection"])
    if gate_report_path is not None:
        raise ValueError(
            "post-hoc memory utility gate reports cannot override the jointly trained causal gate"
        )
    release_checks = _final_reliability_release_checks(
        dynamic,
        reliability_validation,
    )
    candidate_union = dict(dynamic["candidate_union"])
    if int(candidate_union["final_k"]) > int(candidate_union["static_k"]):
        raise ValueError("Stage4 final_k must not exceed static_k")
    payload = {
        "schema_version": "stage4_selection_v1",
        "status": "ok",
        "release_status": (
            "ok" if all(release_checks.values()) else "action_required"
        ),
        "dynamic_release_status": str(dynamic["release_status"]),
        "release_checks": release_checks,
        "selected_checkpoint_path": str(
            Path(dynamic["selected_checkpoint_path"]).resolve()
        ),
        "selected_checkpoint_sha256": str(
            dynamic["selected_checkpoint_sha256"]
        ),
        "selected_validation_report_path": str(
            Path(dynamic["selected_validation_report_path"]).resolve()
        ),
        "selected_validation_report_sha256": str(
            dynamic["selected_validation_report_sha256"]
        ),
        "router_integrity": dict(dynamic["router_integrity"]),
        "stage2_baseline": dict(dynamic["stage2_baseline"]),
        "candidate_union": candidate_union,
        "safe_fused_selection": dict(dynamic["safe_fused_selection"]),
        "reliability": reliability,
        "reliability_validation": reliability_validation,
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    _atomic_write_json(Path(output_path), payload)
    return payload
