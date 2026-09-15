from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import torch

from clstr.full_base_train import (
    _compute_full_base_loss,
    _equivalent_skill_ids_by_skill_id,
    _normalize_loss_weights,
    _sample_full_base_batch,
    _build_loss_buckets,
)


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_metrics(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    keys = [
        "transition_skill_recall@1",
        "transition_skill_recall@5",
        "transition_skill_mrr",
        "transition_prior_skill_recall@1",
        "transition_prior_skill_recall@5",
        "transition_residual_skill_recall@1",
        "transition_residual_skill_recall@5",
        "transition_skill_self_recall@1",
        "transition_skill_self_recall@5",
        "transition_skill_switch_recall@1",
        "transition_skill_switch_recall@5",
        "stage0_prior_transition_skill_recall@1",
        "stage0_prior_transition_skill_recall@5",
        "stage0_prior_transition_skill_mrr",
        "transition_delta_vs_stage0_prior_recall@1",
        "transition_delta_vs_stage0_prior_recall@5",
        "transition_delta_vs_stage0_prior_mrr",
        "transition_worse_than_stage0_prior_fraction",
        "transition_skill_ce_loss",
        "transition_skill_ce_candidate_count",
        "transition_positive_mean_count",
        "transition_positive_max_count",
        "transition_multi_positive_rows",
    ]
    output: dict[str, float] = {}
    for key in keys:
        value = _mean_metric(rows, key)
        if value is not None:
            output[f"{prefix}{key}"] = value
    return output


def evaluate_transition_lambda_sweep(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    lambdas: Iterable[float],
    batch_size: int = 4,
    max_eval_batches: int = 20,
    seed: int = 17,
    sampling_strategy: str = "balanced_random",
    transition_inventory_mask_mode: str = "auto",
    transition_inventory_min_candidates: int = 64,
    transition_loss_type: str = "listwise_nll",
    transition_positive_mode: str = "gold_plus_equivalent",
    transition_scoring_mode: str = "stage0_rank_prior_plus_transition_residual",
    device: torch.device | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("lambda sweep requires at least one training row")
    lambda_values = [float(value) for value in lambdas]
    if not lambda_values:
        raise ValueError("lambda sweep requires at least one lambda")

    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skills) if row.get("skill_id")}
    if not skill_id_to_idx:
        raise ValueError("lambda sweep requires non-empty skill_id mapping")
    equivalent_skill_ids_by_skill_id = _equivalent_skill_ids_by_skill_id(skills)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    loss_weights = _normalize_loss_weights(
        {
            "L_policy": 0.0,
            "hard_negative_margin": 0.0,
            "Q_success": 0.0,
            "routing": 0.0,
            "STOP": 0.0,
            "L_trans": 0.0,
            "L_trans_skill_ce": 1.0,
            "transition_hard_negative_margin": 0.0,
            "belief": 0.0,
        }
    )
    loss_buckets = _build_loss_buckets(rows, loss_weights)
    transition_rows = loss_buckets.get("L_trans_skill_ce") or []
    if not transition_rows:
        raise ValueError("lambda sweep requires rows with L_trans_skill_ce enabled")

    max_eval_batches = max(1, int(max_eval_batches))
    batch_size = max(1, int(batch_size))
    lambda_metrics: dict[str, list[dict[str, Any]]] = {str(value): [] for value in lambda_values}
    sampled_rows = 0
    sampled_benchmarks: Counter[str] = Counter()
    sampled_relations: Counter[str] = Counter()

    with torch.no_grad():
        for step_idx in range(1, max_eval_batches + 1):
            batch = _sample_full_base_batch(
                rows,
                step_idx,
                batch_size,
                loss_weights,
                loss_buckets=loss_buckets,
                sampling_strategy=sampling_strategy,
                sampler_seed=seed,
            )
            batch = [row for row in batch if (row.get("loss_mask") or {}).get("L_trans_skill_ce")]
            if not batch:
                continue
            sampled_rows += len(batch)
            sampled_benchmarks.update(str(row.get("benchmark") or "unknown") for row in batch)
            sampled_relations.update(
                "self" if str(row.get("skill_id") or "") == str(row.get("next_skill_id") or "") else "switch"
                for row in batch
            )
            for lambda_value in lambda_values:
                _loss, metrics = _compute_full_base_loss(
                    model=model,
                    action_adapter=object(),
                    batch=batch,
                    skill_id_to_idx=skill_id_to_idx,
                    device=device,
                    loss_weights=loss_weights,
                    transition_inventory_mask_mode=transition_inventory_mask_mode,
                    transition_inventory_min_candidates=transition_inventory_min_candidates,
                    transition_loss_type=transition_loss_type,
                    transition_positive_mode=transition_positive_mode,
                    transition_scoring_mode=transition_scoring_mode,
                    equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                    transition_residual_lambda=lambda_value,
                )
                lambda_metrics[str(lambda_value)].append(metrics)

    reports: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        key = str(lambda_value)
        summary = _summarize_metrics(lambda_metrics[key])
        reports.append(
            {
                "lambda": lambda_value,
                "metric_batch_count": len(lambda_metrics[key]),
                **summary,
            }
        )
    best_by_recall5 = max(reports, key=lambda row: float(row.get("transition_skill_recall@5", -1.0)))
    best_by_recall1 = max(reports, key=lambda row: float(row.get("transition_skill_recall@1", -1.0)))
    return {
        "status": "ok",
        "method": "offline_transition_residual_lambda_sweep",
        "interpretation_scope": (
            "diagnostic_only: reweights prior/residual logits in an already trained checkpoint; "
            "does not estimate the result of retraining with that lambda"
        ),
        "device": str(device),
        "batch_size": batch_size,
        "max_eval_batches": max_eval_batches,
        "sampled_transition_rows": sampled_rows,
        "sampled_benchmark_counts": dict(sorted(sampled_benchmarks.items())),
        "sampled_relation_counts": dict(sorted(sampled_relations.items())),
        "transition_inventory_mask_mode": transition_inventory_mask_mode,
        "transition_inventory_min_candidates": int(transition_inventory_min_candidates),
        "transition_loss_type": transition_loss_type,
        "transition_positive_mode": transition_positive_mode,
        "transition_scoring_mode": str(transition_scoring_mode),
        "lambda_reports": reports,
        "best_lambda_by_transition_recall@5": best_by_recall5["lambda"],
        "best_lambda_by_transition_recall@1": best_by_recall1["lambda"],
    }
