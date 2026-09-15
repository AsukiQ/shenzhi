from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from clstr.action_adapter import UniversalActionAdapter
from clstr.full_base_train import (
    LOSS_WEIGHT_KEYS,
    TRANSITION_INVENTORY_MASK_MODES,
    TRANSITION_LOSS_TYPES,
    TRANSITION_POSITIVE_MODES,
    _apply_skill_text_format,
    _attach_auto_replay_prefixes,
    _attach_full_base_embedding_cache,
    _attach_policy_embedding_cache,
    _attach_stage0_topm_candidates,
    _build_loss_buckets,
    _cap_rows_by_benchmark,
    _compute_full_base_loss,
    _embedding_cache_policy,
    _equivalent_skill_ids_by_skill_id,
    _filter_rows_by_allowed_benchmarks,
    _filter_rows_by_train_split,
    _limit_rows_for_smoke,
    _normalize_loss_weights,
    _read_jsonl,
    _sample_full_base_batch,
    _select_stage0_handoff_training_subset,
    _skill_id,
    _skipped_embedding_cache_report,
)
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model


DEFAULT_STAGE2_V2_LOSS_WEIGHTS = {
    "L_policy": 0.2,
    "routing": 0.0,
    "STOP": 0.1,
    "L_trans": 0.3,
    "L_trans_skill_ce": 1.0,
    "belief": 0.1,
}

METRIC_WEIGHT_LOSSES = {
    "policy_ce_loss": "L_policy",
    "policy_expert_recall@1": "L_policy",
    "transition_cosine_loss": "L_trans",
    "transition_skill_ce_loss": "L_trans_skill_ce",
    "transition_skill_recall@1": "L_trans_skill_ce",
    "transition_skill_recall@5": "L_trans_skill_ce",
    "transition_skill_mrr": "L_trans_skill_ce",
    "transition_skill_ce_candidate_count": "L_trans_skill_ce",
    "transition_skill_ce_count": "L_trans_skill_ce",
    "transition_inventory_mask_applied_rows": "L_trans_skill_ce",
    "transition_inventory_mask_removed_candidates": "L_trans_skill_ce",
    "transition_inventory_mask_missing_rows": "L_trans_skill_ce",
    "transition_inventory_mask_positive_missing_rows": "L_trans_skill_ce",
    "transition_positive_mean_count": "L_trans_skill_ce",
    "transition_positive_max_count": "L_trans_skill_ce",
    "transition_multi_positive_rows": "L_trans_skill_ce",
    "transition_skill_self_count": "L_trans_skill_ce",
    "transition_skill_self_recall@1": "L_trans_skill_ce",
    "transition_skill_self_recall@5": "L_trans_skill_ce",
    "transition_skill_self_mrr": "L_trans_skill_ce",
    "transition_skill_switch_count": "L_trans_skill_ce",
    "transition_skill_switch_recall@1": "L_trans_skill_ce",
    "transition_skill_switch_recall@5": "L_trans_skill_ce",
    "transition_skill_switch_mrr": "L_trans_skill_ce",
    "belief_cosine_loss": "belief",
    "stop_bce_loss": "STOP",
    "retrieval_contrastive_loss": "routing",
}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _loss_mask_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, enabled in (row.get("loss_mask") or {}).items():
            if enabled:
                counts[str(key)] += 1
    return {key: int(counts.get(key, 0)) for key in LOSS_WEIGHT_KEYS}


def aggregate_eval_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_sums: Counter[str] = Counter()
    metric_weights: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    evaluated_rows = 0

    for row in metric_rows:
        batch_size = int(row.get("batch_size") or 0)
        evaluated_rows += batch_size
        loss_counts = {str(k): int(v) for k, v in (row.get("loss_mask_counts") or {}).items()}
        for key, value in loss_counts.items():
            total_counts[key] += int(value)
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            if key == "weighted_loss_terms" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            loss_key = METRIC_WEIGHT_LOSSES.get(str(key))
            weight = loss_counts.get(loss_key, 0) if loss_key else batch_size
            if weight <= 0:
                continue
            metric_sums[str(key)] += float(value) * float(weight)
            metric_weights[str(key)] += int(weight)

    return {
        "evaluated_batches": len(metric_rows),
        "evaluated_rows": evaluated_rows,
        "loss_mask_counts": {key: int(total_counts.get(key, 0)) for key in LOSS_WEIGHT_KEYS},
        "metrics": {
            key: float(metric_sums[key] / metric_weights[key])
            for key in sorted(metric_sums)
            if metric_weights[key] > 0
        },
        "metric_weights": {key: int(metric_weights[key]) for key in sorted(metric_weights)},
    }


def _select_real_topm_eval_rows(
    rows: list[dict[str, Any]],
    *,
    max_eval_batches: int | None,
    batch_size: int,
    loss_weights: dict[str, float],
    sample_multiplier: float | None,
    sampling_strategy: str,
    sampler_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_eval_batches is None:
        return rows, {
            "enabled": False,
            "source_rows": len(rows),
            "selected_rows": len(rows),
            "max_eval_batches": None,
            "batch_size": int(batch_size),
            "sample_multiplier": sample_multiplier,
            "sampling_strategy": str(sampling_strategy),
            "reason": "full_eval_keeps_all_rows",
        }
    return _select_stage0_handoff_training_subset(
        rows,
        max_steps=max(1, int(max_eval_batches)),
        batch_size=max(1, int(batch_size)),
        loss_weights=loss_weights,
        sample_multiplier=sample_multiplier,
        sampling_strategy=sampling_strategy,
        sampler_seed=int(sampler_seed),
    )


def build_real_topm_report(
    *,
    output_dir: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    handoff_report: dict[str, Any],
    aggregate_report: dict[str, Any],
    min_current_coverage: float = 0.85,
    min_next_coverage: float = 0.85,
    min_transition_recall_at_5: float = 0.5,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    policy = str(handoff_report.get("positive_missing_policy") or "")
    injected_rows = int(handoff_report.get("injected_positive_rows") or 0)
    retained_rows = int(handoff_report.get("retained_rows") or 0)
    current_coverage = handoff_report.get("current_positive_coverage@M")
    next_coverage = handoff_report.get("next_positive_coverage@M")
    metrics = aggregate_report.get("metrics") or {}
    metric_weights = aggregate_report.get("metric_weights") or {}

    if policy != "skip":
        blockers.append("not_real_topm_no_inject_policy")
    if injected_rows > 0:
        blockers.append("gold_positive_injected")
    if retained_rows <= 0:
        blockers.append("no_retained_real_topm_rows")
    if current_coverage is not None and float(current_coverage) < float(min_current_coverage):
        blockers.append("current_positive_coverage_below_threshold")
    if next_coverage is not None and float(next_coverage) < float(min_next_coverage):
        blockers.append("next_positive_coverage_below_threshold")

    recall_weight = int(metric_weights.get("transition_skill_recall@5") or 0)
    recall_at_5 = metrics.get("transition_skill_recall@5")
    if recall_weight <= 0:
        blockers.append("no_transition_skill_eval_rows")
    elif recall_at_5 is not None and float(recall_at_5) < float(min_transition_recall_at_5):
        blockers.append("transition_recall_at_5_below_threshold")

    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "output_dir": str(output_dir),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "stage0_candidate_handoff": handoff_report,
        "aggregate": aggregate_report,
        "thresholds": {
            "min_current_coverage": float(min_current_coverage),
            "min_next_coverage": float(min_next_coverage),
            "min_transition_recall_at_5": float(min_transition_recall_at_5),
        },
        "paper_scope_note": (
            "This is an offline no-inject Stage2 diagnostic over Stage0 top-M candidates. "
            "It is not a closed-loop benchmark and does not use gold candidate injection."
        ),
        **(extra or {}),
    }


def evaluate_stage2_real_topm(
    *,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    top_m: int = 350,
    batch_size: int = 8,
    max_rows: int | None = None,
    max_eval_batches: int | None = None,
    seed: int = 17,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = None,
    sampling_strategy: str = "balanced_random",
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    skill_text_format: str | None = None,
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 100,
    min_current_coverage: float = 0.85,
    min_next_coverage: float = 0.85,
    min_transition_recall_at_5: float = 0.5,
    transition_inventory_mask_mode: str = "off",
    transition_loss_type: str = "cross_entropy",
    transition_positive_mode: str = "single",
    auto_replay_prefix_max_steps: int = 3,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage0_checkpoint_path = Path(stage0_checkpoint_path)
    stage2_checkpoint_path = Path(stage2_checkpoint_path)
    train_path = Path(train_path)
    skills_path = Path(skills_path)
    transition_inventory_mask_mode = str(transition_inventory_mask_mode or "off")
    transition_loss_type = str(transition_loss_type or "cross_entropy")
    transition_positive_mode = str(transition_positive_mode or "single")
    if transition_inventory_mask_mode not in TRANSITION_INVENTORY_MASK_MODES:
        raise ValueError(f"unsupported transition inventory mask mode: {transition_inventory_mask_mode}")
    if transition_loss_type not in TRANSITION_LOSS_TYPES:
        raise ValueError(f"unsupported transition loss type: {transition_loss_type}")
    if transition_positive_mode not in TRANSITION_POSITIVE_MODES:
        raise ValueError(f"unsupported transition positive mode: {transition_positive_mode}")

    torch.manual_seed(int(seed))
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=stage0_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        stage2_checkpoint_path,
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    rows = _read_jsonl(train_path)
    raw_row_count = len(rows)
    rows, train_split_filter_report = _filter_rows_by_train_split(rows)
    rows, benchmark_filter_report = _filter_rows_by_allowed_benchmarks(rows, allowed_benchmarks)
    rows, benchmark_caps_report = _cap_rows_by_benchmark(rows, benchmark_caps)
    rows, max_rows_filter_report = _limit_rows_for_smoke(rows, max_rows)
    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    equivalent_skill_ids_by_skill_id = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    skill_text_format_report = _apply_skill_text_format(model, model_config, skill_text_format)
    loss_weights = _normalize_loss_weights(DEFAULT_STAGE2_V2_LOSS_WEIGHTS)
    rows, subset_report = _select_real_topm_eval_rows(
        rows,
        max_eval_batches=max_eval_batches,
        batch_size=max(1, int(batch_size)),
        loss_weights=loss_weights,
        sample_multiplier=stage0_handoff_sample_multiplier,
        sampling_strategy=sampling_strategy,
        sampler_seed=int(seed),
    )
    setup_status_path = output_dir / "setup_status.jsonl"
    if setup_status_path.exists():
        setup_status_path.unlink()
    rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=stage0_checkpoint_path,
        manifest_path=output_dir / "stage0_candidate_handoff_no_inject.json",
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        setup_status_path=setup_status_path,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
    )
    rows, auto_replay_prefix_report = _attach_auto_replay_prefixes(
        rows,
        max_steps=auto_replay_prefix_max_steps,
    )

    qwen_external_encoder = bool(
        getattr(model, "qwen_external_metadata", {})
        and getattr(model, "qwen_external_metadata", {}).get("qwen_external_encoder")
    )
    cache_policy = _embedding_cache_policy(
        mode=embedding_cache_mode,
        row_count=len(rows),
        qwen_external_encoder=qwen_external_encoder,
        max_rows=embedding_cache_max_rows,
    )
    embedding_cache_batch_size = 8 if qwen_external_encoder else 256
    if rows and cache_policy["cache_enabled"]:
        full_base_embedding_cache_report = _attach_full_base_embedding_cache(
            model,
            rows,
            encode_batch_size=embedding_cache_batch_size,
        )
        policy_embedding_cache_report = _attach_policy_embedding_cache(
            model,
            rows,
            encode_batch_size=embedding_cache_batch_size,
        )
    else:
        full_base_embedding_cache_report = _skipped_embedding_cache_report("full_base_replay", rows, cache_policy)
        policy_embedding_cache_report = _skipped_embedding_cache_report("policy_candidates", rows, cache_policy)

    action_adapter = UniversalActionAdapter(int(model_config.get("d", 128)), hidden_dim=int(model_config.get("d", 128))).to(device)
    if hasattr(model, "eval"):
        model.eval()
    action_adapter.eval()
    batch_size = max(1, int(batch_size))
    metric_rows: list[dict[str, Any]] = []
    max_batches = None if max_eval_batches is None else max(1, int(max_eval_batches))
    loss_buckets = _build_loss_buckets(rows, loss_weights)
    with torch.no_grad():
        if max_batches is None:
            batch_iter = (
                rows[start : start + batch_size]
                for start in range(0, len(rows), batch_size)
            )
        else:
            batch_iter = (
                _sample_full_base_batch(
                    rows,
                    step_idx=step_idx,
                    batch_size=batch_size,
                    loss_weights=loss_weights,
                    loss_buckets=loss_buckets,
                    sampling_strategy=sampling_strategy,
                    sampler_seed=int(seed),
                )
                for step_idx in range(1, max_batches + 1)
            )
        for batch_idx, batch in enumerate(batch_iter, start=1):
            if not batch:
                continue
            _loss, metrics = _compute_full_base_loss(
                model,
                action_adapter,
                batch,
                skill_id_to_idx,
                device,
                loss_weights=loss_weights,
                transition_inventory_mask_mode=transition_inventory_mask_mode,
                transition_loss_type=transition_loss_type,
                transition_positive_mode=transition_positive_mode,
                equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                trainable_replay_prefix=False,
            )
            metric_rows.append(
                {
                    "batch_index": batch_idx,
                    "batch_size": len(batch),
                    "loss_mask_counts": _loss_mask_counts(batch),
                    "metrics": metrics,
                }
            )

    metrics_path = output_dir / "real_topm_eval_metrics.jsonl"
    _write_jsonl(metrics_path, metric_rows)
    aggregate = aggregate_eval_metrics(metric_rows)
    report = build_real_topm_report(
        output_dir=output_dir,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        train_path=train_path,
        skills_path=skills_path,
        handoff_report=handoff_report,
        aggregate_report=aggregate,
        min_current_coverage=min_current_coverage,
        min_next_coverage=min_next_coverage,
        min_transition_recall_at_5=min_transition_recall_at_5,
        extra={
            "metrics_path": str(metrics_path),
            "config": {
                "top_m": int(top_m),
                "batch_size": int(batch_size),
                "max_rows": max_rows,
                "max_eval_batches": max_eval_batches,
                "seed": int(seed),
                "stage0_handoff_query_mode": str(stage0_handoff_query_mode),
                "stage0_handoff_sample_multiplier": stage0_handoff_sample_multiplier,
                "sampling_strategy": str(sampling_strategy),
                "embedding_cache_mode": str(embedding_cache_mode),
                "embedding_cache_max_rows": int(embedding_cache_max_rows),
                "transition_inventory_mask_mode": transition_inventory_mask_mode,
                "transition_loss_type": transition_loss_type,
                "transition_positive_mode": transition_positive_mode,
                "auto_replay_prefix_max_steps": int(auto_replay_prefix_max_steps),
            },
            "auto_replay_prefix": auto_replay_prefix_report,
            "data_filters": {
                "raw_row_count": raw_row_count,
                "train_split_filter": train_split_filter_report,
                "benchmark_filter": benchmark_filter_report,
                "benchmark_caps": benchmark_caps_report,
                "max_rows_filter": max_rows_filter_report,
                "stage0_handoff_subset": subset_report,
            },
            "model_load": {
                "routing_init": routing_report,
                "stage2_load": stage2_load_report,
                "skill_text_format": skill_text_format_report,
            },
            "embedding_cache": {
                "policy": cache_policy,
                "full_base": full_base_embedding_cache_report,
                "policy_candidates": policy_embedding_cache_report,
            },
        },
    )
    _write_json(output_dir / "real_topm_eval_report.json", report)
    return report
