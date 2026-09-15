from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from clstr.stage_quality_common import (
    EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    EXPECTED_TRANSITION_SCORING_MODE,
    artifact_exists as _path_exists,
    bounded_window_size,
    max_counter,
    read_json_file as _read_json,
    read_jsonl_file as _read_jsonl,
    transition_scoring_safety as _transition_scoring_safety,
    window_mean as _window_mean,
    write_json_report,
)


DEFAULT_REQUIRED_LOSS_TERMS = ["L_policy", "L_trans", "L_trans_skill_ce", "belief"]
LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER = "legacy_prior_residual"
UNIFIED_MEMORY_ROUTE_SCORER = "unified_memory"


def _max_step(rows: list[dict[str, Any]], train_report: dict[str, Any]) -> int:
    return max_counter(rows, train_report, row_keys=("step", "update"), report_keys=("max_steps", "step"))


def _missing_required_loss_terms(train_report: dict[str, Any], required_terms: list[str]) -> list[str]:
    activation_counts = train_report.get("loss_activation_counts")
    sampled_counts = train_report.get("sampled_loss_activation_counts")
    activation_counts = activation_counts if isinstance(activation_counts, dict) else {}
    sampled_counts = sampled_counts if isinstance(sampled_counts, dict) else {}
    missing: list[str] = []
    for term in required_terms:
        activated = int(activation_counts.get(term) or 0) > 0
        sampled = int(sampled_counts.get(term) or 0) > 0
        if not activated or not sampled:
            missing.append(term)
    return missing


def _route_scorer_safety(
    train_report: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_route_scorer: str | None = None,
) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []

    def add(source: str, value: Any) -> None:
        if value is None:
            return
        text = str(value or "")
        if text:
            observed.append({"source": source, "route_scorer": text})

    add("train_report", train_report.get("route_scorer"))
    transition_candidate_training = train_report.get("transition_candidate_training")
    if isinstance(transition_candidate_training, dict):
        add("transition_candidate_training", transition_candidate_training.get("route_scorer"))
    transition_skill_ce = train_report.get("transition_skill_ce")
    if isinstance(transition_skill_ce, dict):
        add("transition_skill_ce", transition_skill_ce.get("route_scorer"))
    for idx, row in enumerate(rows[:20], start=1):
        add(f"metric_row_{idx}", row.get("route_scorer"))

    observed_values = sorted({item["route_scorer"] for item in observed})
    expected = str(expected_route_scorer or "")
    if not expected:
        if observed_values:
            expected = observed_values[0] if len(observed_values) == 1 else ""
        else:
            expected = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER

    matched_sources = [
        item["source"]
        for item in observed
        if item["route_scorer"] == expected
    ]
    route_matched = bool(matched_sources) if observed else expected == LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER

    uses_stage0_prior = train_report.get("uses_stage0_prior_at_inference")
    if expected == UNIFIED_MEMORY_ROUTE_SCORER:
        stage0_prior_ok = uses_stage0_prior is False
    else:
        stage0_prior_ok = True

    head_types = sorted({
        str(row.get("transition_skill_head_type") or "")
        for row in rows
        if row.get("transition_skill_head_type")
    })
    if expected == UNIFIED_MEMORY_ROUTE_SCORER:
        head_type_ok = not head_types or set(head_types) <= {
            "unified_memory_retriever",
            "unified_memory_retriever_full_pool",
        }
    else:
        head_type_ok = True

    return {
        "expected": expected,
        "observed": observed,
        "observed_values": observed_values,
        "matched_sources": matched_sources,
        "matched": bool(route_matched and stage0_prior_ok and head_type_ok),
        "uses_stage0_prior_at_inference": uses_stage0_prior,
        "stage0_prior_ok": bool(stage0_prior_ok),
        "transition_skill_head_types": head_types,
        "transition_skill_head_type_ok": bool(head_type_ok),
    }


def _next_skill_pool_safety(
    train_report: dict[str, Any],
    expected_next_skill_pool_mode: str | None,
) -> dict[str, Any]:
    candidate_training = train_report.get("transition_candidate_training")
    candidate_training = candidate_training if isinstance(candidate_training, dict) else {}
    transition_skill_ce = train_report.get("transition_skill_ce")
    transition_skill_ce = transition_skill_ce if isinstance(transition_skill_ce, dict) else {}
    expected = str(expected_next_skill_pool_mode or "")
    observed = str(candidate_training.get("next_skill_pool_mode") or "")
    inventory_mask_mode = str(candidate_training.get("inventory_mask_mode") or "")
    candidate_pool = str(transition_skill_ce.get("candidate_pool") or "")
    transition_objective = str(train_report.get("transition_objective") or "")
    memory_utility_calibration_owner = str(
        candidate_training.get("memory_utility_calibration_owner") or ""
    )
    raw_counterfactual_weight = candidate_training.get("counterfactual_utility_loss_weight")
    try:
        counterfactual_weight = float(raw_counterfactual_weight)
    except (TypeError, ValueError):
        counterfactual_weight = float("nan")
    counterfactual_enabled = math.isfinite(counterfactual_weight) and counterfactual_weight > 0.0
    legacy_counterfactual_contract = (
        transition_objective == "full_pool_causal_next_skill_with_counterfactual_utility"
        and counterfactual_enabled
    )
    stage4_cmc_contract = (
        transition_objective == "full_pool_causal_next_skill_nll"
        and not counterfactual_enabled
        and memory_utility_calibration_owner == "stage4_cmc"
    )
    pool_mode_matched = not expected or observed == expected
    full_pool_expected = expected == "full_pool"
    full_pool_contract_matched = (
        not full_pool_expected
        or (
            inventory_mask_mode == "explicit_only"
            and candidate_pool == "declared_legal_full_skill_pool"
            and (legacy_counterfactual_contract or stage4_cmc_contract)
        )
    )
    return {
        "expected": expected or None,
        "observed": observed or None,
        "matched": bool(pool_mode_matched and full_pool_contract_matched),
        "pool_mode_matched": bool(pool_mode_matched),
        "full_pool_expected": bool(full_pool_expected),
        "inventory_mask_mode": inventory_mask_mode or None,
        "candidate_pool": candidate_pool or None,
        "transition_objective": transition_objective or None,
        "counterfactual_utility_loss_weight": (
            counterfactual_weight if math.isfinite(counterfactual_weight) else None
        ),
        "counterfactual_utility_enabled": bool(counterfactual_enabled),
        "memory_utility_calibration_owner": memory_utility_calibration_owner or None,
        "legacy_counterfactual_contract": bool(legacy_counterfactual_contract),
        "stage4_cmc_contract": bool(stage4_cmc_contract),
    }


def _safe_memory_training_safety(train_report: dict[str, Any]) -> dict[str, Any]:
    trainable_modules = {
        str(item) for item in (train_report.get("trainable_modules") or [])
    }
    candidate_training = train_report.get("transition_candidate_training")
    candidate_training = candidate_training if isinstance(candidate_training, dict) else {}
    objective = str(candidate_training.get("counterfactual_training_objective") or "")
    raw_bound = candidate_training.get("safe_memory_residual_bound")
    try:
        residual_bound = float(raw_bound)
    except (TypeError, ValueError):
        residual_bound = float("nan")
    raw_sizes = candidate_training.get("safe_local_candidate_sizes")
    sizes = [int(item) for item in raw_sizes] if isinstance(raw_sizes, list) else []
    blockers: list[str] = []
    if "route_memory_utility_gate" not in trainable_modules:
        blockers.append("route_memory_utility_gate_not_trainable")
    if {"initial_belief_head", "unified_retriever"}.intersection(trainable_modules):
        blockers.append("static_routing_foundation_trainable")
    if objective != "safe_local_candidate_rank_v1":
        blockers.append("safe_local_objective_mismatch")
    if not math.isfinite(residual_bound) or residual_bound <= 0.0:
        blockers.append("safe_memory_residual_bound_invalid")
    if not sizes or any(size < 2 for size in sizes):
        blockers.append("safe_local_candidate_sizes_invalid")
    return {
        "matched": not blockers,
        "blockers": blockers,
        "trainable_modules": sorted(trainable_modules),
        "counterfactual_training_objective": objective or None,
        "safe_memory_residual_bound": residual_bound if math.isfinite(residual_bound) else None,
        "safe_local_candidate_sizes": sizes,
    }


def _anchored_routing_foundation_safety(
    train_report: dict[str, Any],
    *,
    max_static_regression: float = 0.005,
) -> dict[str, Any]:
    trainable_modules = {
        str(item) for item in (train_report.get("trainable_modules") or [])
    }
    required_modules = {
        "transition",
        "gate",
        "action_proj",
        "initial_belief_head",
        "unified_retriever",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    }
    forbidden_modules = {"route_memory_utility_gate"}
    anchor = train_report.get("stage2_static_route_anchor")
    anchor = anchor if isinstance(anchor, dict) else {}
    blockers: list[str] = []
    if not required_modules.issubset(trainable_modules) or bool(
        forbidden_modules.intersection(trainable_modules)
    ):
        blockers.append("anchored_router_trainable_scope_mismatch")
    if train_report.get("qwen_and_skill_table_frozen") is not True:
        blockers.append("qwen_or_skill_table_geometry_not_frozen")
    if not bool(anchor.get("enabled")):
        blockers.append("stage2_static_route_anchor_disabled")
    try:
        weight = float(anchor.get("weight"))
    except (TypeError, ValueError):
        weight = float("nan")
    if not math.isfinite(weight) or weight <= 0.0:
        blockers.append("stage2_static_route_anchor_weight_invalid")
    if not str(anchor.get("teacher_digest") or ""):
        blockers.append("stage2_static_route_teacher_digest_missing")
    try:
        row_count = int(anchor.get("row_count") or 0)
    except (TypeError, ValueError):
        row_count = 0
    if row_count <= 0:
        blockers.append("stage2_static_route_anchor_rows_missing")
    try:
        static_delta = float(anchor.get("static_mrr_delta_vs_teacher"))
    except (TypeError, ValueError):
        static_delta = float("nan")
    if not math.isfinite(static_delta) or static_delta < -abs(
        float(max_static_regression)
    ):
        blockers.append("stage2_static_route_anchor_regression")
    try:
        dynamic_delta = float(anchor.get("dynamic_mrr_delta_vs_step_zero"))
    except (TypeError, ValueError):
        dynamic_delta = float("nan")
    if not math.isfinite(dynamic_delta) or dynamic_delta <= 0.0:
        blockers.append("stage2_dynamic_route_not_improved")
    return {
        "matched": not blockers,
        "blockers": blockers,
        "trainable_modules": sorted(trainable_modules),
        "required_trainable_modules": sorted(required_modules),
        "forbidden_trainable_modules": sorted(forbidden_modules),
        "weight": weight if math.isfinite(weight) else None,
        "teacher_digest": str(anchor.get("teacher_digest") or "") or None,
        "row_count": row_count,
        "static_mrr_delta_vs_teacher": (
            static_delta if math.isfinite(static_delta) else None
        ),
        "dynamic_mrr_delta_vs_step_zero": (
            dynamic_delta if math.isfinite(dynamic_delta) else None
        ),
        "max_static_regression": abs(float(max_static_regression)),
    }


def audit_stage2_full_base_quality(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    train_report_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    min_steps: int = 10000,
    first_window: int = 200,
    last_window: int = 200,
    min_loss_drop: float = 0.02,
    required_loss_terms: list[str] | None = None,
    max_transition_recall_at_5_drop: float = 0.1,
    max_transition_skill_ce_increase: float = 0.2,
    min_transition_recall_at_5_tail: float = 0.5,
    max_stage2_recall_at_5_drop_vs_stage0_prior: float = 0.01,
    max_stage2_mrr_drop_vs_stage0_prior: float = 0.01,
    max_stage2_worse_than_stage0_prior_fraction: float = 0.5,
    expected_transition_scoring_mode: str = EXPECTED_TRANSITION_SCORING_MODE,
    expected_transition_residual_lambda: float | None = EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    expected_route_scorer: str | None = None,
    expected_next_skill_pool_mode: str | None = None,
    static_route_anchor_max_regression: float = 0.005,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_report_path = Path(train_report_path) if train_report_path is not None else output_dir / "train_report.json"
    train_report = _read_json(train_report_path)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(train_report.get("checkpoint") or output_dir / "checkpoints" / f"clstr_full_base-step{min_steps}.pt")
    )
    metrics_path = Path(metrics_path) if metrics_path is not None else Path(train_report.get("training_metrics_path") or output_dir / "training_metrics.jsonl")
    required_loss_terms = list(required_loss_terms or DEFAULT_REQUIRED_LOSS_TERMS)
    blockers: list[str] = []

    if not train_report_path.is_file():
        blockers.append("missing_train_report")
    if train_report and train_report.get("status") != "ok":
        blockers.append("stage2_train_report_not_ok")
    if train_report and train_report.get("training_objective") != "component_complete_masked_multi_loss":
        blockers.append("unexpected_training_objective")
    if train_report and train_report.get("training_regime") != "offline_replay_supervised_pretraining":
        blockers.append("unexpected_training_regime")
    if train_report and train_report.get("valid_or_test_used_for_training") is True:
        blockers.append("valid_or_test_used_for_training")
    if train_report and train_report.get("on_policy_rollout_used") is True:
        blockers.append("stage2_should_not_use_on_policy_rollout")
    anchor_report = train_report.get("stage2_static_route_anchor")
    anchor_report = anchor_report if isinstance(anchor_report, dict) else {}
    anchored_routing_foundation = bool(anchor_report.get("enabled"))
    if train_report and anchored_routing_foundation:
        if train_report.get("frozen_routing_foundation") is not False:
            blockers.append("anchored_routing_foundation_report_mismatch")
    elif train_report and train_report.get("frozen_routing_foundation") is not True:
        blockers.append("routing_foundation_not_frozen")
    transition_skill_ce = train_report.get("transition_skill_ce")
    if train_report and (not isinstance(transition_skill_ce, dict) or transition_skill_ce.get("enabled") is not True):
        blockers.append("transition_skill_ce_not_enabled")

    if not _path_exists(checkpoint_path):
        blockers.append("missing_stage2_checkpoint")
    rows = _read_jsonl(metrics_path)
    if not metrics_path.is_file():
        blockers.append("missing_training_metrics")
    transition_scoring_safety = _transition_scoring_safety(
        train_report,
        rows,
        expected_mode=str(expected_transition_scoring_mode or EXPECTED_TRANSITION_SCORING_MODE),
        expected_residual_lambda=expected_transition_residual_lambda,
    )
    route_scorer_safety = _route_scorer_safety(train_report, rows, expected_route_scorer=expected_route_scorer)
    next_skill_pool_safety = _next_skill_pool_safety(train_report, expected_next_skill_pool_mode)
    safe_memory_training_safety = _safe_memory_training_safety(train_report)
    anchored_routing_foundation_safety = _anchored_routing_foundation_safety(
        train_report,
        max_static_regression=static_route_anchor_max_regression,
    )
    unified_route_scorer = route_scorer_safety.get("expected") == UNIFIED_MEMORY_ROUTE_SCORER
    if train_report and route_scorer_safety["matched"] is not True:
        blockers.append("stage2_route_scorer_mismatch")
    if train_report and expected_next_skill_pool_mode and next_skill_pool_safety["pool_mode_matched"] is not True:
        blockers.append("stage2_next_skill_pool_mismatch")
    if train_report and expected_next_skill_pool_mode == "full_pool":
        if next_skill_pool_safety["inventory_mask_mode"] != "explicit_only":
            blockers.append("stage2_full_pool_inventory_not_explicit_only")
        if next_skill_pool_safety["candidate_pool"] != "declared_legal_full_skill_pool":
            blockers.append("stage2_full_pool_candidate_pool_mismatch")
        if next_skill_pool_safety["matched"] is not True:
            blockers.append("stage2_full_pool_objective_mismatch")
        if next_skill_pool_safety["legacy_counterfactual_contract"]:
            if unified_route_scorer and safe_memory_training_safety["matched"] is not True:
                blockers.extend(safe_memory_training_safety["blockers"])
        elif next_skill_pool_safety["stage4_cmc_contract"]:
            if unified_route_scorer and anchored_routing_foundation_safety["matched"] is not True:
                blockers.extend(anchored_routing_foundation_safety["blockers"])
    if train_report and not unified_route_scorer and transition_scoring_safety["matched"] is not True:
        blockers.append(
            "stage2_transition_scoring_not_prior_residual"
            if str(expected_transition_scoring_mode) == EXPECTED_TRANSITION_SCORING_MODE
            else "stage2_transition_scoring_mismatch"
        )
    max_step = _max_step(rows, train_report)
    if max_step < int(min_steps):
        blockers.append("insufficient_training_steps")

    first_window = bounded_window_size(first_window, len(rows))
    last_window = bounded_window_size(last_window, len(rows))
    first_loss = _window_mean(rows, "loss", first_window, tail=False)
    last_loss = _window_mean(rows, "loss", last_window, tail=True)
    loss_drop = None if first_loss is None or last_loss is None else round(float(first_loss - last_loss), 12)
    if loss_drop is None or loss_drop < float(min_loss_drop):
        blockers.append("insufficient_learning_signal")

    transition_recall_first = _window_mean(rows, "transition_skill_recall@5", first_window, tail=False)
    transition_recall_last = _window_mean(rows, "transition_skill_recall@5", last_window, tail=True)
    transition_recall_drop = (
        None
        if transition_recall_first is None or transition_recall_last is None
        else round(float(transition_recall_first - transition_recall_last), 12)
    )
    transition_ce_first = _window_mean(rows, "transition_skill_ce_loss", first_window, tail=False)
    transition_ce_last = _window_mean(rows, "transition_skill_ce_loss", last_window, tail=True)
    transition_ce_increase = (
        None
        if transition_ce_first is None or transition_ce_last is None
        else round(float(transition_ce_last - transition_ce_first), 12)
    )
    if transition_recall_drop is None or transition_ce_increase is None:
        blockers.append("missing_transition_quality_metrics")
    else:
        if transition_recall_drop > float(max_transition_recall_at_5_drop):
            blockers.append("transition_recall_at_5_regressed")
        if transition_ce_increase > float(max_transition_skill_ce_increase):
            blockers.append("transition_skill_ce_regressed")
        if transition_recall_last is not None and transition_recall_last < float(min_transition_recall_at_5_tail):
            blockers.append("transition_recall_at_5_too_low")

    stage0_prior_recall_last = _window_mean(rows, "transition_prior_skill_recall@5", last_window, tail=True)
    stage2_mrr_last = _window_mean(rows, "transition_skill_mrr", last_window, tail=True)
    stage0_prior_mrr_last = _window_mean(rows, "transition_prior_skill_mrr", last_window, tail=True)
    worse_than_stage0_prior_last = _window_mean(
        rows,
        "transition_worse_than_stage0_prior_fraction",
        last_window,
        tail=True,
    )
    recall_drop_vs_stage0_prior = (
        None
        if transition_recall_last is None or stage0_prior_recall_last is None
        else round(float(stage0_prior_recall_last - transition_recall_last), 12)
    )
    mrr_drop_vs_stage0_prior = (
        None
        if stage2_mrr_last is None or stage0_prior_mrr_last is None
        else round(float(stage0_prior_mrr_last - stage2_mrr_last), 12)
    )
    requires_stage0_prior_gate = (
        not unified_route_scorer
        and str(expected_transition_scoring_mode or "") == EXPECTED_TRANSITION_SCORING_MODE
    )
    if requires_stage0_prior_gate:
        if (
            stage0_prior_recall_last is None
            or stage2_mrr_last is None
            or stage0_prior_mrr_last is None
            or worse_than_stage0_prior_last is None
        ):
            blockers.append("missing_stage0_prior_delta_metrics")
        else:
            if recall_drop_vs_stage0_prior is not None and recall_drop_vs_stage0_prior > float(max_stage2_recall_at_5_drop_vs_stage0_prior):
                blockers.append("stage2_recall_at_5_below_stage0_prior")
            if mrr_drop_vs_stage0_prior is not None and mrr_drop_vs_stage0_prior > float(max_stage2_mrr_drop_vs_stage0_prior):
                blockers.append("stage2_mrr_below_stage0_prior")
            if worse_than_stage0_prior_last > float(max_stage2_worse_than_stage0_prior_fraction):
                blockers.append("stage2_worse_than_stage0_prior_too_high")

    missing_terms = _missing_required_loss_terms(train_report, required_loss_terms)
    if missing_terms:
        blockers.append("missing_required_loss_terms")

    latest_checkpoint = train_report.get("latest_checkpoint") or output_dir / "checkpoints" / "latest.pt"
    loss_curve = train_report.get("loss_curve_path") or output_dir / "loss_curve.svg"
    latest_exists = _path_exists(latest_checkpoint)
    loss_curve_exists = _path_exists(loss_curve)
    if not latest_exists:
        blockers.append("missing_latest_checkpoint")
    if not loss_curve_exists:
        blockers.append("missing_loss_curve")

    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_report_path": str(train_report_path),
        "metrics_path": str(metrics_path),
        "metrics_summary": {
            "metric_count": len(rows),
            "max_step": max_step,
            "first_window": first_window,
            "last_window": last_window,
            "first_loss_mean": first_loss,
            "last_loss_mean": last_loss,
            "loss_drop": loss_drop,
            "min_steps": int(min_steps),
            "min_loss_drop": float(min_loss_drop),
        },
        "transition_quality": {
            "first_transition_skill_recall@5": transition_recall_first,
            "last_transition_skill_recall@5": transition_recall_last,
            "transition_skill_recall@5_drop": transition_recall_drop,
            "max_transition_recall_at_5_drop": float(max_transition_recall_at_5_drop),
            "min_transition_recall_at_5_tail": float(min_transition_recall_at_5_tail),
            "first_transition_skill_ce_loss": transition_ce_first,
            "last_transition_skill_ce_loss": transition_ce_last,
            "transition_skill_ce_increase": transition_ce_increase,
            "max_transition_skill_ce_increase": float(max_transition_skill_ce_increase),
            "transition_scoring_safety": transition_scoring_safety,
            "route_scorer_safety": route_scorer_safety,
        },
        "stage0_prior_comparison": {
            "last_stage2_transition_skill_recall@5": transition_recall_last,
            "last_stage0_prior_transition_skill_recall@5": stage0_prior_recall_last,
            "recall@5_drop_vs_stage0_prior": recall_drop_vs_stage0_prior,
            "max_stage2_recall_at_5_drop_vs_stage0_prior": float(max_stage2_recall_at_5_drop_vs_stage0_prior),
            "last_stage2_transition_skill_mrr": stage2_mrr_last,
            "last_stage0_prior_transition_skill_mrr": stage0_prior_mrr_last,
            "mrr_drop_vs_stage0_prior": mrr_drop_vs_stage0_prior,
            "max_stage2_mrr_drop_vs_stage0_prior": float(max_stage2_mrr_drop_vs_stage0_prior),
            "last_worse_than_stage0_prior_fraction": worse_than_stage0_prior_last,
            "max_stage2_worse_than_stage0_prior_fraction": float(max_stage2_worse_than_stage0_prior_fraction),
            "required": bool(requires_stage0_prior_gate),
        },
        "required_loss_terms": {
            "required": required_loss_terms,
            "loss_activation_counts": train_report.get("loss_activation_counts") if isinstance(train_report.get("loss_activation_counts"), dict) else {},
            "sampled_loss_activation_counts": train_report.get("sampled_loss_activation_counts") if isinstance(train_report.get("sampled_loss_activation_counts"), dict) else {},
            "missing_or_unsampled": missing_terms,
        },
        "train_safety": {
            "valid_or_test_used_for_training": bool(train_report.get("valid_or_test_used_for_training")),
            "on_policy_rollout_used": bool(train_report.get("on_policy_rollout_used")),
            "not_rl_fine_tuning": bool(train_report.get("not_rl_fine_tuning")),
            "training_regime": train_report.get("training_regime"),
        },
        "route_scorer_safety": route_scorer_safety,
        "next_skill_pool_safety": next_skill_pool_safety,
        "safe_memory_training_safety": safe_memory_training_safety,
        "anchored_routing_foundation_safety": anchored_routing_foundation_safety,
        "artifacts": {
            "checkpoint_exists": _path_exists(checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint),
            "latest_checkpoint_exists": latest_exists,
            "loss_curve_path": str(loss_curve),
            "loss_curve_exists": loss_curve_exists,
        },
        "paper_scope_note": (
            "This gate only proves Stage2 is safe to hand off to causal Stage4 and evaluation. "
            "It does not replace TRAJECT/ToolBench-G3/ToolRet benchmark evaluation."
        ),
    }
    if output_path is not None:
        write_json_report(output_path, report)
    return report
