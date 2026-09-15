from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from clstr.stage4_safe_memory import (
    validate_stage4_candidate_admission_delta_state_dict,
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_delta_state_dict,
)
from clstr.stage_quality_common import (
    EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    EXPECTED_TRANSITION_SCORING_MODE,
    artifact_exists as _path_exists,
    benchmark_counts as _benchmark_counts,
    bounded_window_size,
    finite_float as _as_float,
    max_counter,
    mean as _mean,
    read_json_file as _read_json,
    read_jsonl_file as _read_jsonl,
    transition_scoring_safety as _transition_scoring_safety,
    window_mean as _window_mean,
    write_json_report,
)


DEFAULT_EXPECTED_BENCHMARKS = ["toolbench_g3", "traject_bench", "alfworld", "webshop"]
FULL_POOL_TRAINING_OBJECTIVE = "full_pool_causal_next_skill_with_counterfactual_utility"
CMC_TRAINING_OBJECTIVE = "counterfactual_memory_calibration"
CANDIDATE_ADMISSION_TRAINING_OBJECTIVE = "candidate_admission_constrained_residual"
ALLOWED_TRAINING_OBJECTIVES = {
    FULL_POOL_TRAINING_OBJECTIVE,
    CMC_TRAINING_OBJECTIVE,
    CANDIDATE_ADMISSION_TRAINING_OBJECTIVE,
}


def _candidate_admission_contract_blockers(
    *,
    selection: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    train_report: dict[str, Any],
    router_integrity: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    candidate = dict(selection.get("candidate_admission_selection") or {})
    gradient = dict(selection.get("gradient_health") or {})
    if selection.get("promotion_status") != "promoted":
        blockers.append("stage4_candidate_admission_not_promoted")
    if checkpoint_payload.get("stage4_method") != "candidate_admission_residual_v1":
        blockers.append("stage4_candidate_admission_checkpoint_method_mismatch")
    if train_report.get("stage4_method") != "candidate_admission_residual_v1":
        blockers.append("stage4_candidate_admission_train_method_mismatch")
    if train_report.get("training_objective") != CANDIDATE_ADMISSION_TRAINING_OBJECTIVE:
        blockers.append("stage4_candidate_admission_objective_mismatch")
    if float(candidate.get("admission_auprc", float("-inf"))) <= float(
        candidate.get("admission_prevalence", float("inf"))
    ):
        blockers.append(
            "stage4_candidate_admission_signal_not_better_than_prevalence"
        )
    if int(candidate.get("dynamic_extra_positive_count", 0)) <= 0:
        blockers.append("stage4_candidate_admission_missing_positive_support")
    if float(candidate.get("balanced_macro_mrr", float("-inf"))) < float(
        candidate.get("static_balanced_macro_mrr", float("inf"))
    ):
        blockers.append("stage4_candidate_admission_static_regression")
    if float(
        candidate.get("no_regret_margin_violation_rate", float("inf"))
    ) > float(candidate.get("no_regret_tolerance", float("-inf"))):
        blockers.append("stage4_candidate_admission_no_regret_violation")
    if gradient.get("finite") is not True or gradient.get("nonzero") is not True:
        blockers.append("stage4_candidate_admission_gradient_health_failed")
    base_sha = str(checkpoint_payload.get("base_cmc_checkpoint_sha256") or "")
    if not base_sha or base_sha != str(
        train_report.get("base_cmc_checkpoint_sha256") or ""
    ):
        blockers.append("stage4_candidate_admission_base_cmc_mismatch")
    if (
        checkpoint_payload.get("parent_stage2_full_router_digest")
        != router_integrity.get("full_router_digest")
        or checkpoint_payload.get("parent_stage2_fast_router_digest")
        != router_integrity.get("fast_router_digest")
    ):
        blockers.append("stage4_candidate_admission_parent_router_mismatch")
    if (
        router_integrity.get("full_router_digest_unchanged") is not True
        or router_integrity.get("fast_router_digest_unchanged") is not True
    ):
        blockers.append("stage4_candidate_admission_parent_router_changed")
    freeze_report = dict(train_report.get("freeze_report") or {})
    if freeze_report.get("trainable_modules") != [
        "route_memory_candidate_admission_residual"
    ]:
        blockers.append("stage4_candidate_admission_optimizer_scope_mismatch")
    if freeze_report.get("frozen_cmc_residual_adapter") is not True:
        blockers.append("stage4_candidate_admission_cmc_adapter_not_frozen")
    return blockers
ROUTING_FOUNDATION_PREFIXES = ("encoder.", "cross_encoder.", "skill_table.")
HEALTH_ONLY_BLOCKERS = {
    "missing_stage4_next_skill_quality_metrics",
    "stage4_next_skill_recall_at_5_too_low",
    "stage4_act_loss_regressed",
}
CMC_LEGACY_ONLY_BLOCKERS = {
    "nonfinite_stage4_full_pool_main_loss",
    "missing_stage4_full_pool_eligible_rows",
    "nonfinite_stage4_counterfactual_utility_loss",
    "missing_stage4_counterfactual_eligible_rows",
    "missing_stage4_counterfactual_weak_rows",
    "missing_stage4_counterfactual_strong_rows",
    "missing_stage4_counterfactual_gain_violation_rows",
    "missing_stage4_counterfactual_safety_violation_rows",
    "missing_stage4_static_hit_rows",
    "missing_stage4_static_miss_rows",
    "stage4_counterfactual_row_counts_inconsistent",
}


def _self_hashed_json(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = _read_json(path)
    digest_payload = dict(payload)
    recorded = str(digest_payload.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(digest_payload):
        raise ValueError(f"{label} self-hash mismatch")
    return payload


def _selected_release_blockers(
    *,
    selection_path: str | Path,
    stage2_baseline_report_path: str | Path,
    checkpoint_path: Path,
    train_report: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    selection = _self_hashed_json(selection_path, label="Stage4 selection")
    if selection.get("schema_version") != "stage4_selection_v1":
        blockers.append("unsupported_stage4_selection_schema")
    if selection.get("status") != "ok" or selection.get("release_status") != "ok":
        blockers.append("stage4_selection_not_release_safe")
    release_checks = selection.get("release_checks") or {}
    if not isinstance(release_checks, dict) or not release_checks or not all(
        value is True for value in release_checks.values()
    ):
        blockers.append("stage4_selection_release_checks_failed")
    selected_checkpoint = Path(
        str(selection.get("selected_checkpoint_path") or "")
    ).resolve()
    if selected_checkpoint != checkpoint_path.resolve():
        blockers.append("stage4_selected_checkpoint_disagreement")
    selected_checkpoint_identity = sha256_path(selected_checkpoint)
    if selected_checkpoint_identity["sha256"] != str(
        selection.get("selected_checkpoint_sha256") or ""
    ):
        blockers.append("stage4_selected_checkpoint_digest_mismatch")
    checkpoint_payload = torch.load(selected_checkpoint, map_location="cpu")
    delta_state = (
        checkpoint_payload.get("model_state_dict")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    candidate_admission_enabled = (
        selection.get("stage4_method") == "candidate_admission_residual_v1"
    )
    cmc_enabled = selection.get("stage4_method") == "counterfactual_memory_calibration_v1"
    try:
        delta_validation = (
            validate_stage4_candidate_admission_delta_state_dict(delta_state or {})
            if candidate_admission_enabled
            else validate_stage4_cmc_delta_state_dict(delta_state or {})
            if cmc_enabled
            else validate_stage4_delta_state_dict(delta_state or {})
        )
    except (TypeError, ValueError):
        blockers.append("invalid_stage4_delta_checkpoint")
        delta_validation = {"status": "invalid"}
    validation_path = Path(
        str(selection.get("selected_validation_report_path") or "")
    ).resolve()
    if sha256_path(validation_path)["sha256"] != str(
        selection.get("selected_validation_report_sha256") or ""
    ):
        blockers.append("stage4_selected_validation_digest_mismatch")
    validation = _self_hashed_json(
        validation_path,
        label="Stage4 selected validation report",
    )
    baseline_path = Path(stage2_baseline_report_path).resolve()
    baseline_identity = sha256_path(baseline_path)
    baseline_payload = _self_hashed_json(
        baseline_path,
        label="Stage2 baseline report",
    )
    selection_baseline = dict(selection.get("stage2_baseline") or {})
    if selection_baseline.get("report_sha256") != baseline_identity["sha256"]:
        blockers.append("stage2_baseline_identity_mismatch")
    if dict(validation.get("stage2_baseline") or {}) != selection_baseline:
        blockers.append("selected_validation_baseline_mismatch")
    if dict(selection.get("router_integrity") or {}) != dict(
        validation.get("router_integrity") or {}
    ):
        blockers.append("stage4_router_integrity_mismatch")
    router_integrity = dict(selection.get("router_integrity") or {})
    if (
        router_integrity.get("static_logits_exact") is not True
        or float(
            router_integrity.get(
                "max_abs_static_logit_difference",
                float("inf"),
            )
        )
        != 0.0
        or router_integrity.get("full_router_digest")
        != selection_baseline.get("full_router_digest")
    ):
        blockers.append("stage4_static_router_not_exact")
    candidate_union = dict(selection.get("candidate_union") or {})
    if int(candidate_union.get("final_k", 1)) > int(
        candidate_union.get("static_k", 0)
    ):
        blockers.append("stage4_candidate_union_budget_invalid")
    macro = dict(validation.get("balanced_macro") or {})
    by_benchmark = dict(validation.get("by_benchmark") or {})
    baseline_by_benchmark = dict(selection_baseline.get("by_benchmark") or {})
    reliability_validation = dict(selection.get("reliability_validation") or {})
    reliability = dict(selection.get("reliability") or {})
    if candidate_admission_enabled:
        if validation.get("stage4_method") != "candidate_admission_residual_v1":
            blockers.append("stage4_candidate_admission_validation_method_mismatch")
        if dict(selection.get("candidate_admission_selection") or {}) != dict(
            validation.get("candidate_admission") or {}
        ):
            blockers.append("stage4_candidate_admission_selection_mismatch")
        blockers.extend(
            _candidate_admission_contract_blockers(
                selection=selection,
                checkpoint_payload=checkpoint_payload,
                train_report=train_report,
                router_integrity=router_integrity,
            )
        )
        if (
            reliability.get("mode") != "candidate_admission_residual"
            or reliability.get("fixed_alpha") is not None
            or reliability.get("gate_checkpoint") is not None
            or float(
                reliability.get("safe_memory_residual_bound", float("nan"))
            )
            != 2.0
            or float(
                reliability.get("feature_update_count_cap", float("nan"))
            )
            != 16.0
            or float(
                reliability.get("feature_candidate_count_cap", float("nan"))
            )
            != 256.0
            or reliability.get("deployed_reliability_source")
            != "candidate_admission_constrained_residual"
        ):
            blockers.append("stage4_candidate_admission_reliability_mismatch")
        if str(reliability.get("base_cmc_checkpoint_sha256") or "") != str(
            checkpoint_payload.get("base_cmc_checkpoint_sha256") or ""
        ):
            blockers.append("stage4_candidate_admission_reliability_lineage_mismatch")
        if train_report.get("valid_or_test_used_for_training") is not False:
            blockers.append("valid_or_test_used_for_training")
        if train_report.get("validation_used_for_checkpoint_selection") is not True:
            blockers.append("validation_not_used_for_checkpoint_selection")
        return blockers, {
            "selection": selection,
            "selection_identity": sha256_path(selection_path),
            "selected_checkpoint": selected_checkpoint_identity,
            "selected_validation_report": sha256_path(validation_path),
            "stage2_baseline_report": baseline_identity,
            "stage2_baseline": baseline_payload,
            "delta_validation": delta_validation,
        }
    if cmc_enabled:
        cmc_fused = dict(selection.get("cmc_fused_selection") or {})
        validation_cmc_fused = dict(validation.get("cmc_fused") or {})
        gradient_health = dict(selection.get("gradient_health") or {})
        if selection.get("promotion_status") != "promoted":
            blockers.append("stage4_cmc_not_promoted")
        if validation.get("stage4_method") != "counterfactual_memory_calibration_v1":
            blockers.append("stage4_cmc_validation_method_mismatch")
        if checkpoint_payload.get("stage4_method") != "counterfactual_memory_calibration_v1":
            blockers.append("stage4_cmc_checkpoint_method_mismatch")
        if train_report.get("stage4_method") != "counterfactual_memory_calibration_v1":
            blockers.append("stage4_cmc_train_method_mismatch")
        if cmc_fused != validation_cmc_fused:
            blockers.append("stage4_cmc_fused_selection_mismatch")
        if reliability.get("mode") != "cmc_candidate_gate":
            blockers.append("stage4_cmc_reliability_mode_mismatch")
        if reliability_validation != cmc_fused:
            blockers.append("stage4_cmc_reliability_validation_mismatch")
        if float(cmc_fused.get("balanced_macro_mrr", float("-inf"))) < float(
            selection_baseline.get("static_macro_mrr", float("inf"))
        ) + 0.005:
            blockers.append("stage4_cmc_fused_macro_improvement_too_small")
        if any(
            float(by_benchmark[benchmark].get("fused_mrr", float("-inf")))
            < float(metrics["static_mrr"]) - 0.01
            for benchmark, metrics in baseline_by_benchmark.items()
        ):
            blockers.append("stage4_cmc_fused_benchmark_regression")
        if float(cmc_fused.get("regret", float("inf"))) >= float(
            selection_baseline.get("raw_dynamic_regret", float("-inf"))
        ):
            blockers.append("stage4_cmc_regret_not_reduced")
        if not all(
            cmc_fused.get(key) is True
            for key in (
                "alpha_zero_exact",
                "alpha_one_exact",
                "zero_history_exact",
            )
        ):
            blockers.append("stage4_cmc_endpoints_not_exact")
        if (
            gradient_health.get("finite") is not True
            or gradient_health.get("nonzero") is not True
        ):
            blockers.append("stage4_cmc_gradient_health_failed")
        if (
            router_integrity.get("full_router_digest_unchanged") is not True
            or router_integrity.get("fast_router_digest_unchanged") is not True
        ):
            blockers.append("stage4_cmc_parent_router_changed")
        if (
            checkpoint_payload.get("parent_stage2_full_router_digest")
            != router_integrity.get("full_router_digest")
            or checkpoint_payload.get("parent_stage2_fast_router_digest")
            != router_integrity.get("fast_router_digest")
        ):
            blockers.append("stage4_cmc_parent_router_digest_mismatch")
        if (
            train_report.get("safe_memory_protocol_version")
            != "counterfactual_memory_calibration_v1"
        ):
            blockers.append("stage4_cmc_protocol_missing")
        if train_report.get("valid_or_test_used_for_training") is not False:
            blockers.append("valid_or_test_used_for_training")
        if train_report.get("validation_used_for_checkpoint_selection") is not True:
            blockers.append("validation_not_used_for_checkpoint_selection")
        if train_report.get("final_benchmark_used_for_checkpoint_selection") is True:
            blockers.append("final_benchmark_used_for_checkpoint_selection")
        if train_report.get("tau2_used_for_checkpoint_selection") is True:
            blockers.append("tau2_used_for_checkpoint_selection")
        return blockers, {
            "selection": selection,
            "selection_identity": sha256_path(selection_path),
            "selected_checkpoint": selected_checkpoint_identity,
            "selected_validation_report": sha256_path(validation_path),
            "stage2_baseline_report": baseline_identity,
            "stage2_baseline": baseline_payload,
            "delta_validation": delta_validation,
        }
    if reliability.get("mode") == "causal_gate":
        if dict(selection.get("safe_fused_selection") or {}) != reliability_validation:
            blockers.append("stage4_safe_fused_selection_mismatch")
        if float(macro.get("safe_fused_mrr", float("-inf"))) < float(
            selection_baseline.get("safe_fused_macro_mrr", float("inf"))
        ) - 0.01:
            blockers.append("stage4_safe_fused_stage2_regression")
        if float(macro.get("safe_fused_minus_static_mrr", float("-inf"))) < -0.005:
            blockers.append("stage4_safe_fused_static_regression")
        if sum(
            float(metrics.get("safe_fused_minus_static_mrr", float("-inf")))
            >= -0.01
            for metrics in by_benchmark.values()
        ) < 3:
            blockers.append("stage4_too_few_safe_benchmarks")
        if any(
            float(baseline_by_benchmark[benchmark]["safe_fused_mrr"])
            - float(metrics["safe_fused_mrr"])
            > 0.03
            for benchmark, metrics in by_benchmark.items()
        ):
            blockers.append("stage4_safe_fused_benchmark_regression")
    else:
        if float(macro.get("dynamic_mrr", float("-inf"))) < float(
            selection_baseline.get("dynamic_macro_mrr", float("inf"))
        ) + 0.01:
            blockers.append("stage4_dynamic_macro_improvement_too_small")
        if float(macro.get("dynamic_minus_static_mrr", float("-inf"))) < 0.01:
            blockers.append("stage4_memory_active_delta_too_small")
        if sum(
            float(metrics.get("dynamic_minus_static_mrr", float("-inf"))) >= 0.0
            for metrics in by_benchmark.values()
        ) < 3:
            blockers.append("stage4_too_few_nonnegative_benchmarks")
        if any(
            float(baseline_by_benchmark[benchmark]["dynamic_mrr"])
            - float(metrics["dynamic_mrr"])
            > 0.03
            for benchmark, metrics in by_benchmark.items()
        ):
            blockers.append("stage4_dynamic_benchmark_regression")
        reference = max(
            float(selection_baseline.get("static_macro_mrr", float("inf"))),
            float(selection_baseline.get("dynamic_macro_mrr", float("inf"))),
        )
        if float(
            reliability_validation.get("balanced_macro_mrr", float("-inf"))
        ) < reference + 0.005:
            blockers.append("stage4_fused_macro_improvement_too_small")
    fused_by_benchmark = dict(reliability_validation.get("by_benchmark") or {})
    if any(
        float(fused_by_benchmark[benchmark]["mrr"])
        < float(metrics["static_mrr"]) - 0.01
        for benchmark, metrics in baseline_by_benchmark.items()
    ):
        blockers.append("stage4_fused_benchmark_regression")
    if reliability_validation.get("zero_history_exact") is not True:
        blockers.append("stage4_zero_history_not_exact")
    if train_report.get("safe_memory_protocol_version") != "stage4_safe_memory_v1":
        blockers.append("stage4_safe_memory_protocol_missing")
    if train_report.get("valid_or_test_used_for_training") is not False:
        blockers.append("valid_or_test_used_for_training")
    if train_report.get("validation_used_for_checkpoint_selection") is not True:
        blockers.append("validation_not_used_for_checkpoint_selection")
    if train_report.get("final_benchmark_used_for_checkpoint_selection") is True:
        blockers.append("final_benchmark_used_for_checkpoint_selection")
    if train_report.get("tau2_used_for_checkpoint_selection") is True:
        blockers.append("tau2_used_for_checkpoint_selection")
    return blockers, {
        "selection": selection,
        "selection_identity": sha256_path(selection_path),
        "selected_checkpoint": selected_checkpoint_identity,
        "selected_validation_report": sha256_path(validation_path),
        "stage2_baseline_report": baseline_identity,
        "stage2_baseline": baseline_payload,
        "delta_validation": delta_validation,
    }


def _max_step(rows: list[dict[str, Any]], train_report: dict[str, Any]) -> int:
    return max_counter(rows, train_report, row_keys=("step", "update"), report_keys=("max_steps", "step"))


def _has_stage4_act_metrics(rows: list[dict[str, Any]], train_report: dict[str, Any]) -> bool:
    return any("stage4_act_loss" in row and "stage4_act_count" in row for row in rows)


def _latest_finite_metric(
    rows: list[dict[str, Any]],
    last_metrics: dict[str, Any],
    key: str,
) -> float | None:
    if key in last_metrics:
        return _as_float(last_metrics.get(key))
    values = [parsed for row in rows if (parsed := _as_float(row.get(key))) is not None]
    return values[-1] if values else None


def audit_stage4_act_quality(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    train_report_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    min_steps: int = 2000,
    first_window: int = 200,
    last_window: int = 200,
    expected_benchmarks: list[str] | None = None,
    require_stage0_handoff: bool = True,
    require_transition_trainable: bool = True,
    require_routing_foundation_protection: bool = True,
    min_stage4_recall_at_5_tail: float = 0.5,
    max_stage4_act_loss_increase: float = 0.2,
    expected_transition_scoring_mode: str | None = None,
    expected_transition_residual_lambda: float | None = None,
    selection_path: str | Path | None = None,
    stage2_baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_report_path = Path(train_report_path) if train_report_path is not None else output_dir / "train_stdout.json"
    train_report = _read_json(train_report_path)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(train_report.get("checkpoint") or output_dir / "checkpoints" / f"clstr_stage4_act-step{min_steps}.pt")
    )
    metrics_path = Path(metrics_path) if metrics_path is not None else Path(train_report.get("training_metrics_path") or output_dir / "training_metrics.jsonl")
    expected_benchmarks = list(expected_benchmarks or DEFAULT_EXPECTED_BENCHMARKS)
    blockers: list[str] = []

    if not train_report_path.is_file():
        blockers.append("missing_train_report")
    if train_report and train_report.get("status") != "ok":
        blockers.append("stage4_train_report_not_ok")
    if train_report and train_report.get("stage") != "clstr_stage4_transition_conditioned_next_skill":
        blockers.append("unexpected_stage4_stage")
    training_objective = train_report.get("training_objective")
    cmc_enabled = training_objective == CMC_TRAINING_OBJECTIVE
    candidate_admission_enabled = (
        training_objective == CANDIDATE_ADMISSION_TRAINING_OBJECTIVE
    )
    if train_report and training_objective not in ALLOWED_TRAINING_OBJECTIVES:
        blockers.append("unexpected_training_objective")
    if train_report and train_report.get("training_regime") != "offline_train_split_causal_next_skill":
        blockers.append("unexpected_training_regime")
    if train_report and train_report.get("valid_or_test_used_for_training") is True:
        blockers.append("valid_or_test_used_for_training")
    if train_report and train_report.get("on_policy_rollout_used") is True:
        blockers.append("stage4_should_not_use_on_policy_rollout")

    checkpoint_init_report = (
        train_report.get("checkpoint_init_report")
        if isinstance(train_report.get("checkpoint_init_report"), dict)
        else {}
    )
    head_overrides = [
        str(item)
        for item in checkpoint_init_report.get("head_overrides_routing_keys", [])
        if str(item)
    ] if isinstance(checkpoint_init_report.get("head_overrides_routing_keys"), list) else []
    routing_foundation_head_allowlist = sorted(
        str(item)
        for item in checkpoint_init_report.get("routing_foundation_head_allowlist", [])
        if str(item)
    ) if isinstance(checkpoint_init_report.get("routing_foundation_head_allowlist"), list) else []
    head_overrode_routing_foundation_keys = sorted(
        key
        for key in head_overrides
        if key.startswith(ROUTING_FOUNDATION_PREFIXES)
        and key not in routing_foundation_head_allowlist
    )
    skipped_head_routing_foundation_keys = [
        str(item)
        for item in checkpoint_init_report.get("skipped_head_routing_foundation_keys", [])
        if str(item)
    ] if isinstance(checkpoint_init_report.get("skipped_head_routing_foundation_keys"), list) else []
    protect_routing_foundation = checkpoint_init_report.get("protect_routing_foundation") is True
    if train_report and require_routing_foundation_protection and not protect_routing_foundation:
        blockers.append("stage4_checkpoint_init_not_protected")
    if train_report and head_overrode_routing_foundation_keys:
        blockers.append("stage4_head_checkpoint_overrode_routing_foundation")

    freeze_report = train_report.get("freeze_report") if isinstance(train_report.get("freeze_report"), dict) else {}
    if train_report and freeze_report.get("frozen_routing_foundation") is not True:
        blockers.append("routing_foundation_not_frozen")
    trainable_modules = [
        str(item)
        for item in freeze_report.get("trainable_modules", [])
        if str(item)
    ] if isinstance(freeze_report.get("trainable_modules"), list) else []
    if (
        require_transition_trainable
        and train_report
        and not cmc_enabled
        and not candidate_admission_enabled
    ):
        if freeze_report.get("train_transition") is not True or "transition" not in trainable_modules:
            blockers.append("stage4_transition_not_trainable")
    if cmc_enabled and train_report:
        if freeze_report.get("train_transition") is not False:
            blockers.append("stage4_cmc_transition_must_remain_frozen")
        if trainable_modules != [
            "route_memory_residual_adapter",
            "route_memory_candidate_utility_gate",
        ]:
            blockers.append("stage4_cmc_optimizer_scope_mismatch")
    if candidate_admission_enabled and train_report:
        if freeze_report.get("train_transition") is not False:
            blockers.append("stage4_candidate_admission_transition_must_remain_frozen")
        if trainable_modules != ["route_memory_candidate_admission_residual"]:
            blockers.append("stage4_candidate_admission_optimizer_scope_mismatch")
        if freeze_report.get("frozen_cmc_residual_adapter") is not True:
            blockers.append("stage4_candidate_admission_cmc_adapter_not_frozen")
    if train_report.get("safe_memory_protocol_version") == "stage4_safe_memory_v1":
        if "route_memory_utility_gate" not in trainable_modules:
            blockers.append("stage4_route_memory_utility_gate_not_trainable")
        if {"initial_belief_head", "unified_retriever"}.intersection(trainable_modules):
            blockers.append("stage4_static_routing_foundation_trainable")

    if not _path_exists(checkpoint_path):
        blockers.append("missing_stage4_checkpoint")

    rows = _read_jsonl(metrics_path)
    if not metrics_path.is_file():
        blockers.append("missing_training_metrics")
    route_scorer = str(train_report.get("route_scorer") or "")
    if train_report and route_scorer != "unified_memory":
        blockers.append("stage4_route_scorer_not_unified_memory")
    if training_objective in {
        FULL_POOL_TRAINING_OBJECTIVE,
        CMC_TRAINING_OBJECTIVE,
        CANDIDATE_ADMISSION_TRAINING_OBJECTIVE,
    }:
        expected_scoring_mode = "unified_memory"
        expected_residual_lambda = 0.0
    else:
        expected_scoring_mode = (
            str(expected_transition_scoring_mode)
            if expected_transition_scoring_mode is not None
            else EXPECTED_TRANSITION_SCORING_MODE
        )
        expected_residual_lambda = (
            float(expected_transition_residual_lambda)
            if expected_transition_residual_lambda is not None
            else EXPECTED_TRANSITION_RESIDUAL_LAMBDA
        )
    transition_scoring_safety = _transition_scoring_safety(
        train_report,
        rows,
        expected_mode=expected_scoring_mode,
        expected_residual_lambda=expected_residual_lambda,
    )
    if train_report and transition_scoring_safety["matched"] is not True:
        blockers.append("stage4_transition_scoring_not_effective_route_mode")
    max_step = _max_step(rows, train_report)
    if max_step < int(min_steps):
        blockers.append("insufficient_training_steps")
    if cmc_enabled or candidate_admission_enabled:
        if not any(
            "stage4_total_loss" in row and "stage4_act_count" in row
            for row in rows
        ):
            blockers.append("missing_stage4_cmc_metrics")
    elif not _has_stage4_act_metrics(rows, train_report):
        blockers.append("missing_stage4_act_metrics")
    first_window = bounded_window_size(first_window, len(rows))
    last_window = bounded_window_size(last_window, len(rows))
    monitored_loss_key = (
        "stage4_total_loss"
        if cmc_enabled or candidate_admission_enabled
        else "stage4_act_loss"
    )
    first_stage4_act_loss = _window_mean(
        rows,
        monitored_loss_key,
        first_window,
        tail=False,
    )
    last_stage4_act_loss = _window_mean(
        rows,
        monitored_loss_key,
        last_window,
        tail=True,
    )
    stage4_act_loss_increase = (
        None
        if first_stage4_act_loss is None or last_stage4_act_loss is None
        else round(float(last_stage4_act_loss - first_stage4_act_loss), 12)
    )
    last_stage4_recall_at_5 = _window_mean(
        rows,
        "stage4_next_skill_recall@5",
        last_window,
        tail=True,
    )
    if (cmc_enabled or candidate_admission_enabled) and stage4_act_loss_increase is None:
        blockers.append("missing_stage4_cmc_loss_metrics")
    elif not cmc_enabled and not candidate_admission_enabled and (
        last_stage4_recall_at_5 is None or stage4_act_loss_increase is None
    ):
        blockers.append("missing_stage4_next_skill_quality_metrics")
    elif not cmc_enabled and not candidate_admission_enabled:
        if last_stage4_recall_at_5 < float(min_stage4_recall_at_5_tail):
            blockers.append("stage4_next_skill_recall_at_5_too_low")
        if stage4_act_loss_increase > float(max_stage4_act_loss_increase):
            blockers.append("stage4_act_loss_regressed")
    elif stage4_act_loss_increase > float(max_stage4_act_loss_increase):
        blockers.append("stage4_act_loss_regressed")

    last_metrics = train_report.get("last_metrics") if isinstance(train_report.get("last_metrics"), dict) else {}
    post_action_update_rows = _as_float(last_metrics.get("stage4_post_action_update_rows"))
    next_state_rows = _as_float(last_metrics.get("stage4_next_state_rows"))
    if post_action_update_rows is None:
        post_action_values = [
            value for row in rows if (value := _as_float(row.get("stage4_post_action_update_rows"))) is not None
        ]
        post_action_update_rows = post_action_values[-1] if post_action_values else None
    if next_state_rows is None:
        next_state_values = [
            value for row in rows if (value := _as_float(row.get("stage4_next_state_rows"))) is not None
        ]
        next_state_rows = next_state_values[-1] if next_state_values else None
    if post_action_update_rows is None or post_action_update_rows <= 0:
        blockers.append("missing_stage4_post_action_update_rows")
    if next_state_rows is None or next_state_rows <= 0:
        blockers.append("missing_stage4_next_state_rows")

    next_skill_pool_mode = str(train_report.get("next_skill_pool_mode") or "")
    if train_report and next_skill_pool_mode != "full_pool":
        blockers.append("stage4_next_skill_pool_mode_not_full_pool")
    full_pool_main_loss = _latest_finite_metric(rows, last_metrics, "stage4_act_loss")
    full_pool_eligible_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_full_pool_eligible_rows",
    )
    counterfactual_utility_loss = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_utility_loss",
    )
    counterfactual_eligible_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_eligible_rows",
    )
    counterfactual_weak_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_weak_rows",
    )
    counterfactual_strong_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_strong_rows",
    )
    counterfactual_gain_violation_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_gain_violation_rows",
    )
    counterfactual_safety_violation_rows = _latest_finite_metric(
        rows,
        last_metrics,
        "stage4_counterfactual_safety_violation_rows",
    )
    static_hit_rows = _latest_finite_metric(rows, last_metrics, "stage4_static_hit_train_rows")
    static_miss_rows = _latest_finite_metric(rows, last_metrics, "stage4_static_miss_train_rows")
    if full_pool_main_loss is None:
        blockers.append("nonfinite_stage4_full_pool_main_loss")
    if full_pool_eligible_rows is None or full_pool_eligible_rows <= 0:
        blockers.append("missing_stage4_full_pool_eligible_rows")
    if counterfactual_utility_loss is None:
        blockers.append("nonfinite_stage4_counterfactual_utility_loss")
    if counterfactual_eligible_rows is None or counterfactual_eligible_rows <= 0:
        blockers.append("missing_stage4_counterfactual_eligible_rows")
    if counterfactual_weak_rows is None:
        blockers.append("missing_stage4_counterfactual_weak_rows")
    if counterfactual_strong_rows is None:
        blockers.append("missing_stage4_counterfactual_strong_rows")
    if counterfactual_gain_violation_rows is None:
        blockers.append("missing_stage4_counterfactual_gain_violation_rows")
    if counterfactual_safety_violation_rows is None:
        blockers.append("missing_stage4_counterfactual_safety_violation_rows")
    if static_hit_rows is None:
        blockers.append("missing_stage4_static_hit_rows")
    if static_miss_rows is None:
        blockers.append("missing_stage4_static_miss_rows")
    if (
        counterfactual_eligible_rows is not None
        and counterfactual_weak_rows is not None
        and counterfactual_strong_rows is not None
        and abs(counterfactual_eligible_rows - counterfactual_weak_rows - counterfactual_strong_rows) > 1.0e-6
    ):
        blockers.append("stage4_counterfactual_row_counts_inconsistent")
    cmc_objective_metrics = {
        key: _latest_finite_metric(rows, last_metrics, key)
        for key in (
            "stage4_total_loss",
            "stage4_cmc_dynamic_loss",
            "stage4_cmc_fused_loss",
            "stage4_cmc_no_regret_loss",
            "stage4_cmc_eligible_rows",
        )
    }
    if cmc_enabled or candidate_admission_enabled:
        blockers = [
            blocker
            for blocker in blockers
            if blocker not in CMC_LEGACY_ONLY_BLOCKERS
        ]
    if candidate_admission_enabled:
        candidate_metrics = {
            key: _latest_finite_metric(rows, last_metrics, key)
            for key in (
                "stage4_total_loss",
                "stage4_candidate_admission_listwise_loss",
                "stage4_candidate_admission_bce_loss",
                "stage4_candidate_admission_no_regret_loss",
                "stage4_candidate_admission_positive_count",
            )
        }
        for key, value in candidate_metrics.items():
            if value is None:
                blockers.append(f"missing_{key}")
    if cmc_enabled:
        for key, value in cmc_objective_metrics.items():
            if value is None:
                blockers.append(f"missing_{key}")
        if (
            cmc_objective_metrics["stage4_cmc_eligible_rows"] is not None
            and cmc_objective_metrics["stage4_cmc_eligible_rows"] <= 0
        ):
            blockers.append("missing_stage4_cmc_eligible_rows")

    data_report = train_report.get("data_report") if isinstance(train_report.get("data_report"), dict) else {}
    stage4_rows = int(data_report.get("stage4_rows") or 0)
    if stage4_rows <= 0:
        blockers.append("no_stage4_rows")
    stage0_candidate_handoff = (
        data_report.get("stage0_candidate_handoff")
        if isinstance(data_report.get("stage0_candidate_handoff"), dict)
        else {}
    )
    if require_stage0_handoff and train_report:
        if data_report.get("candidate_source") != "declared_legal_full_skill_pool":
            blockers.append("stage4_candidate_source_not_declared_legal_full_pool")
        if data_report.get("static_candidate_source") != "stage0_topm_online":
            blockers.append("stage4_static_candidate_source_not_stage0_topm")
        if stage0_candidate_handoff.get("enabled") is not True:
            blockers.append("stage4_stage0_handoff_not_enabled")
        if stage0_candidate_handoff.get("candidate_source") != "stage0_topm_online":
            blockers.append("stage4_handoff_candidate_source_not_stage0_topm")
        if str(stage0_candidate_handoff.get("positive_missing_policy") or "") != "skip":
            blockers.append("stage4_positive_missing_policy_not_skip")
        if int(data_report.get("positive_injected_rows") or 0) > 0:
            blockers.append("stage4_positive_injected_rows")
        if int(stage0_candidate_handoff.get("injected_positive_rows") or 0) > 0:
            blockers.append("stage4_stage0_handoff_injected_positive_rows")
    counts = _benchmark_counts(train_report)
    missing_benchmarks = [name for name in expected_benchmarks if int(counts.get(name, 0)) <= 0]
    if missing_benchmarks:
        blockers.append("missing_expected_benchmarks")

    latest_checkpoint = train_report.get("latest_checkpoint") or output_dir / "checkpoints" / "latest.pt"
    loss_curve = train_report.get("loss_curve_path") or output_dir / "loss_curve.svg"
    latest_exists = _path_exists(latest_checkpoint)
    loss_curve_exists = _path_exists(loss_curve)
    if not latest_exists:
        blockers.append("missing_latest_checkpoint")
    if not loss_curve_exists:
        blockers.append("missing_loss_curve")

    losses = [value for row in rows if (value := _as_float(row.get("loss"))) is not None]
    act_losses = [value for row in rows if (value := _as_float(row.get("stage4_act_loss"))) is not None]
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
            "min_steps": int(min_steps),
            "first_window": first_window,
            "last_window": last_window,
            "loss_mean": _mean(losses),
            "stage4_act_loss_mean": _mean(act_losses),
            "last_metrics": train_report.get("last_metrics") if isinstance(train_report.get("last_metrics"), dict) else {},
        },
        "stage4_quality": {
            "first_stage4_act_loss": first_stage4_act_loss,
            "last_stage4_act_loss": last_stage4_act_loss,
            "stage4_act_loss_increase": stage4_act_loss_increase,
            "max_stage4_act_loss_increase": float(max_stage4_act_loss_increase),
            "last_stage4_next_skill_recall@5": last_stage4_recall_at_5,
            "min_stage4_recall_at_5_tail": float(min_stage4_recall_at_5_tail),
            "transition_scoring_safety": transition_scoring_safety,
        },
        "full_pool_objective": {
            "next_skill_pool_mode": next_skill_pool_mode,
            "main_loss": full_pool_main_loss,
            "eligible_rows": full_pool_eligible_rows,
            "counterfactual_utility_loss": counterfactual_utility_loss,
            "counterfactual_eligible_rows": counterfactual_eligible_rows,
            "counterfactual_weak_rows": counterfactual_weak_rows,
            "counterfactual_strong_rows": counterfactual_strong_rows,
            "counterfactual_gain_violation_rows": counterfactual_gain_violation_rows,
            "counterfactual_safety_violation_rows": counterfactual_safety_violation_rows,
            "static_hit_rows": static_hit_rows,
            "static_miss_rows": static_miss_rows,
            "cmc": cmc_objective_metrics if cmc_enabled else None,
        },
        "causal_update": {
            "post_action_update_rows": post_action_update_rows,
            "next_state_rows": next_state_rows,
        },
        "data_coverage": {
            "stage4_rows": stage4_rows,
            "expected_benchmarks": expected_benchmarks,
            "benchmark_counts": counts,
            "missing_expected_benchmarks": missing_benchmarks,
            "candidate_source": data_report.get("candidate_source"),
            "static_candidate_source": data_report.get("static_candidate_source"),
            "positive_injected_rows": int(data_report.get("positive_injected_rows") or 0),
            "stage0_candidate_rows": int(data_report.get("stage0_candidate_rows") or 0),
            "stage0_candidate_handoff": stage0_candidate_handoff,
            "require_stage0_handoff": bool(require_stage0_handoff),
        },
        "train_safety": {
            "valid_or_test_used_for_training": bool(train_report.get("valid_or_test_used_for_training")),
            "on_policy_rollout_used": bool(train_report.get("on_policy_rollout_used")),
            "training_regime": train_report.get("training_regime"),
            "training_objective": training_objective,
            "route_scorer": route_scorer,
            "require_transition_trainable": bool(require_transition_trainable),
            "train_transition": bool(freeze_report.get("train_transition")),
            "trainable_modules": trainable_modules,
        },
        "checkpoint_init": {
            "protect_routing_foundation": bool(protect_routing_foundation),
            "require_routing_foundation_protection": bool(require_routing_foundation_protection),
            "head_overrides_routing_keys": head_overrides,
            "head_overrode_routing_foundation_keys": head_overrode_routing_foundation_keys,
            "routing_foundation_head_allowlist": routing_foundation_head_allowlist,
            "skipped_head_routing_foundation_keys": skipped_head_routing_foundation_keys,
        },
        "artifacts": {
            "checkpoint_exists": _path_exists(checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint),
            "latest_checkpoint_exists": latest_exists,
            "loss_curve_path": str(loss_curve),
            "loss_curve_exists": loss_curve_exists,
        },
        "paper_scope": {
            "ready_for_phase_h_handoff": not bool(blockers),
            "metric_scope": "Stage4 transition-conditioned next-skill L_act training quality gate; not benchmark evaluation.",
            "note": "This gate only validates Stage4 training artifacts before Phase H evaluation.",
        },
    }
    if selection_path is not None or stage2_baseline_report_path is not None:
        if selection_path is None or stage2_baseline_report_path is None:
            raise ValueError(
                "safe Stage4 quality requires selection and Stage2 baseline paths"
            )
        release_blockers, release_evidence = _selected_release_blockers(
            selection_path=selection_path,
            stage2_baseline_report_path=stage2_baseline_report_path,
            checkpoint_path=checkpoint_path,
            train_report=train_report,
        )
        structural_blockers = [
            blocker for blocker in blockers if blocker not in HEALTH_ONLY_BLOCKERS
        ]
        final_blockers = sorted(set(structural_blockers + release_blockers))
        report["status"] = "ok" if not final_blockers else "action_required"
        report["blockers"] = final_blockers
        report["release_authority"] = "fixed_validation_selection_v1"
        report["selection_path"] = str(Path(selection_path).resolve())
        report["stage2_baseline_report_path"] = str(
            Path(stage2_baseline_report_path).resolve()
        )
        report["selected_checkpoint"] = release_evidence[
            "selected_checkpoint"
        ]
        report["selected_validation_report"] = release_evidence[
            "selected_validation_report"
        ]
        report["stage4_selection"] = release_evidence["selection"]
        report["stage4_selection_identity"] = release_evidence[
            "selection_identity"
        ]
        report["stage2_baseline_report"] = release_evidence[
            "stage2_baseline_report"
        ]
        report["delta_validation"] = release_evidence["delta_validation"]
        report["health_diagnostics"] = {
            "tail_recall@5": last_stage4_recall_at_5,
            "stage4_act_loss_increase": stage4_act_loss_increase,
            "health_only_blockers": sorted(
                set(blockers).intersection(HEALTH_ONLY_BLOCKERS)
            ),
        }
        report["paper_scope"]["ready_for_phase_h_handoff"] = not bool(
            final_blockers
        )
    if output_path is not None:
        write_json_report(output_path, report)
    return report
