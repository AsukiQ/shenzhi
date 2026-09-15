from __future__ import annotations

from typing import Any


def _nested(report: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = report or {}
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def audit_clstr_promotion_readiness(
    *,
    stage2_report: dict[str, Any] | None,
    stage4_report: dict[str, Any] | None,
    belief_audit_report: dict[str, Any] | None,
    route_report: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    belief_semantics = _nested(stage2_report, "belief_memory_semantics")
    if not isinstance(belief_semantics, dict):
        blockers.append("missing_belief_memory_semantics")
    else:
        if belief_semantics.get("memory_source") != "skill_table.belief_logits_via_subspace_obs":
            blockers.append("stage2_memory_not_belief_logits_subspace_obs")
        if belief_semantics.get("stage1_stage2_train_eval_memory_consistent") is not True:
            blockers.append("stage2_train_eval_memory_not_consistent")
        if belief_semantics.get("belief_calibration_trainable") is not True:
            blockers.append("belief_calibration_not_trainable")

    policy_semantics = _nested(stage2_report, "policy_feature_semantics")
    if not isinstance(policy_semantics, dict):
        blockers.append("missing_policy_feature_semantics")
    elif policy_semantics.get("train_eval_default_aligned") is not True:
        blockers.append("policy_train_eval_default_not_aligned")

    retrieval_loss = _nested(stage2_report, "retrieval_loss")
    if isinstance(retrieval_loss, dict):
        if retrieval_loss.get("claim_scope") != "diagnostic_not_retrieval_improvement_evidence":
            blockers.append("routing_loss_not_marked_diagnostic")
    else:
        warnings.append("missing_retrieval_loss_claim_scope")

    if not isinstance(belief_audit_report, dict):
        blockers.append("missing_belief_audit_report")
    elif belief_audit_report.get("status") not in {"ok", "pass"}:
        blockers.append("belief_audit_not_ok")

    if isinstance(stage4_report, dict):
        injected = _nested(stage4_report, "data_report", "positive_injected_rows")
        if injected is None:
            injected = _nested(stage4_report, "train_report", "data_report", "positive_injected_rows")
        if int(injected or 0) > 0:
            blockers.append("stage4_positive_injected_rows_nonzero")
        retained_fraction = _nested(stage4_report, "data_report", "stage4_retained_fraction")
        if retained_fraction is None:
            retained_fraction = _nested(stage4_report, "train_report", "data_report", "stage4_retained_fraction")
        if retained_fraction is None:
            warnings.append("missing_stage4_retained_fraction")

    if isinstance(route_report, dict):
        metric_contract = route_report.get("metric_contract")
        if not isinstance(metric_contract, dict):
            blockers.append("missing_route_metric_contract")
        else:
            if metric_contract.get("primary_metric_scope") != "strict":
                blockers.append("route_primary_metric_not_strict")
            if metric_contract.get("retained_metrics_scope") != "diagnostic_only":
                blockers.append("route_retained_metrics_not_diagnostic")
        if "strict" not in route_report:
            blockers.append("missing_strict_route_metrics")

    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "warnings": warnings,
        "promotion_scope": "post_fix_final_clstr" if not blockers else "not_promotable",
    }
