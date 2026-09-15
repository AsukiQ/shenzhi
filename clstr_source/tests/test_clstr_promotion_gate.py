from __future__ import annotations

from clstr.promotion_gate import audit_clstr_promotion_readiness


def test_promotion_gate_blocks_legacy_stage2_without_belief_semantics():
    report = audit_clstr_promotion_readiness(
        stage2_report={
            "stage": "clstr_full_base_component_complete",
            "metric_averages": {"transition_skill_recall@1": 0.8},
        },
        stage4_report=None,
        belief_audit_report=None,
        route_report=None,
    )

    assert report["status"] == "action_required"
    assert "missing_belief_memory_semantics" in report["blockers"]
    assert "missing_belief_audit_report" in report["blockers"]


def test_promotion_gate_accepts_repaired_strict_artifacts():
    report = audit_clstr_promotion_readiness(
        stage2_report={
            "belief_memory_semantics": {
                "memory_source": "skill_table.belief_logits_via_subspace_obs",
                "stage1_stage2_train_eval_memory_consistent": True,
                "belief_calibration_params_present": True,
                "belief_calibration_trainable": True,
            },
            "policy_feature_semantics": {
                "train_eval_default_aligned": True,
            },
            "retrieval_loss": {
                "claim_scope": "diagnostic_not_retrieval_improvement_evidence",
            },
        },
        stage4_report={
            "data_report": {
                "positive_injected_rows": 0,
                "stage4_retained_fraction": 0.75,
            }
        },
        belief_audit_report={
            "status": "ok",
            "blockers": [],
        },
        route_report={
            "metric_contract": {
                "primary_metric_scope": "strict",
                "retained_metrics_scope": "diagnostic_only",
            },
            "strict": {"stage4": {"strict_stage4_next_skill_recall@1": 0.3}},
        },
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
