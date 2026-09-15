import json
import subprocess
import sys
from pathlib import Path

from clstr.stage2_quality_gate import (
    DEFAULT_REQUIRED_LOSS_TERMS,
    _anchored_routing_foundation_safety,
    _next_skill_pool_safety,
    _safe_memory_training_safety,
    audit_stage2_full_base_quality,
)


def test_stage2_canonical_quality_gate_does_not_require_stop_or_dead_losses():
    assert DEFAULT_REQUIRED_LOSS_TERMS == [
        "L_policy",
        "L_trans",
        "L_trans_skill_ce",
        "belief",
    ]


def test_anchored_routing_foundation_safety_accepts_static_anchor_and_dynamic_gain() -> None:
    safety = _anchored_routing_foundation_safety(
        {
            "qwen_and_skill_table_frozen": True,
            "trainable_modules": [
                "transition",
                "gate",
                "action_proj",
                "initial_belief_head",
                "unified_retriever",
                "skill_table.logit_scale_belief",
                "skill_table.skill_bias_belief",
            ],
            "stage2_static_route_anchor": {
                "enabled": True,
                "weight": 0.1,
                "teacher_digest": "teacher-digest",
                "row_count": 64,
                "static_mrr_delta_vs_teacher": -0.004,
                "dynamic_mrr_delta_vs_step_zero": 0.002,
            },
        },
        max_static_regression=0.005,
    )

    assert safety["matched"] is True
    assert safety["blockers"] == []


def test_anchored_routing_foundation_safety_rejects_static_drop_and_no_dynamic_gain() -> None:
    safety = _anchored_routing_foundation_safety(
        {
            "qwen_and_skill_table_frozen": False,
            "trainable_modules": ["transition", "initial_belief_head"],
            "stage2_static_route_anchor": {
                "enabled": True,
                "weight": 0.1,
                "teacher_digest": "teacher-digest",
                "row_count": 64,
                "static_mrr_delta_vs_teacher": -0.006,
                "dynamic_mrr_delta_vs_step_zero": 0.0,
            },
        },
        max_static_regression=0.005,
    )

    assert safety["matched"] is False
    assert "anchored_router_trainable_scope_mismatch" in safety["blockers"]
    assert "qwen_or_skill_table_geometry_not_frozen" in safety["blockers"]
    assert "stage2_static_route_anchor_regression" in safety["blockers"]
    assert "stage2_dynamic_route_not_improved" in safety["blockers"]


def test_stage2_quality_gate_accepts_anchored_router_with_frozen_qwen_geometry(
    tmp_path,
) -> None:
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step2.pt"
    report = _stage2_report(output_dir, checkpoint)
    report.update(
        {
            "route_scorer": "unified_memory",
            "uses_stage0_prior_at_inference": False,
            "frozen_routing_foundation": False,
            "qwen_and_skill_table_frozen": True,
            "transition_objective": "full_pool_causal_next_skill_nll",
            "trainable_modules": [
                "transition",
                "gate",
                "action_proj",
                "initial_belief_head",
                "unified_retriever",
                "skill_table.logit_scale_belief",
                "skill_table.skill_bias_belief",
            ],
            "stage2_static_route_anchor": {
                "enabled": True,
                "weight": 0.1,
                "teacher_digest": "teacher-digest",
                "row_count": 8,
                "static_mrr_delta_vs_teacher": -0.004,
                "dynamic_mrr_delta_vs_step_zero": 0.002,
            },
        }
    )
    report["transition_skill_ce"].update(
        {
            "scoring_mode": "unified_memory",
            "route_scorer": "unified_memory",
            "candidate_pool": "declared_legal_full_skill_pool",
        }
    )
    report["transition_candidate_training"].update(
        {
            "scoring_mode": "unified_memory",
            "route_scorer": "unified_memory",
            "next_skill_pool_mode": "full_pool",
            "inventory_mask_mode": "explicit_only",
            "counterfactual_utility_loss_weight": 0.0,
            "memory_utility_calibration_owner": "stage4_cmc",
        }
    )
    report["transition_input_semantics"].update(
        {"transition_scoring_mode": "unified_memory", "route_scorer": "unified_memory"}
    )
    report["loss_activation_counts"] = {
        "L_policy": 2,
        "L_trans": 2,
        "L_trans_skill_ce": 2,
        "belief": 2,
    }
    report["sampled_loss_activation_counts"] = dict(report["loss_activation_counts"])
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@5": 0.8,
                "route_scorer": "unified_memory",
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 2,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@5": 0.81,
                "route_scorer": "unified_memory",
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=2,
        first_window=1,
        last_window=1,
        min_loss_drop=0.2,
        min_transition_recall_at_5_tail=0.0,
        expected_route_scorer="unified_memory",
        expected_next_skill_pool_mode="full_pool",
        static_route_anchor_max_regression=0.005,
    )

    assert audit["status"] == "ok"
    assert "routing_foundation_not_frozen" not in audit["blockers"]
    assert audit["anchored_routing_foundation_safety"]["matched"] is True


def test_stage2_full_pool_contract_assigns_memory_calibration_to_stage4_cmc():
    safety = _next_skill_pool_safety(
        {
            "transition_objective": "full_pool_causal_next_skill_nll",
            "transition_skill_ce": {
                "candidate_pool": "declared_legal_full_skill_pool",
            },
            "transition_candidate_training": {
                "next_skill_pool_mode": "full_pool",
                "inventory_mask_mode": "explicit_only",
                "memory_utility_calibration_owner": "stage4_cmc",
            },
        },
        "full_pool",
    )

    assert safety["matched"] is True
    assert safety["counterfactual_utility_enabled"] is False
    assert safety["memory_utility_calibration_owner"] == "stage4_cmc"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stage2_report(output_dir: Path, checkpoint: Path) -> dict:
    metrics_path = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    latest = output_dir / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")
    loss_curve.write_text("<svg>rolling loss recall</svg>", encoding="utf-8")
    return {
        "status": "ok",
        "training_objective": "component_complete_masked_multi_loss",
        "training_regime": "offline_replay_supervised_pretraining",
        "transition_skill_ce": {
            "enabled": True,
            "scoring_mode": "stage0_rank_prior_plus_transition_residual",
            "residual_lambda": 0.25,
        },
        "checkpoint": str(checkpoint),
        "training_metrics_path": str(metrics_path),
        "loss_curve_path": str(loss_curve),
        "latest_checkpoint": str(latest),
        "max_steps": 6,
        "sample_count": 128,
        "skill_count": 16,
        "valid_or_test_used_for_training": False,
        "on_policy_rollout_used": False,
        "not_rl_fine_tuning": True,
        "frozen_routing_foundation": True,
        "loss_activation_counts": {
            "L_policy": 10,
            "L_trans": 12,
            "L_trans_skill_ce": 9,
            "STOP": 8,
            "routing": 0,
            "belief": 6,
        },
        "sampled_loss_activation_counts": {
            "L_policy": 3,
            "L_trans": 3,
            "L_trans_skill_ce": 2,
            "STOP": 3,
            "routing": 0,
            "belief": 2,
        },
        "metric_averages": {
            "loss": 1.4,
            "policy_loss": 0.5,
            "transition_skill_ce_loss": 0.4,
        },
        "transition_candidate_training": {
            "scoring_mode": "stage0_rank_prior_plus_transition_residual",
            "residual_lambda": 0.25,
        },
        "transition_input_semantics": {
            "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
            "transition_residual_lambda": 0.25,
        },
    }


def test_safe_memory_training_safety_requires_frozen_static_and_trainable_gate() -> None:
    report = {
        "trainable_modules": ["transition", "gate", "action_proj", "route_memory_utility_gate"],
        "transition_candidate_training": {
            "counterfactual_training_objective": "safe_local_candidate_rank_v1",
            "safe_memory_residual_bound": 2.0,
            "safe_local_candidate_sizes": [2, 3, 4, 5, 8, 10],
        },
    }

    safety = _safe_memory_training_safety(report)

    assert safety["matched"] is True
    assert safety["blockers"] == []


def test_safe_memory_training_safety_rejects_router_drift_and_missing_gate() -> None:
    report = {
        "trainable_modules": ["transition", "unified_retriever"],
        "transition_candidate_training": {
            "counterfactual_training_objective": "full_pool_log_utility",
            "safe_memory_residual_bound": 0.0,
            "safe_local_candidate_sizes": [],
        },
    }

    safety = _safe_memory_training_safety(report)

    assert safety["matched"] is False
    assert "route_memory_utility_gate_not_trainable" in safety["blockers"]
    assert "static_routing_foundation_trainable" in safety["blockers"]
    assert "safe_local_objective_mismatch" in safety["blockers"]


def test_stage2_quality_gate_accepts_complete_full_base_training(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "policy_loss": 0.8,
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@5": 0.8,
                "transition_prior_skill_recall@5": 0.78,
                "transition_skill_mrr": 0.5,
                "transition_prior_skill_mrr": 0.49,
                "transition_worse_than_stage0_prior_fraction": 0.2,
            },
            {
                "step": 2,
                "loss": 1.8,
                "policy_loss": 0.7,
                "transition_skill_ce_loss": 0.6,
                "transition_skill_recall@5": 0.8,
                "transition_prior_skill_recall@5": 0.78,
                "transition_skill_mrr": 0.5,
                "transition_prior_skill_mrr": 0.49,
                "transition_worse_than_stage0_prior_fraction": 0.2,
            },
            {
                "step": 3,
                "loss": 1.6,
                "policy_loss": 0.6,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@5": 0.82,
                "transition_prior_skill_recall@5": 0.79,
                "transition_skill_mrr": 0.52,
                "transition_prior_skill_mrr": 0.5,
                "transition_worse_than_stage0_prior_fraction": 0.18,
            },
            {
                "step": 4,
                "loss": 1.4,
                "policy_loss": 0.5,
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@5": 0.83,
                "transition_prior_skill_recall@5": 0.8,
                "transition_skill_mrr": 0.53,
                "transition_prior_skill_mrr": 0.5,
                "transition_worse_than_stage0_prior_fraction": 0.18,
            },
            {
                "step": 5,
                "loss": 1.2,
                "policy_loss": 0.4,
                "transition_skill_ce_loss": 0.3,
                "transition_skill_recall@5": 0.84,
                "transition_prior_skill_recall@5": 0.8,
                "transition_skill_mrr": 0.54,
                "transition_prior_skill_mrr": 0.5,
                "transition_worse_than_stage0_prior_fraction": 0.18,
            },
            {
                "step": 6,
                "loss": 1.0,
                "policy_loss": 0.3,
                "transition_skill_ce_loss": 0.2,
                "transition_skill_recall@5": 0.85,
                "transition_prior_skill_recall@5": 0.8,
                "transition_skill_mrr": 0.55,
                "transition_prior_skill_mrr": 0.5,
                "transition_worse_than_stage0_prior_fraction": 0.18,
            },
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
    )

    assert audit["status"] == "ok"
    assert audit["blockers"] == []
    assert audit["metrics_summary"]["loss_drop"] == 0.8
    assert audit["required_loss_terms"]["missing_or_unsampled"] == []
    assert audit["transition_quality"]["transition_skill_recall@5_drop"] <= 0.0
    assert audit["artifacts"]["latest_checkpoint_exists"] is True
    assert audit["artifacts"]["loss_curve_exists"] is True


def test_stage2_quality_gate_accepts_expected_v4_1b_transition_scoring(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    for section_name, mode_key, lambda_key in (
        ("transition_skill_ce", "scoring_mode", "residual_lambda"),
        ("transition_candidate_training", "scoring_mode", "residual_lambda"),
        ("transition_input_semantics", "transition_scoring_mode", "transition_residual_lambda"),
    ):
        report[section_name][mode_key] = "v4_1b_action_observation"
        report[section_name][lambda_key] = 0.0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.7, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.6, "transition_skill_recall@5": 0.81},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.5, "transition_skill_recall@5": 0.82},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.4, "transition_skill_recall@5": 0.83},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.3, "transition_skill_recall@5": 0.84},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.2, "transition_skill_recall@5": 0.85},
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        expected_transition_scoring_mode="v4_1b_action_observation",
        expected_transition_residual_lambda=0.0,
    )

    assert audit["status"] == "ok"
    assert audit["transition_quality"]["transition_scoring_safety"]["matched"] is True


def test_stage2_quality_gate_accepts_unified_memory_without_stage0_prior_delta_metrics(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    report["route_scorer"] = "unified_memory"
    report["uses_stage0_prior_at_inference"] = False
    report["transition_skill_ce"]["scoring_mode"] = "unified_memory"
    report["transition_skill_ce"]["route_scorer"] = "unified_memory"
    report["transition_candidate_training"]["scoring_mode"] = "unified_memory"
    report["transition_candidate_training"]["route_scorer"] = "unified_memory"
    report["transition_input_semantics"]["transition_scoring_mode"] = "unified_memory"
    report["transition_input_semantics"]["route_scorer"] = "unified_memory"
    report["transition_objective"] = "full_pool_causal_next_skill_with_counterfactual_utility"
    report["transition_skill_ce"]["candidate_pool"] = "declared_legal_full_skill_pool"
    report["transition_candidate_training"].update(
        {
            "next_skill_pool_mode": "full_pool",
            "inventory_mask_mode": "explicit_only",
            "counterfactual_utility_loss_weight": 0.05,
            "counterfactual_training_objective": "safe_local_candidate_rank_v1",
            "safe_memory_residual_bound": 2.0,
            "safe_local_candidate_sizes": [2, 3, 4, 5, 8, 10],
        }
    )
    report["trainable_modules"] = [
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ]
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@5": 0.8,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 2,
                "loss": 1.8,
                "transition_skill_ce_loss": 0.6,
                "transition_skill_recall@5": 0.81,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 3,
                "loss": 1.6,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@5": 0.82,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 4,
                "loss": 1.4,
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@5": 0.83,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 5,
                "loss": 1.2,
                "transition_skill_ce_loss": 0.3,
                "transition_skill_recall@5": 0.84,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
            {
                "step": 6,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.2,
                "transition_skill_recall@5": 0.85,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever_full_pool",
                "transition_scoring_mode": "unified_memory",
            },
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        expected_route_scorer="unified_memory",
        expected_next_skill_pool_mode="full_pool",
    )

    assert audit["status"] == "ok"
    assert audit["stage0_prior_comparison"]["required"] is False
    assert audit["route_scorer_safety"]["matched"] is True
    assert audit["next_skill_pool_safety"]["matched"] is True
    assert audit["next_skill_pool_safety"]["counterfactual_utility_enabled"] is True


def test_stage2_quality_gate_rejects_candidate_limited_unified_checkpoint_when_full_pool_expected(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    report["route_scorer"] = "unified_memory"
    report["uses_stage0_prior_at_inference"] = False
    report["transition_skill_ce"].update(
        {
            "scoring_mode": "unified_memory",
            "route_scorer": "unified_memory",
            "candidate_pool": "stage0_topm_candidate_set",
        }
    )
    report["transition_candidate_training"].update(
        {
            "scoring_mode": "unified_memory",
            "route_scorer": "unified_memory",
            "next_skill_pool_mode": "stage0_candidates",
            "inventory_mask_mode": "explicit_only",
            "counterfactual_utility_loss_weight": 0.05,
        }
    )
    report["transition_input_semantics"].update(
        {"transition_scoring_mode": "unified_memory", "route_scorer": "unified_memory"}
    )
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 2.2 - 0.2 * step,
                "transition_skill_ce_loss": 0.8 - 0.1 * step,
                "transition_skill_recall@5": 0.75 + 0.01 * step,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_scoring_mode": "unified_memory",
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        expected_route_scorer="unified_memory",
        expected_next_skill_pool_mode="full_pool",
    )

    assert audit["status"] == "action_required"
    assert "stage2_next_skill_pool_mismatch" in audit["blockers"]
    assert audit["next_skill_pool_safety"]["matched"] is False


def test_stage2_quality_gate_rejects_legacy_transition_scoring_metadata(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    report["transition_skill_ce"] = {"enabled": True, "scoring_mode": "action_observation_concat"}
    report["transition_candidate_training"].pop("residual_lambda")
    report["transition_input_semantics"].pop("transition_residual_lambda")
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "policy_loss": 0.8, "transition_skill_ce_loss": 0.7},
            {"step": 2, "loss": 1.8, "policy_loss": 0.7, "transition_skill_ce_loss": 0.6, "transition_skill_recall@5": 0.8},
            {"step": 3, "loss": 1.6, "policy_loss": 0.6, "transition_skill_ce_loss": 0.5, "transition_skill_recall@5": 0.82},
            {"step": 4, "loss": 1.4, "policy_loss": 0.5, "transition_skill_ce_loss": 0.4, "transition_skill_recall@5": 0.83},
            {"step": 5, "loss": 1.2, "policy_loss": 0.4, "transition_skill_ce_loss": 0.3, "transition_skill_recall@5": 0.84},
            {"step": 6, "loss": 1.0, "policy_loss": 0.3, "transition_skill_ce_loss": 0.2, "transition_skill_recall@5": 0.85},
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
    )

    assert audit["status"] == "action_required"
    assert "stage2_transition_scoring_not_prior_residual" in audit["blockers"]


def test_stage2_quality_gate_rejects_checkpoint_without_learning_signal_or_policy_terms(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    report["valid_or_test_used_for_training"] = True
    report["loss_activation_counts"]["L_policy"] = 0
    report["sampled_loss_activation_counts"]["L_trans_skill_ce"] = 0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [{"step": step, "loss": 2.0} for step in range(1, 7)],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
    )

    assert audit["status"] == "action_required"
    assert "valid_or_test_used_for_training" in audit["blockers"]
    assert "insufficient_learning_signal" in audit["blockers"]
    assert "missing_required_loss_terms" in audit["blockers"]
    assert audit["required_loss_terms"]["missing_or_unsampled"] == ["L_policy", "L_trans_skill_ce"]


def test_stage2_quality_gate_rejects_transition_metric_regression_even_when_total_loss_drops(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@5": 0.95},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@5": 0.93},
            {"step": 3, "loss": 1.5, "transition_skill_ce_loss": 1.0, "transition_skill_recall@5": 0.9},
            {"step": 4, "loss": 1.2, "transition_skill_ce_loss": 1.1, "transition_skill_recall@5": 0.7},
            {"step": 5, "loss": 1.0, "transition_skill_ce_loss": 1.4, "transition_skill_recall@5": 0.6},
            {"step": 6, "loss": 0.8, "transition_skill_ce_loss": 1.5, "transition_skill_recall@5": 0.55},
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        max_transition_recall_at_5_drop=0.1,
        max_transition_skill_ce_increase=0.2,
    )

    assert audit["status"] == "action_required"
    assert "transition_recall_at_5_regressed" in audit["blockers"]
    assert "transition_skill_ce_regressed" in audit["blockers"]
    assert audit["transition_quality"]["transition_skill_recall@5_drop"] > 0.1
    assert audit["transition_quality"]["transition_skill_ce_increase"] > 0.2


def test_stage2_quality_gate_rejects_stage2_below_stage0_prior(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    report = _stage2_report(output_dir, checkpoint)
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.8,
                "transition_skill_recall@5": 0.3,
                "transition_prior_skill_recall@5": 0.6,
                "transition_skill_mrr": 0.2,
                "transition_prior_skill_mrr": 0.4,
                "transition_worse_than_stage0_prior_fraction": 0.8,
            },
            {
                "step": 2,
                "loss": 1.8,
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@5": 0.3,
                "transition_prior_skill_recall@5": 0.6,
                "transition_skill_mrr": 0.2,
                "transition_prior_skill_mrr": 0.4,
                "transition_worse_than_stage0_prior_fraction": 0.8,
            },
            {
                "step": 3,
                "loss": 1.6,
                "transition_skill_ce_loss": 0.6,
                "transition_skill_recall@5": 0.31,
                "transition_prior_skill_recall@5": 0.61,
                "transition_skill_mrr": 0.21,
                "transition_prior_skill_mrr": 0.41,
                "transition_worse_than_stage0_prior_fraction": 0.78,
            },
            {
                "step": 4,
                "loss": 1.4,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@5": 0.32,
                "transition_prior_skill_recall@5": 0.62,
                "transition_skill_mrr": 0.22,
                "transition_prior_skill_mrr": 0.42,
                "transition_worse_than_stage0_prior_fraction": 0.76,
            },
            {
                "step": 5,
                "loss": 1.2,
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@5": 0.32,
                "transition_prior_skill_recall@5": 0.62,
                "transition_skill_mrr": 0.22,
                "transition_prior_skill_mrr": 0.42,
                "transition_worse_than_stage0_prior_fraction": 0.76,
            },
            {
                "step": 6,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.3,
                "transition_skill_recall@5": 0.32,
                "transition_prior_skill_recall@5": 0.62,
                "transition_skill_mrr": 0.22,
                "transition_prior_skill_mrr": 0.42,
                "transition_worse_than_stage0_prior_fraction": 0.76,
            },
        ],
    )

    audit = audit_stage2_full_base_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
    )

    assert audit["status"] == "action_required"
    assert "stage2_recall_at_5_below_stage0_prior" in audit["blockers"]
    assert "stage2_mrr_below_stage0_prior" in audit["blockers"]
    assert "stage2_worse_than_stage0_prior_too_high" in audit["blockers"]


def test_stage2_quality_gate_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_dir = tmp_path / "stage2"
    checkpoint = output_dir / "checkpoints/clstr_full_base-step6.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    _write_json(output_dir / "train_report.json", {"status": "ok", "checkpoint": str(checkpoint)})
    _write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0}])
    output_path = tmp_path / "stage2_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage2_quality.py",
            "--output_dir",
            str(output_dir),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output_path),
            "--min_steps",
            "6",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "action_required"
    assert "insufficient_training_steps" in written["blockers"]


def test_stage2_quality_gate_cli_defaults_to_action_aware_progressive_mainline():
    script = Path("scripts/audit_clstr_stage2_quality.py").read_text(encoding="utf-8")

    assert (
        'default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"'
        in script
    )
    assert "clstr_unified_stage2_v3_traj_retrieval_full_base" not in script
    assert "--max_stage2_recall_at_5_drop_vs_stage0_prior" in script
    assert "max_stage2_recall_at_5_drop_vs_stage0_prior=args.max_stage2_recall_at_5_drop_vs_stage0_prior" in script
    assert "--max_stage2_mrr_drop_vs_stage0_prior" in script
    assert "max_stage2_mrr_drop_vs_stage0_prior=args.max_stage2_mrr_drop_vs_stage0_prior" in script
    assert "--max_stage2_worse_than_stage0_prior_fraction" in script
    assert "max_stage2_worse_than_stage0_prior_fraction=args.max_stage2_worse_than_stage0_prior_fraction" in script
