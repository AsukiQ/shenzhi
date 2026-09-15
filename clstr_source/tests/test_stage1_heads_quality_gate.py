import json
import subprocess
import sys
from pathlib import Path

from clstr.stage1_heads_quality_gate import (
    DEFAULT_REQUIRED_LOSS_TERMS,
    STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
    audit_stage1_heads_quality,
)


def test_stage1_canonical_quality_gate_does_not_require_stop_or_dead_losses():
    assert DEFAULT_REQUIRED_LOSS_TERMS == [
        "L_policy",
        "L_trans",
        "L_trans_skill_ce",
        "belief",
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stage1_report(output_dir: Path, checkpoint: Path) -> dict:
    metrics_path = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    latest = output_dir / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")
    loss_curve.write_text("<svg>stage1 loss</svg>", encoding="utf-8")
    return {
        "status": "ok",
        "stage": "clstr_stage1_heads_init",
        "training_objective": "stage1_heads_init_topm_supervised",
        "checkpoint": str(checkpoint),
        "training_metrics_path": str(metrics_path),
        "loss_curve_path": str(loss_curve),
        "latest_checkpoint": str(latest),
        "max_steps": 6,
        "frozen_routing_foundation": True,
        "checkpoint_excludes_frozen_routing_foundation": True,
        "excluded_state_key_prefixes": ["encoder.", "cross_encoder.", "skill_table."],
        "transition_input_semantics": {
            "action_channel": "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding",
            "observation_channel": "next_observation_text_embedding",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": True,
            "action_text_used_as_observation": False,
            "transition_scoring_mode": STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
            "transition_residual_lambda": 0.25,
        },
        "stage0_candidate_handoff": {
            "candidate_source": "stage0_topm_online",
            "positive_missing_policy": "skip",
            "injected_positive_rows": 0,
        },
        "loss_activation_counts": {
            "L_policy": 6,
            "L_trans": 6,
            "L_trans_skill_ce": 6,
            "STOP": 6,
            "belief": 6,
        },
        "sampled_loss_activation_counts": {
            "L_policy": 6,
            "L_trans": 6,
            "L_trans_skill_ce": 6,
            "STOP": 6,
            "belief": 6,
        },
        "loss_weights": {
            "L_policy": 0.7,
            "L_trans": 0.2,
            "L_trans_skill_ce": 0.5,
            "STOP": 0.1,
            "belief": 0.1,
            "transition_hard_negative_margin": 0.0,
        },
        "transition_candidate_training": {
            "hard_negative_loss_weight": 0.0,
            "hard_negative_margin": 1.0,
            "scoring_mode": STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
            "residual_lambda": 0.25,
        },
        "metrics": {
            "transition_hard_negative_pair_count": 2.0,
        },
}


def test_stage1_heads_quality_gate_rejects_legacy_transition_scoring_metadata(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["transition_input_semantics"].pop("transition_scoring_mode")
    report["transition_input_semantics"].pop("transition_residual_lambda")
    report["transition_candidate_training"]["scoring_mode"] = "action_observation_concat"
    report["transition_candidate_training"].pop("residual_lambda")
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "stage1_transition_scoring_mismatch" in audit["blockers"]


def test_stage1_heads_quality_gate_accepts_historical_topm_metrics_without_scoring_metadata(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report.pop("transition_input_semantics")
    report["transition_candidate_training"] = {
        "hard_negative_loss_weight": 0.0,
        "hard_negative_margin": 1.0,
        "inventory_mask_mode": "auto",
        "inventory_min_candidates": 64,
        "loss_type": "listwise_nll",
        "positive_mode": "gold_plus_equivalent",
    }
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.8,
                "transition_skill_recall@1": 0.55,
                "transition_skill_recall@5": 0.8,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
            {
                "step": 2,
                "loss": 1.8,
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@1": 0.56,
                "transition_skill_recall@5": 0.82,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
            {
                "step": 3,
                "loss": 1.6,
                "transition_skill_ce_loss": 0.6,
                "transition_skill_recall@1": 0.57,
                "transition_skill_recall@5": 0.84,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
            {
                "step": 4,
                "loss": 1.4,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@1": 0.58,
                "transition_skill_recall@5": 0.85,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
            {
                "step": 5,
                "loss": 1.2,
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@1": 0.59,
                "transition_skill_recall@5": 0.86,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
            {
                "step": 6,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.3,
                "transition_skill_recall@1": 0.6,
                "transition_skill_recall@5": 0.87,
                "transition_skill_head_type": "native_trans_head_stage0_topm",
                "transition_loss_type": "listwise_nll",
                "transition_positive_mode": "gold_plus_equivalent",
            },
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "ok"
    assert audit["transition_input_safety"]["historical_stage1_topm_metadata_compatible"] is True


def test_stage1_heads_quality_gate_accepts_healthy_transition_ranking(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    _write_json(output_dir / "train_report.json", _stage1_report(output_dir, checkpoint))
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "ok"
    assert audit["blockers"] == []
    assert audit["transition_quality"]["last_transition_skill_recall@1"] >= 0.5
    assert audit["transition_quality"]["last_transition_skill_recall@5"] >= 0.75


def test_stage1_heads_quality_gate_rejects_transition_regression_even_when_loss_drops(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    _write_json(output_dir / "train_report.json", _stage1_report(output_dir, checkpoint))
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.45, "transition_skill_recall@5": 0.95},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.44, "transition_skill_recall@5": 0.93},
            {"step": 3, "loss": 1.5, "transition_skill_ce_loss": 1.0, "transition_skill_recall@1": 0.4, "transition_skill_recall@5": 0.85},
            {"step": 4, "loss": 1.2, "transition_skill_ce_loss": 1.1, "transition_skill_recall@1": 0.35, "transition_skill_recall@5": 0.75},
            {"step": 5, "loss": 1.0, "transition_skill_ce_loss": 1.4, "transition_skill_recall@1": 0.25, "transition_skill_recall@5": 0.6},
            {"step": 6, "loss": 0.8, "transition_skill_ce_loss": 1.5, "transition_skill_recall@1": 0.2, "transition_skill_recall@5": 0.55},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        max_transition_recall_at_5_drop=0.1,
        max_transition_skill_ce_increase=0.2,
        min_transition_recall_at_1_tail=0.35,
        min_transition_recall_at_5_tail=0.7,
    )

    assert audit["status"] == "action_required"
    assert "transition_recall_at_1_too_low" in audit["blockers"]
    assert "transition_recall_at_5_too_low" in audit["blockers"]
    assert "transition_recall_at_5_regressed" in audit["blockers"]
    assert "transition_skill_ce_regressed" in audit["blockers"]


def test_stage1_heads_quality_gate_requires_head_checkpoint_to_exclude_routing_foundation(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["checkpoint_excludes_frozen_routing_foundation"] = False
    report["excluded_state_key_prefixes"] = []
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "stage1_checkpoint_includes_frozen_routing_foundation" in audit["blockers"]


def test_stage1_heads_quality_gate_requires_action_aware_transition_input_semantics(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report.pop("transition_input_semantics")
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "stage1_transition_input_semantics_not_action_aware" in audit["blockers"]


def test_stage1_heads_quality_gate_rejects_injected_stage0_positives(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["stage0_candidate_handoff"] = {
        "candidate_source": "stage0_topm_online",
        "positive_missing_policy": "inject",
        "injected_positive_rows": 2,
    }
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "stage1_positive_missing_policy_not_skip" in audit["blockers"]
    assert "stage1_injected_stage0_positives" in audit["blockers"]


def test_stage1_heads_quality_gate_accepts_disabled_transition_hard_negative_margin(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "ok"
    assert "transition_hard_negative_margin_not_enabled" not in audit["blockers"]
    assert audit["transition_hard_negative_safety"]["hard_negative_loss_weight"] == 0.0


def test_stage1_heads_quality_gate_requires_canonical_heads_init_loss_terms(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    for key in ("L_trans", "STOP", "belief"):
        report["loss_activation_counts"][key] = 0
        report["sampled_loss_activation_counts"][key] = 0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "missing_required_loss_terms" in audit["blockers"]
    assert audit["required_loss_terms"]["missing_or_unsampled"] == ["L_trans", "belief"]


def test_stage1_heads_quality_gate_accepts_unified_no_replay_without_belief_loss(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["route_scorer"] = "unified_memory"
    report["uses_stage0_prior_at_inference"] = False
    report["auto_replay_prefix"] = {
        "enabled": False,
        "max_steps": 0,
        "row_count": 12,
        "rows_with_auto_replay_prefix": 0,
        "total_prefix_steps": 0,
    }
    report["trainable_replay_prefix"] = False
    report["transition_candidate_training"]["route_scorer"] = "unified_memory"
    report["transition_input_semantics"] = {
        "route_scorer": "unified_memory",
        "score_function": "unified_route_logits(h_t, m_t)",
        "static_memory": "initial_belief(h_t)",
        "dynamic_memory": "replay_prefix_belief_when_prefix_available_else_initial_belief",
        "action_channel": "actual_action_text_embedding_used_only_for_m_t_update_when_replay_prefix_available",
        "observation_channel": "next_observation_text_embedding_used_only_for_transition_auxiliary_or_m_t_update",
        "uses_actual_action_text_when_available": True,
        "uses_next_observation_as_observation": True,
        "action_text_used_as_observation": False,
        "transition_scoring_mode": "unified_memory",
        "transition_residual_lambda": 0.0,
        "transition_prior_branch": "disabled_as_main_score_in_unified_memory",
        "transition_residual_branch": "disabled_as_main_score_in_unified_memory",
    }
    report["loss_activation_counts"]["belief"] = 0
    report["sampled_loss_activation_counts"]["belief"] = 0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.8,
                "transition_skill_recall@1": 0.55,
                "transition_skill_recall@5": 0.8,
            },
            {
                "step": 2,
                "loss": 1.8,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.7,
                "transition_skill_recall@1": 0.56,
                "transition_skill_recall@5": 0.82,
            },
            {
                "step": 3,
                "loss": 1.6,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.6,
                "transition_skill_recall@1": 0.57,
                "transition_skill_recall@5": 0.84,
            },
            {
                "step": 4,
                "loss": 1.4,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@1": 0.58,
                "transition_skill_recall@5": 0.85,
            },
            {
                "step": 5,
                "loss": 1.2,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.4,
                "transition_skill_recall@1": 0.59,
                "transition_skill_recall@5": 0.86,
            },
            {
                "step": 6,
                "loss": 1.0,
                "route_scorer": "unified_memory",
                "uses_stage0_prior_at_inference": False,
                "transition_skill_head_type": "unified_memory_retriever",
                "transition_skill_ce_loss": 0.3,
                "transition_skill_recall@1": 0.6,
                "transition_skill_recall@5": 0.87,
            },
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
        expected_route_scorer="unified_memory",
    )

    assert audit["status"] == "ok"
    assert audit["required_loss_terms"]["missing_or_unsampled"] == []
    assert audit["route_scorer_safety"]["matched"] is True


def test_stage1_heads_quality_gate_rejects_inconsistent_transition_hard_negative_report(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["loss_weights"]["transition_hard_negative_margin"] = 0.1
    report["transition_candidate_training"]["hard_negative_loss_weight"] = 0.0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "transition_hard_negative_margin_inconsistent" in audit["blockers"]
    assert audit["transition_hard_negative_safety"]["loss_weight"] == 0.1
    assert audit["transition_hard_negative_safety"]["candidate_training_loss_weight"] == 0.0


def test_stage1_heads_quality_gate_requires_transition_hard_negative_pairs(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    report = _stage1_report(output_dir, checkpoint)
    report["loss_weights"]["transition_hard_negative_margin"] = 0.1
    report["transition_candidate_training"]["hard_negative_loss_weight"] = 0.1
    report["metrics"]["transition_hard_negative_pair_count"] = 0.0
    _write_json(output_dir / "train_report.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "transition_skill_ce_loss": 0.8, "transition_skill_recall@1": 0.55, "transition_skill_recall@5": 0.8},
            {"step": 2, "loss": 1.8, "transition_skill_ce_loss": 0.7, "transition_skill_recall@1": 0.56, "transition_skill_recall@5": 0.82},
            {"step": 3, "loss": 1.6, "transition_skill_ce_loss": 0.6, "transition_skill_recall@1": 0.57, "transition_skill_recall@5": 0.84},
            {"step": 4, "loss": 1.4, "transition_skill_ce_loss": 0.5, "transition_skill_recall@1": 0.58, "transition_skill_recall@5": 0.85},
            {"step": 5, "loss": 1.2, "transition_skill_ce_loss": 0.4, "transition_skill_recall@1": 0.59, "transition_skill_recall@5": 0.86},
            {"step": 6, "loss": 1.0, "transition_skill_ce_loss": 0.3, "transition_skill_recall@1": 0.6, "transition_skill_recall@5": 0.87},
        ],
    )

    audit = audit_stage1_heads_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_transition_recall_at_1_tail=0.5,
        min_transition_recall_at_5_tail=0.75,
    )

    assert audit["status"] == "action_required"
    assert "transition_hard_negative_pairs_missing" in audit["blockers"]
    assert audit["transition_hard_negative_safety"]["hard_negative_pair_count"] == 0.0


def test_stage1_heads_quality_gate_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints/clstr_stage1_heads-step6.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    _write_json(output_dir / "train_report.json", {"status": "ok", "checkpoint": str(checkpoint)})
    _write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0}])
    output_path = tmp_path / "stage1_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage1_heads_quality.py",
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


def test_stage1_heads_quality_gate_cli_defaults_to_action_aware_progressive_mainline():
    script = Path("scripts/audit_clstr_stage1_heads_quality.py").read_text(encoding="utf-8")
    from clstr.stage1_heads_quality_gate import DEFAULT_REQUIRED_LOSS_TERMS

    assert (
        'default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"'
        in script
    )
    assert "clstr_unified_stage1_v3_traj_retrieval_heads_init" not in script
    assert DEFAULT_REQUIRED_LOSS_TERMS == ["L_policy", "L_trans", "L_trans_skill_ce", "belief"]
