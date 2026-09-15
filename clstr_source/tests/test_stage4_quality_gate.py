import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from clstr.stage4_quality_gate import (
    _candidate_admission_contract_blockers,
    _selected_release_blockers,
    audit_stage4_act_quality,
)


def test_candidate_admission_quality_contract_rejects_signal_and_lineage_drift() -> None:
    selection = {
        "stage4_method": "candidate_admission_residual_v1",
        "promotion_status": "promoted",
        "candidate_admission_selection": {
            "balanced_macro_mrr": 0.64,
            "static_balanced_macro_mrr": 0.60,
            "admission_auprc": 0.50,
            "admission_prevalence": 0.25,
            "dynamic_extra_positive_count": 4,
            "no_regret_margin_violation_rate": 0.0,
            "no_regret_tolerance": 0.01,
        },
        "gradient_health": {"finite": True, "nonzero": True},
    }
    checkpoint = {
        "stage4_method": "candidate_admission_residual_v1",
        "base_cmc_checkpoint_sha256": "a" * 64,
        "parent_stage2_full_router_digest": "full",
        "parent_stage2_fast_router_digest": "fast",
    }
    train_report = {
        "stage4_method": "candidate_admission_residual_v1",
        "training_objective": "candidate_admission_constrained_residual",
        "base_cmc_checkpoint_sha256": "a" * 64,
        "freeze_report": {
            "trainable_modules": ["route_memory_candidate_admission_residual"],
            "frozen_cmc_residual_adapter": True,
        },
    }
    router_integrity = {
        "full_router_digest": "full",
        "fast_router_digest": "fast",
        "full_router_digest_unchanged": True,
        "fast_router_digest_unchanged": True,
    }

    assert _candidate_admission_contract_blockers(
        selection=selection,
        checkpoint_payload=checkpoint,
        train_report=train_report,
        router_integrity=router_integrity,
    ) == []

    bad_signal = dict(selection)
    bad_signal["candidate_admission_selection"] = {
        **selection["candidate_admission_selection"],
        "admission_auprc": 0.25,
    }
    assert "stage4_candidate_admission_signal_not_better_than_prevalence" in (
        _candidate_admission_contract_blockers(
            selection=bad_signal,
            checkpoint_payload=checkpoint,
            train_report=train_report,
            router_integrity=router_integrity,
        )
    )

    bad_lineage = dict(checkpoint)
    bad_lineage["base_cmc_checkpoint_sha256"] = "b" * 64
    assert "stage4_candidate_admission_base_cmc_mismatch" in (
        _candidate_admission_contract_blockers(
            selection=selection,
            checkpoint_payload=bad_lineage,
            train_report=train_report,
            router_integrity=router_integrity,
        )
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stage4_report(output_dir: Path, checkpoint: Path) -> dict:
    metrics_path = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    latest = output_dir / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")
    loss_curve.write_text("<svg>stage4 loss</svg>", encoding="utf-8")
    return {
        "status": "ok",
        "stage": "clstr_stage4_transition_conditioned_next_skill",
        "training_objective": "full_pool_causal_next_skill_with_counterfactual_utility",
        "training_regime": "offline_train_split_causal_next_skill",
        "checkpoint": str(checkpoint),
        "training_metrics_path": str(metrics_path),
        "loss_curve_path": str(loss_curve),
        "latest_checkpoint": str(latest),
        "route_scorer": "unified_memory",
        "next_skill_pool_mode": "full_pool",
        "counterfactual_utility_weight": 0.05,
        "counterfactual_gain_margin": 0.1,
        "counterfactual_safety_tolerance": 0.01,
        "counterfactual_gain_weight": 1.0,
        "counterfactual_safety_weight": 1.0,
        "counterfactual_warmup_fraction": 0.05,
        "transition_scoring_mode": "unified_memory",
        "transition_residual_lambda": 0.0,
        "valid_or_test_used_for_training": False,
        "on_policy_rollout_used": False,
        "data_report": {
            "stage4_rows": 24,
            "candidate_source": "declared_legal_full_skill_pool",
            "static_candidate_source": "stage0_topm_online",
            "next_skill_pool_mode": "full_pool",
            "positive_injected_rows": 0,
            "stage0_candidate_rows": 24,
            "stage0_next_static_hit_rows": 18,
            "stage0_next_static_miss_retained_rows": 6,
            "stage0_candidate_handoff": {
                "enabled": True,
                "candidate_source": "stage0_topm_online",
                "top_m": 350,
                "positive_missing_policy": "skip",
                "query_mode": "skillrouter_state",
                "injected_positive_rows": 0,
            },
            "benchmark_filter_enabled": True,
            "benchmark_counts": {"alfworld": 6, "toolbench_g3": 6, "traject_bench": 6, "webshop": 6},
        },
        "freeze_report": {
            "frozen_routing_foundation": True,
            "frozen_belief_gate": False,
            "train_transition": True,
            "trainable_modules": [
                "initial_belief_head",
                "transition",
                "gate",
                "action_proj",
                "unified_retriever",
            ],
        },
        "checkpoint_init_report": {
            "partial_load_mode": "stage0_routing_plus_act_init_heads_for_joint_stage4",
            "protect_routing_foundation": True,
            "head_overrides_routing_keys": ["transition.obs_proj.weight"],
            "skipped_head_routing_foundation_keys": ["skill_table.E"],
        },
        "last_metrics": {
            "loss": 0.7,
            "stage4_act_loss": 0.7,
            "stage4_total_loss": 0.71,
            "stage4_act_count": 4.0,
            "stage4_full_pool_rows": 4.0,
            "stage4_full_pool_eligible_rows": 4.0,
            "stage4_counterfactual_utility_loss": 0.2,
            "stage4_counterfactual_eligible_rows": 4.0,
            "stage4_counterfactual_weak_rows": 2.0,
            "stage4_counterfactual_strong_rows": 2.0,
            "stage4_counterfactual_gain_violation_rows": 1.0,
            "stage4_counterfactual_safety_violation_rows": 0.0,
            "stage4_static_hit_train_rows": 3.0,
            "stage4_static_miss_train_rows": 1.0,
            "stage4_next_skill_recall@1": 0.5,
            "stage4_next_skill_recall@5": 1.0,
            "route_scorer": "unified_memory",
            "transition_scoring_mode": "unified_memory",
            "transition_residual_lambda": 0.0,
            "stage4_post_action_update_rows": 4.0,
            "stage4_next_state_rows": 4.0,
        },
    }


def _write_safe_selection_artifacts(
    output_dir: Path,
    report: dict,
) -> tuple[Path, Path, Path]:
    checkpoint = output_dir / "checkpoints" / "clstr_stage4_safe-step800.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
                "route_memory_utility_gate.output.weight": torch.zeros(1, 1),
            },
            "train_report": report,
        },
        checkpoint,
    )
    baseline_path = output_dir / "validation" / "stage2_baseline_report.json"
    baseline_payload = {
        "schema_version": "stage4_stage2_baseline_v1",
        "status": "ok",
        "stage2_baseline": {
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.60,
            "dynamic_macro_mrr": 0.62,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60, "dynamic_mrr": 0.62}
                for benchmark in (
                    "toolbench_g3",
                    "traject_bench",
                    "alfworld",
                    "webshop",
                )
            },
        },
    }
    baseline_payload["manifest_sha256"] = canonical_digest(baseline_payload)
    _write_json(baseline_path, baseline_payload)
    stage2_baseline = {
        **baseline_payload["stage2_baseline"],
        "report_path": str(baseline_path.resolve()),
        "report_sha256": sha256_path(baseline_path)["sha256"],
        "report_manifest_sha256": baseline_payload["manifest_sha256"],
    }
    validation_path = output_dir / "validation" / "step800.json"
    validation_payload = {
        "schema_version": "stage4_validation_report_v1",
        "status": "ok",
        "release_status": "ok",
        "step": 800,
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": stage2_baseline,
        "candidate_union": {"static_k": 500, "dynamic_extra_k": 64, "final_k": 64},
        "balanced_macro": {
            "dynamic_mrr": 0.68,
            "dynamic_minus_static_mrr": 0.06,
            "dynamic_recall@5": 0.80,
        },
        "by_benchmark": {
            benchmark: {
                "static_mrr": 0.62,
                "dynamic_mrr": 0.68,
                "dynamic_minus_static_mrr": 0.06,
            }
            for benchmark in (
                "toolbench_g3",
                "traject_bench",
                "alfworld",
                "webshop",
            )
        },
    }
    validation_payload["manifest_sha256"] = canonical_digest(validation_payload)
    _write_json(validation_path, validation_payload)
    reliability = {
        "mode": "fixed_alpha",
        "fixed_alpha": 0.5,
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    selection = {
        "schema_version": "stage4_selection_v1",
        "status": "ok",
        "release_status": "ok",
        "dynamic_release_status": "ok",
        "release_checks": {
            "dynamic_release_ok": True,
            "fused_macro_improvement": True,
            "fused_per_benchmark_nonregression": True,
            "zero_history_exact": True,
        },
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(validation_path.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_path)["sha256"],
        "router_integrity": validation_payload["router_integrity"],
        "stage2_baseline": stage2_baseline,
        "candidate_union": validation_payload["candidate_union"],
        "reliability": reliability,
        "reliability_validation": {
            "balanced_macro_mrr": 0.64,
            "zero_history_exact": True,
            "by_benchmark": {
                benchmark: {"mrr": 0.64}
                for benchmark in (
                    "toolbench_g3",
                    "traject_bench",
                    "alfworld",
                    "webshop",
                )
            },
        },
    }
    selection["manifest_sha256"] = canonical_digest(selection)
    selection_path = output_dir / "stage4_selection.json"
    _write_json(selection_path, selection)
    return selection_path, baseline_path, checkpoint


def _write_cmc_selection_artifacts(
    output_dir: Path,
    train_report: dict,
) -> tuple[Path, Path, Path]:
    benchmarks = (
        "toolbench_g3",
        "traject_bench",
        "alfworld",
        "webshop",
    )
    checkpoint = output_dir / "checkpoints" / "clstr_stage4_cmc-step800.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": "counterfactual_memory_calibration_v1",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "parent_stage2_full_router_digest": "router-full",
            "parent_stage2_fast_router_digest": "router-fast",
            "model_state_dict": {
                "route_memory_residual_adapter.net.3.weight": torch.zeros(1, 1),
                "route_memory_candidate_utility_gate.net.0.weight": torch.zeros(1, 1),
            },
            "train_report": train_report,
        },
        checkpoint,
    )
    baseline_path = output_dir / "validation" / "stage2_baseline_report.json"
    baseline_payload = {
        "schema_version": "stage4_stage2_baseline_v1",
        "status": "ok",
        "stage2_baseline": {
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.60,
            "raw_dynamic_macro_mrr": 0.62,
            "raw_dynamic_regret": 0.04,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60, "raw_dynamic_mrr": 0.62}
                for benchmark in benchmarks
            },
        },
    }
    baseline_payload["manifest_sha256"] = canonical_digest(baseline_payload)
    _write_json(baseline_path, baseline_payload)
    stage2_baseline = {
        **baseline_payload["stage2_baseline"],
        "report_path": str(baseline_path.resolve()),
        "report_sha256": sha256_path(baseline_path)["sha256"],
        "report_manifest_sha256": baseline_payload["manifest_sha256"],
    }
    validation_path = output_dir / "validation" / "step800.json"
    router_integrity = {
        "full_router_digest": "router-full",
        "fast_router_digest": "router-fast",
        "full_router_digest_unchanged": True,
        "fast_router_digest_unchanged": True,
        "static_logits_exact": True,
        "max_abs_static_logit_difference": 0.0,
    }
    cmc_fused = {
        "balanced_macro_mrr": 0.606,
        "regret": 0.01,
        "alpha_zero_exact": True,
        "alpha_one_exact": True,
        "zero_history_exact": True,
        "by_benchmark": {
            benchmark: {"mrr": 0.606} for benchmark in benchmarks
        },
    }
    validation_payload = {
        "schema_version": "stage4_validation_report_v1",
        "status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "release_status": "ok",
        "step": 800,
        "router_integrity": router_integrity,
        "stage2_baseline": stage2_baseline,
        "candidate_union": {"static_k": 500, "dynamic_extra_k": 64, "final_k": 64},
        "balanced_macro": {
            "raw_dynamic_mrr": 0.64,
            "fused_mrr": 0.606,
            "fused_minus_static_mrr": 0.006,
            "fused_regret": 0.01,
        },
        "by_benchmark": {
            benchmark: {
                "static_mrr": 0.60,
                "raw_dynamic_mrr": 0.64,
                "fused_mrr": 0.606,
                "fused_minus_static_mrr": 0.006,
            }
            for benchmark in benchmarks
        },
        "cmc_fused": cmc_fused,
        "gradient_health": {"finite": True, "nonzero": True},
    }
    validation_payload["manifest_sha256"] = canonical_digest(validation_payload)
    _write_json(validation_path, validation_payload)
    release_checks = {
        "nonzero_selected_step": True,
        "fused_macro_improvement": True,
        "per_regime_nonregression": True,
        "regret_reduced": True,
        "exact_endpoints": True,
        "finite_nonzero_gradients": True,
        "stage2_router_unchanged": True,
    }
    reliability = {
        "mode": "cmc_candidate_gate",
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    selection = {
        "schema_version": "stage4_selection_v1",
        "status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "release_status": "ok",
        "promotion_status": "promoted",
        "dynamic_release_status": "ok",
        "release_checks": release_checks,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(validation_path.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_path)["sha256"],
        "router_integrity": router_integrity,
        "stage2_baseline": stage2_baseline,
        "candidate_union": validation_payload["candidate_union"],
        "cmc_fused_selection": cmc_fused,
        "gradient_health": validation_payload["gradient_health"],
        "reliability": reliability,
        "reliability_validation": cmc_fused,
    }
    selection["manifest_sha256"] = canonical_digest(selection)
    selection_path = output_dir / "stage4_selection.json"
    _write_json(selection_path, selection)
    return selection_path, baseline_path, checkpoint


def test_cmc_selected_release_blockers_accept_promoted_delta(tmp_path) -> None:
    output_dir = tmp_path / "stage4_cmc"
    train_report = {
        "status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "safe_memory_protocol_version": "counterfactual_memory_calibration_v1",
        "valid_or_test_used_for_training": False,
        "validation_used_for_checkpoint_selection": True,
    }
    selection_path, baseline_path, checkpoint = _write_cmc_selection_artifacts(
        output_dir,
        train_report,
    )

    blockers, evidence = _selected_release_blockers(
        selection_path=selection_path,
        stage2_baseline_report_path=baseline_path,
        checkpoint_path=checkpoint,
        train_report=train_report,
    )

    assert blockers == []
    assert evidence["delta_validation"]["status"] == "ok"


def test_stage4_quality_gate_accepts_promoted_cmc_selection(tmp_path) -> None:
    output_dir = tmp_path / "stage4_cmc"
    placeholder = output_dir / "placeholder.pt"
    report = _stage4_report(output_dir, placeholder)
    report.update(
        {
            "stage4_method": "counterfactual_memory_calibration_v1",
            "safe_memory_protocol_version": "counterfactual_memory_calibration_v1",
            "training_objective": "counterfactual_memory_calibration",
            "validation_used_for_checkpoint_selection": True,
        }
    )
    report["freeze_report"] = {
        "frozen_routing_foundation": True,
        "frozen_belief_gate": True,
        "train_transition": False,
        "trainable_modules": [
            "route_memory_residual_adapter",
            "route_memory_candidate_utility_gate",
        ],
    }
    report["last_metrics"] = {
        "loss": 0.6,
        "stage4_total_loss": 0.6,
        "stage4_act_count": 4.0,
        "stage4_cmc_dynamic_loss": 0.2,
        "stage4_cmc_fused_loss": 0.2,
        "stage4_cmc_no_regret_loss": 0.2,
        "stage4_cmc_eligible_rows": 4.0,
        "stage4_post_action_update_rows": 4.0,
        "stage4_next_state_rows": 4.0,
        "route_scorer": "unified_memory",
        "transition_scoring_mode": "unified_memory",
        "transition_residual_lambda": 0.0,
    }
    selection_path, baseline_path, checkpoint = _write_cmc_selection_artifacts(
        output_dir,
        report,
    )
    report["checkpoint"] = str(checkpoint)
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7 - step * 0.01,
                "stage4_total_loss": 0.7 - step * 0.01,
                "stage4_act_count": 4.0,
                "stage4_cmc_dynamic_loss": 0.2,
                "stage4_cmc_fused_loss": 0.2,
                "stage4_cmc_no_regret_loss": 0.2,
                "stage4_cmc_eligible_rows": 4.0,
                "stage4_post_action_update_rows": 4.0,
                "stage4_next_state_rows": 4.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        selection_path=selection_path,
        stage2_baseline_report_path=baseline_path,
        min_steps=6,
        expected_benchmarks=[
            "toolbench_g3",
            "traject_bench",
            "alfworld",
            "webshop",
        ],
    )

    assert audit["status"] == "ok", audit["blockers"]
    assert audit["blockers"] == []
    assert audit["release_authority"] == "fixed_validation_selection_v1"


def test_stage4_quality_gate_uses_fixed_validation_selection_not_tail_loss(
    tmp_path,
):
    output_dir = tmp_path / "stage4_safe"
    placeholder = output_dir / "placeholder.pt"
    report = _stage4_report(output_dir, placeholder)
    report.update(
        {
            "safe_memory_protocol_version": "stage4_safe_memory_v1",
            "validation_used_for_checkpoint_selection": True,
        }
    )
    report["freeze_report"]["trainable_modules"] = [
        "transition",
        "gate",
        "action_proj",
        "route_memory_utility_gate",
    ]
    selection_path, baseline_path, checkpoint = _write_safe_selection_artifacts(
        output_dir,
        report,
    )
    report["checkpoint"] = str(checkpoint)
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 9.0 + step,
                "stage4_act_loss": 9.0 + step,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        selection_path=selection_path,
        stage2_baseline_report_path=baseline_path,
        min_steps=6,
        expected_benchmarks=[
            "toolbench_g3",
            "traject_bench",
            "alfworld",
            "webshop",
        ],
    )

    assert audit["status"] == "ok"
    assert audit["blockers"] == []
    assert audit["release_authority"] == "fixed_validation_selection_v1"
    assert audit["selected_checkpoint"]["path"] == str(checkpoint.resolve())
    assert audit["health_diagnostics"]["tail_recall@5"] == 0.0


def test_stage4_quality_gate_accepts_complete_transition_act_training(tmp_path):
    output_dir = tmp_path / "stage4"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    _write_json(output_dir / "train_stdout.json", _stage4_report(output_dir, checkpoint))
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 1.2,
                "stage4_act_loss": 1.2,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.75,
            },
            {
                "step": 2,
                "loss": 1.0,
                "stage4_act_loss": 1.0,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.8,
            },
            {
                "step": 3,
                "loss": 0.9,
                "stage4_act_loss": 0.9,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.85,
            },
            {
                "step": 4,
                "loss": 0.8,
                "stage4_act_loss": 0.8,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.9,
            },
            {
                "step": 5,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            },
            {
                "step": 6,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            },
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "ok"
    assert audit["blockers"] == []
    assert audit["metrics_summary"]["max_step"] == 6
    assert audit["data_coverage"]["missing_expected_benchmarks"] == []
    assert audit["artifacts"]["latest_checkpoint_exists"] is True
    assert audit["train_safety"]["training_objective"] == "full_pool_causal_next_skill_with_counterfactual_utility"
    assert audit["full_pool_objective"]["next_skill_pool_mode"] == "full_pool"
    assert audit["full_pool_objective"]["eligible_rows"] == 4.0
    assert audit["full_pool_objective"]["counterfactual_eligible_rows"] == 4.0
    assert audit["full_pool_objective"]["static_hit_rows"] == 3.0
    assert audit["full_pool_objective"]["static_miss_rows"] == 1.0
    assert audit["causal_update"]["post_action_update_rows"] == 4.0
    assert audit["causal_update"]["next_state_rows"] == 4.0
    assert "preference_aux" not in audit
    assert audit["paper_scope"]["ready_for_phase_h_handoff"] is True


def test_stage4_quality_gate_rejects_legacy_transition_scoring_metadata(tmp_path):
    output_dir = tmp_path / "stage4_legacy_scoring"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["transition_scoring_mode"] = "action_observation_concat"
    report.pop("transition_residual_lambda")
    report["last_metrics"].pop("transition_residual_lambda")
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_transition_scoring_not_effective_route_mode" in audit["blockers"]


def test_stage4_quality_gate_rejects_legacy_route_scorer_even_with_explicit_zero_residual(tmp_path):
    output_dir = tmp_path / "stage4_v4_1b_listwise"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["route_scorer"] = "legacy_prior_residual"
    report["transition_scoring_mode"] = "skill_prior_plus_action_observation_residual"
    report["transition_residual_lambda"] = 0.0
    report["last_metrics"]["transition_scoring_mode"] = "skill_prior_plus_action_observation_residual"
    report["last_metrics"]["transition_residual_lambda"] = 0.0
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
                "transition_scoring_mode": "skill_prior_plus_action_observation_residual",
                "transition_residual_lambda": 0.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
        expected_transition_scoring_mode="skill_prior_plus_action_observation_residual",
        expected_transition_residual_lambda=0.0,
    )

    assert audit["status"] == "action_required"
    assert "stage4_route_scorer_not_unified_memory" in audit["blockers"]
    assert audit["stage4_quality"]["transition_scoring_safety"]["expected_residual_lambda"] == 0.0


def test_stage4_quality_gate_rejects_joint_preference_objective(tmp_path):
    output_dir = tmp_path / "stage4_joint_legacy"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["training_objective"] = "joint_transition_act_with_optional_preference"
    report["lambda_pref"] = 0.0
    report["preference_aux_enabled"] = False
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "unexpected_training_objective" in audit["blockers"]
    assert audit["train_safety"]["training_objective"] == "joint_transition_act_with_optional_preference"
    assert "preference_aux" not in audit


def test_stage4_quality_gate_requires_causal_post_action_evidence(tmp_path):
    output_dir = tmp_path / "stage4_missing_causal_update"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["last_metrics"].pop("stage4_post_action_update_rows")
    report["last_metrics"].pop("stage4_next_state_rows")
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "missing_stage4_post_action_update_rows" in audit["blockers"]
    assert "missing_stage4_next_state_rows" in audit["blockers"]


def test_stage4_quality_gate_requires_full_pool_counterfactual_evidence(tmp_path):
    output_dir = tmp_path / "stage4_missing_full_pool_evidence"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["next_skill_pool_mode"] = "stage0_candidates"
    for key in (
        "stage4_full_pool_eligible_rows",
        "stage4_counterfactual_eligible_rows",
        "stage4_counterfactual_weak_rows",
        "stage4_counterfactual_strong_rows",
        "stage4_static_hit_train_rows",
        "stage4_static_miss_train_rows",
    ):
        report["last_metrics"].pop(key)
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_next_skill_pool_mode_not_full_pool" in audit["blockers"]
    assert "missing_stage4_full_pool_eligible_rows" in audit["blockers"]
    assert "missing_stage4_counterfactual_eligible_rows" in audit["blockers"]
    assert "missing_stage4_counterfactual_weak_rows" in audit["blockers"]
    assert "missing_stage4_counterfactual_strong_rows" in audit["blockers"]
    assert "missing_stage4_static_hit_rows" in audit["blockers"]
    assert "missing_stage4_static_miss_rows" in audit["blockers"]


def test_stage4_quality_gate_rejects_nonfinite_full_pool_losses(tmp_path):
    output_dir = tmp_path / "stage4_nonfinite_full_pool_losses"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["last_metrics"]["stage4_act_loss"] = float("nan")
    report["last_metrics"]["stage4_counterfactual_utility_loss"] = float("inf")
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "nonfinite_stage4_full_pool_main_loss" in audit["blockers"]
    assert "nonfinite_stage4_counterfactual_utility_loss" in audit["blockers"]


def test_stage4_quality_gate_rejects_random_or_injected_candidate_handoff(tmp_path):
    output_dir = tmp_path / "stage4_bad_candidates"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["data_report"]["candidate_source"] = "explicit_or_random_fallback"
    report["data_report"]["static_candidate_source"] = "explicit_or_random_fallback"
    report["data_report"]["positive_injected_rows"] = 3
    report["data_report"]["stage0_candidate_handoff"] = {
        "enabled": False,
        "candidate_source": "full_pool_debug",
        "positive_missing_policy": "inject",
        "injected_positive_rows": 2,
    }
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": step, "loss": 0.7, "stage4_act_loss": 0.7, "stage4_act_count": 4.0}
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_candidate_source_not_declared_legal_full_pool" in audit["blockers"]
    assert "stage4_static_candidate_source_not_stage0_topm" in audit["blockers"]
    assert "stage4_stage0_handoff_not_enabled" in audit["blockers"]
    assert "stage4_positive_injected_rows" in audit["blockers"]
    assert "stage4_stage0_handoff_injected_positive_rows" in audit["blockers"]
    assert "stage4_positive_missing_policy_not_skip" in audit["blockers"]


def test_stage4_quality_gate_rejects_bad_next_skill_ranking_even_when_loss_exists(tmp_path):
    output_dir = tmp_path / "stage4_bad_ranking"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["last_metrics"]["stage4_next_skill_recall@5"] = 0.0
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.9 + step * 0.1,
                "stage4_act_loss": 0.9 + step * 0.1,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 0.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        first_window=1,
        last_window=1,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
        min_stage4_recall_at_5_tail=0.5,
        max_stage4_act_loss_increase=0.2,
    )

    assert audit["status"] == "action_required"
    assert "stage4_next_skill_recall_at_5_too_low" in audit["blockers"]
    assert "stage4_act_loss_regressed" in audit["blockers"]
    assert audit["stage4_quality"]["last_stage4_next_skill_recall@5"] == 0.0


def test_stage4_quality_gate_rejects_frozen_transition_for_mainline_act(tmp_path):
    output_dir = tmp_path / "stage4_frozen_transition"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["freeze_report"]["train_transition"] = False
    report["freeze_report"]["trainable_modules"] = ["trans_head", "action_proj"]
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_transition_not_trainable" in audit["blockers"]


def test_stage4_quality_gate_rejects_head_checkpoint_that_overrode_routing_foundation(tmp_path):
    output_dir = tmp_path / "stage4_head_overrode_routing"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["checkpoint_init_report"] = {
        "partial_load_mode": "stage0_routing_plus_act_init_heads_for_joint_stage4",
        "head_overrides_routing_keys": [
            "skill_table.E",
            "skill_table.W.weight",
            "encoder.proj.weight",
            "transition.net.0.weight",
        ],
    }
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_head_checkpoint_overrode_routing_foundation" in audit["blockers"]
    assert audit["checkpoint_init"]["head_overrode_routing_foundation_keys"] == [
        "encoder.proj.weight",
        "skill_table.E",
        "skill_table.W.weight",
    ]


def test_stage4_quality_gate_accepts_declared_belief_calibration_allowlist(tmp_path):
    output_dir = tmp_path / "stage4_allowlisted_belief_calibration"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    allowlist = [
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    ]
    report["checkpoint_init_report"] = {
        "partial_load_mode": "stage0_routing_plus_act_init_heads_for_joint_stage4",
        "protect_routing_foundation": True,
        "head_overrides_routing_keys": [
            "transition.obs_proj.weight",
            *allowlist,
        ],
        "routing_foundation_head_allowlist": allowlist,
        "skipped_head_routing_foundation_keys": [],
    }
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "ok"
    assert "stage4_head_checkpoint_overrode_routing_foundation" not in audit["blockers"]
    assert audit["checkpoint_init"]["head_overrode_routing_foundation_keys"] == []
    assert audit["checkpoint_init"]["routing_foundation_head_allowlist"] == allowlist


def test_stage4_quality_gate_rejects_missing_routing_foundation_protection_metadata(tmp_path):
    output_dir = tmp_path / "stage4_unprotected_init"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["checkpoint_init_report"] = {
        "partial_load_mode": "stage0_routing_plus_act_init_heads_for_joint_stage4",
        "protect_routing_foundation": False,
        "head_overrides_routing_keys": ["transition.obs_proj.weight"],
    }
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "stage4_checkpoint_init_not_protected" in audit["blockers"]
    assert audit["checkpoint_init"]["protect_routing_foundation"] is False


def test_stage4_quality_gate_rejects_checkpoint_without_quality_evidence(tmp_path):
    output_dir = tmp_path / "stage4"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["valid_or_test_used_for_training"] = True
    report["freeze_report"]["frozen_routing_foundation"] = False
    report["data_report"]["stage4_rows"] = 0
    report["data_report"]["benchmark_counts"] = {"traject_bench": 24}
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0}])

    audit = audit_stage4_act_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=6,
        expected_benchmarks=["traject_bench", "toolbench_g3"],
    )

    assert audit["status"] == "action_required"
    assert "valid_or_test_used_for_training" in audit["blockers"]
    assert "routing_foundation_not_frozen" in audit["blockers"]
    assert "insufficient_training_steps" in audit["blockers"]
    assert "no_stage4_rows" in audit["blockers"]
    assert "missing_expected_benchmarks" in audit["blockers"]
    assert "missing_stage4_act_metrics" in audit["blockers"]


def test_stage4_quality_gate_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_dir = tmp_path / "stage4"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    _write_json(output_dir / "train_stdout.json", {"status": "ok", "checkpoint": str(checkpoint)})
    _write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0}])
    output_path = tmp_path / "stage4_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage4_quality.py",
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


def test_stage4_quality_gate_cli_cannot_override_full_pool_route_semantics(tmp_path):
    output_dir = tmp_path / "stage4_v4_1b_cli"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["transition_scoring_mode"] = "skill_prior_plus_action_observation_residual"
    report["transition_residual_lambda"] = 0.0
    report["last_metrics"]["transition_scoring_mode"] = "skill_prior_plus_action_observation_residual"
    report["last_metrics"]["transition_residual_lambda"] = 0.0
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
                "transition_scoring_mode": "skill_prior_plus_action_observation_residual",
                "transition_residual_lambda": 0.0,
            }
            for step in range(1, 7)
        ],
    )
    output_path = tmp_path / "stage4_v4_1b_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage4_quality.py",
            "--output_dir",
            str(output_dir),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output_path),
            "--min_steps",
            "6",
            "--expected_benchmarks",
            "traject_bench",
            "toolbench_g3",
            "--expected_transition_scoring_mode",
            "skill_prior_plus_action_observation_residual",
            "--expected_transition_residual_lambda",
            "0.0",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "action_required"
    assert "stage4_transition_scoring_not_effective_route_mode" in written["blockers"]
    assert written["stage4_quality"]["transition_scoring_safety"]["expected_residual_lambda"] == 0.0


def test_stage4_quality_gate_cli_can_explicitly_audit_legacy_unprotected_init(tmp_path):
    output_dir = tmp_path / "stage4_legacy_unprotected"
    checkpoint = output_dir / "checkpoints/clstr_stage4_act-step6.pt"
    report = _stage4_report(output_dir, checkpoint)
    report["checkpoint_init_report"] = {
        "partial_load_mode": "legacy_stage4_without_protection_metadata",
        "protect_routing_foundation": False,
        "head_overrides_routing_keys": ["transition.obs_proj.weight"],
    }
    _write_json(output_dir / "train_stdout.json", report)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 0.7,
                "stage4_act_loss": 0.7,
                "stage4_act_count": 4.0,
                "stage4_next_skill_recall@5": 1.0,
            }
            for step in range(1, 7)
        ],
    )
    output_path = tmp_path / "stage4_legacy_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage4_quality.py",
            "--output_dir",
            str(output_dir),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output_path),
            "--min_steps",
            "6",
            "--expected_benchmarks",
            "traject_bench",
            "toolbench_g3",
            "--allow_unprotected_routing_init",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert written["checkpoint_init"]["require_routing_foundation_protection"] is False
    assert written["checkpoint_init"]["protect_routing_foundation"] is False


def test_stage4_quality_gate_cli_defaults_to_progressive_final_mainline():
    script = Path("scripts/audit_clstr_stage4_quality.py").read_text(encoding="utf-8")

    assert 'default="outputs/clstr_unified_memory_stage4_full_pool_counterfactual"' in script
    assert "clstr_unified_stage4_v3_traj_retrieval_act" not in script
