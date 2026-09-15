from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


CANONICAL_STAGE0_OUTPUT_DIR = "outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE"
CANONICAL_STAGE0_EVAL_METRICS = f"{CANONICAL_STAGE0_OUTPUT_DIR}/full_retrieval_eval/metrics.json"
CANONICAL_DATA_ROOT = "data/clstr_unified_pretrain_v4_2_progressive_final"
CANONICAL_STAGE1_OUTPUT_DIR = (
    "outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"
)
CANONICAL_STAGE2_OUTPUT_DIR = (
    "outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"
)


def _load_audit_module():
    path = Path("scripts/audit_clstr_unified_training_readiness.py")
    spec = importlib.util.spec_from_file_location("audit_clstr_unified_training_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_train_safe_stage1_checkpoint(path: Path, skill_count: int, dim: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "sampling_strategy": "batch_stride",
            "shuffle_queries": True,
            "train_safety": {
                "unified_v2_train_safe_retrieval": True,
                "excluded_retrieval_splits": ["public_train_or_eval_unlabeled"],
            },
            "model_state_dict": {
                "skill_table.E": torch.zeros(skill_count, dim),
                "skill_table.W.weight": torch.eye(dim),
                "skill_table.logit_scale_retr": torch.zeros(1),
                "skill_table.skill_bias_retr": torch.zeros(skill_count),
            },
        },
        path,
    )


def _write_stage0_baseline_metrics(path: Path, value: float = 0.1) -> None:
    _write_json(
        path,
        {
            "method": "stage0_skillrouter_frozen_baseline",
            "metrics": {
                "Recall@20": value,
                "Recall@50": value,
                "Recall@100": value,
            },
        },
    )


def _write_stage0_eval_metrics(path: Path, value: float = 0.5) -> None:
    _write_json(
        path,
        {
            "method": "clstr_unified_stage0_biencoder",
            "status": "ok",
            "metrics": {
                "Recall@20": value,
                "Recall@50": value,
                "Recall@100": value,
            },
        },
    )


def _write_stage0_quality_output(output_dir: Path, baseline_metrics_path: Path, *, skill_count: int, value: float = 0.5) -> Path:
    checkpoint = output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    _write_train_safe_stage1_checkpoint(checkpoint, skill_count=skill_count)
    _write_stage0_baseline_metrics(baseline_metrics_path, value=0.1)
    _write_stage0_eval_metrics(output_dir / "full_retrieval_eval/metrics.json", value=value)
    (output_dir / "loss_curve.svg").write_text("<svg>loss</svg>", encoding="utf-8")
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 2.0,
                "recall_at_20": 0.2,
                "recall_at_50": 0.25,
                "recall_at_100": 0.3,
                "query_sources": ["skillret", "toolret_training"],
            }
            for step in range(1, 201)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "recall_at_20": 0.4,
                "recall_at_50": 0.5,
                "recall_at_100": 0.6,
                "query_sources": ["toolbench_g3", "traject_bench"],
            }
            for step in range(4801, 5001)
        ],
    )
    return checkpoint


def _write_stage2_quality_output(output_dir: Path, *, steps: int = 10000) -> Path:
    checkpoint = output_dir / f"checkpoints/clstr_full_base-step{steps}.pt"
    latest = output_dir / "checkpoints/latest.pt"
    metrics = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("stage2 checkpoint", encoding="utf-8")
    latest.write_text("latest checkpoint", encoding="utf-8")
    loss_curve.write_text("<svg>rolling loss</svg>", encoding="utf-8")
    _write_jsonl(
        metrics,
        [
            {
                "step": step,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.8,
                "transition_skill_recall@5": 0.8,
                "transition_prior_skill_recall@5": 0.78,
                "transition_skill_mrr": 0.5,
                "transition_prior_skill_mrr": 0.49,
                "transition_worse_than_stage0_prior_fraction": 0.2,
                "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "transition_residual_lambda": 0.25,
            }
            for step in range(1, 201)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@5": 0.85,
                "transition_prior_skill_recall@5": 0.83,
                "transition_skill_mrr": 0.55,
                "transition_prior_skill_mrr": 0.52,
                "transition_worse_than_stage0_prior_fraction": 0.18,
                "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "transition_residual_lambda": 0.25,
            }
            for step in range(steps - 199, steps + 1)
        ],
    )
    _write_json(
        output_dir / "train_report.json",
        {
            "status": "ok",
            "training_objective": "component_complete_masked_multi_loss",
            "training_regime": "offline_replay_supervised_pretraining",
            "route_scorer": "unified_memory",
            "uses_stage0_prior_at_inference": False,
            "transition_objective": "full_pool_causal_next_skill_with_counterfactual_utility",
            "transition_skill_ce": {
                "enabled": True,
                "scoring_mode": "unified_memory",
                "residual_lambda": 0.0,
                "route_scorer": "unified_memory",
                "candidate_pool": "declared_legal_full_skill_pool",
            },
            "checkpoint": str(checkpoint),
            "training_metrics_path": str(metrics),
            "loss_curve_path": str(loss_curve),
            "latest_checkpoint": str(latest),
            "max_steps": steps,
            "sample_count": 32,
            "skill_count": 2,
            "valid_or_test_used_for_training": False,
            "on_policy_rollout_used": False,
            "not_rl_fine_tuning": True,
            "frozen_routing_foundation": True,
            "loss_activation_counts": {
                "L_policy": 1,
                "L_trans": 1,
                "L_trans_skill_ce": 1,
                "STOP": 1,
                "routing": 1,
            },
            "sampled_loss_activation_counts": {
                "L_policy": 1,
                "L_trans": 1,
                "L_trans_skill_ce": 1,
                "STOP": 1,
                "routing": 1,
            },
            "metric_averages": {"loss": 1.2},
            "transition_candidate_training": {
                "scoring_mode": "unified_memory",
                "residual_lambda": 0.0,
                "route_scorer": "unified_memory",
                "next_skill_pool_mode": "full_pool",
                "inventory_mask_mode": "explicit_only",
                "counterfactual_utility_loss_weight": 0.05,
            },
            "transition_input_semantics": {
                "transition_scoring_mode": "unified_memory",
                "transition_residual_lambda": 0.0,
                "route_scorer": "unified_memory",
            },
        },
    )
    return checkpoint


def _write_stage1_heads_quality_output(output_dir: Path, *, steps: int = 3000) -> Path:
    checkpoint = output_dir / f"checkpoints/clstr_stage1_heads-step{steps}.pt"
    latest = output_dir / "checkpoints/latest.pt"
    metrics = output_dir / "training_metrics.jsonl"
    loss_curve = output_dir / "loss_curve.svg"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("stage1 heads checkpoint", encoding="utf-8")
    latest.write_text("latest stage1 checkpoint", encoding="utf-8")
    loss_curve.write_text("<svg>stage1 loss</svg>", encoding="utf-8")
    _write_jsonl(
        metrics,
        [
            {
                "step": step,
                "loss": 2.0,
                "transition_skill_ce_loss": 0.8,
                "transition_skill_recall@1": 0.45,
                "transition_skill_recall@5": 0.8,
                "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "transition_residual_lambda": 0.25,
            }
            for step in range(1, 101)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "transition_skill_ce_loss": 0.5,
                "transition_skill_recall@1": 0.5,
                "transition_skill_recall@5": 0.85,
                "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "transition_residual_lambda": 0.25,
            }
            for step in range(steps - 99, steps + 1)
        ],
    )
    _write_json(
        output_dir / "train_report.json",
        {
            "status": "ok",
            "stage": "clstr_stage1_heads_init",
            "training_objective": "stage1_heads_init_topm_supervised",
            "checkpoint": str(checkpoint),
            "training_metrics_path": str(metrics),
            "loss_curve_path": str(loss_curve),
            "latest_checkpoint": str(latest),
            "max_steps": steps,
            "frozen_routing_foundation": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "excluded_state_key_prefixes": ["encoder.", "cross_encoder.", "skill_table."],
            "transition_input_semantics": {
                "action_channel": "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding",
                "observation_channel": "next_observation_text_embedding",
                "uses_actual_action_text_when_available": True,
                "uses_next_observation_as_observation": True,
                "action_text_used_as_observation": False,
                "transition_scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "transition_residual_lambda": 0.25,
            },
            "transition_candidate_training": {
                "scoring_mode": "stage0_rank_prior_plus_transition_residual",
                "residual_lambda": 0.25,
            },
            "stage0_candidate_handoff": {
                "candidate_source": "stage0_topm_online",
                "positive_missing_policy": "skip",
                "injected_positive_rows": 0,
            },
            "loss_activation_counts": {
                "L_policy": 1,
                "L_trans": 1,
                "L_trans_skill_ce": 1,
                "STOP": 1,
                "belief": 1,
            },
            "sampled_loss_activation_counts": {
                "L_policy": 1,
                "L_trans": 1,
                "L_trans_skill_ce": 1,
                "STOP": 1,
                "belief": 1,
            },
        },
    )
    return checkpoint


def test_unified_training_readiness_reports_stage_gates_and_benchmark_filter(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {
                "total_rows": 4,
                "traject_bench": {"available": True, "total_rows": 2},
                "toolbench_g3": {"available": False, "total_rows": 0},
            },
            "retrieval_stream": {
                "total_pairs": 3,
                "traject_bench": {"available": True, "total_pairs": 1},
                "counts": {"toolret_training_pair": 2},
            },
            "skill_pool": {"total_skills": 3},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}, {"skill_id": "skill/c"}],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {"source": "traject_bench", "query_text": "q", "positive_skill_id": "skill/a"},
            {"source": "toolret_training", "query_text": "q2", "positive_skill_id": "skill/b"},
            {"source": "toolret_training", "query_text": "q3", "positive_skill_id": "skill/c"},
        ],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "admissible_actions": ["call a"],
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "state_text": "state 2",
                "action_text": "call b",
                "expert_action": "call b",
                "admissible_actions": ["call b"],
                "skill_id": "skill/b",
                "next_skill_id": "skill/c",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "alfworld",
                "state_text": "aux",
                "action_text": "look",
                "expert_action": "look",
                "admissible_actions": ["look"],
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "state_text": "dev",
                "action_text": "call",
                "expert_action": "call",
                "admissible_actions": ["call"],
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "dev"},
            },
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage0 = _write_stage0_quality_output(stage0_output_dir, baseline_metrics, skill_count=3)
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    assert report["status"] == "action_required"
    assert report["leakage_audit"]["status"] == "ok"
    assert report["source_inventory"]["missing_required_target_sources"] == ["toolbench_g3"]
    assert report["benchmark_filter"]["allowed_benchmarks"] == ["toolbench_g3", "traject_bench"]
    assert report["benchmark_filter"]["retained_train_trajectory_rows"] == 2
    assert report["benchmark_filter"]["skipped_by_benchmark"] == {"alfworld": 1, "toolbench_g3": 1}
    assert set(report["stage_gates"]) == {
        "stage0_routing",
        "stage1_heads",
        "stage2_full_base",
        "stage4_causal_next_skill",
    }
    assert report["stage_gates"]["stage0_routing"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage1_heads"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage4_causal_next_skill"]["ready_to_submit"] is False
    assert report["stage_gates"]["stage4_causal_next_skill"]["data_rows"] == 2
    assert report["stage_gates"]["stage4_causal_next_skill"]["post_action_training_rows"] == 2
    assert report["paper_readiness"]["full_three_benchmark_data_ready"] is False


def test_unified_training_readiness_reports_toolret_eval_readiness(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    toolret_eval = tmp_path / "data/toolret_eval"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "state",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_jsonl(toolret_eval / "queries.jsonl", [{"query_id": "q1"}])
    _write_jsonl(toolret_eval / "skills.jsonl", [{"skill_id": "tool/a", "name": "a"}])
    _write_jsonl(toolret_eval / "qrels.jsonl", [{"query_id": "q1", "skill_id": "tool/a", "relevance": 1}])

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage1_checkpoint=tmp_path / "outputs/missing_stage1.pt",
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
        toolret_eval_dir=toolret_eval,
    )

    toolret = report["benchmark_eval_readiness"]["toolret"]
    assert toolret["status"] == "ok"
    assert toolret["query_count"] == 1
    assert toolret["skill_count"] == 1
    assert toolret["positive_qrel_count"] == 1
    assert report["paper_readiness"]["toolret_eval_ready"] is True


def test_unified_training_readiness_blocks_stage1_when_skill_pool_borderline_review_pending(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {
                "total_skills": 2,
                "borderline_review": {
                    "status": "pending_manual_or_llm_review",
                    "candidate_count": 3,
                    "method": "fuzzy_name_or_partial_param_overlap_v1",
                },
            },
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [{"skill_id": "skill/a", "description": "A"}, {"skill_id": "skill/b", "description": "B"}],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        stage1_checkpoint=tmp_path / "outputs/missing_stage1.pt",
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    skill_pool_quality = report["skill_pool_quality"]
    assert skill_pool_quality["borderline_review_status"] == "pending_manual_or_llm_review"
    assert skill_pool_quality["borderline_candidate_count"] == 3
    assert "skill_pool_borderline_review_pending" in skill_pool_quality["blockers"]
    assert "skill_pool_borderline_review_pending" in report["stage_gates"]["stage0_routing"]["blockers"]
    assert report["stage_gates"]["stage0_routing"]["ready_to_submit"] is False


def test_unified_training_readiness_accepts_completed_conservative_borderline_review(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    review_report = tmp_path / "outputs/skill_dedup_borderline_review_report.json"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {
                "total_skills": 2,
                "borderline_review": {
                    "status": "pending_manual_or_llm_review",
                    "candidate_count": 3,
                    "method": "fuzzy_name_or_partial_param_overlap_v1",
                },
            },
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [{"skill_id": "skill/a", "description": "A"}, {"skill_id": "skill/b", "description": "B"}],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_json(
        review_report,
        {
            "status": "complete",
            "method": "conservative_threshold_review_v1",
            "reviewed_candidate_count": 3,
            "keep_separate_count": 3,
            "unresolved_candidate_count": 0,
        },
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        stage1_checkpoint=tmp_path / "outputs/missing_stage1.pt",
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
        skill_dedup_borderline_review_report_path=review_report,
    )

    skill_pool_quality = report["skill_pool_quality"]
    assert skill_pool_quality["borderline_review_status"] == "complete_by_conservative_review"
    assert skill_pool_quality["borderline_candidate_count"] == 0
    assert skill_pool_quality["borderline_review_report"]["status"] == "complete"
    assert "skill_pool_borderline_review_pending" not in skill_pool_quality["blockers"]
    assert "skill_pool_borderline_review_pending" not in report["stage_gates"]["stage0_routing"]["blockers"]


def test_unified_training_readiness_reports_traject_eval_readiness(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    traject_eval = tmp_path / "data/traject_eval"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "state",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_jsonl(traject_eval / "queries.jsonl", [{"query_id": "q1"}])
    _write_jsonl(traject_eval / "skills.jsonl", [{"skill_id": "tool/a", "name": "a"}])
    _write_jsonl(traject_eval / "qrels.jsonl", [{"query_id": "q1", "skill_id": "tool/a", "relevance": 1}])

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage1_checkpoint=tmp_path / "outputs/missing_stage1.pt",
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
        traject_eval_dir=traject_eval,
    )

    traject = report["benchmark_eval_readiness"]["traject"]
    assert traject["status"] == "ok"
    assert traject["query_count"] == 1
    assert traject["skill_count"] == 1
    assert traject["positive_qrel_count"] == 1
    assert report["paper_readiness"]["traject_eval_ready"] is True


def test_unified_training_readiness_reports_toolbench_g3_data_blockers(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    toolbench_source = tmp_path / "ToolBench/data_example"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "state",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_json(toolbench_source / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(toolbench_source / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        toolbench_g3_data_dir=tmp_path / "data/toolbench_g3",
        toolbench_g3_source_root=toolbench_source,
    )

    toolbench = report["benchmark_eval_readiness"]["toolbench_g3"]
    assert toolbench["status"] == "action_required"
    assert "missing_normalized_toolbench_g3_files" in toolbench["blockers"]
    assert "source_root_is_data_example_not_official_g3" in toolbench["blockers"]
    assert toolbench["may_use_for_paper_main_table"] is False
    assert report["paper_readiness"]["toolbench_g3_data_ready"] is False


def test_unified_training_readiness_blocks_toolbench_g3_when_raw_answer_count_incomplete(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    toolbench_data = tmp_path / "data/toolbench_g3"
    toolbench_source = tmp_path / "ToolBench/data"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/forecast"},
            {"skill_id": "toolbench-g3/maps/route"},
        ],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_text": "q",
                "positive_skill_id": "toolbench-g3/weather/forecast",
                "negative_skill_ids": ["toolbench-g3/maps/route"],
            }
        ],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "state_text": "state",
                "action_text": "call",
                "expert_action": "call",
                "skill_id": "toolbench-g3/weather/forecast",
                "next_skill_id": "toolbench-g3/maps/route",
                "provenance": {"split": "train_or_released_g3"},
            }
        ],
    )
    _write_jsonl(
        toolbench_data / "skills.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/forecast", "name": "forecast"},
            {"skill_id": "toolbench-g3/maps/route", "name": "route"},
        ],
    )
    _write_jsonl(
        toolbench_data / "retrieval.jsonl",
        [
            {
                "query_id": "toolbench-g3-1",
                "query_text": "q",
                "positive_skill_id": "toolbench-g3/weather/forecast",
                "negative_skill_ids": ["toolbench-g3/maps/route"],
            }
        ],
    )
    _write_jsonl(
        toolbench_data / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "state_text": "state",
                "expert_action": "call",
                "skill_id": "toolbench-g3/weather/forecast",
                "next_skill_id": "toolbench-g3/maps/route",
            }
        ],
    )
    _write_json(toolbench_source / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(toolbench_source / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        toolbench_g3_data_dir=toolbench_data,
        toolbench_g3_source_root=toolbench_source,
        expected_toolbench_g3_answer_files=2,
    )

    toolbench = report["benchmark_eval_readiness"]["toolbench_g3"]
    assert toolbench["status"] == "action_required"
    assert "source_root_incomplete_g3_answer_files" in toolbench["blockers"]
    assert report["paper_readiness"]["toolbench_g3_data_ready"] is False


def test_unified_training_readiness_reports_toolret_eval_blockers(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    toolret_eval = tmp_path / "data/toolret_eval"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 0},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(data_root / "trajectories.jsonl", [])
    _write_jsonl(toolret_eval / "queries.jsonl", [{"query_id": "q1"}])
    _write_jsonl(toolret_eval / "skills.jsonl", [{"skill_id": "tool/a", "name": "a"}])
    _write_jsonl(toolret_eval / "qrels.jsonl", [{"query_id": "q1", "skill_id": "missing", "relevance": 1}])

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        toolret_eval_dir=toolret_eval,
    )

    toolret = report["benchmark_eval_readiness"]["toolret"]
    assert toolret["status"] == "action_required"
    assert "missing_positive_qrel_skill_ids" in toolret["blockers"]
    assert report["paper_readiness"]["toolret_eval_ready"] is False


def test_unified_training_readiness_keeps_stage2_base_train_separate_from_target_public_data(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 2},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_id": "q", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "traject_bench",
                "state_text": "public target",
                "action_text": "call target",
                "expert_action": "call target",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "public_train_or_eval_unlabeled"},
            },
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage0 = _write_stage0_quality_output(stage0_output_dir, baseline_metrics, skill_count=2)
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    assert report["base_train_filter"]["retained_train_trajectory_rows"] == 1
    assert report["base_train_filter"]["retained_by_benchmark"] == {"alfworld": 1}
    assert report["benchmark_filter"]["retained_train_trajectory_rows"] == 0
    assert report["benchmark_filter"]["skipped_by_split"]["public_train_or_eval_unlabeled"] == 1
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage2_full_base"]["data_rows"] == 1
    assert report["stage_gates"]["stage4_causal_next_skill"]["ready_to_submit"] is False
    assert "no_stage4_post_action_rows" in report["stage_gates"]["stage4_causal_next_skill"]["blockers"]


def test_unified_training_readiness_blocks_stage2_when_skill_pool_has_placeholder_records(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": ["toolbench_g3"]},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": False, "train_allowed": False},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {"skill_id": "skill/a", "description": "real skill"},
            {
                "skill_id": "traject/query-only",
                "description": "traject/query-only",
                "provenance": {
                    "missing_source_record": True,
                    "dedup_method": "identity_skill_id_v2",
                },
            },
        ],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_id": "q", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage1 = tmp_path / "outputs/stage1.pt"
    _write_train_safe_stage1_checkpoint(stage1, skill_count=2)

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage1_checkpoint=stage1,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    assert report["skill_pool_quality"]["status"] == "action_required"
    assert report["skill_pool_quality"]["missing_source_record_count"] == 1
    assert report["skill_pool_quality"]["missing_source_record_by_prefix"] == {"traject": 1}
    assert report["stage_gates"]["stage0_routing"]["ready_to_submit"] is False
    assert "skill_pool_has_placeholder_records" in report["stage_gates"]["stage0_routing"]["blockers"]
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is False
    assert "skill_pool_has_placeholder_records" in report["stage_gates"]["stage2_full_base"]["blockers"]


def test_unified_training_readiness_rejects_legacy_stage0_checkpoint_without_train_safety(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage0 = tmp_path / "outputs/legacy_stage0.pt"
    stage0.parent.mkdir(parents=True)
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "model_state_dict": {"skill_table.E": torch.zeros(1, 8)},
        },
        stage0,
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    stage2_gate = report["stage_gates"]["stage2_full_base"]
    assert stage2_gate["ready_to_submit"] is False
    assert "stage0_checkpoint_not_train_safe" in stage2_gate["blockers"]
    assert stage2_gate["stage0_checkpoint_train_safety"]["status"] == "not_train_safe"


def test_unified_training_readiness_blocks_stage2_when_stage0_quality_gate_fails(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    stage0_output_dir = tmp_path / "outputs/stage0"
    stage0 = stage0_output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_train_safe_stage1_checkpoint(stage0, skill_count=1)
    _write_stage0_baseline_metrics(baseline_metrics, value=0.1)
    _write_stage0_eval_metrics(stage0_output_dir / "full_retrieval_eval/metrics.json", value=0.5)
    _write_jsonl(
        stage0_output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 10.2, "recall_at_50": 0.0},
            {"step": 2, "loss": 10.2, "recall_at_50": 0.0},
        ],
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    stage2_gate = report["stage_gates"]["stage2_full_base"]
    assert stage2_gate["ready_to_submit"] is False
    assert "stage0_quality_gate_not_ok" in stage2_gate["blockers"]
    assert stage2_gate["stage0_quality_gate"]["status"] == "action_required"
    assert "missing_loss_curve" in stage2_gate["stage0_quality_gate"]["blockers"]


def test_unified_training_readiness_blocks_stage2_until_stage1_heads_gate_ok(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    stage0_output_dir = tmp_path / "outputs/stage0"
    stage0 = stage0_output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(data_root / "retrieval.jsonl", [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_train_safe_stage1_checkpoint(stage0, skill_count=1)
    _write_stage0_baseline_metrics(baseline_metrics, value=0.1)
    _write_stage0_eval_metrics(stage0_output_dir / "full_retrieval_eval/metrics.json", value=0.5)
    (stage0_output_dir / "loss_curve.svg").write_text("<svg>loss</svg>", encoding="utf-8")
    _write_jsonl(
        stage0_output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 2.0,
                "recall_at_20": 0.2,
                "recall_at_50": 0.25,
                "recall_at_100": 0.3,
                "query_sources": ["skillret"],
            }
            for step in range(1, 201)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "recall_at_20": 0.4,
                "recall_at_50": 0.5,
                "recall_at_100": 0.6,
                "query_sources": ["toolbench_g3"],
            }
            for step in range(4801, 5001)
        ],
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1_output_dir / "checkpoints/missing-stage1.pt",
        stage1_output_dir=stage1_output_dir,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    assert report["stage_gates"]["stage0_routing"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage1_heads"]["ready_to_submit"] is False
    assert "missing_stage1_checkpoint" in report["stage_gates"]["stage1_heads"]["blockers"]
    assert report["stage_gates"]["stage1_heads"]["stage1_quality_gate"]["status"] == "missing"
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is False
    assert "missing_stage1_checkpoint" in report["stage_gates"]["stage2_full_base"]["blockers"]


def test_unified_training_readiness_allows_stage2_after_stage1_heads_gate_ok(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    stage0_output_dir = tmp_path / "outputs/stage0"
    stage0 = stage0_output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(data_root / "retrieval.jsonl", [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "base train",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_train_safe_stage1_checkpoint(stage0, skill_count=1)
    _write_stage0_baseline_metrics(baseline_metrics, value=0.1)
    _write_stage0_eval_metrics(stage0_output_dir / "full_retrieval_eval/metrics.json", value=0.5)
    (stage0_output_dir / "loss_curve.svg").write_text("<svg>loss</svg>", encoding="utf-8")
    _write_jsonl(
        stage0_output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 2.0,
                "recall_at_20": 0.2,
                "recall_at_50": 0.25,
                "recall_at_100": 0.3,
                "query_sources": ["skillret"],
            }
            for step in range(1, 201)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "recall_at_20": 0.4,
                "recall_at_50": 0.5,
                "recall_at_100": 0.6,
                "query_sources": ["toolbench_g3"],
            }
            for step in range(4801, 5001)
        ],
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_checkpoint=tmp_path / "outputs/missing_stage2.pt",
    )

    assert report["stage_gates"]["stage1_heads"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage1_heads"]["stage1_quality_gate"]["status"] == "ok"
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is True
    assert report["stage_gates"]["stage2_full_base"]["stage1_checkpoint"] == str(stage1)
    assert report["stage_gates"]["stage2_full_base"]["stage1_quality_gate"]["status"] == "ok"


def test_unified_training_readiness_sbatch_entrypoint_is_audit_only_and_local():
    script = Path("scripts/sbatch/run_clstr_unified_readiness_audit.sh").read_text(encoding="utf-8")

    assert "scripts/audit_clstr_unified_training_readiness.py" in script
    assert "STAGE0_OUTPUT_DIR" in script
    assert "--stage0_output_dir" in script
    assert "STAGE0_CHECKPOINT" in script
    assert "--stage0_checkpoint" in script
    assert "STAGE0_EVAL_METRICS_PATH" in script
    assert "--stage0_eval_metrics_path" in script
    assert CANONICAL_STAGE0_EVAL_METRICS in script
    assert "STAGE0_BASELINE_METRICS_PATH" in script
    assert "--stage0_baseline_metrics_path" in script
    assert CANONICAL_STAGE1_OUTPUT_DIR in script
    assert CANONICAL_STAGE2_OUTPUT_DIR in script
    assert "SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH" in script
    assert "--skill_dedup_borderline_review_report_path" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json" in script
    assert "STAGE2_OUTPUT_DIR" in script
    assert "--stage2_output_dir" in script
    assert "STAGE3_OUTPUT_DIR" not in script
    assert "--stage3_output_dir" not in script
    assert CANONICAL_DATA_ROOT in script
    assert "TOOLRET_EVAL_DIR" in script
    assert "--toolret_eval_dir" in script
    assert "data/toolret_eval" in script
    assert "EXPECTED_TOOLBENCH_G3_ANSWER_FILES" in script
    assert "--expected_toolbench_g3_answer_files" in script
    assert "ALLOWED_BENCHMARKS" in script
    assert "toolbench_g3,traject_bench,alfworld,webshop" in script
    assert "MEMORY_UTILITY_GATE_MODE=${MEMORY_UTILITY_GATE_MODE:-dynamic}" in script
    assert "MEMORY_UTILITY_AUDIT_REPORT_PATH=${MEMORY_UTILITY_AUDIT_REPORT_PATH:-}" in script
    assert "MEMORY_UTILITY_GATE_CHECKPOINT_PATH=${MEMORY_UTILITY_GATE_CHECKPOINT_PATH:-}" in script
    assert '--memory_utility_gate_mode "${MEMORY_UTILITY_GATE_MODE}"' in script
    assert 'extra_args+=(--memory_utility_audit_report_path "${MEMORY_UTILITY_AUDIT_REPORT_PATH}")' in script
    assert 'extra_args+=(--memory_utility_gate_checkpoint_path "${MEMORY_UTILITY_GATE_CHECKPOINT_PATH}")' in script
    assert "sbatch" not in script
    assert "python" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_unified_training_readiness_cli_runs_from_repo_root(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "state",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    stage0 = stage0_output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    _write_train_safe_stage1_checkpoint(stage0, skill_count=1)
    _write_stage0_baseline_metrics(baseline_metrics, value=0.1)
    _write_stage0_eval_metrics(stage0_output_dir / "full_retrieval_eval/metrics.json", value=0.5)
    (stage0_output_dir / "loss_curve.svg").write_text("<svg>loss</svg>", encoding="utf-8")
    _write_jsonl(
        stage0_output_dir / "training_metrics.jsonl",
        [
            {
                "step": step,
                "loss": 2.0,
                "recall_at_20": 0.2,
                "recall_at_50": 0.25,
                "recall_at_100": 0.3,
                "query_sources": ["skillret", "toolret_training"],
            }
            for step in range(1, 201)
        ]
        + [
            {
                "step": step,
                "loss": 1.0,
                "recall_at_20": 0.4,
                "recall_at_50": 0.5,
                "recall_at_100": 0.6,
                "query_sources": ["toolbench_g3", "traject_bench"],
            }
            for step in range(4801, 5001)
        ],
    )
    output_path = tmp_path / "readiness_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_unified_training_readiness.py",
            "--data_root",
            str(data_root),
            "--stage0_checkpoint",
            str(stage0),
            "--stage0_output_dir",
            str(stage0_output_dir),
            "--stage0_baseline_metrics_path",
            str(baseline_metrics),
            "--stage1_checkpoint",
            str(stage1),
            "--stage1_output_dir",
            str(stage1_output_dir),
            "--stage2_checkpoint",
            str(tmp_path / "outputs/missing_stage2.pt"),
            "--output_path",
            str(output_path),
        ],
        cwd=Path.cwd(),
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["stage_gates"]["stage2_full_base"]["ready_to_submit"] is True


def test_unified_training_readiness_cli_defaults_use_toolbench_g3_stage_paths():
    script = Path("scripts/audit_clstr_unified_training_readiness.py").read_text(encoding="utf-8")

    assert CANONICAL_DATA_ROOT in script
    assert f'DEFAULT_STAGE0_OUTPUT_DIR = "{CANONICAL_STAGE0_OUTPUT_DIR}"' in script
    assert 'DEFAULT_STAGE0_EVAL_METRICS_NAME = "full_retrieval_eval/metrics.json"' in script
    assert "checkpoints/clstr_unified_retrieval_v2-step5000.pt" in script
    assert CANONICAL_STAGE1_OUTPUT_DIR in script
    assert CANONICAL_STAGE2_OUTPUT_DIR in script
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json" in script
    assert 'DEFAULT_STAGE2_CHECKPOINT = f"{DEFAULT_STAGE2_OUTPUT_DIR}/checkpoints/clstr_full_base-step10000.pt"' in script
    assert "DEFAULT_STAGE3" not in script
    assert "--stage3" not in script
    assert "clstr_unified_stage1_v3_traj_retrieval_heads_init" not in script
    assert "clstr_unified_stage1_train_safe_retrieval_warmup" not in script


def test_unified_training_readiness_stage4_uses_stage2_checkpoint(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage0 = _write_stage0_quality_output(stage0_output_dir, baseline_metrics, skill_count=2)
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    stage2_output_dir = tmp_path / "outputs/stage2"
    stage2 = _write_stage2_quality_output(stage2_output_dir)

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_output_dir=stage2_output_dir,
        stage2_checkpoint=stage2,
    )

    stage4_gate = report["stage_gates"]["stage4_causal_next_skill"]
    assert stage4_gate["ready_to_submit"] is True
    assert stage4_gate["stage2_checkpoint_exists"] is True
    assert stage4_gate["head_checkpoint_exists"] is True
    assert stage4_gate["act_init_source"] == "stage2_full_base_checkpoint"
    assert stage4_gate["training_objective"] == "causal_transition_conditioned_next_skill_ce"


def test_unified_training_readiness_stage4_requires_stage2_quality_gate(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage0 = _write_stage0_quality_output(stage0_output_dir, baseline_metrics, skill_count=2)
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    stage2_output_dir = tmp_path / "outputs/stage2"
    stage2_checkpoint = stage2_output_dir / "checkpoints/clstr_full_base-step10000.pt"
    stage2_checkpoint.parent.mkdir(parents=True)
    stage2_checkpoint.write_text("stage2 checkpoint", encoding="utf-8")
    _write_json(stage2_output_dir / "train_report.json", {"status": "ok", "checkpoint": str(stage2_checkpoint)})
    _write_jsonl(stage2_output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0}])
    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_output_dir=stage2_output_dir,
        stage2_checkpoint=stage2_checkpoint,
    )

    stage4_gate = report["stage_gates"]["stage4_causal_next_skill"]
    assert stage4_gate["ready_to_submit"] is False
    assert "stage2_quality_gate_not_ok" in stage4_gate["blockers"]
    assert stage4_gate["stage2_quality_gate"]["status"] == "action_required"
    assert "insufficient_training_steps" in stage4_gate["stage2_quality_gate"]["blockers"]


def test_unified_training_readiness_stage4_ready_when_stage2_quality_gate_passes(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state",
                "action_text": "call a",
                "expert_action": "call a",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )
    stage0_output_dir = tmp_path / "outputs/stage0"
    baseline_metrics = tmp_path / "outputs/stage0_baseline/metrics.json"
    stage0 = _write_stage0_quality_output(stage0_output_dir, baseline_metrics, skill_count=2)
    stage1_output_dir = tmp_path / "outputs/stage1_heads"
    stage1 = _write_stage1_heads_quality_output(stage1_output_dir)
    stage2_output_dir = tmp_path / "outputs/stage2"
    stage2 = _write_stage2_quality_output(stage2_output_dir)

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(
        data_root=data_root,
        allowed_benchmarks={"traject_bench", "toolbench_g3"},
        stage0_checkpoint=stage0,
        stage0_output_dir=stage0_output_dir,
        stage0_baseline_metrics_path=baseline_metrics,
        stage1_checkpoint=stage1,
        stage1_output_dir=stage1_output_dir,
        stage2_output_dir=stage2_output_dir,
        stage2_checkpoint=stage2,
    )

    stage4_gate = report["stage_gates"]["stage4_causal_next_skill"]
    assert stage4_gate["ready_to_submit"] is True
    assert stage4_gate["stage2_quality_gate"]["status"] == "ok"


def test_unified_training_readiness_rejects_candidate_limited_stage2_checkpoint(tmp_path):
    module = _load_audit_module()
    output_dir = tmp_path / "outputs/stage2"
    checkpoint = _write_stage2_quality_output(output_dir)
    train_report_path = output_dir / "train_report.json"
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    train_report["transition_candidate_training"]["next_skill_pool_mode"] = "stage0_candidates"
    train_report["transition_skill_ce"]["candidate_pool"] = "stage0_topm_candidate_set"
    train_report["transition_objective"] = "causal_post_action_unified_retrieval_plus_optional_next_belief_cosine"
    _write_json(train_report_path, train_report)

    gate = module._stage2_quality_gate_report(output_dir, checkpoint)

    assert gate["status"] == "action_required"
    assert "stage2_next_skill_pool_mismatch" in gate["blockers"]


def test_unified_training_readiness_cli_missing_checkpoint_does_not_require_torch(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 1},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "alfworld",
                "state_text": "state",
                "action_text": "look",
                "expert_action": "look",
                "skill_id": "skill/a",
                "provenance": {"split": "train"},
            }
        ],
    )
    output_path = tmp_path / "readiness_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/audit_clstr_unified_training_readiness.py",
            "--data_root",
            str(data_root),
            "--stage0_checkpoint",
            str(tmp_path / "outputs/missing_stage0.pt"),
            "--stage0_output_dir",
            str(tmp_path / "outputs/missing_stage0_dir"),
            "--stage1_checkpoint",
            str(tmp_path / "outputs/missing_stage1.pt"),
            "--stage2_checkpoint",
            str(tmp_path / "outputs/missing_stage2.pt"),
            "--output_path",
            str(output_path),
        ],
        cwd=Path.cwd(),
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["stage_gates"]["stage2_full_base"]["ready_to_submit"] is False
    assert "missing_stage0_checkpoint" in report["stage_gates"]["stage2_full_base"]["blockers"]


def test_unified_training_readiness_exposes_only_four_stage_causal_mainline(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    _write_json(
        data_root / "manifest.json",
        {
            "status": "ok",
            "schema_version": "v2",
            "trajectory_stream": {"total_rows": 1},
            "retrieval_stream": {"total_pairs": 1},
            "skill_pool": {"total_skills": 2},
            "source_inventory": {"missing_required_target_sources": []},
            "leakage_audit": {"status": "ok"},
        },
    )
    _write_json(data_root / "leakage_audit.json", {"status": "ok"})
    _write_jsonl(
        data_root / "source_inventory.jsonl",
        [
            {"source_id": "traject_bench", "available": True, "train_allowed": True},
            {"source_id": "toolbench_g3", "available": True, "train_allowed": True},
            {"source_id": "toolret_training", "available": True, "train_allowed": True},
        ],
    )
    _write_jsonl(data_root / "skill_pool.jsonl", [{"skill_id": "skill/a"}, {"skill_id": "skill/b"}])
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [{"source": "toolret_training", "query_text": "q", "positive_skill_id": "skill/a"}],
    )
    _write_jsonl(
        data_root / "trajectories.jsonl",
        [
            {
                "benchmark": "traject_bench",
                "state_text": "state zero",
                "action_text": "call a",
                "next_observation_text": "observed result",
                "next_state_text": "state one",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "provenance": {"split": "train"},
            }
        ],
    )

    audit = _load_audit_module()
    report = audit.audit_unified_training_readiness(data_root=data_root)

    assert set(report["stage_gates"]) == {
        "stage0_routing",
        "stage1_heads",
        "stage2_full_base",
        "stage4_causal_next_skill",
    }
    stage4_gate = report["stage_gates"]["stage4_causal_next_skill"]
    assert stage4_gate["training_objective"] == "causal_transition_conditioned_next_skill_ce"
    assert stage4_gate["post_action_training_rows"] == 1
    assert "stage3_checkpoint" not in json.dumps(report, sort_keys=True)
    assert report["candidate_recall_readiness"]["status"] == "ok"
    assert report["memory_utility_reliability_readiness"]["status"] == "ok"
    assert report["memory_utility_reliability_readiness"]["memory_utility_gate_mode"] == "dynamic"
    assert report["paper_readiness"]["memory_utility_reliability_ready"] is True


def test_memory_candidate_recall_readiness_accepts_strict_global_contract():
    audit = _load_audit_module()

    report = audit.audit_memory_candidate_recall_readiness(
        candidate_recall_mode="static_plus_dynamic_extra",
        route_scorer="unified_memory",
        static_k=500,
        dynamic_extra_k=64,
        final_k=64,
        denominator_scope="all_eligible_source_strict",
        gold_positive_injected=False,
        legal_pool_source="declared_global_skill_pool",
        equal_budget_comparator_mode="same_final_scorer_static_top_m_plus_d",
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["candidate_union_version"] == "memory_union_v1"
    assert report["candidate_selection_version"] == "stable_declared_pool_v1"
    assert report["tie_break_policy"] == "declared_pool_index_ascending"
    assert report["requested_dynamic_extra_d"] == 64
    assert report["equal_budget_comparator_mode"] == (
        "same_final_scorer_static_top_m_plus_d"
    )


def test_memory_candidate_recall_readiness_rejects_unsupported_claim_contracts():
    audit = _load_audit_module()
    valid = {
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "route_scorer": "unified_memory",
        "static_k": 100,
        "dynamic_extra_k": 20,
        "final_k": 20,
        "denominator_scope": "all_eligible_source_strict",
        "gold_positive_injected": False,
        "legal_pool_source": "declared_global_skill_pool",
        "equal_budget_comparator_mode": "same_final_scorer_static_top_m_plus_d",
    }
    cases = [
        (
            {"denominator_scope": "retained_rows_only"},
            "candidate_recall_denominator_not_all_source_strict",
        ),
        ({"gold_positive_injected": True}, "candidate_recall_gold_positive_injected"),
        (
            {"legal_pool_source": "target_namespace_inferred"},
            "candidate_recall_target_derived_legal_pool",
        ),
        (
            {"equal_budget_comparator_mode": "missing"},
            "candidate_recall_equal_budget_control_missing",
        ),
        ({"final_k": 101}, "candidate_recall_final_k_exceeds_static_k"),
    ]

    for overrides, expected_blocker in cases:
        report = audit.audit_memory_candidate_recall_readiness(**{**valid, **overrides})
        assert report["status"] == "action_required"
        assert expected_blocker in report["blockers"]


def test_memory_utility_reliability_readiness_accepts_only_nonlearned_contract_modes():
    audit = _load_audit_module()

    for mode in ("static", "dynamic", "fixed_alpha", "heuristic"):
        report = audit.audit_memory_utility_reliability_readiness(mode=mode)

        assert report["status"] == "ok"
        assert report["blockers"] == []
        assert report["memory_utility_gate_mode"] == mode
        assert report["zero_history_fallback"] == "exact_static"
        assert report["reliability_feature_schema"] == "memory_utility_features_v1"
        assert report["reliability_changes_memory_state"] is False
        assert report["learned_gate_enabled"] is False


def test_memory_utility_reliability_readiness_rejects_learned_without_audit_and_checkpoint():
    audit = _load_audit_module()

    report = audit.audit_memory_utility_reliability_readiness(mode="learned")

    assert report["status"] == "action_required"
    assert report["learned_gate_enabled"] is False
    assert "memory_utility_audit_report_missing" in report["blockers"]
    assert "memory_utility_gate_checkpoint_missing" in report["blockers"]


def test_memory_utility_reliability_readiness_accepts_hash_bound_passing_gate(tmp_path):
    audit = _load_audit_module()
    audit_path = tmp_path / "memory_utility_audit.json"
    audit_payload = {
        "status": "ok",
        "learned_gate_recommended": True,
        "recommendation_blockers": [],
        "record_schema_version": "memory_utility_route_record_v1",
        "feature_schema": "memory_utility_features_v1",
        "route_manifest_identity": {
            "candidate_union_version": "memory_union_v1",
            "candidate_selection_version": "stable_declared_pool_v1",
            "pool_protocol": "known_global",
        },
    }
    _write_json(audit_path, audit_payload)
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    checkpoint_path = tmp_path / "memory_utility_gate.pt"
    torch.save(
        {
            "stage": "clstr_memory_utility_gate",
            "schema_version": "memory_utility_gate_checkpoint_v1",
            "audit_report_sha256": audit_digest,
            "feature_schema": "memory_utility_features_v1",
            "zero_history_fallback": "exact_static",
            "reliability_changes_memory_state": False,
            "memory_utility_gate_state_dict": {"net.0.weight": torch.ones(1, 11)},
        },
        checkpoint_path,
    )

    report = audit.audit_memory_utility_reliability_readiness(
        mode="learned",
        audit_report_path=audit_path,
        gate_checkpoint_path=checkpoint_path,
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["learned_gate_enabled"] is True
    assert report["audit_report_sha256"] == audit_digest


def test_memory_utility_reliability_readiness_rejects_negative_or_non_global_audit(tmp_path):
    audit = _load_audit_module()
    audit_path = tmp_path / "memory_utility_audit.json"
    _write_json(
        audit_path,
        {
            "status": "ok",
            "learned_gate_recommended": False,
            "recommendation_blockers": ["utility_not_predictable"],
            "feature_schema": "memory_utility_features_v1",
            "route_manifest_identity": {
                "candidate_union_version": "memory_union_v1",
                "candidate_selection_version": "stable_declared_pool_v1",
                "pool_protocol": "appended_untrained",
            },
        },
    )

    report = audit.audit_memory_utility_reliability_readiness(
        mode="learned",
        audit_report_path=audit_path,
    )

    assert report["status"] == "action_required"
    assert "memory_utility_audit_did_not_recommend_learned_gate" in report["blockers"]
    assert "memory_utility_audit_pool_protocol_not_known_global" in report["blockers"]


def test_memory_utility_reliability_readiness_rejects_non_mapping_checkpoint(tmp_path):
    audit = _load_audit_module()
    audit_path = tmp_path / "memory_utility_audit.json"
    _write_json(
        audit_path,
        {
            "status": "ok",
            "learned_gate_recommended": True,
            "recommendation_blockers": [],
            "feature_schema": "memory_utility_features_v1",
            "route_manifest_identity": {
                "candidate_union_version": "memory_union_v1",
                "candidate_selection_version": "stable_declared_pool_v1",
                "pool_protocol": "known_global",
            },
        },
    )
    checkpoint_path = tmp_path / "not_a_gate.pt"
    torch.save(torch.ones(1), checkpoint_path)

    report = audit.audit_memory_utility_reliability_readiness(
        mode="learned",
        audit_report_path=audit_path,
        gate_checkpoint_path=checkpoint_path,
    )

    assert report["status"] == "action_required"
    assert "memory_utility_gate_checkpoint_payload_invalid" in report["blockers"]
