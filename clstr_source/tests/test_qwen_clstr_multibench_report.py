from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from clstr.qwen_clstr_multibench_submit import QwenClstrEvalStage
from clstr.qwen_clstr_multibench_submit import QWEN_CLSTR_MULTIBENCH_STAGE_ORDER


CHAIN_DIGEST = "1" * 64
CHAIN_MANIFEST_SHA = "2" * 64
CORPUS_MANIFEST_SHA = "3" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _self_hash(payload: dict, field: str) -> dict:
    value = dict(payload)
    value[field] = canonical_digest(payload)
    return value


def _frozen_stage(tmp_path: Path) -> QwenClstrEvalStage:
    output_dir = tmp_path / "frozen" / "tau2"
    return QwenClstrEvalStage(
        key="frozen/tau2",
        job_name="test",
        launcher=tmp_path / "launcher.sh",
        exports={
            "OUTPUT_DIR": str(output_dir),
            "EVAL_SCOPE": "smoke",
            "CHECKPOINT_CHAIN_DIGEST": CHAIN_DIGEST,
            "FINAL_CHAIN_MANIFEST_SHA256": CHAIN_MANIFEST_SHA,
            "CORPUS_MANIFEST_SHA256": CORPUS_MANIFEST_SHA,
            "MAX_EVAL_ROWS": "2",
        },
        partition="gpu_a800",
        time_limit="00:10:00",
    )


def test_tau2_full_denominators_follow_the_pinned_base_manifest(tmp_path: Path) -> None:
    from clstr.qwen_clstr_multibench_report import (
        _expected_frozen_rows,
        _expected_native_rows,
    )

    manifest = _self_hash(
        {
            "schema_version": 1,
            "benchmark": "tau2",
            "split": "base",
            "source_row_count": 1208,
        },
        "manifest_sha256",
    )
    manifest_path = tmp_path / "tau2_base_manifest.json"
    _write_json(manifest_path, manifest)
    common_exports = {
        "EVAL_SCOPE": "full",
        "CORPUS_MANIFEST_PATH": str(manifest_path),
        "CORPUS_MANIFEST_SHA256": manifest["manifest_sha256"],
    }
    frozen_exports = {**_frozen_stage(tmp_path).exports, **common_exports}
    frozen_exports.pop("MAX_EVAL_ROWS", None)
    frozen = replace(_frozen_stage(tmp_path), exports=frozen_exports)
    native_exports = {
        **_native_stage(tmp_path).exports,
        **common_exports,
        "TASK_SPLIT": "base",
    }
    native_exports.pop("EXPECTED_SOURCE_ROWS", None)
    native = replace(
        _native_stage(tmp_path),
        exports=native_exports,
    )

    assert _expected_frozen_rows(frozen) == 1208
    assert _expected_native_rows(native) == 1208


def _write_valid_frozen_artifacts(stage: QwenClstrEvalStage) -> tuple[Path, Path, Path]:
    output_dir = Path(stage.exports["OUTPUT_DIR"])
    identity = _self_hash(
        {
            "schema_version": "qwen06_clstr_frozen_route_identity_v1",
            "checkpoint_chain_digest": CHAIN_DIGEST,
            "final_chain_manifest_sha256": CHAIN_MANIFEST_SHA,
            "benchmark_manifest_sha256": CORPUS_MANIFEST_SHA,
            "evaluated_rows": 2,
            "memory_active_rows": 0,
        },
        "identity_sha256",
    )
    identity_path = output_dir / "evaluation_identity.json"
    _write_json(identity_path, identity)
    predictions = []
    for index in range(2):
        predictions.append(
            _self_hash(
                {
                    "row_id": f"row-{index}",
                    "positive_skill_id": "skill/a",
                    "positive_rank": 1,
                    "declared_candidate_count": 2,
                    "ranked_skill_ids": ["skill/a", "skill/b"],
                    "ranked_scores": [1.0, 0.0],
                    "evaluation_identity_sha256": identity["identity_sha256"],
                    "benchmark_manifest_sha256": CORPUS_MANIFEST_SHA,
                },
                "prediction_sha256",
            )
        )
    predictions_path = output_dir / "frozen_route_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = _self_hash(
        {
            "status": "ok",
            "blockers": [],
            "benchmark": "tau2",
            "result_label": "frozen",
            "metric_scope": "next_tool_action_routing",
            "task_success": False,
            "source_rows": 13907,
            "evaluated_rows": 2,
            "prediction_rows": 2,
            "memory_active_rows": 0,
            "zero_history_fallback": "exact_static",
            "skill_merge": {
                "base_skill_count": 67557,
                "benchmark_skill_count": 2,
                "known_benchmark_skill_count": 1,
                "appended_benchmark_skill_count": 1,
                "merged_skill_count": 67558,
            },
            "evaluation_identity_sha256": identity["identity_sha256"],
            "checkpoint_chain_digest": CHAIN_DIGEST,
            "benchmark_manifest_sha256": CORPUS_MANIFEST_SHA,
            "identity_path": str(identity_path),
            "predictions_path": str(predictions_path),
            "routing_metrics": {
                "status": "ok",
                "source_rows": 2,
                "prediction_rows": 2,
                "recall@1": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
                "memory_active_rows": 0,
            },
        },
        "report_sha256",
    )
    report_path = output_dir / "frozen_route_eval_report.json"
    _write_json(report_path, report)
    return report_path, identity_path, predictions_path


def test_frozen_stage_validation_accepts_complete_finite_routing_artifacts(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    _write_valid_frozen_artifacts(stage)

    result = validate_frozen_route_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["metric_scope"] == "routing"
    assert result["task_success"] is False
    assert result["prediction_rows"] == 2
    assert result["memory_active_rows"] == 0
    assert result["zero_history_fallback"] == "exact_static"
    assert result["skill_merge"]["known_benchmark_skill_count"] == 1
    assert result["skill_merge"]["appended_benchmark_skill_count"] == 1


def test_frozen_stage_validation_rejects_inconsistent_skill_merge_counts(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    report_path, _identity, _predictions = _write_valid_frozen_artifacts(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("report_sha256")
    report["skill_merge"]["merged_skill_count"] = 67557
    _write_json(report_path, _self_hash(report, "report_sha256"))

    result = validate_frozen_route_stage(stage)

    assert "invalid_skill_merge_counts" in result["blockers"]


def test_frozen_stage_validation_rejects_incomplete_predictions(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    _report, _identity, predictions_path = _write_valid_frozen_artifacts(stage)
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    predictions_path.write_text(lines[0] + "\n", encoding="utf-8")

    result = validate_frozen_route_stage(stage)

    assert "prediction_row_count_mismatch" in result["blockers"]


def test_frozen_stage_validation_rejects_nonfinite_scores(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    _report, _identity, predictions_path = _write_valid_frozen_artifacts(stage)
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["ranked_scores"][0] = float("nan")
    payload = dict(rows[0])
    payload.pop("prediction_sha256")
    rows[0] = _self_hash(payload, "prediction_sha256")
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = validate_frozen_route_stage(stage)

    assert "nonfinite_prediction_scores" in result["blockers"]


def test_frozen_stage_validation_rejects_checkpoint_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    report_path, _identity, _predictions = _write_valid_frozen_artifacts(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("report_sha256")
    report["checkpoint_chain_digest"] = "9" * 64
    _write_json(report_path, _self_hash(report, "report_sha256"))

    result = validate_frozen_route_stage(stage)

    assert "checkpoint_chain_digest_mismatch" in result["blockers"]


def test_frozen_stage_validation_rejects_routing_labeled_as_task_success(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_frozen_route_stage

    stage = _frozen_stage(tmp_path)
    report_path, _identity, _predictions = _write_valid_frozen_artifacts(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("report_sha256")
    report["task_success"] = True
    _write_json(report_path, _self_hash(report, "report_sha256"))

    result = validate_frozen_route_stage(stage)

    assert "routing_mislabeled_as_task_success" in result["blockers"]


def _native_stage(tmp_path: Path) -> QwenClstrEvalStage:
    output_dir = tmp_path / "native" / "tau2"
    return QwenClstrEvalStage(
        key="native/tau2",
        job_name="test",
        launcher=tmp_path / "launcher.sh",
        exports={
            "OUTPUT_DIR": str(output_dir),
            "EVAL_SCOPE": "smoke",
            "CHECKPOINT_CHAIN_DIGEST": CHAIN_DIGEST,
            "FINAL_CHAIN_MANIFEST_SHA256": CHAIN_MANIFEST_SHA,
            "CORPUS_MANIFEST_SHA256": CORPUS_MANIFEST_SHA,
            "STAGE0_CHECKPOINT_PATH": str(tmp_path / "stage0.pt"),
            "STAGE2_CHECKPOINT_PATH": str(tmp_path / "stage2.pt"),
            "STAGE4_CHECKPOINT_PATH": str(tmp_path / "stage4.pt"),
            "TRAINING_SKILLS_PATH": str(tmp_path / "training_skills.jsonl"),
            "EXPECTED_SOURCE_ROWS": "2",
        },
        partition="gpu_a800",
        time_limit="00:10:00",
    )


def _write_valid_native_artifact(stage: QwenClstrEvalStage) -> Path:
    benchmark = stage.key.split("/", 1)[1]
    candidate_count = int(stage.exports.get("CANDIDATE_COUNT") or 2)
    report = {
        "status": "ok",
        "blockers": [],
        "benchmark": benchmark,
        "route_scorer": "unified_memory",
        "task_success": False,
        "stage0_checkpoint_path": stage.exports["STAGE0_CHECKPOINT_PATH"],
        "stage2_checkpoint_path": stage.exports["STAGE2_CHECKPOINT_PATH"],
        "stage4_checkpoint_path": stage.exports["STAGE4_CHECKPOINT_PATH"],
        "source_eval_rows": 2,
        "retained_eval_rows": 2,
        "route_data_report": {
            "source_rows": 2,
            "stage4_rows": 2,
            "positive_injected_rows": 0,
        },
        "stage0_prior_eval": {"stage4_next_skill_mrr": 0.5},
        "stage4_eval": {
            "stage4_next_skill_recall@1": 0.5,
            "stage4_next_skill_recall@5": 1.0,
            "stage4_next_skill_mrr": 0.75,
            "stage4_candidate_count": float(candidate_count),
            "candidate_recall_final_k": candidate_count,
            "stage4_replay_prefix_used_count": 1.0,
        },
        "strict": {
            "stage4": {
                "strict_source_rows": 2.0,
                "strict_retained_rows": 2.0,
                "strict_stage4_next_skill_mrr": 0.75,
            }
        },
        "model_load": {
            "skill_pool_adapter": {
                "training_skills_path": stage.exports["TRAINING_SKILLS_PATH"],
                "checkpoint_load_order": [
                    "stage0",
                    "stage2",
                    "stage4",
                    "append_benchmark_skills",
                ],
                "trainable_backbone_parameters": 0,
                "stage0": {"stage0_checkpoint_path": stage.exports["STAGE0_CHECKPOINT_PATH"]},
                "stage2": {"head_checkpoint_path": stage.exports["STAGE2_CHECKPOINT_PATH"]},
                "stage4": {"head_checkpoint_path": stage.exports["STAGE4_CHECKPOINT_PATH"]},
            }
        },
    }
    filename = {
        "toolbench_g3": "full_clstr_route_eval_report.json",
        "toolsandbox": "toolsandbox_full_clstr_route_eval_report.json",
        "tau2": "tau2_full_clstr_route_eval_report.json",
    }[benchmark]
    path = Path(stage.exports["OUTPUT_DIR"]) / filename
    _write_json(path, report)
    return path


def test_native_stage_validation_accepts_causal_replay_routing_artifact(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    stage = _native_stage(tmp_path)
    _write_valid_native_artifact(stage)

    result = validate_native_route_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["metric_scope"] == "routing"
    assert result["task_success"] is False
    assert result["memory_active_rows"] == 1.0


def test_native_stage_validation_rejects_denominator_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    stage = _native_stage(tmp_path)
    report_path = _write_valid_native_artifact(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_eval_rows"] = 1
    _write_json(report_path, report)

    result = validate_native_route_stage(stage)

    assert "source_row_denominator_mismatch" in result["blockers"]


def test_native_stage_validation_rejects_checkpoint_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    stage = _native_stage(tmp_path)
    report_path = _write_valid_native_artifact(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stage4_checkpoint_path"] = str(tmp_path / "wrong.pt")
    _write_json(report_path, report)

    result = validate_native_route_stage(stage)

    assert "stage4_checkpoint_path_mismatch" in result["blockers"]


def test_native_toolbench_validation_rejects_candidate_budget_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    base = _native_stage(tmp_path)
    exports = dict(base.exports)
    exports["OUTPUT_DIR"] = str(tmp_path / "native" / "toolbench_g3")
    exports["CANDIDATE_COUNT"] = "100"
    stage = replace(base, key="native/toolbench_g3", exports=exports)
    report_path = _write_valid_native_artifact(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stage4_eval"]["stage4_candidate_count"] = 64.0
    report["stage4_eval"]["candidate_recall_final_k"] = 64
    _write_json(report_path, report)

    result = validate_native_route_stage(stage)

    assert "candidate_budget_mismatch" in result["blockers"]


def test_native_stage_validation_rejects_missing_replay_evidence(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    stage = _native_stage(tmp_path)
    report_path = _write_valid_native_artifact(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stage4_eval"]["stage4_replay_prefix_used_count"] = 0.0
    _write_json(report_path, report)

    result = validate_native_route_stage(stage)

    assert "no_causal_replay_evidence" in result["blockers"]


def test_native_stage_validation_rejects_routing_labeled_as_task_success(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_native_route_stage

    stage = _native_stage(tmp_path)
    report_path = _write_valid_native_artifact(stage)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["task_success"] = True
    _write_json(report_path, report)

    result = validate_native_route_stage(stage)

    assert "routing_mislabeled_as_task_success" in result["blockers"]


def _alfworld_stage(tmp_path: Path) -> QwenClstrEvalStage:
    output_dir = tmp_path / "closed_loop" / "alfworld_valid_seen"
    return QwenClstrEvalStage(
        key="closed_loop/alfworld_valid_seen",
        job_name="test",
        launcher=tmp_path / "launcher.sh",
        exports={
            "OUTPUT_DIR": str(output_dir),
            "EVAL_SCOPE": "smoke",
            "CHECKPOINT_CHAIN_DIGEST": CHAIN_DIGEST,
            "FINAL_CHAIN_MANIFEST_SHA256": CHAIN_MANIFEST_SHA,
            "CORPUS_MANIFEST_SHA256": CORPUS_MANIFEST_SHA,
            "STAGE0_CHECKPOINT_PATH": str(tmp_path / "stage0.pt"),
            "STAGE2_CHECKPOINT_PATH": str(tmp_path / "stage2.pt"),
            "STAGE4_CHECKPOINT_PATH": str(tmp_path / "stage4.pt"),
            "TRAINING_SKILLS_PATH": str(tmp_path / "training_skills.jsonl"),
            "ALFWORLD_SKILLS_PATH": str(tmp_path / "alfworld_skills.jsonl"),
            "SPLIT": "valid_seen",
            "SCORER_MODE": "unified_memory_admissible_action",
            "MEMORY_PROTOCOL": "stateful_post_action_v1",
            "RELIABILITY_MODE": "causal_gate",
            "SAFE_MEMORY_RESIDUAL_BOUND": "2.0",
            "EXPECTED_EPISODES": "2",
        },
        partition="gpu_a800",
        time_limit="00:10:00",
    )


def _write_valid_alfworld_artifacts(stage: QwenClstrEvalStage) -> tuple[Path, Path]:
    output_dir = Path(stage.exports["OUTPUT_DIR"])
    metrics = {
        "status": "ok",
        "method": "test",
        "split": "valid_seen",
        "success_rate": 0.5,
        "average_reward": 0.5,
        "average_goal_condition_points": 0.75,
        "average_episode_steps": 2.0,
        "episode_count": 2,
        "episodes": 2,
        "stage0_checkpoint_path": stage.exports["STAGE0_CHECKPOINT_PATH"],
        "checkpoint_path": stage.exports["STAGE2_CHECKPOINT_PATH"],
        "stage4_checkpoint_path": stage.exports["STAGE4_CHECKPOINT_PATH"],
        "skill_rows_path_override": stage.exports["TRAINING_SKILLS_PATH"],
        "benchmark_skill_rows_path": stage.exports["ALFWORLD_SKILLS_PATH"],
        "candidate_scorer_mode": "unified_memory_admissible_action",
        "memory_active_protocol": True,
        "memory_protocol": "stateful_post_action_v1",
        "memory_utility_reliability_mode": "causal_gate",
        "safe_memory_residual_bound": 2.0,
        "causal_update_count": 1,
        "stage4_overlay_loaded": True,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    run_rows = []
    for episode in range(2):
        update_count = int(episode)
        run_rows.append(
            {
                "episode_index": episode,
                "split": "valid_seen",
                "gamefile": f"game-{episode}",
                "success": bool(episode),
                "points": float(episode),
                "goal_condition_points": 1.0,
                "steps": 2.0,
                "causal_update_count_final": 1,
                "memory_protocol": "stateful_post_action_v1",
                "action_trace": ["look", "go to fridge 1"],
                "policy_metadata_trace": [
                    {
                        "policy_family": "clstr_unified_memory_admissible_action_scorer",
                        "causal_update_count": update_count,
                        "uses_recurrent_m_t": bool(update_count),
                        "memory_protocol": "stateful_post_action_v1",
                        "candidate_skill_mappings": [
                            {
                                "action": "look",
                                "mapped_skill_id": "alfworld_action/look",
                            },
                            {
                                "action": "go to fridge 1",
                                "mapped_skill_id": "alfworld_action/go to fridge 1",
                            },
                        ],
                    }
                ],
            }
        )
    run_path = output_dir / "run.jsonl"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        "".join(json.dumps(row) + "\n" for row in run_rows),
        encoding="utf-8",
    )
    return metrics_path, run_path


def _write_valid_alfworld_executor_artifacts(
    stage: QwenClstrEvalStage,
) -> tuple[Path, Path, Path]:
    output_dir = Path(stage.exports["OUTPUT_DIR"])
    split = stage.exports["SPLIT"]
    qwen_only_metrics = {
        "status": "ok",
        "method": "qwen_clstr_executor_gate_qwen_only",
        "split": split,
        "success_rate": 0.0,
        "average_reward": 0.0,
        "average_goal_condition_points": 0.0,
        "average_episode_steps": 2.0,
        "episode_count": 2,
        "episodes": 2,
        "policy_family": "qwen_direct_admissible",
        "uses_clstr": False,
        "qwen_direct_baseline": True,
        "loop_guard": True,
        "scoring_method": "generate",
        "model_name_or_path": str(stage.exports.get("QWEN_MODEL_NAME_OR_PATH") or "qwen"),
        "executor_gate": True,
    }
    routing_init_report = {
        "benchmark_skill_source_path": stage.exports["ALFWORLD_SKILLS_PATH"],
        "final_chain_adapter": {
            "checkpoint_load_order": [
                "stage0",
                "stage2",
                "stage4",
                "append_benchmark_skills",
            ],
            "training_skills_path": stage.exports["TRAINING_SKILLS_PATH"],
            "trainable_backbone_parameters": 0,
            "stage0": {
                "stage0_checkpoint_path": stage.exports["STAGE0_CHECKPOINT_PATH"],
            },
        },
    }
    clstr_metrics = {
        "status": "ok",
        "method": "qwen_clstr_executor_gate_qwen_plus_clstr",
        "split": split,
        "success_rate": 0.5,
        "average_reward": 0.5,
        "average_goal_condition_points": 0.5,
        "average_episode_steps": 2.0,
        "episode_count": 2,
        "episodes": 2,
        "policy_family": "qwen_clstr_hybrid_executor_gate",
        "uses_clstr": True,
        "is_clstr_result": True,
        "qwen_direct_baseline": False,
        "executor_gate": True,
        "scoring_method": "generate",
        "qwen_weight": 1.0,
        "clstr_weight": 0.25,
        "qwen_score_mode": "proposal_bonus",
        "score_normalization": "qwen:proposal_bonus;clstr:row_zscore",
        "loop_guard": True,
        "checkpoint_path": stage.exports["STAGE2_CHECKPOINT_PATH"],
        "stage4_checkpoint_path": stage.exports["STAGE4_CHECKPOINT_PATH"],
        "stage4_overlay_loaded": True,
        "skill_rows_path_override": stage.exports["TRAINING_SKILLS_PATH"],
        "clstr_prior_mode": "unified_memory_concrete_action",
        "memory_utility_reliability_mode": "cmc",
        "routing_init_report": routing_init_report,
    }
    qwen_metrics_path = output_dir / "qwen_only" / "metrics.json"
    clstr_metrics_path = output_dir / "qwen_plus_clstr" / "metrics.json"
    _write_json(qwen_metrics_path, qwen_only_metrics)
    _write_json(clstr_metrics_path, clstr_metrics)

    qwen_rows = []
    clstr_rows = []
    for episode in range(2):
        gamefile = f"game-{episode}"
        qwen_rows.append(
            {
                "episode_index": episode,
                "split": split,
                "gamefile": gamefile,
                "success": False,
                "points": 0.0,
                "goal_condition_points": 0.0,
                "steps": 2.0,
            }
        )
        clstr_rows.append(
            {
                "episode_index": episode,
                "split": split,
                "gamefile": gamefile,
                "success": bool(episode),
                "points": float(episode),
                "goal_condition_points": float(episode),
                "steps": 2.0,
                "candidate_trace": [["look", "inventory"], ["look", "inventory"]],
                "chosen_indices": [0, 1],
                "chosen_action_trace": ["look", "inventory"],
                "policy_metadata_trace": [
                    {
                        "executor_gate": True,
                        "clstr_route_scorer": "unified_memory_concrete_action",
                        "clstr_memory_source": "initial_belief",
                        "clstr_replay_prefix_len": 0,
                        "clstr_replay_prefix_used_count": 0,
                        "uses_recurrent_m_t": False,
                    },
                    {
                        "executor_gate": True,
                        "clstr_route_scorer": "unified_memory_concrete_action",
                        "clstr_memory_source": "replay_prefix",
                        "clstr_replay_prefix_len": 1,
                        "clstr_replay_prefix_used_count": 1,
                        "uses_recurrent_m_t": True,
                    },
                ],
            }
        )
    qwen_run_path = output_dir / "qwen_only" / "run.jsonl"
    clstr_run_path = output_dir / "qwen_plus_clstr" / "run.jsonl"
    for path, rows in ((qwen_run_path, qwen_rows), (clstr_run_path, clstr_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    gate_summary = {
        "status": "ok",
        "method": "qwen_clstr_executor_gate",
        "splits": [split],
        "max_episodes": 2,
        "max_steps": 50,
        "qwen_only": qwen_only_metrics,
        "qwen_plus_clstr": clstr_metrics,
        "delta": {"success_rate": 0.5},
    }
    gate_path = output_dir / "gate_summary.json"
    _write_json(gate_path, gate_summary)
    return clstr_metrics_path, clstr_run_path, gate_path


def _write_valid_alfworld_control_import(stage: QwenClstrEvalStage) -> Path:
    output_dir = Path(stage.exports["OUTPUT_DIR"])
    qwen_dir = output_dir / "qwen_only"
    source_dir = output_dir / "control_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_metrics = source_dir / "metrics.json"
    source_run = source_dir / "run.jsonl"
    source_identity = source_dir / "method_executor_identity.json"
    source_report = source_dir / "method_alfworld_executor_report.json"
    source_metrics.write_bytes((qwen_dir / "metrics.json").read_bytes())
    source_run.write_bytes((qwen_dir / "run.jsonl").read_bytes())
    _write_json(source_identity, {"status": "canonical"})
    _write_json(source_report, {"status": "canonical"})
    gamefiles = ["game-0", "game-1"]
    payload = {
        "schema_version": "alfworld_qwen_control_import_v1",
        "status": "ok",
        "split": "valid_seen",
        "source": {
            "metrics": sha256_path(source_metrics),
            "run": sha256_path(source_run),
            "executor_identity": sha256_path(source_identity),
            "executor_report": sha256_path(source_report),
        },
        "imported": {
            "metrics": sha256_path(qwen_dir / "metrics.json"),
            "run": sha256_path(qwen_dir / "run.jsonl"),
        },
        "split_manifest": None,
        "episode_count": 2,
        "gamefile_digest": canonical_digest(sorted(gamefiles)),
        "executor_protocol_version": "method_alfworld_qwen3_14b_executor_v1",
        "qwen_model_digest": "c8c9a22462bf04dee2cbce3e8d216819e28b46cbe50b247083d91cdb546acfe5",
        "prefix_equivalence": {
            "rows": 2,
            "all_equal": True,
            "reference_run": sha256_path(source_run),
            "compared_fields": [
                "gamefile",
                "chosen_action_trace",
                "success",
                "steps",
            ],
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    path = output_dir / "control_import.json"
    _write_json(path, payload)
    return path


def test_alfworld_stage_validation_accepts_complete_memory_active_task_success(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _write_valid_alfworld_artifacts(stage)

    result = validate_alfworld_closed_loop_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["metric_scope"] == "task_success"
    assert result["task_success"] is True
    assert result["episode_rows"] == 2
    assert result["memory_active_steps"] == 1
    assert result["unmapped_candidate_count"] == 0


def test_alfworld_stage_validation_accepts_executor_gate_artifacts(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    base = _alfworld_stage(tmp_path)
    exports = {
        **base.exports,
        "SPLITS": base.exports["SPLIT"],
        "RELIABILITY_MODE": "cmc",
        "QWEN_MODEL_NAME_OR_PATH": str(tmp_path / "qwen"),
        "QWEN_WEIGHT": "1.0",
        "CLSTR_WEIGHT": "0.25",
        "CLSTR_PRIOR_MODE": "unified_memory_concrete_action",
        "SCORING_METHOD": "generate",
        "LOOP_GUARD": "1",
        "MAX_STEPS": "50",
    }
    exports.pop("SPLIT")
    stage = replace(base, exports=exports)
    fixture_stage = replace(stage, exports={**exports, "SPLIT": "valid_seen"})
    _write_valid_alfworld_executor_artifacts(fixture_stage)

    result = validate_alfworld_closed_loop_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["artifact_layout"] == "executor_gate"
    assert result["episode_rows"] == 2
    assert result["control_episode_rows"] == 2
    assert result["memory_active_steps"] == 2
    assert result["replay_prefix_steps"] == 2
    assert result["success_rate"] == 0.5
    assert result["control_success_rate"] == 0.0


def test_alfworld_stage_validation_accepts_provenance_bound_imported_control(
    tmp_path,
):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    base = _alfworld_stage(tmp_path)
    exports = {
        **base.exports,
        "SPLITS": base.exports["SPLIT"],
        "RELIABILITY_MODE": "cmc",
        "QWEN_MODEL_NAME_OR_PATH": str(tmp_path / "qwen"),
        "QWEN_WEIGHT": "1.0",
        "CLSTR_WEIGHT": "0.25",
        "CLSTR_PRIOR_MODE": "unified_memory_concrete_action",
        "SCORING_METHOD": "generate",
        "LOOP_GUARD": "1",
        "MAX_STEPS": "50",
        "RUN_QWEN_ONLY": "0",
    }
    exports.pop("SPLIT")
    stage = replace(base, exports=exports)
    fixture_stage = replace(stage, exports={**exports, "SPLIT": "valid_seen"})
    _write_valid_alfworld_executor_artifacts(fixture_stage)
    import_path = _write_valid_alfworld_control_import(fixture_stage)
    stage = replace(
        stage,
        exports={**stage.exports, "QWEN_CONTROL_IMPORT_MANIFEST": str(import_path)},
    )

    result = validate_alfworld_closed_loop_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["control_import_manifest_path"] == str(import_path)


def test_alfworld_stage_validation_rejects_missing_import_provenance(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    base = _alfworld_stage(tmp_path)
    exports = {
        **base.exports,
        "SPLITS": base.exports["SPLIT"],
        "RUN_QWEN_ONLY": "0",
        "QWEN_CONTROL_IMPORT_MANIFEST": str(tmp_path / "missing-import.json"),
    }
    exports.pop("SPLIT")
    stage = replace(base, exports=exports)
    fixture_stage = replace(stage, exports={**exports, "SPLIT": "valid_seen"})
    _write_valid_alfworld_executor_artifacts(fixture_stage)

    result = validate_alfworld_closed_loop_stage(stage)

    assert "missing_or_invalid_qwen_control_import" in result["blockers"]


def test_alfworld_stage_validation_rejects_executor_without_qwen_control(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _write_valid_alfworld_executor_artifacts(stage)
    (Path(stage.exports["OUTPUT_DIR"]) / "qwen_only" / "metrics.json").unlink()

    result = validate_alfworld_closed_loop_stage(stage)

    assert "missing_or_invalid_qwen_only_metrics" in result["blockers"]


def test_alfworld_stage_validation_rejects_executor_gate_summary_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _metrics, _run, gate_path = _write_valid_alfworld_executor_artifacts(stage)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["qwen_plus_clstr"]["success_rate"] = 0.25
    _write_json(gate_path, gate)

    result = validate_alfworld_closed_loop_stage(stage)

    assert "executor_gate_summary_mismatch" in result["blockers"]


def test_alfworld_stage_validation_derives_executable_game_denominator(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    base = _alfworld_stage(tmp_path)
    exports = dict(base.exports)
    exports.pop("EXPECTED_EPISODES")
    exports["DATA_DIR"] = str(tmp_path / "alfworld_data")
    stage = replace(base, exports=exports)
    split_root = Path(exports["DATA_DIR"]) / "json_2.1.1" / "valid_seen"
    for index in range(3):
        trial = split_root / f"task-{index}" / "trial-0"
        trial.mkdir(parents=True)
        (trial / "traj_data.json").write_text("{}\n", encoding="utf-8")
        if index < 2:
            (trial / "game.tw-pddl").write_text("game\n", encoding="utf-8")
    _write_valid_alfworld_artifacts(stage)

    result = validate_alfworld_closed_loop_stage(stage)

    assert result["status"] == "ok"
    assert result["blockers"] == []
    assert result["expected_episodes"] == 2


def test_alfworld_stage_validation_rejects_episode_denominator_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _metrics, run_path = _write_valid_alfworld_artifacts(stage)
    lines = run_path.read_text(encoding="utf-8").splitlines()
    run_path.write_text(lines[0] + "\n", encoding="utf-8")

    result = validate_alfworld_closed_loop_stage(stage)

    assert "episode_row_count_mismatch" in result["blockers"]


def test_alfworld_stage_validation_rejects_checkpoint_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    metrics_path, _run = _write_valid_alfworld_artifacts(stage)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["stage0_checkpoint_path"] = str(tmp_path / "wrong.pt")
    _write_json(metrics_path, metrics)

    result = validate_alfworld_closed_loop_stage(stage)

    assert "stage0_checkpoint_path_mismatch" in result["blockers"]


def test_alfworld_stage_validation_rejects_safe_memory_bound_drift(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    metrics_path, _run = _write_valid_alfworld_artifacts(stage)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["safe_memory_residual_bound"] = 1.0
    _write_json(metrics_path, metrics)

    result = validate_alfworld_closed_loop_stage(stage)

    assert "safe_memory_residual_bound_mismatch" in result["blockers"]


def test_alfworld_stage_validation_rejects_unmapped_actions(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _metrics, run_path = _write_valid_alfworld_artifacts(stage)
    rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["policy_metadata_trace"][0]["candidate_skill_mappings"][0][
        "mapped_skill_id"
    ] = None
    run_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = validate_alfworld_closed_loop_stage(stage)

    assert "unmapped_admissible_actions" in result["blockers"]


def test_alfworld_stage_validation_rejects_no_stateful_update_evidence(tmp_path):
    from clstr.qwen_clstr_multibench_report import validate_alfworld_closed_loop_stage

    stage = _alfworld_stage(tmp_path)
    _metrics, run_path = _write_valid_alfworld_artifacts(stage)
    rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        for metadata in row["policy_metadata_trace"]:
            metadata["causal_update_count"] = 0
            metadata["uses_recurrent_m_t"] = False
    run_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = validate_alfworld_closed_loop_stage(stage)

    assert "no_causal_stateful_update_evidence" in result["blockers"]


def _all_stages(tmp_path: Path) -> list[QwenClstrEvalStage]:
    stages = []
    for index, key in enumerate(QWEN_CLSTR_MULTIBENCH_STAGE_ORDER):
        stages.append(
            QwenClstrEvalStage(
                key=key,
                job_name=f"job-{index}",
                launcher=tmp_path / "launcher.sh",
                exports={
                    "OUTPUT_DIR": str(tmp_path / key),
                    "EVAL_SCOPE": "smoke",
                    "CHECKPOINT_CHAIN_DIGEST": CHAIN_DIGEST,
                    "FINAL_CHAIN_MANIFEST_SHA256": CHAIN_MANIFEST_SHA,
                    "CORPUS_MANIFEST_SHA256": f"{index + 4:064x}",
                },
                partition="gpu_a800",
                time_limit="00:10:00",
            )
        )
    return stages


def _write_valid_job_evidence(
    tmp_path: Path,
    stages: list[QwenClstrEvalStage],
) -> Path:
    registry_path = tmp_path / "submission-registry.json"
    stage_order = [stage.key for stage in stages]
    submission_identity = {
        "stage_order": stage_order,
        "checkpoint_chain_digest": CHAIN_DIGEST,
        "final_chain_manifest_sha256": CHAIN_MANIFEST_SHA,
        "scope": "smoke",
        "corpus_manifest_sha256_by_stage": {
            stage.key: stage.exports["CORPUS_MANIFEST_SHA256"]
            for stage in stages
        },
    }
    registry = {
        "schema_version": 1,
        "status": "submitted",
        "stage_order": stage_order,
        "submission_identity": submission_identity,
        "stages": {
            stage.key: {
                "status": "submitted",
                "job_id": str(1201 + index),
            }
            for index, stage in enumerate(stages)
        },
    }
    _write_json(registry_path, registry)
    evidence = _self_hash(
        {
            "schema_version": "qwen06_clstr_multibench_job_evidence_v1",
            "status": "ok",
            "blockers": [],
            "stage_order": stage_order,
            "input_fingerprint": "8" * 64,
            "plan_fingerprint": "9" * 64,
            "submission_identity": submission_identity,
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "jobs": {
                stage.key: {
                    "job_id": str(1201 + index),
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                }
                for index, stage in enumerate(stages)
            },
        },
        "evidence_sha256",
    )
    path = tmp_path / "job-evidence.json"
    _write_json(path, evidence)
    return path


def test_multibench_report_separates_routing_from_task_success_and_self_hashes(
    tmp_path,
    monkeypatch,
):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)

    def _routing(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    def _task_success(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
            "success_rate": 0.5,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        _task_success,
    )
    output_path = tmp_path / "smoke_gate.json"
    markdown_path = tmp_path / "smoke_gate.md"

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=output_path,
        markdown_path=markdown_path,
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert list(report["routing"]["frozen"]) == [
        "toolbench_g3",
        "toolsandbox",
        "tau2",
        "alfworld_offline",
    ]
    assert list(report["routing"]["native"]) == [
        "toolbench_g3",
        "toolsandbox",
        "tau2",
    ]
    assert list(report["task_success"]) == [
        "alfworld_valid_seen",
        "alfworld_valid_unseen",
    ]
    assert all(not row["task_success"] for row in report["routing"]["frozen"].values())
    assert all(not row["task_success"] for row in report["routing"]["native"].values())
    assert all(row["task_success"] for row in report["task_success"].values())
    payload = dict(report)
    recorded = payload.pop("report_sha256")
    assert canonical_digest(payload) == recorded
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert markdown_path.read_text(encoding="utf-8") == (
        report_module.render_qwen_clstr_multibench_markdown(report)
    )


def test_multibench_report_fails_closed_when_any_stage_has_blockers(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)

    def _routing(stage):
        blockers = ["no_causal_replay_evidence"] if stage.key == "native/tau2" else []
        return {
            "status": "action_required" if blockers else "ok",
            "blockers": blockers,
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "blocked.json",
    )

    assert report["status"] == "action_required"
    assert "native/tau2:no_causal_replay_evidence" in report["blockers"]


def test_full_multibench_report_records_one_accepted_smoke_gate(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    accepted_gate_sha = "5" * 64
    stages = [
        replace(
            stage,
            exports={
                **stage.exports,
                "EVAL_SCOPE": "full",
                "ACCEPTED_SMOKE_GATE_SHA256": accepted_gate_sha,
            },
        )
        for stage in _all_stages(tmp_path)
    ]

    def _routing(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "full-report.json",
    )

    assert report["status"] == "ok"
    assert report["accepted_smoke_gate_sha256"] == accepted_gate_sha


def test_multibench_report_includes_exact_terminal_job_evidence(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    evidence_path = _write_valid_job_evidence(tmp_path, stages)

    def _routing(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "report-with-jobs.json",
        job_evidence_path=evidence_path,
    )

    assert report["status"] == "ok"
    assert report["execution"]["status"] == "ok"
    assert report["execution"]["jobs"]["native/tau2"] == {
        "job_id": "1207",
        "state": "COMPLETED",
        "exit_code": "0:0",
    }


def test_multibench_report_rejects_job_evidence_from_another_chain(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    evidence_path = _write_valid_job_evidence(tmp_path, stages)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("evidence_sha256")
    evidence["submission_identity"]["checkpoint_chain_digest"] = "6" * 64
    _write_json(evidence_path, _self_hash(evidence, "evidence_sha256"))

    def _routing(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "wrong-chain-jobs.json",
        job_evidence_path=evidence_path,
    )

    assert report["status"] == "action_required"
    assert "jobs:checkpoint_chain_digest_mismatch" in report["blockers"]


def test_multibench_report_rejects_job_ids_that_drift_from_the_registry(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    evidence_path = _write_valid_job_evidence(tmp_path, stages)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("evidence_sha256")
    evidence["jobs"]["native/tau2"]["job_id"] = "9999"
    _write_json(evidence_path, _self_hash(evidence, "evidence_sha256"))

    def _routing(stage):
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "wrong-job-id.json",
        job_evidence_path=evidence_path,
    )

    assert report["status"] == "action_required"
    assert "jobs:native/tau2:registry_job_id_mismatch" in report["blockers"]


def test_multibench_report_rejects_stage_order_drift(tmp_path):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    stages[0], stages[1] = stages[1], stages[0]

    with pytest.raises(ValueError, match="fixed nine-stage order"):
        report_module.build_qwen_clstr_multibench_report(
            stages=stages,
            output_path=tmp_path / "invalid-order.json",
        )


@pytest.mark.parametrize(
    ("export_name", "drifted_value", "message"),
    [
        ("CHECKPOINT_CHAIN_DIGEST", "9" * 64, "checkpoint-chain digest"),
        ("FINAL_CHAIN_MANIFEST_SHA256", "8" * 64, "final-chain manifest"),
        ("EVAL_SCOPE", "full", "evaluation scope"),
    ],
)
def test_multibench_report_rejects_cross_stage_identity_drift(
    tmp_path,
    export_name,
    drifted_value,
    message,
):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    drifted_exports = dict(stages[-1].exports)
    drifted_exports[export_name] = drifted_value
    stages[-1] = replace(stages[-1], exports=drifted_exports)

    with pytest.raises(ValueError, match=message):
        report_module.build_qwen_clstr_multibench_report(
            stages=stages,
            output_path=tmp_path / "identity-drift.json",
        )


def test_multibench_report_rejects_an_unpinned_corpus_manifest(tmp_path):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)
    drifted_exports = dict(stages[-1].exports)
    drifted_exports["CORPUS_MANIFEST_SHA256"] = ""
    stages[-1] = replace(stages[-1], exports=drifted_exports)

    with pytest.raises(ValueError, match="corpus manifest"):
        report_module.build_qwen_clstr_multibench_report(
            stages=stages,
            output_path=tmp_path / "missing-corpus.json",
        )


def test_multibench_report_fails_closed_on_validator_scope_drift(tmp_path, monkeypatch):
    from clstr import qwen_clstr_multibench_report as report_module

    stages = _all_stages(tmp_path)

    def _routing(stage):
        drifted = stage.key == "native/tau2"
        return {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success" if drifted else "routing",
            "task_success": drifted,
        }

    monkeypatch.setattr(report_module, "validate_frozen_route_stage", _routing)
    monkeypatch.setattr(report_module, "validate_native_route_stage", _routing)
    monkeypatch.setattr(
        report_module,
        "validate_alfworld_closed_loop_stage",
        lambda stage: {
            "status": "ok",
            "blockers": [],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
        },
    )

    report = report_module.build_qwen_clstr_multibench_report(
        stages=stages,
        output_path=tmp_path / "scope-drift.json",
    )

    assert report["status"] == "action_required"
    assert "native/tau2:routing_scope_mismatch" in report["blockers"]
    assert "native/tau2:routing_mislabeled_as_task_success" in report["blockers"]


def test_multibench_markdown_keeps_routing_and_task_success_in_separate_tables():
    from clstr.qwen_clstr_multibench_report import (
        render_qwen_clstr_multibench_markdown,
    )

    report = {
        "status": "ok",
        "blockers": [],
        "scope": "full",
        "checkpoint_chain_digest": CHAIN_DIGEST,
        "accepted_smoke_gate_sha256": "5" * 64,
        "report_sha256": "7" * 64,
        "execution": {
            "status": "ok",
            "jobs": {
                "frozen/toolbench_g3": {
                    "job_id": "1201",
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                }
            },
        },
        "routing": {
            "frozen": {
                "toolbench_g3": {
                    "status": "ok",
                    "evaluated_rows": 1362,
                    "memory_active_rows": 0,
                    "skill_merge": {
                        "known_benchmark_skill_count": 398,
                        "appended_benchmark_skill_count": 0,
                    },
                    "routing_metrics": {
                        "recall@1": 0.6,
                        "recall@5": 0.8,
                        "mrr": 0.7,
                    },
                }
            },
            "native": {
                "tau2": {
                    "status": "ok",
                    "source_rows": 13907,
                    "retained_rows": 13900,
                    "memory_active_rows": 12000,
                    "stage4_eval": {
                        "stage4_next_skill_recall@1": 0.5,
                        "stage4_next_skill_recall@5": 0.75,
                        "stage4_next_skill_mrr": 0.625,
                    },
                }
            },
        },
        "task_success": {
            "alfworld_valid_seen": {
                "status": "ok",
                "episode_rows": 251,
                "success_rate": 0.42,
                "memory_active_steps": 100,
            }
        },
    }

    markdown = render_qwen_clstr_multibench_markdown(report)

    assert "## Frozen comparable routing (not task success)" in markdown
    assert "| toolbench_g3 | 1362 | 0.6000 | 0.8000 | 0.7000 | 398 | 0 | 0 | ok |" in markdown
    assert "## Native causal routing (not task success)" in markdown
    assert "| tau2 | 13907 | 13900 | 0.5000 | 0.7500 | 0.6250 | 12000 | ok |" in markdown
    assert "## ALFWorld closed-loop task success" in markdown
    assert "| alfworld_valid_seen | 251 | 0.4200 | 100 | ok |" in markdown
    assert f"Checkpoint chain: `{CHAIN_DIGEST}`" in markdown
    assert f"Accepted smoke gate: `{'5' * 64}`" in markdown
    assert "## Slurm execution evidence" in markdown
    assert "| frozen/toolbench_g3 | 1201 | COMPLETED | 0:0 |" in markdown
    assert "- None." in markdown


def test_multibench_report_cli_forwards_config_scope_and_output(monkeypatch, tmp_path):
    import scripts.build_qwen06_clstr_multibench_report as cli

    config_path = tmp_path / "evaluation-config.json"
    output_path = tmp_path / "smoke-gate.json"
    markdown_path = tmp_path / "smoke-gate.md"
    job_evidence_path = tmp_path / "job-evidence.json"
    _write_json(config_path, {"project_root": "/repo"})
    captured = {}
    fake_stages = [object()]

    def fake_build(config, *, scope, partition):
        captured["build"] = {
            "config": config,
            "scope": scope,
            "partition": partition,
        }
        return fake_stages

    def fake_report(**kwargs):
        captured["report"] = kwargs
        return {"status": "ok", "report_sha256": "7" * 64}

    monkeypatch.setattr(cli, "build_qwen_clstr_multibench_stages", fake_build)
    monkeypatch.setattr(cli, "build_qwen_clstr_multibench_report", fake_report)
    args = cli.build_parser().parse_args(
        [
            "--evaluation_config_path",
            str(config_path),
            "--output_path",
            str(output_path),
            "--markdown_path",
            str(markdown_path),
            "--job_evidence_path",
            str(job_evidence_path),
            "--scope",
            "smoke",
            "--partition",
            "gpu_a800,gpu_h100",
        ]
    )

    report = cli.run_from_args(args)

    assert report == {"status": "ok", "report_sha256": "7" * 64}
    assert captured["build"] == {
        "config": {"project_root": "/repo"},
        "scope": "smoke",
        "partition": "gpu_a800,gpu_h100",
    }
    assert captured["report"] == {
        "stages": fake_stages,
        "output_path": str(output_path),
        "markdown_path": str(markdown_path),
        "job_evidence_path": str(job_evidence_path),
    }
