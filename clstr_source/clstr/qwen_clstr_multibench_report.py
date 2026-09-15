from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_multibench_submit import (
    QWEN_CLSTR_MULTIBENCH_STAGE_ORDER,
    QwenClstrEvalStage,
)


FULL_FROZEN_DENOMINATORS = {
    "toolbench_g3": 1362,
    "toolsandbox": 115,
    "tau2": 13907,
    "alfworld_offline": 5858,
}
FULL_NATIVE_DENOMINATORS = {
    "toolbench_g3": 1362,
    "toolsandbox": 115,
    "tau2": 13907,
}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object: {source}")
    return payload


def _self_hash_valid(payload: Mapping[str, Any], *, field: str) -> bool:
    recorded = str(payload.get(field) or "")
    digest_payload = dict(payload)
    digest_payload.pop(field, None)
    return bool(recorded) and canonical_digest(digest_payload) == recorded


def _add(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _pinned_corpus_source_rows(stage: QwenClstrEvalStage) -> int | None:
    raw_path = str(stage.exports.get("CORPUS_MANIFEST_PATH") or "").strip()
    if not raw_path:
        return None
    manifest = _read_json(raw_path, label="pinned corpus manifest")
    if not _self_hash_valid(manifest, field="manifest_sha256"):
        raise ValueError("pinned corpus manifest self-hash mismatch")
    if manifest.get("manifest_sha256") != stage.exports.get(
        "CORPUS_MANIFEST_SHA256"
    ):
        raise ValueError("pinned corpus manifest identity mismatch")
    source_rows = int(manifest.get("source_row_count", -1))
    if source_rows <= 0:
        raise ValueError("pinned corpus manifest source row count is invalid")
    return source_rows


def _expected_frozen_rows(stage: QwenClstrEvalStage) -> int:
    benchmark = stage.key.split("/", 1)[1]
    full = _pinned_corpus_source_rows(stage)
    if full is None:
        full = FULL_FROZEN_DENOMINATORS[benchmark]
    raw_limit = str(stage.exports.get("MAX_EVAL_ROWS") or "").strip()
    if raw_limit and raw_limit != "ALL":
        return min(full, max(0, int(raw_limit)))
    return full


def validate_frozen_route_stage(stage: QwenClstrEvalStage) -> dict[str, Any]:
    if not stage.key.startswith("frozen/"):
        raise ValueError("frozen route validation requires a frozen stage")
    output_dir = Path(str(stage.exports.get("OUTPUT_DIR") or ""))
    report_path = output_dir / "frozen_route_eval_report.json"
    blockers: list[str] = []
    try:
        report = _read_json(report_path, label="frozen route report")
    except ValueError:
        return {
            "status": "action_required",
            "blockers": ["missing_or_invalid_report"],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
            "prediction_rows": 0,
        }

    if not _self_hash_valid(report, field="report_sha256"):
        _add(blockers, "report_self_hash_mismatch")
    if report.get("status") != "ok" or report.get("blockers"):
        _add(blockers, "stage_report_not_ok")
    if report.get("result_label") != "frozen" or report.get("metric_scope") != "next_tool_action_routing":
        _add(blockers, "routing_scope_mismatch")
    if report.get("task_success") is not False:
        _add(blockers, "routing_mislabeled_as_task_success")
    if report.get("checkpoint_chain_digest") != stage.exports.get("CHECKPOINT_CHAIN_DIGEST"):
        _add(blockers, "checkpoint_chain_digest_mismatch")
    if report.get("benchmark_manifest_sha256") != stage.exports.get("CORPUS_MANIFEST_SHA256"):
        _add(blockers, "corpus_manifest_sha256_mismatch")
    if int(report.get("memory_active_rows") or 0) != 0:
        _add(blockers, "frozen_route_memory_must_be_inactive")
    if report.get("zero_history_fallback") != "exact_static":
        _add(blockers, "frozen_route_fallback_mismatch")
    raw_skill_merge = report.get("skill_merge")
    skill_merge = dict(raw_skill_merge) if isinstance(raw_skill_merge, Mapping) else {}
    try:
        base_skill_count = int(skill_merge.get("base_skill_count"))
        benchmark_skill_count = int(skill_merge.get("benchmark_skill_count"))
        known_skill_count = int(skill_merge.get("known_benchmark_skill_count"))
        appended_skill_count = int(skill_merge.get("appended_benchmark_skill_count"))
        merged_skill_count = int(skill_merge.get("merged_skill_count"))
    except (TypeError, ValueError):
        base_skill_count = benchmark_skill_count = known_skill_count = -1
        appended_skill_count = merged_skill_count = -1
    if (
        min(
            base_skill_count,
            benchmark_skill_count,
            known_skill_count,
            appended_skill_count,
            merged_skill_count,
        )
        < 0
        or benchmark_skill_count != known_skill_count + appended_skill_count
        or merged_skill_count != base_skill_count + appended_skill_count
    ):
        _add(blockers, "invalid_skill_merge_counts")
    if not _all_finite(report.get("routing_metrics") or {}):
        _add(blockers, "nonfinite_routing_metrics")

    expected_rows = _expected_frozen_rows(stage)
    if int(report.get("evaluated_rows") or -1) != expected_rows:
        _add(blockers, "evaluated_row_denominator_mismatch")
    if int(report.get("prediction_rows") or -1) != expected_rows:
        _add(blockers, "prediction_row_count_mismatch")

    identity_path = Path(str(report.get("identity_path") or output_dir / "evaluation_identity.json"))
    try:
        identity = _read_json(identity_path, label="frozen route identity")
    except ValueError:
        identity = {}
        _add(blockers, "missing_or_invalid_identity")
    if identity:
        if not _self_hash_valid(identity, field="identity_sha256"):
            _add(blockers, "identity_self_hash_mismatch")
        if identity.get("checkpoint_chain_digest") != stage.exports.get("CHECKPOINT_CHAIN_DIGEST"):
            _add(blockers, "identity_checkpoint_chain_digest_mismatch")
        if identity.get("final_chain_manifest_sha256") != stage.exports.get("FINAL_CHAIN_MANIFEST_SHA256"):
            _add(blockers, "identity_final_chain_manifest_mismatch")
        if identity.get("benchmark_manifest_sha256") != stage.exports.get("CORPUS_MANIFEST_SHA256"):
            _add(blockers, "identity_corpus_manifest_mismatch")
        if identity.get("identity_sha256") != report.get("evaluation_identity_sha256"):
            _add(blockers, "report_identity_sha256_mismatch")

    predictions_path = Path(
        str(report.get("predictions_path") or output_dir / "frozen_route_predictions.jsonl")
    )
    prediction_rows = 0
    row_ids: set[str] = set()
    try:
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prediction_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    _add(blockers, "invalid_prediction_json")
                    continue
                if not isinstance(row, dict):
                    _add(blockers, "invalid_prediction_row")
                    continue
                if not _self_hash_valid(row, field="prediction_sha256"):
                    _add(blockers, "prediction_self_hash_mismatch")
                row_id = str(row.get("row_id") or "")
                if not row_id or row_id in row_ids:
                    _add(blockers, "duplicate_or_empty_prediction_row_id")
                row_ids.add(row_id)
                if row.get("evaluation_identity_sha256") != report.get("evaluation_identity_sha256"):
                    _add(blockers, "prediction_identity_mismatch")
                if row.get("benchmark_manifest_sha256") != stage.exports.get("CORPUS_MANIFEST_SHA256"):
                    _add(blockers, "prediction_corpus_manifest_mismatch")
                scores = row.get("ranked_scores")
                if not isinstance(scores, list) or not _all_finite(scores):
                    _add(blockers, "nonfinite_prediction_scores")
                candidate_count = int(row.get("declared_candidate_count") or 0)
                positive_rank = int(row.get("positive_rank") or 0)
                if candidate_count <= 0 or positive_rank <= 0 or positive_rank > candidate_count:
                    _add(blockers, "invalid_positive_rank")
    except OSError:
        _add(blockers, "missing_predictions")

    if prediction_rows != expected_rows or prediction_rows != int(report.get("prediction_rows") or -1):
        _add(blockers, "prediction_row_count_mismatch")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "stage": stage.key,
        "metric_scope": "routing",
        "task_success": False,
        "source_rows": int(report.get("source_rows") or 0),
        "evaluated_rows": int(report.get("evaluated_rows") or 0),
        "prediction_rows": prediction_rows,
        "memory_active_rows": int(report.get("memory_active_rows") or 0),
        "zero_history_fallback": report.get("zero_history_fallback"),
        "skill_merge": skill_merge,
        "routing_metrics": report.get("routing_metrics") or {},
        "report_path": str(report_path),
    }


def _native_report_path(stage: QwenClstrEvalStage) -> Path:
    output_dir = Path(str(stage.exports.get("OUTPUT_DIR") or ""))
    benchmark = stage.key.split("/", 1)[1]
    filename = {
        "toolbench_g3": "full_clstr_route_eval_report.json",
        "toolsandbox": "toolsandbox_full_clstr_route_eval_report.json",
        "tau2": "tau2_full_clstr_route_eval_report.json",
    }[benchmark]
    return output_dir / filename


def _expected_native_rows(stage: QwenClstrEvalStage) -> int | None:
    explicit = str(stage.exports.get("EXPECTED_SOURCE_ROWS") or "").strip()
    if explicit:
        return max(0, int(explicit))
    if stage.exports.get("EVAL_SCOPE") == "full":
        pinned = _pinned_corpus_source_rows(stage)
        if pinned is not None:
            return pinned
        return FULL_NATIVE_DENOMINATORS[stage.key.split("/", 1)[1]]
    return None


def _same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def _replay_evidence(value: Any) -> float:
    total = 0.0
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and "replay_prefix" in name
                and ("used_count" in name or "rows_with_replay_prefix" in name)
                and math.isfinite(float(item))
            ):
                total += max(0.0, float(item))
            else:
                total += _replay_evidence(item)
    elif isinstance(value, (list, tuple)):
        total += sum(_replay_evidence(item) for item in value)
    return total


def validate_native_route_stage(stage: QwenClstrEvalStage) -> dict[str, Any]:
    if not stage.key.startswith("native/"):
        raise ValueError("native route validation requires a native stage")
    report_path = _native_report_path(stage)
    blockers: list[str] = []
    try:
        report = _read_json(report_path, label="native route report")
    except ValueError:
        return {
            "status": "action_required",
            "blockers": ["missing_or_invalid_report"],
            "stage": stage.key,
            "metric_scope": "routing",
            "task_success": False,
            "memory_active_rows": 0.0,
        }

    if report.get("status") != "ok" or report.get("blockers"):
        _add(blockers, "stage_report_not_ok")
    if report.get("route_scorer") != "unified_memory":
        _add(blockers, "route_scorer_mismatch")
    if report.get("task_success") is True:
        _add(blockers, "routing_mislabeled_as_task_success")
    expected_reliability = str(stage.exports.get("RELIABILITY_MODE") or "")
    if expected_reliability:
        config = report.get("config") or {}
        if config.get("reliability_mode") != expected_reliability:
            _add(blockers, "reliability_mode_mismatch")
        if expected_reliability == "learned":
            gate_report = config.get("reliability_gate") or {}
            if gate_report.get("checkpoint_sha256") != stage.exports.get(
                "MEMORY_UTILITY_GATE_CHECKPOINT_SHA256"
            ):
                _add(blockers, "reliability_gate_identity_mismatch")
    for role in ("stage0", "stage2", "stage4"):
        if not _same_path(
            report.get(f"{role}_checkpoint_path"),
            stage.exports.get(f"{role.upper()}_CHECKPOINT_PATH"),
        ):
            _add(blockers, f"{role}_checkpoint_path_mismatch")

    source_rows = int(report.get("source_eval_rows") or 0)
    retained_rows = int(report.get("retained_eval_rows") or 0)
    expected_rows = _expected_native_rows(stage)
    if source_rows <= 0:
        _add(blockers, "no_source_eval_rows")
    if retained_rows <= 0 or retained_rows > source_rows:
        _add(blockers, "invalid_retained_eval_rows")
    if expected_rows is not None and source_rows != expected_rows:
        _add(blockers, "source_row_denominator_mismatch")
    raw_max_rows = str(stage.exports.get("MAX_EVAL_ROWS") or "").strip()
    if expected_rows is None and raw_max_rows and raw_max_rows != "ALL":
        if source_rows > int(raw_max_rows):
            _add(blockers, "source_rows_exceed_smoke_cap")
    if int((report.get("route_data_report") or {}).get("positive_injected_rows") or 0) > 0:
        _add(blockers, "gold_positive_injected")

    metric_payload = {
        "stage0_prior_eval": report.get("stage0_prior_eval") or report.get("prior_eval") or {},
        "stage4_eval": report.get("stage4_eval") or {},
        "strict": report.get("strict") or {},
    }
    if not _all_finite(metric_payload):
        _add(blockers, "nonfinite_native_metrics")
    memory_active_rows = _replay_evidence(report.get("stage4_eval") or {})
    if memory_active_rows <= 0:
        _add(blockers, "no_causal_replay_evidence")

    benchmark = stage.key.split("/", 1)[1]
    if benchmark == "toolbench_g3":
        stage4_eval = report.get("stage4_eval") or {}
        try:
            expected_candidate_count = int(stage.exports.get("CANDIDATE_COUNT"))
            reported_candidate_count = int(
                float(stage4_eval.get("stage4_candidate_count"))
            )
            reported_final_k = int(stage4_eval.get("candidate_recall_final_k"))
        except (TypeError, ValueError):
            expected_candidate_count = reported_candidate_count = reported_final_k = -1
        if (
            expected_candidate_count <= 0
            or reported_candidate_count != expected_candidate_count
            or reported_final_k != expected_candidate_count
        ):
            _add(blockers, "candidate_budget_mismatch")
    adapter = ((report.get("model_load") or {}).get("skill_pool_adapter") or {})
    if benchmark in {"tau2", "toolsandbox"}:
        if adapter.get("checkpoint_load_order") != [
            "stage0",
            "stage2",
            "stage4",
            "append_benchmark_skills",
        ]:
            _add(blockers, "native_skill_pool_adapter_order_mismatch")
        if int(adapter.get("trainable_backbone_parameters") or 0) != 0:
            _add(blockers, "trainable_backbone_parameters")
        if not _same_path(
            adapter.get("training_skills_path"),
            stage.exports.get("TRAINING_SKILLS_PATH"),
        ):
            _add(blockers, "training_skill_pool_path_mismatch")

    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "stage": stage.key,
        "metric_scope": "routing",
        "task_success": False,
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "memory_active_rows": memory_active_rows,
        "stage4_eval": report.get("stage4_eval") or {},
        "strict": report.get("strict") or {},
        "report_path": str(report_path),
    }


def _expected_alfworld_episodes(stage: QwenClstrEvalStage) -> int | None:
    explicit = str(stage.exports.get("EXPECTED_EPISODES") or "").strip()
    if explicit:
        return max(0, int(explicit))
    max_episodes = str(stage.exports.get("MAX_EPISODES") or "").strip()
    if max_episodes and max_episodes != "ALL":
        return max(0, int(max_episodes))
    data_dir = Path(str(stage.exports.get("DATA_DIR") or ""))
    split = str(stage.exports.get("SPLIT") or "")
    split_root = data_dir / "json_2.1.1" / split
    if split_root.is_dir():
        return sum(1 for _path in split_root.rglob("game.tw-pddl"))
    return None


def _alfworld_stage_split(stage: QwenClstrEvalStage) -> str:
    raw = str(
        stage.exports.get("SPLIT") or stage.exports.get("SPLITS") or ""
    ).strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts[0] if len(parts) == 1 else ""


def _read_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return rows, False
                if not isinstance(row, dict):
                    return rows, False
                rows.append(row)
    except OSError:
        return rows, False
    return rows, True


def _executor_summary_matches(
    summary: Any,
    metrics: Mapping[str, Any],
) -> bool:
    if not isinstance(summary, Mapping):
        return False
    for key in ("status", "method", "split", "episode_count"):
        if summary.get(key) != metrics.get(key):
            return False
    try:
        return float(summary.get("success_rate")) == float(metrics.get("success_rate"))
    except (TypeError, ValueError):
        return False


def validate_alfworld_closed_loop_stage(stage: QwenClstrEvalStage) -> dict[str, Any]:
    if not stage.key.startswith("closed_loop/alfworld_"):
        raise ValueError("ALFWorld validation requires a closed-loop ALFWorld stage")
    output_dir = Path(str(stage.exports.get("OUTPUT_DIR") or ""))
    executor_layout = (output_dir / "qwen_plus_clstr" / "metrics.json").exists() or (
        output_dir / "gate_summary.json"
    ).exists()
    artifact_layout = "executor_gate" if executor_layout else "legacy"
    metrics_path = output_dir / (
        "qwen_plus_clstr/metrics.json" if executor_layout else "metrics.json"
    )
    run_path = output_dir / (
        "qwen_plus_clstr/run.jsonl" if executor_layout else "run.jsonl"
    )
    blockers: list[str] = []
    try:
        metrics = _read_json(metrics_path, label="ALFWorld metrics")
    except ValueError:
        return {
            "status": "action_required",
            "blockers": ["missing_or_invalid_metrics"],
            "stage": stage.key,
            "metric_scope": "task_success",
            "task_success": True,
            "artifact_layout": artifact_layout,
            "episode_rows": 0,
            "memory_active_steps": 0,
            "unmapped_candidate_count": 0,
        }

    expected_split = _alfworld_stage_split(stage)
    if metrics.get("status") != "ok":
        _add(blockers, "metrics_not_ok")
    if not expected_split or metrics.get("split") != expected_split:
        _add(blockers, "split_mismatch")
    if not _all_finite(metrics):
        _add(blockers, "nonfinite_task_success_metrics")
    success_rate = float(metrics.get("success_rate") or 0.0)
    if success_rate < 0.0 or success_rate > 1.0:
        _add(blockers, "invalid_success_rate")
    routing_init_report = metrics.get("routing_init_report") or {}
    final_chain_adapter = routing_init_report.get("final_chain_adapter") or {}
    reported_checkpoints = {
        "stage0": metrics.get("stage0_checkpoint_path")
        or (final_chain_adapter.get("stage0") or {}).get("stage0_checkpoint_path"),
        "stage2": metrics.get("checkpoint_path"),
        "stage4": metrics.get("stage4_checkpoint_path"),
    }
    for role, reported_path in reported_checkpoints.items():
        if not _same_path(
            reported_path,
            stage.exports.get(f"{role.upper()}_CHECKPOINT_PATH"),
        ):
            _add(blockers, f"{role}_checkpoint_path_mismatch")
    if not _same_path(
        metrics.get("skill_rows_path_override"),
        stage.exports.get("TRAINING_SKILLS_PATH"),
    ):
        _add(blockers, "training_skill_pool_path_mismatch")
    if not _same_path(
        metrics.get("benchmark_skill_rows_path")
        or routing_init_report.get("benchmark_skill_source_path"),
        stage.exports.get("ALFWORLD_SKILLS_PATH"),
    ):
        _add(blockers, "alfworld_skill_pool_path_mismatch")
    expected_scorer = str(
        stage.exports.get("CLSTR_PRIOR_MODE")
        or stage.exports.get("SCORER_MODE")
        or "unified_memory_admissible_action"
    )
    reported_scorer = metrics.get("clstr_prior_mode") or metrics.get(
        "candidate_scorer_mode"
    )
    if reported_scorer != expected_scorer:
        _add(blockers, "candidate_scorer_mode_mismatch")
    expected_reliability = str(stage.exports.get("RELIABILITY_MODE") or "")
    if expected_reliability and metrics.get(
        "memory_utility_reliability_mode"
    ) != expected_reliability:
        _add(blockers, "reliability_mode_mismatch")
    if expected_reliability == "learned" and metrics.get(
        "memory_utility_gate_checkpoint_sha256"
    ) != stage.exports.get("MEMORY_UTILITY_GATE_CHECKPOINT_SHA256"):
        _add(blockers, "reliability_gate_identity_mismatch")
    expected_bound_text = str(
        stage.exports.get("SAFE_MEMORY_RESIDUAL_BOUND") or ""
    ).strip()
    if expected_bound_text and (
        not executor_layout or "safe_memory_residual_bound" in metrics
    ):
        try:
            expected_bound = float(expected_bound_text)
            reported_bound = float(metrics.get("safe_memory_residual_bound"))
        except (TypeError, ValueError):
            expected_bound = reported_bound = float("nan")
        if (
            not math.isfinite(expected_bound)
            or expected_bound <= 0.0
            or not math.isfinite(reported_bound)
            or reported_bound != expected_bound
        ):
            _add(blockers, "safe_memory_residual_bound_mismatch")
    expected_memory_protocol = str(
        stage.exports.get("MEMORY_PROTOCOL") or "stateful_post_action_v1"
    )
    if not executor_layout:
        if not bool(metrics.get("memory_active_protocol")):
            _add(blockers, "memory_active_protocol_disabled")
        if metrics.get("memory_protocol") != expected_memory_protocol:
            _add(blockers, "memory_protocol_mismatch")
    if metrics.get("stage4_overlay_loaded") is not True:
        _add(blockers, "stage4_overlay_not_loaded")
    if executor_layout:
        if metrics.get("executor_gate") is not True:
            _add(blockers, "executor_gate_disabled")
        if metrics.get("uses_clstr") is not True or metrics.get("is_clstr_result") is not True:
            _add(blockers, "executor_clstr_identity_mismatch")
        if metrics.get("qwen_direct_baseline") is not False:
            _add(blockers, "executor_clstr_mislabeled_as_qwen_only")
        if metrics.get("scoring_method") != str(
            stage.exports.get("SCORING_METHOD") or "generate"
        ):
            _add(blockers, "executor_scoring_method_mismatch")
        for export_name, metric_name, default in (
            ("QWEN_WEIGHT", "qwen_weight", 1.0),
            ("CLSTR_WEIGHT", "clstr_weight", 0.25),
        ):
            try:
                expected_weight = float(stage.exports.get(export_name) or default)
                reported_weight = float(metrics.get(metric_name))
            except (TypeError, ValueError):
                expected_weight = reported_weight = float("nan")
            if not math.isfinite(reported_weight) or reported_weight != expected_weight:
                _add(blockers, f"executor_{metric_name}_mismatch")
        if metrics.get("qwen_score_mode") != "proposal_bonus":
            _add(blockers, "executor_qwen_score_mode_mismatch")
        if metrics.get("score_normalization") != "qwen:proposal_bonus;clstr:row_zscore":
            _add(blockers, "executor_score_normalization_mismatch")
        expected_loop_guard = str(stage.exports.get("LOOP_GUARD") or "0") == "1"
        if bool(metrics.get("loop_guard")) != expected_loop_guard:
            _add(blockers, "executor_loop_guard_mismatch")
        if final_chain_adapter.get("checkpoint_load_order") != [
            "stage0",
            "stage2",
            "stage4",
            "append_benchmark_skills",
        ]:
            _add(blockers, "executor_checkpoint_load_order_mismatch")
        if int(final_chain_adapter.get("trainable_backbone_parameters") or 0) != 0:
            _add(blockers, "trainable_backbone_parameters")

    rows, rows_valid = _read_jsonl_objects(run_path)
    if not rows_valid:
        _add(blockers, "missing_episode_records")
    episode_rows = len(rows)
    episode_ids: set[int] = set()
    episode_gamefiles: set[str] = set()
    memory_active_steps = 0
    replay_prefix_steps = 0
    causal_update_total = 0
    candidate_mapping_count = 0
    unmapped_candidate_count = 0
    for row in rows:
        try:
            episode_index = int(row.get("episode_index"))
        except (TypeError, ValueError):
            episode_index = -1
        if episode_index < 0 or episode_index in episode_ids:
            _add(blockers, "duplicate_or_invalid_episode_index")
        episode_ids.add(episode_index)
        episode_gamefiles.add(str(row.get("gamefile") or ""))
        if row.get("split") != expected_split:
            _add(blockers, "episode_split_mismatch")
        if not _all_finite(
            {
                "points": row.get("points"),
                "goal_condition_points": row.get("goal_condition_points"),
                "steps": row.get("steps"),
            }
        ):
            _add(blockers, "nonfinite_episode_metrics")
        metadata_trace = row.get("policy_metadata_trace")
        if not isinstance(metadata_trace, list) or not metadata_trace:
            _add(blockers, "missing_policy_metadata_trace")
            continue
        if executor_layout:
            candidate_trace = row.get("candidate_trace")
            chosen_indices = row.get("chosen_indices")
            chosen_actions = row.get("chosen_action_trace")
            if not all(
                isinstance(value, list)
                for value in (candidate_trace, chosen_indices, chosen_actions)
            ) or not (
                len(candidate_trace)
                == len(chosen_indices)
                == len(chosen_actions)
                == len(metadata_trace)
            ):
                _add(blockers, "executor_trace_length_mismatch")
                continue
            for candidates, chosen_index, chosen_action, metadata in zip(
                candidate_trace,
                chosen_indices,
                chosen_actions,
                metadata_trace,
            ):
                if not isinstance(metadata, Mapping):
                    _add(blockers, "invalid_policy_metadata")
                    continue
                if metadata.get("clstr_route_scorer") != expected_scorer:
                    _add(blockers, "policy_scorer_mode_mismatch")
                recurrent = bool(metadata.get("uses_recurrent_m_t"))
                replay_len = int(metadata.get("clstr_replay_prefix_len") or 0)
                replay_used = int(
                    metadata.get("clstr_replay_prefix_used_count") or 0
                )
                if recurrent:
                    memory_active_steps += 1
                    if (
                        replay_len <= 0
                        or replay_used <= 0
                        or metadata.get("clstr_memory_source") != "replay_prefix"
                    ):
                        _add(blockers, "invalid_replay_memory_evidence")
                    else:
                        replay_prefix_steps += 1
                        causal_update_total += replay_used
                if not isinstance(candidates, list) or not candidates:
                    _add(blockers, "missing_concrete_action_candidates")
                    continue
                candidate_mapping_count += len(candidates)
                try:
                    selected = int(chosen_index)
                except (TypeError, ValueError):
                    selected = -1
                if (
                    selected < 0
                    or selected >= len(candidates)
                    or candidates[selected] != chosen_action
                ):
                    unmapped_candidate_count += 1
        else:
            try:
                final_update_count = int(row.get("causal_update_count_final") or 0)
            except (TypeError, ValueError):
                final_update_count = -1
            if final_update_count < 0:
                _add(blockers, "invalid_final_causal_update_count")
            else:
                causal_update_total += final_update_count
            for metadata in metadata_trace:
                if not isinstance(metadata, Mapping):
                    _add(blockers, "invalid_policy_metadata")
                    continue
                try:
                    update_count = int(metadata.get("causal_update_count") or 0)
                except (TypeError, ValueError):
                    update_count = -1
                if update_count < 0:
                    _add(blockers, "invalid_causal_update_count")
                    continue
                if metadata.get("memory_protocol") != expected_memory_protocol:
                    _add(blockers, "policy_memory_protocol_mismatch")
                if bool(metadata.get("uses_recurrent_m_t")) and update_count > 0:
                    memory_active_steps += 1
                mappings = metadata.get("candidate_skill_mappings")
                if not isinstance(mappings, list) or not mappings:
                    _add(blockers, "missing_candidate_skill_mappings")
                    continue
                for mapping in mappings:
                    candidate_mapping_count += 1
                    if not isinstance(mapping, Mapping) or not str(
                        mapping.get("mapped_skill_id") or ""
                    ):
                        unmapped_candidate_count += 1

    control_episode_rows = 0
    control_success_rate: float | None = None
    control_metrics_path: Path | None = None
    control_run_path: Path | None = None
    gate_summary_path: Path | None = None
    if executor_layout:
        control_metrics_path = output_dir / "qwen_only" / "metrics.json"
        control_run_path = output_dir / "qwen_only" / "run.jsonl"
        gate_summary_path = output_dir / "gate_summary.json"
        try:
            control_metrics = _read_json(
                control_metrics_path,
                label="ALFWorld Qwen-only metrics",
            )
        except ValueError:
            control_metrics = {}
            _add(blockers, "missing_or_invalid_qwen_only_metrics")
        if control_metrics:
            if (
                control_metrics.get("status") != "ok"
                or control_metrics.get("split") != expected_split
                or control_metrics.get("qwen_direct_baseline") is not True
                or bool(control_metrics.get("uses_clstr"))
                or control_metrics.get("executor_gate") is not True
            ):
                _add(blockers, "qwen_only_control_identity_mismatch")
            if control_metrics.get("scoring_method") != str(
                stage.exports.get("SCORING_METHOD") or "generate"
            ):
                _add(blockers, "qwen_only_scoring_method_mismatch")
            expected_qwen_model = stage.exports.get("QWEN_MODEL_NAME_OR_PATH")
            if expected_qwen_model and not _same_path(
                control_metrics.get("model_name_or_path"), expected_qwen_model
            ):
                _add(blockers, "qwen_only_model_identity_mismatch")
            try:
                control_success_rate = float(control_metrics.get("success_rate"))
            except (TypeError, ValueError):
                control_success_rate = None
            if (
                control_success_rate is None
                or not math.isfinite(control_success_rate)
                or not 0.0 <= control_success_rate <= 1.0
            ):
                _add(blockers, "invalid_qwen_only_success_rate")
        control_rows, control_rows_valid = _read_jsonl_objects(control_run_path)
        control_episode_rows = len(control_rows)
        if not control_rows_valid:
            _add(blockers, "missing_qwen_only_episode_records")
        control_gamefiles = {str(row.get("gamefile") or "") for row in control_rows}
        if control_gamefiles != episode_gamefiles:
            _add(blockers, "executor_episode_set_mismatch")
        if control_metrics and control_episode_rows != int(
            control_metrics.get("episode_count") or control_metrics.get("episodes") or 0
        ):
            _add(blockers, "qwen_only_episode_row_count_mismatch")
        try:
            gate_summary = _read_json(
                gate_summary_path,
                label="ALFWorld executor gate summary",
            )
        except ValueError:
            gate_summary = {}
            _add(blockers, "missing_or_invalid_executor_gate_summary")
        if gate_summary:
            if (
                gate_summary.get("status") != "ok"
                or gate_summary.get("splits") != [expected_split]
                or not _executor_summary_matches(
                    gate_summary.get("qwen_plus_clstr"), metrics
                )
                or not _executor_summary_matches(
                    gate_summary.get("qwen_only"), control_metrics
                )
            ):
                _add(blockers, "executor_gate_summary_mismatch")
            expected_steps = int(stage.exports.get("MAX_STEPS") or 0)
            if expected_steps and int(gate_summary.get("max_steps") or 0) != expected_steps:
                _add(blockers, "executor_max_steps_mismatch")

    expected_episodes = _expected_alfworld_episodes(stage)
    metric_episodes = int(metrics.get("episode_count") or metrics.get("episodes") or 0)
    if episode_rows != metric_episodes:
        _add(blockers, "episode_row_count_mismatch")
    if expected_episodes is not None and episode_rows != expected_episodes:
        _add(blockers, "episode_row_count_mismatch")
    if candidate_mapping_count <= 0:
        _add(blockers, "no_candidate_mapping_evidence")
    if unmapped_candidate_count > 0:
        _add(blockers, "unmapped_admissible_actions")
    if memory_active_steps <= 0 or causal_update_total <= 0:
        _add(blockers, "no_causal_stateful_update_evidence")

    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "stage": stage.key,
        "metric_scope": "task_success",
        "task_success": True,
        "artifact_layout": artifact_layout,
        "episode_rows": episode_rows,
        "control_episode_rows": control_episode_rows,
        "expected_episodes": expected_episodes,
        "success_rate": success_rate,
        "control_success_rate": control_success_rate,
        "memory_active_steps": memory_active_steps,
        "replay_prefix_steps": replay_prefix_steps,
        "causal_update_total": causal_update_total,
        "candidate_mapping_count": candidate_mapping_count,
        "unmapped_candidate_count": unmapped_candidate_count,
        "metrics_path": str(metrics_path),
        "run_path": str(run_path),
        "control_metrics_path": str(control_metrics_path) if control_metrics_path else None,
        "control_run_path": str(control_run_path) if control_run_path else None,
        "gate_summary_path": str(gate_summary_path) if gate_summary_path else None,
    }


def _shared_stage_export(
    stages: list[QwenClstrEvalStage],
    *,
    export_name: str,
    label: str,
) -> str:
    values = {
        str(stage.exports.get(export_name) or "").strip()
        for stage in stages
    }
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"all multibench stages must use one {label}")
    return next(iter(values))


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _atomic_write_text(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _markdown_score(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return f"{number:.4f}"
    return "N/A"


def _markdown_count(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return str(int(number)) if number.is_integer() else f"{number:.4f}"
    return "N/A"


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "N/A").replace("|", "\\|").replace("\n", " ")


def render_qwen_clstr_multibench_markdown(report: Mapping[str, Any]) -> str:
    blockers = report.get("blockers")
    blocker_rows = blockers if isinstance(blockers, list) else ["invalid_report_blockers"]
    lines = [
        "# Qwen 0.6B CLSTR Multi-Benchmark Report",
        "",
        f"- Status: `{_markdown_cell(report.get('status'))}`",
        f"- Scope: `{_markdown_cell(report.get('scope'))}`",
        f"- Checkpoint chain: `{_markdown_cell(report.get('checkpoint_chain_digest'))}`",
    ]
    if report.get("accepted_smoke_gate_sha256"):
        lines.append(
            f"- Accepted smoke gate: `{_markdown_cell(report.get('accepted_smoke_gate_sha256'))}`"
        )
    lines.extend(
        [
            f"- Report SHA-256: `{_markdown_cell(report.get('report_sha256'))}`",
            "",
            "## Blockers",
            "",
        ]
    )
    if blocker_rows:
        lines.extend(f"- `{_markdown_cell(blocker)}`" for blocker in blocker_rows)
    else:
        lines.append("- None.")

    execution = report.get("execution") if isinstance(report.get("execution"), Mapping) else None
    if execution is not None:
        jobs = execution.get("jobs") if isinstance(execution.get("jobs"), Mapping) else {}
        lines.extend(
            [
                "",
                "## Slurm execution evidence",
                "",
                "| Stage | Job ID | State | Exit code |",
                "|---|---:|---|---|",
            ]
        )
        for stage, row in jobs.items():
            values = row if isinstance(row, Mapping) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(stage),
                        _markdown_cell(values.get("job_id")),
                        _markdown_cell(values.get("state")),
                        _markdown_cell(values.get("exit_code")),
                    ]
                )
                + " |"
            )

    routing = report.get("routing") if isinstance(report.get("routing"), Mapping) else {}
    frozen = routing.get("frozen") if isinstance(routing.get("frozen"), Mapping) else {}
    lines.extend(
        [
            "",
            "## Frozen comparable routing (not task success)",
            "",
            "| Benchmark | Evaluated rows | Recall@1 | Recall@5 | MRR | Known skills | Appended skills | Memory-active rows | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for benchmark, row in frozen.items():
        values = row if isinstance(row, Mapping) else {}
        metrics = values.get("routing_metrics") if isinstance(values.get("routing_metrics"), Mapping) else {}
        skill_merge = values.get("skill_merge") if isinstance(values.get("skill_merge"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(benchmark),
                    _markdown_count(values.get("evaluated_rows")),
                    _markdown_score(metrics.get("recall@1")),
                    _markdown_score(metrics.get("recall@5")),
                    _markdown_score(metrics.get("mrr")),
                    _markdown_count(skill_merge.get("known_benchmark_skill_count")),
                    _markdown_count(skill_merge.get("appended_benchmark_skill_count")),
                    _markdown_count(values.get("memory_active_rows")),
                    _markdown_cell(values.get("status")),
                ]
            )
            + " |"
        )

    native = routing.get("native") if isinstance(routing.get("native"), Mapping) else {}
    lines.extend(
        [
            "",
            "## Native causal routing (not task success)",
            "",
            "| Benchmark | Source rows | Retained rows | Recall@1 | Recall@5 | MRR | Memory-active rows | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for benchmark, row in native.items():
        values = row if isinstance(row, Mapping) else {}
        metrics = values.get("stage4_eval") if isinstance(values.get("stage4_eval"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(benchmark),
                    _markdown_count(values.get("source_rows")),
                    _markdown_count(values.get("retained_rows")),
                    _markdown_score(metrics.get("stage4_next_skill_recall@1")),
                    _markdown_score(metrics.get("stage4_next_skill_recall@5")),
                    _markdown_score(metrics.get("stage4_next_skill_mrr")),
                    _markdown_count(values.get("memory_active_rows")),
                    _markdown_cell(values.get("status")),
                ]
            )
            + " |"
        )

    task_success = report.get("task_success") if isinstance(report.get("task_success"), Mapping) else {}
    lines.extend(
        [
            "",
            "## ALFWorld closed-loop task success",
            "",
            "| Benchmark | Episodes | Success rate | Memory-active steps | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for benchmark, row in task_success.items():
        values = row if isinstance(row, Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(benchmark),
                    _markdown_count(values.get("episode_rows")),
                    _markdown_score(values.get("success_rate")),
                    _markdown_count(values.get("memory_active_steps")),
                    _markdown_cell(values.get("status")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Frozen and native routing scores measure next-skill/action ranking. Only the ALFWorld closed-loop section reports end-to-end task success.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_qwen_clstr_multibench_job_evidence(
    path: str | Path,
    *,
    stages: list[QwenClstrEvalStage],
) -> dict[str, Any]:
    evidence_path = Path(path).resolve()
    blockers: list[str] = []
    try:
        evidence = _read_json(evidence_path, label="multibench job evidence")
    except ValueError:
        return {
            "status": "action_required",
            "blockers": ["missing_or_invalid_job_evidence"],
            "evidence_path": str(evidence_path),
            "jobs": {},
        }
    if not _self_hash_valid(evidence, field="evidence_sha256"):
        _add(blockers, "job_evidence_self_hash_mismatch")
    raw_blockers = evidence.get("blockers")
    if evidence.get("status") != "ok" or raw_blockers:
        _add(blockers, "job_evidence_not_ok")
        if isinstance(raw_blockers, list):
            for blocker in raw_blockers:
                _add(blockers, str(blocker))

    expected_stage_order = [stage.key for stage in stages]
    if evidence.get("stage_order") != expected_stage_order:
        _add(blockers, "job_stage_order_mismatch")
    identity = evidence.get("submission_identity")
    if not isinstance(identity, Mapping):
        _add(blockers, "missing_submission_identity")
        identity = {}
    expected_chain = str(stages[0].exports.get("CHECKPOINT_CHAIN_DIGEST") or "")
    expected_manifest = str(stages[0].exports.get("FINAL_CHAIN_MANIFEST_SHA256") or "")
    expected_scope = str(stages[0].exports.get("EVAL_SCOPE") or "")
    expected_smoke_gate = str(
        stages[0].exports.get("ACCEPTED_SMOKE_GATE_SHA256") or ""
    )
    expected_corpora = {
        stage.key: str(stage.exports.get("CORPUS_MANIFEST_SHA256") or "")
        for stage in stages
    }
    if identity.get("stage_order") != expected_stage_order:
        _add(blockers, "submission_stage_order_mismatch")
    if identity.get("checkpoint_chain_digest") != expected_chain:
        _add(blockers, "checkpoint_chain_digest_mismatch")
    if identity.get("final_chain_manifest_sha256") != expected_manifest:
        _add(blockers, "final_chain_manifest_sha256_mismatch")
    if identity.get("scope") != expected_scope:
        _add(blockers, "evaluation_scope_mismatch")
    if expected_scope == "full" and identity.get(
        "accepted_smoke_gate_sha256"
    ) != expected_smoke_gate:
        _add(blockers, "accepted_smoke_gate_sha256_mismatch")
    if identity.get("corpus_manifest_sha256_by_stage") != expected_corpora:
        _add(blockers, "corpus_manifest_sha256_by_stage_mismatch")

    registry_path = Path(str(evidence.get("registry_path") or ""))
    try:
        registry_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    except OSError:
        _add(blockers, "missing_submission_registry")
        registry_digest = ""
    if registry_digest != str(evidence.get("registry_sha256") or ""):
        _add(blockers, "submission_registry_sha256_mismatch")
    try:
        registry = _read_json(registry_path, label="multibench submission registry")
    except ValueError:
        _add(blockers, "missing_or_invalid_submission_registry")
        registry = {}
    if registry.get("status") != "submitted":
        _add(blockers, "submission_registry_not_submitted")
    if registry.get("stage_order") != expected_stage_order:
        _add(blockers, "submission_registry_stage_order_mismatch")
    if registry.get("submission_identity") != identity:
        _add(blockers, "submission_registry_identity_mismatch")
    registry_stages = registry.get("stages")
    if not isinstance(registry_stages, Mapping) or set(registry_stages) != set(expected_stage_order):
        _add(blockers, "submission_registry_job_set_mismatch")
        registry_stages = registry_stages if isinstance(registry_stages, Mapping) else {}

    raw_jobs = evidence.get("jobs")
    jobs: dict[str, dict[str, str]] = {}
    if not isinstance(raw_jobs, Mapping) or set(raw_jobs) != set(expected_stage_order):
        _add(blockers, "job_set_mismatch")
        raw_jobs = raw_jobs if isinstance(raw_jobs, Mapping) else {}
    for stage in expected_stage_order:
        row = raw_jobs.get(stage)
        if not isinstance(row, Mapping):
            _add(blockers, f"{stage}:missing_job_state")
            continue
        job_id = str(row.get("job_id") or "")
        state = str(row.get("state") or "")
        exit_code = str(row.get("exit_code") or "")
        jobs[stage] = {
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
        }
        if not re.fullmatch(r"[0-9]+", job_id):
            _add(blockers, f"{stage}:invalid_job_id")
        registry_row = registry_stages.get(stage)
        if (
            not isinstance(registry_row, Mapping)
            or registry_row.get("status") != "submitted"
            or str(registry_row.get("job_id") or "") != job_id
        ):
            _add(blockers, f"{stage}:registry_job_id_mismatch")
        if state != "COMPLETED" or exit_code != "0:0":
            _add(blockers, f"{stage}:job_not_completed:{state}:{exit_code}")

    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence.get("evidence_sha256"),
        "registry_path": str(registry_path),
        "registry_sha256": evidence.get("registry_sha256"),
        "jobs": jobs,
    }


def build_qwen_clstr_multibench_report(
    *,
    stages: list[QwenClstrEvalStage],
    output_path: str | Path,
    markdown_path: str | Path | None = None,
    job_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    stage_order = tuple(stage.key for stage in stages)
    if stage_order != QWEN_CLSTR_MULTIBENCH_STAGE_ORDER:
        raise ValueError("multibench report requires the fixed nine-stage order")

    checkpoint_chain_digest = _shared_stage_export(
        stages,
        export_name="CHECKPOINT_CHAIN_DIGEST",
        label="checkpoint-chain digest",
    )
    final_chain_manifest_sha256 = _shared_stage_export(
        stages,
        export_name="FINAL_CHAIN_MANIFEST_SHA256",
        label="final-chain manifest",
    )
    scope = _shared_stage_export(
        stages,
        export_name="EVAL_SCOPE",
        label="evaluation scope",
    )
    if scope == "full":
        accepted_smoke_gate_sha256 = _shared_stage_export(
            stages,
            export_name="ACCEPTED_SMOKE_GATE_SHA256",
            label="accepted smoke gate",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", accepted_smoke_gate_sha256):
            raise ValueError("accepted smoke gate must be a SHA-256 digest")
    else:
        accepted_smoke_gate_sha256 = None
    corpus_manifest_sha256_by_stage = {
        stage.key: str(stage.exports.get("CORPUS_MANIFEST_SHA256") or "")
        for stage in stages
    }
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in corpus_manifest_sha256_by_stage.values()
    ):
        raise ValueError("every multibench stage must pin a corpus manifest SHA-256")

    frozen_routing: dict[str, dict[str, Any]] = {}
    native_routing: dict[str, dict[str, Any]] = {}
    task_success: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    execution = None
    if job_evidence_path is not None:
        execution = validate_qwen_clstr_multibench_job_evidence(
            job_evidence_path,
            stages=stages,
        )
        for blocker in execution.get("blockers") or []:
            _add(blockers, f"jobs:{blocker}")

    for stage in stages:
        family, benchmark = stage.key.split("/", 1)
        if family == "frozen":
            result = validate_frozen_route_stage(stage)
            frozen_routing[benchmark] = result
        elif family == "native":
            result = validate_native_route_stage(stage)
            native_routing[benchmark] = result
        else:
            result = validate_alfworld_closed_loop_stage(stage)
            task_success[benchmark] = result

        raw_stage_blockers = result.get("blockers")
        if isinstance(raw_stage_blockers, list):
            stage_blockers = list(raw_stage_blockers)
        else:
            stage_blockers = ["invalid_stage_blockers"]
        if result.get("stage") != stage.key:
            _add(stage_blockers, "stage_result_identity_mismatch")
        if family == "closed_loop":
            if result.get("metric_scope") != "task_success":
                _add(stage_blockers, "task_success_scope_mismatch")
            if result.get("task_success") is not True:
                _add(stage_blockers, "task_success_mislabeled_as_routing")
        else:
            if result.get("metric_scope") != "routing":
                _add(stage_blockers, "routing_scope_mismatch")
            if result.get("task_success") is not False:
                _add(stage_blockers, "routing_mislabeled_as_task_success")
        if result.get("status") != "ok" and not stage_blockers:
            stage_blockers = ["stage_status_not_ok"]
        for blocker in stage_blockers:
            _add(blockers, f"{stage.key}:{blocker}")

    report: dict[str, Any] = {
        "schema_version": "qwen06_clstr_multibench_report_v1",
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "scope": scope,
        "stage_order": list(stage_order),
        "checkpoint_chain_digest": checkpoint_chain_digest,
        "final_chain_manifest_sha256": final_chain_manifest_sha256,
        "accepted_smoke_gate_sha256": accepted_smoke_gate_sha256,
        "corpus_manifest_sha256_by_stage": corpus_manifest_sha256_by_stage,
        "routing": {
            "frozen": frozen_routing,
            "native": native_routing,
        },
        "task_success": task_success,
    }
    if execution is not None:
        report["execution"] = execution
    report["report_sha256"] = canonical_digest(report)
    _atomic_write_json(output_path, report)
    if markdown_path is not None:
        _atomic_write_text(markdown_path, render_qwen_clstr_multibench_markdown(report))
    return report
