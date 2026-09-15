#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolret_eval_audit import audit_toolret_eval_data
from clstr.traject_eval_audit import audit_traject_eval_data
from clstr.toolbench_g3_audit import audit_toolbench_g3_data


TRAIN_SPLIT_NAMES = {"train", "training", "train_or_released_g3", "released_train"}
UNSAFE_RETRIEVAL_SPLITS = {
    "dev",
    "eval",
    "evaluation",
    "public_train_or_eval_unlabeled",
    "test",
    "valid",
    "validation",
}
DEFAULT_STAGE0_OUTPUT_DIR = "outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE"
DEFAULT_STAGE0_CHECKPOINT = f"{DEFAULT_STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt"
DEFAULT_STAGE0_EVAL_METRICS_NAME = "full_retrieval_eval/metrics.json"
DEFAULT_STAGE0_EVAL_METRICS_PATH = f"{DEFAULT_STAGE0_OUTPUT_DIR}/{DEFAULT_STAGE0_EVAL_METRICS_NAME}"
DEFAULT_STAGE1_OUTPUT_DIR = "outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"
DEFAULT_STAGE1_CHECKPOINT = f"{DEFAULT_STAGE1_OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt"
DEFAULT_STAGE2_OUTPUT_DIR = "outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"
DEFAULT_STAGE2_CHECKPOINT = f"{DEFAULT_STAGE2_OUTPUT_DIR}/checkpoints/clstr_full_base-step10000.pt"
EXPECTED_CANDIDATE_RECALL_MODE = "static_plus_dynamic_extra"
EXPECTED_CANDIDATE_UNION_VERSION = "memory_union_v1"
EXPECTED_CANDIDATE_SELECTION_VERSION = "stable_declared_pool_v1"
EXPECTED_CANDIDATE_TIE_BREAK_POLICY = "declared_pool_index_ascending"
EXPECTED_EQUAL_BUDGET_COMPARATOR_MODE = "same_final_scorer_static_top_m_plus_d"
EXPECTED_RELIABILITY_FEATURE_SCHEMA = "memory_utility_features_v1"
EXPECTED_ZERO_HISTORY_FALLBACK = "exact_static"
MEMORY_UTILITY_RELIABILITY_MODES = {
    "static",
    "dynamic",
    "fixed_alpha",
    "heuristic",
    "learned",
}


def audit_memory_utility_reliability_readiness(
    *,
    mode: str = "dynamic",
    zero_history_fallback: str = EXPECTED_ZERO_HISTORY_FALLBACK,
    reliability_feature_schema: str = EXPECTED_RELIABILITY_FEATURE_SCHEMA,
    reliability_changes_memory_state: bool = False,
    audit_report_path: str | Path | None = None,
    gate_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_mode = str(mode or "").strip()
    blockers: list[str] = []
    if normalized_mode not in MEMORY_UTILITY_RELIABILITY_MODES:
        blockers.append("memory_utility_reliability_mode_unsupported")
    if str(zero_history_fallback) != EXPECTED_ZERO_HISTORY_FALLBACK:
        blockers.append("memory_utility_zero_history_fallback_mismatch")
    if str(reliability_feature_schema) != EXPECTED_RELIABILITY_FEATURE_SCHEMA:
        blockers.append("memory_utility_feature_schema_mismatch")
    if bool(reliability_changes_memory_state):
        blockers.append("memory_utility_reliability_must_not_change_memory_state")

    audit_path = Path(audit_report_path) if audit_report_path is not None else None
    gate_path = Path(gate_checkpoint_path) if gate_checkpoint_path is not None else None
    audit_report: dict[str, Any] = {}
    audit_digest = None
    checkpoint_report: dict[str, Any] = {}
    if normalized_mode == "learned":
        if audit_path is None or not audit_path.is_file():
            blockers.append("memory_utility_audit_report_missing")
        else:
            audit_report = _read_json(audit_path)
            audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            if audit_report.get("status") != "ok":
                blockers.append("memory_utility_audit_status_not_ok")
            if not bool(audit_report.get("learned_gate_recommended")) or bool(
                audit_report.get("recommendation_blockers")
            ):
                blockers.append("memory_utility_audit_did_not_recommend_learned_gate")
            if audit_report.get("feature_schema") != EXPECTED_RELIABILITY_FEATURE_SCHEMA:
                blockers.append("memory_utility_audit_feature_schema_mismatch")
            manifest_identity = (
                audit_report.get("route_manifest_identity")
                if isinstance(audit_report.get("route_manifest_identity"), dict)
                else {}
            )
            if manifest_identity.get("candidate_union_version") != EXPECTED_CANDIDATE_UNION_VERSION:
                blockers.append("memory_utility_audit_candidate_union_version_mismatch")
            if (
                manifest_identity.get("candidate_selection_version")
                != EXPECTED_CANDIDATE_SELECTION_VERSION
            ):
                blockers.append("memory_utility_audit_candidate_selection_version_mismatch")
            if manifest_identity.get("pool_protocol") != "known_global":
                blockers.append("memory_utility_audit_pool_protocol_not_known_global")

        if gate_path is None or not gate_path.is_file():
            blockers.append("memory_utility_gate_checkpoint_missing")
        else:
            try:
                import torch

                loaded = torch.load(gate_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                loaded = {}
                checkpoint_report["load_error"] = f"{type(exc).__name__}: {exc}"
                blockers.append("memory_utility_gate_checkpoint_unreadable")
            if isinstance(loaded, dict):
                checkpoint_report = {
                    **checkpoint_report,
                    "stage": loaded.get("stage"),
                    "schema_version": loaded.get("schema_version"),
                    "audit_report_sha256": loaded.get("audit_report_sha256"),
                    "feature_schema": loaded.get("feature_schema"),
                }
                if loaded.get("stage") != "clstr_memory_utility_gate":
                    blockers.append("memory_utility_gate_checkpoint_stage_mismatch")
                if loaded.get("schema_version") != "memory_utility_gate_checkpoint_v1":
                    blockers.append("memory_utility_gate_checkpoint_schema_mismatch")
                if audit_digest is None or loaded.get("audit_report_sha256") != audit_digest:
                    blockers.append("memory_utility_gate_audit_hash_mismatch")
                if loaded.get("feature_schema") != EXPECTED_RELIABILITY_FEATURE_SCHEMA:
                    blockers.append("memory_utility_gate_feature_schema_mismatch")
                if loaded.get("zero_history_fallback") != EXPECTED_ZERO_HISTORY_FALLBACK:
                    blockers.append("memory_utility_gate_zero_history_fallback_mismatch")
                if bool(loaded.get("reliability_changes_memory_state")):
                    blockers.append("memory_utility_gate_changes_memory_state")
                state_dict = loaded.get("memory_utility_gate_state_dict")
                if not isinstance(state_dict, dict) or not state_dict:
                    blockers.append("memory_utility_gate_state_dict_missing")
            else:
                blockers.append("memory_utility_gate_checkpoint_payload_invalid")

    blockers = list(dict.fromkeys(blockers))
    learned_enabled = normalized_mode == "learned" and not blockers
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "memory_utility_gate_mode": normalized_mode,
        "zero_history_fallback": str(zero_history_fallback),
        "reliability_feature_schema": str(reliability_feature_schema),
        "reliability_changes_memory_state": bool(reliability_changes_memory_state),
        "learned_gate_enabled": learned_enabled,
        "audit_report_path": None if audit_path is None else str(audit_path),
        "audit_report_sha256": audit_digest,
        "audit_report": audit_report,
        "gate_checkpoint_path": None if gate_path is None else str(gate_path),
        "gate_checkpoint": checkpoint_report,
    }


def audit_memory_candidate_recall_readiness(
    *,
    candidate_recall_mode: str = EXPECTED_CANDIDATE_RECALL_MODE,
    route_scorer: str = "unified_memory",
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    final_k: int = 64,
    denominator_scope: str = "all_eligible_source_strict",
    gold_positive_injected: bool = False,
    legal_pool_source: str = "declared_global_skill_pool",
    equal_budget_comparator_mode: str = EXPECTED_EQUAL_BUDGET_COMPARATOR_MODE,
) -> dict[str, Any]:
    static_budget = int(static_k)
    dynamic_budget = int(dynamic_extra_k)
    final_budget = int(final_k)
    blockers: list[str] = []
    if str(candidate_recall_mode) != EXPECTED_CANDIDATE_RECALL_MODE:
        blockers.append("candidate_recall_mode_mismatch")
    if str(route_scorer) != "unified_memory":
        blockers.append("candidate_recall_route_scorer_mismatch")
    if static_budget <= 0 or dynamic_budget < 0 or final_budget <= 0:
        blockers.append("candidate_recall_invalid_budget")
    if final_budget > static_budget:
        blockers.append("candidate_recall_final_k_exceeds_static_k")
    if str(denominator_scope) != "all_eligible_source_strict":
        blockers.append("candidate_recall_denominator_not_all_source_strict")
    if bool(gold_positive_injected):
        blockers.append("candidate_recall_gold_positive_injected")
    normalized_pool_source = str(legal_pool_source or "").strip().lower()
    if any(token in normalized_pool_source for token in ("target", "gold", "namespace")):
        blockers.append("candidate_recall_target_derived_legal_pool")
    elif normalized_pool_source not in {
        "declared_global_skill_pool",
        "declared_legal_full_skill_pool",
    }:
        blockers.append("candidate_recall_legal_pool_source_unsupported")
    if str(equal_budget_comparator_mode) != EXPECTED_EQUAL_BUDGET_COMPARATOR_MODE:
        blockers.append("candidate_recall_equal_budget_control_missing")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "candidate_recall_mode": str(candidate_recall_mode),
        "candidate_union_version": EXPECTED_CANDIDATE_UNION_VERSION,
        "candidate_selection_version": EXPECTED_CANDIDATE_SELECTION_VERSION,
        "tie_break_policy": EXPECTED_CANDIDATE_TIE_BREAK_POLICY,
        "route_scorer": str(route_scorer),
        "requested_static_m": static_budget,
        "requested_dynamic_extra_d": dynamic_budget,
        "final_k": final_budget,
        "denominator_scope": str(denominator_scope),
        "gold_positive_injected": bool(gold_positive_injected),
        "legal_pool_source": str(legal_pool_source),
        "equal_budget_comparator_mode": str(equal_budget_comparator_mode),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_stream(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt JSONL at {path} line {line_no}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_allowed_benchmarks(value: str | set[str] | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    parsed = {str(item).strip() for item in items if str(item).strip()}
    return parsed or None


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _split_name(row: dict[str, Any]) -> str:
    provenance = _provenance(row)
    return str(row.get("split") or provenance.get("split") or "").strip().lower()


def _is_train_row(row: dict[str, Any]) -> bool:
    return _split_name(row) in TRAIN_SPLIT_NAMES


def _is_safe_retrieval_row(row: dict[str, Any]) -> bool:
    return _split_name(row) not in UNSAFE_RETRIEVAL_SPLITS


def _count_jsonl(path: Path) -> int:
    return sum(1 for _row in _read_jsonl_stream(path))


def _file_report(data_root: Path) -> dict[str, Any]:
    names = [
        "manifest.json",
        "leakage_audit.json",
        "source_inventory.jsonl",
        "skill_pool.jsonl",
        "retrieval.jsonl",
        "trajectories.jsonl",
    ]
    return {
        name: {
            "exists": (data_root / name).exists(),
            "bytes": (data_root / name).stat().st_size if (data_root / name).exists() else 0,
        }
        for name in names
    }


def _source_inventory_report(data_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = list(_read_jsonl_stream(data_root / "source_inventory.jsonl"))
    by_id = {str(row.get("source_id") or ""): row for row in rows if row.get("source_id")}
    missing = list((manifest.get("source_inventory") or {}).get("missing_required_target_sources") or [])
    if not missing:
        missing = sorted(
            source_id
            for source_id in ("traject_bench", "toolbench_g3", "toolret_training")
            if by_id.get(source_id) and not bool(by_id[source_id].get("available"))
        )
    return {
        "source_count": len(rows),
        "available_source_count": sum(1 for row in rows if row.get("available")),
        "missing_required_target_sources": missing,
        "target_sources": {
            source_id: {
                "available": bool(by_id.get(source_id, {}).get("available", False)),
                "train_allowed": bool(by_id.get(source_id, {}).get("train_allowed", False)),
                "reason_if_missing": by_id.get(source_id, {}).get("reason_if_missing"),
            }
            for source_id in ("traject_bench", "toolbench_g3", "toolret_training")
        },
    }


def _trajectory_filter_report(data_root: Path, allowed_benchmarks: set[str] | None) -> dict[str, Any]:
    allowed = allowed_benchmarks or set()
    enabled = allowed_benchmarks is not None
    retained_by_benchmark: Counter[str] = Counter()
    skipped_by_benchmark: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    total_rows = 0
    retained_train_rows = 0
    post_action_rows = 0
    skipped_by_split: Counter[str] = Counter()
    for row in _read_jsonl_stream(data_root / "trajectories.jsonl"):
        total_rows += 1
        benchmark = str(row.get("benchmark") or "")
        split = _split_name(row)
        split_counts[split or "<missing>"] += 1
        split_allowed = _is_train_row(row)
        benchmark_allowed = not enabled or benchmark in allowed
        retained = benchmark_allowed and split_allowed
        if retained:
            retained_train_rows += 1
            retained_by_benchmark[benchmark] += 1
            if (
                not bool(row.get("done"))
                and str(row.get("state_text") or "").strip()
                and str(row.get("action_text") or "").strip()
                and str(row.get("skill_id") or "").strip()
                and str(row.get("next_skill_id") or "").strip()
            ):
                post_action_rows += 1
        else:
            skipped_by_benchmark[benchmark] += 1
            if not split_allowed:
                skipped_by_split[split or "<missing>"] += 1
    return {
        "enabled": enabled,
        "allowed_benchmarks": sorted(allowed),
        "source_trajectory_rows": total_rows,
        "retained_train_trajectory_rows": retained_train_rows,
        "stage4_post_action_rows": post_action_rows,
        "skipped_rows": total_rows - retained_train_rows,
        "retained_by_benchmark": dict(sorted(retained_by_benchmark.items())),
        "skipped_by_benchmark": dict(sorted(skipped_by_benchmark.items())),
        "skipped_by_split": dict(sorted(skipped_by_split.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }


def _retrieval_filter_report(data_root: Path) -> dict[str, Any]:
    total_rows = 0
    retained_rows = 0
    retained_by_source: Counter[str] = Counter()
    skipped_by_source: Counter[str] = Counter()
    skipped_by_split: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for row in _read_jsonl_stream(data_root / "retrieval.jsonl"):
        total_rows += 1
        source = str(row.get("source") or "")
        split = _split_name(row)
        split_counts[split or "<missing>"] += 1
        if _is_safe_retrieval_row(row):
            retained_rows += 1
            retained_by_source[source] += 1
        else:
            skipped_by_source[source] += 1
            skipped_by_split[split or "<missing>"] += 1
    return {
        "source_retrieval_rows": total_rows,
        "retained_train_retrieval_rows": retained_rows,
        "skipped_rows": total_rows - retained_rows,
        "retained_by_source": dict(sorted(retained_by_source.items())),
        "skipped_by_source": dict(sorted(skipped_by_source.items())),
        "skipped_by_split": dict(sorted(skipped_by_split.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "unsafe_splits": sorted(UNSAFE_RETRIEVAL_SPLITS),
    }


def _skill_pool_quality_report(
    data_root: Path,
    manifest: dict[str, Any],
    skill_dedup_borderline_review_report_path: Path | None = None,
) -> dict[str, Any]:
    skill_pool_manifest = manifest.get("skill_pool") or {}
    borderline_review = skill_pool_manifest.get("borderline_review") or {}
    conservative_review_report = (
        _read_json(skill_dedup_borderline_review_report_path)
        if skill_dedup_borderline_review_report_path is not None
        else {}
    )
    borderline_review_status = str(borderline_review.get("status") or "unknown")
    borderline_candidate_count = int(borderline_review.get("candidate_count") or 0)
    borderline_review_method = borderline_review.get("method")
    if (
        borderline_review_status == "pending_manual_or_llm_review"
        and conservative_review_report.get("status") == "complete"
        and int(conservative_review_report.get("unresolved_candidate_count") or 0) == 0
    ):
        borderline_review_status = "complete_by_conservative_review"
        borderline_candidate_count = 0
        borderline_review_method = conservative_review_report.get("method") or borderline_review_method
    total_rows = 0
    missing_source_record_count = 0
    missing_by_prefix: Counter[str] = Counter()
    missing_samples: list[str] = []
    empty_description_count = 0
    placeholder_description_count = 0
    for row in _read_jsonl_stream(data_root / "skill_pool.jsonl"):
        total_rows += 1
        sid = str(row.get("skill_id") or "")
        provenance = _provenance(row)
        description = str(row.get("description") or "").strip()
        if provenance.get("missing_source_record"):
            missing_source_record_count += 1
            prefix = sid.split("/", 1)[0] if "/" in sid else sid or "<missing>"
            missing_by_prefix[prefix] += 1
            if len(missing_samples) < 20:
                missing_samples.append(sid)
        if not description:
            empty_description_count += 1
        if description and description == sid:
            placeholder_description_count += 1
    blockers = []
    if missing_source_record_count > 0:
        blockers.append("skill_pool_has_placeholder_records")
    if (
        borderline_review_status == "pending_manual_or_llm_review"
        and borderline_candidate_count > 0
    ):
        blockers.append("skill_pool_borderline_review_pending")
    if total_rows <= 0:
        blockers.append("missing_skill_pool")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "skill_count": total_rows,
        "missing_source_record_count": missing_source_record_count,
        "missing_source_record_by_prefix": dict(sorted(missing_by_prefix.items())),
        "missing_source_record_samples": missing_samples,
        "empty_description_count": empty_description_count,
        "placeholder_description_count": placeholder_description_count,
        "borderline_review_status": borderline_review_status,
        "borderline_candidate_count": borderline_candidate_count,
        "borderline_review_method": borderline_review_method,
        "borderline_review_report_path": (
            str(skill_dedup_borderline_review_report_path)
            if skill_dedup_borderline_review_report_path is not None
            else None
        ),
        "borderline_review_report": conservative_review_report,
    }


def _stage_gate(name: str, ready: bool, blockers: list[str], data_rows: int, **extra: Any) -> dict[str, Any]:
    return {
        "stage": name,
        "ready_to_submit": bool(ready),
        "blockers": blockers,
        "data_rows": int(data_rows),
        **extra,
    }


def _checkpoint_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def _stage0_checkpoint_report(data_root: Path, checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"status": "missing", "checkpoint_exists": False}
    if not checkpoint_path.exists():
        return {
            "status": "missing",
            "checkpoint_exists": False,
            "checkpoint_path": str(checkpoint_path),
        }
    try:
        from clstr.stage_preflight import validate_stage2_routing_checkpoint

        report = validate_stage2_routing_checkpoint(
            checkpoint_path=checkpoint_path,
            skills_path=data_root / "skill_pool.jsonl",
        )
    except Exception as exc:
        message = str(exc)
        status = "not_train_safe" if "train-safe retrieval metadata" in message else "invalid"
        return {
            "status": status,
            "checkpoint_exists": True,
            "checkpoint_path": str(checkpoint_path),
            "error": message,
        }
    return {
        "status": "ok",
        "checkpoint_exists": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_stage": report.get("checkpoint_stage"),
        "train_safety": report.get("train_safety") or {},
        "skill_count": report.get("skill_count"),
        "embedding_shape": report.get("embedding_shape"),
        "has_retrieval_scale_and_bias": report.get("has_retrieval_scale_and_bias"),
    }


def _stage0_checkpoint_blocker(checkpoint_report: dict[str, Any]) -> str:
    status = str(checkpoint_report.get("status") or "")
    if status == "not_train_safe":
        return "stage0_checkpoint_not_train_safe"
    if status == "invalid":
        return "stage0_checkpoint_invalid"
    return "missing_stage0_checkpoint"


def _stage0_quality_gate_report(
    stage0_output_dir: Path | None,
    stage0_checkpoint: Path | None,
    stage0_eval_metrics_path: Path | None,
    stage0_baseline_metrics_path: Path | None,
) -> dict[str, Any]:
    if stage0_output_dir is None:
        return {"enabled": False, "status": "not_run"}
    if stage0_checkpoint is None or not stage0_checkpoint.exists():
        return {
            "enabled": True,
            "status": "missing",
            "output_dir": str(stage0_output_dir),
            "checkpoint_path": str(stage0_checkpoint) if stage0_checkpoint else None,
            "blockers": ["missing_stage0_checkpoint"],
        }
    try:
        from clstr.stage0_quality_gate import audit_stage0_biencoder_quality

        report = audit_stage0_biencoder_quality(
            output_dir=stage0_output_dir,
            checkpoint_path=stage0_checkpoint,
            stage0_eval_metrics_path=stage0_eval_metrics_path,
            baseline_metrics_path=stage0_baseline_metrics_path,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "invalid",
            "output_dir": str(stage0_output_dir),
            "checkpoint_path": str(stage0_checkpoint),
            "blockers": ["stage0_quality_gate_error"],
            "error": str(exc),
        }
    report = dict(report)
    report["enabled"] = True
    return report


def _stage2_quality_gate_report(
    stage2_output_dir: Path | None,
    stage2_checkpoint: Path | None,
) -> dict[str, Any]:
    if stage2_output_dir is None:
        return {"enabled": False, "status": "not_run"}
    if stage2_checkpoint is None or not stage2_checkpoint.exists():
        return {
            "enabled": True,
            "status": "missing",
            "output_dir": str(stage2_output_dir),
            "checkpoint_path": str(stage2_checkpoint) if stage2_checkpoint else None,
            "blockers": ["missing_stage2_checkpoint"],
        }
    try:
        from clstr.stage2_quality_gate import audit_stage2_full_base_quality

        report = audit_stage2_full_base_quality(
            output_dir=stage2_output_dir,
            checkpoint_path=stage2_checkpoint,
            expected_route_scorer="unified_memory",
            expected_next_skill_pool_mode="full_pool",
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "invalid",
            "output_dir": str(stage2_output_dir),
            "checkpoint_path": str(stage2_checkpoint),
            "blockers": ["stage2_quality_gate_error"],
            "error": str(exc),
        }
    report = dict(report)
    report["enabled"] = True
    return report


def _stage1_heads_quality_gate_report(
    stage1_output_dir: Path | None,
    stage1_checkpoint: Path | None,
) -> dict[str, Any]:
    if stage1_output_dir is None:
        return {"enabled": False, "status": "not_run"}
    if stage1_checkpoint is None or not stage1_checkpoint.exists():
        return {
            "enabled": True,
            "status": "missing",
            "output_dir": str(stage1_output_dir),
            "checkpoint_path": str(stage1_checkpoint) if stage1_checkpoint else None,
            "blockers": ["missing_stage1_checkpoint"],
        }
    try:
        from clstr.stage1_heads_quality_gate import audit_stage1_heads_quality

        report = audit_stage1_heads_quality(
            output_dir=stage1_output_dir,
            checkpoint_path=stage1_checkpoint,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "status": "invalid",
            "output_dir": str(stage1_output_dir),
            "checkpoint_path": str(stage1_checkpoint),
            "blockers": ["stage1_quality_gate_error"],
            "error": str(exc),
        }
    report = dict(report)
    report["enabled"] = True
    return report


def _stage_gates(
    *,
    data_root: Path,
    manifest: dict[str, Any],
    leakage_ok: bool,
    retrieval_filter: dict[str, Any],
    base_train_filter: dict[str, Any],
    target_train_filter: dict[str, Any],
    skill_pool_quality: dict[str, Any],
    stage0_checkpoint: Path | None,
    stage0_output_dir: Path | None,
    stage0_eval_metrics_path: Path | None,
    stage0_baseline_metrics_path: Path | None,
    stage1_output_dir: Path | None,
    stage1_checkpoint: Path | None,
    stage2_output_dir: Path | None,
    stage2_checkpoint: Path | None,
) -> dict[str, Any]:
    skill_count = int((manifest.get("skill_pool") or {}).get("total_skills") or _count_jsonl(data_root / "skill_pool.jsonl"))
    retrieval_pairs = int(retrieval_filter["retained_train_retrieval_rows"])
    retained_rows = int(base_train_filter["retained_train_trajectory_rows"])
    stage4_rows = int(target_train_filter["stage4_post_action_rows"])

    stage0_blockers = []
    if not leakage_ok:
        stage0_blockers.append("leakage_audit_not_ok")
    if retrieval_pairs <= 0:
        stage0_blockers.append("missing_retrieval_pairs")
    if skill_count <= 0:
        stage0_blockers.append("missing_skill_pool")

    skill_pool_quality_blockers = list(skill_pool_quality.get("blockers") or [])
    stage0_blockers.extend(skill_pool_quality_blockers)
    stage0_ckpt_exists = _checkpoint_exists(stage0_checkpoint)
    stage0_checkpoint_report = _stage0_checkpoint_report(data_root, stage0_checkpoint)
    stage0_checkpoint_usable = stage0_checkpoint_report["status"] == "ok"
    stage0_quality_gate = _stage0_quality_gate_report(
        stage0_output_dir,
        stage0_checkpoint,
        stage0_eval_metrics_path,
        stage0_baseline_metrics_path,
    )
    stage0_quality_gate_ok = (
        not bool(stage0_quality_gate.get("enabled"))
        or str(stage0_quality_gate.get("status")) == "ok"
    )

    stage1_blockers = []
    if not leakage_ok:
        stage1_blockers.append("leakage_audit_not_ok")
    stage1_blockers.extend(skill_pool_quality_blockers)
    if not stage0_checkpoint_usable:
        stage1_blockers.append(_stage0_checkpoint_blocker(stage0_checkpoint_report))
    elif not stage0_quality_gate_ok:
        stage1_blockers.append("stage0_quality_gate_not_ok")
    if retained_rows <= 0:
        stage1_blockers.append("no_allowed_train_trajectory_rows")
    if skill_count <= 0:
        stage1_blockers.append("missing_skill_pool")
    stage1_ckpt_exists = _checkpoint_exists(stage1_checkpoint)
    stage1_quality_gate = _stage1_heads_quality_gate_report(stage1_output_dir, stage1_checkpoint)
    stage1_quality_gate_ok = (
        bool(stage1_quality_gate.get("enabled"))
        and str(stage1_quality_gate.get("status")) == "ok"
    )
    if not stage1_ckpt_exists:
        stage1_blockers.append("missing_stage1_checkpoint")
    elif not stage1_quality_gate_ok:
        stage1_blockers.append("stage1_quality_gate_not_ok")

    stage2_blockers = []
    if not leakage_ok:
        stage2_blockers.append("leakage_audit_not_ok")
    stage2_blockers.extend(skill_pool_quality_blockers)
    if not stage0_checkpoint_usable:
        stage2_blockers.append(_stage0_checkpoint_blocker(stage0_checkpoint_report))
    elif not stage0_quality_gate_ok:
        stage2_blockers.append("stage0_quality_gate_not_ok")
    if not stage1_ckpt_exists:
        stage2_blockers.append("missing_stage1_checkpoint")
    elif not stage1_quality_gate_ok:
        stage2_blockers.append("stage1_quality_gate_not_ok")
    if retained_rows <= 0:
        stage2_blockers.append("no_allowed_train_trajectory_rows")
    if skill_count <= 0:
        stage2_blockers.append("missing_skill_pool")

    stage2_ckpt_exists = _checkpoint_exists(stage2_checkpoint)
    stage2_quality_gate = _stage2_quality_gate_report(stage2_output_dir, stage2_checkpoint)
    stage2_quality_gate_ok = (
        not bool(stage2_quality_gate.get("enabled"))
        or str(stage2_quality_gate.get("status")) == "ok"
    )

    stage4_blockers = []
    if not leakage_ok:
        stage4_blockers.append("leakage_audit_not_ok")
    stage4_blockers.extend(skill_pool_quality_blockers)
    if not stage0_checkpoint_usable:
        stage4_blockers.append(_stage0_checkpoint_blocker(stage0_checkpoint_report))
    elif not stage0_quality_gate_ok:
        stage4_blockers.append("stage0_quality_gate_not_ok")
    if not stage2_ckpt_exists:
        stage4_blockers.append("missing_stage2_checkpoint")
    elif not stage2_quality_gate_ok:
        stage4_blockers.append("stage2_quality_gate_not_ok")
    if stage4_rows <= 0:
        stage4_blockers.append("no_stage4_post_action_rows")

    stage0_gate = _stage_gate(
        "stage0_routing",
        not stage0_blockers,
        stage0_blockers,
        retrieval_pairs,
        skill_count=skill_count,
    )
    return {
        "stage0_routing": stage0_gate,
        "stage1_heads": _stage_gate(
            "stage1_heads",
            not stage1_blockers,
            stage1_blockers,
            retained_rows,
            stage0_checkpoint=str(stage0_checkpoint) if stage0_checkpoint else None,
            stage0_checkpoint_exists=stage0_ckpt_exists,
            stage0_checkpoint_train_safety=stage0_checkpoint_report,
            stage0_quality_gate=stage0_quality_gate,
            stage1_checkpoint=str(stage1_checkpoint) if stage1_checkpoint else None,
            stage1_checkpoint_exists=stage1_ckpt_exists,
            stage1_quality_gate=stage1_quality_gate,
        ),
        "stage2_full_base": _stage_gate(
            "stage2_full_base",
            not stage2_blockers,
            stage2_blockers,
            retained_rows,
            stage0_checkpoint=str(stage0_checkpoint) if stage0_checkpoint else None,
            stage0_checkpoint_exists=stage0_ckpt_exists,
            stage0_checkpoint_train_safety=stage0_checkpoint_report,
            stage0_quality_gate=stage0_quality_gate,
            stage1_checkpoint=str(stage1_checkpoint) if stage1_checkpoint else None,
            stage1_checkpoint_exists=stage1_ckpt_exists,
            stage1_quality_gate=stage1_quality_gate,
        ),
        "stage4_causal_next_skill": _stage_gate(
            "stage4_causal_next_skill",
            not stage4_blockers,
            stage4_blockers,
            stage4_rows,
            training_objective="causal_transition_conditioned_next_skill_ce",
            training_regime="offline_train_split_causal_next_skill",
            post_action_training_rows=stage4_rows,
            on_policy_rollout_used=False,
            stage0_checkpoint_exists=stage0_ckpt_exists,
            stage0_checkpoint_train_safety=stage0_checkpoint_report,
            stage0_quality_gate=stage0_quality_gate,
            stage1_checkpoint_exists=stage1_ckpt_exists,
            stage1_quality_gate=stage1_quality_gate,
            stage2_checkpoint=str(stage2_checkpoint) if stage2_checkpoint else None,
            stage2_checkpoint_exists=stage2_ckpt_exists,
            stage2_quality_gate=stage2_quality_gate,
            act_init_checkpoint=str(stage2_checkpoint) if stage2_checkpoint else None,
            act_init_source="stage2_full_base_checkpoint",
            head_checkpoint_exists=stage2_ckpt_exists,
        ),
    }


def audit_unified_training_readiness(
    *,
    data_root: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final",
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | str | None = None,
    skill_dedup_borderline_review_report_path: str | Path | None = None,
    stage0_checkpoint: str | Path | None = None,
    stage0_output_dir: str | Path | None = None,
    stage0_eval_metrics_path: str | Path | None = None,
    stage0_baseline_metrics_path: str | Path | None = "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json",
    stage1_checkpoint: str | Path | None = None,
    stage1_output_dir: str | Path | None = None,
    stage2_output_dir: str | Path | None = None,
    stage2_checkpoint: str | Path | None = None,
    toolret_eval_dir: str | Path = "data/toolret_eval",
    traject_eval_dir: str | Path = "data/traject_eval_traject_split_test",
    toolbench_g3_data_dir: str | Path = "data/toolbench_g3",
    toolbench_g3_source_root: str | Path | None = "../ToolBench/data",
    expected_toolbench_g3_answer_files: int | None = None,
    candidate_recall_mode: str = EXPECTED_CANDIDATE_RECALL_MODE,
    candidate_recall_route_scorer: str = "unified_memory",
    candidate_recall_static_k: int = 500,
    candidate_recall_dynamic_extra_k: int = 64,
    candidate_recall_final_k: int = 64,
    candidate_recall_denominator_scope: str = "all_eligible_source_strict",
    candidate_recall_gold_positive_injected: bool = False,
    candidate_recall_legal_pool_source: str = "declared_global_skill_pool",
    candidate_recall_equal_budget_comparator_mode: str = EXPECTED_EQUAL_BUDGET_COMPARATOR_MODE,
    memory_utility_gate_mode: str = "dynamic",
    memory_utility_zero_history_fallback: str = EXPECTED_ZERO_HISTORY_FALLBACK,
    memory_utility_reliability_feature_schema: str = EXPECTED_RELIABILITY_FEATURE_SCHEMA,
    memory_utility_reliability_changes_memory_state: bool = False,
    memory_utility_audit_report_path: str | Path | None = None,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    allowed = _parse_allowed_benchmarks(allowed_benchmarks)
    manifest = _read_json(data_root / "manifest.json")
    leakage = _read_json(data_root / "leakage_audit.json")
    leakage_status = str(leakage.get("status") or (manifest.get("leakage_audit") or {}).get("status") or "missing")
    leakage_ok = leakage_status == "ok"
    source_inventory = _source_inventory_report(data_root, manifest)
    benchmark_filter = _trajectory_filter_report(data_root, allowed)
    base_train_filter = _trajectory_filter_report(data_root, None)
    retrieval_filter = _retrieval_filter_report(data_root)
    skill_pool_quality = _skill_pool_quality_report(
        data_root,
        manifest,
        Path(skill_dedup_borderline_review_report_path) if skill_dedup_borderline_review_report_path else None,
    )
    toolret_eval_report = audit_toolret_eval_data(data_dir=toolret_eval_dir)
    traject_eval_report = audit_traject_eval_data(data_dir=traject_eval_dir)
    toolbench_g3_report = audit_toolbench_g3_data(
        data_dir=toolbench_g3_data_dir,
        source_root=toolbench_g3_source_root,
        expected_answer_files=expected_toolbench_g3_answer_files,
    )
    candidate_recall_readiness = audit_memory_candidate_recall_readiness(
        candidate_recall_mode=candidate_recall_mode,
        route_scorer=candidate_recall_route_scorer,
        static_k=candidate_recall_static_k,
        dynamic_extra_k=candidate_recall_dynamic_extra_k,
        final_k=candidate_recall_final_k,
        denominator_scope=candidate_recall_denominator_scope,
        gold_positive_injected=candidate_recall_gold_positive_injected,
        legal_pool_source=candidate_recall_legal_pool_source,
        equal_budget_comparator_mode=candidate_recall_equal_budget_comparator_mode,
    )
    memory_utility_reliability_readiness = audit_memory_utility_reliability_readiness(
        mode=memory_utility_gate_mode,
        zero_history_fallback=memory_utility_zero_history_fallback,
        reliability_feature_schema=memory_utility_reliability_feature_schema,
        reliability_changes_memory_state=memory_utility_reliability_changes_memory_state,
        audit_report_path=memory_utility_audit_report_path,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
    )
    effective_stage0_eval_metrics_path = stage0_eval_metrics_path
    if effective_stage0_eval_metrics_path is None and stage0_output_dir is not None:
        effective_stage0_eval_metrics_path = Path(stage0_output_dir) / DEFAULT_STAGE0_EVAL_METRICS_NAME
    gates = _stage_gates(
        data_root=data_root,
        manifest=manifest,
        leakage_ok=leakage_ok,
        retrieval_filter=retrieval_filter,
        base_train_filter=base_train_filter,
        target_train_filter=benchmark_filter,
        skill_pool_quality=skill_pool_quality,
        stage0_checkpoint=Path(stage0_checkpoint) if stage0_checkpoint else None,
        stage0_output_dir=Path(stage0_output_dir) if stage0_output_dir else None,
        stage0_eval_metrics_path=Path(effective_stage0_eval_metrics_path) if effective_stage0_eval_metrics_path else None,
        stage0_baseline_metrics_path=Path(stage0_baseline_metrics_path) if stage0_baseline_metrics_path else None,
        stage1_output_dir=Path(stage1_output_dir) if stage1_output_dir else None,
        stage1_checkpoint=Path(stage1_checkpoint) if stage1_checkpoint else None,
        stage2_output_dir=Path(stage2_output_dir) if stage2_output_dir else None,
        stage2_checkpoint=Path(stage2_checkpoint) if stage2_checkpoint else None,
    )
    target_sources = source_inventory["target_sources"]
    full_three_benchmark_ready = all(
        bool(target_sources.get(source_id, {}).get("available"))
        for source_id in ("traject_bench", "toolbench_g3", "toolret_training")
    )
    toolret_eval_ready = toolret_eval_report.get("status") == "ok"
    traject_eval_ready = traject_eval_report.get("status") == "ok"
    toolbench_g3_data_ready = toolbench_g3_report.get("status") == "ok"
    status = (
        "ready"
        if all(gate["ready_to_submit"] for gate in gates.values())
        and full_three_benchmark_ready
        and candidate_recall_readiness["status"] == "ok"
        and memory_utility_reliability_readiness["status"] == "ok"
        else "action_required"
    )
    report = {
        "status": status,
        "data_root": str(data_root),
        "allowed_benchmarks": sorted(allowed or []),
        "files": _file_report(data_root),
        "manifest": {
            "status": manifest.get("status"),
            "schema_version": manifest.get("schema_version"),
            "trajectory_rows": (manifest.get("trajectory_stream") or {}).get("total_rows"),
            "retrieval_pairs": (manifest.get("retrieval_stream") or {}).get("total_pairs"),
            "skill_count": (manifest.get("skill_pool") or {}).get("total_skills"),
        },
        "leakage_audit": {
            **leakage,
            "status": leakage_status,
        },
        "source_inventory": source_inventory,
        "skill_pool_quality": skill_pool_quality,
        "retrieval_filter": retrieval_filter,
        "base_train_filter": base_train_filter,
        "benchmark_filter": benchmark_filter,
        "benchmark_eval_readiness": {
            "toolret": toolret_eval_report,
            "traject": traject_eval_report,
            "toolbench_g3": toolbench_g3_report,
        },
        "stage_gates": gates,
        "candidate_recall_readiness": candidate_recall_readiness,
        "memory_utility_reliability_readiness": memory_utility_reliability_readiness,
        "paper_readiness": {
            "full_three_benchmark_data_ready": full_three_benchmark_ready,
            "toolret_eval_ready": toolret_eval_ready,
            "traject_eval_ready": traject_eval_ready,
            "toolbench_g3_data_ready": toolbench_g3_data_ready,
            "candidate_recall_ready": candidate_recall_readiness["status"] == "ok",
            "memory_utility_reliability_ready": (
                memory_utility_reliability_readiness["status"] == "ok"
            ),
            "active_training_path": [
                "stage0_routing",
                "stage1_heads",
                "stage2_full_base",
                "stage4_causal_next_skill",
            ],
            "appworld_is_secondary": True,
            "missing_required_target_sources": source_inventory["missing_required_target_sources"],
        },
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unified CLSTR training data and stage submission readiness.")
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final")
    parser.add_argument("--allowed_benchmarks", default="toolbench_g3,traject_bench,alfworld,webshop")
    parser.add_argument(
        "--skill_dedup_borderline_review_report_path",
        default="outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json",
    )
    parser.add_argument(
        "--stage0_checkpoint",
        default=DEFAULT_STAGE0_CHECKPOINT,
    )
    parser.add_argument(
        "--stage0_output_dir",
        default=DEFAULT_STAGE0_OUTPUT_DIR,
    )
    parser.add_argument(
        "--stage0_eval_metrics_path",
        default=None,
    )
    parser.add_argument(
        "--stage0_baseline_metrics_path",
        default="outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        default=DEFAULT_STAGE1_CHECKPOINT,
    )
    parser.add_argument(
        "--stage1_output_dir",
        default=DEFAULT_STAGE1_OUTPUT_DIR,
    )
    parser.add_argument(
        "--stage2_output_dir",
        default=DEFAULT_STAGE2_OUTPUT_DIR,
    )
    parser.add_argument(
        "--stage2_checkpoint",
        default=DEFAULT_STAGE2_CHECKPOINT,
    )
    parser.add_argument("--toolret_eval_dir", default="data/toolret_eval")
    parser.add_argument("--traject_eval_dir", default="data/traject_eval_traject_split_test")
    parser.add_argument("--toolbench_g3_data_dir", default="data/toolbench_g3")
    parser.add_argument("--toolbench_g3_source_root", default="../ToolBench/data")
    parser.add_argument("--expected_toolbench_g3_answer_files", type=int, default=None)
    parser.add_argument("--candidate_recall_mode", default=EXPECTED_CANDIDATE_RECALL_MODE)
    parser.add_argument("--candidate_recall_route_scorer", default="unified_memory")
    parser.add_argument("--candidate_recall_static_k", type=int, default=500)
    parser.add_argument("--candidate_recall_dynamic_extra_k", type=int, default=64)
    parser.add_argument("--candidate_recall_final_k", type=int, default=64)
    parser.add_argument("--candidate_recall_denominator_scope", default="all_eligible_source_strict")
    parser.add_argument("--candidate_recall_gold_positive_injected", action="store_true")
    parser.add_argument("--candidate_recall_legal_pool_source", default="declared_global_skill_pool")
    parser.add_argument(
        "--candidate_recall_equal_budget_comparator_mode",
        default=EXPECTED_EQUAL_BUDGET_COMPARATOR_MODE,
    )
    parser.add_argument(
        "--memory_utility_gate_mode",
        default="dynamic",
        choices=sorted(MEMORY_UTILITY_RELIABILITY_MODES),
    )
    parser.add_argument(
        "--memory_utility_zero_history_fallback",
        default=EXPECTED_ZERO_HISTORY_FALLBACK,
    )
    parser.add_argument(
        "--memory_utility_reliability_feature_schema",
        default=EXPECTED_RELIABILITY_FEATURE_SCHEMA,
    )
    parser.add_argument(
        "--memory_utility_reliability_changes_memory_state",
        action="store_true",
    )
    parser.add_argument("--memory_utility_audit_report_path")
    parser.add_argument("--memory_utility_gate_checkpoint_path")
    parser.add_argument("--output_path", default="outputs/clstr_unified_readiness_audit/readiness_report.json")
    args = parser.parse_args()
    report = audit_unified_training_readiness(
        data_root=args.data_root,
        allowed_benchmarks=args.allowed_benchmarks,
        skill_dedup_borderline_review_report_path=args.skill_dedup_borderline_review_report_path,
        stage0_checkpoint=args.stage0_checkpoint,
        stage0_output_dir=args.stage0_output_dir,
        stage0_eval_metrics_path=args.stage0_eval_metrics_path,
        stage0_baseline_metrics_path=args.stage0_baseline_metrics_path,
        stage1_checkpoint=args.stage1_checkpoint,
        stage1_output_dir=args.stage1_output_dir,
        stage2_output_dir=args.stage2_output_dir,
        stage2_checkpoint=args.stage2_checkpoint,
        toolret_eval_dir=args.toolret_eval_dir,
        traject_eval_dir=args.traject_eval_dir,
        toolbench_g3_data_dir=args.toolbench_g3_data_dir,
        toolbench_g3_source_root=args.toolbench_g3_source_root,
        expected_toolbench_g3_answer_files=args.expected_toolbench_g3_answer_files,
        candidate_recall_mode=args.candidate_recall_mode,
        candidate_recall_route_scorer=args.candidate_recall_route_scorer,
        candidate_recall_static_k=args.candidate_recall_static_k,
        candidate_recall_dynamic_extra_k=args.candidate_recall_dynamic_extra_k,
        candidate_recall_final_k=args.candidate_recall_final_k,
        candidate_recall_denominator_scope=args.candidate_recall_denominator_scope,
        candidate_recall_gold_positive_injected=args.candidate_recall_gold_positive_injected,
        candidate_recall_legal_pool_source=args.candidate_recall_legal_pool_source,
        candidate_recall_equal_budget_comparator_mode=(
            args.candidate_recall_equal_budget_comparator_mode
        ),
        memory_utility_gate_mode=args.memory_utility_gate_mode,
        memory_utility_zero_history_fallback=args.memory_utility_zero_history_fallback,
        memory_utility_reliability_feature_schema=(
            args.memory_utility_reliability_feature_schema
        ),
        memory_utility_reliability_changes_memory_state=(
            args.memory_utility_reliability_changes_memory_state
        ),
        memory_utility_audit_report_path=args.memory_utility_audit_report_path,
        memory_utility_gate_checkpoint_path=args.memory_utility_gate_checkpoint_path,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
