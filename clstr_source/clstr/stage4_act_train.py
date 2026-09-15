from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.candidate_admission_residual import (
    CANDIDATE_ADMISSION_RESIDUAL_V1,
    CandidateAdmissionResidualObjective,
    CandidateAdmissionScoringOutput,
    candidate_admission_residual_loss,
    score_candidate_admission_residual,
)
from clstr.counterfactual_memory_calibration import (
    CMC_FEATURE_CANDIDATE_COUNT_CAP,
    CMC_FEATURE_UPDATE_COUNT_CAP,
    CounterfactualMemoryCalibrationOutput,
    counterfactual_memory_calibration_loss,
    score_cmc_candidates,
)
from clstr.counterfactual_ranking import full_pool_causal_route_objective
from clstr.external_data import write_json
from clstr.full_base_train import (
    DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    NEXT_SKILL_POOL_MODES,
    ROUTE_SCORERS,
    STATE_QUERY_ROLE,
    SAFE_LOCAL_CANDIDATE_SIZES,
    TRANSITION_SCORING_MODE,
    TRANSITION_SCORING_MODES,
    TRANSITION_TEXT_ROLE,
    UNIFIED_MEMORY_ROUTE_SCORER,
    _apply_replay_prefix_beliefs,
    _attach_adjacent_next_states,
    _attach_auto_replay_prefixes,
    _attach_full_base_embedding_cache,
    _prepare_stage0_topm_candidates_with_cache,
    _batch_action_text_embedding_or_none,
    _batch_cached_or_encode,
    _cap_rows_by_benchmark,
    _checkpoint_state_dict,
    _counterfactual_warmup_scale,
    _dynamic_static_rank_metrics,
    _equivalent_skill_ids_by_skill_id,
    _filter_transition_candidates_by_inventory,
    _post_action_memory,
    _ranking_metrics_from_logits,
    _skill_logits_and_memory,
    _transition_candidate_logits_for_mode,
    _validate_counterfactual_hyperparameters,
)
from clstr.memory_candidate_recall import (
    CandidateUnion,
    build_static_dynamic_union,
    candidate_provenance_mask,
    explicit_inventory_skill_ids_ordered,
    full_pool_positive_mask,
    legal_skill_pool_mask,
)
from clstr.memory_utility_records import canonical_digest
from clstr.memory_utility_gate import (
    effective_memory_alpha,
    memory_utility_features,
)
from clstr.qwen_clstr_lineage import sha256_path
from clstr.safe_memory_ranking import (
    bounded_memory_fusion,
    build_local_candidate_masks,
    safe_local_route_objective,
)
from clstr.stage4_data_protocol import (
    BenchmarkBalancedStage4Batcher,
    build_stage4_data_protocol,
)
from clstr.stage4_safe_memory import (
    freeze_stage4_candidate_admission,
    freeze_stage4_cmc,
    freeze_stage4_safe_memory,
    router_state_digest,
    set_stage4_candidate_admission_training_mode,
    set_stage4_cmc_training_mode,
    set_stage4_safe_training_mode,
    stage4_cmc_delta_state_dict,
    stage4_candidate_admission_delta_state_dict,
    stage4_delta_state_dict,
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_candidate_admission_delta_state_dict,
    validate_stage4_cmc_parent_router,
    validate_stage4_delta_state_dict,
)
from clstr.stage4_validation import (
    build_stage4_validation_report,
    evaluate_stage4_validation,
    persist_stage4_validation_artifacts,
    select_stage4_candidate_admission_checkpoint,
    select_stage4_cmc_checkpoint,
    select_stage4_dynamic_checkpoint,
    stage2_baseline_from_evaluation,
    validate_static_baseline_batches,
    write_stage4_dynamic_selection,
)
from clstr.state_query_prompt import resolve_state_query_prompt_contract
from clstr.training_monitor import TrainingMonitor, append_setup_status, reset_setup_status


TRAIN_SPLIT_NAMES = {"train", "training", "train_or_released_g3", "released_train"}
STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE = "stage0_rank_prior_plus_transition_residual"
STAGE4_TRANSITION_SCORING_MODES = set(TRANSITION_SCORING_MODES) | {STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE}
STAGE4_SCORE_CALIBRATOR_NAME = "stage4_score_calibrator"
LEGACY_SAFE_MEMORY_STAGE4_V1 = "stage4_safe_memory_v1"
COUNTERFACTUAL_MEMORY_CALIBRATION_V1 = "counterfactual_memory_calibration_v1"
STAGE4_METHODS = {
    LEGACY_SAFE_MEMORY_STAGE4_V1,
    COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
    CANDIDATE_ADMISSION_RESIDUAL_V1,
}
STAGE4_SCORE_CALIBRATOR_FEATURE_NAMES = (
    "prior_logit",
    "transition_residual_logit",
    "online_memory_logit",
    "prior_x_online_memory",
    "transition_residual_x_online_memory",
    "online_memory_hit",
)


def _read_jsonl_stream(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt JSONL at {path} line {line_no}: {exc}") from exc


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(_read_jsonl_stream(path))


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
    split = _split_name(row)
    return split in TRAIN_SPLIT_NAMES


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [value]
    return [value]


def _candidate_skill_ids(row: dict[str, Any]) -> list[str]:
    for key in (
        "candidate_next_skill_ids",
        "candidates_next_skill_ids",
        "admissible_next_skill_ids",
        "candidate_skill_ids",
    ):
        values = [str(item) for item in _as_list(row.get(key)) if str(item)]
        if values:
            return values
    return []


def _stage0_next_candidate_skill_ids(row: dict[str, Any], skill_ids: list[str]) -> list[str]:
    values = [str(item) for item in _as_list(row.get("stage0_next_candidate_skill_ids")) if str(item)]
    if values:
        return values
    indices = []
    for item in _as_list(row.get("stage0_next_candidate_skill_indices")):
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(skill_ids):
            indices.append(str(skill_ids[idx]))
    return indices


def _stable_negative_fill(
    *,
    seed_text: str,
    existing: list[str],
    skill_ids,
    candidate_count: int | None,
) -> list[str]:
    if candidate_count is None or len(existing) >= int(candidate_count):
        return existing
    wanted = max(1, int(candidate_count))
    used = set(existing)
    pool_size = len(skill_ids)
    if pool_size <= 0:
        return existing
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    sample_size = min(pool_size, max(wanted * 4, wanted + len(used)))
    sampled_indices = rng.sample(range(pool_size), k=sample_size)
    filled = list(existing)
    for idx in sampled_indices:
        skill_id = str(skill_ids[idx])
        if skill_id in used:
            continue
        used.add(skill_id)
        filled.append(skill_id)
        if len(filled) >= wanted:
            break
    if len(filled) >= wanted:
        return filled

    # Small or heavily filtered pools may not fill from the random sample.
    # Walk deterministically from a seed offset without materializing the pool.
    start = seed % pool_size
    for offset in range(pool_size):
        skill_id = str(skill_ids[(start + offset) % pool_size])
        if skill_id in used:
            continue
        used.add(skill_id)
        filled.append(skill_id)
        if len(filled) >= min(wanted, pool_size):
            break
    return filled


def _candidate_rank_prior_scores(candidate_count: int) -> list[float]:
    return [-math.log(float(rank)) for rank in range(1, max(0, int(candidate_count)) + 1)]


def _candidate_prior_scores_from_stage0_row(
    row: dict[str, Any],
    candidate_indices: list[int],
) -> list[float]:
    score_map: dict[int, float] = {}
    indices = row.get("stage0_next_candidate_skill_indices")
    scores = row.get("stage0_next_candidate_skill_scores")
    if isinstance(indices, list) and isinstance(scores, list) and len(indices) == len(scores):
        for idx, score in zip(indices, scores):
            try:
                score_map[int(idx)] = float(score)
            except (TypeError, ValueError):
                continue
    rank_scores = _candidate_rank_prior_scores(len(candidate_indices))
    if not score_map:
        return rank_scores
    return [
        float(score_map.get(int(idx), rank_scores[pos]))
        for pos, idx in enumerate(candidate_indices)
    ]


def _stage4_causal_field_skip_reason(row: dict[str, Any]) -> str | None:
    adjacency_reason = str(row.get("causal_next_state_skip_reason") or "").strip()
    if adjacency_reason:
        return adjacency_reason
    for key, reason in (
        ("state_text", "missing_state_text"),
        ("action_text", "missing_action_text"),
        ("next_observation_text", "missing_next_observation_text"),
        ("next_state_text", "missing_next_state_text"),
        ("skill_id", "missing_current_skill"),
        ("next_skill_id", "missing_next_skill"),
    ):
        if not str(row.get(key) or "").strip():
            return reason
    return None


def _eligible_stage4_source_rows(
    source_rows_iter,
    skill_id_to_idx: dict[str, int],
    *,
    allowed_benchmarks: set[str] | None = None,
    max_source_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_rows = 0
    skipped_benchmarks: Counter[str] = Counter()
    allowed_report = sorted(allowed_benchmarks) if allowed_benchmarks is not None else []

    for row in source_rows_iter:
        source_rows += 1
        benchmark = str(row.get("benchmark") or "")
        if allowed_benchmarks is not None and benchmark not in allowed_benchmarks:
            skipped["benchmark_not_allowed"] += 1
            skipped_benchmarks[benchmark] += 1
            continue
        if not _is_train_row(row):
            skipped["split_not_train"] += 1
            continue
        if bool(row.get("done")):
            skipped["missing_next_skill"] += 1
            continue
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if not current_skill_id:
            skipped["missing_current_skill"] += 1
            continue
        if not next_skill_id:
            skipped["missing_next_skill"] += 1
            continue
        causal_skip_reason = _stage4_causal_field_skip_reason(row)
        if causal_skip_reason is not None:
            skipped[causal_skip_reason] += 1
            continue
        if current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if next_skill_id not in skill_id_to_idx:
            skipped["next_skill_not_in_pool"] += 1
            continue

        copied = dict(row)
        copied.setdefault("loss_mask", {"routing": True, "L_trans_skill_ce": True})
        rows.append(copied)
        if max_source_rows is not None and len(rows) >= int(max_source_rows):
            break

    return rows, {
        "source_rows": source_rows,
        "eligible_source_rows": len(rows),
        "benchmark_filter_enabled": allowed_benchmarks is not None,
        "allowed_benchmarks": allowed_report,
        "skipped_benchmarks": dict(sorted(skipped_benchmarks.items())),
        "skipped_reasons": dict(skipped),
        "max_source_rows": max_source_rows,
    }


def _build_stage4_next_skill_rows_from_source_rows(
    source_rows_list: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    candidate_count: int | None = None,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    stage0_candidate_handoff_report: dict[str, Any] | None = None,
    next_skill_pool_mode: str = "stage0_candidates",
    require_next_state_text: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    next_skill_pool_mode = str(next_skill_pool_mode or "stage0_candidates")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_counts_by_benchmark: Counter[str] = Counter()
    skipped_by_benchmark: Counter[str] = Counter()
    positive_missing_skip_by_benchmark: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    stage0_candidate_filter_audit: Counter[str] = Counter()
    injected = 0
    stage0_candidate_rows = 0
    explicit_candidate_rows = 0
    random_negative_fill_rows = 0
    stage0_next_static_hit_rows = 0
    stage0_next_static_miss_retained_rows = 0
    empty_natural_candidate_rows = 0
    skill_ids = list(skill_id_to_idx)
    skill_ids_by_idx = {int(idx): str(skill_id) for skill_id, idx in skill_id_to_idx.items()}
    allowed_report = sorted(allowed_benchmarks) if allowed_benchmarks is not None else []
    handoff_enabled = bool(stage0_candidate_handoff_report and stage0_candidate_handoff_report.get("enabled"))

    def record_skip(reason: str, benchmark: str, *, positive_missing: bool = False) -> None:
        skipped[reason] += 1
        skipped_by_benchmark[benchmark] += 1
        if positive_missing:
            positive_missing_skip_by_benchmark[benchmark] += 1

    for source_rows, row in enumerate(source_rows_list, start=1):
        benchmark = str(row.get("benchmark") or "")
        source_counts_by_benchmark[benchmark] += 1
        causal_skip_reason = _stage4_causal_field_skip_reason(row)
        if causal_skip_reason == "missing_next_state_text" and not require_next_state_text:
            causal_skip_reason = None
        if causal_skip_reason is not None:
            record_skip(causal_skip_reason, benchmark)
            continue
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        candidate_ids: list[str] = []
        seen: set[str] = set()
        positive_injected = False
        stage0_candidate_ids = _stage0_next_candidate_skill_ids(row, skill_ids)
        if next_skill_pool_mode == "full_pool":
            if stage0_candidate_ids:
                stage0_candidate_rows += 1
            else:
                empty_natural_candidate_rows += 1
            for candidate_id in stage0_candidate_ids:
                if candidate_id in skill_id_to_idx and candidate_id not in seen:
                    seen.add(candidate_id)
                    candidate_ids.append(candidate_id)
            next_positive_hit = bool(row.get("stage0_next_positive_hit")) if row.get("stage0_next_positive_hit") is not None else next_skill_id in seen
            if next_positive_hit:
                stage0_next_static_hit_rows += 1
            else:
                stage0_next_static_miss_retained_rows += 1
        elif stage0_candidate_ids:
            stage0_candidate_rows += 1
            if candidate_count is not None and int(candidate_count) > 0:
                original_indices = [
                    int(skill_id_to_idx[candidate_id])
                    for candidate_id in stage0_candidate_ids
                    if candidate_id in skill_id_to_idx
                ]
                if original_indices:
                    filtered_indices, filter_audit = _filter_transition_candidates_by_inventory(
                        rows=[row],
                        candidate_rows=[original_indices],
                        labels=torch.tensor([int(skill_id_to_idx[next_skill_id])], dtype=torch.long),
                        skill_ids_by_idx=skill_ids_by_idx,
                        mode="stage0_topk_trajectory_prior",
                        min_candidates=int(candidate_count),
                    )
                    stage0_candidate_filter_audit.update(filter_audit)
                    stage0_candidate_ids = [
                        skill_ids_by_idx[int(idx)]
                        for idx in (filtered_indices[0] if filtered_indices else [])
                        if int(idx) in skill_ids_by_idx
                    ]
            for candidate_id in stage0_candidate_ids:
                if candidate_id in skill_id_to_idx and candidate_id not in seen:
                    seen.add(candidate_id)
                    candidate_ids.append(candidate_id)
            if next_skill_id not in seen:
                record_skip("stage0_next_positive_missing_from_topm", benchmark, positive_missing=True)
                continue
        else:
            if handoff_enabled:
                record_skip("missing_stage0_next_candidates", benchmark)
                continue
            for candidate_id in _candidate_skill_ids(row):
                if candidate_id in skill_id_to_idx and candidate_id not in seen:
                    seen.add(candidate_id)
                    candidate_ids.append(candidate_id)
            if candidate_ids:
                explicit_candidate_rows += 1
            if next_skill_id not in seen:
                candidate_ids.append(next_skill_id)
                positive_injected = True
                injected += 1
            before_fill = len(candidate_ids)
            candidate_ids = _stable_negative_fill(
                seed_text=str(row.get("task_id") or row.get("trajectory_id") or source_rows),
                existing=candidate_ids,
                skill_ids=skill_ids,
                candidate_count=candidate_count,
            )
            if len(candidate_ids) > before_fill:
                random_negative_fill_rows += 1
        if not candidate_ids and next_skill_pool_mode == "stage0_candidates":
            record_skip("empty_candidates", benchmark)
            continue

        positive_pos = candidate_ids.index(next_skill_id) if next_skill_id in candidate_ids else None
        candidate_indices = [int(skill_id_to_idx[item]) for item in candidate_ids]
        stage4_row = {
            "task_id": row.get("task_id"),
            "trajectory_id": row.get("trajectory_id"),
            "step_index": row.get("step_index"),
            "source_benchmark": benchmark,
            "state_text": str(row["state_text"]),
            "action_text": str(row["action_text"]),
            "next_observation_text": str(row["next_observation_text"]),
            "next_state_text": str(row.get("next_state_text") or ""),
            "skill_id": current_skill_id,
            "next_skill_id": next_skill_id,
            "skill_idx": int(skill_id_to_idx[current_skill_id]),
            "positive_next_skill_idx": int(skill_id_to_idx[next_skill_id]),
            "positive_next_skill_position": None if positive_pos is None else int(positive_pos),
            "candidate_next_skill_ids": candidate_ids,
            "candidate_next_skill_indices": candidate_indices,
            "candidate_next_prior_scores": _candidate_prior_scores_from_stage0_row(row, candidate_indices),
            "positive_injected": bool(positive_injected),
            "stage0_current_positive_hit": row.get("stage0_current_positive_hit"),
            "stage0_next_positive_hit": (
                bool(row.get("stage0_next_positive_hit"))
                if row.get("stage0_next_positive_hit") is not None
                else next_skill_id in candidate_ids
            ),
            "provenance": {
                "source": "stage4_transition_conditioned_next_skill",
                "original_provenance": _provenance(row),
                "candidate_count_requested": candidate_count,
            },
        }
        for inventory_key in (
            "visible_inventory_skill_ids",
            "available_skill_ids",
            "available_skills",
            "admissible_skill_ids",
            "skill_inventory_ids",
            "stage0_allowed_skill_ids",
        ):
            if row.get(inventory_key):
                stage4_row[inventory_key] = row.get(inventory_key)
        rows.append(stage4_row)
        benchmark_counts[benchmark] += 1
        if max_rows is not None and len(rows) >= int(max_rows):
            break

    skip_distribution_by_benchmark = {}
    for benchmark, source_count in sorted(source_counts_by_benchmark.items()):
        retained = int(benchmark_counts.get(benchmark, 0))
        skipped_count = int(skipped_by_benchmark.get(benchmark, 0))
        skip_distribution_by_benchmark[benchmark] = {
            "source_rows": int(source_count),
            "stage4_rows": retained,
            "skipped_rows": skipped_count,
            "positive_missing_skip_rows": int(positive_missing_skip_by_benchmark.get(benchmark, 0)),
            "retained_fraction": retained / int(source_count) if int(source_count) > 0 else 0.0,
        }

    return rows, {
        "source_rows": len(source_rows_list),
        "stage4_rows": len(rows),
        "stage4_retained_fraction": len(rows) / len(source_rows_list) if source_rows_list else 0.0,
        "positive_injected_rows": injected,
        "positive_missing_skip_rows": int(sum(positive_missing_skip_by_benchmark.values())),
        "positive_missing_skip_by_benchmark": dict(sorted(positive_missing_skip_by_benchmark.items())),
        "skip_distribution_by_benchmark": skip_distribution_by_benchmark,
        "candidate_source": (
            "declared_legal_full_skill_pool"
            if next_skill_pool_mode == "full_pool"
            else "stage0_topm_online"
            if handoff_enabled
            else "explicit_or_random_fallback"
        ),
        "static_candidate_source": "stage0_topm_online" if handoff_enabled else "explicit_or_random_fallback",
        "next_skill_pool_mode": next_skill_pool_mode,
        "require_next_state_text": bool(require_next_state_text),
        "stage0_candidate_rows": stage0_candidate_rows,
        "stage0_next_static_hit_rows": stage0_next_static_hit_rows,
        "stage0_next_static_miss_retained_rows": stage0_next_static_miss_retained_rows,
        "empty_natural_candidate_rows": empty_natural_candidate_rows,
        "explicit_candidate_rows": explicit_candidate_rows,
        "random_negative_fill_rows": random_negative_fill_rows,
        "candidate_count_requested": candidate_count,
        "stage0_candidate_filter": dict(sorted(stage0_candidate_filter_audit.items())),
        "benchmark_filter_enabled": allowed_benchmarks is not None,
        "allowed_benchmarks": allowed_report,
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "stage0_candidate_handoff": stage0_candidate_handoff_report or {"enabled": False},
        "skipped_reasons": dict(skipped),
    }


def build_stage4_next_skill_rows(
    trajectories_path: str | Path,
    skill_id_to_idx: dict[str, int],
    *,
    candidate_count: int | None = None,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    benchmark_caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_source_rows = list(_read_jsonl_stream(trajectories_path))
    prepared_source_rows, causal_next_state_report = _attach_adjacent_next_states(raw_source_rows)
    source_rows, source_report = _eligible_stage4_source_rows(
        prepared_source_rows,
        skill_id_to_idx,
        allowed_benchmarks=allowed_benchmarks,
        max_source_rows=None if benchmark_caps is not None else max_rows,
    )
    source_rows, benchmark_caps_report = _cap_rows_by_benchmark(source_rows, benchmark_caps)
    rows, report = _build_stage4_next_skill_rows_from_source_rows(
        source_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
        max_rows=max_rows,
        allowed_benchmarks=allowed_benchmarks,
    )
    report = {
        **report,
        "source_path": str(trajectories_path),
        "source_rows": source_report["source_rows"],
        "eligible_source_rows": source_report["eligible_source_rows"],
        "skipped_benchmarks": source_report["skipped_benchmarks"],
        "skipped_reasons": dict(Counter(source_report["skipped_reasons"]) + Counter(report["skipped_reasons"])),
        "benchmark_caps": benchmark_caps_report,
        "causal_next_state": causal_next_state_report,
    }
    return rows, report


def _stage4_handoff_source_limit(max_rows: int | None, sample_multiplier: int) -> int | None:
    if max_rows is None:
        return None
    return max(1, int(max_rows)) * max(1, int(sample_multiplier))


def _as_stage4_handoff_row(
    row: dict[str, Any],
    *,
    next_skill_pool_mode: str,
) -> dict[str, Any]:
    copied = dict(row)
    original_loss_mask = row.get("loss_mask") or {}
    loss_mask = dict(original_loss_mask) if isinstance(original_loss_mask, dict) else {}
    if next_skill_pool_mode == "stage0_candidates":
        loss_mask["routing"] = False
    loss_mask["L_trans_skill_ce"] = True
    copied["loss_mask"] = loss_mask
    if original_loss_mask:
        copied["stage4_original_loss_mask"] = original_loss_mask
    return copied


def _pad_candidate_indices(rows: list[dict[str, Any]], device: torch.device) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    max_width = max(len(row["candidate_next_skill_indices"]) for row in rows)
    padded: list[list[int]] = []
    masks: list[list[bool]] = []
    labels: list[int] = []
    for row in rows:
        candidates = [int(idx) for idx in row["candidate_next_skill_indices"]]
        if not candidates:
            raise ValueError(f"Stage4 ACT candidate row has empty candidates: task_id={row.get('task_id')}")
        positive_idx = int(row["positive_next_skill_idx"])
        if positive_idx not in candidates:
            raise ValueError(
                "Stage4 ACT candidate row missing positive next skill: "
                f"task_id={row.get('task_id')} next_skill_id={row.get('next_skill_id')}"
            )
        positive_pos = candidates.index(positive_idx)
        if "positive_next_skill_position" in row and int(row["positive_next_skill_position"]) != positive_pos:
            raise ValueError(
                "Stage4 ACT row has stale positive next skill position: "
                f"task_id={row.get('task_id')} expected={positive_pos} "
                f"actual={row.get('positive_next_skill_position')}"
            )
        label = positive_pos
        if len(candidates) < max_width:
            candidates = candidates + [candidates[-1]] * (max_width - len(candidates))
            mask = [True] * len(row["candidate_next_skill_indices"]) + [False] * (max_width - len(row["candidate_next_skill_indices"]))
        else:
            mask = [True] * len(candidates)
        padded.append(candidates)
        masks.append(mask)
        labels.append(label)
    return padded, torch.tensor(masks, dtype=torch.bool, device=device), torch.tensor(labels, dtype=torch.long, device=device)


def _pad_candidate_prior_scores(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    padded: list[list[float]] = []
    for row in rows:
        candidate_count = len(row["candidate_next_skill_indices"])
        raw_scores = row.get("candidate_next_prior_scores")
        if raw_scores is None:
            scores = _candidate_rank_prior_scores(candidate_count)
        else:
            scores = [float(item) for item in raw_scores]
            if len(scores) != candidate_count:
                raise ValueError(
                    "Stage4 ACT row has stale candidate prior scores: "
                    f"task_id={row.get('task_id')} expected={candidate_count} actual={len(scores)}"
                )
        if len(scores) < int(width):
            scores = scores + [0.0] * (int(width) - len(scores))
        padded.append(scores[: int(width)])
    return torch.tensor(padded, dtype=torch.float32, device=device)


def _pad_candidate_online_memory_scores(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    padded: list[list[float]] = []
    for row in rows:
        candidate_count = len(row["candidate_next_skill_indices"])
        raw_scores = row.get("candidate_next_online_memory_scores")
        if raw_scores is None:
            scores = [0.0] * candidate_count
        else:
            scores = [float(item) for item in raw_scores]
            if len(scores) != candidate_count:
                raise ValueError(
                    "Stage4 ACT row has stale online memory scores: "
                    f"task_id={row.get('task_id')} expected={candidate_count} actual={len(scores)}"
                )
        if len(scores) < int(width):
            scores = scores + [0.0] * (int(width) - len(scores))
        padded.append(scores[: int(width)])
    return torch.tensor(padded, dtype=torch.float32, device=device)


def _stage4_skill_id_to_idx(rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in rows:
        for skill_key, idx_key in (
            ("skill_id", "skill_idx"),
            ("next_skill_id", "positive_next_skill_idx"),
        ):
            skill_id = str(row.get(skill_key) or "").strip()
            if skill_id and row.get(idx_key) is not None:
                mapping[skill_id] = int(row[idx_key])
        candidate_ids = list(row.get("candidate_next_skill_ids") or [])
        candidate_indices = list(row.get("candidate_next_skill_indices") or [])
        for skill_id, idx in zip(candidate_ids, candidate_indices):
            skill_text = str(skill_id or "").strip()
            if skill_text:
                mapping[skill_text] = int(idx)
        prefix = row.get("replay_prefix")
        if isinstance(prefix, list):
            for step in prefix:
                if not isinstance(step, dict):
                    continue
                skill_id = str(step.get("skill_id") or "").strip()
                if skill_id and step.get("skill_idx") is not None:
                    mapping[skill_id] = int(step["skill_idx"])
    return mapping


def _first_floating_parameter(module: Any) -> torch.nn.Parameter | None:
    if not hasattr(module, "parameters"):
        return None
    for param in module.parameters():
        if torch.is_floating_point(param):
            return param
    return None


def _ensure_stage4_score_calibrator(model: Any) -> torch.nn.Module:
    """Attach a zero-initialized residual score calibrator if one is missing."""

    feature_count = len(STAGE4_SCORE_CALIBRATOR_FEATURE_NAMES)
    existing = getattr(model, STAGE4_SCORE_CALIBRATOR_NAME, None)
    if existing is not None:
        in_features = getattr(existing, "in_features", feature_count)
        if int(in_features) != feature_count:
            raise ValueError(
                "Stage4 score calibrator has incompatible feature count: "
                f"expected={feature_count} actual={in_features}"
            )
        return existing

    calibrator = torch.nn.Linear(feature_count, 1, bias=False)
    torch.nn.init.zeros_(calibrator.weight)
    ref_param = _first_floating_parameter(model)
    if ref_param is not None:
        calibrator.to(device=ref_param.device, dtype=ref_param.dtype)
    setattr(model, STAGE4_SCORE_CALIBRATOR_NAME, calibrator)
    return calibrator


def _stage4_score_calibrator_features(
    *,
    prior_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    online_memory_logits: torch.Tensor,
    transition_residual_lambda: float,
    online_memory_weight: float,
) -> torch.Tensor:
    dtype = prior_logits.dtype
    prior_feature = prior_logits.to(dtype=dtype)
    residual_feature = float(transition_residual_lambda) * residual_logits.to(dtype=dtype)
    memory_feature = float(online_memory_weight) * online_memory_logits.to(dtype=dtype)
    memory_hit = (memory_feature.abs() > 0.0).to(dtype=dtype)
    return torch.stack(
        [
            prior_feature,
            residual_feature,
            memory_feature,
            prior_feature * memory_feature,
            residual_feature * memory_feature,
            memory_hit,
        ],
        dim=-1,
    )


def _apply_stage4_score_calibrator(
    model: Any,
    logits: torch.Tensor,
    *,
    prior_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    online_memory_logits: torch.Tensor,
    transition_residual_lambda: float,
    online_memory_weight: float,
) -> tuple[torch.Tensor, bool, float]:
    calibrator = getattr(model, STAGE4_SCORE_CALIBRATOR_NAME, None)
    if calibrator is None:
        return logits, False, 0.0
    active_rows = (
        (online_memory_logits.abs().amax(dim=-1) > 0.0)
        & torch.tensor(float(online_memory_weight) != 0.0, device=online_memory_logits.device)
    )
    features = _stage4_score_calibrator_features(
        prior_logits=prior_logits,
        residual_logits=residual_logits,
        online_memory_logits=online_memory_logits,
        transition_residual_lambda=transition_residual_lambda,
        online_memory_weight=online_memory_weight,
    )
    ref_param = _first_floating_parameter(calibrator)
    if ref_param is not None:
        features = features.to(device=ref_param.device, dtype=ref_param.dtype)
    delta = calibrator(features).squeeze(-1).to(device=logits.device, dtype=logits.dtype)
    active_rows = active_rows.to(device=logits.device)
    delta = delta * active_rows.to(dtype=delta.dtype).unsqueeze(-1)
    active_fraction = float(active_rows.float().mean().detach().cpu().item()) if active_rows.numel() else 0.0
    return logits + delta, True, active_fraction


def _masked_ranking_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0:
        return {
            "stage4_next_skill_recall@1": 0.0,
            "stage4_next_skill_recall@5": 0.0,
            "stage4_next_skill_mrr": 0.0,
            "stage4_candidate_count": 0.0,
        }
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    order = torch.argsort(masked.detach(), dim=-1, descending=True)
    ranks: list[int] = []
    for row_idx, label in enumerate(labels.detach().cpu().tolist()):
        positions = (order[row_idx].detach().cpu() == int(label)).nonzero(as_tuple=False)
        ranks.append(int(positions[0].item()) + 1 if positions.numel() else masked.size(-1) + 1)
    candidate_count = float(mask.float().sum(dim=-1).mean().detach().cpu().item())
    return {
        "stage4_next_skill_recall@1": sum(1 for rank in ranks if rank <= 1) / len(ranks),
        "stage4_next_skill_recall@5": sum(1 for rank in ranks if rank <= 5) / len(ranks),
        "stage4_next_skill_mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "stage4_candidate_count": candidate_count,
    }


def _validate_stage4_full_skill_mapping(
    model: Any,
    skill_id_to_idx: dict[str, int] | None,
) -> dict[str, int]:
    if not isinstance(skill_id_to_idx, dict) or not skill_id_to_idx:
        raise ValueError("full-pool Stage4 requires the complete skill_id_to_idx mapping")
    normalized = {str(skill_id): int(idx) for skill_id, idx in skill_id_to_idx.items()}
    skill_table = getattr(model, "skill_table", None)
    embeddings = getattr(skill_table, "E", None)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("full-pool Stage4 requires model.skill_table.E")
    skill_count = int(embeddings.size(0))
    if len(normalized) != skill_count:
        raise ValueError("full-pool Stage4 skill mapping size must match model.skill_table.E")
    if sorted(normalized.values()) != list(range(skill_count)):
        raise ValueError("full-pool Stage4 skill mapping indices must be contiguous in declared order")
    declared_ids = [skill_id for skill_id, _idx in sorted(normalized.items(), key=lambda item: item[1])]
    model_skill_ids = getattr(skill_table, "skill_ids", None)
    if model_skill_ids is not None and [str(item) for item in model_skill_ids] != declared_ids:
        raise ValueError("full-pool Stage4 declared skill order must match the model skill table")
    return normalized


@dataclass(frozen=True)
class Stage4SafeRouteBatch:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    dynamic_full_logits: torch.Tensor
    static_full_logits: torch.Tensor
    safe_fused_full_logits: torch.Tensor
    positive_mask: torch.Tensor
    valid_mask: torch.Tensor
    dynamic_memory: torch.Tensor
    static_memory: torch.Tensor
    causal_update_count: torch.Tensor
    safe_alpha: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Stage4CMCRouteBatch:
    total_loss: torch.Tensor
    objective: CounterfactualMemoryCalibrationOutput
    raw_dynamic_full_logits: torch.Tensor
    dynamic_full_logits: torch.Tensor
    static_full_logits: torch.Tensor
    fused_full_logits: torch.Tensor
    positive_mask: torch.Tensor
    valid_mask: torch.Tensor
    dynamic_memory: torch.Tensor
    static_memory: torch.Tensor
    causal_update_count: torch.Tensor
    raw_alpha: torch.Tensor
    effective_alpha: torch.Tensor
    metrics: dict[str, Any]


@dataclass(frozen=True)
class Stage4CandidateAdmissionRouteBatch:
    total_loss: torch.Tensor
    objective: CandidateAdmissionResidualObjective
    scoring: CandidateAdmissionScoringOutput
    candidate_union: CandidateUnion
    candidate_positive_mask: torch.Tensor
    candidate_valid_mask: torch.Tensor
    dynamic_extra_mask: torch.Tensor
    static_full_logits: torch.Tensor
    dynamic_full_logits: torch.Tensor
    static_memory: torch.Tensor
    dynamic_memory: torch.Tensor
    causal_update_count: torch.Tensor
    positive_full_mask: torch.Tensor
    valid_full_mask: torch.Tensor
    metrics: dict[str, Any]


def _cmc_training_row_exclusion_reason(
    row: dict[str, Any],
    skill_id_to_idx: dict[str, int],
) -> str | None:
    required_text = (
        ("trajectory_id", "missing_trajectory_id"),
        ("state_text", "missing_state_text"),
        ("action_text", "missing_action_text"),
        ("next_observation_text", "missing_next_observation_text"),
        ("next_state_text", "missing_next_state_text"),
    )
    for key, reason in required_text:
        if not str(row.get(key) or "").strip():
            return reason
    if str(row.get("skill_id") or "") not in skill_id_to_idx:
        return "missing_current_skill"
    if str(row.get("next_skill_id") or "") not in skill_id_to_idx:
        return "missing_positive_next_skill"
    return None


def _build_stage4_cmc_route_batch(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    feature_update_count_cap: float = CMC_FEATURE_UPDATE_COUNT_CAP,
    feature_candidate_count_cap: float = CMC_FEATURE_CANDIDATE_COUNT_CAP,
) -> Stage4CMCRouteBatch:
    if not rows:
        raise ValueError("CMC Stage4 route batch requires rows")
    full_skill_mapping = _validate_stage4_full_skill_mapping(
        model,
        skill_id_to_idx,
    )
    exclusion_counts: Counter[str] = Counter()
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        reason = _cmc_training_row_exclusion_reason(row, full_skill_mapping)
        if reason is None:
            eligible_rows.append(row)
        else:
            exclusion_counts[reason] += 1
    if not eligible_rows:
        raise ValueError("CMC Stage4 batch has no trustworthy causal rows")
    rows = eligible_rows
    full_logits_fn = getattr(model, "unified_route_full_logits", None)
    if not callable(full_logits_fn):
        raise ValueError("CMC Stage4 requires model.unified_route_full_logits")
    adapter = getattr(model, "route_memory_residual_adapter", None)
    candidate_gate = getattr(model, "route_memory_candidate_utility_gate", None)
    if not callable(adapter) or not callable(candidate_gate):
        raise ValueError("CMC Stage4 requires residual adapter and candidate utility gate")
    skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
    if not isinstance(skill_embeddings, torch.Tensor) or skill_embeddings.ndim != 2:
        raise ValueError("CMC Stage4 requires model.skill_table.E")
    skill_count = int(skill_embeddings.size(0))

    with torch.no_grad():
        h = _batch_cached_or_encode(
            model,
            rows,
            "_state_embedding",
            "state_text",
            device,
            text_role=STATE_QUERY_ROLE,
        )
        memory = model.initial_belief(h)
        memory, replay_prefix_used_count = _apply_replay_prefix_beliefs(
            model,
            rows,
            memory,
            full_skill_mapping,
            skill_count,
            device,
            trainable=False,
        )
        observation_embeddings = _batch_cached_or_encode(
            model,
            rows,
            "_next_observation_embedding",
            "next_observation_text",
            device,
            text_role=TRANSITION_TEXT_ROLE,
        )
        action_embeddings = _batch_action_text_embedding_or_none(model, rows, device)
        current_skill_labels = torch.tensor(
            [full_skill_mapping[str(row["skill_id"])] for row in rows],
            dtype=torch.long,
            device=device,
        )
        h_next = _batch_cached_or_encode(
            model,
            rows,
            "_next_state_embedding",
            "next_state_text",
            device,
            text_role=STATE_QUERY_ROLE,
        )
        _predicted_memory, _observation_memory, dynamic_memory = _post_action_memory(
            model,
            m_t=memory,
            current_skill_labels=current_skill_labels,
            action_embeddings=action_embeddings,
            observation_embeddings=observation_embeddings,
            h_next=h_next,
        )
        static_memory = model.initial_belief(h_next)
        static_full = full_logits_fn(h_next, static_memory)
        raw_dynamic_full = full_logits_fn(h_next, dynamic_memory)

    expected_shape = (len(rows), skill_count)
    if (
        tuple(static_full.shape) != expected_shape
        or tuple(raw_dynamic_full.shape) != expected_shape
    ):
        raise ValueError("CMC Stage4 logits must match rows and declared skill table")
    positive_mask = full_pool_positive_mask(
        rows,
        full_skill_mapping,
        equivalent_skill_ids_by_skill_id,
        skill_count=skill_count,
        device=device,
    )
    legal_pool = legal_skill_pool_mask(
        rows,
        full_skill_mapping,
        skill_count=skill_count,
        device=device,
    )
    causal_update_count = torch.tensor(
        [1 + len(row.get("replay_prefix") or []) for row in rows],
        dtype=raw_dynamic_full.dtype,
        device=device,
    )
    cmc_scoring = score_cmc_candidates(
        model,
        h=h_next,
        static_memory=static_memory,
        dynamic_memory=dynamic_memory,
        static_logits=static_full,
        raw_dynamic_logits=raw_dynamic_full,
        candidate_embeddings=skill_embeddings,
        valid_mask=legal_pool.mask,
        causal_update_count=causal_update_count,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
    )
    dynamic_full = cmc_scoring.dynamic_logits
    features = cmc_scoring.features
    raw_alpha = cmc_scoring.raw_alpha
    effective_alpha = cmc_scoring.effective_alpha
    objective = counterfactual_memory_calibration_loss(
        static_logits=static_full,
        dynamic_logits=dynamic_full,
        alpha=effective_alpha,
        positive_mask=positive_mask,
        valid_mask=legal_pool.mask,
    )
    floor = torch.finfo(dynamic_full.dtype).min
    valid_positive = positive_mask & legal_pool.mask
    labels = valid_positive.to(torch.float32).argmax(dim=-1).to(torch.long)
    static_ranking = _ranking_metrics_from_logits(
        objective.static_logits.masked_fill(~legal_pool.mask, floor),
        labels,
        rows=rows,
        positive_mask=valid_positive,
    )
    dynamic_ranking = _ranking_metrics_from_logits(
        objective.dynamic_logits.masked_fill(~legal_pool.mask, floor),
        labels,
        rows=rows,
        positive_mask=valid_positive,
    )
    fused_ranking = _ranking_metrics_from_logits(
        objective.fused_logits.masked_fill(~legal_pool.mask, floor),
        labels,
        rows=rows,
        positive_mask=valid_positive,
    )
    metrics = {
        "stage4_method": COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
        "stage4_act_count": float(len(rows)),
        "stage4_cmc_input_rows": float(len(rows) + sum(exclusion_counts.values())),
        "stage4_cmc_eligible_rows": float(len(rows)),
        "stage4_cmc_excluded_rows": float(sum(exclusion_counts.values())),
        "stage4_cmc_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "stage4_cmc_dynamic_loss": float(objective.dynamic_loss.detach().cpu().item()),
        "stage4_cmc_fused_loss": float(objective.fused_loss.detach().cpu().item()),
        "stage4_cmc_no_regret_loss": float(
            objective.no_regret_loss.detach().cpu().item()
        ),
        "stage4_total_loss": float(objective.loss.detach().cpu().item()),
        "stage4_cmc_alpha_mean": float(effective_alpha.mean().detach().cpu().item()),
        "stage4_cmc_alpha_min": float(effective_alpha.min().detach().cpu().item()),
        "stage4_cmc_alpha_max": float(effective_alpha.max().detach().cpu().item()),
        "stage4_replay_prefix_used_count": float(replay_prefix_used_count),
        "stage4_replay_prefix_trainable_enabled": False,
        "stage4_post_action_update_rows": float(len(rows)),
        "stage4_next_state_rows": float(len(rows)),
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "next_skill_pool_mode": "full_pool",
        "transition_skill_head_type": "cmc_unified_memory_retriever_full_pool",
        "stage4_cmc_static_mrr": float(static_ranking["transition_skill_mrr"]),
        "stage4_cmc_dynamic_mrr": float(dynamic_ranking["transition_skill_mrr"]),
        "stage4_cmc_fused_mrr": float(fused_ranking["transition_skill_mrr"]),
    }
    return Stage4CMCRouteBatch(
        total_loss=objective.loss,
        objective=objective,
        raw_dynamic_full_logits=raw_dynamic_full,
        dynamic_full_logits=dynamic_full,
        static_full_logits=static_full,
        fused_full_logits=objective.fused_logits,
        positive_mask=positive_mask,
        valid_mask=legal_pool.mask,
        dynamic_memory=dynamic_memory,
        static_memory=static_memory,
        causal_update_count=causal_update_count,
        raw_alpha=raw_alpha,
        effective_alpha=effective_alpha,
        metrics=metrics,
    )


def _candidate_union_local_tensors(
    candidate_union: CandidateUnion,
    *,
    static_full: torch.Tensor,
    dynamic_full: torch.Tensor,
    positive_full: torch.Tensor,
    skill_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_rows = candidate_union.candidate_rows
    width = max((len(row) for row in candidate_rows), default=0)
    if width <= 0 or any(not row for row in candidate_rows):
        raise ValueError("candidate-admission union requires candidates in every row")
    batch_size = len(candidate_rows)
    candidate_ids = torch.zeros(
        (batch_size, width),
        dtype=torch.long,
        device=static_full.device,
    )
    valid = torch.zeros_like(candidate_ids, dtype=torch.bool)
    skill_count = int(static_full.size(1))
    for row_idx, row in enumerate(candidate_rows):
        normalized = [int(item) for item in row]
        if any(item < 0 or item >= skill_count for item in normalized):
            raise ValueError("candidate-admission union contains an invalid skill index")
        row_ids = torch.tensor(
            normalized,
            dtype=torch.long,
            device=static_full.device,
        )
        candidate_ids[row_idx, : len(normalized)] = row_ids
        valid[row_idx, : len(normalized)] = True
    static_local = static_full.gather(1, candidate_ids)
    dynamic_local = dynamic_full.gather(1, candidate_ids)
    positive_local = positive_full.gather(1, candidate_ids) & valid
    embeddings = skill_embeddings.detach().to(
        device=static_full.device,
        dtype=static_full.dtype,
    )
    candidate_embeddings = embeddings.index_select(
        0,
        candidate_ids.reshape(-1),
    ).reshape(batch_size, width, -1)
    return static_local, dynamic_local, positive_local, valid, candidate_embeddings


def _candidate_admission_average_precision(
    admission_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> float:
    eligible = eligible_mask.to(device=admission_logits.device, dtype=torch.bool)
    labels = positive_mask.to(device=admission_logits.device, dtype=torch.bool)[eligible]
    scores = admission_logits.detach()[eligible]
    positive_count = int(labels.sum().detach().cpu().item())
    if positive_count <= 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_labels = labels.index_select(0, order).to(torch.float32)
    precision = ordered_labels.cumsum(dim=0) / torch.arange(
        1,
        int(ordered_labels.numel()) + 1,
        dtype=torch.float32,
        device=ordered_labels.device,
    )
    return float(
        (precision * ordered_labels).sum().div(float(positive_count)).cpu().item()
    )


def _validate_candidate_admission_training_support(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    values: list[float] = []
    for metrics in history:
        value = float(metrics.get("stage4_candidate_admission_positive_count", 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "candidate-admission training support count must be finite and nonnegative"
            )
        values.append(value)
    positive_count = float(sum(values))
    if positive_count <= 0.0:
        raise ValueError(
            "candidate-admission complete training schedule has no dynamic-extra positive"
        )
    return {
        "status": "ok",
        "scheduled_batch_count": len(history),
        "dynamic_extra_positive_count": positive_count,
    }


def _build_stage4_candidate_admission_route_batch(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    admission_weight: float = 1.0,
    counterfactual_weight: float = 1.0,
    gain_margin: float = 0.1,
) -> Stage4CandidateAdmissionRouteBatch:
    if not rows:
        raise ValueError("candidate-admission Stage4 route batch requires rows")
    if any(bool(row.get("positive_injected")) for row in rows):
        raise ValueError("candidate-admission Stage4 forbids positive injection")
    full_skill_mapping = _validate_stage4_full_skill_mapping(
        model,
        skill_id_to_idx,
    )
    exclusion_counts: Counter[str] = Counter()
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        reason = _cmc_training_row_exclusion_reason(row, full_skill_mapping)
        if reason is None:
            eligible_rows.append(row)
        else:
            exclusion_counts[reason] += 1
    if not eligible_rows:
        raise ValueError("candidate-admission Stage4 batch has no trustworthy causal rows")
    rows = eligible_rows
    full_logits_fn = getattr(model, "unified_route_full_logits", None)
    adapter = getattr(model, "route_memory_residual_adapter", None)
    head = getattr(model, "route_memory_candidate_admission_residual", None)
    skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
    if not callable(full_logits_fn):
        raise ValueError(
            "candidate-admission Stage4 requires model.unified_route_full_logits"
        )
    if not callable(adapter) or not callable(head):
        raise ValueError(
            "candidate-admission Stage4 requires frozen CMC adapter and candidate head"
        )
    if not isinstance(skill_embeddings, torch.Tensor) or skill_embeddings.ndim != 2:
        raise ValueError("candidate-admission Stage4 requires model.skill_table.E")
    skill_count = int(skill_embeddings.size(0))

    with torch.no_grad():
        h = _batch_cached_or_encode(
            model,
            rows,
            "_state_embedding",
            "state_text",
            device,
            text_role=STATE_QUERY_ROLE,
        )
        memory = model.initial_belief(h)
        memory, replay_prefix_used_count = _apply_replay_prefix_beliefs(
            model,
            rows,
            memory,
            full_skill_mapping,
            skill_count,
            device,
            trainable=False,
        )
        observation_embeddings = _batch_cached_or_encode(
            model,
            rows,
            "_next_observation_embedding",
            "next_observation_text",
            device,
            text_role=TRANSITION_TEXT_ROLE,
        )
        action_embeddings = _batch_action_text_embedding_or_none(model, rows, device)
        current_skill_labels = torch.tensor(
            [full_skill_mapping[str(row["skill_id"])] for row in rows],
            dtype=torch.long,
            device=device,
        )
        h_next = _batch_cached_or_encode(
            model,
            rows,
            "_next_state_embedding",
            "next_state_text",
            device,
            text_role=STATE_QUERY_ROLE,
        )
        _predicted_memory, _observation_memory, dynamic_memory = _post_action_memory(
            model,
            m_t=memory,
            current_skill_labels=current_skill_labels,
            action_embeddings=action_embeddings,
            observation_embeddings=observation_embeddings,
            h_next=h_next,
        )
        static_memory = model.initial_belief(h_next)
        static_full = full_logits_fn(h_next, static_memory)
        raw_dynamic_full = full_logits_fn(h_next, dynamic_memory)
        route_residual = adapter(h_next, dynamic_memory - static_memory)
        dynamic_full = raw_dynamic_full + route_residual @ skill_embeddings.detach().to(
            device=raw_dynamic_full.device,
            dtype=raw_dynamic_full.dtype,
        ).t()

    expected_shape = (len(rows), skill_count)
    if tuple(static_full.shape) != expected_shape or tuple(dynamic_full.shape) != expected_shape:
        raise ValueError(
            "candidate-admission Stage4 logits must match rows and declared skill table"
        )
    positive_full = full_pool_positive_mask(
        rows,
        full_skill_mapping,
        equivalent_skill_ids_by_skill_id,
        skill_count=skill_count,
        device=device,
    )
    legal_pool = legal_skill_pool_mask(
        rows,
        full_skill_mapping,
        skill_count=skill_count,
        device=device,
    )
    candidate_union = build_static_dynamic_union(
        static_full,
        dynamic_full,
        legal_pool.mask,
        static_k=int(static_k),
        dynamic_extra_k=int(dynamic_extra_k),
    )
    (
        static_local,
        dynamic_local,
        positive_local,
        candidate_valid,
        candidate_embeddings,
    ) = _candidate_union_local_tensors(
        candidate_union,
        static_full=static_full,
        dynamic_full=dynamic_full,
        positive_full=positive_full & legal_pool.mask,
        skill_embeddings=skill_embeddings,
    )
    dynamic_extra_mask = candidate_provenance_mask(
        candidate_union.candidate_rows,
        candidate_union.dynamic_extra_rows,
        candidate_valid,
    )
    causal_update_count = torch.tensor(
        [1 + len(row.get("replay_prefix") or []) for row in rows],
        dtype=static_local.dtype,
        device=device,
    )
    scoring = score_candidate_admission_residual(
        head,
        h=h_next,
        memory_delta=dynamic_memory - static_memory,
        candidate_embeddings=candidate_embeddings,
        static_logits=static_local,
        dynamic_logits=dynamic_local,
        dynamic_extra_mask=dynamic_extra_mask,
        valid_mask=candidate_valid,
        causal_update_count=causal_update_count,
    )
    objective = candidate_admission_residual_loss(
        scoring=scoring,
        positive_mask=positive_local,
        valid_mask=candidate_valid,
        dynamic_extra_mask=dynamic_extra_mask,
        admission_weight=float(admission_weight),
        counterfactual_weight=float(counterfactual_weight),
        gain_margin=float(gain_margin),
    )
    extra_eligible = dynamic_extra_mask & candidate_valid
    extra_positive = extra_eligible & positive_local
    shared = candidate_valid & ~dynamic_extra_mask
    rescue_rows = extra_positive.any(dim=-1) & ~(shared & positive_local).any(dim=-1)
    extra_count = int(extra_eligible.sum().detach().cpu().item())
    admission_prevalence = (
        float(objective.admission_positive_count) / float(extra_count)
        if extra_count > 0
        else 0.0
    )
    metrics = {
        "stage4_method": CANDIDATE_ADMISSION_RESIDUAL_V1,
        "stage4_act_count": float(len(rows)),
        "stage4_candidate_admission_input_rows": float(
            len(rows) + sum(exclusion_counts.values())
        ),
        "stage4_candidate_admission_excluded_rows": float(
            sum(exclusion_counts.values())
        ),
        "stage4_candidate_admission_exclusion_counts": dict(
            sorted(exclusion_counts.items())
        ),
        "stage4_candidate_admission_listwise_loss": float(
            objective.listwise_loss.detach().cpu().item()
        ),
        "stage4_candidate_admission_bce_loss": float(
            objective.admission_loss.detach().cpu().item()
        ),
        "stage4_candidate_admission_no_regret_loss": float(
            objective.no_regret_loss.detach().cpu().item()
        ),
        "stage4_candidate_admission_safety_loss": float(
            objective.safety_loss.detach().cpu().item()
        ),
        "stage4_candidate_admission_gain_loss": float(
            objective.gain_loss.detach().cpu().item()
        ),
        "stage4_candidate_admission_auprc": _candidate_admission_average_precision(
            scoring.admission_logits,
            positive_local,
            extra_eligible,
        ),
        "stage4_candidate_admission_prevalence": admission_prevalence,
        "stage4_candidate_admission_positive_count": float(
            objective.admission_positive_count
        ),
        "stage4_candidate_admission_negative_count": float(
            objective.admission_negative_count
        ),
        "stage4_candidate_admission_shared_positive_count": float(
            (shared & positive_local).sum().detach().cpu().item()
        ),
        "stage4_candidate_admission_shared_negative_count": float(
            (shared & ~positive_local).sum().detach().cpu().item()
        ),
        "stage4_candidate_admission_dynamic_rescue_rows": float(
            rescue_rows.sum().detach().cpu().item()
        ),
        "stage4_candidate_admission_positive_missing_union_rows": float(
            (~positive_local.any(dim=-1)).sum().detach().cpu().item()
        ),
        "stage4_total_loss": float(objective.loss.detach().cpu().item()),
        "stage4_candidate_count": float(
            candidate_valid.sum(dim=-1).to(torch.float32).mean().cpu().item()
        ),
        "stage4_replay_prefix_used_count": float(replay_prefix_used_count),
        "stage4_replay_prefix_trainable_enabled": False,
        "stage4_post_action_update_rows": float(len(rows)),
        "stage4_next_state_rows": float(len(rows)),
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "next_skill_pool_mode": "full_pool",
        "transition_skill_head_type": "candidate_admission_residual_full_pool",
    }
    return Stage4CandidateAdmissionRouteBatch(
        total_loss=objective.loss,
        objective=objective,
        scoring=scoring,
        candidate_union=candidate_union,
        candidate_positive_mask=positive_local,
        candidate_valid_mask=candidate_valid,
        dynamic_extra_mask=dynamic_extra_mask,
        static_full_logits=static_full,
        dynamic_full_logits=dynamic_full,
        static_memory=static_memory,
        dynamic_memory=dynamic_memory,
        causal_update_count=causal_update_count,
        positive_full_mask=positive_full,
        valid_full_mask=legal_pool.mask,
        metrics=metrics,
    )


def _build_stage4_safe_route_batch(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    skill_id_to_idx: dict[str, int],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    trainable_replay_prefix: bool,
    counterfactual_utility_weight: float,
    counterfactual_gain_margin: float,
    counterfactual_safety_tolerance: float,
    counterfactual_gain_weight: float,
    counterfactual_safety_weight: float,
    counterfactual_scale: float,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
) -> Stage4SafeRouteBatch:
    if not rows:
        raise ValueError("safe-memory Stage4 route batch requires rows")
    if any(bool(row.get("positive_injected")) for row in rows):
        raise ValueError("full-pool Stage4 forbids positive injection")
    if not callable(getattr(model, "initial_belief", None)):
        raise ValueError("safe-memory Stage4 requires model.initial_belief")
    full_logits_fn = getattr(model, "unified_route_full_logits", None)
    if not callable(full_logits_fn):
        raise ValueError("full-pool Stage4 requires model.unified_route_full_logits")

    skill_count = int(getattr(getattr(model, "skill_table", None), "E").size(0))
    full_skill_mapping = _validate_stage4_full_skill_mapping(
        model,
        skill_id_to_idx,
    )
    h = _batch_cached_or_encode(
        model,
        rows,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    memory = model.initial_belief(h)
    memory, replay_prefix_used_count = _apply_replay_prefix_beliefs(
        model,
        rows,
        memory,
        full_skill_mapping,
        skill_count,
        device,
        trainable=bool(trainable_replay_prefix),
    )
    observation_embeddings = _batch_cached_or_encode(
        model,
        rows,
        "_next_observation_embedding",
        "next_observation_text",
        device,
        text_role=TRANSITION_TEXT_ROLE,
    )
    action_embeddings = _batch_action_text_embedding_or_none(model, rows, device)
    current_skill_labels = torch.tensor(
        [int(row["skill_idx"]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    missing_next_state_rows = [
        row
        for row in rows
        if "_next_state_embedding" not in row
        and not str(row.get("next_state_text") or "").strip()
    ]
    if missing_next_state_rows:
        raise ValueError("causal unified Stage4 requires next_state_text")
    h_next = _batch_cached_or_encode(
        model,
        rows,
        "_next_state_embedding",
        "next_state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    _predicted_memory, _observation_memory, next_memory = _post_action_memory(
        model,
        m_t=memory,
        current_skill_labels=current_skill_labels,
        action_embeddings=action_embeddings,
        observation_embeddings=observation_embeddings,
        h_next=h_next,
    )
    dynamic_full = full_logits_fn(h_next, next_memory)
    with torch.no_grad():
        static_memory = model.initial_belief(h_next)
        static_full = full_logits_fn(h_next, static_memory)
    expected_shape = (len(rows), skill_count)
    if (
        tuple(dynamic_full.shape) != expected_shape
        or tuple(static_full.shape) != expected_shape
    ):
        raise ValueError(
            "full-pool Stage4 logits must match causal rows and declared skill table"
        )

    positive_mask = full_pool_positive_mask(
        rows,
        full_skill_mapping,
        equivalent_skill_ids_by_skill_id,
        skill_count=skill_count,
        device=device,
    )
    legal_pool = legal_skill_pool_mask(
        rows,
        full_skill_mapping,
        skill_count=skill_count,
        device=device,
    )
    causal_update_count = torch.tensor(
        [1 + len(row.get("replay_prefix") or []) for row in rows],
        dtype=dynamic_full.dtype,
        device=device,
    )
    objective = full_pool_causal_route_objective(
        dynamic_logits=dynamic_full,
        static_logits=static_full,
        positive_mask=positive_mask,
        valid_mask=legal_pool.mask,
        row_weights=torch.ones(
            len(rows),
            dtype=dynamic_full.dtype,
            device=device,
        ),
        gain_margin=counterfactual_gain_margin,
        safety_tolerance=counterfactual_safety_tolerance,
        gain_weight=counterfactual_gain_weight,
        safety_weight=counterfactual_safety_weight,
    )
    route_alpha_fn = getattr(model, "route_memory_alpha", None)
    if callable(route_alpha_fn):
        safe_alpha = route_alpha_fn(
            h_next,
            static_memory.detach(),
            next_memory,
            causal_update_count,
        )
        safe_gate_available = True
    else:
        safe_alpha = dynamic_full.new_ones(len(rows))
        safe_gate_available = False
    safe_fused_full = bounded_memory_fusion(
        static_full,
        dynamic_full,
        safe_alpha,
        legal_pool.mask,
        residual_bound=safe_memory_residual_bound,
    )
    explicit_row_mask = torch.tensor(
        [bool(explicit_inventory_skill_ids_ordered(row)) for row in rows],
        dtype=torch.bool,
        device=device,
    )
    local_masks = build_local_candidate_masks(
        static_full,
        dynamic_full,
        positive_mask,
        legal_pool.mask,
        explicit_row_mask,
        candidate_sizes=safe_local_candidate_sizes,
    )
    local_objective = safe_local_route_objective(
        fused_logits=safe_fused_full,
        static_logits=static_full,
        positive_mask=positive_mask,
        candidate_masks=local_masks.masks,
        source_row_indices=local_masks.source_row_indices,
        row_weights=torch.ones(len(rows), dtype=dynamic_full.dtype, device=device),
        gain_margin=counterfactual_gain_margin,
        safety_tolerance=counterfactual_safety_tolerance,
    )
    scaled_counterfactual = local_objective.loss * counterfactual_scale
    weighted_counterfactual = (
        counterfactual_utility_weight * scaled_counterfactual
    )
    total_loss = objective.main_loss + weighted_counterfactual
    valid = legal_pool.mask
    legal_positive = positive_mask & valid
    eligible = objective.eligible_mask
    floor = torch.finfo(dynamic_full.dtype).min
    eligible_dynamic = dynamic_full.masked_fill(~valid, floor)[eligible]
    eligible_static = static_full.masked_fill(~valid, floor)[eligible]
    eligible_positive = legal_positive[eligible]
    eligible_rows = [
        row
        for row, keep in zip(rows, eligible.detach().cpu().tolist())
        if keep
    ]
    eligible_labels = (
        eligible_positive.to(torch.float32).argmax(dim=-1).to(torch.long)
        if bool(eligible.any())
        else torch.zeros(0, dtype=torch.long, device=device)
    )
    dynamic_ranking = _ranking_metrics_from_logits(
        eligible_dynamic,
        eligible_labels,
        rows=eligible_rows,
        positive_mask=eligible_positive,
    )
    static_ranking = _ranking_metrics_from_logits(
        eligible_static,
        eligible_labels,
        rows=eligible_rows,
        positive_mask=eligible_positive,
    )
    fused_ranking = _ranking_metrics_from_logits(
        safe_fused_full.masked_fill(~valid, floor)[eligible],
        eligible_labels,
        rows=eligible_rows,
        positive_mask=eligible_positive,
    )
    static_hits = 0
    for row, row_positive in zip(rows, positive_mask.detach().cpu()):
        natural = {
            int(idx)
            for idx in row.get("candidate_next_skill_indices") or []
            if 0 <= int(idx) < skill_count
        }
        positive_indices = set(
            row_positive.nonzero(as_tuple=False).view(-1).tolist()
        )
        static_hits += int(bool(natural.intersection(positive_indices)))
    metrics = {
        "stage4_act_loss": float(objective.main_loss.detach().cpu().item()),
        "stage4_total_loss": float(total_loss.detach().cpu().item()),
        "stage4_act_count": float(len(rows)),
        "stage4_full_pool_rows": float(len(rows)),
        "stage4_full_pool_eligible_rows": float(
            objective.exclusion_counts["eligible_rows"]
        ),
        "next_skill_pool_mode": "full_pool",
        "transition_scoring_mode": UNIFIED_MEMORY_ROUTE_SCORER,
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "uses_stage0_prior_at_inference": False,
        "transition_residual_lambda": 0.0,
        "online_memory_weight": 0.0,
        "transition_skill_head_type": "unified_memory_retriever_full_pool",
        "stage4_score_calibrator_enabled": False,
        "stage4_score_calibrator_active_row_fraction": 0.0,
        "stage4_score_calibrator_feature_count": len(
            STAGE4_SCORE_CALIBRATOR_FEATURE_NAMES
        ),
        "stage4_recurrent_belief_memory_enabled": True,
        "stage4_replay_prefix_used_count": float(replay_prefix_used_count),
        "stage4_replay_prefix_trainable_used_count": float(
            replay_prefix_used_count if trainable_replay_prefix else 0
        ),
        "stage4_replay_prefix_trainable_enabled": bool(
            trainable_replay_prefix
        ),
        "stage4_counterfactual_utility_weight": counterfactual_utility_weight,
        "stage4_counterfactual_utility_loss": float(
            objective.counterfactual.loss.detach().cpu().item()
        ),
        "stage4_counterfactual_training_objective": "safe_local_candidate_rank_v1",
        "stage4_counterfactual_scaled_loss": float(
            scaled_counterfactual.detach().cpu().item()
        ),
        "stage4_weighted_counterfactual_utility_loss": float(
            weighted_counterfactual.detach().cpu().item()
        ),
        "stage4_counterfactual_scale": counterfactual_scale,
        "stage4_counterfactual_gain_loss": float(
            objective.counterfactual.gain_loss.detach().cpu().item()
        ),
        "stage4_counterfactual_safety_loss": float(
            objective.counterfactual.safety_loss.detach().cpu().item()
        ),
        "stage4_counterfactual_weak_rows": float(
            objective.counterfactual.weak_count
        ),
        "stage4_counterfactual_strong_rows": float(
            objective.counterfactual.strong_count
        ),
        "stage4_counterfactual_eligible_rows": float(
            objective.counterfactual.weak_count
            + objective.counterfactual.strong_count
        ),
        "stage4_counterfactual_gain_violation_rows": float(
            objective.counterfactual.gain_violation_count
        ),
        "stage4_counterfactual_safety_violation_rows": float(
            objective.counterfactual.safety_violation_count
        ),
        "stage4_safe_memory_gate_available": bool(safe_gate_available),
        "stage4_safe_memory_residual_bound": float(safe_memory_residual_bound),
        "stage4_safe_memory_alpha_mean": float(safe_alpha.mean().detach().cpu().item()),
        "stage4_safe_memory_alpha_min": float(safe_alpha.min().detach().cpu().item()),
        "stage4_safe_memory_alpha_max": float(safe_alpha.max().detach().cpu().item()),
        "stage4_safe_local_candidate_sizes": list(safe_local_candidate_sizes),
        "stage4_safe_local_mask_count": float(local_masks.masks.size(0)),
        "stage4_safe_local_loss": float(local_objective.loss.detach().cpu().item()),
        "stage4_safe_local_nll_loss": float(local_objective.nll_loss.detach().cpu().item()),
        "stage4_safe_local_gain_loss": float(local_objective.gain_loss.detach().cpu().item()),
        "stage4_safe_local_safety_loss": float(local_objective.safety_loss.detach().cpu().item()),
        "stage4_safe_local_eligible_masks": float(local_objective.eligible_count),
        "stage4_safe_local_gain_violation_masks": float(
            local_objective.gain_violation_count
        ),
        "stage4_safe_local_safety_violation_masks": float(
            local_objective.safety_violation_count
        ),
        "stage4_static_hit_train_rows": float(static_hits),
        "stage4_static_miss_train_rows": float(len(rows) - static_hits),
        "positive_not_in_skill_table_rows": float(
            (~positive_mask.any(dim=-1)).sum().detach().cpu().item()
        ),
        "positive_outside_legal_pool_rows": float(
            (
                positive_mask.any(dim=-1)
                & ~(positive_mask & valid).any(dim=-1)
            )
            .sum()
            .detach()
            .cpu()
            .item()
        ),
        "explicit_inventory_no_known_skill_rows": float(
            legal_pool.explicit_inventory_no_known_skill_mask.sum()
            .detach()
            .cpu()
            .item()
        ),
        "no_legal_negative_rows": float(
            objective.exclusion_counts["no_valid_negative_rows"]
        ),
        "nonfinite_logit_rows": float(
            objective.exclusion_counts["nonfinite_logit_rows"]
        ),
        "stage4_explicit_inventory_rows": float(
            sum(bool(explicit_inventory_skill_ids_ordered(row)) for row in rows)
        ),
        "stage4_post_action_update_rows": float(len(rows)),
        "stage4_next_state_rows": float(len(rows)),
        "stage4_next_skill_recall@1": fused_ranking[
            "transition_skill_recall@1"
        ],
        "stage4_next_skill_recall@5": fused_ranking[
            "transition_skill_recall@5"
        ],
        "stage4_next_skill_mrr": fused_ranking["transition_skill_mrr"],
        "stage4_candidate_count": float(
            valid.sum(dim=-1).float().mean().detach().cpu().item()
        ),
        "stage4_unified_dynamic_next_skill_recall@1": dynamic_ranking[
            "transition_skill_recall@1"
        ],
        "stage4_unified_dynamic_next_skill_recall@5": dynamic_ranking[
            "transition_skill_recall@5"
        ],
        "stage4_unified_dynamic_next_skill_mrr": dynamic_ranking[
            "transition_skill_mrr"
        ],
        "stage4_unified_safe_fused_next_skill_recall@1": fused_ranking[
            "transition_skill_recall@1"
        ],
        "stage4_unified_safe_fused_next_skill_recall@5": fused_ranking[
            "transition_skill_recall@5"
        ],
        "stage4_unified_safe_fused_next_skill_mrr": fused_ranking[
            "transition_skill_mrr"
        ],
        "stage4_unified_static_next_skill_recall@1": static_ranking[
            "transition_skill_recall@1"
        ],
        "stage4_unified_static_next_skill_recall@5": static_ranking[
            "transition_skill_recall@5"
        ],
        "stage4_unified_static_next_skill_mrr": static_ranking[
            "transition_skill_mrr"
        ],
    }
    metrics.update(
        _dynamic_static_rank_metrics(
            eligible_dynamic,
            eligible_static,
            eligible_labels,
            positive_mask=eligible_positive,
            prefix="stage4_unified",
        )
    )
    return Stage4SafeRouteBatch(
        total_loss=total_loss,
        main_loss=objective.main_loss,
        dynamic_full_logits=dynamic_full,
        static_full_logits=static_full,
        safe_fused_full_logits=safe_fused_full,
        positive_mask=positive_mask,
        valid_mask=valid,
        dynamic_memory=next_memory,
        static_memory=static_memory,
        causal_update_count=causal_update_count,
        safe_alpha=safe_alpha,
        metrics=metrics,
    )


def _compute_stage4_act_loss(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    *,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    online_memory_weight: float = 0.0,
    score_calibrator_enabled: bool = True,
    use_replay_prefix_belief: bool = True,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    next_skill_pool_mode: str = "stage0_candidates",
    skill_id_to_idx: dict[str, int] | None = None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
    counterfactual_utility_weight: float = 0.05,
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_scale: float = 1.0,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
    stage4_method: str = LEGACY_SAFE_MEMORY_STAGE4_V1,
    candidate_admission_static_k: int = 500,
    candidate_admission_dynamic_extra_k: int = 64,
    candidate_admission_weight: float = 1.0,
    candidate_admission_counterfactual_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not rows:
        param = next(model.parameters())
        return param.sum() * 0.0, {"stage4_act_count": 0.0}
    transition_residual_lambda = float(transition_residual_lambda)
    transition_scoring_mode = str(transition_scoring_mode or TRANSITION_SCORING_MODE)
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    next_skill_pool_mode = str(next_skill_pool_mode or "stage0_candidates")
    counterfactual_utility_weight = float(counterfactual_utility_weight)
    counterfactual_gain_margin = float(counterfactual_gain_margin)
    counterfactual_safety_tolerance = float(counterfactual_safety_tolerance)
    counterfactual_gain_weight = float(counterfactual_gain_weight)
    counterfactual_safety_weight = float(counterfactual_safety_weight)
    counterfactual_scale = float(counterfactual_scale)
    safe_memory_residual_bound = float(safe_memory_residual_bound)
    safe_local_candidate_sizes = tuple(int(size) for size in safe_local_candidate_sizes)
    stage4_method = str(stage4_method or LEGACY_SAFE_MEMORY_STAGE4_V1)
    if transition_scoring_mode not in STAGE4_TRANSITION_SCORING_MODES:
        raise ValueError(f"unsupported transition scoring mode: {transition_scoring_mode}")
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    if stage4_method not in STAGE4_METHODS:
        raise ValueError(f"unsupported stage4_method: {stage4_method}")
    if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1 and (
        route_scorer != UNIFIED_MEMORY_ROUTE_SCORER
        or next_skill_pool_mode != "full_pool"
    ):
        raise ValueError(
            "candidate-admission Stage4 requires unified-memory full-pool routing"
        )
    if next_skill_pool_mode == "full_pool" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("next_skill_pool_mode=full_pool requires route_scorer=unified_memory")
    _validate_counterfactual_hyperparameters(
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
    )
    if not math.isfinite(counterfactual_utility_weight) or counterfactual_utility_weight < 0.0:
        raise ValueError("counterfactual_utility_weight must be finite and nonnegative")
    if not 0.0 <= counterfactual_scale <= 1.0:
        raise ValueError("counterfactual_scale must be in [0, 1]")
    if not math.isfinite(safe_memory_residual_bound) or safe_memory_residual_bound <= 0.0:
        raise ValueError("safe_memory_residual_bound must be finite and positive")
    if not safe_local_candidate_sizes or any(size < 2 for size in safe_local_candidate_sizes):
        raise ValueError("safe_local_candidate_sizes must contain integers of at least two")
    effective_transition_scoring_mode = (
        UNIFIED_MEMORY_ROUTE_SCORER
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else transition_scoring_mode
    )
    effective_transition_residual_lambda = (
        0.0
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else transition_residual_lambda
    )
    if (
        route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        and next_skill_pool_mode == "full_pool"
    ):
        if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1:
            batch = _build_stage4_candidate_admission_route_batch(
                model,
                rows,
                device,
                skill_id_to_idx=skill_id_to_idx or {},
                equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                static_k=int(candidate_admission_static_k),
                dynamic_extra_k=int(candidate_admission_dynamic_extra_k),
                admission_weight=float(candidate_admission_weight),
                counterfactual_weight=float(
                    candidate_admission_counterfactual_weight
                ),
                gain_margin=counterfactual_gain_margin,
            )
            return batch.total_loss, batch.metrics
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1:
            batch = _build_stage4_cmc_route_batch(
                model,
                rows,
                device,
                skill_id_to_idx=skill_id_to_idx or {},
                equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            )
            return batch.total_loss, batch.metrics
        batch = _build_stage4_safe_route_batch(
            model,
            rows,
            device,
            skill_id_to_idx=skill_id_to_idx or {},
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            trainable_replay_prefix=trainable_replay_prefix,
            counterfactual_utility_weight=counterfactual_utility_weight,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            counterfactual_scale=counterfactual_scale,
            safe_memory_residual_bound=safe_memory_residual_bound,
            safe_local_candidate_sizes=safe_local_candidate_sizes,
        )
        return batch.total_loss, batch.metrics
    h = _batch_cached_or_encode(
        model,
        rows,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    skill_count = getattr(getattr(model, "skill_table", None), "E").size(0)
    full_skill_mapping = (
        _validate_stage4_full_skill_mapping(model, skill_id_to_idx)
        if next_skill_pool_mode == "full_pool"
        else None
    )
    _logits, m_obs = _skill_logits_and_memory(model, h, skill_count)
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        if not callable(getattr(model, "initial_belief", None)) or not callable(getattr(model, "unified_route_logits", None)):
            raise ValueError("route_scorer=unified_memory requires model.initial_belief and model.unified_route_logits")
        m_obs = model.initial_belief(h)
    m_static = m_obs
    replay_prefix_used_count = 0
    if use_replay_prefix_belief:
        m_obs, replay_prefix_used_count = _apply_replay_prefix_beliefs(
            model,
            rows,
            m_obs,
            full_skill_mapping or _stage4_skill_id_to_idx(rows),
            int(skill_count),
            device,
            trainable=bool(trainable_replay_prefix),
        )
    obs_emb = _batch_cached_or_encode(
        model,
        rows,
        "_next_observation_embedding",
        "next_observation_text",
        device,
        text_role=TRANSITION_TEXT_ROLE,
    )
    action_emb = _batch_action_text_embedding_or_none(model, rows, device)
    labels = torch.tensor([int(row["skill_idx"]) for row in rows], dtype=torch.long, device=device)
    route_h = h
    route_memory = m_obs
    route_static_memory = m_static
    post_action_update_rows = 0
    next_state_rows = 0
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        missing_next_state_rows = [
            row
            for row in rows
            if "_next_state_embedding" not in row and not str(row.get("next_state_text") or "").strip()
        ]
        if missing_next_state_rows:
            raise ValueError("causal unified Stage4 requires next_state_text")
        h_next = _batch_cached_or_encode(
            model,
            rows,
            "_next_state_embedding",
            "next_state_text",
            device,
            text_role=STATE_QUERY_ROLE,
        )
        _predicted_memory, _observation_memory, next_memory = _post_action_memory(
            model,
            m_t=m_obs,
            current_skill_labels=labels,
            action_embeddings=action_emb,
            observation_embeddings=obs_emb,
            h_next=h_next,
        )
        route_h = h_next
        route_memory = next_memory
        with torch.no_grad():
            route_static_memory = model.initial_belief(h_next)
        post_action_update_rows = len(rows)
        next_state_rows = len(rows)
    candidate_rows, mask, target_positions = _pad_candidate_indices(rows, device)
    candidate_ids = torch.tensor(candidate_rows, dtype=torch.long, device=device)
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        logits = model.unified_route_logits(route_h, route_memory, candidate_rows=candidate_rows)
        static_logits = model.unified_route_logits(route_h, route_static_memory, candidate_rows=candidate_rows)
        prior_logits = static_logits
        residual_logits = logits - static_logits
        online_memory_logits = torch.zeros_like(logits)
        head_type = "unified_memory_retriever"
        score_calibrator_applied = False
        score_calibrator_active_row_fraction = 0.0
    else:
        helper_scoring_mode = (
            TRANSITION_SCORING_MODE
            if transition_scoring_mode == STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
            else transition_scoring_mode
        )
        logits, head_type, prior_logits, residual_logits = _transition_candidate_logits_for_mode(
            model,
            h,
            m_obs,
            labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=mask,
            residual_lambda=transition_residual_lambda,
            scoring_mode=helper_scoring_mode,
        )
        if transition_scoring_mode == STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE:
            prior_logits = _pad_candidate_prior_scores(rows, candidate_ids.size(1), device).to(dtype=residual_logits.dtype)
            logits = prior_logits + transition_residual_lambda * residual_logits
            head_type = f"stage0_rank_prior+{head_type}"
        online_memory_logits = _pad_candidate_online_memory_scores(rows, candidate_ids.size(1), device).to(dtype=logits.dtype)
        if float(online_memory_weight) != 0.0:
            logits = logits + float(online_memory_weight) * online_memory_logits
        if bool(score_calibrator_enabled):
            logits, score_calibrator_applied, score_calibrator_active_row_fraction = _apply_stage4_score_calibrator(
                model,
                logits,
                prior_logits=prior_logits,
                residual_logits=residual_logits,
                online_memory_logits=online_memory_logits,
                transition_residual_lambda=transition_residual_lambda,
                online_memory_weight=online_memory_weight,
            )
        else:
            score_calibrator_applied = False
            score_calibrator_active_row_fraction = 0.0
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    ce_loss = F.cross_entropy(logits, target_positions)
    loss = ce_loss
    metrics = {
        "stage4_act_loss": float(ce_loss.detach().cpu().item()),
        "stage4_total_loss": float(loss.detach().cpu().item()),
        "stage4_act_count": float(len(rows)),
        "transition_scoring_mode": effective_transition_scoring_mode,
        "route_scorer": route_scorer,
        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_residual_lambda": effective_transition_residual_lambda,
        "online_memory_weight": float(online_memory_weight),
        "transition_skill_head_type": head_type,
        "stage4_score_calibrator_enabled": bool(score_calibrator_applied),
        "stage4_score_calibrator_active_row_fraction": score_calibrator_active_row_fraction,
        "stage4_score_calibrator_feature_count": len(STAGE4_SCORE_CALIBRATOR_FEATURE_NAMES),
        "stage4_recurrent_belief_memory_enabled": bool(use_replay_prefix_belief),
        "stage4_replay_prefix_used_count": float(replay_prefix_used_count),
        "stage4_replay_prefix_trainable_used_count": float(
            replay_prefix_used_count if trainable_replay_prefix else 0
        ),
        "stage4_replay_prefix_trainable_enabled": bool(trainable_replay_prefix),
        "next_skill_pool_mode": next_skill_pool_mode,
        "stage4_post_action_update_rows": float(post_action_update_rows),
        "stage4_next_state_rows": float(next_state_rows),
    }
    metrics.update(_masked_ranking_metrics(logits, target_positions, mask))
    comparison_metrics = _masked_ranking_metrics(
        prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min),
        target_positions,
        mask,
    )
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        metrics.update({f"stage4_unified_dynamic_{key.removeprefix('stage4_')}": value for key, value in _masked_ranking_metrics(logits, target_positions, mask).items()})
        metrics.update({f"stage4_unified_static_{key.removeprefix('stage4_')}": value for key, value in comparison_metrics.items()})
        metrics.update(
            _dynamic_static_rank_metrics(
                logits.masked_fill(~mask, torch.finfo(logits.dtype).min),
                prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min),
                target_positions,
                prefix="stage4_unified",
            )
        )
    else:
        residual_metrics = _masked_ranking_metrics(
            residual_logits.masked_fill(~mask, torch.finfo(residual_logits.dtype).min),
            target_positions,
            mask,
        )
        online_memory_metrics = _masked_ranking_metrics(
            online_memory_logits.masked_fill(~mask, torch.finfo(online_memory_logits.dtype).min),
            target_positions,
            mask,
        )
        metrics.update({f"stage4_prior_{key.removeprefix('stage4_')}": value for key, value in comparison_metrics.items()})
        metrics.update({f"stage4_residual_{key.removeprefix('stage4_')}": value for key, value in residual_metrics.items()})
        metrics.update(
            {f"stage4_online_memory_{key.removeprefix('stage4_')}": value for key, value in online_memory_metrics.items()}
        )
        metrics["stage4_unified_dynamic_vs_static_delta_mrr"] = 0.0
        metrics["stage4_unified_positive_rank_improved_rows"] = 0.0
        metrics["stage4_unified_positive_rank_worsened_rows"] = 0.0
        metrics["stage4_unified_argmax_changed_rows"] = 0.0
    return loss, metrics


def _batch_for_step(rows: list[dict[str, Any]], step_idx: int, batch_size: int) -> list[dict[str, Any]]:
    start = ((step_idx - 1) * max(1, int(batch_size))) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(max(1, int(batch_size)))]


def _stage4_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step <= warmup_steps:
        return max(0.0, min(1.0, float(step) / float(warmup_steps)))
    progress = min(
        1.0,
        float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _set_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(trainable)


def _freeze_for_stage4_act(
    model: Any,
    train_transition: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    stage4_method: str = LEGACY_SAFE_MEMORY_STAGE4_V1,
) -> dict[str, Any]:
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    stage4_method = str(stage4_method or LEGACY_SAFE_MEMORY_STAGE4_V1)
    if stage4_method not in STAGE4_METHODS:
        raise ValueError(f"unsupported stage4_method: {stage4_method}")
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad_(False)
    trainable_modules: list[str] = []
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1:
            return freeze_stage4_candidate_admission(model)
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1:
            return freeze_stage4_cmc(model)
        return freeze_stage4_safe_memory(model)
    else:
        for name in ("trans_head", "action_proj"):
            module = getattr(model, name, None)
            _set_trainable(module, True)
            if module is not None:
                trainable_modules.append(name)
        if train_transition:
            _set_trainable(getattr(model, "transition", None), True)
            if getattr(model, "transition", None) is not None:
                trainable_modules.append("transition")
        effective_train_transition = bool(train_transition)
    return {
        "frozen_routing_foundation": True,
        "frozen_belief_gate": bool(
            getattr(model, "gate", None) is not None and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER
        ),
        "train_transition": effective_train_transition,
        "trainable_modules": trainable_modules,
        "route_scorer": route_scorer,
    }


SAFE_STAGE4_BENCHMARKS = (
    "toolbench_g3",
    "traject_bench",
    "alfworld",
    "webshop",
)


def _safe_stage4_row_id(row: dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or "").strip()
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not benchmark or not trajectory_id or row.get("step_index") is None:
        raise ValueError(
            "safe-memory Stage4 rows require benchmark, trajectory_id, and step_index"
        )
    return str(
        row.get("row_id")
        or f"{benchmark}/{trajectory_id}/step-{int(row['step_index'])}"
    )


def _stable_safe_stage4_train_cap(
    rows: list[dict[str, Any]],
    *,
    benchmarks: tuple[str, ...],
    benchmark_caps: dict[str, int] | None,
    max_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    caps = dict(benchmark_caps or {})
    for benchmark in benchmarks:
        benchmark_rows = [
            dict(row)
            for row in rows
            if str(row.get("benchmark") or "") == benchmark
        ]
        ordered = sorted(
            benchmark_rows,
            key=lambda row: (
                canonical_digest(
                    ["stage4_train_cap_v1", seed, benchmark, _safe_stage4_row_id(row)]
                ),
                _safe_stage4_row_id(row),
            ),
        )
        cap = int(caps.get(benchmark, -1))
        retained = ordered if cap < 0 else ordered[:cap]
        counts[benchmark] = len(retained)
        selected.extend(retained)
    selected = sorted(
        selected,
        key=lambda row: (
            canonical_digest(
                [
                    "stage4_global_train_cap_v1",
                    seed,
                    str(row.get("benchmark") or ""),
                    _safe_stage4_row_id(row),
                ]
            ),
            _safe_stage4_row_id(row),
        ),
    )
    if max_rows is not None:
        selected = selected[: max(0, int(max_rows))]
        counts = {
            benchmark: sum(
                str(row.get("benchmark") or "") == benchmark for row in selected
            )
            for benchmark in benchmarks
        }
    missing = [benchmark for benchmark, count in counts.items() if count <= 0]
    if missing:
        raise ValueError(
            f"safe-memory Stage4 training cap removed benchmark: {missing[0]}"
        )
    return selected, {
        "strategy": "lowest_stable_hash_per_benchmark_v1",
        "seed": int(seed),
        "benchmark_caps": {
            benchmark: int(caps.get(benchmark, -1)) for benchmark in benchmarks
        },
        "max_rows": None if max_rows is None else int(max_rows),
        "capped_train_rows_by_benchmark": counts,
        "capped_train_row_ids": sorted(
            _safe_stage4_row_id(row) for row in selected
        ),
    }


def _train_stage4_safe_memory_with_model(
    *,
    model: Any,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    minimum_learning_rate: float,
    learning_rate_warmup_fraction: float,
    validation_fraction: float,
    validation_rows_per_benchmark: int,
    minimum_validation_rows_per_benchmark: int,
    gate_rows_per_benchmark: int,
    validation_interval_steps: int,
    resume_checkpoint_path: str | Path | None,
    seed: int,
    candidate_count: int | None,
    max_rows: int | None,
    counterfactual_utility_weight: float,
    counterfactual_gain_margin: float,
    counterfactual_safety_tolerance: float,
    counterfactual_gain_weight: float,
    counterfactual_safety_weight: float,
    counterfactual_warmup_fraction: float,
    safe_memory_residual_bound: float,
    safe_local_candidate_sizes: tuple[int, ...],
    allowed_benchmarks: set[str] | None,
    benchmark_caps: dict[str, int] | None,
    stage0_top_m: int | None,
    stage0_positive_missing_policy: str,
    stage0_handoff_query_mode: str,
    stage0_handoff_sample_multiplier: int,
    stage0_inventory_min_candidates: int,
    stage0_candidate_encode_batch_size: int,
    stage0_candidate_progress_interval_batches: int,
    stage0_handoff_cache_mode: str,
    stage0_handoff_cache_dir: str | Path,
    stage0_handoff_cache_format: str,
    stage0_handoff_cache_shard_size: int,
    routing_checkpoint_path: str | Path | None,
    checkpoint_init_report: dict[str, Any] | None,
    setup_status_path: str | Path | None,
    auto_replay_prefix_max_steps: int,
    trainable_replay_prefix: bool,
    stage4_method: str,
    base_cmc_checkpoint_sha256: str | None,
    candidate_admission_static_k: int,
    candidate_admission_dynamic_extra_k: int,
) -> dict[str, Any]:
    del stage0_handoff_sample_multiplier
    stage4_method = str(stage4_method or LEGACY_SAFE_MEMORY_STAGE4_V1)
    if stage4_method not in STAGE4_METHODS:
        raise ValueError(f"unsupported stage4_method: {stage4_method}")
    cmc_enabled = stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
    candidate_admission_enabled = stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1
    if (cmc_enabled or candidate_admission_enabled) and trainable_replay_prefix:
        raise ValueError(
            "CMC-derived Stage4 methods require frozen replay-prefix expansion"
        )
    if candidate_admission_enabled and not str(base_cmc_checkpoint_sha256 or ""):
        raise ValueError(
            "candidate-admission Stage4 requires base_cmc_checkpoint_sha256"
        )
    safe_memory_residual_bound = float(safe_memory_residual_bound)
    safe_local_candidate_sizes = tuple(int(size) for size in safe_local_candidate_sizes)
    if not math.isfinite(safe_memory_residual_bound) or safe_memory_residual_bound <= 0.0:
        raise ValueError("safe_memory_residual_bound must be finite and positive")
    if not safe_local_candidate_sizes or any(size < 2 for size in safe_local_candidate_sizes):
        raise ValueError("safe_local_candidate_sizes must contain integers of at least two")
    total_steps = max(1, int(max_steps))
    peak_learning_rate = float(learning_rate)
    minimum_lr = float(minimum_learning_rate)
    warmup_fraction = float(learning_rate_warmup_fraction)
    if (
        not math.isfinite(peak_learning_rate)
        or not math.isfinite(minimum_lr)
        or peak_learning_rate <= 0.0
        or minimum_lr <= 0.0
        or minimum_lr > peak_learning_rate
    ):
        raise ValueError(
            "safe-memory Stage4 learning rates must be finite, positive, and ordered"
        )
    if int(validation_interval_steps) <= 0:
        raise ValueError("validation_interval_steps must be positive")
    minimum_ratio = minimum_lr / peak_learning_rate
    _stage4_lr_multiplier(
        0,
        total_steps=total_steps,
        warmup_fraction=warmup_fraction,
        minimum_ratio=minimum_ratio,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    resume_dir = output / "resume"
    validation_dir = output / "validation"
    for directory in (checkpoint_dir, resume_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    setup_status = (
        Path(setup_status_path)
        if setup_status_path is not None
        else output / "setup_status.jsonl"
    )
    if setup_status_path is None:
        reset_setup_status(setup_status)

    skills = _read_jsonl(skills_path)
    append_setup_status(
        setup_status,
        "skills_loaded",
        skills_path=str(skills_path),
        skill_count=len(skills),
    )
    skill_id_to_idx = {
        str(row.get("skill_id") or row.get("canonical_skill_id")): index
        for index, row in enumerate(skills)
    }
    equivalent_skill_ids = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    _validate_stage4_full_skill_mapping(model, skill_id_to_idx)
    append_setup_status(setup_status, "model_moved_to_device", device=str(device))

    raw_source_rows = list(_read_jsonl_stream(trajectories_path))
    prepared_source_rows, causal_next_state_report = _attach_adjacent_next_states(
        raw_source_rows
    )
    append_setup_status(
        setup_status,
        "stage4_causal_next_state_prepared",
        **causal_next_state_report,
    )
    source_rows, source_report = _eligible_stage4_source_rows(
        prepared_source_rows,
        skill_id_to_idx,
        allowed_benchmarks=allowed_benchmarks,
        max_source_rows=None,
    )
    requested_benchmarks = set(
        allowed_benchmarks
        or {str(row.get("benchmark") or "") for row in source_rows}
    )
    benchmarks = tuple(
        benchmark
        for benchmark in SAFE_STAGE4_BENCHMARKS
        if benchmark in requested_benchmarks
    ) + tuple(sorted(requested_benchmarks.difference(SAFE_STAGE4_BENCHMARKS)))
    if not benchmarks:
        raise ValueError("safe-memory Stage4 requires declared benchmarks")
    source_file_identity = sha256_path(trajectories_path)
    source_data_identity = {
        benchmark: {
            "source_path": source_file_identity["path"],
            "source_sha256": source_file_identity["sha256"],
            "eligible_rows_sha256": canonical_digest(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if not str(key).startswith("_")
                    }
                    for row in source_rows
                    if str(row.get("benchmark") or "") == benchmark
                ]
            ),
        }
        for benchmark in benchmarks
    }
    resolved_prompt = resolve_state_query_prompt_contract(
        prompt_version="clstr_causal_state_v1",
        max_chars=2000,
        truncation="head_tail_v1",
    )
    prompt_contract = {
        "prompt_mode": resolved_prompt["state_query_prompt_version"],
        "prompt_sha256": canonical_digest(resolved_prompt),
        **resolved_prompt,
    }
    protocol = build_stage4_data_protocol(
        source_rows,
        expected_benchmarks=benchmarks,
        seed=int(seed),
        validation_fraction=float(validation_fraction),
        validation_rows_per_benchmark=int(validation_rows_per_benchmark),
        minimum_validation_rows_per_benchmark=int(
            minimum_validation_rows_per_benchmark
        ),
        gate_rows_per_benchmark=int(gate_rows_per_benchmark),
        source_data_identity=source_data_identity,
        prompt_contract=prompt_contract,
    )
    capped_train_source_rows, cap_report = _stable_safe_stage4_train_cap(
        protocol.train_rows,
        benchmarks=benchmarks,
        benchmark_caps=benchmark_caps,
        max_rows=max_rows,
        seed=int(seed),
    )
    data_protocol = dict(protocol.manifest)
    data_protocol.pop("manifest_sha256", None)
    data_protocol.update(cap_report)
    data_protocol["manifest_sha256"] = canonical_digest(data_protocol)
    data_protocol_path = output / "stage4_data_protocol.json"
    write_json(data_protocol_path, data_protocol)
    append_setup_status(
        setup_status,
        "stage4_data_protocol_created",
        data_protocol=data_protocol,
        data_protocol_path=str(data_protocol_path),
    )

    stage0_handoff_enabled = stage0_top_m is not None and int(stage0_top_m) > 0

    def prepare_partition(
        partition_name: str,
        partition_source_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        handoff_report: dict[str, Any] | None = None
        prepared_rows = [dict(row) for row in partition_source_rows]
        if stage0_handoff_enabled:
            handoff_rows = [
                _as_stage4_handoff_row(
                    row,
                    next_skill_pool_mode="full_pool",
                )
                for row in prepared_rows
            ]
            manifest_name = (
                "stage0_candidate_handoff.json"
                if partition_name == "train"
                else f"stage0_candidate_handoff_{partition_name}.json"
            )
            prepared_rows, handoff_report = (
                _prepare_stage0_topm_candidates_with_cache(
                    model,
                    handoff_rows,
                    skills,
                    skill_id_to_idx,
                    top_m=int(stage0_top_m),
                    positive_missing_policy=stage0_positive_missing_policy,
                    query_mode=stage0_handoff_query_mode,
                    allow_full_pool_stage2_debug=False,
                    routing_checkpoint_path=routing_checkpoint_path,
                    manifest_path=output / manifest_name,
                    encode_batch_size=stage0_candidate_encode_batch_size,
                    device=device,
                    setup_status_path=setup_status,
                    progress_interval_batches=(
                        stage0_candidate_progress_interval_batches
                    ),
                    inventory_min_candidates=stage0_inventory_min_candidates,
                    next_skill_pool_mode="full_pool",
                    cache_mode=stage0_handoff_cache_mode,
                    cache_dir=stage0_handoff_cache_dir,
                    cache_format=stage0_handoff_cache_format,
                    cache_shard_size=stage0_handoff_cache_shard_size,
                    route_scorer=UNIFIED_MEMORY_ROUTE_SCORER,
                )
            )
        stage4_rows, partition_report = (
            _build_stage4_next_skill_rows_from_source_rows(
                prepared_rows,
                skill_id_to_idx,
                candidate_count=candidate_count,
                max_rows=None,
                allowed_benchmarks=set(benchmarks),
                stage0_candidate_handoff_report=handoff_report,
                next_skill_pool_mode="full_pool",
            )
        )
        stage4_rows, replay_report = _attach_auto_replay_prefixes(
            stage4_rows,
            max_steps=auto_replay_prefix_max_steps,
        )
        for row in stage4_rows:
            row["benchmark"] = str(row.get("source_benchmark") or "")
            row["row_id"] = str(
                row.get("row_id")
                or f"{row['benchmark']}/{row['trajectory_id']}/step-{row['step_index']}"
            )
        partition_report = {
            **partition_report,
            "partition": partition_name,
            "auto_replay_prefix": replay_report,
            "trainable_replay_prefix": bool(trainable_replay_prefix),
        }
        return stage4_rows, partition_report, handoff_report

    train_rows, train_data_report, train_handoff_report = prepare_partition(
        "train",
        capped_train_source_rows,
    )
    validation_rows, validation_data_report, validation_handoff_report = (
        prepare_partition("validation", protocol.validation_rows)
    )
    gate_rows, gate_data_report, gate_handoff_report = prepare_partition(
        "gate",
        protocol.gate_rows,
    )
    cmc_optimizer_row_filter: dict[str, Any] | None = None
    if cmc_enabled or candidate_admission_enabled:
        retained_train_rows: list[dict[str, Any]] = []
        exclusion_counts: Counter[str] = Counter()
        for row in train_rows:
            reason = _cmc_training_row_exclusion_reason(row, skill_id_to_idx)
            if reason is None:
                retained_train_rows.append(row)
            else:
                exclusion_counts[reason] += 1
        cmc_optimizer_row_filter = {
            "input_rows": len(train_rows),
            "retained_rows": len(retained_train_rows),
            "excluded_rows": sum(exclusion_counts.values()),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "zero_history_policy": "validation_control_not_forced_into_optimizer",
        }
        train_rows = retained_train_rows
    if stage0_handoff_enabled:
        append_setup_status(
            setup_status,
            "stage0_candidate_handoff_prepared",
            train=train_handoff_report,
            validation=validation_handoff_report,
            gate=gate_handoff_report,
        )
    if not train_rows or not validation_rows or not gate_rows:
        raise ValueError("safe-memory Stage4 partitions must all contain route rows")
    data_report = {
        **train_data_report,
        "source_path": str(trajectories_path),
        "source_rows": source_report["source_rows"],
        "eligible_source_rows": source_report["eligible_source_rows"],
        "skipped_benchmarks": source_report["skipped_benchmarks"],
        "skipped_reasons": source_report["skipped_reasons"],
        "benchmark_caps": cap_report,
        "causal_next_state": causal_next_state_report,
        "validation_partition": validation_data_report,
        "gate_partition": gate_data_report,
        "data_protocol": data_protocol,
        "cmc_optimizer_row_filter": cmc_optimizer_row_filter,
    }
    append_setup_status(setup_status, "stage4_rows_built", data_report=data_report)

    freeze_report = (
        freeze_stage4_candidate_admission(model)
        if candidate_admission_enabled
        else freeze_stage4_cmc(model)
        if cmc_enabled
        else freeze_stage4_safe_memory(model)
    )
    append_setup_status(
        setup_status,
        "stage4_freeze_applied",
        freeze_report=freeze_report,
    )
    batcher = BenchmarkBalancedStage4Batcher(
        train_rows,
        benchmarks=benchmarks,
        batch_size=int(batch_size),
        seed=int(seed),
    )
    scheduled_train_rows_by_id: dict[str, dict[str, Any]] = {}
    scheduled_train_reference_count = 0
    for scheduled_step in range(1, total_steps + 1):
        for row in batcher.batch_for_step(scheduled_step):
            scheduled_train_reference_count += 1
            scheduled_train_rows_by_id.setdefault(
                _safe_stage4_row_id(row),
                row,
            )
    scheduled_train_rows = list(scheduled_train_rows_by_id.values())
    cache_batch_size = max(1, int(stage0_candidate_encode_batch_size))
    train_cache_report = _attach_full_base_embedding_cache(
        model,
        scheduled_train_rows,
        encode_batch_size=cache_batch_size,
    )
    validation_cache_report = _attach_full_base_embedding_cache(
        model,
        validation_rows,
        encode_batch_size=cache_batch_size,
    )
    gate_cache_report = _attach_full_base_embedding_cache(
        model,
        gate_rows,
        encode_batch_size=cache_batch_size,
    )
    frozen_embedding_cache = {
        "schema_version": "stage4_frozen_embedding_cache_v1",
        "mode": "scheduled_rows_v1",
        "backbone_frozen": True,
        "encode_batch_size": cache_batch_size,
        "train": {
            **train_cache_report,
            "source_row_count": len(train_rows),
            "scheduled_reference_count": scheduled_train_reference_count,
            "scheduled_unique_row_count": len(scheduled_train_rows),
            "scheduled_row_ids_sha256": canonical_digest(
                sorted(scheduled_train_rows_by_id)
            ),
        },
        "validation": validation_cache_report,
        "gate": gate_cache_report,
        "data_protocol_manifest_sha256": data_protocol["manifest_sha256"],
        "fast_router_digest": freeze_report["fast_router_digest"],
    }
    frozen_embedding_cache["cache_identity_sha256"] = canonical_digest(
        frozen_embedding_cache
    )
    append_setup_status(
        setup_status,
        "stage4_frozen_embedding_cache_prepared",
        frozen_embedding_cache=frozen_embedding_cache,
    )
    set_training_mode = (
        set_stage4_candidate_admission_training_mode
        if candidate_admission_enabled
        else set_stage4_cmc_training_mode
        if cmc_enabled
        else set_stage4_safe_training_mode
    )
    delta_state_fn = (
        stage4_candidate_admission_delta_state_dict
        if candidate_admission_enabled
        else stage4_cmc_delta_state_dict
        if cmc_enabled
        else stage4_delta_state_dict
    )
    validate_delta_fn = (
        validate_stage4_candidate_admission_delta_state_dict
        if candidate_admission_enabled
        else validate_stage4_cmc_delta_state_dict
        if cmc_enabled
        else lambda state: validate_stage4_delta_state_dict(
            state,
            require_route_memory_utility_gate=True,
        )
    )
    set_training_mode(model)
    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not optimizer_parameters:
        raise ValueError("safe-memory Stage4 has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=peak_learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _stage4_lr_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            minimum_ratio=minimum_ratio,
        ),
    )
    batch_protocol = {
        "batch_size": int(batch_size),
        "benchmarks": list(benchmarks),
        "rows_per_benchmark": int(batch_size) // len(benchmarks),
        "seed": int(seed),
    }
    scheduler_config = {
        "type": "linear_warmup_cosine_v1",
        "peak_learning_rate": peak_learning_rate,
        "minimum_learning_rate": minimum_lr,
        "warmup_fraction": warmup_fraction,
        "total_steps": total_steps,
    }
    optimizer_report = {
        "type": "AdamW",
        "peak_learning_rate": peak_learning_rate,
        "weight_decay": 0.01,
        "parameter_names": list(freeze_report["optimizer_parameter_names"]),
    }
    monitor = TrainingMonitor(output, checkpoint_dir=checkpoint_dir)
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    validation_reports: list[dict[str, Any]] = []
    checkpoint_paths: dict[int, Path] = {}
    validation_report_paths: dict[int, Path] = {}
    stage2_baseline: dict[str, Any] | None = None
    static_baseline_batches: list[torch.Tensor] | None = None
    static_baseline_path = validation_dir / "stage2_static_baseline.pt"
    stage2_baseline_report_path = validation_dir / "stage2_baseline_report.json"
    start_step = 0

    if resume_checkpoint_path is not None:
        resume_checkpoint_identity = sha256_path(resume_checkpoint_path)
        resume_payload = torch.load(resume_checkpoint_path, map_location="cpu")
        if resume_payload.get("stage") != "clstr_stage4_transition_conditioned_next_skill":
            raise ValueError("Stage4 resume checkpoint stage mismatch")
        if resume_payload.get("stage4_method") != stage4_method:
            raise ValueError("Stage4 resume method mismatch")
        if resume_payload.get("data_protocol_manifest_sha256") != data_protocol[
            "manifest_sha256"
        ]:
            raise ValueError("Stage4 resume data protocol mismatch")
        if resume_payload.get("batch_protocol") != batch_protocol:
            raise ValueError("Stage4 resume sampler identity mismatch")
        if resume_payload.get("full_router_digest") != freeze_report[
            "full_router_digest"
        ]:
            raise ValueError("Stage4 resume router digest mismatch")
        if resume_payload.get("optimizer_parameter_names") != freeze_report[
            "optimizer_parameter_names"
        ]:
            raise ValueError("Stage4 resume optimizer parameter mismatch")
        if resume_payload.get("scheduler_config") != scheduler_config:
            raise ValueError("Stage4 resume scheduler configuration mismatch")
        if resume_payload.get("frozen_embedding_cache_identity_sha256") != (
            frozen_embedding_cache["cache_identity_sha256"]
        ):
            raise ValueError("Stage4 resume frozen embedding cache mismatch")
        validate_delta_fn(resume_payload["model_state_dict"])
        if candidate_admission_enabled and str(
            resume_payload.get("base_cmc_checkpoint_sha256") or ""
        ) != str(base_cmc_checkpoint_sha256):
            raise ValueError("Stage4 resume base CMC identity mismatch")
        if cmc_enabled:
            validate_stage4_cmc_parent_router(resume_payload, model)
        model.load_state_dict(resume_payload["model_state_dict"], strict=False)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        start_step = int(resume_payload["step"])
        history = list(resume_payload.get("history") or [])
        validation_history = list(resume_payload.get("validation_history") or [])
        validation_reports = list(resume_payload.get("validation_reports") or [])
        checkpoint_paths = {
            int(step): Path(path)
            for step, path in dict(
                resume_payload.get("checkpoint_paths") or {}
            ).items()
        }
        validation_report_paths = {
            int(step): Path(path)
            for step, path in dict(
                resume_payload.get("validation_report_paths") or {}
            ).items()
        }
        current_report_path = validation_dir / f"step{start_step}.json"
        if not current_report_path.is_file():
            raise ValueError("Stage4 resume validation report is missing")
        current_report = json.loads(
            current_report_path.read_text(encoding="utf-8")
        )
        current_report_without_hash = dict(current_report)
        current_report_hash = str(
            current_report_without_hash.pop("manifest_sha256", "")
        )
        if current_report_hash != canonical_digest(current_report_without_hash):
            raise ValueError("Stage4 resume validation report self-hash mismatch")
        if dict(current_report.get("resume_checkpoint") or {}) != (
            resume_checkpoint_identity
        ):
            raise ValueError("Stage4 resume validation checkpoint identity mismatch")
        current_step = int(current_report.get("step", -1))
        if current_step != start_step:
            raise ValueError("Stage4 resume validation step mismatch")
        if not any(int(item.get("step", -1)) == start_step for item in validation_reports):
            validation_reports.append(current_report)
        checkpoint_paths[start_step] = Path(
            current_report["stage4_checkpoint"]["path"]
        )
        validation_report_paths[start_step] = current_report_path
        if not any(int(item.get("step", -1)) == start_step for item in validation_history):
            validation_history.append(
                {
                    "step": start_step,
                    "release_status": current_report["release_status"],
                    "dynamic_mrr": current_report["balanced_macro"][
                        "dynamic_mrr"
                    ],
                    "dynamic_minus_static_mrr": current_report[
                        "balanced_macro"
                    ]["dynamic_minus_static_mrr"],
                    "report_path": str(current_report_path),
                }
            )
        if stage2_baseline_report_path.is_file() and static_baseline_path.is_file():
            baseline_payload = json.loads(
                stage2_baseline_report_path.read_text(encoding="utf-8")
            )
            stage2_baseline = dict(baseline_payload["stage2_baseline"])
            static_baseline_batches = torch.load(
                static_baseline_path,
                map_location="cpu",
            )

    objective_config = (
        {
            "stage4_method": CANDIDATE_ADMISSION_RESIDUAL_V1,
            "next_skill_pool_mode": "full_pool",
            "counterfactual_training_objective": (
                "candidate_admission_listwise_bce_static_no_regret_v1"
            ),
            "candidate_admission_static_k": int(candidate_admission_static_k),
            "candidate_admission_dynamic_extra_k": int(
                candidate_admission_dynamic_extra_k
            ),
            "candidate_admission_residual_bound": 2.0,
            "base_cmc_checkpoint_sha256": str(base_cmc_checkpoint_sha256),
            "stage4_resume_supported": True,
        }
        if candidate_admission_enabled
        else
        {
            "stage4_method": COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
            "next_skill_pool_mode": "full_pool",
            "counterfactual_training_objective": (
                "dynamic_plus_fused_log_utility_with_static_no_regret"
            ),
            "feature_schema": "memory_utility_features_v1",
            "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
            "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
            "stage4_resume_supported": True,
        }
        if cmc_enabled
        else {
            "stage4_method": LEGACY_SAFE_MEMORY_STAGE4_V1,
            "next_skill_pool_mode": "full_pool",
            "counterfactual_utility_weight": float(counterfactual_utility_weight),
            "counterfactual_gain_margin": float(counterfactual_gain_margin),
            "counterfactual_safety_tolerance": float(
                counterfactual_safety_tolerance
            ),
            "counterfactual_gain_weight": float(counterfactual_gain_weight),
            "counterfactual_safety_weight": float(counterfactual_safety_weight),
            "counterfactual_warmup_fraction": float(
                counterfactual_warmup_fraction
            ),
            "counterfactual_warmup_step_source": "fresh_stage4_local_step",
            "counterfactual_training_objective": "safe_local_candidate_rank_v1",
            "safe_memory_residual_bound": safe_memory_residual_bound,
            "safe_local_candidate_sizes": list(safe_local_candidate_sizes),
            "stage4_resume_supported": True,
        }
    )

    def compact_checkpoint_payload(step: int) -> dict[str, Any]:
        return {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": stage4_method,
            "step": int(step),
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": delta_state_fn(model),
            **(
                {
                    "base_cmc_checkpoint_sha256": str(
                        base_cmc_checkpoint_sha256
                    )
                }
                if candidate_admission_enabled
                else {}
            ),
            "parent_stage2_full_router_digest": freeze_report[
                "full_router_digest"
            ],
            "parent_stage2_fast_router_digest": freeze_report[
                "fast_router_digest"
            ],
            "train_report": {
                "status": "running" if step < total_steps else "ok",
                "safe_memory_protocol_version": stage4_method,
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "data_protocol": data_protocol,
                "batch_protocol": batch_protocol,
                "freeze_report": freeze_report,
                "frozen_embedding_cache": frozen_embedding_cache,
                "checkpoint_init_report": checkpoint_init_report or {},
                "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
                **objective_config,
            },
            "setup_status_path": str(setup_status),
        }

    def run_validation(step: int) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal stage2_baseline, static_baseline_batches
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient_health = {
            "finite": bool(
                all(
                    torch.isfinite(gradient).all()
                    for gradient in trainable_gradients
                )
            ),
            "nonzero": bool(
                any(
                    bool((gradient.detach().abs() > 0).any())
                    for gradient in trainable_gradients
                )
            ),
            "gradient_tensor_count": len(trainable_gradients),
        }
        if int(step) == 0 and not trainable_gradients:
            gradient_health["finite"] = True
        model.eval()
        current_fast_digest = router_state_digest(model, scope="fast")
        current_full_digest = router_state_digest(model, scope="full")
        if current_fast_digest != freeze_report["fast_router_digest"]:
            raise ValueError("safe-memory Stage4 fast router digest drifted")
        if current_full_digest != freeze_report["full_router_digest"]:
            raise ValueError("safe-memory Stage4 full router digest drifted")
        delta_state = delta_state_fn(model)
        delta_validation = validate_delta_fn(delta_state)
        checkpoint_tag = (
            "candidate_admission"
            if candidate_admission_enabled
            else "cmc"
            if cmc_enabled
            else "safe"
        )
        compact_path = checkpoint_dir / f"clstr_stage4_{checkpoint_tag}-step{step}.pt"
        torch.save(compact_checkpoint_payload(step), compact_path)
        evaluation = evaluate_stage4_validation(
            model=model,
            validation_rows=validation_rows,
            gate_rows=gate_rows,
            device=device,
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids,
            batch_size=max(1, int(batch_size)),
            static_k=(
                int(candidate_admission_static_k)
                if candidate_admission_enabled
                else 500
            ),
            dynamic_extra_k=(
                int(candidate_admission_dynamic_extra_k)
                if candidate_admission_enabled
                else 64
            ),
            final_k=(
                min(64, int(candidate_admission_static_k))
                if candidate_admission_enabled
                else 64
            ),
            counterfactual_utility_weight=counterfactual_utility_weight,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            safe_memory_residual_bound=safe_memory_residual_bound,
            stage4_method=stage4_method,
            validation_step=step,
        )
        if stage2_baseline is None:
            baseline_core = stage2_baseline_from_evaluation(
                evaluation,
                full_router_digest=freeze_report["full_router_digest"],
            )
            baseline_payload = {
                "schema_version": "stage4_stage2_baseline_v1",
                "status": "ok",
                "stage2_baseline": baseline_core,
            }
            baseline_payload["manifest_sha256"] = canonical_digest(
                baseline_payload
            )
            write_json(stage2_baseline_report_path, baseline_payload)
            baseline_identity = sha256_path(stage2_baseline_report_path)
            stage2_baseline = {
                **baseline_core,
                "report_path": baseline_identity["path"],
                "report_sha256": baseline_identity["sha256"],
                "report_manifest_sha256": baseline_payload[
                    "manifest_sha256"
                ],
            }
            static_baseline_batches = list(
                evaluation["validation"]["static_logits_batches"]
            )
            torch.save(static_baseline_batches, static_baseline_path)
        if static_baseline_batches is None or stage2_baseline is None:
            raise ValueError("safe-memory Stage4 baseline was not initialized")
        static_integrity = validate_static_baseline_batches(
            static_baseline_batches,
            evaluation["validation"]["static_logits_batches"],
        )
        router_integrity = {
            "full_router_digest": current_full_digest,
            "fast_router_digest": current_fast_digest,
            "full_router_digest_unchanged": (
                current_full_digest == freeze_report["full_router_digest"]
            ),
            "fast_router_digest_unchanged": (
                current_fast_digest == freeze_report["fast_router_digest"]
            ),
            **static_integrity,
        }
        report = build_stage4_validation_report(
            evaluation,
            step=step,
            stage2_baseline=stage2_baseline,
            router_integrity=router_integrity,
            delta_state_validation=delta_validation,
            stage4_method=stage4_method,
            gradient_health=gradient_health,
            validation_interval_steps=validation_interval_steps,
        )
        if candidate_admission_enabled:
            report["base_cmc_checkpoint_sha256"] = str(
                base_cmc_checkpoint_sha256
            )
        resume_path = resume_dir / f"clstr_stage4_safe-step{step}.pt"
        resume_payload = {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": stage4_method,
            "safe_memory_protocol_version": stage4_method,
            "step": int(step),
            "model_state_dict": delta_state,
            **(
                {
                    "base_cmc_checkpoint_sha256": str(
                        base_cmc_checkpoint_sha256
                    )
                }
                if candidate_admission_enabled
                else {}
            ),
            "parent_stage2_full_router_digest": freeze_report[
                "full_router_digest"
            ],
            "parent_stage2_fast_router_digest": freeze_report[
                "fast_router_digest"
            ],
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "data_protocol_manifest_sha256": data_protocol["manifest_sha256"],
            "batch_protocol": batch_protocol,
            "full_router_digest": freeze_report["full_router_digest"],
            "optimizer_parameter_names": freeze_report[
                "optimizer_parameter_names"
            ],
            "scheduler_config": scheduler_config,
            "frozen_embedding_cache_identity_sha256": frozen_embedding_cache[
                "cache_identity_sha256"
            ],
            "history": history,
            "validation_history": validation_history,
            "validation_reports": validation_reports,
            "checkpoint_paths": {
                str(key): str(value) for key, value in checkpoint_paths.items()
            },
            "validation_report_paths": {
                str(key): str(value)
                for key, value in validation_report_paths.items()
            },
        }
        torch.save(resume_payload, resume_path)
        resume_identity = sha256_path(resume_path)
        persisted_report, report_path = persist_stage4_validation_artifacts(
            output_dir=output,
            step=step,
            report=report,
            evaluation=evaluation,
            compact_checkpoint_path=compact_path,
            skill_id_to_idx=skill_id_to_idx,
            resume_checkpoint=resume_identity,
        )
        checkpoint_paths[int(step)] = compact_path
        validation_report_paths[int(step)] = report_path
        validation_reports.append(persisted_report)
        validation_history.append(
            {
                "step": int(step),
                "release_status": persisted_report["release_status"],
                "dynamic_mrr": persisted_report["balanced_macro"].get(
                    "raw_dynamic_mrr",
                    persisted_report["balanced_macro"].get("dynamic_mrr"),
                ),
                "dynamic_minus_static_mrr": persisted_report[
                    "balanced_macro"
                ].get(
                    "fused_minus_static_mrr",
                    persisted_report["balanced_macro"].get(
                        "dynamic_minus_static_mrr"
                    ),
                ),
                "fused_mrr": persisted_report["balanced_macro"].get(
                    "fused_mrr"
                ),
                "fused_regret": persisted_report["balanced_macro"].get(
                    "fused_regret"
                ),
                "report_path": str(report_path),
            }
        )
        selection = (
            select_stage4_candidate_admission_checkpoint(
                validation_reports,
                checkpoint_paths=checkpoint_paths,
                validation_report_paths=validation_report_paths,
                terminal_step=total_steps,
            )
            if candidate_admission_enabled
            else select_stage4_cmc_checkpoint(
                validation_reports,
                checkpoint_paths=checkpoint_paths,
                validation_report_paths=validation_report_paths,
                terminal_step=total_steps,
            )
            if cmc_enabled
            else select_stage4_dynamic_checkpoint(
                validation_reports,
                checkpoint_paths=checkpoint_paths,
                validation_report_paths=validation_report_paths,
            )
        )
        selection_path = output / "stage4_dynamic_selection.json"
        written_selection = write_stage4_dynamic_selection(
            selection_path,
            selection=selection,
        )
        monitor.save_latest(resume_payload)
        set_training_mode(model)
        return written_selection, {
            "resume_path": str(resume_path),
            "resume_checkpoint": resume_identity,
            "selection_path": str(selection_path),
        }

    append_setup_status(
        setup_status,
        "training_started",
        max_steps=total_steps,
        batch_size=int(batch_size),
    )
    latest_selection: dict[str, Any] | None = None
    latest_artifacts: dict[str, Any] = {}
    if start_step == 0:
        latest_selection, latest_artifacts = run_validation(0)
    for step_index in range(start_step + 1, total_steps + 1):
        set_training_mode(model)
        batch = batcher.batch_for_step(step_index)
        counterfactual_scale = (
            1.0
            if cmc_enabled or candidate_admission_enabled
            else _counterfactual_warmup_scale(
                step_index,
                total_steps,
                counterfactual_warmup_fraction,
            )
        )
        loss, metrics = _compute_stage4_act_loss(
            model,
            batch,
            device,
            transition_residual_lambda=0.0,
            transition_scoring_mode=TRANSITION_SCORING_MODE,
            use_replay_prefix_belief=True,
            trainable_replay_prefix=trainable_replay_prefix,
            route_scorer=UNIFIED_MEMORY_ROUTE_SCORER,
            next_skill_pool_mode="full_pool",
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids,
            counterfactual_utility_weight=counterfactual_utility_weight,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            counterfactual_scale=counterfactual_scale,
            safe_memory_residual_bound=safe_memory_residual_bound,
            safe_local_candidate_sizes=safe_local_candidate_sizes,
            stage4_method=stage4_method,
            candidate_admission_static_k=int(candidate_admission_static_k),
            candidate_admission_dynamic_extra_k=int(
                candidate_admission_dynamic_extra_k
            ),
        )
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        scheduler.step()
        metrics = {
            **metrics,
            "step": int(step_index),
            "loss": float(loss.detach().cpu().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(metrics)
        monitor.record(metrics, None)
        if (
            step_index % int(validation_interval_steps) == 0
            or step_index == total_steps
        ):
            latest_selection, latest_artifacts = run_validation(step_index)
    if latest_selection is None:
        latest_selection, latest_artifacts = run_validation(total_steps)
    candidate_admission_training_support = (
        _validate_candidate_admission_training_support(history)
        if candidate_admission_enabled
        else None
    )
    final_full_digest = router_state_digest(model, scope="full")
    if final_full_digest != freeze_report["full_router_digest"]:
        raise ValueError("safe-memory Stage4 full router digest drifted")
    selected_checkpoint = str(latest_selection["selected_checkpoint_path"])
    report = {
        "status": "ok",
        "stage": "clstr_stage4_transition_conditioned_next_skill",
        "stage4_method": stage4_method,
        "safe_memory_protocol_version": stage4_method,
        "training_objective": (
            "candidate_admission_constrained_residual"
            if candidate_admission_enabled
            else "counterfactual_memory_calibration"
            if cmc_enabled
            else "full_pool_causal_next_skill_with_counterfactual_utility"
        ),
        "training_regime": "offline_train_split_causal_next_skill",
        "valid_or_test_used_for_training": False,
        "validation_used_for_checkpoint_selection": True,
        "on_policy_rollout_used": False,
        "checkpoint": selected_checkpoint,
        "dynamic_selection": latest_selection,
        "dynamic_selection_path": latest_artifacts["selection_path"],
        "stage4_data_protocol_path": str(data_protocol_path),
        "stage2_baseline_report_path": str(stage2_baseline_report_path),
        "data_protocol": data_protocol,
        "batch_protocol": batch_protocol,
        "data_report": data_report,
        "freeze_report": freeze_report,
        "candidate_admission_training_support": (
            candidate_admission_training_support
        ),
        **(
            {
                "base_cmc_checkpoint_sha256": str(
                    base_cmc_checkpoint_sha256
                )
            }
            if candidate_admission_enabled
            else {}
        ),
        "frozen_embedding_cache": frozen_embedding_cache,
        "checkpoint_init_report": checkpoint_init_report or {},
        "optimizer": optimizer_report,
        "scheduler": scheduler_config,
        "route_scorer": UNIFIED_MEMORY_ROUTE_SCORER,
        "next_skill_pool_mode": "full_pool",
        "uses_stage0_prior_at_inference": False,
        "transition_scoring_mode": UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_scoring_mode_requested": TRANSITION_SCORING_MODE,
        "transition_residual_lambda": 0.0,
        "transition_residual_lambda_requested": 0.0,
        **objective_config,
        "auto_replay_prefix": train_data_report["auto_replay_prefix"],
        "trainable_replay_prefix": bool(trainable_replay_prefix),
        "last_metrics": history[-1] if history else {},
        "history": history,
        "validation_history": validation_history,
        "latest_resume_checkpoint": latest_artifacts["resume_checkpoint"],
        "setup_status_path": str(setup_status),
        **monitor.paths_report(),
    }
    write_json(output / "train_report.json", report)
    write_json(output / "train_stdout.json", report)
    return report


def train_stage4_act_with_model(
    *,
    model: Any,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 3.0e-5,
    minimum_learning_rate: float = 3.0e-6,
    learning_rate_warmup_fraction: float = 0.05,
    validation_fraction: float = 0.10,
    validation_rows_per_benchmark: int = 256,
    minimum_validation_rows_per_benchmark: int = 128,
    gate_rows_per_benchmark: int = 512,
    validation_interval_steps: int = 400,
    resume_checkpoint_path: str | Path | None = None,
    seed: int = 17,
    candidate_count: int | None = 64,
    train_transition: bool = False,
    max_rows: int | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    next_skill_pool_mode: str = "full_pool",
    stage4_method: str = LEGACY_SAFE_MEMORY_STAGE4_V1,
    counterfactual_utility_weight: float = 0.05,
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_warmup_fraction: float = 0.05,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
    allowed_benchmarks: set[str] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    stage0_top_m: int | None = None,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: int = 8,
    stage0_inventory_min_candidates: int = 0,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    stage0_handoff_cache_mode: str = "off",
    stage0_handoff_cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    stage0_handoff_cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    stage0_handoff_cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    routing_checkpoint_path: str | Path | None = None,
    checkpoint_init_report: dict[str, Any] | None = None,
    setup_status_path: str | Path | None = None,
    auto_replay_prefix_max_steps: int = 3,
    trainable_replay_prefix: bool = False,
    base_cmc_checkpoint_sha256: str | None = None,
    candidate_admission_static_k: int = 500,
    candidate_admission_dynamic_extra_k: int = 64,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    transition_residual_lambda = float(transition_residual_lambda)
    transition_scoring_mode = str(transition_scoring_mode or TRANSITION_SCORING_MODE)
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    next_skill_pool_mode = str(next_skill_pool_mode or "full_pool")
    stage4_method = str(stage4_method or LEGACY_SAFE_MEMORY_STAGE4_V1)
    counterfactual_utility_weight = float(counterfactual_utility_weight)
    counterfactual_gain_margin = float(counterfactual_gain_margin)
    counterfactual_safety_tolerance = float(counterfactual_safety_tolerance)
    counterfactual_gain_weight = float(counterfactual_gain_weight)
    counterfactual_safety_weight = float(counterfactual_safety_weight)
    counterfactual_warmup_fraction = float(counterfactual_warmup_fraction)
    safe_memory_residual_bound = float(safe_memory_residual_bound)
    safe_local_candidate_sizes = tuple(int(size) for size in safe_local_candidate_sizes)
    if transition_scoring_mode not in TRANSITION_SCORING_MODES:
        raise ValueError(f"unsupported transition scoring mode: {transition_scoring_mode}")
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    if stage4_method not in STAGE4_METHODS:
        raise ValueError(f"unsupported stage4_method: {stage4_method}")
    if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1 and (
        route_scorer != UNIFIED_MEMORY_ROUTE_SCORER
        or next_skill_pool_mode != "full_pool"
    ):
        raise ValueError("CMC Stage4 requires unified_memory full-pool routing")
    if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1 and trainable_replay_prefix:
        raise ValueError("CMC Stage4 requires frozen replay-prefix expansion")
    if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1 and (
        route_scorer != UNIFIED_MEMORY_ROUTE_SCORER
        or next_skill_pool_mode != "full_pool"
    ):
        raise ValueError(
            "candidate-admission Stage4 requires unified_memory full-pool routing"
        )
    if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1 and trainable_replay_prefix:
        raise ValueError(
            "candidate-admission Stage4 requires frozen replay-prefix expansion"
        )
    if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1 and not str(
        base_cmc_checkpoint_sha256 or ""
    ):
        raise ValueError(
            "candidate-admission Stage4 requires base_cmc_checkpoint_sha256"
        )
    if next_skill_pool_mode == "full_pool" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("next_skill_pool_mode=full_pool requires route_scorer=unified_memory")
    _validate_counterfactual_hyperparameters(
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
    )
    if not math.isfinite(counterfactual_utility_weight) or counterfactual_utility_weight < 0.0:
        raise ValueError("counterfactual_utility_weight must be finite and nonnegative")
    _counterfactual_warmup_scale(0, max_steps, counterfactual_warmup_fraction)
    if not math.isfinite(safe_memory_residual_bound) or safe_memory_residual_bound <= 0.0:
        raise ValueError("safe_memory_residual_bound must be finite and positive")
    if not safe_local_candidate_sizes or any(size < 2 for size in safe_local_candidate_sizes):
        raise ValueError("safe_local_candidate_sizes must contain integers of at least two")
    effective_transition_scoring_mode = (
        UNIFIED_MEMORY_ROUTE_SCORER
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else transition_scoring_mode
    )
    effective_transition_residual_lambda = (
        0.0
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else transition_residual_lambda
    )
    training_objective = (
        "candidate_admission_constrained_residual"
        if stage4_method == CANDIDATE_ADMISSION_RESIDUAL_V1
        else "counterfactual_memory_calibration"
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
        else "full_pool_causal_next_skill_with_counterfactual_utility"
        if next_skill_pool_mode == "full_pool"
        else "causal_transition_conditioned_next_skill_ce"
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else "legacy_transition_conditioned_next_skill_ce"
    )
    training_regime = (
        "offline_train_split_causal_next_skill"
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else "offline_train_split_transition_conditioned_next_skill"
    )
    if (
        route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        and next_skill_pool_mode == "full_pool"
        and int(validation_rows_per_benchmark) > 0
    ):
        return _train_stage4_safe_memory_with_model(
            model=model,
            trajectories_path=trajectories_path,
            skills_path=skills_path,
            output_dir=output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            minimum_learning_rate=minimum_learning_rate,
            learning_rate_warmup_fraction=learning_rate_warmup_fraction,
            validation_fraction=validation_fraction,
            validation_rows_per_benchmark=validation_rows_per_benchmark,
            minimum_validation_rows_per_benchmark=(
                minimum_validation_rows_per_benchmark
            ),
            gate_rows_per_benchmark=gate_rows_per_benchmark,
            validation_interval_steps=validation_interval_steps,
            resume_checkpoint_path=resume_checkpoint_path,
            seed=seed,
            candidate_count=candidate_count,
            max_rows=max_rows,
            counterfactual_utility_weight=counterfactual_utility_weight,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            counterfactual_warmup_fraction=counterfactual_warmup_fraction,
            safe_memory_residual_bound=safe_memory_residual_bound,
            safe_local_candidate_sizes=safe_local_candidate_sizes,
            allowed_benchmarks=allowed_benchmarks,
            benchmark_caps=benchmark_caps,
            stage0_top_m=stage0_top_m,
            stage0_positive_missing_policy=stage0_positive_missing_policy,
            stage0_handoff_query_mode=stage0_handoff_query_mode,
            stage0_handoff_sample_multiplier=stage0_handoff_sample_multiplier,
            stage0_inventory_min_candidates=stage0_inventory_min_candidates,
            stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
            stage0_candidate_progress_interval_batches=(
                stage0_candidate_progress_interval_batches
            ),
            stage0_handoff_cache_mode=stage0_handoff_cache_mode,
            stage0_handoff_cache_dir=stage0_handoff_cache_dir,
            stage0_handoff_cache_format=stage0_handoff_cache_format,
            stage0_handoff_cache_shard_size=stage0_handoff_cache_shard_size,
            routing_checkpoint_path=routing_checkpoint_path,
            checkpoint_init_report=checkpoint_init_report,
            setup_status_path=setup_status_path,
            auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
            trainable_replay_prefix=trainable_replay_prefix,
            stage4_method=stage4_method,
            base_cmc_checkpoint_sha256=base_cmc_checkpoint_sha256,
            candidate_admission_static_k=int(candidate_admission_static_k),
            candidate_admission_dynamic_extra_k=int(
                candidate_admission_dynamic_extra_k
            ),
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_status = Path(setup_status_path) if setup_status_path is not None else output / "setup_status.jsonl"
    if setup_status_path is None:
        reset_setup_status(setup_status)

    skills = _read_jsonl(skills_path)
    append_setup_status(setup_status, "skills_loaded", skills_path=str(skills_path), skill_count=len(skills))
    skill_id_to_idx = {str(row.get("skill_id") or row.get("canonical_skill_id")): idx for idx, row in enumerate(skills)}
    equivalent_skill_ids = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if next_skill_pool_mode == "full_pool":
        _validate_stage4_full_skill_mapping(model, skill_id_to_idx)
    append_setup_status(setup_status, "model_moved_to_device", device=str(device))

    stage0_handoff_enabled = stage0_top_m is not None and int(stage0_top_m) > 0
    raw_source_rows = list(_read_jsonl_stream(trajectories_path))
    prepared_source_rows, causal_next_state_report = _attach_adjacent_next_states(raw_source_rows)
    append_setup_status(setup_status, "stage4_causal_next_state_prepared", **causal_next_state_report)
    source_rows, source_report = _eligible_stage4_source_rows(
        prepared_source_rows,
        skill_id_to_idx,
        allowed_benchmarks=allowed_benchmarks,
        max_source_rows=(
            None
            if benchmark_caps is not None
            else
            _stage4_handoff_source_limit(max_rows, stage0_handoff_sample_multiplier)
            if stage0_handoff_enabled
            else max_rows
        ),
    )
    source_rows, benchmark_caps_report = _cap_rows_by_benchmark(source_rows, benchmark_caps)
    append_setup_status(setup_status, "stage4_source_rows_filtered", data_report=source_report)
    append_setup_status(setup_status, "stage4_benchmark_caps_applied", benchmark_caps_report=benchmark_caps_report)
    stage0_candidate_handoff_report: dict[str, Any] | None = None
    if stage0_handoff_enabled:
        source_rows_for_handoff = [
            _as_stage4_handoff_row(row, next_skill_pool_mode=next_skill_pool_mode)
            for row in source_rows
        ]
        source_rows, stage0_candidate_handoff_report = _prepare_stage0_topm_candidates_with_cache(
            model,
            source_rows_for_handoff,
            skills,
            skill_id_to_idx,
            top_m=stage0_top_m,
            positive_missing_policy=stage0_positive_missing_policy,
            query_mode=stage0_handoff_query_mode,
            allow_full_pool_stage2_debug=False,
            routing_checkpoint_path=routing_checkpoint_path,
            manifest_path=output / "stage0_candidate_handoff.json",
            encode_batch_size=stage0_candidate_encode_batch_size,
            device=device,
            setup_status_path=setup_status,
            progress_interval_batches=stage0_candidate_progress_interval_batches,
            inventory_min_candidates=stage0_inventory_min_candidates,
            next_skill_pool_mode=next_skill_pool_mode,
            cache_mode=stage0_handoff_cache_mode,
            cache_dir=stage0_handoff_cache_dir,
            cache_format=stage0_handoff_cache_format,
            cache_shard_size=stage0_handoff_cache_shard_size,
            route_scorer=route_scorer,
        )
        append_setup_status(
            setup_status,
            "stage0_candidate_handoff_prepared",
            **stage0_candidate_handoff_report,
        )
    rows, data_report = _build_stage4_next_skill_rows_from_source_rows(
        source_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
        max_rows=max_rows,
        allowed_benchmarks=allowed_benchmarks,
        stage0_candidate_handoff_report=stage0_candidate_handoff_report,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    rows, auto_replay_prefix_report = _attach_auto_replay_prefixes(
        rows,
        max_steps=auto_replay_prefix_max_steps,
    )
    data_report = {
        **data_report,
        "source_path": str(trajectories_path),
        "source_rows": source_report["source_rows"],
        "eligible_source_rows": source_report["eligible_source_rows"],
        "stage4_handoff_sample_multiplier": int(stage0_handoff_sample_multiplier),
        "skipped_benchmarks": source_report["skipped_benchmarks"],
        "skipped_reasons": dict(Counter(source_report["skipped_reasons"]) + Counter(data_report["skipped_reasons"])),
        "benchmark_caps": benchmark_caps_report,
        "causal_next_state": causal_next_state_report,
        "auto_replay_prefix": auto_replay_prefix_report,
        "trainable_replay_prefix": bool(trainable_replay_prefix),
    }
    append_setup_status(setup_status, "stage4_rows_built", data_report=data_report)
    row_order_report = {
        "strategy": "seeded_shuffle",
        "seed": int(seed),
        "stage4_rows": len(rows),
    }
    random.Random(int(seed)).shuffle(rows)
    append_setup_status(setup_status, "stage4_rows_shuffled", row_order=row_order_report)
    if not rows:
        append_setup_status(setup_status, "stage4_blocked", error="no_stage4_rows")
        report = {
            "status": "blocked",
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "error": "no_stage4_rows",
            "data_report": data_report,
            "setup_status_path": str(setup_status),
        }
        write_json(output / "blocker_report.json", report)
        return report

    freeze_report = _freeze_for_stage4_act(
        model,
        train_transition=train_transition,
        route_scorer=route_scorer,
        stage4_method=stage4_method,
    )
    append_setup_status(setup_status, "stage4_freeze_applied", freeze_report=freeze_report)
    parent_stage2_digest_metadata = (
        {
            "parent_stage2_full_router_digest": freeze_report[
                "full_router_digest"
            ],
            "parent_stage2_fast_router_digest": freeze_report[
                "fast_router_digest"
            ],
        }
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
        else {}
    )
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate)) if params else None

    def checkpoint_delta_state() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1:
            state = stage4_cmc_delta_state_dict(model)
            validation = validate_stage4_cmc_delta_state_dict(state)
            return state, {
                "checkpoint_excludes_frozen_qwen_backbone": True,
                "checkpoint_excludes_frozen_routing_foundation": True,
                "checkpoint_state_key_count": len(state),
                "stage4_cmc_delta_validation": validation,
            }
        return _checkpoint_state_dict(
            model,
            exclude_frozen_qwen_backbone=True,
            exclude_frozen_routing_foundation=True,
        )
    history: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    objective_config = (
        {
            "stage4_method": COUNTERFACTUAL_MEMORY_CALIBRATION_V1,
            "next_skill_pool_mode": next_skill_pool_mode,
            "counterfactual_training_objective": (
                "dynamic_plus_fused_log_utility_with_static_no_regret"
            ),
            "feature_schema": "memory_utility_features_v1",
            "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
            "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
            "stage4_resume_supported": False,
        }
        if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
        else {
            "stage4_method": stage4_method,
            "next_skill_pool_mode": next_skill_pool_mode,
            "counterfactual_utility_weight": counterfactual_utility_weight,
            "counterfactual_gain_margin": counterfactual_gain_margin,
            "counterfactual_safety_tolerance": counterfactual_safety_tolerance,
            "counterfactual_gain_weight": counterfactual_gain_weight,
            "counterfactual_safety_weight": counterfactual_safety_weight,
            "counterfactual_warmup_fraction": counterfactual_warmup_fraction,
            "counterfactual_warmup_step_source": "fresh_stage4_local_step",
            "counterfactual_training_objective": "safe_local_candidate_rank_v1",
            "safe_memory_residual_bound": safe_memory_residual_bound,
            "safe_local_candidate_sizes": list(safe_local_candidate_sizes),
            "stage4_resume_supported": False,
        }
    )
    monitor = TrainingMonitor(output, checkpoint_dir=checkpoint_dir)
    initial_checkpoint_state, initial_checkpoint_report = checkpoint_delta_state()
    monitor.save_latest(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": stage4_method,
            "step": 0,
            **parent_stage2_digest_metadata,
            "model_state_dict": initial_checkpoint_state,
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "train_report": {
                "status": "running",
                "training_objective": training_objective,
                "training_regime": training_regime,
                "on_policy_rollout_used": False,
                "data_report": data_report,
                "row_order": row_order_report,
                "freeze_report": freeze_report,
                "checkpoint_init_report": checkpoint_init_report or {},
                "transition_scoring_mode": effective_transition_scoring_mode,
                "transition_scoring_mode_requested": transition_scoring_mode,
                "route_scorer": route_scorer,
                "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
                "transition_residual_lambda": effective_transition_residual_lambda,
                "transition_residual_lambda_requested": transition_residual_lambda,
                **objective_config,
                "auto_replay_prefix": auto_replay_prefix_report,
                "trainable_replay_prefix": bool(trainable_replay_prefix),
                "last_metrics": {},
            },
            "setup_status_path": str(setup_status),
            **initial_checkpoint_report,
        }
    )
    append_setup_status(setup_status, "training_started", max_steps=int(max_steps), batch_size=int(batch_size))
    for step_idx in range(1, max(1, int(max_steps)) + 1):
        batch = _batch_for_step(rows, step_idx, batch_size)
        counterfactual_scale = (
            1.0
            if stage4_method == COUNTERFACTUAL_MEMORY_CALIBRATION_V1
            else _counterfactual_warmup_scale(
                step_idx,
                max_steps,
                counterfactual_warmup_fraction,
            )
        )
        loss, metrics = _compute_stage4_act_loss(
            model,
            batch,
            device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            use_replay_prefix_belief=True,
            trainable_replay_prefix=trainable_replay_prefix,
            route_scorer=route_scorer,
            next_skill_pool_mode=next_skill_pool_mode,
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids,
            counterfactual_utility_weight=counterfactual_utility_weight,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            counterfactual_scale=counterfactual_scale,
            safe_memory_residual_bound=safe_memory_residual_bound,
            safe_local_candidate_sizes=safe_local_candidate_sizes,
            stage4_method=stage4_method,
        )
        if optimizer is not None:
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        metrics = {**metrics, "step": step_idx, "loss": float(loss.detach().cpu().item())}
        last_metrics = metrics
        history.append(metrics)
        step_checkpoint_payload = None
        if monitor.should_save_checkpoint(step_idx):
            step_checkpoint_state, step_checkpoint_report = checkpoint_delta_state()
            step_checkpoint_payload = {
                    "stage": "clstr_stage4_transition_conditioned_next_skill",
                    "stage4_method": stage4_method,
                    "step": step_idx,
                    **parent_stage2_digest_metadata,
                    "model_state_dict": step_checkpoint_state,
                    "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
                    "train_report": {
                        "status": "running",
                        "training_objective": training_objective,
                        "training_regime": training_regime,
                        "on_policy_rollout_used": False,
                        "data_report": data_report,
                        "row_order": row_order_report,
                        "freeze_report": freeze_report,
                        "checkpoint_init_report": checkpoint_init_report or {},
                        "transition_scoring_mode": effective_transition_scoring_mode,
                        "transition_scoring_mode_requested": transition_scoring_mode,
                        "route_scorer": route_scorer,
                        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
                        "transition_residual_lambda": effective_transition_residual_lambda,
                        "transition_residual_lambda_requested": transition_residual_lambda,
                        **objective_config,
                        "auto_replay_prefix": auto_replay_prefix_report,
                        "trainable_replay_prefix": bool(trainable_replay_prefix),
                        "last_metrics": last_metrics,
                    },
                    "setup_status_path": str(setup_status),
                    **step_checkpoint_report,
                }
        monitor.record(
            metrics,
            step_checkpoint_payload,
        )

    checkpoint_state, checkpoint_report = checkpoint_delta_state()
    checkpoint_path = checkpoint_dir / f"clstr_stage4_act-step{max_steps}.pt"
    payload = {
        "stage": "clstr_stage4_transition_conditioned_next_skill",
        "stage4_method": stage4_method,
        "step": max_steps,
        **parent_stage2_digest_metadata,
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "train_report": {
            "status": "ok",
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "training_objective": training_objective,
            "training_regime": training_regime,
            "valid_or_test_used_for_training": False,
            "on_policy_rollout_used": False,
            "data_report": data_report,
            "row_order": row_order_report,
            "freeze_report": freeze_report,
            "checkpoint_init_report": checkpoint_init_report or {},
            "transition_scoring_mode": effective_transition_scoring_mode,
            "transition_scoring_mode_requested": transition_scoring_mode,
            "route_scorer": route_scorer,
            "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
            "transition_residual_lambda": effective_transition_residual_lambda,
            "transition_residual_lambda_requested": transition_residual_lambda,
            **objective_config,
            "auto_replay_prefix": auto_replay_prefix_report,
            "trainable_replay_prefix": bool(trainable_replay_prefix),
            "last_metrics": last_metrics,
            "history": history,
        },
        "setup_status_path": str(setup_status),
        **monitor.paths_report(),
        **checkpoint_report,
    }
    torch.save(payload, checkpoint_path)
    monitor.save_latest(payload)
    report = {
        "status": "ok",
        "stage": "clstr_stage4_transition_conditioned_next_skill",
        "stage4_method": stage4_method,
        "training_objective": training_objective,
        "training_regime": training_regime,
        "valid_or_test_used_for_training": False,
        "on_policy_rollout_used": False,
        "checkpoint": str(checkpoint_path),
        "data_report": data_report,
        "row_order": row_order_report,
        "freeze_report": freeze_report,
        "checkpoint_init_report": checkpoint_init_report or {},
        "transition_scoring_mode": effective_transition_scoring_mode,
        "transition_scoring_mode_requested": transition_scoring_mode,
        "route_scorer": route_scorer,
        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
        "transition_residual_lambda": effective_transition_residual_lambda,
        "transition_residual_lambda_requested": transition_residual_lambda,
        **objective_config,
        "auto_replay_prefix": auto_replay_prefix_report,
        "trainable_replay_prefix": bool(trainable_replay_prefix),
        "last_metrics": last_metrics,
        "history": history,
        "setup_status_path": str(setup_status),
        **monitor.paths_report(),
        **checkpoint_report,
    }
    write_json(output / "train_report.json", report)
    write_json(output / "train_stdout.json", report)
    return report
