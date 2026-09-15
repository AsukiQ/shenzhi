from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.action_adapter import UniversalActionAdapter
from clstr.aux_pretrain import _build_model_from_routing_init, _encode_text_batches, _gpu_report
from clstr.belief import subspace_obs
from clstr.counterfactual_ranking import (
    full_pool_causal_route_objective,
    multi_positive_log_utility,
    shuffled_history_utility_loss,
)
from clstr.external_data import write_json
from clstr.gated_temporal_reranker import GatedTemporalConfig, GatedTemporalReranker, prior_preserving_loss
from clstr.history_channel import (
    audit_history_channel_rows,
    materialize_history_free_state,
    router_state_text,
    strip_history_sections,
)
from clstr.memory_candidate_recall import (
    CANDIDATE_SELECTION_VERSION,
    TIE_BREAK_POLICY,
    explicit_inventory_skill_ids_ordered as _shared_explicit_inventory_skill_ids_ordered,
    full_pool_positive_mask,
    legal_skill_pool_mask,
    stable_masked_topk_rows,
)
from clstr.model import _skill_text_serializer
from clstr.safe_memory_ranking import (
    bounded_memory_fusion,
    build_local_candidate_masks,
    safe_local_route_objective,
)
from clstr.state_query_prompt import RAW_STATE_V1
from clstr.stage0_handoff_acceleration import (
    LEGACY_EXECUTION_SCHEDULE_VERSION,
    ROW_SHARDED_CACHE_FORMAT,
    RawStage0Candidates,
    append_row_sharded_cache,
    build_legacy_schedule_row_key_plan,
    load_row_sharded_cache,
    reset_row_sharded_cache,
    stage0_handoff_global_identity,
)
from clstr.stage_checkpoint_init import (
    build_clstr_model_from_stage0_checkpoint,
    load_compatible_state_dict,
    load_head_checkpoint_into_model,
)
from clstr.stage2_static_route_anchor import (
    Stage2StaticRouteTeacher,
    static_route_anchor_kl,
)
from clstr.success_value import QSuccessHead, compute_q_success_loss, q_success_scores
from clstr.training_monitor import TrainingMonitor, append_setup_status, reset_setup_status
from clstr.transition_utils import model_action_embeddings, transition_action_input


LOSS_WEIGHT_KEYS = (
    "L_policy",
    "hard_negative_margin",
    "Q_success",
    "routing",
    "STOP",
    "L_trans",
    "L_trans_skill_ce",
    "transition_hard_negative_margin",
    "counterfactual_utility",
    "counterfactual_history",
    "belief",
)
SAFE_LOCAL_CANDIDATE_SIZES = (2, 3, 4, 5, 8, 10)
DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND = 2.0
TRAIN_SPLIT_NAMES = {"train", "training", "train_or_released_g3", "released_train"}
STAGE0_HANDOFF_QUERY_MODES = {
    "raw_state",
    "skillrouter_state",
    "checkpoint_state_query",
}
STATE_QUERY_ROLE = "state_query"
TRANSITION_TEXT_ROLE = "transition_text"
BENCHMARK_TRANSITION_BALANCED_RANDOM = "benchmark_transition_balanced_random"
BENCHMARK_TRANSITION_QUOTA_RANDOM = "benchmark_transition_quota_random"
GROUPED_SAMPLING_STRATEGIES = {BENCHMARK_TRANSITION_BALANCED_RANDOM, BENCHMARK_TRANSITION_QUOTA_RANDOM}
SAMPLING_STRATEGIES = {
    "balanced_deterministic",
    "balanced_random",
    BENCHMARK_TRANSITION_BALANCED_RANDOM,
    BENCHMARK_TRANSITION_QUOTA_RANDOM,
}
TRANSITION_INVENTORY_MASK_MODES = {
    "off",
    "auto",
    "explicit_only",
    "root_namespace",
    "stage0_topk_trajectory_prior",
}
NEXT_SKILL_POOL_MODES = {"stage0_candidates", "full_pool"}
TRANSITION_LOSS_TYPES = {"cross_entropy", "listwise_nll"}
TRANSITION_POSITIVE_MODES = {"single", "gold_plus_equivalent"}
SKILL_PRIOR_TRANSITION_SCORING_MODE = "skill_prior_plus_action_observation_residual"
STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE = "stage0_rank_prior_plus_transition_residual"
GATED_TEMPORAL_TRANSITION_SCORING_MODE = "stage0_prior_gated_temporal_residual"
TRANSITION_SCORING_MODE = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
V4_1B_TRANSITION_SCORING_MODE = "v4_1b_action_observation"
TRANSITION_SCORING_MODES = {
    TRANSITION_SCORING_MODE,
    SKILL_PRIOR_TRANSITION_SCORING_MODE,
    GATED_TEMPORAL_TRANSITION_SCORING_MODE,
    V4_1B_TRANSITION_SCORING_MODE,
}
LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER = "legacy_prior_residual"
UNIFIED_MEMORY_ROUTE_SCORER = "unified_memory"
ROUTE_SCORERS = {LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER, UNIFIED_MEMORY_ROUTE_SCORER}
DEFAULT_TRANSITION_RESIDUAL_LAMBDA = 0.25
DEFAULT_GATED_TEMPORAL_LAMBDA_MAX = 0.5
DEFAULT_GATED_TEMPORAL_KL_ALPHA = 0.03
DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA = 0.05
DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K = 64
STAGE0_SCORE_PRIOR_CALIBRATION = "off"
STAGE0_SCORE_PRIOR_CALIBRATIONS = {"off", "rank", "rank_std", "raw", "none"}
STAGE0_HANDOFF_CACHE_VERSION = "stage0_handoff_cache_v2_unified_static"
STAGE0_HANDOFF_CACHE_MODES = {"auto", "off", "refresh"}
LEGACY_STAGE0_HANDOFF_CACHE_FORMAT = "legacy_jsonl"
STAGE0_HANDOFF_CACHE_FORMATS = {LEGACY_STAGE0_HANDOFF_CACHE_FORMAT, ROW_SHARDED_CACHE_FORMAT}
DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE = 2048
DEFAULT_STAGE0_HANDOFF_CACHE_DIR = Path("outputs/cache/stage0_handoff")
BENCHMARK_ROOT_SKILL_PREFIX = {
    "alfworld": "alfworld",
    "scienceworld": "scienceworld",
    "toolbench_g3": "toolbench-g3",
    "traject_bench": "traject",
    "toolret": "toolret",
}
EXPLICIT_INVENTORY_SKILL_ID_KEYS = (
    "visible_inventory_skill_ids",
    "available_skill_ids",
    "available_skills",
    "admissible_skill_ids",
    "skill_inventory_ids",
    "stage0_allowed_skill_ids",
)
EXPLICIT_POSITIVE_SKILL_ID_KEYS = (
    "positive_next_skill_ids",
    "equivalent_next_skill_ids",
    "next_positive_skill_ids",
    "equivalent_skill_ids",
    "positive_skill_ids",
)
DEFAULT_LOSS_WEIGHTS = {
    "L_policy": 0.2,
    "hard_negative_margin": 0.2,
    "Q_success": 0.5,
    "routing": 0.0,
    "STOP": 0.1,
    "L_trans": 0.3,
    "L_trans_skill_ce": 1.0,
    "transition_hard_negative_margin": 0.0,
    "counterfactual_utility": 0.0,
    "counterfactual_history": 0.0,
    "belief": 0.1,
}
CANONICAL_STAGE_ACTIVE_LOSS_WEIGHTS = {
    "stage1": {
        "L_policy": 0.6,
        "L_trans": 0.2,
        "L_trans_skill_ce": 0.6,
        "belief": 0.1,
    },
    "stage2": {
        "L_policy": 0.2,
        "L_trans": 0.3,
        "L_trans_skill_ce": 1.0,
        "counterfactual_history": 0.0,
        "belief": 0.1,
    },
}
CANONICAL_REMOVED_LOSS_KEYS = {
    "routing",
    "hard_negative_margin",
    "Q_success",
    "transition_hard_negative_margin",
    "STOP",
    "counterfactual_utility",
}
BELIEF_CALIBRATION_STATE_KEYS = {
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
}


def canonical_stage_loss_weights(
    stage_name: str,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    stage_name = str(stage_name or "")
    if stage_name not in CANONICAL_STAGE_ACTIVE_LOSS_WEIGHTS:
        raise ValueError(f"unsupported canonical stage loss contract: {stage_name}")
    active = dict(CANONICAL_STAGE_ACTIVE_LOSS_WEIGHTS[stage_name])
    for key, raw_value in (overrides or {}).items():
        value = float(raw_value)
        if key not in active:
            if value != 0.0:
                raise ValueError(f"removed canonical {stage_name} loss: {key}")
            continue
        active[key] = value
    return {key: float(active.get(key, 0.0)) for key in LOSS_WEIGHT_KEYS}


def _reported_loss_weights(loss_weights: dict[str, float]) -> dict[str, float]:
    normalized = {key: float(loss_weights.get(key, 0.0)) for key in LOSS_WEIGHT_KEYS}
    if all(normalized[key] == 0.0 for key in CANONICAL_REMOVED_LOSS_KEYS):
        return {key: value for key, value in normalized.items() if value > 0.0}
    return normalized


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _counterfactual_memory_permutation(
    rows: list[dict[str, Any]],
    device: torch.device,
    state_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    groups: dict[tuple[str, int, str], list[int]] = {}
    for row_index, row in enumerate(rows):
        benchmark = str(row.get("source_benchmark") or row.get("benchmark") or "unknown")
        replay_prefix = row.get("replay_prefix")
        replay_length = len(replay_prefix) if isinstance(replay_prefix, list) else 0
        if replay_length == 0:
            continue
        current_skill = str(row.get("skill_id") or "")
        groups.setdefault((benchmark, replay_length, current_skill), []).append(row_index)

    normalized_states: torch.Tensor | None = None
    if isinstance(state_embeddings, torch.Tensor):
        if state_embeddings.ndim != 2 or int(state_embeddings.size(0)) != len(rows):
            raise ValueError("counterfactual donor state embeddings must align with rows")
        normalized_states = F.normalize(state_embeddings.detach().float(), dim=-1)

    def candidate_set(row: dict[str, Any]) -> set[str]:
        for key in (
            "candidate_next_skill_ids",
            "candidate_next_skill_indices",
            "stage0_next_candidate_skill_indices",
            "visible_inventory_skill_ids",
            "tool_inventory_skill_ids",
        ):
            values = row.get(key)
            if isinstance(values, list) and values:
                return {str(value) for value in values if str(value)}
        return set()

    donors = [-1] * len(rows)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for position, row_index in enumerate(indices):
            target = rows[row_index]
            target_trajectory = str(
                target.get("trajectory_id")
                or target.get("task_id")
                or f"row-{row_index}"
            )
            target_candidates = candidate_set(target)
            ranked_donors: list[tuple[float, float, int, int]] = []
            for offset in range(1, len(indices)):
                donor_index = indices[(position + offset) % len(indices)]
                donor = rows[donor_index]
                donor_trajectory = str(
                    donor.get("trajectory_id")
                    or donor.get("task_id")
                    or f"row-{donor_index}"
                )
                if donor_trajectory == target_trajectory:
                    continue
                target_next_skill = str(target.get("next_skill_id") or "")
                donor_next_skill = str(donor.get("next_skill_id") or "")
                if (
                    target_next_skill
                    and donor_next_skill
                    and donor_next_skill == target_next_skill
                ):
                    continue
                donor_candidates = candidate_set(donor)
                union = target_candidates | donor_candidates
                overlap = (
                    len(target_candidates & donor_candidates) / len(union)
                    if union
                    else 0.0
                )
                similarity = (
                    float((normalized_states[row_index] * normalized_states[donor_index]).sum().item())
                    if normalized_states is not None
                    else 0.0
                )
                ranked_donors.append((overlap, similarity, -offset, donor_index))
            if ranked_donors:
                donors[row_index] = max(ranked_donors)[-1]
    return torch.tensor(donors, dtype=torch.long, device=device)


def _json_safe_for_report(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe_for_report(value.detach().cpu().item())
        return [_json_safe_for_report(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe_for_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_for_report(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _raise_if_nonfinite_loss(
    loss: torch.Tensor,
    *,
    metrics: dict[str, Any],
    step_idx: int,
    output_dir: str | Path,
) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    report = {
        "status": "action_required",
        "reason": "nonfinite_loss",
        "step": int(step_idx),
        "loss": _json_safe_for_report(loss.detach()),
        "metrics": _json_safe_for_report(metrics),
    }
    write_json(Path(output_dir) / "blocker_report.json", report)
    raise FloatingPointError(f"non-finite CLSTR full-base loss at step {step_idx}")


def _auto_replay_prefix_step(prev_row: dict[str, Any], next_row: dict[str, Any] | None = None) -> dict[str, Any]:
    next_observation = prev_row.get("next_observation_text")
    observation_source = str(prev_row.get("observation_source") or "row_next_observation_text")
    if (next_observation is None or not str(next_observation)) and next_row is not None:
        next_observation = router_state_text(next_row)
        observation_source = "next_state_without_tool_result"
    step = {
        "observation_text": router_state_text(prev_row),
        "action_text": str(prev_row.get("action_text") or prev_row.get("expert_action") or ""),
        "next_observation_text": str(next_observation or ""),
        "skill_id": str(prev_row.get("skill_id") or ""),
        "observation_source": observation_source,
    }
    if "trajectory_id" in prev_row:
        step["trajectory_id"] = str(prev_row.get("trajectory_id") or "")
    if "step_index" in prev_row:
        step["step_index"] = prev_row.get("step_index")
    if "skill_idx" in prev_row:
        step["skill_idx"] = prev_row.get("skill_idx")
    return step


def _attach_auto_replay_prefixes(
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_steps = max(0, int(max_steps or 0))
    existing_prefix_rows = sum(1 for row in rows if isinstance(row.get("replay_prefix"), list) and row.get("replay_prefix"))
    if max_steps <= 0:
        return rows, {
            "enabled": False,
            "max_steps": max_steps,
            "row_count": len(rows),
            "rows_with_existing_replay_prefix": existing_prefix_rows,
            "rows_with_auto_replay_prefix": 0,
            "total_prefix_steps": 0,
            "max_prefix_len": 0,
            "trajectory_count": 0,
        }

    grouped: dict[tuple[str, str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for original_idx, row in enumerate(rows):
        identity = _adjacency_identity(row)
        if not identity[3]:
            continue
        grouped.setdefault(identity, []).append((original_idx, row))

    prepared = list(rows)
    auto_prefix_rows = 0
    total_prefix_steps = 0
    max_prefix_len = 0
    for _identity, indexed_rows in grouped.items():
        step_counts = Counter(_strict_step_index(row) for _idx, row in indexed_rows)
        ordered = sorted(
            indexed_rows,
            key=lambda item: (
                _strict_step_index(item[1]) is None,
                _strict_step_index(item[1]) or 0,
                item[0],
            ),
        )
        previous_rows: list[dict[str, Any]] = []
        previous_step: int | None = None
        for original_idx, row in ordered:
            step_index = _strict_step_index(row)
            if step_index is None or step_counts[step_index] != 1:
                previous_rows = []
                previous_step = None
                continue
            if previous_step is None or step_index != previous_step + 1:
                previous_rows = []
            existing_prefix = row.get("replay_prefix")
            if isinstance(existing_prefix, list) and existing_prefix:
                previous_rows.append(row)
                previous_step = step_index
                continue
            prefix_source = previous_rows[-max_steps:]
            if prefix_source:
                prefix_context = [*prefix_source, row]
                prefix = [
                    _auto_replay_prefix_step(prev_row, next_row=prefix_context[offset + 1])
                    for offset, prev_row in enumerate(prefix_source)
                ]
                updated = dict(row)
                updated["replay_prefix"] = prefix
                prepared[original_idx] = updated
                auto_prefix_rows += 1
                total_prefix_steps += len(prefix)
                max_prefix_len = max(max_prefix_len, len(prefix))
            previous_rows.append(row)
            previous_step = step_index

    return prepared, {
        "enabled": True,
        "max_steps": max_steps,
        "row_count": len(rows),
        "rows_with_existing_replay_prefix": existing_prefix_rows,
        "rows_with_auto_replay_prefix": auto_prefix_rows,
        "total_prefix_steps": total_prefix_steps,
        "max_prefix_len": max_prefix_len,
        "trajectory_count": len(grouped),
    }


def _normalize_allowed_benchmarks(allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if allowed_benchmarks is None:
        return None
    normalized = {str(item).strip() for item in allowed_benchmarks if str(item).strip()}
    return normalized or None


def _normalize_benchmark_caps(benchmark_caps: dict[str, int] | None) -> dict[str, int] | None:
    if benchmark_caps is None:
        return None
    normalized = {str(key).strip(): int(value) for key, value in benchmark_caps.items() if str(key).strip()}
    return normalized or None


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


def _source_namespace(row: dict[str, Any]) -> str:
    provenance = _provenance(row)
    for key in ("source_namespace", "source_id", "source_dataset", "source"):
        value = row.get(key) or provenance.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _strict_step_index(row: dict[str, Any]) -> int | None:
    value = row.get("step_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    return None


def _adjacency_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("benchmark") or "").strip(),
        _source_namespace(row),
        _split_name(row),
        str(row.get("trajectory_id") or "").strip(),
    )


def _attach_adjacent_next_states(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach only next states proven by a unique, strict t+1 trajectory row."""
    prepared = [dict(row) for row in rows]
    indexed_steps: dict[tuple[str, str, str, str], dict[int, list[int]]] = {}
    step_by_index: dict[int, int] = {}
    for idx, row in enumerate(rows):
        step = _strict_step_index(row)
        if step is None:
            continue
        step_by_index[idx] = step
        indexed_steps.setdefault(_adjacency_identity(row), {}).setdefault(step, []).append(idx)

    skipped: Counter[str] = Counter()
    causal_rows = 0
    attached_rows = 0
    materialized_rows = 0
    preserved_rows = 0
    masked_rows = 0

    def reject(idx: int, reason: str) -> None:
        nonlocal masked_rows
        updated = prepared[idx]
        updated.pop("next_state_text", None)
        updated.pop("next_state_source", None)
        updated.pop("_next_state_embedding", None)
        loss_mask = updated.get("loss_mask")
        copied_mask = dict(loss_mask) if isinstance(loss_mask, dict) else {}
        copied_mask["L_trans_skill_ce"] = False
        updated["loss_mask"] = copied_mask
        updated["causal_next_state_skip_reason"] = reason
        skipped[reason] += 1
        masked_rows += 1

    for idx, row in enumerate(rows):
        loss_mask = row.get("loss_mask")
        is_causal = bool(str(row.get("next_skill_id") or "").strip()) or bool(
            isinstance(loss_mask, dict) and loss_mask.get("L_trans_skill_ce")
        )
        if not is_causal:
            continue
        causal_rows += 1
        missing_current_field_reason = next(
            (
                reason
                for key, reason in (
                    ("action_text", "missing_action_text"),
                    ("next_observation_text", "missing_next_observation_text"),
                    ("skill_id", "missing_current_skill"),
                    ("next_skill_id", "missing_next_skill"),
                )
                if not str(row.get(key) or "").strip()
            ),
            None,
        )
        if not router_state_text(row):
            missing_current_field_reason = "missing_state_text"
        if missing_current_field_reason is not None:
            reject(idx, missing_current_field_reason)
            continue
        step = step_by_index.get(idx)
        if step is None:
            reject(idx, "invalid_step_index")
            continue
        identity = _adjacency_identity(row)
        benchmark, _source_namespace_value, split, trajectory_id = identity
        if not benchmark or not split or not trajectory_id:
            reject(idx, "missing_adjacency_identity")
            continue
        identity_steps = indexed_steps.get(identity, {})
        if len(identity_steps.get(step, [])) != 1 or len(identity_steps.get(step + 1, [])) > 1:
            reject(idx, "duplicate_step_index")
            continue
        successor_indices = identity_steps.get(step + 1, [])
        if not successor_indices:
            reject(idx, "missing_adjacent_step")
            continue

        successor = rows[successor_indices[0]]
        next_skill_id = str(row.get("next_skill_id") or "").strip()
        successor_skill_id = str(successor.get("skill_id") or "").strip()
        if not next_skill_id or next_skill_id != successor_skill_id:
            reject(idx, "adjacent_skill_mismatch")
            continue
        successor_state = router_state_text(successor)
        if not successor_state:
            reject(idx, "empty_next_state")
            continue
        existing_state = strip_history_sections(str(row.get("next_state_text") or ""))
        successor_base_state = ""
        if bool(successor.get("available_actions_planner_context")):
            successor_base_state = str(successor.get("_available_actions_base_state_text") or "").strip()
        if existing_state:
            if existing_state == successor_state:
                preserved_rows += 1
            elif successor_base_state and existing_state == successor_base_state:
                prepared[idx]["next_state_text"] = successor_state
                prepared[idx].pop("_next_state_embedding", None)
                materialized_rows += 1
            else:
                reject(idx, "existing_next_state_mismatch")
                continue
        else:
            prepared[idx]["next_state_text"] = successor_state
            prepared[idx].pop("_next_state_embedding", None)
            materialized_rows += 1
        prepared[idx]["next_state_source"] = "adjacent_trajectory_row"
        prepared[idx].pop("causal_next_state_skip_reason", None)
        attached_rows += 1

    for updated in prepared:
        updated.pop("_available_actions_base_state_text", None)

    return prepared, {
        "row_count": len(rows),
        "causal_rows": causal_rows,
        "attached_rows": attached_rows,
        "materialized_rows": materialized_rows,
        "preserved_rows": preserved_rows,
        "masked_rows": masked_rows,
        "skip_reasons": dict(sorted(skipped.items())),
    }


def _filter_rows_by_train_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    skipped_by_split: Counter[str] = Counter()
    for row in rows:
        split = _split_name(row)
        if split in TRAIN_SPLIT_NAMES:
            retained.append(row)
        else:
            skipped_by_split[split or "<missing>"] += 1
    return retained, {
        "enabled": True,
        "accepted_splits": sorted(TRAIN_SPLIT_NAMES),
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "skipped_rows": len(rows) - len(retained),
        "skipped_by_split": dict(sorted(skipped_by_split.items())),
    }


def _filter_rows_by_allowed_benchmarks(
    rows: list[dict[str, Any]],
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = _normalize_allowed_benchmarks(allowed_benchmarks)
    if allowed is None:
        return rows, {
            "enabled": False,
            "allowed_benchmarks": [],
            "source_rows": len(rows),
            "retained_rows": len(rows),
            "skipped_rows": 0,
            "skipped_by_benchmark": {},
        }
    retained: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        benchmark = str(row.get("benchmark") or "")
        if benchmark in allowed:
            retained.append(row)
        else:
            skipped[benchmark] += 1
    return retained, {
        "enabled": True,
        "allowed_benchmarks": sorted(allowed),
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "skipped_rows": len(rows) - len(retained),
        "skipped_by_benchmark": dict(sorted(skipped.items())),
    }


def _cap_rows_by_benchmark(
    rows: list[dict[str, Any]],
    benchmark_caps: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    caps = _normalize_benchmark_caps(benchmark_caps)
    if caps is None:
        return rows, {
            "enabled": False,
            "benchmark_caps": {},
            "source_rows": len(rows),
            "retained_rows": len(rows),
            "skipped_rows": 0,
            "retained_by_benchmark": dict(sorted(Counter(str(row.get("benchmark") or "") for row in rows).items())),
            "skipped_by_benchmark": {},
        }
    retained: list[dict[str, Any]] = []
    retained_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    for row in rows:
        benchmark = str(row.get("benchmark") or "")
        cap = caps.get(benchmark)
        if cap is not None and cap >= 0 and retained_counts[benchmark] >= cap:
            skipped_counts[benchmark] += 1
            continue
        retained.append(row)
        retained_counts[benchmark] += 1
    return retained, {
        "enabled": True,
        "benchmark_caps": dict(sorted(caps.items())),
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "skipped_rows": len(rows) - len(retained),
        "retained_by_benchmark": dict(sorted(retained_counts.items())),
        "skipped_by_benchmark": dict(sorted(skipped_counts.items())),
    }


def _limit_rows_for_smoke(rows: list[dict[str, Any]], max_rows: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_rows is None:
        return rows, {
            "enabled": False,
            "source_rows": len(rows),
            "retained_rows": len(rows),
            "skipped_rows": 0,
            "max_rows": None,
        }
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive when provided, got {max_rows}")
    limited = rows[:max_rows]
    return limited, {
        "enabled": True,
        "source_rows": len(rows),
        "retained_rows": len(limited),
        "skipped_rows": max(0, len(rows) - len(limited)),
        "max_rows": max_rows,
    }


def _skill_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or idx)


def _inject_positive_candidate(candidates: list[int], positive: int, top_m: int) -> tuple[list[int], bool]:
    if positive in candidates:
        return candidates, False
    width = max(1, int(top_m))
    output = list(candidates[:width])
    if len(output) < width:
        output.append(positive)
    else:
        output[-1] = positive
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in output:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    if positive not in seen:
        deduped.append(positive)
    return deduped[:width], True


def _skillrouter_query_text_from_raw(query: str) -> str:
    return (
        "Instruct: Given a task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery:"
        f"{str(query)[:1500]}"
    )


def _stage0_handoff_raw_query(row: dict[str, Any], *, target: str) -> str:
    state = router_state_text(row)
    if target == "current":
        return state
    if target != "next":
        raise ValueError(f"unsupported Stage0 handoff query target: {target}")
    next_state = strip_history_sections(str(row.get("next_state_text") or ""))
    if next_state:
        return next_state
    action = str(row.get("action_text") or row.get("expert_action") or "").strip()
    next_observation = str(row.get("next_observation_text") or state).strip()
    parts = [state]
    if action:
        parts.append(f"previous_action: {action}")
    if next_observation:
        parts.append(f"next_observation: {next_observation}")
    return "\n".join(part for part in parts if part)


def _stage0_handoff_query_text(row: dict[str, Any], *, target: str, mode: str = "skillrouter_state") -> str:
    if mode not in STAGE0_HANDOFF_QUERY_MODES:
        raise ValueError(f"unsupported Stage0 handoff query mode: {mode}")
    raw = _stage0_handoff_raw_query(row, target=target)
    if mode in {"raw_state", "checkpoint_state_query"}:
        return raw
    return _skillrouter_query_text_from_raw(raw)


def _stage0_topm_handoff_report(
    *,
    stage0_checkpoint: str | Path | None,
    top_m: int | None,
    positive_missing_policy: str,
    query_mode: str,
    allow_full_pool_stage2_debug: bool,
    source_rows: int,
    retained_rows: int,
    skipped_reasons: Counter[str],
    injected_rows: int,
    current_positive_required_rows: int = 0,
    current_positive_covered_rows: int = 0,
    next_positive_required_rows: int = 0,
    next_positive_covered_rows: int = 0,
    stage0_topm_next_positive_covered_rows: int = 0,
    masked_next_skill_ce_rows: int = 0,
    current_skill_candidate_added_rows: int = 0,
    current_skill_candidate_positive_rows: int = 0,
    current_skill_candidate_hard_negative_rows: int = 0,
    next_positive_rank_bucket_counts: Counter[str] | dict[str, int] | None = None,
    learnable_correction_rows_rank_2_to_100: int = 0,
    inventory_candidate_rows: int = 0,
    inventory_candidate_missing_rows: int = 0,
    inventory_candidate_removed_candidates: int = 0,
    inventory_candidate_topk_backfilled_rows: int = 0,
    inventory_candidate_topk_backfilled_candidates: int = 0,
    tool_inventory_backfill_report: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    enabled = top_m is not None and int(top_m) > 0
    candidate_source = "stage0_topm_online" if enabled else "full_pool_debug"
    current_coverage = (
        current_positive_covered_rows / current_positive_required_rows
        if current_positive_required_rows > 0
        else None
    )
    next_coverage = (
        next_positive_covered_rows / next_positive_required_rows
        if next_positive_required_rows > 0
        else None
    )
    stage0_topm_next_coverage = (
        stage0_topm_next_positive_covered_rows / next_positive_required_rows
        if next_positive_required_rows > 0
        else None
    )
    return {
        "enabled": enabled,
        "stage0_checkpoint": None if stage0_checkpoint is None else str(stage0_checkpoint),
        "candidate_source": candidate_source,
        "static_candidate_scorer": "unified_static" if enabled else None,
        "top_m": None if top_m is None else int(top_m),
        "positive_missing_policy": str(positive_missing_policy),
        "query_mode": str(query_mode),
        "source_rows": int(source_rows),
        "retained_rows": int(retained_rows),
        "skipped_rows": int(source_rows - retained_rows),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "injected_positive_rows": int(injected_rows),
        "current_positive_required_rows": int(current_positive_required_rows),
        "current_positive_covered_rows": int(current_positive_covered_rows),
        "current_positive_coverage@M": current_coverage,
        "next_positive_required_rows": int(next_positive_required_rows),
        "next_positive_covered_rows": int(next_positive_covered_rows),
        "next_positive_coverage@M": next_coverage,
        "stage0_topm_next_positive_covered_rows": int(stage0_topm_next_positive_covered_rows),
        "stage0_topm_next_positive_coverage@M": stage0_topm_next_coverage,
        "masked_next_skill_ce_rows": int(masked_next_skill_ce_rows),
        "current_skill_candidate_added_rows": int(current_skill_candidate_added_rows),
        "current_skill_candidate_positive_rows": int(current_skill_candidate_positive_rows),
        "current_skill_candidate_hard_negative_rows": int(current_skill_candidate_hard_negative_rows),
        "next_positive_rank_bucket_counts": dict(
            sorted((next_positive_rank_bucket_counts or Counter()).items())
        ),
        "learnable_correction_rows_rank_2_to_100": int(learnable_correction_rows_rank_2_to_100),
        "inventory_candidate_rows": int(inventory_candidate_rows),
        "inventory_candidate_missing_rows": int(inventory_candidate_missing_rows),
        "inventory_candidate_removed_candidates": int(inventory_candidate_removed_candidates),
        "inventory_candidate_topk_backfilled_rows": int(inventory_candidate_topk_backfilled_rows),
        "inventory_candidate_topk_backfilled_candidates": int(inventory_candidate_topk_backfilled_candidates),
        "tool_inventory_backfill": tool_inventory_backfill_report or {},
        "full_pool_stage2_debug_allowed": bool(allow_full_pool_stage2_debug),
        "manifest_path": None if manifest_path is None else str(manifest_path),
    }


def _stable_json_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_stat_digest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    resolved = Path(path)
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _path_content_digest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    resolved = Path(path)
    if not resolved.exists():
        return {"path": str(resolved), "exists": False, "sha256": None}
    digest = hashlib.sha256()
    total_size = 0
    files = [resolved] if resolved.is_file() else sorted(item for item in resolved.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if resolved.is_file() else str(item.relative_to(resolved))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                total_size += len(block)
                digest.update(block)
    return {
        "path": str(resolved),
        "exists": True,
        "file_count": len(files),
        "size": int(total_size),
        "sha256": digest.hexdigest(),
    }


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _effective_unified_static_scorer_digest(model: Any) -> str:
    prefixes = (
        "encoder.",
        "skill_table.W",
        "skill_table.E",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
        "initial_belief_head.",
        "unified_retriever.",
    )
    state = getattr(model, "state_dict", lambda: {})()
    payload = []
    for name, tensor in sorted(state.items()):
        if any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes):
            payload.append((name, _tensor_content_digest(tensor)))
    if not payload:
        raise ValueError("unified static scorer digest found no scorer tensors")
    return _stable_json_digest(payload)


def _stage2_route_teacher_digest(
    teacher: Stage2StaticRouteTeacher,
    model: Any,
) -> str:
    payload = [
        (name, _tensor_content_digest(tensor))
        for name, tensor in sorted(teacher.state_dict().items())
    ]
    skill_embeddings = getattr(getattr(model, "skill_table", None), "E", None)
    if not isinstance(skill_embeddings, torch.Tensor):
        raise ValueError("stage2 route teacher digest requires skill_table.E")
    payload.append(("skill_table.E", _tensor_content_digest(skill_embeddings)))
    return _stable_json_digest(payload)


def _stage0_handoff_rows_digest(rows: list[dict[str, Any]]) -> str:
    compact = []
    for idx, row in enumerate(rows):
        compact.append(
            {
                "idx": idx,
                "row_id": row.get("row_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index"),
                "benchmark": row.get("benchmark"),
                "skill_id": row.get("skill_id"),
                "next_skill_id": row.get("next_skill_id"),
                "state_text": router_state_text(row),
                "state_text_full": row.get("state_text_full"),
                "next_observation_text": row.get("next_observation_text"),
                "next_state_text": row.get("next_state_text"),
                "loss_mask": row.get("loss_mask"),
                "replay_prefix": None,
                "tool_inventory_skill_ids": row.get("tool_inventory_skill_ids"),
                "explicit_inventory": {
                    key: row.get(key)
                    for key in EXPLICIT_INVENTORY_SKILL_ID_KEYS
                    if key in row
                },
            }
        )
    return _stable_json_digest(compact)


def _stage0_handoff_skills_digest(skills: list[dict[str, Any]]) -> str:
    return _stable_json_digest(
        [
            {
                "idx": idx,
                "skill_id": _skill_id(skill, idx),
                "canonical_skill_id": skill.get("canonical_skill_id"),
                "name": skill.get("name"),
            }
            for idx, skill in enumerate(skills)
        ]
    )


def _stage0_handoff_cache_key(
    *,
    rows_digest: str,
    skills_digest: str,
    checkpoint_digest: str | dict[str, Any],
    top_m: int | None,
    positive_missing_policy: str,
    query_mode: str,
    inventory_min_candidates: int,
    allow_full_pool_stage2_debug: bool,
    static_candidate_scorer: str = "unified_static",
    initial_belief_top_k: int | None = None,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    state_text_format: str = "default",
    skill_text_format: str = "default",
    skill_embedding_digest: str = "",
    declared_pool_order_digest: str = "",
    candidate_selection_version: str = CANDIDATE_SELECTION_VERSION,
    tie_break_policy: str = TIE_BREAK_POLICY,
    unified_static_scorer_digest: str = "",
    next_skill_pool_mode: str = "stage0_candidates",
) -> dict[str, Any]:
    key = {
        "version": STAGE0_HANDOFF_CACHE_VERSION,
        "rows_digest": str(rows_digest),
        "skills_digest": str(skills_digest),
        "checkpoint_digest": checkpoint_digest,
        "top_m": None if top_m is None else int(top_m),
        "positive_missing_policy": str(positive_missing_policy),
        "query_mode": str(query_mode),
        "inventory_min_candidates": int(inventory_min_candidates),
        "allow_full_pool_stage2_debug": bool(allow_full_pool_stage2_debug),
        "static_candidate_scorer": str(static_candidate_scorer),
        "initial_belief_top_k": None if initial_belief_top_k is None else int(initial_belief_top_k),
        "route_scorer": str(route_scorer),
        "state_text_format": str(state_text_format),
        "skill_text_format": str(skill_text_format),
        "skill_embedding_digest": str(skill_embedding_digest),
        "declared_pool_order_digest": str(declared_pool_order_digest),
        "candidate_selection_version": str(candidate_selection_version),
        "tie_break_policy": str(tie_break_policy),
        "unified_static_scorer_digest": str(unified_static_scorer_digest),
        "next_skill_pool_mode": str(next_skill_pool_mode),
    }
    key["cache_key"] = _stable_json_digest(key)
    return key


def _stage0_handoff_cache_entry_dir(cache_root: str | Path, cache_key: dict[str, Any]) -> Path:
    key = str(cache_key.get("cache_key") or "")
    if not key:
        raise ValueError("stage0 handoff cache key is missing cache_key")
    return Path(cache_root) / key[:2] / key


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_stage0_handoff_cache(
    cache_root: str | Path,
    cache_key: dict[str, Any],
    rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    entry_dir = _stage0_handoff_cache_entry_dir(cache_root, cache_key)
    entry_dir.mkdir(parents=True, exist_ok=True)
    rows_path = entry_dir / "rows.jsonl"
    report_path = entry_dir / "report.json"
    manifest_path = entry_dir / "manifest.json"
    _write_jsonl_rows(rows_path, rows)
    cached_report = dict(report)
    cached_report["cache_hit"] = False
    cached_report["cache_key"] = cache_key["cache_key"]
    cached_report["cache_path"] = str(entry_dir)
    write_json(report_path, cached_report)
    manifest = {
        "key": cache_key,
        "rows_path": str(rows_path.name),
        "report_path": str(report_path.name),
        "retained_rows": len(rows),
    }
    write_json(manifest_path, manifest)
    return {
        "status": "written",
        "cache_key": cache_key["cache_key"],
        "cache_path": str(entry_dir),
        "retained_rows": len(rows),
    }


def _load_stage0_handoff_cache(
    cache_root: str | Path,
    cache_key: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    entry_dir = _stage0_handoff_cache_entry_dir(cache_root, cache_key)
    manifest_path = entry_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("key") != cache_key:
        return None
    rows_path = entry_dir / str(manifest.get("rows_path") or "rows.jsonl")
    report_path = entry_dir / str(manifest.get("report_path") or "report.json")
    if not rows_path.is_file() or not report_path.is_file():
        return None
    rows = _read_jsonl(rows_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if len(rows) != int(manifest.get("retained_rows") or -1):
        return None
    report = dict(report)
    report["cache_hit"] = True
    report["cache_key"] = cache_key["cache_key"]
    report["cache_path"] = str(entry_dir)
    return rows, report


def _stage0_handoff_row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("row_id"),
        row.get("trajectory_id"),
        row.get("step_index"),
        row.get("benchmark"),
        row.get("skill_id"),
        row.get("next_skill_id"),
        router_state_text(row),
        row.get("next_observation_text"),
        row.get("next_state_text"),
    )


def _merge_stage0_handoff_cached_rows(
    current_rows: list[dict[str, Any]],
    cached_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in current_rows:
        current_by_identity.setdefault(_stage0_handoff_row_identity(row), []).append(row)

    merged_rows: list[dict[str, Any]] = []
    missing_current_rows = 0
    for cached_row in cached_rows:
        identity = _stage0_handoff_row_identity(cached_row)
        current_bucket = current_by_identity.get(identity) or []
        current_row = current_bucket.pop(0) if current_bucket else None
        if current_row is None:
            missing_current_rows += 1
            continue
        merged_row = dict(current_row)
        preserve_current_causal_mask = bool(current_row.get("causal_next_state_skip_reason"))
        for key, value in cached_row.items():
            if key in {
                "replay_prefix",
                "next_state_text",
                "next_state_source",
                "_next_state_embedding",
                "causal_next_state_skip_reason",
            }:
                continue
            if key == "loss_mask" and preserve_current_causal_mask:
                continue
            merged_row[key] = value
        merged_rows.append(merged_row)

    missing_cached_rows = sum(len(bucket) for bucket in current_by_identity.values())

    return merged_rows, {
        "cached_rows": int(len(cached_rows)),
        "merged_rows": int(len(merged_rows)),
        "missing_current_rows": int(missing_current_rows),
        "missing_cached_rows": int(missing_cached_rows),
        "valid": bool(missing_current_rows == 0),
    }


def _rank_bucket(rank: int | None) -> str:
    if rank is None or int(rank) <= 0:
        return "missing"
    rank = int(rank)
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 20:
        return "6-20"
    if rank <= 100:
        return "21-100"
    return ">100"


def _stage0_topk_indices_with_explicit_inventory(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
    k: int,
    min_candidates: int = 0,
) -> tuple[list[list[int]], Counter[str]]:
    output: list[list[int]] = []
    audit: Counter[str] = Counter()
    k = max(1, int(k))
    min_candidates = min(max(0, int(min_candidates or 0)), k, int(skill_count))
    if logits.ndim != 2 or int(logits.size(0)) != len(rows) or int(logits.size(1)) < int(skill_count):
        raise ValueError("Stage0 handoff logits must cover every row and declared skill")
    global_valid = torch.ones(
        len(rows),
        int(skill_count),
        dtype=torch.bool,
        device=logits.device,
    )
    for row_idx, row in enumerate(rows):
        inventory_ids = _explicit_inventory_skill_ids_ordered(row)
        inventory_indices = [
            int(skill_id_to_idx[skill_id])
            for skill_id in inventory_ids
            if skill_id in skill_id_to_idx
        ]
        if inventory_indices:
            audit["inventory_candidate_rows"] += 1
            audit["inventory_candidate_removed_candidates"] += max(0, int(skill_count) - len(set(inventory_indices)))
            inventory_valid = torch.zeros(1, int(skill_count), dtype=torch.bool, device=logits.device)
            inventory_valid[0, torch.tensor(inventory_indices, dtype=torch.long, device=logits.device)] = True
            selected = stable_masked_topk_rows(
                logits[row_idx : row_idx + 1, :skill_count],
                inventory_valid,
                k=k,
            )[0]
            if min_candidates > 0 and len(selected) < min_candidates:
                seen = set(selected)
                global_top = stable_masked_topk_rows(
                    logits[row_idx : row_idx + 1, :skill_count],
                    global_valid[row_idx : row_idx + 1],
                    k=k,
                )[0]
                before_backfill = len(selected)
                for idx in global_top:
                    idx = int(idx)
                    if idx in seen:
                        continue
                    selected.append(idx)
                    seen.add(idx)
                    if len(selected) >= min_candidates:
                        break
                backfilled = len(selected) - before_backfill
                if backfilled > 0:
                    audit["inventory_candidate_topk_backfilled_rows"] += 1
                    audit["inventory_candidate_topk_backfilled_candidates"] += backfilled
            output.append(selected)
            continue
        audit["inventory_candidate_missing_rows"] += 1
        selected = stable_masked_topk_rows(
            logits[row_idx : row_idx + 1, :skill_count],
            global_valid[row_idx : row_idx + 1],
            k=k,
        )[0]
        output.append(selected)
    return output, audit


def _stage0_topk_with_explicit_inventory(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
    k: int,
    min_candidates: int = 0,
) -> tuple[list[list[int]], list[list[float]], Counter[str]]:
    output, audit = _stage0_topk_indices_with_explicit_inventory(
        logits,
        rows,
        skill_id_to_idx=skill_id_to_idx,
        skill_count=skill_count,
        k=k,
        min_candidates=min_candidates,
    )
    output_scores = [
        [float(logits[row_idx, int(idx)].detach().cpu().item()) for idx in selected]
        for row_idx, selected in enumerate(output)
    ]
    return output, output_scores, audit


def _stage0_topk_with_explicit_inventory_vectorized_scores(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
    k: int,
    min_candidates: int = 0,
) -> tuple[list[list[int]], list[list[float]], Counter[str]]:
    output, audit = _stage0_topk_indices_with_explicit_inventory(
        logits,
        rows,
        skill_id_to_idx=skill_id_to_idx,
        skill_count=skill_count,
        k=k,
        min_candidates=min_candidates,
    )
    max_width = max((len(selected) for selected in output), default=0)
    if max_width == 0:
        return output, [[] for _ in output], audit
    gather_indices_cpu = torch.zeros((len(output), max_width), dtype=torch.long)
    for row_idx, selected in enumerate(output):
        if selected:
            gather_indices_cpu[row_idx, : len(selected)] = torch.tensor(selected, dtype=torch.long)
    gathered_scores = (
        logits[:, :skill_count]
        .gather(1, gather_indices_cpu.to(logits.device))
        .detach()
        .cpu()
    )
    output_scores = [
        [float(value) for value in gathered_scores[row_idx, : len(selected)].tolist()]
        for row_idx, selected in enumerate(output)
    ]
    return output, output_scores, audit


def _compute_stage0_raw_candidates_exact_schedule(
    model: Any,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    top_m: int,
    query_mode: str,
    encode_batch_size: int = 8,
    device: torch.device | None = None,
    setup_status_path: str | Path | None = None,
    progress_interval_batches: int = 100,
    inventory_min_candidates: int = 0,
) -> tuple[list[RawStage0Candidates], dict[str, Any]]:
    if query_mode not in STAGE0_HANDOFF_QUERY_MODES:
        raise ValueError(f"unsupported Stage0 handoff query mode: {query_mode}")
    skill_count = len(skills)
    if skill_count <= 0:
        raise ValueError("Stage0 handoff raw candidates require non-empty skill pool")
    k = min(max(1, int(top_m)), skill_count)
    encode_batch_size = max(1, int(encode_batch_size))
    progress_interval_batches = max(1, int(progress_interval_batches))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_query_count = 2 * len(rows)
    total_scoring_batches = (len(rows) + encode_batch_size - 1) // encode_batch_size
    report: dict[str, Any] = {
        "mode": "legacy_schedule_vectorized_scores",
        "source_rows": len(rows),
        "source_query_count": source_query_count,
        "encoded_query_count": source_query_count,
        "encode_batch_size": encode_batch_size,
        "execution_schedule_version": LEGACY_EXECUTION_SCHEDULE_VERSION,
        "scoring_batches": total_scoring_batches,
    }
    if not rows:
        report["inventory_audit"] = {}
        return [], report

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    inventory_audit: Counter[str] = Counter()
    raw_candidates: list[RawStage0Candidates] = []
    try:
        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(rows), encode_batch_size), start=1):
                end = min(len(rows), start + encode_batch_size)
                batch = rows[start:end]
                current_queries = [
                    _stage0_handoff_query_text(row, target="current", mode=query_mode)
                    for row in batch
                ]
                next_queries = [
                    _stage0_handoff_query_text(row, target="next", mode=query_mode)
                    for row in batch
                ]
                current_h = _encode_stage0_handoff_queries(
                    model,
                    current_queries,
                    query_mode=query_mode,
                ).to(device)
                current_logits = _stage0_unified_static_logits(model, current_h, skill_count)
                next_h = _encode_stage0_handoff_queries(
                    model,
                    next_queries,
                    query_mode=query_mode,
                ).to(device)
                next_logits = _stage0_unified_static_logits(model, next_h, skill_count)
                current_top, current_scores, current_audit = (
                    _stage0_topk_with_explicit_inventory_vectorized_scores(
                        current_logits[:, :skill_count],
                        batch,
                        skill_id_to_idx=skill_id_to_idx,
                        skill_count=skill_count,
                        k=k,
                        min_candidates=inventory_min_candidates,
                    )
                )
                next_top, next_scores, next_audit = (
                    _stage0_topk_with_explicit_inventory_vectorized_scores(
                        next_logits[:, :skill_count],
                        batch,
                        skill_id_to_idx=skill_id_to_idx,
                        skill_count=skill_count,
                        k=k,
                        min_candidates=inventory_min_candidates,
                    )
                )
                inventory_audit.update(current_audit)
                inventory_audit.update(next_audit)
                raw_candidates.extend(
                    RawStage0Candidates(
                        current_indices=tuple(current_row),
                        current_scores=tuple(current_score_row),
                        next_indices=tuple(next_row),
                        next_scores=tuple(next_score_row),
                    )
                    for current_row, current_score_row, next_row, next_score_row in zip(
                        current_top,
                        current_scores,
                        next_top,
                        next_scores,
                        strict=True,
                    )
                )
                if setup_status_path is not None and (
                    batch_idx == 1
                    or batch_idx % progress_interval_batches == 0
                    or batch_idx == total_scoring_batches
                ):
                    append_setup_status(
                        setup_status_path,
                        "stage0_candidate_handoff_progress",
                        rows_done=end,
                        source_rows=len(rows),
                        batch_idx=batch_idx,
                        total_batches=total_scoring_batches,
                        progress=end / max(len(rows), 1),
                    )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    report["inventory_audit"] = dict(inventory_audit)
    return raw_candidates, report


def _attach_stage0_topm_candidates(
    model: Any,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    top_m: int | None,
    positive_missing_policy: str = "skip",
    query_mode: str = "skillrouter_state",
    allow_full_pool_stage2_debug: bool = False,
    routing_checkpoint_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    encode_batch_size: int = 8,
    device: torch.device | None = None,
    setup_status_path: str | Path | None = None,
    progress_interval_batches: int = 100,
    inventory_min_candidates: int = 0,
    next_skill_pool_mode: str = "stage0_candidates",
    raw_candidates: list[RawStage0Candidates] | None = None,
    raw_candidate_report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = len(rows)
    tool_inventory_backfill_report = {
        "enabled": False,
        "reason": "disabled_for_official_handoff_no_gt_inventory",
    }
    if query_mode not in STAGE0_HANDOFF_QUERY_MODES:
        raise ValueError(f"unsupported Stage0 handoff query mode: {query_mode}")
    next_skill_pool_mode = str(next_skill_pool_mode or "stage0_candidates")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    if top_m is None or int(top_m) <= 0:
        if not allow_full_pool_stage2_debug:
            raise ValueError("Stage2 requires Stage0 top-M candidates unless allow_full_pool_stage2_debug is enabled")
        report = _stage0_topm_handoff_report(
            stage0_checkpoint=routing_checkpoint_path,
            top_m=None,
            positive_missing_policy=positive_missing_policy,
            query_mode=query_mode,
            allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
            source_rows=source_rows,
            retained_rows=source_rows,
            skipped_reasons=Counter(),
            injected_rows=0,
            tool_inventory_backfill_report=tool_inventory_backfill_report,
            manifest_path=manifest_path,
        )
        report["next_skill_pool_mode"] = next_skill_pool_mode
        if manifest_path is not None:
            write_json(manifest_path, report)
        return rows, report

    top_m = max(1, int(top_m))
    policy = str(positive_missing_policy)
    if policy not in {"skip", "inject", "skip_or_inject_with_provenance"}:
        raise ValueError(f"unsupported stage0 positive missing policy: {positive_missing_policy}")
    inject_missing = (
        policy in {"inject", "skip_or_inject_with_provenance"}
        and next_skill_pool_mode == "stage0_candidates"
    )
    skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
    skill_count = len(skill_ids)
    if skill_count <= 0:
        raise ValueError("Stage2 candidate handoff requires non-empty skill pool")
    k = min(top_m, skill_count)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_top: list[list[int]] = []
    next_top: list[list[int]] = []
    state_top_scores: list[list[float]] = []
    next_top_scores: list[list[float]] = []
    encode_batch_size = max(1, int(encode_batch_size))
    total_batches = (len(rows) + encode_batch_size - 1) // encode_batch_size
    progress_interval_batches = max(1, int(progress_interval_batches))
    if setup_status_path is not None:
        append_setup_status(
            setup_status_path,
            "stage0_candidate_handoff_started",
            source_rows=len(rows),
            skill_count=skill_count,
            top_m=k,
            encode_batch_size=encode_batch_size,
            total_batches=total_batches,
        )
    inventory_audit: Counter[str] = Counter()
    if raw_candidates is None:
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        try:
            with torch.no_grad():
                for batch_idx, start in enumerate(range(0, len(rows), encode_batch_size), start=1):
                    batch = rows[start : start + encode_batch_size]
                    state_texts = [_stage0_handoff_query_text(row, target="current", mode=query_mode) for row in batch]
                    next_texts = [_stage0_handoff_query_text(row, target="next", mode=query_mode) for row in batch]
                    state_h = _encode_stage0_handoff_queries(
                        model,
                        state_texts,
                        query_mode=query_mode,
                    ).to(device)
                    state_logits = _stage0_unified_static_logits(model, state_h, skill_count)
                    next_h = _encode_stage0_handoff_queries(
                        model,
                        next_texts,
                        query_mode=query_mode,
                    ).to(device)
                    next_logits = _stage0_unified_static_logits(model, next_h, skill_count)
                    state_batch_top, state_batch_scores, state_inventory_audit = _stage0_topk_with_explicit_inventory(
                        state_logits[:, :skill_count],
                        batch,
                        skill_id_to_idx=skill_id_to_idx,
                        skill_count=skill_count,
                        k=k,
                        min_candidates=inventory_min_candidates,
                    )
                    next_batch_top, next_batch_scores, next_inventory_audit = _stage0_topk_with_explicit_inventory(
                        next_logits[:, :skill_count],
                        batch,
                        skill_id_to_idx=skill_id_to_idx,
                        skill_count=skill_count,
                        k=k,
                        min_candidates=inventory_min_candidates,
                    )
                    state_top.extend(state_batch_top)
                    next_top.extend(next_batch_top)
                    state_top_scores.extend(state_batch_scores)
                    next_top_scores.extend(next_batch_scores)
                    inventory_audit.update(state_inventory_audit)
                    inventory_audit.update(next_inventory_audit)
                    if setup_status_path is not None and (
                        batch_idx == 1 or batch_idx % progress_interval_batches == 0 or batch_idx == total_batches
                    ):
                        rows_done = min(len(rows), start + len(batch))
                        append_setup_status(
                            setup_status_path,
                            "stage0_candidate_handoff_progress",
                            rows_done=rows_done,
                            source_rows=len(rows),
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            progress=rows_done / max(len(rows), 1),
                        )
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
    else:
        if len(raw_candidates) != len(rows):
            raise ValueError("raw Stage0 candidate count must match source rows")
        for raw in raw_candidates:
            state_top.append(list(raw.current_indices))
            state_top_scores.append(list(raw.current_scores))
            next_top.append(list(raw.next_indices))
            next_top_scores.append(list(raw.next_scores))
        inventory_audit.update((raw_candidate_report or {}).get("inventory_audit") or {})

    retained: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    injected_rows = 0
    current_positive_required_rows = 0
    current_positive_covered_rows = 0
    next_positive_required_rows = 0
    next_positive_covered_rows = 0
    stage0_topm_next_positive_covered_rows = 0
    masked_next_skill_ce_rows = 0
    current_skill_candidate_added_rows = 0
    current_skill_candidate_positive_rows = 0
    current_skill_candidate_hard_negative_rows = 0
    next_positive_rank_bucket_counts: Counter[str] = Counter()
    learnable_correction_rows_rank_2_to_100 = 0
    for row, current_candidates_raw, current_scores_raw, next_candidates_raw, next_scores_raw in zip(
        rows,
        state_top,
        state_top_scores,
        next_top,
        next_top_scores,
        strict=True,
    ):
        current_candidates = [int(idx) for idx in current_candidates_raw]
        next_candidates = [int(idx) for idx in next_candidates_raw]
        current_candidate_scores = [float(score) for score in current_scores_raw]
        next_candidate_scores = [float(score) for score in next_scores_raw]
        current_injected = False
        next_injected = False
        current_skill_candidate_added = False
        current_skill_candidate_role = "none"
        current_positive_hit: bool | None = None
        next_positive_hit: bool | None = None
        next_skill_ce_masked = False
        missing_reason: str | None = None
        loss_mask = row.get("loss_mask") or {}
        skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if loss_mask.get("routing") and skill_id in skill_id_to_idx:
            current_positive_required_rows += 1
            positive = int(skill_id_to_idx[skill_id])
            current_positive_hit = positive in current_candidates
            if positive not in current_candidates:
                if next_skill_pool_mode == "full_pool":
                    loss_mask = dict(loss_mask)
                    loss_mask["routing"] = False
                elif inject_missing:
                    old_candidates = list(current_candidates)
                    current_candidates, current_injected = _inject_positive_candidate(current_candidates, positive, k)
                    current_score_map = dict(zip(old_candidates, current_candidate_scores))
                    fallback_score = min(current_candidate_scores) - 1.0 if current_candidate_scores else 0.0
                    current_candidate_scores = [
                        float(current_score_map.get(int(idx), fallback_score))
                        for idx in current_candidates
                    ]
                else:
                    missing_reason = "routing_positive_missing_from_stage0_topm"
            else:
                current_positive_covered_rows += 1
        if (
            next_skill_pool_mode == "stage0_candidates"
            and missing_reason is None
            and loss_mask.get("L_trans_skill_ce")
            and skill_id in skill_id_to_idx
        ):
            current_skill_idx = int(skill_id_to_idx[skill_id])
            if current_skill_idx not in next_candidates:
                next_candidates.append(current_skill_idx)
                fallback_score = min(next_candidate_scores) - 1.0 if next_candidate_scores else 0.0
                next_candidate_scores.append(float(fallback_score))
                current_skill_candidate_added = True
                current_skill_candidate_added_rows += 1
            if next_skill_id == skill_id:
                current_skill_candidate_role = "positive"
                current_skill_candidate_positive_rows += 1
            else:
                current_skill_candidate_role = "hard_negative"
                current_skill_candidate_hard_negative_rows += 1
        if missing_reason is None and loss_mask.get("L_trans_skill_ce") and next_skill_id in skill_id_to_idx:
            next_positive_required_rows += 1
            positive = int(skill_id_to_idx[next_skill_id])
            next_positive_hit = positive in [int(idx) for idx in next_candidates_raw]
            if next_positive_hit:
                stage0_topm_next_positive_covered_rows += 1
            if positive not in next_candidates:
                if next_skill_pool_mode == "full_pool":
                    next_positive_rank_bucket_counts["missing"] += 1
                elif inject_missing:
                    old_candidates = list(next_candidates)
                    next_candidates, next_injected = _inject_positive_candidate(next_candidates, positive, k)
                    next_score_map = dict(zip(old_candidates, next_candidate_scores))
                    fallback_score = min(next_candidate_scores) - 1.0 if next_candidate_scores else 0.0
                    next_candidate_scores = [
                        float(next_score_map.get(int(idx), fallback_score))
                        for idx in next_candidates
                    ]
                else:
                    loss_mask = dict(loss_mask)
                    loss_mask["L_trans_skill_ce"] = False
                    next_skill_ce_masked = True
                    masked_next_skill_ce_rows += 1
                    next_positive_rank_bucket_counts["missing"] += 1
            else:
                next_positive_covered_rows += 1
            if positive in next_candidates:
                positive_rank = next_candidates.index(positive) + 1
                next_positive_rank_bucket_counts[_rank_bucket(positive_rank)] += 1
                if 2 <= positive_rank <= 100:
                    learnable_correction_rows_rank_2_to_100 += 1
        if missing_reason is not None:
            skipped[missing_reason] += 1
            continue
        copied = dict(row)
        if loss_mask is not row.get("loss_mask"):
            copied["loss_mask"] = loss_mask
        copied["stage0_candidate_skill_indices"] = current_candidates
        copied["stage0_candidate_skill_ids"] = [skill_ids[idx] for idx in current_candidates]
        copied["stage0_candidate_skill_scores"] = current_candidate_scores
        copied["stage0_next_candidate_skill_indices"] = next_candidates
        copied["stage0_next_candidate_skill_ids"] = [skill_ids[idx] for idx in next_candidates]
        copied["stage0_next_candidate_skill_scores"] = next_candidate_scores
        copied["stage0_candidate_source"] = "stage0_topm_online"
        copied["stage0_current_positive_hit"] = current_positive_hit
        copied["stage0_next_positive_hit"] = next_positive_hit
        copied["stage0_positive_injected"] = bool(current_injected or next_injected)
        copied["stage0_current_skill_candidate_added"] = bool(current_skill_candidate_added)
        copied["stage0_current_skill_candidate_role"] = current_skill_candidate_role
        loss_mask_changed = loss_mask is not row.get("loss_mask")
        if current_injected or next_injected or loss_mask_changed or current_skill_candidate_added:
            provenance = dict(_provenance(copied))
            provenance["stage0_candidate_handoff"] = {
                "positive_missing_policy": policy,
                "query_mode": query_mode,
                "current_positive_injected": bool(current_injected),
                "next_positive_injected": bool(next_injected),
                "next_skill_ce_masked": bool(next_skill_ce_masked),
                "current_skill_candidate_added": bool(current_skill_candidate_added),
                "current_skill_candidate_role": current_skill_candidate_role,
            }
            copied["provenance"] = provenance
        if current_injected or next_injected:
            injected_rows += 1
        retained.append(copied)

    report = _stage0_topm_handoff_report(
        stage0_checkpoint=routing_checkpoint_path,
        top_m=k,
        positive_missing_policy=policy,
        query_mode=query_mode,
        allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
        source_rows=source_rows,
        retained_rows=len(retained),
        skipped_reasons=skipped,
        injected_rows=injected_rows,
        current_positive_required_rows=current_positive_required_rows,
        current_positive_covered_rows=current_positive_covered_rows,
        next_positive_required_rows=next_positive_required_rows,
        next_positive_covered_rows=next_positive_covered_rows,
        stage0_topm_next_positive_covered_rows=stage0_topm_next_positive_covered_rows,
        masked_next_skill_ce_rows=masked_next_skill_ce_rows,
        current_skill_candidate_added_rows=current_skill_candidate_added_rows,
        current_skill_candidate_positive_rows=current_skill_candidate_positive_rows,
        current_skill_candidate_hard_negative_rows=current_skill_candidate_hard_negative_rows,
        next_positive_rank_bucket_counts=next_positive_rank_bucket_counts,
        learnable_correction_rows_rank_2_to_100=learnable_correction_rows_rank_2_to_100,
        inventory_candidate_rows=inventory_audit["inventory_candidate_rows"],
        inventory_candidate_missing_rows=inventory_audit["inventory_candidate_missing_rows"],
        inventory_candidate_removed_candidates=inventory_audit["inventory_candidate_removed_candidates"],
        inventory_candidate_topk_backfilled_rows=inventory_audit["inventory_candidate_topk_backfilled_rows"],
        inventory_candidate_topk_backfilled_candidates=inventory_audit["inventory_candidate_topk_backfilled_candidates"],
        tool_inventory_backfill_report=tool_inventory_backfill_report,
        manifest_path=manifest_path,
    )
    report["encode_batch_size"] = encode_batch_size
    report["candidate_selection_version"] = CANDIDATE_SELECTION_VERSION
    report["tie_break_policy"] = TIE_BREAK_POLICY
    report["initial_belief_top_k"] = getattr(getattr(model, "config", None), "initial_belief_top_k", None)
    report["next_skill_pool_mode"] = next_skill_pool_mode
    if raw_candidate_report is not None:
        report["raw_candidate_computation"] = dict(raw_candidate_report)
    if manifest_path is not None:
        write_json(manifest_path, report)
    return retained, report


def _inventory_audit_from_raw_candidates(
    rows: list[dict[str, Any]],
    raw_candidates: list[RawStage0Candidates],
    *,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
) -> Counter[str]:
    if len(rows) != len(raw_candidates):
        raise ValueError("raw Stage0 inventory audit requires one candidate record per row")
    audit: Counter[str] = Counter()
    for row, raw in zip(rows, raw_candidates, strict=True):
        inventory_indices = {
            int(skill_id_to_idx[skill_id])
            for skill_id in _explicit_inventory_skill_ids_ordered(row)
            if skill_id in skill_id_to_idx
        }
        for candidate_indices in (raw.current_indices, raw.next_indices):
            if not inventory_indices:
                audit["inventory_candidate_missing_rows"] += 1
                continue
            audit["inventory_candidate_rows"] += 1
            audit["inventory_candidate_removed_candidates"] += max(
                0,
                int(skill_count) - len(inventory_indices),
            )
            backfilled = sum(1 for idx in candidate_indices if int(idx) not in inventory_indices)
            if backfilled > 0:
                audit["inventory_candidate_topk_backfilled_rows"] += 1
                audit["inventory_candidate_topk_backfilled_candidates"] += backfilled
    return audit


def _prepare_stage0_topm_candidates_with_cache(
    model: Any,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    top_m: int | None,
    positive_missing_policy: str = "skip",
    query_mode: str = "skillrouter_state",
    allow_full_pool_stage2_debug: bool = False,
    routing_checkpoint_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    encode_batch_size: int = 8,
    device: torch.device | None = None,
    setup_status_path: str | Path | None = None,
    progress_interval_batches: int = 100,
    inventory_min_candidates: int = 0,
    next_skill_pool_mode: str = "stage0_candidates",
    cache_mode: str = "off",
    cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    model_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_mode = str(cache_mode or "off")
    cache_format = str(cache_format or LEGACY_STAGE0_HANDOFF_CACHE_FORMAT)
    cache_shard_size = int(cache_shard_size)
    if cache_mode not in STAGE0_HANDOFF_CACHE_MODES:
        raise ValueError(f"unsupported stage0_handoff_cache_mode: {cache_mode}")
    if cache_format not in STAGE0_HANDOFF_CACHE_FORMATS:
        raise ValueError(f"unsupported stage0_handoff_cache_format: {cache_format}")
    if cache_shard_size <= 0:
        raise ValueError("stage0_handoff_cache_shard_size must be positive")
    if cache_format == LEGACY_STAGE0_HANDOFF_CACHE_FORMAT:
        cache_report: dict[str, Any] = {
            "mode": cache_mode,
            "format": cache_format,
            "enabled": cache_mode != "off",
            "cache_hit": False,
        }
        cache_key: dict[str, Any] | None = None
        cached_handoff: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
        if cache_mode != "off" and top_m is not None and int(top_m) > 0:
            skill_table = getattr(model, "skill_table", None)
            skill_embeddings = getattr(skill_table, "E", None)
            if not isinstance(skill_embeddings, torch.Tensor):
                raise ValueError("Stage0 handoff cache requires model.skill_table.E")
            effective_model_config = getattr(model, "config", None)
            fallback_config = model_config or {}
            initial_belief_top_k = getattr(
                effective_model_config,
                "initial_belief_top_k",
                fallback_config.get("initial_belief_top_k"),
            )
            state_text_format = getattr(
                effective_model_config,
                "state_text_format",
                fallback_config.get("state_text_format", "default"),
            )
            effective_skill_text_format = getattr(
                effective_model_config,
                "skill_text_format",
                fallback_config.get("skill_text_format", "default"),
            )
            cache_key = _stage0_handoff_cache_key(
                rows_digest=_stage0_handoff_rows_digest(rows),
                skills_digest=_stage0_handoff_skills_digest(skills),
                checkpoint_digest=_path_content_digest(routing_checkpoint_path),
                top_m=top_m,
                positive_missing_policy=positive_missing_policy,
                query_mode=query_mode,
                inventory_min_candidates=inventory_min_candidates,
                allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
                static_candidate_scorer="unified_static",
                initial_belief_top_k=initial_belief_top_k,
                route_scorer=route_scorer,
                state_text_format=str(state_text_format),
                skill_text_format=str(effective_skill_text_format),
                skill_embedding_digest=_tensor_content_digest(skill_embeddings),
                declared_pool_order_digest=_stable_json_digest(
                    [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
                ),
                candidate_selection_version=CANDIDATE_SELECTION_VERSION,
                tie_break_policy=TIE_BREAK_POLICY,
                unified_static_scorer_digest=_effective_unified_static_scorer_digest(model),
                next_skill_pool_mode=next_skill_pool_mode,
            )
            cache_report.update(
                {
                    "cache_key": cache_key["cache_key"],
                    "cache_dir": str(cache_dir),
                }
            )
            if setup_status_path is not None:
                append_setup_status(
                    setup_status_path,
                    "stage0_candidate_handoff_cache_lookup",
                    **cache_report,
                )
            if cache_mode != "refresh":
                cached_handoff = _load_stage0_handoff_cache(cache_dir, cache_key)
        if cached_handoff is not None:
            cached_rows, cached_report = cached_handoff
            merged_rows, merge_report = _merge_stage0_handoff_cached_rows(rows, cached_rows)
            if merge_report["valid"]:
                report = dict(cached_report)
                report["manifest_path"] = None if manifest_path is None else str(manifest_path)
                report["cache_merge"] = merge_report
                cache_report.update(
                    {
                        "cache_hit": True,
                        "cache_path": report.get("cache_path"),
                        "retained_rows": len(merged_rows),
                        "cache_merge": merge_report,
                    }
                )
                report["cache"] = cache_report
                if manifest_path is not None:
                    write_json(manifest_path, report)
                if setup_status_path is not None:
                    append_setup_status(
                        setup_status_path,
                        "stage0_candidate_handoff_cache_hit",
                        **cache_report,
                    )
                return merged_rows, report
            cache_report.update(
                {
                    "cache_invalidated": True,
                    "cache_invalidation_reason": "cached_rows_do_not_match_current_next_state_identity",
                    "cache_merge": merge_report,
                }
            )
            if setup_status_path is not None:
                append_setup_status(
                    setup_status_path,
                    "stage0_candidate_handoff_cache_invalidated",
                    **cache_report,
                )

        retained, report = _attach_stage0_topm_candidates(
            model,
            rows,
            skills,
            skill_id_to_idx,
            top_m=top_m,
            positive_missing_policy=positive_missing_policy,
            query_mode=query_mode,
            allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
            routing_checkpoint_path=routing_checkpoint_path,
            manifest_path=manifest_path,
            encode_batch_size=encode_batch_size,
            device=device,
            setup_status_path=setup_status_path,
            progress_interval_batches=progress_interval_batches,
            inventory_min_candidates=inventory_min_candidates,
            next_skill_pool_mode=next_skill_pool_mode,
        )
        if cache_key is not None:
            for key in (
                "static_candidate_scorer",
                "initial_belief_top_k",
                "route_scorer",
                "state_text_format",
                "skill_text_format",
                "checkpoint_digest",
                "skills_digest",
                "skill_embedding_digest",
                "declared_pool_order_digest",
                "candidate_selection_version",
                "tie_break_policy",
                "unified_static_scorer_digest",
                "next_skill_pool_mode",
            ):
                report[key] = cache_key[key]
            write_report = _write_stage0_handoff_cache(
                cache_dir,
                cache_key,
                retained,
                report,
            )
            cache_report.update(write_report)
            if setup_status_path is not None:
                append_setup_status(
                    setup_status_path,
                    "stage0_candidate_handoff_cache_written",
                    **cache_report,
                )
        report["cache"] = cache_report
        if manifest_path is not None:
            write_json(manifest_path, report)
        return retained, report

    if top_m is None or int(top_m) <= 0:
        retained, report = _attach_stage0_topm_candidates(
            model,
            rows,
            skills,
            skill_id_to_idx,
            top_m=top_m,
            positive_missing_policy=positive_missing_policy,
            query_mode=query_mode,
            allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
            routing_checkpoint_path=routing_checkpoint_path,
            manifest_path=manifest_path,
            encode_batch_size=encode_batch_size,
            device=device,
            setup_status_path=setup_status_path,
            progress_interval_batches=progress_interval_batches,
            inventory_min_candidates=inventory_min_candidates,
            next_skill_pool_mode=next_skill_pool_mode,
        )
        report["cache"] = {
            "mode": cache_mode,
            "format": cache_format,
            "enabled": False,
            "cache_hit": False,
        }
        if manifest_path is not None:
            write_json(manifest_path, report)
        return retained, report

    top_m = max(1, int(top_m))
    encode_batch_size = max(1, int(encode_batch_size))
    skill_count = len(skills)
    if skill_count <= 0:
        raise ValueError("Stage0 handoff cache requires non-empty skill pool")
    candidate_count = min(top_m, skill_count)
    cache_enabled = cache_mode != "off"
    current_queries = [
        _stage0_handoff_query_text(row, target="current", mode=query_mode)
        for row in rows
    ]
    next_queries = [
        _stage0_handoff_query_text(row, target="next", mode=query_mode)
        for row in rows
    ]
    ordered_inventories = [
        _explicit_inventory_skill_ids_ordered(row)
        for row in rows
    ]
    schedule_plan = build_legacy_schedule_row_key_plan(
        current_queries,
        next_queries,
        ordered_inventories,
        batch_size=encode_batch_size,
    )
    row_keys = list(schedule_plan.row_keys)
    global_identity: dict[str, object] | None = None
    hits: dict[str, RawStage0Candidates] = {}
    cache_lookup_report: dict[str, object] = {
        "format": cache_format,
        "cache_path": str(cache_dir),
        "cached_rows": 0,
        "hit_rows": 0,
        "miss_rows": len(row_keys),
    }
    if cache_enabled:
        skill_table = getattr(model, "skill_table", None)
        skill_embeddings = getattr(skill_table, "E", None)
        if not isinstance(skill_embeddings, torch.Tensor):
            raise ValueError("Stage0 handoff cache requires model.skill_table.E")
        effective_config = getattr(model, "config", None)
        checkpoint_digest = _path_content_digest(routing_checkpoint_path)
        global_identity = stage0_handoff_global_identity(
            checkpoint_digest={
                key: checkpoint_digest.get(key)
                for key in ("exists", "file_count", "size", "sha256")
                if key in checkpoint_digest
            },
            skills_digest=_stage0_handoff_skills_digest(skills),
            top_m=top_m,
            candidate_count=candidate_count,
            query_mode=query_mode,
            inventory_min_candidates=inventory_min_candidates,
            initial_belief_top_k=getattr(effective_config, "initial_belief_top_k", None),
            state_text_format=str(getattr(effective_config, "state_text_format", "default")),
            skill_text_format=str(getattr(effective_config, "skill_text_format", "default")),
            state_query_prompt_version=str(
                getattr(effective_config, "state_query_prompt_version", "unspecified")
            ),
            skill_embedding_digest=_tensor_content_digest(skill_embeddings),
            declared_pool_order_digest=_stable_json_digest(
                [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
            ),
            candidate_selection_version=CANDIDATE_SELECTION_VERSION,
            tie_break_policy=TIE_BREAK_POLICY,
            unified_static_scorer_digest=_effective_unified_static_scorer_digest(model),
            encode_batch_size=encode_batch_size,
        )
        if cache_mode == "refresh":
            reset_row_sharded_cache(cache_dir, global_identity)
        hits, cache_lookup_report = load_row_sharded_cache(
            cache_dir,
            global_identity,
            row_keys,
        )

    raw_hit_rows = int(cache_lookup_report.get("hit_rows") or 0)
    missing_batch_ranges: list[tuple[int, int]] = []
    effective_hit_rows = 0
    for start, end in schedule_plan.batch_ranges:
        batch_keys = row_keys[start:end]
        if all(row_key in hits for row_key in batch_keys):
            effective_hit_rows += end - start
            continue
        missing_batch_ranges.append((start, end))
        for row_key in batch_keys:
            hits.pop(row_key, None)
    computed_row_count = sum(end - start for start, end in missing_batch_ranges)
    cache_lookup_report = dict(cache_lookup_report)
    cache_lookup_report["raw_hit_rows"] = raw_hit_rows
    cache_lookup_report["partial_hit_rows_discarded"] = max(0, raw_hit_rows - effective_hit_rows)
    cache_lookup_report["hit_rows"] = effective_hit_rows
    cache_lookup_report["miss_rows"] = computed_row_count

    computed_records: dict[str, RawStage0Candidates] = {}
    if missing_batch_ranges:
        missing_rows = [
            row
            for start, end in missing_batch_ranges
            for row in rows[start:end]
        ]
        missing_keys = [
            row_key
            for start, end in missing_batch_ranges
            for row_key in row_keys[start:end]
        ]
        missing_raw, compute_report = _compute_stage0_raw_candidates_exact_schedule(
            model,
            missing_rows,
            skills,
            skill_id_to_idx,
            top_m=top_m,
            query_mode=query_mode,
            encode_batch_size=encode_batch_size,
            device=device,
            setup_status_path=setup_status_path,
            progress_interval_batches=progress_interval_batches,
            inventory_min_candidates=inventory_min_candidates,
        )
        for row_key, record in zip(missing_keys, missing_raw, strict=True):
            existing = computed_records.get(row_key)
            if existing is not None and existing != record:
                raise ValueError(f"row_sharded_v1 conflicting computed schedule key: {row_key}")
            computed_records[row_key] = record
        compute_report = dict(compute_report)
        compute_report.update(
            {
                "schedule_batches_total": len(schedule_plan.batch_ranges),
                "schedule_batches_computed": len(missing_batch_ranges),
            }
        )
    else:
        compute_report = {
            "mode": "row_sharded_cache_hits_only",
            "source_rows": 0,
            "source_query_count": 0,
            "encoded_query_count": 0,
            "encode_batch_size": encode_batch_size,
            "execution_schedule_version": LEGACY_EXECUTION_SCHEDULE_VERSION,
            "scoring_batches": 0,
            "schedule_batches_total": len(schedule_plan.batch_ranges),
            "schedule_batches_computed": 0,
        }
    write_report: dict[str, object] = {}
    if cache_enabled and computed_records:
        if global_identity is None:
            raise RuntimeError("row_sharded_v1 cache identity was not prepared")
        write_report = append_row_sharded_cache(
            cache_dir,
            global_identity,
            computed_records,
            shard_size=cache_shard_size,
        )
    raw_candidates = [
        hits[row_key] if row_key in hits else computed_records[row_key]
        for row_key in row_keys
    ]
    compute_report = dict(compute_report)
    compute_report["inventory_audit"] = dict(
        _inventory_audit_from_raw_candidates(
            rows,
            raw_candidates,
            skill_id_to_idx=skill_id_to_idx,
            skill_count=skill_count,
        )
    )
    retained, report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=top_m,
        positive_missing_policy=positive_missing_policy,
        query_mode=query_mode,
        allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
        routing_checkpoint_path=routing_checkpoint_path,
        manifest_path=None,
        encode_batch_size=encode_batch_size,
        device=device,
        setup_status_path=setup_status_path,
        progress_interval_batches=progress_interval_batches,
        inventory_min_candidates=inventory_min_candidates,
        next_skill_pool_mode=next_skill_pool_mode,
        raw_candidates=raw_candidates,
        raw_candidate_report=compute_report,
    )
    cache_report = {
        "mode": cache_mode,
        "format": cache_format,
        "enabled": cache_enabled,
        "cache_hit": bool(cache_enabled and int(cache_lookup_report["miss_rows"]) == 0),
        "cache_dir": str(cache_dir),
        "cache_key": None if global_identity is None else global_identity["global_key"],
        "cache_path": cache_lookup_report.get("cache_path"),
        "cached_rows": int(cache_lookup_report.get("cached_rows") or 0),
        "hit_rows": int(cache_lookup_report.get("hit_rows") or 0),
        "miss_rows": int(cache_lookup_report.get("miss_rows") or 0),
        "raw_hit_rows": int(cache_lookup_report.get("raw_hit_rows") or 0),
        "partial_hit_rows_discarded": int(cache_lookup_report.get("partial_hit_rows_discarded") or 0),
        "computed_rows": computed_row_count,
        "schedule_batches": len(schedule_plan.batch_ranges),
        "computed_batches": len(missing_batch_ranges),
        "unique_requested_rows": len(set(row_keys)),
        "shard_size": cache_shard_size,
        **write_report,
    }
    report["cache"] = cache_report
    if global_identity is not None:
        report["raw_candidate_cache_identity"] = global_identity
    if manifest_path is not None:
        write_json(manifest_path, report)
    return retained, report


def _available_actions_planner_state_text(state_text: str, admissible_actions: list[str]) -> str:
    action_lines = "\n".join(f"{idx}. {action}" for idx, action in enumerate(admissible_actions, start=1))
    return "\n".join(
        [
            str(state_text),
            "",
            "AVAILABLE ACTIONS:",
            action_lines,
            "",
            "CLSTR skill-routing query:",
            "Infer the high-level skill intent and target objects/receptacles needed for the next step.",
            "Do not directly execute or copy an action; CLSTR will rank the admissible actions.",
        ]
    )


def _augment_rows_with_available_actions_planner_context(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    count = 0
    for row in rows:
        new_row = dict(row)
        loss_mask = new_row.get("loss_mask") or {}
        candidates = [str(item) for item in new_row.get("admissible_actions") or []]
        if loss_mask.get("L_policy") and candidates:
            base_state_text = str(
                new_row.get("_available_actions_base_state_text")
                or router_state_text(new_row)
                or ""
            )
            new_row["_available_actions_base_state_text"] = base_state_text
            routed_state_text = _available_actions_planner_state_text(base_state_text, candidates)
            new_row["state_text"] = routed_state_text
            new_row["state_text_current"] = routed_state_text
            new_row["available_actions_planner_context"] = True
            count += 1
        augmented.append(new_row)
    return augmented, {
        "enabled": True,
        "mode": "latent_available_actions_query",
        "augmented_row_count": count,
        "row_count": len(rows),
    }


def _set_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(trainable)


def _set_belief_calibration_trainable(model: Any, trainable_modules: list[str]) -> None:
    skill_table = getattr(model, "skill_table", None)
    if skill_table is None:
        return
    for key in ("logit_scale_belief", "skill_bias_belief"):
        param = getattr(skill_table, key, None)
        if isinstance(param, torch.nn.Parameter):
            param.requires_grad_(True)
            module_name = f"skill_table.{key}"
            if module_name not in trainable_modules:
                trainable_modules.append(module_name)


def _freeze_for_full_base(
    model: Any,
    *,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    freeze_gated_temporal_only: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    anchored_routing_foundation: bool = False,
    active_loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    effective_transition_scoring_mode = (
        UNIFIED_MEMORY_ROUTE_SCORER
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else str(transition_scoring_mode)
    )
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad_(False)
    trainable_modules: list[str] = []
    if (
        bool(freeze_gated_temporal_only)
        and effective_transition_scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE
    ):
        _set_trainable(getattr(model, "gated_temporal_reranker", None), True)
        if getattr(model, "gated_temporal_reranker", None) is not None:
            trainable_modules.append("gated_temporal_reranker")
        _set_belief_calibration_trainable(model, trainable_modules)
        return {
            "frozen_routing_foundation": True,
            "qwen_and_skill_table_frozen": True,
            "frozen_modules": ["encoder.backbone", "encoder.proj", "skill_table retrieval path", "legacy CLSTR heads"],
            "trainable_modules": trainable_modules,
            "freeze_gated_temporal_only": True,
        }
    _set_trainable(getattr(model, "transition", None), True)
    if getattr(model, "transition", None) is not None:
        trainable_modules.append("transition")
    _set_trainable(getattr(model, "gate", None), True)
    if getattr(model, "gate", None) is not None:
        trainable_modules.append("gate")
    stop_enabled = active_loss_weights is None or float(active_loss_weights.get("STOP", 0.0)) > 0.0
    _set_trainable(getattr(model, "stop_head", None), stop_enabled)
    if stop_enabled and getattr(model, "stop_head", None) is not None:
        trainable_modules.append("stop_head")
    _set_trainable(getattr(model, "skill_head", None), True)
    if getattr(model, "skill_head", None) is not None:
        trainable_modules.append("skill_head")
    _set_trainable(getattr(model, "trans_head", None), route_scorer != UNIFIED_MEMORY_ROUTE_SCORER)
    if route_scorer != UNIFIED_MEMORY_ROUTE_SCORER and getattr(model, "trans_head", None) is not None:
        trainable_modules.append("trans_head")
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER and anchored_routing_foundation:
        _set_trainable(getattr(model, "initial_belief_head", None), True)
        if getattr(model, "initial_belief_head", None) is not None:
            trainable_modules.append("initial_belief_head")
        _set_trainable(getattr(model, "unified_retriever", None), True)
        if getattr(model, "unified_retriever", None) is not None:
            trainable_modules.append("unified_retriever")
        _set_belief_calibration_trainable(model, trainable_modules)
    elif route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        route_memory_gate_enabled = (
            active_loss_weights is None
            or float(active_loss_weights.get("counterfactual_utility", 0.0)) > 0.0
        )
        _set_trainable(
            getattr(model, "route_memory_utility_gate", None),
            route_memory_gate_enabled,
        )
        if (
            route_memory_gate_enabled
            and getattr(model, "route_memory_utility_gate", None) is not None
        ):
            trainable_modules.append("route_memory_utility_gate")
    _set_trainable(getattr(model, "action_proj", None), True)
    if getattr(model, "action_proj", None) is not None:
        trainable_modules.append("action_proj")
    _set_trainable(getattr(model, "action_emb", None), True)
    if getattr(model, "action_emb", None) is not None and "action_emb" not in trainable_modules:
        trainable_modules.append("action_emb")
    q_success_enabled = active_loss_weights is None or float(active_loss_weights.get("Q_success", 0.0)) > 0.0
    _set_trainable(getattr(model, "q_success_head", None), q_success_enabled)
    if q_success_enabled and getattr(model, "q_success_head", None) is not None:
        trainable_modules.append("q_success_head")
    if route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        _set_belief_calibration_trainable(model, trainable_modules)
    if effective_transition_scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE:
        _set_trainable(getattr(model, "gated_temporal_reranker", None), True)
        if (
            getattr(model, "gated_temporal_reranker", None) is not None
            and "gated_temporal_reranker" not in trainable_modules
        ):
            trainable_modules.append("gated_temporal_reranker")
    frozen_modules = [
        "encoder.backbone",
        "encoder.proj",
        "skill_table retrieval and belief paths",
    ]
    if not anchored_routing_foundation:
        frozen_modules.extend(["initial_belief_head", "unified_retriever"])
    return {
        "frozen_routing_foundation": not bool(anchored_routing_foundation),
        "qwen_and_skill_table_frozen": True,
        "frozen_modules": frozen_modules,
        "trainable_modules": trainable_modules,
        "freeze_gated_temporal_only": False,
        "route_scorer": route_scorer,
        "anchored_routing_foundation": bool(anchored_routing_foundation),
    }


def _external_encoder_metadata(model: Any, model_config: dict[str, Any], routing_report: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in (
        getattr(model, "qwen_external_metadata", None),
        routing_report,
        model_config,
    ):
        if isinstance(source, dict):
            for key in (
                "qwen_model",
                "qwen_model_name_or_path",
                "qwen_external_encoder",
                "qwen_frozen",
                "qwen_quantization_mode",
                "qwen_direct_generator",
                "external_encoder_type",
                "encoder_output_dim",
            ):
                if key in source:
                    metadata[key] = source[key]
    if metadata.get("qwen_external_encoder"):
        metadata.setdefault("qwen_model", "Qwen/Qwen3-8B")
        metadata.setdefault("qwen_frozen", True)
        metadata.setdefault("qwen_quantization_mode", "none")
        metadata.setdefault("qwen_direct_generator", False)
    return metadata


def _frozen_encoder_checkpoint_exclusion(
    model: Any,
    external_encoder_metadata: dict[str, Any],
) -> bool:
    if bool(external_encoder_metadata.get("qwen_external_encoder", False)):
        return True
    config = getattr(model, "config", None)
    if bool(getattr(config, "freeze_backbone", False)):
        return True
    backbone = getattr(getattr(model, "encoder", None), "backbone", None)
    parameters = list(backbone.parameters()) if hasattr(backbone, "parameters") else []
    return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)


def _checkpoint_state_dict(
    model: Any,
    exclude_frozen_qwen_backbone: bool = False,
    exclude_frozen_routing_foundation: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not hasattr(model, "state_dict"):
        return {}, {
            "checkpoint_excludes_frozen_qwen_backbone": bool(exclude_frozen_qwen_backbone),
            "checkpoint_excludes_frozen_routing_foundation": bool(exclude_frozen_routing_foundation),
            "checkpoint_state_key_count": 0,
            "excluded_state_key_prefixes": ["encoder.backbone."] if exclude_frozen_qwen_backbone else [],
            "preserved_routing_calibration_keys": [],
        }
    state = dict(model.state_dict())
    excluded_prefixes: list[str] = []
    preserved_routing_calibration_keys: list[str] = []
    if exclude_frozen_routing_foundation:
        excluded_prefixes.extend(
            [
                "encoder.",
                "cross_encoder.",
                "skill_table.",
            ]
        )
    if exclude_frozen_qwen_backbone:
        excluded_prefixes.extend([
            "encoder.backbone.",
            "encoder.proj.",
            "cross_encoder.backbone.",
            "cross_encoder.proj.",
            "skill_table.encoder_fn.backbone.",
            "skill_table.encoder_fn.proj.",
            "skill_table.W.",
            "skill_table.E",
        ])
    if excluded_prefixes:
        preserved_routing_calibration_keys = sorted(key for key in state if key in BELIEF_CALIBRATION_STATE_KEYS)
        state = {
            key: value
            for key, value in state.items()
            if key in BELIEF_CALIBRATION_STATE_KEYS
            or not any(str(key).startswith(prefix) for prefix in excluded_prefixes)
        }
    return state, {
        "checkpoint_excludes_frozen_qwen_backbone": bool(exclude_frozen_qwen_backbone),
        "checkpoint_excludes_frozen_routing_foundation": bool(exclude_frozen_routing_foundation),
        "checkpoint_state_key_count": len(state),
        "excluded_state_key_prefixes": excluded_prefixes,
        "preserved_routing_calibration_keys": preserved_routing_calibration_keys,
    }


def _encode(model: Any, texts: list[str]) -> torch.Tensor:
    if hasattr(model, "encode_observations"):
        return model.encode_observations(texts)
    raise TypeError("model must expose encode_observations(texts)")


def _encode_state_queries(model: Any, texts: list[str]) -> torch.Tensor:
    encode = getattr(model, "encode_states", None)
    if callable(encode):
        return encode(texts)
    config = getattr(model, "config", None)
    prompt_version = getattr(config, "state_query_prompt_version", RAW_STATE_V1)
    if prompt_version != RAW_STATE_V1:
        raise TypeError("prompted model must expose encode_states(texts)")
    return _encode(model, texts)


def _encode_stage0_handoff_queries(
    model: Any,
    texts: list[str],
    *,
    query_mode: str,
) -> torch.Tensor:
    if query_mode not in STAGE0_HANDOFF_QUERY_MODES:
        raise ValueError(f"unsupported Stage0 handoff query mode: {query_mode}")
    if query_mode == "checkpoint_state_query":
        return _encode_state_queries(model, texts)
    return _encode(model, texts)


def _stage0_unified_static_logits(
    model: Any,
    h: torch.Tensor,
    skill_count: int,
) -> torch.Tensor:
    initial_belief = getattr(model, "initial_belief", None)
    unified_full_logits = getattr(model, "unified_route_full_logits", None)
    if not callable(initial_belief) or not callable(unified_full_logits):
        raise ValueError(
            "unified Stage0 handoff requires model.initial_belief and "
            "model.unified_route_full_logits"
        )
    memory = initial_belief(h)
    logits = unified_full_logits(h, memory)
    expected_shape = (int(h.size(0)), int(skill_count))
    if logits.ndim != 2 or tuple(logits.shape) != expected_shape:
        raise ValueError("unified Stage0 handoff logits must match batch and declared skill pool")
    return logits


class _StateQueryBatchEncoder:
    def __init__(self, model: Any):
        self.model = model

    def encode_observations(self, texts: list[str]) -> torch.Tensor:
        return _encode_state_queries(self.model, texts)


def _encode_state_text_batches(
    model: Any,
    texts: list[str],
    batch_size: int,
    *,
    label: str,
) -> torch.Tensor:
    return _encode_text_batches(
        _StateQueryBatchEncoder(model),
        texts,
        batch_size,
        label=label,
    )


def _cached_embedding(row: dict[str, Any], key: str, device: torch.device) -> torch.Tensor | None:
    value = row.get(key)
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value.to(device)
    return None


def _batch_cached_or_encode(
    model: Any,
    rows: list[dict[str, Any]],
    embedding_key: str,
    text_key: str,
    device: torch.device,
    *,
    text_role: str,
) -> torch.Tensor:
    cached = [_cached_embedding(row, embedding_key, device) for row in rows]
    if cached and all(item is not None for item in cached):
        return torch.cat([item for item in cached if item is not None], dim=0)
    if text_role == STATE_QUERY_ROLE:
        if text_key == "state_text":
            texts = [router_state_text(row) for row in rows]
        else:
            texts = [strip_history_sections(str(row.get(text_key) or "")) for row in rows]
        return _encode_state_queries(model, texts).to(device)
    if text_role == TRANSITION_TEXT_ROLE:
        texts = [str(row.get(text_key) or "") for row in rows]
        return _encode(model, texts).to(device)
    raise ValueError(f"unsupported text_role: {text_role}")


def _skill_logits_and_memory(model: Any, h: torch.Tensor, skill_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    skill_table = getattr(model, "skill_table", None)
    if skill_table is None:
        logits = torch.zeros(h.size(0), skill_count, device=h.device)
        return logits, h
    if hasattr(skill_table, "retrieval_logits"):
        logits = skill_table.retrieval_logits(h)
    elif hasattr(skill_table, "logits"):
        logits = skill_table.logits(h)
    else:
        logits = skill_table(h)
        return logits, h
    emb = getattr(skill_table, "E", None)
    if emb is not None and hasattr(skill_table, "belief_logits"):
        m_obs = subspace_obs(skill_table, h)
    elif emb is not None:
        m_obs = torch.softmax(logits, dim=-1) @ emb.to(h.device)
    else:
        m_obs = h
    return logits, m_obs


def _belief_memory_semantics_report(model: Any, freeze_audit: dict[str, Any]) -> dict[str, Any]:
    skill_table = getattr(model, "skill_table", None)
    has_retrieval_logits = callable(getattr(skill_table, "retrieval_logits", None))
    has_belief_logits = callable(getattr(skill_table, "belief_logits", None))
    has_skill_embeddings = getattr(skill_table, "E", None) is not None
    trainable_modules = set(str(item) for item in (freeze_audit.get("trainable_modules") or []))
    belief_keys = {"skill_table.logit_scale_belief", "skill_table.skill_bias_belief"}
    params_present = all(isinstance(getattr(skill_table, key.split(".", 1)[1], None), torch.nn.Parameter) for key in belief_keys)
    params_trainable = bool(params_present and belief_keys.issubset(trainable_modules))
    if has_belief_logits and has_skill_embeddings:
        memory_source = "skill_table.belief_logits_via_subspace_obs"
        train_eval_consistent = True
    elif has_skill_embeddings:
        memory_source = "legacy_retrieval_softmax_over_skill_embeddings"
        train_eval_consistent = False
    else:
        memory_source = "hidden_state_fallback_without_skill_subspace"
        train_eval_consistent = False
    return {
        "routing_logits_source": "skill_table.retrieval_logits" if has_retrieval_logits else "skill_table.logits_or_forward",
        "memory_source": memory_source,
        "stage1_stage2_train_eval_memory_consistent": train_eval_consistent,
        "belief_calibration_params_present": bool(params_present),
        "belief_calibration_trainable": params_trainable,
    }


def _policy_feature_semantics_report(model: Any) -> dict[str, Any]:
    return {
        "train_candidate_encoder": "action_text_bi_encoder_embeddings",
        "train_context": "belief",
        "skill_candidate_policy_eval_interface": "removed",
        "skill_routing_eval_uses": "route_logits_from_candidates_stage0_retrieval_prior",
        "train_eval_default_aligned": True,
    }


def _row_candidate_indices(row: dict[str, Any], key: str, skill_count: int) -> list[int]:
    values = row.get(key)
    if values is None:
        return []
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= skill_count or idx in seen:
            continue
        seen.add(idx)
        output.append(idx)
    return output


def _skill_root_namespace(skill_id: str) -> str:
    return str(skill_id or "").split("/", 1)[0]


def _flatten_skill_id_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("skill_id", "canonical_skill_id", "id", "name"):
            if value.get(key):
                return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_skill_id_values(item))
        return output
    return [str(value)] if str(value).strip() else []


def _explicit_inventory_skill_ids_ordered(row: dict[str, Any]) -> list[str]:
    return _shared_explicit_inventory_skill_ids_ordered(row)


def _explicit_inventory_skill_ids(row: dict[str, Any]) -> set[str]:
    return set(_explicit_inventory_skill_ids_ordered(row))


def _explicit_positive_skill_ids(row: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for key in EXPLICIT_POSITIVE_SKILL_ID_KEYS:
        output.update(_flatten_skill_id_values(row.get(key)))
    return {item for item in output if item}


def _equivalent_skill_ids_by_skill_id(skills: list[dict[str, Any]]) -> dict[str, list[str]]:
    canonical_to_aliases: dict[str, set[str]] = {}
    for row in skills:
        skill_id = str(row.get("skill_id") or row.get("id") or "")
        if not skill_id:
            continue
        canonical_id = str(row.get("canonical_skill_id") or skill_id)
        aliases = set(_flatten_skill_id_values(row.get("alias_skill_ids")))
        aliases.add(skill_id)
        canonical_to_aliases.setdefault(canonical_id, set()).update(alias for alias in aliases if alias)
    output: dict[str, list[str]] = {}
    for row in skills:
        skill_id = str(row.get("skill_id") or row.get("id") or "")
        if not skill_id:
            continue
        canonical_id = str(row.get("canonical_skill_id") or skill_id)
        aliases = canonical_to_aliases.get(canonical_id, {skill_id})
        output[skill_id] = sorted(alias for alias in aliases if alias and alias != skill_id)
    return output


def _transition_inventory_root(row: dict[str, Any]) -> str | None:
    benchmark = str(row.get("benchmark") or "")
    if benchmark in BENCHMARK_ROOT_SKILL_PREFIX:
        return BENCHMARK_ROOT_SKILL_PREFIX[benchmark]
    for key in ("next_skill_id", "skill_id"):
        value = str(row.get(key) or "")
        if "/" in value:
            return _skill_root_namespace(value)
    return None


def _filter_transition_candidates_by_inventory(
    *,
    rows: list[dict[str, Any]],
    candidate_rows: list[list[int]],
    labels: torch.Tensor,
    skill_ids_by_idx: dict[int, str],
    mode: str = "off",
    min_candidates: int = 0,
    preserve_positive: bool = True,
) -> tuple[list[list[int]], dict[str, int]]:
    mode = str(mode or "off")
    if mode not in TRANSITION_INVENTORY_MASK_MODES:
        raise ValueError(f"unsupported transition inventory mask mode: {mode}")
    min_candidates = max(0, int(min_candidates or 0))
    audit = {
        "inventory_mask_applied_rows": 0,
        "inventory_mask_missing_rows": 0,
        "inventory_mask_removed_candidates": 0,
        "inventory_mask_positive_missing_rows": 0,
        "inventory_mask_backfilled_rows": 0,
        "inventory_mask_backfilled_candidates": 0,
    }
    if mode == "off":
        return [list(row) for row in candidate_rows], audit

    filtered_rows: list[list[int]] = []
    label_values = [int(item) for item in labels.detach().cpu().tolist()]
    for row, candidates, positive_idx in zip(rows, candidate_rows, label_values):
        original = [int(idx) for idx in candidates]
        if mode == "stage0_topk_trajectory_prior":
            target_count = max(1, min_candidates or 50)
            explicit_ids = _explicit_inventory_skill_ids(row)
            prior_indices = [
                idx
                for idx in original
                if str(skill_ids_by_idx.get(int(idx), "")) in explicit_ids
            ]
            prior_budget = min(len(prior_indices), max(0, target_count // 4))
            score_budget = max(0, target_count - prior_budget)
            allowed: list[int] = []
            seen: set[int] = set()

            def add_candidate(idx: int) -> None:
                idx = int(idx)
                if idx in seen or len(allowed) >= target_count:
                    return
                allowed.append(idx)
                seen.add(idx)

            for idx in original[:score_budget]:
                add_candidate(idx)
            for idx in prior_indices:
                add_candidate(idx)
            for idx in original:
                add_candidate(idx)
            if bool(preserve_positive) and positive_idx in original and positive_idx not in seen:
                if allowed:
                    prior_set = {int(idx) for idx in prior_indices}
                    replace_at = len(allowed) - 1
                    for candidate_pos in range(len(allowed) - 1, -1, -1):
                        if int(allowed[candidate_pos]) not in prior_set:
                            replace_at = candidate_pos
                            break
                    removed = allowed.pop(replace_at)
                    seen.discard(int(removed))
                    allowed.append(int(positive_idx))
                    seen.add(int(positive_idx))
                else:
                    allowed.append(int(positive_idx))
                    seen.add(int(positive_idx))
                audit["inventory_mask_positive_missing_rows"] += 1
            audit["inventory_mask_applied_rows"] += 1
            audit["inventory_mask_removed_candidates"] += max(0, len(original) - len(allowed))
            filtered_rows.append(allowed)
            continue
        explicit_ids = (
            _explicit_inventory_skill_ids(row)
            if mode in {"auto", "explicit_only"}
            else set()
        )
        allowed_root = (
            None
            if explicit_ids or mode == "explicit_only"
            else _transition_inventory_root(row)
        )
        if not explicit_ids and not allowed_root:
            audit["inventory_mask_missing_rows"] += 1
            filtered_rows.append(original)
            continue
        allowed: list[int] = []
        for idx in original:
            skill_id = str(skill_ids_by_idx.get(int(idx), ""))
            keep = skill_id in explicit_ids if explicit_ids else _skill_root_namespace(skill_id) == allowed_root
            if keep:
                allowed.append(int(idx))
        if bool(preserve_positive) and positive_idx in original and positive_idx not in allowed:
            allowed.append(int(positive_idx))
            audit["inventory_mask_positive_missing_rows"] += 1
        if not allowed:
            audit["inventory_mask_missing_rows"] += 1
            filtered_rows.append(original)
            continue
        before_backfill = len(allowed)
        if min_candidates > 0 and before_backfill < min_candidates:
            seen = set(allowed)
            for idx in original:
                idx = int(idx)
                if idx in seen:
                    continue
                allowed.append(idx)
                seen.add(idx)
                if len(allowed) >= min_candidates:
                    break
            backfilled = len(allowed) - before_backfill
            if backfilled > 0:
                audit["inventory_mask_backfilled_rows"] += 1
                audit["inventory_mask_backfilled_candidates"] += backfilled
        audit["inventory_mask_applied_rows"] += 1
        audit["inventory_mask_removed_candidates"] += max(0, len(original) - len(allowed))
        filtered_rows.append(allowed)
    return filtered_rows, audit


def _multi_positive_listwise_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.ndim != 2 or positive_mask.ndim != 2:
        raise ValueError("multi-positive listwise NLL expects rank-2 logits and positive_mask")
    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have matching shapes")
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("each row must contain at least one positive")
    min_value = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, min_value)
    losses = torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(positive_logits.float(), dim=-1)
    if reduction == "none":
        return losses.to(logits.dtype)
    if reduction == "sum":
        return losses.sum().to(logits.dtype)
    if reduction == "mean":
        return losses.mean().to(logits.dtype)
    raise ValueError(f"unsupported reduction: {reduction}")


def _transition_positive_mask(
    *,
    rows: list[dict[str, Any]],
    candidate_rows: list[list[int]],
    labels: torch.Tensor,
    skill_ids_by_idx: dict[int, str],
    positive_mode: str,
    device: torch.device,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_mode = str(positive_mode or "single")
    if positive_mode not in TRANSITION_POSITIVE_MODES:
        raise ValueError(f"unsupported transition positive mode: {positive_mode}")
    max_width = max((len(row) for row in candidate_rows), default=0)
    mask = torch.zeros((len(candidate_rows), max_width), dtype=torch.bool, device=device)
    counts: list[int] = []
    label_values = [int(item) for item in labels.detach().cpu().tolist()]
    for row_idx, (row, candidates, label) in enumerate(zip(rows, candidate_rows, label_values)):
        if 0 <= label < len(candidates):
            mask[row_idx, label] = True
        if positive_mode == "gold_plus_equivalent":
            explicit_ids = _explicit_positive_skill_ids(row)
            next_skill_id = str(row.get("next_skill_id") or "")
            explicit_ids.add(next_skill_id)
            if equivalent_skill_ids_by_skill_id is not None:
                explicit_ids.update(equivalent_skill_ids_by_skill_id.get(next_skill_id, []))
            for local_idx, global_idx in enumerate(candidates):
                if str(skill_ids_by_idx.get(int(global_idx), "")) in explicit_ids:
                    mask[row_idx, local_idx] = True
        count = int(mask[row_idx, : len(candidates)].sum().detach().cpu().item())
        counts.append(count)
    if counts:
        mean_count = sum(counts) / len(counts)
        max_count = max(counts)
        multi_positive_rows = sum(1 for count in counts if count > 1)
    else:
        mean_count = 0.0
        max_count = 0
        multi_positive_rows = 0
    return mask, {
        "transition_positive_mean_count": float(mean_count),
        "transition_positive_max_count": float(max_count),
        "transition_multi_positive_rows": float(multi_positive_rows),
    }


def _current_skill_candidate_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_rows: list[list[int]],
    logits: torch.Tensor,
    skill_id_to_idx: dict[str, int],
) -> dict[str, float]:
    candidate_row_count = 0
    keep_rows = 0
    switch_rows = 0
    keep_top1 = 0
    switch_current_top1 = 0
    ranks: list[float] = []
    for row_idx, (row, candidates) in enumerate(zip(rows, candidate_rows)):
        skill_id = str(row.get("skill_id") or "")
        current_idx = skill_id_to_idx.get(skill_id)
        if current_idx is None or int(current_idx) not in candidates:
            continue
        local_idx = candidates.index(int(current_idx))
        valid_logits = logits[row_idx, : len(candidates)].detach().float()
        if valid_logits.numel() == 0:
            continue
        candidate_row_count += 1
        current_logit = valid_logits[local_idx]
        rank = float((valid_logits > current_logit).sum().detach().cpu().item() + 1)
        ranks.append(rank)
        pred_local = int(torch.argmax(valid_logits).detach().cpu().item())
        is_keep = str(row.get("next_skill_id") or "") == skill_id
        if is_keep:
            keep_rows += 1
            keep_top1 += int(pred_local == local_idx)
        else:
            switch_rows += 1
            switch_current_top1 += int(pred_local == local_idx)
    return {
        "transition_current_skill_candidate_rows": float(candidate_row_count),
        "transition_current_skill_keep_rows": float(keep_rows),
        "transition_current_skill_switch_rows": float(switch_rows),
        "transition_current_skill_keep_recall@1": float(keep_top1 / max(keep_rows, 1)),
        "transition_current_skill_switch_false_positive@1": float(switch_current_top1 / max(switch_rows, 1)),
        "transition_current_skill_mean_rank": float(sum(ranks) / max(len(ranks), 1)),
    }


def _gather_candidate_logits_and_labels(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    labels: torch.Tensor,
    candidate_key: str,
    skill_count: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    candidate_rows = [_row_candidate_indices(row, candidate_key, skill_count) for row in rows]
    if not candidate_rows or not all(candidate_rows):
        return None, None, None
    max_width = max(len(row) for row in candidate_rows)
    gathered = torch.full(
        (len(candidate_rows), max_width),
        torch.finfo(logits.dtype).min,
        dtype=logits.dtype,
        device=logits.device,
    )
    local_labels: list[int] = []
    padded_indices = torch.zeros(len(candidate_rows), max_width, dtype=torch.long, device=logits.device)
    for row_idx, (candidate_indices, label) in enumerate(zip(candidate_rows, labels.detach().cpu().tolist())):
        label_int = int(label)
        if label_int not in candidate_indices:
            return None, None, None
        width = len(candidate_indices)
        index_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=logits.device)
        gathered[row_idx, :width] = logits[row_idx].index_select(0, index_tensor)
        padded_indices[row_idx, :width] = index_tensor
        local_labels.append(candidate_indices.index(label_int))
    return gathered, torch.tensor(local_labels, dtype=torch.long, device=logits.device), padded_indices


def _next_belief_target_from_embeddings(model: Any, h_next: torch.Tensor, skill_count: int) -> torch.Tensor:
    _logits, m_tilde_next = _skill_logits_and_memory(model, h_next, skill_count)
    return m_tilde_next.detach()


def _next_belief_target(model: Any, texts: list[str], skill_count: int, device: torch.device) -> torch.Tensor:
    h_next = _encode(model, texts).to(device)
    return _next_belief_target_from_embeddings(model, h_next, skill_count)


def _prefix_step_embedding(
    model: Any,
    step: dict[str, Any],
    embedding_key: str,
    text_key: str,
    device: torch.device,
) -> torch.Tensor:
    cached = _cached_embedding(step, embedding_key, device)
    if cached is not None:
        return cached
    return _encode(model, [str(step.get(text_key) or "")]).to(device)


def _batch_action_text_embedding_or_none(
    model: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor | None:
    if not rows:
        return None
    if not all("_action_embedding" in row or str(row.get("action_text") or "").strip() for row in rows):
        return None
    return _batch_cached_or_encode(
        model,
        rows,
        "_action_embedding",
        "action_text",
        device,
        text_role=TRANSITION_TEXT_ROLE,
    )


def _replay_prefix_belief(
    model: Any,
    replay_prefix: list[dict[str, Any]],
    fallback_m: torch.Tensor,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
    device: torch.device,
    *,
    trainable: bool = False,
) -> tuple[torch.Tensor, bool]:
    if not replay_prefix:
        return fallback_m, False
    first_observation = str(replay_prefix[0].get("observation_text") or "")
    if not first_observation:
        return fallback_m, False

    def _run() -> tuple[torch.Tensor, bool]:
        h_step = _prefix_step_embedding(model, replay_prefix[0], "_observation_embedding", "observation_text", device)
        if trainable:
            h_step = h_step.detach()
        initial_belief = getattr(model, "initial_belief", None)
        if not callable(initial_belief):
            raise ValueError("causal replay requires model.initial_belief")
        m = initial_belief(h_step)
        if not trainable:
            m = m.detach()
        used = False
        for step in replay_prefix:
            if not isinstance(step, dict):
                continue
            action_text = str(step.get("action_text") or "")
            next_observation = step.get("next_observation_text")
            if not action_text:
                continue
            action_emb = _prefix_step_embedding(model, step, "_action_embedding", "action_text", device)
            if trainable:
                action_emb = action_emb.detach()
            labels = torch.tensor(
                [
                    int(step["skill_idx"])
                    if step.get("skill_idx") is not None
                    else skill_id_to_idx.get(str(step.get("skill_id")), 0)
                ],
                dtype=torch.long,
                device=device,
            )
            if next_observation:
                h_next = _prefix_step_embedding(model, step, "_next_observation_embedding", "next_observation_text", device)
                if trainable:
                    h_next = h_next.detach()
                pred = _transition_prediction(model, h_step, m, labels, h_next, action_emb=action_emb)
                m_tilde_next = _next_belief_target_from_embeddings(model, h_next, skill_count)
                m = _belief_prediction(model, pred, m_tilde_next, h_next)
                h_step = h_next
            else:
                pred = _transition_prediction(model, h_step, m, labels, h_step, action_emb=action_emb)
                m = pred
            used = True
        return m, used

    if trainable:
        return _run()
    with torch.no_grad():
        replayed, used = _run()
        return replayed.detach(), used


def _apply_replay_prefix_beliefs(
    model: Any,
    batch: list[dict[str, Any]],
    m_obs: torch.Tensor,
    skill_id_to_idx: dict[str, int],
    skill_count: int,
    device: torch.device,
    *,
    trainable: bool = False,
) -> tuple[torch.Tensor, int]:
    rows: list[torch.Tensor] = []
    used_count = 0
    for idx, row in enumerate(batch):
        prefix = row.get("replay_prefix")
        if isinstance(prefix, list) and prefix:
            replayed, used = _replay_prefix_belief(
                model,
                prefix,
                m_obs[idx : idx + 1],
                skill_id_to_idx,
                skill_count,
                device,
                trainable=trainable,
            )
            rows.append(replayed)
            used_count += int(used)
        else:
            rows.append(m_obs[idx : idx + 1])
    return torch.cat(rows, dim=0), used_count


def _stop_logits(model: Any, h: torch.Tensor, m_obs: torch.Tensor) -> torch.Tensor:
    stop_head = getattr(model, "stop_head", None)
    if stop_head is None:
        return torch.zeros(h.size(0), device=h.device)
    try:
        out = stop_head(h, m_obs)
    except TypeError:
        out = stop_head(torch.cat([h, m_obs], dim=-1))
    return out.squeeze(-1)


def _transition_prediction(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    labels: torch.Tensor,
    obs_emb: torch.Tensor,
    action_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    transition = getattr(model, "transition", None)
    if transition is None:
        return h
    action_input = _transition_action_input(model, labels, obs_emb, action_emb=action_emb)
    try:
        return transition(m_obs, action_input, obs_emb)
    except TypeError:
        return transition(torch.cat([h, obs_emb], dim=-1))


def _transition_prior_prediction(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    labels: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    return _transition_prediction(model, h, m_obs, labels, torch.zeros_like(like), action_emb=None)


def _transition_action_input(
    model: Any,
    labels: torch.Tensor,
    like: torch.Tensor,
    action_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if action_emb is None:
        return transition_action_input(model, labels, like=like)
    action_input = action_emb.to(device=like.device, dtype=like.dtype)
    action_proj = getattr(model, "action_proj", None)
    if action_proj is not None:
        action_input = action_proj(action_input)
    return action_input.to(device=like.device, dtype=like.dtype)


def _belief_prediction(model: Any, pred: torch.Tensor, m_obs: torch.Tensor, obs_emb: torch.Tensor) -> torch.Tensor:
    gate = getattr(model, "gate", None)
    if gate is None:
        return pred
    try:
        gamma = gate(pred, m_obs, obs_emb)
    except TypeError:
        gamma = torch.sigmoid(gate(torch.cat([pred, m_obs, obs_emb], dim=-1)))
    return gamma * m_obs + (1.0 - gamma) * pred


def _post_action_memory(
    model: Any,
    *,
    m_t: torch.Tensor,
    current_skill_labels: torch.Tensor,
    action_embeddings: torch.Tensor | None,
    observation_embeddings: torch.Tensor,
    h_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transition = getattr(model, "transition", None)
    gate = getattr(model, "gate", None)
    skill_table = getattr(model, "skill_table", None)
    if transition is None or gate is None:
        raise ValueError("causal unified memory requires model.transition and model.gate")
    if skill_table is None or getattr(skill_table, "E", None) is None or not callable(
        getattr(skill_table, "belief_logits", None)
    ):
        raise ValueError("causal unified memory requires skill_table.E and skill_table.belief_logits")
    if action_embeddings is None:
        raise ValueError("causal unified memory requires current action embeddings")
    action_input = _transition_action_input(
        model,
        current_skill_labels,
        observation_embeddings,
        action_emb=action_embeddings,
    )
    predicted_memory = transition(m_t, action_input, observation_embeddings)
    observation_memory = subspace_obs(skill_table, h_next)
    gamma = gate(predicted_memory, observation_memory, observation_embeddings)
    next_memory = gamma * observation_memory + (1.0 - gamma) * predicted_memory
    return predicted_memory, observation_memory, next_memory


def _candidate_batch(rows: list[dict[str, Any]]) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
    candidate_rows: list[list[str]] = []
    labels: list[int] = []
    max_width = 1
    for row in rows:
        candidates = [str(item) for item in row.get("admissible_actions") or []]
        expert = str(row.get("expert_action") or row.get("action_text") or "")
        if expert not in candidates:
            candidates = [expert]
        labels.append(candidates.index(expert))
        candidate_rows.append(candidates)
        max_width = max(max_width, len(candidates))
    padded: list[list[str]] = []
    masks: list[list[bool]] = []
    for candidates in candidate_rows:
        if len(candidates) < max_width:
            pad = candidates[-1]
            padded.append(candidates + [pad] * (max_width - len(candidates)))
            masks.append([True] * len(candidates) + [False] * (max_width - len(candidates)))
        else:
            padded.append(candidates)
            masks.append([True] * len(candidates))
    return padded, torch.tensor(masks, dtype=torch.bool), torch.tensor(labels, dtype=torch.long)


def _attach_policy_embedding_cache(
    model: Any,
    rows: list[dict[str, Any]],
    encode_batch_size: int = 256,
) -> dict[str, Any]:
    policy_rows = [row for row in rows if (row.get("loss_mask") or {}).get("L_policy")]
    if not policy_rows:
        return {"used": False, "policy_rows": 0, "action_vocab_size": 0}

    state_texts = [router_state_text(row) for row in policy_rows]
    action_texts: list[str] = []
    action_text_to_idx: dict[str, int] = {}
    labels: list[int] = []
    row_candidate_indices: list[list[int]] = []
    for row in policy_rows:
        candidates = [str(item) for item in row.get("admissible_actions") or []]
        expert = str(row.get("expert_action") or row.get("action_text") or "")
        if expert not in candidates:
            candidates = [expert]
        labels.append(candidates.index(expert))
        indices: list[int] = []
        for action in candidates:
            if action not in action_text_to_idx:
                action_text_to_idx[action] = len(action_texts)
                action_texts.append(action)
            indices.append(action_text_to_idx[action])
        row_candidate_indices.append(indices)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    state_embeddings = _encode_state_text_batches(
        model,
        state_texts,
        encode_batch_size,
        label="full-base policy state embeddings",
    )
    action_embeddings = _encode_text_batches(
        model,
        action_texts,
        encode_batch_size,
        label="full-base policy action embeddings",
    )
    if was_training and hasattr(model, "train"):
        model.train()

    for row, state_emb, candidate_indices, label in zip(policy_rows, state_embeddings, row_candidate_indices, labels):
        index_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        row["_policy_state_embedding"] = state_emb.detach().cpu().float()
        row["_policy_candidate_embeddings"] = action_embeddings.index_select(0, index_tensor).detach().cpu().float()
        row["_policy_label"] = int(label)

    return {
        "used": True,
        "policy_rows": len(policy_rows),
        "action_vocab_size": len(action_texts),
        "encode_batch_size": max(1, int(encode_batch_size)),
    }


def _attach_full_base_embedding_cache(
    model: Any,
    rows: list[dict[str, Any]],
    encode_batch_size: int = 256,
) -> dict[str, Any]:
    state_refs: list[tuple[dict[str, Any], str, str]] = []
    transition_refs: list[tuple[dict[str, Any], str, str]] = []
    for row in rows:
        current_state = router_state_text(row)
        if current_state:
            state_refs.append((row, "_state_embedding", current_state))
        next_state = strip_history_sections(str(row.get("next_state_text") or ""))
        if next_state:
            state_refs.append((row, "_next_state_embedding", next_state))
        for text_key, embedding_key in (
            ("action_text", "_action_embedding"),
            ("next_observation_text", "_next_observation_embedding"),
        ):
            text = row.get(text_key)
            if text is not None and str(text):
                transition_refs.append((row, embedding_key, str(text)))
        prefix = row.get("replay_prefix")
        if isinstance(prefix, list):
            for step in prefix:
                if not isinstance(step, dict):
                    continue
                for text_key, embedding_key in (
                    ("observation_text", "_observation_embedding"),
                    ("action_text", "_action_embedding"),
                    ("next_observation_text", "_next_observation_embedding"),
                ):
                    text = step.get(text_key)
                    if text is not None and str(text):
                        transition_refs.append((step, embedding_key, str(text)))
    if not state_refs and not transition_refs:
        return {"used": False, "row_count": len(rows), "unique_text_count": 0}

    def unique_texts(
        refs: list[tuple[dict[str, Any], str, str]],
    ) -> tuple[list[str], dict[str, int]]:
        text_to_idx: dict[str, int] = {}
        texts: list[str] = []
        for _container, _embedding_key, text in refs:
            if text not in text_to_idx:
                text_to_idx[text] = len(texts)
                texts.append(text)
        return texts, text_to_idx

    state_texts, state_text_to_idx = unique_texts(state_refs)
    transition_texts, transition_text_to_idx = unique_texts(transition_refs)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    state_embeddings = _encode_state_text_batches(
        model,
        state_texts,
        encode_batch_size,
        label="full-base state query embeddings",
    )
    transition_embeddings = _encode_text_batches(
        model,
        transition_texts,
        encode_batch_size,
        label="full-base transition text embeddings",
    )
    if was_training and hasattr(model, "train"):
        model.train()

    for container, embedding_key, text in state_refs:
        container[embedding_key] = (
            state_embeddings[state_text_to_idx[text]].detach().cpu().float()
        )
    for container, embedding_key, text in transition_refs:
        container[embedding_key] = (
            transition_embeddings[transition_text_to_idx[text]].detach().cpu().float()
        )

    return {
        "used": True,
        "row_count": len(rows),
        "unique_text_count": len(state_texts) + len(transition_texts),
        "state_query_unique_text_count": len(state_texts),
        "transition_text_unique_text_count": len(transition_texts),
        "encoded_reference_count": len(state_refs) + len(transition_refs),
        "encode_batch_size": max(1, int(encode_batch_size)),
    }


def _cached_policy_batch(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not rows or not all("_policy_state_embedding" in row and "_policy_candidate_embeddings" in row for row in rows):
        return None
    state_embs = torch.stack([row["_policy_state_embedding"] for row in rows]).to(device)
    candidate_lists = [row["_policy_candidate_embeddings"] for row in rows]
    max_width = max(int(candidates.size(0)) for candidates in candidate_lists)
    dim = int(candidate_lists[0].size(-1))
    padded = torch.zeros(len(rows), max_width, dim, dtype=torch.float32)
    mask = torch.zeros(len(rows), max_width, dtype=torch.bool)
    labels = []
    for idx, candidates in enumerate(candidate_lists):
        width = int(candidates.size(0))
        padded[idx, :width] = candidates
        mask[idx, :width] = True
        labels.append(int(rows[idx].get("_policy_label", 0)))
    return state_embs, padded.to(device), mask.to(device), torch.tensor(labels, dtype=torch.long, device=device)


def _native_policy_scores(
    model: Any,
    candidate_embs: torch.Tensor,
    belief_embs: torch.Tensor,
    candidate_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    skill_head = getattr(model, "skill_head", None)
    if skill_head is None:
        return None
    try:
        scores = skill_head(candidate_embs, belief_embs)
    except TypeError:
        return None
    if scores.ndim == 3 and scores.size(-1) == 1:
        scores = scores.squeeze(-1)
    if scores.ndim != 2:
        return None
    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask.to(scores.device).to(torch.bool), torch.finfo(scores.dtype).min)
    return scores


def _has_native_policy_head(model: Any) -> bool:
    return getattr(model, "skill_head", None) is not None


def _loss_activation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, enabled in (row.get("loss_mask") or {}).items():
            if enabled:
                counts[str(key)] += 1
    return {key: counts.get(key, 0) for key in LOSS_WEIGHT_KEYS}


def _batch_for_step(rows: list[dict[str, Any]], step_idx: int, batch_size: int) -> list[dict[str, Any]]:
    start = ((step_idx - 1) * batch_size) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def _build_loss_buckets(rows: list[dict[str, Any]], loss_weights: dict[str, float] | None = None) -> dict[str, list[int]]:
    weights = _normalize_loss_weights(loss_weights)
    return {
        key: [
            idx
            for idx, row in enumerate(rows)
            if float(weights.get(key, 0.0)) > 0.0 and bool((row.get("loss_mask") or {}).get(key))
        ]
        for key in LOSS_WEIGHT_KEYS
    }


def _row_transition_sampling_group(row: dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or "unknown")
    skill_id = str(row.get("skill_id") or "")
    next_skill_id = str(row.get("next_skill_id") or "")
    if not skill_id or not next_skill_id or next_skill_id.lower() == "none":
        relation = "none"
    elif skill_id == next_skill_id:
        relation = "self"
    else:
        relation = "switch"
    return f"{benchmark}:{relation}"


def _is_real_transition_skill_row(row: dict[str, Any]) -> bool:
    loss_mask = row.get("loss_mask") or {}
    next_skill_id = str(row.get("next_skill_id") or "")
    return bool(loss_mask.get("L_trans_skill_ce")) and bool(next_skill_id) and next_skill_id.lower() != "none" and not bool(row.get("stage0_positive_injected"))


def _is_switch_transition_row(row: dict[str, Any]) -> bool:
    skill_id = str(row.get("skill_id") or "")
    next_skill_id = str(row.get("next_skill_id") or "")
    return bool(skill_id and next_skill_id and next_skill_id.lower() != "none" and skill_id != next_skill_id)


def _batch_composition_metrics(batch: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark_counts: Counter[str] = Counter()
    loss_mask_counts: Counter[str] = Counter()
    transition_relation_counts: Counter[str] = Counter()
    transition_real_rows = 0
    transition_injected_rows = 0
    transition_switch_real_rows = 0
    policy_rows = 0
    for row in batch:
        benchmark = str(row.get("benchmark") or "unknown")
        benchmark_counts[benchmark] += 1
        loss_mask = row.get("loss_mask") or {}
        for key, enabled in loss_mask.items():
            if enabled:
                loss_mask_counts[str(key)] += 1
        if loss_mask.get("L_policy"):
            policy_rows += 1
        relation = _row_transition_sampling_group(row).split(":", 1)[-1]
        transition_relation_counts[relation] += 1
        if bool(loss_mask.get("L_trans_skill_ce")):
            if bool(row.get("stage0_positive_injected")):
                transition_injected_rows += 1
            else:
                transition_real_rows += 1
                if _is_switch_transition_row(row):
                    transition_switch_real_rows += 1
    return {
        "batch_size_actual": float(len(batch)),
        "batch_benchmark_counts": dict(sorted(benchmark_counts.items())),
        "batch_loss_mask_counts": {key: loss_mask_counts.get(key, 0) for key in LOSS_WEIGHT_KEYS},
        "batch_transition_relation_counts": dict(sorted(transition_relation_counts.items())),
        "batch_policy_rows": float(policy_rows),
        "batch_transition_real_rows": float(transition_real_rows),
        "batch_transition_injected_rows": float(transition_injected_rows),
        "batch_transition_switch_real_rows": float(transition_switch_real_rows),
    }


def _group_indices_by_transition_relation(
    rows: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx in indices:
        buckets.setdefault(_row_transition_sampling_group(rows[idx]), []).append(idx)
    return buckets


def _build_transition_grouped_loss_buckets(
    rows: list[dict[str, Any]],
    loss_buckets: dict[str, list[int]],
) -> dict[str, dict[str, list[int]]]:
    return {
        key: _group_indices_by_transition_relation(rows, bucket)
        for key, bucket in loss_buckets.items()
        if bucket
    }


def _build_quota_grouped_buckets(
    rows: list[dict[str, Any]],
    loss_buckets: dict[str, list[int]],
) -> dict[str, dict[str, list[int]]]:
    real_transition_indices = [
        idx
        for idx in loss_buckets.get("L_trans_skill_ce", [])
        if _is_real_transition_skill_row(rows[idx])
    ]
    switch_transition_indices = [
        idx
        for idx in real_transition_indices
        if _is_switch_transition_row(rows[idx])
    ]
    policy_indices = list(loss_buckets.get("L_policy", []))
    belief_or_trans_indices = list(loss_buckets.get("belief", [])) or list(loss_buckets.get("L_trans", []))
    stop_or_routing_indices = sorted(set(loss_buckets.get("STOP", [])) | set(loss_buckets.get("routing", [])))
    return {
        "switch_transition": _group_indices_by_transition_relation(rows, switch_transition_indices),
        "real_transition": _group_indices_by_transition_relation(rows, real_transition_indices),
        "policy": _group_indices_by_transition_relation(rows, policy_indices),
        "belief_or_trans": _group_indices_by_transition_relation(rows, belief_or_trans_indices),
        "stop_or_routing": _group_indices_by_transition_relation(rows, stop_or_routing_indices),
    }


def _sample_full_base_benchmark_transition_balanced_batch(
    rows: list[dict[str, Any]],
    step_idx: int,
    batch_size: int,
    loss_weights: dict[str, float] | None = None,
    loss_buckets: dict[str, list[int]] | None = None,
    sampler_seed: int | None = None,
    grouped_loss_buckets: dict[str, dict[str, list[int]]] | None = None,
    all_group_buckets: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    weights = _normalize_loss_weights(loss_weights)
    batch_size = max(1, int(batch_size))
    buckets = loss_buckets if loss_buckets is not None else _build_loss_buckets(rows, weights)
    grouped_loss_buckets = grouped_loss_buckets or _build_transition_grouped_loss_buckets(rows, buckets)
    all_group_buckets = all_group_buckets or _group_indices_by_transition_relation(rows, list(range(len(rows))))
    all_groups = sorted(group for group, bucket in all_group_buckets.items() if bucket)
    selected: list[int] = []
    selected_set: set[int] = set()
    rng = random.Random((0 if sampler_seed is None else int(sampler_seed)) * 1_000_003 + int(step_idx) * 97_409)

    def add_from_grouped_bucket(grouped_bucket: dict[str, list[int]], group_offset: int) -> bool:
        if len(selected) >= batch_size:
            return False
        groups = [group for group in all_groups if grouped_bucket.get(group)]
        if not groups:
            return False
        start_group = (max(1, int(step_idx)) - 1 + int(group_offset)) % len(groups)
        for group_probe in range(len(groups)):
            group = groups[(start_group + group_probe) % len(groups)]
            bucket = grouped_bucket[group]
            start_row = rng.randrange(len(bucket))
            for row_probe in range(len(bucket)):
                idx = bucket[(start_row + row_probe) % len(bucket)]
                if idx not in selected_set or len(selected_set) >= len(rows):
                    selected.append(idx)
                    selected_set.add(idx)
                    return True
        return False

    priority = ["L_policy", "L_trans_skill_ce", "STOP", "belief", "routing", "L_trans"]
    for offset, key in enumerate(priority):
        add_from_grouped_bucket(grouped_loss_buckets.get(key, {}), group_offset=offset)
        if len(selected) >= batch_size:
            break

    fallback_offset = len(priority)
    while len(selected) < batch_size:
        added = add_from_grouped_bucket(all_group_buckets, group_offset=fallback_offset)
        fallback_offset += 1
        if added:
            continue
        if selected:
            selected.append(selected[-1])
        else:
            selected.append(0)
        if fallback_offset > len(all_groups) * 2 + len(priority):
            break

    return [rows[idx] for idx in selected[:batch_size]]


def _sample_full_base_benchmark_transition_quota_batch(
    rows: list[dict[str, Any]],
    step_idx: int,
    batch_size: int,
    loss_weights: dict[str, float] | None = None,
    loss_buckets: dict[str, list[int]] | None = None,
    sampler_seed: int | None = None,
    all_group_buckets: dict[str, list[int]] | None = None,
    grouped_quota_buckets: dict[str, dict[str, list[int]]] | None = None,
) -> list[dict[str, Any]]:
    weights = _normalize_loss_weights(loss_weights)
    batch_size = max(1, int(batch_size))
    buckets = loss_buckets if loss_buckets is not None else _build_loss_buckets(rows, weights)
    all_group_buckets = all_group_buckets or _group_indices_by_transition_relation(rows, list(range(len(rows))))
    grouped_quota_buckets = grouped_quota_buckets or _build_quota_grouped_buckets(rows, buckets)
    rng = random.Random((0 if sampler_seed is None else int(sampler_seed)) * 1_000_003 + int(step_idx) * 97_409)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add_from_grouped(grouped_bucket: dict[str, list[int]], count: int, group_offset: int) -> int:
        added = 0
        groups = sorted(group for group, bucket in grouped_bucket.items() if bucket)
        if not groups:
            return added
        for slot in range(max(0, int(count))):
            if len(selected) >= batch_size:
                break
            start_group = (max(1, int(step_idx)) - 1 + int(group_offset) + slot) % len(groups)
            did_add = False
            for group_probe in range(len(groups)):
                group = groups[(start_group + group_probe) % len(groups)]
                bucket = grouped_bucket[group]
                start_row = rng.randrange(len(bucket))
                for row_probe in range(len(bucket)):
                    idx = bucket[(start_row + row_probe) % len(bucket)]
                    if idx in selected_set and len(selected_set) < len(rows):
                        continue
                    selected.append(idx)
                    selected_set.add(idx)
                    added += 1
                    did_add = True
                    break
                if did_add:
                    break
            if not did_add:
                break
        return added

    base_quarter = max(1, batch_size // 4)
    base_eighth = max(0, batch_size // 8)
    quota_plan = [
        (grouped_quota_buckets.get("switch_transition", {}), base_quarter, 0),
        (grouped_quota_buckets.get("real_transition", {}), base_quarter, 3),
        (grouped_quota_buckets.get("policy", {}), base_quarter, 6),
        (grouped_quota_buckets.get("belief_or_trans", {}), base_eighth, 9),
        (grouped_quota_buckets.get("stop_or_routing", {}), base_eighth, 12),
    ]
    for grouped_bucket, count, offset in quota_plan:
        add_from_grouped(grouped_bucket, count, offset)

    fallback_offset = 16
    while len(selected) < batch_size:
        before = len(selected)
        add_from_grouped(all_group_buckets, 1, fallback_offset)
        fallback_offset += 1
        if len(selected) > before:
            continue
        if selected:
            selected.append(selected[-1])
        else:
            selected.append(0)
        if fallback_offset > len(all_group_buckets) * 2 + 32:
            break

    return [rows[idx] for idx in selected[:batch_size]]


def _sample_full_base_batch(
    rows: list[dict[str, Any]],
    step_idx: int,
    batch_size: int,
    loss_weights: dict[str, float] | None = None,
    loss_buckets: dict[str, list[int]] | None = None,
    sampling_strategy: str = "balanced_deterministic",
    sampler_seed: int | None = None,
    grouped_loss_buckets: dict[str, dict[str, list[int]]] | None = None,
    all_group_buckets: dict[str, list[int]] | None = None,
    grouped_quota_buckets: dict[str, dict[str, list[int]]] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    strategy = str(sampling_strategy or "balanced_deterministic")
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"unsupported full-base sampling_strategy: {strategy}")
    if strategy == BENCHMARK_TRANSITION_BALANCED_RANDOM:
        return _sample_full_base_benchmark_transition_balanced_batch(
            rows,
            step_idx,
            batch_size,
            loss_weights=loss_weights,
            loss_buckets=loss_buckets,
            sampler_seed=sampler_seed,
            grouped_loss_buckets=grouped_loss_buckets,
            all_group_buckets=all_group_buckets,
        )
    if strategy == BENCHMARK_TRANSITION_QUOTA_RANDOM:
        return _sample_full_base_benchmark_transition_quota_batch(
            rows,
            step_idx,
            batch_size,
            loss_weights=loss_weights,
            loss_buckets=loss_buckets,
            sampler_seed=sampler_seed,
            all_group_buckets=all_group_buckets,
            grouped_quota_buckets=grouped_quota_buckets,
        )
    weights = _normalize_loss_weights(loss_weights)
    batch_size = max(1, int(batch_size))
    buckets = loss_buckets if loss_buckets is not None else _build_loss_buckets(rows, weights)
    selected: list[int] = []
    selected_set: set[int] = set()
    rng = random.Random((0 if sampler_seed is None else int(sampler_seed)) * 1_000_003 + int(step_idx) * 97_409)

    def add_from_bucket(key: str, offset: int = 0) -> None:
        if len(selected) >= batch_size:
            return
        bucket = buckets.get(key) or []
        if not bucket:
            return
        start = rng.randrange(len(bucket)) if strategy == "balanced_random" else (step_idx - 1 + offset) % len(bucket)
        for probe in range(len(bucket)):
            idx = bucket[(start + probe) % len(bucket)]
            if idx not in selected_set or len(selected_set) >= len(rows):
                selected.append(idx)
                selected_set.add(idx)
                return

    priority = ["L_policy", "L_trans_skill_ce", "STOP", "belief", "routing", "L_trans"]
    for offset, key in enumerate(priority):
        add_from_bucket(key, offset=offset)

    if strategy == "balanced_random":
        if len(selected) >= batch_size:
            return [rows[idx] for idx in selected[:batch_size]]
        start = rng.randrange(len(rows))
        for offset in range(len(rows)):
            idx = (start + offset) % len(rows)
            if len(selected) >= batch_size:
                break
            if idx not in selected_set or len(selected_set) >= len(rows):
                selected.append(idx)
                selected_set.add(idx)
    else:
        start = ((step_idx - 1) * batch_size) % len(rows)
        offset = 0
        while len(selected) < batch_size:
            idx = (start + offset) % len(rows)
            if idx not in selected_set or len(selected_set) >= len(rows):
                selected.append(idx)
                selected_set.add(idx)
            offset += 1
            if offset > len(rows) * 2 and selected:
                selected.append(selected[-1])

    return [rows[idx] for idx in selected[:batch_size]]


def _select_stage0_handoff_training_subset(
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
    batch_size: int,
    loss_weights: dict[str, float],
    sample_multiplier: float | None,
    sampling_strategy: str = "balanced_deterministic",
    sampler_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_multiplier is None or float(sample_multiplier) <= 0.0:
        return rows, {
            "enabled": False,
            "source_rows": len(rows),
            "selected_rows": len(rows),
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "sample_multiplier": None,
            "budget_steps": int(max_steps),
            "sampling_strategy": str(sampling_strategy),
            "reason": "disabled",
        }
    if not rows:
        return rows, {
            "enabled": True,
            "source_rows": 0,
            "selected_rows": 0,
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "sample_multiplier": float(sample_multiplier),
            "budget_steps": 0,
            "sampling_strategy": str(sampling_strategy),
            "reason": "empty_rows",
        }
    max_steps = max(1, int(max_steps))
    batch_size = max(1, int(batch_size))
    multiplier = float(sample_multiplier)
    budget_steps = max(max_steps, int(max_steps * multiplier + 0.999999))
    buckets = _build_loss_buckets(rows, loss_weights)
    grouped_loss_buckets = (
        _build_transition_grouped_loss_buckets(rows, buckets)
        if str(sampling_strategy) == BENCHMARK_TRANSITION_BALANCED_RANDOM
        else None
    )
    all_group_buckets = (
        _group_indices_by_transition_relation(rows, list(range(len(rows))))
        if str(sampling_strategy) in GROUPED_SAMPLING_STRATEGIES
        else None
    )
    grouped_quota_buckets = (
        _build_quota_grouped_buckets(rows, buckets)
        if str(sampling_strategy) == BENCHMARK_TRANSITION_QUOTA_RANDOM
        else None
    )
    row_id_to_index = {id(row): idx for idx, row in enumerate(rows)}
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    for step_idx in range(1, budget_steps + 1):
        for row in _sample_full_base_batch(
            rows,
            step_idx,
            batch_size,
            loss_weights,
            loss_buckets=buckets,
            sampling_strategy=sampling_strategy,
            sampler_seed=sampler_seed,
            grouped_loss_buckets=grouped_loss_buckets,
            all_group_buckets=all_group_buckets,
            grouped_quota_buckets=grouped_quota_buckets,
        ):
            idx = row_id_to_index[id(row)]
            if idx in selected_set:
                continue
            selected_set.add(idx)
            selected_indices.append(idx)
            if len(selected_indices) >= len(rows):
                break
        if len(selected_indices) >= len(rows):
            break
    selected_rows = [rows[idx] for idx in selected_indices]
    return selected_rows, {
        "enabled": True,
        "source_rows": len(rows),
        "selected_rows": len(selected_rows),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "sample_multiplier": multiplier,
        "budget_steps": budget_steps,
        "sample_slots": budget_steps * batch_size,
        "selection_order": (
            "seeded_benchmark_transition_quota_training_sampler"
            if str(sampling_strategy) == BENCHMARK_TRANSITION_QUOTA_RANDOM
            else
            "seeded_benchmark_transition_balanced_training_sampler"
            if str(sampling_strategy) == BENCHMARK_TRANSITION_BALANCED_RANDOM
            else
            "seeded_balanced_random_training_sampler"
            if str(sampling_strategy) == "balanced_random"
            else "deterministic_training_sampler_order"
        ),
        "sampling_strategy": str(sampling_strategy),
        "reason": "avoid_precomputing_stage0_topm_for_rows_unreachable_by_current_training_budget",
    }


def _normalize_loss_weights(loss_weights: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_LOSS_WEIGHTS)
    for key, value in (loss_weights or {}).items():
        if key in merged:
            merged[key] = float(value)
    return merged


def _counterfactual_warmup_scale(global_step: int, max_steps: int, fraction: float) -> float:
    fraction = float(fraction)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("counterfactual_warmup_fraction must be in [0, 1]")
    warmup_steps = fraction * max(1.0, float(max_steps))
    if warmup_steps <= 0.0:
        return 1.0
    return min(1.0, max(0.0, float(global_step) / warmup_steps))


def _validate_counterfactual_hyperparameters(
    *,
    counterfactual_gain_margin: float,
    counterfactual_safety_tolerance: float,
    counterfactual_gain_weight: float,
    counterfactual_safety_weight: float,
) -> None:
    values = {
        "counterfactual_gain_margin": float(counterfactual_gain_margin),
        "counterfactual_safety_tolerance": float(counterfactual_safety_tolerance),
        "counterfactual_gain_weight": float(counterfactual_gain_weight),
        "counterfactual_safety_weight": float(counterfactual_safety_weight),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _embedding_cache_policy(
    *,
    mode: str,
    row_count: int,
    qwen_external_encoder: bool,
    max_rows: int,
) -> dict[str, Any]:
    mode = str(mode or "auto")
    if mode not in {"auto", "always", "never"}:
        raise ValueError(f"unsupported embedding_cache_mode: {mode}")
    max_rows = max(0, int(max_rows))
    if mode == "never":
        cache_enabled = False
        reason = "disabled_by_embedding_cache_mode"
    elif mode == "always":
        cache_enabled = True
        reason = "forced_by_embedding_cache_mode"
    elif int(row_count) > max_rows:
        cache_enabled = False
        reason = "qwen_external_large_dataset_auto_skip" if qwen_external_encoder else "large_dataset_auto_skip"
    else:
        cache_enabled = True
        reason = "auto_cache_enabled"
    return {
        "mode": mode,
        "cache_enabled": bool(cache_enabled),
        "reason": reason,
        "row_count": int(row_count),
        "max_rows": int(max_rows),
        "qwen_external_encoder": bool(qwen_external_encoder),
    }


def _skipped_embedding_cache_report(kind: str, rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "used": False,
        "kind": str(kind),
        "row_count": len(rows),
        "reason": str(policy["reason"]),
        "embedding_cache_mode": str(policy["mode"]),
        "max_rows": int(policy["max_rows"]),
        "qwen_external_encoder": bool(policy["qwen_external_encoder"]),
    }


def _apply_skill_text_format(model: Any, model_config: dict[str, Any], skill_text_format: str | None) -> dict[str, Any]:
    current = str(model_config.get("skill_text_format") or getattr(getattr(model, "config", None), "skill_text_format", "clstr"))
    if not skill_text_format:
        return {"skill_text_format": current, "skill_text_format_override": False}
    serializer = _skill_text_serializer(str(skill_text_format))
    skill_table = getattr(model, "skill_table", None)
    if skill_table is not None and hasattr(skill_table, "skill_text_fn"):
        skill_table.skill_text_fn = serializer
    if getattr(model, "config", None) is not None:
        setattr(model.config, "skill_text_format", str(skill_text_format))
    model_config["skill_text_format"] = str(skill_text_format)
    return {
        "skill_text_format": str(skill_text_format),
        "skill_text_format_override": str(skill_text_format) != current,
        "previous_skill_text_format": current,
    }


def _routing_checkpoint_payload(checkpoint_path: str | Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"routing checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"routing checkpoint must be a dict payload, got {type(payload).__name__}")
    return payload


def _checkpoint_state_from_payload(payload: Any) -> dict[str, Any]:
    raw_state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_state, dict):
        raise TypeError(f"routing checkpoint must contain a state dict, got {type(raw_state).__name__}")
    return raw_state


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_full_base_warm_start_model_state(
    model: Any,
    warm_start_checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    if warm_start_checkpoint_path is None:
        return {"enabled": False, "path": None, "loaded": False}
    path = Path(warm_start_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"warm-start checkpoint must be a dict payload, got {type(payload).__name__}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("warm-start checkpoint must contain model_state_dict")
    load_report = load_compatible_state_dict(
        model,
        state,
        partial_load_mode="stage2_counterfactual_warm_start_compatible_state",
    )
    if not bool(load_report.get("loaded")):
        raise ValueError(f"warm-start checkpoint loaded no compatible model keys: {path}")
    return {
        "enabled": True,
        "path": str(path),
        "checkpoint_sha256": _file_sha256(path),
        "checkpoint_stage": payload.get("stage"),
        "checkpoint_step": int(payload.get("step", 0) or 0),
        "optimizer_loaded": False,
        "optimizer_load_reason": "fresh_counterfactual_repair_optimizer",
        **load_report,
    }


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_full_base_resume_model_state(
    *,
    model: Any,
    action_adapter: UniversalActionAdapter,
    native_policy_head: bool,
    resume_checkpoint_path: str | Path | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if resume_checkpoint_path is None:
        return None, {
            "enabled": False,
            "path": None,
            "start_step": 0,
            "trained_steps_this_run": 0,
            "final_step": 0,
            "model_loaded": False,
            "optimizer_loaded": False,
            "optimizer_load_reason": "disabled",
        }
    path = Path(resume_checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"resume checkpoint must be a dict payload, got {type(payload).__name__}")
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise TypeError("resume checkpoint must contain model_state_dict")
    model_load_report = load_compatible_state_dict(
        model,
        raw_state,
        partial_load_mode="full_base_resume_checkpoint_compatible_state",
    )
    adapter_report: dict[str, Any] = {"loaded": False, "reason": "native_policy_head"}
    adapter_state = payload.get("universal_action_adapter_state_dict")
    if not native_policy_head:
        if isinstance(adapter_state, dict):
            result = action_adapter.load_state_dict(adapter_state, strict=False)
            adapter_report = {
                "loaded": True,
                "missing_keys": sorted(getattr(result, "missing_keys", []) or []),
                "unexpected_keys": sorted(getattr(result, "unexpected_keys", []) or []),
            }
        else:
            adapter_report = {"loaded": False, "reason": "missing_universal_action_adapter_state_dict"}
    try:
        start_step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        start_step = 0
    return payload, {
        "enabled": True,
        "path": str(path),
        "checkpoint_stage": payload.get("stage"),
        "start_step": start_step,
        "trained_steps_this_run": 0,
        "final_step": start_step,
        "model_loaded": bool(model_load_report.get("loaded")),
        "model_load_report": model_load_report,
        "universal_action_adapter_load_report": adapter_report,
        "optimizer_loaded": False,
        "optimizer_load_reason": "pending_optimizer_init",
    }


def _load_full_base_resume_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    resume_payload: dict[str, Any] | None,
    resume_report: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    report = dict(resume_report)
    if resume_payload is None:
        return report
    optimizer_state = resume_payload.get("optimizer_state_dict")
    if optimizer_state is None:
        report["optimizer_loaded"] = False
        report["optimizer_load_reason"] = "missing_optimizer_state_dict"
        return report
    if not isinstance(optimizer_state, dict):
        raise TypeError(f"optimizer_state_dict must be a dict, got {type(optimizer_state).__name__}")
    optimizer.load_state_dict(optimizer_state)
    _optimizer_state_to_device(optimizer, device)
    report["optimizer_loaded"] = True
    report["optimizer_load_reason"] = "loaded"
    return report


def _can_skip_skill_table_rebuild_from_routing_checkpoint(model: Any, payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    raw_state = _checkpoint_state_from_payload(payload)
    if "skill_table.E" not in raw_state:
        return False
    current = model.state_dict() if hasattr(model, "state_dict") else {}
    if "skill_table.E" not in current:
        return False
    value = raw_state["skill_table.E"]
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tuple(current["skill_table.E"].shape) == tuple(tensor.shape)


def _load_routing_checkpoint_into_model(
    model: Any,
    checkpoint_path: str | Path | None,
    payload: dict[str, Any] | None = None,
    skill_table_rebuild_skipped: bool = False,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "loaded": False,
            "path": None,
            "reason": "not_provided",
            "loaded_keys": [],
            "skipped_keys": [],
            "skill_table_rebuild_skipped": bool(skill_table_rebuild_skipped),
        }
    path = Path(checkpoint_path)
    payload = payload if payload is not None else _routing_checkpoint_payload(path)
    raw_state = _checkpoint_state_from_payload(payload)
    current = model.state_dict() if hasattr(model, "state_dict") else {}
    allowed_exact = {
        "skill_table.E",
        "skill_table.logit_scale_retr",
        "skill_table.skill_bias_retr",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    }
    allowed_prefixes = ("encoder.proj.", "skill_table.W.")
    compatible: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []
    shape_mismatched: dict[str, dict[str, list[int]]] = {}
    for key, value in raw_state.items():
        key = str(key)
        if key not in allowed_exact and not any(key.startswith(prefix) for prefix in allowed_prefixes):
            skipped_keys.append(key)
            continue
        if key not in current:
            skipped_keys.append(key)
            continue
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tuple(current[key].shape) != tuple(tensor.shape):
            shape_mismatched[key] = {
                "checkpoint": list(tensor.shape),
                "model": list(current[key].shape),
            }
            skipped_keys.append(key)
            continue
        compatible[key] = tensor
    load_report = load_compatible_state_dict(
        model,
        compatible,
        partial_load_mode="manual_copy_compatible_routing_keys",
    )
    return {
        "path": str(path),
        "stage": payload.get("stage") if isinstance(payload, dict) else None,
        "skill_table_rebuild_skipped": bool(skill_table_rebuild_skipped),
        "allowed_keys": sorted(list(allowed_exact) + list(allowed_prefixes)),
        **load_report,
        "skipped_keys": sorted(set(skipped_keys) | set(load_report.get("skipped_keys") or [])),
        "shape_mismatched": {
            **shape_mismatched,
            **dict(load_report.get("shape_mismatched") or {}),
        },
    }


def _policy_candidate_text_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    candidate_rows: list[list[str]] = []
    for row in rows:
        candidates = [str(item) for item in row.get("admissible_actions") or []]
        expert = str(row.get("expert_action") or row.get("action_text") or "")
        if expert and expert not in candidates:
            candidates = [expert]
        candidate_rows.append(candidates)
    return candidate_rows


def _q_success_targets(
    rows: list[dict[str, Any]],
    max_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros(len(rows), max_width, dtype=torch.float32, device=device)
    weights = torch.zeros(len(rows), max_width, dtype=torch.float32, device=device)
    for row_idx, row in enumerate(rows):
        raw_labels = list(row.get("q_success_labels") or [])
        raw_weights = list(row.get("q_success_label_weights") or [])
        candidates = [str(item) for item in row.get("admissible_actions") or []]
        expert = str(row.get("expert_action") or row.get("action_text") or "")
        if not raw_labels and candidates:
            raw_labels = [1.0 if action == expert else 0.0 for action in candidates]
        if not raw_weights and raw_labels:
            raw_weights = [1.0 for _ in raw_labels]
        for col_idx in range(min(max_width, len(raw_labels))):
            labels[row_idx, col_idx] = float(raw_labels[col_idx])
            weight_value = raw_weights[col_idx] if col_idx < len(raw_weights) else 1.0
            weights[row_idx, col_idx] = float(weight_value)
    return labels, weights


def _retrieval_contrastive_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_negatives: int = 32,
    hard_ratio: float = 0.5,
    return_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if logits.ndim != 2:
        raise ValueError("retrieval logits must be rank-2 [batch, num_skills]")
    if labels.ndim != 1:
        raise ValueError("retrieval labels must be rank-1 [batch]")
    device = logits.device
    losses: list[torch.Tensor] = []
    audit: dict[str, Any] = {"positive_indices": [], "negative_indices": []}
    width = logits.size(-1)
    if width <= 0:
        loss = logits.sum() * 0.0
        return (loss, audit) if return_audit else loss

    for row_idx in range(logits.size(0)):
        pos_idx = int(labels[row_idx].detach().cpu().item())
        if pos_idx < 0 or pos_idx >= width:
            continue
        row = logits[row_idx]
        masked = row.clone()
        masked[pos_idx] = float("-inf")
        neg_budget = min(max(int(num_negatives), 0), max(width - 1, 0))
        hard_count = min(int(round(neg_budget * float(hard_ratio))), neg_budget)
        rand_count = max(neg_budget - hard_count, 0)

        hard_idx = torch.empty(0, dtype=torch.long, device=device)
        if hard_count > 0:
            hard_idx = torch.topk(masked, k=hard_count).indices

        rand_idx: list[int] = []
        if rand_count > 0:
            forbidden = {pos_idx, *[int(idx) for idx in hard_idx.detach().cpu().tolist()]}
            available = [idx for idx in range(width) if idx not in forbidden]
            if available:
                perm = torch.randperm(len(available), device=device)
                rand_idx = [available[idx] for idx in perm[:rand_count].detach().cpu().tolist()]

        negative_idx = [int(idx) for idx in hard_idx.detach().cpu().tolist()] + rand_idx
        candidate_logits = [row[pos_idx]]
        if negative_idx:
            candidate_logits.append(row[torch.tensor(negative_idx, dtype=torch.long, device=device)])
            candidates = torch.cat([candidate_logits[0].unsqueeze(0), candidate_logits[1]])
        else:
            candidates = candidate_logits[0].unsqueeze(0)
        target = torch.tensor([0], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(candidates.unsqueeze(0), target))
        audit["positive_indices"].append(pos_idx)
        audit["negative_indices"].append(negative_idx)

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = logits.sum() * 0.0
    return (loss, audit) if return_audit else loss


def _weighted_term(metrics: dict[str, float], key: str, loss: torch.Tensor, weight: float) -> torch.Tensor:
    weighted = loss * float(weight)
    metrics["weighted_loss_terms"][key] = float(weighted.detach().cpu().item())
    return weighted


def _pad_or_trim_logits(logits: torch.Tensor, skill_count: int) -> torch.Tensor:
    if logits.size(-1) < skill_count:
        pad = torch.zeros(logits.size(0), skill_count - logits.size(-1), dtype=logits.dtype, device=logits.device)
        logits = torch.cat([logits, pad], dim=-1)
    return logits[:, :skill_count]


def _transition_skill_logits(model: Any, pred: torch.Tensor, skill_count: int) -> tuple[torch.Tensor, str]:
    trans_head = getattr(model, "trans_head", None)
    if trans_head is not None:
        try:
            candidate_ids = torch.arange(skill_count, dtype=torch.long, device=pred.device)
            candidate_embs = model_action_embeddings(model, candidate_ids)
            if candidate_embs is None:
                raise AttributeError("model has no action_embeddings")
            if candidate_embs.ndim == 2:
                candidate_embs = candidate_embs.unsqueeze(0).expand(pred.size(0), -1, -1)
            logits = trans_head(pred, candidate_embs)
            if logits.ndim == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            if logits.ndim == 2:
                return _pad_or_trim_logits(logits, skill_count), "native_trans_head"
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    action_emb = getattr(model, "action_emb", None)
    if trans_head is not None and action_emb is not None and callable(action_emb):
        try:
            candidate_ids = torch.arange(skill_count, dtype=torch.long, device=pred.device)
            candidate_embs = action_emb(candidate_ids)
            if candidate_embs.ndim == 2:
                candidate_embs = candidate_embs.unsqueeze(0).expand(pred.size(0), -1, -1)
            logits = trans_head(pred, candidate_embs)
            if logits.ndim == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            if logits.ndim == 2:
                return _pad_or_trim_logits(logits, skill_count), "native_trans_head"
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    skill_table = getattr(model, "skill_table", None)
    if skill_table is not None and hasattr(skill_table, "retrieval_logits"):
        logits = skill_table.retrieval_logits(pred)
    elif skill_table is not None and hasattr(skill_table, "logits"):
        logits = skill_table.logits(pred)
    elif skill_table is not None and callable(skill_table):
        logits = skill_table(pred)
    else:
        emb = getattr(skill_table, "E", None) if skill_table is not None else None
        if emb is None:
            return torch.zeros(pred.size(0), skill_count, dtype=pred.dtype, device=pred.device), "zero_fallback"
        logits = pred @ emb.to(pred.device).t()
    return _pad_or_trim_logits(logits, skill_count), "skill_table_logits"


def _candidate_transition_logits(
    model: Any,
    pred: torch.Tensor,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
    candidate_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    if candidate_ids is None:
        return _transition_skill_logits(model, pred, skill_count)
    trans_head = getattr(model, "trans_head", None)
    candidate_embs = model_action_embeddings(model, candidate_ids)
    if candidate_embs is None:
        action_emb = getattr(model, "action_emb", None)
        if action_emb is not None and callable(action_emb):
            try:
                candidate_embs = action_emb(candidate_ids)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                candidate_embs = None
    if candidate_embs is None:
        emb = getattr(getattr(model, "skill_table", None), "E", None)
        if emb is not None:
            flat = candidate_ids.reshape(-1).to(device=pred.device, dtype=torch.long)
            candidate_embs = emb.to(pred.device).index_select(0, flat).view(*candidate_ids.shape, -1)
    if trans_head is not None and candidate_embs is not None:
        logits = trans_head(pred, candidate_embs)
        if logits.ndim == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        head_type = "native_trans_head_stage0_topm"
    else:
        full_logits, base_head_type = _transition_skill_logits(model, pred, skill_count)
        logits = full_logits.gather(1, candidate_ids.to(device=full_logits.device, dtype=torch.long))
        head_type = f"{base_head_type}_stage0_topm"
    if candidate_valid_mask is not None:
        logits = logits.masked_fill(
            ~candidate_valid_mask.to(device=logits.device),
            torch.finfo(logits.dtype).min,
        )
    return logits, head_type


def _stage0_rank_prior_logits_like(
    logits: torch.Tensor,
    candidate_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    width = int(logits.size(-1))
    if width <= 0:
        return logits.sum(dim=-1, keepdim=True)[:, :0]
    scores = [-math.log(float(rank)) for rank in range(1, width + 1)]
    prior = torch.tensor(scores, dtype=logits.dtype, device=logits.device).unsqueeze(0).expand(logits.size(0), -1)
    if candidate_valid_mask is not None:
        prior = prior.masked_fill(
            ~candidate_valid_mask.to(device=prior.device).to(torch.bool),
            torch.finfo(prior.dtype).min,
        )
    return prior


def _stage0_candidate_prior_scores_tensor(
    rows: list[dict[str, Any]],
    candidate_rows: list[list[int]],
    *,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
) -> torch.Tensor | None:
    if not rows or not candidate_rows:
        return None
    padded_tensors: list[torch.Tensor] = []
    any_real_score = False
    calibration = str(calibration or STAGE0_SCORE_PRIOR_CALIBRATION)
    if calibration == "off":
        return None
    if calibration not in STAGE0_SCORE_PRIOR_CALIBRATIONS:
        raise ValueError(f"unsupported stage0 score prior calibration: {calibration}")
    for row, candidates in zip(rows, candidate_rows):
        score_map: dict[int, float] = {}
        for index_key, score_key in (
            ("stage0_next_candidate_skill_indices", "stage0_next_candidate_skill_scores"),
            ("stage0_candidate_skill_indices", "stage0_candidate_skill_scores"),
        ):
            indices = row.get(index_key)
            scores = row.get(score_key)
            if not isinstance(indices, list) or not isinstance(scores, list) or len(indices) != len(scores):
                continue
            for idx, score in zip(indices, scores):
                try:
                    score_map.setdefault(int(idx), float(score))
                except (TypeError, ValueError):
                    continue
        candidate_width = max(1, len(candidates))
        rank_fallback = _stage0_rank_prior_logits_like(
            torch.zeros(1, candidate_width, dtype=dtype, device=device)
        ).squeeze(0)
        row_scores: list[float] = []
        for pos, idx in enumerate(candidates):
            idx = int(idx)
            if idx in score_map:
                row_scores.append(float(score_map[idx]))
                any_real_score = True
            else:
                row_scores.append(float(rank_fallback[min(pos, rank_fallback.numel() - 1)].detach().cpu().item()))
        if len(row_scores) < int(width):
            row_scores.extend([0.0] * (int(width) - len(row_scores)))
        row_tensor = torch.tensor(row_scores[: int(width)], dtype=dtype, device=device)
        valid_width = min(len(candidates), int(width))
        if calibration == "rank_std" and valid_width > 1:
            values = row_tensor[:valid_width]
            target = rank_fallback[:valid_width].to(device=device, dtype=dtype)
            source_std = values.float().std(unbiased=False)
            target_std = target.float().std(unbiased=False)
            if float(source_std.detach().cpu().item()) > 1.0e-8 and float(target_std.detach().cpu().item()) > 0.0:
                centered = (values.float() - values.float().mean()) / source_std
                calibrated = centered * target_std + target.float().mean()
                row_tensor = row_tensor.clone()
                row_tensor[:valid_width] = calibrated.to(dtype=dtype)
            else:
                row_tensor = row_tensor.clone()
                row_tensor[:valid_width] = target.to(dtype=dtype)
        elif calibration == "rank":
            row_tensor = row_tensor.clone()
            valid_width = min(len(candidates), int(width))
            if valid_width > 0:
                row_tensor[:valid_width] = rank_fallback[:valid_width].to(device=device, dtype=dtype)
        padded_tensors.append(row_tensor)
    if not any_real_score:
        return None
    return torch.stack(padded_tensors, dim=0)


def _candidate_embeddings_for_ids(
    model: Any,
    query: torch.Tensor,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_ids is None:
        ids = torch.arange(skill_count, dtype=torch.long, device=query.device).unsqueeze(0).expand(query.size(0), -1)
    else:
        ids = candidate_ids.to(device=query.device, dtype=torch.long)
    candidate_embs = model_action_embeddings(model, ids)
    if candidate_embs is None:
        action_emb = getattr(model, "action_emb", None)
        if action_emb is not None and callable(action_emb):
            try:
                candidate_embs = action_emb(ids)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                candidate_embs = None
    if candidate_embs is None:
        emb = getattr(getattr(model, "skill_table", None), "E", None)
        if emb is not None:
            flat = ids.reshape(-1)
            candidate_embs = emb.to(query.device).index_select(0, flat).view(*ids.shape, -1)
    if candidate_embs is None:
        eye = torch.eye(max(skill_count, query.size(-1)), dtype=query.dtype, device=query.device)[:skill_count, : query.size(-1)]
        candidate_embs = eye.index_select(0, ids.reshape(-1)).view(*ids.shape, -1)
    return ids, candidate_embs.to(device=query.device, dtype=query.dtype)


def _ensure_gated_temporal_reranker(
    model: Any,
    *,
    dim: int,
    query_dim: int,
    lambda_max: float,
    context_top_k: int,
    device: torch.device,
) -> GatedTemporalReranker:
    reranker = getattr(model, "gated_temporal_reranker", None)
    existing_query_dim = int(getattr(getattr(reranker, "config", None), "query_dim", 0) or getattr(getattr(reranker, "config", None), "dim", 0))
    if (
        not isinstance(reranker, GatedTemporalReranker)
        or int(reranker.config.dim) != int(dim)
        or existing_query_dim != int(query_dim)
    ):
        reranker = GatedTemporalReranker(
            GatedTemporalConfig(
                dim=int(dim),
                query_dim=int(query_dim),
                lambda_max=float(lambda_max),
                context_top_k=int(context_top_k),
            )
        )
        setattr(model, "gated_temporal_reranker", reranker)
    reranker.config.lambda_max = float(lambda_max)
    reranker.config.context_top_k = int(context_top_k)
    return reranker.to(device)


def _gated_temporal_dims_for_model(model: Any, model_config: dict[str, Any]) -> tuple[int, int]:
    query_dim = int(model_config.get("d") or model_config.get("hidden_size") or 128)
    action_proj = getattr(model, "action_proj", None)
    if action_proj is not None and hasattr(action_proj, "out_features"):
        candidate_dim = int(action_proj.out_features)
    else:
        action_emb = getattr(model, "action_emb", None)
        if action_emb is not None and hasattr(action_emb, "embedding_dim"):
            candidate_dim = int(action_emb.embedding_dim)
        else:
            candidate_dim = int(model_config.get("d_a") or query_dim)
    return candidate_dim, query_dim


def _prepare_gated_temporal_reranker_for_training(
    model: Any,
    model_config: dict[str, Any],
    *,
    lambda_max: float,
    context_top_k: int,
    device: torch.device,
) -> GatedTemporalReranker:
    candidate_dim, query_dim = _gated_temporal_dims_for_model(model, model_config)
    reranker = getattr(model, "gated_temporal_reranker", None)
    existing_query_dim = int(getattr(getattr(reranker, "config", None), "query_dim", 0) or getattr(getattr(reranker, "config", None), "dim", 0))
    if (
        not isinstance(reranker, GatedTemporalReranker)
        or int(reranker.config.dim) != int(candidate_dim)
        or existing_query_dim != int(query_dim)
    ):
        reranker = GatedTemporalReranker(
            GatedTemporalConfig(
                dim=int(candidate_dim),
                query_dim=int(query_dim),
                lambda_max=float(lambda_max),
                context_top_k=int(context_top_k),
            )
        )
        setattr(model, "gated_temporal_reranker", reranker)
    reranker.config.lambda_max = float(lambda_max)
    reranker.config.context_top_k = int(context_top_k)
    return reranker.to(device)


def _transition_prior_residual_logits(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    current_labels: torch.Tensor,
    obs_emb: torch.Tensor,
    action_emb: torch.Tensor | None,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
    candidate_valid_mask: torch.Tensor | None = None,
    residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
    prior_pred = _transition_prior_prediction(model, h, m_obs, current_labels, obs_emb)
    residual_pred = _transition_prediction(
        model,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb=action_emb,
    )
    prior_logits, prior_head_type = _candidate_transition_logits(
        model,
        prior_pred,
        skill_count,
        candidate_ids=candidate_ids,
        candidate_valid_mask=candidate_valid_mask,
    )
    residual_logits, residual_head_type = _candidate_transition_logits(
        model,
        residual_pred,
        skill_count,
        candidate_ids=candidate_ids,
        candidate_valid_mask=candidate_valid_mask,
    )
    combined = prior_logits + float(residual_lambda) * residual_logits
    head_type = prior_head_type if prior_head_type == residual_head_type else f"{prior_head_type}+{residual_head_type}"
    return combined, head_type, prior_logits, residual_logits


def _transition_stage0_rank_prior_residual_logits(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    current_labels: torch.Tensor,
    obs_emb: torch.Tensor,
    action_emb: torch.Tensor | None,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
    candidate_valid_mask: torch.Tensor | None = None,
    candidate_stage0_prior_scores: torch.Tensor | None = None,
    residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
    residual_pred = _transition_prediction(
        model,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb=action_emb,
    )
    residual_logits, residual_head_type = _candidate_transition_logits(
        model,
        residual_pred,
        skill_count,
        candidate_ids=candidate_ids,
        candidate_valid_mask=candidate_valid_mask,
    )
    if candidate_stage0_prior_scores is not None:
        prior_logits = candidate_stage0_prior_scores.to(device=residual_logits.device, dtype=residual_logits.dtype)
        if candidate_valid_mask is not None:
            prior_logits = prior_logits.masked_fill(
                ~candidate_valid_mask.to(device=prior_logits.device).to(torch.bool),
                torch.finfo(prior_logits.dtype).min,
            )
        prior_name = "stage0_score_prior"
    else:
        prior_logits = _stage0_rank_prior_logits_like(residual_logits, candidate_valid_mask=candidate_valid_mask)
        prior_name = "stage0_rank_prior"
    combined = prior_logits + float(residual_lambda) * residual_logits
    return combined, f"{prior_name}+{residual_head_type}", prior_logits, residual_logits


def _transition_stage0_prior_gated_temporal_logits(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    current_labels: torch.Tensor,
    obs_emb: torch.Tensor,
    action_emb: torch.Tensor | None,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
    candidate_valid_mask: torch.Tensor | None = None,
    candidate_stage0_prior_scores: torch.Tensor | None = None,
    lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
    query = _transition_prediction(
        model,
        h,
        m_obs,
        current_labels,
        obs_emb,
        action_emb=action_emb,
    )
    ids, candidate_embs = _candidate_embeddings_for_ids(
        model,
        query,
        skill_count,
        candidate_ids=candidate_ids,
    )
    if candidate_stage0_prior_scores is not None:
        prior_logits = candidate_stage0_prior_scores.to(device=query.device, dtype=query.dtype)
        if candidate_valid_mask is not None:
            prior_logits = prior_logits.masked_fill(
                ~candidate_valid_mask.to(device=prior_logits.device).to(torch.bool),
                torch.finfo(prior_logits.dtype).min,
            )
    else:
        prior_logits = _stage0_rank_prior_logits_like(
            torch.zeros(candidate_embs.size(0), candidate_embs.size(1), dtype=query.dtype, device=query.device),
            candidate_valid_mask=candidate_valid_mask,
        )
    reranker = _ensure_gated_temporal_reranker(
        model,
        dim=int(candidate_embs.size(-1)),
        query_dim=int(query.size(-1)),
        lambda_max=float(lambda_max),
        context_top_k=int(context_top_k),
        device=query.device,
    )
    output = reranker(
        query,
        candidate_embs,
        prior_logits,
        candidate_valid_mask,
        current_labels=current_labels,
        candidate_ids=ids,
    )
    setattr(
        model,
        "_last_gated_temporal_aux",
        {
            "lambda_t": output.lambda_t,
            "candidate_context": output.candidate_context,
            "gate_features": output.gate_features,
        },
    )
    return output.final_logits, GATED_TEMPORAL_TRANSITION_SCORING_MODE, output.prior_logits, output.residual_logits


def _transition_candidate_logits_for_mode(
    model: Any,
    h: torch.Tensor,
    m_obs: torch.Tensor,
    current_labels: torch.Tensor,
    obs_emb: torch.Tensor,
    action_emb: torch.Tensor | None,
    skill_count: int,
    *,
    candidate_ids: torch.Tensor | None = None,
    candidate_valid_mask: torch.Tensor | None = None,
    candidate_stage0_prior_scores: torch.Tensor | None = None,
    residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor]:
    setattr(model, "_last_gated_temporal_aux", None)
    if scoring_mode == TRANSITION_SCORING_MODE:
        return _transition_stage0_rank_prior_residual_logits(
            model,
            h,
            m_obs,
            current_labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=candidate_valid_mask,
            candidate_stage0_prior_scores=candidate_stage0_prior_scores,
            residual_lambda=residual_lambda,
        )
    if scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE:
        return _transition_stage0_prior_gated_temporal_logits(
            model,
            h,
            m_obs,
            current_labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=candidate_valid_mask,
            candidate_stage0_prior_scores=candidate_stage0_prior_scores,
            lambda_max=gated_temporal_lambda_max,
            context_top_k=gated_temporal_context_top_k,
        )
    if scoring_mode == SKILL_PRIOR_TRANSITION_SCORING_MODE:
        return _transition_prior_residual_logits(
            model,
            h,
            m_obs,
            current_labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=candidate_valid_mask,
            residual_lambda=residual_lambda,
        )
    if scoring_mode == V4_1B_TRANSITION_SCORING_MODE:
        # v4.1b trained transition skill CE with action text as the transition
        # observation argument. Keep this as an explicit compatibility mode so
        # conservative rollback runs do not silently use the newer residual path.
        legacy_obs_emb = action_emb if action_emb is not None else obs_emb
        pred = _transition_prediction(model, h, m_obs, current_labels, legacy_obs_emb, action_emb=None)
        logits, head_type = _candidate_transition_logits(
            model,
            pred,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=candidate_valid_mask,
        )
        return logits, head_type, logits, logits
    raise ValueError(f"unsupported transition scoring mode: {scoring_mode}")


def _ranking_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    rows: list[dict[str, Any]] | None = None,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if logits.numel() == 0 or labels.numel() == 0:
        return {
            "transition_skill_recall@1": 0.0,
            "transition_skill_recall@5": 0.0,
            "transition_skill_mrr": 0.0,
            "transition_skill_ce_candidate_count": 0.0,
        }
    ranks = _ranking_ranks_from_logits(logits, labels, positive_mask=positive_mask)
    metrics = {
        "transition_skill_recall@1": sum(1 for rank in ranks if rank <= 1) / len(ranks),
        "transition_skill_recall@5": sum(1 for rank in ranks if rank <= 5) / len(ranks),
        "transition_skill_mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "transition_skill_ce_candidate_count": float(logits.size(-1)),
    }
    if rows is not None:
        split_ranks: dict[str, list[int]] = {"self": [], "switch": []}
        for row, rank in zip(rows, ranks):
            key = "self" if str(row.get("skill_id") or "") == str(row.get("next_skill_id") or "") else "switch"
            split_ranks[key].append(rank)
        for key, values in split_ranks.items():
            metrics[f"transition_skill_{key}_count"] = float(len(values))
            metrics[f"transition_skill_{key}_recall@1"] = (
                sum(1 for rank in values if rank <= 1) / len(values) if values else 0.0
            )
            metrics[f"transition_skill_{key}_recall@5"] = (
                sum(1 for rank in values if rank <= 5) / len(values) if values else 0.0
            )
            metrics[f"transition_skill_{key}_mrr"] = (
                sum(1.0 / rank for rank in values) / len(values) if values else 0.0
            )
    return metrics


def _ranking_ranks_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
) -> list[int]:
    ranks: list[int] = []
    order = torch.argsort(logits.detach(), dim=-1, descending=True)
    for row_idx, label in enumerate(labels.detach().cpu().tolist()):
        if positive_mask is None:
            positive_indices = torch.tensor([int(label)], dtype=torch.long)
        else:
            positive_indices = positive_mask[row_idx].detach().cpu().nonzero(as_tuple=False).view(-1).to(torch.long)
            if positive_indices.numel() <= 0:
                positive_indices = torch.tensor([int(label)], dtype=torch.long)
        ordered = order[row_idx].detach().cpu()
        positive_positions: list[int] = []
        for positive_idx in positive_indices.tolist():
            positions = (ordered == int(positive_idx)).nonzero(as_tuple=False)
            if positions.numel():
                positive_positions.append(int(positions[0].item()) + 1)
        ranks.append(min(positive_positions) if positive_positions else logits.size(-1) + 1)
    return ranks


def _rank_delta_metrics_vs_prior(
    logits: torch.Tensor,
    prior_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if logits.numel() == 0 or prior_logits.numel() == 0 or labels.numel() == 0:
        return {
            "transition_delta_vs_stage0_prior_recall@1": 0.0,
            "transition_delta_vs_stage0_prior_recall@5": 0.0,
            "transition_delta_vs_stage0_prior_mrr": 0.0,
            "transition_improved_vs_stage0_prior_fraction": 0.0,
            "transition_worse_than_stage0_prior_fraction": 0.0,
        }
    ranks = _ranking_ranks_from_logits(logits, labels, positive_mask=positive_mask)
    prior_ranks = _ranking_ranks_from_logits(prior_logits, labels, positive_mask=positive_mask)
    denom = max(len(ranks), 1)

    def recall_delta(k: int) -> float:
        current = sum(1 for rank in ranks if rank <= k) / denom
        prior = sum(1 for rank in prior_ranks if rank <= k) / denom
        return float(current - prior)

    mrr = sum(1.0 / rank for rank in ranks) / denom
    prior_mrr = sum(1.0 / rank for rank in prior_ranks) / denom
    return {
        "transition_delta_vs_stage0_prior_recall@1": recall_delta(1),
        "transition_delta_vs_stage0_prior_recall@5": recall_delta(5),
        "transition_delta_vs_stage0_prior_mrr": float(mrr - prior_mrr),
        "transition_improved_vs_stage0_prior_fraction": sum(
            1 for rank, prior_rank in zip(ranks, prior_ranks) if rank < prior_rank
        )
        / denom,
        "transition_worse_than_stage0_prior_fraction": sum(
            1 for rank, prior_rank in zip(ranks, prior_ranks) if rank > prior_rank
        )
        / denom,
    }


def _dynamic_static_rank_metrics(
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
    prefix: str,
) -> dict[str, float]:
    if dynamic_logits.numel() == 0 or static_logits.numel() == 0 or labels.numel() == 0:
        return {
            f"{prefix}_dynamic_vs_static_delta_mrr": 0.0,
            f"{prefix}_positive_rank_improved_rows": 0.0,
            f"{prefix}_positive_rank_worsened_rows": 0.0,
            f"{prefix}_argmax_changed_rows": 0.0,
        }
    dynamic_ranks = _ranking_ranks_from_logits(dynamic_logits, labels, positive_mask=positive_mask)
    static_ranks = _ranking_ranks_from_logits(static_logits, labels, positive_mask=positive_mask)
    denom = max(len(dynamic_ranks), 1)
    dynamic_mrr = sum(1.0 / rank for rank in dynamic_ranks) / denom
    static_mrr = sum(1.0 / rank for rank in static_ranks) / denom
    dynamic_argmax = dynamic_logits.detach().argmax(dim=-1)
    static_argmax = static_logits.detach().argmax(dim=-1)
    return {
        f"{prefix}_dynamic_vs_static_delta_mrr": float(dynamic_mrr - static_mrr),
        f"{prefix}_positive_rank_improved_rows": float(
            sum(1 for rank, static_rank in zip(dynamic_ranks, static_ranks) if rank < static_rank)
        ),
        f"{prefix}_positive_rank_worsened_rows": float(
            sum(1 for rank, static_rank in zip(dynamic_ranks, static_ranks) if rank > static_rank)
        ),
        f"{prefix}_argmax_changed_rows": float((dynamic_argmax != static_argmax).sum().detach().cpu().item()),
    }


def _transition_ce_row_weights(
    rows: list[dict[str, Any]],
    *,
    real_multiplier: float,
    injected_multiplier: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        float(injected_multiplier) if bool(row.get("stage0_positive_injected")) else float(real_multiplier)
        for row in rows
    ]
    weights = torch.tensor(values, dtype=dtype, device=device)
    if torch.all(weights <= 0):
        weights = torch.ones_like(weights)
    return weights


def _transition_hard_negative_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    margin: float,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    for row_idx, (label, row) in enumerate(zip(labels.detach().cpu().tolist(), rows)):
        if bool(row.get("stage0_positive_injected")):
            continue
        label_int = int(label)
        if label_int < 0 or label_int >= logits.size(1) or logits.size(1) <= 1:
            continue
        row_logits = logits[row_idx]
        masked = row_logits.clone()
        masked[label_int] = torch.finfo(masked.dtype).min
        hardest_negative = torch.max(masked)
        positive = row_logits[label_int]
        losses.append(F.relu(torch.as_tensor(float(margin), dtype=row_logits.dtype, device=row_logits.device) - positive + hardest_negative))
    if not losses:
        return logits.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def _compute_full_base_loss(
    model: Any,
    action_adapter: UniversalActionAdapter,
    batch: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    transition_real_candidate_ce_multiplier: float = 1.0,
    transition_injected_candidate_ce_multiplier: float = 1.0,
    transition_hard_negative_margin: float = 1.0,
    transition_inventory_mask_mode: str = "off",
    transition_inventory_min_candidates: int = 0,
    transition_loss_type: str = "cross_entropy",
    transition_positive_mode: str = "single",
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    next_skill_pool_mode: str = "stage0_candidates",
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_scale: float = 1.0,
    counterfactual_history_margin: float = 0.1,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
    static_route_teacher: Stage2StaticRouteTeacher | Any | None = None,
    static_route_anchor_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    transition_inventory_mask_mode = str(transition_inventory_mask_mode or "off")
    transition_inventory_min_candidates = max(0, int(transition_inventory_min_candidates or 0))
    transition_loss_type = str(transition_loss_type or "cross_entropy")
    transition_positive_mode = str(transition_positive_mode or "single")
    next_skill_pool_mode = str(next_skill_pool_mode or "stage0_candidates")
    counterfactual_gain_margin = float(counterfactual_gain_margin)
    counterfactual_safety_tolerance = float(counterfactual_safety_tolerance)
    counterfactual_gain_weight = float(counterfactual_gain_weight)
    counterfactual_safety_weight = float(counterfactual_safety_weight)
    counterfactual_scale = float(counterfactual_scale)
    counterfactual_history_margin = float(counterfactual_history_margin)
    safe_memory_residual_bound = float(safe_memory_residual_bound)
    safe_local_candidate_sizes = tuple(int(size) for size in safe_local_candidate_sizes)
    static_route_anchor_weight = float(static_route_anchor_weight)
    _validate_counterfactual_hyperparameters(
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
    )
    transition_residual_lambda = float(transition_residual_lambda)
    transition_scoring_mode = str(transition_scoring_mode or TRANSITION_SCORING_MODE)
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    gated_temporal_lambda_max = float(gated_temporal_lambda_max)
    gated_temporal_kl_alpha = float(gated_temporal_kl_alpha)
    gated_temporal_rank_drop_beta = float(gated_temporal_rank_drop_beta)
    gated_temporal_context_top_k = int(gated_temporal_context_top_k)
    stage0_score_prior_calibration = str(stage0_score_prior_calibration or STAGE0_SCORE_PRIOR_CALIBRATION)
    if stage0_score_prior_calibration not in STAGE0_SCORE_PRIOR_CALIBRATIONS:
        raise ValueError(f"unsupported stage0 score prior calibration: {stage0_score_prior_calibration}")
    if transition_inventory_mask_mode not in TRANSITION_INVENTORY_MASK_MODES:
        raise ValueError(f"unsupported transition inventory mask mode: {transition_inventory_mask_mode}")
    if transition_loss_type not in TRANSITION_LOSS_TYPES:
        raise ValueError(f"unsupported transition loss type: {transition_loss_type}")
    if transition_positive_mode not in TRANSITION_POSITIVE_MODES:
        raise ValueError(f"unsupported transition positive mode: {transition_positive_mode}")
    if transition_scoring_mode not in TRANSITION_SCORING_MODES:
        raise ValueError(f"unsupported transition scoring mode: {transition_scoring_mode}")
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    if next_skill_pool_mode == "full_pool" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("next_skill_pool_mode=full_pool requires route_scorer=unified_memory")
    if next_skill_pool_mode == "full_pool" and transition_inventory_mask_mode != "explicit_only":
        raise ValueError("full-pool next-skill training requires transition_inventory_mask_mode=explicit_only")
    if not 0.0 <= counterfactual_scale <= 1.0:
        raise ValueError("counterfactual_scale must be in [0, 1]")
    if not math.isfinite(counterfactual_history_margin) or counterfactual_history_margin < 0.0:
        raise ValueError("counterfactual_history_margin must be finite and nonnegative")
    if not math.isfinite(safe_memory_residual_bound) or safe_memory_residual_bound <= 0.0:
        raise ValueError("safe_memory_residual_bound must be finite and positive")
    if not safe_local_candidate_sizes or any(size < 2 for size in safe_local_candidate_sizes):
        raise ValueError("safe_local_candidate_sizes must contain integers of at least two")
    if not math.isfinite(static_route_anchor_weight) or static_route_anchor_weight < 0.0:
        raise ValueError("static_route_anchor_weight must be finite and nonnegative")
    if static_route_anchor_weight > 0.0 and static_route_teacher is None:
        raise ValueError("positive static_route_anchor_weight requires a step-zero teacher")
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
    weights = _normalize_loss_weights(loss_weights)
    h = _batch_cached_or_encode(
        model,
        batch,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    skill_count = max(1, len(skill_id_to_idx))
    skill_ids_by_idx = {idx: skill_id for skill_id, idx in skill_id_to_idx.items()}
    logits, m_obs = _skill_logits_and_memory(model, h, skill_count)
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        if not callable(getattr(model, "initial_belief", None)) or not callable(getattr(model, "unified_route_logits", None)):
            raise ValueError("route_scorer=unified_memory requires model.initial_belief and model.unified_route_logits")
        m_obs = model.initial_belief(h)
        logits = model.unified_route_logits(h, m_obs)
    m_static = m_obs
    losses: list[torch.Tensor] = []
    metrics: dict[str, Any] = {
        "weighted_loss_terms": {key: 0.0 for key in LOSS_WEIGHT_KEYS},
        "route_scorer": route_scorer,
        "next_skill_pool_mode": next_skill_pool_mode,
        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
        "stage2_static_route_anchor_rows": 0.0,
        "stage2_static_route_anchor_loss": 0.0,
        "stage2_static_route_anchor_weighted_loss": 0.0,
        "stage2_static_route_teacher_mrr": 0.0,
        "stage2_static_route_student_mrr": 0.0,
        "stage2_step_zero_dynamic_mrr": 0.0,
        "stage2_dynamic_mrr": 0.0,
        "stage2_static_mrr_delta_vs_teacher": 0.0,
        "stage2_dynamic_mrr_delta_vs_step_zero": 0.0,
        "counterfactual_history_loss": 0.0,
        "counterfactual_history_rank_loss": 0.0,
        "counterfactual_history_static_no_regret_loss": 0.0,
        "counterfactual_history_donor_rows": 0.0,
        "counterfactual_history_eligible_rows": 0.0,
        "counterfactual_history_violation_rows": 0.0,
        "counterfactual_history_true_utility": 0.0,
        "counterfactual_history_shuffled_utility": 0.0,
        "counterfactual_history_true_minus_shuffled_utility": 0.0,
        "counterfactual_history_mismatch_utility": 0.0,
        "counterfactual_history_true_minus_mismatch_utility": 0.0,
    }
    m_obs, replay_prefix_used_count = _apply_replay_prefix_beliefs(
        model,
        batch,
        m_obs,
        skill_id_to_idx,
        skill_count,
        device,
        trainable=bool(trainable_replay_prefix),
    )
    metrics["offline_full_replay_prefix_used_count"] = float(replay_prefix_used_count)
    metrics["offline_replay_prefix_trainable_used_count"] = float(
        replay_prefix_used_count if trainable_replay_prefix else 0
    )
    metrics["offline_replay_prefix_trainable_enabled"] = float(bool(trainable_replay_prefix))

    policy_indices = [idx for idx, row in enumerate(batch) if (row.get("loss_mask") or {}).get("L_policy")]
    policy_items = [batch[idx] for idx in policy_indices]
    if policy_items:
        cached_policy = _cached_policy_batch(policy_items, device)
        if cached_policy is not None:
            policy_h, action_embs, candidate_mask, labels = cached_policy
        else:
            policy_h = _batch_cached_or_encode(
                model,
                policy_items,
                "_state_embedding",
                "state_text",
                device,
                text_role=STATE_QUERY_ROLE,
            )
            candidate_rows, candidate_mask, labels = _candidate_batch(policy_items)
            flat_candidates = [text for row in candidate_rows for text in row]
            action_embs = _encode(model, flat_candidates).to(device).view(len(candidate_rows), len(candidate_rows[0]), -1)
            candidate_mask = candidate_mask.to(device)
            labels = labels.to(device)
        policy_m = m_obs.index_select(0, torch.tensor(policy_indices, dtype=torch.long, device=device))
        scores = _native_policy_scores(model, action_embs, policy_m, candidate_mask)
        if scores is not None:
            metrics["policy_head_type"] = "native_skill_head"
        else:
            scores = action_adapter(policy_h, action_embs, candidate_mask.to(device))
            metrics["policy_head_type"] = "universal_action_adapter"
        policy_loss = F.cross_entropy(scores, labels)
        losses.append(_weighted_term(metrics, "L_policy", policy_loss, weights["L_policy"]))
        metrics["policy_ce_loss"] = float(policy_loss.detach().cpu().item())
        with torch.no_grad():
            pred_labels = torch.argmax(scores.detach(), dim=-1)
            metrics["policy_expert_recall@1"] = float((pred_labels == labels).float().mean().detach().cpu().item())

        candidate_text_rows = _policy_candidate_text_rows(policy_items)
        hard_negative_items: list[tuple[int, int]] = []
        if weights["hard_negative_margin"] > 0.0:
            for local_idx, row in enumerate(policy_items):
                if not (row.get("loss_mask") or {}).get("hard_negative_margin"):
                    continue
                hard_negative = str(row.get("hard_negative_action") or "")
                candidates = candidate_text_rows[local_idx]
                if hard_negative and hard_negative in candidates and int(labels[local_idx].detach().cpu().item()) < scores.size(1):
                    hard_negative_items.append((local_idx, candidates.index(hard_negative)))
        if hard_negative_items:
            row_indices = torch.tensor([item[0] for item in hard_negative_items], dtype=torch.long, device=device)
            neg_indices = torch.tensor([item[1] for item in hard_negative_items], dtype=torch.long, device=device)
            pos_indices = labels.index_select(0, row_indices)
            pos_scores = scores.index_select(0, row_indices).gather(1, pos_indices.unsqueeze(1)).squeeze(1)
            neg_scores = scores.index_select(0, row_indices).gather(1, neg_indices.unsqueeze(1)).squeeze(1)
            margin = torch.tensor(1.0, dtype=scores.dtype, device=device)
            hard_negative_loss = F.relu(margin - pos_scores + neg_scores).mean()
            losses.append(_weighted_term(metrics, "hard_negative_margin", hard_negative_loss, weights["hard_negative_margin"]))
            metrics["hard_negative_margin_loss"] = float(hard_negative_loss.detach().cpu().item())
            metrics["hard_negative_pair_count"] = float(len(hard_negative_items))
        else:
            metrics["hard_negative_margin_loss"] = 0.0
            metrics["hard_negative_pair_count"] = 0.0

        q_success_items = [
            idx
            for idx, row in enumerate(policy_items)
            if weights["Q_success"] > 0.0 and (row.get("loss_mask") or {}).get("Q_success")
        ]
        q_success_head = getattr(model, "q_success_head", None)
        if q_success_items and q_success_head is not None:
            local_indices = torch.tensor(q_success_items, dtype=torch.long, device=device)
            q_h = policy_h.index_select(0, local_indices)
            q_m = policy_m.index_select(0, local_indices)
            q_action_embs = action_embs.index_select(0, local_indices)
            q_mask = candidate_mask.index_select(0, local_indices).to(device)
            q_rows = [policy_items[idx] for idx in q_success_items]
            q_labels, q_weights = _q_success_targets(q_rows, q_action_embs.size(1), device)
            q_logits = q_success_scores(q_success_head, q_h, q_m, q_action_embs, q_mask)
            q_loss, q_metrics = compute_q_success_loss(q_logits, q_labels, q_weights, q_mask)
            losses.append(_weighted_term(metrics, "Q_success", q_loss, weights["Q_success"]))
            metrics.update(q_metrics)
        else:
            metrics["q_success_bce_loss"] = 0.0
            metrics["q_success_sample_count"] = 0.0
            metrics["q_success_accuracy@0.5"] = 0.0
    else:
        metrics["policy_ce_loss"] = 0.0
        metrics["policy_expert_recall@1"] = 0.0
        metrics["policy_head_type"] = "none"
        metrics["hard_negative_margin_loss"] = 0.0
        metrics["hard_negative_pair_count"] = 0.0
        metrics["q_success_bce_loss"] = 0.0
        metrics["q_success_sample_count"] = 0.0
        metrics["q_success_accuracy@0.5"] = 0.0

    routing_items = [
        idx
        for idx, row in enumerate(batch)
        if weights["routing"] > 0.0
        and (row.get("loss_mask") or {}).get("routing")
        and row.get("skill_id") in skill_id_to_idx
    ]
    if routing_items:
        indices = torch.tensor(routing_items, dtype=torch.long, device=device)
        labels = torch.tensor([skill_id_to_idx[str(batch[idx]["skill_id"])] for idx in routing_items], dtype=torch.long, device=device)
        routing_logits = logits.index_select(0, indices)
        if routing_logits.size(-1) >= len(skill_id_to_idx):
            routing_logits = routing_logits[:, : len(skill_id_to_idx)]
        routing_candidate_logits, routing_candidate_labels, _routing_candidate_indices = _gather_candidate_logits_and_labels(
            routing_logits,
            [batch[idx] for idx in routing_items],
            labels,
            "stage0_candidate_skill_indices",
            len(skill_id_to_idx),
        )
        if routing_candidate_logits is not None and routing_candidate_labels is not None:
            routing_logits = routing_candidate_logits
            labels = routing_candidate_labels
            metrics["routing_candidate_source"] = "stage0_topm"
        else:
            metrics["routing_candidate_source"] = "full_pool"
        routing_loss = _retrieval_contrastive_loss_from_logits(
            routing_logits,
            labels,
            num_negatives=retrieval_num_negatives,
            hard_ratio=retrieval_hard_ratio,
        )
        losses.append(_weighted_term(metrics, "routing", routing_loss, weights["routing"]))
        metrics["retrieval_contrastive_loss"] = float(routing_loss.detach().cpu().item())
        metrics["routing_ce_loss"] = 0.0
    else:
        metrics["retrieval_contrastive_loss"] = 0.0
        metrics["routing_ce_loss"] = 0.0

    stop_items = [
        idx
        for idx, row in enumerate(batch)
        if weights["STOP"] > 0.0 and (row.get("loss_mask") or {}).get("STOP")
    ]
    if stop_items:
        indices = torch.tensor(stop_items, dtype=torch.long, device=device)
        stop = _stop_logits(model, h.index_select(0, indices), m_obs.index_select(0, indices))
        labels = torch.tensor([1.0 if batch[idx].get("done") else 0.0 for idx in stop_items], dtype=torch.float32, device=device)
        stop_loss = F.binary_cross_entropy_with_logits(stop, labels)
        losses.append(_weighted_term(metrics, "STOP", stop_loss, weights["STOP"]))
        metrics["stop_bce_loss"] = float(stop_loss.detach().cpu().item())
    else:
        metrics["stop_bce_loss"] = 0.0

    trans_items = [
        idx
        for idx, row in enumerate(batch)
        if weights["L_trans"] > 0.0 and (row.get("loss_mask") or {}).get("L_trans")
    ]
    if trans_items:
        indices = torch.tensor(trans_items, dtype=torch.long, device=device)
        trans_rows = [batch[idx] for idx in trans_items]
        target_h = _batch_cached_or_encode(
            model,
            trans_rows,
            "_next_observation_embedding",
            "next_observation_text",
            device,
            text_role=TRANSITION_TEXT_ROLE,
        )
        action_h = _batch_action_text_embedding_or_none(model, trans_rows, device)
        target = _next_belief_target_from_embeddings(model, target_h, skill_count)
        labels = torch.tensor(
            [skill_id_to_idx.get(str(batch[idx].get("skill_id")), 0) for idx in trans_items],
            dtype=torch.long,
            device=device,
        )
        pred = _transition_prediction(
            model,
            h.index_select(0, indices),
            m_obs.index_select(0, indices),
            labels,
            target_h,
            action_emb=action_h,
        )
        trans_loss = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
        losses.append(_weighted_term(metrics, "L_trans", trans_loss, weights["L_trans"]))
        metrics["transition_cosine_loss"] = float(trans_loss.detach().cpu().item())
    else:
        metrics["transition_cosine_loss"] = 0.0

    trans_skill_items = [
        idx
        for idx, row in enumerate(batch)
        if (row.get("loss_mask") or {}).get("L_trans_skill_ce")
        and row.get("skill_id") in skill_id_to_idx
        and (next_skill_pool_mode == "full_pool" or row.get("next_skill_id") in skill_id_to_idx)
    ]
    if trans_skill_items:
        indices = torch.tensor(trans_skill_items, dtype=torch.long, device=device)
        trans_skill_rows = [batch[idx] for idx in trans_skill_items]
        obs_emb = _batch_cached_or_encode(
            model,
            trans_skill_rows,
            "_next_observation_embedding",
            "next_observation_text",
            device,
            text_role=TRANSITION_TEXT_ROLE,
        )
        action_h = _batch_action_text_embedding_or_none(model, trans_skill_rows, device)
        current_labels = torch.tensor(
            [skill_id_to_idx.get(str(batch[idx].get("skill_id")), 0) for idx in trans_skill_items],
            dtype=torch.long,
            device=device,
        )
        next_labels_global = torch.tensor(
            [skill_id_to_idx.get(str(batch[idx].get("next_skill_id")), 0) for idx in trans_skill_items],
            dtype=torch.long,
            device=device,
        )
        h_selected = h.index_select(0, indices)
        m_selected = m_obs.index_select(0, indices)
        m_static_selected = m_static.index_select(0, indices)
        next_rows = [batch[idx] for idx in trans_skill_items]
        route_h_selected = h_selected
        route_m_selected = m_selected
        route_static_memory = m_static_selected
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
            missing_next_state_rows = [
                row
                for row in next_rows
                if "_next_state_embedding" not in row and not str(row.get("next_state_text") or "").strip()
            ]
            if missing_next_state_rows:
                raise ValueError("causal unified memory next-skill CE requires next_state_text")
            h_next = _batch_cached_or_encode(
                model,
                next_rows,
                "_next_state_embedding",
                "next_state_text",
                device,
                text_role=STATE_QUERY_ROLE,
            )
            _predicted_memory, _observation_memory, next_memory = _post_action_memory(
                model,
                m_t=m_selected,
                current_skill_labels=current_labels,
                action_embeddings=action_h,
                observation_embeddings=obs_emb,
                h_next=h_next,
            )
            route_h_selected = h_next
            route_m_selected = next_memory
            route_static_memory = model.initial_belief(h_next)
            metrics["transition_post_action_update_rows"] = float(len(next_rows))
            metrics["transition_next_state_rows"] = float(len(next_rows))
        else:
            metrics["transition_post_action_update_rows"] = 0.0
            metrics["transition_next_state_rows"] = 0.0
        if next_skill_pool_mode == "full_pool":
            full_logits_fn = getattr(model, "unified_route_full_logits", None)
            if not callable(full_logits_fn):
                raise ValueError("full-pool next-skill training requires model.unified_route_full_logits")
            dynamic_full = full_logits_fn(route_h_selected, route_m_selected)
            if static_route_anchor_weight > 0.0 and static_route_teacher is not None:
                static_full = full_logits_fn(route_h_selected, route_static_memory)
            else:
                with torch.no_grad():
                    static_full = full_logits_fn(route_h_selected, route_static_memory)
            expected_shape = (len(next_rows), skill_count)
            if tuple(dynamic_full.shape) != expected_shape or tuple(static_full.shape) != expected_shape:
                raise ValueError("full-pool route logits must match causal rows and declared skill table")

            known_positive_mask = full_pool_positive_mask(
                next_rows,
                skill_id_to_idx,
                equivalent_skill_ids_by_skill_id,
                skill_count=skill_count,
                device=device,
            )
            legal_pool = legal_skill_pool_mask(
                next_rows,
                skill_id_to_idx,
                skill_count=skill_count,
                device=device,
            )
            row_weights = _transition_ce_row_weights(
                next_rows,
                real_multiplier=transition_real_candidate_ce_multiplier,
                injected_multiplier=transition_injected_candidate_ce_multiplier,
                device=device,
                dtype=dynamic_full.dtype,
            )
            objective = full_pool_causal_route_objective(
                dynamic_logits=dynamic_full,
                static_logits=static_full,
                positive_mask=known_positive_mask,
                valid_mask=legal_pool.mask,
                row_weights=row_weights,
                gain_margin=counterfactual_gain_margin,
                safety_tolerance=counterfactual_safety_tolerance,
                gain_weight=counterfactual_gain_weight,
                safety_weight=counterfactual_safety_weight,
                compute_counterfactual=(
                    weights["counterfactual_utility"] > 0.0
                    or weights["counterfactual_history"] > 0.0
                ),
            )
            losses.append(
                _weighted_term(
                    metrics,
                    "L_trans_skill_ce",
                    objective.main_loss,
                    weights["L_trans_skill_ce"],
                )
            )
            history_rank_loss = dynamic_full.new_zeros(())
            history_static_no_regret_loss = dynamic_full.new_zeros(())
            history_loss = dynamic_full.new_zeros(())
            history_eligible_count = 0
            history_violation_count = 0
            history_true_utility = 0.0
            history_shuffled_utility = 0.0
            history_donor_count = 0
            if weights["counterfactual_history"] > 0.0:
                history_static_no_regret_loss = objective.counterfactual.loss
                donor_permutation = _counterfactual_memory_permutation(
                    next_rows,
                    device,
                    state_embeddings=h_selected,
                )
                history_row_indices = (donor_permutation >= 0).nonzero(as_tuple=False).view(-1)
                history_donor_count = int(history_row_indices.numel())
                if history_donor_count:
                    donor_indices = donor_permutation.index_select(0, history_row_indices)
                    shuffled_current_memory = m_selected.detach().index_select(0, donor_indices)
                    history_current_labels = current_labels.index_select(0, history_row_indices)
                    history_action_embeddings = (
                        action_h.index_select(0, history_row_indices)
                        if isinstance(action_h, torch.Tensor)
                        else None
                    )
                    history_observation_embeddings = obs_emb.index_select(0, history_row_indices)
                    history_h_next = route_h_selected.index_select(0, history_row_indices)
                    _shuffled_predicted, _shuffled_observation, shuffled_next_memory = _post_action_memory(
                        model,
                        m_t=shuffled_current_memory,
                        current_skill_labels=history_current_labels,
                        action_embeddings=history_action_embeddings,
                        observation_embeddings=history_observation_embeddings,
                        h_next=history_h_next,
                    )
                    shuffled_full = full_logits_fn(history_h_next, shuffled_next_memory)
                    true_history_full = dynamic_full.index_select(0, history_row_indices)
                    history_positive = known_positive_mask.index_select(0, history_row_indices)
                    history_valid = legal_pool.mask.index_select(0, history_row_indices)
                    history_rank_loss = shuffled_history_utility_loss(
                        true_logits=true_history_full,
                        shuffled_logits=shuffled_full,
                        positive_mask=history_positive,
                        valid_mask=history_valid,
                        margin=counterfactual_history_margin,
                    )
                    history_legal_positive = history_positive & history_valid
                    history_finite = torch.where(
                        history_valid,
                        torch.isfinite(true_history_full) & torch.isfinite(shuffled_full),
                        True,
                    ).all(dim=-1)
                    history_eligible = (
                        history_legal_positive.any(dim=-1)
                        & (history_valid & ~history_legal_positive).any(dim=-1)
                        & history_finite
                    )
                    history_eligible_count = int(history_eligible.sum().detach().cpu().item())
                    if history_eligible_count:
                        true_utility = multi_positive_log_utility(
                            true_history_full[history_eligible],
                            history_legal_positive[history_eligible],
                            history_valid[history_eligible],
                        )
                        shuffled_utility = multi_positive_log_utility(
                            shuffled_full[history_eligible],
                            history_legal_positive[history_eligible],
                            history_valid[history_eligible],
                        )
                        violations = (
                            float(counterfactual_history_margin)
                            + shuffled_utility
                            - true_utility
                        ) > 0.0
                        history_violation_count = int(violations.sum().detach().cpu().item())
                        history_true_utility = float(true_utility.mean().detach().cpu().item())
                        history_shuffled_utility = float(
                            shuffled_utility.mean().detach().cpu().item()
                        )
                history_loss = history_rank_loss + history_static_no_regret_loss
                losses.append(
                    _weighted_term(
                        metrics,
                        "counterfactual_history",
                        history_loss,
                        weights["counterfactual_history"],
                    )
                )
            metrics["counterfactual_history_loss"] = float(history_loss.detach().cpu().item())
            metrics["counterfactual_history_rank_loss"] = float(
                history_rank_loss.detach().cpu().item()
            )
            metrics["counterfactual_history_static_no_regret_loss"] = float(
                history_static_no_regret_loss.detach().cpu().item()
            )
            metrics["counterfactual_history_donor_rows"] = float(history_donor_count)
            metrics["counterfactual_history_eligible_rows"] = float(history_eligible_count)
            metrics["counterfactual_history_violation_rows"] = float(history_violation_count)
            metrics["counterfactual_history_true_utility"] = history_true_utility
            metrics["counterfactual_history_shuffled_utility"] = history_shuffled_utility
            metrics["counterfactual_history_true_minus_shuffled_utility"] = (
                history_true_utility - history_shuffled_utility
            )
            metrics["counterfactual_history_mismatch_utility"] = history_shuffled_utility
            metrics["counterfactual_history_true_minus_mismatch_utility"] = (
                history_true_utility - history_shuffled_utility
            )
            anchor_local_indices = [
                row_idx
                for row_idx, row in enumerate(next_rows)
                if static_route_anchor_weight > 0.0
                and static_route_teacher is not None
                and not bool(row.get("replay_prefix"))
                and bool(objective.eligible_mask[row_idx].detach().cpu().item())
            ]
            if anchor_local_indices:
                anchor_indices = torch.tensor(
                    anchor_local_indices,
                    dtype=torch.long,
                    device=device,
                )
                anchor_h_current = h_selected.index_select(0, anchor_indices)
                anchor_h_next = route_h_selected.index_select(0, anchor_indices)
                anchor_current_labels = current_labels.index_select(0, anchor_indices)
                anchor_action_embeddings = (
                    action_h.index_select(0, anchor_indices)
                    if isinstance(action_h, torch.Tensor)
                    else None
                )
                anchor_observation_embeddings = obs_emb.index_select(0, anchor_indices)
                anchor_valid = legal_pool.mask.index_select(0, anchor_indices)
                anchor_positive = known_positive_mask.index_select(0, anchor_indices)
                anchor_rows = [next_rows[idx] for idx in anchor_local_indices]
                with torch.no_grad():
                    teacher_static_logits = static_route_teacher.full_logits(
                        model,
                        anchor_h_next,
                    )
                student_static_logits = static_full.index_select(0, anchor_indices)
                anchor_loss = static_route_anchor_kl(
                    model,
                    static_route_teacher,
                    anchor_h_next,
                    anchor_valid,
                    student_logits=student_static_logits,
                    teacher_logits=teacher_static_logits,
                )
                weighted_anchor_loss = anchor_loss * static_route_anchor_weight
                losses.append(weighted_anchor_loss)
                with torch.no_grad():
                    teacher_dynamic_logits = static_route_teacher.post_action_full_logits(
                        model,
                        h_current=anchor_h_current,
                        current_skill_labels=anchor_current_labels,
                        action_embeddings=anchor_action_embeddings,
                        observation_embeddings=anchor_observation_embeddings,
                        h_next=anchor_h_next,
                    )
                    student_dynamic_logits = dynamic_full.index_select(0, anchor_indices)
                    floor = torch.finfo(student_dynamic_logits.dtype).min
                    teacher_static_logits = teacher_static_logits.to(student_dynamic_logits).masked_fill(
                        ~anchor_valid,
                        floor,
                    )
                    teacher_dynamic_logits = teacher_dynamic_logits.to(student_dynamic_logits).masked_fill(
                        ~anchor_valid,
                        floor,
                    )
                    student_static_logits = student_static_logits.masked_fill(~anchor_valid, floor)
                    student_dynamic_logits = student_dynamic_logits.masked_fill(~anchor_valid, floor)
                    anchor_labels = anchor_positive.to(torch.float32).argmax(dim=-1).to(torch.long)
                    teacher_static_mrr = _ranking_metrics_from_logits(
                        teacher_static_logits,
                        anchor_labels,
                        rows=anchor_rows,
                        positive_mask=anchor_positive,
                    )["transition_skill_mrr"]
                    student_static_mrr = _ranking_metrics_from_logits(
                        student_static_logits,
                        anchor_labels,
                        rows=anchor_rows,
                        positive_mask=anchor_positive,
                    )["transition_skill_mrr"]
                    teacher_dynamic_mrr = _ranking_metrics_from_logits(
                        teacher_dynamic_logits,
                        anchor_labels,
                        rows=anchor_rows,
                        positive_mask=anchor_positive,
                    )["transition_skill_mrr"]
                    student_dynamic_mrr = _ranking_metrics_from_logits(
                        student_dynamic_logits,
                        anchor_labels,
                        rows=anchor_rows,
                        positive_mask=anchor_positive,
                    )["transition_skill_mrr"]
                metrics["stage2_static_route_anchor_rows"] = float(len(anchor_rows))
                metrics["stage2_static_route_anchor_loss"] = float(anchor_loss.detach().cpu().item())
                metrics["stage2_static_route_anchor_weighted_loss"] = float(
                    weighted_anchor_loss.detach().cpu().item()
                )
                metrics["stage2_static_route_teacher_mrr"] = float(teacher_static_mrr)
                metrics["stage2_static_route_student_mrr"] = float(student_static_mrr)
                metrics["stage2_step_zero_dynamic_mrr"] = float(teacher_dynamic_mrr)
                metrics["stage2_dynamic_mrr"] = float(student_dynamic_mrr)
                metrics["stage2_static_mrr_delta_vs_teacher"] = float(
                    student_static_mrr - teacher_static_mrr
                )
                metrics["stage2_dynamic_mrr_delta_vs_step_zero"] = float(
                    student_dynamic_mrr - teacher_dynamic_mrr
                )
            legacy_counterfactual_enabled = weights["counterfactual_utility"] > 0.0
            if legacy_counterfactual_enabled:
                causal_update_count = torch.tensor(
                    [1 + len(row.get("replay_prefix") or []) for row in next_rows],
                    dtype=dynamic_full.dtype,
                    device=device,
                )
                route_alpha_fn = getattr(model, "route_memory_alpha", None)
                if callable(route_alpha_fn):
                    safe_alpha = route_alpha_fn(
                        route_h_selected,
                        route_static_memory.detach(),
                        route_m_selected,
                        causal_update_count,
                    )
                    safe_gate_available = True
                else:
                    safe_alpha = dynamic_full.new_ones(len(next_rows))
                    safe_gate_available = False
                safe_fused_full = bounded_memory_fusion(
                    static_full,
                    dynamic_full,
                    safe_alpha,
                    legal_pool.mask,
                    residual_bound=safe_memory_residual_bound,
                )
                explicit_row_mask = torch.tensor(
                    [bool(_explicit_inventory_skill_ids_ordered(row)) for row in next_rows],
                    dtype=torch.bool,
                    device=device,
                )
                local_masks = build_local_candidate_masks(
                    static_full,
                    dynamic_full,
                    known_positive_mask,
                    legal_pool.mask,
                    explicit_row_mask,
                    candidate_sizes=safe_local_candidate_sizes,
                )
                local_objective = safe_local_route_objective(
                    fused_logits=safe_fused_full,
                    static_logits=static_full,
                    positive_mask=known_positive_mask,
                    candidate_masks=local_masks.masks,
                    source_row_indices=local_masks.source_row_indices,
                    row_weights=row_weights,
                    gain_margin=counterfactual_gain_margin,
                    safety_tolerance=counterfactual_safety_tolerance,
                )
                scaled_counterfactual = local_objective.loss * counterfactual_scale
                losses.append(
                    _weighted_term(
                        metrics,
                        "counterfactual_utility",
                        scaled_counterfactual,
                        weights["counterfactual_utility"],
                    )
                )
            else:
                safe_gate_available = False
                scaled_counterfactual = dynamic_full.new_zeros(())

            valid = legal_pool.mask
            legal_positive = known_positive_mask & valid
            eligible = objective.eligible_mask
            floor = torch.finfo(dynamic_full.dtype).min
            masked_dynamic = dynamic_full.masked_fill(~valid, floor)
            masked_static = static_full.masked_fill(~valid, floor)
            masked_residual = (dynamic_full - static_full).masked_fill(~valid, floor)
            eligible_dynamic = masked_dynamic[eligible]
            eligible_static = masked_static[eligible]
            eligible_residual = masked_residual[eligible]
            eligible_positive = legal_positive[eligible]
            eligible_rows = [row for row, keep in zip(next_rows, eligible.detach().cpu().tolist()) if keep]
            eligible_labels = (
                eligible_positive.to(torch.float32).argmax(dim=-1).to(torch.long)
                if bool(eligible.any())
                else torch.zeros(0, dtype=torch.long, device=device)
            )

            metrics["transition_skill_ce_loss"] = float(objective.main_loss.detach().cpu().item())
            metrics["transition_skill_head_type"] = "unified_memory_retriever_full_pool"
            metrics["transition_scoring_mode"] = UNIFIED_MEMORY_ROUTE_SCORER
            metrics["transition_residual_lambda"] = 0.0
            metrics["transition_inventory_mask_mode"] = "explicit_only"
            metrics["transition_loss_type"] = "multi_positive_full_pool_nll"
            metrics["transition_positive_mode"] = transition_positive_mode
            metrics["transition_full_pool_rows"] = float(len(next_rows))
            metrics["transition_skill_ce_count"] = float(objective.exclusion_counts["eligible_rows"])
            metrics["transition_skill_ce_candidate_count"] = float(
                valid.sum(dim=-1).float().mean().detach().cpu().item() if len(next_rows) else 0.0
            )
            static_hits = 0
            for row, row_positive in zip(next_rows, known_positive_mask.detach().cpu()):
                natural = _row_candidate_indices(row, "stage0_next_candidate_skill_indices", skill_count)
                positive_ids = set(row_positive.nonzero(as_tuple=False).view(-1).tolist())
                static_hits += int(bool(positive_ids.intersection(natural)))
            metrics["transition_static_hit_train_rows"] = float(static_hits)
            metrics["transition_static_miss_train_rows"] = float(len(next_rows) - static_hits)
            metrics["counterfactual_utility_loss"] = float(objective.counterfactual.loss.detach().cpu().item())
            metrics["counterfactual_scaled_loss"] = float(scaled_counterfactual.detach().cpu().item())
            metrics["counterfactual_scale"] = float(counterfactual_scale if legacy_counterfactual_enabled else 0.0)
            metrics["counterfactual_training_objective"] = (
                "safe_local_candidate_rank_v1"
                if legacy_counterfactual_enabled
                else "stage2_cf_static_no_regret_v1"
                if weights["counterfactual_history"] > 0.0
                else "disabled_stage4_cmc_owned"
            )
            metrics["counterfactual_gain_loss"] = float(objective.counterfactual.gain_loss.detach().cpu().item())
            metrics["counterfactual_safety_loss"] = float(objective.counterfactual.safety_loss.detach().cpu().item())
            metrics["counterfactual_weak_rows"] = float(objective.counterfactual.weak_count)
            metrics["counterfactual_strong_rows"] = float(objective.counterfactual.strong_count)
            metrics["counterfactual_eligible_rows"] = float(
                objective.counterfactual.weak_count + objective.counterfactual.strong_count
            )
            metrics["counterfactual_gain_violation_rows"] = float(objective.counterfactual.gain_violation_count)
            metrics["counterfactual_safety_violation_rows"] = float(objective.counterfactual.safety_violation_count)
            metrics["safe_memory_gate_available"] = bool(safe_gate_available)
            metrics["safe_memory_residual_bound"] = float(
                safe_memory_residual_bound if legacy_counterfactual_enabled else 0.0
            )
            metrics["safe_local_candidate_sizes"] = (
                list(safe_local_candidate_sizes) if legacy_counterfactual_enabled else []
            )
            if legacy_counterfactual_enabled:
                metrics["safe_memory_alpha_mean"] = float(safe_alpha.mean().detach().cpu().item())
                metrics["safe_memory_alpha_min"] = float(safe_alpha.min().detach().cpu().item())
                metrics["safe_memory_alpha_max"] = float(safe_alpha.max().detach().cpu().item())
                metrics["safe_local_mask_count"] = float(local_masks.masks.size(0))
                metrics["safe_local_candidate_count"] = float(
                    local_masks.masks.sum(dim=-1).float().mean().detach().cpu().item()
                    if int(local_masks.masks.size(0))
                    else 0.0
                )
                metrics["safe_local_loss"] = float(local_objective.loss.detach().cpu().item())
                metrics["safe_local_nll_loss"] = float(local_objective.nll_loss.detach().cpu().item())
                metrics["safe_local_gain_loss"] = float(local_objective.gain_loss.detach().cpu().item())
                metrics["safe_local_safety_loss"] = float(local_objective.safety_loss.detach().cpu().item())
                metrics["safe_local_eligible_masks"] = float(local_objective.eligible_count)
                metrics["safe_local_weak_masks"] = float(local_objective.weak_count)
                metrics["safe_local_strong_masks"] = float(local_objective.strong_count)
                metrics["safe_local_gain_violation_masks"] = float(local_objective.gain_violation_count)
                metrics["safe_local_safety_violation_masks"] = float(local_objective.safety_violation_count)
            else:
                for key in (
                    "safe_memory_alpha_mean",
                    "safe_memory_alpha_min",
                    "safe_memory_alpha_max",
                    "safe_local_mask_count",
                    "safe_local_candidate_count",
                    "safe_local_loss",
                    "safe_local_nll_loss",
                    "safe_local_gain_loss",
                    "safe_local_safety_loss",
                    "safe_local_eligible_masks",
                    "safe_local_weak_masks",
                    "safe_local_strong_masks",
                    "safe_local_gain_violation_masks",
                    "safe_local_safety_violation_masks",
                ):
                    metrics[key] = 0.0
            metrics["positive_not_in_skill_table_rows"] = float(
                (~known_positive_mask.any(dim=-1)).sum().detach().cpu().item()
            )
            metrics["positive_outside_legal_pool_rows"] = float(
                (
                    known_positive_mask.any(dim=-1)
                    & ~(known_positive_mask & valid).any(dim=-1)
                ).sum().detach().cpu().item()
            )
            metrics["explicit_inventory_no_known_skill_rows"] = float(
                legal_pool.explicit_inventory_no_known_skill_mask.sum().detach().cpu().item()
            )
            metrics["no_legal_negative_rows"] = float(
                objective.exclusion_counts["no_valid_negative_rows"]
            )
            metrics["nonfinite_logit_rows"] = float(
                objective.exclusion_counts["nonfinite_logit_rows"]
            )
            explicit_rows = sum(bool(_explicit_inventory_skill_ids_ordered(row)) for row in next_rows)
            metrics["transition_inventory_mask_applied_rows"] = float(explicit_rows)
            metrics["transition_inventory_mask_missing_rows"] = float(len(next_rows) - explicit_rows)
            metrics["transition_inventory_mask_removed_candidates"] = float(
                (skill_count - valid.sum(dim=-1)).sum().detach().cpu().item()
            )
            metrics["transition_inventory_mask_positive_missing_rows"] = metrics[
                "positive_outside_legal_pool_rows"
            ]
            metrics["transition_inventory_mask_backfilled_rows"] = 0.0
            metrics["transition_inventory_mask_backfilled_candidates"] = 0.0
            positive_counts = legal_positive.sum(dim=-1).float()
            metrics["transition_positive_mean_count"] = float(
                positive_counts.mean().detach().cpu().item() if len(next_rows) else 0.0
            )
            metrics["transition_positive_max_count"] = float(
                positive_counts.max().detach().cpu().item() if len(next_rows) else 0.0
            )
            metrics["transition_multi_positive_rows"] = float(
                (positive_counts > 1).sum().detach().cpu().item()
            )

            dynamic_metrics = _ranking_metrics_from_logits(
                eligible_dynamic,
                eligible_labels,
                rows=eligible_rows,
                positive_mask=eligible_positive,
            )
            static_metrics = _ranking_metrics_from_logits(
                eligible_static,
                eligible_labels,
                rows=eligible_rows,
                positive_mask=eligible_positive,
            )
            residual_metrics = _ranking_metrics_from_logits(
                eligible_residual,
                eligible_labels,
                rows=eligible_rows,
                positive_mask=eligible_positive,
            )
            metrics.update(dynamic_metrics)
            metrics.update(
                {f"transition_unified_dynamic_{key.removeprefix('transition_')}": value for key, value in dynamic_metrics.items()}
            )
            metrics.update(
                {f"transition_unified_static_{key.removeprefix('transition_')}": value for key, value in static_metrics.items()}
            )
            metrics.update(
                {f"transition_prior_{key.removeprefix('transition_')}": value for key, value in static_metrics.items()}
            )
            metrics.update({f"stage0_prior_{key}": value for key, value in static_metrics.items()})
            metrics.update(
                {f"transition_residual_{key.removeprefix('transition_')}": value for key, value in residual_metrics.items()}
            )
            metrics.update(
                _dynamic_static_rank_metrics(
                    eligible_dynamic,
                    eligible_static,
                    eligible_labels,
                    positive_mask=eligible_positive,
                    prefix="transition_unified",
                )
            )
            metrics.update(
                _rank_delta_metrics_vs_prior(
                    eligible_dynamic,
                    eligible_static,
                    eligible_labels,
                    positive_mask=eligible_positive,
                )
            )
            full_candidate_row = list(range(skill_count))
            metrics.update(
                _current_skill_candidate_metrics(
                    rows=eligible_rows,
                    candidate_rows=[full_candidate_row] * len(eligible_rows),
                    logits=eligible_dynamic,
                    skill_id_to_idx=skill_id_to_idx,
                )
            )
            real_rows = sum(1 for row in next_rows if not bool(row.get("stage0_positive_injected")))
            metrics["transition_real_candidate_rows"] = float(real_rows)
            metrics["transition_injected_candidate_rows"] = float(len(next_rows) - real_rows)
            if weights["transition_hard_negative_margin"] > 0.0:
                hard_negative_loss, hard_negative_count = _transition_hard_negative_margin_loss(
                    eligible_dynamic,
                    eligible_labels,
                    eligible_rows,
                    margin=transition_hard_negative_margin,
                )
            else:
                hard_negative_loss = eligible_dynamic.new_zeros(())
                hard_negative_count = 0
            losses.append(
                _weighted_term(
                    metrics,
                    "transition_hard_negative_margin",
                    hard_negative_loss,
                    weights["transition_hard_negative_margin"],
                )
            )
            metrics["transition_hard_negative_margin_loss"] = float(
                hard_negative_loss.detach().cpu().item()
            )
            metrics["transition_hard_negative_pair_count"] = float(hard_negative_count)
            metrics["gated_temporal_lambda_mean"] = 0.0
            metrics["gated_temporal_lambda_min"] = 0.0
            metrics["gated_temporal_lambda_max"] = 0.0
            metrics["gated_temporal_prior_kl_loss"] = 0.0
            metrics["gated_temporal_rank_drop_loss"] = 0.0
            metrics["gated_temporal_kl_alpha"] = 0.0
            metrics["gated_temporal_rank_drop_beta"] = 0.0
            metrics["gated_temporal_context_top_k"] = 0.0

            belief_items = [
                idx for idx, row in enumerate(batch) if (row.get("loss_mask") or {}).get("belief")
            ]
            if belief_items:
                belief_indices = torch.tensor(belief_items, dtype=torch.long, device=device)
                belief_rows = [batch[idx] for idx in belief_items]
                target_h = _batch_cached_or_encode(
                    model,
                    belief_rows,
                    "_next_observation_embedding",
                    "next_observation_text",
                    device,
                    text_role=TRANSITION_TEXT_ROLE,
                )
                belief_action_h = _batch_action_text_embedding_or_none(model, belief_rows, device)
                target = _next_belief_target_from_embeddings(model, target_h, skill_count)
                belief_labels = torch.tensor(
                    [skill_id_to_idx.get(str(batch[idx].get("skill_id")), 0) for idx in belief_items],
                    dtype=torch.long,
                    device=device,
                )
                pred = _transition_prediction(
                    model,
                    h.index_select(0, belief_indices),
                    m_obs.index_select(0, belief_indices),
                    belief_labels,
                    target_h,
                    action_emb=belief_action_h,
                )
                belief = _belief_prediction(model, pred, target, target_h)
                belief_loss = 1.0 - F.cosine_similarity(belief.float(), target.float(), dim=-1).mean()
                losses.append(_weighted_term(metrics, "belief", belief_loss, weights["belief"]))
                metrics["belief_cosine_loss"] = float(belief_loss.detach().cpu().item())
            else:
                metrics["belief_cosine_loss"] = 0.0
            loss = sum(losses) if losses else h[:, :0].sum()
            metrics["loss"] = float(loss.detach().cpu().item())
            return loss, metrics
        candidate_rows = [
            _row_candidate_indices(row, "stage0_next_candidate_skill_indices", len(skill_id_to_idx))
            or _row_candidate_indices(row, "stage0_candidate_skill_indices", len(skill_id_to_idx))
            for row in next_rows
        ]
        inventory_audit = {
            "inventory_mask_applied_rows": 0,
            "inventory_mask_missing_rows": 0,
            "inventory_mask_removed_candidates": 0,
            "inventory_mask_positive_missing_rows": 0,
            "inventory_mask_backfilled_rows": 0,
            "inventory_mask_backfilled_candidates": 0,
        }
        next_label_values = [int(label) for label in next_labels_global.detach().cpu().tolist()]
        use_stage0_candidates = (
            bool(candidate_rows)
            and all(candidate_rows)
            and all(label in row for label, row in zip(next_label_values, candidate_rows))
        )
        candidate_rows_for_positive_mask: list[list[int]]
        if use_stage0_candidates:
            candidate_rows, inventory_audit = _filter_transition_candidates_by_inventory(
                rows=next_rows,
                candidate_rows=candidate_rows,
                labels=next_labels_global,
                skill_ids_by_idx=skill_ids_by_idx,
                mode=transition_inventory_mask_mode,
                min_candidates=transition_inventory_min_candidates,
            )
            use_stage0_candidates = all(
                row and label in row for label, row in zip(next_label_values, candidate_rows)
            )
        if use_stage0_candidates:
            max_width = max(len(row) for row in candidate_rows)
            candidate_ids = torch.zeros(len(candidate_rows), max_width, dtype=torch.long, device=device)
            candidate_valid_mask = torch.zeros(len(candidate_rows), max_width, dtype=torch.bool, device=device)
            local_next_labels: list[int] = []
            for row_idx, (row, label) in enumerate(zip(candidate_rows, next_label_values)):
                width = len(row)
                candidate_ids[row_idx, :width] = torch.tensor(row, dtype=torch.long, device=device)
                candidate_valid_mask[row_idx, :width] = True
                local_next_labels.append(row.index(label))
            next_labels = torch.tensor(local_next_labels, dtype=torch.long, device=device)
            candidate_rows_for_positive_mask = candidate_rows
            candidate_stage0_prior_scores = _stage0_candidate_prior_scores_tensor(
                next_rows,
                candidate_rows,
                width=max_width,
                device=device,
                dtype=h_selected.dtype,
                calibration=stage0_score_prior_calibration,
            )
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
                trans_skill_logits = model.unified_route_logits(
                    route_h_selected,
                    route_m_selected,
                    candidate_rows=candidate_rows,
                )
                prior_logits = model.unified_route_logits(
                    route_h_selected,
                    route_static_memory,
                    candidate_rows=candidate_rows,
                )
                residual_logits = trans_skill_logits - prior_logits
                transition_skill_head_type = "unified_memory_retriever"
                trans_skill_logits = trans_skill_logits.masked_fill(~candidate_valid_mask, torch.finfo(trans_skill_logits.dtype).min)
                prior_logits = prior_logits.masked_fill(~candidate_valid_mask, torch.finfo(prior_logits.dtype).min)
                residual_logits = residual_logits.masked_fill(~candidate_valid_mask, torch.finfo(residual_logits.dtype).min)
            else:
                trans_skill_logits, transition_skill_head_type, prior_logits, residual_logits = _transition_candidate_logits_for_mode(
                    model,
                    h_selected,
                    m_selected,
                    current_labels,
                    obs_emb,
                    action_h,
                    skill_count,
                    candidate_ids=candidate_ids,
                    candidate_valid_mask=candidate_valid_mask,
                    candidate_stage0_prior_scores=candidate_stage0_prior_scores,
                    residual_lambda=transition_residual_lambda,
                    scoring_mode=transition_scoring_mode,
                    gated_temporal_lambda_max=gated_temporal_lambda_max,
                    gated_temporal_context_top_k=gated_temporal_context_top_k,
                )
        else:
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
                trans_skill_logits = model.unified_route_logits(route_h_selected, route_m_selected)
                prior_logits = model.unified_route_logits(route_h_selected, route_static_memory)
                residual_logits = trans_skill_logits - prior_logits
                transition_skill_head_type = "unified_memory_retriever"
            else:
                trans_skill_logits, transition_skill_head_type, prior_logits, residual_logits = _transition_candidate_logits_for_mode(
                    model,
                    h_selected,
                    m_selected,
                    current_labels,
                    obs_emb,
                    action_h,
                    skill_count,
                    residual_lambda=transition_residual_lambda,
                    scoring_mode=transition_scoring_mode,
                    gated_temporal_lambda_max=gated_temporal_lambda_max,
                    gated_temporal_context_top_k=gated_temporal_context_top_k,
                )
            next_labels = next_labels_global
            candidate_rows_for_positive_mask = [list(range(trans_skill_logits.size(1))) for _ in next_rows]
        positive_mask, positive_counts = _transition_positive_mask(
            rows=next_rows,
            candidate_rows=candidate_rows_for_positive_mask,
            labels=next_labels,
            skill_ids_by_idx=skill_ids_by_idx,
            positive_mode=transition_positive_mode,
            device=trans_skill_logits.device,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
        )
        if transition_loss_type == "listwise_nll":
            per_row_trans_skill_loss = _multi_positive_listwise_nll(
                trans_skill_logits,
                positive_mask,
                reduction="none",
            )
        else:
            per_row_trans_skill_loss = F.cross_entropy(trans_skill_logits, next_labels, reduction="none")
        trans_skill_row_weights = _transition_ce_row_weights(
            next_rows,
            real_multiplier=transition_real_candidate_ce_multiplier,
            injected_multiplier=transition_injected_candidate_ce_multiplier,
            device=trans_skill_logits.device,
            dtype=per_row_trans_skill_loss.dtype,
        )
        trans_skill_loss = (per_row_trans_skill_loss * trans_skill_row_weights).sum() / trans_skill_row_weights.sum().clamp_min(1.0)
        losses.append(_weighted_term(metrics, "L_trans_skill_ce", trans_skill_loss, weights["L_trans_skill_ce"]))
        metrics["transition_skill_ce_loss"] = float(trans_skill_loss.detach().cpu().item())
        gated_aux = getattr(model, "_last_gated_temporal_aux", None)
        if effective_transition_scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE:
            prior_losses = prior_preserving_loss(
                final_logits=trans_skill_logits,
                prior_logits=prior_logits,
                positive_mask=positive_mask,
                valid_mask=candidate_valid_mask if use_stage0_candidates else None,
                rank_drop_k=5,
            )
            gated_kl_loss = prior_losses["kl_loss"]
            gated_rank_drop_loss = prior_losses["rank_drop_loss"]
            losses.append(gated_kl_loss * gated_temporal_kl_alpha + gated_rank_drop_loss * gated_temporal_rank_drop_beta)
            if isinstance(gated_aux, dict) and isinstance(gated_aux.get("lambda_t"), torch.Tensor):
                lambda_t = gated_aux["lambda_t"]
                metrics["gated_temporal_lambda_mean"] = float(lambda_t.detach().float().mean().cpu().item())
                metrics["gated_temporal_lambda_min"] = float(lambda_t.detach().float().min().cpu().item())
                metrics["gated_temporal_lambda_max"] = float(lambda_t.detach().float().max().cpu().item())
            else:
                metrics["gated_temporal_lambda_mean"] = 0.0
                metrics["gated_temporal_lambda_min"] = 0.0
                metrics["gated_temporal_lambda_max"] = 0.0
            metrics["gated_temporal_prior_kl_loss"] = float(gated_kl_loss.detach().cpu().item())
            metrics["gated_temporal_rank_drop_loss"] = float(gated_rank_drop_loss.detach().cpu().item())
            metrics["gated_temporal_kl_alpha"] = gated_temporal_kl_alpha
            metrics["gated_temporal_rank_drop_beta"] = gated_temporal_rank_drop_beta
            metrics["gated_temporal_context_top_k"] = float(gated_temporal_context_top_k)
        else:
            metrics["gated_temporal_lambda_mean"] = 0.0
            metrics["gated_temporal_lambda_min"] = 0.0
            metrics["gated_temporal_lambda_max"] = 0.0
            metrics["gated_temporal_prior_kl_loss"] = 0.0
            metrics["gated_temporal_rank_drop_loss"] = 0.0
            metrics["gated_temporal_kl_alpha"] = 0.0
            metrics["gated_temporal_rank_drop_beta"] = 0.0
            metrics["gated_temporal_context_top_k"] = 0.0
        metrics["transition_skill_head_type"] = transition_skill_head_type
        metrics["transition_scoring_mode"] = effective_transition_scoring_mode
        metrics["transition_residual_lambda"] = effective_transition_residual_lambda
        metrics["transition_inventory_mask_mode"] = transition_inventory_mask_mode
        metrics["transition_loss_type"] = transition_loss_type
        metrics["transition_positive_mode"] = transition_positive_mode
        metrics["transition_inventory_mask_applied_rows"] = float(inventory_audit["inventory_mask_applied_rows"])
        metrics["transition_inventory_mask_removed_candidates"] = float(inventory_audit["inventory_mask_removed_candidates"])
        metrics["transition_inventory_mask_missing_rows"] = float(inventory_audit["inventory_mask_missing_rows"])
        metrics["transition_inventory_mask_positive_missing_rows"] = float(inventory_audit["inventory_mask_positive_missing_rows"])
        metrics["transition_inventory_mask_backfilled_rows"] = float(inventory_audit["inventory_mask_backfilled_rows"])
        metrics["transition_inventory_mask_backfilled_candidates"] = float(inventory_audit["inventory_mask_backfilled_candidates"])
        metrics.update(positive_counts)
        metrics.update(_ranking_metrics_from_logits(trans_skill_logits, next_labels, rows=next_rows, positive_mask=positive_mask))
        prior_metrics = _ranking_metrics_from_logits(prior_logits, next_labels, rows=next_rows, positive_mask=positive_mask)
        residual_metrics = _ranking_metrics_from_logits(residual_logits, next_labels, rows=next_rows, positive_mask=positive_mask)
        metrics.update({f"transition_prior_{key.removeprefix('transition_')}": value for key, value in prior_metrics.items()})
        metrics.update({f"stage0_prior_{key}": value for key, value in prior_metrics.items()})
        metrics.update({f"transition_residual_{key.removeprefix('transition_')}": value for key, value in residual_metrics.items()})
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
            dynamic_metrics = _ranking_metrics_from_logits(
                trans_skill_logits,
                next_labels,
                rows=next_rows,
                positive_mask=positive_mask,
            )
            static_metrics = _ranking_metrics_from_logits(
                prior_logits,
                next_labels,
                rows=next_rows,
                positive_mask=positive_mask,
            )
            metrics.update(
                {f"transition_unified_dynamic_{key.removeprefix('transition_')}": value for key, value in dynamic_metrics.items()}
            )
            metrics.update(
                {f"transition_unified_static_{key.removeprefix('transition_')}": value for key, value in static_metrics.items()}
            )
            metrics.update(
                _dynamic_static_rank_metrics(
                    trans_skill_logits,
                    prior_logits,
                    next_labels,
                    positive_mask=positive_mask,
                    prefix="transition_unified",
                )
            )
        else:
            metrics["transition_unified_dynamic_vs_static_delta_mrr"] = 0.0
            metrics["transition_unified_positive_rank_improved_rows"] = 0.0
            metrics["transition_unified_positive_rank_worsened_rows"] = 0.0
            metrics["transition_unified_argmax_changed_rows"] = 0.0
        metrics.update(
            _rank_delta_metrics_vs_prior(
                trans_skill_logits,
                prior_logits,
                next_labels,
                positive_mask=positive_mask,
            )
        )
        metrics.update(
            _current_skill_candidate_metrics(
                rows=next_rows,
                candidate_rows=candidate_rows_for_positive_mask,
                logits=trans_skill_logits,
                skill_id_to_idx=skill_id_to_idx,
            )
        )
        metrics["transition_skill_ce_count"] = float(len(trans_skill_items))
        real_rows = sum(1 for row in next_rows if not bool(row.get("stage0_positive_injected")))
        injected_rows = len(next_rows) - real_rows
        metrics["transition_real_candidate_rows"] = float(real_rows)
        metrics["transition_injected_candidate_rows"] = float(injected_rows)
        if weights["transition_hard_negative_margin"] > 0.0:
            hard_negative_loss, hard_negative_count = _transition_hard_negative_margin_loss(
                trans_skill_logits,
                next_labels,
                next_rows,
                margin=transition_hard_negative_margin,
            )
        else:
            hard_negative_loss = trans_skill_logits.new_zeros(())
            hard_negative_count = 0
        losses.append(
            _weighted_term(
                metrics,
                "transition_hard_negative_margin",
                hard_negative_loss,
                weights["transition_hard_negative_margin"],
            )
        )
        metrics["transition_hard_negative_margin_loss"] = float(hard_negative_loss.detach().cpu().item())
        metrics["transition_hard_negative_pair_count"] = float(hard_negative_count)
        metrics["counterfactual_utility_loss"] = 0.0
        metrics["counterfactual_scaled_loss"] = 0.0
        metrics["counterfactual_scale"] = float(counterfactual_scale)
        metrics["counterfactual_gain_loss"] = 0.0
        metrics["counterfactual_safety_loss"] = 0.0
        metrics["counterfactual_weak_rows"] = 0.0
        metrics["counterfactual_strong_rows"] = 0.0
        metrics["counterfactual_eligible_rows"] = 0.0
        metrics["counterfactual_gain_violation_rows"] = 0.0
        metrics["counterfactual_safety_violation_rows"] = 0.0
    else:
        metrics["transition_skill_ce_loss"] = 0.0
        metrics["transition_skill_head_type"] = "none"
        metrics["transition_scoring_mode"] = effective_transition_scoring_mode
        metrics["transition_residual_lambda"] = effective_transition_residual_lambda
        metrics["gated_temporal_lambda_mean"] = 0.0
        metrics["gated_temporal_lambda_min"] = 0.0
        metrics["gated_temporal_lambda_max"] = 0.0
        metrics["gated_temporal_prior_kl_loss"] = 0.0
        metrics["gated_temporal_rank_drop_loss"] = 0.0
        metrics["gated_temporal_kl_alpha"] = 0.0
        metrics["gated_temporal_rank_drop_beta"] = 0.0
        metrics["gated_temporal_context_top_k"] = 0.0
        metrics["transition_skill_recall@1"] = 0.0
        metrics["transition_skill_recall@5"] = 0.0
        metrics["transition_skill_mrr"] = 0.0
        metrics["transition_skill_ce_candidate_count"] = 0.0
        metrics["transition_skill_ce_count"] = 0.0
        metrics["transition_real_candidate_rows"] = 0.0
        metrics["transition_injected_candidate_rows"] = 0.0
        metrics["transition_hard_negative_margin_loss"] = 0.0
        metrics["transition_hard_negative_pair_count"] = 0.0
        metrics["transition_full_pool_rows"] = 0.0
        metrics["transition_static_hit_train_rows"] = 0.0
        metrics["transition_static_miss_train_rows"] = 0.0
        metrics["counterfactual_utility_loss"] = 0.0
        metrics["counterfactual_scaled_loss"] = 0.0
        metrics["counterfactual_scale"] = float(counterfactual_scale)
        metrics["counterfactual_gain_loss"] = 0.0
        metrics["counterfactual_safety_loss"] = 0.0
        metrics["counterfactual_weak_rows"] = 0.0
        metrics["counterfactual_strong_rows"] = 0.0
        metrics["counterfactual_eligible_rows"] = 0.0
        metrics["counterfactual_gain_violation_rows"] = 0.0
        metrics["counterfactual_safety_violation_rows"] = 0.0
        metrics["positive_not_in_skill_table_rows"] = 0.0
        metrics["positive_outside_legal_pool_rows"] = 0.0
        metrics["explicit_inventory_no_known_skill_rows"] = 0.0
        metrics["no_legal_negative_rows"] = 0.0
        metrics["nonfinite_logit_rows"] = 0.0
        metrics["transition_unified_dynamic_vs_static_delta_mrr"] = 0.0
        metrics["transition_unified_positive_rank_improved_rows"] = 0.0
        metrics["transition_unified_positive_rank_worsened_rows"] = 0.0
        metrics["transition_unified_argmax_changed_rows"] = 0.0
        metrics["transition_post_action_update_rows"] = 0.0
        metrics["transition_next_state_rows"] = 0.0
        metrics["transition_inventory_mask_mode"] = transition_inventory_mask_mode
        metrics["transition_loss_type"] = transition_loss_type
        metrics["transition_positive_mode"] = transition_positive_mode
        metrics["transition_inventory_mask_applied_rows"] = 0.0
        metrics["transition_inventory_mask_removed_candidates"] = 0.0
        metrics["transition_inventory_mask_missing_rows"] = 0.0
        metrics["transition_inventory_mask_positive_missing_rows"] = 0.0
        metrics["transition_inventory_mask_backfilled_rows"] = 0.0
        metrics["transition_inventory_mask_backfilled_candidates"] = 0.0
        metrics["transition_positive_mean_count"] = 0.0
        metrics["transition_positive_max_count"] = 0.0
        metrics["transition_multi_positive_rows"] = 0.0
        metrics["transition_skill_self_count"] = 0.0
        metrics["transition_skill_self_recall@1"] = 0.0
        metrics["transition_skill_self_recall@5"] = 0.0
        metrics["transition_skill_self_mrr"] = 0.0
        metrics["transition_skill_switch_count"] = 0.0
        metrics["transition_skill_switch_recall@1"] = 0.0
        metrics["transition_skill_switch_recall@5"] = 0.0
        metrics["transition_skill_switch_mrr"] = 0.0

    belief_items = [
        idx
        for idx, row in enumerate(batch)
        if weights["belief"] > 0.0 and (row.get("loss_mask") or {}).get("belief")
    ]
    if belief_items:
        indices = torch.tensor(belief_items, dtype=torch.long, device=device)
        belief_rows = [batch[idx] for idx in belief_items]
        target_h = _batch_cached_or_encode(
            model,
            belief_rows,
            "_next_observation_embedding",
            "next_observation_text",
            device,
            text_role=TRANSITION_TEXT_ROLE,
        )
        action_h = _batch_action_text_embedding_or_none(model, belief_rows, device)
        target = _next_belief_target_from_embeddings(model, target_h, skill_count)
        labels = torch.tensor(
            [skill_id_to_idx.get(str(batch[idx].get("skill_id")), 0) for idx in belief_items],
            dtype=torch.long,
            device=device,
        )
        pred = _transition_prediction(
            model,
            h.index_select(0, indices),
            m_obs.index_select(0, indices),
            labels,
            target_h,
            action_emb=action_h,
        )
        belief = _belief_prediction(model, pred, target, target_h)
        belief_loss = 1.0 - F.cosine_similarity(belief.float(), target.float(), dim=-1).mean()
        losses.append(_weighted_term(metrics, "belief", belief_loss, weights["belief"]))
        metrics["belief_cosine_loss"] = float(belief_loss.detach().cpu().item())
    else:
        metrics["belief_cosine_loss"] = 0.0

    if not losses:
        loss = h.sum() * 0.0
    else:
        loss = sum(losses)
    metrics["loss"] = float(loss.detach().cpu().item())
    return loss, metrics


def train_clstr_full_base_with_model(
    model: Any,
    model_config: dict[str, Any],
    routing_report: dict[str, Any],
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 1000,
    target_total_steps: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    minimum_learning_rate: float | None = None,
    checkpoint_interval_steps: int = 400,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    transition_real_candidate_ce_multiplier: float = 1.0,
    transition_injected_candidate_ce_multiplier: float = 1.0,
    transition_hard_negative_margin: float = 1.0,
    transition_inventory_mask_mode: str = "off",
    transition_inventory_min_candidates: int = 0,
    transition_loss_type: str = "cross_entropy",
    transition_positive_mode: str = "single",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
    freeze_gated_temporal_only: bool = False,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    routing_checkpoint_path: str | Path | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    max_rows: int | None = None,
    stage0_top_m: int | None = None,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = None,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    stage0_handoff_cache_mode: str = "auto",
    stage0_handoff_cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    stage0_handoff_cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    stage0_handoff_cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    allow_full_pool_stage2_debug: bool = False,
    setup_status_path: str | Path | None = None,
    stage_name: str = "clstr_full_base_component_complete",
    checkpoint_prefix: str = "clstr_full_base",
    training_objective: str = "component_complete_masked_multi_loss",
    sampling_strategy: str = "balanced_deterministic",
    auto_replay_prefix_max_steps: int = 0,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    anchored_routing_foundation: bool = False,
    static_route_anchor_weight: float = 0.1,
    static_route_anchor_max_regression: float = 0.005,
    next_skill_pool_mode: str = "stage0_candidates",
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_warmup_fraction: float = 0.1,
    counterfactual_history_margin: float = 0.1,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
    warm_start_checkpoint_path: str | Path | None = None,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    train_path = Path(train_path)
    skills_path = Path(skills_path)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    setup_status = Path(setup_status_path) if setup_status_path is not None else output_dir / "setup_status.jsonl"
    if setup_status_path is None:
        reset_setup_status(setup_status)
    random.seed(seed)
    torch.manual_seed(seed)
    learning_rate = float(learning_rate)
    minimum_learning_rate = (
        learning_rate if minimum_learning_rate is None else float(minimum_learning_rate)
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if (
        not math.isfinite(minimum_learning_rate)
        or minimum_learning_rate <= 0.0
        or minimum_learning_rate > learning_rate
    ):
        raise ValueError("minimum_learning_rate must be finite, positive, and no greater than learning_rate")
    checkpoint_interval_steps = max(1, int(checkpoint_interval_steps))
    raw_rows = _read_jsonl(train_path)
    rows = [
        materialize_history_free_state(row, replace_state_text=True)
        for row in raw_rows
    ]
    initial_history_channel_report = audit_history_channel_rows(
        rows,
        require_explicit_current=True,
    )
    if initial_history_channel_report["status"] != "ok":
        raise ValueError(
            "full-base router state still contains execution history: "
            f"{initial_history_channel_report}"
        )
    if include_available_actions_in_state:
        rows, available_actions_preparation_report = _augment_rows_with_available_actions_planner_context(rows)
    else:
        available_actions_preparation_report = {
            "enabled": False,
            "mode": "none",
            "augmented_row_count": 0,
            "row_count": len(rows),
        }
    rows, causal_next_state_report = _attach_adjacent_next_states(rows)
    skills = _read_jsonl(skills_path)
    append_setup_status(
        setup_status,
        "data_loaded",
        train_path=str(train_path),
        skills_path=str(skills_path),
        raw_row_count=len(rows),
        skill_count=len(skills),
        causal_next_state=causal_next_state_report,
        history_channel=initial_history_channel_report,
    )
    rows, train_split_filter_report = _filter_rows_by_train_split(rows)
    rows, benchmark_filter_report = _filter_rows_by_allowed_benchmarks(rows, allowed_benchmarks)
    rows, benchmark_caps_report = _cap_rows_by_benchmark(rows, benchmark_caps)
    rows, max_rows_filter_report = _limit_rows_for_smoke(rows, max_rows)
    append_setup_status(
        setup_status,
        "rows_filtered",
        row_count=len(rows),
        train_split_filter=train_split_filter_report,
        benchmark_filter=benchmark_filter_report,
        benchmark_caps=benchmark_caps_report,
        max_rows_filter=max_rows_filter_report,
    )
    if not rows:
        raise ValueError(
            f"no full-base train rows in {train_path} after filters "
            f"{train_split_filter_report} {benchmark_filter_report}"
        )
    available_actions_context_report = {
        **available_actions_preparation_report,
        "preparation_row_count": int(available_actions_preparation_report["row_count"]),
        "preparation_augmented_row_count": int(
            available_actions_preparation_report["augmented_row_count"]
        ),
        "row_count": len(rows),
        "augmented_row_count": sum(
            1 for row in rows if bool(row.get("available_actions_planner_context"))
        ),
    }
    rows, auto_replay_prefix_report = _attach_auto_replay_prefixes(
        rows,
        max_steps=auto_replay_prefix_max_steps,
    )
    replay_history_channel_report = audit_history_channel_rows(
        rows,
        require_explicit_current=True,
        require_actual_replay_observation=True,
    )
    if replay_history_channel_report["status"] != "ok":
        raise ValueError(
            "full-base replay prefix is not an executed action-result trace: "
            f"{replay_history_channel_report}"
        )
    append_setup_status(
        setup_status,
        "auto_replay_prefix_prepared",
        **auto_replay_prefix_report,
        trainable_replay_prefix=bool(trainable_replay_prefix),
        history_channel=replay_history_channel_report,
    )
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skills) if row.get("skill_id")}
    equivalent_skill_ids_by_skill_id = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(model, "to"):
        model.to(device)
    append_setup_status(setup_status, "model_moved_to_device", device=str(device))
    skill_text_format_report = _apply_skill_text_format(model, model_config, skill_text_format)
    routing_checkpoint_payload = _routing_checkpoint_payload(routing_checkpoint_path)
    skill_table_rebuild_skipped = False
    if bool(getattr(getattr(model, "config", None), "defer_skill_table_init", False)) and hasattr(model, "rebuild_skill_table"):
        skill_table_rebuild_skipped = _can_skip_skill_table_rebuild_from_routing_checkpoint(
            model,
            routing_checkpoint_payload,
        )
        if not skill_table_rebuild_skipped:
            append_setup_status(setup_status, "skill_table_rebuild_started")
            model.rebuild_skill_table()
            append_setup_status(setup_status, "skill_table_rebuilt")
        else:
            append_setup_status(setup_status, "skill_table_rebuild_skipped", reason="routing_checkpoint_can_seed_table")
    else:
        append_setup_status(setup_status, "skill_table_rebuild_skipped", reason="not_deferred_or_no_rebuild_method")
    routing_checkpoint_report = _load_routing_checkpoint_into_model(
        model,
        routing_checkpoint_path,
        payload=routing_checkpoint_payload,
        skill_table_rebuild_skipped=skill_table_rebuild_skipped,
    )
    append_setup_status(
        setup_status,
        "routing_checkpoint_loaded",
        routing_checkpoint_path=str(routing_checkpoint_path) if routing_checkpoint_path is not None else None,
        loaded=bool(routing_checkpoint_report.get("loaded", False)),
        skipped=bool(routing_checkpoint_report.get("skipped", False)),
    )
    warm_start_report = _load_full_base_warm_start_model_state(
        model,
        warm_start_checkpoint_path,
    )
    routing_report = {
        **routing_report,
        "stage2_counterfactual_warm_start": warm_start_report,
    }
    append_setup_status(
        setup_status,
        "stage2_counterfactual_warm_start_loaded",
        warm_start=warm_start_report,
    )
    normalized_loss_weights = _normalize_loss_weights(loss_weights)
    reported_loss_weights = _reported_loss_weights(normalized_loss_weights)
    transition_inventory_mask_mode = str(transition_inventory_mask_mode or "off")
    transition_inventory_min_candidates = max(0, int(transition_inventory_min_candidates or 0))
    transition_loss_type = str(transition_loss_type or "cross_entropy")
    transition_positive_mode = str(transition_positive_mode or "single")
    next_skill_pool_mode = str(next_skill_pool_mode or "stage0_candidates")
    counterfactual_gain_margin = float(counterfactual_gain_margin)
    counterfactual_safety_tolerance = float(counterfactual_safety_tolerance)
    counterfactual_gain_weight = float(counterfactual_gain_weight)
    counterfactual_safety_weight = float(counterfactual_safety_weight)
    counterfactual_warmup_fraction = float(counterfactual_warmup_fraction)
    counterfactual_history_margin = float(counterfactual_history_margin)
    static_route_anchor_weight = float(static_route_anchor_weight)
    static_route_anchor_max_regression = float(static_route_anchor_max_regression)
    safe_memory_residual_bound = float(safe_memory_residual_bound)
    safe_local_candidate_sizes = tuple(int(size) for size in safe_local_candidate_sizes)
    _validate_counterfactual_hyperparameters(
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
    )
    counterfactual_utility_weight = float(normalized_loss_weights.get("counterfactual_utility", 0.0))
    if not math.isfinite(counterfactual_utility_weight) or counterfactual_utility_weight < 0.0:
        raise ValueError("loss_weights[counterfactual_utility] must be finite and nonnegative")
    if not math.isfinite(counterfactual_history_margin) or counterfactual_history_margin < 0.0:
        raise ValueError("counterfactual_history_margin must be finite and nonnegative")
    transition_residual_lambda = float(transition_residual_lambda)
    transition_scoring_mode = str(transition_scoring_mode or TRANSITION_SCORING_MODE)
    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    gated_temporal_lambda_max = float(gated_temporal_lambda_max)
    gated_temporal_kl_alpha = float(gated_temporal_kl_alpha)
    gated_temporal_rank_drop_beta = float(gated_temporal_rank_drop_beta)
    gated_temporal_context_top_k = int(gated_temporal_context_top_k)
    stage0_score_prior_calibration = str(stage0_score_prior_calibration or STAGE0_SCORE_PRIOR_CALIBRATION)
    if stage0_score_prior_calibration not in STAGE0_SCORE_PRIOR_CALIBRATIONS:
        raise ValueError(f"unsupported stage0 score prior calibration: {stage0_score_prior_calibration}")
    if transition_inventory_mask_mode not in TRANSITION_INVENTORY_MASK_MODES:
        raise ValueError(f"unsupported transition inventory mask mode: {transition_inventory_mask_mode}")
    if transition_loss_type not in TRANSITION_LOSS_TYPES:
        raise ValueError(f"unsupported transition loss type: {transition_loss_type}")
    if transition_positive_mode not in TRANSITION_POSITIVE_MODES:
        raise ValueError(f"unsupported transition positive mode: {transition_positive_mode}")
    if transition_scoring_mode not in TRANSITION_SCORING_MODES:
        raise ValueError(f"unsupported transition scoring mode: {transition_scoring_mode}")
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    if anchored_routing_foundation and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("anchored_routing_foundation requires route_scorer=unified_memory")
    if not math.isfinite(static_route_anchor_weight) or static_route_anchor_weight < 0.0:
        raise ValueError("static_route_anchor_weight must be finite and nonnegative")
    if anchored_routing_foundation and static_route_anchor_weight <= 0.0:
        raise ValueError("anchored_routing_foundation requires positive static_route_anchor_weight")
    if (
        not math.isfinite(static_route_anchor_max_regression)
        or static_route_anchor_max_regression < 0.0
    ):
        raise ValueError("static_route_anchor_max_regression must be finite and nonnegative")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    if next_skill_pool_mode == "full_pool" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("next_skill_pool_mode=full_pool requires route_scorer=unified_memory")
    if next_skill_pool_mode == "full_pool" and transition_inventory_mask_mode != "explicit_only":
        raise ValueError("full-pool next-skill training requires transition_inventory_mask_mode=explicit_only")
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
    stage0_handoff_cache_mode = str(stage0_handoff_cache_mode or "off")
    if stage0_handoff_cache_mode not in STAGE0_HANDOFF_CACHE_MODES:
        raise ValueError(f"unsupported stage0_handoff_cache_mode: {stage0_handoff_cache_mode}")
    stage0_handoff_cache_format = str(
        stage0_handoff_cache_format or LEGACY_STAGE0_HANDOFF_CACHE_FORMAT
    )
    if stage0_handoff_cache_format not in STAGE0_HANDOFF_CACHE_FORMATS:
        raise ValueError(
            f"unsupported stage0_handoff_cache_format: {stage0_handoff_cache_format}"
        )
    stage0_handoff_cache_shard_size = int(stage0_handoff_cache_shard_size)
    if stage0_handoff_cache_shard_size <= 0:
        raise ValueError("stage0_handoff_cache_shard_size must be positive")
    if effective_transition_scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE:
        _prepare_gated_temporal_reranker_for_training(
            model,
            model_config,
            lambda_max=gated_temporal_lambda_max,
            context_top_k=gated_temporal_context_top_k,
            device=device,
        )
    sampling_strategy = str(sampling_strategy or "balanced_deterministic")
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"unsupported full-base sampling_strategy: {sampling_strategy}")
    rows, stage0_candidate_handoff_subset_report = _select_stage0_handoff_training_subset(
        rows,
        max_steps=max_steps,
        batch_size=batch_size,
        loss_weights=normalized_loss_weights,
        sample_multiplier=stage0_handoff_sample_multiplier,
        sampling_strategy=sampling_strategy,
        sampler_seed=seed,
    )
    append_setup_status(
        setup_status,
        "stage0_candidate_handoff_subset_prepared",
        **stage0_candidate_handoff_subset_report,
    )
    stage0_handoff_manifest_path = output_dir / "stage0_candidate_handoff.json"
    rows, stage0_candidate_handoff_report = _prepare_stage0_topm_candidates_with_cache(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=stage0_top_m,
        positive_missing_policy=stage0_positive_missing_policy,
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
        routing_checkpoint_path=routing_checkpoint_path,
        manifest_path=stage0_handoff_manifest_path,
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        setup_status_path=setup_status,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
        inventory_min_candidates=transition_inventory_min_candidates,
        next_skill_pool_mode=next_skill_pool_mode,
        cache_mode=stage0_handoff_cache_mode,
        cache_dir=stage0_handoff_cache_dir,
        cache_format=stage0_handoff_cache_format,
        cache_shard_size=stage0_handoff_cache_shard_size,
        route_scorer=route_scorer,
        model_config=model_config,
    )
    append_setup_status(
        setup_status,
        "stage0_candidate_handoff_prepared",
        **stage0_candidate_handoff_report,
    )
    if not rows:
        raise ValueError(
            "no full-base train rows remain after Stage0 candidate handoff "
            f"{stage0_candidate_handoff_report}"
        )
    static_route_teacher: Stage2StaticRouteTeacher | None = None
    static_route_teacher_digest: str | None = None
    if anchored_routing_foundation:
        static_route_teacher = Stage2StaticRouteTeacher.from_model(model).to(device)
        static_route_teacher.requires_grad_(False)
        static_route_teacher.eval()
        static_route_teacher_digest = _stage2_route_teacher_digest(
            static_route_teacher,
            model,
        )
    static_route_anchor_config = {
        "enabled": bool(anchored_routing_foundation),
        "weight": float(static_route_anchor_weight if anchored_routing_foundation else 0.0),
        "max_static_regression": float(static_route_anchor_max_regression),
        "teacher_digest": static_route_teacher_digest,
        "teacher_scope": (
            "initial_belief_transition_gate_action_projection_unified_retriever_no_qwen_no_skill_table_copy"
            if anchored_routing_foundation
            else None
        ),
    }
    append_setup_status(
        setup_status,
        "stage2_static_route_teacher_prepared",
        **static_route_anchor_config,
    )
    if (
        normalized_loss_weights["Q_success"] > 0.0
        and any((row.get("loss_mask") or {}).get("Q_success") for row in rows)
        and getattr(model, "q_success_head", None) is None
    ):
        model.q_success_head = QSuccessHead(int(model_config.get("d", 128))).to(device)
    freeze_audit = _freeze_for_full_base(
        model,
        transition_scoring_mode=transition_scoring_mode,
        freeze_gated_temporal_only=freeze_gated_temporal_only,
        route_scorer=route_scorer,
        anchored_routing_foundation=anchored_routing_foundation,
        active_loss_weights=normalized_loss_weights,
    )
    effective_freeze_gated_temporal_only = bool(freeze_audit["freeze_gated_temporal_only"])
    external_encoder_metadata = _external_encoder_metadata(model, model_config, routing_report)
    qwen_external_encoder = bool(external_encoder_metadata.get("qwen_external_encoder", False))
    exclude_frozen_encoder_backbone = _frozen_encoder_checkpoint_exclusion(
        model,
        external_encoder_metadata,
    )
    embedding_cache_batch_size = 8 if qwen_external_encoder else 256
    embedding_cache_policy = _embedding_cache_policy(
        mode=embedding_cache_mode,
        row_count=len(rows),
        qwen_external_encoder=qwen_external_encoder,
        max_rows=embedding_cache_max_rows,
    )
    append_setup_status(
        setup_status,
        "embedding_cache_started",
        policy=embedding_cache_policy,
        batch_size=embedding_cache_batch_size,
    )
    if embedding_cache_policy["cache_enabled"]:
        full_base_embedding_cache_report = _attach_full_base_embedding_cache(model, rows, encode_batch_size=embedding_cache_batch_size)
        if normalized_loss_weights["L_policy"] > 0.0:
            policy_embedding_cache_report = _attach_policy_embedding_cache(
                model,
                rows,
                encode_batch_size=embedding_cache_batch_size,
            )
        else:
            policy_embedding_cache_report = _skipped_embedding_cache_report(
                "policy_candidates_zero_weight",
                rows,
                embedding_cache_policy,
            )
    else:
        full_base_embedding_cache_report = _skipped_embedding_cache_report("full_base_replay", rows, embedding_cache_policy)
        policy_embedding_cache_report = _skipped_embedding_cache_report("policy_candidates", rows, embedding_cache_policy)
    append_setup_status(
        setup_status,
        "embedding_cache_prepared",
        full_base_embedding_cache=full_base_embedding_cache_report,
        policy_embedding_cache=policy_embedding_cache_report,
    )
    action_adapter = UniversalActionAdapter(int(model_config.get("d", 128)), hidden_dim=int(model_config.get("d", 128))).to(device)
    native_policy_head = _has_native_policy_head(model)
    policy_head_type = "native_skill_head" if native_policy_head else "universal_action_adapter"
    legacy_universal_action_adapter_role = "fallback_only" if native_policy_head else "primary_policy_head"
    resume_payload, resume_report = _load_full_base_resume_model_state(
        model=model,
        action_adapter=action_adapter,
        native_policy_head=native_policy_head,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    loss_activation_counts = _loss_activation_counts(rows)
    clstr_native_act_trained = bool(native_policy_head and loss_activation_counts.get("L_policy", 0) > 0)
    if not native_policy_head and "universal_action_adapter" not in freeze_audit["trainable_modules"]:
        freeze_audit["trainable_modules"].append("universal_action_adapter")
    belief_memory_semantics = _belief_memory_semantics_report(model, freeze_audit)
    policy_feature_semantics = _policy_feature_semantics_report(model)
    params = [param for param in model.parameters() if param.requires_grad] if hasattr(model, "parameters") else []
    if not native_policy_head:
        params.extend(action_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    resume_report = _load_full_base_resume_optimizer_state(
        optimizer=optimizer,
        resume_payload=resume_payload,
        resume_report=resume_report,
        device=device,
    )
    resume_start_step = int(resume_report.get("start_step", 0) or 0)
    if target_total_steps is None:
        trained_steps_this_run = max(1, int(max_steps))
        final_step = resume_start_step + trained_steps_this_run
    else:
        target_total_steps = int(target_total_steps)
        if target_total_steps > int(max_steps):
            raise ValueError("target_total_steps must not exceed the max_steps schedule budget")
        if target_total_steps <= resume_start_step:
            raise ValueError("target_total_steps must be greater than the resume checkpoint step")
        trained_steps_this_run = target_total_steps - resume_start_step
        final_step = target_total_steps
    resume_report["target_total_steps"] = (
        None if target_total_steps is None else int(target_total_steps)
    )
    resume_report["trained_steps_this_run"] = int(trained_steps_this_run)
    resume_report["final_step"] = int(final_step)
    metrics: dict[str, Any] = {}
    metric_sums: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    weighted_term_sums: Counter[str] = Counter()
    weighted_term_counts: Counter[str] = Counter()
    sampled_loss_activation_counts: Counter[str] = Counter()
    model.train()
    if native_policy_head:
        action_adapter.eval()
    else:
        action_adapter.train()
    batch_size = max(1, int(batch_size))
    loss_buckets = _build_loss_buckets(rows, normalized_loss_weights)
    grouped_loss_buckets = (
        _build_transition_grouped_loss_buckets(rows, loss_buckets)
        if sampling_strategy == BENCHMARK_TRANSITION_BALANCED_RANDOM
        else None
    )
    all_group_buckets = (
        _group_indices_by_transition_relation(rows, list(range(len(rows))))
        if sampling_strategy in GROUPED_SAMPLING_STRATEGIES
        else None
    )
    grouped_quota_buckets = (
        _build_quota_grouped_buckets(rows, loss_buckets)
        if sampling_strategy == BENCHMARK_TRANSITION_QUOTA_RANDOM
        else None
    )
    monitor = TrainingMonitor(
        output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_steps=checkpoint_interval_steps,
        curve_interval_steps=checkpoint_interval_steps,
    )
    initial_checkpoint_state, initial_checkpoint_report = _checkpoint_state_dict(
        model,
        exclude_frozen_qwen_backbone=exclude_frozen_encoder_backbone,
        exclude_frozen_routing_foundation=bool(
            freeze_audit["frozen_routing_foundation"]
        ),
    )
    monitor.save_latest(
        {
            "stage": stage_name,
            "step": resume_start_step,
            "config": model_config,
            "model_state_dict": initial_checkpoint_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "universal_action_adapter_state_dict": None if native_policy_head else action_adapter.state_dict(),
            "train_path": str(train_path),
            "skills_path": str(skills_path),
            "loss_weights": reported_loss_weights,
            "frozen_routing_foundation": freeze_audit["frozen_routing_foundation"],
            "routing_foundation_trainable_for_retrieval": bool(
                anchored_routing_foundation
            ),
            "qwen_and_skill_table_frozen": freeze_audit[
                "qwen_and_skill_table_frozen"
            ],
            "trainable_modules": freeze_audit["trainable_modules"],
            "stage2_static_route_anchor": static_route_anchor_config,
            "transition_candidate_training": {
                "real_candidate_ce_multiplier": float(transition_real_candidate_ce_multiplier),
                "injected_candidate_ce_multiplier": float(transition_injected_candidate_ce_multiplier),
                "hard_negative_margin": float(transition_hard_negative_margin),
                "hard_negative_loss_weight": float(normalized_loss_weights.get("transition_hard_negative_margin", 0.0)),
                "inventory_mask_mode": transition_inventory_mask_mode,
                "inventory_min_candidates": int(transition_inventory_min_candidates),
                "loss_type": transition_loss_type,
                "positive_mode": transition_positive_mode,
                "scoring_mode": effective_transition_scoring_mode,
                "residual_lambda": effective_transition_residual_lambda,
                "gated_temporal_lambda_max": gated_temporal_lambda_max,
                "gated_temporal_kl_alpha": gated_temporal_kl_alpha,
                "gated_temporal_rank_drop_beta": gated_temporal_rank_drop_beta,
                "gated_temporal_context_top_k": int(gated_temporal_context_top_k),
                "freeze_gated_temporal_only": effective_freeze_gated_temporal_only,
                "freeze_gated_temporal_only_requested": bool(freeze_gated_temporal_only),
                "route_scorer": route_scorer,
                "next_skill_pool_mode": next_skill_pool_mode,
                "counterfactual_gain_margin": counterfactual_gain_margin,
                "counterfactual_safety_tolerance": counterfactual_safety_tolerance,
                "counterfactual_gain_weight": counterfactual_gain_weight,
                "counterfactual_safety_weight": counterfactual_safety_weight,
                "counterfactual_warmup_fraction": counterfactual_warmup_fraction,
                "counterfactual_history_margin": counterfactual_history_margin,
                "safe_memory_residual_bound": safe_memory_residual_bound,
                "safe_local_candidate_sizes": list(safe_local_candidate_sizes),
            },
            "routing_init": routing_report,
            "routing_checkpoint": routing_checkpoint_report,
            "auto_replay_prefix": auto_replay_prefix_report,
            "causal_next_state": causal_next_state_report,
            "history_channel": replay_history_channel_report,
            "trainable_replay_prefix": bool(trainable_replay_prefix),
            "stage0_candidate_handoff_subset": stage0_candidate_handoff_subset_report,
            "stage0_candidate_handoff": stage0_candidate_handoff_report,
            "benchmark_caps": benchmark_caps_report,
            "max_rows_filter": max_rows_filter_report,
            "resume": resume_report,
            "metrics": metrics,
            "setup_status_path": str(setup_status),
            **initial_checkpoint_report,
            **external_encoder_metadata,
        }
    )
    append_setup_status(
        setup_status,
        "training_started",
        max_steps=int(max_steps),
        start_step=int(resume_start_step),
        final_step=int(final_step),
        trained_steps_this_run=int(trained_steps_this_run),
        batch_size=batch_size,
        sampling_strategy=sampling_strategy,
        sampler_seed=int(seed),
        resume=resume_report,
    )
    for local_step_idx in range(1, trained_steps_this_run + 1):
        step_idx = resume_start_step + local_step_idx
        schedule_progress = float(local_step_idx - 1) / float(
            max(trained_steps_this_run - 1, 1)
        )
        current_learning_rate = learning_rate + (
            minimum_learning_rate - learning_rate
        ) * schedule_progress
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        batch = _sample_full_base_batch(
            rows,
            step_idx,
            batch_size,
            normalized_loss_weights,
            loss_buckets=loss_buckets,
            sampling_strategy=sampling_strategy,
            sampler_seed=seed,
            grouped_loss_buckets=grouped_loss_buckets,
            all_group_buckets=all_group_buckets,
            grouped_quota_buckets=grouped_quota_buckets,
        )
        for row in batch:
            for key, enabled in (row.get("loss_mask") or {}).items():
                if enabled:
                    sampled_loss_activation_counts[str(key)] += 1
        loss, metrics = _compute_full_base_loss(
            model,
            action_adapter,
            batch,
            skill_id_to_idx,
            device,
            loss_weights=normalized_loss_weights,
            retrieval_num_negatives=retrieval_num_negatives,
            retrieval_hard_ratio=retrieval_hard_ratio,
            transition_real_candidate_ce_multiplier=transition_real_candidate_ce_multiplier,
            transition_injected_candidate_ce_multiplier=transition_injected_candidate_ce_multiplier,
            transition_hard_negative_margin=transition_hard_negative_margin,
            transition_inventory_mask_mode=transition_inventory_mask_mode,
            transition_inventory_min_candidates=transition_inventory_min_candidates,
            transition_loss_type=transition_loss_type,
            transition_positive_mode=transition_positive_mode,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            gated_temporal_lambda_max=gated_temporal_lambda_max,
            gated_temporal_kl_alpha=gated_temporal_kl_alpha,
            gated_temporal_rank_drop_beta=gated_temporal_rank_drop_beta,
            gated_temporal_context_top_k=gated_temporal_context_top_k,
            stage0_score_prior_calibration=stage0_score_prior_calibration,
            trainable_replay_prefix=trainable_replay_prefix,
            route_scorer=route_scorer,
            next_skill_pool_mode=next_skill_pool_mode,
            counterfactual_gain_margin=counterfactual_gain_margin,
            counterfactual_safety_tolerance=counterfactual_safety_tolerance,
            counterfactual_gain_weight=counterfactual_gain_weight,
            counterfactual_safety_weight=counterfactual_safety_weight,
            counterfactual_scale=_counterfactual_warmup_scale(
                step_idx,
                max_steps,
                counterfactual_warmup_fraction,
            ),
            counterfactual_history_margin=counterfactual_history_margin,
            safe_memory_residual_bound=safe_memory_residual_bound,
            safe_local_candidate_sizes=safe_local_candidate_sizes,
            static_route_teacher=static_route_teacher,
            static_route_anchor_weight=(
                static_route_anchor_weight if anchored_routing_foundation else 0.0
            ),
        )
        metrics.update(_batch_composition_metrics(batch))
        metrics["learning_rate"] = float(current_learning_rate)
        _raise_if_nonfinite_loss(loss, metrics=metrics, step_idx=step_idx, output_dir=output_dir)
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            metrics["optimizer_step_skipped_no_grad"] = 0.0
        else:
            metrics["optimizer_step_skipped_no_grad"] = 1.0
        anchor_rows_in_batch = float(metrics.get("stage2_static_route_anchor_rows", 0.0))
        anchor_quality_metrics = {
            "stage2_static_route_anchor_loss",
            "stage2_static_route_anchor_weighted_loss",
            "stage2_static_route_teacher_mrr",
            "stage2_static_route_student_mrr",
            "stage2_step_zero_dynamic_mrr",
            "stage2_dynamic_mrr",
            "stage2_static_mrr_delta_vs_teacher",
            "stage2_dynamic_mrr_delta_vs_step_zero",
        }
        for key, value in metrics.items():
            if key == "weighted_loss_terms" and isinstance(value, dict):
                for term_key, term_value in value.items():
                    weighted_term_sums[str(term_key)] += float(term_value)
                    weighted_term_counts[str(term_key)] += 1
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in anchor_quality_metrics and anchor_rows_in_batch <= 0.0:
                    continue
                metric_sums[str(key)] += float(value)
                metric_counts[str(key)] += 1
        step_metrics = {**metrics, "step": step_idx}
        step_checkpoint_payload = None
        if monitor.should_save_checkpoint(step_idx):
            step_checkpoint_state, step_checkpoint_report = _checkpoint_state_dict(
                model,
                exclude_frozen_qwen_backbone=exclude_frozen_encoder_backbone,
                exclude_frozen_routing_foundation=bool(
                    freeze_audit["frozen_routing_foundation"]
                ),
            )
            step_checkpoint_payload = {
                    "stage": stage_name,
                    "step": step_idx,
                    "config": model_config,
                    "model_state_dict": step_checkpoint_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "universal_action_adapter_state_dict": None if native_policy_head else action_adapter.state_dict(),
                    "train_path": str(train_path),
                    "skills_path": str(skills_path),
                    "loss_weights": reported_loss_weights,
                    "frozen_routing_foundation": freeze_audit[
                        "frozen_routing_foundation"
                    ],
                    "routing_foundation_trainable_for_retrieval": bool(
                        anchored_routing_foundation
                    ),
                    "qwen_and_skill_table_frozen": freeze_audit[
                        "qwen_and_skill_table_frozen"
                    ],
                    "trainable_modules": freeze_audit["trainable_modules"],
                    "stage2_static_route_anchor": {
                        **static_route_anchor_config,
                        "row_count": int(metrics.get("stage2_static_route_anchor_rows", 0.0)),
                        "anchor_kl": float(metrics.get("stage2_static_route_anchor_loss", 0.0)),
                        "static_mrr_delta_vs_teacher": float(
                            metrics.get("stage2_static_mrr_delta_vs_teacher", 0.0)
                        ),
                        "dynamic_mrr_delta_vs_step_zero": float(
                            metrics.get("stage2_dynamic_mrr_delta_vs_step_zero", 0.0)
                        ),
                    },
                    "transition_candidate_training": {
                        "real_candidate_ce_multiplier": float(transition_real_candidate_ce_multiplier),
                        "injected_candidate_ce_multiplier": float(transition_injected_candidate_ce_multiplier),
                        "hard_negative_margin": float(transition_hard_negative_margin),
                        "hard_negative_loss_weight": float(normalized_loss_weights.get("transition_hard_negative_margin", 0.0)),
                        "inventory_mask_mode": transition_inventory_mask_mode,
                        "inventory_min_candidates": int(transition_inventory_min_candidates),
                        "loss_type": transition_loss_type,
                        "positive_mode": transition_positive_mode,
                        "scoring_mode": effective_transition_scoring_mode,
                        "residual_lambda": effective_transition_residual_lambda,
                        "gated_temporal_lambda_max": gated_temporal_lambda_max,
                        "gated_temporal_kl_alpha": gated_temporal_kl_alpha,
                        "gated_temporal_rank_drop_beta": gated_temporal_rank_drop_beta,
                        "gated_temporal_context_top_k": int(gated_temporal_context_top_k),
                        "freeze_gated_temporal_only": effective_freeze_gated_temporal_only,
                        "freeze_gated_temporal_only_requested": bool(freeze_gated_temporal_only),
                        "route_scorer": route_scorer,
                        "next_skill_pool_mode": next_skill_pool_mode,
                        "counterfactual_gain_margin": counterfactual_gain_margin,
                        "counterfactual_safety_tolerance": counterfactual_safety_tolerance,
                        "counterfactual_gain_weight": counterfactual_gain_weight,
                        "counterfactual_safety_weight": counterfactual_safety_weight,
                        "counterfactual_warmup_fraction": counterfactual_warmup_fraction,
                        "counterfactual_history_margin": counterfactual_history_margin,
                        "safe_memory_residual_bound": safe_memory_residual_bound,
                        "safe_local_candidate_sizes": list(safe_local_candidate_sizes),
                    },
                    "routing_init": routing_report,
                    "routing_checkpoint": routing_checkpoint_report,
                    "auto_replay_prefix": auto_replay_prefix_report,
                    "causal_next_state": causal_next_state_report,
                    "history_channel": replay_history_channel_report,
                    "trainable_replay_prefix": bool(trainable_replay_prefix),
                    "stage0_candidate_handoff_subset": stage0_candidate_handoff_subset_report,
                    "stage0_candidate_handoff": stage0_candidate_handoff_report,
                    "benchmark_caps": benchmark_caps_report,
                    "max_rows_filter": max_rows_filter_report,
                    "resume": resume_report,
                    "metrics": step_metrics,
                    "setup_status_path": str(setup_status),
                    **step_checkpoint_report,
                    **external_encoder_metadata,
                }
        monitor.record(
            step_metrics,
            step_checkpoint_payload,
        )

    metric_averages = {
        key: metric_sums[key] / max(metric_counts[key], 1)
        for key in sorted(metric_sums)
    }
    metric_averages["weighted_loss_terms"] = {
        key: weighted_term_sums[key] / max(weighted_term_counts[key], 1)
        for key in LOSS_WEIGHT_KEYS
    }
    stage2_static_route_anchor_report = {
        **static_route_anchor_config,
        "row_count": int(metric_sums.get("stage2_static_route_anchor_rows", 0.0)),
        "anchor_kl": float(metric_averages.get("stage2_static_route_anchor_loss", 0.0)),
        "weighted_anchor_loss": float(
            metric_averages.get("stage2_static_route_anchor_weighted_loss", 0.0)
        ),
        "teacher_static_mrr": float(
            metric_averages.get("stage2_static_route_teacher_mrr", 0.0)
        ),
        "student_static_mrr": float(
            metric_averages.get("stage2_static_route_student_mrr", 0.0)
        ),
        "step_zero_dynamic_mrr": float(
            metric_averages.get("stage2_step_zero_dynamic_mrr", 0.0)
        ),
        "dynamic_mrr": float(metric_averages.get("stage2_dynamic_mrr", 0.0)),
        "static_mrr_delta_vs_teacher": float(
            metric_averages.get("stage2_static_mrr_delta_vs_teacher", 0.0)
        ),
        "dynamic_mrr_delta_vs_step_zero": float(
            metric_averages.get("stage2_dynamic_mrr_delta_vs_step_zero", 0.0)
        ),
    }

    benchmark_counts = dict(sorted(Counter(str(row.get("benchmark")) for row in rows).items()))
    bucket_counts = dict(sorted(Counter(str(row.get("source_quality")) for row in rows).items()))
    split_values = [str((row.get("provenance") or {}).get("split", "")).lower() for row in rows]
    valid_or_test_used = any(split in {"valid", "valid_seen", "valid_unseen", "test"} for split in split_values)
    offline_rollout_source_counts = dict(
        sorted(
            Counter(str(row.get("source_quality") or "unknown") for row in rows if bool(row.get("on_policy_rollout"))).items()
        )
    )
    offline_rollout_source_rows_used = sum(offline_rollout_source_counts.values())
    offline_rollout_source_note = (
        "Rows may originate from logged or corrected rollouts, but this Stage2 trainer only consumes "
        "offline replay rows and does not execute simulator/on-policy rollouts during training."
    )
    on_policy_rollout_used = False
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}-step{final_step}.pt"
    transition_objective = "supervised_next_skill_ce_plus_optional_next_belief_cosine"
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
    transition_candidate_pool = (
        "declared_legal_full_skill_pool"
        if next_skill_pool_mode == "full_pool"
        else "stage0_topm_candidate_set"
        if stage0_candidate_handoff_report.get("candidate_source") == "stage0_topm_online"
        else "full skills.jsonl skill table"
    )
    transition_candidate_training_config = {
        "real_candidate_ce_multiplier": float(transition_real_candidate_ce_multiplier),
        "injected_candidate_ce_multiplier": float(transition_injected_candidate_ce_multiplier),
        "next_skill_pool_mode": next_skill_pool_mode,
        "inventory_mask_mode": transition_inventory_mask_mode,
        "inventory_min_candidates": int(transition_inventory_min_candidates),
        "loss_type": transition_loss_type,
        "positive_mode": transition_positive_mode,
        "scoring_mode": effective_transition_scoring_mode,
        "residual_lambda": effective_transition_residual_lambda,
        "stage0_score_prior_calibration": stage0_score_prior_calibration,
        "gated_temporal_lambda_max": gated_temporal_lambda_max,
        "gated_temporal_kl_alpha": gated_temporal_kl_alpha,
        "gated_temporal_rank_drop_beta": gated_temporal_rank_drop_beta,
        "gated_temporal_context_top_k": int(gated_temporal_context_top_k),
        "freeze_gated_temporal_only": effective_freeze_gated_temporal_only,
        "freeze_gated_temporal_only_requested": bool(freeze_gated_temporal_only),
        "route_scorer": route_scorer,
    }
    if normalized_loss_weights["transition_hard_negative_margin"] > 0.0:
        transition_candidate_training_config.update(
            {
                "hard_negative_margin": float(transition_hard_negative_margin),
                "hard_negative_loss_weight": float(
                    normalized_loss_weights["transition_hard_negative_margin"]
                ),
            }
        )
    if normalized_loss_weights["counterfactual_utility"] > 0.0:
        transition_candidate_training_config.update(
            {
                "counterfactual_utility_loss_weight": float(
                    normalized_loss_weights["counterfactual_utility"]
                ),
                "counterfactual_gain_margin": counterfactual_gain_margin,
                "counterfactual_safety_tolerance": counterfactual_safety_tolerance,
                "counterfactual_gain_weight": counterfactual_gain_weight,
                "counterfactual_safety_weight": counterfactual_safety_weight,
                "counterfactual_warmup_fraction": counterfactual_warmup_fraction,
                "counterfactual_history_margin": counterfactual_history_margin,
                "counterfactual_training_objective": "safe_local_candidate_rank_v1",
                "safe_memory_residual_bound": safe_memory_residual_bound,
                "safe_local_candidate_sizes": list(safe_local_candidate_sizes),
            }
        )
    else:
        transition_candidate_training_config["memory_utility_calibration_owner"] = "stage4_cmc"
    if normalized_loss_weights["counterfactual_history"] > 0.0:
        transition_candidate_training_config.update(
            {
                "counterfactual_history_loss_weight": float(
                    normalized_loss_weights["counterfactual_history"]
                ),
                "counterfactual_history_margin": counterfactual_history_margin,
                "counterfactual_history_objective": "true_history_over_hard_cross_trajectory_mismatch_v2",
                "counterfactual_history_donor_matching": (
                    "same_benchmark_prefix_length_current_skill_different_trajectory_and_next_skill_then_candidate_overlap_and_clean_state_similarity"
                ),
            }
        )
    if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
        transition_objective = (
            (
                "full_pool_causal_next_skill_with_counterfactual_history"
                if normalized_loss_weights["counterfactual_history"] > 0.0
                else "full_pool_causal_next_skill_with_counterfactual_utility"
                if normalized_loss_weights["counterfactual_utility"] > 0.0
                else "full_pool_causal_next_skill_nll"
            )
            if next_skill_pool_mode == "full_pool"
            else "causal_post_action_unified_retrieval_plus_optional_next_belief_cosine"
        )
        transition_input_semantics = {
            "route_scorer": route_scorer,
            "score_function": "unified_route_logits(h_{t+1}, m_{t+1})",
            "static_memory": "initial_belief(h_{t+1})",
            "dynamic_memory": "post_action_update_of_replayed_or_initial_m_t_for_every_causal_stage2_row",
            "action_channel": "actual_action_text_embedding_updates_m_{t+1}_for_every_causal_stage2_row",
            "observation_channel": "next_observation_text_embedding_updates_m_{t+1}_for_every_causal_stage2_row",
            "next_state_channel": "next_state_text_embedding_is_h_{t+1}_for_routing_and_observation_correction",
            "post_action_update": "required_for_every_causal_stage2_next_skill_ce_row",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": True,
            "action_text_used_as_observation": False,
            "transition_scoring_mode": effective_transition_scoring_mode,
            "transition_residual_lambda": effective_transition_residual_lambda,
            "transition_prior_branch": "disabled_as_main_score_in_unified_memory",
            "transition_residual_branch": "disabled_as_main_score_in_unified_memory",
        }
    elif transition_scoring_mode == V4_1B_TRANSITION_SCORING_MODE:
        transition_input_semantics = {
            "compatibility_mode": "v4_1b_conservative_rollback",
            "action_channel": "not_separate_for_transition_skill_ce",
            "observation_channel": "action_text_embedding_as_legacy_observation_proxy",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": False,
            "action_text_used_as_observation": True,
            "fallback_action_channel": "next_observation_embedding_when_action_text_missing",
            "transition_scoring_mode": transition_scoring_mode,
            "transition_residual_lambda": transition_residual_lambda,
            "transition_prior_branch": "disabled_in_this_mode",
            "transition_residual_branch": "disabled_in_this_mode",
        }
    elif transition_scoring_mode == STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE:
        transition_prior_branch = (
            "stage0_candidate_score_prior_rank_std_calibrated_when_scores_available_else_rank_prior"
            if stage0_score_prior_calibration == "rank_std"
            else "stage0_candidate_score_prior_when_scores_available_else_rank_prior"
            if stage0_score_prior_calibration in {"raw", "none"}
            else "stage0_candidate_rank_prior"
        )
        transition_input_semantics = {
            "action_channel": "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding",
            "observation_channel": "next_observation_text_embedding",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": True,
            "action_text_used_as_observation": False,
            "fallback_action_channel": "current_skill_embedding_when_action_text_missing",
            "transition_scoring_mode": transition_scoring_mode,
            "transition_residual_lambda": transition_residual_lambda,
            "transition_prior_branch": transition_prior_branch,
            "transition_residual_branch": "actual_action_text_plus_next_observation",
        }
    elif transition_scoring_mode == GATED_TEMPORAL_TRANSITION_SCORING_MODE:
        transition_input_semantics = {
            "action_channel": "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding",
            "observation_channel": "next_observation_text_embedding",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": True,
            "action_text_used_as_observation": False,
            "fallback_action_channel": "current_skill_embedding_when_action_text_missing",
            "transition_scoring_mode": transition_scoring_mode,
            "transition_residual_lambda": transition_residual_lambda,
            "transition_prior_branch": "stage0_candidate_score_prior_rank_std_calibrated_when_scores_available_else_rank_prior",
            "transition_residual_branch": "set_aware_gated_temporal_residual",
            "candidate_context": "topM_candidate_context_pooling",
            "lambda_control": "state_conditioned_sigmoid_gate_times_lambda_max",
            "prior_preserving_loss": {
                "kl_alpha": gated_temporal_kl_alpha,
                "rank_drop_beta": gated_temporal_rank_drop_beta,
            },
            "freeze_gated_temporal_only": bool(freeze_gated_temporal_only),
        }
    else:
        transition_input_semantics = {
            "action_channel": "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding",
            "observation_channel": "next_observation_text_embedding",
            "uses_actual_action_text_when_available": True,
            "uses_next_observation_as_observation": True,
            "action_text_used_as_observation": False,
            "fallback_action_channel": "current_skill_embedding_when_action_text_missing",
            "transition_scoring_mode": transition_scoring_mode,
            "transition_residual_lambda": transition_residual_lambda,
            "transition_prior_branch": "current_skill_conditioned_neutral_observation",
            "transition_residual_branch": "actual_action_text_plus_next_observation",
        }
    transition_skill_ce_report = {
        "enabled": loss_activation_counts.get("L_trans_skill_ce", 0) > 0,
        "target": "next_skill_id",
        "candidate_pool": transition_candidate_pool,
        "loss": (
            f"{transition_loss_type}(unified_route_logits_over_stage0_candidates, {transition_positive_mode})"
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER and transition_candidate_pool == "stage0_topm_candidate_set"
            else f"{transition_loss_type}(unified_route_logits_over_skills, {transition_positive_mode})"
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
            else f"{transition_loss_type}(prior_scores_over_stage0_candidates, {transition_positive_mode})"
            if transition_candidate_pool == "stage0_topm_candidate_set"
            else f"{transition_loss_type}(prior_scores_over_skills, {transition_positive_mode})"
        ),
        "inventory_mask_mode": transition_inventory_mask_mode,
        "inventory_min_candidates": int(transition_inventory_min_candidates),
        "positive_mode": transition_positive_mode,
        "scoring_mode": effective_transition_scoring_mode,
        "residual_lambda": effective_transition_residual_lambda,
        "route_scorer": route_scorer,
        "stage0_score_prior_calibration": stage0_score_prior_calibration,
        "observation_cosine_role": "optional_auxiliary_not_primary",
        "input_semantics": transition_input_semantics,
    }
    checkpoint_state, checkpoint_state_report = _checkpoint_state_dict(
        model,
        exclude_frozen_qwen_backbone=exclude_frozen_encoder_backbone,
        exclude_frozen_routing_foundation=bool(
            freeze_audit["frozen_routing_foundation"]
        ),
    )
    payload = {
        "stage": stage_name,
        "step": final_step,
        "config": model_config,
        "training_objective": training_objective,
        "transition_objective": transition_objective,
        "transition_input_semantics": transition_input_semantics,
        "belief_memory_semantics": belief_memory_semantics,
        "policy_feature_semantics": policy_feature_semantics,
        "transition_skill_ce": transition_skill_ce_report,
        "route_scorer": route_scorer,
        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
        "training_regime": "offline_replay_supervised_pretraining",
        "training_data": "CLSTR full-base registry train_allowed rows",
        "include_available_actions_in_state": bool(include_available_actions_in_state),
        "available_actions_planner_context": available_actions_context_report,
        "auto_replay_prefix": auto_replay_prefix_report,
        "causal_next_state": causal_next_state_report,
        "history_channel": replay_history_channel_report,
        "trainable_replay_prefix": bool(trainable_replay_prefix),
        "train_split_filter": train_split_filter_report,
        "benchmark_filter": benchmark_filter_report,
        "benchmark_caps": benchmark_caps_report,
        "max_rows_filter": max_rows_filter_report,
        "qdoc_adapter_used": False,
        "frozen_routing_foundation": freeze_audit["frozen_routing_foundation"],
        "routing_foundation_trainable_for_retrieval": bool(
            anchored_routing_foundation
        ),
        "qwen_and_skill_table_frozen": freeze_audit[
            "qwen_and_skill_table_frozen"
        ],
        "on_policy_rollout_used": on_policy_rollout_used,
        "offline_rollout_source_rows_used": offline_rollout_source_rows_used,
        "offline_rollout_source_counts": offline_rollout_source_counts,
        "offline_rollout_source_note": offline_rollout_source_note,
        "grpo_policy_loss_used": False,
        "clstr_native_act_trained": clstr_native_act_trained,
        "not_full_clstr_act": not clstr_native_act_trained,
        "policy_head_type": policy_head_type,
        "native_clstr_heads_used": native_policy_head,
        "legacy_universal_action_adapter_role": legacy_universal_action_adapter_role,
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "universal_action_adapter_state_dict": None if native_policy_head else action_adapter.state_dict(),
        "trainable_modules": freeze_audit["trainable_modules"],
        "frozen_modules": freeze_audit["frozen_modules"],
        "loss_weights": reported_loss_weights,
        "stage2_static_route_anchor": stage2_static_route_anchor_report,
        "transition_candidate_training": transition_candidate_training_config,
        "loss_activation_counts": loss_activation_counts,
        "sampling_strategy": sampling_strategy,
        "sampler_seed": int(seed),
        "sampled_loss_activation_counts": {key: sampled_loss_activation_counts.get(key, 0) for key in LOSS_WEIGHT_KEYS},
        "embedding_cache_policy": embedding_cache_policy,
        "full_base_embedding_cache": full_base_embedding_cache_report,
        "policy_embedding_cache": policy_embedding_cache_report,
        "metrics": metric_averages,
        "metric_averages": metric_averages,
        "last_batch_metrics": metrics,
        "routing_init": routing_report,
        "routing_checkpoint": routing_checkpoint_report,
        "stage0_candidate_handoff_subset": stage0_candidate_handoff_subset_report,
        "stage0_candidate_handoff": stage0_candidate_handoff_report,
        "resume": resume_report,
        "setup_status_path": str(setup_status),
        **monitor.paths_report(),
        **checkpoint_state_report,
        **external_encoder_metadata,
    }
    torch.save(payload, checkpoint_path)
    monitor.save_latest(payload)
    report = {
        "status": "ok",
        "stage": stage_name,
        "training_objective": training_objective,
        "transition_objective": transition_objective,
        "transition_input_semantics": transition_input_semantics,
        "belief_memory_semantics": belief_memory_semantics,
        "policy_feature_semantics": policy_feature_semantics,
        "transition_skill_ce": transition_skill_ce_report,
        "route_scorer": route_scorer,
        "uses_stage0_prior_at_inference": route_scorer != UNIFIED_MEMORY_ROUTE_SCORER,
        "training_regime": "offline_replay_supervised_pretraining",
        "checkpoint": str(checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "skill_text_format": skill_text_format_report["skill_text_format"],
        "skill_text_format_report": skill_text_format_report,
        "include_available_actions_in_state": bool(include_available_actions_in_state),
        "available_actions_planner_context": available_actions_context_report,
        "auto_replay_prefix": auto_replay_prefix_report,
        "causal_next_state": causal_next_state_report,
        "history_channel": replay_history_channel_report,
        "trainable_replay_prefix": bool(trainable_replay_prefix),
        "train_split_filter": train_split_filter_report,
        "benchmark_filter": benchmark_filter_report,
        "benchmark_caps": benchmark_caps_report,
        "max_rows_filter": max_rows_filter_report,
        "sample_count": len(rows),
        "max_rows": max_rows_filter_report["max_rows"],
        "skill_count": len(skills),
        "benchmark_counts": benchmark_counts,
        "bucket_counts": bucket_counts,
        "loss_activation_counts": loss_activation_counts,
        "trainable_modules": freeze_audit["trainable_modules"],
        "frozen_modules": freeze_audit["frozen_modules"],
        "frozen_routing_foundation": freeze_audit["frozen_routing_foundation"],
        "routing_foundation_trainable_for_retrieval": bool(
            anchored_routing_foundation
        ),
        "qwen_and_skill_table_frozen": freeze_audit[
            "qwen_and_skill_table_frozen"
        ],
        "qdoc_adapter_used": False,
        "valid_or_test_used_for_training": valid_or_test_used,
        "on_policy_rollout_used": on_policy_rollout_used,
        "offline_rollout_source_rows_used": offline_rollout_source_rows_used,
        "offline_rollout_source_counts": offline_rollout_source_counts,
        "offline_rollout_source_note": offline_rollout_source_note,
        "grpo_policy_loss_used": False,
        "clstr_native_act_trained": clstr_native_act_trained,
        "not_full_clstr_act": not clstr_native_act_trained,
        "policy_head_type": policy_head_type,
        "native_clstr_heads_used": native_policy_head,
        "legacy_universal_action_adapter_role": legacy_universal_action_adapter_role,
        "leakage_exclusions": {
            "alfworld_valid_or_test": "excluded_by_registry",
            "webshop_test_split": "eval_only_not_train_allowed",
            "synthetic_as_official_replay": "disallowed",
        },
        "loss_weights": reported_loss_weights,
        "stage2_static_route_anchor": stage2_static_route_anchor_report,
        "transition_candidate_training": transition_candidate_training_config,
        "retrieval_loss": {
            "type": "contrastive",
            "num_negatives": int(retrieval_num_negatives),
            "hard_ratio": float(retrieval_hard_ratio),
            "positive_excluded_from_negatives": True,
            "legacy_routing_ce_loss_weight": 0.0,
            "routing_foundation_frozen": freeze_audit["frozen_routing_foundation"],
            "trainable_effect": (
                "static_kl_anchored_causal_next_skill_training"
                if anchored_routing_foundation
                else "monitor_only_no_encoder_or_skill_table_gradient"
            ),
            "claim_scope": (
                "anchored_stage2_routing_foundation"
                if anchored_routing_foundation
                else "diagnostic_not_retrieval_improvement_evidence"
            ),
        },
        "sampling_strategy": sampling_strategy,
        "sampler_seed": int(seed),
        "sampled_loss_activation_counts": {key: sampled_loss_activation_counts.get(key, 0) for key in LOSS_WEIGHT_KEYS},
        "embedding_cache_policy": embedding_cache_policy,
        "full_base_embedding_cache": full_base_embedding_cache_report,
        "policy_embedding_cache": policy_embedding_cache_report,
        "max_steps": max_steps,
        "trained_steps_this_run": trained_steps_this_run,
        "final_step": final_step,
        "resume": resume_report,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "metrics": metric_averages,
        "metric_averages": metric_averages,
        "last_batch_metrics": metrics,
        "routing_init": routing_report,
        "routing_checkpoint": routing_checkpoint_report,
        "stage0_candidate_handoff_subset": stage0_candidate_handoff_subset_report,
        "stage0_candidate_handoff": stage0_candidate_handoff_report,
        "gpu": _gpu_report(device),
        "not_rl_fine_tuning": True,
        "setup_status_path": str(setup_status),
        **monitor.paths_report(),
        **checkpoint_state_report,
        **external_encoder_metadata,
    }
    write_json(output_dir / "train_report.json", report)
    return report


def train_clstr_stage1_heads_with_model(
    model: Any,
    model_config: dict[str, Any],
    routing_report: dict[str, Any],
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 3000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    transition_hard_negative_margin: float = 1.0,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    routing_checkpoint_path: str | Path | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    max_rows: int | None = None,
    stage0_top_m: int | None = 350,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = None,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    stage0_handoff_cache_mode: str = "auto",
    stage0_handoff_cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    stage0_handoff_cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    stage0_handoff_cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    transition_inventory_mask_mode: str = "auto",
    transition_inventory_min_candidates: int = 64,
    transition_loss_type: str = "listwise_nll",
    transition_positive_mode: str = "gold_plus_equivalent",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
    freeze_gated_temporal_only: bool = False,
    setup_status_path: str | Path | None = None,
    sampling_strategy: str = "balanced_random",
    auto_replay_prefix_max_steps: int = 0,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    stage1_loss_weights = canonical_stage_loss_weights("stage1")
    if loss_weights is not None:
        stage1_loss_weights.update(loss_weights)
    return train_clstr_full_base_with_model(
        model=model,
        model_config=model_config,
        routing_report={
            **routing_report,
            "stage1_role": "same_clstr_model_heads_initialization_on_frozen_stage0_topm",
        },
        train_path=train_path,
        skills_path=skills_path,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_weights=stage1_loss_weights,
        retrieval_num_negatives=retrieval_num_negatives,
        retrieval_hard_ratio=retrieval_hard_ratio,
        transition_hard_negative_margin=transition_hard_negative_margin,
        include_available_actions_in_state=include_available_actions_in_state,
        skill_text_format=skill_text_format,
        routing_checkpoint_path=routing_checkpoint_path,
        embedding_cache_mode=embedding_cache_mode,
        embedding_cache_max_rows=embedding_cache_max_rows,
        allowed_benchmarks=allowed_benchmarks,
        benchmark_caps=benchmark_caps,
        max_rows=max_rows,
        stage0_top_m=stage0_top_m,
        stage0_positive_missing_policy=stage0_positive_missing_policy,
        stage0_handoff_query_mode=stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=stage0_handoff_sample_multiplier,
        stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
        stage0_handoff_cache_mode=stage0_handoff_cache_mode,
        stage0_handoff_cache_dir=stage0_handoff_cache_dir,
        stage0_handoff_cache_format=stage0_handoff_cache_format,
        stage0_handoff_cache_shard_size=stage0_handoff_cache_shard_size,
        transition_inventory_mask_mode=transition_inventory_mask_mode,
        transition_inventory_min_candidates=transition_inventory_min_candidates,
        transition_loss_type=transition_loss_type,
        transition_positive_mode=transition_positive_mode,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        gated_temporal_lambda_max=gated_temporal_lambda_max,
        gated_temporal_kl_alpha=gated_temporal_kl_alpha,
        gated_temporal_rank_drop_beta=gated_temporal_rank_drop_beta,
        gated_temporal_context_top_k=gated_temporal_context_top_k,
        stage0_score_prior_calibration=stage0_score_prior_calibration,
        freeze_gated_temporal_only=freeze_gated_temporal_only,
        setup_status_path=setup_status_path,
        allow_full_pool_stage2_debug=False,
        stage_name="clstr_stage1_heads_init",
        checkpoint_prefix="clstr_stage1_heads",
        training_objective="stage1_heads_init_topm_supervised",
        sampling_strategy=sampling_strategy,
        auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
        trainable_replay_prefix=trainable_replay_prefix,
        route_scorer=route_scorer,
        next_skill_pool_mode="stage0_candidates",
        resume_checkpoint_path=resume_checkpoint_path,
    )


def run_clstr_stage1_heads_init(
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    routing_checkpoint_path: str | Path,
    max_steps: int = 3000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    transition_hard_negative_margin: float = 1.0,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    max_rows: int | None = None,
    stage0_top_m: int | None = 350,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = None,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    stage0_handoff_cache_mode: str = "auto",
    stage0_handoff_cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    stage0_handoff_cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    stage0_handoff_cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    transition_inventory_mask_mode: str = "auto",
    transition_inventory_min_candidates: int = 64,
    transition_loss_type: str = "listwise_nll",
    transition_positive_mode: str = "gold_plus_equivalent",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
    freeze_gated_temporal_only: bool = False,
    sampling_strategy: str = "balanced_random",
    auto_replay_prefix_max_steps: int = 0,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=routing_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=Path(output_dir) / "model_cache",
    )
    return train_clstr_stage1_heads_with_model(
        model=model,
        model_config=model_config,
        routing_report=routing_report,
        train_path=train_path,
        skills_path=skills_path,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_weights=loss_weights,
        retrieval_num_negatives=retrieval_num_negatives,
        retrieval_hard_ratio=retrieval_hard_ratio,
        transition_hard_negative_margin=transition_hard_negative_margin,
        include_available_actions_in_state=include_available_actions_in_state,
        skill_text_format=skill_text_format,
        routing_checkpoint_path=routing_checkpoint_path,
        embedding_cache_mode=embedding_cache_mode,
        embedding_cache_max_rows=embedding_cache_max_rows,
        allowed_benchmarks=allowed_benchmarks,
        benchmark_caps=benchmark_caps,
        max_rows=max_rows,
        stage0_top_m=stage0_top_m,
        stage0_positive_missing_policy=stage0_positive_missing_policy,
        stage0_handoff_query_mode=stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=stage0_handoff_sample_multiplier,
        stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
        stage0_handoff_cache_mode=stage0_handoff_cache_mode,
        stage0_handoff_cache_dir=stage0_handoff_cache_dir,
        stage0_handoff_cache_format=stage0_handoff_cache_format,
        stage0_handoff_cache_shard_size=stage0_handoff_cache_shard_size,
        transition_inventory_mask_mode=transition_inventory_mask_mode,
        transition_inventory_min_candidates=transition_inventory_min_candidates,
        transition_loss_type=transition_loss_type,
        transition_positive_mode=transition_positive_mode,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        gated_temporal_lambda_max=gated_temporal_lambda_max,
        gated_temporal_kl_alpha=gated_temporal_kl_alpha,
        gated_temporal_rank_drop_beta=gated_temporal_rank_drop_beta,
        gated_temporal_context_top_k=gated_temporal_context_top_k,
        stage0_score_prior_calibration=stage0_score_prior_calibration,
        freeze_gated_temporal_only=freeze_gated_temporal_only,
        sampling_strategy=sampling_strategy,
        auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
        trainable_replay_prefix=trainable_replay_prefix,
        route_scorer=route_scorer,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def run_clstr_full_base_train(
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    routing_init_manifest: str | Path | None = None,
    max_steps: int = 1000,
    target_total_steps: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    minimum_learning_rate: float | None = None,
    checkpoint_interval_steps: int = 400,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    transition_real_candidate_ce_multiplier: float = 1.0,
    transition_injected_candidate_ce_multiplier: float = 1.0,
    transition_hard_negative_margin: float = 1.0,
    transition_inventory_mask_mode: str = "off",
    transition_inventory_min_candidates: int = 0,
    transition_loss_type: str = "cross_entropy",
    transition_positive_mode: str = "single",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = STAGE0_SCORE_PRIOR_CALIBRATION,
    freeze_gated_temporal_only: bool = False,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    routing_checkpoint_path: str | Path | None = None,
    stage1_checkpoint_path: str | Path | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    max_rows: int | None = None,
    stage0_top_m: int | None = None,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = None,
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    stage0_handoff_cache_mode: str = "auto",
    stage0_handoff_cache_dir: str | Path = DEFAULT_STAGE0_HANDOFF_CACHE_DIR,
    stage0_handoff_cache_format: str = LEGACY_STAGE0_HANDOFF_CACHE_FORMAT,
    stage0_handoff_cache_shard_size: int = DEFAULT_STAGE0_HANDOFF_CACHE_SHARD_SIZE,
    allow_full_pool_stage2_debug: bool = False,
    sampling_strategy: str = "balanced_deterministic",
    auto_replay_prefix_max_steps: int = 0,
    trainable_replay_prefix: bool = False,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    anchored_routing_foundation: bool = False,
    static_route_anchor_weight: float = 0.1,
    static_route_anchor_max_regression: float = 0.005,
    next_skill_pool_mode: str = "stage0_candidates",
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_warmup_fraction: float = 0.1,
    counterfactual_history_margin: float = 0.1,
    safe_memory_residual_bound: float = DEFAULT_SAFE_MEMORY_RESIDUAL_BOUND,
    safe_local_candidate_sizes: tuple[int, ...] = SAFE_LOCAL_CANDIDATE_SIZES,
    warm_start_checkpoint_path: str | Path | None = None,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if routing_init_manifest is not None:
        raise ValueError(
            "canonical Stage2 requires routing_checkpoint_path from the gated "
            "Stage0 checkpoint; run_clstr_full_base_train does not accept "
            "routing_init_manifest. Use "
            "run_legacy_clstr_full_base_train_from_routing_init for old "
            "routing_init_manifest experiments."
        )
    if routing_checkpoint_path is None:
        raise ValueError("Stage2 canonical training requires routing_checkpoint_path from the gated Stage0 checkpoint")
    if stage1_checkpoint_path is None:
        raise ValueError("Stage2 canonical training requires stage1_checkpoint_path from Stage1 heads initialization")
    if warm_start_checkpoint_path is not None and resume_checkpoint_path is not None:
        raise ValueError("warm-start and exact resume checkpoints are mutually exclusive")
    loss_weights = canonical_stage_loss_weights("stage2", loss_weights)
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=routing_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=Path(output_dir) / "model_cache",
    )
    stage1_initialization_report: dict[str, Any] = {"enabled": False}
    if stage1_checkpoint_path is not None:
        stage1_initialization_report = {
            "enabled": True,
            "role": "continue_same_clstr_model_from_stage1_heads_init",
            **load_head_checkpoint_into_model(
                model,
                head_checkpoint_path=stage1_checkpoint_path,
                partial_load_mode="stage1_checkpoint_compatible_state",
            ),
        }
        routing_report = {
            **routing_report,
            "stage1_initialization": stage1_initialization_report,
        }
    return train_clstr_full_base_with_model(
        model=model,
        model_config=model_config,
        routing_report=routing_report,
        train_path=train_path,
        skills_path=skills_path,
        output_dir=output_dir,
        max_steps=max_steps,
        target_total_steps=target_total_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        checkpoint_interval_steps=checkpoint_interval_steps,
        seed=seed,
        loss_weights=loss_weights,
        retrieval_num_negatives=retrieval_num_negatives,
        retrieval_hard_ratio=retrieval_hard_ratio,
        transition_real_candidate_ce_multiplier=transition_real_candidate_ce_multiplier,
        transition_injected_candidate_ce_multiplier=transition_injected_candidate_ce_multiplier,
        transition_hard_negative_margin=transition_hard_negative_margin,
        transition_inventory_mask_mode=transition_inventory_mask_mode,
        transition_inventory_min_candidates=transition_inventory_min_candidates,
        transition_loss_type=transition_loss_type,
        transition_positive_mode=transition_positive_mode,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        gated_temporal_lambda_max=gated_temporal_lambda_max,
        gated_temporal_kl_alpha=gated_temporal_kl_alpha,
        gated_temporal_rank_drop_beta=gated_temporal_rank_drop_beta,
        gated_temporal_context_top_k=gated_temporal_context_top_k,
        stage0_score_prior_calibration=stage0_score_prior_calibration,
        freeze_gated_temporal_only=freeze_gated_temporal_only,
        include_available_actions_in_state=include_available_actions_in_state,
        skill_text_format=skill_text_format,
        routing_checkpoint_path=routing_checkpoint_path,
        embedding_cache_mode=embedding_cache_mode,
        embedding_cache_max_rows=embedding_cache_max_rows,
        allowed_benchmarks=allowed_benchmarks,
        benchmark_caps=benchmark_caps,
        max_rows=max_rows,
        stage0_top_m=stage0_top_m,
        stage0_positive_missing_policy=stage0_positive_missing_policy,
        stage0_handoff_query_mode=stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=stage0_handoff_sample_multiplier,
        stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
        stage0_handoff_cache_mode=stage0_handoff_cache_mode,
        stage0_handoff_cache_dir=stage0_handoff_cache_dir,
        stage0_handoff_cache_format=stage0_handoff_cache_format,
        stage0_handoff_cache_shard_size=stage0_handoff_cache_shard_size,
        allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
        sampling_strategy=sampling_strategy,
        auto_replay_prefix_max_steps=auto_replay_prefix_max_steps,
        trainable_replay_prefix=trainable_replay_prefix,
        route_scorer=route_scorer,
        anchored_routing_foundation=anchored_routing_foundation,
        static_route_anchor_weight=static_route_anchor_weight,
        static_route_anchor_max_regression=static_route_anchor_max_regression,
        next_skill_pool_mode=next_skill_pool_mode,
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
        counterfactual_warmup_fraction=counterfactual_warmup_fraction,
        counterfactual_history_margin=counterfactual_history_margin,
        safe_memory_residual_bound=safe_memory_residual_bound,
        safe_local_candidate_sizes=safe_local_candidate_sizes,
        warm_start_checkpoint_path=warm_start_checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )


def run_legacy_clstr_full_base_train_from_routing_init(
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    routing_init_manifest: str | Path,
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    max_rows: int | None = None,
    allow_full_pool_stage2_debug: bool = True,
) -> dict[str, Any]:
    skills = _read_jsonl(skills_path)
    model, model_config, routing_report = _build_model_from_routing_init(
        routing_init_manifest,
        skills,
        Path(output_dir) / "model_cache",
    )
    routing_report = {
        **routing_report,
        "legacy_routing_init_manifest_used": True,
        "not_canonical_stage2_unified_mainline": True,
    }
    report = train_clstr_full_base_with_model(
        model=model,
        model_config=model_config,
        routing_report=routing_report,
        train_path=train_path,
        skills_path=skills_path,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_weights=loss_weights,
        retrieval_num_negatives=retrieval_num_negatives,
        retrieval_hard_ratio=retrieval_hard_ratio,
        include_available_actions_in_state=include_available_actions_in_state,
        skill_text_format=skill_text_format,
        routing_checkpoint_path=None,
        embedding_cache_mode=embedding_cache_mode,
        embedding_cache_max_rows=embedding_cache_max_rows,
        allowed_benchmarks=allowed_benchmarks,
        max_rows=max_rows,
        stage0_top_m=None,
        stage0_positive_missing_policy="skip",
        allow_full_pool_stage2_debug=allow_full_pool_stage2_debug,
    )
    report["legacy_routing_init_manifest_used"] = True
    report["not_canonical_stage2_unified_mainline"] = True
    write_json(Path(output_dir) / "train_report.json", report)
    return report
