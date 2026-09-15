from __future__ import annotations

import hashlib
import json
import math
import re
import time
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from clstr.full_base_train import (
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    ROUTE_SCORERS,
    TRANSITION_SCORING_MODE,
    UNIFIED_MEMORY_ROUTE_SCORER,
    _attach_adjacent_next_states,
    _attach_auto_replay_prefixes,
    _attach_stage0_topm_candidates,
    _cap_rows_by_benchmark,
    _auto_replay_prefix_step,
)
from clstr.current_state_route_eval import (
    _build_current_state_route_batch,
    _compute_current_state_route_loss,
)
from clstr.history_channel import router_state_text
from clstr.memory_candidate_recall import (
    CANDIDATE_RECALL_MODES,
    CANDIDATE_SELECTION_VERSION,
    CANDIDATE_UNION_VERSION,
    row_positive_skill_ids,
)
from clstr.memory_utility_gate import (
    HEURISTIC_ALPHA_VERSION,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    RELIABILITY_MODES,
)
from clstr.memory_utility_records import (
    build_memory_utility_route_records,
    persist_memory_utility_route_records,
    validate_memory_utility_route_record_target,
)
from clstr.logged_online_trajectory import read_jsonl_stream
from clstr.stage4_act_train import (
    _as_stage4_handoff_row,
    _build_stage4_next_skill_rows_from_source_rows,
    _compute_stage4_act_loss,
    _candidate_prior_scores_from_stage0_row,
    _eligible_stage4_source_rows,
    _ensure_stage4_score_calibrator,
    _freeze_for_stage4_act,
    _stable_negative_fill,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl_stream(path))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _resolve_logged_online_device(model: Any, device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    elif torch.cuda.is_available() and callable(getattr(model, "to", None)):
        resolved = torch.device("cuda")
    else:
        resolved = torch.device(getattr(model, "device", "cpu"))
    if callable(getattr(model, "to", None)):
        model.to(resolved)
    return resolved


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or "").strip()


def _first_string(values: Any) -> str:
    if isinstance(values, list) and values:
        return str(values[0] or "").strip()
    if isinstance(values, tuple) and values:
        return str(values[0] or "").strip()
    return str(values or "").strip()


def _unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _row_benchmark(row: dict[str, Any]) -> str:
    return str(row.get("source_benchmark") or row.get("benchmark") or "unknown")


def _benchmark_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_row_benchmark(row) for row in rows).items()))


def _cap_stage4_rows_by_benchmark(
    rows: list[dict[str, Any]],
    benchmark_caps: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("benchmark"):
            normalized_rows.append(row)
        else:
            copied = dict(row)
            copied["benchmark"] = _row_benchmark(row)
            normalized_rows.append(copied)
    capped, report = _cap_rows_by_benchmark(normalized_rows, benchmark_caps)
    for row in capped:
        row.pop("benchmark", None) if "source_benchmark" in row else None
    return capped, report


def _full_state_text_from_logged_step(step: dict[str, Any]) -> str:
    explicit_full = str(step.get("state_text_full") or "").strip()
    if explicit_full:
        return explicit_full
    direct = str(step.get("state_text") or "").strip()
    if direct:
        history = str(step.get("history_text") or "").strip()
        if history and not any(
            line.strip().lower().startswith("history:")
            for line in direct.splitlines()
        ):
            return f"{direct}\nhistory: {history}"
        return direct
    parts: list[str] = []
    goal = str(step.get("goal") or step.get("goal_text") or step.get("task_text") or "").strip()
    history = str(step.get("history_text") or "").strip()
    observation = str(step.get("observation_text") or step.get("state") or "").strip()
    if goal:
        parts.append(f"goal: {goal}")
    if history:
        parts.append(f"history: {history}")
    if observation:
        parts.append(f"observation: {observation}")
    return "\n".join(parts)


def _state_text_from_logged_step(step: dict[str, Any]) -> str:
    return router_state_text({"state_text": _full_state_text_from_logged_step(step)})


def build_logged_online_stage4_rows(
    logged_steps: Iterable[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    candidate_count: int | None = None,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    positive_missing_policy: str = "skip",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert normalized logged-online steps into Stage4 ACT rows.

    The output format is intentionally the same as Stage4 ACT rows used by
    `clstr.stage4_act_train`, so the online simulator can reuse the exact
    next-skill ranking loss instead of defining a new objective.
    """

    if positive_missing_policy not in {"skip", "inject"}:
        raise ValueError(f"unsupported positive_missing_policy: {positive_missing_policy}")
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    source_steps = 0
    positive_injected_rows = 0
    logged_candidate_rows = 0
    stable_fill_rows = 0
    skill_ids = list(skill_id_to_idx)

    for step in logged_steps:
        if not isinstance(step, dict):
            continue
        source_steps += 1
        benchmark = str(step.get("benchmark") or "unknown")
        if allowed_benchmarks is not None and benchmark not in allowed_benchmarks:
            skipped["benchmark_not_allowed"] += 1
            continue
        current_skill_id = _first_string(step.get("gt_skill_ids") or step.get("skill_id"))
        next_skill_id = _first_string(step.get("gt_next_skill_ids") or step.get("next_skill_id"))
        declared_positive_ids = row_positive_skill_ids(
            {
                "next_skill_id": next_skill_id,
                "positive_next_skill_ids": step.get("gt_next_skill_ids"),
                "equivalent_next_skill_ids": step.get("equivalent_next_skill_ids"),
            }
        )
        in_pool_positive_ids = [
            skill_id for skill_id in declared_positive_ids if skill_id in skill_id_to_idx
        ]
        if not current_skill_id:
            skipped["missing_current_skill"] += 1
            continue
        if not next_skill_id:
            skipped["missing_next_skill"] += 1
            continue
        if current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if not in_pool_positive_ids:
            skipped["next_skill_not_in_pool"] += 1
            continue

        raw_candidates = _unique_strings(step.get("candidate_skill_ids") or step.get("candidate_next_skill_ids") or [])
        candidate_ids = [skill_id for skill_id in raw_candidates if skill_id in skill_id_to_idx]
        positive_injected = False
        if candidate_ids:
            logged_candidate_rows += 1
            if not any(skill_id in candidate_ids for skill_id in in_pool_positive_ids):
                if positive_missing_policy == "skip":
                    skipped["next_positive_missing_from_candidates"] += 1
                    continue
                candidate_ids.append(in_pool_positive_ids[0])
                positive_injected = True
        else:
            if candidate_count is None:
                skipped["empty_candidates"] += 1
                continue
            candidate_ids = [in_pool_positive_ids[0]]
            positive_injected = True
            stable_fill_rows += 1

        before_fill = len(candidate_ids)
        candidate_ids = _stable_negative_fill(
            seed_text=str(step.get("task_id") or step.get("trajectory_id") or source_steps),
            existing=candidate_ids,
            skill_ids=skill_ids,
            candidate_count=candidate_count,
        )
        if len(candidate_ids) > before_fill:
            stable_fill_rows += 1
        if not candidate_ids:
            skipped["empty_candidates"] += 1
            continue
        ranked_positives = sorted(
            (
                (candidate_ids.index(skill_id), skill_id)
                for skill_id in in_pool_positive_ids
                if skill_id in candidate_ids
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked_positives:
            skipped["next_positive_missing_from_candidates"] += 1
            continue

        positive_positions = [int(position) for position, _skill_id in ranked_positives]
        positive_ids = [skill_id for _position, skill_id in ranked_positives]
        positive_indices = [int(skill_id_to_idx[skill_id]) for skill_id in positive_ids]
        positive_pos = positive_positions[0]
        if positive_injected:
            positive_injected_rows += 1
        candidate_indices = [int(skill_id_to_idx[skill_id]) for skill_id in candidate_ids]
        state_text_current = _state_text_from_logged_step(step)
        row = {
            "task_id": step.get("task_id"),
            "trajectory_id": step.get("trajectory_id"),
            "step_index": step.get("step_idx", step.get("step_index")),
            "benchmark": benchmark,
            "source_benchmark": benchmark,
            "state_text": state_text_current,
            "state_text_current": state_text_current,
            "state_text_full": _full_state_text_from_logged_step(step),
            "history_text": str(step.get("history_text") or ""),
            "action_text": str(step.get("action_text") or ""),
            "next_observation_text": str(step.get("next_observation_text") or ""),
            "observation_source": str(step.get("observation_source") or "logged_environment_result"),
            "skill_id": current_skill_id,
            "next_skill_id": next_skill_id,
            "equivalent_next_skill_ids": [
                skill_id for skill_id in declared_positive_ids if skill_id != next_skill_id
            ],
            "skill_idx": int(skill_id_to_idx[current_skill_id]),
            "positive_next_skill_ids": positive_ids,
            "positive_next_skill_indices": positive_indices,
            "positive_next_skill_positions": positive_positions,
            "positive_next_skill_idx": positive_indices[0],
            "positive_next_skill_position": int(positive_pos),
            "candidate_next_skill_ids": candidate_ids,
            "candidate_next_skill_indices": candidate_indices,
            "candidate_next_prior_scores": _candidate_prior_scores_from_stage0_row(step, candidate_indices),
            "positive_injected": bool(positive_injected),
            "provenance": {
                "source": "executor_free_logged_online_stage4",
                "reward_type": step.get("reward_type", "logged_gt"),
                "source_quality": step.get("source_quality"),
            },
        }
        for inventory_key in _INVENTORY_SKILL_ID_KEYS:
            if step.get(inventory_key):
                row[inventory_key] = step.get(inventory_key)
        rows.append(row)
        benchmark_counts[benchmark] += 1
        if benchmark_caps is None and max_rows is not None and len(rows) >= int(max_rows):
            break

    rows, benchmark_caps_report = _cap_stage4_rows_by_benchmark(rows, benchmark_caps)
    if max_rows is not None and len(rows) > int(max_rows):
        rows = rows[: int(max_rows)]
    benchmark_counts = Counter(_row_benchmark(row) for row in rows)

    if logged_candidate_rows and stable_fill_rows:
        candidate_source = "mixed_logged_candidates_and_stable_fill"
    elif logged_candidate_rows:
        candidate_source = "logged_candidates"
    elif stable_fill_rows:
        candidate_source = "stable_negative_fill"
    else:
        candidate_source = "none"
    return rows, {
        "source_steps": source_steps,
        "stage4_rows": len(rows),
        "candidate_source": candidate_source,
        "candidate_count_requested": candidate_count,
        "positive_missing_policy": positive_missing_policy,
        "positive_injected_rows": positive_injected_rows,
        "logged_candidate_rows": logged_candidate_rows,
        "stable_fill_rows": stable_fill_rows,
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "benchmark_caps": benchmark_caps_report,
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def build_logged_online_stage4_rows_with_stage0_handoff(
    *,
    model: Any,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    stage0_top_m: int = 350,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 8,
    stage0_candidate_progress_interval_batches: int = 100,
    candidate_count: int | None = None,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    routing_checkpoint_path: str | Path | None = None,
    device: torch.device | str | None = None,
    next_skill_pool_mode: str = "stage0_candidates",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build executor-free Stage4 rows using live Stage0 top-M candidates.

    This is the logged-online equivalent of the Stage4 ACT handoff path: raw
    trajectory rows are filtered into trainable next-skill rows, Stage0 supplies
    the candidate set, and the existing Stage4 row builder produces the final
    ranking examples.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {_skill_id(row): idx for idx, row in enumerate(skills) if _skill_id(row)}
    if not skill_id_to_idx:
        raise ValueError("logged-online Stage4 handoff requires a non-empty skill pool")

    raw_source_rows = list(read_jsonl_stream(trajectories_path))
    prepared_source_rows, causal_next_state_report = _attach_adjacent_next_states(raw_source_rows)
    source_rows, source_report = _eligible_stage4_source_rows(
        prepared_source_rows,
        skill_id_to_idx,
        allowed_benchmarks=allowed_benchmarks,
        max_source_rows=None if benchmark_caps is not None else max_rows,
    )
    source_rows, benchmark_caps_report = _cap_rows_by_benchmark(source_rows, benchmark_caps)
    source_rows_for_handoff = [
        _as_stage4_handoff_row(row, next_skill_pool_mode=next_skill_pool_mode)
        for row in source_rows
    ]

    resolved_device = _resolve_logged_online_device(model, device)

    retained_rows, handoff_report = _attach_stage0_topm_candidates(
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
        device=resolved_device,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    rows, data_report = _build_stage4_next_skill_rows_from_source_rows(
        retained_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
        max_rows=max_rows,
        allowed_benchmarks=allowed_benchmarks,
        stage0_candidate_handoff_report=handoff_report,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    report = {
        **data_report,
        "source_path": str(trajectories_path),
        "skills_path": str(skills_path),
        "source_rows": source_report["source_rows"],
        "eligible_source_rows": source_report["eligible_source_rows"],
        "skipped_benchmarks": source_report["skipped_benchmarks"],
        "skipped_reasons": dict(Counter(source_report["skipped_reasons"]) + Counter(data_report["skipped_reasons"])),
        "benchmark_caps": benchmark_caps_report,
        "stage0_top_m": int(stage0_top_m),
        "stage0_handoff_query_mode": str(stage0_handoff_query_mode),
        "causal_next_state": causal_next_state_report,
    }
    _write_json(output / "logged_online_stage4_rows_report.json", report)
    return rows, report


def _batch_rows(rows: list[dict[str, Any]], start: int, batch_size: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    batch_size = max(1, int(batch_size))
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def _row_step_index(row: dict[str, Any]) -> int:
    value = row.get("step_index", row.get("step_idx", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_rank_bucket(row: dict[str, Any]) -> str:
    value = row.get("positive_next_skill_position")
    try:
        rank = int(value) + 1
    except (TypeError, ValueError):
        return "missing_position"
    if rank <= 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 20:
        return "6-20"
    if rank <= 100:
        return "21-100"
    return ">100"


def _positive_rank_bucket_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(_positive_rank_bucket(row) for row in rows))


def attach_replay_prefixes_from_source_rows(
    target_rows: Iterable[dict[str, Any]],
    *,
    source_rows: Iterable[dict[str, Any]],
    max_steps: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach replay prefixes to target rows from earlier rows in the same trajectory.

    This is used when the eval set is pre-materialized as one row per trajectory
    suffix while a separate full Stage4 row file contains the prefix steps.
    """

    max_steps = max(0, int(max_steps or 0))
    targets = [dict(row) for row in target_rows]
    if max_steps <= 0:
        return targets, {
            "enabled": False,
            "reason": "max_steps_le_0",
            "target_rows": len(targets),
            "source_rows": 0,
            "rows_with_replay_prefix": 0,
            "total_prefix_steps": 0,
            "max_prefix_len": 0,
            "matched_target_rows": 0,
            "missing_trajectory_rows": 0,
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    target_trajectory_ids = {
        str(row.get("trajectory_id") or "").strip()
        for row in targets
        if str(row.get("trajectory_id") or "").strip()
    }
    source_count = 0
    retained_source_rows = 0
    for row in source_rows:
        source_count += 1
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            continue
        if trajectory_id not in target_trajectory_ids:
            continue
        grouped.setdefault(trajectory_id, []).append(dict(row))
        retained_source_rows += 1
    for trajectory_id, rows in list(grouped.items()):
        grouped[trajectory_id] = sorted(rows, key=_row_step_index)

    output: list[dict[str, Any]] = []
    rows_with_prefix = 0
    total_prefix_steps = 0
    max_prefix_len = 0
    matched_target_rows = 0
    missing_trajectory_rows = 0
    for row in targets:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id or trajectory_id not in grouped:
            missing_trajectory_rows += 1
            output.append(row)
            continue
        ordered = grouped[trajectory_id]
        target_step = _row_step_index(row)
        prior_rows = [source_row for source_row in ordered if _row_step_index(source_row) < target_step]
        if not prior_rows:
            matched_target_rows += 1
            output.append(row)
            continue
        prefix_source = prior_rows[-max_steps:]
        prefix: list[dict[str, Any]] = []
        start_position = len(prior_rows) - len(prefix_source)
        for offset, source_row in enumerate(prefix_source):
            source_position = start_position + offset
            next_row = ordered[source_position + 1] if source_position + 1 < len(ordered) else row
            prefix.append(_auto_replay_prefix_step(source_row, next_row=next_row))
        copied = dict(row)
        copied["replay_prefix"] = prefix
        output.append(copied)
        matched_target_rows += 1
        rows_with_prefix += 1
        total_prefix_steps += len(prefix)
        max_prefix_len = max(max_prefix_len, len(prefix))

    return output, {
        "enabled": True,
        "max_steps": max_steps,
        "target_rows": len(targets),
        "source_rows": source_count,
        "retained_source_rows": retained_source_rows,
        "source_trajectory_count": len(grouped),
        "matched_target_rows": matched_target_rows,
        "missing_trajectory_rows": missing_trajectory_rows,
        "rows_with_replay_prefix": rows_with_prefix,
        "rows_without_replay_prefix": len(targets) - rows_with_prefix,
        "total_prefix_steps": total_prefix_steps,
        "max_prefix_len": max_prefix_len,
    }


_STATE_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Keep state_conditioned available as an explicit ablation, but leave it out of
# the default auto gate because small calibration splits showed high variance.
ONLINE_MEMORY_AUTO_VARIANTS = ("latest_exact", "inventory_remaining")
ONLINE_MEMORY_MODES = {"latest_exact", "exact_count", "state_conditioned", "inventory_remaining"}
_INVENTORY_SKILL_ID_KEYS = (
    "visible_inventory_skill_ids",
    "available_skill_ids",
    "available_skills",
    "admissible_skill_ids",
    "skill_inventory_ids",
    "stage0_allowed_skill_ids",
)


def _state_similarity_text(row: dict[str, Any]) -> str:
    parts = [
        router_state_text(row),
        str(row.get("observation_text") or ""),
        str(row.get("next_observation_text") or ""),
    ]
    return " ".join(part for part in parts if part.strip())


def _state_tokens(row: dict[str, Any]) -> set[str]:
    return {token for token in _STATE_TOKEN_RE.findall(_state_similarity_text(row).lower()) if len(token) > 1}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _flatten_inventory_values(value: Any) -> list[str]:
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
            output.extend(_flatten_inventory_values(item))
        return output
    text = str(value).strip()
    return [text] if text else []


def _row_inventory_skill_ids(row: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for key in _INVENTORY_SKILL_ID_KEYS:
        for item in _flatten_inventory_values(row.get(key)):
            if item and item not in seen:
                seen.add(item)
                output.append(item)
    return output


def _trajectory_prefix_signal_report(
    stream_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, int]:
    stream_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in stream_rows:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if trajectory_id:
            stream_by_trajectory.setdefault(trajectory_id, []).append(row)

    same_prefix_rows = 0
    next_skill_seen = 0
    current_skill_seen = 0
    transition_seen = 0
    for row in eval_rows:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            continue
        eval_step = _row_step_index(row)
        prefix = [
            item
            for item in stream_by_trajectory.get(trajectory_id, [])
            if _row_step_index(item) < eval_step
        ]
        if not prefix:
            continue
        same_prefix_rows += 1
        next_skill_id = str(row.get("next_skill_id") or "").strip()
        current_skill_id = str(row.get("skill_id") or "").strip()
        prefix_next_skills = {str(item.get("next_skill_id") or "").strip() for item in prefix}
        prefix_current_skills = {str(item.get("skill_id") or "").strip() for item in prefix}
        prefix_transitions = {
            (str(item.get("skill_id") or "").strip(), str(item.get("next_skill_id") or "").strip())
            for item in prefix
        }
        if next_skill_id and next_skill_id in prefix_next_skills:
            next_skill_seen += 1
        if current_skill_id and current_skill_id in prefix_current_skills:
            current_skill_seen += 1
        if current_skill_id and next_skill_id and (current_skill_id, next_skill_id) in prefix_transitions:
            transition_seen += 1
    return {
        "eval_same_trajectory_prefix_rows": same_prefix_rows,
        "eval_next_skill_seen_in_prefix_count": next_skill_seen,
        "eval_current_skill_seen_in_prefix_count": current_skill_seen,
        "eval_current_to_next_transition_seen_in_prefix_count": transition_seen,
    }


def attach_trajectory_prefix_online_memory_scores(
    target_rows: Iterable[dict[str, Any]],
    *,
    feedback_rows: Iterable[dict[str, Any]],
    next_skill_bonus: float = 0.0,
    exact_transition_bonus: float = 5.0,
    memory_mode: str = "latest_exact",
    state_similarity_threshold: float = 0.2,
    state_similarity_temperature: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach same-trajectory prefix memory scores without using future rows."""

    memory_mode = str(memory_mode or "latest_exact")
    if memory_mode not in ONLINE_MEMORY_MODES:
        raise ValueError(f"unsupported online memory mode: {memory_mode}")
    feedback_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in feedback_rows:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if trajectory_id:
            feedback_by_trajectory.setdefault(trajectory_id, []).append(row)
    inventory_prefix_by_trajectory: dict[str, tuple[list[int], list[frozenset[str]]]] = {}
    if memory_mode == "inventory_remaining":
        for trajectory_id, feedback_items in feedback_by_trajectory.items():
            by_step: dict[int, list[dict[str, Any]]] = {}
            for item in feedback_items:
                by_step.setdefault(_row_step_index(item), []).append(item)
            used_skill_ids: set[str] = set()
            steps: list[int] = []
            used_after_steps: list[frozenset[str]] = []
            for step in sorted(by_step):
                for item in by_step[step]:
                    prefix_current = str(item.get("skill_id") or "").strip()
                    if prefix_current:
                        used_skill_ids.add(prefix_current)
                steps.append(step)
                used_after_steps.append(frozenset(used_skill_ids))
            inventory_prefix_by_trajectory[trajectory_id] = (steps, used_after_steps)

    scored_rows: list[dict[str, Any]] = []
    memory_rows = 0
    positive_boosted_rows = 0
    exact_transition_rows = 0
    state_conditioned_rows = 0
    inventory_remaining_rows = 0
    total_positive_score = 0.0
    for row in target_rows:
        copied = dict(row)
        candidates = [str(item) for item in copied.get("candidate_next_skill_ids") or []]
        scores = [0.0] * len(candidates)
        trajectory_id = str(copied.get("trajectory_id") or "").strip()
        current_skill_id = str(copied.get("skill_id") or "").strip()
        target_step = _row_step_index(copied)
        prefix = [
            item
            for item in feedback_by_trajectory.get(trajectory_id, [])
            if _row_step_index(item) < target_step
        ]
        prefix = sorted(prefix, key=_row_step_index)
        next_skill_counts: Counter[str] = Counter()
        transition_counts: Counter[tuple[str, str]] = Counter()
        latest_exact_next_skill = ""
        target_state_tokens = _state_tokens(copied)
        state_conditioned_scores: Counter[str] = Counter()
        used_skill_ids: set[str] = {current_skill_id} if current_skill_id else set()
        if memory_mode == "inventory_remaining":
            steps, used_after_steps = inventory_prefix_by_trajectory.get(trajectory_id, ([], []))
            prefix_idx = bisect_left(steps, target_step) - 1
            if prefix_idx >= 0:
                used_skill_ids.update(used_after_steps[prefix_idx])
        else:
            for item in prefix:
                prefix_next = str(item.get("next_skill_id") or "").strip()
                prefix_current = str(item.get("skill_id") or "").strip()
                if prefix_current:
                    used_skill_ids.add(prefix_current)
                if prefix_next:
                    next_skill_counts[prefix_next] += 1
                if prefix_current and prefix_next:
                    transition_counts[(prefix_current, prefix_next)] += 1
                    if prefix_current == current_skill_id:
                        latest_exact_next_skill = prefix_next
                        if memory_mode == "state_conditioned":
                            similarity = _jaccard_similarity(target_state_tokens, _state_tokens(item))
                            if similarity >= float(state_similarity_threshold):
                                temperature = max(1.0e-6, float(state_similarity_temperature))
                                state_conditioned_scores[prefix_next] += float(exact_transition_bonus) * min(
                                    1.0,
                                    similarity / temperature,
                                )
        for idx, candidate_id in enumerate(candidates):
            if memory_mode != "inventory_remaining":
                scores[idx] += float(next_skill_bonus) * float(next_skill_counts.get(candidate_id, 0))
            if current_skill_id and memory_mode == "exact_count":
                scores[idx] += float(exact_transition_bonus) * float(
                    transition_counts.get((current_skill_id, candidate_id), 0)
                )
            elif current_skill_id and memory_mode == "latest_exact" and candidate_id == latest_exact_next_skill:
                scores[idx] += float(exact_transition_bonus)
            elif current_skill_id and memory_mode == "state_conditioned":
                scores[idx] += float(state_conditioned_scores.get(candidate_id, 0.0))
            elif memory_mode == "inventory_remaining":
                inventory = set(_row_inventory_skill_ids(copied))
                if candidate_id in inventory:
                    scores[idx] += float(next_skill_bonus)
                    if candidate_id not in used_skill_ids:
                        scores[idx] += float(exact_transition_bonus)
        if any(score != 0.0 for score in scores):
            memory_rows += 1
            if memory_mode == "state_conditioned":
                state_conditioned_rows += 1
            if memory_mode == "inventory_remaining":
                inventory_remaining_rows += 1
        positive_id = str(copied.get("next_skill_id") or "").strip()
        if positive_id and positive_id in candidates:
            positive_score = scores[candidates.index(positive_id)]
            total_positive_score += positive_score
            if positive_score > 0.0:
                positive_boosted_rows += 1
            if (
                current_skill_id
                and (
                    (memory_mode == "exact_count" and transition_counts.get((current_skill_id, positive_id), 0) > 0)
                    or (memory_mode == "latest_exact" and latest_exact_next_skill == positive_id)
                    or (memory_mode == "state_conditioned" and state_conditioned_scores.get(positive_id, 0.0) > 0)
                    or (
                        memory_mode == "inventory_remaining"
                        and positive_id in set(_row_inventory_skill_ids(copied))
                        and positive_id not in used_skill_ids
                    )
                )
            ):
                exact_transition_rows += 1
        copied["candidate_next_online_memory_scores"] = scores
        scored_rows.append(copied)

    row_count = len(scored_rows)
    return scored_rows, {
        "online_memory_rows": memory_rows,
        "online_memory_positive_boosted_rows": positive_boosted_rows,
        "online_memory_exact_transition_rows": exact_transition_rows,
        "online_memory_state_conditioned_rows": state_conditioned_rows,
        "online_memory_inventory_remaining_rows": inventory_remaining_rows,
        "online_memory_mean_positive_score": total_positive_score / max(1, row_count),
        "online_memory_next_skill_bonus": float(next_skill_bonus),
        "online_memory_exact_transition_bonus": float(exact_transition_bonus),
        "online_memory_mode": memory_mode,
        "online_memory_state_similarity_threshold": float(state_similarity_threshold),
        "online_memory_state_similarity_temperature": float(state_similarity_temperature),
    }


def decide_online_memory_weight_from_calibration(
    *,
    requested_weight: float,
    prior_eval: dict[str, Any],
    memory_eval: dict[str, Any],
    min_delta_mrr: float = 0.0,
    min_delta_recall5: float = 0.0,
) -> dict[str, Any]:
    requested_weight = float(requested_weight)
    prior_mrr = float(prior_eval.get("stage4_next_skill_mrr") or 0.0)
    memory_mrr = float(memory_eval.get("stage4_next_skill_mrr") or 0.0)
    prior_r5 = float(prior_eval.get("stage4_next_skill_recall@5") or 0.0)
    memory_r5 = float(memory_eval.get("stage4_next_skill_recall@5") or 0.0)
    delta_mrr = memory_mrr - prior_mrr
    delta_r5 = memory_r5 - prior_r5
    enabled = requested_weight != 0.0 and delta_mrr >= float(min_delta_mrr) and delta_r5 >= float(min_delta_recall5)
    return {
        "enabled": bool(enabled),
        "requested_online_memory_weight": requested_weight,
        "effective_online_memory_weight": requested_weight if enabled else 0.0,
        "delta_mrr": delta_mrr,
        "delta_recall@5": delta_r5,
        "min_delta_mrr": float(min_delta_mrr),
        "min_delta_recall@5": float(min_delta_recall5),
        "prior_mrr": prior_mrr,
        "memory_mrr": memory_mrr,
        "prior_recall@5": prior_r5,
        "memory_recall@5": memory_r5,
    }


def decide_online_memory_weight_by_benchmark_from_calibration(
    *,
    requested_weight: float,
    prior_eval_by_benchmark: dict[str, dict[str, Any]],
    memory_eval_by_benchmark: dict[str, dict[str, Any]],
    min_delta_mrr: float = 0.0,
    min_delta_recall5: float = 0.0,
) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    enabled: dict[str, bool] = {}
    benchmarks = sorted(set(prior_eval_by_benchmark) | set(memory_eval_by_benchmark))
    for benchmark in benchmarks:
        decision = decide_online_memory_weight_from_calibration(
            requested_weight=requested_weight,
            prior_eval=prior_eval_by_benchmark.get(benchmark, {}),
            memory_eval=memory_eval_by_benchmark.get(benchmark, {}),
            min_delta_mrr=min_delta_mrr,
            min_delta_recall5=min_delta_recall5,
        )
        by_benchmark[benchmark] = decision
        enabled[benchmark] = bool(decision["enabled"])
    enabled_benchmarks = [benchmark for benchmark in benchmarks if enabled.get(benchmark)]
    disabled_benchmarks = [benchmark for benchmark in benchmarks if not enabled.get(benchmark)]
    return {
        "enabled": bool(enabled_benchmarks),
        "memory_enabled": bool(enabled_benchmarks),
        "requested_online_memory_weight": float(requested_weight),
        "effective_online_memory_weight": float(requested_weight) if enabled_benchmarks else 0.0,
        "enabled_by_benchmark": enabled,
        "enabled_benchmarks": enabled_benchmarks,
        "disabled_benchmarks": disabled_benchmarks,
        "by_benchmark": by_benchmark,
    }


def decide_online_memory_variant_by_benchmark_from_calibration(
    *,
    prior_eval_by_benchmark: dict[str, dict[str, Any]],
    variant_eval_by_benchmark: dict[str, dict[str, dict[str, Any]]],
    requested_weight: float = 1.0,
    min_delta_mrr: float = 0.0,
    min_delta_recall5: float = 0.0,
) -> dict[str, Any]:
    selected_mode_by_benchmark: dict[str, str] = {}
    by_benchmark: dict[str, dict[str, Any]] = {}
    benchmarks = sorted(set(prior_eval_by_benchmark) | set(variant_eval_by_benchmark))
    for benchmark in benchmarks:
        prior_eval = prior_eval_by_benchmark.get(benchmark, {})
        prior_mrr = float(prior_eval.get("stage4_next_skill_mrr") or 0.0)
        prior_r5 = float(prior_eval.get("stage4_next_skill_recall@5") or 0.0)
        best_mode = "none"
        best_score = (0.0, 0.0)
        variant_decisions: dict[str, dict[str, Any]] = {}
        for mode, memory_eval in sorted(variant_eval_by_benchmark.get(benchmark, {}).items()):
            memory_mrr = float(memory_eval.get("stage4_next_skill_mrr") or 0.0)
            memory_r5 = float(memory_eval.get("stage4_next_skill_recall@5") or 0.0)
            delta_mrr = memory_mrr - prior_mrr
            delta_r5 = memory_r5 - prior_r5
            safe = delta_mrr >= float(min_delta_mrr) and delta_r5 >= float(min_delta_recall5)
            variant_decisions[mode] = {
                "enabled": bool(safe),
                "delta_mrr": delta_mrr,
                "delta_recall@5": delta_r5,
                "prior_mrr": prior_mrr,
                "memory_mrr": memory_mrr,
                "prior_recall@5": prior_r5,
                "memory_recall@5": memory_r5,
            }
            score = (delta_mrr, delta_r5)
            if safe and score > best_score:
                best_mode = mode
                best_score = score
        selected_mode_by_benchmark[benchmark] = best_mode
        by_benchmark[benchmark] = {
            "selected_mode": best_mode,
            "enabled": best_mode != "none",
            "variants": variant_decisions,
        }
    enabled_benchmarks = [
        benchmark for benchmark, mode in selected_mode_by_benchmark.items() if mode != "none"
    ]
    disabled_benchmarks = [
        benchmark for benchmark, mode in selected_mode_by_benchmark.items() if mode == "none"
    ]
    return {
        "enabled": bool(enabled_benchmarks),
        "memory_enabled": bool(enabled_benchmarks),
        "requested_online_memory_weight": float(requested_weight),
        "effective_online_memory_weight": float(requested_weight) if enabled_benchmarks else 0.0,
        "selected_mode_by_benchmark": dict(sorted(selected_mode_by_benchmark.items())),
        "enabled_benchmarks": enabled_benchmarks,
        "disabled_benchmarks": disabled_benchmarks,
        "by_benchmark": by_benchmark,
    }


def _attach_selected_variant_online_memory_scores(
    target_rows: Iterable[dict[str, Any]],
    *,
    feedback_rows: Iterable[dict[str, Any]],
    selected_mode_by_benchmark: dict[str, str],
    next_skill_bonus: float = 0.0,
    exact_transition_bonus: float = 5.0,
    state_similarity_threshold: float = 0.2,
    state_similarity_temperature: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_list = list(target_rows)
    selected_modes = {
        str(mode)
        for mode in selected_mode_by_benchmark.values()
        if str(mode) and str(mode) != "none"
    }
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    variant_reports: dict[str, dict[str, Any]] = {}
    for mode in sorted(selected_modes):
        scored_rows, report = attach_trajectory_prefix_online_memory_scores(
            row_list,
            feedback_rows=feedback_rows,
            next_skill_bonus=next_skill_bonus,
            exact_transition_bonus=exact_transition_bonus,
            memory_mode=mode,
            state_similarity_threshold=state_similarity_threshold,
            state_similarity_temperature=state_similarity_temperature,
        )
        variant_rows[mode] = scored_rows
        variant_reports[mode] = report
    output_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for idx, row in enumerate(row_list):
        benchmark = _row_benchmark(row)
        mode = str(selected_mode_by_benchmark.get(benchmark, "none") or "none")
        if mode != "none" and mode in variant_rows:
            copied = dict(variant_rows[mode][idx])
        else:
            copied = dict(row)
            scores = copied.get("candidate_next_online_memory_scores")
            if scores:
                copied["candidate_next_online_memory_scores"] = [0.0 for _ in scores]
            else:
                copied["candidate_next_online_memory_scores"] = [
                    0.0 for _ in copied.get("candidate_next_skill_ids") or []
                ]
            mode = "none"
        selected_counts[mode] += 1
        output_rows.append(copied)
    return output_rows, {
        "online_memory_mode": "auto_variant",
        "selected_mode_by_benchmark": dict(sorted(selected_mode_by_benchmark.items())),
        "selected_row_counts_by_mode": dict(sorted(selected_counts.items())),
        "variant_reports": variant_reports,
    }


def _zero_online_memory_scores_for_disabled_benchmarks(
    rows: Iterable[dict[str, Any]],
    enabled_by_benchmark: dict[str, bool],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        benchmark = _row_benchmark(copied)
        if not enabled_by_benchmark.get(benchmark, False):
            scores = copied.get("candidate_next_online_memory_scores") or []
            copied["candidate_next_online_memory_scores"] = [0.0 for _ in scores]
        filtered.append(copied)
    return filtered


def _candidate_recall_mapping_digest(
    skill_id_to_idx: dict[str, int] | None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
) -> str | None:
    if not skill_id_to_idx:
        return None
    ordered_ids = [
        str(skill_id)
        for skill_id, _idx in sorted(
            ((str(skill_id), int(idx)) for skill_id, idx in skill_id_to_idx.items()),
            key=lambda item: item[1],
        )
    ]
    equivalents = {
        str(skill_id): sorted(str(item) for item in values)
        for skill_id, values in sorted((equivalent_skill_ids_by_skill_id or {}).items())
    }
    payload = json.dumps(
        {"ordered_skill_ids": ordered_ids, "equivalent_skill_ids": equivalents},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_recall_identity(
    *,
    candidate_recall_mode: str,
    skill_id_to_idx: dict[str, int] | None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    static_k: int,
    dynamic_extra_k: int,
    final_k: int,
) -> dict[str, Any]:
    return {
        "candidate_recall_mode": str(candidate_recall_mode),
        "candidate_union_version": (
            CANDIDATE_UNION_VERSION
            if candidate_recall_mode == "static_plus_dynamic_extra"
            else "disabled"
        ),
        "candidate_recall_static_k": int(static_k),
        "candidate_recall_dynamic_extra_k": int(dynamic_extra_k),
        "candidate_recall_final_k": int(final_k),
        "candidate_recall_mapping_digest": _candidate_recall_mapping_digest(
            skill_id_to_idx,
            equivalent_skill_ids_by_skill_id,
        ),
    }


def _reliability_identity(
    *,
    reliability_mode: str,
    fixed_alpha: float,
    memory_utility_gate: torch.nn.Module | None,
    feature_update_count_cap: float,
    feature_candidate_count_cap: float,
) -> dict[str, Any]:
    return {
        "memory_utility_reliability_mode": str(reliability_mode),
        "memory_utility_fixed_alpha": float(fixed_alpha),
        "memory_utility_gate_supplied": memory_utility_gate is not None,
        "reliability_feature_update_count_cap": float(feature_update_count_cap),
        "reliability_feature_candidate_count_cap": float(feature_candidate_count_cap),
    }


def _accumulate_candidate_recall_metrics(
    totals: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    required_raw_keys: set[str] = set()
    for prefix in ("candidate_recall_all_", "candidate_recall_memory_active_"):
        for metric_name, count_name in (
            ("static_recall", "static_hit_rows"),
            ("dynamic_top_recall", "dynamic_top_hit_rows"),
            ("dynamic_extra_recall", "dynamic_extra_hit_rows"),
            ("union_recall", "union_hit_rows"),
            ("static_equal_budget_recall", "static_equal_budget_hit_rows"),
        ):
            if f"{prefix}{metric_name}" in metrics:
                required_raw_keys.update(
                    {f"{prefix}source_rows", f"{prefix}{count_name}"}
                )
        if f"{prefix}static_miss_recovery_rate" in metrics:
            required_raw_keys.update(
                {f"{prefix}static_miss_rows", f"{prefix}dynamic_rescue_rows"}
            )
        if f"{prefix}static_dynamic_overlap" in metrics:
            required_raw_keys.update(
                {f"{prefix}source_rows", f"{prefix}static_dynamic_overlap_sum"}
            )
        if f"{prefix}dynamic_only_candidate_count" in metrics:
            required_raw_keys.update(
                {f"{prefix}source_rows", f"{prefix}dynamic_only_candidate_total"}
            )
    if "candidate_recall_memory_active_coverage" in metrics:
        required_raw_keys.update(
            {
                "candidate_recall_all_source_rows",
                "candidate_recall_memory_active_source_rows",
            }
        )
    missing = sorted(key for key in required_raw_keys if key not in metrics)
    if missing:
        raise ValueError(
            "candidate recall aggregation requires raw count metrics: "
            + ", ".join(missing)
        )
    for key, value in metrics.items():
        if key.startswith("candidate_recall_") and isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0.0) + float(value)


def _finalize_candidate_recall_metrics(totals: dict[str, float]) -> dict[str, float]:
    result = dict(totals)
    for prefix in ("candidate_recall_all_", "candidate_recall_memory_active_"):
        source_rows = float(totals.get(f"{prefix}source_rows", 0.0))
        for metric_name, count_name in (
            ("static_recall", "static_hit_rows"),
            ("dynamic_top_recall", "dynamic_top_hit_rows"),
            ("dynamic_extra_recall", "dynamic_extra_hit_rows"),
            ("union_recall", "union_hit_rows"),
            ("static_equal_budget_recall", "static_equal_budget_hit_rows"),
        ):
            result[f"{prefix}{metric_name}"] = (
                float(totals.get(f"{prefix}{count_name}", 0.0)) / source_rows
                if source_rows > 0
                else 0.0
            )
        static_miss_rows = float(totals.get(f"{prefix}static_miss_rows", 0.0))
        result[f"{prefix}static_miss_recovery_rate"] = (
            float(totals.get(f"{prefix}dynamic_rescue_rows", 0.0)) / static_miss_rows
            if static_miss_rows > 0
            else 0.0
        )
        result[f"{prefix}static_dynamic_overlap"] = (
            float(totals.get(f"{prefix}static_dynamic_overlap_sum", 0.0)) / source_rows
            if source_rows > 0
            else 0.0
        )
        result[f"{prefix}dynamic_only_candidate_count"] = (
            float(totals.get(f"{prefix}dynamic_only_candidate_total", 0.0)) / source_rows
            if source_rows > 0
            else 0.0
        )
    all_rows = float(totals.get("candidate_recall_all_source_rows", 0.0))
    active_rows = float(totals.get("candidate_recall_memory_active_source_rows", 0.0))
    result["candidate_recall_memory_active_coverage"] = (
        active_rows / all_rows if all_rows > 0 else 0.0
    )
    return result


def evaluate_logged_online_stage4_rows_by_benchmark(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    online_memory_weight: float = 0.0,
    score_calibrator_enabled: bool = True,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    candidate_recall_mode: str = "stage0_candidates",
    skill_id_to_idx: dict[str, int] | None = None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    final_k: int = 64,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate: torch.nn.Module | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    route_records_path: str | Path | None = None,
    route_record_manifest_path: str | Path | None = None,
    route_record_pool_protocol: str | None = None,
    route_record_model_digest: str | None = None,
    route_record_sequential_benchmarks: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    route_scorer = str(route_scorer or UNIFIED_MEMORY_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_row_benchmark(row), []).append(row)
    return {
        benchmark: evaluate_logged_online_stage4_rows(
            model,
            benchmark_rows,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=online_memory_weight,
            score_calibrator_enabled=score_calibrator_enabled,
            route_scorer=route_scorer,
            candidate_recall_mode=candidate_recall_mode,
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            static_k=static_k,
            dynamic_extra_k=dynamic_extra_k,
            final_k=final_k,
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
            route_records_path=route_records_path,
            route_record_manifest_path=route_record_manifest_path,
            route_record_pool_protocol=route_record_pool_protocol,
            route_record_model_digest=route_record_model_digest,
            route_record_sequential_benchmarks=route_record_sequential_benchmarks,
        )
        for benchmark, benchmark_rows in sorted(groups.items())
    }


def _numeric_metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in current.items():
        if not isinstance(value, (int, float)):
            continue
        base_value = baseline.get(key)
        if not isinstance(base_value, (int, float)):
            continue
        deltas[key] = float(value) - float(base_value)
    return deltas


def split_logged_online_stage4_rows(
    rows: Iterable[dict[str, Any]],
    *,
    eval_rows: int = 256,
    eval_split_mode: str = "sequential_tail",
    trajectory_eval_steps: int = 1,
    eval_benchmark_caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    row_list = list(rows)
    eval_count = max(1, min(int(eval_rows), len(row_list))) if row_list else 0
    eval_split_mode = str(eval_split_mode or "sequential_tail")
    normalized_eval_caps = None
    eval_caps_report = {
        "enabled": False,
        "benchmark_caps": {},
        "source_rows": len(row_list),
        "retained_rows": 0,
        "skipped_rows": 0,
        "retained_by_benchmark": {},
        "skipped_by_benchmark": {},
    }
    if eval_benchmark_caps is not None:
        normalized_eval_caps = {
            str(key).strip(): int(value)
            for key, value in eval_benchmark_caps.items()
            if str(key).strip()
        }
        eval_caps_report = {
            "enabled": True,
            "benchmark_caps": dict(sorted(normalized_eval_caps.items())),
            "source_rows": len(row_list),
            "retained_rows": 0,
            "skipped_rows": 0,
            "retained_by_benchmark": {},
            "skipped_by_benchmark": {},
        }

    def can_select_eval(row: dict[str, Any], counts: Counter[str]) -> bool:
        if len(eval_set) >= eval_count:
            return False
        if normalized_eval_caps is None:
            return True
        benchmark = _row_benchmark(row)
        cap = normalized_eval_caps.get(benchmark)
        return cap is None or cap < 0 or counts[benchmark] < cap

    if eval_split_mode == "sequential_tail":
        if normalized_eval_caps is None:
            eval_set = row_list[-eval_count:] if eval_count else []
            stream_rows = row_list[:-eval_count] if len(row_list) > eval_count else list(row_list)
        else:
            eval_indices: set[int] = set()
            eval_set = []
            eval_counts: Counter[str] = Counter()
            skipped_counts: Counter[str] = Counter()
            for idx in range(len(row_list) - 1, -1, -1):
                row = row_list[idx]
                if can_select_eval(row, eval_counts):
                    eval_indices.add(idx)
                    eval_set.append(row)
                    eval_counts[_row_benchmark(row)] += 1
                elif len(eval_set) < eval_count:
                    skipped_counts[_row_benchmark(row)] += 1
                if len(eval_set) >= eval_count:
                    break
            eval_set.reverse()
            stream_rows = [row for idx, row in enumerate(row_list) if idx not in eval_indices]
            eval_caps_report.update(
                {
                    "retained_rows": len(eval_set),
                    "skipped_rows": sum(skipped_counts.values()),
                    "retained_by_benchmark": dict(sorted(eval_counts.items())),
                    "skipped_by_benchmark": dict(sorted(skipped_counts.items())),
                }
            )
        return stream_rows, eval_set, {
            "eval_split_mode": "sequential_tail",
            "eval_rows_requested": int(eval_rows),
            "eval_row_count": len(eval_set),
            "eval_benchmark_counts": _benchmark_counts(eval_set),
            "eval_benchmark_caps": eval_caps_report,
            "stream_row_count": len(stream_rows),
            "stream_positive_rank_bucket_counts": _positive_rank_bucket_counts(stream_rows),
            "eval_positive_rank_bucket_counts": _positive_rank_bucket_counts(eval_set),
            **_trajectory_prefix_signal_report(stream_rows, eval_set),
        }
    if eval_split_mode != "trajectory_prefix":
        raise ValueError(f"unsupported logged-online Stage4 eval split mode: {eval_split_mode}")

    groups: dict[str, list[dict[str, Any]]] = {}
    groupless: list[dict[str, Any]] = []
    for row in row_list:
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            groupless.append(row)
            continue
        groups.setdefault(trajectory_id, []).append(row)

    stream_rows: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []
    eval_counts: Counter[str] = Counter()
    skipped_by_eval_cap: Counter[str] = Counter()
    selected_trajectory_count = 0
    single_step_trajectory_count = 0
    trajectory_eval_steps = max(1, int(trajectory_eval_steps))
    for _trajectory_id, group in groups.items():
        ordered = sorted(group, key=_row_step_index)
        remaining_eval_slots = max(0, eval_count - len(eval_set))
        if len(ordered) <= trajectory_eval_steps or remaining_eval_slots <= 0:
            stream_rows.extend(ordered)
            if len(ordered) <= trajectory_eval_steps:
                single_step_trajectory_count += 1
            continue
        benchmark = _row_benchmark(ordered[-1])
        if normalized_eval_caps is not None:
            cap = normalized_eval_caps.get(benchmark)
            if cap is not None and cap >= 0:
                remaining_benchmark_slots = max(0, cap - eval_counts[benchmark])
                if remaining_benchmark_slots <= 0:
                    skipped_by_eval_cap[benchmark] += min(trajectory_eval_steps, len(ordered))
                    stream_rows.extend(ordered)
                    continue
                remaining_eval_slots = min(remaining_eval_slots, remaining_benchmark_slots)
        suffix_count = min(trajectory_eval_steps, remaining_eval_slots)
        split_at = len(ordered) - suffix_count
        stream_rows.extend(ordered[:split_at])
        selected_suffix = ordered[split_at:]
        eval_set.extend(selected_suffix)
        eval_counts.update(_row_benchmark(row) for row in selected_suffix)
        selected_trajectory_count += 1
    stream_rows.extend(groupless)
    if normalized_eval_caps is not None:
        eval_caps_report.update(
            {
                "retained_rows": len(eval_set),
                "skipped_rows": sum(skipped_by_eval_cap.values()),
                "retained_by_benchmark": dict(sorted(eval_counts.items())),
                "skipped_by_benchmark": dict(sorted(skipped_by_eval_cap.items())),
            }
        )
    return stream_rows, eval_set, {
        "eval_split_mode": "trajectory_prefix",
        "eval_rows_requested": int(eval_rows),
        "eval_row_count": len(eval_set),
        "eval_benchmark_counts": _benchmark_counts(eval_set),
        "eval_benchmark_caps": eval_caps_report,
        "stream_row_count": len(stream_rows),
        "trajectory_eval_steps": int(trajectory_eval_steps),
        "trajectory_count": len(groups),
        "selected_trajectory_count": selected_trajectory_count,
        "single_step_trajectory_count": single_step_trajectory_count,
        "groupless_row_count": len(groupless),
        "stream_positive_rank_bucket_counts": _positive_rank_bucket_counts(stream_rows),
        "eval_positive_rank_bucket_counts": _positive_rank_bucket_counts(eval_set),
        **_trajectory_prefix_signal_report(stream_rows, eval_set),
    }


def evaluate_logged_online_stage4_rows(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    online_memory_weight: float = 0.0,
    score_calibrator_enabled: bool = True,
    auto_replay_prefix_max_steps: int = 3,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    progress_path: str | Path | None = None,
    progress_label: str = "stage4_eval",
    candidate_recall_mode: str = "stage0_candidates",
    skill_id_to_idx: dict[str, int] | None = None,
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
    static_k: int = 500,
    dynamic_extra_k: int = 64,
    final_k: int = 64,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate: torch.nn.Module | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    route_records_path: str | Path | None = None,
    route_record_manifest_path: str | Path | None = None,
    route_record_pool_protocol: str | None = None,
    route_record_model_digest: str | None = None,
    route_record_sequential_benchmarks: set[str] | None = None,
) -> dict[str, Any]:
    route_scorer = str(route_scorer or UNIFIED_MEMORY_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    candidate_recall_mode = str(candidate_recall_mode or "stage0_candidates")
    if candidate_recall_mode not in CANDIDATE_RECALL_MODES:
        raise ValueError(f"unsupported candidate_recall_mode: {candidate_recall_mode}")
    if candidate_recall_mode == "static_plus_dynamic_extra" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("memory candidate recall requires route_scorer=unified_memory")
    if int(static_k) < 0 or int(dynamic_extra_k) < 0 or int(final_k) <= 0:
        raise ValueError("candidate budgets must be nonnegative and final_k must be positive")
    reliability_mode = str(reliability_mode or "dynamic")
    if reliability_mode not in RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    if not math.isfinite(float(fixed_alpha)) or not 0.0 <= float(fixed_alpha) <= 1.0:
        raise ValueError("fixed_alpha must be finite and in [0, 1]")
    if (
        not math.isfinite(float(feature_update_count_cap))
        or not math.isfinite(float(feature_candidate_count_cap))
        or float(feature_update_count_cap) <= 0
        or float(feature_candidate_count_cap) <= 0
    ):
        raise ValueError("reliability feature caps must be finite and positive")
    route_recording_enabled = route_records_path is not None
    normalized_sequential_benchmarks = {
        str(item).strip()
        for item in (route_record_sequential_benchmarks or set())
        if str(item).strip()
    }
    if route_recording_enabled:
        if route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
            raise ValueError("route recording requires route_scorer=unified_memory")
        if candidate_recall_mode != "static_plus_dynamic_extra":
            raise ValueError("route recording requires static_plus_dynamic_extra candidates")
        if not str(route_record_pool_protocol or "").strip():
            raise ValueError("route recording requires route_record_pool_protocol")
        if not str(route_record_model_digest or "").strip():
            raise ValueError("route recording requires route_record_model_digest")
        if route_record_sequential_benchmarks is None:
            raise ValueError("route recording requires route_record_sequential_benchmarks")
    candidate_identity = {
        **_candidate_recall_identity(
            candidate_recall_mode=candidate_recall_mode,
            skill_id_to_idx=skill_id_to_idx,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
            static_k=static_k,
            dynamic_extra_k=dynamic_extra_k,
            final_k=final_k,
        ),
        **_reliability_identity(
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
        ),
    }
    route_manifest_identity = {
        "candidate_union_version": CANDIDATE_UNION_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "pool_protocol": str(route_record_pool_protocol or "").strip(),
        "static_k": int(static_k),
        "dynamic_extra_k": int(dynamic_extra_k),
        "final_k": int(final_k),
        "feature_schema": RELIABILITY_FEATURE_SCHEMA_VERSION,
        "heuristic_alpha_version": HEURISTIC_ALPHA_VERSION,
        "feature_update_count_cap": float(feature_update_count_cap),
        "feature_candidate_count_cap": float(feature_candidate_count_cap),
        "model_checkpoint_chain_digest": str(route_record_model_digest or "").strip(),
        "skill_mapping_digest": candidate_identity["candidate_recall_mapping_digest"],
        "sequential_benchmarks": sorted(normalized_sequential_benchmarks),
    }
    if route_recording_enabled:
        validate_memory_utility_route_record_target(
            route_records_path,
            manifest_identity=route_manifest_identity,
            manifest_path=route_record_manifest_path,
        )
    if not rows:
        return {
            "stage4_act_count": 0.0,
            "route_scorer": route_scorer,
            **candidate_identity,
        }
    rows, auto_replay_prefix_report = _attach_auto_replay_prefixes(
        rows,
        max_steps=auto_replay_prefix_max_steps,
    )
    device = torch.device(device or getattr(model, "device", "cpu"))
    weighted: dict[str, float] = {}
    candidate_recall_totals: dict[str, float] = {}
    route_records: list[dict[str, Any]] = []
    total = 0
    last_metrics: dict[str, Any] = {}
    progress_file = Path(progress_path) if progress_path is not None else None
    batch_log_path = (
        progress_file.with_name(f"{progress_file.stem}_batches.jsonl") if progress_file is not None else None
    )
    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "eval", None)):
        model.eval()
    with torch.no_grad():
        eval_start_time = time.time()
        resolved_batch_size = max(1, int(batch_size))
        total_batches = (len(rows) + resolved_batch_size - 1) // resolved_batch_size
        completed_batches: dict[int, dict[str, Any]] = {}
        if not route_recording_enabled and batch_log_path is not None and batch_log_path.exists():
            for saved in _read_jsonl(batch_log_path):
                if not isinstance(saved, dict) or saved.get("status") != "batch_complete":
                    continue
                if int(saved.get("total_rows", -1)) != int(len(rows)):
                    continue
                if int(saved.get("batch_size", -1)) != int(resolved_batch_size):
                    continue
                if str(saved.get("label", "")) != str(progress_label):
                    continue
                if str(saved.get("route_scorer", "")) != str(route_scorer):
                    continue
                if any(saved.get(key) != value for key, value in candidate_identity.items()):
                    continue
                start_idx = int(saved.get("start", -1))
                if start_idx < 0:
                    continue
                completed_batches[start_idx] = saved
            for saved in sorted(completed_batches.values(), key=lambda item: int(item.get("start", 0))):
                metrics = saved.get("metrics") if isinstance(saved.get("metrics"), dict) else {}
                weight = int(saved.get("weight", 0))
                if weight <= 0:
                    continue
                total += weight
                last_metrics = dict(metrics)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key.startswith("candidate_recall_"):
                            continue
                        weighted[key] = weighted.get(key, 0.0) + float(value) * weight
                _accumulate_candidate_recall_metrics(candidate_recall_totals, metrics)
        for batch_idx, start in enumerate(range(0, len(rows), resolved_batch_size), start=1):
            batch = rows[start : start + resolved_batch_size]
            if start in completed_batches:
                if progress_file is not None:
                    elapsed = max(1.0e-6, time.time() - eval_start_time)
                    rows_per_second = total / elapsed
                    remaining_rows = max(0, len(rows) - total)
                    eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else None
                    _write_json(
                        progress_file,
                        {
                            "status": "resuming",
                            "label": str(progress_label),
                            "processed_rows": int(total),
                            "total_rows": int(len(rows)),
                            "batch_index": int(batch_idx),
                            "total_batches": int(total_batches),
                            "batch_size": int(resolved_batch_size),
                            "elapsed_seconds": round(float(elapsed), 3),
                            "rows_per_second": round(float(rows_per_second), 6),
                            "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                            "route_scorer": route_scorer,
                            "resumed_completed_batches": int(len(completed_batches)),
                            **candidate_identity,
                        },
                    )
                continue
            if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER:
                if route_recording_enabled:
                    batch_output = _build_current_state_route_batch(
                        model,
                        batch,
                        device,
                        skill_id_to_idx=skill_id_to_idx,
                        equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                        static_k=static_k,
                        dynamic_extra_k=dynamic_extra_k,
                        final_k=final_k,
                        reliability_mode=reliability_mode,
                        fixed_alpha=fixed_alpha,
                        memory_utility_gate=memory_utility_gate,
                        feature_update_count_cap=feature_update_count_cap,
                        feature_candidate_count_cap=feature_candidate_count_cap,
                    )
                    _loss, metrics = batch_output.loss, batch_output.metrics
                    route_records.extend(
                        build_memory_utility_route_records(
                            batch,
                            batch_output,
                            skill_id_to_idx=skill_id_to_idx or {},
                            sequential_benchmarks=normalized_sequential_benchmarks,
                            source_start=start,
                        )
                    )
                else:
                    _loss, metrics = _compute_current_state_route_loss(
                        model,
                        batch,
                        device,
                        candidate_recall_mode=candidate_recall_mode,
                        skill_id_to_idx=skill_id_to_idx,
                        equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                        static_k=static_k,
                        dynamic_extra_k=dynamic_extra_k,
                        final_k=final_k,
                        reliability_mode=reliability_mode,
                        fixed_alpha=fixed_alpha,
                        memory_utility_gate=memory_utility_gate,
                        feature_update_count_cap=feature_update_count_cap,
                        feature_candidate_count_cap=feature_candidate_count_cap,
                    )
            else:
                _loss, metrics = _compute_stage4_act_loss(
                    model,
                    batch,
                    device,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                    online_memory_weight=online_memory_weight,
                    score_calibrator_enabled=score_calibrator_enabled,
                    use_replay_prefix_belief=True,
                    trainable_replay_prefix=False,
                    route_scorer=route_scorer,
                )
            last_metrics = metrics
            weight = len(batch)
            total += weight
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key.startswith("candidate_recall_"):
                        continue
                    weighted[key] = weighted.get(key, 0.0) + float(value) * weight
            _accumulate_candidate_recall_metrics(candidate_recall_totals, metrics)
            if batch_log_path is not None:
                _append_jsonl(
                    batch_log_path,
                    {
                        "status": "batch_complete",
                        "label": str(progress_label),
                        "batch_index": int(batch_idx),
                        "total_batches": int(total_batches),
                        "start": int(start),
                        "end": int(start + weight),
                        "weight": int(weight),
                        "total_rows": int(len(rows)),
                        "batch_size": int(resolved_batch_size),
                        "route_scorer": route_scorer,
                        **candidate_identity,
                        "metrics": {
                            key: float(value)
                            for key, value in metrics.items()
                            if isinstance(value, (int, float))
                        },
                    },
                )
            if progress_path is not None:
                elapsed = max(1.0e-6, time.time() - eval_start_time)
                rows_per_second = total / elapsed
                remaining_rows = max(0, len(rows) - total)
                eta_seconds = remaining_rows / rows_per_second if rows_per_second > 0 else None
                _write_json(
                    progress_path,
                    {
                        "status": "running",
                        "label": str(progress_label),
                        "processed_rows": int(total),
                        "total_rows": int(len(rows)),
                        "batch_index": int(batch_idx),
                        "total_batches": int(total_batches),
                        "batch_size": int(resolved_batch_size),
                        "elapsed_seconds": round(float(elapsed), 3),
                        "rows_per_second": round(float(rows_per_second), 6),
                        "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                        "route_scorer": route_scorer,
                        **candidate_identity,
                    },
                )
    if was_training and callable(getattr(model, "train", None)):
        model.train()
    result = {key: value / max(1, total) for key, value in weighted.items()}
    if candidate_recall_totals:
        result.update(_finalize_candidate_recall_metrics(candidate_recall_totals))
    result["stage4_act_count"] = float(total)
    if progress_path is not None:
        _write_json(
            progress_path,
            {
                "status": "complete",
                "label": str(progress_label),
                "processed_rows": int(total),
                "total_rows": int(len(rows)),
                "batch_size": int(max(1, int(batch_size))),
                "route_scorer": route_scorer,
                **candidate_identity,
            },
        )
    for key in (
        "route_scorer",
        "transition_skill_head_type",
        "uses_stage0_prior_at_inference",
        "candidate_union_version",
        "candidate_selection_version",
        "candidate_tie_break_policy",
        "memory_utility_reliability_mode",
        "zero_history_fallback",
        "reliability_feature_schema",
        "memory_utility_heuristic_version",
        "reliability_changes_memory_state",
    ):
        if key in last_metrics:
            result[key] = last_metrics[key]
    result.setdefault("route_scorer", route_scorer)
    result.update(candidate_identity)
    result.update(
        {
            f"stage4_auto_replay_prefix_{key}": value
            for key, value in auto_replay_prefix_report.items()
            if isinstance(value, (int, float, bool))
        }
    )
    if route_recording_enabled:
        persisted = persist_memory_utility_route_records(
            route_records_path,
            route_records,
            manifest_identity=route_manifest_identity,
            manifest_path=route_record_manifest_path,
        )
        result.update(
            {
                "route_record_written_count": int(persisted["written_count"]),
                "route_record_total_count": int(persisted["total_count"]),
                "route_record_manifest_path": persisted["manifest_path"],
                "route_record_schema_version": "memory_utility_route_record_v1",
                "route_record_forced_full_rescore": True,
            }
        )
    return result


def _save_latest_checkpoint(model: Any, output_dir: Path, report: dict[str, Any]) -> str:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"
    freeze_report = report.get("freeze_report") if isinstance(report.get("freeze_report"), dict) else {}
    calibrator_only = bool(report.get("train_score_calibrator") or freeze_report.get("train_score_calibrator"))
    model_state_dict = (
        _stage4_score_calibrator_state_dict(model)
        if calibrator_only
        else (model.state_dict() if hasattr(model, "state_dict") else {})
    )
    torch.save(
        {
            "stage": "executor_free_logged_online_stage4",
            "checkpoint_format": "stage4_score_calibrator_only" if calibrator_only else "full_model_state_dict",
            "model_state_dict": model_state_dict,
            "train_report": report,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def _stage4_score_calibrator_state_dict(model: Any) -> dict[str, torch.Tensor]:
    calibrator = getattr(model, "stage4_score_calibrator", None)
    if calibrator is None or not hasattr(calibrator, "state_dict"):
        return {}
    return {
        f"stage4_score_calibrator.{name}": tensor.detach().cpu().clone()
        for name, tensor in calibrator.state_dict().items()
    }


def _restore_stage4_score_calibrator_state(model: Any, state_dict: dict[str, torch.Tensor]) -> bool:
    calibrator = getattr(model, "stage4_score_calibrator", None)
    if calibrator is None or not hasattr(calibrator, "load_state_dict") or not state_dict:
        return False
    device = next(calibrator.parameters()).device
    dtype = next(calibrator.parameters()).dtype
    stripped = {
        key.removeprefix("stage4_score_calibrator."): value.to(device=device, dtype=dtype)
        for key, value in state_dict.items()
        if key.startswith("stage4_score_calibrator.")
    }
    if not stripped:
        return False
    calibrator.load_state_dict(stripped)
    return True


def _save_stage4_score_calibrator_checkpoint(
    model: Any,
    output_dir: Path,
    report: dict[str, Any],
    *,
    filename: str = "best_score_calibrator.pt",
) -> str:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(
        {
            "stage": "executor_free_logged_online_stage4_score_calibrator",
            "stage4_score_calibrator_state_dict": _stage4_score_calibrator_state_dict(model),
            "train_report": report,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def run_logged_online_stage4_adaptation_on_rows(
    *,
    model: Any,
    rows: Iterable[dict[str, Any]],
    data_report: dict[str, Any],
    output_dir: str | Path,
    max_updates: int = 100,
    batch_size: int = 1,
    eval_interval: int = 20,
    eval_rows: int = 256,
    learning_rate: float = 5.0e-5,
    train_transition: bool = False,
    train_score_calibrator: bool = False,
    device: torch.device | str | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    online_memory_weight: float = 0.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    online_memory_mode: str = "latest_exact",
    online_memory_state_similarity_threshold: float = 0.2,
    online_memory_state_similarity_temperature: float = 1.0,
    online_memory_auto_gate: bool = False,
    online_memory_gate_scope: str = "global",
    online_memory_gate_eval_rows: int = 64,
    online_memory_gate_min_delta_mrr: float = 0.0,
    online_memory_gate_min_delta_recall5: float = 0.0,
    eval_split_mode: str = "sequential_tail",
    trajectory_eval_steps: int = 1,
    eval_benchmark_caps: dict[str, int] | None = None,
    input_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "online_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    route_scorer = str(route_scorer or LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")

    row_list = list(rows)
    input_report = dict(input_report or {})
    if not row_list:
        report = {
            "status": "blocked",
            "blocker": "no_logged_online_stage4_rows",
            "data_report": data_report,
            "input_report": input_report,
            "on_policy_rollout_used": False,
        }
        report.update(input_report)
        _write_json(output / "blocker_report.json", report)
        return report

    stream_rows, eval_set, split_report = split_logged_online_stage4_rows(
        row_list,
        eval_rows=eval_rows,
        eval_split_mode=eval_split_mode,
        trajectory_eval_steps=trajectory_eval_steps,
        eval_benchmark_caps=eval_benchmark_caps,
    )
    auto_variant_mode = str(online_memory_mode or "") == "auto_variant"
    if auto_variant_mode:
        stream_rows, stream_memory_report = _attach_selected_variant_online_memory_scores(
            stream_rows,
            feedback_rows=stream_rows,
            selected_mode_by_benchmark={},
            next_skill_bonus=online_memory_next_skill_bonus,
            exact_transition_bonus=online_memory_exact_transition_bonus,
            state_similarity_threshold=online_memory_state_similarity_threshold,
            state_similarity_temperature=online_memory_state_similarity_temperature,
        )
        eval_set, eval_memory_report = _attach_selected_variant_online_memory_scores(
            eval_set,
            feedback_rows=stream_rows,
            selected_mode_by_benchmark={},
            next_skill_bonus=online_memory_next_skill_bonus,
            exact_transition_bonus=online_memory_exact_transition_bonus,
            state_similarity_threshold=online_memory_state_similarity_threshold,
            state_similarity_temperature=online_memory_state_similarity_temperature,
        )
    else:
        stream_rows, stream_memory_report = attach_trajectory_prefix_online_memory_scores(
            stream_rows,
            feedback_rows=stream_rows,
            next_skill_bonus=online_memory_next_skill_bonus,
            exact_transition_bonus=online_memory_exact_transition_bonus,
            memory_mode=online_memory_mode,
            state_similarity_threshold=online_memory_state_similarity_threshold,
            state_similarity_temperature=online_memory_state_similarity_temperature,
        )
        eval_set, eval_memory_report = attach_trajectory_prefix_online_memory_scores(
            eval_set,
            feedback_rows=stream_rows,
            next_skill_bonus=online_memory_next_skill_bonus,
            exact_transition_bonus=online_memory_exact_transition_bonus,
            memory_mode=online_memory_mode,
            state_similarity_threshold=online_memory_state_similarity_threshold,
            state_similarity_temperature=online_memory_state_similarity_temperature,
        )
    device = _resolve_logged_online_device(model, device)

    effective_online_memory_weight = float(online_memory_weight)
    online_memory_gate_report: dict[str, Any] = {
        "enabled": False,
        "reason": "auto_gate_disabled",
        "gate_scope": str(online_memory_gate_scope),
        "requested_online_memory_weight": float(online_memory_weight),
        "effective_online_memory_weight": effective_online_memory_weight,
    }
    if bool(online_memory_auto_gate) and float(online_memory_weight) != 0.0 and len(stream_rows) >= 2:
        gate_stream_rows, gate_eval_rows, gate_split_report = split_logged_online_stage4_rows(
            stream_rows,
            eval_rows=max(1, int(online_memory_gate_eval_rows)),
            eval_split_mode="trajectory_prefix",
            trajectory_eval_steps=trajectory_eval_steps,
            eval_benchmark_caps=eval_benchmark_caps,
        )
        if auto_variant_mode:
            gate_eval_rows, gate_memory_report = _attach_selected_variant_online_memory_scores(
                gate_eval_rows,
                feedback_rows=gate_stream_rows,
                selected_mode_by_benchmark={},
                next_skill_bonus=online_memory_next_skill_bonus,
                exact_transition_bonus=online_memory_exact_transition_bonus,
                state_similarity_threshold=online_memory_state_similarity_threshold,
                state_similarity_temperature=online_memory_state_similarity_temperature,
            )
        else:
            gate_eval_rows, gate_memory_report = attach_trajectory_prefix_online_memory_scores(
                gate_eval_rows,
                feedback_rows=gate_stream_rows,
                next_skill_bonus=online_memory_next_skill_bonus,
                exact_transition_bonus=online_memory_exact_transition_bonus,
                memory_mode=online_memory_mode,
                state_similarity_threshold=online_memory_state_similarity_threshold,
                state_similarity_temperature=online_memory_state_similarity_temperature,
            )
        if gate_eval_rows:
            prior_gate_eval = evaluate_logged_online_stage4_rows(
                model,
                gate_eval_rows,
                batch_size=batch_size,
                device=device,
                transition_residual_lambda=transition_residual_lambda,
                transition_scoring_mode=transition_scoring_mode,
                route_scorer=route_scorer,
                online_memory_weight=0.0,
            )
            memory_gate_eval = evaluate_logged_online_stage4_rows(
                model,
                gate_eval_rows,
                batch_size=batch_size,
                device=device,
                transition_residual_lambda=transition_residual_lambda,
                transition_scoring_mode=transition_scoring_mode,
                route_scorer=route_scorer,
                online_memory_weight=online_memory_weight,
            )
            gate_scope = str(online_memory_gate_scope or "global")
            if auto_variant_mode and gate_scope == "source_benchmark":
                prior_gate_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
                    model,
                    gate_eval_rows,
                    batch_size=batch_size,
                    device=device,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                    route_scorer=route_scorer,
                    online_memory_weight=0.0,
                )
                variant_eval_by_benchmark: dict[str, dict[str, dict[str, Any]]] = {
                    benchmark: {} for benchmark in prior_gate_eval_by_benchmark
                }
                variant_memory_reports: dict[str, dict[str, Any]] = {}
                for variant_mode in ONLINE_MEMORY_AUTO_VARIANTS:
                    variant_gate_rows, variant_memory_report = attach_trajectory_prefix_online_memory_scores(
                        gate_eval_rows,
                        feedback_rows=gate_stream_rows,
                        next_skill_bonus=online_memory_next_skill_bonus,
                        exact_transition_bonus=online_memory_exact_transition_bonus,
                        memory_mode=variant_mode,
                        state_similarity_threshold=online_memory_state_similarity_threshold,
                        state_similarity_temperature=online_memory_state_similarity_temperature,
                    )
                    variant_memory_reports[variant_mode] = variant_memory_report
                    variant_eval = evaluate_logged_online_stage4_rows_by_benchmark(
                        model,
                        variant_gate_rows,
                        batch_size=batch_size,
                        device=device,
                        transition_residual_lambda=transition_residual_lambda,
                        transition_scoring_mode=transition_scoring_mode,
                        route_scorer=route_scorer,
                        online_memory_weight=online_memory_weight,
                    )
                    for benchmark, metrics in variant_eval.items():
                        variant_eval_by_benchmark.setdefault(benchmark, {})[variant_mode] = metrics
                gate_decision = decide_online_memory_variant_by_benchmark_from_calibration(
                    prior_eval_by_benchmark=prior_gate_eval_by_benchmark,
                    variant_eval_by_benchmark=variant_eval_by_benchmark,
                    requested_weight=online_memory_weight,
                    min_delta_mrr=online_memory_gate_min_delta_mrr,
                    min_delta_recall5=online_memory_gate_min_delta_recall5,
                )
                stream_rows, stream_memory_report = _attach_selected_variant_online_memory_scores(
                    stream_rows,
                    feedback_rows=stream_rows,
                    selected_mode_by_benchmark=gate_decision["selected_mode_by_benchmark"],
                    next_skill_bonus=online_memory_next_skill_bonus,
                    exact_transition_bonus=online_memory_exact_transition_bonus,
                    state_similarity_threshold=online_memory_state_similarity_threshold,
                    state_similarity_temperature=online_memory_state_similarity_temperature,
                )
                eval_set, eval_memory_report = _attach_selected_variant_online_memory_scores(
                    eval_set,
                    feedback_rows=stream_rows,
                    selected_mode_by_benchmark=gate_decision["selected_mode_by_benchmark"],
                    next_skill_bonus=online_memory_next_skill_bonus,
                    exact_transition_bonus=online_memory_exact_transition_bonus,
                    state_similarity_threshold=online_memory_state_similarity_threshold,
                    state_similarity_temperature=online_memory_state_similarity_temperature,
                )
                selected_gate_rows, selected_gate_memory_report = _attach_selected_variant_online_memory_scores(
                    gate_eval_rows,
                    feedback_rows=gate_stream_rows,
                    selected_mode_by_benchmark=gate_decision["selected_mode_by_benchmark"],
                    next_skill_bonus=online_memory_next_skill_bonus,
                    exact_transition_bonus=online_memory_exact_transition_bonus,
                    state_similarity_threshold=online_memory_state_similarity_threshold,
                    state_similarity_temperature=online_memory_state_similarity_temperature,
                )
                memory_gate_eval = evaluate_logged_online_stage4_rows(
                    model,
                    selected_gate_rows,
                    batch_size=batch_size,
                    device=device,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                    route_scorer=route_scorer,
                    online_memory_weight=online_memory_weight,
                )
                gate_extra = {
                    "prior_gate_eval_by_benchmark": prior_gate_eval_by_benchmark,
                    "variant_eval_by_benchmark": variant_eval_by_benchmark,
                    "variant_memory_reports": variant_memory_reports,
                    "selected_gate_memory_report": selected_gate_memory_report,
                }
            elif gate_scope == "source_benchmark":
                prior_gate_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
                    model,
                    gate_eval_rows,
                    batch_size=batch_size,
                    device=device,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                    route_scorer=route_scorer,
                    online_memory_weight=0.0,
                )
                memory_gate_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
                    model,
                    gate_eval_rows,
                    batch_size=batch_size,
                    device=device,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                    route_scorer=route_scorer,
                    online_memory_weight=online_memory_weight,
                )
                gate_decision = decide_online_memory_weight_by_benchmark_from_calibration(
                    requested_weight=online_memory_weight,
                    prior_eval_by_benchmark=prior_gate_eval_by_benchmark,
                    memory_eval_by_benchmark=memory_gate_eval_by_benchmark,
                    min_delta_mrr=online_memory_gate_min_delta_mrr,
                    min_delta_recall5=online_memory_gate_min_delta_recall5,
                )
                stream_rows = _zero_online_memory_scores_for_disabled_benchmarks(
                    stream_rows,
                    gate_decision["enabled_by_benchmark"],
                )
                eval_set = _zero_online_memory_scores_for_disabled_benchmarks(
                    eval_set,
                    gate_decision["enabled_by_benchmark"],
                )
                gate_extra = {
                    "prior_gate_eval_by_benchmark": prior_gate_eval_by_benchmark,
                    "memory_gate_eval_by_benchmark": memory_gate_eval_by_benchmark,
                }
            elif gate_scope == "global":
                gate_decision = decide_online_memory_weight_from_calibration(
                    requested_weight=online_memory_weight,
                    prior_eval=prior_gate_eval,
                    memory_eval=memory_gate_eval,
                    min_delta_mrr=online_memory_gate_min_delta_mrr,
                    min_delta_recall5=online_memory_gate_min_delta_recall5,
                )
                gate_extra = {}
            else:
                raise ValueError(f"unsupported online memory gate scope: {gate_scope}")
            online_memory_gate_report = {
                **gate_decision,
                "memory_enabled": bool(gate_decision["enabled"]),
                "auto_gate_enabled": True,
                "gate_scope": gate_scope,
                "gate_split": gate_split_report,
                "gate_memory_report": gate_memory_report,
                "prior_gate_eval": prior_gate_eval,
                "memory_gate_eval": memory_gate_eval,
                **gate_extra,
            }
            effective_online_memory_weight = float(online_memory_gate_report["effective_online_memory_weight"])
        else:
            online_memory_gate_report = {
                "enabled": False,
                "auto_gate_enabled": True,
                "reason": "no_gate_eval_rows",
                "gate_scope": str(online_memory_gate_scope),
                "requested_online_memory_weight": float(online_memory_weight),
                "effective_online_memory_weight": effective_online_memory_weight,
                "gate_split": gate_split_report,
            }

    freeze_report = _freeze_for_stage4_act(model, train_transition=bool(train_transition))
    if bool(train_score_calibrator):
        _ensure_stage4_score_calibrator(model)
        if hasattr(model, "named_parameters"):
            for name, param in model.named_parameters():
                param.requires_grad_(name.startswith("stage4_score_calibrator."))
        freeze_report = {
            **freeze_report,
            "train_transition": False,
            "train_score_calibrator": True,
            "trainable_modules": ["stage4_score_calibrator"],
            "score_calibrator_training_mode": "calibrator_only",
        }
    else:
        freeze_report = {
            **freeze_report,
            "train_score_calibrator": False,
        }
    params = [param for param in model.parameters() if param.requires_grad] if hasattr(model, "parameters") else []
    if not params:
        report = {
            "status": "blocked",
            "blocker": "no_trainable_stage4_parameters",
            "data_report": data_report,
            "freeze_report": freeze_report,
            "input_report": input_report,
            "on_policy_rollout_used": False,
        }
        report.update(input_report)
        _write_json(output / "blocker_report.json", report)
        return report
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate))

    initial_eval = evaluate_logged_online_stage4_rows(
        model,
        eval_set,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=effective_online_memory_weight,
        progress_path=output / "initial_eval_progress.json",
        progress_label="initial_eval",
    )
    best_selection_metric = "stage4_next_skill_mrr"
    best_update = 0
    best_eval = dict(initial_eval)
    best_score_calibrator_state = (
        _stage4_score_calibrator_state_dict(model) if bool(train_score_calibrator) else {}
    )
    best_checkpoint = ""
    if bool(train_score_calibrator):
        best_checkpoint = _save_stage4_score_calibrator_checkpoint(
            model,
            output,
            {
                "status": "running",
                "best_update": best_update,
                "best_eval": best_eval,
                "selection_metric": best_selection_metric,
            },
        )
    history: list[dict[str, Any]] = []
    latest_checkpoint = ""
    last_metrics: dict[str, Any] = {}
    max_updates = max(0, int(max_updates))
    eval_interval = max(1, int(eval_interval))
    for update_idx in range(1, max_updates + 1):
        batch = _batch_rows(stream_rows, update_idx - 1, batch_size)
        if bool(train_score_calibrator):
            if callable(getattr(model, "eval", None)):
                model.eval()
            calibrator = getattr(model, "stage4_score_calibrator", None)
            if callable(getattr(calibrator, "train", None)):
                calibrator.train()
        elif callable(getattr(model, "train", None)):
            model.train()
        loss, train_metrics = _compute_stage4_act_loss(
            model,
            batch,
            device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            route_scorer=route_scorer,
            online_memory_weight=effective_online_memory_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics: dict[str, Any] = {
            "online_update": update_idx,
            "batch_size": len(batch),
            "updated": True,
            **train_metrics,
            "loss": float(loss.detach().cpu().item()),
        }
        if update_idx % eval_interval == 0 or update_idx == max_updates:
            metrics["eval"] = evaluate_logged_online_stage4_rows(
                model,
                eval_set,
                batch_size=batch_size,
                device=device,
                transition_residual_lambda=transition_residual_lambda,
                transition_scoring_mode=transition_scoring_mode,
                route_scorer=route_scorer,
                online_memory_weight=effective_online_memory_weight,
            )
            if bool(train_score_calibrator):
                current_metric = float(metrics["eval"].get(best_selection_metric, float("-inf")))
                best_metric = float(best_eval.get(best_selection_metric, float("-inf")))
                if current_metric >= best_metric:
                    best_update = update_idx
                    best_eval = dict(metrics["eval"])
                    best_score_calibrator_state = _stage4_score_calibrator_state_dict(model)
                    best_checkpoint = _save_stage4_score_calibrator_checkpoint(
                        model,
                        output,
                        {
                            "status": "running",
                            "best_update": best_update,
                            "best_eval": best_eval,
                            "selection_metric": best_selection_metric,
                        },
                    )
        _append_jsonl(metrics_path, metrics)
        history.append(metrics)
        last_metrics = metrics
        latest_checkpoint = _save_latest_checkpoint(
            model,
            output,
            {
                "status": "running",
                "online_update_count": update_idx,
                "data_report": data_report,
                "freeze_report": freeze_report,
                "last_metrics": last_metrics,
            },
        )

    restored_best_score_calibrator = False
    if bool(train_score_calibrator) and best_score_calibrator_state:
        restored_best_score_calibrator = _restore_stage4_score_calibrator_state(
            model,
            best_score_calibrator_state,
        )

    eval_only_noop = (
        max_updates <= 0
        and not bool(train_score_calibrator)
        and float(effective_online_memory_weight) == 0.0
    )
    eval_benchmarks = sorted(
        {
            str(row.get("source_benchmark") or row.get("benchmark") or "unknown")
            for row in eval_set
            if isinstance(row, dict)
        }
    )
    if eval_only_noop:
        prior_eval = dict(initial_eval)
        prior_eval["reused_from_eval"] = "initial_eval"
        final_eval = dict(initial_eval)
        final_eval["reused_from_eval"] = "initial_eval"
        if len(eval_benchmarks) == 1:
            prior_eval_by_benchmark = {eval_benchmarks[0]: dict(prior_eval)}
            final_eval_by_benchmark = {eval_benchmarks[0]: dict(final_eval)}
        else:
            prior_eval_by_benchmark = {}
            final_eval_by_benchmark = {}
        _write_json(
            output / "eval_reuse_report.json",
            {
                "status": "reused_initial_eval",
                "reason": "max_updates=0_and_no_online_memory_or_score_calibrator_change",
                "benchmarks": eval_benchmarks,
            },
        )
    else:
        prior_eval = evaluate_logged_online_stage4_rows(
            model,
            eval_set,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            route_scorer=route_scorer,
            online_memory_weight=0.0,
            score_calibrator_enabled=False,
            progress_path=output / "prior_eval_progress.json",
            progress_label="no_online_memory_eval",
        )
        prior_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
            model,
            eval_set,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            route_scorer=route_scorer,
            online_memory_weight=0.0,
            score_calibrator_enabled=False,
        )
        final_eval = evaluate_logged_online_stage4_rows(
            model,
            eval_set,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            route_scorer=route_scorer,
            online_memory_weight=effective_online_memory_weight,
            progress_path=output / "final_eval_progress.json",
            progress_label="final_eval",
        )
        final_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
            model,
            eval_set,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            route_scorer=route_scorer,
            online_memory_weight=effective_online_memory_weight,
        )
    delta_final_vs_prior_by_benchmark = {
        benchmark: _numeric_metric_delta(
            final_eval_by_benchmark.get(benchmark, {}),
            prior_eval_by_benchmark.get(benchmark, {}),
        )
        for benchmark in sorted(set(prior_eval_by_benchmark) | set(final_eval_by_benchmark))
    }
    report = {
        "status": "ok",
        "stage": "executor_free_logged_online_stage4",
        "training_regime": "executor_free_logged_online_adaptation_simulation",
        "training_objective": "logged_next_skill_candidate_ranking",
        "on_policy_rollout_used": False,
        "logged_feedback_used": True,
        "valid_or_test_used_for_training": False,
        "input_report": input_report,
        "data_report": data_report,
        "stream_row_count": len(stream_rows),
        "eval_row_count": len(eval_set),
        "eval_split": split_report,
        "online_update_count": max_updates,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "train_score_calibrator": bool(train_score_calibrator),
        "freeze_report": freeze_report,
        "best_selection_metric": best_selection_metric,
        "best_update": int(best_update),
        "best_eval": best_eval,
        "best_checkpoint": best_checkpoint,
        "restored_best_score_calibrator": bool(restored_best_score_calibrator),
        "transition_residual_lambda": float(transition_residual_lambda),
        "transition_scoring_mode": str(transition_scoring_mode),
        "route_scorer": route_scorer,
        "online_memory_weight": float(online_memory_weight),
        "online_memory_mode": str(online_memory_mode),
        "online_memory_state_similarity_threshold": float(online_memory_state_similarity_threshold),
        "online_memory_state_similarity_temperature": float(online_memory_state_similarity_temperature),
        "effective_online_memory_weight": effective_online_memory_weight,
        "online_memory_gate": online_memory_gate_report,
        "online_memory_report": {
            "stream": stream_memory_report,
            "eval": eval_memory_report,
        },
        "prior_eval": prior_eval,
        "prior_eval_by_benchmark": prior_eval_by_benchmark,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "final_eval_by_benchmark": final_eval_by_benchmark,
        "delta_final_vs_prior": _numeric_metric_delta(final_eval, prior_eval),
        "delta_final_vs_prior_by_benchmark": delta_final_vs_prior_by_benchmark,
        "last_metrics": last_metrics,
        "history": history,
        "metrics_path": str(metrics_path),
        "latest_checkpoint": latest_checkpoint,
    }
    report.update(input_report)
    _write_json(output / "train_stdout.json", report)
    _save_latest_checkpoint(model, output, report)
    return report


def run_logged_online_stage4_adaptation_with_model(
    *,
    model: Any,
    logged_steps_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    max_updates: int = 100,
    batch_size: int = 1,
    eval_interval: int = 20,
    eval_rows: int = 256,
    learning_rate: float = 5.0e-5,
    candidate_count: int | None = None,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | None = None,
    positive_missing_policy: str = "skip",
    train_transition: bool = False,
    train_score_calibrator: bool = False,
    device: torch.device | str | None = None,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    route_scorer: str = LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
    online_memory_weight: float = 0.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    online_memory_mode: str = "latest_exact",
    online_memory_state_similarity_threshold: float = 0.2,
    online_memory_state_similarity_temperature: float = 1.0,
    online_memory_auto_gate: bool = False,
    online_memory_gate_scope: str = "global",
    online_memory_gate_eval_rows: int = 64,
    online_memory_gate_min_delta_mrr: float = 0.0,
    online_memory_gate_min_delta_recall5: float = 0.0,
    eval_split_mode: str = "sequential_tail",
    trajectory_eval_steps: int = 1,
    benchmark_caps: dict[str, int] | None = None,
    eval_benchmark_caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {_skill_id(row): idx for idx, row in enumerate(skills) if _skill_id(row)}
    rows, data_report = build_logged_online_stage4_rows(
        _read_jsonl(logged_steps_path),
        skill_id_to_idx,
        candidate_count=candidate_count,
        max_rows=max_rows,
        allowed_benchmarks=allowed_benchmarks,
        benchmark_caps=benchmark_caps,
        positive_missing_policy=positive_missing_policy,
    )
    return run_logged_online_stage4_adaptation_on_rows(
        model=model,
        rows=rows,
        data_report=data_report,
        output_dir=output,
        max_updates=max_updates,
        batch_size=batch_size,
        eval_interval=eval_interval,
        eval_rows=eval_rows,
        learning_rate=learning_rate,
        train_transition=train_transition,
        train_score_calibrator=train_score_calibrator,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=online_memory_weight,
        online_memory_next_skill_bonus=online_memory_next_skill_bonus,
        online_memory_exact_transition_bonus=online_memory_exact_transition_bonus,
        online_memory_mode=online_memory_mode,
        online_memory_state_similarity_threshold=online_memory_state_similarity_threshold,
        online_memory_state_similarity_temperature=online_memory_state_similarity_temperature,
        online_memory_auto_gate=online_memory_auto_gate,
        online_memory_gate_scope=online_memory_gate_scope,
        online_memory_gate_eval_rows=online_memory_gate_eval_rows,
        online_memory_gate_min_delta_mrr=online_memory_gate_min_delta_mrr,
        online_memory_gate_min_delta_recall5=online_memory_gate_min_delta_recall5,
        eval_split_mode=eval_split_mode,
        trajectory_eval_steps=trajectory_eval_steps,
        eval_benchmark_caps=eval_benchmark_caps,
        input_report={
            "logged_steps_path": str(logged_steps_path),
            "skills_path": str(skills_path),
        },
    )
