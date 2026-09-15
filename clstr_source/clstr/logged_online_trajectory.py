from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


TRAIN_SPLIT_NAMES = {"train", "training", "train_or_released_g3", "released_train"}


def read_jsonl_stream(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt JSONL at {path} line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def load_skill_id_set(path: str | Path) -> set[str]:
    skill_ids: set[str] = set()
    for row in read_jsonl_stream(path):
        skill_id = str(row.get("skill_id") or row.get("id") or "").strip()
        if skill_id:
            skill_ids.add(skill_id)
    return skill_ids


def load_stage0_handoff_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stage0 handoff report must be a JSON object: {path}")
    keys = (
        "enabled",
        "top_m",
        "positive_missing_policy",
        "current_positive_required_rows",
        "current_positive_covered_rows",
        "current_positive_coverage@M",
        "next_positive_required_rows",
        "next_positive_covered_rows",
        "next_positive_coverage@M",
        "skipped_reasons",
    )
    return {key: payload[key] for key in keys if key in payload}


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
    return str(row.get("split") or provenance.get("split") or "unknown").strip().lower()


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


def _candidate_skill_ids(row: dict[str, Any]) -> list[str]:
    candidates: list[Any] = []
    for key in (
        "stage0_next_candidate_skill_ids",
        "candidate_next_skill_ids",
        "candidates_next_skill_ids",
        "admissible_next_skill_ids",
        "candidate_skill_ids",
        "visible_inventory_skill_ids",
        "available_skill_ids",
        "admissible_skill_ids",
        "skill_inventory_ids",
        "stage0_allowed_skill_ids",
    ):
        candidates.extend(_as_list(row.get(key)))
    return _unique_strings(candidates)


def _api_refs(row: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("gt_api_refs", "api_refs", "valid_api_refs"):
        values.extend(_as_list(row.get(key)))
    return _unique_strings(values)


def _normalize_logged_online_step(
    row: dict[str, Any],
    *,
    skill_ids: set[str] | None = None,
    include_splits: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    split = _split_name(row)
    if include_splits is not None and split not in include_splits:
        return None, "split_not_included"

    current_skill_id = str(row.get("skill_id") or "").strip()
    if not current_skill_id:
        return None, "missing_current_skill"
    if skill_ids is not None and current_skill_id not in skill_ids:
        return None, "current_skill_not_in_pool"

    next_skill_id = str(row.get("next_skill_id") or "").strip()
    next_skill_mapped = bool(next_skill_id) and (skill_ids is None or next_skill_id in skill_ids)
    benchmark = str(row.get("benchmark") or row.get("environment") or "unknown")
    trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "")
    task_id = row.get("task_id")
    step_idx_raw = row.get("step_index", row.get("step_idx", 0))
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0

    candidates = _candidate_skill_ids(row)
    return (
        {
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "benchmark": benchmark,
            "split": split,
            "step_idx": step_idx,
            "goal": str(row.get("goal_text") or row.get("task_text") or "").strip(),
            "task_text": str(row.get("task_text") or "").strip(),
            "history_text": str(row.get("history_text") or "").strip(),
            "observation_text": str(row.get("state_text") or row.get("observation_text") or "").strip(),
            "action_text": str(row.get("action_text") or row.get("expert_action") or "").strip(),
            "next_observation_text": str(row.get("next_observation_text") or "").strip(),
            "candidate_skill_ids": candidates,
            "visible_inventory_skill_ids": _unique_strings(_as_list(row.get("visible_inventory_skill_ids"))),
            "gt_skill_ids": [current_skill_id],
            "gt_next_skill_ids": [next_skill_id] if next_skill_mapped else [],
            "raw_next_skill_id": next_skill_id,
            "gt_api_refs": _api_refs(row),
            "done": bool(row.get("done")) if row.get("done") is not None else False,
            "reward": row.get("reward"),
            "reward_type": "logged_gt",
            "source_quality": row.get("source_quality"),
            "provenance": {
                "source": "logged_online_trajectory",
                "original_provenance": _provenance(row),
            },
        },
        None,
    )


def iter_logged_online_steps(
    trajectories_path: str | Path,
    skill_ids: set[str] | None = None,
    *,
    include_splits: set[str] | None = None,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    for row in read_jsonl_stream(trajectories_path):
        step, _reason = _normalize_logged_online_step(
            row,
            skill_ids=skill_ids,
            include_splits=include_splits,
        )
        if step is None:
            continue
        yield step
        emitted += 1
        if max_rows is not None and emitted >= int(max_rows):
            break


def _empty_stats() -> Counter[str]:
    return Counter(
        {
            "steps": 0,
            "current_skill_mapped": 0,
            "next_skill_mapped": 0,
            "candidate_rows": 0,
            "current_gt_in_candidates": 0,
            "next_gt_in_candidates": 0,
            "label_updateable_steps": 0,
            "stage4_updateable_steps": 0,
        }
    )


def _rates(stats: dict[str, int]) -> dict[str, float]:
    steps = max(1, int(stats.get("steps") or 0))
    candidate_rows = max(1, int(stats.get("candidate_rows") or 0))
    return {
        "current_skill_mapping_rate": float(stats.get("current_skill_mapped", 0)) / steps,
        "next_skill_mapping_rate": float(stats.get("next_skill_mapped", 0)) / steps,
        "candidate_row_rate": float(stats.get("candidate_rows", 0)) / steps,
        "current_candidate_coverage": float(stats.get("current_gt_in_candidates", 0)) / candidate_rows,
        "next_candidate_coverage": float(stats.get("next_gt_in_candidates", 0)) / candidate_rows,
        "label_updateable_rate": float(stats.get("label_updateable_steps", 0)) / steps,
        "stage4_updateable_rate": float(stats.get("stage4_updateable_steps", 0)) / steps,
    }


def audit_logged_online_coverage(
    *,
    trajectories_path: str | Path,
    skill_pool_path: str | Path,
    include_splits: set[str] | None = None,
    max_source_rows: int | None = None,
    stage0_handoff_report_path: str | Path | None = None,
) -> dict[str, Any]:
    skill_ids = load_skill_id_set(skill_pool_path)
    include_splits = set(TRAIN_SPLIT_NAMES if include_splits is None else include_splits)
    skipped: Counter[str] = Counter()
    overall = _empty_stats()
    by_benchmark: defaultdict[str, Counter[str]] = defaultdict(_empty_stats)
    source_rows = 0
    emitted_steps = 0

    for row in read_jsonl_stream(trajectories_path):
        if max_source_rows is not None and source_rows >= int(max_source_rows):
            break
        source_rows += 1
        step, reason = _normalize_logged_online_step(
            row,
            skill_ids=skill_ids,
            include_splits=include_splits,
        )
        if step is None:
            skipped[str(reason or "unknown")] += 1
            continue

        emitted_steps += 1
        benchmark = str(step.get("benchmark") or "unknown")
        counters = (overall, by_benchmark[benchmark])
        current_skill = step["gt_skill_ids"][0]
        next_skills = step["gt_next_skill_ids"]
        raw_next = str(step.get("raw_next_skill_id") or "")
        candidates = [str(item) for item in step.get("candidate_skill_ids") or []]

        for stats in counters:
            stats["steps"] += 1
            stats["current_skill_mapped"] += 1
            if raw_next and next_skills:
                stats["next_skill_mapped"] += 1
            if candidates:
                stats["candidate_rows"] += 1
            if current_skill in candidates:
                stats["current_gt_in_candidates"] += 1
            if next_skills and next_skills[0] in candidates:
                stats["next_gt_in_candidates"] += 1
            if not step.get("done") and next_skills:
                stats["label_updateable_steps"] += 1
            if not step.get("done") and next_skills and next_skills[0] in candidates:
                stats["stage4_updateable_steps"] += 1

    overall_dict = dict(overall)
    by_benchmark_dict = {key: dict(value) for key, value in sorted(by_benchmark.items())}
    return {
        "source_path": str(trajectories_path),
        "skill_pool_path": str(skill_pool_path),
        "skill_pool_size": len(skill_ids),
        "include_splits": sorted(include_splits),
        "source_rows": source_rows,
        "emitted_steps": emitted_steps,
        "skipped_reasons": dict(sorted(skipped.items())),
        "overall": {**overall_dict, **_rates(overall_dict)},
        "by_benchmark": {
            key: {**stats, **_rates(stats)}
            for key, stats in by_benchmark_dict.items()
        },
        "stage0_topm_handoff": (
            load_stage0_handoff_summary(stage0_handoff_report_path)
            if stage0_handoff_report_path is not None
            else {"enabled": False}
        ),
    }
