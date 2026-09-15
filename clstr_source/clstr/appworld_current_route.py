from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr")
DEFAULT_OUTPUT_DIR = Path("data/clstr_appworld_current_route_v1")


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split(row: dict[str, Any], default: str = "") -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, str) and provenance.strip():
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    return str(row.get("split") or provenance.get("split") or default).strip().lower()


def _is_train_row(row: dict[str, Any]) -> bool:
    split = _split(row, default="train")
    if split not in {"", "train", "training"}:
        return False
    if row.get("train_allowed") is False:
        return False
    return True


def _skill_id(skill: dict[str, Any], idx: int) -> str:
    return str(skill.get("skill_id") or skill.get("canonical_skill_id") or skill.get("id") or idx)


def _normalize_skill_pool(
    skills: list[dict[str, Any]],
    *,
    executor_domain: str,
    appworld_executor_compatible: bool,
    source_role: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, skill in enumerate(skills):
        skill_id = _skill_id(skill, idx)
        copied = dict(skill)
        copied["skill_id"] = skill_id
        copied.setdefault("canonical_skill_id", skill_id)
        copied.setdefault("source", copied.get("source_dataset") or "SkillX-AppWorld")
        copied.setdefault("executor_domain", executor_domain)
        copied.setdefault("appworld_executor_compatible", bool(appworld_executor_compatible))
        copied.setdefault("skill_pool_role", source_role)
        copied.setdefault(
            "provenance",
            {
                "source_dataset": copied.get("source_dataset") or "SkillX-AppWorld",
                "source_path": copied.get("source_path"),
                "adapter": "clstr_appworld_current_route_v1",
            },
        )
        normalized.append(copied)
    return normalized


def _normalize_extra_skill_pool(skills: list[dict[str, Any]], *, source_path: str | Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, skill in enumerate(skills):
        skill_id = _skill_id(skill, idx)
        is_appworld = skill_id.startswith("skillx/appworld/")
        copied = dict(skill)
        copied["skill_id"] = skill_id
        copied.setdefault("canonical_skill_id", skill_id)
        copied["executor_domain"] = "appworld" if is_appworld else "non_appworld_distractor"
        copied["appworld_executor_compatible"] = bool(is_appworld)
        copied["skill_pool_role"] = "large_pool_distractor" if not is_appworld else "extra_appworld_skill"
        copied.setdefault("source", copied.get("source_dataset") or "extra_skill_pool")
        copied.setdefault(
            "provenance",
            {
                "source_dataset": copied.get("source_dataset") or "extra_skill_pool",
                "source_path": str(source_path),
                "adapter": "clstr_appworld_current_route_v1",
                "role": copied["skill_pool_role"],
            },
        )
        normalized.append(copied)
    return normalized


def _normalize_checkpoint_base_skill_pool(skills: list[dict[str, Any]], *, source_path: str | Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, skill in enumerate(skills):
        skill_id = _skill_id(skill, idx)
        is_appworld = skill_id.startswith("skillx/appworld/")
        copied = dict(skill)
        copied["skill_id"] = skill_id
        copied.setdefault("canonical_skill_id", skill_id)
        copied.setdefault("executor_domain", "appworld" if is_appworld else copied.get("source_dataset") or "checkpoint_base")
        copied["appworld_executor_compatible"] = bool(is_appworld or copied.get("appworld_executor_compatible") is True)
        copied["skill_pool_role"] = "checkpoint_base"
        copied.setdefault(
            "provenance",
            {
                "source_dataset": copied.get("source_dataset") or "checkpoint_base",
                "source_path": str(source_path),
                "adapter": "clstr_appworld_current_route_v1",
                "role": "checkpoint_base",
            },
        )
        normalized.append(copied)
    return normalized


def _merge_checkpoint_base_and_appworld_skills(
    *,
    base_skills: list[dict[str, Any]],
    appworld_skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = list(base_skills)
    seen = {_skill_id(skill, idx) for idx, skill in enumerate(base_skills)}
    appended_ids: list[str] = []
    skipped_duplicates: list[str] = []
    for offset, row in enumerate(appworld_skills):
        skill_id = _skill_id(row, len(merged) + offset)
        if skill_id in seen:
            skipped_duplicates.append(skill_id)
            continue
        seen.add(skill_id)
        copied = dict(row)
        copied["skill_id"] = skill_id
        copied.setdefault("canonical_skill_id", skill_id)
        copied["appworld_executor_compatible"] = True
        copied["executor_domain"] = "appworld"
        copied["skill_pool_role"] = "appworld_appended_after_checkpoint"
        copied.setdefault("is_appended_after_checkpoint", True)
        copied.setdefault("retrieval_seen_count", 0)
        copied.setdefault("transition_seen_count", 0)
        copied.setdefault("act_seen_count", 0)
        merged.append(copied)
        appended_ids.append(skill_id)
    return merged, {
        "enabled": True,
        "pool_order": "base_then_appworld_append",
        "checkpoint_prefix_skill_count": len(base_skills),
        "appended_appworld_skill_count": len(appended_ids),
        "appended_appworld_skill_ids": appended_ids,
        "skipped_duplicate_appworld_skill_ids": skipped_duplicates,
    }


def _merge_extra_skill_pools(
    *,
    root: Path,
    primary_skills: list[dict[str, Any]],
    extra_skill_pool_paths: Iterable[str | Path] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = list(primary_skills)
    seen = {_skill_id(skill, idx) for idx, skill in enumerate(primary_skills)}
    skipped: Counter[str] = Counter()
    path_reports: list[dict[str, Any]] = []
    for raw_path in extra_skill_pool_paths or []:
        if raw_path is None or not str(raw_path).strip():
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        raw_rows = _read_jsonl(path)
        normalized_rows = _normalize_extra_skill_pool(raw_rows, source_path=raw_path)
        added = 0
        duplicate = 0
        for row in normalized_rows:
            skill_id = _skill_id(row, len(merged))
            if skill_id in seen:
                duplicate += 1
                skipped["duplicate_skill_id"] += 1
                continue
            seen.add(skill_id)
            merged.append(row)
            added += 1
        path_reports.append(
            {
                "path": str(raw_path),
                "source_rows": len(raw_rows),
                "added_rows": added,
                "duplicate_rows": duplicate,
            }
        )
    return merged, {
        "enabled": bool(path_reports),
        "extra_skill_pool_count": max(0, len(merged) - len(primary_skills)),
        "paths": path_reports,
        "skipped": dict(sorted(skipped.items())),
    }


def _skill_name(skill_id: str, skill_by_id: dict[str, dict[str, Any]]) -> str:
    skill = skill_by_id.get(skill_id) or {}
    return str(skill.get("name") or skill_id)


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _history_text(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    return "\n".join(f"{idx}. skill={skill}; observation={obs}" for idx, (skill, obs) in enumerate(history, start=1))


def _state_text(*, query: str, history: list[tuple[str, str]], observation: str = "") -> str:
    history_block = _history_text(history) or "<empty>"
    return "\n".join(
        [
            "[User Goal]",
            str(query or ""),
            "",
            "[Previous Skill Trace]",
            history_block,
            "",
            "[Current Observation]",
            str(observation or ""),
            "",
            "[Routing Task]",
            "Choose the next useful AppWorld SkillX skill for the current execution state.",
        ]
    )


def _state_before_text(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        return str(value or ""), "", ""
    query = str(value.get("query") or value.get("instruction_text") or "")
    raw_history = value.get("history") or []
    history_pairs: list[tuple[str, str]] = []
    if isinstance(raw_history, list):
        for item in raw_history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                history_pairs.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                history_pairs.append((str(item.get("skill") or item.get("action") or ""), str(item.get("observation") or item.get("obs") or "")))
            elif item:
                history_pairs.append((str(item), ""))
    observation = str(value.get("observation") or "")
    return _state_text(query=query, history=history_pairs, observation=observation), query, _history_text(history_pairs)


def _loss_mask(*, has_next: bool, belief: bool = True) -> dict[str, bool]:
    return {
        "L_policy": True,
        "L_trans": bool(has_next),
        "L_trans_skill_ce": bool(has_next),
        "belief": bool(has_next and belief),
        "STOP": True,
        "routing": True,
    }


def _build_task_index(tasks: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    task_by_id: dict[str, dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("query_id") or "")
        query_id = str(task.get("query_id") or task_id)
        if not task_id or not query_id:
            skipped["missing_task_or_query_id"] += 1
            continue
        if not _is_train_row(task):
            skipped["non_train_task"] += 1
            continue
        copied = dict(task)
        copied["task_id"] = task_id
        copied["query_id"] = query_id
        copied["split"] = "train"
        task_by_id[task_id] = copied
        task_by_id[query_id] = copied
    return task_by_id, skipped


def _routing_retrieval_rows(
    *,
    tasks: dict[str, dict[str, Any]],
    qrels: list[dict[str, Any]],
    skill_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for qrel in qrels:
        if not _is_train_row(qrel):
            skipped["non_train_qrel"] += 1
            continue
        if int(qrel.get("relevance", 1)) <= 0:
            skipped["non_positive_qrel"] += 1
            continue
        query_id = str(qrel.get("query_id") or "")
        skill_id = str(qrel.get("skill_id") or qrel.get("positive_skill_id") or "")
        task = tasks.get(query_id)
        if task is None:
            skipped["missing_train_task"] += 1
            continue
        if skill_id not in skill_ids:
            skipped["skill_not_in_pool"] += 1
            continue
        rows.append(
            {
                "source": "appworld_task_skillx_qrel",
                "query_id": query_id,
                "query_text": str(task.get("query") or task.get("instruction_text") or ""),
                "positive_skill_id": skill_id,
                "negative_skill_ids": [],
                "split": "train",
                "provenance": {
                    "source_dataset": "data/appworld_routing/train_qrels.jsonl",
                    "split": "train",
                    "task_id": task.get("task_id"),
                    "query_id": query_id,
                },
            }
        )
    return rows, {"rows": len(rows), "skipped": dict(sorted(skipped.items()))}


def _replay_trajectory_rows(
    *,
    replay_rows: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    skill_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skill_ids = set(skill_by_id)
    for replay in replay_rows:
        task_id = str(replay.get("task_id") or "")
        task = task_by_id.get(task_id)
        if task is None or not _is_train_row(replay):
            skipped["non_train_or_missing_task"] += 1
            continue
        steps = replay.get("steps") or []
        if not isinstance(steps, list) or not steps:
            skipped["empty_steps"] += 1
            continue
        query = str(replay.get("query") or replay.get("instruction_text") or task.get("query") or task.get("instruction_text") or "")
        history: list[tuple[str, str]] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                skipped["invalid_step"] += 1
                continue
            skill_id = str(step.get("skill_id") or "")
            if skill_id not in skill_ids:
                skipped["skill_not_in_pool"] += 1
                continue
            next_step = steps[step_index + 1] if step_index + 1 < len(steps) and isinstance(steps[step_index + 1], dict) else None
            next_skill_id = str(next_step.get("skill_id") or "") if next_step else ""
            has_next = next_skill_id in skill_ids
            observation = str(step.get("observation") or "")
            skill_name = str(step.get("skill_name") or _skill_name(skill_id, skill_by_id))
            next_skill_name = str(next_step.get("skill_name") or _skill_name(next_skill_id, skill_by_id)) if has_next and next_step else ""
            state_observation = history[-1][1] if history else ""
            row = {
                "benchmark": "appworld",
                "task_id": task_id,
                "trajectory_id": f"appworld::{task_id}",
                "step_index": int(step_index),
                "goal_text": str(task.get("instruction_text") or replay.get("instruction_text") or ""),
                "task_text": str(task.get("query") or query),
                "state_text": _state_text(query=query, history=history, observation=state_observation),
                "history_text": _history_text(history),
                "action_text": skill_name,
                "expert_action": skill_name,
                "admissible_actions": [str(item.get("skill_name") or _skill_name(str(item.get("skill_id") or ""), skill_by_id)) for item in steps if isinstance(item, dict)],
                "next_action_text": next_skill_name,
                "next_observation_text": observation,
                "skill_id": skill_id,
                "next_skill_id": next_skill_id if has_next else "",
                "done": not has_next,
                "reward": replay.get("reward") if not has_next else None,
                "loss_mask": _loss_mask(has_next=has_next, belief=False),
                "source_quality": "appworld_oracle_skillx_replay",
                "candidate_source": "appworld_oracle_skill_sequence",
                "on_policy_rollout": False,
                "m_t_source": "appworld_train_oracle_replay_prefix",
                "split": "train",
                "provenance": {
                    "source_id": "appworld_train_replay",
                    "source_dataset": "data/appworld_routing/train_replay.jsonl",
                    "split": "train",
                    "task_id": task_id,
                    "success": bool(replay.get("success", False)),
                },
            }
            rows.append(row)
            history.append((skill_name, observation))
    return rows, {"rows": len(rows), "skipped": dict(sorted(skipped.items()))}


def _verified_pair_trajectory_rows(
    *,
    pair_rows: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    skills: list[dict[str, Any]],
    skill_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
    skill_by_id = {skill_id: skill for skill_id, skill in zip(skill_ids, skills)}
    for pair in pair_rows:
        task_id = str(pair.get("task_id") or "")
        state_before = pair.get("state_before") if isinstance(pair.get("state_before"), dict) else {}
        split = _split(pair, default=str(state_before.get("artifact", {}).get("split") if isinstance(state_before.get("artifact"), dict) else "train"))
        if split not in {"", "train", "training"} or task_by_id.get(task_id) is None:
            skipped["non_train_or_missing_task"] += 1
            continue
        try:
            action_idx = int(pair["action_at_t"]) + int(skill_index_offset)
            next_idx = int(pair["a_next_plus"]) + int(skill_index_offset)
        except (KeyError, TypeError, ValueError):
            skipped["missing_action_or_next_index"] += 1
            continue
        if action_idx < 0 or action_idx >= len(skill_ids) or next_idx < 0 or next_idx >= len(skill_ids):
            skipped["skill_index_out_of_range"] += 1
            continue
        candidate_indices: list[int] = []
        for value in pair.get("candidates_next") or []:
            try:
                idx = int(value) + int(skill_index_offset)
            except (TypeError, ValueError):
                skipped["invalid_candidate_index"] += 1
                continue
            if 0 <= idx < len(skill_ids) and idx not in candidate_indices:
                candidate_indices.append(idx)
        if next_idx not in candidate_indices:
            candidate_indices.append(next_idx)
        skill_id = skill_ids[action_idx]
        next_skill_id = skill_ids[next_idx]
        candidate_ids = [skill_ids[idx] for idx in candidate_indices]
        state_text, query, history_text = _state_before_text(state_before)
        current_name = _skill_name(skill_id, skill_by_id)
        next_name = _skill_name(next_skill_id, skill_by_id)
        step_idx = int(pair.get("step_idx", pair.get("step_index", 0)))
        rows.append(
            {
                "benchmark": "appworld",
                "task_id": task_id,
                "trajectory_id": f"appworld-verified::{task_id}",
                "step_index": step_idx,
                "goal_text": str((task_by_id.get(task_id) or {}).get("instruction_text") or query),
                "task_text": query,
                "state_text": state_text,
                "history_text": history_text,
                "action_text": current_name,
                "expert_action": current_name,
                "admissible_actions": [_skill_name(candidate_id, skill_by_id) for candidate_id in candidate_ids],
                "next_action_text": next_name,
                "next_observation_text": str(pair.get("obs_at_t") or ""),
                "skill_id": skill_id,
                "next_skill_id": next_skill_id,
                "candidate_next_skill_ids": candidate_ids,
                "candidate_next_skill_indices": candidate_indices,
                "done": False,
                "reward": None,
                "loss_mask": _loss_mask(has_next=True, belief=True),
                "source_quality": "appworld_oracle_verified_pair",
                "candidate_source": "appworld_verified_pair_topk",
                "on_policy_rollout": False,
                "m_t_source": "appworld_verified_pair_replay_prefix",
                "split": "train",
                "provenance": {
                    "source_id": "appworld_verified_pairs_train",
                    "source_dataset": "data/appworld_act/verified_pairs_train.jsonl",
                    "split": "train",
                    "task_id": task_id,
                    "step_idx": step_idx,
                    "was_in_raw_topk": bool(pair.get("was_in_raw_topk", False)),
                },
            }
        )
    return rows, {"rows": len(rows), "skipped": dict(sorted(skipped.items()))}


def _trajectory_retrieval_rows(trajectories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in trajectories:
        task_id = str(row.get("task_id") or "")
        step_index = int(row.get("step_index", 0))
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        source_id = str(provenance.get("source_id") or "appworld_trajectory")
        base = {
            "source_dataset": "clstr_appworld_current_route_v1",
            "source": "trajectory_derived",
            "benchmark": "appworld",
            "split": "train",
            "task_id": task_id,
            "trajectory_id": row.get("trajectory_id"),
            "step_index": step_index,
            "source_id": source_id,
        }
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if current_skill_id:
            rows.append(
                {
                    "source": "appworld_trajectory_derived_current",
                    "query_id": f"appworld-trajectory::{task_id}::{step_index}::current"
                    if source_id == "appworld_train_replay"
                    else f"appworld-verified-pair::{task_id}::{step_index}::current",
                    "query_text": str(row.get("state_text") or ""),
                    "positive_skill_id": current_skill_id,
                    "negative_skill_ids": _unique([next_skill_id]),
                    "split": "train",
                    "provenance": {**base, "target": "current"},
                }
            )
        else:
            skipped["missing_current_skill"] += 1
        if next_skill_id:
            rows.append(
                {
                    "source": "appworld_trajectory_derived_next",
                    "query_id": f"appworld-trajectory::{task_id}::{step_index}::next"
                    if source_id == "appworld_train_replay"
                    else f"appworld-verified-pair::{task_id}::{step_index}::next",
                    "query_text": "\n".join(
                        [
                            str(row.get("state_text") or ""),
                            "",
                            "[After Current Skill Observation]",
                            str(row.get("next_observation_text") or ""),
                        ]
                    ),
                    "positive_skill_id": next_skill_id,
                    "negative_skill_ids": _unique([current_skill_id, *[str(item) for item in row.get("candidate_next_skill_ids") or [] if str(item) != next_skill_id]]),
                    "split": "train",
                    "provenance": {**base, "target": "next"},
                }
            )
        else:
            skipped["missing_next_skill"] += 1
    return rows, {"rows": len(rows), "skipped": dict(sorted(skipped.items()))}


def build_appworld_current_route_dataset(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    train_tasks_path: str | Path = "data/appworld_routing/train_tasks.jsonl",
    train_qrels_path: str | Path = "data/appworld_routing/train_qrels.jsonl",
    train_replay_path: str | Path = "data/appworld_routing/train_replay.jsonl",
    verified_pairs_path: str | Path = "data/appworld_act/verified_pairs_train.jsonl",
    extra_skill_pool_paths: Iterable[str | Path] | None = None,
    base_skill_pool_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_skills = _read_jsonl(root / skill_pool_path)
    primary_skills = _normalize_skill_pool(
        raw_skills,
        executor_domain="appworld",
        appworld_executor_compatible=True,
        source_role="appworld_primary",
    )
    appworld_skill_index_offset = 0
    dynamic_skill_registry_report: dict[str, Any] = {
        "enabled": False,
        "pool_order": "appworld_primary_first",
        "checkpoint_prefix_skill_count": 0,
        "appended_appworld_skill_count": 0,
    }
    if base_skill_pool_path is not None and str(base_skill_pool_path).strip():
        base_path = Path(base_skill_pool_path)
        if not base_path.is_absolute():
            base_path = root / base_path
        base_skills = _normalize_checkpoint_base_skill_pool(_read_jsonl(base_path), source_path=base_skill_pool_path)
        skills, dynamic_skill_registry_report = _merge_checkpoint_base_and_appworld_skills(
            base_skills=base_skills,
            appworld_skills=primary_skills,
        )
        appworld_skill_index_offset = int(dynamic_skill_registry_report["checkpoint_prefix_skill_count"])
    else:
        skills = list(primary_skills)
    skills, extra_skill_pool_report = _merge_extra_skill_pools(
        root=root,
        primary_skills=skills,
        extra_skill_pool_paths=extra_skill_pool_paths,
    )
    skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
    skill_by_id = {skill_id: skill for skill_id, skill in zip(skill_ids, skills)}

    task_by_id, task_report = _build_task_index(_read_jsonl(root / train_tasks_path))
    routing_rows, routing_report = _routing_retrieval_rows(
        tasks=task_by_id,
        qrels=_read_jsonl(root / train_qrels_path),
        skill_ids=set(skill_ids),
    )
    replay_trajectories, replay_report = _replay_trajectory_rows(
        replay_rows=_read_jsonl(root / train_replay_path),
        task_by_id=task_by_id,
        skill_by_id=skill_by_id,
    )
    verified_trajectories, verified_report = _verified_pair_trajectory_rows(
        pair_rows=_read_jsonl(root / verified_pairs_path),
        task_by_id=task_by_id,
        skills=skills,
        skill_index_offset=appworld_skill_index_offset,
    )
    trajectories = replay_trajectories + verified_trajectories
    trajectory_retrieval, trajectory_retrieval_report = _trajectory_retrieval_rows(trajectories)
    retrieval_rows = routing_rows + trajectory_retrieval

    skill_write_count = _write_jsonl(out_dir / "skill_pool.jsonl", skills)
    retrieval_write_count = _write_jsonl(out_dir / "retrieval.jsonl", retrieval_rows)
    trajectory_write_count = _write_jsonl(out_dir / "trajectories.jsonl", trajectories)
    source_inventory = [
        {
            "source_id": "appworld_task_skillx_qrel",
            "available": bool(routing_rows),
            "rows": len(routing_rows),
            "path": str(train_qrels_path),
        },
        {
            "source_id": "appworld_train_replay",
            "available": bool(replay_trajectories),
            "rows": len(replay_trajectories),
            "path": str(train_replay_path),
        },
        {
            "source_id": "appworld_verified_pairs_train",
            "available": bool(verified_trajectories),
            "rows": len(verified_trajectories),
            "path": str(verified_pairs_path),
        },
    ]
    _write_jsonl(out_dir / "source_inventory.jsonl", source_inventory)

    trajectory_counts = Counter(str(row.get("provenance", {}).get("source_id") if isinstance(row.get("provenance"), dict) else "unknown") for row in trajectories)
    status = "ok" if skill_write_count > 0 and retrieval_write_count > 0 and trajectory_write_count > 0 else "blocked"
    manifest = {
        "status": status,
        "schema_version": "appworld_current_route_v1",
        "output_dir": str(out_dir.relative_to(root)) if out_dir.is_relative_to(root) else str(out_dir),
        "files": {
            "skill_pool": "skill_pool.jsonl",
            "retrieval": "retrieval.jsonl",
            "trajectories": "trajectories.jsonl",
            "source_inventory": "source_inventory.jsonl",
        },
        "skill_pool": {
            "skill_count": skill_write_count,
            "primary_appworld_skill_count": len(primary_skills),
            "extra_skill_pool_count": int(extra_skill_pool_report["extra_skill_pool_count"]),
            "extra_skill_pools": extra_skill_pool_report,
            "base_skill_pool_path": None if base_skill_pool_path is None else str(base_skill_pool_path),
            "source_path": str(skill_pool_path),
        },
        "dynamic_skill_registry": dynamic_skill_registry_report,
        "tasks": {
            "train_task_keys": len(task_by_id),
            "skipped": dict(sorted(task_report.items())),
        },
        "retrieval_stream": {
            "total_pairs": retrieval_write_count,
            "routing_pairs": len(routing_rows),
            "trajectory_derived_pairs": len(trajectory_retrieval),
            "routing": routing_report,
            "trajectory_derived": trajectory_retrieval_report,
        },
        "trajectory_stream": {
            "total_rows": trajectory_write_count,
            "rows_by_source": dict(sorted(trajectory_counts.items())),
            "train_replay": replay_report,
            "verified_pairs": verified_report,
        },
        "source_inventory": {
            "available_sources": [row["source_id"] for row in source_inventory if row["available"]],
            "missing_sources": [row["source_id"] for row in source_inventory if not row["available"]],
        },
        "leakage_boundary": {
            "train_only": True,
            "non_train_rows_skipped": True,
            "note": "This adapter only consumes AppWorld train routing/replay/verified-pair files and skips rows whose split is not train.",
        },
        "note": "Data preparation only. Use the current CLSTR Stage0/Stage1/Stage2/Stage4 scripts with this data root; do not mix checkpoints from a different skill pool.",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest
