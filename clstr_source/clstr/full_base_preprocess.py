from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from clstr.alfworld_action_skills import alfworld_action_to_skill_id
from clstr.external_data import write_json, write_jsonl
from clstr.full_base_data import FullBaseExample, LOSS_KEYS, allowed_loss_mask, validate_full_base_example


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSONL row: {exc}") from exc
    return rows


def _history_text(history: Any) -> str:
    if isinstance(history, list):
        return " | ".join(str(item) for item in history if str(item).strip())
    return str(history or "")


def _state_text(goal: str, task: str, observation: str, history: str) -> str:
    return "\n".join(
        [
            f"goal: {goal}",
            f"task_type: {task}",
            f"observation: {observation}",
            f"history: {history or '<empty>'}",
        ]
    )


def _infer_skill_id(benchmark: str, action_text: str, known_skill_ids: set[str]) -> str | None:
    action = str(action_text or "").strip().lower()
    candidates: list[str] = []
    if benchmark == "alfworld":
        return alfworld_action_to_skill_id(action, known_skill_ids)
    elif benchmark == "scienceworld":
        if action.startswith(("teleport ", "teleport to ", "go to ")):
            candidates = ["scienceworld/scienceworld-room-navigator", "scienceworld/scienceworld-room-teleporter"]
        elif action.startswith(("measure ", "use thermometer")) or "temperature" in action:
            candidates = ["scienceworld/scienceworld-temperature-measurer", "scienceworld/scienceworld-measurement-taker"]
        elif action.startswith(("look", "examine", "inspect")):
            candidates = ["scienceworld/scienceworld-room-scanner", "scienceworld/scienceworld-container-inspector"]
        elif action.startswith(("pick up ", "take ", "get ")):
            candidates = ["scienceworld/scienceworld-object-retriever", "scienceworld/scienceworld-item-fetcher"]
    elif benchmark == "webshop":
        if action.startswith("search["):
            candidates = ["webshop/webshop-search-executor"]
        elif "buy now" in action:
            candidates = ["webshop/webshop-purchase-executor"]
        elif action.startswith("click["):
            candidates = ["webshop/webshop-action-executor", "webshop/webshop-product-selector"]
    for candidate in candidates:
        if not known_skill_ids or candidate in known_skill_ids:
            return candidate
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _replay_step_record(row: dict[str, Any], skill_id: str | None = None) -> dict[str, Any]:
    record = {
        "step_index": _safe_int(row.get("step_index"), 0),
        "observation_text": str(row.get("observation_t") or row.get("observation_text") or ""),
        "action_text": str(row.get("expert_action_t") or row.get("action_text") or ""),
        "next_observation_text": row.get("next_observation_t") or row.get("next_observation_text"),
        "observation_source": str(
            row.get("observation_source") or "recorded_environment_observation"
        ),
    }
    if skill_id:
        record["skill_id"] = skill_id
    return record


def _load_skill_rows_from_sources(repo_root: Path, source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skill_rows: dict[str, dict[str, Any]] = {}
    for record in source_records:
        path_value = record.get("resolved_path") or record.get("path")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = repo_root / path
        candidate_files = []
        if path.is_dir():
            candidate_files.extend([path / "pseudo_skills.jsonl", path / "skills.jsonl"])
        elif path.name.endswith("skills.jsonl"):
            candidate_files.append(path)
        for skill_path in candidate_files:
            if not skill_path.exists():
                continue
            for row in _read_jsonl(skill_path):
                skill_id = str(row.get("skill_id") or "").strip()
                if not skill_id:
                    continue
                skill_rows.setdefault(
                    skill_id,
                    {
                        "skill_id": skill_id,
                        "name": row.get("name") or skill_id.rsplit("/", 1)[-1],
                        "description": row.get("description") or "",
                        "environment": row.get("environment") or skill_id.split("/", 1)[0],
                        "source": row.get("source") or "clstr_full_base",
                    },
                )
    return sorted(skill_rows.values(), key=lambda row: row["skill_id"])


def _convert_official_alfworld_replay(
    source: dict[str, Any],
    path: Path,
    known_skill_ids: set[str],
) -> Iterable[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        if str(row.get("split")) != "train":
            continue
        gamefile = str(row.get("gamefile") or row.get("task_id") or "")
        episode = str(row.get("episode_index") if row.get("episode_index") is not None else "")
        grouped_rows[(gamefile, episode)].append(row)
    for (gamefile, episode), group in sorted(grouped_rows.items(), key=lambda item: item[0]):
        del episode
        usable_rows: list[dict[str, Any]] = []
        for row in sorted(group, key=lambda item: _safe_int(item.get("step_index"), 0)):
            candidates = [str(item) for item in row.get("admissible_commands_t") or []]
            expert = str(row.get("expert_action_t") or "")
            if expert in candidates:
                usable_rows.append(row)
        prefix: list[dict[str, Any]] = []
        trajectory_step_count = len(usable_rows)
        for pos, row in enumerate(usable_rows):
            candidates = [str(item) for item in row.get("admissible_commands_t") or []]
            expert = str(row.get("expert_action_t") or "")
            history = _history_text(row.get("history_t"))
            goal = str(row.get("goal_text") or "")
            task = str(row.get("task_type") or "")
            observation = str(row.get("observation_t") or "")
            skill_id = _infer_skill_id("alfworld", expert, known_skill_ids)
            next_action = None
            next_skill_id = None
            if pos + 1 < trajectory_step_count:
                next_action = str(usable_rows[pos + 1].get("expert_action_t") or "") or None
                next_skill_id = _infer_skill_id("alfworld", next_action, known_skill_ids) if next_action else None
            step_index = _safe_int(row.get("step_index"), pos)
            mask = allowed_loss_mask(
                list(source.get("allowed_losses") or []),
                candidates,
                expert,
                row.get("next_observation_t"),
                bool(row.get("done_t")) if row.get("done_t") is not None else None,
                skill_id,
                history,
                next_skill_id=next_skill_id,
            )
            example = FullBaseExample(
                benchmark="alfworld",
                task_id=f"{gamefile}:{step_index}",
                trajectory_id=gamefile,
                step_index=step_index,
                trajectory_step_count=trajectory_step_count,
                goal_text=goal,
                task_text=task,
                state_text=_state_text(goal, task, observation, history),
                history_text=history,
                replay_prefix=list(prefix),
                action_text=expert,
                admissible_actions=candidates,
                expert_action=expert,
                next_action_text=next_action,
                next_skill_id=next_skill_id,
                next_observation_text=row.get("next_observation_t"),
                done=bool(row.get("done_t")) if row.get("done_t") is not None else None,
                reward=None,
                skill_id=skill_id,
                loss_mask=mask,
                source_quality=str(source.get("bucket")),
                candidate_source="alfworld_admissible_commands_t",
                on_policy_rollout=False,
                m_t_source="offline_full_replay_prefix_or_observation",
                provenance={
                    "source_id": source.get("source_id"),
                    "source_dataset": source.get("source_dataset"),
                    "bucket": source.get("bucket"),
                    "split": row.get("split"),
                    "gamefile": row.get("gamefile"),
                },
            )
            yield example.to_record()
            prefix.append(_replay_step_record(row, skill_id=skill_id))


def _convert_aux_skillnet(source: dict[str, Any], path: Path) -> Iterable[dict[str, Any]]:
    for trajectory in _read_jsonl(path / "trajectories.jsonl"):
        if str(trajectory.get("split")) != "train":
            continue
        benchmark = str(trajectory.get("environment") or "unknown")
        goal = str(trajectory.get("goal_text") or trajectory.get("task") or "")
        task = str(trajectory.get("task_type") or "")
        history_items: list[str] = []
        prefix: list[dict[str, Any]] = []
        steps = [step for step in trajectory.get("steps") or [] if isinstance(step, dict)]
        trajectory_id = str(trajectory.get("task_id") or "")
        for step_pos, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            if step.get("usable_for_skillnet_training") is False:
                history_items.append(str(step.get("action_text") or ""))
                prefix.append(
                    {
                        "step_index": _safe_int(step.get("t"), step_pos),
                        "observation_text": str(step.get("observation_text") or ""),
                        "action_text": str(step.get("action_text") or ""),
                        "next_observation_text": step.get("next_observation_text"),
                    }
                )
                continue
            action = str(step.get("action_text") or "")
            observation = str(step.get("observation_text") or "")
            history = _history_text(history_items)
            skill_id = step.get("pseudo_skill_id") or step.get("skillnet_skill_id")
            step_index = _safe_int(step.get("t"), step_pos)
            next_action = None
            next_skill_id = None
            for future in steps[step_pos + 1 :]:
                if future.get("usable_for_skillnet_training") is False:
                    continue
                next_action = str(future.get("action_text") or "") or None
                next_skill = future.get("pseudo_skill_id") or future.get("skillnet_skill_id")
                next_skill_id = str(next_skill) if next_skill else None
                break
            done = bool(step.get("done")) if step.get("done") is not None else False
            reward = step.get("reward")
            if isinstance(reward, (int, float)):
                reward_value = float(reward)
            else:
                reward_value = None
            mask = allowed_loss_mask(
                list(source.get("allowed_losses") or []),
                [],
                action,
                step.get("next_observation_text"),
                done,
                str(skill_id or ""),
                history,
                next_skill_id=next_skill_id,
            )
            example = FullBaseExample(
                benchmark=benchmark,
                task_id=f"{trajectory_id}:{step_index}",
                trajectory_id=trajectory_id,
                step_index=step_index,
                trajectory_step_count=len(steps),
                goal_text=goal,
                task_text=task,
                state_text=_state_text(goal, task, observation, history),
                history_text=history,
                replay_prefix=[],
                action_text=action,
                admissible_actions=[],
                expert_action=action,
                next_action_text=next_action,
                next_skill_id=next_skill_id,
                next_observation_text=step.get("next_observation_text"),
                done=done,
                reward=reward_value,
                skill_id=str(skill_id) if skill_id else None,
                loss_mask=mask,
                source_quality=str(source.get("bucket")),
                candidate_source="aux_trajectory_no_admissible_actions",
                on_policy_rollout=False,
                m_t_source="offline_history_text_only_or_observation",
                provenance={
                    "source_id": source.get("source_id"),
                    "source_dataset": source.get("source_dataset"),
                    "bucket": source.get("bucket"),
                    "split": trajectory.get("split"),
                    "trajectory_task_id": trajectory.get("task_id"),
                },
            )
            yield example.to_record()
            history_items.append(action)
            prefix.append(
                {
                    "step_index": step_index,
                    "observation_text": observation,
                    "action_text": action,
                    "next_observation_text": step.get("next_observation_text"),
                }
            )


def _convert_verified_policy_replay(
    source: dict[str, Any],
    path: Path,
    known_skill_ids: set[str],
) -> Iterable[dict[str, Any]]:
    for row in _read_jsonl(path):
        if str(row.get("split")) != "train":
            continue
        candidates = [str(item) for item in row.get("admissible_commands_t") or []]
        expert = str(row.get("expert_action_t") or "")
        if expert not in candidates:
            continue
        benchmark = str(row.get("benchmark") or source.get("benchmark") or "unknown")
        goal = str(row.get("goal_text") or "")
        task = str(row.get("task_type") or "")
        observation = str(row.get("observation_t") or "")
        history = _history_text(row.get("history_t"))
        skill_id = row.get("skill_id") or _infer_skill_id(benchmark, expert, known_skill_ids)
        next_action = str(row.get("next_action_text") or row.get("next_action_t") or "")
        next_skill_id = row.get("next_skill_id") or (
            _infer_skill_id(benchmark, next_action, known_skill_ids) if next_action else None
        )
        done = bool(row.get("done_t")) if row.get("done_t") is not None else None
        reward = row.get("reward_t", row.get("reward"))
        reward_value = float(reward) if isinstance(reward, (int, float)) else None
        mask = allowed_loss_mask(
            list(source.get("allowed_losses") or []),
            candidates,
            expert,
            row.get("next_observation_t"),
            done,
            str(skill_id or ""),
            history,
            next_skill_id=str(next_skill_id or "") or None,
        )
        trajectory_id = str(row.get("trajectory_id") or row.get("gamefile") or "")
        step_index = _safe_int(row.get("step_index"), 0)
        example = FullBaseExample(
            benchmark=benchmark,
            task_id=f"{trajectory_id}:{step_index}",
            trajectory_id=trajectory_id,
            step_index=step_index,
            trajectory_step_count=_safe_int(row.get("trajectory_step_count"), step_index + 1),
            goal_text=goal,
            task_text=task,
            state_text=_state_text(goal, task, observation, history),
            history_text=history,
            replay_prefix=list(row.get("replay_prefix") or []),
            action_text=expert,
            admissible_actions=candidates,
            expert_action=expert,
            next_action_text=next_action or None,
            next_skill_id=str(next_skill_id) if next_skill_id else None,
            next_observation_text=row.get("next_observation_t"),
            done=done,
            reward=reward_value,
            skill_id=str(skill_id) if skill_id else None,
            loss_mask=mask,
            source_quality=str(source.get("bucket")),
            candidate_source=str(row.get("candidate_source") or "official_harness_valid_actions"),
            on_policy_rollout=False,
            m_t_source="offline_verified_replay_prefix_or_observation",
            provenance={
                "source_id": source.get("source_id"),
                "source_dataset": source.get("source_dataset"),
                "bucket": source.get("bucket"),
                "split": row.get("split"),
                "trajectory_id": trajectory_id,
                **(row.get("provenance") if isinstance(row.get("provenance"), dict) else {}),
            },
        )
        yield example.to_record()


def _converter_for_source(source: dict[str, Any]):
    source_id = str(source.get("source_id") or "")
    if source_id == "alfworld_official_train_replay":
        return "alfworld_official"
    if source_id == "aux_skillnet_rebuilt":
        return "aux_skillnet"
    if source.get("bucket") == "verified_replay" and source.get("has_admissible_actions") and source.get("has_expert_action"):
        return "verified_policy_replay"
    return None


def build_clstr_full_base_data(
    registry_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    repo_root: str | Path = ".",
    max_records: int | None = None,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    repo_root = Path(repo_root)
    source_records = _read_jsonl(registry_path)
    skill_rows = _load_skill_rows_from_sources(repo_root, source_records)
    known_skill_ids = {str(row["skill_id"]) for row in skill_rows}
    records: list[dict[str, Any]] = []
    skipped_sources: Counter[str] = Counter()
    for source in source_records:
        if not source.get("train_allowed"):
            skipped_sources[str(source.get("source_id"))] += 1
            continue
        path_value = source.get("resolved_path") or source.get("path")
        if not path_value:
            skipped_sources[str(source.get("source_id"))] += 1
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            skipped_sources[str(source.get("source_id"))] += 1
            continue
        converter = _converter_for_source(source)
        if converter == "alfworld_official":
            iterator = _convert_official_alfworld_replay(source, path, known_skill_ids)
        elif converter == "aux_skillnet":
            iterator = _convert_aux_skillnet(source, path)
        elif converter == "verified_policy_replay":
            iterator = _convert_verified_policy_replay(source, path, known_skill_ids)
        else:
            skipped_sources[str(source.get("source_id"))] += 1
            continue
        for record in iterator:
            validate_full_base_example(record)
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
        if max_records is not None and len(records) >= max_records:
            break

    benchmark_counts = Counter(str(row["benchmark"]) for row in records)
    bucket_counts = Counter(str(row["source_quality"]) for row in records)
    loss_counts = Counter()
    for row in records:
        for key in LOSS_KEYS:
            if row["loss_mask"].get(key):
                loss_counts[key] += 1
    uses_valid_or_test = any(
        str(row.get("provenance", {}).get("split", "")).lower() in {"valid", "valid_seen", "valid_unseen", "test"}
        for row in records
    )
    if not skill_rows:
        skill_ids = sorted({str(row.get("skill_id")) for row in records if row.get("skill_id")})
        skill_rows = [
            {"skill_id": skill_id, "name": skill_id.rsplit("/", 1)[-1], "description": "", "source": "full_base_inferred"}
            for skill_id in skill_ids
        ]
    manifest = {
        "status": "ok" if records else "missing",
        "record_count": len(records),
        "skill_count": len(skill_rows),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "loss_activation_counts": {key: loss_counts.get(key, 0) for key in LOSS_KEYS},
        "uses_valid_or_test_for_training": uses_valid_or_test,
        "supports_offline_replay_prefix": any("replay_prefix" in row for row in records),
        "replay_prefix_policy": {
            "official_replay": "full_prefix",
            "verified_replay": "verified_prefix_if_available",
            "aux_only": "empty_prefix_history_text_only",
        },
        "supports_on_policy_rollout": False,
        "training_regime": "offline_replay_supervised_pretraining",
        "on_policy_rollout_used": False,
        "skipped_sources": dict(sorted(skipped_sources.items())),
        "train_path": str(output_dir / "train.jsonl"),
        "skills_path": str(output_dir / "skills.jsonl"),
    }
    write_jsonl(output_dir / "train.jsonl", records)
    write_jsonl(output_dir / "skills.jsonl", skill_rows)
    write_json(output_dir / "manifest.json", manifest)
    write_json(report_dir / "preprocess_report.json", manifest)
    return manifest
