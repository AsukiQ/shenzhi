from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATASET_NAMES = {
    "sft_alfworld": "moroqq/sft_alfworld_trajectory_dataset_v5_cleaned",
    "auto_dreamer": "uyffg/auto-dreamer",
    "eto_sft": "agent-eto/eto-sft-trajectory",
}


@dataclass(frozen=True)
class PseudoSkillMapping:
    pseudo_skill_id: str
    name: str
    description: str
    low_confidence: bool = False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _iter_jsonish_files(root: Path) -> list[Path]:
    suffixes = {".json", ".jsonl", ".gz", ".parquet"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and (path.suffix in suffixes or path.name.endswith(".jsonl.gz"))
    )


def _iter_json_records(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"pyarrow is required to read parquet file: {path}") from exc
        table = pq.read_table(path)
        rows = table.to_pylist()
        for idx, row in enumerate(rows):
            if limit is not None and idx >= limit:
                break
            if isinstance(row, dict):
                yield row
        return

    opener = gzip.open if path.name.endswith(".gz") else open
    text = ""
    with opener(path, "rt", encoding="utf-8") as handle:
        if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
            return
        text = handle.read()
    payload = json.loads(text)
    rows = payload if isinstance(payload, list) else [payload]
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        if isinstance(row, dict):
            yield row


def _file_format(path: Path) -> str:
    if path.suffix == ".parquet":
        return "parquet"
    if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
        return "jsonl"
    if path.suffix == ".json":
        return "json"
    return path.suffix.lstrip(".") or "unknown"


def _split_from_path(path: Path) -> str:
    bits = [part.lower() for part in path.parts]
    for split in ("train", "validation", "valid", "val", "dev", "test"):
        if split in bits:
            return "dev" if split in {"validation", "valid", "val"} else split
    name = path.name.lower()
    for split in ("train", "validation", "valid", "val", "dev", "test"):
        if split in name:
            return "dev" if split in {"validation", "valid", "val"} else split
    return "unknown"


def build_aux_trajectory_schema_report(
    dataset_name: str,
    dataset_key: str,
    source_root: str | Path,
    max_samples_per_file: int = 8,
) -> dict[str, Any]:
    source_root = Path(source_root)
    files = _iter_jsonish_files(source_root) if source_root.exists() else []
    fields: set[str] = set()
    nested_fields: dict[str, set[str]] = defaultdict(set)
    format_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    sample_count = 0
    file_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        fmt = _file_format(path)
        format_counts[fmt] += 1
        split = _split_from_path(path.relative_to(source_root))
        split_counts[split] += 1
        file_sample_count = 0
        try:
            for row in _iter_json_records(path, limit=max_samples_per_file):
                sample_count += 1
                file_sample_count += 1
                for key, value in row.items():
                    fields.add(str(key))
                    if isinstance(value, dict):
                        nested_fields[str(key)].update(str(k) for k in value)
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        nested_fields[str(key)].update(str(k) for k in value[0])
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path.relative_to(source_root)), "error": str(exc)})
        file_reports.append(
            {
                "path": str(path.relative_to(source_root)),
                "format": fmt,
                "split": split,
                "sample_count": file_sample_count,
            }
        )
    return {
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "source_root": str(source_root),
        "status": "ok" if files else "missing",
        "file_count": len(files),
        "sample_count": sample_count,
        "fields": sorted(fields),
        "nested_fields": {key: sorted(values) for key, values in sorted(nested_fields.items())},
        "format_counts": dict(sorted(format_counts.items())),
        "split_file_counts": dict(sorted(split_counts.items())),
        "files": file_reports,
        "errors": errors,
    }


def _environment_from_row(dataset_key: str, row: dict[str, Any], path: Path) -> str:
    raw = (
        row.get("environment")
        or row.get("benchmark")
        or row.get("source")
        or row.get("task_type")
        or row.get("game_file")
        or ""
    )
    text = f"{raw} {'/'.join(path.parts)} {dataset_key}".lower()
    if "scienceworld" in text or "sciworld" in text:
        return "scienceworld"
    if "alfworld" in text:
        return "alfworld"
    if "webshop" in text:
        return "webshop"
    if "crafter" in text:
        return "crafter"
    return dataset_key.replace("_", "-")


def _task_id(row: dict[str, Any], fallback: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("task_id")
        or row.get("id")
        or row.get("task")
        or metadata.get("task_id")
        or metadata.get("id")
        or fallback
    )


def _task_type(row: dict[str, Any], environment: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("task_type")
        or row.get("type")
        or row.get("game_file")
        or metadata.get("task_type")
        or environment
    )


def _success(row: dict[str, Any]) -> bool | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        value = container.get("success")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "success", "succeeded"}:
            return True
        if normalized in {"false", "0", "no", "failure", "failed"}:
            return False
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _reward(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        value = container.get("reward")
        if value is None:
            value = container.get("total_reward")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def _step_field(step: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in step and step[key] is not None:
            return step[key]
    return None


def _action_command_text(text: str) -> str:
    matches = re.findall(r"(?im)^\s*(?:action|act)\s*:\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip()
    return text


def _steps_from_explicit_list(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = row.get("steps") or row.get("trajectory") or row.get("traj") or row.get("actions")
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_steps):
        if isinstance(item, dict):
            obs = _step_field(item, "observation", "obs", "state", "input", "prompt")
            action = _step_field(item, "action", "action_text", "assistant", "response", "output", "action_taken")
            next_obs = _step_field(
                item,
                "next_observation",
                "next_observation_text",
                "next_obs",
                "next_state",
                "env_feedback_raw",
                "feedback",
            )
            done = _step_field(item, "done", "terminated", "is_done")
            reward = _step_field(item, "reward", "r")
        else:
            obs = ""
            action = str(item)
            next_obs = None
            done = None
            reward = None
        if action is None:
            continue
        steps.append(
            {
                "t": idx,
                "observation_text": "" if obs is None else str(obs),
                "action_text": _action_command_text(str(action)),
                "next_observation_text": None if next_obs is None else str(next_obs),
                "done": None if done is None else bool(done),
                "reward": float(reward) if isinstance(reward, (int, float)) else None,
            }
        )
    return steps


def _steps_from_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if not isinstance(messages, list):
        return []
    steps: list[dict[str, Any]] = []
    last_observation = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").lower()
        content = item.get("content")
        if content is None:
            content = item.get("value")
        content_text = "" if content is None else str(content)
        if role in {"user", "human", "system", "environment"}:
            if steps and steps[-1]["next_observation_text"] is None:
                steps[-1]["next_observation_text"] = content_text
            last_observation = content_text
        elif role in {"assistant", "gpt", "agent"}:
            steps.append(
                {
                    "t": len(steps),
                    "observation_text": last_observation,
                    "action_text": _action_command_text(content_text),
                    "next_observation_text": None,
                    "done": None,
                    "reward": None,
                }
            )
    if steps:
        steps[-1]["done"] = True if steps[-1]["done"] is None else steps[-1]["done"]
    return steps


def _normalize_verb(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def map_action_to_pseudo_skill(environment: str, action_text: str) -> PseudoSkillMapping:
    env = (environment or "generic").lower().replace("-", "_")
    action = _normalize_verb(_action_command_text(action_text))
    low_confidence = False
    if env == "alfworld":
        if action.startswith(("go to ", "go ", "walk to ", "move to ")):
            suffix = "go_to_object"
        elif action.startswith(("open ", "close ")):
            suffix = "open_object" if action.startswith("open ") else "close_object"
        elif action.startswith(("take ", "pick up ", "grab ")):
            suffix = "take_object"
        elif action.startswith(("put ", "place ")):
            suffix = "put_object"
        elif action.startswith(("look", "examine", "inspect")):
            suffix = "inspect_object"
        else:
            suffix = "text_action"
            low_confidence = True
    elif env == "scienceworld":
        if action.startswith(("measure ", "read ")):
            suffix = "measure_property"
        elif action.startswith(("mix ", "combine ")):
            suffix = "mix_objects"
        elif action.startswith(("focus ", "look", "examine")):
            suffix = "inspect_object"
        elif action.startswith(("use ", "activate ")):
            suffix = "use_tool"
        else:
            suffix = "text_action"
            low_confidence = True
    else:
        suffix = "text_action"
        low_confidence = True
    pseudo_skill_id = f"{env}/{suffix}"
    name = suffix.replace("_", " ")
    return PseudoSkillMapping(
        pseudo_skill_id=pseudo_skill_id,
        name=name,
        description=f"Pseudo-skill for {env} action template: {name}.",
        low_confidence=low_confidence,
    )


def _normalize_row(
    dataset_key: str,
    dataset_name: str,
    source_root: Path,
    source_path: Path,
    row: dict[str, Any],
    row_idx: int,
) -> dict[str, Any] | None:
    rel_path = source_path.relative_to(source_root)
    environment = _environment_from_row(dataset_key, row, rel_path)
    steps = _steps_from_explicit_list(row) or _steps_from_messages(row)
    if not steps:
        return None
    normalized_steps = []
    for idx, step in enumerate(steps):
        mapping = map_action_to_pseudo_skill(environment, step["action_text"])
        normalized_steps.append(
            {
                "t": int(step.get("t", idx)),
                "observation_text": str(step.get("observation_text") or ""),
                "action_text": str(step.get("action_text") or ""),
                "next_observation_text": step.get("next_observation_text"),
                "done": step.get("done"),
                "reward": step.get("reward"),
                "pseudo_skill_id": mapping.pseudo_skill_id,
                "pseudo_skill_low_confidence": mapping.low_confidence,
            }
        )
    return {
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "source_path": str(rel_path),
        "split": _split_from_path(rel_path),
        "environment": environment,
        "task_id": _task_id(row, f"{dataset_key}:{rel_path}:{row_idx}"),
        "task_type": _task_type(row, environment),
        "success": _success(row),
        "reward": _reward(row),
        "steps": normalized_steps,
        "metadata": {
            "game_file": row.get("game_file"),
            "source": row.get("source"),
        },
    }


def _auto_dreamer_episode_metadata(root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("episodes.jsonl")):
        for row in _iter_json_records(path):
            episode_id = row.get("episode_id")
            if episode_id is not None:
                metadata[str(episode_id)] = row
    return metadata


def _auto_dreamer_step_to_standard(row: dict[str, Any]) -> dict[str, Any] | None:
    action = row.get("action_taken")
    if action is None:
        action = row.get("action")
    if action is None:
        action = row.get("model_output_raw")
    if action is None:
        return None
    done = _coerce_bool(row.get("done"))
    return {
        "t": int(row.get("step_idx", 0) or 0),
        "observation_text": str(row.get("observation_raw") or row.get("observation") or ""),
        "action_text": _action_command_text(str(action)),
        "next_observation_text": None
        if row.get("env_feedback_raw") is None
        else str(row.get("env_feedback_raw")),
        "done": done,
        "reward": _coerce_float(row.get("reward")),
    }


def _auto_dreamer_record_from_steps(
    dataset_name: str,
    source_root: Path,
    source_path: Path,
    episode_rows: list[dict[str, Any]],
    episode_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not episode_rows:
        return None
    first = episode_rows[0]
    episode_id = str(first.get("episode_id") or "")
    metadata = episode_metadata.get(episode_id, {})
    rel_path = source_path.relative_to(source_root)
    environment = _environment_from_row("auto_dreamer", first, rel_path)
    raw_steps = [_auto_dreamer_step_to_standard(row) for row in episode_rows]
    steps = [step for step in raw_steps if step is not None]
    if not steps:
        return None
    steps.sort(key=lambda step: int(step.get("t", 0)))
    normalized_steps = []
    for idx, step in enumerate(steps):
        mapping = map_action_to_pseudo_skill(environment, step["action_text"])
        normalized_steps.append(
            {
                "t": int(step.get("t", idx)),
                "observation_text": str(step.get("observation_text") or ""),
                "action_text": str(step.get("action_text") or ""),
                "next_observation_text": step.get("next_observation_text"),
                "done": step.get("done"),
                "reward": step.get("reward"),
                "pseudo_skill_id": mapping.pseudo_skill_id,
                "pseudo_skill_low_confidence": mapping.low_confidence,
            }
        )
    success = _success(first)
    if success is None:
        success = _success(metadata)
    reward = _coerce_float(metadata.get("total_reward"))
    if reward is None:
        reward = _reward(first)
    return {
        "dataset": dataset_name,
        "dataset_key": "auto_dreamer",
        "source_path": str(rel_path),
        "split": str(first.get("split") or metadata.get("split") or _split_from_path(rel_path)),
        "environment": environment,
        "task_id": _task_id(first, episode_id or f"auto_dreamer:{rel_path}"),
        "task_type": _task_type(first, environment),
        "success": success,
        "reward": reward,
        "steps": normalized_steps,
        "metadata": {
            "episode_id": episode_id,
            "game_file": first.get("game_file") or metadata.get("game_file"),
            "source": first.get("source") or metadata.get("source"),
            "model_name": first.get("model_name") or metadata.get("model_name"),
        },
    }


def _iter_auto_dreamer_step_records(
    root: Path,
    dataset_name: str,
    max_records: int | None,
) -> Iterable[dict[str, Any]]:
    episode_metadata = _auto_dreamer_episode_metadata(root)
    emitted = 0
    for source_path in sorted(root.rglob("steps.jsonl")):
        current_episode: str | None = None
        current_rows: list[dict[str, Any]] = []

        def flush() -> dict[str, Any] | None:
            return _auto_dreamer_record_from_steps(dataset_name, root, source_path, current_rows, episode_metadata)

        for row in _iter_json_records(source_path):
            episode_id = str(row.get("episode_id") or f"{source_path}:{row.get('step_idx', len(current_rows))}")
            if current_episode is None:
                current_episode = episode_id
            if episode_id != current_episode:
                record = flush()
                if record is not None:
                    yield record
                    emitted += 1
                    if max_records is not None and emitted >= max_records:
                        return
                current_episode = episode_id
                current_rows = []
            current_rows.append(row)
        record = flush()
        if record is not None:
            yield record
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


def import_aux_trajectory_datasets(
    dataset_roots: dict[str, str | Path],
    output_dir: str | Path,
    report_dir: str | Path,
    max_records_per_dataset: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    trajectories: list[dict[str, Any]] = []
    pseudo_skills: dict[str, dict[str, Any]] = {}
    dataset_stats: dict[str, Any] = {}
    missing_fields: dict[str, Counter[str]] = {}
    low_confidence_count = 0

    for dataset_key, root_value in dataset_roots.items():
        root = Path(root_value)
        dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
        schema = build_aux_trajectory_schema_report(dataset_name, dataset_key, root)
        write_json(report_dir / f"{dataset_key}_schema_report.json", schema)
        count = 0
        missing = Counter()
        if dataset_key == "auto_dreamer":
            record_iter = _iter_auto_dreamer_step_records(root, dataset_name, max_records_per_dataset)
            for record in record_iter:
                if record["success"] is None:
                    missing["success"] += 1
                if record["reward"] is None:
                    missing["reward"] += 1
                for step in record["steps"]:
                    if step["next_observation_text"] is None:
                        missing["next_observation_text"] += 1
                    if step["reward"] is None:
                        missing["step_reward"] += 1
                    mapping = map_action_to_pseudo_skill(record["environment"], step["action_text"])
                    if step["pseudo_skill_low_confidence"]:
                        low_confidence_count += 1
                    pseudo_skills.setdefault(
                        mapping.pseudo_skill_id,
                        {
                            "skill_id": mapping.pseudo_skill_id,
                            "name": mapping.name,
                            "description": mapping.description,
                            "source": "auxiliary_pseudo_skill",
                            "environment": record["environment"],
                            "low_confidence_mapping": mapping.low_confidence,
                            "not_skillsbench_skill": True,
                        },
                    )
                trajectories.append(record)
                count += 1
            if count == 0:
                for source_path in _iter_jsonish_files(root):
                    try:
                        rows = _iter_json_records(source_path)
                        for row_idx, row in enumerate(rows):
                            if max_records_per_dataset is not None and count >= max_records_per_dataset:
                                break
                            record = _normalize_row(dataset_key, dataset_name, root, source_path, row, row_idx)
                            if record is None:
                                missing["steps"] += 1
                                continue
                            if record["success"] is None:
                                missing["success"] += 1
                            if record["reward"] is None:
                                missing["reward"] += 1
                            for step in record["steps"]:
                                if step["next_observation_text"] is None:
                                    missing["next_observation_text"] += 1
                                if step["reward"] is None:
                                    missing["step_reward"] += 1
                                mapping = map_action_to_pseudo_skill(record["environment"], step["action_text"])
                                if step["pseudo_skill_low_confidence"]:
                                    low_confidence_count += 1
                                pseudo_skills.setdefault(
                                    mapping.pseudo_skill_id,
                                    {
                                        "skill_id": mapping.pseudo_skill_id,
                                        "name": mapping.name,
                                        "description": mapping.description,
                                        "source": "auxiliary_pseudo_skill",
                                        "environment": record["environment"],
                                        "low_confidence_mapping": mapping.low_confidence,
                                        "not_skillsbench_skill": True,
                                    },
                                )
                            trajectories.append(record)
                            count += 1
                        if max_records_per_dataset is not None and count >= max_records_per_dataset:
                            break
                    except Exception as exc:  # noqa: BLE001
                        missing["read_errors"] += 1
                        schema.setdefault("errors", []).append({"path": str(source_path), "error": str(exc)})
        else:
            for source_path in _iter_jsonish_files(root):
                try:
                    rows = _iter_json_records(source_path)
                    for row_idx, row in enumerate(rows):
                        if max_records_per_dataset is not None and count >= max_records_per_dataset:
                            break
                        record = _normalize_row(dataset_key, dataset_name, root, source_path, row, row_idx)
                        if record is None:
                            missing["steps"] += 1
                            continue
                        if record["success"] is None:
                            missing["success"] += 1
                        if record["reward"] is None:
                            missing["reward"] += 1
                        for step in record["steps"]:
                            if step["next_observation_text"] is None:
                                missing["next_observation_text"] += 1
                            if step["reward"] is None:
                                missing["step_reward"] += 1
                            mapping = map_action_to_pseudo_skill(record["environment"], step["action_text"])
                            if step["pseudo_skill_low_confidence"]:
                                low_confidence_count += 1
                            pseudo_skills.setdefault(
                                mapping.pseudo_skill_id,
                                {
                                    "skill_id": mapping.pseudo_skill_id,
                                    "name": mapping.name,
                                    "description": mapping.description,
                                    "source": "auxiliary_pseudo_skill",
                                    "environment": record["environment"],
                                    "low_confidence_mapping": mapping.low_confidence,
                                    "not_skillsbench_skill": True,
                                },
                            )
                        trajectories.append(record)
                        count += 1
                    if max_records_per_dataset is not None and count >= max_records_per_dataset:
                        break
                except Exception as exc:  # noqa: BLE001
                    missing["read_errors"] += 1
                    schema.setdefault("errors", []).append({"path": str(source_path), "error": str(exc)})
        dataset_stats[dataset_key] = {"trajectory_count": count, "schema_status": schema["status"]}
        missing_fields[dataset_key] = missing

    split_counts = Counter(record["split"] for record in trajectories)
    environment_counts = Counter(record["environment"] for record in trajectories)
    step_count = sum(len(record["steps"]) for record in trajectories)
    success_present = sum(1 for record in trajectories if record["success"] is not None)
    reward_present = sum(1 for record in trajectories if record["reward"] is not None)
    manifest = {
        "status": "ok" if trajectories else "missing",
        "data_role": "auxiliary_trajectory_not_skillsbench_clean_router",
        "source_datasets": DATASET_NAMES,
        "output_dir": str(output_dir),
        "dataset_stats": dataset_stats,
        "record_counts": {
            "trajectories": len(trajectories),
            "steps": step_count,
        },
        "split_counts": dict(sorted(split_counts.items())),
        "environment_counts": dict(sorted(environment_counts.items())),
        "coverage": {
            "success": {"present": success_present, "missing": len(trajectories) - success_present},
            "reward": {"present": reward_present, "missing": len(trajectories) - reward_present},
        },
        "missing_fields": {
            key: dict(sorted(counter.items())) for key, counter in sorted(missing_fields.items())
        },
        "pseudo_skill_count": len(pseudo_skills),
        "low_confidence_mapping_count": low_confidence_count,
        "forbidden_outputs": {
            "skillsbench_successful_trajectories": "not_written",
            "skillsbench_harness_results": "not_written",
            "clean_router_train_jsonl": "not_written",
        },
        "note": "Auxiliary trajectories may train transition/belief/STOP/imitation objectives; they are not SkillsBench clean_router_data.",
    }
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_jsonl(output_dir / "pseudo_skills.jsonl", sorted(pseudo_skills.values(), key=lambda row: row["skill_id"]))
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_aux_training_rows(data_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_root = Path(data_root)
    skills_path = data_root / "pseudo_skills.jsonl"
    trajectories_path = data_root / "trajectories.jsonl"
    if not skills_path.exists() or not trajectories_path.exists():
        raise FileNotFoundError(f"auxiliary trajectory data is missing under {data_root}")
    skills = [json.loads(line) for line in skills_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    trajectories = [
        json.loads(line)
        for line in trajectories_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return skills, trajectories
