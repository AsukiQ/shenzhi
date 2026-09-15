from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from clstr.data import ExecutionState, VerifiedPair, read_jsonl, verified_pair_to_dict


SPLITS = ("train", "dev", "test")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _task_ids_from_audit(path: Path) -> set[str]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return {str(item) for item in payload}
    return {str(item) for item in payload.get("task_ids", [])}


def _load_audit_splits(audit_dir: Path) -> tuple[dict[str, set[str]], set[str]]:
    splits = {
        split: _task_ids_from_audit(audit_dir / f"skillsbench_{split}_task_ids.json")
        for split in SPLITS
    }
    eval_query_ids = _task_ids_from_audit(audit_dir / "skillrouter_eval_query_ids.json")
    return splits, eval_query_ids


def _success_from_result(result: dict[str, Any], reward_text: str | None) -> bool:
    if isinstance(result.get("success"), bool):
        return result["success"]
    reward = result.get("reward")
    if reward is None and isinstance(result.get("rewards"), dict):
        reward = result["rewards"].get("reward")
    if reward is None and isinstance(result.get("rewards"), list) and result["rewards"]:
        reward = result["rewards"][-1]
    if reward is None and reward_text:
        try:
            reward = float(reward_text.strip())
        except ValueError:
            reward = None
    return bool(reward is not None and float(reward) > 0)


def _condition_from_run(path_text: str, config: dict[str, Any]) -> str:
    lower = path_text.lower().replace("_", "-")
    if "without-skills" in lower or "withoutskills" in lower:
        return "without-skills"
    if config.get("skills_dir") or "with-skills" in lower or "withskills" in lower:
        return "with-skills"
    return "unknown"


def _task_id_from_run(run_dir: Path, result: dict[str, Any], config: dict[str, Any]) -> str:
    if result.get("task_id") is not None:
        return str(result["task_id"])
    if result.get("task_name") is not None:
        return str(result["task_name"])
    task_path = str(config.get("task_path") or "")
    if task_path:
        return Path(task_path).name
    return run_dir.name.split("__", 1)[0]


def _load_skill_maps(skill_pool_path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    records = read_jsonl(skill_pool_path)
    by_id = {str(record["skill_id"]): record for record in records if record.get("skill_id")}
    by_task_name: dict[tuple[str, str], str] = {}
    for skill_id, record in by_id.items():
        task_id = str(record.get("task_id") or skill_id.split("/", 1)[0])
        names = {
            str(record.get("name", "")),
            str(record.get("skill_id", "")),
            str(record.get("skill_id", "")).split("/")[-1],
        }
        for name in names:
            if name:
                by_task_name[(task_id, name)] = skill_id
    return by_id, by_task_name


def _resolve_skill_id(
    raw: Any,
    task_id: str,
    skill_by_id: dict[str, dict[str, Any]],
    skill_by_task_name: dict[tuple[str, str], str],
) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    if value in skill_by_id:
        return value
    if (task_id, value) in skill_by_task_name:
        return skill_by_task_name[(task_id, value)]
    candidate = f"{task_id}/{value}"
    if candidate in skill_by_id:
        return candidate
    return None


def _candidate_values(event: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ("raw_topk", "raw_candidates"):
        if isinstance(event.get(key), list):
            values.extend(event[key])
    return values


def _texts_from_content(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            texts.append(value["text"])
        if isinstance(value.get("content"), (dict, list, str)):
            texts.extend(_texts_from_content(value["content"]))
    if isinstance(value, list):
        for item in value:
            texts.extend(_texts_from_content(item))
    return texts


def _skill_from_launch_text(event: dict[str, Any]) -> str | None:
    if str(event.get("title", "")).lower() != "skill":
        return None
    for text in _texts_from_content(event.get("content")):
        match = re.search(r"Launching skill:\s*([^\n`]+)", text)
        if match:
            return match.group(1).strip()
    return None


def _extract_skill_steps(
    trajectory: list[dict[str, Any]],
    task_id: str,
    skill_by_id: dict[str, dict[str, Any]],
    skill_by_task_name: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], int]:
    steps: list[dict[str, Any]] = []
    missing = 0

    def consume_event(event: dict[str, Any]) -> None:
        nonlocal missing
        raw_skill = event.get("skill_id") or event.get("skill_name") or event.get("skill") or _skill_from_launch_text(event)
        if raw_skill is None and isinstance(event.get("tool_call"), dict):
            raw_skill = event["tool_call"].get("skill_id") or event["tool_call"].get("name")
        if raw_skill is None:
            return
        skill_id = _resolve_skill_id(raw_skill, task_id, skill_by_id, skill_by_task_name)
        if skill_id is None:
            missing += 1
            return
        step: dict[str, Any] = {"skill_id": skill_id}
        if event.get("observation") is not None:
            step["observation"] = str(event["observation"])
        for key in ("raw_topk", "raw_candidates"):
            if isinstance(event.get(key), list):
                mapped = [
                    resolved
                    for item in event[key]
                    if (resolved := _resolve_skill_id(item, task_id, skill_by_id, skill_by_task_name)) is not None
                ]
                if mapped:
                    step[key] = mapped
        steps.append(step)

    for event in trajectory:
        if not isinstance(event, dict):
            continue
        nested_steps = event.get("steps")
        if isinstance(nested_steps, list):
            for nested in nested_steps:
                if isinstance(nested, dict):
                    consume_event(nested)
            continue
        consume_event(event)
    return steps, missing


def _read_run(run_dir: Path, result_path: Path, dataset_root: Path) -> dict[str, Any]:
    result = _read_json(result_path)
    config_path = run_dir / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    reward_path = run_dir / "verifier" / "reward.txt"
    reward_text = reward_path.read_text(encoding="utf-8").strip() if reward_path.exists() else ""
    trajectory_path = run_dir / "trajectory" / "acp_trajectory.jsonl"
    trajectory = _read_jsonl_lenient(trajectory_path) if trajectory_path.exists() else []
    task_id = _task_id_from_run(run_dir, result, config)
    rel_run_dir = run_dir.relative_to(dataset_root)
    return {
        "task_id": task_id,
        "agent": str(result.get("agent") or config.get("agent") or ""),
        "model": str(result.get("model") or config.get("model") or ""),
        "condition": _condition_from_run(str(rel_run_dir), config),
        "success": _success_from_result(result, reward_text),
        "status": "completed" if _success_from_result(result, reward_text) else "failed",
        "result_path": str(result_path.relative_to(dataset_root)),
        "run_path": str(rel_run_dir),
        "trajectory": trajectory,
        "verifier_output": reward_text,
        "error": result.get("error") or result.get("verifier_error"),
    }


def _discover_runs(dataset_root: Path) -> list[dict[str, Any]]:
    return [
        _read_run(path.parent, path, dataset_root)
        for path in sorted(dataset_root.rglob("result.json"))
    ]


def _split_for_task(task_id: str, splits: dict[str, set[str]]) -> str:
    for split, ids in splits.items():
        if task_id in ids:
            return split
    return "unknown"


def _schema_report(runs: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    return {
        "status": "ok" if runs else "missing",
        "dataset_root": str(dataset_root),
        "run_count": len(runs),
        "task_count": len({run["task_id"] for run in runs}),
        "agent_counts": dict(Counter(run["agent"] for run in runs)),
        "model_counts": dict(Counter(run["model"] for run in runs)),
        "condition_counts": dict(Counter(run["condition"] for run in runs)),
        "sample_runs": [run["run_path"] for run in runs[:10]],
    }


def import_historical_skillsbench_trajectories(
    dataset_root: str | Path,
    skillsbench_root: str | Path,
    audit_dir: str | Path,
    output_dir: str | Path,
    harness_results_path: str | Path,
    successful_trajectories_path: str | Path,
    clstr_harness_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    skillsbench_root = Path(skillsbench_root)
    audit_dir = Path(audit_dir)
    output_dir = Path(output_dir)
    harness_results_path = Path(harness_results_path)
    successful_trajectories_path = Path(successful_trajectories_path)
    clstr_harness_output_dir = Path(clstr_harness_output_dir) if clstr_harness_output_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_by_id, skill_by_task_name = _load_skill_maps(skillsbench_root / "skill_pool.jsonl")
    splits, eval_query_ids = _load_audit_splits(audit_dir)
    runs = _discover_runs(dataset_root) if dataset_root.exists() else []
    schema_report = _schema_report(runs, dataset_root)
    _write_json(output_dir / "schema_report.json", schema_report)

    excluded = Counter()
    split_seen = {split: 0 for split in (*SPLITS, "unknown")}
    split_kept = {split: 0 for split in (*SPLITS, "unknown")}
    split_excluded = {split: 0 for split in (*SPLITS, "unknown")}
    split_exclusion_reasons = {split: Counter() for split in (*SPLITS, "unknown")}
    harness_records: list[dict[str, Any]] = []
    successful_records: list[dict[str, Any]] = []

    def exclude(split: str, reason: str) -> None:
        excluded[reason] += 1
        split_excluded[split] += 1
        split_exclusion_reasons[split][reason] += 1

    for run in runs:
        task_id = run["task_id"]
        split = _split_for_task(task_id, splits)
        split_seen[split] += 1
        if task_id in eval_query_ids:
            exclude(split, "skillrouter_eval_overlap")
            continue
        if split != "train":
            exclude(split, "not_train_split")
            continue
        if run["condition"] != "with-skills":
            exclude(split, "without_skills")
            continue
        if not run["success"]:
            exclude(split, "not_successful")
            continue
        steps, missing = _extract_skill_steps(run["trajectory"], task_id, skill_by_id, skill_by_task_name)
        if missing or not steps:
            exclude(split, "missing_skill_mapping")
            continue

        split_kept[split] += 1
        harness_records.append(
            {
                "task_id": task_id,
                "success": True,
                "status": "completed",
                "result_path": run["result_path"],
                "trajectory_source_path": run["run_path"],
                "trajectory": [{"steps": steps, "source": "skillsbench_historical_agent_trajectory"}],
                "verifier_output": run["verifier_output"],
                "error": None,
                "agent": run["agent"],
                "model": run["model"],
                "condition": run["condition"],
                "dataset": "benchflow/skillsbench-trajectories-apr2026",
            }
        )
        successful_records.append(
            {
                "task_id": task_id,
                "success": True,
                "steps": steps,
                "source": "skillsbench_historical_agent_trajectory",
                "status": "completed",
                "agent": run["agent"],
                "model": run["model"],
                "trajectory_source_path": run["run_path"],
            }
        )

    if harness_records:
        _write_jsonl(harness_results_path, harness_records)
    elif harness_results_path.exists():
        harness_results_path.unlink()
    if successful_records:
        _write_jsonl(successful_trajectories_path, successful_records)
    elif successful_trajectories_path.exists():
        successful_trajectories_path.unlink()

    if clstr_harness_output_dir is not None:
        clstr_harness_output_dir.mkdir(parents=True, exist_ok=True)
        if harness_records:
            _write_jsonl(clstr_harness_output_dir / "harness_results.jsonl", harness_records)
        if successful_records:
            _write_jsonl(clstr_harness_output_dir / "successful_trajectories.jsonl", successful_records)

    filter_report = {
        "dataset": "benchflow/skillsbench-trajectories-apr2026",
        "dataset_root": str(dataset_root),
        "total_runs": len(runs),
        "kept_train_successful_with_skills": len(successful_records),
        "excluded": {key: int(value) for key, value in sorted(excluded.items())},
        "split_seen": split_seen,
        "split_kept_for_training": split_kept,
        "split_excluded": split_excluded,
        "split_exclusion_reasons": {
            split: {key: int(value) for key, value in sorted(reasons.items())}
            for split, reasons in split_exclusion_reasons.items()
        },
        "harness_results_path": str(harness_results_path),
        "successful_trajectories_path": str(successful_trajectories_path) if successful_records else None,
    }
    _write_json(output_dir / "filter_report.json", filter_report)
    manifest = {"schema": schema_report, "filter": filter_report}
    _write_json(output_dir / "historical_import_manifest.json", manifest)
    _write_json(skillsbench_root / "historical_import_manifest.json", manifest)
    return manifest


def _raw_candidate_values_for_pair(current_step: dict[str, Any], next_step: dict[str, Any]) -> list[Any]:
    for source in (next_step, current_step):
        values = _candidate_values(source)
        if values:
            return values
    return []


def build_verified_pairs_from_successful_trajectories(
    successful_trajectories_path: str | Path,
    skill_pool_path: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    successful_trajectories_path = Path(successful_trajectories_path)
    skill_pool_path = Path(skill_pool_path)
    output_jsonl = Path(output_jsonl)
    manifest_path = Path(manifest_path)
    skill_pool = read_jsonl(skill_pool_path)
    skill_id_to_idx = {str(record["skill_id"]): idx for idx, record in enumerate(skill_pool)}
    skill_by_id = {str(record["skill_id"]): record for record in skill_pool}
    skill_by_task_name = {
        (str(record.get("task_id") or str(record["skill_id"]).split("/", 1)[0]), str(record.get("name"))): str(record["skill_id"])
        for record in skill_pool
        if record.get("name")
    }
    rows = read_jsonl(successful_trajectories_path) if successful_trajectories_path.exists() else []
    output_rows: list[dict[str, Any]] = []
    pair_candidate_count = 0
    missing_raw = 0
    missing_mapping = 0

    for row in rows:
        task_id = str(row.get("task_id"))
        query = str(row.get("query") or task_id)
        steps = row.get("steps", [])
        if not isinstance(steps, list):
            continue
        history: list[tuple[str, str]] = []
        for idx in range(len(steps) - 1):
            current = steps[idx]
            nxt = steps[idx + 1]
            if not isinstance(current, dict) or not isinstance(nxt, dict):
                continue
            pair_candidate_count += 1
            current_skill = _resolve_skill_id(current.get("skill_id") or current.get("skill"), task_id, skill_by_id, skill_by_task_name)
            next_skill = _resolve_skill_id(nxt.get("skill_id") or nxt.get("skill"), task_id, skill_by_id, skill_by_task_name)
            if current_skill is None or next_skill is None:
                missing_mapping += 1
                continue
            raw_values = _raw_candidate_values_for_pair(current, nxt)
            if not raw_values:
                missing_raw += 1
                history.append((current_skill, str(current.get("observation", ""))))
                continue
            raw_ids = [
                resolved
                for value in raw_values
                if (resolved := _resolve_skill_id(value, task_id, skill_by_id, skill_by_task_name)) is not None
            ]
            if not raw_ids:
                missing_raw += 1
                history.append((current_skill, str(current.get("observation", ""))))
                continue
            raw_candidate_indices = [skill_id_to_idx[skill_id] for skill_id in raw_ids if skill_id in skill_id_to_idx]
            next_idx = skill_id_to_idx[next_skill]
            was_in_raw_topk = next_idx in raw_candidate_indices
            candidates_next = list(raw_candidate_indices)
            if not was_in_raw_topk:
                candidates_next.append(next_idx)
            state = ExecutionState(
                query=query,
                history=list(history),
                observation=str(current.get("observation", "")),
                artifact={},
                error=None,
            )
            pair = VerifiedPair(
                task_id=task_id,
                step_idx=idx,
                state_before=state,
                action_at_t=skill_id_to_idx[current_skill],
                obs_at_t=str(current.get("observation", "")),
                candidates_next=candidates_next,
                a_next_plus=next_idx,
                was_in_raw_topk=was_in_raw_topk,
            )
            output_rows.append(verified_pair_to_dict(pair))
            history.append((current_skill, str(current.get("observation", ""))))

    if output_rows:
        _write_jsonl(output_jsonl, output_rows)
    elif output_jsonl.exists():
        output_jsonl.unlink()
    status = "ok" if output_rows else "missing_raw_candidates"
    manifest = {
        "status": status,
        "successful_trajectories_path": str(successful_trajectories_path),
        "skill_pool_path": str(skill_pool_path),
        "output_jsonl": str(output_jsonl) if output_rows else None,
        "pair_candidate_count": pair_candidate_count,
        "emitted_pair_count": len(output_rows),
        "missing_raw_candidate_count": missing_raw,
        "missing_skill_mapping_count": missing_mapping,
        "note": "verified pairs are emitted only when raw_topk/raw_candidates are explicit; candidates_next may include injected positives for training, while was_in_raw_topk records raw recall truth",
    }
    _write_json(manifest_path, manifest)
    return manifest
