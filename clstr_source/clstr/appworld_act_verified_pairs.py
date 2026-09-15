from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from clstr.appworld_routing import (
    load_api_catalog,
    load_skill_pool,
    match_api_call,
    read_json,
    read_jsonl,
    score_skill_for_task,
    write_json,
    write_jsonl,
)
from clstr.appworld_routing import _skill_api_refs as skill_api_refs
from clstr.candidate_utils import inject_positive_candidate
from clstr.data import ExecutionState, ReplayStep, VerifiedPair, verified_pair_to_dict


@dataclass(frozen=True)
class _MappedStep:
    skill_id: str
    skill_idx: int
    skill_name: str
    api_ref: str
    method: str
    url: str
    observation: str


def _unique(items: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[Any] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _task_query(task: dict[str, Any]) -> str:
    return str(task.get("query") or task.get("instruction_text") or task.get("instruction") or "")


def _task_with_ref_context(task: dict[str, Any], api_ref: str) -> dict[str, Any]:
    refs = _unique([api_ref, *[str(ref) for ref in task.get("api_refs", [])]])
    apps = _unique([api_ref.split(".", 1)[0], *[str(app) for app in task.get("api_apps", [])]])
    return {**task, "api_refs": refs, "api_apps": apps}


def _build_ref_index(skills: Sequence[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    by_ref: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, skill in enumerate(skills):
        for api_ref in skill_api_refs(skill):
            by_ref.setdefault(api_ref, []).append((idx, skill))
    return by_ref


def _best_skill_for_ref(
    *,
    task: dict[str, Any],
    api_ref: str,
    skills_for_ref: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[int, dict[str, Any]] | None:
    if not skills_for_ref:
        return None
    task_for_ref = _task_with_ref_context(task, api_ref)
    return max(
        skills_for_ref,
        key=lambda item: score_skill_for_task(task_for_ref, item[1], item[0]),
    )


def _state_for_history(task: dict[str, Any], history: list[tuple[str, str]]) -> ExecutionState:
    return ExecutionState(
        query=_task_query(task),
        history=list(history),
        observation=history[-1][1] if history else "",
        artifact={
            "source": "appworld_train_oracle_api_calls_mapped_to_skillx",
            "task_id": task.get("task_id") or task.get("query_id"),
            "split": task.get("split"),
        },
        error=None,
    )


def _map_api_calls_to_skill_steps(
    *,
    appworld_root: Path,
    task: dict[str, Any],
    skills: Sequence[dict[str, Any]],
    ref_index: dict[str, list[tuple[int, dict[str, Any]]]],
    compress_consecutive: bool,
) -> tuple[list[_MappedStep], Counter[str]]:
    task_id = str(task.get("task_id") or task.get("query_id"))
    calls = read_json(appworld_root / "data" / "tasks" / task_id / "ground_truth" / "api_calls.json", default=[]) or []
    catalog = load_api_catalog(appworld_root)
    steps: list[_MappedStep] = []
    skipped: Counter[str] = Counter()

    for call in calls:
        ref = match_api_call(call, catalog)
        if ref is None:
            skipped["unmatched_api_call"] += 1
            continue
        if ref.app in {"admin", "api_docs", "supervisor"}:
            skipped["non_skill_app"] += 1
            continue
        selected = _best_skill_for_ref(
            task=task,
            api_ref=ref.ref,
            skills_for_ref=ref_index.get(ref.ref, []),
        )
        if selected is None:
            skipped["missing_skill_for_api_ref"] += 1
            continue

        skill_idx, skill = selected
        skill_id = str(skill.get("skill_id") or skill_idx)
        if compress_consecutive and steps and steps[-1].skill_id == skill_id:
            skipped["compressed_consecutive_duplicate"] += 1
            continue
        method = str(call.get("method") or ref.method).upper()
        url = str(call.get("url") or call.get("path") or ref.path)
        steps.append(
            _MappedStep(
                skill_id=skill_id,
                skill_idx=skill_idx,
                skill_name=str(skill.get("name") or skill_id),
                api_ref=ref.ref,
                method=method,
                url=url,
                observation=f"AppWorld oracle API call: {ref.ref} {method} {url}",
            )
        )
    return steps, skipped


def _candidate_indices(
    *,
    target: int,
    task: dict[str, Any],
    skills: Sequence[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    sequence_steps: Sequence[_MappedStep],
    top_k: int,
) -> tuple[list[int], bool]:
    raw_ids = [str(skill_id) for skill_id in task.get("positive_skill_ids", [])]
    raw_indices = [skill_id_to_idx[skill_id] for skill_id in raw_ids if skill_id in skill_id_to_idx]
    raw_indices.extend(step.skill_idx for step in sequence_steps)

    required_apps = {str(app) for app in task.get("required_apps", [])}
    if required_apps:
        for idx, skill in enumerate(skills):
            text = " ".join(
                [
                    str(skill.get("skill_id", "")),
                    str(skill.get("name", "")),
                    str(skill.get("executor_desc", "")),
                ]
            ).lower()
            if any(app.lower() in text or app.lower().replace("_", " ") in text for app in required_apps):
                raw_indices.append(idx)

    raw_indices.extend(idx for idx in range(len(skills)))
    raw_unique = [int(idx) for idx in _unique(raw_indices)]
    was_in_raw_topk = target in raw_unique[:top_k]
    return inject_positive_candidate(raw_unique, target, top_k), was_in_raw_topk


def _verified_pairs_for_sequence(
    *,
    task: dict[str, Any],
    skills: Sequence[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    sequence_steps: Sequence[_MappedStep],
    top_k: int,
) -> list[VerifiedPair]:
    pairs: list[VerifiedPair] = []
    states: list[ExecutionState] = []
    history: list[tuple[str, str]] = []
    for step in sequence_steps:
        states.append(_state_for_history(task, history))
        history.append((step.skill_name, step.observation))

    replay_steps = [
        ReplayStep(x=states[pos], skill_idx=step.skill_idx, obs=step.observation)
        for pos, step in enumerate(sequence_steps)
    ]
    for pos in range(len(sequence_steps) - 1):
        current = sequence_steps[pos]
        nxt = sequence_steps[pos + 1]
        candidates, was_in_raw_topk = _candidate_indices(
            target=nxt.skill_idx,
            task=task,
            skills=skills,
            skill_id_to_idx=skill_id_to_idx,
            sequence_steps=sequence_steps,
            top_k=top_k,
        )
        pairs.append(
            VerifiedPair(
                task_id=task.get("task_id") or task.get("query_id"),
                step_idx=pos,
                state_before=states[pos],
                action_at_t=current.skill_idx,
                obs_at_t=current.observation,
                candidates_next=candidates,
                a_next_plus=nxt.skill_idx,
                was_in_raw_topk=was_in_raw_topk,
                replay_prefix=replay_steps[:pos],
                replay_prefix_is_essential_subsequence=True,
            )
        )
    return pairs


def build_appworld_act_verified_pairs(
    *,
    appworld_root: str | Path,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path | None = None,
    top_k: int = 20,
    max_tasks: int | None = None,
    compress_consecutive: bool = True,
) -> dict[str, Any]:
    appworld_root = Path(appworld_root)
    tasks = read_jsonl(tasks_path)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    skills = load_skill_pool(skill_pool_path)
    skill_id_to_idx = {
        str(skill.get("skill_id") or idx): idx
        for idx, skill in enumerate(skills)
    }
    ref_index = _build_ref_index(skills)

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    tasks_with_sequences = 0
    split_values: set[str] = set()
    sequence_lengths: list[int] = []

    for task in tasks:
        split_values.add(str(task.get("split") or ""))
        if str(task.get("split") or "train") != "train":
            skipped["non_train_task"] += 1
            continue
        sequence, sequence_skipped = _map_api_calls_to_skill_steps(
            appworld_root=appworld_root,
            task=task,
            skills=skills,
            ref_index=ref_index,
            compress_consecutive=compress_consecutive,
        )
        skipped.update(sequence_skipped)
        if len(sequence) < 2:
            skipped["sequence_too_short"] += 1
            continue
        tasks_with_sequences += 1
        sequence_lengths.append(len(sequence))
        pairs = _verified_pairs_for_sequence(
            task=task,
            skills=skills,
            skill_id_to_idx=skill_id_to_idx,
            sequence_steps=sequence,
            top_k=top_k,
        )
        rows.extend(verified_pair_to_dict(pair) for pair in pairs)

    write_count = write_jsonl(output_jsonl, rows)
    dev_test_used = any(split not in {"", "train"} for split in split_values)
    report = {
        "status": "ok" if write_count > 0 else "blocked",
        "appworld_root": str(appworld_root),
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "output_jsonl": str(output_jsonl),
        "task_count": len(tasks),
        "skill_count": len(skills),
        "tasks_with_skill_sequences": tasks_with_sequences,
        "verified_pair_count": write_count,
        "top_k": int(top_k),
        "compress_consecutive": bool(compress_consecutive),
        "dev_test_used_for_training": dev_test_used,
        "supervision_source": "AppWorld train split ground_truth/api_calls.json mapped to SkillX skills by API refs.",
        "avg_sequence_length": round(sum(sequence_lengths) / max(len(sequence_lengths), 1), 3),
        "skipped": dict(sorted(skipped.items())),
    }
    if manifest_path is not None:
        write_json(manifest_path, report)
    return report
