from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


_GENERIC_TOKENS = {
    "a",
    "all",
    "album",
    "albums",
    "and",
    "any",
    "app",
    "apps",
    "are",
    "been",
    "by",
    "comma",
    "across",
    "day",
    "days",
    "from",
    "for",
    "give",
    "have",
    "how",
    "i",
    "in",
    "library",
    "libraries",
    "list",
    "many",
    "me",
    "more",
    "my",
    "of",
    "on",
    "or",
    "playlist",
    "playlists",
    "separated",
    "song",
    "songs",
    "spotify",
    "the",
    "there",
    "that",
    "title",
    "titles",
    "to",
    "top",
    "user",
    "with",
    "which",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", str(text).lower()):
        if raw.isdigit():
            continue
        if not raw or raw in _GENERIC_TOKENS:
            continue
        variants = {raw}
        if raw.endswith("s") and len(raw) > 3:
            variants.add(raw[:-1])
        if raw.endswith("ed") and len(raw) > 4:
            variants.add(raw[:-2])
        if raw.endswith("ing") and len(raw) > 5:
            variants.add(raw[:-3])
        tokens.update(token for token in variants if token and token not in _GENERIC_TOKENS)
    return tokens


def _skill_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or fallback)


def _is_appworld_compatible(row: dict[str, Any], skill_id: str) -> bool:
    return bool(
        row.get("appworld_executor_compatible") is True
        or str(row.get("executor_domain") or "").lower() == "appworld"
        or str(skill_id).startswith("skillx/appworld/")
    )


def _skill_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("name") or ""),
            str(row.get("description") or ""),
            str(row.get("executor_desc") or ""),
            str(row.get("body") or ""),
        ]
    )


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_skill_id")
    if values is None:
        values = row.get("skill_id")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if item]


def _task_instruction(row: dict[str, Any]) -> str:
    return str(row.get("instruction_text") or row.get("user_goal") or row.get("query") or "")


def _semantic_scores(
    *,
    instruction: str,
    skill_rows: Sequence[dict[str, Any]],
    skill_ids: Sequence[str],
) -> list[tuple[int, str, dict[str, Any], list[str]]]:
    query_tokens = _tokenize(instruction)
    scored: list[tuple[int, str, dict[str, Any], list[str]]] = []
    for skill_id, skill in zip(skill_ids, skill_rows):
        overlap = sorted(query_tokens & _tokenize(_skill_text(skill)))
        scored.append((len(overlap), str(skill_id), skill, overlap))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored


def _audit_one_task(
    task: dict[str, Any],
    *,
    skill_by_id: dict[str, dict[str, Any]],
    candidate_skills: Sequence[dict[str, Any]],
    candidate_skill_ids: Sequence[str],
    top_k: int,
    score_margin: int,
) -> dict[str, Any]:
    instruction = _task_instruction(task)
    positives = _positive_skill_ids(task)
    scores = _semantic_scores(instruction=instruction, skill_rows=candidate_skills, skill_ids=candidate_skill_ids)
    best = scores[: max(1, int(top_k))]
    positive_scores = [
        (
            len(_tokenize(instruction) & _tokenize(_skill_text(skill_by_id[skill_id]))),
            skill_id,
            sorted(_tokenize(instruction) & _tokenize(_skill_text(skill_by_id[skill_id]))),
        )
        for skill_id in positives
        if skill_id in skill_by_id
    ]
    positive_best = max((item[0] for item in positive_scores), default=0)
    semantic_best = best[0][0] if best else 0
    best_ids = [item[1] for item in best]
    flags: list[str] = []
    if positives and not (set(best_ids) & set(positives)):
        flags.append("semantic_best_misses_positive")
    if semantic_best >= positive_best + int(score_margin) and positives:
        flags.append("semantic_best_scores_higher_than_positive")
    if positive_best <= 0 and positives:
        flags.append("positive_has_no_intent_token_overlap")
    return {
        "task_id": str(task.get("task_id") or task.get("query_id") or ""),
        "query_id": str(task.get("query_id") or task.get("task_id") or ""),
        "instruction_text": instruction,
        "task_semantic_tokens": sorted(_tokenize(instruction)),
        "positive_skill_ids": positives,
        "positive_skill_names": [str((skill_by_id.get(skill_id) or {}).get("name") or skill_id) for skill_id in positives],
        "positive_best_semantic_overlap": positive_best,
        "positive_semantic_scores": [
            {"skill_id": skill_id, "overlap": score, "overlap_tokens": tokens}
            for score, skill_id, tokens in positive_scores
        ],
        "semantic_best_skill_ids": best_ids,
        "semantic_best_skill_names": [str(item[2].get("name") or item[1]) for item in best],
        "semantic_best_overlap": semantic_best,
        "semantic_best_overlap_tokens": best[0][3] if best else [],
        "semantic_best_positive_overlap": bool(set(best_ids) & set(positives)),
        "flags": flags,
        "suspicious": bool(flags),
    }


def _audit_executor_runs(
    *,
    runs_path: str | Path,
    skill_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    runs = _read_jsonl(runs_path)
    flagged: list[dict[str, Any]] = []
    step_count = 0
    selected_better = 0
    for run in runs:
        instruction = _task_instruction(run)
        if not instruction:
            instruction = str(run.get("user_goal") or "")
        query_tokens = _tokenize(instruction)
        for step in run.get("steps") or []:
            if not isinstance(step, dict):
                continue
            step_count += 1
            selected = [str(item) for item in step.get("selected_skill_ids") or []]
            positives = [str(item) for item in step.get("positive_skill_ids") or []]
            selected_best = max((len(query_tokens & _tokenize(_skill_text(skill_by_id.get(skill_id, {})))) for skill_id in selected), default=0)
            positive_best = max((len(query_tokens & _tokenize(_skill_text(skill_by_id.get(skill_id, {})))) for skill_id in positives), default=0)
            hit = bool(set(selected) & set(positives)) if positives else None
            if hit is False and selected_best > positive_best:
                selected_better += 1
                flagged.append(
                    {
                        "task_id": str(run.get("task_id") or run.get("query_id") or ""),
                        "step_idx": int(step.get("step_idx", 0)),
                        "selected_skill_ids": selected,
                        "positive_skill_ids": positives,
                        "selected_best_semantic_overlap": selected_best,
                        "positive_best_semantic_overlap": positive_best,
                    }
                )
    return {
        "runs_path": str(runs_path),
        "step_count": step_count,
        "selected_better_than_positive_count": selected_better,
        "flagged_steps": flagged[:50],
    }


def audit_appworld_task_positive_labels(
    *,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_path: str | Path | None = None,
    runs_path: str | Path | None = None,
    max_tasks: int | None = None,
    top_k: int = 5,
    score_margin: int = 1,
) -> dict[str, Any]:
    tasks = _read_jsonl(tasks_path)
    if max_tasks is not None and int(max_tasks) > 0:
        tasks = tasks[: int(max_tasks)]
    raw_skills = _read_jsonl(skill_pool_path)
    skill_ids = [_skill_id(row, str(idx)) for idx, row in enumerate(raw_skills)]
    skill_by_id = {skill_id: row for skill_id, row in zip(skill_ids, raw_skills)}
    candidate_pairs = [
        (skill_id, row)
        for skill_id, row in zip(skill_ids, raw_skills)
        if _is_appworld_compatible(row, skill_id)
    ]
    candidate_skill_ids = [item[0] for item in candidate_pairs]
    candidate_skills = [item[1] for item in candidate_pairs]
    audited = [
        _audit_one_task(
            task,
            skill_by_id=skill_by_id,
            candidate_skills=candidate_skills,
            candidate_skill_ids=candidate_skill_ids,
            top_k=top_k,
            score_margin=score_margin,
        )
        for task in tasks
    ]
    suspicious = [row for row in audited if row["suspicious"]]
    flag_counts = Counter(flag for row in suspicious for flag in row["flags"])
    report: dict[str, Any] = {
        "status": "ok",
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "task_count": len(audited),
        "skill_count": len(raw_skills),
        "appworld_compatible_skill_count": len(candidate_skills),
        "suspicious_task_count": len(suspicious),
        "suspicious_task_rate": round(len(suspicious) / max(len(audited), 1), 6),
        "flag_counts": dict(sorted(flag_counts.items())),
        "suspicious_tasks": suspicious[:50],
        "top_k": int(top_k),
        "score_margin": int(score_margin),
        "interpretation": "CPU diagnostic for task-level AppWorld positives; suspicious rows should be reviewed before using step hit metrics or Stage4 rewards.",
    }
    if runs_path is not None and str(runs_path).strip():
        report["executor_run_alignment"] = _audit_executor_runs(runs_path=runs_path, skill_by_id=skill_by_id)
    if output_path is not None:
        _write_json(output_path, report)
    return report
