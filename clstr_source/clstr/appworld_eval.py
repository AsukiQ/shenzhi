from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from clstr.appworld_routing import read_jsonl, write_json, write_jsonl


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _skillrouter_query_text(task: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"query: {task.get('query', task.get('instruction_text', ''))}",
            f"required_apps: {task.get('required_apps', [])}",
            f"api_refs: {task.get('api_refs', [])}",
            "instruction: rank reusable skills for this AppWorld task",
        ]
    )


def _skillrouter_skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"name: {skill.get('name', '')}",
            f"description: {skill.get('description', '')}",
            f"executor: {skill.get('executor_desc', '')}",
            f"input_schema: {json.dumps(skill.get('input_schema', {}), ensure_ascii=False, sort_keys=True)}",
            f"output_schema: {json.dumps(skill.get('output_schema', {}), ensure_ascii=False, sort_keys=True)}",
            f"failure_modes: {json.dumps(skill.get('failure_modes', []), ensure_ascii=False, sort_keys=True)}",
            f"body: {skill.get('body', '')}",
        ]
    )


def _skill_api_refs(skill: dict[str, Any]) -> set[str]:
    text = _skillrouter_skill_text(skill)
    return {
        f"{match.group(1)}.{match.group(2)}"
        for match in re.finditer(r"apis\.([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", text)
    }


def _skill_apps(skill: dict[str, Any]) -> set[str]:
    apps = {ref.split(".", 1)[0] for ref in _skill_api_refs(skill)}
    text = _skillrouter_skill_text(skill).lower()
    for app in ("amazon", "file_system", "gmail", "phone", "simple_note", "splitwise", "spotify", "supervisor", "todoist", "venmo"):
        if app in text or app.replace("_", " ") in text:
            apps.add(app)
    return apps


def skillrouter_lexical_rank(task: dict[str, Any], skills: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    query_tokens = _tokenize(_skillrouter_query_text(task))
    task_refs = {str(item) for item in task.get("api_refs", [])}
    task_apps = {str(item) for item in task.get("required_apps", [])} | {str(item) for item in task.get("api_apps", [])}
    scored: list[tuple[str, float]] = []
    for idx, skill in enumerate(skills):
        skill_tokens = _tokenize(_skillrouter_skill_text(skill))
        denom = math.sqrt(max(len(query_tokens), 1) * max(len(skill_tokens), 1))
        token_score = len(query_tokens & skill_tokens) / denom if denom else 0.0
        api_score = 8.0 * len(task_refs & _skill_api_refs(skill))
        app_score = 3.0 * len(task_apps & _skill_apps(skill))
        tie_break = 1.0e-9 * (len(skills) - idx)
        scored.append((str(skill.get("skill_id")), float(api_score + app_score + token_score + tie_break)))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        if int(row.get("relevance", 1)) <= 0:
            continue
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["skill_id"]))
    return qrels


def compute_retrieval_metrics(
    predictions: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    query_ids = [query_id for query_id in predictions if query_id in qrels]
    metrics: dict[str, float] = {}
    for k in ks:
        hits = 0
        reciprocal_ranks: list[float] = []
        for query_id in query_ids:
            positives = qrels.get(query_id, set())
            ranked = predictions[query_id][:k]
            hit_rank = next((idx + 1 for idx, skill_id in enumerate(ranked) if skill_id in positives), None)
            if hit_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)
        denom = max(len(query_ids), 1)
        metrics[f"recall@{k}"] = round(hits / denom, 6)
        metrics[f"mrr@{k}"] = round(sum(reciprocal_ranks) / denom, 6)
    metrics["evaluated_queries"] = float(len(query_ids))
    return metrics


def run_appworld_skillrouter_baseline(
    *,
    tasks_path: str | Path = "data/appworld_routing/dev_tasks.jsonl",
    qrels_path: str | Path = "data/appworld_routing/dev_qrels.jsonl",
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    output_dir: str | Path = "outputs/appworld_skillrouter_baseline",
    top_k: int = 20,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    skills = read_jsonl(skill_pool_path)
    qrels = _load_qrels(qrels_path)
    if not tasks or not skills or not qrels:
        report = {
            "status": "blocked",
            "method": "skillrouter_serialization_lexical",
            "tasks_path": str(tasks_path),
            "qrels_path": str(qrels_path),
            "skill_pool_path": str(skill_pool_path),
            "query_count": len(tasks),
            "skill_count": len(skills),
            "qrel_query_count": len(qrels),
            "blocker": "missing_tasks_skills_or_qrels",
        }
        write_json(output_dir / "report.json", report)
        return report

    predictions: dict[str, list[str]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for task in tasks:
        ranked = skillrouter_lexical_rank(task, skills)[:top_k]
        query_id = str(task.get("query_id") or task.get("task_id"))
        predictions[query_id] = [skill_id for skill_id, _score in ranked]
        prediction_rows.append(
            {
                "query_id": query_id,
                "task_id": task.get("task_id"),
                "split": task.get("split"),
                "ranked_skill_ids": [skill_id for skill_id, _score in ranked],
                "scores": [score for _skill_id, score in ranked],
            }
        )

    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    metrics = compute_retrieval_metrics(predictions, qrels, ks=(1, 5, 10, top_k))
    report = {
        "status": "ok",
        "method": "skillrouter_serialization_lexical",
        "baseline_family": "SkillRouter-style static query-to-skill routing",
        "caveat": "This is an AppWorld routing baseline using SkillRouter serialization and static lexical scoring; it is not a DB-state task-completion agent.",
        "tasks_path": str(tasks_path),
        "qrels_path": str(qrels_path),
        "skill_pool_path": str(skill_pool_path),
        "predictions_path": str(predictions_path),
        "top_k": top_k,
        "query_count": len(tasks),
        "skill_count": len(skills),
        "qrel_query_count": len(qrels),
        "metrics": metrics,
    }
    write_json(output_dir / "report.json", report)
    return report


def run_appworld_routing_eval_from_scores(
    *,
    predictions: dict[str, list[str]],
    qrels_path: str | Path,
    output_dir: str | Path,
    method: str,
    top_k: int = 20,
) -> dict[str, Any]:
    qrels = _load_qrels(qrels_path)
    metrics = compute_retrieval_metrics(predictions, qrels, ks=(1, 5, 10, top_k))
    report = {
        "status": "ok",
        "method": method,
        "qrels_path": str(qrels_path),
        "top_k": top_k,
        "metrics": metrics,
    }
    write_json(Path(output_dir) / "report.json", report)
    return report
