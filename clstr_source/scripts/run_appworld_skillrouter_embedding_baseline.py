#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_eval import compute_retrieval_metrics
from clstr.appworld_routing import read_jsonl, write_json, write_jsonl
from clstr.skillret_official import _encode_hf_texts, _rank_embedding_rows


def _load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        if int(row.get("relevance", 1)) <= 0:
            continue
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["skill_id"]))
    return qrels


def _query_text(task: dict) -> str:
    query = str(task.get("query") or task.get("instruction_text") or "")
    return (
        "Instruct: Given a task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery:"
        f"{query[:1500]}"
    )


def _skill_text(skill: dict) -> str:
    name = str(skill.get("name", "")).strip()
    desc = str(skill.get("description", "")).strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    executor = str(skill.get("executor_desc", "")).strip()
    return f"{name} | {desc} | {executor} | {body}"


def run_appworld_skillrouter_embedding_baseline(
    *,
    tasks_path: str | Path,
    qrels_path: str | Path,
    skill_pool_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path,
    top_k: int = 20,
    batch_size: int = 8,
    max_length: int = 1024,
) -> dict:
    output_dir = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    skills = read_jsonl(skill_pool_path)
    qrels = _load_qrels(qrels_path)
    query_ids = [str(row.get("query_id") or row.get("task_id")) for row in tasks]
    skill_ids = [str(row["skill_id"]) for row in skills]
    query_embs = _encode_hf_texts(str(model_name_or_path), [_query_text(row) for row in tasks], batch_size, max_length)
    skill_embs = _encode_hf_texts(str(model_name_or_path), [_skill_text(row) for row in skills], batch_size, max_length)
    ranked_by_query = _rank_embedding_rows(query_ids, query_embs, skill_embs, skill_ids, top_k)
    predictions = {query_id: [skill_id for skill_id, _score in ranked] for query_id, ranked in ranked_by_query.items()}
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(
        predictions_path,
        (
            {
                "query_id": query_id,
                "ranked_skill_ids": [skill_id for skill_id, _score in ranked],
                "scores": [score for _skill_id, score in ranked],
            }
            for query_id, ranked in ranked_by_query.items()
        ),
    )
    metrics = compute_retrieval_metrics(predictions, qrels, ks=(1, 5, 10, top_k))
    report = {
        "status": "ok",
        "method": "skillrouter_embedding_0.6b",
        "baseline_family": "SkillRouter static embedding retrieval",
        "model_name_or_path": str(model_name_or_path),
        "tasks_path": str(tasks_path),
        "qrels_path": str(qrels_path),
        "skill_pool_path": str(skill_pool_path),
        "predictions_path": str(predictions_path),
        "top_k": top_k,
        "batch_size": batch_size,
        "max_length": max_length,
        "query_count": len(tasks),
        "skill_count": len(skills),
        "qrel_query_count": len(qrels),
        "metrics": metrics,
        "caveat": "SkillRouter embedding retrieval on AppWorld routing qrels; not DB-state task-completion.",
    }
    write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkillRouter-Embedding AppWorld routing baseline.")
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--qrels_path", default="data/appworld_routing/dev_qrels.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_skillrouter_embedding_baseline/dev")
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    report = run_appworld_skillrouter_embedding_baseline(
        tasks_path=args.tasks_path,
        qrels_path=args.qrels_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
