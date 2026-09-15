from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _row_id(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    return None


def audit_toolret_eval_data(
    *,
    data_dir: str | Path = "data/toolret_eval",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    queries_path = data_dir / "queries.jsonl"
    skills_path = data_dir / "skills.jsonl"
    qrels_path = data_dir / "qrels.jsonl"

    missing_files = [
        str(path)
        for path in (queries_path, skills_path, qrels_path)
        if not path.exists() or path.stat().st_size <= 0
    ]
    if missing_files:
        report = {
            "status": "action_required",
            "data_dir": str(data_dir),
            "blockers": ["missing_required_files"],
            "missing_files": missing_files,
        }
        if output_path is not None:
            _write_json(output_path, report)
        return report

    queries = _read_jsonl(queries_path)
    skills = _read_jsonl(skills_path)
    qrels = _read_jsonl(qrels_path)

    query_ids = {
        query_id
        for row in queries
        if (query_id := _row_id(row, "query_id", "id", "qid", "task_id")) is not None
    }
    skill_ids = [
        skill_id
        for row in skills
        if (skill_id := _row_id(row, "skill_id", "id", "tool_id")) is not None
    ]
    skill_id_counts = Counter(skill_ids)
    duplicate_skill_ids = sorted(skill_id for skill_id, count in skill_id_counts.items() if count > 1)
    skill_id_set = set(skill_ids)

    positive_qrels = [
        row
        for row in qrels
        if int(row.get("relevance", 1)) > 0
    ]
    qrel_query_ids = {
        query_id
        for row in positive_qrels
        if (query_id := _row_id(row, "query_id", "id", "qid", "task_id")) is not None
    }
    qrel_skill_ids = {
        skill_id
        for row in positive_qrels
        if (skill_id := _row_id(row, "skill_id", "id", "tool_id", "doc_id")) is not None
    }
    missing_positive_qrel_skill_ids = sorted(qrel_skill_ids - skill_id_set)
    missing_positive_qrel_query_ids = sorted(qrel_query_ids - query_ids)

    blockers: list[str] = []
    if missing_positive_qrel_skill_ids:
        blockers.append("missing_positive_qrel_skill_ids")
    if missing_positive_qrel_query_ids:
        blockers.append("missing_positive_qrel_query_ids")
    if duplicate_skill_ids:
        blockers.append("duplicate_skill_ids")
    if not positive_qrels:
        blockers.append("no_positive_qrels")

    report = {
        "status": "ok" if not blockers else "action_required",
        "data_dir": str(data_dir),
        "blockers": blockers,
        "query_count": len(queries),
        "unique_query_count": len(query_ids),
        "skill_count": len(skills),
        "unique_skill_count": len(skill_id_set),
        "qrel_count": len(qrels),
        "positive_qrel_count": len(positive_qrels),
        "missing_positive_qrel_skill_ids": missing_positive_qrel_skill_ids,
        "missing_positive_qrel_query_ids": missing_positive_qrel_query_ids,
        "duplicate_skill_ids": duplicate_skill_ids,
        "files": {
            "queries": str(queries_path),
            "skills": str(skills_path),
            "qrels": str(qrels_path),
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
