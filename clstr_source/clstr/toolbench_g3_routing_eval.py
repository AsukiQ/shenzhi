from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _parse_provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance") or {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _split(row: dict[str, Any]) -> str:
    provenance = _parse_provenance(row)
    return str(row.get("split") or provenance.get("split") or "unknown")


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_tool_ids")
    if values is None:
        value = row.get("positive_skill_id") or row.get("skill_id") or row.get("tool_id")
        values = [value] if value else []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if value]


def prepare_toolbench_g3_routing_eval(
    *,
    retrieval_path: str | Path = "data/toolbench_g3/retrieval.jsonl",
    output_dir: str | Path = "data/toolbench_g3_routing_eval",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build ToolBench-G3 routing eval queries/qrels from normalized retrieval rows.

    The normalized importer stores one row per positive API. Routing eval needs
    one query row per ToolBench-G3 query and one positive qrel per relevant API.
    """

    output = Path(output_dir)
    queries_path = output / "queries.jsonl"
    qrels_path = output / "qrels.jsonl"
    manifest_path = Path(report_path) if report_path is not None else output / "manifest.json"

    queries: dict[str, dict[str, Any]] = {}
    qrels: list[dict[str, Any]] = []
    seen_qrels: set[tuple[str, str]] = set()
    source_rows = 0
    skipped_missing_query = 0
    skipped_missing_positive = 0
    query_text_conflicts = 0

    for row in _read_jsonl(retrieval_path):
        source_rows += 1
        query_id = row.get("query_id") or row.get("qid") or row.get("task_id")
        if not query_id:
            skipped_missing_query += 1
            continue
        query_id = str(query_id)
        query_text = str(row.get("query_text") or row.get("query") or row.get("text") or "")
        split = _split(row)
        source = str(row.get("source") or "toolbench_g3")

        if query_id not in queries:
            queries[query_id] = {
                "query_id": query_id,
                "query_text": query_text,
                "source": source,
                "split": split,
            }
        elif query_text and queries[query_id].get("query_text") not in {"", query_text}:
            query_text_conflicts += 1

        positives = _positive_skill_ids(row)
        if not positives:
            skipped_missing_positive += 1
            continue
        for skill_id in positives:
            key = (query_id, skill_id)
            if key in seen_qrels:
                continue
            seen_qrels.add(key)
            qrels.append(
                {
                    "query_id": query_id,
                    "skill_id": skill_id,
                    "relevance": 1,
                    "source": source,
                    "split": split,
                }
            )

    _write_jsonl(queries_path, queries.values())
    _write_jsonl(qrels_path, qrels)
    report = {
        "status": "ok",
        "benchmark": "toolbench_g3_routing",
        "retrieval_path": str(retrieval_path),
        "output_dir": str(output),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "source_rows": source_rows,
        "query_count": len(queries),
        "positive_qrels": len(qrels),
        "skipped_missing_query": skipped_missing_query,
        "skipped_missing_positive": skipped_missing_positive,
        "query_text_conflicts": query_text_conflicts,
        "metric_scope": "static routing/retrieval recall over normalized ToolBench-G3 relevant APIs; not StableToolBench pass-rate.",
    }
    _write_json(manifest_path, report)
    return report
