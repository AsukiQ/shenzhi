from __future__ import annotations

import json
from collections import Counter
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_tool_ids")
    if values is None:
        single = row.get("positive_skill_id") or row.get("skill_id") or row.get("tool_id") or row.get("doc_id")
        values = [single] if single else []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if item]


def _negative_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("negative_skill_ids")
    if values is None:
        values = row.get("negative_tool_ids")
    if values is None:
        values = row.get("negative_doc_ids")
    if values is None:
        values = []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if item]


def _source_and_split(row: dict[str, Any]) -> tuple[str, str]:
    provenance = _parse_provenance(row)
    source = str(row.get("source") or provenance.get("source") or provenance.get("source_dataset") or "unknown")
    split = str(row.get("split") or provenance.get("split") or "unknown")
    return source, split


def export_retrieval_qrels(
    *,
    retrieval_path: str | Path,
    output_path: str | Path,
    include_negatives: bool = False,
    allowed_sources: set[str] | None = None,
    allowed_splits: set[str] | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    positive_qrels = 0
    negative_qrels = 0
    skipped_missing_fields = 0
    skipped_by_source: Counter[str] = Counter()
    skipped_by_split: Counter[str] = Counter()

    with output.open("w", encoding="utf-8") as handle:
        for row in _read_jsonl(retrieval_path):
            source_rows += 1
            source, split = _source_and_split(row)
            if allowed_sources is not None and source not in allowed_sources:
                skipped_by_source[source] += 1
                continue
            if allowed_splits is not None and split not in allowed_splits:
                skipped_by_split[split] += 1
                continue

            query_id = row.get("query_id") or row.get("qid") or row.get("task_id")
            positives = _positive_skill_ids(row)
            if not query_id or not positives:
                skipped_missing_fields += 1
                continue

            for skill_id in positives:
                handle.write(
                    json.dumps(
                        {
                            "query_id": str(query_id),
                            "skill_id": skill_id,
                            "relevance": 1,
                            "source": source,
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                positive_qrels += 1

            if include_negatives:
                for skill_id in _negative_skill_ids(row):
                    handle.write(
                        json.dumps(
                            {
                                "query_id": str(query_id),
                                "skill_id": skill_id,
                                "relevance": 0,
                                "source": source,
                                "split": split,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    negative_qrels += 1

    report = {
        "status": "ok",
        "retrieval_path": str(retrieval_path),
        "qrels_path": str(output),
        "include_negatives": include_negatives,
        "allowed_sources": sorted(allowed_sources) if allowed_sources is not None else None,
        "allowed_splits": sorted(allowed_splits) if allowed_splits is not None else None,
        "source_rows": source_rows,
        "positive_qrels": positive_qrels,
        "negative_qrels": negative_qrels,
        "written_qrels": positive_qrels + negative_qrels,
        "skipped_missing_fields": skipped_missing_fields,
        "skipped_by_source": dict(sorted(skipped_by_source.items())),
        "skipped_by_split": dict(sorted(skipped_by_split.items())),
    }
    if report_path is not None:
        _write_json(report_path, report)
    return report
