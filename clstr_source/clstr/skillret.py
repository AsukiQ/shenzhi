from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SKILLRET_DATASET = "ThakiCloud/SKILLRET"
SPLITS = ("train", "test")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _schema_for_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path), "count": 0, "fields": []}
    count = 0
    fields: list[str] = []
    for row in _read_jsonl(path):
        count += 1
        if not fields:
            fields = sorted(row.keys())
    return {"status": "ok", "path": str(path), "count": count, "fields": fields}


def _source_path(source_root: Path, subset: str, split: str) -> Path:
    return source_root / "data" / subset / f"{split}.jsonl"


def build_skillret_schema_report(source_root: str | Path) -> dict[str, Any]:
    source_root = Path(source_root)
    subsets: dict[str, dict[str, Any]] = {}
    for subset in ("skills", "queries", "qrels"):
        subsets[subset] = {
            split: _schema_for_file(_source_path(source_root, subset, split))
            for split in SPLITS
        }
    return {
        "source_dataset": SKILLRET_DATASET,
        "source_root": str(source_root),
        "subsets": subsets,
    }


def _normalize_skill(row: dict[str, Any], split: str) -> dict[str, Any]:
    skill_md = row.get("skill_md") or row.get("body") or ""
    return {
        "skill_id": str(row["id"]),
        "id": str(row["id"]),
        "name": str(row.get("name", "")),
        "description": str(row.get("description", "")),
        "input_schema": {},
        "output_schema": {},
        "executor_desc": str(row.get("description", "")),
        "failure_modes": [],
        "body": str(skill_md),
        "skill_md": str(skill_md),
        "split": split,
        "source_dataset": SKILLRET_DATASET,
        "metadata": {
            key: row.get(key)
            for key in (
                "namespace",
                "author",
                "license",
                "repo",
                "source_url",
                "raw_url",
                "major",
                "sub",
                "primary_action",
                "primary_object",
                "domain",
                "stars",
                "installs",
            )
            if key in row
        },
    }


def _normalize_query(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "query_id": str(row["id"]),
        "id": str(row["id"]),
        "query": str(row.get("query", "")),
        "positive_skill_ids": [str(item) for item in row.get("skill_ids", [])],
        "skill_names": [str(item) for item in row.get("skill_names", [])],
        "k": int(row.get("k", len(row.get("skill_ids", [])))),
        "split": split,
        "source_dataset": SKILLRET_DATASET,
        "metadata": {
            "original_id": row.get("original_id"),
            "generator_model": row.get("generator_model"),
        },
    }


def _normalize_qrel(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "query_id": str(row["query_id"]),
        "skill_id": str(row["skill_id"]),
        "relevance": int(row.get("relevance", 1)),
        "split": split,
        "source_dataset": SKILLRET_DATASET,
    }


def import_skillret_dataset(
    source_root: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    schema_report = build_skillret_schema_report(source_root)
    _write_json(report_dir / "schema_report.json", schema_report)

    split_counts: dict[str, dict[str, int]] = {
        split: {"skill_count": 0, "query_count": 0, "qrel_count": 0}
        for split in SPLITS
    }

    def skills() -> Iterable[dict[str, Any]]:
        for split in SPLITS:
            path = _source_path(source_root, "skills", split)
            for row in _read_jsonl(path):
                split_counts[split]["skill_count"] += 1
                yield _normalize_skill(row, split)

    def queries() -> Iterable[dict[str, Any]]:
        for split in SPLITS:
            path = _source_path(source_root, "queries", split)
            for row in _read_jsonl(path):
                split_counts[split]["query_count"] += 1
                yield _normalize_query(row, split)

    def qrels() -> Iterable[dict[str, Any]]:
        for split in SPLITS:
            path = _source_path(source_root, "qrels", split)
            for row in _read_jsonl(path):
                split_counts[split]["qrel_count"] += 1
                yield _normalize_qrel(row, split)

    skill_count = _write_jsonl(output_dir / "skills.jsonl", skills())
    query_count = _write_jsonl(output_dir / "queries.jsonl", queries())
    qrel_count = _write_jsonl(output_dir / "qrels.jsonl", qrels())

    manifest = {
        "status": "ready",
        "source_dataset": SKILLRET_DATASET,
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "data_role": "retrieval_pretraining_not_skillsbench_trajectory",
        "splits": split_counts,
        "record_counts": {
            "skills": skill_count,
            "queries": query_count,
            "qrels": qrel_count,
        },
        "files": {
            "skills": str(output_dir / "skills.jsonl"),
            "queries": str(output_dir / "queries.jsonl"),
            "qrels": str(output_dir / "qrels.jsonl"),
        },
        "notes": [
            "SKILLRET provides query-skill relevance supervision for retrieval warmup.",
            "SKILLRET is not a SkillsBench trajectory source and is not clean_router_data.",
            "SkillRouter benchmark eval labels are not used by this importer.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_skillret_training_rows(data_root: str | Path, split: str = "train") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    data_root = Path(data_root)
    skills = [row for row in _read_jsonl(data_root / "skills.jsonl") if row.get("split") == split]
    queries = [row for row in _read_jsonl(data_root / "queries.jsonl") if row.get("split") == split]
    positives: dict[str, list[str]] = defaultdict(list)
    for row in _read_jsonl(data_root / "qrels.jsonl"):
        if row.get("split") == split and int(row.get("relevance", 1)) > 0:
            positives[str(row["query_id"])].append(str(row["skill_id"]))
    return skills, queries, dict(positives)
