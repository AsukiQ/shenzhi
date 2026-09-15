from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "").strip()


def _query_text(row: dict[str, Any]) -> str:
    return str(row.get("query_text") or row.get("query") or row.get("state_text") or "").strip()


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if row.get("positive_skill_id"):
        values.append(str(row["positive_skill_id"]).strip())
    raw_many = row.get("positive_skill_ids")
    if isinstance(raw_many, list):
        values.extend(str(item).strip() for item in raw_many)
    return [item for item in dict.fromkeys(values) if item]


def _source_name(root: Path, explicit: str | None = None) -> str:
    return str(explicit or root.name).strip() or "toolrex_source"


def _skills_path(root: Path) -> Path:
    for name in ("skills.jsonl", "skill_pool.jsonl"):
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"missing skills.jsonl or skill_pool.jsonl under {root}")


def _load_retrieval_source(root: Path, *, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    skills = _read_jsonl(_skills_path(root))
    retrieval_path = root / "retrieval.jsonl"
    if not retrieval_path.exists():
        raise FileNotFoundError(f"missing retrieval.jsonl under {root}")
    valid_skill_ids = {_skill_id(skill) for skill in skills if _skill_id(skill)}
    trajectories: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for idx, row in enumerate(_read_jsonl(retrieval_path)):
        query = _query_text(row)
        positives = _positive_skill_ids(row)
        if not query:
            skipped["missing_query_text"] += 1
            continue
        if not positives:
            skipped["missing_positive_skill_id"] += 1
            continue
        for positive_idx, skill_id in enumerate(positives):
            if skill_id not in valid_skill_ids:
                skipped["positive_missing_from_skills"] += 1
                continue
            query_id = str(row.get("query_id") or f"{source}-{idx}").strip()
            provenance = dict(row.get("provenance") or {})
            provenance.setdefault("split", "train")
            provenance.setdefault("source_dataset", source)
            trajectories.append(
                {
                    "task_id": f"{source}/{query_id}/{positive_idx}",
                    "query_id": f"{source}/{query_id}",
                    "state_text": query,
                    "query": query,
                    "next_skill_id": skill_id,
                    "benchmark": source,
                    "source_benchmark": source,
                    "loss_mask": {"routing": True},
                    "negative_skill_ids": [
                        str(item).strip()
                        for item in row.get("negative_skill_ids", [])
                        if str(item).strip() in valid_skill_ids
                    ],
                    "provenance": provenance,
                }
            )
            retrieval_rows.append(
                {
                    "source": source,
                    "query_id": f"{source}/{query_id}",
                    "query_text": query,
                    "positive_skill_id": skill_id,
                    "negative_skill_ids": trajectories[-1]["negative_skill_ids"],
                    "provenance": provenance,
                }
            )
    report = {
        "format": "retrieval_jsonl",
        "skill_count": len(skills),
        "source_query_rows": len(_read_jsonl(retrieval_path)),
        "query_positive_count": len(trajectories),
        "skipped_reasons": dict(sorted(skipped.items())),
    }
    return skills, trajectories, report | {"retrieval_rows": retrieval_rows}


def _load_qrels_source(root: Path, *, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    skills = _read_jsonl(_skills_path(root))
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels.jsonl"
    if not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(f"missing queries.jsonl or qrels.jsonl under {root}")
    query_by_id = {
        str(row.get("query_id") or row.get("id") or "").strip(): row
        for row in _read_jsonl(queries_path)
        if str(row.get("query_id") or row.get("id") or "").strip()
    }
    valid_skill_ids = {_skill_id(skill) for skill in skills if _skill_id(skill)}
    grouped: dict[str, list[str]] = defaultdict(list)
    split_by_query: dict[str, str] = {}
    skipped: Counter[str] = Counter()
    for row in _read_jsonl(qrels_path):
        qid = str(row.get("query_id") or "").strip()
        sid = str(row.get("skill_id") or "").strip()
        if not qid or not sid:
            skipped["missing_query_or_skill"] += 1
            continue
        if sid not in valid_skill_ids:
            skipped["positive_missing_from_skills"] += 1
            continue
        grouped[qid].append(sid)
        split_by_query.setdefault(qid, str(row.get("split") or "train"))
    trajectories: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for qid, skill_ids in sorted(grouped.items()):
        query_row = query_by_id.get(qid)
        if query_row is None:
            skipped["missing_query_row"] += 1
            continue
        query = _query_text(query_row)
        if not query:
            skipped["missing_query_text"] += 1
            continue
        for positive_idx, skill_id in enumerate(dict.fromkeys(skill_ids)):
            provenance = {
                "split": split_by_query.get(qid, "train"),
                "source_dataset": source,
            }
            trajectories.append(
                {
                    "task_id": f"{source}/{qid}/{positive_idx}",
                    "query_id": f"{source}/{qid}",
                    "state_text": query,
                    "query": query,
                    "next_skill_id": skill_id,
                    "benchmark": source,
                    "source_benchmark": source,
                    "loss_mask": {"routing": True},
                    "negative_skill_ids": [],
                    "provenance": provenance,
                }
            )
            retrieval_rows.append(
                {
                    "source": source,
                    "query_id": f"{source}/{qid}",
                    "query_text": query,
                    "positive_skill_id": skill_id,
                    "negative_skill_ids": [],
                    "provenance": provenance,
                }
            )
    report = {
        "format": "queries_qrels",
        "skill_count": len(skills),
        "source_query_rows": len(query_by_id),
        "query_positive_count": len(trajectories),
        "skipped_reasons": dict(sorted(skipped.items())),
    }
    return skills, trajectories, report | {"retrieval_rows": retrieval_rows}


def _load_source(root: str | Path) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    source = _source_name(root)
    if (root / "retrieval.jsonl").exists():
        skills, trajectories, report = _load_retrieval_source(root, source=source)
    elif (root / "queries.jsonl").exists() and (root / "qrels.jsonl").exists():
        skills, trajectories, report = _load_qrels_source(root, source=source)
    else:
        raise FileNotFoundError(f"unsupported ToolRex source format under {root}")
    retrieval_rows = report.pop("retrieval_rows")
    report["root"] = str(root)
    return source, skills, trajectories, {**report, "retrieval_rows": retrieval_rows}


def build_toolrex_unified_data(
    *,
    source_roots: list[str | Path],
    output_dir: str | Path,
    max_rows: int | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    skill_by_id: dict[str, dict[str, Any]] = {}
    all_trajectories: list[dict[str, Any]] = []
    all_retrieval_rows: list[dict[str, Any]] = []
    source_reports: dict[str, dict[str, Any]] = {}
    for root in source_roots:
        source, skills, trajectories, report = _load_source(root)
        for skill in skills:
            sid = _skill_id(skill)
            if sid and sid not in skill_by_id:
                skill_by_id[sid] = {**skill, "source": skill.get("source") or source}
        all_trajectories.extend(trajectories)
        all_retrieval_rows.extend(report.pop("retrieval_rows"))
        source_reports[source] = report
    rng = random.Random(seed)
    rng.shuffle(all_trajectories)
    if max_rows is not None:
        all_trajectories = all_trajectories[: max(0, int(max_rows))]
        kept_query_ids = {str(row.get("query_id")) for row in all_trajectories}
        all_retrieval_rows = [
            row
            for row in all_retrieval_rows
            if str(row.get("query_id")) in kept_query_ids
        ]
    skills = [skill_by_id[sid] for sid in sorted(skill_by_id)]
    _write_jsonl(output_dir / "skill_pool.jsonl", skills)
    _write_jsonl(output_dir / "trajectories.jsonl", all_trajectories)
    _write_jsonl(output_dir / "retrieval.jsonl", all_retrieval_rows)
    benchmark_counts = dict(sorted(Counter(str(row.get("benchmark")) for row in all_trajectories).items()))
    report = {
        "status": "ok",
        "output_dir": str(output_dir),
        "source_roots": [str(Path(root)) for root in source_roots],
        "skill_count": len(skills),
        "trajectory_count": len(all_trajectories),
        "retrieval_row_count": len(all_retrieval_rows),
        "benchmark_counts": benchmark_counts,
        "source_reports": source_reports,
        "max_rows": max_rows,
        "seed": int(seed),
    }
    _write_json(output_dir / "manifest.json", report)
    return report
