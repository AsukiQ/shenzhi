from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


QUERY_DATASET = "mangopy/ToolRet-Queries"
TOOL_DATASET = "mangopy/ToolRet-Tools"
TOOLRET_TASK_TO_CATEGORY = {
    "craft-math-algebra": "code",
    "craft-tabmwp": "code",
    "craft-vqa": "code",
    "gorilla-huggingface": "code",
    "gorilla-pytorch": "code",
    "gorilla-tensor": "code",
    "toolink": "code",
    "apibank": "web",
    "apigen": "web",
    "mnms": "web",
    "reversechain": "web",
    "rotbench": "web",
    "t-eval-dialog": "web",
    "t-eval-step": "web",
    "taskbench-daily": "web",
    "toolace": "web",
    "toolbench": "web",
    "toolemu": "web",
    "tooleyes": "web",
    "toollens": "web",
    "ultratool": "web",
    "autotools-food": "web",
    "autotools-music": "web",
    "autotools-weather": "web",
    "restgpt-spotify": "web",
    "restgpt-tmdb": "web",
    "appbench": "customized",
    "gpt4tools": "customized",
    "gta": "customized",
    "taskbench-huggingface": "customized",
    "taskbench-multimedia": "customized",
    "metatool": "customized",
    "tool-be-honest": "customized",
    "toolalpaca": "customized",
    "toolbench-sam": "customized",
}
TOOLRET_TASKS = tuple(TOOLRET_TASK_TO_CATEGORY)
TOOLRET_CATEGORIES = tuple(sorted(set(TOOLRET_TASK_TO_CATEGORY.values())))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_jsonl_source(source: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(source)
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if path.is_dir() and not paths:
        paths = sorted(path.glob("**/*.jsonl"))
    for item in paths:
        yield from _read_jsonl(item)


def _iter_hf_queries(tasks: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required when --query_source is omitted. Download ToolRet-Queries "
            "on the login node first or run in an environment with datasets installed."
        ) from exc
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    for task in _expand_toolret_tasks(tasks):
        dataset = load_dataset(QUERY_DATASET, task, split="queries")
        for row in dataset:
            yield dict(row)


def _iter_hf_tools(categories: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required when --tool_source is omitted. Download ToolRet-Tools "
            "on the login node first or run in an environment with datasets installed."
        ) from exc
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    for category in _expand_toolret_categories(categories):
        dataset = load_dataset(TOOL_DATASET, category, split="tools")
        for row in dataset:
            yield dict(row)


def _expand_toolret_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    if not tasks or any(str(task).lower() == "all" for task in tasks):
        return TOOLRET_TASKS
    return tuple(str(task).strip().lower() for task in tasks if str(task).strip())


def _expand_toolret_categories(categories: Sequence[str] | None) -> tuple[str, ...]:
    if not categories or any(str(category).lower() == "all" for category in categories):
        return TOOLRET_CATEGORIES
    return tuple(str(category).strip().lower() for category in categories if str(category).strip())


def _iter_queries(source: str | Path | None, tasks: Sequence[str] | None) -> Iterable[dict[str, Any]]:
    if source is not None:
        yield from _iter_jsonl_source(source)
        return
    yield from _iter_hf_queries(tasks)


def _iter_tools(source: str | Path | None, categories: Sequence[str] | None) -> Iterable[dict[str, Any]]:
    if source is not None:
        yield from _iter_jsonl_source(source)
        return
    yield from _iter_hf_tools(categories)


def _parse_labels(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _tool_doc(row: dict[str, Any]) -> dict[str, Any]:
    doc = row.get("doc")
    return doc if isinstance(doc, dict) else {}


def _tool_name(row: dict[str, Any]) -> str:
    doc = _tool_doc(row)
    return str(row.get("name") or doc.get("name") or row.get("id") or "").strip()


def _tool_description(row: dict[str, Any]) -> str:
    doc = _tool_doc(row)
    return str(row.get("description") or doc.get("description") or row.get("documentation") or "")


def _query_text(row: dict[str, Any]) -> str:
    query = str(row.get("query") or row.get("text") or "").strip()
    instruction = str(row.get("instruction") or "").strip()
    if instruction and query:
        return f"{instruction}\nTask: {query}"
    return instruction or query


def _query_record(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    query_id = row.get("id") or row.get("query_id")
    if not query_id:
        return None
    return {
        "query_id": str(query_id),
        "query_text": _query_text(row),
        "query": str(row.get("query") or row.get("text") or ""),
        "instruction": str(row.get("instruction") or ""),
        "task": str(row.get("task") or row.get("source_task") or "unknown"),
        "source": "ToolRet-Queries",
        "split": split,
    }


def _skill_record(row: dict[str, Any]) -> dict[str, Any] | None:
    raw_id = row.get("id") or row.get("tool_id") or row.get("skill_id")
    if not raw_id:
        return None
    doc = _tool_doc(row)
    parameters = row.get("parameters") or doc.get("parameters") or row.get("input_schema") or {}
    description = _tool_description(row)
    return {
        "skill_id": str(raw_id),
        "canonical_skill_id": str(raw_id),
        "name": _tool_name(row),
        "description": description,
        "source": "ToolRet-Tools",
        "environment": row.get("category") or row.get("domain"),
        "executor_desc": description,
        "input_schema": parameters if isinstance(parameters, dict) else {},
        "output_schema": row.get("output_schema") or {},
        "body": str(row.get("documentation") or row.get("body") or ""),
        "provenance": {
            "source_dataset": TOOL_DATASET,
            "raw_tool_id": str(raw_id),
        },
    }


def _qrel_records(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    query_id = row.get("id") or row.get("query_id")
    if not query_id:
        return []
    task = str(row.get("task") or row.get("source_task") or "unknown")
    qrels: list[dict[str, Any]] = []
    for label in _parse_labels(row.get("labels")):
        skill_id = label.get("id") or label.get("tool_id") or label.get("skill_id")
        if not skill_id:
            continue
        qrels.append(
            {
                "query_id": str(query_id),
                "skill_id": str(skill_id),
                "relevance": int(label.get("relevance", 1)),
                "task": task,
                "source": "ToolRet",
                "split": split,
            }
        )
    return qrels


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def import_toolret_eval(
    *,
    query_source: str | Path | None,
    tool_source: str | Path | None,
    output_dir: str | Path = "data/toolret_eval",
    split: str = "test",
    tasks: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    max_queries: int | None = None,
    max_tools: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_iter_queries(query_source, tasks)):
        if max_queries is not None and idx >= max_queries:
            break
        query = _query_record(row, split)
        if query is not None:
            query_rows.append(query)
            qrel_rows.extend(_qrel_records(row, split))

    skill_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_iter_tools(tool_source, categories)):
        if max_tools is not None and idx >= max_tools:
            break
        skill = _skill_record(row)
        if skill is not None:
            skill_rows.append(skill)

    _write_jsonl(out_dir / "queries.jsonl", query_rows)
    _write_jsonl(out_dir / "skills.jsonl", skill_rows)
    _write_jsonl(out_dir / "qrels.jsonl", qrel_rows)

    skill_ids = {row["skill_id"] for row in skill_rows}
    qrel_skill_ids = {row["skill_id"] for row in qrel_rows if int(row.get("relevance", 0)) > 0}
    manifest = {
        "status": "ok",
        "source_dataset": {
            "queries": QUERY_DATASET,
            "tools": TOOL_DATASET,
        },
        "query_source": str(query_source) if query_source is not None else f"hf://datasets/{QUERY_DATASET}",
        "tool_source": str(tool_source) if tool_source is not None else f"hf://datasets/{TOOL_DATASET}",
        "output_dir": str(out_dir),
        "split": split,
        "query_count": len(query_rows),
        "skill_count": len(skill_rows),
        "qrel_count": len(qrel_rows),
        "missing_qrel_skill_ids": sorted(qrel_skill_ids - skill_ids),
        "files": {
            "queries": "queries.jsonl",
            "skills": "skills.jsonl",
            "qrels": "qrels.jsonl",
        },
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
