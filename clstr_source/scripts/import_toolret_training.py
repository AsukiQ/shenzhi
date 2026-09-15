#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


SOURCE_DATASET = "mangopy/ToolRet-Training-20w"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_parquet(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for local parquet ToolRet shards. Use the reasoning_trap env "
            "or convert the shards to JSONL on the login node first."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            yield row


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_jsonable(key): _sanitize_jsonable(item)
            for key, item in value.items()
        }
    return value


def _iter_source_rows(source_path: Path | None, max_rows: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    if source_path is None:
        yield from _iter_hf_rows(max_rows=max_rows)
        return

    paths: list[Path]
    if source_path.is_dir():
        paths = sorted(source_path.glob("*.jsonl"))
        if not paths:
            paths = sorted(source_path.glob("**/*.jsonl"))
        parquet_paths = sorted(source_path.glob("*.parquet"))
        if not parquet_paths:
            parquet_paths = sorted(source_path.glob("**/*.parquet"))
        paths.extend(path for path in parquet_paths if path not in paths)
    else:
        paths = [source_path]

    yielded = 0
    for path in paths:
        if path.suffix == ".jsonl":
            rows = _read_jsonl(path)
        elif path.suffix == ".parquet":
            rows = _read_parquet(path)
        else:
            continue
        for row in rows:
            yield yielded, {**row, "_source_path": str(path)}
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def _iter_hf_rows(max_rows: int | None = None) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required when --source_path is omitted. Install it on the login node "
            "or provide a local JSONL file downloaded from ToolRet-Training-20w."
        ) from exc

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    dataset = load_dataset(SOURCE_DATASET, split="train", streaming=True)
    for idx, row in enumerate(dataset):
        if max_rows is not None and idx >= max_rows:
            break
        yield idx, dict(row)


def _parse_tool(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"name": text[:80], "description": text}
    return parsed if isinstance(parsed, dict) else None


def _tool_record(tool: dict[str, Any]) -> dict[str, Any] | None:
    tool = _sanitize_jsonable(tool)
    name = str(tool.get("name") or tool.get("tool_name") or "").strip()
    if not name:
        return None
    sid = f"toolret/{_slug(name)}"
    parameters = tool.get("parameters") or tool.get("input_schema") or {}
    return {
        "skill_id": sid,
        "canonical_skill_id": sid,
        "name": name,
        "description": str(tool.get("description") or tool.get("doc") or name),
        "alternate_descriptions": [],
        "environment": tool.get("category") or tool.get("domain"),
        "source": "ToolRet-Training-20w",
        "source_files": [],
        "executor_desc": str(tool.get("description") or ""),
        "input_schema": parameters if isinstance(parameters, dict) else {},
        "output_schema": tool.get("output_schema") or {},
        "body": "",
        "provenance": {
            "dedup_method": "toolret_name_identity_v2",
            "source_dataset": SOURCE_DATASET,
            "raw_name": name,
        },
    }


def _query_text(row: dict[str, Any]) -> str:
    row = _sanitize_jsonable(row)
    query = str(row.get("query") or row.get("task") or "").strip()
    prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
    if prompt and query:
        return f"{prompt}\nTask: {query}"
    return prompt or query


def import_toolret_training(
    source_path: str | Path | None,
    output_dir: str | Path = "data/toolret_training",
    max_rows: int | None = None,
) -> dict[str, Any]:
    source = Path(source_path) if source_path is not None else None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = out_dir / "retrieval.jsonl"
    skills_path = out_dir / "skills.jsonl"
    manifest_path = out_dir / "manifest.json"

    skills: dict[str, dict[str, Any]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    skipped_rows = 0
    seen_rows = 0

    for row_index, row in _iter_source_rows(source, max_rows=max_rows):
        seen_rows += 1
        positives = row.get("pos") or row.get("positive") or row.get("positives") or []
        negatives = row.get("neg") or row.get("negative") or row.get("negatives") or []
        if isinstance(positives, (str, dict)):
            positives = [positives]
        if isinstance(negatives, (str, dict)):
            negatives = [negatives]

        positive_records = [
            record for record in (_tool_record(tool) for tool in (_parse_tool(item) for item in positives)) if record
        ]
        negative_records = [
            record for record in (_tool_record(tool) for tool in (_parse_tool(item) for item in negatives)) if record
        ]
        if not positive_records:
            skipped_rows += 1
            continue

        positive = positive_records[0]
        for record in positive_records + negative_records:
            if record["skill_id"] not in skills:
                skills[record["skill_id"]] = record

        source_row_path = row.get("_source_path") or (str(source) if source is not None else f"hf://datasets/{SOURCE_DATASET}")
        retrieval_rows.append(
            {
                "query_id": str(row.get("query_id") or row.get("id") or f"toolret-train-{row_index}"),
                "query_text": _query_text(row),
                "positive_skill_id": positive["skill_id"],
                "negative_skill_ids": [record["skill_id"] for record in negative_records],
                "provenance": {
                    "source_dataset": SOURCE_DATASET,
                    "source_path": source_row_path,
                    "row_index": row_index,
                    "split": row.get("split", "train"),
                },
            }
        )

    with retrieval_path.open("w", encoding="utf-8") as f:
        for row in retrieval_rows:
            f.write(json.dumps(_sanitize_jsonable(row), ensure_ascii=False) + "\n")
    with skills_path.open("w", encoding="utf-8") as f:
        for sid in sorted(skills):
            f.write(json.dumps(_sanitize_jsonable(skills[sid]), ensure_ascii=False) + "\n")

    manifest = {
        "status": "ok",
        "source_dataset": SOURCE_DATASET,
        "source_path": str(source) if source is not None else f"hf://datasets/{SOURCE_DATASET}",
        "source_format": "local" if source is not None else "hf_streaming",
        "output_dir": str(out_dir),
        "files": {
            "retrieval": "retrieval.jsonl",
            "skills": "skills.jsonl",
        },
        "seen_rows": seen_rows,
        "retrieval_pairs": len(retrieval_rows),
        "skills": len(skills),
        "skipped_rows": skipped_rows,
        "max_rows": max_rows,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    }
    manifest_path.write_text(
        json.dumps(_sanitize_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ToolRet-Training-20w into CLSTR unified v2 JSONL inputs.")
    parser.add_argument(
        "--source_path",
        default=None,
        help="Local ToolRet-Training-20w JSONL file or directory. If omitted, uses datasets.load_dataset with HF mirror.",
    )
    parser.add_argument("--output_dir", default="data/toolret_training")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    manifest = import_toolret_training(
        source_path=args.source_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
