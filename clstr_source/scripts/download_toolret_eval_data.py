#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolret_eval_import import (
    QUERY_DATASET,
    TOOL_DATASET,
    _expand_toolret_categories,
    _expand_toolret_tasks,
)


DEFAULT_OUTPUT_DIR = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval")


def _sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, dict):
        return {str(_sanitize_jsonable(key)): _sanitize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_sanitize_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_dataset_rows(repo_id: str, config: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to download ToolRet eval data. Run this on the login node "
            "inside the reasoning_trap env, or provide local JSONL files to import_toolret_eval.py."
        ) from exc
    dataset = load_dataset(repo_id, config, split=split)
    return [dict(row) for row in dataset]


def download_toolret_eval_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dry_run: bool = False,
    tasks: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    max_queries_per_config: int | None = None,
    max_tools_per_config: int | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_endpoint = os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache")

    query_configs = list(_expand_toolret_tasks(tasks))
    tool_configs = list(_expand_toolret_categories(categories))
    queries_path = out_dir / "queries.jsonl"
    tools_path = out_dir / "tools.jsonl"
    manifest = {
        "status": "dry_run" if dry_run else "ok",
        "source_dataset": {
            "queries": QUERY_DATASET,
            "tools": TOOL_DATASET,
        },
        "hf_endpoint": hf_endpoint,
        "output_dir": str(out_dir),
        "query_configs": query_configs,
        "tool_configs": tool_configs,
        "files": {
            "queries": "queries.jsonl",
            "tools": "tools.jsonl",
            "manifest": "download_manifest.json",
        },
        "query_count": 0,
        "tool_count": 0,
        "max_queries_per_config": max_queries_per_config,
        "max_tools_per_config": max_tools_per_config,
    }
    if dry_run:
        return manifest

    query_rows: list[dict[str, Any]] = []
    for config in query_configs:
        rows = _load_dataset_rows(QUERY_DATASET, config, "queries")
        if max_queries_per_config is not None:
            rows = rows[:max_queries_per_config]
        for row in rows:
            row.setdefault("task", config)
            query_rows.append(row)

    tool_rows: list[dict[str, Any]] = []
    for config in tool_configs:
        rows = _load_dataset_rows(TOOL_DATASET, config, "tools")
        if max_tools_per_config is not None:
            rows = rows[:max_tools_per_config]
        for row in rows:
            row.setdefault("category", config)
            tool_rows.append(row)

    manifest["query_count"] = _write_jsonl(queries_path, query_rows)
    manifest["tool_count"] = _write_jsonl(tools_path, tool_rows)
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download ToolRet official eval queries/tools into local JSONL files on the login node. "
            "Do not run this from compute nodes."
        )
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--max_queries_per_config", type=int)
    parser.add_argument("--max_tools_per_config", type=int)
    args = parser.parse_args()

    manifest = download_toolret_eval_data(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        tasks=args.tasks,
        categories=args.categories,
        max_queries_per_config=args.max_queries_per_config,
        max_tools_per_config=args.max_tools_per_config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
