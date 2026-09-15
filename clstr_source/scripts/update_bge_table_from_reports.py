#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def _benchmark_label(value: Any) -> str:
    label = str(value or "").strip()
    mapping = {
        "toolbench_g3": "ToolBench-G3",
        "toolbench-g3": "ToolBench-G3",
        "toolsandbox": "ToolSandbox",
        "tau2": "tau2",
    }
    return mapping.get(label.lower(), label or "unknown")


def _method_label(report: dict[str, Any]) -> tuple[str, str]:
    embedding_family = str(report.get("embedding_model_family") or "").lower()
    reranker_path = report.get("reranker_model_name_or_path")
    embedding_path_label = str(report.get("embedding_model_name_or_path") or "").lower()
    reranker_path_label = str(reranker_path or "").lower()
    if embedding_family == "bge_m3":
        if "bge_toolrex" in embedding_path_label or "bge-m3-tool-embed" in embedding_path_label:
            if reranker_path:
                return "BGE-Tool-Embed + BGE-Tool-Rank", "1.2B"
            return "BGE-Tool-Embed retrieval-only", "0.6B"
        if "bge_sr_embedding" in embedding_path_label or "bge-m3-sr-emb" in embedding_path_label:
            if reranker_path:
                return "BGE-SR-Emb + BGE-SR-Rank", "1.2B"
            return "BGE-SR-Emb", "0.6B"
        if "bge_tool_rank" in reranker_path_label:
            return "BGE-Tool-Embed + BGE-Tool-Rank", "1.2B"
        if "bge_sr_rank" in reranker_path_label:
            return "BGE-SR-Emb + BGE-SR-Rank", "1.2B"
        if reranker_path:
            return "BGE-M3 + BGE-reranker-v2-M3", "1.2B"
        return "BGE-M3 retrieval-only", "0.6B"
    return str(report.get("method") or "unknown"), "unknown"


def _candidate_scope(report: dict[str, Any]) -> str:
    candidate_source = str(report.get("candidate_source") or "").strip()
    embedding_report = report.get("embedding_report")
    if not isinstance(embedding_report, dict):
        embedding_report = {}
    if candidate_source == "full_pool":
        top_k = embedding_report.get("top_k") or report.get("top_k")
        return f"full pool -> top{int(top_k)}" if top_k is not None else "full pool"
    if candidate_source == "row_candidates":
        return "row candidates"
    return candidate_source or "unknown"


def route_report_to_table_row(report_path: str | Path) -> dict[str, str]:
    report_path = Path(report_path)
    report = _load_json(report_path)
    strict = report.get("strict")
    if not isinstance(strict, dict):
        strict = {}
    method, params = _method_label(report)
    retained = report.get("retained_eval_rows")
    source = report.get("source_eval_rows")
    if retained is not None and source is not None:
        rows = f"{int(retained)}/{int(source)} strict"
    else:
        rows = "N/A"
    return {
        "benchmark": _benchmark_label(report.get("benchmark")),
        "method": method,
        "params": params,
        "r1": _fmt_metric(strict.get("strict_recall@1")),
        "r5": _fmt_metric(strict.get("strict_recall@5")),
        "mrr": _fmt_metric(strict.get("strict_mrr")),
        "rows": rows,
        "candidate_scope": _candidate_scope(report),
        "report_path": str(report_path),
    }


def _markdown_row(row: dict[str, str]) -> str:
    return (
        f"| {row['benchmark']} | {row['method']} | {row['params']} | {row['r1']} | {row['r5']} | "
        f"{row['mrr']} | {row['rows']} | {row['candidate_scope']} | `{row['report_path']}` |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print BGE route report rows for finalwork/table_bge_m3.md.")
    parser.add_argument("reports", nargs="+", help="Path(s) to route_eval_report.json")
    args = parser.parse_args()

    for report in args.reports:
        print(_markdown_row(route_report_to_table_row(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
