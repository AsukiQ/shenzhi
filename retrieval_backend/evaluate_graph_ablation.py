"""Compare text-only and graph-enabled paper retrieval.

The default report is an operational ablation (graph trigger/fallback,
candidate changes, and latency).  If the input rows contain judgments, the
script also emits the existing MRR/Recall metrics for both engines; it does
not manufacture relevance labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

try:
    from .evaluate_retrieval import _filters, _judgments, evaluate_rows, read_jsonl
    from .neo4j_filter import Neo4jHttpPaperFilter
    from .paper_search import PaperSearchIndex
    from .retrieval_pipeline import HybridPaperSearch
except ImportError:
    from evaluate_retrieval import _filters, _judgments, evaluate_rows, read_jsonl
    from neo4j_filter import Neo4jHttpPaperFilter
    from paper_search import PaperSearchIndex
    from retrieval_pipeline import HybridPaperSearch


def _operational_report(
    engine: HybridPaperSearch,
    rows: Iterable[dict[str, Any]],
    *,
    top_k: int,
    max_queries: int | None,
) -> tuple[dict[str, Any], list[list[str]]]:
    count = 0
    latencies: list[float] = []
    operation_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    changed = 0
    added = 0
    returned = 0
    result_ids_by_query: list[list[str]] = []
    for row in rows:
        if max_queries is not None and count >= int(max_queries):
            break
        query = str(row.get("query_text") or row.get("query") or "").strip()
        if not query:
            continue
        started = time.perf_counter()
        results, state = engine.search(query, filters=_filters(row), top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        operation_counts.update(state.executed_operations)
        failure_counts.update(state.failed_operations)
        baseline_ids = set(row.get("baseline_result_ids") or [])
        result_ids = {result.paper_id for result in results}
        result_ids_by_query.append([result.paper_id for result in results])
        if baseline_ids:
            changed += int(result_ids != baseline_ids)
            added += len(result_ids - baseline_ids)
        returned += len(results)
        count += 1
    if not count:
        raise ValueError("no nonempty queries")
    ordered = sorted(latencies)
    return {
        "query_count": count,
        "returned_count": returned,
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "operation_counts": dict(sorted(operation_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "graph_operation_count": sum(
            value
            for name, value in operation_counts.items()
            if "NEO4J" in name
        ),
        "fallback_operation_count": sum(
            value
            for name, value in operation_counts.items()
            if "FALLBACK" in name or "SQLITE" in name
        ),
        "changed_query_count": changed,
        "added_result_count": added,
    }, result_ids_by_query


def run_ablation(
    *,
    db: str | Path,
    rows: list[dict[str, Any]],
    top_k: int = 10,
    max_queries: int | None = None,
    graph_filter: Any | None = None,
    graph_weight: float = 0.8,
    graph_seed_k: int = 12,
    graph_expand_k: int = 100,
    graph_value_limit: int = 16,
) -> dict[str, Any]:
    effective_rows = [
        row
        for row in rows
        if str(row.get("query_text") or row.get("query") or "").strip()
    ]
    if max_queries is not None:
        effective_rows = effective_rows[: max(0, int(max_queries))]
    baseline = HybridPaperSearch(
        PaperSearchIndex(db),
        graph_intent_enabled=False,
    )
    graph_engine = HybridPaperSearch(
        PaperSearchIndex(db),
        graph_filter=graph_filter,
        graph_weight=graph_weight,
        graph_seed_k=graph_seed_k,
        graph_expand_k=graph_expand_k,
        graph_value_limit=graph_value_limit,
    )
    baseline_report, baseline_ids = _operational_report(
        baseline, effective_rows, top_k=top_k, max_queries=None
    )
    compared_rows = [
        {**row, "baseline_result_ids": result_ids}
        for row, result_ids in zip(effective_rows, baseline_ids)
    ]
    graph_report, _graph_ids = _operational_report(
        graph_engine, compared_rows, top_k=top_k, max_queries=None
    )
    quality: dict[str, Any] | None = None
    if any(
        any(int(value) > 0 for value in _judgments(row).values())
        for row in effective_rows
    ):
        quality = {
            "baseline": evaluate_rows(
                baseline, effective_rows, top_k=top_k, max_queries=None
            ),
            "graph": evaluate_rows(
                graph_engine, effective_rows, top_k=top_k, max_queries=None
            ),
        }
    return {
        "status": "ok",
        "protocol": "shenzhi_graph_ablation_v1",
        "graph_enabled": graph_filter is not None,
        "parameters": {
            "top_k": int(top_k),
            "graph_weight": float(graph_weight),
            "graph_seed_k": int(graph_seed_k),
            "graph_expand_k": int(graph_expand_k),
            "graph_value_limit": int(graph_value_limit),
        },
        "baseline": baseline_report,
        "graph": graph_report,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output")
    parser.add_argument("--neo4j-http-uri")
    parser.add_argument("--neo4j-user")
    parser.add_argument("--neo4j-password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--graph-weight", type=float, default=0.8)
    parser.add_argument("--graph-seed-k", type=int, default=12)
    parser.add_argument("--graph-expand-k", type=int, default=100)
    parser.add_argument("--graph-value-limit", type=int, default=16)
    args = parser.parse_args()
    rows = [
        row
        for row in read_jsonl(args.queries)
        if args.split == "all" or str(row.get("split") or "") == args.split
    ]
    graph = None
    if args.neo4j_http_uri:
        if not args.neo4j_user:
            parser.error("--neo4j-http-uri requires --neo4j-user")
        import os

        password = os.environ.get(args.neo4j_password_env)
        if not password:
            parser.error(
                f"Neo4j password environment variable is empty: {args.neo4j_password_env}"
            )
        graph = Neo4jHttpPaperFilter(
            http_uri=args.neo4j_http_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
        )
        graph.healthcheck()
    report = run_ablation(
        db=args.db,
        rows=rows,
        top_k=args.top_k,
        max_queries=args.max_queries,
        graph_filter=graph,
        graph_weight=args.graph_weight,
        graph_seed_k=args.graph_seed_k,
        graph_expand_k=args.graph_expand_k,
        graph_value_limit=args.graph_value_limit,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
