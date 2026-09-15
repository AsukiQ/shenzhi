"""Exact multi-process BM25 evaluation for the full frozen paper test split."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any

from evaluate_retrieval import _dcg, _filters, _judgments, read_jsonl
from paper_search import PaperSearchIndex


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "query_id",
        "query_text",
        "query",
        "positive_skill_id",
        "positive_skill_ids",
        "positive_paper_id",
        "positive_paper_ids",
        "judgments",
        "qrels",
        "filters",
        "year_gte",
        "year_lte",
        "conference",
        "author",
        "keyword",
        "subject",
    )
    return {name: row[name] for name in fields if name in row}


def _evaluate_chunk(
    db_path: str,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    index = PaperSearchIndex(db_path)
    totals: Counter[str] = Counter()
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    latencies_ms: list[float] = []
    misses: list[dict[str, Any]] = []
    evaluated = 0
    try:
        for order, row in indexed_rows:
            query_id = str(row.get("query_id") or "").strip()
            query = str(row.get("query_text") or row.get("query") or "").strip()
            judgments = _judgments(row)
            positives = {
                paper_id for paper_id, relevance in judgments.items() if relevance > 0
            }
            if not query_id or not query or not positives:
                continue
            started = time.perf_counter()
            filters = _filters(row)
            has_filters = bool(
                filters.year_gte is not None
                or filters.year_lte is not None
                or filters.conference
                or filters.author
                or filters.keyword
                or filters.subject
            )
            if has_filters:
                results, _state = index.search(
                    query,
                    filters=filters,
                    top_k=top_k,
                )
                ranked_ids = [result.paper_id for result in results]
            else:
                # The frozen weak test has no metadata constraints.  Directly
                # evaluating BM25 Top-K is rank-identical to search(), while
                # avoiding hydration of 1,000 full abstracts per query.
                ranked_ids = [
                    hit.paper_id for hit in index.recall(query, limit=top_k)
                ]
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            ranks = [
                rank
                for rank, paper_id in enumerate(ranked_ids, start=1)
                if paper_id in positives
            ]
            first_rank = min(ranks) if ranks else None
            reciprocal_rank_sum += 0.0 if first_rank is None else 1.0 / first_rank
            for cutoff in (1, 5, 10, 20, 50, 100):
                if cutoff <= top_k:
                    retrieved = len(positives.intersection(ranked_ids[:cutoff]))
                    totals[f"hit_at_{cutoff}"] += int(retrieved > 0)
                    totals[f"recall_at_{cutoff}"] += float(retrieved) / len(positives)
            relevances = [
                int(judgments.get(paper_id, 0)) for paper_id in ranked_ids[:top_k]
            ]
            ideal = sorted(
                [int(value) for value in judgments.values() if int(value) > 0],
                reverse=True,
            )[:top_k]
            ideal.extend([0] * max(0, top_k - len(ideal)))
            ideal_dcg = _dcg(ideal)
            ndcg_sum += _dcg(relevances) / ideal_dcg if ideal_dcg else 0.0
            evaluated += 1
            if first_rank is None:
                misses.append(
                    {
                        "order": order,
                        "query_id": query_id,
                        "query": query,
                        "positive_ids": sorted(positives),
                        "returned_ids": ranked_ids[:10],
                    }
                )
    finally:
        index.close()
    return {
        "evaluated": evaluated,
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "ndcg_sum": ndcg_sum,
        "totals": dict(totals),
        "latencies_ms": latencies_ms,
        "misses": misses,
    }


def evaluate_parallel(
    *,
    db_path: str | Path,
    rows: list[dict[str, Any]],
    top_k: int,
    workers: int,
) -> dict[str, Any]:
    indexed = list(enumerate(rows))
    worker_count = max(1, min(int(workers), len(indexed)))
    chunks = [indexed[offset::worker_count] for offset in range(worker_count)]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        partials = list(
            executor.map(
                _evaluate_chunk,
                [str(Path(db_path).resolve())] * worker_count,
                chunks,
                [int(top_k)] * worker_count,
            )
        )
    evaluated = sum(int(row["evaluated"]) for row in partials)
    if not evaluated:
        raise ValueError("no evaluable BM25 rows")
    totals: Counter[str] = Counter()
    latencies_ms: list[float] = []
    misses: list[dict[str, Any]] = []
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    for row in partials:
        totals.update(row["totals"])
        latencies_ms.extend(float(value) for value in row["latencies_ms"])
        misses.extend(row["misses"])
        reciprocal_rank_sum += float(row["reciprocal_rank_sum"])
        ndcg_sum += float(row["ndcg_sum"])
    ordered_latency = sorted(latencies_ms)
    metrics = {
        "mrr": reciprocal_rank_sum / evaluated,
        f"ndcg_at_{top_k}": ndcg_sum / evaluated,
        **{name: float(value) / evaluated for name, value in sorted(totals.items())},
    }
    misses.sort(key=lambda row: int(row["order"]))
    for row in misses:
        row.pop("order", None)
    return {
        "status": "ok",
        "query_count": evaluated,
        "top_k": int(top_k),
        "worker_count": worker_count,
        "metrics": metrics,
        "latency": {
            "mean_ms": statistics.fmean(latencies_ms),
            "p50_ms": ordered_latency[int(0.50 * (len(ordered_latency) - 1))],
            "p95_ms": ordered_latency[int(0.95 * (len(ordered_latency) - 1))],
            "max_ms": ordered_latency[-1],
        },
        "miss_examples": misses[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-local-copy", action="store_true")
    args = parser.parse_args()
    rows = [
        _compact_row(row)
        for row in read_jsonl(args.queries)
        if args.split == "all" or str(row.get("split") or "") == args.split
    ]
    if not rows:
        raise ValueError(f"no query rows found for split={args.split}")
    started = time.perf_counter()
    if args.no_local_copy:
        report = evaluate_parallel(
            db_path=args.db,
            rows=rows,
            top_k=max(1, int(args.top_k)),
            workers=args.workers,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="shenzhi-bm25-eval-") as tmp:
            local_db = Path(tmp) / "papers.db"
            shutil.copyfile(args.db, local_db)
            report = evaluate_parallel(
                db_path=local_db,
                rows=rows,
                top_k=max(1, int(args.top_k)),
                workers=args.workers,
            )
    report.update(
        {
            "backend": "bm25",
            "split": args.split,
            "protocol": "exact_multiprocess_bm25_v1",
            "wall_time_seconds": time.perf_counter() - started,
            "interpretation": (
                "Weak-bootstrap title/abstract queries are for regression, not "
                "as sole evidence of user-query generalization."
            ),
        }
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
