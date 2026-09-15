"""Evaluate a retrieval backend against paper query/qrel JSONL data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

try:
    from .paper_search import PaperSearchIndex, SearchFilters
    from .retrieval_pipeline import (
        DensePaperSearch,
        HybridPaperSearch,
        SkillRouterPaperReranker,
        Stage0DenseRetriever,
    )
except ImportError:
    from paper_search import PaperSearchIndex, SearchFilters
    from retrieval_pipeline import (
        DensePaperSearch,
        HybridPaperSearch,
        SkillRouterPaperReranker,
        Stage0DenseRetriever,
    )


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _judgments(row: dict[str, Any]) -> dict[str, int]:
    graded = row.get("judgments") or row.get("qrels") or []
    output: dict[str, int] = {}
    if isinstance(graded, dict):
        for paper_id, relevance in graded.items():
            if relevance is not None:
                output[str(paper_id)] = int(relevance)
    elif isinstance(graded, list):
        for item in graded:
            if not isinstance(item, dict):
                continue
            paper_id = item.get("paper_id") or item.get("skill_id")
            relevance = item.get("relevance", item.get("label", 1))
            if paper_id and relevance is not None:
                output[str(paper_id)] = int(relevance)
    values = row.get("positive_skill_ids") or row.get("positive_paper_ids") or []
    if isinstance(values, str):
        values = [values]
    single = row.get("positive_skill_id") or row.get("positive_paper_id")
    if single:
        values = [*values, single]
    for value in values:
        if str(value):
            output.setdefault(str(value), 1)
    return output


def _filters(row: dict[str, Any]) -> SearchFilters:
    raw = row.get("filters") or {}
    if not isinstance(raw, dict):
        raw = {}

    def values(name: str) -> list[str]:
        value = raw.get(name, row.get(name))
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value if str(item).strip()]

    return SearchFilters(
        year_gte=raw.get("year_gte", row.get("year_gte")),
        year_lte=raw.get("year_lte", row.get("year_lte")),
        conference=values("conference"),
        author=values("author"),
        keyword=values("keyword"),
        subject=values("subject"),
    )


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2.0 ** int(relevance) - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(relevances, start=1)
    )


def evaluate_rows(
    engine: Any,
    rows: Iterable[dict[str, Any]],
    *,
    top_k: int = 100,
    max_queries: int | None = None,
) -> dict[str, Any]:
    top_k = max(1, int(top_k))
    totals: Counter[str] = Counter()
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0
    examples: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for row in rows:
        if max_queries is not None and evaluated >= int(max_queries):
            break
        query_id = str(row.get("query_id") or "").strip()
        query = str(row.get("query_text") or row.get("query") or "").strip()
        judgments = _judgments(row)
        positives = {paper_id for paper_id, relevance in judgments.items() if relevance > 0}
        if not query_id or not query or not positives:
            continue
        started = time.perf_counter()
        results, _state = engine.search(query, filters=_filters(row), top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        ranked_ids = [result.paper_id for result in results]
        ranks = [index + 1 for index, paper_id in enumerate(ranked_ids) if paper_id in positives]
        first_rank = min(ranks) if ranks else None
        reciprocal_rank_sum += 0.0 if first_rank is None else 1.0 / first_rank
        for cutoff in (1, 5, 10, 20, 50, 100):
            if cutoff <= top_k:
                retrieved = len(positives.intersection(ranked_ids[:cutoff]))
                totals[f"hit_at_{cutoff}"] += int(retrieved > 0)
                totals[f"recall_at_{cutoff}"] += float(retrieved) / len(positives)
        relevances = [int(judgments.get(paper_id, 0)) for paper_id in ranked_ids[:top_k]]
        ideal = sorted(
            [int(value) for value in judgments.values() if int(value) > 0],
            reverse=True,
        )[:top_k]
        ideal.extend([0] * max(0, top_k - len(ideal)))
        ideal_dcg = _dcg(ideal)
        ndcg_sum += _dcg(relevances) / ideal_dcg if ideal_dcg else 0.0
        evaluated += 1
        if len(examples) < 20 and first_rank is None:
            examples.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "positive_ids": sorted(positives),
                    "returned_ids": ranked_ids[:10],
                }
            )
    if not evaluated:
        raise ValueError("no evaluable retrieval rows")
    metrics: dict[str, float] = {
        "mrr": reciprocal_rank_sum / evaluated,
        f"ndcg_at_{top_k}": ndcg_sum / evaluated,
    }
    for key, value in sorted(totals.items()):
        metrics[key] = value / evaluated
    ordered_latency = sorted(latencies_ms)
    latency = {
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": ordered_latency[int(0.50 * (len(ordered_latency) - 1))],
        "p95_ms": ordered_latency[int(0.95 * (len(ordered_latency) - 1))],
        "max_ms": ordered_latency[-1],
    }
    return {
        "status": "ok",
        "query_count": evaluated,
        "top_k": top_k,
        "metrics": metrics,
        "latency": latency,
        "miss_examples": examples,
    }


def build_engine(args: argparse.Namespace) -> tuple[Any, str]:
    lexical = PaperSearchIndex(args.db)
    if args.backend == "bm25" and not args.reranker_model:
        return lexical, "bm25"
    dense = None
    if args.backend in {"dense", "hybrid"}:
        missing = [
            name
            for name, value in (
                ("--stage0-checkpoint", args.stage0_checkpoint),
                ("--stage0-skills", args.stage0_skills),
                ("--clstr-source", args.clstr_source),
                ("--model-cache-dir", args.model_cache_dir),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"{args.backend} backend requires: {', '.join(missing)}")
        dense = Stage0DenseRetriever(
            checkpoint_path=args.stage0_checkpoint,
            skills_path=args.stage0_skills,
            clstr_source=args.clstr_source,
            model_cache_dir=args.model_cache_dir,
            device=args.device,
        )
    reranker = None
    if args.reranker_model:
        if not args.clstr_source:
            raise ValueError("--reranker-model requires --clstr-source")
        reranker = SkillRouterPaperReranker(
            model_path=args.reranker_model,
            clstr_source=args.clstr_source,
            batch_size=args.reranker_batch_size,
            device=args.device,
        )
    if args.backend == "dense":
        if reranker is not None:
            raise ValueError("dense-only evaluation does not support --reranker-model")
        assert dense is not None
        return (
            DensePaperSearch(lexical, dense, recall_k=args.recall_k),
            "dense",
        )
    return (
        HybridPaperSearch(
            lexical,
            dense_retriever=dense,
            reranker=reranker,
            lexical_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            recall_k=args.recall_k,
            rerank_k=args.rerank_k,
        ),
        "+".join(
            [
                "bm25",
                *(["dense"] if dense is not None else []),
                *(["reranker"] if reranker is not None else []),
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shenzhi paper retrieval")
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output")
    parser.add_argument("--backend", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--stage0-checkpoint")
    parser.add_argument("--stage0-skills")
    parser.add_argument("--clstr-source")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--recall-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--device")
    args = parser.parse_args()
    rows = (
        row
        for row in read_jsonl(args.queries)
        if args.split == "all" or str(row.get("split") or "") == args.split
    )
    engine, backend_name = build_engine(args)
    report = evaluate_rows(
        engine,
        rows,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    report["backend"] = backend_name
    report["split"] = args.split
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
