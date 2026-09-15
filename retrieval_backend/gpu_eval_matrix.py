"""Evaluate dense, hybrid and reranked retrieval while loading each model once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_retrieval import evaluate_rows, read_jsonl
from paper_search import PaperSearchIndex
from retrieval_pipeline import (
    DensePaperSearch,
    HybridPaperSearch,
    SkillRouterPaperReranker,
    Stage0DenseRetriever,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--clstr-source", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--recall-k", type=int, default=1000)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--rerank-max-queries", type=int, default=40)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    args = parser.parse_args()

    rows = list(read_jsonl(args.queries))
    if not rows:
        raise ValueError("GPU evaluation query file is empty")
    lexical = PaperSearchIndex(args.db)
    dense = Stage0DenseRetriever(
        checkpoint_path=args.checkpoint,
        skills_path=args.skills,
        clstr_source=args.clstr_source,
        model_cache_dir=args.model_cache_dir,
        device="cuda",
    )
    dense_only = DensePaperSearch(
        lexical,
        dense,
        recall_k=args.recall_k,
    )
    hybrid = HybridPaperSearch(
        lexical,
        dense_retriever=dense,
        recall_k=args.recall_k,
    )
    reports = {
        "dense": evaluate_rows(dense_only, rows, top_k=args.top_k),
        "hybrid": evaluate_rows(hybrid, rows, top_k=args.top_k),
    }
    reranker = SkillRouterPaperReranker(
        model_path=args.reranker_model,
        clstr_source=args.clstr_source,
        batch_size=args.reranker_batch_size,
        max_length=1024,
        device="cuda",
    )
    reranked = HybridPaperSearch(
        lexical,
        dense_retriever=dense,
        reranker=reranker,
        recall_k=args.recall_k,
        rerank_k=args.rerank_k,
    )
    reports["hybrid_reranker"] = evaluate_rows(
        reranked,
        rows,
        top_k=min(args.top_k, args.rerank_k),
        max_queries=args.rerank_max_queries,
    )
    report = {
        "status": "ok",
        "protocol": "weak_test_fixed_hash_gpu_matrix_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "skills": str(Path(args.skills).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "query_pool_size": len(rows),
        "rerank_max_queries": args.rerank_max_queries,
        "dense_load_report": dense.load_report,
        "reports": reports,
        "interpretation": (
            "Weak-bootstrap queries are text-derived and are suitable for regression, "
            "not as the sole evidence of user-query generalization."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
