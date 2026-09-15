"""Real-model end-to-end smoke for Stage0 dense recall and SkillRouter reranking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from paper_search import PaperSearchIndex
from retrieval_pipeline import (
    HybridPaperSearch,
    SkillRouterPaperReranker,
    Stage0DenseRetriever,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--clstr-source", required=True)
    parser.add_argument("--model-cache-dir", required=True)
    parser.add_argument("--reranker-model", required=True)
    parser.add_argument("--query", default="knowledge graph completion with graph neural networks")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dense = Stage0DenseRetriever(
        checkpoint_path=args.checkpoint,
        skills_path=args.skills,
        clstr_source=args.clstr_source,
        model_cache_dir=args.model_cache_dir,
        device="cuda",
    )
    reranker = SkillRouterPaperReranker(
        model_path=args.reranker_model,
        clstr_source=args.clstr_source,
        batch_size=4,
        max_length=1024,
        device="cuda",
    )
    engine = HybridPaperSearch(
        PaperSearchIndex(args.db),
        dense_retriever=dense,
        reranker=reranker,
        recall_k=30,
        rerank_k=10,
    )
    results, state = engine.search(args.query, top_k=5)
    if not results:
        raise RuntimeError("real-model hybrid smoke returned no papers")
    if not all("reranker" in result.source_scores for result in results):
        raise RuntimeError("real-model hybrid smoke did not apply the reranker")
    report = {
        "status": "ok",
        "query": args.query,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "skills": str(Path(args.skills).resolve()),
        "reranker_model": str(Path(args.reranker_model).resolve()),
        "dense_load_report": dense.load_report,
        "results": [asdict(result) for result in results],
        "state": asdict(state),
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
