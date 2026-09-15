#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bge_reranker import BGERerankerConfig, BGERerankerScorer
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test local BGE-M3 embedding and BGE reranker backends.")
    parser.add_argument("--embedding_model_name_or_path", default="models/BAAI/bge-m3")
    parser.add_argument("--reranker_model_name_or_path", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output_dir", default="outputs/bge_backend_smoke")
    parser.add_argument("--embedding_batch_size", type=int, default=2)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--reranker_batch_size", type=int, default=2)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--skip_reranker", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    def progress(phase: str, **extra: object) -> None:
        _write_json(
            progress_path,
            {
                "status": "running",
                "phase": phase,
                "elapsed_seconds": time.perf_counter() - started,
                "embedding_model_name_or_path": args.embedding_model_name_or_path,
                "reranker_model_name_or_path": None if args.skip_reranker else args.reranker_model_name_or_path,
                **extra,
            },
        )

    queries = [
        "Instruct: Given a task description, retrieve the most relevant skill document.\nQuery: summarize a PDF and extract key claims",
        "Instruct: Given a task description, retrieve the most relevant skill document.\nQuery: call an API to book a flight",
    ]
    documents = [
        "pdf summarization | Extract key points, tables, and claims from PDF documents.",
        "flight booking api | Search flights, compare fares, and reserve an itinerary.",
    ]

    progress("encoding_embedding_texts", text_count=len(queries) + len(documents))
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=args.embedding_model_name_or_path,
        texts=queries,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    ).float()
    doc_embs = _encode_skillrouter_texts(
        model_name_or_path=args.embedding_model_name_or_path,
        texts=documents,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    ).float()
    query_embs = F.normalize(query_embs, p=2, dim=-1)
    doc_embs = F.normalize(doc_embs, p=2, dim=-1)
    cosine = query_embs @ doc_embs.t()

    reranker_scores: list[float] | None = None
    if not args.skip_reranker:
        progress("scoring_reranker_pairs", embedding_dim=int(query_embs.size(-1)))
        scorer = BGERerankerScorer(
            BGERerankerConfig(
                model_name_or_path=args.reranker_model_name_or_path,
                torch_dtype=args.torch_dtype,
                local_files_only=True,
                batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length,
            )
        )
        reranker_scores = scorer.score_pairs(queries=queries, documents=documents)

    metrics = {
        "status": "ok",
        "embedding_model_name_or_path": args.embedding_model_name_or_path,
        "reranker_model_name_or_path": None if args.skip_reranker else args.reranker_model_name_or_path,
        "embedding_dim": int(query_embs.size(-1)),
        "query_count": len(queries),
        "document_count": len(documents),
        "query_norms": [float(item) for item in torch.linalg.norm(query_embs, dim=-1).cpu().tolist()],
        "document_norms": [float(item) for item in torch.linalg.norm(doc_embs, dim=-1).cpu().tolist()],
        "cosine_scores": cosine.detach().cpu().tolist(),
        "reranker_scores": reranker_scores,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(progress_path, {**metrics, "phase": "done"})
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
