#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bge_toolrex_reranker_train import run_bge_toolrex_reranker_train


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip().upper() == "ALL":
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="BGE ToolRex Tool-Rank true/false pair training.")
    parser.add_argument("--tool_embed_output_dir", required=True)
    parser.add_argument("--tool_embed_checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reranker_model_path", default="models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--negatives_per_query", type=int, default=5)
    parser.add_argument("--encoder_max_length", type=int, default=512)
    parser.add_argument("--encoder_batch_size", type=int, default=64)
    parser.add_argument("--rank_batch_size", type=int, default=256)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=400)
    parser.add_argument("--max_train_queries")
    parser.add_argument("--max_eval_queries")
    parser.add_argument("--max_train_pairs")
    parser.add_argument("--max_eval_pairs")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--resume_checkpoint_path")
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=2)
    parser.add_argument("--candidate_cache_dir")
    args = parser.parse_args()

    report = run_bge_toolrex_reranker_train(
        tool_embed_output_dir=args.tool_embed_output_dir,
        tool_embed_checkpoint_path=args.tool_embed_checkpoint_path,
        output_dir=args.output_dir,
        reranker_model_path=args.reranker_model_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        negatives_per_query=args.negatives_per_query,
        encoder_max_length=args.encoder_max_length,
        encoder_batch_size=args.encoder_batch_size,
        rank_batch_size=args.rank_batch_size,
        reranker_max_length=args.reranker_max_length,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        max_train_queries=_optional_int(args.max_train_queries),
        max_eval_queries=_optional_int(args.max_eval_queries),
        max_train_pairs=_optional_int(args.max_train_pairs),
        max_eval_pairs=_optional_int(args.max_eval_pairs),
        freeze_backbone=args.freeze_backbone,
        resume_checkpoint_path=args.resume_checkpoint_path,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        candidate_cache_dir=args.candidate_cache_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
