#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bge_sr_embedding_full_train import run_bge_sr_embedding_full_train


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip().upper() == "ALL":
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-style BGE SkillRouter embedding full encoder finetune.")
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean")
    parser.add_argument("--prepared_corpus_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_model_path", default="models/BAAI/bge-m3")
    parser.add_argument("--max_rows")
    parser.add_argument("--max_skills")
    parser.add_argument("--eval_rows", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--negatives_per_query", type=int, default=7)
    parser.add_argument("--hard_negative_top_k", type=int, default=32)
    parser.add_argument("--max_hard_negative_queries")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--mine_score_batch_size", type=int, default=128)
    parser.add_argument("--torch_dtype", default="float32")
    parser.add_argument("--use_bf16_autocast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mining_use_bf16_autocast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=400)
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=2)
    parser.add_argument("--resume_checkpoint_path")
    parser.add_argument("--nonfinite_gradient_action", default="error", choices=["error", "zero"])
    parser.add_argument("--max_nonfinite_gradient_fraction", type=float, default=0.0)
    args = parser.parse_args()

    report = run_bge_sr_embedding_full_train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        prepared_corpus_dir=args.prepared_corpus_dir,
        encoder_model_path=args.encoder_model_path,
        max_rows=_optional_int(args.max_rows),
        max_skills=_optional_int(args.max_skills),
        eval_rows=args.eval_rows,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        negatives_per_query=args.negatives_per_query,
        hard_negative_top_k=args.hard_negative_top_k,
        max_hard_negative_queries=_optional_int(args.max_hard_negative_queries),
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        mine_score_batch_size=args.mine_score_batch_size,
        torch_dtype=args.torch_dtype,
        use_bf16_autocast=args.use_bf16_autocast,
        mining_use_bf16_autocast=args.mining_use_bf16_autocast,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        max_checkpoints_to_keep=args.max_checkpoints_to_keep,
        resume_checkpoint_path=args.resume_checkpoint_path,
        nonfinite_gradient_action=args.nonfinite_gradient_action,
        max_nonfinite_gradient_fraction=args.max_nonfinite_gradient_fraction,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
