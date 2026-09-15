#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.unified_skillrouter_finetune import run_unified_skillrouter_finetune


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip().upper() == "ALL":
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a SkillRouter-style bi-encoder adapter on unified CLSTR trajectories."
    )
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_model_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--max_rows")
    parser.add_argument("--max_skills")
    parser.add_argument("--eval_rows", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--encoder_max_length", type=int, default=2048)
    parser.add_argument("--encoder_batch_size", type=int, default=16)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--tokenizer_padding_side", default="left")
    parser.add_argument("--projection_init", default="identity", choices=["identity", "default"])
    parser.add_argument("--train_doc_projection", action="store_true")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=400)
    parser.add_argument("--pooling", default="auto", choices=["auto", "last_token", "cls", "masked_mean"])
    parser.add_argument("--query_text_mode", default="auto", choices=["auto", "skillrouter", "raw", "raw_state"])
    args = parser.parse_args()

    report = run_unified_skillrouter_finetune(
        data_root=args.data_root,
        output_dir=args.output_dir,
        encoder_model_path=args.encoder_model_path,
        max_rows=_optional_int(args.max_rows),
        max_skills=_optional_int(args.max_skills),
        eval_rows=args.eval_rows,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        encoder_max_length=args.encoder_max_length,
        encoder_batch_size=args.encoder_batch_size,
        torch_dtype=args.torch_dtype,
        tokenizer_padding_side=args.tokenizer_padding_side,
        projection_init=args.projection_init,
        train_doc_projection=args.train_doc_projection,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        pooling=args.pooling,
        query_text_mode=args.query_text_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
