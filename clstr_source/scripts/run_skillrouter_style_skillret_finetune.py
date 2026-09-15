#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillrouter_style import run_skillrouter_style_finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkillRouter-style bi-encoder adapter finetune on SKILLRET.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--output_dir", default="outputs/skillret_warmup_skillrouter_style")
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--learning_rate", type=float, default=1.0e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--skill_batch_size", type=int, default=1)
    parser.add_argument("--query_batch_size", type=int)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--tokenizer_padding_side", default="left")
    parser.add_argument("--projection_init", default="identity")
    args = parser.parse_args()
    report = run_skillrouter_style_finetune(
        data_root=args.data_root,
        base_model_name=args.base_model_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        max_skills=args.max_skills,
        max_queries=args.max_queries,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        max_length=args.max_length,
        skill_batch_size=args.skill_batch_size,
        query_batch_size=args.query_batch_size,
        torch_dtype=args.torch_dtype,
        tokenizer_padding_side=args.tokenizer_padding_side,
        projection_init=args.projection_init,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
