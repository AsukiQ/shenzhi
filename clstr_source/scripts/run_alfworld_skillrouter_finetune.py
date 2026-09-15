#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_skillrouter_finetune import run_alfworld_skillrouter_action_adapter_finetune


def main() -> int:
    parser = argparse.ArgumentParser(description="Finetune a lightweight SkillRouter action adapter on ALFWorld train replay.")
    parser.add_argument("--replay_path", default="data/alfworld_policy_replay/train_replay.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--projection_init", default="identity", choices=["identity", "default"])
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--torch_dtype", default="bfloat16")
    args = parser.parse_args()

    report = run_alfworld_skillrouter_action_adapter_finetune(
        replay_path=args.replay_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        projection_init=args.projection_init,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
