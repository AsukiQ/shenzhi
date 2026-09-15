#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_teacher_rollout import extract_qwen_teacher_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen3 train-split ALFWorld rollout with official expert correction.")
    parser.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/alfworld_data")
    parser.add_argument("--data_output_dir", default="data/alfworld_qwen3_expert_corrected_rollout")
    parser.add_argument("--report_output_dir", default="outputs/alfworld_qwen3_expert_corrected_rollout")
    parser.add_argument("--max_games", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--qwen_model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    report = extract_qwen_teacher_rollout(
        official_repo=Path(args.official_repo),
        data_dir=Path(args.data_dir),
        data_output_dir=Path(args.data_output_dir),
        report_output_dir=Path(args.report_output_dir),
        max_games=args.max_games,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        qwen_model_name_or_path=Path(args.qwen_model_name_or_path),
        cache_dir=Path(args.cache_dir),
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
