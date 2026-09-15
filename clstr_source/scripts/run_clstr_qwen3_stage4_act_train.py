#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_stage4_act_train import run_clstr_qwen3_stage4_act_train


def _parse_allowed_benchmarks(value: str | None) -> set[str] | None:
    if value is None:
        return None
    benchmarks = {item.strip() for item in str(value).split(",") if item.strip()}
    return benchmarks or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CLSTR Stage 4 transition-conditioned next-skill L_act with a frozen Qwen3 encoder."
    )
    parser.add_argument("--trajectories_path", default="data/clstr_unified_pretrain_v2/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v2/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage4_act")
    parser.add_argument("--qwen_model_path", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument(
        "--routing_checkpoint_path",
        required=True,
        help="Stage 1 retrieval warmup checkpoint carrying routing foundation keys.",
    )
    parser.add_argument(
        "--head_checkpoint_path",
        required=True,
        help="Stage 2 or Stage 3 checkpoint carrying transition/ACT head keys.",
    )
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--candidate_count", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--allowed_benchmarks",
        default=None,
        help="Comma-separated benchmark names retained for Stage 4, e.g. traject_bench,toolbench_g3.",
    )
    parser.add_argument(
        "--train_transition",
        action="store_true",
        help="Also train TransitionPredictor. Default trains TransHead/action_proj only.",
    )
    args = parser.parse_args()

    report = run_clstr_qwen3_stage4_act_train(
        trajectories_path=Path(args.trajectories_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        qwen_model_path=Path(args.qwen_model_path),
        cache_dir=Path(args.cache_dir),
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        head_checkpoint_path=Path(args.head_checkpoint_path),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        candidate_count=args.candidate_count,
        train_transition=args.train_transition,
        max_rows=args.max_rows,
        allowed_benchmarks=_parse_allowed_benchmarks(args.allowed_benchmarks),
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        model_dim=args.model_dim,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
