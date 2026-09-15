#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_full_base_train import run_clstr_qwen3_full_base_train


def _parse_allowed_benchmarks(value: str | None) -> set[str] | None:
    if value is None:
        return None
    benchmarks = {item.strip() for item in str(value).split(",") if item.strip()}
    return benchmarks or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLSTR-native ACT heads with frozen Qwen3-8B external encoder.")
    parser.add_argument("--train_path", default="data/clstr_full_base_train_trans_skill_ce/train.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_full_base_train_trans_skill_ce/skills.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_qwen3_8b_full_base_gate")
    parser.add_argument("--qwen_model_path", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--transition_loss_weight", type=float, default=0.05)
    parser.add_argument("--transition_skill_ce_loss_weight", type=float, default=0.2)
    parser.add_argument("--belief_loss_weight", type=float, default=0.1)
    parser.add_argument("--stop_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_num_negatives", type=int, default=32)
    parser.add_argument("--retrieval_hard_ratio", type=float, default=0.5)
    parser.add_argument(
        "--embedding_cache_mode",
        default="auto",
        choices=["auto", "always", "never"],
        help="Control full-dataset embedding cache before training. auto skips large Qwen datasets.",
    )
    parser.add_argument("--embedding_cache_max_rows", type=int, default=20000)
    parser.add_argument(
        "--routing_checkpoint_path",
        default=None,
        help="Optional Stage 1 retrieval warmup checkpoint used to initialize routing foundation before Stage 2.",
    )
    parser.add_argument(
        "--include_available_actions_in_state",
        action="store_true",
        help="Augment L_policy state text with AVAILABLE ACTIONS + CLSTR skill-routing query.",
    )
    parser.add_argument(
        "--allowed_benchmarks",
        default=None,
        help="Comma-separated benchmark names retained for Stage 2 training, e.g. traject_bench,toolbench_g3.",
    )
    parser.add_argument("--stage0_top_m", type=int, default=None)
    parser.add_argument(
        "--stage0_positive_missing_policy",
        default="skip",
        choices=["skip", "inject"],
    )
    parser.add_argument("--allow_full_pool_stage2_debug", action="store_true")
    args = parser.parse_args()

    report = run_clstr_qwen3_full_base_train(
        train_path=Path(args.train_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        qwen_model_path=Path(args.qwen_model_path),
        cache_dir=Path(args.cache_dir),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        model_dim=args.model_dim,
        loss_weights={
            "L_policy": args.policy_loss_weight,
            "L_trans": args.transition_loss_weight,
            "L_trans_skill_ce": args.transition_skill_ce_loss_weight,
            "belief": args.belief_loss_weight,
            "STOP": args.stop_loss_weight,
            "routing": args.retrieval_loss_weight,
        },
        retrieval_num_negatives=args.retrieval_num_negatives,
        retrieval_hard_ratio=args.retrieval_hard_ratio,
        include_available_actions_in_state=args.include_available_actions_in_state,
        routing_checkpoint_path=Path(args.routing_checkpoint_path) if args.routing_checkpoint_path else None,
        embedding_cache_mode=args.embedding_cache_mode,
        embedding_cache_max_rows=args.embedding_cache_max_rows,
        allowed_benchmarks=_parse_allowed_benchmarks(args.allowed_benchmarks),
        stage0_top_m=args.stage0_top_m,
        stage0_positive_missing_policy=args.stage0_positive_missing_policy,
        allow_full_pool_stage2_debug=args.allow_full_pool_stage2_debug,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
