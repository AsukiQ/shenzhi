#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.trajectbench_skillrouter_eval import run_trajectbench_skillrouter_finetuned_eval


def _optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip().upper() in {"", "ALL", "NONE"}:
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TrajectBench SkillRouter finetuned adapter on complete-CLSTR Stage0 top-M candidate rows."
    )
    parser.add_argument("--trajectories_path", default=".tmp/stage4_sanitized_trajectbench/trajectories_visible_global_full.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--routing_checkpoint_path",
        default="outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/checkpoints/clstr_unified_retrieval_v2-step300.pt",
    )
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument(
        "--adapter_checkpoint_path",
        default="outputs/unified_skillrouter_finetune/function_aug_v2_full_20260628_212251/checkpoints/unified_skillrouter_finetune-step2000.pt",
    )
    parser.add_argument("--prebuilt_stage4_rows_path")
    parser.add_argument("--max_rows")
    parser.add_argument("--eval_rows", type=int, default=256)
    parser.add_argument("--eval_split_mode", default="trajectory_prefix", choices=["sequential_tail", "trajectory_prefix"])
    parser.add_argument("--trajectory_eval_steps", type=int, default=1)
    parser.add_argument("--candidate_count", type=int)
    parser.add_argument("--stage0_top_m", type=int, default=350)
    parser.add_argument("--stage0_positive_missing_policy", default="skip", choices=["skip", "inject", "skip_or_inject_with_provenance"])
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    report = run_trajectbench_skillrouter_finetuned_eval(
        trajectories_path=args.trajectories_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        routing_checkpoint_path=args.routing_checkpoint_path,
        model_name_or_path=args.model_name_or_path,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
        prebuilt_stage4_rows_path=args.prebuilt_stage4_rows_path,
        max_rows=_optional_int(args.max_rows),
        eval_rows=args.eval_rows,
        eval_split_mode=args.eval_split_mode,
        trajectory_eval_steps=args.trajectory_eval_steps,
        candidate_count=args.candidate_count,
        stage0_top_m=args.stage0_top_m,
        stage0_positive_missing_policy=args.stage0_positive_missing_policy,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
