#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import (
    TRANSITION_INVENTORY_MASK_MODES,
    TRANSITION_LOSS_TYPES,
    TRANSITION_POSITIVE_MODES,
)
from clstr.stage2_real_topm_eval import evaluate_stage2_real_topm


def _parse_allowed_benchmarks(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()} or None


def _parse_benchmark_caps(value: str | None) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    caps: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"benchmark cap must use name=value format: {item}")
        name, raw_cap = item.split("=", 1)
        caps[name.strip()] = int(raw_cap.strip())
    return caps or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage2 checkpoint with real Stage0 top-M candidates and no gold candidate injection."
    )
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--stage2_checkpoint_path", required=True)
    parser.add_argument("--train_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage2_real_topm_eval")
    parser.add_argument("--top_m", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allowed_benchmarks", default=None)
    parser.add_argument(
        "--benchmark_caps",
        default="toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000",
    )
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_handoff_sample_multiplier", type=float, default=None)
    parser.add_argument("--sampling_strategy", default="balanced_random", choices=["balanced_deterministic", "balanced_random"])
    parser.add_argument("--embedding_cache_mode", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--embedding_cache_max_rows", type=int, default=20000)
    parser.add_argument("--skill_text_format", default=None)
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--min_current_coverage", type=float, default=0.85)
    parser.add_argument("--min_next_coverage", type=float, default=0.85)
    parser.add_argument("--min_transition_recall_at_5", type=float, default=0.5)
    parser.add_argument("--transition_inventory_mask_mode", default="off", choices=sorted(TRANSITION_INVENTORY_MASK_MODES))
    parser.add_argument("--transition_loss_type", default="cross_entropy", choices=sorted(TRANSITION_LOSS_TYPES))
    parser.add_argument("--transition_positive_mode", default="single", choices=sorted(TRANSITION_POSITIVE_MODES))
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = evaluate_stage2_real_topm(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        top_m=args.top_m,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
        allowed_benchmarks=_parse_allowed_benchmarks(args.allowed_benchmarks),
        benchmark_caps=_parse_benchmark_caps(args.benchmark_caps),
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=args.stage0_handoff_sample_multiplier,
        sampling_strategy=args.sampling_strategy,
        embedding_cache_mode=args.embedding_cache_mode,
        embedding_cache_max_rows=args.embedding_cache_max_rows,
        skill_text_format=args.skill_text_format,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        min_current_coverage=args.min_current_coverage,
        min_next_coverage=args.min_next_coverage,
        min_transition_recall_at_5=args.min_transition_recall_at_5,
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_loss_type=args.transition_loss_type,
        transition_positive_mode=args.transition_positive_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
