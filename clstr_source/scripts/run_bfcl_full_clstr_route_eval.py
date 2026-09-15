#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bfcl_route_eval import run_bfcl_full_clstr_route_eval


def _categories(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate full CLSTR route on official BFCL-derived next-function routing rows."
    )
    parser.add_argument(
        "--data_root",
        default=".tmp/benchmark_probe_direct/gorilla-llm__Berkeley-Function-Calling-Leaderboard",
    )
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument("--stage4_checkpoint_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--categories")
    parser.add_argument("--max_rows_per_category", type=int)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_batch_size", type=int, default=16)
    parser.add_argument(
        "--online_memory_mode",
        default="latest_exact",
        choices=["latest_exact", "exact_count", "state_conditioned", "inventory_remaining"],
    )
    parser.add_argument("--online_memory_weight", type=float, default=1.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    parser.add_argument("--transition_residual_lambda", type=float, default=0.25)
    args = parser.parse_args()
    report = run_bfcl_full_clstr_route_eval(
        data_root=args.data_root,
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        output_dir=args.output_dir,
        categories=_categories(args.categories),
        max_rows_per_category=args.max_rows_per_category,
        max_eval_rows=args.max_eval_rows,
        batch_size=args.batch_size,
        stage0_candidate_batch_size=args.stage0_candidate_batch_size,
        online_memory_mode=args.online_memory_mode,
        online_memory_weight=args.online_memory_weight,
        online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
        online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
        transition_residual_lambda=args.transition_residual_lambda,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
