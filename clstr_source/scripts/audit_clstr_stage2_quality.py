#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage_quality_common import EXPECTED_TRANSITION_RESIDUAL_LAMBDA, EXPECTED_TRANSITION_SCORING_MODE
from clstr.stage2_quality_gate import DEFAULT_REQUIRED_LOSS_TERMS, audit_stage2_full_base_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR Stage 2 full-base quality before Stage 3.")
    parser.add_argument(
        "--output_dir",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025",
    )
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--train_report_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--output_path")
    parser.add_argument("--min_steps", type=int, default=10000)
    parser.add_argument("--first_window", type=int, default=200)
    parser.add_argument("--last_window", type=int, default=200)
    parser.add_argument("--min_loss_drop", type=float, default=0.02)
    parser.add_argument("--max_transition_recall_at_5_drop", type=float, default=0.1)
    parser.add_argument("--max_transition_skill_ce_increase", type=float, default=0.2)
    parser.add_argument("--min_transition_recall_at_5_tail", type=float, default=0.5)
    parser.add_argument("--max_stage2_recall_at_5_drop_vs_stage0_prior", type=float, default=0.01)
    parser.add_argument("--max_stage2_mrr_drop_vs_stage0_prior", type=float, default=0.01)
    parser.add_argument("--max_stage2_worse_than_stage0_prior_fraction", type=float, default=0.5)
    parser.add_argument("--expected_transition_scoring_mode", default=EXPECTED_TRANSITION_SCORING_MODE)
    parser.add_argument("--expected_transition_residual_lambda", type=float, default=EXPECTED_TRANSITION_RESIDUAL_LAMBDA)
    parser.add_argument("--expected_route_scorer", default="unified_memory")
    parser.add_argument("--expected_next_skill_pool_mode", default="full_pool", choices=["stage0_candidates", "full_pool"])
    parser.add_argument("--static_route_anchor_max_regression", type=float, default=0.005)
    parser.add_argument("--required_loss_terms", nargs="*", default=DEFAULT_REQUIRED_LOSS_TERMS)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stage2_full_base_quality(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        train_report_path=args.train_report_path,
        metrics_path=args.metrics_path,
        output_path=args.output_path,
        min_steps=args.min_steps,
        first_window=args.first_window,
        last_window=args.last_window,
        min_loss_drop=args.min_loss_drop,
        required_loss_terms=args.required_loss_terms,
        max_transition_recall_at_5_drop=args.max_transition_recall_at_5_drop,
        max_transition_skill_ce_increase=args.max_transition_skill_ce_increase,
        min_transition_recall_at_5_tail=args.min_transition_recall_at_5_tail,
        max_stage2_recall_at_5_drop_vs_stage0_prior=args.max_stage2_recall_at_5_drop_vs_stage0_prior,
        max_stage2_mrr_drop_vs_stage0_prior=args.max_stage2_mrr_drop_vs_stage0_prior,
        max_stage2_worse_than_stage0_prior_fraction=args.max_stage2_worse_than_stage0_prior_fraction,
        expected_transition_scoring_mode=args.expected_transition_scoring_mode,
        expected_transition_residual_lambda=args.expected_transition_residual_lambda,
        expected_route_scorer=args.expected_route_scorer,
        expected_next_skill_pool_mode=args.expected_next_skill_pool_mode,
        static_route_anchor_max_regression=args.static_route_anchor_max_regression,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
