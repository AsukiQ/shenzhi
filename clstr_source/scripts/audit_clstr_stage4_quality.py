#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage4_quality_gate import DEFAULT_EXPECTED_BENCHMARKS, audit_stage4_act_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit full-pool causal CLSTR Stage 4 quality before evaluation.")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_memory_stage4_full_pool_counterfactual")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--train_report_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--output_path")
    parser.add_argument("--selection_path", default=None)
    parser.add_argument("--stage2_baseline_report_path", default=None)
    parser.add_argument("--min_steps", type=int, default=2000)
    parser.add_argument("--first_window", type=int, default=200)
    parser.add_argument("--last_window", type=int, default=200)
    parser.add_argument("--min_stage4_recall_at_5_tail", type=float, default=0.5)
    parser.add_argument("--max_stage4_act_loss_increase", type=float, default=0.2)
    parser.add_argument(
        "--expected_transition_scoring_mode",
        default=None,
        help="Optional override; by default infer the expected scoring mode from route_scorer.",
    )
    parser.add_argument(
        "--expected_transition_residual_lambda",
        type=float,
        default=None,
        help="Optional override; by default infer the expected residual weight from route_scorer.",
    )
    parser.add_argument("--expected_benchmarks", nargs="*", default=DEFAULT_EXPECTED_BENCHMARKS)
    parser.add_argument(
        "--allow_frozen_transition",
        action="store_true",
        help="Legacy/ablation mode only: do not require Stage4 to train TransitionPredictor.",
    )
    parser.add_argument(
        "--allow_unprotected_routing_init",
        action="store_true",
        help=(
            "Legacy/ablation mode only: allow Stage4 artifacts whose checkpoint init report "
            "does not prove Stage0 routing foundation protection."
        ),
    )
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stage4_act_quality(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        train_report_path=args.train_report_path,
        metrics_path=args.metrics_path,
        output_path=args.output_path,
        min_steps=args.min_steps,
        first_window=args.first_window,
        last_window=args.last_window,
        expected_benchmarks=args.expected_benchmarks,
        require_transition_trainable=not args.allow_frozen_transition,
        require_routing_foundation_protection=not args.allow_unprotected_routing_init,
        min_stage4_recall_at_5_tail=args.min_stage4_recall_at_5_tail,
        max_stage4_act_loss_increase=args.max_stage4_act_loss_increase,
        expected_transition_scoring_mode=args.expected_transition_scoring_mode,
        expected_transition_residual_lambda=args.expected_transition_residual_lambda,
        selection_path=args.selection_path,
        stage2_baseline_report_path=args.stage2_baseline_report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
