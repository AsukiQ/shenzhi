#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage1_heads_quality_gate import (
    DEFAULT_REQUIRED_LOSS_TERMS,
    STAGE1_EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
    audit_stage1_heads_quality,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR Stage1 heads-init quality before Stage2.")
    parser.add_argument(
        "--output_dir",
        default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init",
    )
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--train_report_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--output_path")
    parser.add_argument("--min_steps", type=int, default=3000)
    parser.add_argument("--first_window", type=int, default=100)
    parser.add_argument("--last_window", type=int, default=100)
    parser.add_argument("--min_loss_drop", type=float, default=0.0)
    parser.add_argument("--max_transition_recall_at_5_drop", type=float, default=0.1)
    parser.add_argument("--max_transition_skill_ce_increase", type=float, default=0.2)
    parser.add_argument("--min_transition_recall_at_1_tail", type=float, default=0.35)
    parser.add_argument("--min_transition_recall_at_5_tail", type=float, default=0.7)
    parser.add_argument("--required_loss_terms", nargs="*", default=DEFAULT_REQUIRED_LOSS_TERMS)
    parser.add_argument(
        "--expected_transition_scoring_mode",
        default=STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
        help="Expected Stage1 transition scoring mode; override only for historical audits.",
    )
    parser.add_argument(
        "--expected_transition_residual_lambda",
        type=float,
        default=STAGE1_EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    )
    parser.add_argument("--expected_route_scorer")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stage1_heads_quality(
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
        min_transition_recall_at_1_tail=args.min_transition_recall_at_1_tail,
        min_transition_recall_at_5_tail=args.min_transition_recall_at_5_tail,
        expected_transition_scoring_mode=args.expected_transition_scoring_mode,
        expected_transition_residual_lambda=args.expected_transition_residual_lambda,
        expected_route_scorer=args.expected_route_scorer,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
