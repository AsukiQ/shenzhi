#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage2_transition_error_audit import audit_stage2_transition_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage2 transition ranking from existing train/eval artifacts.")
    parser.add_argument("--train_metrics_path", required=True)
    parser.add_argument("--eval_metrics_path", required=True)
    parser.add_argument("--eval_report_path", required=True)
    parser.add_argument("--baseline_eval_report_path")
    parser.add_argument("--output_json_path")
    parser.add_argument("--output_markdown_path")
    parser.add_argument("--first_window", type=int, default=200)
    parser.add_argument("--tail_window", type=int, default=1000)
    parser.add_argument("--min_meaningful_recall5_gain", type=float, default=0.005)
    parser.add_argument("--min_eval_recall5", type=float, default=0.5)
    args = parser.parse_args()

    report = audit_stage2_transition_outputs(
        train_metrics_path=args.train_metrics_path,
        eval_metrics_path=args.eval_metrics_path,
        eval_report_path=args.eval_report_path,
        baseline_eval_report_path=args.baseline_eval_report_path,
        output_json_path=args.output_json_path,
        output_markdown_path=args.output_markdown_path,
        first_window=args.first_window,
        tail_window=args.tail_window,
        min_meaningful_recall5_gain=args.min_meaningful_recall5_gain,
        min_eval_recall5=args.min_eval_recall5,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
