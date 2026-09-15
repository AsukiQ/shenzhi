#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage0_quality_gate import (
    DEFAULT_EXPECTED_SOURCES,
    DEFAULT_REQUIRED_RECALL_KEYS,
    audit_stage0_biencoder_quality,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR Stage0 bi-encoder coarse recall quality.")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--stage0_eval_metrics_path")
    parser.add_argument("--baseline_metrics_path", default="outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json")
    parser.add_argument("--output_path")
    parser.add_argument("--min_steps", type=int, default=5000)
    parser.add_argument("--first_window", type=int, default=200)
    parser.add_argument("--last_window", type=int, default=200)
    parser.add_argument("--min_loss_drop", type=float, default=0.05)
    parser.add_argument("--baseline_tolerance", type=float, default=0.0)
    parser.add_argument("--expected_sources", nargs="*", default=DEFAULT_EXPECTED_SOURCES)
    parser.add_argument("--required_recall_keys", nargs="*", default=DEFAULT_REQUIRED_RECALL_KEYS)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stage0_biencoder_quality(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        stage0_eval_metrics_path=args.stage0_eval_metrics_path,
        baseline_metrics_path=args.baseline_metrics_path,
        output_path=args.output_path,
        min_steps=args.min_steps,
        first_window=args.first_window,
        last_window=args.last_window,
        min_loss_drop=args.min_loss_drop,
        baseline_tolerance=args.baseline_tolerance,
        expected_sources=args.expected_sources,
        required_recall_keys=args.required_recall_keys,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
