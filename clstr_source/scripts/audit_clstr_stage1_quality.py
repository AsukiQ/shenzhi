#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage1_quality_gate import DEFAULT_EXPECTED_SOURCES, audit_stage1_retrieval_warmup_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR Stage 1 retrieval warmup quality before Stage 2.")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--metrics_path")
    parser.add_argument("--output_path")
    parser.add_argument("--min_steps", type=int, default=5000)
    parser.add_argument("--first_window", type=int, default=200)
    parser.add_argument("--last_window", type=int, default=200)
    parser.add_argument("--min_loss_drop", type=float, default=0.05)
    parser.add_argument("--min_last_recall", type=float, default=0.02)
    parser.add_argument("--expected_sources", nargs="*", default=DEFAULT_EXPECTED_SOURCES)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stage1_retrieval_warmup_quality(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        output_path=args.output_path,
        min_steps=args.min_steps,
        first_window=args.first_window,
        last_window=args.last_window,
        min_loss_drop=args.min_loss_drop,
        min_last_recall=args.min_last_recall,
        expected_sources=args.expected_sources,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
