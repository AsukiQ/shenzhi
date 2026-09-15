#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.prior_damage_slices import (
    build_prior_damage_slice_report,
    load_prior_damage_inputs,
    write_prior_damage_slice_report,
)


def _default_run_id(run_dir: Path) -> str:
    return run_dir.name or "clstr_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CLSTR Stage0-prior damage and candidate coverage slices.")
    parser.add_argument("--run_dir", required=True, help="Run output directory containing stage artifacts.")
    parser.add_argument("--output_dir", required=True, help="Directory where slice_report.json will be written.")
    parser.add_argument("--run_id", default="", help="Optional stable run id for the report.")
    parser.add_argument("--tail_window", type=int, default=100, help="Metric rows to average for tail diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    inputs = load_prior_damage_inputs(run_dir)
    report = build_prior_damage_slice_report(
        row_records=inputs["row_records"],
        handoff_report=inputs["handoff_report"],
        metric_rows=inputs["metric_rows"],
        tail_window=args.tail_window,
        run_id=args.run_id or _default_run_id(run_dir),
    )
    report["input_paths"] = {
        "run_dir": inputs["run_dir"],
        "metrics_path": inputs["metrics_path"],
        "handoff_path": inputs["handoff_path"],
        "row_records_path": inputs["row_records_path"],
    }
    write_prior_damage_slice_report(output_dir / "slice_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
