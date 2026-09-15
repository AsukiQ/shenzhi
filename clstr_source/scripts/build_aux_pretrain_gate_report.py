#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.aux_pretrain import build_aux_pretrain_gate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auxiliary pretrain gate report.")
    parser.add_argument("--train_report_path", required=True)
    parser.add_argument("--eval_report_path", required=True)
    parser.add_argument("--retention_metrics_path", required=True)
    parser.add_argument("--output_path", default="outputs/aux_trajectory_pretrain_gate/gate_report.json")
    args = parser.parse_args()

    report = build_aux_pretrain_gate_report(
        train_report_path=Path(args.train_report_path),
        eval_report_path=Path(args.eval_report_path),
        retention_metrics_path=Path(args.retention_metrics_path),
        output_path=Path(args.output_path),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
