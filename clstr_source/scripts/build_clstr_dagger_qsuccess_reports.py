#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.dagger_qsuccess_report import build_dagger_qsuccess_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR-Qwen DAgger/Q_success reports.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_table_path", default="outputs/clstr_dagger_qsuccess_comparison_table.md")
    parser.add_argument("--output_summary_path", default="outputs/clstr_dagger_qsuccess_comparison_summary.json")
    parser.add_argument("--output_paper_md", default="outputs/clstr_dagger_qsuccess_paper_report.md")
    parser.add_argument("--output_paper_json", default="outputs/clstr_dagger_qsuccess_paper_report.json")
    args = parser.parse_args()

    report = build_dagger_qsuccess_reports(
        output_root=Path(args.output_root),
        data_root=Path(args.data_root),
        output_table_path=Path(args.output_table_path),
        output_summary_path=Path(args.output_summary_path),
        output_paper_md=Path(args.output_paper_md),
        output_paper_json=Path(args.output_paper_json),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
