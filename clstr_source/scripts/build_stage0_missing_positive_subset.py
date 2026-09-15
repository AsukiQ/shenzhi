#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.stage0_retrieval_coverage_audit import build_stage0_missing_positive_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a balanced source-filtered subset from Stage0 missing-positive correction rows."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_allowlist", required=True, help="Comma-separated base source names to keep.")
    parser.add_argument("--per_source_cap", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = build_stage0_missing_positive_subset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        source_allowlist=[item.strip() for item in args.source_allowlist.split(",") if item.strip()],
        per_source_cap=args.per_source_cap,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
