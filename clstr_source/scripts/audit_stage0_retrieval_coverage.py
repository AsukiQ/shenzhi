#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.stage0_retrieval_coverage_audit import build_stage0_retrieval_coverage_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Stage0 candidate coverage for Stream-B retrieval rows against exported predictions. "
            "Writes source-sliced recall metrics and missing-positive correction candidates."
        )
    )
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--skill_pool_path",
        help=(
            "Optional JSONL skill pool used to separate positives absent from the "
            "search space from in-pool positives missing from top-k predictions."
        ),
    )
    parser.add_argument("--k_values", nargs="+", type=int, default=[20, 50, 100, 350])
    parser.add_argument("--negative_k", type=int, default=50)
    parser.add_argument("--max_missing_rows", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = build_stage0_retrieval_coverage_audit(
        retrieval_rows_path=args.retrieval_rows_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        skill_pool_path=args.skill_pool_path,
        k_values=args.k_values,
        negative_k=args.negative_k,
        max_missing_rows=args.max_missing_rows,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
