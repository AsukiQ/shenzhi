#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.retrieval_qrels import export_retrieval_qrels


def _optional_set(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize CLSTR unified retrieval rows into generic qrels.jsonl."
    )
    parser.add_argument("--retrieval_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--report_path")
    parser.add_argument("--include_negatives", action="store_true")
    parser.add_argument("--allowed_sources", nargs="+")
    parser.add_argument("--allowed_splits", nargs="+")
    args = parser.parse_args()

    report = export_retrieval_qrels(
        retrieval_path=args.retrieval_path,
        output_path=args.output_path,
        include_negatives=args.include_negatives,
        allowed_sources=_optional_set(args.allowed_sources),
        allowed_splits=_optional_set(args.allowed_splits),
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
