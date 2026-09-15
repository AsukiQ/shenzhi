#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.traject_sequence_proxy import evaluate_traject_sequence_proxy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TRAJECT-Bench selection-only trajectory sequence proxy metrics."
    )
    parser.add_argument("--queries_path", required=True)
    parser.add_argument("--qrels_path", required=True)
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_format", choices=["jsonl", "trec"], default="trec")
    parser.add_argument("--method", default="unknown")
    args = parser.parse_args()

    report = evaluate_traject_sequence_proxy(
        queries_path=args.queries_path,
        qrels_path=args.qrels_path,
        run_path=args.run_path,
        output_dir=args.output_dir,
        run_format=args.run_format,
        method=args.method,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
