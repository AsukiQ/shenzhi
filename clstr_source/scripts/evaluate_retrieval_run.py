#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.retrieval_metrics import evaluate_retrieval_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a generic CLSTR retrieval run with NDCG/Recall/MAP metrics."
    )
    parser.add_argument("--qrels_path", required=True)
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_format", choices=["jsonl", "trec"], default="jsonl")
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--benchmark", default="generic_retrieval")
    parser.add_argument("--method", default="unknown")
    args = parser.parse_args()

    report = evaluate_retrieval_run(
        qrels_path=args.qrels_path,
        run_path=args.run_path,
        output_dir=args.output_dir,
        run_format=args.run_format,
        k_values=args.k_values,
        benchmark=args.benchmark,
        method=args.method,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
