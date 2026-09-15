#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_g3_routing_eval import prepare_toolbench_g3_routing_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ToolBench-G3 static routing eval queries/qrels from normalized retrieval rows."
    )
    parser.add_argument("--retrieval_path", default="data/toolbench_g3/retrieval.jsonl")
    parser.add_argument("--output_dir", default="data/toolbench_g3_routing_eval")
    parser.add_argument("--report_path")
    args = parser.parse_args()

    report = prepare_toolbench_g3_routing_eval(
        retrieval_path=args.retrieval_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
