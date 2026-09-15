#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_clstr_queries import (
    build_stabletoolbench_queries_from_clstr_run,
    export_stabletoolbench_retrieval_queries,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build CLSTR-routed StableToolBench query inputs, or export StableToolBench queries for CLSTR retrieval."
    )
    parser.add_argument("--original_query_file")
    parser.add_argument("--query_file")
    parser.add_argument("--skills_path")
    parser.add_argument("--run_path")
    parser.add_argument("--output_query_file")
    parser.add_argument("--output_path")
    parser.add_argument("--report_path")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--export_retrieval_queries", action="store_true")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    if args.export_retrieval_queries:
        query_file = args.query_file or args.original_query_file
        output_path = args.output_path or args.output_query_file
        if not query_file or not output_path:
            parser.error("--export_retrieval_queries requires --query_file/--original_query_file and --output_path")
        report = export_stabletoolbench_retrieval_queries(
            query_file=query_file,
            output_path=output_path,
            report_path=args.report_path,
        )
    else:
        missing = [
            name
            for name, value in {
                "--original_query_file": args.original_query_file,
                "--skills_path": args.skills_path,
                "--run_path": args.run_path,
                "--output_query_file": args.output_query_file,
            }.items()
            if not value
        ]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
        report = build_stabletoolbench_queries_from_clstr_run(
            original_query_file=args.original_query_file,
            skills_path=args.skills_path,
            run_path=args.run_path,
            output_query_file=args.output_query_file,
            report_path=args.report_path,
            top_k=args.top_k,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
