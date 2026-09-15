#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_g3_audit import audit_toolbench_g3_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit normalized ToolBench-G3 data and reject data_example/static smoke substitutes."
    )
    parser.add_argument("--data_dir", default="data/toolbench_g3")
    parser.add_argument("--source_root")
    parser.add_argument("--output_path")
    parser.add_argument("--expected_answer_files", type=int, default=None)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_toolbench_g3_data(
        data_dir=args.data_dir,
        source_root=args.source_root,
        output_path=args.output_path,
        expected_answer_files=args.expected_answer_files,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
