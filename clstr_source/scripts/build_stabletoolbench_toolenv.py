#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_toolenv import build_stabletoolbench_toolenv_from_queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a StableToolBench G3 toolenv/tools subset from official solvable queries.")
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = build_stabletoolbench_toolenv_from_queries(
        query_file=args.query_file,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
