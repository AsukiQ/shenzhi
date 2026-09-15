#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_answer_conversion import convert_stabletoolbench_answers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit or run StableToolBench raw-answer to converted-answer preprocessing."
    )
    parser.add_argument("--stabletoolbench_root", required=True)
    parser.add_argument("--raw_answer_path", required=True)
    parser.add_argument("--converted_answer_path", required=True)
    parser.add_argument("--candidate_model", required=True)
    parser.add_argument("--test_set", default="G3_instruction")
    parser.add_argument("--method", default="CLSTR@1")
    parser.add_argument("--command_root")
    parser.add_argument("--output_path")
    parser.add_argument("--run_convert", action="store_true")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = convert_stabletoolbench_answers(
        stabletoolbench_root=args.stabletoolbench_root,
        raw_answer_path=args.raw_answer_path,
        converted_answer_path=args.converted_answer_path,
        candidate_model=args.candidate_model,
        test_set=args.test_set,
        method=args.method,
        command_root=args.command_root,
        output_path=args.output_path,
        run_convert=bool(args.run_convert),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
