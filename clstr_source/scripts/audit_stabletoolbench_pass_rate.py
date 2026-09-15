#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_pass_rate import audit_stabletoolbench_pass_rate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether official StableToolBench SoPR can run for a converted prediction set."
    )
    parser.add_argument("--stabletoolbench_root", required=True)
    parser.add_argument("--converted_answer_path", required=True)
    parser.add_argument("--api_pool_file")
    parser.add_argument("--candidate_model", required=True)
    parser.add_argument("--test_set", default="G3_instruction")
    parser.add_argument("--save_path", default="outputs/toolbench_g3/stabletoolbench_pass_rate")
    parser.add_argument("--test_ids_dir")
    parser.add_argument("--evaluator", default="tooleval_gpt-3.5-turbo_default")
    parser.add_argument("--max_eval_threads", type=int, default=1)
    parser.add_argument("--evaluate_times", type=int, default=3)
    parser.add_argument("--command_root")
    parser.add_argument("--output_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=args.stabletoolbench_root,
        converted_answer_path=args.converted_answer_path,
        api_pool_file=args.api_pool_file,
        candidate_model=args.candidate_model,
        test_set=args.test_set,
        save_path=args.save_path,
        test_ids_dir=args.test_ids_dir,
        evaluator=args.evaluator,
        max_eval_threads=args.max_eval_threads,
        evaluate_times=args.evaluate_times,
        command_root=args.command_root,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
