#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stabletoolbench_raw_generation import audit_stabletoolbench_raw_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit StableToolBench official raw-answer generation prerequisites.")
    parser.add_argument("--stabletoolbench_root", required=True)
    parser.add_argument("--tool_root_dir", required=True)
    parser.add_argument("--raw_answer_path", required=True)
    parser.add_argument("--candidate_model", required=True)
    parser.add_argument("--test_set", default="G3_instruction")
    parser.add_argument("--input_query_file")
    parser.add_argument("--method", default="CLSTR@1")
    parser.add_argument("--backbone_model", default="chatgpt_function")
    parser.add_argument("--openai_key", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--service_url", default=None)
    parser.add_argument("--toolbench_key", default=None)
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--chatgpt_model", default="gpt-4-turbo-2024-04-09")
    parser.add_argument("--max_observation_length", type=int, default=1024)
    parser.add_argument("--single_chain_max_step", type=int, default=50)
    parser.add_argument("--max_query_count", type=int, default=200)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--command_root")
    parser.add_argument("--output_path")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_stabletoolbench_raw_generation(
        stabletoolbench_root=args.stabletoolbench_root,
        tool_root_dir=args.tool_root_dir,
        raw_answer_path=args.raw_answer_path,
        candidate_model=args.candidate_model,
        test_set=args.test_set,
        input_query_file=args.input_query_file,
        method=args.method,
        backbone_model=args.backbone_model,
        openai_key=args.openai_key if args.openai_key is not None else os.environ.get("OPENAI_KEY"),
        model_path=args.model_path,
        service_url=args.service_url if args.service_url is not None else os.environ.get("SERVICE_URL"),
        toolbench_key=args.toolbench_key if args.toolbench_key is not None else os.environ.get("TOOLBENCH_KEY"),
        base_url=args.base_url,
        chatgpt_model=args.chatgpt_model,
        max_observation_length=args.max_observation_length,
        single_chain_max_step=args.single_chain_max_step,
        max_query_count=args.max_query_count,
        num_thread=args.num_thread,
        command_root=args.command_root,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
