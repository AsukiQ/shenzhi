#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_corrective_preference import write_api_correction_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build diagnostic counterfactual AppWorld correction samples by mapping "
            "official solution API refs to candidate SkillX skills."
        )
    )
    parser.add_argument("--rollouts_path", required=True)
    parser.add_argument("--skill_pool_path", required=True)
    parser.add_argument("--appworld_tasks_root", required=True)
    parser.add_argument("--output_jsonl_path", required=True)
    parser.add_argument("--report_json_path", required=True)
    parser.add_argument(
        "--stage0_retrieval_jsonl_path",
        help="Optional output path for Stage0 retrieval correction rows when GT API skills are missing from candidates.",
    )
    parser.add_argument("--max_targets", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    dataset = write_api_correction_dataset(
        rollouts_path=args.rollouts_path,
        skill_pool_path=args.skill_pool_path,
        appworld_tasks_root=args.appworld_tasks_root,
        output_jsonl_path=args.output_jsonl_path,
        report_json_path=args.report_json_path,
        max_targets=int(args.max_targets),
        stage0_retrieval_jsonl_path=args.stage0_retrieval_jsonl_path,
    )
    print(json.dumps(dataset.report, ensure_ascii=False, sort_keys=True))
    return dataset.report


if __name__ == "__main__":
    main()
