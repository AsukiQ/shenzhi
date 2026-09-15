#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_stage0_correction_audit import build_train_stage0_correction_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-side AppWorld Stage0 API correction/audit rows. Without a candidate file, "
            "this only reports pool-positive availability; with candidates, it writes correction rows "
            "only for positives missing from candidate_skill_ids."
        )
    )
    parser.add_argument("--train_tasks_path", default="data/appworld_routing/train_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_train_stage0_correction_audit")
    parser.add_argument("--candidate_rows_path", default="")
    parser.add_argument("--max_targets", type=int, default=5)
    parser.add_argument("--max_tasks", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = build_train_stage0_correction_audit(
        train_tasks_path=args.train_tasks_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        candidate_rows_path=args.candidate_rows_path or None,
        max_targets=int(args.max_targets),
        max_tasks=args.max_tasks,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
