#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_routing_diagnostics import (
    build_duplicate_aware_eval_report,
    build_executor_failure_topk_report,
    build_train_dev_overlap_report,
)


def _parse_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected METHOD=PATH, got: {value}")
        method, path = value.split("=", 1)
        if not method or not path:
            raise SystemExit(f"expected METHOD=PATH, got: {value}")
        mapping[method] = path
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AppWorld routing diagnostic reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dup = subparsers.add_parser("duplicate-aware")
    dup.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    dup.add_argument("--qrels_path", default="data/appworld_routing/dev_qrels.jsonl")
    dup.add_argument("--predictions_path", required=True)
    dup.add_argument("--output_dir", required=True)
    dup.add_argument("--method", required=True)

    overlap = subparsers.add_parser("train-dev-overlap")
    overlap.add_argument("--train_tasks_path", default="data/appworld_routing/train_tasks.jsonl")
    overlap.add_argument("--dev_tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    overlap.add_argument("--output_dir", required=True)

    failure = subparsers.add_parser("executor-failure-topk")
    failure.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    failure.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    failure.add_argument("--qrels_path", default="data/appworld_routing/dev_qrels.jsonl")
    failure.add_argument("--run", action="append", default=[], help="METHOD=PATH to executor runs.jsonl")
    failure.add_argument("--predictions", action="append", default=[], help="METHOD=PATH to routing predictions.jsonl")
    failure.add_argument("--output_dir", required=True)
    failure.add_argument("--focus_method", default="clstr_base")
    failure.add_argument("--reference_method", action="append", default=["skillrouter_base", "qwen_only"])
    failure.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "duplicate-aware":
        report = build_duplicate_aware_eval_report(
            skill_pool_path=args.skill_pool_path,
            qrels_path=args.qrels_path,
            predictions_path=args.predictions_path,
            output_dir=args.output_dir,
            method=args.method,
        )
    elif args.command == "train-dev-overlap":
        report = build_train_dev_overlap_report(
            train_tasks_path=args.train_tasks_path,
            dev_tasks_path=args.dev_tasks_path,
            output_dir=args.output_dir,
        )
    else:
        report = build_executor_failure_topk_report(
            skill_pool_path=args.skill_pool_path,
            tasks_path=args.tasks_path,
            qrels_path=args.qrels_path,
            run_paths=_parse_mapping(args.run),
            prediction_paths=_parse_mapping(args.predictions),
            output_dir=args.output_dir,
            focus_method=args.focus_method,
            reference_methods=args.reference_method,
            top_k=args.top_k,
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
