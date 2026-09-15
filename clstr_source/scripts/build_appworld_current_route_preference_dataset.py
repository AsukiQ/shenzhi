#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_current_route_preference import (
    build_current_route_disagreement_preference_dataset,
    write_current_route_preference_dataset,
)
from clstr.appworld_routing import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean Stage4 preference/imitation samples from current-route AppWorld rollout traces."
    )
    parser.add_argument("--rollouts_path", required=True)
    parser.add_argument("--output_jsonl_path", required=True)
    parser.add_argument("--report_json_path", required=True)
    parser.add_argument("--current_runs_path", default=None)
    parser.add_argument("--qwen_runs_path", default=None)
    parser.add_argument("--reference_runs_path", default=None)
    parser.add_argument("--negative_weight", type=float, default=0.2)
    parser.add_argument(
        "--adapter_mode",
        choices=["conservative", "disagreement"],
        default="conservative",
        help=(
            "conservative uses only rollout policy_signal labels; disagreement additionally "
            "uses baseline route splits and is intended for explicit ablations."
        ),
    )
    parser.add_argument(
        "--allow_ambiguous_disagreement",
        action="store_true",
        help=(
            "Allow disagreement suppress samples from ambiguous current failures such as "
            "wrong_completion. Leave off for the main Stage4 route."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    rollouts = read_jsonl(args.rollouts_path)
    if str(args.adapter_mode) == "disagreement":
        dataset = build_current_route_disagreement_preference_dataset(
            rollouts,
            current_runs=read_jsonl(args.current_runs_path) if args.current_runs_path else None,
            qwen_runs=read_jsonl(args.qwen_runs_path) if args.qwen_runs_path else None,
            reference_runs=read_jsonl(args.reference_runs_path) if args.reference_runs_path else None,
            negative_weight=float(args.negative_weight),
            allow_ambiguous_disagreement=bool(args.allow_ambiguous_disagreement),
        )
        output_path = Path(args.output_jsonl_path)
        report_path = Path(args.report_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            for sample in dataset.samples:
                fp.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        report_path.write_text(
            json.dumps(dataset.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        dataset = write_current_route_preference_dataset(
            rollouts,
            output_jsonl_path=args.output_jsonl_path,
            report_json_path=args.report_json_path,
        )
    print(json.dumps(dataset.report, ensure_ascii=False, sort_keys=True))
    return dataset.report


if __name__ == "__main__":
    main()
