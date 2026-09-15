#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_replay import extract_alfworld_policy_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ALFWorld train replay for CLSTR L_policy training.")
    parser.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/alfworld_data")
    parser.add_argument("--data_output_dir", default="data/alfworld_policy_replay")
    parser.add_argument("--report_output_dir", default="outputs/alfworld_policy_replay")
    parser.add_argument("--max_games", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--expert_type", default="handcoded", choices=["handcoded", "planner"])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = extract_alfworld_policy_replay(
        official_repo=Path(args.official_repo),
        data_dir=Path(args.data_dir),
        data_output_dir=Path(args.data_output_dir),
        report_output_dir=Path(args.report_output_dir),
        max_games=args.max_games,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        expert_type=args.expert_type,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
