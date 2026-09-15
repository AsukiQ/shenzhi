#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.historical_trajectories import build_verified_pairs_from_successful_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified pairs from historical SkillsBench successful trajectories.")
    parser.add_argument("--successful_trajectories_path", default="/root/autodl-tmp/skillsbench/successful_trajectories.jsonl")
    parser.add_argument("--skill_pool_path", default="/root/autodl-tmp/skillsbench/skill_pool.jsonl")
    parser.add_argument("--output_jsonl", default="data/verified_pairs/skillsbench_train_verified_pairs.jsonl")
    parser.add_argument("--manifest_path", default="data/verified_pairs/manifest.json")
    args = parser.parse_args()

    report = build_verified_pairs_from_successful_trajectories(
        successful_trajectories_path=Path(args.successful_trajectories_path),
        skill_pool_path=Path(args.skill_pool_path),
        output_jsonl=Path(args.output_jsonl),
        manifest_path=Path(args.manifest_path),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
