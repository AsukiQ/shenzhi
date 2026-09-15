#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_routing import build_appworld_routing_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AppWorld task-to-SkillX routing data for CLSTR.")
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="data/appworld_routing")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test_normal", "test_challenge"])
    parser.add_argument("--positives_per_task", type=int, default=5)
    parser.add_argument("--max_tasks_per_split", type=int)
    args = parser.parse_args()

    report = build_appworld_routing_corpus(
        appworld_root=args.appworld_root,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        splits=args.splits,
        positives_per_task=args.positives_per_task,
        max_tasks_per_split=args.max_tasks_per_split,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
