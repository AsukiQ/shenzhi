#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_eval import run_appworld_skillrouter_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkillRouter-style static routing baseline on AppWorld tasks.")
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--qrels_path", default="data/appworld_routing/dev_qrels.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_skillrouter_baseline")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    report = run_appworld_skillrouter_baseline(
        tasks_path=args.tasks_path,
        qrels_path=args.qrels_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
