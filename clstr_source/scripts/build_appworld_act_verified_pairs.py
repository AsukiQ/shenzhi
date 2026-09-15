#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_act_verified_pairs import build_appworld_act_verified_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only AppWorld CLSTR-act verified pairs.")
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--tasks_path", default="data/appworld_routing/train_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_jsonl", default="data/appworld_act/verified_pairs_train.jsonl")
    parser.add_argument("--manifest_path", default="data/appworld_act/manifest.json")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--compress_consecutive", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    report = build_appworld_act_verified_pairs(
        appworld_root=args.appworld_root,
        tasks_path=args.tasks_path,
        skill_pool_path=args.skill_pool_path,
        output_jsonl=args.output_jsonl,
        manifest_path=args.manifest_path,
        top_k=args.top_k,
        max_tasks=args.max_tasks,
        compress_consecutive=bool(args.compress_consecutive),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
