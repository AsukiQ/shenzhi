#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_multistep import build_appworld_eval_step_label_tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval-only AppWorld tasks augmented with oracle API-trace step labels."
    )
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl")
    parser.add_argument("--output_path", default="data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl")
    parser.add_argument("--manifest_path", default="data/appworld_multistep/dev_step_labels_dynamic_manifest.json")
    parser.add_argument("--allowed_split", default="dev")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--compress_consecutive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--appworld_executor_compatible_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    manifest = build_appworld_eval_step_label_tasks(
        appworld_root=args.appworld_root,
        tasks_path=args.tasks_path,
        skill_pool_path=args.skill_pool_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        allowed_split=args.allowed_split,
        max_tasks=args.max_tasks,
        compress_consecutive=bool(args.compress_consecutive),
        appworld_executor_compatible_only=bool(args.appworld_executor_compatible_only),
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
