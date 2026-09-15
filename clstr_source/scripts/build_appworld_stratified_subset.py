#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_eval_subset import build_stratified_task_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small required-app stratified AppWorld task subset.")
    parser.add_argument("--input_path", default="data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl")
    parser.add_argument("--output_path", default="data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl")
    parser.add_argument("--manifest_path", default="data/appworld_multistep/dev10_stratified_step_labels_dynamic_manifest.json")
    parser.add_argument("--max_tasks", type=int, default=10)
    args = parser.parse_args()
    manifest = build_stratified_task_subset(
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        max_tasks=args.max_tasks,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
