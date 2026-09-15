#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_multistep import build_train_multistep_trajectories_from_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export train-only AppWorld multi-step trajectories from runs.jsonl.")
    parser.add_argument("--runs_path", required=True)
    parser.add_argument("--output_path", default="data/appworld_multistep/train_trajectories.jsonl")
    parser.add_argument("--manifest_path", default="data/appworld_multistep/manifest.json")
    parser.add_argument("--allowed_split", default="train")
    parser.add_argument("--require_success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--supervision_type", default="train_success_rollout")
    args = parser.parse_args()
    manifest = build_train_multistep_trajectories_from_runs(
        runs_path=args.runs_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        allowed_split=args.allowed_split,
        require_success=bool(args.require_success),
        supervision_type=args.supervision_type,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
