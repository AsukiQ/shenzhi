#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.envs.appworld_env import DEFAULT_APPWORLD_CACHE, DEFAULT_APPWORLD_ROOT, run_appworld_adapter_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal CLSTR AppWorld adapter smoke.")
    parser.add_argument("--appworld_root", default=str(DEFAULT_APPWORLD_ROOT))
    parser.add_argument("--appworld_cache", default=str(DEFAULT_APPWORLD_CACHE))
    parser.add_argument("--output_dir", default="outputs/appworld_adapter_smoke")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task_id")
    parser.add_argument("--attempt_reset", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    report = run_appworld_adapter_smoke(
        appworld_root=args.appworld_root,
        appworld_cache=args.appworld_cache,
        output_dir=args.output_dir,
        split=args.split,
        task_id=args.task_id,
        attempt_reset=args.attempt_reset,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
