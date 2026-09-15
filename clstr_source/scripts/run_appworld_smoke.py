#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_smoke import run_appworld_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal AppWorld availability smoke check.")
    parser.add_argument("--output_dir", default="outputs/appworld_smoke")
    parser.add_argument("--attempt_reset", action="store_true")
    parser.add_argument("--appworld_root")
    parser.add_argument("--appworld_cache")
    args = parser.parse_args()

    report = run_appworld_smoke(
        output_dir=args.output_dir,
        attempt_reset=args.attempt_reset,
        appworld_root=args.appworld_root,
        appworld_cache=args.appworld_cache,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
