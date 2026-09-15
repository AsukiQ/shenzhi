#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_executor import build_executor_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AppWorld executor task-success comparison.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/appworld_executor_comparison")
    args = parser.parse_args()
    payload = build_executor_comparison([Path(item) for item in args.reports], Path(args.output_dir))
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
