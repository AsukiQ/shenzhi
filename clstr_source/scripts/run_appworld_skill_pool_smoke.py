#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bridges.skillx.appworld_adapter import summarize_appworld_skill_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the normalized SkillX AppWorld skill pool.")
    parser.add_argument("--pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_skill_pool_smoke")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / "report.json"
    report = summarize_appworld_skill_pool(args.pool_path, output_path=output_path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
