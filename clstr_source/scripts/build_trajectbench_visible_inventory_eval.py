#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.trajectbench_visible_inventory_eval import build_trajectbench_visible_inventory_eval


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a TrajectBench-only CLSTR eval JSONL with public visible inventory and no GT inventory."
    )
    parser.add_argument(
        "--trajectories_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl",
    )
    parser.add_argument(
        "--skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl",
    )
    parser.add_argument(
        "--output_path",
        default=".tmp/stage4_sanitized_trajectbench/trajectories_visible_global_full.jsonl",
    )
    parser.add_argument(
        "--report_path",
        default=".tmp/stage4_sanitized_trajectbench/trajectories_visible_global_full_report.json",
    )
    parser.add_argument(
        "--visible_inventory_mode",
        choices=["global", "domain_local"],
        default="global",
    )
    args = parser.parse_args()

    report = build_trajectbench_visible_inventory_eval(
        trajectories_path=args.trajectories_path,
        skills_path=args.skills_path,
        output_path=args.output_path,
        report_path=args.report_path,
        visible_inventory_mode=args.visible_inventory_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
