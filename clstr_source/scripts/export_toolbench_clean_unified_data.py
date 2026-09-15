#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_clean_training_export import export_toolbench_clean_unified_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export CLSTR unified data with ToolBench-G3 eval trajectories removed from training sources."
    )
    parser.add_argument(
        "--data_root",
        default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2",
    )
    parser.add_argument(
        "--eval_trajectories_path",
        default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = export_toolbench_clean_unified_data(
        data_root=args.data_root,
        eval_trajectories_path=args.eval_trajectories_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

