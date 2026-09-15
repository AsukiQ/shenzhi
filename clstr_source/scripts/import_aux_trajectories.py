#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.aux_trajectories import import_aux_trajectory_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Import auxiliary ALFWorld/ScienceWorld/AutoDreamer trajectories into CLSTR JSONL.")
    parser.add_argument("--sft_alfworld_root", default="/root/autodl-tmp/aux_trajectories/sft_alfworld_trajectory_dataset_v5_cleaned")
    parser.add_argument("--auto_dreamer_root", default="/root/autodl-tmp/aux_trajectories/auto-dreamer")
    parser.add_argument("--eto_sft_root", default="/root/autodl-tmp/aux_trajectories/eto-sft-trajectory")
    parser.add_argument("--output_dir", default="data/aux_trajectories")
    parser.add_argument("--report_dir", default="outputs/aux_trajectory_import")
    parser.add_argument("--max_records_per_dataset", type=int)
    args = parser.parse_args()

    manifest = import_aux_trajectory_datasets(
        dataset_roots={
            "sft_alfworld": Path(args.sft_alfworld_root),
            "auto_dreamer": Path(args.auto_dreamer_root),
            "eto_sft": Path(args.eto_sft_root),
        },
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
        max_records_per_dataset=args.max_records_per_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
