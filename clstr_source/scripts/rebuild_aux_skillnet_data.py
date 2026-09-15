#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillnet_aux_rebuild import rebuild_aux_with_skillnet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild CLSTR auxiliary trajectories with benchmark-specific SkillNet skill pools."
    )
    parser.add_argument("--source_data_dir", default="data/aux_trajectories")
    parser.add_argument("--skillnet_root", default="/tmp/SkillNet/experiments/src/skills")
    parser.add_argument("--output_dir", default="data/aux_skillnet_rebuilt")
    parser.add_argument("--report_dir", default="outputs/aux_skillnet_rebuild")
    parser.add_argument(
        "--training_split",
        action="append",
        default=["train"],
        help="Split usable for training. Repeatable; defaults to train only.",
    )
    args = parser.parse_args()

    manifest = rebuild_aux_with_skillnet(
        source_data_dir=Path(args.source_data_dir),
        skillnet_root=Path(args.skillnet_root),
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
        training_splits=args.training_split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
