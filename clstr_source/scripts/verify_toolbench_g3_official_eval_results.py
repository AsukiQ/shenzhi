#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_skillrouter_official import (
    export_verified_toolbench_existing_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ToolBench results while preserving an existing official split",
    )
    parser.add_argument("--source_eval_trajectories_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--verification_skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    report = export_verified_toolbench_existing_eval(
        source_eval_trajectories_path=args.source_eval_trajectories_path,
        skills_path=args.skills_path,
        verification_skills_path=args.verification_skills_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
