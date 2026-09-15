#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_skillrouter_official import export_toolbench_g3_trajectory_skillrouter_eval_core


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ToolBench-G3 trajectory next-skill rows to the official SkillRouter eval_core layout."
    )
    parser.add_argument(
        "--trajectories_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl",
    )
    parser.add_argument(
        "--skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_train_rows", type=int)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--verify_executed_result_provenance", action="store_true")
    parser.add_argument("--verification_skills_path")
    args = parser.parse_args()
    report = export_toolbench_g3_trajectory_skillrouter_eval_core(
        trajectories_path=args.trajectories_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        max_skills=args.max_skills,
        verify_executed_result_provenance=args.verify_executed_result_provenance,
        verification_skills_path=args.verification_skills_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
