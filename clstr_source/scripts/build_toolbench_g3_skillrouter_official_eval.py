#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_skillrouter_official import export_toolbench_g3_skillrouter_eval_core


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ToolBench-G3 to official SkillRouter eval_core layout.")
    parser.add_argument("--queries_path", default="data/toolbench_g3_routing_eval/queries.jsonl")
    parser.add_argument("--qrels_path", default="data/toolbench_g3_routing_eval/qrels.jsonl")
    parser.add_argument("--skills_path", default="data/toolbench_g3/skills.jsonl")
    parser.add_argument("--output_dir", default="data/toolbench_g3_skillrouter_official_eval")
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_train_queries", type=int)
    parser.add_argument("--max_eval_queries", type=int)
    parser.add_argument("--max_skills", type=int)
    args = parser.parse_args()
    report = export_toolbench_g3_skillrouter_eval_core(
        queries_path=args.queries_path,
        qrels_path=args.qrels_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
        max_skills=args.max_skills,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

