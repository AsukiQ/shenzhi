#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_label_audit import audit_appworld_task_positive_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AppWorld task-level positive labels against SkillX semantic evidence.")
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl")
    parser.add_argument("--runs_path", default=None)
    parser.add_argument("--output_path", default="outputs/appworld_label_quality/dev_dynamic/report.json")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--score_margin", type=int, default=1)
    args = parser.parse_args()
    report = audit_appworld_task_positive_labels(
        tasks_path=args.tasks_path,
        skill_pool_path=args.skill_pool_path,
        runs_path=args.runs_path,
        output_path=args.output_path,
        max_tasks=args.max_tasks,
        top_k=args.top_k,
        score_margin=args.score_margin,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
