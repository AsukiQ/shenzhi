#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_dynamic_controller_candidate_audit import audit_appworld_dynamic_controller_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether CLSTRMultiStepController keeps AppWorld dynamic SkillX "
            "positives in its candidate set after applying available-skill masks."
        )
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--base_skill_pool_path", required=True)
    parser.add_argument("--dynamic_skill_pool_path", required=True)
    parser.add_argument("--retrieval_path", required=True)
    parser.add_argument("--tasks_path")
    parser.add_argument("--appworld_root")
    parser.add_argument("--output_path", default="outputs/appworld_dynamic_controller_candidate_audit/report.json")
    parser.add_argument("--model_cache_dir")
    parser.add_argument("--candidate_top_k", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_pairs", type=int)
    parser.add_argument("--sampling_strategy", choices=["head", "stride"], default="head")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--query_text_format",
        default="auto",
        choices=["auto", "raw", "skillrouter", "skillrouter_state", "appworld_skillrouter_query"],
    )
    parser.add_argument(
        "--state_text_source",
        default="retrieval_query",
        choices=["retrieval_query", "official_state", "official_schema_state"],
    )
    parser.add_argument("--ranking_mode", default="skill_table")
    parser.add_argument("--candidate_source", default="routing")
    parser.add_argument("--appworld_executor_compatible_only", action="store_true")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_appworld_dynamic_controller_candidates(
        checkpoint_path=args.checkpoint_path,
        base_skill_pool_path=args.base_skill_pool_path,
        dynamic_skill_pool_path=args.dynamic_skill_pool_path,
        retrieval_path=args.retrieval_path,
        output_path=args.output_path,
        tasks_path=args.tasks_path,
        appworld_root=args.appworld_root,
        model_cache_dir=args.model_cache_dir,
        candidate_top_k=args.candidate_top_k,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        sampling_strategy=args.sampling_strategy,
        device=args.device,
        query_text_format=args.query_text_format,
        state_text_source=args.state_text_source,
        ranking_mode=args.ranking_mode,
        candidate_source=args.candidate_source,
        appworld_executor_compatible_only=bool(args.appworld_executor_compatible_only),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
