#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_clstr_eval import run_appworld_clstr_routing_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR checkpoint routing eval on AppWorld SkillX qrels.")
    parser.add_argument("--model_config", default="configs/model/appworld_mini.yaml")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--qrels_path", default="data/appworld_routing/dev_qrels.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_clstr_eval")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--ranking_mode", choices=["skill_table", "policy_head", "policy_blend"], default="skill_table")
    parser.add_argument("--candidate_top_k", type=int, default=None)
    parser.add_argument("--policy_blend_alpha", type=float, default=0.25)
    parser.add_argument("--allow_legacy_policy_skill_router", action="store_true")
    args = parser.parse_args()
    report = run_appworld_clstr_routing_eval(
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint_path,
        tasks_path=args.tasks_path,
        qrels_path=args.qrels_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        ranking_mode=args.ranking_mode,
        candidate_top_k=args.candidate_top_k,
        policy_blend_alpha=args.policy_blend_alpha,
        allow_legacy_policy_skill_router=bool(args.allow_legacy_policy_skill_router),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
