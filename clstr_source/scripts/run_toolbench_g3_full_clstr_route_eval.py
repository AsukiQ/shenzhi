#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_full_clstr_route_eval import run_toolbench_full_clstr_route_eval
from clstr.full_base_train import ROUTE_SCORERS, UNIFIED_MEMORY_ROUTE_SCORER
from clstr.memory_utility_gate import RELIABILITY_MODES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate full CLSTR route on the fixed ToolBench-G3 trajectory split."
    )
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument("--stage4_checkpoint_path")
    parser.add_argument(
        "--train_trajectories_path",
        default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/train_trajectories.jsonl",
    )
    parser.add_argument(
        "--eval_trajectories_path",
        default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    )
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--route_records_path")
    parser.add_argument("--route_record_manifest_path")
    parser.add_argument("--route_record_model_digest")
    parser.add_argument("--top_m", "--static_k", dest="top_m", type=int, default=350)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument("--candidate_count", "--final_k", dest="candidate_count", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--feedback_mode",
        default="eval_prefix",
        choices=["train_only", "eval_prefix", "train_plus_eval_prefix"],
    )
    parser.add_argument("--online_memory_mode", default="latest_exact", choices=["latest_exact", "exact_count", "state_conditioned", "inventory_remaining"])
    parser.add_argument("--online_memory_weight", type=float, default=1.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    parser.add_argument("--transition_residual_lambda", type=float, default=0.25)
    parser.add_argument("--route_scorer", default=UNIFIED_MEMORY_ROUTE_SCORER, choices=sorted(ROUTE_SCORERS))
    parser.add_argument(
        "--reliability_mode",
        default="dynamic",
        choices=sorted(RELIABILITY_MODES),
    )
    parser.add_argument("--fixed_alpha", type=float, default=1.0)
    parser.add_argument("--feature_update_count_cap", type=float, default=1.0)
    parser.add_argument("--feature_candidate_count_cap", type=float, default=1.0)
    parser.add_argument("--memory_utility_gate_checkpoint_path")
    parser.add_argument("--expected_memory_utility_gate_checkpoint_sha256")
    parser.add_argument("--expected_memory_utility_gate_audit_sha256")
    parser.add_argument("--transition_inventory_mask_mode", default="off")
    parser.add_argument("--transition_inventory_min_candidates", type=int, default=0)
    parser.add_argument(
        "--stage0_handoff_query_mode",
        default="skillrouter_state",
        choices=["raw_state", "skillrouter_state", "checkpoint_state_query"],
    )
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--max_train_rows", type=int)
    parser.add_argument("--max_eval_rows", type=int)
    args = parser.parse_args()

    report = run_toolbench_full_clstr_route_eval(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        train_trajectories_path=args.train_trajectories_path,
        eval_trajectories_path=args.eval_trajectories_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        top_m=args.top_m,
        dynamic_extra_k=args.dynamic_extra_k,
        candidate_count=args.candidate_count,
        batch_size=args.batch_size,
        feedback_mode=args.feedback_mode,
        online_memory_mode=args.online_memory_mode,
        online_memory_weight=args.online_memory_weight,
        online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
        online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
        transition_residual_lambda=args.transition_residual_lambda,
        route_scorer=args.route_scorer,
        reliability_mode=args.reliability_mode,
        fixed_alpha=args.fixed_alpha,
        feature_update_count_cap=args.feature_update_count_cap,
        feature_candidate_count_cap=args.feature_candidate_count_cap,
        memory_utility_gate_checkpoint_path=args.memory_utility_gate_checkpoint_path,
        expected_memory_utility_gate_checkpoint_sha256=args.expected_memory_utility_gate_checkpoint_sha256,
        expected_memory_utility_gate_audit_sha256=args.expected_memory_utility_gate_audit_sha256,
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_inventory_min_candidates=args.transition_inventory_min_candidates,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        route_records_path=args.route_records_path,
        route_record_manifest_path=args.route_record_manifest_path,
        route_record_model_digest=args.route_record_model_digest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
