#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolsandbox_route_eval import run_toolsandbox_full_clstr_route_eval
from clstr.full_base_train import ROUTE_SCORERS, UNIFIED_MEMORY_ROUTE_SCORER
from clstr.memory_utility_gate import RELIABILITY_MODES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the full CLSTR route on ToolSandbox source-derived required-tool routing rows."
    )
    parser.add_argument(
        "--scenarios_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios",
    )
    parser.add_argument(
        "--tools_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools",
    )
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/checkpoints/clstr_unified_retrieval_v2-step300.pt",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        default="outputs/clstr_unified_stage2_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_trainl025_calib_full/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument(
        "--stage4_checkpoint_path",
        default=None,
        help="Optional Stage4 checkpoint; omit for Stage2-only causal-memory evaluation.",
    )
    parser.add_argument(
        "--training_skills_path",
        help="Optional exact Stage0 training skill pool to restore before appending ToolSandbox skills.",
    )
    parser.add_argument(
        "--model_skill_pool_mode",
        default="checkpoint_faithful",
        choices=["checkpoint_faithful", "local_table_rebuild"],
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_scenarios", type=int)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--candidate_count", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_batch_size", type=int, default=16)
    parser.add_argument(
        "--online_memory_mode",
        default="latest_exact",
        choices=["latest_exact", "exact_count", "state_conditioned", "inventory_remaining"],
    )
    parser.add_argument("--online_memory_weight", type=float, default=1.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    parser.add_argument("--transition_residual_lambda", type=float, default=0.5)
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
    parser.add_argument("--write_mt_ablation_report", action="store_true")
    parser.add_argument("--mt_ablation_auto_replay_prefix_max_steps", type=int, default=3)
    parser.add_argument("--include_mt_effect_diagnostics", action="store_true")
    parser.add_argument(
        "--toolsandbox_replay_mode",
        default="action_only",
        choices=["legacy_auto", "action_only", "corrected_causal"],
    )
    args = parser.parse_args()

    report = run_toolsandbox_full_clstr_route_eval(
        scenarios_root=args.scenarios_root,
        tools_root=args.tools_root,
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        training_skills_path=args.training_skills_path,
        model_skill_pool_mode=args.model_skill_pool_mode,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        max_eval_rows=args.max_eval_rows,
        candidate_count=args.candidate_count,
        batch_size=args.batch_size,
        stage0_candidate_batch_size=args.stage0_candidate_batch_size,
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
        write_mt_ablation_report=args.write_mt_ablation_report,
        mt_ablation_auto_replay_prefix_max_steps=args.mt_ablation_auto_replay_prefix_max_steps,
        include_mt_effect_diagnostics=args.include_mt_effect_diagnostics,
        toolsandbox_replay_mode=args.toolsandbox_replay_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
