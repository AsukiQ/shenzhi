#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.tau2_route_eval import run_tau2_full_clstr_route_eval
from clstr.full_base_train import ROUTE_SCORERS, UNIFIED_MEMORY_ROUTE_SCORER
from clstr.memory_utility_gate import RELIABILITY_MODES


def _domains(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the full CLSTR route on tau2/tau-bench oracle next-tool routing rows."
    )
    parser.add_argument(
        "--data_root",
        default=".tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data",
        help="Local tau2-bench-data root containing domains/<domain>/tasks.json.",
    )
    parser.add_argument("--prebuilt_source_rows_path", help="Optional frozen tau2 source rows JSONL.")
    parser.add_argument("--prebuilt_skills_path", help="Optional frozen tau2 skill pool JSONL.")
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument(
        "--stage4_checkpoint_path",
        help="Optional Stage4 checkpoint to load after Stage2 before evaluating the full route.",
    )
    parser.add_argument(
        "--training_skills_path",
        help="Optional exact Stage0 training skill pool to restore before appending tau2 skills.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domains", default="airline,retail,telecom", help="Comma-separated tau2 domains.")
    parser.add_argument(
        "--task_split",
        default="base",
        choices=["base", "train", "test", "full"],
    )
    parser.add_argument("--max_tasks_per_domain", type=int)
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
    parser.add_argument("--write_mt_ablation_report", action="store_true")
    parser.add_argument("--mt_ablation_auto_replay_prefix_max_steps", type=int, default=3)
    parser.add_argument("--include_mt_effect_diagnostics", action="store_true")
    args = parser.parse_args()

    report = run_tau2_full_clstr_route_eval(
        data_root=args.data_root,
        prebuilt_source_rows_path=args.prebuilt_source_rows_path,
        prebuilt_skills_path=args.prebuilt_skills_path,
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        training_skills_path=args.training_skills_path,
        output_dir=args.output_dir,
        domains=_domains(args.domains),
        task_split=args.task_split,
        max_tasks_per_domain=args.max_tasks_per_domain,
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
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
