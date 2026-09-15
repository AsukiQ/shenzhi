#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import ROUTE_SCORERS, UNIFIED_MEMORY_ROUTE_SCORER
from clstr.tau2_route_eval import run_tau2_full_clstr_route_eval


def _domains(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the full CLSTR route on tau3/tau-bench oracle next-tool routing rows."
    )
    parser.add_argument(
        "--data_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench/data/tau2",
        help="Local tau3/tau2-bench data root containing domains/<domain>/tasks.json.",
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
        default="outputs/clstr_unified_stage4_function_aug_v2_true_adapt_s0_top500_stage0prior50_rankprior_l050_joint_act/checkpoints/clstr_stage4_act-step2000.pt",
        help="Optional Stage4 checkpoint to load after Stage2 before evaluating the full route.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domains", default="airline,retail,telecom,banking_knowledge")
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
    parser.add_argument("--transition_residual_lambda", type=float, default=0.5)
    parser.add_argument("--route_scorer", default=UNIFIED_MEMORY_ROUTE_SCORER, choices=sorted(ROUTE_SCORERS))
    args = parser.parse_args()

    report = run_tau2_full_clstr_route_eval(
        benchmark_name="tau3",
        data_root=args.data_root,
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        output_dir=args.output_dir,
        domains=_domains(args.domains),
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
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
