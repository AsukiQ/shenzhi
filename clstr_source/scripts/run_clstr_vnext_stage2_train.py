#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_stage2_train import train_vnext_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical CLSTR vNext Stage2")
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--candidate_checkpoint_path")
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--trajectory_rows_path", required=True)
    parser.add_argument("--trajectory_dev_rows_path", required=True)
    parser.add_argument("--causal_pair_support_rows_path", required=True)
    parser.add_argument("--causal_pair_support_dev_rows_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--data_contract_path", required=True)
    parser.add_argument("--causal_branch_pairs_path", required=True)
    parser.add_argument("--causal_branch_dev_pairs_path", required=True)
    parser.add_argument("--causal_order_pairs_path")
    parser.add_argument("--causal_order_dev_pairs_path")
    parser.add_argument("--causal_outcome_pairs_path")
    parser.add_argument("--causal_outcome_dev_pairs_path")
    parser.add_argument("--one_error_prefix_rows_path")
    parser.add_argument("--one_error_prefix_dev_rows_path")
    parser.add_argument("--two_or_more_error_prefix_rows_path")
    parser.add_argument("--two_or_more_error_prefix_dev_rows_path")
    parser.add_argument("--recovery_prefix_rows_path")
    parser.add_argument("--recovery_prefix_dev_rows_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--curriculum_total_steps", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--candidate_learning_rate_scale", type=float, default=1.0)
    parser.add_argument("--static_route_learning_rate_scale", type=float, default=1.0)
    parser.add_argument("--transition_scale_initial", type=float, default=0.05)
    parser.add_argument("--result_scale_initial", type=float, default=0.05)
    parser.add_argument("--recall_scale_initial", type=float, default=0.10)
    parser.add_argument(
        "--enable_native_synchronization",
        action="store_true",
        help="Enable the CTM-inspired latent-trace readout in native Stage2.",
    )
    parser.add_argument("--synchronization_pair_dim", type=int, default=96)
    parser.add_argument("--synchronization_trace_length", type=int, default=8)
    parser.add_argument(
        "--synchronization_scale_initial",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max_horizon", type=int, default=16)
    parser.add_argument(
        "--family_first_horizon_sampling",
        action="store_true",
    )
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--compressed_m", type=int, default=64)
    parser.add_argument("--final_k", type=int, default=100)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--frozen_cache_dir")
    parser.add_argument(
        "--frozen_cache_read_only",
        action="store_true",
        help="Require a complete existing frozen cache and never extend it.",
    )
    parser.add_argument("--lambda_recall", type=float, default=0.2)
    parser.add_argument("--lambda_compression", type=float, default=0.2)
    parser.add_argument("--lambda_anchor", type=float, default=1.0e-4)
    parser.add_argument("--teacher_retention_start", type=float, default=1.0)
    parser.add_argument("--teacher_retention_end", type=float, default=0.0)
    parser.add_argument("--lambda_safety", type=float, default=0.05)
    parser.add_argument("--lambda_raw_route", type=float, default=0.5)
    parser.add_argument("--lambda_route_topk", type=float, default=0.0)
    parser.add_argument("--route_topk", type=int, default=5)
    parser.add_argument("--route_topk_margin", type=float, default=0.05)
    parser.add_argument("--route_topk_recoverable_weight", type=float, default=4.0)
    parser.add_argument("--route_topk_listwise_weight", type=float, default=0.05)
    parser.add_argument("--lambda_mixture", type=float, default=0.2)
    parser.add_argument("--mixture_utility_scale", type=float, default=0.5)
    parser.add_argument("--lambda_history", type=float, default=0.2)
    parser.add_argument("--lambda_order", type=float, default=0.0)
    parser.add_argument("--lambda_result", type=float, default=0.0)
    parser.add_argument("--counterfactual_margin", type=float, default=0.2)
    parser.add_argument("--no_regret_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--ordinary_mrr_noninferiority_tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint_interval", type=int, default=400)
    parser.add_argument("--validation_interval", type=int, default=400)
    parser.add_argument("--max_dev_pairs_per_kind", type=int, default=128)
    parser.add_argument("--max_ordinary_dev_rows", type=int, default=1024)
    parser.add_argument("--ordinary_dev_batch_size", type=int, default=64)
    parser.add_argument("--max_robust_dev_rows_per_kind", type=int, default=128)
    parser.add_argument("--minimum_dev_score_gain", type=float, default=0.0)
    parser.add_argument("--minimum_gradient_norm", type=float, default=1.0e-10)
    parser.add_argument("--minimum_dev_clusters_per_enabled_kind", type=int, default=20)
    parser.add_argument("--minimum_dev_clusters_per_source", type=int, default=5)
    parser.add_argument(
        "--minimum_ordinary_dev_clusters_per_stratum",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--minimum_ordinary_dev_stratum_coverage",
        type=float,
        default=0.90,
    )
    parser.add_argument("--minimum_full_pool_clusters", type=int, default=20)
    parser.add_argument(
        "--require_robust_prefix_curriculum",
        action="store_true",
        help="Enable the optional verified executed-error/recovery curriculum",
    )
    parser.set_defaults(require_robust_prefix_curriculum=False)
    parser.add_argument(
        "--minimum_robust_prefix_training_exposure_rate",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--robust_mrr_noninferiority_tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--resume_checkpoint_path")
    parser.add_argument("--warm_start_checkpoint_path")
    parser.add_argument("--require_clean_source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_vnext_stage2(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        candidate_checkpoint_path=args.candidate_checkpoint_path,
        skills_path=args.skills_path,
        trajectory_rows_path=args.trajectory_rows_path,
        trajectory_dev_rows_path=args.trajectory_dev_rows_path,
        causal_pair_support_rows_path=args.causal_pair_support_rows_path,
        causal_pair_support_dev_rows_path=args.causal_pair_support_dev_rows_path,
        inventory_catalogs_path=args.inventory_catalogs_path,
        data_contract_path=args.data_contract_path,
        causal_branch_pairs_path=args.causal_branch_pairs_path,
        causal_branch_dev_pairs_path=args.causal_branch_dev_pairs_path,
        causal_order_pairs_path=args.causal_order_pairs_path,
        causal_order_dev_pairs_path=args.causal_order_dev_pairs_path,
        causal_outcome_pairs_path=args.causal_outcome_pairs_path,
        causal_outcome_dev_pairs_path=args.causal_outcome_dev_pairs_path,
        one_error_prefix_rows_path=args.one_error_prefix_rows_path,
        one_error_prefix_dev_rows_path=args.one_error_prefix_dev_rows_path,
        two_or_more_error_prefix_rows_path=args.two_or_more_error_prefix_rows_path,
        two_or_more_error_prefix_dev_rows_path=(
            args.two_or_more_error_prefix_dev_rows_path
        ),
        recovery_prefix_rows_path=args.recovery_prefix_rows_path,
        recovery_prefix_dev_rows_path=args.recovery_prefix_dev_rows_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        curriculum_total_steps=args.curriculum_total_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        candidate_learning_rate_scale=args.candidate_learning_rate_scale,
        static_route_learning_rate_scale=args.static_route_learning_rate_scale,
        transition_scale_initial=args.transition_scale_initial,
        result_scale_initial=args.result_scale_initial,
        recall_scale_initial=args.recall_scale_initial,
        synchronization_enabled=args.enable_native_synchronization,
        synchronization_pair_dim=args.synchronization_pair_dim,
        synchronization_trace_length=args.synchronization_trace_length,
        synchronization_scale_initial=args.synchronization_scale_initial,
        max_horizon=args.max_horizon,
        family_first_horizon_sampling=args.family_first_horizon_sampling,
        coarse_k=args.coarse_k,
        compressed_m=args.compressed_m,
        final_k=args.final_k,
        belief_top_k=args.belief_top_k,
        cache_batch_size=args.cache_batch_size,
        cache_shard_size=args.cache_shard_size,
        frozen_cache_dir=args.frozen_cache_dir,
        frozen_cache_read_only=args.frozen_cache_read_only,
        lambda_recall=args.lambda_recall,
        lambda_compression=args.lambda_compression,
        lambda_anchor=args.lambda_anchor,
        teacher_retention_start=args.teacher_retention_start,
        teacher_retention_end=args.teacher_retention_end,
        lambda_safety=args.lambda_safety,
        lambda_raw_route=args.lambda_raw_route,
        lambda_route_topk=args.lambda_route_topk,
        route_topk=args.route_topk,
        route_topk_margin=args.route_topk_margin,
        route_topk_recoverable_weight=args.route_topk_recoverable_weight,
        route_topk_listwise_weight=args.route_topk_listwise_weight,
        lambda_mixture=args.lambda_mixture,
        mixture_utility_scale=args.mixture_utility_scale,
        lambda_history=args.lambda_history,
        lambda_order=args.lambda_order,
        lambda_result=args.lambda_result,
        counterfactual_margin=args.counterfactual_margin,
        no_regret_tolerance=args.no_regret_tolerance,
        ordinary_mrr_noninferiority_tolerance=(
            args.ordinary_mrr_noninferiority_tolerance
        ),
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        max_dev_pairs_per_kind=args.max_dev_pairs_per_kind,
        max_ordinary_dev_rows=args.max_ordinary_dev_rows,
        ordinary_dev_batch_size=args.ordinary_dev_batch_size,
        max_robust_dev_rows_per_kind=args.max_robust_dev_rows_per_kind,
        minimum_dev_score_gain=args.minimum_dev_score_gain,
        minimum_gradient_norm=args.minimum_gradient_norm,
        minimum_dev_clusters_per_enabled_kind=(
            args.minimum_dev_clusters_per_enabled_kind
        ),
        minimum_dev_clusters_per_source=args.minimum_dev_clusters_per_source,
        minimum_ordinary_dev_clusters_per_stratum=(
            args.minimum_ordinary_dev_clusters_per_stratum
        ),
        minimum_ordinary_dev_stratum_coverage=(
            args.minimum_ordinary_dev_stratum_coverage
        ),
        minimum_full_pool_clusters=args.minimum_full_pool_clusters,
        require_robust_prefix_curriculum=args.require_robust_prefix_curriculum,
        minimum_robust_prefix_training_exposure_rate=(
            args.minimum_robust_prefix_training_exposure_rate
        ),
        robust_mrr_noninferiority_tolerance=(
            args.robust_mrr_noninferiority_tolerance
        ),
        max_rows=args.max_rows,
        resume_checkpoint_path=args.resume_checkpoint_path,
        warm_start_checkpoint_path=args.warm_start_checkpoint_path,
        require_clean_source=args.require_clean_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
