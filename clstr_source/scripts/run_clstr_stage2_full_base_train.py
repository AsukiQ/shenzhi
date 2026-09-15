#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import (
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    NEXT_SKILL_POOL_MODES,
    SAMPLING_STRATEGIES,
    TRANSITION_INVENTORY_MASK_MODES,
    TRANSITION_LOSS_TYPES,
    TRANSITION_POSITIVE_MODES,
    ROUTE_SCORERS,
    TRANSITION_SCORING_MODE,
    TRANSITION_SCORING_MODES,
    STAGE0_HANDOFF_QUERY_MODES,
    STAGE0_HANDOFF_CACHE_FORMATS,
    run_clstr_full_base_train,
)


def _parse_allowed_benchmarks(value: str | None) -> set[str] | None:
    if value is None:
        return None
    benchmarks = {item.strip() for item in str(value).split(",") if item.strip()}
    return benchmarks or None


def _parse_benchmark_caps(value: str | None) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    caps: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"benchmark cap must use name=value format: {item}")
        name, raw_cap = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"benchmark cap has empty benchmark name: {item}")
        caps[name] = int(raw_cap.strip())
    return caps or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CLSTR-native Stage2 heads with full-pool causal next-skill routing. "
            "Qwen is intentionally not part of this mainline training entry."
        )
    )
    parser.add_argument("--train_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument(
        "--output_dir",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025",
    )
    parser.add_argument(
        "--routing_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage1_checkpoint_path",
        default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init/checkpoints/clstr_stage1_heads-step3000.pt",
        help="Stage1 heads-init checkpoint to continue the same CLSTR model from.",
    )
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument(
        "--target_total_steps",
        type=int,
        default=None,
        help=(
            "Optional absolute final step for exact resume. max_steps remains the "
            "data/sampling schedule budget."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum_learning_rate", type=float, default=None)
    parser.add_argument("--checkpoint_interval_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy_loss_weight", type=float, default=0.2)
    parser.add_argument("--transition_loss_weight", type=float, default=0.3)
    parser.add_argument("--transition_skill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_history_loss_weight", type=float, default=0.0)
    parser.add_argument("--counterfactual_history_margin", type=float, default=0.1)
    parser.add_argument("--transition_real_candidate_ce_multiplier", type=float, default=1.0)
    parser.add_argument("--transition_injected_candidate_ce_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--next_skill_pool_mode",
        default="full_pool",
        choices=sorted(NEXT_SKILL_POOL_MODES),
        help="Use the declared legal skill pool for causal next-skill supervision by default.",
    )
    parser.add_argument(
        "--transition_inventory_mask_mode",
        default="explicit_only",
        choices=sorted(TRANSITION_INVENTORY_MASK_MODES),
        help="Only explicit row inventory may restrict the full declared skill pool.",
    )
    parser.add_argument(
        "--transition_inventory_min_candidates",
        type=int,
        default=64,
        help="Minimum candidate count after inventory filtering; backfills from original Stage0 top-M order when positive.",
    )
    parser.add_argument(
        "--transition_loss_type",
        default="listwise_nll",
        choices=sorted(TRANSITION_LOSS_TYPES),
        help="Transition skill loss over Stage0 top-M candidates.",
    )
    parser.add_argument(
        "--transition_positive_mode",
        default="gold_plus_equivalent",
        choices=sorted(TRANSITION_POSITIVE_MODES),
        help="Positive-label policy for transition skill ranking.",
    )
    parser.add_argument(
        "--transition_residual_lambda",
        type=float,
        default=DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
        help="Weight on the action-aware residual branch in prior + residual transition scoring.",
    )
    parser.add_argument(
        "--transition_scoring_mode",
        default=TRANSITION_SCORING_MODE,
        choices=sorted(TRANSITION_SCORING_MODES),
        help="Transition skill scoring semantics. v4_1b_action_observation replays the conservative v4.1b path.",
    )
    parser.add_argument(
        "--route_scorer",
        default="unified_memory",
        choices=sorted(ROUTE_SCORERS),
        help="Main Stage1/2 route scorer. unified_memory trains z_t dot E instead of Stage0 prior plus residual.",
    )
    parser.add_argument("--anchored_routing_foundation", action="store_true")
    parser.add_argument("--static_route_anchor_weight", type=float, default=0.1)
    parser.add_argument("--static_route_anchor_max_regression", type=float, default=0.005)
    parser.add_argument("--stage0_score_prior_calibration", default="off", choices=["off", "rank", "rank_std", "raw", "none"])
    parser.add_argument("--belief_loss_weight", type=float, default=0.1)
    parser.add_argument("--embedding_cache_mode", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--embedding_cache_max_rows", type=int, default=20000)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional small-sample smoke limit applied after train/benchmark filters.")
    parser.add_argument("--include_available_actions_in_state", action="store_true")
    parser.add_argument("--skill_text_format", default=None)
    parser.add_argument(
        "--allowed_benchmarks",
        default="toolbench_g3,traject_bench,alfworld,webshop",
        help="Comma-separated benchmark names retained for Stage2 training, e.g. traject_bench,toolbench_g3.",
    )
    parser.add_argument(
        "--benchmark_caps",
        default="toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1",
        help=(
            "Comma-separated benchmark=row_cap controls applied after allowed_benchmarks. "
            "Use -1 for unlimited, e.g. toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000."
        ),
    )
    parser.add_argument(
        "--stage0_top_m",
        type=int,
        default=350,
        help="Number of Stage0 coarse-retriever candidates consumed by Stage2.",
    )
    parser.add_argument(
        "--stage0_positive_missing_policy",
        default="skip",
        choices=["skip", "inject"],
        help="How Stage2 handles rows whose positive skill is missing from Stage0 top-M.",
    )
    parser.add_argument(
        "--stage0_handoff_query_mode",
        default="skillrouter_state",
        choices=sorted(STAGE0_HANDOFF_QUERY_MODES),
        help="Text protocol used to compute Stage0 handoff candidates for Stage2.",
    )
    parser.add_argument(
        "--stage0_handoff_sample_multiplier",
        type=float,
        default=None,
        help=(
            "Optional multiplier over max_steps used to precompute Stage0 top-M only for rows "
            "reachable by the current training budget. Disabled when omitted."
        ),
    )
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--stage0_handoff_cache_mode", default="auto", choices=["auto", "off", "refresh"])
    parser.add_argument("--stage0_handoff_cache_dir", default="outputs/cache/stage0_handoff")
    parser.add_argument(
        "--stage0_handoff_cache_format",
        default="legacy_jsonl",
        choices=sorted(STAGE0_HANDOFF_CACHE_FORMATS),
    )
    parser.add_argument("--stage0_handoff_cache_shard_size", type=int, default=2048)
    parser.add_argument(
        "--sampling_strategy",
        default="balanced_random",
        choices=sorted(SAMPLING_STRATEGIES),
        help="Stage2 defaults to seeded balanced random batches; relation-aware samplers are explicit diagnostics.",
    )
    parser.add_argument("--auto_replay_prefix_max_steps", type=int, default=0)
    parser.add_argument("--trainable_replay_prefix", action="store_true")
    parser.add_argument(
        "--warm_start_checkpoint_path",
        default=None,
        help=(
            "Optional Stage2 checkpoint whose compatible model weights initialize a fresh "
            "optimizer and step-zero counterfactual repair run."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        default=None,
        help=(
            "Optional Stage2 latest/final checkpoint to continue from. "
            "max_steps is interpreted as additional steps after the checkpoint step."
        ),
    )
    parser.add_argument(
        "--allow_full_pool_stage2_debug",
        action="store_true",
        help="Explicit debug escape hatch. Without this, Stage2 requires Stage0 top-M candidates.",
    )
    args = parser.parse_args()

    report = run_clstr_full_base_train(
        train_path=Path(args.train_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        target_total_steps=args.target_total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        seed=args.seed,
        loss_weights={
            "L_policy": args.policy_loss_weight,
            "L_trans": args.transition_loss_weight,
            "L_trans_skill_ce": args.transition_skill_ce_loss_weight,
            "counterfactual_history": args.counterfactual_history_loss_weight,
            "belief": args.belief_loss_weight,
        },
        transition_real_candidate_ce_multiplier=args.transition_real_candidate_ce_multiplier,
        transition_injected_candidate_ce_multiplier=args.transition_injected_candidate_ce_multiplier,
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_inventory_min_candidates=args.transition_inventory_min_candidates,
        transition_loss_type=args.transition_loss_type,
        transition_positive_mode=args.transition_positive_mode,
        transition_residual_lambda=args.transition_residual_lambda,
        transition_scoring_mode=args.transition_scoring_mode,
        route_scorer=args.route_scorer,
        anchored_routing_foundation=args.anchored_routing_foundation,
        static_route_anchor_weight=args.static_route_anchor_weight,
        static_route_anchor_max_regression=args.static_route_anchor_max_regression,
        next_skill_pool_mode=args.next_skill_pool_mode,
        stage0_score_prior_calibration=args.stage0_score_prior_calibration,
        include_available_actions_in_state=args.include_available_actions_in_state,
        skill_text_format=args.skill_text_format,
        routing_checkpoint_path=Path(args.routing_checkpoint_path) if args.routing_checkpoint_path else None,
        stage1_checkpoint_path=Path(args.stage1_checkpoint_path) if args.stage1_checkpoint_path else None,
        embedding_cache_mode=args.embedding_cache_mode,
        embedding_cache_max_rows=args.embedding_cache_max_rows,
        max_rows=args.max_rows,
        allowed_benchmarks=_parse_allowed_benchmarks(args.allowed_benchmarks),
        benchmark_caps=_parse_benchmark_caps(args.benchmark_caps),
        stage0_top_m=args.stage0_top_m,
        stage0_positive_missing_policy=args.stage0_positive_missing_policy,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=args.stage0_handoff_sample_multiplier,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        stage0_handoff_cache_mode=args.stage0_handoff_cache_mode,
        stage0_handoff_cache_dir=Path(args.stage0_handoff_cache_dir),
        stage0_handoff_cache_format=args.stage0_handoff_cache_format,
        stage0_handoff_cache_shard_size=args.stage0_handoff_cache_shard_size,
        allow_full_pool_stage2_debug=args.allow_full_pool_stage2_debug,
        sampling_strategy=args.sampling_strategy,
        auto_replay_prefix_max_steps=args.auto_replay_prefix_max_steps,
        trainable_replay_prefix=args.trainable_replay_prefix,
        counterfactual_history_margin=args.counterfactual_history_margin,
        warm_start_checkpoint_path=(
            Path(args.warm_start_checkpoint_path)
            if args.warm_start_checkpoint_path
            else None
        ),
        resume_checkpoint_path=Path(args.resume_checkpoint_path) if args.resume_checkpoint_path else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
