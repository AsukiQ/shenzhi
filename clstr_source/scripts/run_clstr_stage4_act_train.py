#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import (
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    NEXT_SKILL_POOL_MODES,
    ROUTE_SCORERS,
    STAGE0_HANDOFF_CACHE_FORMATS,
    STAGE0_HANDOFF_QUERY_MODES,
    TRANSITION_SCORING_MODE,
    TRANSITION_SCORING_MODES,
    UNIFIED_MEMORY_ROUTE_SCORER,
)
from clstr.stage4_act_train import STAGE4_METHODS, train_stage4_act_with_model
from clstr.stage4_safe_memory import (
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_cmc_parent_router,
)
from clstr.stage_checkpoint_init import (
    build_clstr_model_from_stage0_checkpoint,
    checkpoint_payload,
    checkpoint_state,
    load_routing_and_head_checkpoints,
)
from clstr.qwen_clstr_lineage import sha256_path
from clstr.training_monitor import append_setup_status, reset_setup_status


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


def _parse_candidate_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("safe local candidate sizes must be comma-separated integers of at least two")
    return sizes


def _load_candidate_admission_base_cmc(
    model,
    checkpoint_path: str | Path,
) -> dict:
    path = Path(checkpoint_path)
    payload = checkpoint_payload(path, "base CMC")
    if payload.get("stage4_method") != "counterfactual_memory_calibration_v1":
        raise ValueError("candidate-admission base checkpoint must be CMC Stage4")
    state = checkpoint_state(payload, "base CMC")
    delta_validation = validate_stage4_cmc_delta_state_dict(state)
    parent_validation = validate_stage4_cmc_parent_router(payload, model)
    model.load_state_dict(state, strict=False)
    identity = sha256_path(path)
    return {
        "status": "ok",
        "base_cmc_checkpoint_path": identity["path"],
        "base_cmc_checkpoint_sha256": identity["sha256"],
        "delta_validation": delta_validation,
        "parent_validation": parent_validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train causal CLSTR-native Stage4 next-skill routing over the declared legal skill pool, "
            "using the Stage0 shortlist only as a static-recall diagnostic. "
            "Qwen is intentionally not part of this mainline entry."
        )
    )
    parser.add_argument("--trajectories_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_memory_stage4_full_pool_counterfactual")
    parser.add_argument(
        "--routing_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--head_checkpoint_path",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument("--base_cmc_checkpoint_path", default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3.0e-5)
    parser.add_argument("--minimum_learning_rate", type=float, default=3.0e-6)
    parser.add_argument("--learning_rate_warmup_fraction", type=float, default=0.05)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--validation_rows_per_benchmark", type=int, default=256)
    parser.add_argument(
        "--minimum_validation_rows_per_benchmark",
        type=int,
        default=128,
    )
    parser.add_argument("--gate_rows_per_benchmark", type=int, default=512)
    parser.add_argument("--validation_interval_steps", type=int, default=400)
    parser.add_argument("--resume_checkpoint_path", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--candidate_count", type=int, default=64)
    parser.add_argument("--max_rows", type=int, default=None)
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
        default=UNIFIED_MEMORY_ROUTE_SCORER,
        choices=sorted(ROUTE_SCORERS),
        help="Main Stage4 route scorer. unified_memory is the causal paper mainline; legacy mode is an explicit ablation.",
    )
    parser.add_argument("--next_skill_pool_mode", default="full_pool", choices=sorted(NEXT_SKILL_POOL_MODES))
    parser.add_argument(
        "--stage4_method",
        default="stage4_safe_memory_v1",
        choices=sorted(STAGE4_METHODS),
    )
    parser.add_argument("--counterfactual_utility_weight", type=float, default=0.05)
    parser.add_argument("--counterfactual_gain_margin", type=float, default=0.1)
    parser.add_argument("--counterfactual_safety_tolerance", type=float, default=0.01)
    parser.add_argument("--counterfactual_gain_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_safety_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_warmup_fraction", type=float, default=0.05)
    parser.add_argument("--safe_memory_residual_bound", type=float, default=2.0)
    parser.add_argument("--safe_local_candidate_sizes", default="2,3,4,5,8,10")
    parser.add_argument(
        "--stage0_top_m",
        type=int,
        default=350,
        help="If set, build Stage4 next-skill candidates from online Stage0 top-M handoff.",
    )
    parser.add_argument(
        "--stage0_positive_missing_policy",
        default="skip",
        choices=["skip", "inject", "skip_or_inject_with_provenance"],
        help="Policy passed to the shared Stage0 top-M candidate handoff.",
    )
    parser.add_argument(
        "--stage0_handoff_query_mode",
        default="skillrouter_state",
        choices=sorted(STAGE0_HANDOFF_QUERY_MODES),
        help="Query formatting used for Stage0 candidate handoff.",
    )
    parser.add_argument(
        "--stage0_handoff_sample_multiplier",
        type=int,
        default=8,
        help="For smoke runs with --max_rows, pre-handoff source rows are capped at max_rows times this value.",
    )
    parser.add_argument(
        "--stage0_inventory_min_candidates",
        type=int,
        default=0,
        help="Minimum Stage0-backed candidates retained when a row has an explicit/visible skill inventory.",
    )
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--stage0_handoff_cache_mode", default="off", choices=["auto", "off", "refresh"])
    parser.add_argument("--stage0_handoff_cache_dir", default="outputs/cache/stage0_handoff")
    parser.add_argument(
        "--stage0_handoff_cache_format",
        default="legacy_jsonl",
        choices=sorted(STAGE0_HANDOFF_CACHE_FORMATS),
    )
    parser.add_argument("--stage0_handoff_cache_shard_size", type=int, default=2048)
    parser.add_argument(
        "--allowed_benchmarks",
        default="toolbench_g3,traject_bench,alfworld,webshop",
        help="Comma-separated benchmark names retained for Stage4, e.g. traject_bench,toolbench_g3.",
    )
    parser.add_argument(
        "--benchmark_caps",
        default=None,
        help=(
            "Comma-separated benchmark=row_cap controls applied after allowed_benchmarks. "
            "Use -1 for unlimited, e.g. toolbench_g3=4000,traject_bench=4000,alfworld=-1,webshop=4000."
        ),
    )
    parser.add_argument(
        "--train_transition",
        action="store_true",
        help=(
            "Enable TransitionPredictor training for legacy scoring mode. "
            "Unified-memory mode always trains the causal transition/gate modules."
        ),
    )
    parser.add_argument("--auto_replay_prefix_max_steps", type=int, default=3)
    parser.add_argument("--trainable_replay_prefix", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_status_path = output_dir / "setup_status.jsonl"
    reset_setup_status(setup_status_path)
    append_setup_status(
        setup_status_path,
        "stage4_clstr_native_started",
        trajectories_path=str(args.trajectories_path),
        skills_path=str(args.skills_path),
        routing_checkpoint_path=str(args.routing_checkpoint_path),
        head_checkpoint_path=str(args.head_checkpoint_path),
        base_cmc_checkpoint_path=args.base_cmc_checkpoint_path,
        transition_residual_lambda=float(args.transition_residual_lambda),
        transition_scoring_mode=str(args.transition_scoring_mode),
        route_scorer=str(args.route_scorer),
        next_skill_pool_mode=str(args.next_skill_pool_mode),
        stage4_method=str(args.stage4_method),
        counterfactual_utility_weight=float(args.counterfactual_utility_weight),
        counterfactual_gain_margin=float(args.counterfactual_gain_margin),
        counterfactual_safety_tolerance=float(args.counterfactual_safety_tolerance),
        counterfactual_gain_weight=float(args.counterfactual_gain_weight),
        counterfactual_safety_weight=float(args.counterfactual_safety_weight),
        counterfactual_warmup_fraction=float(args.counterfactual_warmup_fraction),
        safe_memory_residual_bound=float(args.safe_memory_residual_bound),
        safe_local_candidate_sizes=_parse_candidate_sizes(args.safe_local_candidate_sizes),
        minimum_learning_rate=float(args.minimum_learning_rate),
        learning_rate_warmup_fraction=float(args.learning_rate_warmup_fraction),
        validation_fraction=float(args.validation_fraction),
        validation_rows_per_benchmark=int(args.validation_rows_per_benchmark),
        minimum_validation_rows_per_benchmark=int(
            args.minimum_validation_rows_per_benchmark
        ),
        gate_rows_per_benchmark=int(args.gate_rows_per_benchmark),
        validation_interval_steps=int(args.validation_interval_steps),
        resume_checkpoint_path=args.resume_checkpoint_path,
        stage0_top_m=args.stage0_top_m,
        stage0_positive_missing_policy=str(args.stage0_positive_missing_policy),
        stage0_handoff_query_mode=str(args.stage0_handoff_query_mode),
        stage0_handoff_sample_multiplier=int(args.stage0_handoff_sample_multiplier),
        stage0_inventory_min_candidates=int(args.stage0_inventory_min_candidates),
        auto_replay_prefix_max_steps=int(args.auto_replay_prefix_max_steps),
        trainable_replay_prefix=bool(args.trainable_replay_prefix),
        benchmark_caps=args.benchmark_caps,
    )
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(args.routing_checkpoint_path),
        skills_path=Path(args.skills_path),
        model_cache_dir=output_dir / "model_cache",
    )
    checkpoint_init_report = load_routing_and_head_checkpoints(
        model,
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        head_checkpoint_path=Path(args.head_checkpoint_path),
        partial_load_mode="stage0_routing_plus_act_init_heads_for_joint_stage4",
        protect_routing_foundation=True,
    )
    base_cmc_report = None
    if str(args.stage4_method) == "candidate_admission_residual_v1":
        if not args.base_cmc_checkpoint_path:
            raise ValueError(
                "candidate-admission Stage4 requires --base_cmc_checkpoint_path"
            )
        base_cmc_report = _load_candidate_admission_base_cmc(
            model,
            args.base_cmc_checkpoint_path,
        )
        checkpoint_init_report = {
            **checkpoint_init_report,
            "candidate_admission_base_cmc": base_cmc_report,
        }
    append_setup_status(setup_status_path, "stage4_checkpoint_init_loaded", checkpoint_init_report=checkpoint_init_report)
    report = train_stage4_act_with_model(
        model=model,
        trajectories_path=Path(args.trajectories_path),
        skills_path=Path(args.skills_path),
        output_dir=output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        learning_rate_warmup_fraction=args.learning_rate_warmup_fraction,
        validation_fraction=args.validation_fraction,
        validation_rows_per_benchmark=args.validation_rows_per_benchmark,
        minimum_validation_rows_per_benchmark=(
            args.minimum_validation_rows_per_benchmark
        ),
        gate_rows_per_benchmark=args.gate_rows_per_benchmark,
        validation_interval_steps=args.validation_interval_steps,
        resume_checkpoint_path=(
            None
            if args.resume_checkpoint_path is None
            else Path(args.resume_checkpoint_path)
        ),
        seed=args.seed,
        candidate_count=args.candidate_count,
        train_transition=args.train_transition,
        max_rows=args.max_rows,
        transition_residual_lambda=args.transition_residual_lambda,
        transition_scoring_mode=args.transition_scoring_mode,
        route_scorer=args.route_scorer,
        next_skill_pool_mode=args.next_skill_pool_mode,
        stage4_method=args.stage4_method,
        counterfactual_utility_weight=args.counterfactual_utility_weight,
        counterfactual_gain_margin=args.counterfactual_gain_margin,
        counterfactual_safety_tolerance=args.counterfactual_safety_tolerance,
        counterfactual_gain_weight=args.counterfactual_gain_weight,
        counterfactual_safety_weight=args.counterfactual_safety_weight,
        counterfactual_warmup_fraction=args.counterfactual_warmup_fraction,
        safe_memory_residual_bound=args.safe_memory_residual_bound,
        safe_local_candidate_sizes=_parse_candidate_sizes(args.safe_local_candidate_sizes),
        allowed_benchmarks=_parse_allowed_benchmarks(args.allowed_benchmarks),
        benchmark_caps=_parse_benchmark_caps(args.benchmark_caps),
        stage0_top_m=args.stage0_top_m,
        stage0_positive_missing_policy=args.stage0_positive_missing_policy,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_handoff_sample_multiplier=args.stage0_handoff_sample_multiplier,
        stage0_inventory_min_candidates=args.stage0_inventory_min_candidates,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        stage0_handoff_cache_mode=args.stage0_handoff_cache_mode,
        stage0_handoff_cache_dir=Path(args.stage0_handoff_cache_dir),
        stage0_handoff_cache_format=args.stage0_handoff_cache_format,
        stage0_handoff_cache_shard_size=args.stage0_handoff_cache_shard_size,
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        checkpoint_init_report=checkpoint_init_report,
        base_cmc_checkpoint_sha256=(
            None
            if base_cmc_report is None
            else base_cmc_report["base_cmc_checkpoint_sha256"]
        ),
        setup_status_path=setup_status_path,
        auto_replay_prefix_max_steps=args.auto_replay_prefix_max_steps,
        trainable_replay_prefix=args.trainable_replay_prefix,
    )
    report["checkpoint_init"] = checkpoint_init_report
    report["routing_init"] = routing_report
    report["model_config"] = model_config
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
