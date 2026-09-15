#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FUNCTION_AUG_V2_DATA_ROOT = Path("data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2")
DEFAULT_TRANSITION_RESIDUAL_LAMBDA = 0.25
DEFAULT_GATED_TEMPORAL_LAMBDA_MAX = 0.5
DEFAULT_GATED_TEMPORAL_KL_ALPHA = 0.03
DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA = 0.05
DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K = 64
TRANSITION_SCORING_MODE = "stage0_rank_prior_plus_transition_residual"
TRANSITION_INVENTORY_MASK_MODES = {
    "off",
    "auto",
    "explicit_only",
    "root_namespace",
    "stage0_topk_trajectory_prior",
}
NEXT_SKILL_POOL_MODES = {"stage0_candidates", "full_pool"}
TRANSITION_LOSS_TYPES = {"cross_entropy", "listwise_nll"}
TRANSITION_POSITIVE_MODES = {"single", "gold_plus_equivalent"}
TRANSITION_SCORING_MODES = {
    "stage0_rank_prior_plus_transition_residual",
    "stage0_prior_gated_temporal_residual",
    "skill_prior_plus_action_observation_residual",
    "v4_1b_action_observation",
}
LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER = "legacy_prior_residual"
UNIFIED_MEMORY_ROUTE_SCORER = "unified_memory"
ROUTE_SCORERS = {LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER, UNIFIED_MEMORY_ROUTE_SCORER}
SAMPLING_STRATEGIES = {
    "balanced_deterministic",
    "balanced_random",
    "benchmark_transition_balanced_random",
    "benchmark_transition_quota_random",
}

run_clstr_full_base_train = None
run_clstr_stage1_heads_init = None
audit_stage1_heads_quality = None
audit_stage2_full_base_quality = None


def _load_training_stack() -> None:
    global audit_stage1_heads_quality
    global audit_stage2_full_base_quality
    global run_clstr_full_base_train
    global run_clstr_stage1_heads_init
    if (
        run_clstr_full_base_train is not None
        and run_clstr_stage1_heads_init is not None
        and audit_stage1_heads_quality is not None
        and audit_stage2_full_base_quality is not None
    ):
        return
    from clstr.full_base_train import (  # noqa: PLC0415
        run_clstr_full_base_train as _run_clstr_full_base_train,
    )
    from clstr.full_base_train import (  # noqa: PLC0415
        run_clstr_stage1_heads_init as _run_clstr_stage1_heads_init,
    )
    from clstr.stage1_heads_quality_gate import audit_stage1_heads_quality as _audit_stage1_heads_quality  # noqa: PLC0415
    from clstr.stage2_quality_gate import audit_stage2_full_base_quality as _audit_stage2_full_base_quality  # noqa: PLC0415

    run_clstr_full_base_train = _run_clstr_full_base_train
    run_clstr_stage1_heads_init = _run_clstr_stage1_heads_init
    audit_stage1_heads_quality = _audit_stage1_heads_quality
    audit_stage2_full_base_quality = _audit_stage2_full_base_quality


def _parse_allowed_benchmarks(value: str | None) -> set[str] | None:
    if value is None:
        return None
    values = {item.strip() for item in str(value).split(",") if item.strip()}
    return values or None


def _parse_benchmark_caps(value: str | None) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    caps: dict[str, int] = {}
    for raw_item in str(value).replace(":", ",").split(","):
        item = raw_item.strip()
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


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"expected optional bool true/false/auto, got: {value}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _checkpoint_from_report(report: dict[str, Any], fallback: Path) -> Path:
    raw = report.get("checkpoint") or report.get("latest_checkpoint")
    checkpoint = Path(raw) if raw else fallback
    if not checkpoint.is_file():
        raise FileNotFoundError(f"expected checkpoint was not written: {checkpoint}")
    return checkpoint


def _raise_if_gate_failed(name: str, gate: dict[str, Any]) -> None:
    if gate.get("status") != "ok":
        blockers = gate.get("blockers")
        raise RuntimeError(f"{name} gate failed: {blockers}")


def run_stage12_consolidated_train(
    *,
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    routing_checkpoint_path: str | Path,
    stage1_max_steps: int = 3000,
    stage2_max_steps: int = 10000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    max_rows: int | None = None,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    stage0_top_m: int = 500,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_handoff_sample_multiplier: float | None = 4.0,
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 50,
    stage0_handoff_cache_mode: str = "auto",
    stage0_handoff_cache_dir: str | Path = "outputs/cache/stage0_handoff",
    transition_inventory_mask_mode: str = "explicit_only",
    transition_inventory_min_candidates: int = 50,
    transition_loss_type: str = "listwise_nll",
    transition_positive_mode: str = "gold_plus_equivalent",
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
    gated_temporal_lambda_max: float = DEFAULT_GATED_TEMPORAL_LAMBDA_MAX,
    gated_temporal_kl_alpha: float = DEFAULT_GATED_TEMPORAL_KL_ALPHA,
    gated_temporal_rank_drop_beta: float = DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA,
    gated_temporal_context_top_k: int = DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K,
    stage0_score_prior_calibration: str = "off",
    freeze_gated_temporal_only: bool = False,
    stage1_freeze_gated_temporal_only: bool | None = None,
    stage2_freeze_gated_temporal_only: bool | None = None,
    transition_hard_negative_margin: float = 1.0,
    next_skill_pool_mode: str = "full_pool",
    counterfactual_gain_margin: float = 0.1,
    counterfactual_safety_tolerance: float = 0.01,
    counterfactual_gain_weight: float = 1.0,
    counterfactual_safety_weight: float = 1.0,
    counterfactual_warmup_fraction: float = 0.1,
    sampling_strategy: str = "benchmark_transition_quota_random",
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    include_available_actions_in_state: bool = False,
    skill_text_format: str | None = None,
    stage1_policy_loss_weight: float = 0.6,
    stage1_transition_loss_weight: float = 0.2,
    stage1_transition_skill_ce_loss_weight: float = 0.6,
    stage1_belief_loss_weight: float = 0.1,
    stage1_stop_loss_weight: float = 0.1,
    stage2_policy_loss_weight: float = 0.2,
    stage2_policy_hard_negative_margin_loss_weight: float = 0.0,
    stage2_q_success_loss_weight: float = 0.0,
    stage2_transition_loss_weight: float = 0.3,
    stage2_transition_skill_ce_loss_weight: float = 1.0,
    stage2_transition_hard_negative_margin_loss_weight: float = 0.0,
    stage2_counterfactual_utility_loss_weight: float = 0.05,
    stage2_belief_loss_weight: float = 0.1,
    stage2_stop_loss_weight: float = 0.1,
    retrieval_loss_weight: float = 0.0,
    retrieval_num_negatives: int = 32,
    retrieval_hard_ratio: float = 0.5,
    stage1_gate_first_window: int = 100,
    stage1_gate_last_window: int = 100,
    stage1_gate_min_loss_drop: float = 0.0,
    stage1_gate_min_transition_recall_at_1_tail: float = 0.35,
    stage1_gate_min_transition_recall_at_5_tail: float = 0.7,
    stage2_gate_first_window: int = 200,
    stage2_gate_last_window: int = 200,
    stage2_gate_min_loss_drop: float = 0.02,
    stage2_gate_min_transition_recall_at_5_tail: float = 0.5,
    stage2_gate_max_recall_at_5_drop_vs_stage0_prior: float = 0.03,
    stage2_gate_max_mrr_drop_vs_stage0_prior: float = 0.01,
    stage2_gate_max_worse_than_stage0_prior_fraction: float = 0.5,
    auto_replay_prefix_max_steps: int = 0,
    trainable_replay_prefix: bool = False,
    stage1_auto_replay_prefix_max_steps: int | None = None,
    stage2_auto_replay_prefix_max_steps: int | None = None,
    stage1_trainable_replay_prefix: bool | None = None,
    stage2_trainable_replay_prefix: bool | None = None,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
) -> dict[str, Any]:
    train_path = Path(train_path)
    skills_path = Path(skills_path)
    output_dir = Path(output_dir)
    routing_checkpoint_path = Path(routing_checkpoint_path)
    if not routing_checkpoint_path.is_file():
        raise FileNotFoundError(f"routing checkpoint not found: {routing_checkpoint_path}")
    route_scorer = str(route_scorer or UNIFIED_MEMORY_ROUTE_SCORER)
    if route_scorer not in ROUTE_SCORERS:
        raise ValueError(f"unsupported route_scorer: {route_scorer}")
    next_skill_pool_mode = str(next_skill_pool_mode or "full_pool")
    if next_skill_pool_mode not in NEXT_SKILL_POOL_MODES:
        raise ValueError(f"unsupported next_skill_pool_mode: {next_skill_pool_mode}")
    transition_inventory_mask_mode = str(transition_inventory_mask_mode or "explicit_only")
    if transition_inventory_mask_mode not in TRANSITION_INVENTORY_MASK_MODES:
        raise ValueError(f"unsupported transition_inventory_mask_mode: {transition_inventory_mask_mode}")
    if next_skill_pool_mode == "full_pool" and route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        raise ValueError("next_skill_pool_mode=full_pool requires route_scorer=unified_memory")
    if next_skill_pool_mode == "full_pool" and transition_inventory_mask_mode != "explicit_only":
        raise ValueError("full-pool Stage2 requires transition_inventory_mask_mode=explicit_only")
    counterfactual_warmup_fraction = float(counterfactual_warmup_fraction)
    if not 0.0 <= counterfactual_warmup_fraction <= 1.0:
        raise ValueError("counterfactual_warmup_fraction must be in [0, 1]")
    _load_training_stack()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = output_dir / "stage1_heads_init"
    stage2_dir = output_dir / "stage2_full_base"
    stage1_freeze = bool(freeze_gated_temporal_only) if stage1_freeze_gated_temporal_only is None else bool(stage1_freeze_gated_temporal_only)
    stage2_freeze = bool(freeze_gated_temporal_only) if stage2_freeze_gated_temporal_only is None else bool(stage2_freeze_gated_temporal_only)
    stage1_replay_steps = 0 if stage1_auto_replay_prefix_max_steps is None else max(0, int(stage1_auto_replay_prefix_max_steps))
    stage2_replay_steps = (
        max(0, int(auto_replay_prefix_max_steps))
        if stage2_auto_replay_prefix_max_steps is None
        else max(0, int(stage2_auto_replay_prefix_max_steps))
    )
    stage1_replay_trainable = False if stage1_trainable_replay_prefix is None else bool(stage1_trainable_replay_prefix)
    stage2_replay_trainable = (
        bool(trainable_replay_prefix)
        if stage2_trainable_replay_prefix is None
        else bool(stage2_trainable_replay_prefix)
    )

    shared = {
        "train_path": train_path,
        "skills_path": skills_path,
        "routing_checkpoint_path": routing_checkpoint_path,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "retrieval_num_negatives": retrieval_num_negatives,
        "retrieval_hard_ratio": retrieval_hard_ratio,
        "transition_hard_negative_margin": transition_hard_negative_margin,
        "include_available_actions_in_state": include_available_actions_in_state,
        "skill_text_format": skill_text_format,
        "embedding_cache_mode": embedding_cache_mode,
        "embedding_cache_max_rows": embedding_cache_max_rows,
        "allowed_benchmarks": allowed_benchmarks,
        "benchmark_caps": benchmark_caps,
        "max_rows": max_rows,
        "stage0_top_m": stage0_top_m,
        "stage0_positive_missing_policy": stage0_positive_missing_policy,
        "stage0_handoff_query_mode": stage0_handoff_query_mode,
        "stage0_handoff_sample_multiplier": stage0_handoff_sample_multiplier,
        "stage0_candidate_encode_batch_size": stage0_candidate_encode_batch_size,
        "stage0_candidate_progress_interval_batches": stage0_candidate_progress_interval_batches,
        "stage0_handoff_cache_mode": stage0_handoff_cache_mode,
        "stage0_handoff_cache_dir": Path(stage0_handoff_cache_dir),
        "transition_inventory_mask_mode": transition_inventory_mask_mode,
        "transition_inventory_min_candidates": transition_inventory_min_candidates,
        "transition_loss_type": transition_loss_type,
        "transition_positive_mode": transition_positive_mode,
        "transition_residual_lambda": transition_residual_lambda,
        "transition_scoring_mode": transition_scoring_mode,
        "route_scorer": route_scorer,
        "gated_temporal_lambda_max": gated_temporal_lambda_max,
        "gated_temporal_kl_alpha": gated_temporal_kl_alpha,
        "gated_temporal_rank_drop_beta": gated_temporal_rank_drop_beta,
        "gated_temporal_context_top_k": gated_temporal_context_top_k,
        "stage0_score_prior_calibration": stage0_score_prior_calibration,
        "sampling_strategy": sampling_strategy,
    }

    stage1_report = run_clstr_stage1_heads_init(
        output_dir=stage1_dir,
        max_steps=stage1_max_steps,
        loss_weights={
            "L_policy": stage1_policy_loss_weight,
            "L_trans": stage1_transition_loss_weight,
            "L_trans_skill_ce": stage1_transition_skill_ce_loss_weight,
            "belief": stage1_belief_loss_weight,
            "STOP": stage1_stop_loss_weight,
            "routing": retrieval_loss_weight,
            "hard_negative_margin": 0.0,
            "Q_success": 0.0,
            "transition_hard_negative_margin": 0.0,
            "counterfactual_utility": 0.0,
        },
        freeze_gated_temporal_only=stage1_freeze,
        auto_replay_prefix_max_steps=stage1_replay_steps,
        trainable_replay_prefix=stage1_replay_trainable,
        **shared,
    )
    stage1_checkpoint = _checkpoint_from_report(
        stage1_report,
        stage1_dir / "checkpoints" / f"clstr_stage1_heads-step{int(stage1_max_steps)}.pt",
    )
    stage1_gate = audit_stage1_heads_quality(
        output_dir=stage1_dir,
        checkpoint_path=stage1_checkpoint,
        output_path=output_dir / "stage1_quality_gate.json",
        min_steps=stage1_max_steps,
        first_window=stage1_gate_first_window,
        last_window=stage1_gate_last_window,
        min_loss_drop=stage1_gate_min_loss_drop,
        min_transition_recall_at_1_tail=stage1_gate_min_transition_recall_at_1_tail,
        min_transition_recall_at_5_tail=stage1_gate_min_transition_recall_at_5_tail,
        expected_transition_scoring_mode=transition_scoring_mode,
        expected_transition_residual_lambda=transition_residual_lambda,
        expected_route_scorer=route_scorer,
    )
    _raise_if_gate_failed("Stage1/2 warmup", stage1_gate)

    stage2_report = run_clstr_full_base_train(
        output_dir=stage2_dir,
        max_steps=stage2_max_steps,
        stage1_checkpoint_path=stage1_checkpoint,
        loss_weights={
            "L_policy": stage2_policy_loss_weight,
            "hard_negative_margin": stage2_policy_hard_negative_margin_loss_weight,
            "Q_success": stage2_q_success_loss_weight,
            "L_trans": stage2_transition_loss_weight,
            "L_trans_skill_ce": stage2_transition_skill_ce_loss_weight,
            "transition_hard_negative_margin": stage2_transition_hard_negative_margin_loss_weight,
            "counterfactual_utility": stage2_counterfactual_utility_loss_weight,
            "belief": stage2_belief_loss_weight,
            "STOP": stage2_stop_loss_weight,
            "routing": retrieval_loss_weight,
        },
        freeze_gated_temporal_only=stage2_freeze,
        auto_replay_prefix_max_steps=stage2_replay_steps,
        trainable_replay_prefix=stage2_replay_trainable,
        next_skill_pool_mode=next_skill_pool_mode,
        counterfactual_gain_margin=counterfactual_gain_margin,
        counterfactual_safety_tolerance=counterfactual_safety_tolerance,
        counterfactual_gain_weight=counterfactual_gain_weight,
        counterfactual_safety_weight=counterfactual_safety_weight,
        counterfactual_warmup_fraction=counterfactual_warmup_fraction,
        **shared,
    )
    stage2_checkpoint = _checkpoint_from_report(
        stage2_report,
        stage2_dir / "checkpoints" / f"clstr_full_base-step{int(stage2_max_steps)}.pt",
    )
    stage2_gate = audit_stage2_full_base_quality(
        output_dir=stage2_dir,
        checkpoint_path=stage2_checkpoint,
        output_path=output_dir / "stage2_quality_gate.json",
        min_steps=stage2_max_steps,
        first_window=stage2_gate_first_window,
        last_window=stage2_gate_last_window,
        min_loss_drop=stage2_gate_min_loss_drop,
        min_transition_recall_at_5_tail=stage2_gate_min_transition_recall_at_5_tail,
        max_stage2_recall_at_5_drop_vs_stage0_prior=stage2_gate_max_recall_at_5_drop_vs_stage0_prior,
        max_stage2_mrr_drop_vs_stage0_prior=stage2_gate_max_mrr_drop_vs_stage0_prior,
        max_stage2_worse_than_stage0_prior_fraction=stage2_gate_max_worse_than_stage0_prior_fraction,
        expected_transition_scoring_mode=transition_scoring_mode,
        expected_transition_residual_lambda=transition_residual_lambda,
        expected_route_scorer=route_scorer,
        expected_next_skill_pool_mode=next_skill_pool_mode,
    )
    _raise_if_gate_failed("Stage1/2 main", stage2_gate)

    report = {
        "status": "ok",
        "stage": "clstr_stage12_supervised_heads",
        "training_objective": "consolidated_stage1_warmup_plus_stage2_main_supervised_heads",
        "checkpoint": str(stage2_checkpoint),
        "stage1": {
            "output_dir": str(stage1_dir),
            "checkpoint": str(stage1_checkpoint),
            "report": stage1_report,
            "gate": stage1_gate,
        },
        "stage2": {
            "output_dir": str(stage2_dir),
            "checkpoint": str(stage2_checkpoint),
            "report": stage2_report,
            "gate": stage2_gate,
        },
        "routing_checkpoint_path": str(routing_checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "config": {
            "stage0_top_m": int(stage0_top_m),
            "transition_inventory_mask_mode": transition_inventory_mask_mode,
            "transition_inventory_min_candidates": int(transition_inventory_min_candidates),
            "transition_loss_type": transition_loss_type,
            "transition_positive_mode": transition_positive_mode,
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": transition_scoring_mode,
            "route_scorer": route_scorer,
            "next_skill_pool_mode": next_skill_pool_mode,
            "stage2_counterfactual_utility_loss_weight": float(stage2_counterfactual_utility_loss_weight),
            "counterfactual_gain_margin": float(counterfactual_gain_margin),
            "counterfactual_safety_tolerance": float(counterfactual_safety_tolerance),
            "counterfactual_gain_weight": float(counterfactual_gain_weight),
            "counterfactual_safety_weight": float(counterfactual_safety_weight),
            "counterfactual_warmup_fraction": counterfactual_warmup_fraction,
            "gated_temporal_lambda_max": float(gated_temporal_lambda_max),
            "gated_temporal_kl_alpha": float(gated_temporal_kl_alpha),
            "gated_temporal_rank_drop_beta": float(gated_temporal_rank_drop_beta),
            "gated_temporal_context_top_k": int(gated_temporal_context_top_k),
            "stage0_score_prior_calibration": stage0_score_prior_calibration,
            "freeze_gated_temporal_only": bool(freeze_gated_temporal_only),
            "stage1_freeze_gated_temporal_only": bool(stage1_freeze),
            "stage2_freeze_gated_temporal_only": bool(stage2_freeze),
            "sampling_strategy": sampling_strategy,
            "stage0_handoff_cache_mode": stage0_handoff_cache_mode,
            "stage0_handoff_cache_dir": str(stage0_handoff_cache_dir),
            "stage1_max_steps": int(stage1_max_steps),
            "stage2_max_steps": int(stage2_max_steps),
            "auto_replay_prefix_max_steps": int(auto_replay_prefix_max_steps),
            "trainable_replay_prefix": bool(trainable_replay_prefix),
            "stage1_auto_replay_prefix_max_steps": int(stage1_replay_steps),
            "stage2_auto_replay_prefix_max_steps": int(stage2_replay_steps),
            "stage1_trainable_replay_prefix": bool(stage1_replay_trainable),
            "stage2_trainable_replay_prefix": bool(stage2_replay_trainable),
        },
    }
    _write_json(output_dir / "consolidated_train_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run consolidated Stage1/2 supervised CLSTR-head training.")
    parser.add_argument("--train_path", default=str(FUNCTION_AUG_V2_DATA_ROOT / "trajectories.jsonl"))
    parser.add_argument("--skills_path", default=str(FUNCTION_AUG_V2_DATA_ROOT / "skill_pool.jsonl"))
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage12_function_aug_v2_consolidated")
    parser.add_argument("--routing_checkpoint_path", required=True)
    parser.add_argument("--stage1_max_steps", type=int, default=3000)
    parser.add_argument("--stage2_max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--allowed_benchmarks", default="toolbench_g3,traject_bench,alfworld,webshop")
    parser.add_argument("--benchmark_caps", default="toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1")
    parser.add_argument("--stage0_top_m", type=int, default=500)
    parser.add_argument("--stage0_positive_missing_policy", default="skip", choices=["skip", "inject"])
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_handoff_sample_multiplier", type=float, default=4.0)
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=50)
    parser.add_argument("--stage0_handoff_cache_mode", default="auto", choices=["auto", "off", "refresh"])
    parser.add_argument("--stage0_handoff_cache_dir", default="outputs/cache/stage0_handoff")
    parser.add_argument("--transition_inventory_mask_mode", default="explicit_only", choices=sorted(TRANSITION_INVENTORY_MASK_MODES))
    parser.add_argument("--transition_inventory_min_candidates", type=int, default=50)
    parser.add_argument("--transition_loss_type", default="listwise_nll", choices=sorted(TRANSITION_LOSS_TYPES))
    parser.add_argument("--transition_positive_mode", default="gold_plus_equivalent", choices=sorted(TRANSITION_POSITIVE_MODES))
    parser.add_argument("--transition_residual_lambda", type=float, default=DEFAULT_TRANSITION_RESIDUAL_LAMBDA)
    parser.add_argument("--transition_scoring_mode", default=TRANSITION_SCORING_MODE, choices=sorted(TRANSITION_SCORING_MODES))
    parser.add_argument("--route_scorer", default=UNIFIED_MEMORY_ROUTE_SCORER, choices=sorted(ROUTE_SCORERS))
    parser.add_argument("--next_skill_pool_mode", default="full_pool", choices=sorted(NEXT_SKILL_POOL_MODES))
    parser.add_argument("--gated_temporal_lambda_max", type=float, default=DEFAULT_GATED_TEMPORAL_LAMBDA_MAX)
    parser.add_argument("--gated_temporal_kl_alpha", type=float, default=DEFAULT_GATED_TEMPORAL_KL_ALPHA)
    parser.add_argument("--gated_temporal_rank_drop_beta", type=float, default=DEFAULT_GATED_TEMPORAL_RANK_DROP_BETA)
    parser.add_argument("--gated_temporal_context_top_k", type=int, default=DEFAULT_GATED_TEMPORAL_CONTEXT_TOP_K)
    parser.add_argument("--stage0_score_prior_calibration", default="off", choices=["off", "rank", "rank_std", "raw", "none"])
    parser.add_argument("--freeze_gated_temporal_only", action="store_true")
    parser.add_argument("--stage1_freeze_gated_temporal_only", default="auto")
    parser.add_argument("--stage2_freeze_gated_temporal_only", default="auto")
    parser.add_argument("--transition_hard_negative_margin", type=float, default=1.0)
    parser.add_argument("--counterfactual_gain_margin", type=float, default=0.1)
    parser.add_argument("--counterfactual_safety_tolerance", type=float, default=0.01)
    parser.add_argument("--counterfactual_gain_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_safety_weight", type=float, default=1.0)
    parser.add_argument("--counterfactual_warmup_fraction", type=float, default=0.1)
    parser.add_argument("--sampling_strategy", default="benchmark_transition_quota_random", choices=sorted(SAMPLING_STRATEGIES))
    parser.add_argument("--embedding_cache_mode", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--embedding_cache_max_rows", type=int, default=20000)
    parser.add_argument("--include_available_actions_in_state", action="store_true")
    parser.add_argument("--skill_text_format", default=None)
    parser.add_argument("--stage1_policy_loss_weight", type=float, default=0.6)
    parser.add_argument("--stage1_transition_loss_weight", type=float, default=0.2)
    parser.add_argument("--stage1_transition_skill_ce_loss_weight", type=float, default=0.6)
    parser.add_argument("--stage1_belief_loss_weight", type=float, default=0.1)
    parser.add_argument("--stage1_stop_loss_weight", type=float, default=0.1)
    parser.add_argument("--stage2_policy_loss_weight", type=float, default=0.2)
    parser.add_argument("--stage2_policy_hard_negative_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--stage2_q_success_loss_weight", type=float, default=0.0)
    parser.add_argument("--stage2_transition_loss_weight", type=float, default=0.3)
    parser.add_argument("--stage2_transition_skill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--stage2_transition_hard_negative_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--stage2_counterfactual_utility_loss_weight", type=float, default=0.05)
    parser.add_argument("--stage2_belief_loss_weight", type=float, default=0.1)
    parser.add_argument("--stage2_stop_loss_weight", type=float, default=0.1)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.0)
    parser.add_argument("--retrieval_num_negatives", type=int, default=32)
    parser.add_argument("--retrieval_hard_ratio", type=float, default=0.5)
    parser.add_argument("--stage1_gate_first_window", type=int, default=100)
    parser.add_argument("--stage1_gate_last_window", type=int, default=100)
    parser.add_argument("--stage1_gate_min_loss_drop", type=float, default=0.0)
    parser.add_argument("--stage1_gate_min_transition_recall_at_1_tail", type=float, default=0.35)
    parser.add_argument("--stage1_gate_min_transition_recall_at_5_tail", type=float, default=0.7)
    parser.add_argument("--stage2_gate_first_window", type=int, default=200)
    parser.add_argument("--stage2_gate_last_window", type=int, default=200)
    parser.add_argument("--stage2_gate_min_loss_drop", type=float, default=0.02)
    parser.add_argument("--stage2_gate_min_transition_recall_at_5_tail", type=float, default=0.5)
    parser.add_argument("--stage2_gate_max_recall_at_5_drop_vs_stage0_prior", type=float, default=0.03)
    parser.add_argument("--stage2_gate_max_mrr_drop_vs_stage0_prior", type=float, default=0.01)
    parser.add_argument("--stage2_gate_max_worse_than_stage0_prior_fraction", type=float, default=0.5)
    parser.add_argument("--auto_replay_prefix_max_steps", type=int, default=0)
    parser.add_argument("--trainable_replay_prefix", action="store_true")
    parser.add_argument("--stage1_auto_replay_prefix_max_steps", type=int, default=None)
    parser.add_argument("--stage2_auto_replay_prefix_max_steps", type=int, default=None)
    parser.add_argument("--stage1_trainable_replay_prefix", default="auto")
    parser.add_argument("--stage2_trainable_replay_prefix", default="auto")
    args = parser.parse_args()

    report = run_stage12_consolidated_train(
        train_path=Path(args.train_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        stage1_max_steps=args.stage1_max_steps,
        stage2_max_steps=args.stage2_max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
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
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_inventory_min_candidates=args.transition_inventory_min_candidates,
        transition_loss_type=args.transition_loss_type,
        transition_positive_mode=args.transition_positive_mode,
        transition_residual_lambda=args.transition_residual_lambda,
        transition_scoring_mode=args.transition_scoring_mode,
        route_scorer=args.route_scorer,
        next_skill_pool_mode=args.next_skill_pool_mode,
        gated_temporal_lambda_max=args.gated_temporal_lambda_max,
        gated_temporal_kl_alpha=args.gated_temporal_kl_alpha,
        gated_temporal_rank_drop_beta=args.gated_temporal_rank_drop_beta,
        gated_temporal_context_top_k=args.gated_temporal_context_top_k,
        stage0_score_prior_calibration=args.stage0_score_prior_calibration,
        freeze_gated_temporal_only=args.freeze_gated_temporal_only,
        stage1_freeze_gated_temporal_only=_parse_optional_bool(args.stage1_freeze_gated_temporal_only),
        stage2_freeze_gated_temporal_only=_parse_optional_bool(args.stage2_freeze_gated_temporal_only),
        transition_hard_negative_margin=args.transition_hard_negative_margin,
        counterfactual_gain_margin=args.counterfactual_gain_margin,
        counterfactual_safety_tolerance=args.counterfactual_safety_tolerance,
        counterfactual_gain_weight=args.counterfactual_gain_weight,
        counterfactual_safety_weight=args.counterfactual_safety_weight,
        counterfactual_warmup_fraction=args.counterfactual_warmup_fraction,
        sampling_strategy=args.sampling_strategy,
        embedding_cache_mode=args.embedding_cache_mode,
        embedding_cache_max_rows=args.embedding_cache_max_rows,
        include_available_actions_in_state=args.include_available_actions_in_state,
        skill_text_format=args.skill_text_format,
        stage1_policy_loss_weight=args.stage1_policy_loss_weight,
        stage1_transition_loss_weight=args.stage1_transition_loss_weight,
        stage1_transition_skill_ce_loss_weight=args.stage1_transition_skill_ce_loss_weight,
        stage1_belief_loss_weight=args.stage1_belief_loss_weight,
        stage1_stop_loss_weight=args.stage1_stop_loss_weight,
        stage2_policy_loss_weight=args.stage2_policy_loss_weight,
        stage2_policy_hard_negative_margin_loss_weight=args.stage2_policy_hard_negative_margin_loss_weight,
        stage2_q_success_loss_weight=args.stage2_q_success_loss_weight,
        stage2_transition_loss_weight=args.stage2_transition_loss_weight,
        stage2_transition_skill_ce_loss_weight=args.stage2_transition_skill_ce_loss_weight,
        stage2_transition_hard_negative_margin_loss_weight=args.stage2_transition_hard_negative_margin_loss_weight,
        stage2_counterfactual_utility_loss_weight=args.stage2_counterfactual_utility_loss_weight,
        stage2_belief_loss_weight=args.stage2_belief_loss_weight,
        stage2_stop_loss_weight=args.stage2_stop_loss_weight,
        retrieval_loss_weight=args.retrieval_loss_weight,
        retrieval_num_negatives=args.retrieval_num_negatives,
        retrieval_hard_ratio=args.retrieval_hard_ratio,
        stage1_gate_first_window=args.stage1_gate_first_window,
        stage1_gate_last_window=args.stage1_gate_last_window,
        stage1_gate_min_loss_drop=args.stage1_gate_min_loss_drop,
        stage1_gate_min_transition_recall_at_1_tail=args.stage1_gate_min_transition_recall_at_1_tail,
        stage1_gate_min_transition_recall_at_5_tail=args.stage1_gate_min_transition_recall_at_5_tail,
        stage2_gate_first_window=args.stage2_gate_first_window,
        stage2_gate_last_window=args.stage2_gate_last_window,
        stage2_gate_min_loss_drop=args.stage2_gate_min_loss_drop,
        stage2_gate_min_transition_recall_at_5_tail=args.stage2_gate_min_transition_recall_at_5_tail,
        stage2_gate_max_recall_at_5_drop_vs_stage0_prior=args.stage2_gate_max_recall_at_5_drop_vs_stage0_prior,
        stage2_gate_max_mrr_drop_vs_stage0_prior=args.stage2_gate_max_mrr_drop_vs_stage0_prior,
        stage2_gate_max_worse_than_stage0_prior_fraction=args.stage2_gate_max_worse_than_stage0_prior_fraction,
        auto_replay_prefix_max_steps=args.auto_replay_prefix_max_steps,
        trainable_replay_prefix=args.trainable_replay_prefix,
        stage1_auto_replay_prefix_max_steps=args.stage1_auto_replay_prefix_max_steps,
        stage2_auto_replay_prefix_max_steps=args.stage2_auto_replay_prefix_max_steps,
        stage1_trainable_replay_prefix=_parse_optional_bool(args.stage1_trainable_replay_prefix),
        stage2_trainable_replay_prefix=_parse_optional_bool(args.stage2_trainable_replay_prefix),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
