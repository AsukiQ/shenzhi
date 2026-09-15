#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.external_data import write_json
from clstr.full_base_train import (
    SAMPLING_STRATEGIES,
    TRANSITION_INVENTORY_MASK_MODES,
    TRANSITION_SCORING_MODE,
    TRANSITION_SCORING_MODES,
    _apply_skill_text_format,
    _attach_full_base_embedding_cache,
    _attach_policy_embedding_cache,
    _attach_stage0_topm_candidates,
    _cap_rows_by_benchmark,
    _filter_rows_by_allowed_benchmarks,
    _filter_rows_by_train_split,
    _limit_rows_for_smoke,
    _read_jsonl,
    _select_stage0_handoff_training_subset,
)
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
from clstr.transition_lambda_sweep import evaluate_transition_lambda_sweep


def _parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()} or None


def _parse_caps(value: str | None) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    output: dict[str, int] = {}
    for raw_item in str(value).replace(":", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"benchmark cap must use name=value format: {item}")
        key, raw_value = item.split("=", 1)
        output[key.strip()] = int(raw_value)
    return output or None


def _parse_lambdas(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one lambda is required")
    return values


def _stage1_loss_weights() -> dict[str, float]:
    return {
        "L_policy": 0.7,
        "hard_negative_margin": 0.0,
        "Q_success": 0.0,
        "routing": 0.0,
        "STOP": 0.1,
        "L_trans": 0.2,
        "L_trans_skill_ce": 0.5,
        "transition_hard_negative_margin": 0.0,
        "belief": 0.1,
    }


def _prepare_rows_and_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(args.train_path)
    skills = _read_jsonl(args.skills_path)
    rows, train_split_filter = _filter_rows_by_train_split(rows)
    rows, benchmark_filter = _filter_rows_by_allowed_benchmarks(rows, _parse_csv_set(args.allowed_benchmarks))
    rows, benchmark_caps = _cap_rows_by_benchmark(rows, _parse_caps(args.benchmark_caps))
    rows, max_rows_filter = _limit_rows_for_smoke(rows, args.max_rows)
    if not rows:
        raise ValueError("no rows remain after filters")
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skills) if row.get("skill_id")}
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=args.stage0_checkpoint_path,
        skills_path=args.skills_path,
        model_cache_dir=Path(args.output_dir) / "model_cache",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    _apply_skill_text_format(model, model_config, args.skill_text_format)
    head_report = load_head_checkpoint_into_model(
        model,
        head_checkpoint_path=args.stage1_checkpoint_path,
        partial_load_mode="stage1_lambda_sweep_checkpoint_compatible_state",
    )
    rows, subset_report = _select_stage0_handoff_training_subset(
        rows,
        max_steps=args.max_eval_batches,
        batch_size=args.batch_size,
        loss_weights=_stage1_loss_weights(),
        sample_multiplier=args.stage0_handoff_sample_multiplier,
        sampling_strategy=args.sampling_strategy,
        sampler_seed=args.seed,
    )
    rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=args.stage0_top_m,
        positive_missing_policy="skip",
        query_mode=args.stage0_handoff_query_mode,
        routing_checkpoint_path=args.stage0_checkpoint_path,
        manifest_path=Path(args.output_dir) / "stage0_candidate_handoff.json",
        encode_batch_size=args.stage0_candidate_encode_batch_size,
        device=device,
        progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        inventory_min_candidates=args.transition_inventory_min_candidates,
    )
    if not rows:
        raise ValueError(f"no rows remain after Stage0 handoff: {handoff_report}")
    if args.embedding_cache_mode != "never":
        full_cache = _attach_full_base_embedding_cache(model, rows, encode_batch_size=args.embedding_cache_encode_batch_size)
        policy_cache = _attach_policy_embedding_cache(model, rows, encode_batch_size=args.embedding_cache_encode_batch_size)
    else:
        full_cache = {"used": False, "reason": "disabled_by_embedding_cache_mode"}
        policy_cache = {"used": False, "reason": "disabled_by_embedding_cache_mode"}
    setup = {
        "train_split_filter": train_split_filter,
        "benchmark_filter": benchmark_filter,
        "benchmark_caps": benchmark_caps,
        "max_rows_filter": max_rows_filter,
        "routing_report": routing_report,
        "stage1_head_load": head_report,
        "stage0_handoff_subset": subset_report,
        "stage0_candidate_handoff": handoff_report,
        "full_base_embedding_cache": full_cache,
        "policy_embedding_cache": policy_cache,
    }
    return model, model_config, rows, skills, setup


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline diagnostic sweep for CLSTR transition prior/residual lambda.")
    parser.add_argument("--train_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage1_checkpoint_path",
        default="outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init/checkpoints/clstr_stage1_heads-step3000.pt",
    )
    parser.add_argument("--output_dir", default="outputs/clstr_transition_lambda_sweep/stage1_inventory100_l025")
    parser.add_argument("--lambdas", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=512)
    parser.add_argument("--max_eval_batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allowed_benchmarks")
    parser.add_argument("--benchmark_caps", default="toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1")
    parser.add_argument("--sampling_strategy", default="balanced_random", choices=sorted(SAMPLING_STRATEGIES))
    parser.add_argument("--stage0_top_m", type=int, default=350)
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_handoff_sample_multiplier", type=float, default=None)
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=50)
    parser.add_argument("--transition_inventory_mask_mode", default="auto", choices=sorted(TRANSITION_INVENTORY_MASK_MODES))
    parser.add_argument("--transition_inventory_min_candidates", type=int, default=64)
    parser.add_argument("--transition_loss_type", default="listwise_nll", choices=["cross_entropy", "listwise_nll"])
    parser.add_argument("--transition_positive_mode", default="gold_plus_equivalent", choices=["single", "gold_plus_equivalent"])
    parser.add_argument("--transition_scoring_mode", default=TRANSITION_SCORING_MODE, choices=sorted(TRANSITION_SCORING_MODES))
    parser.add_argument("--skill_text_format")
    parser.add_argument("--embedding_cache_mode", default="never", choices=["always", "never"])
    parser.add_argument("--embedding_cache_encode_batch_size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _model_config, rows, skills, setup = _prepare_rows_and_model(args)
    report = evaluate_transition_lambda_sweep(
        model=model,
        rows=rows,
        skills=skills,
        lambdas=_parse_lambdas(args.lambdas),
        batch_size=args.batch_size,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_inventory_min_candidates=args.transition_inventory_min_candidates,
        transition_loss_type=args.transition_loss_type,
        transition_positive_mode=args.transition_positive_mode,
        transition_scoring_mode=args.transition_scoring_mode,
    )
    report = {
        **report,
        "stage0_checkpoint_path": args.stage0_checkpoint_path,
        "stage1_checkpoint_path": args.stage1_checkpoint_path,
        "train_path": args.train_path,
        "skills_path": args.skills_path,
        "setup": setup,
    }
    write_json(output_dir / "lambda_sweep_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
