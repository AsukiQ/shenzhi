#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_current_route_preference_train import train_current_route_preference_with_model
from clstr.appworld_routing import read_jsonl
from scripts.run_appworld_multistep_executor_eval import _load_dynamic_checkpoint_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small current-route Stage4 preference warmup from clean official-executor "
            "rollout samples. This is not an executor benchmark."
        )
    )
    parser.add_argument("--preference_samples_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skill_pool_path", required=True)
    parser.add_argument("--base_skill_pool_path", required=True)
    parser.add_argument("--clstr_checkpoint_path", required=True)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_transition", action="store_true")
    parser.add_argument(
        "--loss_score_mode",
        choices=["policy_head", "transition_blend", "policy_transition_blend"],
        default="policy_head",
    )
    parser.add_argument("--enable_suppress_loss", action="store_true")
    parser.add_argument("--enable_pairwise_correction_loss", action="store_true")
    parser.add_argument("--pairwise_correction_margin", type=float, default=0.5)
    parser.add_argument("--pairwise_correction_weight", type=float, default=1.0)
    parser.add_argument("--policy_blend_alpha", type=float, default=0.5)
    parser.add_argument("--transition_residual_lambda", type=float, default=0.0)
    parser.add_argument("--transition_scoring_mode", default="v4_1b_action_observation")
    parser.add_argument("--learned_component_min_range", type=float, default=0.1)
    parser.add_argument("--learned_component_trust_top_k", type=int, default=80)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    samples = read_jsonl(args.preference_samples_path)
    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]
    model, load_report = _load_dynamic_checkpoint_model(
        checkpoint_path=str(args.clstr_checkpoint_path),
        base_skill_pool_path=str(args.base_skill_pool_path),
        dynamic_skill_pool_path=str(args.skill_pool_path),
    )
    report = train_current_route_preference_with_model(
        model=model,
        samples=samples,
        output_dir=args.output_dir,
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        train_transition=bool(args.train_transition),
        loss_score_mode=str(args.loss_score_mode),
        enable_suppress_loss=bool(args.enable_suppress_loss),
        enable_pairwise_correction_loss=bool(args.enable_pairwise_correction_loss),
        pairwise_correction_margin=float(args.pairwise_correction_margin),
        pairwise_correction_weight=float(args.pairwise_correction_weight),
        policy_blend_alpha=float(args.policy_blend_alpha),
        transition_residual_lambda=float(args.transition_residual_lambda),
        transition_scoring_mode=str(args.transition_scoring_mode),
        learned_component_min_range=float(args.learned_component_min_range),
        learned_component_trust_top_k=args.learned_component_trust_top_k,
        checkpoint_metadata={
            "source_clstr_checkpoint_path": str(args.clstr_checkpoint_path),
            "base_skill_pool_path": str(args.base_skill_pool_path),
            "dynamic_skill_pool_path": str(args.skill_pool_path),
            "source_model_stage": load_report.get("stage"),
            "dynamic_checkpoint_load_mode": load_report.get("dynamic_checkpoint_load_mode"),
        },
    )
    report["model_load_report"] = load_report
    report_path = Path(args.output_dir) / "train_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
