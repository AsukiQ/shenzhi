#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.aux_pretrain import run_aux_trajectory_pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auxiliary trajectory transition/belief/STOP pretraining.")
    parser.add_argument("--data_root", default="data/aux_trajectories")
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--output_dir", default="outputs/aux_trajectory_pretrain")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_trajectories", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include_splits", nargs="*", default=None)
    parser.add_argument("--exclude_splits", nargs="*", default=None)
    parser.add_argument("--routing_init_manifest", default=None)
    parser.add_argument("--eval_splits", nargs="*", default=None)
    parser.add_argument("--max_eval_steps", type=int, default=2048)
    parser.add_argument("--freeze_routing_foundation", action="store_true")
    parser.add_argument("--use_universal_action_adapter", action="store_true")
    parser.add_argument("--action_negative_k", type=int, default=15)
    parser.add_argument("--candidate_strategy", default="environment_action_pool")
    parser.add_argument(
        "--transition_objective",
        choices=["observation_cosine", "next_action_ce"],
        default="observation_cosine",
    )
    parser.add_argument(
        "--run_role",
        choices=["debug_gate_not_final", "auxiliary_pretrain_full_candidate"],
        default=None,
    )
    args = parser.parse_args()

    report = run_aux_trajectory_pretrain(
        data_root=Path(args.data_root),
        base_model_name=args.base_model_name,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        top_k=args.top_k,
        max_trajectories=args.max_trajectories,
        learning_rate=args.learning_rate,
        seed=args.seed,
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
        routing_init_manifest=Path(args.routing_init_manifest) if args.routing_init_manifest else None,
        eval_splits=args.eval_splits,
        max_eval_steps=args.max_eval_steps,
        freeze_routing_foundation=args.freeze_routing_foundation,
        use_universal_action_adapter=args.use_universal_action_adapter,
        action_negative_k=args.action_negative_k,
        candidate_strategy=args.candidate_strategy,
        transition_objective=args.transition_objective,
        run_role=args.run_role,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
