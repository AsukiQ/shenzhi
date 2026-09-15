#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_mt_fusion_train import train_appworld_mt_fusion_from_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AppWorld CLSTR mt-fusion policy on train-only multi-step trajectories.")
    parser.add_argument("--model_config", default="configs/model/appworld_skillrouter_init_mt_fusion.yaml")
    parser.add_argument(
        "--checkpoint_path",
        default="outputs/appworld_clstr_train_skillrouter_init_routing_full_v1/checkpoints/stage2_base-step500.pt",
    )
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--trajectories_path", default="data/appworld_multistep/oracle_train_trajectories.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_mt_fusion_train/oracle_v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--candidate_top_k", type=int, default=16)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--trajectory_batch_size", type=int, default=8)
    parser.add_argument("--detach_belief_between_steps", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    report = train_appworld_mt_fusion_from_trajectories(
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint_path,
        skill_pool_path=args.skill_pool_path,
        trajectories_path=args.trajectories_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        candidate_top_k=args.candidate_top_k,
        max_trajectories=args.max_trajectories,
        trajectory_batch_size=args.trajectory_batch_size,
        detach_belief_between_steps=bool(args.detach_belief_between_steps),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
