#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.structured_train import run_structured_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLSTR structured controller on verified Qwen teacher rollout.")
    parser.add_argument("--rollout_path", default="data/alfworld_qwen3_verified_rollout/train_rollout.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_full_base_train/skills.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_qwen3_structured_train")
    parser.add_argument("--data_output_dir", default="data/clstr_qwen3_structured_train")
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--transition_loss_weight", type=float, default=0.02)
    parser.add_argument("--transition_skill_ce_loss_weight", type=float, default=0.2)
    parser.add_argument("--belief_loss_weight", type=float, default=0.05)
    parser.add_argument("--stop_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.2)
    args = parser.parse_args()

    report = run_structured_train(
        rollout_path=Path(args.rollout_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        data_output_dir=Path(args.data_output_dir),
        routing_init_manifest=Path(args.routing_init_manifest),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        loss_weights={
            "L_policy": args.policy_loss_weight,
            "L_trans": args.transition_loss_weight,
            "L_trans_skill_ce": args.transition_skill_ce_loss_weight,
            "belief": args.belief_loss_weight,
            "STOP": args.stop_loss_weight,
            "routing": args.retrieval_loss_weight,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
