#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import run_legacy_clstr_full_base_train_from_routing_init


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run legacy CLSTR component-complete full-base masked multi-loss "
            "training from a routing_init_manifest. Unified Stage2 uses "
            "scripts/run_clstr_stage2_full_base_train.py instead."
        )
    )
    parser.add_argument("--train_path", default="data/clstr_full_base_train/train.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_full_base_train/skills.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_full_base_train")
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--transition_loss_weight", type=float, default=0.1)
    parser.add_argument("--transition_skill_ce_loss_weight", type=float, default=0.2)
    parser.add_argument("--belief_loss_weight", type=float, default=0.1)
    parser.add_argument("--stop_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_num_negatives", type=int, default=32)
    parser.add_argument("--retrieval_hard_ratio", type=float, default=0.5)
    parser.add_argument("--skill_text_format", default=None)
    args = parser.parse_args()

    report = run_legacy_clstr_full_base_train_from_routing_init(
        train_path=Path(args.train_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
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
        retrieval_num_negatives=args.retrieval_num_negatives,
        retrieval_hard_ratio=args.retrieval_hard_ratio,
        skill_text_format=args.skill_text_format,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
