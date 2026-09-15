#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_policy_train import run_alfworld_policy_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLSTR ALFWorld goal-conditioned L_policy head.")
    parser.add_argument("--replay_path", default="data/alfworld_policy_replay/train_replay.jsonl")
    parser.add_argument("--output_dir", default="outputs/alfworld_policy_train_gate")
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--aux_data_root", default="data/aux_trajectories")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval_fraction", type=float, default=0.05)
    parser.add_argument("--stop_loss_weight", type=float, default=0.05)
    parser.add_argument("--history_window", type=int, default=6)
    parser.add_argument("--feature_cache_encode_batch_size", type=int, default=128)
    args = parser.parse_args()

    report = run_alfworld_policy_train(
        replay_path=Path(args.replay_path),
        output_dir=Path(args.output_dir),
        routing_init_manifest=Path(args.routing_init_manifest),
        aux_data_root=Path(args.aux_data_root),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        stop_loss_weight=args.stop_loss_weight,
        history_window=args.history_window,
        feature_cache_encode_batch_size=args.feature_cache_encode_batch_size,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
