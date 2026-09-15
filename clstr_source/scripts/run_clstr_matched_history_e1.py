#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_train import train_matched_history_e1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_kind", choices=("serialized", "gru", "transformer", "lstr"), required=True)
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--train_rows_path", required=True)
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frozen_cache_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--validation_batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--max_train_rows_per_benchmark", type=int)
    parser.add_argument("--max_dev_rows_per_benchmark", type=int)
    args = parser.parse_args()
    report = train_matched_history_e1(
        encoder_kind=args.encoder_kind,
        foundation_checkpoint_path=args.foundation_checkpoint_path,
        skills_path=args.skills_path,
        train_rows_path=args.train_rows_path,
        dev_rows_path=args.dev_rows_path,
        inventory_catalogs_path=args.inventory_catalogs_path,
        output_dir=args.output_dir,
        frozen_cache_dir=args.frozen_cache_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        validation_interval=args.validation_interval,
        validation_batch_size=args.validation_batch_size,
        cache_batch_size=args.cache_batch_size,
        cache_shard_size=args.cache_shard_size,
        max_train_rows_per_benchmark=args.max_train_rows_per_benchmark,
        max_dev_rows_per_benchmark=args.max_dev_rows_per_benchmark,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
