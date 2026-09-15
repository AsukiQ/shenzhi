#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.state_probe_run import run_state_probe_e2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--aligned_rows_path", required=True)
    parser.add_argument("--serialized_checkpoint_path", required=True)
    parser.add_argument("--gru_checkpoint_path", required=True)
    parser.add_argument("--transformer_checkpoint_path", required=True)
    parser.add_argument("--lstr_checkpoint_path", required=True)
    parser.add_argument("--expected_e1_seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frozen_cache_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--probe_max_iter", type=int, default=100)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=29)
    args = parser.parse_args()
    run_state_probe_e2(
        foundation_checkpoint_path=args.foundation_checkpoint_path,
        skills_path=args.skills_path,
        aligned_rows_path=args.aligned_rows_path,
        serialized_checkpoint_path=args.serialized_checkpoint_path,
        gru_checkpoint_path=args.gru_checkpoint_path,
        transformer_checkpoint_path=args.transformer_checkpoint_path,
        lstr_checkpoint_path=args.lstr_checkpoint_path,
        expected_e1_seed=args.expected_e1_seed,
        output_dir=args.output_dir,
        frozen_cache_dir=args.frozen_cache_dir,
        batch_size=args.batch_size,
        cache_batch_size=args.cache_batch_size,
        cache_shard_size=args.cache_shard_size,
        probe_max_iter=args.probe_max_iter,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
