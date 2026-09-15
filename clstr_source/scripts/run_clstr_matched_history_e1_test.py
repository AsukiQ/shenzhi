#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_test import evaluate_matched_history_e1_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_kind",
        choices=("serialized", "gru", "transformer", "lstr"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--foundation_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--test_rows_path", required=True)
    parser.add_argument("--test_data_report_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_report_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frozen_cache_dir", required=True)
    parser.add_argument("--validation_batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    args = parser.parse_args()
    report = evaluate_matched_history_e1_test(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
