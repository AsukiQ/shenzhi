#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_precompute import precompute_vnext_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute canonical CLSTR vNext assets")
    for name in (
        "skills_path",
        "retrieval_rows_path",
        "retrieval_dev_rows_path",
        "static_route_rows_path",
        "static_route_dev_rows_path",
        "trajectory_rows_path",
        "trajectory_dev_rows_path",
        "causal_pair_support_rows_path",
        "causal_pair_support_dev_rows_path",
        "data_contract_path",
        "output_dir",
        "cache_root",
        "model_name_or_path",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--skill_table_batch_size", type=int, default=32)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--skill_cache_shard_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow_remote_model_files", action="store_true")
    parser.add_argument("--allow_dirty_source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = precompute_vnext_assets(
        **{
            name: getattr(args, name)
            for name in (
                "skills_path",
                "retrieval_rows_path",
                "retrieval_dev_rows_path",
                "static_route_rows_path",
                "static_route_dev_rows_path",
                "trajectory_rows_path",
                "trajectory_dev_rows_path",
                "causal_pair_support_rows_path",
                "causal_pair_support_dev_rows_path",
                "data_contract_path",
                "output_dir",
                "cache_root",
                "model_name_or_path",
                "model_dim",
                "max_length",
                "torch_dtype",
                "skill_table_batch_size",
                "belief_top_k",
                "cache_batch_size",
                "cache_shard_size",
                "skill_cache_shard_size",
                "seed",
            )
        },
        local_files_only=not args.allow_remote_model_files,
        require_clean_source=not args.allow_dirty_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
