#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_compressor_train import train_vnext_candidate_compressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the memory-independent CLSTR vNext static route adapter"
    )
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--retrieval_rows_path")
    parser.add_argument("--retrieval_dev_rows_path")
    parser.add_argument("--trajectory_rows_path")
    parser.add_argument("--trajectory_dev_rows_path")
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--data_contract_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--compressed_m", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--frozen_cache_dir")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--checkpoint_interval", type=int, default=300)
    parser.add_argument("--validation_interval", type=int, default=300)
    parser.add_argument("--validation_batch_size", type=int, default=16)
    parser.add_argument("--max_dev_rows", type=int, default=4096)
    parser.add_argument("--max_train_rows", type=int)
    parser.add_argument("--minimum_dev_score_gain", type=float, default=0.0)
    parser.add_argument("--coverage_margin", type=float, default=0.05)
    parser.add_argument("--recoverable_row_weight", type=float, default=4.0)
    parser.add_argument("--listwise_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--objective_mode",
        choices=(
            "topk_boundary",
            "static_top500_rerank",
            "static_route_query_residual",
        ),
        default="topk_boundary",
    )
    parser.add_argument("--static_reranker_scale_initial", type=float, default=1.0)
    parser.add_argument("--require_clean_source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_vnext_candidate_compressor(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        skills_path=args.skills_path,
        retrieval_rows_path=args.retrieval_rows_path,
        retrieval_dev_rows_path=args.retrieval_dev_rows_path,
        trajectory_rows_path=args.trajectory_rows_path,
        trajectory_dev_rows_path=args.trajectory_dev_rows_path,
        inventory_catalogs_path=args.inventory_catalogs_path,
        data_contract_path=args.data_contract_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        belief_top_k=args.belief_top_k,
        coarse_k=args.coarse_k,
        compressed_m=args.compressed_m,
        cache_batch_size=args.cache_batch_size,
        cache_shard_size=args.cache_shard_size,
        frozen_cache_dir=args.frozen_cache_dir,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        validation_batch_size=args.validation_batch_size,
        max_dev_rows=args.max_dev_rows,
        max_train_rows=args.max_train_rows,
        minimum_dev_score_gain=args.minimum_dev_score_gain,
        coverage_margin=args.coverage_margin,
        recoverable_row_weight=args.recoverable_row_weight,
        listwise_loss_weight=args.listwise_loss_weight,
        objective_mode=args.objective_mode,
        static_reranker_scale_initial=args.static_reranker_scale_initial,
        require_clean_source=args.require_clean_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
