#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_stage0_train import train_vnext_stage0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical CLSTR vNext Stage0")
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--retrieval_rows_path", required=True)
    parser.add_argument("--retrieval_dev_rows_path", required=True)
    parser.add_argument("--static_route_rows_path", required=True)
    parser.add_argument("--static_route_dev_rows_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--data_contract_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--skill_table_batch_size", type=int, default=32)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--cache_shard_size", type=int, default=4096)
    parser.add_argument("--frozen_cache_dir")
    parser.add_argument("--backbone_snapshot_path")
    parser.add_argument("--skill_cache_shard_size", type=int, default=2048)
    parser.add_argument("--hard_negative_loss_weight", type=float, default=0.2)
    parser.add_argument("--hard_negative_margin", type=float, default=0.1)
    parser.add_argument("--hard_negative_top_k", type=int, default=32)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_retrieval_rows", type=int)
    parser.add_argument("--max_static_rows", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint_interval", type=int, default=400)
    parser.add_argument("--validation_interval", type=int, default=400)
    parser.add_argument("--validation_batch_size", type=int, default=128)
    parser.add_argument("--max_dev_rows_per_kind", type=int, default=4096)
    parser.add_argument("--minimum_dev_score_gain", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint_path")
    parser.add_argument("--legacy_init_checkpoint_path")
    parser.add_argument("--legacy_init_skills_path")
    parser.add_argument("--require_clean_source", action="store_true")
    parser.add_argument("--allow_remote_model_files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_vnext_stage0(
        skills_path=args.skills_path,
        retrieval_rows_path=args.retrieval_rows_path,
        retrieval_dev_rows_path=args.retrieval_dev_rows_path,
        static_route_rows_path=args.static_route_rows_path,
        static_route_dev_rows_path=args.static_route_dev_rows_path,
        inventory_catalogs_path=args.inventory_catalogs_path,
        data_contract_path=args.data_contract_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        model_dim=args.model_dim,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        skill_table_batch_size=args.skill_table_batch_size,
        belief_top_k=args.belief_top_k,
        cache_batch_size=args.cache_batch_size,
        cache_shard_size=args.cache_shard_size,
        frozen_cache_dir=args.frozen_cache_dir,
        backbone_snapshot_path=args.backbone_snapshot_path,
        skill_cache_shard_size=args.skill_cache_shard_size,
        hard_negative_loss_weight=args.hard_negative_loss_weight,
        hard_negative_margin=args.hard_negative_margin,
        hard_negative_top_k=args.hard_negative_top_k,
        max_skills=args.max_skills,
        max_retrieval_rows=args.max_retrieval_rows,
        max_static_rows=args.max_static_rows,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        validation_batch_size=args.validation_batch_size,
        max_dev_rows_per_kind=args.max_dev_rows_per_kind,
        minimum_dev_score_gain=args.minimum_dev_score_gain,
        resume_checkpoint_path=args.resume_checkpoint_path,
        legacy_init_checkpoint_path=args.legacy_init_checkpoint_path,
        legacy_init_skills_path=args.legacy_init_skills_path,
        local_files_only=not args.allow_remote_model_files,
        require_clean_source=args.require_clean_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
