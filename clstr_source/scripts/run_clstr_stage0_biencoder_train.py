#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CLSTR Stage0 SkillRouter-compatible bi-encoder coarse retriever."
    )
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final")
    parser.add_argument("--output_dir", default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE")
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split_path")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--skill_table_batch_size", type=int, default=32)
    parser.add_argument("--encoder_pooling", default="last_token")
    parser.add_argument("--cross_encoder_pooling", default="last_token")
    parser.add_argument("--tokenizer_padding_side", default="left")
    parser.add_argument("--query_text_format", default="skillrouter")
    parser.add_argument("--state_query_prompt_version")
    parser.add_argument("--state_query_max_chars", type=int)
    parser.add_argument(
        "--state_query_truncation",
        choices=["none", "head_v1", "head_tail_v1"],
    )
    parser.add_argument("--unfreeze_backbone", action="store_true")
    parser.add_argument("--retrieval_loss_mode", default="multi_positive_nll", choices=["single_positive_ce", "multi_positive_nll"])
    parser.add_argument(
        "--sampling_strategy",
        default="handoff_balanced",
        choices=["batch_stride", "source_balanced", "handoff_balanced", "handoff_tempered"],
    )
    parser.add_argument("--tempered_correction_fraction", type=float, default=0.2)
    parser.add_argument("--expand_alias_positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_skill_embeddings", action="store_true")
    parser.add_argument("--train_skill_bias", action="store_true")
    parser.add_argument("--train_encoder_backbone", action="store_true")
    parser.add_argument("--encoder_backbone_learning_rate", type=float)
    parser.add_argument("--train_encoder_projection", action="store_true")
    parser.add_argument("--freeze_skill_adapter", action="store_true")
    parser.add_argument("--freeze_retrieval_scale", action="store_true")
    parser.add_argument("--init_checkpoint_path")
    parser.add_argument("--init_checkpoint_skills_path")
    parser.add_argument("--resume_checkpoint_path")
    parser.add_argument(
        "--frozen_backbone_cache_mode",
        choices=["off", "schedule"],
        default="off",
    )
    parser.add_argument(
        "--frozen_backbone_cache_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--resume_skill_table_mode",
        choices=["rebuild", "verified_checkpoint"],
        default="rebuild",
    )
    parser.add_argument("--preservation_anchor_weight", type=float, default=0.0)
    parser.add_argument("--explicit_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--explicit_negative_margin", type=float, default=0.1)
    parser.add_argument("--mined_hard_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--mined_hard_negative_margin", type=float, default=0.1)
    parser.add_argument("--mined_hard_negative_top_k", type=int, default=32)
    parser.add_argument(
        "--route_scorer",
        default="legacy_biencoder",
        choices=["legacy_biencoder", "unified_memory"],
    )
    parser.add_argument("--belief_top_k", type=int, default=64)
    args = parser.parse_args()

    from clstr.retrieval_warmup import run_stage0_biencoder_train

    report = run_stage0_biencoder_train(
        data_root=args.data_root,
        base_model_name=args.model_name_or_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        model_dim=args.model_dim,
        top_k=args.top_k,
        max_skills=args.max_skills,
        max_queries=args.max_queries,
        learning_rate=args.learning_rate,
        seed=args.seed,
        split_path=args.split_path,
        train_split=args.train_split,
        torch_dtype=args.torch_dtype,
        freeze_backbone=not (args.unfreeze_backbone or args.train_encoder_backbone),
        max_length=args.max_length,
        skill_table_batch_size=args.skill_table_batch_size,
        encoder_pooling=args.encoder_pooling,
        cross_encoder_pooling=args.cross_encoder_pooling,
        tokenizer_padding_side=args.tokenizer_padding_side,
        query_text_format=args.query_text_format,
        state_query_prompt_version=args.state_query_prompt_version,
        state_query_max_chars=args.state_query_max_chars,
        state_query_truncation=args.state_query_truncation,
        retrieval_loss_mode=args.retrieval_loss_mode,
        sampling_strategy=args.sampling_strategy,
        tempered_correction_fraction=args.tempered_correction_fraction,
        expand_alias_positives=bool(args.expand_alias_positives),
        train_skill_embeddings=args.train_skill_embeddings,
        train_skill_bias=args.train_skill_bias,
        train_encoder_backbone=bool(args.train_encoder_backbone or args.unfreeze_backbone),
        encoder_backbone_learning_rate=args.encoder_backbone_learning_rate,
        train_encoder_projection=args.train_encoder_projection,
        train_skill_adapter=not args.freeze_skill_adapter,
        train_retrieval_scale=not args.freeze_retrieval_scale,
        init_checkpoint_path=args.init_checkpoint_path,
        init_checkpoint_skills_path=args.init_checkpoint_skills_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        preservation_anchor_weight=args.preservation_anchor_weight,
        explicit_negative_loss_weight=args.explicit_negative_loss_weight,
        explicit_negative_margin=args.explicit_negative_margin,
        mined_hard_negative_loss_weight=args.mined_hard_negative_loss_weight,
        mined_hard_negative_margin=args.mined_hard_negative_margin,
        mined_hard_negative_top_k=args.mined_hard_negative_top_k,
        route_scorer=args.route_scorer,
        belief_top_k=args.belief_top_k,
        frozen_backbone_cache_mode=args.frozen_backbone_cache_mode,
        frozen_backbone_cache_batch_size=args.frozen_backbone_cache_batch_size,
        resume_skill_table_mode=args.resume_skill_table_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
