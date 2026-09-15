from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.retrieval_warmup import run_skillret_retrieval_warmup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR retrieval warmup on SKILLRET or CLSTR unified v2.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--output_dir", default="outputs/skillret_warmup")
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--split_path")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--encoder_pooling", default="masked_mean")
    parser.add_argument("--cross_encoder_pooling", default="masked_mean")
    parser.add_argument("--tokenizer_padding_side")
    parser.add_argument("--torch_dtype")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--projection_init", default="default")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--defer_skill_table_init", action="store_true")
    parser.add_argument("--skill_text_format", default="clstr")
    parser.add_argument("--skill_table_batch_size", type=int, default=32)
    parser.add_argument("--skill_table_adapter_init", default="default")
    parser.add_argument("--disable_cross_encoder", action="store_true")
    parser.add_argument("--query_text_format", default="raw", choices=["raw", "skillrouter"])
    parser.add_argument("--state_query_prompt_version")
    parser.add_argument("--state_query_max_chars", type=int)
    parser.add_argument(
        "--state_query_truncation",
        choices=["none", "head_v1", "head_tail_v1"],
    )
    parser.add_argument("--data_format", default="skillret", choices=["skillret", "unified_v2"])
    parser.add_argument("--no_shuffle_queries", action="store_true")
    args = parser.parse_args()
    report = run_skillret_retrieval_warmup(
        data_root=args.data_root,
        base_model_name=args.base_model_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        top_k=args.top_k,
        max_skills=args.max_skills,
        max_queries=args.max_queries,
        learning_rate=args.learning_rate,
        split_path=args.split_path,
        train_split=args.train_split,
        encoder_pooling=args.encoder_pooling,
        cross_encoder_pooling=args.cross_encoder_pooling,
        tokenizer_padding_side=args.tokenizer_padding_side,
        torch_dtype=args.torch_dtype,
        freeze_backbone=args.freeze_backbone,
        max_length=args.max_length,
        projection_init=args.projection_init,
        normalize_embeddings=args.normalize_embeddings,
        defer_skill_table_init=args.defer_skill_table_init,
        skill_text_format=args.skill_text_format,
        skill_table_batch_size=args.skill_table_batch_size,
        skill_table_adapter_init=args.skill_table_adapter_init,
        use_cross_encoder=not args.disable_cross_encoder,
        query_text_format=args.query_text_format,
        state_query_prompt_version=args.state_query_prompt_version,
        state_query_max_chars=args.state_query_max_chars,
        state_query_truncation=args.state_query_truncation,
        data_format=args.data_format,
        shuffle_queries=not args.no_shuffle_queries,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
