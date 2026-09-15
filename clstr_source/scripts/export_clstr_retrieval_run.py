#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.clstr_retrieval_export import export_clstr_retrieval_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a CLSTR checkpoint as a generic retrieval run over queries.jsonl and skills.jsonl."
    )
    parser.add_argument("--queries_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--encoder_pooling", default="masked_mean")
    parser.add_argument("--cross_encoder_pooling", default="masked_mean")
    parser.add_argument("--tokenizer_padding_side")
    parser.add_argument("--torch_dtype")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--projection_init", default="default")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--skill_text_format", default="clstr")
    parser.add_argument("--skill_table_batch_size", type=int, default=32)
    parser.add_argument("--skill_table_adapter_init", default="default")
    parser.add_argument("--disable_cross_encoder", action="store_true")
    parser.add_argument("--hf_cache_dir")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--query_text_format", choices=["raw", "skillrouter"], default="raw")
    parser.add_argument("--run_name", default="clstr_retrieval")
    args = parser.parse_args()

    report = export_clstr_retrieval_run(
        queries_path=args.queries_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        encoder_pooling=args.encoder_pooling,
        cross_encoder_pooling=args.cross_encoder_pooling,
        tokenizer_padding_side=args.tokenizer_padding_side,
        torch_dtype=args.torch_dtype,
        freeze_backbone=args.freeze_backbone,
        max_length=args.max_length,
        projection_init=args.projection_init,
        normalize_embeddings=args.normalize_embeddings,
        skill_text_format=args.skill_text_format,
        skill_table_batch_size=args.skill_table_batch_size,
        skill_table_adapter_init=args.skill_table_adapter_init,
        use_cross_encoder=not args.disable_cross_encoder,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        query_text_format=args.query_text_format,
        run_name=args.run_name,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
