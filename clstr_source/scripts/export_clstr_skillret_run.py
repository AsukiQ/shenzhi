#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillret_official import evaluate_official_run, export_clstr_official_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CLSTR checkpoint as a SKILLRET official TREC run.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--output_dir", default="outputs/skillret_official_eval/clstr_skillret_warmup")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--model_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--encoder_pooling", default="masked_mean")
    parser.add_argument("--cross_encoder_pooling", default="masked_mean")
    parser.add_argument("--tokenizer_padding_side")
    parser.add_argument("--torch_dtype")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--projection_init", default="default")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--skill_text_format", default="skillret_official")
    parser.add_argument("--skill_table_batch_size", type=int, default=1)
    parser.add_argument("--skill_table_adapter_init", default="default")
    parser.add_argument("--use_cross_encoder", action="store_true")
    parser.add_argument("--run_name", default="clstr_skillret_warmup")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    report = export_clstr_official_run(
        data_root=args.data_root,
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        split=args.split,
        top_k=args.top_k,
        model_dim=args.model_dim,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        max_skills=args.max_skills,
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
        use_cross_encoder=args.use_cross_encoder,
        run_name=args.run_name,
    )
    if args.evaluate:
        metrics = evaluate_official_run(
            data_root=args.data_root,
            run_path=report["run_path"],
            output_dir=args.output_dir,
            split=args.split,
        )
        metrics.update(
            {
                "method": args.run_name,
                "train_data": "SKILLRET train",
                "model_or_checkpoint": str(args.checkpoint_path),
                "init": "SkillRouter-Embedding-0.6B warm-start",
                "trainable_params": "projection + skill table adapter + retrieval head; backbone frozen",
                "frozen_backbone": True,
                "paper_role": "paper_static_retrieval" if args.max_queries is None and args.max_skills is None else "pilot",
            }
        )
        Path(args.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["metrics_path"] = str(Path(args.output_dir, "metrics.json"))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
