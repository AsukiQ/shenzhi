#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skillret_official import evaluate_official_run
from clstr.skillrouter_style import export_skillrouter_style_official_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SkillRouter-style finetune checkpoint as a SKILLRET official TREC run.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", default="outputs/skillret_official_eval/baselines/skillrouter_style_finetune_full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--run_name", default="skillrouter_style_finetune_full")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    report = export_skillrouter_style_official_run(
        data_root=args.data_root,
        base_model_name=args.base_model_name,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        split=args.split,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_queries=args.max_queries,
        max_skills=args.max_skills,
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
                "trainable_params": "SkillRouter-style q/doc projection adapters; backbone frozen",
                "frozen_backbone": True,
                "paper_role": "paper_static_retrieval" if args.max_queries is None and args.max_skills is None else "pilot",
            }
        )
        Path(args.output_dir, "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["metrics_path"] = str(Path(args.output_dir, "metrics.json"))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
