#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.toolbench_skillrouter_finetune import run_toolbench_skillrouter_finetune


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official-compatible SkillRouter finetune on ToolBench-G3 trajectory eval_core data."
    )
    parser.add_argument("--data_root", default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data")
    parser.add_argument("--encoder_model_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--reranker_model_path", default=".cache/hf_models/SkillRouter-Reranker-0.6B")
    parser.add_argument("--official_repo_path", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_tasks", type=int)
    parser.add_argument("--max_eval_tasks", type=int)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--encoder_max_length", type=int, default=4096)
    parser.add_argument("--reranker_max_length", type=int, default=4096)
    parser.add_argument("--encoder_batch_size", type=int, default=16)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--prompt_format", default="flat-full", choices=["flat-full", "flat-nd", "struct"])
    parser.add_argument("--projection_init", default="identity", choices=["identity", "default"])
    parser.add_argument("--train_doc_projection", action="store_true")
    parser.add_argument("--query_variant", default="history", choices=["history", "goal_only"])
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    report = run_toolbench_skillrouter_finetune(
        data_root=args.data_root,
        encoder_model_path=args.encoder_model_path,
        output_dir=args.output_dir,
        official_repo_path=args.official_repo_path,
        reranker_model_path=args.reranker_model_path,
        max_train_tasks=args.max_train_tasks,
        max_eval_tasks=args.max_eval_tasks,
        max_skills=args.max_skills,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        encoder_max_length=args.encoder_max_length,
        reranker_max_length=args.reranker_max_length,
        encoder_batch_size=args.encoder_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        torch_dtype=args.torch_dtype,
        retrieval_top_k=args.retrieval_top_k,
        prompt_format=args.prompt_format,
        projection_init=args.projection_init,
        train_doc_projection=args.train_doc_projection,
        query_variant=args.query_variant,
        log_every=args.log_every,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
