#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_skillrouter_base import run_appworld_skillrouter_base_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SkillRouter-base scorer on AppWorld routing qrels.")
    parser.add_argument("--tasks_path", default="data/appworld_routing/train_tasks.jsonl")
    parser.add_argument("--qrels_path", default="data/appworld_routing/train_qrels.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/appworld_skillrouter_base_train_v1")
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    report = run_appworld_skillrouter_base_train(
        tasks_path=args.tasks_path,
        qrels_path=args.qrels_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
