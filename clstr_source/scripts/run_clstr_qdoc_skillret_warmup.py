#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.clstr_retrieval_adapters import run_clstr_qdoc_warmup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR q/doc adapter warmup on SKILLRET.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--output_dir", default="outputs/skillret_warmup_clstr_qdoc")
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--learning_rate", type=float, default=1.0e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--skill_batch_size", type=int, default=1)
    parser.add_argument("--query_batch_size", type=int)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--tokenizer_padding_side", default="left")
    parser.add_argument("--projection_init", default="identity")
    args = parser.parse_args()
    report = run_clstr_qdoc_warmup(**vars(args))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
