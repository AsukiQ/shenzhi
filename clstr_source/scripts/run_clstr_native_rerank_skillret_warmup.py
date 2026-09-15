#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.native_rerank import run_clstr_native_rerank_warmup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR-native residual listwise rerank warmup on SKILLRET.")
    parser.add_argument("--data_root", default="data/skillret")
    parser.add_argument("--base_checkpoint_path", required=True)
    parser.add_argument("--output_dir", default="outputs/skillret_warmup_clstr_native_rerank")
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--candidate_k", type=int, default=50)
    parser.add_argument("--max_skills", type=int)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--base_score_temperature", type=float, default=0.05)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    args = parser.parse_args()
    report = run_clstr_native_rerank_warmup(**vars(args))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
