#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage0_skillrouter_baseline import run_stage0_skillrouter_frozen_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run same-pool SkillRouter-compatible frozen bi-encoder baseline for CLSTR Stage0."
    )
    parser.add_argument("--data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final")
    parser.add_argument("--output_dir", default="outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak")
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--score_batch_size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--max_skills", type=int)
    args = parser.parse_args()

    report = run_stage0_skillrouter_frozen_baseline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        max_length=args.max_length,
        max_queries=args.max_queries,
        max_skills=args.max_skills,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
