#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage0_handoff_row_diagnostics import audit_stage0_handoff_row_diagnostics_from_checkpoint


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage0 row-level handoff diagnostics for next-skill routing.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_stage0_handoff_row_diagnostics")
    parser.add_argument("--benchmark", default="traject_bench")
    parser.add_argument("--query_mode", default="skillrouter_state")
    parser.add_argument("--max_rows", type=int, default=960)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--k_values", default="20,50,100,200,500")
    parser.add_argument("--model_cache_dir")
    args = parser.parse_args()

    report = audit_stage0_handoff_row_diagnostics_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
        query_mode=args.query_mode,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        top_k=args.top_k,
        k_values=_parse_ints(args.k_values),
        model_cache_dir=args.model_cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
