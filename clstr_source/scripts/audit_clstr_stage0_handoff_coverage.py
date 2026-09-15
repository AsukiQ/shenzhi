#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.stage0_handoff_audit import audit_stage0_handoff_coverage_from_checkpoint


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage0 top-K coverage on Stage2 trajectory handoff queries.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_path", default="outputs/clstr_stage0_handoff_coverage_audit/report.json")
    parser.add_argument("--top_k_values", default="20,50,100,200,500")
    parser.add_argument("--query_modes", default="raw_state,skillrouter_state")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_cache_dir", default=None)
    args = parser.parse_args()

    report = audit_stage0_handoff_coverage_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_path=args.output_path,
        top_k_values=_parse_ints(args.top_k_values),
        query_modes=_parse_modes(args.query_modes),
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        model_cache_dir=args.model_cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
