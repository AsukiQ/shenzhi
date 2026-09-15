#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.logged_online_stage4_diagnostics import run_logged_online_stage4_candidate_diagnostics_from_checkpoint


def _parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Stage0 candidate handoff rows for executor-free logged-online Stage4 training."
    )
    parser.add_argument(
        "--routing_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--trajectories_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl",
    )
    parser.add_argument(
        "--skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/logged_online_stage4_candidate_diagnostics/toolbench_g3_top350_rows1024",
    )
    parser.add_argument("--stage0_top_m", type=int, default=350)
    parser.add_argument("--allowed_benchmarks", default="toolbench_g3")
    parser.add_argument("--max_rows", type=int, default=1024)
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model_cache_dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_logged_online_stage4_candidate_diagnostics_from_checkpoint(
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        trajectories_path=Path(args.trajectories_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        stage0_top_m=args.stage0_top_m,
        allowed_benchmarks=_parse_csv_set(args.allowed_benchmarks),
        max_rows=args.max_rows,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        top_k=args.top_k,
        model_cache_dir=Path(args.model_cache_dir) if args.model_cache_dir else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
