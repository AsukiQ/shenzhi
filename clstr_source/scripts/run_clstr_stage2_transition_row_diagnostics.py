#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import DEFAULT_TRANSITION_RESIDUAL_LAMBDA, TRANSITION_SCORING_MODE, TRANSITION_SCORING_MODES
from clstr.stage2_transition_row_diagnostics import run_stage2_transition_row_diagnostics


def _benchmark_caps(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    output: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        separator = "=" if "=" in item else ":"
        if separator not in item:
            raise ValueError(f"invalid benchmark cap item: {item}")
        key, raw_value = item.split(separator, 1)
        output[str(key).strip()] = int(raw_value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run row-level CLSTR Stage2 transition ranking diagnostics.")
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--stage2_checkpoint_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_m", type=int, default=350)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_diagnostic_rows", type=int)
    parser.add_argument("--allowed_benchmarks", nargs="*")
    parser.add_argument("--benchmark_caps")
    parser.add_argument("--embedding_cache_mode", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--embedding_cache_max_rows", type=int, default=20000)
    parser.add_argument("--skill_text_format")
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state")
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--transition_residual_lambda", type=float, default=DEFAULT_TRANSITION_RESIDUAL_LAMBDA)
    parser.add_argument("--transition_scoring_mode", default=TRANSITION_SCORING_MODE, choices=sorted(TRANSITION_SCORING_MODES))
    args = parser.parse_args()

    report = run_stage2_transition_row_diagnostics(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        top_m=args.top_m,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        max_diagnostic_rows=args.max_diagnostic_rows,
        allowed_benchmarks=args.allowed_benchmarks,
        benchmark_caps=_benchmark_caps(args.benchmark_caps),
        embedding_cache_mode=args.embedding_cache_mode,
        embedding_cache_max_rows=args.embedding_cache_max_rows,
        skill_text_format=args.skill_text_format,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        transition_residual_lambda=args.transition_residual_lambda,
        transition_scoring_mode=args.transition_scoring_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
