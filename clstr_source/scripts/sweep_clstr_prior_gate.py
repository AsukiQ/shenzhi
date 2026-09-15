#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.prior_gate_sweep import DEFAULT_FORMULAS, run_prior_gate_replay_sweep, write_gate_sweep_report


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


def _formulas(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_FORMULAS)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run no-train CLSTR prior/residual gate sweep.")
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--stage_checkpoint_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_m", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--max_diagnostic_rows", type=int)
    parser.add_argument("--allowed_benchmarks", nargs="*")
    parser.add_argument("--benchmark_caps")
    parser.add_argument("--skill_text_format")
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state")
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--transition_scoring_mode", default="stage0_rank_prior_plus_transition_residual")
    parser.add_argument("--transition_inventory_mask_mode", default="stage0_topk_trajectory_prior")
    parser.add_argument("--transition_inventory_min_candidates", type=int, default=50)
    parser.add_argument("--transition_positive_mode", default="gold_plus_equivalent")
    parser.add_argument("--formulas")
    parser.add_argument("--lambda_min", type=float, default=0.05)
    parser.add_argument("--lambda_max", type=float, default=0.5)
    parser.add_argument("--margin_threshold", type=float, default=1.0)
    parser.add_argument("--entropy_low", type=float, default=0.2)
    parser.add_argument("--entropy_high", type=float, default=0.8)
    args = parser.parse_args()

    report = run_prior_gate_replay_sweep(
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage_checkpoint_path=args.stage_checkpoint_path,
        train_path=args.train_path,
        skills_path=args.skills_path,
        output_dir=args.output_dir,
        top_m=args.top_m,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        max_diagnostic_rows=args.max_diagnostic_rows,
        allowed_benchmarks=args.allowed_benchmarks,
        benchmark_caps=_benchmark_caps(args.benchmark_caps),
        skill_text_format=args.skill_text_format,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
        transition_scoring_mode=args.transition_scoring_mode,
        transition_inventory_mask_mode=args.transition_inventory_mask_mode,
        transition_inventory_min_candidates=args.transition_inventory_min_candidates,
        transition_positive_mode=args.transition_positive_mode,
        formulas=_formulas(args.formulas),
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        margin_threshold=args.margin_threshold,
        entropy_low=args.entropy_low,
        entropy_high=args.entropy_high,
    )
    write_gate_sweep_report(Path(args.output_dir) / "gate_sweep_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
