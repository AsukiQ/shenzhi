#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import DEFAULT_TRANSITION_RESIDUAL_LAMBDA, LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER, ROUTE_SCORERS
from clstr.logged_online_stage4_train import (
    attach_replay_prefixes_from_source_rows,
    build_logged_online_stage4_rows_with_stage0_handoff,
    run_logged_online_stage4_adaptation_on_rows,
    run_logged_online_stage4_adaptation_with_model,
)
from clstr.logged_online_trajectory import iter_logged_online_steps, load_skill_id_set
from clstr.stage4_act_train import STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE, STAGE4_TRANSITION_SCORING_MODES
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_routing_and_head_checkpoints


def _parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    items = {item.strip() for item in str(value).split(",") if item.strip()}
    return items or None


def _parse_benchmark_caps(value: str | None) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    caps: dict[str, int] = {}
    normalized = str(value).replace(":", ",")
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, raw_cap = item.split("=", 1)
        else:
            key, raw_cap = item.split(":", 1)
        key = key.strip()
        if not key:
            continue
        caps[key] = int(raw_cap.strip())
    return caps or None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def materialize_logged_steps(
    *,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_path: str | Path,
    include_splits: set[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    skill_ids = load_skill_id_set(skills_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for step in iter_logged_online_steps(
            trajectories_path,
            skill_ids,
            include_splits=include_splits,
            max_rows=max_rows,
        ):
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")
            count += 1
    return {
        "trajectories_path": str(trajectories_path),
        "skills_path": str(skills_path),
        "output_path": str(output_path),
        "materialized_steps": count,
        "include_splits": sorted(include_splits) if include_splits is not None else None,
        "max_rows": max_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run executor-free logged-online Stage4 adaptation simulation. "
            "This consumes logged trajectory feedback, not AppWorld/Qwen executor rollouts."
        )
    )
    parser.add_argument("--trajectories_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl")
    parser.add_argument("--logged_steps_path", default=None)
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", default="outputs/logged_online_stage4_v4_2_progressive_smoke")
    parser.add_argument(
        "--routing_checkpoint_path",
        default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--head_checkpoint_path",
        default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument("--max_materialized_steps", type=int, default=None)
    parser.add_argument("--include_splits", default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_updates", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_rows", type=int, default=256)
    parser.add_argument("--eval_split_mode", default="sequential_tail", choices=["sequential_tail", "trajectory_prefix"])
    parser.add_argument("--trajectory_eval_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--candidate_count", type=int, default=None)
    parser.add_argument("--positive_missing_policy", default="skip", choices=["skip", "inject"])
    parser.add_argument("--use_stage0_handoff", action="store_true")
    parser.add_argument("--stage0_top_m", type=int, default=350)
    parser.add_argument(
        "--stage0_positive_missing_policy",
        default="skip",
        choices=["skip", "inject", "skip_or_inject_with_provenance"],
    )
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=100)
    parser.add_argument("--allowed_benchmarks", default="toolbench_g3,traject_bench,alfworld,webshop")
    parser.add_argument("--benchmark_caps", default=None)
    parser.add_argument("--eval_benchmark_caps", default=None)
    parser.add_argument("--save_stage4_rows_path", default=None)
    parser.add_argument("--prebuilt_stage4_rows_path", default=None)
    parser.add_argument("--replay_prefix_source_rows_path", default=None)
    parser.add_argument("--replay_prefix_max_steps", type=int, default=3)
    parser.add_argument("--train_transition", action="store_true")
    parser.add_argument("--train_score_calibrator", action="store_true")
    parser.add_argument("--transition_residual_lambda", type=float, default=DEFAULT_TRANSITION_RESIDUAL_LAMBDA)
    parser.add_argument("--online_memory_weight", type=float, default=0.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    parser.add_argument(
        "--online_memory_mode",
        default="latest_exact",
        choices=["latest_exact", "exact_count", "state_conditioned", "inventory_remaining", "auto_variant"],
    )
    parser.add_argument("--online_memory_state_similarity_threshold", type=float, default=0.2)
    parser.add_argument("--online_memory_state_similarity_temperature", type=float, default=1.0)
    parser.add_argument("--online_memory_auto_gate", action="store_true")
    parser.add_argument("--online_memory_gate_scope", default="global", choices=["global", "source_benchmark"])
    parser.add_argument("--online_memory_gate_eval_rows", type=int, default=64)
    parser.add_argument("--online_memory_gate_min_delta_mrr", type=float, default=0.0)
    parser.add_argument("--online_memory_gate_min_delta_recall5", type=float, default=0.0)
    parser.add_argument(
        "--transition_scoring_mode",
        default=STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
        choices=sorted(STAGE4_TRANSITION_SCORING_MODES),
    )
    parser.add_argument(
        "--route_scorer",
        default=LEGACY_PRIOR_RESIDUAL_ROUTE_SCORER,
        choices=sorted(ROUTE_SCORERS),
        help="Route scorer used by logged Stage4 train/eval; use unified_memory for current full CLSTR.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logged_steps_path = Path(args.logged_steps_path) if args.logged_steps_path else output_dir / "logged_online_steps.jsonl"
    materialize_report: dict[str, Any] = {"enabled": False, "path": str(logged_steps_path)}
    if args.prebuilt_stage4_rows_path:
        materialize_report = {
            "enabled": False,
            "reason": "prebuilt_stage4_rows",
            "path": None if args.logged_steps_path is None else str(logged_steps_path),
        }
    elif args.use_stage0_handoff:
        materialize_report = {
            "enabled": False,
            "reason": "stage0_handoff_uses_raw_trajectories",
            "path": None if args.logged_steps_path is None else str(logged_steps_path),
        }
    elif args.logged_steps_path is None:
        materialize_report = {
            "enabled": True,
            **materialize_logged_steps(
                trajectories_path=Path(args.trajectories_path),
                skills_path=Path(args.skills_path),
                output_path=logged_steps_path,
                include_splits=_parse_csv_set(args.include_splits),
                max_rows=args.max_materialized_steps,
            ),
        }

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(args.routing_checkpoint_path),
        skills_path=Path(args.skills_path),
        model_cache_dir=output_dir / "model_cache",
    )
    checkpoint_init_report = load_routing_and_head_checkpoints(
        model,
        routing_checkpoint_path=Path(args.routing_checkpoint_path),
        head_checkpoint_path=Path(args.head_checkpoint_path),
        partial_load_mode="stage0_routing_plus_act_init_heads_for_logged_online_stage4",
        protect_routing_foundation=True,
    )
    allowed_benchmarks = _parse_csv_set(args.allowed_benchmarks)
    benchmark_caps = _parse_benchmark_caps(args.benchmark_caps)
    eval_benchmark_caps = _parse_benchmark_caps(args.eval_benchmark_caps)
    if args.prebuilt_stage4_rows_path:
        rows = _read_jsonl(args.prebuilt_stage4_rows_path)
        replay_prefix_report: dict[str, Any] = {"enabled": False}
        if args.replay_prefix_source_rows_path:
            rows, replay_prefix_report = attach_replay_prefixes_from_source_rows(
                rows,
                source_rows=_iter_jsonl(args.replay_prefix_source_rows_path),
                max_steps=args.replay_prefix_max_steps,
            )
        data_report = {
            "candidate_source": "prebuilt_stage4_rows",
            "prebuilt_stage4_rows_path": str(args.prebuilt_stage4_rows_path),
            "stage4_rows": len(rows),
            "stage0_handoff_enabled": bool(args.use_stage0_handoff),
            "benchmark_caps": benchmark_caps,
            "replay_prefix_source_rows_path": (
                None if args.replay_prefix_source_rows_path is None else str(args.replay_prefix_source_rows_path)
            ),
            "replay_prefix_report": replay_prefix_report,
        }
        if args.save_stage4_rows_path:
            _write_jsonl(args.save_stage4_rows_path, rows)
        report = run_logged_online_stage4_adaptation_on_rows(
            model=model,
            rows=rows,
            data_report=data_report,
            output_dir=output_dir,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_rows=args.eval_rows,
            eval_split_mode=args.eval_split_mode,
            trajectory_eval_steps=args.trajectory_eval_steps,
            learning_rate=args.learning_rate,
            train_transition=args.train_transition,
            train_score_calibrator=args.train_score_calibrator,
            transition_residual_lambda=args.transition_residual_lambda,
            transition_scoring_mode=args.transition_scoring_mode,
            route_scorer=args.route_scorer,
            online_memory_weight=args.online_memory_weight,
            online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
            online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
            online_memory_mode=args.online_memory_mode,
            online_memory_state_similarity_threshold=args.online_memory_state_similarity_threshold,
            online_memory_state_similarity_temperature=args.online_memory_state_similarity_temperature,
            online_memory_auto_gate=args.online_memory_auto_gate,
            online_memory_gate_scope=args.online_memory_gate_scope,
            online_memory_gate_eval_rows=args.online_memory_gate_eval_rows,
            online_memory_gate_min_delta_mrr=args.online_memory_gate_min_delta_mrr,
            online_memory_gate_min_delta_recall5=args.online_memory_gate_min_delta_recall5,
            eval_benchmark_caps=eval_benchmark_caps,
            input_report={
                "trajectories_path": str(args.trajectories_path),
                "skills_path": str(args.skills_path),
                "stage0_handoff_enabled": bool(args.use_stage0_handoff),
                "prebuilt_stage4_rows_path": str(args.prebuilt_stage4_rows_path),
                "save_stage4_rows_path": None if args.save_stage4_rows_path is None else str(args.save_stage4_rows_path),
            },
        )
    elif args.use_stage0_handoff:
        rows, data_report = build_logged_online_stage4_rows_with_stage0_handoff(
            model=model,
            trajectories_path=Path(args.trajectories_path),
            skills_path=Path(args.skills_path),
            output_dir=output_dir,
            stage0_top_m=args.stage0_top_m,
            stage0_positive_missing_policy=args.stage0_positive_missing_policy,
            stage0_handoff_query_mode=args.stage0_handoff_query_mode,
            stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
            stage0_candidate_progress_interval_batches=args.stage0_candidate_progress_interval_batches,
            candidate_count=args.candidate_count,
            max_rows=args.max_rows,
            allowed_benchmarks=allowed_benchmarks,
            benchmark_caps=benchmark_caps,
            routing_checkpoint_path=Path(args.routing_checkpoint_path),
        )
        if args.save_stage4_rows_path:
            _write_jsonl(args.save_stage4_rows_path, rows)
        report = run_logged_online_stage4_adaptation_on_rows(
            model=model,
            rows=rows,
            data_report=data_report,
            output_dir=output_dir,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_rows=args.eval_rows,
            eval_split_mode=args.eval_split_mode,
            trajectory_eval_steps=args.trajectory_eval_steps,
            learning_rate=args.learning_rate,
            train_transition=args.train_transition,
            train_score_calibrator=args.train_score_calibrator,
            transition_residual_lambda=args.transition_residual_lambda,
            transition_scoring_mode=args.transition_scoring_mode,
            route_scorer=args.route_scorer,
            online_memory_weight=args.online_memory_weight,
            online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
            online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
            online_memory_mode=args.online_memory_mode,
            online_memory_state_similarity_threshold=args.online_memory_state_similarity_threshold,
            online_memory_state_similarity_temperature=args.online_memory_state_similarity_temperature,
            online_memory_auto_gate=args.online_memory_auto_gate,
            online_memory_gate_scope=args.online_memory_gate_scope,
            online_memory_gate_eval_rows=args.online_memory_gate_eval_rows,
            online_memory_gate_min_delta_mrr=args.online_memory_gate_min_delta_mrr,
            online_memory_gate_min_delta_recall5=args.online_memory_gate_min_delta_recall5,
            eval_benchmark_caps=eval_benchmark_caps,
            input_report={
                "trajectories_path": str(args.trajectories_path),
                "skills_path": str(args.skills_path),
                "stage0_handoff_enabled": True,
                "save_stage4_rows_path": None if args.save_stage4_rows_path is None else str(args.save_stage4_rows_path),
            },
        )
    else:
        report = run_logged_online_stage4_adaptation_with_model(
            model=model,
            logged_steps_path=logged_steps_path,
            skills_path=Path(args.skills_path),
            output_dir=output_dir,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_rows=args.eval_rows,
            eval_split_mode=args.eval_split_mode,
            trajectory_eval_steps=args.trajectory_eval_steps,
            learning_rate=args.learning_rate,
            candidate_count=args.candidate_count,
            max_rows=args.max_rows,
            allowed_benchmarks=allowed_benchmarks,
            benchmark_caps=benchmark_caps,
            positive_missing_policy=args.positive_missing_policy,
            train_transition=args.train_transition,
            train_score_calibrator=args.train_score_calibrator,
            transition_residual_lambda=args.transition_residual_lambda,
            transition_scoring_mode=args.transition_scoring_mode,
            route_scorer=args.route_scorer,
            online_memory_weight=args.online_memory_weight,
            online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
            online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
            online_memory_mode=args.online_memory_mode,
            online_memory_state_similarity_threshold=args.online_memory_state_similarity_threshold,
            online_memory_state_similarity_temperature=args.online_memory_state_similarity_temperature,
            online_memory_auto_gate=args.online_memory_auto_gate,
            online_memory_gate_scope=args.online_memory_gate_scope,
            online_memory_gate_eval_rows=args.online_memory_gate_eval_rows,
            online_memory_gate_min_delta_mrr=args.online_memory_gate_min_delta_mrr,
            online_memory_gate_min_delta_recall5=args.online_memory_gate_min_delta_recall5,
            eval_benchmark_caps=eval_benchmark_caps,
        )
    report["materialize_logged_steps"] = materialize_report
    report["checkpoint_init"] = checkpoint_init_report
    report["routing_init"] = routing_report
    report["model_config"] = model_config
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
