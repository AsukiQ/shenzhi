#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_COLD_SPEEDUP = 1.25
REQUIRED_WARM_SPEEDUP = 5.0
MAX_COMPACT_TO_LEGACY_RATIO = 0.25
DEFAULT_SCORE_ATOL = 1.0e-5


def _handoff_output_row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("row_id"),
        row.get("trajectory_id"),
        row.get("step_index"),
        row.get("benchmark"),
        row.get("state_text"),
        row.get("next_state_text"),
    )


def compare_handoff_outputs(
    legacy_rows: list[dict[str, Any]],
    accelerated_rows: list[dict[str, Any]],
    *,
    score_atol: float = DEFAULT_SCORE_ATOL,
) -> dict[str, Any]:
    score_atol = float(score_atol)
    row_count_equal = len(legacy_rows) == len(accelerated_rows)
    retained_row_identity_equal = row_count_equal and all(
        _handoff_output_row_identity(legacy_row) == _handoff_output_row_identity(accelerated_row)
        for legacy_row, accelerated_row in zip(legacy_rows, accelerated_rows, strict=True)
    )
    candidate_ids_equal = row_count_equal
    loss_masks_equal = row_count_equal
    positive_decisions_equal = row_count_equal
    finite_scores = True
    score_shapes_equal = row_count_equal
    max_score_abs_diff = 0.0
    decision_keys = (
        "stage0_current_positive_hit",
        "stage0_next_positive_hit",
        "stage0_positive_injected",
        "stage0_current_skill_candidate_added",
        "stage0_current_skill_candidate_role",
    )
    if row_count_equal:
        for legacy_row, accelerated_row in zip(legacy_rows, accelerated_rows, strict=True):
            for prefix in ("stage0_candidate", "stage0_next_candidate"):
                index_key = f"{prefix}_skill_indices"
                score_key = f"{prefix}_skill_scores"
                legacy_indices = [int(value) for value in legacy_row.get(index_key) or []]
                accelerated_indices = [int(value) for value in accelerated_row.get(index_key) or []]
                candidate_ids_equal = candidate_ids_equal and legacy_indices == accelerated_indices
                legacy_scores = [float(value) for value in legacy_row.get(score_key) or []]
                accelerated_scores = [float(value) for value in accelerated_row.get(score_key) or []]
                if len(legacy_scores) != len(accelerated_scores):
                    score_shapes_equal = False
                    continue
                for legacy_score, accelerated_score in zip(
                    legacy_scores,
                    accelerated_scores,
                    strict=True,
                ):
                    if not math.isfinite(legacy_score) or not math.isfinite(accelerated_score):
                        finite_scores = False
                        continue
                    max_score_abs_diff = max(
                        max_score_abs_diff,
                        abs(legacy_score - accelerated_score),
                    )
            loss_masks_equal = loss_masks_equal and (
                (legacy_row.get("loss_mask") or {}) == (accelerated_row.get("loss_mask") or {})
            )
            positive_decisions_equal = positive_decisions_equal and all(
                legacy_row.get(key) == accelerated_row.get(key)
                for key in decision_keys
            )
    scores_within_tolerance = bool(
        score_shapes_equal
        and finite_scores
        and max_score_abs_diff <= score_atol
    )
    status = "ok" if all(
        (
            row_count_equal,
            retained_row_identity_equal,
            candidate_ids_equal,
            loss_masks_equal,
            positive_decisions_equal,
            scores_within_tolerance,
        )
    ) else "action_required"
    return {
        "status": status,
        "legacy_retained_rows": len(legacy_rows),
        "accelerated_retained_rows": len(accelerated_rows),
        "row_count_equal": row_count_equal,
        "retained_row_identity_equal": retained_row_identity_equal,
        "candidate_ids_equal": candidate_ids_equal,
        "loss_masks_equal": loss_masks_equal,
        "positive_decisions_equal": positive_decisions_equal,
        "finite_scores": finite_scores,
        "score_shapes_equal": score_shapes_equal,
        "score_atol": score_atol,
        "max_score_abs_diff": max_score_abs_diff,
        "scores_within_tolerance": scores_within_tolerance,
    }


def evaluate_gate(
    *,
    parity_report: dict[str, Any],
    backbone_trainable_parameters: int,
    cold_speedup: float,
    warm_speedup: float,
    compact_to_legacy_ratio: float,
    errors: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    blockers: list[str] = []
    if parity_report.get("status") != "ok":
        blockers.append("parity")
    if int(backbone_trainable_parameters) != 0:
        blockers.append("backbone_trainable_parameters")
    if not math.isfinite(float(cold_speedup)) or float(cold_speedup) < REQUIRED_COLD_SPEEDUP:
        blockers.append("cold_speedup")
    if not math.isfinite(float(warm_speedup)) or float(warm_speedup) < REQUIRED_WARM_SPEEDUP:
        blockers.append("warm_speedup")
    if (
        not math.isfinite(float(compact_to_legacy_ratio))
        or float(compact_to_legacy_ratio) > MAX_COMPACT_TO_LEGACY_RATIO
    ):
        blockers.append("compact_to_legacy_ratio")
    if errors:
        blockers.append("runtime_errors")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "requirements": {
            "cold_speedup_min": REQUIRED_COLD_SPEEDUP,
            "warm_speedup_min": REQUIRED_WARM_SPEEDUP,
            "compact_to_legacy_ratio_max": MAX_COMPACT_TO_LEGACY_RATIO,
        },
        "parity": parity_report,
        "backbone_trainable_parameters": int(backbone_trainable_parameters),
        "cold_speedup": float(cold_speedup),
        "warm_speedup": float(warm_speedup),
        "compact_to_legacy_ratio": float(compact_to_legacy_ratio),
        "errors": [str(error) for error in errors],
    }


def _parse_int_list(value: str) -> list[int]:
    output = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            output.append(int(item))
    if not output:
        raise ValueError("candidate batch size list must not be empty")
    if any(item <= 0 for item in output):
        raise ValueError("candidate batch sizes must be positive")
    return output


def _parse_str_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    output = {item.strip() for item in str(value).split(",") if item.strip()}
    return output or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_cuda_call(callable_):
    _synchronize_cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    value = callable_()
    _synchronize_cuda()
    elapsed = time.perf_counter() - started
    return value, elapsed, int(torch.cuda.max_memory_allocated())


def _backbone_parameter_report(model: Any) -> dict[str, int]:
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", None)
    if backbone is None or not hasattr(backbone, "parameters"):
        raise ValueError("Qwen handoff gate requires model.encoder.backbone")
    parameters = list(backbone.parameters())
    return {
        "total_parameters": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameters": sum(
            int(parameter.numel())
            for parameter in parameters
            if parameter.requires_grad
        ),
    }


def _prepare_gate_rows(
    *,
    train_path: str | Path,
    row_count: int,
    allowed_benchmarks: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from clstr.full_base_train import (
        _attach_adjacent_next_states,
        _read_jsonl,
        _split_name,
    )
    from clstr.stage0_audit_sampling import select_audit_rows
    from clstr.trajectory_inventory import backfill_tool_inventory_from_trajectory_rows

    train_splits = {"train", "training", "train_or_released_g3", "released_train"}
    source_rows = _read_jsonl(train_path)
    adjacent_rows, causal_report = _attach_adjacent_next_states(source_rows)
    filtered = [
        row
        for row in adjacent_rows
        if _split_name(row) in train_splits
        and (allowed_benchmarks is None or str(row.get("benchmark") or "") in allowed_benchmarks)
    ]
    filtered, inventory_report = backfill_tool_inventory_from_trajectory_rows(filtered)
    selected = select_audit_rows(filtered, int(row_count))
    if len(selected) != int(row_count):
        raise ValueError(
            f"handoff acceleration gate requires {row_count} fixed rows, found {len(selected)}"
        )
    return selected, {
        "source_rows": len(source_rows),
        "eligible_rows": len(filtered),
        "selected_rows": len(selected),
        "selection": "benchmark_label_round_robin",
        "causal_next_state": causal_report,
        "tool_inventory_backfill": inventory_report,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from clstr.full_base_train import (
        _attach_stage0_topm_candidates,
        _prepare_stage0_topm_candidates_with_cache,
        _read_jsonl,
        _skill_id,
        _write_jsonl_rows,
    )
    from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen handoff acceleration gate requires CUDA")
    checkpoint_path = Path(args.checkpoint_path)
    train_path = Path(args.train_path)
    skills_path = Path(args.skills_path)
    output_path = Path(args.output_path)
    output_dir = output_path.parent
    for required_path in (checkpoint_path, train_path, skills_path):
        if not required_path.is_file() or required_path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing handoff gate input: {required_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, row_report = _prepare_gate_rows(
        train_path=train_path,
        row_count=int(args.row_count),
        allowed_benchmarks=_parse_str_set(args.allowed_benchmarks),
    )
    skills = _read_jsonl(skills_path)
    skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(skills)]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    if not skills:
        raise ValueError("Qwen handoff acceleration gate requires non-empty skill pool")

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    backbone_report = _backbone_parameter_report(model)

    common = {
        "top_m": int(args.top_m),
        "positive_missing_policy": "skip",
        "query_mode": str(args.query_mode),
        "allow_full_pool_stage2_debug": False,
        "routing_checkpoint_path": checkpoint_path,
        "device": device,
        "inventory_min_candidates": int(args.inventory_min_candidates),
        "next_skill_pool_mode": "full_pool",
    }

    legacy_rows_and_report, legacy_compute_seconds, legacy_peak_memory = _measure_cuda_call(
        lambda: _attach_stage0_topm_candidates(
            model,
            rows,
            skills,
            skill_id_to_idx,
            manifest_path=None,
            encode_batch_size=int(args.legacy_batch_size),
            **common,
        )
    )
    legacy_rows, legacy_handoff_report = legacy_rows_and_report
    legacy_cache_dir = output_dir / "legacy_jsonl"
    if legacy_cache_dir.exists():
        shutil.rmtree(legacy_cache_dir)
    legacy_cache_path = legacy_cache_dir / "rows.jsonl"
    materialization_started = time.perf_counter()
    _write_jsonl_rows(legacy_cache_path, legacy_rows)
    legacy_materialization_seconds = time.perf_counter() - materialization_started
    legacy_total_seconds = legacy_compute_seconds + legacy_materialization_seconds
    legacy_bytes = _directory_size(legacy_cache_dir)

    batch_results: list[dict[str, Any]] = []
    for batch_size in _parse_int_list(args.candidate_batch_sizes):
        cache_dir = output_dir / f"row_sharded_v1_batch{batch_size}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        errors: list[str] = []
        try:
            cold_value, cold_seconds, cold_peak_memory = _measure_cuda_call(
                lambda: _prepare_stage0_topm_candidates_with_cache(
                    model,
                    rows,
                    skills,
                    skill_id_to_idx,
                    manifest_path=output_dir / f"cold_batch{batch_size}.json",
                    encode_batch_size=batch_size,
                    cache_mode="refresh",
                    cache_dir=cache_dir,
                    cache_format="row_sharded_v1",
                    cache_shard_size=int(args.cache_shard_size),
                    **common,
                )
            )
            cold_rows, cold_report = cold_value
            cold_parity = compare_handoff_outputs(
                legacy_rows,
                cold_rows,
                score_atol=float(args.score_atol),
            )
            if cold_report.get("static_candidate_scorer") != "unified_static":
                errors.append("cold run did not use unified_static scorer")
            if cold_report.get("query_mode") != str(args.query_mode):
                errors.append("cold run changed query mode")
            if (cold_report.get("cache") or {}).get("format") != "row_sharded_v1":
                errors.append("cold run did not use row_sharded_v1")

            warm_value, warm_seconds, warm_peak_memory = _measure_cuda_call(
                lambda: _prepare_stage0_topm_candidates_with_cache(
                    model,
                    rows,
                    skills,
                    skill_id_to_idx,
                    manifest_path=output_dir / f"warm_batch{batch_size}.json",
                    encode_batch_size=batch_size,
                    cache_mode="auto",
                    cache_dir=cache_dir,
                    cache_format="row_sharded_v1",
                    cache_shard_size=int(args.cache_shard_size),
                    **common,
                )
            )
            warm_rows, warm_report = warm_value
            warm_parity = compare_handoff_outputs(
                legacy_rows,
                warm_rows,
                score_atol=float(args.score_atol),
            )
            if not bool((warm_report.get("cache") or {}).get("cache_hit")):
                errors.append("warm run did not report a complete cache hit")
            parity_report = {
                "status": (
                    "ok"
                    if cold_parity.get("status") == "ok" and warm_parity.get("status") == "ok"
                    else "action_required"
                ),
                "cold": cold_parity,
                "warm": warm_parity,
            }
            compact_bytes = _directory_size(cache_dir)
            cold_speedup = legacy_total_seconds / max(cold_seconds, 1.0e-12)
            warm_speedup = legacy_total_seconds / max(warm_seconds, 1.0e-12)
            compact_ratio = compact_bytes / max(legacy_bytes, 1)
            gate = evaluate_gate(
                parity_report=parity_report,
                backbone_trainable_parameters=backbone_report["trainable_parameters"],
                cold_speedup=cold_speedup,
                warm_speedup=warm_speedup,
                compact_to_legacy_ratio=compact_ratio,
                errors=errors,
            )
            batch_results.append(
                {
                    **gate,
                    "batch_size": batch_size,
                    "cold_seconds": cold_seconds,
                    "warm_seconds": warm_seconds,
                    "cold_peak_cuda_bytes": cold_peak_memory,
                    "warm_peak_cuda_bytes": warm_peak_memory,
                    "compact_cache_bytes": compact_bytes,
                    "legacy_cache_bytes": legacy_bytes,
                    "cold_cache": cold_report.get("cache"),
                    "warm_cache": warm_report.get("cache"),
                    "raw_candidate_cache_identity": cold_report.get("raw_candidate_cache_identity"),
                }
            )
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            batch_results.append(
                {
                    "status": "action_required",
                    "blockers": ["runtime_errors"],
                    "batch_size": batch_size,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    passing = [result for result in batch_results if result.get("status") == "ok"]
    selected = min(passing, key=lambda result: float(result["cold_seconds"])) if passing else None
    report = {
        "status": "ok" if selected is not None else "action_required",
        "gate": "qwen06_stage0_handoff_acceleration",
        "checkpoint_path": str(checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "row_selection": row_report,
        "skill_count": len(skills),
        "top_m": int(args.top_m),
        "inventory_min_candidates": int(args.inventory_min_candidates),
        "query_mode": str(args.query_mode),
        "legacy_batch_size": int(args.legacy_batch_size),
        "candidate_batch_sizes": _parse_int_list(args.candidate_batch_sizes),
        "score_atol": float(args.score_atol),
        "backbone": backbone_report,
        "legacy": {
            "compute_seconds": legacy_compute_seconds,
            "materialization_seconds": legacy_materialization_seconds,
            "total_seconds": legacy_total_seconds,
            "peak_cuda_bytes": legacy_peak_memory,
            "cache_bytes": legacy_bytes,
            "retained_rows": len(legacy_rows),
            "handoff_report": legacy_handoff_report,
        },
        "batch_results": batch_results,
        "selected_batch_size": None if selected is None else int(selected["batch_size"]),
        "selected_result": selected,
        "cross_stage_reuse": {
            "policy": "reuse_only_when_effective_unified_static_scorer_digest_matches",
            "unsafe_stale_reuse": False,
        },
        "model_config": model_config,
        "routing_init": routing_report,
    }
    _write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parity and performance gate for compact frozen-Qwen Stage0 handoff caching."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--row_count", type=int, default=2048)
    parser.add_argument("--top_m", type=int, default=500)
    parser.add_argument("--inventory_min_candidates", type=int, default=64)
    parser.add_argument("--query_mode", default="checkpoint_state_query")
    parser.add_argument("--legacy_batch_size", type=int, default=16)
    parser.add_argument("--candidate_batch_sizes", default="16,64,128")
    parser.add_argument("--cache_shard_size", type=int, default=2048)
    parser.add_argument("--score_atol", type=float, default=DEFAULT_SCORE_ATOL)
    parser.add_argument(
        "--allowed_benchmarks",
        default="toolbench_g3,traject_bench,alfworld,webshop",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_audit(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "ok" else 2
    except Exception as exc:
        report = {
            "status": "error",
            "gate": "qwen06_stage0_handoff_acceleration",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(args.output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
