from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clstr.stage_quality_common import (
    bounded_window_size,
    finite_float as _as_float,
    mean as _mean,
    read_json_file as _read_json,
    read_jsonl_file as _read_jsonl,
    source_coverage as _source_coverage,
    window_mean as _window_mean,
    write_json_report as _write_json,
)


DEFAULT_EXPECTED_SOURCES = ["skillret", "toolret_training", "toolbench_g3", "traject_bench"]
DEFAULT_REQUIRED_RECALL_KEYS = ["recall_at_20", "recall_at_50", "recall_at_100"]
DEFAULT_STAGE0_CHECKPOINT_NAME = "clstr_unified_retrieval_v2-step5000.pt"
DEFAULT_STAGE0_FULL_EVAL_METRICS = "full_retrieval_eval/metrics.json"


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    return {
        "stage": payload.get("stage"),
        "step": payload.get("step"),
        "stage0_protocol": payload.get("stage0_protocol") if isinstance(payload.get("stage0_protocol"), dict) else {},
        "train_safety": payload.get("train_safety") if isinstance(payload.get("train_safety"), dict) else {},
    }


def _report_metrics(path: str | Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = _read_json(path)
    raw_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        number = _as_float(value)
        if number is not None:
            metric_key = str(key)
            metrics[metric_key] = number
            if metric_key.startswith("recall_at_"):
                metrics.setdefault("Recall@" + metric_key.rsplit("_", 1)[-1], number)
            elif metric_key.startswith("Recall@"):
                metrics.setdefault("recall_at_" + metric_key.rsplit("@", 1)[-1], number)
    return metrics, payload


def _eval_metric_key(recall_key: str) -> str:
    if recall_key.startswith("Recall@"):
        return recall_key
    if recall_key.startswith("recall_at_"):
        return "Recall@" + recall_key.rsplit("_", 1)[-1]
    return recall_key


def _training_recall_tail(rows: list[dict[str, Any]], keys: list[str], window: int) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for key in keys:
        if key.startswith("Recall@"):
            training_key = "recall_at_" + key.rsplit("@", 1)[-1]
        else:
            training_key = key
        values[training_key] = _window_mean(rows, training_key, window, tail=True)
    return values


def audit_stage0_biencoder_quality(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    stage0_eval_metrics_path: str | Path | None = None,
    baseline_metrics_path: str | Path | None = "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json",
    output_path: str | Path | None = None,
    min_steps: int = 5000,
    first_window: int = 200,
    last_window: int = 200,
    min_loss_drop: float = 0.05,
    baseline_tolerance: float = 0.0,
    expected_sources: list[str] | None = None,
    required_recall_keys: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else output_dir / "checkpoints" / DEFAULT_STAGE0_CHECKPOINT_NAME
    )
    metrics_path = Path(metrics_path) if metrics_path is not None else output_dir / "training_metrics.jsonl"
    stage0_eval_metrics_path = (
        Path(stage0_eval_metrics_path)
        if stage0_eval_metrics_path is not None
        else output_dir / DEFAULT_STAGE0_FULL_EVAL_METRICS
    )
    loss_curve_path = output_dir / "loss_curve.svg"
    required_recall_keys = list(required_recall_keys or DEFAULT_REQUIRED_RECALL_KEYS)
    expected_sources = list(expected_sources or [])
    blockers: list[str] = []

    if not checkpoint_path.is_file():
        blockers.append("missing_stage0_checkpoint")
        checkpoint_metadata: dict[str, Any] = {}
    else:
        checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
        stage = str(checkpoint_metadata.get("stage") or "")
        if stage not in {"clstr_unified_stage0_biencoder", "clstr_unified_retrieval_v2"}:
            blockers.append("unexpected_stage0_checkpoint_stage")

    if not metrics_path.is_file():
        blockers.append("missing_training_metrics")
        rows: list[dict[str, Any]] = []
    else:
        rows = _read_jsonl(metrics_path)

    if not loss_curve_path.is_file():
        blockers.append("missing_loss_curve")

    max_step = max([int(row.get("step", 0)) for row in rows if isinstance(row.get("step"), int)] or [0])
    if max_step < int(min_steps):
        blockers.append("insufficient_training_steps")

    first_window = bounded_window_size(first_window, len(rows))
    last_window = bounded_window_size(last_window, len(rows))
    first_loss = _window_mean(rows, "loss", first_window, tail=False)
    last_loss = _window_mean(rows, "loss", last_window, tail=True)
    loss_drop = None if first_loss is None or last_loss is None else float(first_loss - last_loss)
    if loss_drop is None or loss_drop < float(min_loss_drop):
        blockers.append("insufficient_learning_signal")

    training_recall = _training_recall_tail(rows, required_recall_keys, last_window)
    for key in required_recall_keys:
        training_key = "recall_at_" + key.rsplit("@", 1)[-1] if key.startswith("Recall@") else key
        value = training_recall.get(training_key)
        if value is None:
            blockers.append(f"missing_{training_key}")

    stage0_eval_payload: dict[str, Any] = {}
    stage0_eval: dict[str, float] = {}
    if not stage0_eval_metrics_path.is_file():
        blockers.append("missing_stage0_full_eval_metrics")
    else:
        stage0_eval, stage0_eval_payload = _report_metrics(stage0_eval_metrics_path)
        if str(stage0_eval_payload.get("status") or "") not in {"", "ok"}:
            blockers.append("stage0_full_eval_status_not_ok")

    baseline_payload: dict[str, Any] = {}
    baseline: dict[str, float] = {}
    if baseline_metrics_path is None or not Path(baseline_metrics_path).is_file():
        blockers.append("missing_same_pool_skillrouter_baseline_metrics")
    else:
        baseline, baseline_payload = _report_metrics(baseline_metrics_path)
        for key in required_recall_keys:
            stage0_key = _eval_metric_key(key)
            stage0_value = stage0_eval.get(stage0_key)
            baseline_key = _eval_metric_key(key)
            baseline_value = baseline.get(baseline_key)
            if stage0_value is None:
                blockers.append(f"missing_stage0_eval_{stage0_key}")
            if baseline_value is None:
                blockers.append(f"missing_baseline_{baseline_key}")
            elif stage0_value is not None and float(stage0_value) + float(baseline_tolerance) < float(baseline_value):
                blockers.append(f"stage0_{key}_below_same_pool_baseline")

    observed_sources, rows_with_sources = _source_coverage(rows)
    missing_sources = [source for source in expected_sources if source not in set(observed_sources)]
    if expected_sources and rows and rows_with_sources == 0:
        blockers.append("missing_query_source_metrics")
    if missing_sources:
        blockers.append("missing_expected_query_sources")

    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "stage0_eval_metrics_path": str(stage0_eval_metrics_path),
        "loss_curve_path": str(loss_curve_path),
        "baseline_metrics_path": str(baseline_metrics_path) if baseline_metrics_path is not None else None,
        "checkpoint_metadata": checkpoint_metadata,
        "metrics_summary": {
            "metric_count": len(rows),
            "max_step": max_step,
            "first_window": first_window,
            "last_window": last_window,
            "first_loss_mean": first_loss,
            "last_loss_mean": last_loss,
            "loss_drop": loss_drop,
            "min_steps": int(min_steps),
            "min_loss_drop": float(min_loss_drop),
        },
        "coarse_recall_summary": {
            "required_recall_keys": required_recall_keys,
            "metric_parity": "baseline comparison uses full qrels/run retrieval metrics only; training batch recall is reported as learning signal only",
            "training_batch_tail": training_recall,
            "stage0_eval": stage0_eval,
            "baseline": baseline,
            "baseline_tolerance": float(baseline_tolerance),
            "baseline_payload_method": baseline_payload.get("method"),
            "stage0_eval_payload_method": stage0_eval_payload.get("method"),
        },
        "source_coverage": {
            "expected_sources": expected_sources,
            "observed_sources": observed_sources,
            "missing_sources": missing_sources,
            "rows_with_query_sources": rows_with_sources,
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
