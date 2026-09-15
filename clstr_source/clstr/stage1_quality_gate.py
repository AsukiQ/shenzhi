from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from clstr.stage_quality_common import (
    bounded_window_size,
    max_step_or_update,
    read_jsonl_file as _read_jsonl,
    source_coverage as _source_coverage,
    window_mean as _window_mean,
    write_json_report as _write_json,
)


DEFAULT_EXPECTED_SOURCES = ["skillret", "toolret_training", "toolbench_g3", "traject_bench"]


def _recall_key(rows: list[dict[str, Any]]) -> str | None:
    preferred = ["recall_at_50", "recall_at_10", "recall_at_5", "recall_at_1"]
    present = {key for row in rows for key in row.keys()}
    for key in preferred:
        if key in present:
            return key
    for key in sorted(present):
        if key.startswith("recall_at_"):
            return key
    return None


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    return {
        "stage": payload.get("stage"),
        "step": payload.get("step"),
        "sampling_strategy": payload.get("sampling_strategy"),
        "shuffle_queries": payload.get("shuffle_queries"),
        "train_safety": payload.get("train_safety") if isinstance(payload.get("train_safety"), dict) else {},
    }


def audit_stage1_retrieval_warmup_quality(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    min_steps: int = 5000,
    first_window: int = 200,
    last_window: int = 200,
    min_loss_drop: float = 0.05,
    min_last_recall: float = 0.02,
    expected_sources: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else output_dir / "checkpoints" / "clstr_unified_retrieval_v2-step5000.pt"
    )
    metrics_path = Path(metrics_path) if metrics_path is not None else output_dir / "training_metrics.jsonl"
    expected_sources = list(expected_sources or DEFAULT_EXPECTED_SOURCES)
    blockers: list[str] = []

    if not checkpoint_path.is_file():
        blockers.append("missing_stage1_checkpoint")
        checkpoint_metadata: dict[str, Any] = {}
    else:
        checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
        if checkpoint_metadata.get("stage") != "clstr_unified_retrieval_v2":
            blockers.append("unexpected_stage1_checkpoint_stage")
        if checkpoint_metadata.get("sampling_strategy") != "batch_stride":
            blockers.append("stage1_not_batch_stride_sampled")
        if checkpoint_metadata.get("shuffle_queries") is not True:
            blockers.append("stage1_queries_not_shuffled")

    if not metrics_path.is_file():
        blockers.append("missing_training_metrics")
        rows: list[dict[str, Any]] = []
    else:
        rows = _read_jsonl(metrics_path)

    max_step = max_step_or_update(rows)
    if max_step < int(min_steps):
        blockers.append("insufficient_training_steps")

    first_window = bounded_window_size(first_window, len(rows))
    last_window = bounded_window_size(last_window, len(rows))
    first_loss = _window_mean(rows, "loss", first_window, tail=False)
    last_loss = _window_mean(rows, "loss", last_window, tail=True)
    loss_drop = None if first_loss is None or last_loss is None else float(first_loss - last_loss)
    if loss_drop is None or loss_drop < float(min_loss_drop):
        blockers.append("insufficient_learning_signal")

    recall_key = _recall_key(rows)
    last_recall = _window_mean(rows, recall_key, last_window, tail=True) if recall_key is not None else None
    if last_recall is None or last_recall < float(min_last_recall):
        blockers.append("low_last_recall")

    observed_sources, rows_with_sources = _source_coverage(rows)
    if rows and rows_with_sources == 0:
        blockers.append("missing_query_source_metrics")
    missing_sources = [source for source in expected_sources if source not in set(observed_sources)]
    if missing_sources:
        blockers.append("missing_expected_query_sources")

    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "checkpoint_metadata": checkpoint_metadata,
        "metrics_summary": {
            "metric_count": len(rows),
            "max_step": max_step,
            "first_window": first_window,
            "last_window": last_window,
            "first_loss_mean": first_loss,
            "last_loss_mean": last_loss,
            "loss_drop": loss_drop,
            "recall_key": recall_key,
            "last_recall_mean": last_recall,
            "min_steps": int(min_steps),
            "min_loss_drop": float(min_loss_drop),
            "min_last_recall": float(min_last_recall),
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
