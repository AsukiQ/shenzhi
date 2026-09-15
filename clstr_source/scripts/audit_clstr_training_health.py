#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit_training_health(
    output_dir: str | Path,
    *,
    min_metric_rows: int = 1,
    require_latest_checkpoint: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metrics_path = output_dir / "training_metrics.jsonl"
    setup_status_path = output_dir / "setup_status.jsonl"
    latest_checkpoint = output_dir / "checkpoints/latest.pt"
    train_report_path = output_dir / "train_report.json"

    blockers: list[str] = []
    metrics = _read_jsonl(metrics_path)
    setup_rows = _read_jsonl(setup_status_path)
    train_report = {}
    if train_report_path.exists():
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))

    if not metrics_path.exists():
        blockers.append("missing_training_metrics")
    elif len(metrics) < int(min_metric_rows):
        blockers.append("insufficient_metric_rows")

    last_metric = metrics[-1] if metrics else {}
    loss_value = _finite_number(last_metric.get("loss")) if last_metric else None
    if metrics and loss_value is None:
        blockers.append("last_loss_not_finite")

    if require_latest_checkpoint and not latest_checkpoint.exists():
        blockers.append("missing_latest_checkpoint")

    if train_report and train_report.get("status") not in {"ok", None}:
        blockers.append("train_report_not_ok")

    setup_phases = [str(row.get("phase") or "") for row in setup_rows]
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "output_dir": str(output_dir),
        "training_metrics_path": str(metrics_path),
        "metric_rows": len(metrics),
        "min_metric_rows": int(min_metric_rows),
        "last_metric": last_metric,
        "setup_status_path": str(setup_status_path),
        "setup_phase_count": len(setup_phases),
        "last_setup_phase": setup_phases[-1] if setup_phases else None,
        "latest_checkpoint": str(latest_checkpoint),
        "latest_checkpoint_exists": latest_checkpoint.exists(),
        "train_report_path": str(train_report_path),
        "train_report_exists": train_report_path.exists(),
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit early health of a CLSTR training or smoke output directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_metric_rows", type=int, default=1)
    parser.add_argument("--no_require_latest_checkpoint", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_training_health(
        output_dir=args.output_dir,
        min_metric_rows=args.min_metric_rows,
        require_latest_checkpoint=not args.no_require_latest_checkpoint,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
