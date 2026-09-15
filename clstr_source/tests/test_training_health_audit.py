from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_health_module():
    path = Path("scripts/audit_clstr_training_health.py")
    spec = importlib.util.spec_from_file_location("audit_clstr_training_health", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_training_health_audit_accepts_running_smoke_with_metrics_and_latest(tmp_path):
    output_dir = tmp_path / "out"
    _write_jsonl(output_dir / "setup_status.jsonl", [{"phase": "training_started"}])
    _write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.25}])
    latest = output_dir / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"checkpoint")

    audit = _load_health_module()
    report = audit.audit_training_health(output_dir=output_dir, min_metric_rows=1)

    assert report["status"] == "ok"
    assert report["metric_rows"] == 1
    assert report["latest_checkpoint_exists"] is True


def test_training_health_audit_blocks_when_metrics_are_missing(tmp_path):
    output_dir = tmp_path / "out"
    _write_jsonl(output_dir / "setup_status.jsonl", [{"phase": "training_started"}])

    audit = _load_health_module()
    report = audit.audit_training_health(output_dir=output_dir, min_metric_rows=1)

    assert report["status"] == "action_required"
    assert "missing_training_metrics" in report["blockers"]
    assert "missing_latest_checkpoint" in report["blockers"]
