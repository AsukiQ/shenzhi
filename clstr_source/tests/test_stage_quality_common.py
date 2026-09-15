import json
from pathlib import Path

from clstr.stage_quality_common import (
    artifact_exists,
    benchmark_counts,
    read_json_file,
    read_jsonl_file,
    source_coverage,
    window_mean,
    write_json_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage_quality_common_handles_gate_files_and_metrics(tmp_path):
    report_path = tmp_path / "reports" / "gate.json"
    metrics_path = tmp_path / "metrics.jsonl"
    artifact_path = tmp_path / "nonempty.pt"
    empty_artifact_path = tmp_path / "empty.pt"

    write_json_report(report_path, {"status": "ok", "nested": {"count": 1}})
    _write_jsonl(
        metrics_path,
        [
            {"step": 1, "loss": 3.0, "query_sources": ["toolret_training"]},
            {"step": 2, "loss": 2.0, "query_sources": ["skillret", "traject_bench"]},
            {"step": 3, "loss": "ignored", "query_sources": []},
            {"step": 4, "loss": 1.0, "query_sources": ["toolbench_g3"]},
        ],
    )
    artifact_path.write_text("checkpoint", encoding="utf-8")
    empty_artifact_path.write_text("", encoding="utf-8")

    assert read_json_file(report_path)["nested"]["count"] == 1
    assert read_json_file(tmp_path / "missing.json") == {}
    assert len(read_jsonl_file(metrics_path)) == 4
    assert read_jsonl_file(tmp_path / "missing.jsonl") == []
    assert window_mean(read_jsonl_file(metrics_path), "loss", 2, tail=False) == 2.5
    assert window_mean(read_jsonl_file(metrics_path), "loss", 2, tail=True) == 1.0
    assert source_coverage(read_jsonl_file(metrics_path)) == (
        ["skillret", "toolbench_g3", "toolret_training", "traject_bench"],
        3,
    )
    assert artifact_exists(artifact_path) is True
    assert artifact_exists(empty_artifact_path) is False
    assert artifact_exists(None) is False
    assert benchmark_counts({"data_report": {"benchmark_counts": {"a": "2", "b": "bad"}}}) == {"a": 2, "b": 0}
