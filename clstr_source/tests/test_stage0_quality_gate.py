from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.stage0_quality_gate import audit_stage0_biencoder_quality


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_stage0_checkpoint(path: Path, *, stage: str = "clstr_unified_stage0_biencoder", step: int = 5000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "step": step,
            "stage0_protocol": {"role": "SkillRouter-compatible bi-encoder coarse retriever"},
            "model_state_dict": {"skill_table.E": torch.zeros(2, 8)},
        },
        path,
    )


def write_eval_metrics(path: Path, *, recall20: float, recall50: float, recall100: float) -> None:
    write_json(
        path,
        {
            "status": "ok",
            "benchmark": "clstr_unified_stage0",
            "method": "clstr_unified_stage0_biencoder",
            "qrels_path": "baseline/qrels.jsonl",
            "run_path": "stage0/run.tsv",
            "run_format": "trec",
            "metrics": {
                "Recall@20": recall20,
                "Recall@50": recall50,
                "Recall@100": recall100,
            },
        },
    )


def test_stage0_quality_gate_rejects_top1_only_improvement_without_coarse_recall(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/stage0-step5000.pt"
    baseline_dir = tmp_path / "baseline"
    write_stage0_checkpoint(checkpoint)
    write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 2.0, "recall_at_1": 0.1, "recall_at_20": 0.1, "recall_at_50": 0.1, "recall_at_100": 0.1},
            {"step": 5000, "loss": 1.0, "recall_at_1": 0.9, "recall_at_20": 0.2, "recall_at_50": 0.3, "recall_at_100": 0.4},
        ],
    )
    (output_dir / "loss_curve.svg").write_text("<svg/>", encoding="utf-8")
    write_json(
        baseline_dir / "metrics.json",
        {
            "status": "ok",
            "method": "stage0_skillrouter_frozen_baseline",
            "metrics": {"Recall@20": 0.5, "Recall@50": 0.7, "Recall@100": 0.8},
        },
    )
    write_eval_metrics(output_dir / "full_retrieval_eval" / "metrics.json", recall20=0.2, recall50=0.3, recall100=0.4)

    report = audit_stage0_biencoder_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        baseline_metrics_path=baseline_dir / "metrics.json",
        min_steps=5000,
        first_window=1,
        last_window=1,
        min_loss_drop=0.1,
        baseline_tolerance=0.01,
    )

    assert report["status"] == "action_required"
    assert "stage0_recall_at_20_below_same_pool_baseline" in report["blockers"]
    assert "stage0_recall_at_50_below_same_pool_baseline" in report["blockers"]
    assert "stage0_recall_at_100_below_same_pool_baseline" in report["blockers"]
    assert "recall_at_1" not in report["coarse_recall_summary"]["required_recall_keys"]


def test_stage0_quality_gate_requires_full_eval_metrics_before_baseline_comparison(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/stage0-step5000.pt"
    baseline_dir = tmp_path / "baseline"
    write_stage0_checkpoint(checkpoint)
    write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 3.0, "recall_at_20": 0.7, "recall_at_50": 0.8, "recall_at_100": 0.9},
            {"step": 5000, "loss": 1.0, "recall_at_20": 0.99, "recall_at_50": 0.99, "recall_at_100": 0.99},
        ],
    )
    (output_dir / "loss_curve.svg").write_text("<svg/>", encoding="utf-8")
    write_json(
        baseline_dir / "metrics.json",
        {
            "status": "ok",
            "method": "stage0_skillrouter_frozen_baseline",
            "metrics": {"Recall@20": 0.1, "Recall@50": 0.1, "Recall@100": 0.1},
        },
    )

    report = audit_stage0_biencoder_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        baseline_metrics_path=baseline_dir / "metrics.json",
        min_steps=5000,
        first_window=1,
        last_window=1,
        min_loss_drop=0.1,
    )

    assert report["status"] == "action_required"
    assert "missing_stage0_full_eval_metrics" in report["blockers"]
    assert report["coarse_recall_summary"]["stage0_eval"] == {}
    assert report["coarse_recall_summary"]["training_batch_tail"]["recall_at_50"] == 0.99


def test_stage0_quality_gate_accepts_coarse_recall_and_loss_curve(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/stage0-step5000.pt"
    baseline_dir = tmp_path / "baseline"
    write_stage0_checkpoint(checkpoint)
    write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 3.0,
                "recall_at_20": 0.4,
                "recall_at_50": 0.5,
                "recall_at_100": 0.6,
                "query_sources": ["skillret", "toolret_training"],
            },
            {
                "step": 5000,
                "loss": 1.0,
                "recall_at_20": 0.62,
                "recall_at_50": 0.76,
                "recall_at_100": 0.87,
                "query_sources": ["toolbench_g3", "traject_bench"],
            },
        ],
    )
    (output_dir / "loss_curve.svg").write_text("<svg/>", encoding="utf-8")
    write_json(
        baseline_dir / "metrics.json",
        {
            "status": "ok",
            "method": "stage0_skillrouter_frozen_baseline",
            "metrics": {"Recall@20": 0.6, "Recall@50": 0.75, "Recall@100": 0.88},
        },
    )
    write_eval_metrics(output_dir / "full_retrieval_eval" / "metrics.json", recall20=0.62, recall50=0.76, recall100=0.87)

    report = audit_stage0_biencoder_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        baseline_metrics_path=baseline_dir / "metrics.json",
        min_steps=5000,
        first_window=1,
        last_window=1,
        min_loss_drop=0.1,
        baseline_tolerance=0.02,
        expected_sources=["skillret", "toolret_training", "toolbench_g3", "traject_bench"],
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["coarse_recall_summary"]["stage0_eval"]["Recall@50"] == 0.76
    assert report["coarse_recall_summary"]["training_batch_tail"]["recall_at_50"] == 0.76
    assert report["coarse_recall_summary"]["baseline"]["Recall@50"] == 0.75
    assert report["source_coverage"]["missing_sources"] == []


def test_stage0_quality_gate_accepts_lowercase_recall_metrics_from_train_report_payload(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/stage0-step5000.pt"
    baseline_dir = tmp_path / "baseline"
    write_stage0_checkpoint(checkpoint)
    write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 4.0, "recall_at_20": 0.2, "recall_at_50": 0.3, "recall_at_100": 0.4},
            {"step": 5000, "loss": 1.0, "recall_at_20": 0.72, "recall_at_50": 0.83, "recall_at_100": 0.91},
        ],
    )
    (output_dir / "loss_curve.svg").write_text("<svg/>", encoding="utf-8")
    write_json(
        baseline_dir / "train_report.json",
        {
            "status": "ok",
            "metrics": {"recall_at_20": 0.70, "recall_at_50": 0.80, "recall_at_100": 0.90},
        },
    )
    write_json(
        output_dir / "train_report.json",
        {
            "status": "ok",
            "metrics": {"recall_at_20": 0.72, "recall_at_50": 0.83, "recall_at_100": 0.91},
        },
    )

    report = audit_stage0_biencoder_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        stage0_eval_metrics_path=output_dir / "train_report.json",
        baseline_metrics_path=baseline_dir / "train_report.json",
        min_steps=5000,
        first_window=1,
        last_window=1,
        min_loss_drop=0.1,
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["coarse_recall_summary"]["stage0_eval"]["Recall@50"] == 0.83
    assert report["coarse_recall_summary"]["baseline"]["Recall@50"] == 0.80


def test_stage0_quality_gate_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/stage0-step1.pt"
    report_path = tmp_path / "stage0_gate.json"
    write_stage0_checkpoint(checkpoint, step=1)
    write_jsonl(output_dir / "training_metrics.jsonl", [{"step": 1, "loss": 1.0, "recall_at_20": 0.0}])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage0_quality.py",
            "--output_dir",
            str(output_dir),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(report_path),
            "--min_steps",
            "5000",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
    assert "insufficient_training_steps" in report["blockers"]


def test_stage0_quality_gate_default_checkpoint_matches_stage0_training_name(tmp_path):
    output_dir = tmp_path / "stage0"
    checkpoint = output_dir / "checkpoints/clstr_unified_retrieval_v2-step5000.pt"
    baseline_dir = tmp_path / "baseline"
    write_stage0_checkpoint(checkpoint, stage="clstr_unified_retrieval_v2", step=5000)
    write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.0,
                "recall_at_20": 0.70,
                "recall_at_50": 0.72,
                "recall_at_100": 0.74,
            },
            {
                "step": 5000,
                "loss": 1.0,
                "recall_at_20": 0.80,
                "recall_at_50": 0.82,
                "recall_at_100": 0.84,
                "query_sources": ["skillret", "toolret_training", "toolbench_g3", "traject_bench"],
            },
        ],
    )
    (output_dir / "loss_curve.svg").write_text("<svg/>", encoding="utf-8")
    write_json(
        baseline_dir / "metrics.json",
        {
            "status": "ok",
            "method": "stage0_skillrouter_frozen_baseline",
            "metrics": {"Recall@20": 0.70, "Recall@50": 0.72, "Recall@100": 0.74},
        },
    )
    write_eval_metrics(output_dir / "full_retrieval_eval" / "metrics.json", recall20=0.80, recall50=0.82, recall100=0.84)

    report = audit_stage0_biencoder_quality(
        output_dir=output_dir,
        baseline_metrics_path=baseline_dir / "metrics.json",
        first_window=1,
        last_window=1,
        min_loss_drop=0.1,
        expected_sources=["skillret", "toolret_training", "toolbench_g3", "traject_bench"],
    )

    assert report["checkpoint_path"] == str(checkpoint)
    assert report["status"] == "ok"
