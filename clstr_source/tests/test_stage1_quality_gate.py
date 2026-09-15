import json
import subprocess
import sys
from pathlib import Path

import torch

from clstr.stage1_quality_gate import audit_stage1_retrieval_warmup_quality


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_checkpoint(path: Path, *, sampling_strategy: str = "batch_stride", shuffle_queries: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "step": 4,
            "sampling_strategy": sampling_strategy,
            "shuffle_queries": shuffle_queries,
            "train_safety": {
                "unified_v2_train_safe_retrieval": True,
                "excluded_retrieval_splits": ["public_train_or_eval_unlabeled"],
            },
            "model_state_dict": {"skill_table.E": torch.zeros(4, 8)},
        },
        path,
    )


def test_stage1_quality_gate_accepts_learning_signal_and_source_coverage(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints" / "clstr_unified_retrieval_v2-step4.pt"
    _write_checkpoint(checkpoint)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 2.2,
                "recall_at_50": 0.0,
                "query_ids": ["q-skillret-1", "q-toolret-1"],
                "query_sources": ["skillret", "toolret_training"],
            },
            {
                "step": 2,
                "loss": 2.0,
                "recall_at_50": 0.25,
                "query_ids": ["q-toolbench-1", "q-traject-1"],
                "query_sources": ["toolbench_g3", "traject_bench"],
            },
            {
                "step": 3,
                "loss": 1.5,
                "recall_at_50": 0.5,
                "query_ids": ["q-skillret-2", "q-toolret-2"],
                "query_sources": ["skillret", "toolret_training"],
            },
            {
                "step": 4,
                "loss": 1.2,
                "recall_at_50": 0.5,
                "query_ids": ["q-toolbench-2", "q-traject-2"],
                "query_sources": ["toolbench_g3", "traject_bench"],
            },
        ],
    )

    report = audit_stage1_retrieval_warmup_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=4,
        first_window=2,
        last_window=2,
        min_loss_drop=0.2,
        min_last_recall=0.1,
        expected_sources=["skillret", "toolret_training", "toolbench_g3", "traject_bench"],
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["metrics_summary"]["loss_drop"] > 0.2
    assert report["source_coverage"]["observed_sources"] == [
        "skillret",
        "toolbench_g3",
        "toolret_training",
        "traject_bench",
    ]
    assert report["checkpoint_metadata"]["sampling_strategy"] == "batch_stride"


def test_stage1_quality_gate_rejects_old_sliding_metrics_without_learning_signal(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints" / "clstr_unified_retrieval_v2-step4.pt"
    _write_checkpoint(checkpoint, sampling_strategy="sliding_one_row", shuffle_queries=False)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [
            {"step": 1, "loss": 10.21, "recall_at_50": 0.0},
            {"step": 2, "loss": 10.20, "recall_at_50": 0.0},
            {"step": 3, "loss": 10.22, "recall_at_50": 0.0},
            {"step": 4, "loss": 10.21, "recall_at_50": 0.0},
        ],
    )

    report = audit_stage1_retrieval_warmup_quality(
        output_dir=output_dir,
        checkpoint_path=checkpoint,
        min_steps=4,
        first_window=2,
        last_window=2,
        min_loss_drop=0.05,
        min_last_recall=0.02,
        expected_sources=["skillret", "toolret_training"],
    )

    assert report["status"] == "action_required"
    assert "missing_query_source_metrics" in report["blockers"]
    assert "insufficient_learning_signal" in report["blockers"]
    assert "stage1_not_batch_stride_sampled" in report["blockers"]
    assert "stage1_queries_not_shuffled" in report["blockers"]


def test_stage1_quality_gate_cli_writes_report_and_fails_on_action_required(tmp_path):
    output_dir = tmp_path / "stage1"
    checkpoint = output_dir / "checkpoints" / "clstr_unified_retrieval_v2-step4.pt"
    report_path = tmp_path / "gate.json"
    _write_checkpoint(checkpoint, sampling_strategy="sliding_one_row", shuffle_queries=False)
    _write_jsonl(
        output_dir / "training_metrics.jsonl",
        [{"step": 1, "loss": 10.0, "recall_at_50": 0.0}],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_stage1_quality.py",
            "--output_dir",
            str(output_dir),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(report_path),
            "--min_steps",
            "4",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
