from __future__ import annotations

import json
from pathlib import Path

from clstr.stage2_transition_error_audit import (
    audit_stage2_transition_outputs,
    summarize_eval_batches,
    summarize_training_trend,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_training_trend_reports_tail_delta_and_plateau_signal(tmp_path):
    metrics_path = tmp_path / "training_metrics.jsonl"
    rows = []
    for step in range(1, 1001):
        rows.append(
            {
                "step": step,
                "transition_skill_recall@5": 0.40 if step <= 500 else 0.405,
                "transition_skill_ce_loss": 3.0 if step <= 500 else 2.98,
                "loss": 5.0 if step <= 500 else 4.95,
            }
        )
    _write_jsonl(metrics_path, rows)

    trend = summarize_training_trend(metrics_path, first_window=100, tail_window=250)

    assert trend["metric_count"] == 1000
    assert trend["max_step"] == 1000
    assert trend["transition_skill_recall@5"]["first_mean"] == 0.4
    assert trend["transition_skill_recall@5"]["tail_delta"] == 0.0
    assert trend["continue_training_signal"] == "weak_tail_improvement"


def test_eval_batch_summary_uses_transition_counts_as_weights(tmp_path):
    metrics_path = tmp_path / "real_topm_eval_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "batch_index": 1,
                "batch_size": 8,
                "loss_mask_counts": {"L_trans_skill_ce": 2},
                "metrics": {"transition_skill_recall@5": 1.0, "transition_skill_ce_loss": 1.0},
            },
            {
                "batch_index": 2,
                "batch_size": 8,
                "loss_mask_counts": {"L_trans_skill_ce": 6},
                "metrics": {"transition_skill_recall@5": 0.0, "transition_skill_ce_loss": 3.0},
            },
        ],
    )

    summary = summarize_eval_batches(metrics_path)

    assert summary["transition_eval_rows"] == 8
    assert summary["weighted_metrics"]["transition_skill_recall@5"] == 0.25
    assert summary["weighted_metrics"]["transition_skill_ce_loss"] == 2.5
    assert summary["batch_distribution"]["transition_skill_recall@5"]["zero_batch_fraction"] == 0.5


def test_eval_batch_summary_uses_policy_counts_for_policy_metrics(tmp_path):
    metrics_path = tmp_path / "real_topm_eval_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "batch_index": 1,
                "batch_size": 8,
                "loss_mask_counts": {"L_policy": 2},
                "metrics": {"policy_expert_recall@1": 1.0},
            },
            {
                "batch_index": 2,
                "batch_size": 8,
                "loss_mask_counts": {"L_policy": 6},
                "metrics": {"policy_expert_recall@1": 0.0},
            },
        ],
    )

    summary = summarize_eval_batches(metrics_path)

    assert summary["metric_weights"]["policy_expert_recall@1"] == 8
    assert summary["weighted_metrics"]["policy_expert_recall@1"] == 0.25


def test_full_audit_compares_v3_against_baseline_and_flags_no_eval_gain(tmp_path):
    train_metrics = tmp_path / "train.jsonl"
    eval_metrics = tmp_path / "eval.jsonl"
    eval_report = tmp_path / "eval_report.json"
    baseline_report = tmp_path / "baseline_report.json"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    _write_jsonl(
        train_metrics,
        [
            {"step": 1, "transition_skill_recall@5": 0.48, "transition_skill_ce_loss": 3.2, "loss": 4.0},
            {"step": 2, "transition_skill_recall@5": 0.49, "transition_skill_ce_loss": 3.1, "loss": 3.9},
            {"step": 3, "transition_skill_recall@5": 0.49, "transition_skill_ce_loss": 3.1, "loss": 3.8},
            {"step": 4, "transition_skill_recall@5": 0.49, "transition_skill_ce_loss": 3.1, "loss": 3.7},
        ],
    )
    _write_jsonl(
        eval_metrics,
        [
            {
                "batch_index": 1,
                "batch_size": 8,
                "loss_mask_counts": {"L_trans_skill_ce": 8},
                "metrics": {"transition_skill_recall@5": 0.48, "transition_skill_recall@1": 0.20},
            }
        ],
    )
    _write_json(
        eval_report,
        {
            "status": "action_required",
            "blockers": ["transition_recall_at_5_below_threshold"],
            "stage0_candidate_handoff": {"current_positive_coverage@M": 0.9, "next_positive_coverage@M": 0.91},
            "aggregate": {
                "metrics": {"transition_skill_recall@5": 0.48, "transition_skill_recall@1": 0.20},
                "metric_weights": {"transition_skill_recall@5": 8},
            },
        },
    )
    _write_json(
        baseline_report,
        {
            "aggregate": {
                "metrics": {"transition_skill_recall@5": 0.479, "transition_skill_recall@1": 0.18},
                "metric_weights": {"transition_skill_recall@5": 8},
            }
        },
    )

    report = audit_stage2_transition_outputs(
        train_metrics_path=train_metrics,
        eval_metrics_path=eval_metrics,
        eval_report_path=eval_report,
        baseline_eval_report_path=baseline_report,
        output_json_path=output_json,
        output_markdown_path=output_md,
        first_window=1,
        tail_window=2,
    )

    assert report["status"] == "action_required"
    assert "no_meaningful_no_inject_recall5_gain" in report["findings"]
    assert report["baseline_comparison"]["transition_skill_recall@5_delta"] == 0.001
    assert output_json.exists()
    assert output_md.exists()
