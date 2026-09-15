from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clstr.qwen_clstr_stage0_gate import (
    audit_stage0_promotion,
    audit_stage0_release,
    audit_stage0_smoke,
    select_stage0_checkpoint,
)


def _handoff(global_100, global_500, tool_100, tool_500, traj_100, traj_200, traj_500):
    return {
        "query_modes": {
            "checkpoint_state_query": {
                "global": {
                    "next_recall@100": global_100,
                    "next_recall@500": global_500,
                },
                "benchmarks": {
                    "toolbench_g3": {
                        "next_recall@100": tool_100,
                        "next_recall@500": tool_500,
                    },
                    "traject_bench": {
                        "next_recall@100": traj_100,
                        "next_recall@200": traj_200,
                        "next_recall@500": traj_500,
                    },
                },
            }
        }
    }


def test_smoke_gate_requires_finite_loss_mined_pairs_and_frozen_unified_route():
    report = {
        "status": "ok",
        "metrics": {
            "loss": 1.0,
            "mined_hard_negative_rows": 8,
            "mined_hard_negative_pair_count": 256,
        },
        "stage0_protocol": {
            "route_scorer": "unified_memory",
            "state_query_prompt_version": "clstr_causal_state_v1",
            "state_query_max_chars": 2000,
            "state_query_truncation": "head_tail_v1",
        },
        "trainable_parameter_policy": {
            "trains_encoder_backbone": False,
            "optimizer_parameter_names": [
                "initial_belief_head.weight",
                "unified_retriever.query.weight",
            ],
        },
    }

    gate = audit_stage0_smoke(report, {"status": "ok"})

    assert gate["status"] == "ok"


def test_step1200_gate_requires_global_gain_and_hard_domain_safety():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    current = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.68)

    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=baseline,
        target_step=1200,
    )

    assert gate["status"] == "ok"


def test_later_segment_gate_uses_composite_or_hard_domain_gain():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    previous = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.69)
    current = _handoff(0.75, 0.90, 0.58, 0.83, 0.37, 0.53, 0.69)

    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=previous,
        target_step=2400,
    )

    assert gate["status"] == "ok"


def test_release_gate_marks_small_floor_miss_borderline_and_large_miss_failed():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    borderline = _handoff(0.76, 0.91, 0.60, 0.83, 0.40, 0.51, 0.69)
    failed = _handoff(0.76, 0.91, 0.60, 0.80, 0.40, 0.48, 0.65)

    assert audit_stage0_release(baseline, borderline, "step1200.pt")["status"] == "borderline"
    assert audit_stage0_release(baseline, failed, "step1200.pt")["status"] == "action_required"


def test_checkpoint_selection_chooses_highest_safe_composite():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    step1200 = _handoff(0.76, 0.91, 0.60, 0.85, 0.42, 0.54, 0.72)
    step2400 = _handoff(0.78, 0.92, 0.62, 0.86, 0.44, 0.56, 0.74)

    selection = select_stage0_checkpoint(
        baseline_report=baseline,
        candidates=[
            {
                "step": 1200,
                "report": step1200,
                "checkpoint_path": "step1200.pt",
                "lineage_manifest_path": "lineage-step1200.json",
            },
            {
                "step": 2400,
                "report": step2400,
                "checkpoint_path": "step2400.pt",
                "lineage_manifest_path": "lineage-step2400.json",
            },
        ],
    )

    assert selection["status"] == "ok"
    assert selection["selected_step"] == 2400
    assert selection["selected_checkpoint_path"] == "step2400.pt"
    assert selection["selected_lineage_path"] == "lineage-step2400.json"


def test_smoke_gate_rejects_nonfinite_loss_and_backbone_optimizer():
    report = {
        "status": "ok",
        "metrics": {
            "loss": float("nan"),
            "mined_hard_negative_rows": 8,
            "mined_hard_negative_pair_count": 256,
        },
        "stage0_protocol": {
            "route_scorer": "unified_memory",
            "state_query_prompt_version": "clstr_causal_state_v1",
            "state_query_max_chars": 2000,
            "state_query_truncation": "head_tail_v1",
        },
        "trainable_parameter_policy": {
            "trains_encoder_backbone": True,
            "optimizer_parameter_names": [
                "initial_belief_head.weight",
                "unified_retriever.query.weight",
                "encoder.backbone.layers.0.weight",
            ],
        },
    }

    gate = audit_stage0_smoke(report, {"status": "ok"})

    assert gate["status"] == "action_required"
    assert "finite_loss" in gate["failed_checks"]
    assert "encoder_backbone_frozen" in gate["failed_checks"]
    assert "encoder_backbone_not_optimized" in gate["failed_checks"]


def test_later_segment_gate_rejects_recall500_regression():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    previous = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.69)
    current = _handoff(0.80, 0.88, 0.60, 0.81, 0.40, 0.55, 0.67)

    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=previous,
        target_step=2400,
    )

    assert gate["status"] == "action_required"
    assert "global_next500_safe" in gate["failed_checks"]
    assert "toolbench_g3_next500_safe" in gate["failed_checks"]
    assert "traject_bench_next500_safe" in gate["failed_checks"]


def test_later_segment_gate_accepts_exactly_one_percent_recall500_regression():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    previous = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.69)
    current = _handoff(0.75, 0.89, 0.58, 0.82, 0.37, 0.53, 0.68)

    gate = audit_stage0_promotion(
        baseline_report=baseline,
        current_report=current,
        previous_report=previous,
        target_step=2400,
    )

    assert gate["status"] == "ok"


def test_checkpoint_selection_excludes_borderline_candidate():
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    borderline_high = _handoff(0.90, 0.95, 0.90, 0.83, 0.90, 0.51, 0.69)
    safe_lower = _handoff(0.76, 0.91, 0.60, 0.85, 0.42, 0.54, 0.72)

    selection = select_stage0_checkpoint(
        baseline_report=baseline,
        candidates=[
            {
                "step": 1200,
                "report": borderline_high,
                "checkpoint_path": "borderline.pt",
                "lineage_manifest_path": "borderline.json",
            },
            {
                "step": 2400,
                "report": safe_lower,
                "checkpoint_path": "safe.pt",
                "lineage_manifest_path": "safe.json",
            },
        ],
    )

    assert selection["status"] == "ok"
    assert selection["selected_step"] == 2400
    assert selection["selected_checkpoint_path"] == "safe.pt"


def test_stage0_gate_cli_promotes_and_returns_two_on_failed_gate(tmp_path):
    baseline = _handoff(0.70, 0.88, 0.55, 0.82, 0.35, 0.50, 0.68)
    current = _handoff(0.74, 0.90, 0.57, 0.83, 0.36, 0.52, 0.68)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output_path = tmp_path / "promotion.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    script = Path("scripts/audit_qwen06_clstr_stage0_gate.py")
    command = [
        sys.executable,
        str(script),
        "promote",
        "--baseline_report",
        str(baseline_path),
        "--current_report",
        str(current_path),
        "--previous_report",
        str(baseline_path),
        "--target_step",
        "1200",
        "--output_path",
        str(output_path),
    ]

    passed = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True)

    assert passed.returncode == 0, passed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "ok"

    current_path.write_text(json.dumps(baseline), encoding="utf-8")
    failed = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True)

    assert failed.returncode == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "action_required"
