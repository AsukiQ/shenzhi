from __future__ import annotations

from pathlib import Path
import runpy

import torch


SCRIPT = Path("scripts/audit_stage0_frozen_backbone_cache.py")
SBATCH = Path("scripts/sbatch/run_qwen06_stage0_cache_benchmark.sh")


def _namespace():
    return runpy.run_path(str(SCRIPT))


def test_cache_audit_reports_tensor_difference_statistics():
    difference = _namespace()["tensor_difference"]

    report = difference(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 2.0]),
    )

    assert report == {
        "max_abs_diff": 1.0,
        "mean_abs_diff": 0.5,
    }


def test_cache_gate_requires_parity_speed_and_frozen_backbone():
    decide = _namespace()["decide_cache_gate"]
    parity = {
        "pooled_max_abs_diff": 0.0,
        "projected_max_abs_diff": 0.0,
        "logits_max_abs_diff": 0.0,
        "loss_abs_diff": 0.0,
        "gradient_max_abs_diff": 0.0,
        "parameter_update_max_abs_diff": 0.0,
    }

    accepted = decide(
        parity=parity,
        steady_state_speedup=1.6,
        projected_end_to_end_speedup=1.7,
        backbone_trainable_parameter_count=0,
    )
    slow = decide(
        parity=parity,
        steady_state_speedup=1.4,
        projected_end_to_end_speedup=1.7,
        backbone_trainable_parameter_count=0,
    )
    unfrozen = decide(
        parity=parity,
        steady_state_speedup=1.6,
        projected_end_to_end_speedup=1.7,
        backbone_trainable_parameter_count=1,
    )

    assert accepted["status"] == "ok"
    assert slow["status"] == "blocked"
    assert "steady_state_speedup_below_1.5" in slow["blockers"]
    assert unfrozen["status"] == "blocked"
    assert "backbone_not_fully_frozen" in unfrozen["blockers"]


def test_cache_audit_and_slurm_wrapper_expose_required_evidence():
    script = SCRIPT.read_text(encoding="utf-8")
    shell = SBATCH.read_text(encoding="utf-8")

    for key in (
        "pooled_max_abs_diff",
        "projected_max_abs_diff",
        "logits_max_abs_diff",
        "loss_abs_diff",
        "gradient_max_abs_diff",
        "parameter_update_max_abs_diff",
        "steady_state_speedup",
        "projected_end_to_end_speedup",
        "peak_gpu_memory_bytes",
        "backbone_trainable_parameter_count",
        "verified_resume_skill_table",
    ):
        assert key in script
    assert "#SBATCH --gpus=1" in shell
    assert "audit_stage0_frozen_backbone_cache.py" in shell
    assert "TRAIN_ENCODER_BACKBONE=0" in shell
