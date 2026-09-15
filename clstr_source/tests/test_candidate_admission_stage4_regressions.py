import json
from pathlib import Path

from clstr.stage4_quality_gate import audit_stage4_act_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_candidate_admission_quality_gate_uses_unified_memory_scoring(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate_admission"
    checkpoint = output_dir / "checkpoints" / "step1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    metrics = {
        "step": 1,
        "stage4_total_loss": 1.0,
        "stage4_act_count": 2.0,
        "stage4_post_action_update_rows": 2.0,
        "stage4_next_state_rows": 2.0,
        "route_scorer": "unified_memory",
        "transition_scoring_mode": "unified_memory",
        "transition_residual_lambda": 0.0,
    }
    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    report = {
        "status": "ok",
        "stage": "clstr_stage4_transition_conditioned_next_skill",
        "stage4_method": "candidate_admission_residual_v1",
        "training_objective": "candidate_admission_constrained_residual",
        "training_regime": "offline_train_split_causal_next_skill",
        "valid_or_test_used_for_training": False,
        "on_policy_rollout_used": False,
        "checkpoint": str(checkpoint),
        "training_metrics_path": str(metrics_path),
        "route_scorer": "unified_memory",
        "transition_scoring_mode": "unified_memory",
        "transition_residual_lambda": 0.0,
        "freeze_report": {
            "frozen_routing_foundation": True,
            "train_transition": False,
            "trainable_modules": ["route_memory_candidate_admission_residual"],
            "frozen_cmc_residual_adapter": True,
        },
        "checkpoint_init_report": {"protect_routing_foundation": True},
        "last_metrics": metrics,
    }
    report_path = output_dir / "train_report.json"
    _write_json(report_path, report)

    audit = audit_stage4_act_quality(
        output_dir,
        checkpoint_path=checkpoint,
        train_report_path=report_path,
        metrics_path=metrics_path,
        min_steps=1,
        require_stage0_handoff=False,
    )

    assert "stage4_transition_scoring_not_effective_route_mode" not in audit["blockers"]
    scoring_safety = audit["stage4_quality"]["transition_scoring_safety"]
    assert scoring_safety["matched"] is True
    assert scoring_safety["expected_mode"] == "unified_memory"


def test_candidate_admission_smoke_overrides_legacy_32_row_caps() -> None:
    text = (
        PROJECT_ROOT
        / "scripts"
        / "sbatch"
        / "run_qwen06_clstr_candidate_admission_stage4.sh"
    ).read_text(encoding="utf-8")

    assert (
        "SMOKE_BENCHMARK_CAPS=toolbench_g3=1024:traject_bench=1024:"
        "alfworld=1024:webshop=1024"
    ) in text
    assert 'export BENCHMARK_CAPS="${BENCHMARK_CAPS:-${SMOKE_BENCHMARK_CAPS}}"' in text
