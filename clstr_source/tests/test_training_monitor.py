from pathlib import Path
import json


def test_training_monitor_loss_curve_includes_smoothed_loss_and_recall(tmp_path: Path) -> None:
    from clstr.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, curve_interval_steps=1)

    for step, loss, recall in [
        (1, 10.4, 0.05),
        (2, 10.0, 0.15),
        (3, 10.2, 0.10),
        (4, 9.6, 0.35),
        (5, 9.8, 0.30),
        (6, 9.2, 0.45),
    ]:
        monitor.record({"step": step, "loss": loss, "recall_at_50": recall})

    svg = monitor.loss_curve_path.read_text(encoding="utf-8")

    assert "raw loss" in svg
    assert "rolling loss" in svg
    assert "ema loss" in svg
    assert "recall_at_50" in svg
    assert "right axis: recall" in svg


def test_training_monitor_writes_component_diagnostic_curves(tmp_path: Path) -> None:
    from clstr.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, curve_interval_steps=1)

    for step in range(1, 6):
        monitor.record(
            {
                "step": step,
                "loss": 10.0 - step * 0.1,
                "policy_ce_loss": 2.0 - step * 0.03,
                "transition_skill_ce_loss": 4.0 - step * 0.05,
                "transition_skill_recall@5": 0.4 + step * 0.02,
                "stage0_prior_transition_skill_recall@5": 0.5,
                "batch_transition_switch_real_rows": 4,
                "batch_policy_rows": 4,
                "weighted_loss_terms": {"L_policy": 0.3, "L_trans_skill_ce": 1.2},
            }
        )

    paths = monitor.paths_report()
    diagnostic_path = Path(paths["diagnostic_curves_path"])
    svg = diagnostic_path.read_text(encoding="utf-8")

    assert diagnostic_path.exists()
    assert "Component diagnostics" in svg
    assert "policy_ce_loss" in svg
    assert "transition_skill_ce_loss" in svg
    assert "transition_skill_recall@5" in svg
    assert "stage0_prior_transition_skill_recall@5" in svg
    assert "weighted_loss_terms.L_trans_skill_ce" in svg
    assert "batch_transition_switch_real_rows" in svg


def test_training_monitor_writes_progress_json_for_latest_step(tmp_path: Path) -> None:
    from clstr.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, curve_interval_steps=100)

    monitor.record({"step": 1, "loss": 3.0, "train_recall@1": 0.25})
    monitor.record({"step": 2, "loss": 2.5, "train_recall@1": 0.5})

    progress_path = tmp_path / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    assert progress["status"] == "running"
    assert progress["step"] == 2
    assert progress["loss"] == 2.5
    assert progress["train_recall@1"] == 0.5
    assert progress["metrics_path"] == str(tmp_path / "training_metrics.jsonl")


def test_training_monitor_throttles_checkpoint_and_curve_writes(tmp_path: Path) -> None:
    import torch

    from clstr.training_monitor import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, checkpoint_interval_steps=400, curve_interval_steps=400)

    monitor.record({"step": 1, "loss": 3.0}, {"step": 1, "model_state_dict": {}})
    monitor.record({"step": 399, "loss": 2.0}, {"step": 399, "model_state_dict": {}})

    assert not monitor.latest_checkpoint_path.exists()
    assert "raw loss" not in monitor.loss_curve_path.read_text(encoding="utf-8")

    monitor.record({"step": 400, "loss": 1.0}, {"step": 400, "model_state_dict": {}})

    assert torch.load(monitor.latest_checkpoint_path, map_location="cpu")["step"] == 400
    assert "raw loss" in monitor.loss_curve_path.read_text(encoding="utf-8")
