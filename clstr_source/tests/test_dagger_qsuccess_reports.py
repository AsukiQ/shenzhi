import json
from pathlib import Path

from clstr.dagger_qsuccess_report import build_dagger_qsuccess_reports


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_dagger_qsuccess_reports_writes_comparison_and_paper_files(tmp_path):
    outputs = tmp_path / "outputs"
    data = tmp_path / "data"
    _write_json(
        outputs / "alfworld_eval" / "clstr_controller_gate_full_2k" / "metrics.json",
        {"status": "ok", "success_rate": 0.00365, "average_reward": 0.00365, "episodes": 274},
    )
    _write_json(
        outputs / "alfworld_eval" / "clstr_qwen3_structured_gate_valid_seen" / "metrics.json",
        {"status": "ok", "success_rate": 0.1, "average_reward": 0.1, "episodes": 20},
    )
    _write_json(
        outputs / "alfworld_eval" / "qwen3_8b_chat_available_full" / "metrics.json",
        {"status": "ok", "success_rate": 0.109489, "average_reward": 0.109489, "episodes": 274, "not_clstr_result": True},
    )
    _write_json(
        outputs / "alfworld_eval" / "clstr_dagger_qsuccess_gate_valid_seen" / "metrics.json",
        {"status": "ok", "success_rate": 0.15, "average_reward": 0.15, "episodes": 20},
    )
    _write_json(
        outputs / "clstr_loss_ablation" / "ablation_summary.json",
        {
            "status": "ok",
            "recommended_slim_loss_config": {"source_ablation": "policy_plus_q_success", "loss_weights": {"L_policy": 1.0, "Q_success": 0.5}},
            "best_by_success_rate": {"name": "policy_plus_q_success", "checkpoint": "outputs/a/checkpoints/ckpt.pt"},
        },
    )
    _write_json(
        outputs / "clstr_dagger_success_train" / "train_report.json",
        {"checkpoint": "outputs/clstr_dagger_success_train/checkpoints/clstr_full_base-step1000.pt", "metrics": {"policy_expert_recall@1": 0.7}},
    )
    _write_json(
        outputs / "alfworld_qwen3_expert_corrected_rollout" / "extraction_report.json",
        {"extracted_step_count": 100, "expert_action_in_admissible_rate": 1.0, "qwen_action_matches_expert_rate": 0.2},
    )
    _write_json(data / "alfworld_qwen3_expert_corrected_rollout" / "manifest.json", {"split": "train", "bucket": "dagger_expert_corrected_rollout"})

    summary = build_dagger_qsuccess_reports(output_root=outputs, data_root=data)

    assert summary["improvements"]["vs_0_6b_best"]["improved"] is True
    assert summary["improvements"]["vs_previous_structured_clstr_qwen"]["improved"] is True
    assert summary["improvements"]["vs_qwen_direct_reference"]["improved"] is True
    assert (outputs / "clstr_dagger_qsuccess_comparison_table.md").exists()
    assert (outputs / "clstr_dagger_qsuccess_comparison_summary.json").exists()
    assert (outputs / "clstr_dagger_qsuccess_paper_report.md").exists()
    assert (outputs / "clstr_dagger_qsuccess_paper_report.json").exists()
    paper = (outputs / "clstr_dagger_qsuccess_paper_report.md").read_text(encoding="utf-8")
    assert "Q_success is an estimated action-value" in paper
    assert "offline diagnostics are not closed-loop success" in paper


def test_build_dagger_qsuccess_reports_prefers_enriched_postfix_ablation_outputs(tmp_path):
    outputs = tmp_path / "outputs"
    data = tmp_path / "data"
    _write_json(outputs / "alfworld_eval" / "clstr_controller_gate_full_2k" / "metrics.json", {"status": "ok", "success_rate": 0.00365})
    _write_json(outputs / "alfworld_eval" / "clstr_qwen3_structured_gate_valid_seen" / "metrics.json", {"status": "ok", "success_rate": 0.1})
    _write_json(outputs / "alfworld_eval" / "qwen3_8b_chat_available_full" / "metrics.json", {"status": "ok", "success_rate": 0.109489, "not_clstr_result": True})
    _write_json(outputs / "alfworld_eval" / "clstr_dagger_qsuccess_gate_valid_seen" / "metrics.json", {"status": "ok", "success_rate": 0.15})
    _write_json(outputs / "alfworld_eval" / "clstr_dagger_qsuccess_full" / "metrics.json", {"status": "ok", "success_rate": 0.167883})
    _write_json(
        outputs / "clstr_loss_ablation" / "ablation_summary.json",
        {
            "status": "ok",
            "recommended_slim_loss_config": {"source_ablation": "old"},
            "best_by_success_rate": {"checkpoint": "old.pt"},
        },
    )
    _write_json(
        outputs / "clstr_loss_ablation_enriched" / "ablation_summary.json",
        {
            "status": "ok",
            "recommended_slim_loss_config": {"source_ablation": "policy_plus_q_success_trans_skill_ce"},
            "best_by_success_rate": {"checkpoint": "enriched.pt"},
        },
    )
    _write_json(
        outputs / "clstr_loss_ablation_enriched" / "recommended_slim_loss_config.json",
        {"source_ablation": "policy_plus_q_success_trans_skill_ce"},
    )
    _write_json(outputs / "clstr_dagger_success_train_enriched" / "train_report.json", {"checkpoint": "enriched_train.pt", "metrics": {"policy_expert_recall@1": 0.8}})
    _write_json(outputs / "alfworld_qwen3_expert_corrected_rollout" / "extraction_report.json", {"extracted_step_count": 100})
    _write_json(data / "alfworld_qwen3_expert_corrected_rollout" / "manifest.json", {"split": "train"})

    summary = build_dagger_qsuccess_reports(output_root=outputs, data_root=data)

    assert summary["ablation_summary_path"].endswith("clstr_loss_ablation_enriched/ablation_summary.json")
    assert summary["recommended_slim_loss_config"]["source_ablation"] == "policy_plus_q_success_trans_skill_ce"
    assert summary["checkpoint"] == "enriched.pt"
