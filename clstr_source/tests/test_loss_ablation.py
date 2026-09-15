import json
import sys
from pathlib import Path

from clstr.loss_ablation import (
    ABLATION_CONFIGS,
    build_ablation_action_trace_diagnostic,
    build_ablation_summary,
    recommended_slim_config,
)


def test_ablation_configs_include_required_loss_variants():
    names = {cfg["name"] for cfg in ABLATION_CONFIGS}

    assert {
        "policy_only",
        "policy_plus_hard_negative",
        "policy_plus_q_success",
        "policy_plus_q_success_trans_skill_ce",
        "policy_plus_q_success_stop",
        "policy_plus_q_success_belief",
        "policy_plus_q_success_routing",
        "full",
        "full_minus_observation_cosine",
        "full_minus_STOP",
        "full_minus_belief",
        "full_minus_routing",
        "full_minus_trans_skill_ce",
    }.issubset(names)


def test_build_ablation_summary_selects_closed_loop_best_and_writes_outputs(tmp_path):
    reports = [
        {
            "name": "policy_only",
            "output_dir": "outputs/a/policy_only",
            "checkpoint": "outputs/a/policy_only/checkpoints/ckpt.pt",
            "train_report": {"loss_weights": {"L_policy": 1.0}},
            "offline_report": {"policy_recall@1": 0.4},
            "eval_metrics": {"success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 50.0, "episodes": 20},
        },
        {
            "name": "policy_plus_q_success",
            "output_dir": "outputs/a/policy_plus_q_success",
            "checkpoint": "outputs/a/policy_plus_q_success/checkpoints/ckpt.pt",
            "train_report": {"loss_weights": {"L_policy": 1.0, "Q_success": 0.5}},
            "offline_report": {"policy_recall@1": 0.5},
            "eval_metrics": {"success_rate": 0.1, "average_reward": 0.1, "average_episode_steps": 46.0, "episodes": 20},
        },
    ]

    summary = build_ablation_summary(
        ablation_reports=reports,
        output_dir=tmp_path,
    )

    assert summary["best_by_success_rate"]["name"] == "policy_plus_q_success"
    assert (tmp_path / "ablation_summary.json").exists()
    assert (tmp_path / "ablation_table.md").exists()
    assert (tmp_path / "recommended_slim_loss_config.json").exists()
    slim = json.loads((tmp_path / "recommended_slim_loss_config.json").read_text(encoding="utf-8"))
    assert slim["source_ablation"] == "policy_plus_q_success"
    assert slim["loss_weights"]["Q_success"] == 0.5


def test_recommended_slim_config_prefers_fewer_losses_when_success_ties():
    chosen = recommended_slim_config(
        [
            {"name": "full", "eval_metrics": {"success_rate": 0.1}, "train_report": {"loss_weights": {"L_policy": 1.0, "Q_success": 0.5, "STOP": 0.2}}},
            {"name": "policy_plus_q_success", "eval_metrics": {"success_rate": 0.1}, "train_report": {"loss_weights": {"L_policy": 1.0, "Q_success": 0.5}}},
        ]
    )

    assert chosen["source_ablation"] == "policy_plus_q_success"


def test_ablation_runner_releases_training_cuda_before_eval(tmp_path, monkeypatch):
    import scripts.run_clstr_loss_ablation as runner

    calls = []
    train_kwargs = {}

    monkeypatch.setattr(runner, "ABLATION_CONFIGS", [{"name": "policy_only", "loss_weights": {"L_policy": 1.0}}])
    monkeypatch.setattr(
        runner,
        "run_legacy_clstr_full_base_train_from_routing_init",
        lambda **kwargs: train_kwargs.update(kwargs)
        or calls.append("train")
        or {
            "checkpoint": str(tmp_path / "checkpoint.pt"),
            "metrics": {},
            "loss_activation_counts": {},
            "loss_weights": kwargs["loss_weights"],
        },
    )
    monkeypatch.setattr(runner, "_release_training_memory", lambda: calls.append("release"), raising=False)
    monkeypatch.setattr(runner, "_run_eval", lambda **kwargs: calls.append("eval") or {"success_rate": 0.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_clstr_loss_ablation.py",
            "--train_path",
            str(tmp_path / "train.jsonl"),
            "--skills_path",
            str(tmp_path / "skills.jsonl"),
            "--output_dir",
            str(tmp_path / "outputs"),
            "--routing_init_manifest",
            str(tmp_path / "manifest.json"),
            "--official_repo",
            "/tmp/alfworld_repo",
            "--data_dir",
            "/tmp/alfworld_data",
            "--limit",
            "1",
            "--skill_text_format",
            "clstr_enriched",
        ],
    )

    runner.main()

    assert calls[:3] == ["train", "release", "eval"]
    assert train_kwargs["skill_text_format"] == "clstr_enriched"


def test_ablation_runner_skips_completed_eval_when_resuming(tmp_path, monkeypatch):
    import scripts.run_clstr_loss_ablation as runner

    calls = []
    output_dir = tmp_path / "outputs"
    run_output = output_dir / "policy_only"
    eval_dir = output_dir / "policy_only_valid_seen_gate"
    run_output.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (run_output / "train_report.json").write_text(
        json.dumps({"checkpoint": "existing.pt", "loss_weights": {"L_policy": 1.0}, "metrics": {}}),
        encoding="utf-8",
    )
    (run_output / "offline_report.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    (eval_dir / "metrics.json").write_text(json.dumps({"success_rate": 0.2, "average_reward": 0.2, "episodes": 20}), encoding="utf-8")

    monkeypatch.setattr(runner, "ABLATION_CONFIGS", [{"name": "policy_only", "loss_weights": {"L_policy": 1.0}}])
    monkeypatch.setattr(runner, "run_legacy_clstr_full_base_train_from_routing_init", lambda **kwargs: calls.append("train"))
    monkeypatch.setattr(runner, "_run_eval", lambda **kwargs: calls.append("eval"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_clstr_loss_ablation.py",
            "--output_dir",
            str(output_dir),
            "--limit",
            "1",
        ],
    )

    runner.main()

    assert calls == []
    summary = json.loads((output_dir / "ablation_summary.json").read_text(encoding="utf-8"))
    assert summary["best_by_success_rate"]["eval_metrics"]["success_rate"] == 0.2


def test_build_ablation_action_trace_diagnostic_reports_loop_and_qwen_acceptance(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "controller_diagnostic.json").write_text(
        json.dumps(
            {
                "action_trace_analysis": {
                    "episodes": 2,
                    "top_verbs": [["go", 8], ["open", 2]],
                    "stuck_patterns": {
                        "consecutive_repeat_ge_5_episodes": 1,
                        "single_action_only_episodes": 0,
                        "tail_two_action_cycle_episodes": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "run.jsonl").write_text(
        json.dumps(
            {
                "action_trace": ["go to fridge 1", "open fridge 1"],
                "policy_metadata_trace": [
                    {"qwen_proposed_action": "go to fridge 1", "qwen_parse_status": "exact"},
                    {"qwen_proposed_action": "look", "qwen_parse_status": "exact", "qwen_fallback_used": False},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostic = build_ablation_action_trace_diagnostic(
        eval_output_dir=eval_dir,
        train_report={"metrics": {"policy_expert_recall@1": 0.75}},
        offline_report={"metrics": {"policy_ce_loss": 0.4}},
        output_path=tmp_path / "action_trace_diagnostic.json",
    )

    assert diagnostic["top_verbs"][0] == ["go", 8]
    assert diagnostic["loop_or_repeat_episode_rate"] == 1.0
    assert diagnostic["offline_policy_expert_recall@1"] == 0.75
    assert diagnostic["qwen_proposal_stats"]["accepted_count"] == 1
    assert diagnostic["qwen_proposal_stats"]["rejected_count"] == 1
    assert (tmp_path / "action_trace_diagnostic.json").exists()
