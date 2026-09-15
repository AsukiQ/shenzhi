import inspect

from clstr.stage2_real_topm_eval import (
    aggregate_eval_metrics,
    build_real_topm_report,
    evaluate_stage2_real_topm,
    _select_real_topm_eval_rows,
)


def test_real_topm_report_rejects_training_time_injection():
    report = build_real_topm_report(
        output_dir="outputs/eval",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        train_path="train.jsonl",
        skills_path="skills.jsonl",
        handoff_report={
            "positive_missing_policy": "inject",
            "injected_positive_rows": 3,
            "current_positive_coverage@M": 0.9,
            "next_positive_coverage@M": 0.88,
            "retained_rows": 10,
            "skipped_rows": 0,
        },
        aggregate_report={"metrics": {"transition_skill_recall@5": 0.7}, "metric_weights": {"transition_skill_recall@5": 8}},
        min_current_coverage=0.85,
        min_next_coverage=0.85,
        min_transition_recall_at_5=0.5,
    )

    assert report["status"] == "action_required"
    assert "not_real_topm_no_inject_policy" in report["blockers"]
    assert "gold_positive_injected" in report["blockers"]


def test_aggregate_eval_metrics_weights_by_active_loss_counts():
    aggregate = aggregate_eval_metrics(
        [
            {
                "metrics": {"transition_skill_recall@5": 1.0, "policy_expert_recall@1": 0.0},
                "loss_mask_counts": {"L_trans_skill_ce": 1, "L_policy": 2},
                "batch_size": 2,
            },
            {
                "metrics": {"transition_skill_recall@5": 0.0, "policy_expert_recall@1": 1.0},
                "loss_mask_counts": {"L_trans_skill_ce": 3, "L_policy": 2},
                "batch_size": 4,
            },
        ]
    )

    assert aggregate["metrics"]["transition_skill_recall@5"] == 0.25
    assert aggregate["metric_weights"]["transition_skill_recall@5"] == 4
    assert aggregate["metrics"]["policy_expert_recall@1"] == 0.5
    assert aggregate["metric_weights"]["policy_expert_recall@1"] == 4


def test_real_topm_eval_does_not_subset_full_eval_by_default():
    signature = inspect.signature(evaluate_stage2_real_topm)

    assert signature.parameters["max_eval_batches"].default is None
    assert signature.parameters["stage0_handoff_sample_multiplier"].default is None


def test_real_topm_eval_exposes_replay_prefix_memory_by_default():
    signature = inspect.signature(evaluate_stage2_real_topm)

    assert "auto_replay_prefix_max_steps" in signature.parameters
    assert signature.parameters["auto_replay_prefix_max_steps"].default == 3


def test_real_topm_eval_keeps_all_rows_when_eval_batches_unset_even_with_multiplier():
    rows = [{"row_id": idx, "loss_mask": {"L_trans_skill_ce": True}} for idx in range(100)]

    selected, report = _select_real_topm_eval_rows(
        rows,
        max_eval_batches=None,
        batch_size=8,
        loss_weights={"L_trans_skill_ce": 1.0},
        sample_multiplier=4.0,
        sampling_strategy="balanced_random",
        sampler_seed=17,
    )

    assert selected == rows
    assert report["enabled"] is False
    assert report["selected_rows"] == 100
    assert report["reason"] == "full_eval_keeps_all_rows"
