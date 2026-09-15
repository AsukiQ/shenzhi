from __future__ import annotations

import pytest

from clstr.mt_ablation_eval import build_mt_ablation_report
from clstr.mt_ablation_eval import compute_mt_pairwise_effect_diagnostics
from clstr.mt_ablation_eval import evaluate_mt_ablation_rows
from clstr.mt_ablation_eval import _mismatch_replay_prefixes
from clstr.mt_ablation_eval import _order_shuffle_replay_prefixes
from clstr.mt_ablation_eval import summarize_pairwise_effect_records


def test_build_mt_ablation_report_marks_static_variant_as_true_no_replay():
    report = build_mt_ablation_report(
        evaluations={
            "static_no_replay": {
                "stage4_act_count": 10.0,
                "stage4_next_skill_mrr": 0.20,
                "stage4_auto_replay_prefix_used_count": 0,
            },
            "dynamic_replay_no_online": {
                "stage4_act_count": 10.0,
                "stage4_next_skill_mrr": 0.25,
                "stage4_auto_replay_prefix_used_count": 6,
            },
            "dynamic_replay_with_online": {
                "stage4_act_count": 10.0,
                "stage4_next_skill_mrr": 0.30,
                "stage4_auto_replay_prefix_used_count": 6,
            },
            "shuffled_replay_no_online": {
                "stage4_act_count": 10.0,
                "stage4_next_skill_mrr": 0.21,
            },
            "masked_replay_no_online": {
                "stage4_act_count": 10.0,
                "stage4_next_skill_mrr": 0.20,
            },
        },
        source_rows=12,
        retained_rows=10,
    )

    assert report["variant_config"]["static_no_replay"]["auto_replay_prefix_max_steps"] == 0
    assert report["variant_config"]["static_no_replay"]["online_memory_weight"] == 0.0
    assert report["variant_config"]["dynamic_replay_no_online"]["auto_replay_prefix_max_steps"] == 3
    assert report["variant_config"]["dynamic_replay_no_online"]["online_memory_weight"] == 0.0
    assert report["variant_config"]["dynamic_replay_with_online"]["online_memory_weight"] == 1.0
    assert report["delta_dynamic_replay_no_online_vs_static_no_replay"]["stage4_next_skill_mrr"] == pytest.approx(
        0.05
    )
    assert report["delta_dynamic_replay_with_online_vs_dynamic_replay_no_online"][
        "stage4_next_skill_mrr"
    ] == pytest.approx(0.05)
    assert report["delta_true_replay_vs_shuffled_replay"]["stage4_next_skill_mrr"] == pytest.approx(
        0.04
    )
    assert report["delta_shuffled_replay_vs_static_no_replay"]["stage4_next_skill_mrr"] == pytest.approx(
        0.01
    )
    assert report["strict"]["dynamic_replay_no_online"]["source_rows"] == 12.0
    assert report["strict"]["dynamic_replay_no_online"]["retained_rows"] == 10.0


def test_evaluate_mt_ablation_rows_passes_true_replay_off_and_dynamic_variants(monkeypatch):
    calls = []

    def fake_eval(model, rows, **kwargs):
        calls.append(
            {
                "kwargs": kwargs,
                "prefixes": [
                    [dict(step) for step in row.get("replay_prefix") or []]
                    for row in rows
                ],
            }
        )
        return {
            "stage4_act_count": float(len(rows)),
            "stage4_next_skill_mrr": float(kwargs["online_memory_weight"]),
        }

    report = evaluate_mt_ablation_rows(
        model=object(),
        rows=[
            {
                "id": "a",
                "source_benchmark": "toolbench_g3",
                "trajectory_id": "ta",
                "replay_prefix": [
                    {
                        "trajectory_id": "ta",
                        "skill_id": "skill-a",
                        "action_text": "action-a",
                        "observation_text": "observation-a",
                        "next_observation_text": "next-a",
                    }
                ],
            },
            {
                "id": "b",
                "source_benchmark": "toolbench_g3",
                "trajectory_id": "tb",
                "replay_prefix": [
                    {
                        "trajectory_id": "tb",
                        "skill_id": "skill-b",
                        "action_text": "action-b",
                        "observation_text": "observation-b",
                        "next_observation_text": "next-b",
                    }
                ],
            },
        ],
        source_rows=2,
        transition_residual_lambda=0.25,
        transition_scoring_mode="stage0_rank_prior",
        online_memory_weight=1.0,
        auto_replay_prefix_max_steps=3,
        evaluator=fake_eval,
    )

    assert [call["kwargs"]["auto_replay_prefix_max_steps"] for call in calls] == [0, 0, 0, 0, 0, 0]
    assert [call["kwargs"]["online_memory_weight"] for call in calls] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert [call["kwargs"]["score_calibrator_enabled"] for call in calls] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert calls[0]["prefixes"] == [[], []]
    assert [prefix[0]["skill_id"] for prefix in calls[1]["prefixes"]] == ["skill-a", "skill-b"]
    assert [prefix[0]["skill_id"] for prefix in calls[2]["prefixes"]] == ["skill-b", "skill-a"]
    assert [prefix[0]["skill_id"] for prefix in calls[3]["prefixes"]] == ["skill-a", "skill-b"]
    assert calls[4]["prefixes"] == [[], []]
    assert report["evaluations"]["static_no_replay"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["dynamic_replay_no_online"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["mismatch_replay_no_online"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["order_shuffled_replay_no_online"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["shuffled_replay_no_online"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["masked_replay_no_online"]["stage4_next_skill_mrr"] == 0.0
    assert report["evaluations"]["dynamic_replay_with_online"]["stage4_next_skill_mrr"] == 1.0
    assert report["replay_preparation"]["mismatch"]["mismatched_prefix_rows"] == 2
    assert report["replay_preparation"]["order_shuffle"]["order_shuffled_prefix_rows"] == 0


def test_memory_interventions_distinguish_cross_trajectory_mismatch_and_temporal_order():
    rows = [
        {
            "benchmark": "toy",
            "trajectory_id": "a",
            "replay_prefix": [{"step": 0}, {"step": 1}],
        },
        {
            "benchmark": "toy",
            "trajectory_id": "a",
            "replay_prefix": [{"step": 2}, {"step": 3}],
        },
        {
            "benchmark": "toy",
            "trajectory_id": "b",
            "replay_prefix": [{"step": 4}, {"step": 5}],
        },
    ]

    mismatched, mismatch_report = _mismatch_replay_prefixes(rows)
    order_shuffled, order_report = _order_shuffle_replay_prefixes(rows)

    assert mismatched[0]["replay_prefix"] == rows[2]["replay_prefix"]
    assert mismatched[1]["replay_prefix"] == rows[2]["replay_prefix"]
    assert mismatched[2]["replay_prefix"] in (
        rows[0]["replay_prefix"],
        rows[1]["replay_prefix"],
    )
    assert mismatch_report["mismatched_prefix_rows"] == 3
    assert order_shuffled[0]["replay_prefix"] == list(reversed(rows[0]["replay_prefix"]))
    assert order_report["order_shuffled_prefix_rows"] == 3


def test_summarize_pairwise_effect_records_counts_rank_and_argmax_flips():
    summary = summarize_pairwise_effect_records(
        [
            {
                "m_l2": 0.2,
                "m_cosine": 0.9,
                "final_max_abs_diff": 0.3,
                "residual_max_abs_diff": 0.4,
                "static_final_rank": 3,
                "dynamic_final_rank": 2,
                "static_residual_rank": 4,
                "dynamic_residual_rank": 4,
                "static_final_argmax": 1,
                "dynamic_final_argmax": 2,
                "static_residual_argmax": 1,
                "dynamic_residual_argmax": 1,
            },
            {
                "m_l2": 0.0,
                "m_cosine": 1.0,
                "final_max_abs_diff": 0.0,
                "residual_max_abs_diff": 0.0,
                "static_final_rank": 1,
                "dynamic_final_rank": 3,
                "static_residual_rank": 2,
                "dynamic_residual_rank": 1,
                "static_final_argmax": 0,
                "dynamic_final_argmax": 0,
                "static_residual_argmax": 0,
                "dynamic_residual_argmax": 2,
            },
        ]
    )

    assert summary["row_count"] == 2
    assert summary["m_l2_mean"] == 0.1
    assert summary["final_argmax_changed_rows"] == 1
    assert summary["residual_argmax_changed_rows"] == 1
    assert summary["final_positive_rank_improved_rows"] == 1
    assert summary["final_positive_rank_worsened_rows"] == 1
    assert summary["residual_positive_rank_improved_rows"] == 1
    assert summary["residual_positive_rank_worsened_rows"] == 0


def test_pairwise_effect_diagnostics_uses_unified_route_scorer(monkeypatch):
    import torch

    class SkillTable:
        E = torch.eye(2)

        def retrieval_logits(self, h):
            return h @ self.E.t()

    class Model:
        device = torch.device("cpu")
        skill_table = SkillTable()

        def eval(self):
            return self

        def initial_belief(self, h):
            return h

        def unified_route_logits(self, h, m, candidate_rows=None):
            logits = m @ self.skill_table.E.t()
            if candidate_rows is None:
                return logits
            ids = torch.tensor(candidate_rows, dtype=torch.long)
            return logits.gather(1, ids)

    def fail_legacy_transition(*args, **kwargs):
        raise AssertionError("legacy transition scorer should not be used for unified_memory diagnostics")

    monkeypatch.setattr("clstr.full_base_train._transition_candidate_logits_for_mode", fail_legacy_transition)

    report = compute_mt_pairwise_effect_diagnostics(
        Model(),
        [
            {
                "task_id": "row0",
                "trajectory_id": "traj",
                "step_index": 0,
                "_state_embedding": torch.tensor([1.0, 0.0]),
                "_next_observation_embedding": torch.tensor([1.0, 0.0]),
                "_action_embedding": torch.tensor([1.0, 0.0]),
                "state_text": "s0",
                "action_text": "a0",
                "next_observation_text": "s1",
                "skill_id": "a",
                "skill_idx": 0,
                "next_skill_id": "a",
                "positive_next_skill_idx": 0,
                "candidate_next_skill_ids": ["a", "b"],
                "candidate_next_skill_indices": [0, 1],
            }
        ],
        transition_residual_lambda=0.25,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        route_scorer="unified_memory",
    )

    assert report["route_scorer"] == "unified_memory"
    assert report["row_count"] == 1
