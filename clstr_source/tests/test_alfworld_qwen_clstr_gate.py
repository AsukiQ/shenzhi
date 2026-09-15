import torch

from clstr.alfworld_qwen_clstr_gate import QwenClstrHybridActionScorer
from scripts.run_alfworld_qwen_clstr_executor_gate import _LoopGuardComponentScorer, _summarize_policy_trace


class _FakeScorer:
    def __init__(self, rows, metadata):
        self.rows = rows
        self.metadata = metadata
        self.last_metadata = []
        self.calls = []

    def __call__(self, state_texts, candidate_rows):
        self.calls.append((list(state_texts), [list(row) for row in candidate_rows]))
        self.last_metadata = [dict(item) for item in self.metadata]
        return torch.tensor(self.rows, dtype=torch.float32)


def test_hybrid_scorer_preserves_qwen_ranking_when_clstr_weight_is_zero():
    qwen = _FakeScorer(
        [[0.1, 0.9, 0.2]],
        [{"policy_family": "qwen_direct_admissible", "parsed_action": "open fridge"}],
    )
    clstr = _FakeScorer(
        [[5.0, 0.0, 0.0]],
        [{"policy_family": "clstr_native_action_scorer"}],
    )
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=clstr,
        qwen_weight=1.0,
        clstr_weight=0.0,
        normalize_scores=False,
    )

    scores = scorer(["state"], [["look", "open fridge", "inventory"]])

    assert torch.argmax(scores[0]).item() == 1
    assert torch.allclose(scores[0, :3], torch.tensor([0.1, 0.9, 0.2]))
    assert scorer.last_metadata[0]["policy_family"] == "qwen_clstr_hybrid_executor_gate"
    assert scorer.last_metadata[0]["uses_clstr_prior"] is False
    assert scorer.last_metadata[0]["qwen_chosen_action"] == "open fridge"
    assert scorer.last_metadata[0]["raw_model_response"] == ""
    assert scorer.last_metadata[0]["parsed_action"] == "open fridge"
    assert scorer.last_metadata[0]["legacy_coarse_safeguard_applicable"] is False
    assert scorer.last_metadata[0]["clstr_override_reason"] == "allowed_noncoarse_prior"


def test_hybrid_scorer_can_move_ranking_toward_clstr_prior():
    qwen = _FakeScorer(
        [[0.0, 1.0, 0.0]],
        [{"policy_family": "qwen_direct_admissible", "parsed_action": "open fridge"}],
    )
    clstr = _FakeScorer(
        [[3.0, 0.0, 0.0]],
        [{"policy_family": "clstr_native_action_scorer"}],
    )
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=clstr,
        qwen_weight=1.0,
        clstr_weight=1.0,
        normalize_scores=False,
    )

    scores = scorer(["state"], [["look", "open fridge", "inventory"]])

    assert torch.argmax(scores[0]).item() == 0
    assert scorer.last_metadata[0]["qwen_chosen_action"] == "open fridge"
    assert scorer.last_metadata[0]["clstr_chosen_action"] == "look"
    assert scorer.last_metadata[0]["hybrid_chosen_action"] == "look"
    assert scorer.last_metadata[0]["uses_clstr_prior"] is True


def test_hybrid_scorer_can_label_skillrouter_prior():
    qwen = _FakeScorer(
        [[0.0, 1.0, 0.0]],
        [{"policy_family": "qwen_direct_admissible", "parsed_action": "open fridge"}],
    )
    skillrouter = _FakeScorer(
        [[3.0, 0.0, 0.0]],
        [{"policy_family": "skillrouter_frozen_admissible_action"}],
    )
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=skillrouter,
        qwen_weight=1.0,
        clstr_weight=1.0,
        normalize_scores=False,
        prior_name="skillrouter",
        policy_family="qwen_skillrouter_hybrid_executor_gate",
    )

    scorer(["state"], [["look", "open fridge", "inventory"]])

    metadata = scorer.last_metadata[0]
    assert metadata["policy_family"] == "qwen_skillrouter_hybrid_executor_gate"
    assert metadata["uses_skillrouter_prior"] is True
    assert metadata["skillrouter_chosen_action"] == "look"
    assert metadata["action_selection"] == "qwen_scores_plus_skillrouter_prior_over_admissible_actions"


def test_hybrid_scorer_can_treat_generated_qwen_action_as_proposal_bonus():
    qwen = _FakeScorer(
        [[0.0, 1.0, 0.0]],
        [{"policy_family": "qwen_direct_admissible", "parsed_action": "open fridge"}],
    )
    clstr = _FakeScorer(
        [[1.0, 0.0, 0.0]],
        [{"policy_family": "clstr_native_action_scorer"}],
    )
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=clstr,
        qwen_weight=1.0,
        clstr_weight=1.0,
        normalize_scores=True,
        qwen_score_mode="proposal_bonus",
    )

    scores = scorer(["state"], [["look", "open fridge", "inventory"]])

    assert torch.argmax(scores[0]).item() == 0
    assert scorer.last_metadata[0]["qwen_score_mode"] == "proposal_bonus"
    assert scorer.last_metadata[0]["qwen_chosen_action"] == "open fridge"
    assert scorer.last_metadata[0]["hybrid_chosen_action"] == "look"


def test_hybrid_scorer_masks_padding_and_calibrates_rows_independently():
    qwen = _FakeScorer(
        [[0.0, 10.0, torch.finfo(torch.float32).min], [4.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]],
        [{"policy_family": "qwen"}, {"policy_family": "qwen"}],
    )
    clstr = _FakeScorer(
        [[10.0, 0.0, torch.finfo(torch.float32).min], [1.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]],
        [{"policy_family": "clstr"}, {"policy_family": "clstr"}],
    )
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=clstr,
        qwen_weight=1.0,
        clstr_weight=0.5,
        normalize_scores=True,
    )

    scores = scorer(["s1", "s2"], [["look", "open fridge"], ["inventory"]])

    assert scores.shape == (2, 2)
    assert scores[1, 1].item() == torch.finfo(torch.float32).min
    assert scores[1, 0].item() == 0.0
    assert scorer.last_metadata[0]["score_normalization"] == "row_zscore"
    assert scorer.last_metadata[1]["candidate_count"] == 1


def test_gate_trace_summary_reports_qwen_clstr_and_hybrid_disagreement(tmp_path):
    run_path = tmp_path / "run.jsonl"
    run_path.write_text(
        "\n".join(
            [
                '{"policy_metadata_trace": [{"qwen_chosen_action": "open", "clstr_chosen_action": "look", "hybrid_chosen_action": "open"}]}',
                '{"policy_metadata_trace": [{"qwen_chosen_action": "look", "clstr_chosen_action": "look", "hybrid_chosen_action": "look", "qwen_fallback_used": true, "clstr_override_allowed": true}]}',
                '{"policy_metadata_trace": [{"qwen_chosen_action": "open fridge", "clstr_chosen_action": "open cabinet", "hybrid_chosen_action": "open fridge", "clstr_override_allowed": false}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_policy_trace(run_path)

    assert summary["total_steps"] == 3
    assert summary["qwen_clstr_disagreement_rate"] == 0.666667
    assert summary["qwen_hybrid_disagreement_rate"] == 0.0
    assert summary["clstr_hybrid_agreement_rate"] == 0.333333
    assert summary["qwen_fallback_rate"] == 0.333333
    assert summary["clstr_override_allowed_steps"] == 1
    assert summary["clstr_override_blocked_steps"] == 1
    assert summary["clstr_override_blocked_rate"] == 0.333333


def test_loop_guard_component_scorer_returns_zero_components_without_transition_signal():
    scorer = _LoopGuardComponentScorer()
    policy_scores = torch.tensor([[1.0, 2.0]])

    components = scorer(["state"], [["open fridge", "close fridge"]], policy_scores, [["open fridge"]])

    assert set(components) == {"transition_scores", "belief_scores", "stop_logits"}
    assert torch.equal(components["transition_scores"], torch.zeros_like(policy_scores))
    assert torch.equal(components["belief_scores"], torch.zeros_like(policy_scores))
    assert torch.equal(components["stop_logits"], torch.zeros_like(policy_scores))
    assert scorer.last_metadata[0]["component_score_source"] == "loop_guard_only_zero_components"
