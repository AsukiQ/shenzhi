import torch

from clstr.closed_loop_controller import (
    ClosedLoopControllerConfig,
    ClosedLoopControllerState,
    score_candidates_with_components,
)
from clstr.qwen_planner_intent import PlannerIntent
from clstr.structured_controller import (
    StructuredCandidateScorer,
    StructuredComponentScorer,
    planner_match_scores,
)


class _FakePlanner:
    def propose(self, state_texts, candidate_rows):
        return [
            PlannerIntent(
                proposed_action="open fridge 1",
                raw_response="ACTION: open fridge 1",
                parse_status="exact_match",
                fallback_used=False,
            )
            for _ in state_texts
        ]


def test_planner_match_scores_are_candidate_aligned():
    scores = planner_match_scores(
        [["look", "open fridge 1", "close fridge 1"]],
        ["OPEN FRIDGE 1."],
    )

    assert scores.tolist() == [[0.0, 1.0, 0.0]]


def test_controller_uses_planner_scores_as_component_not_direct_execution():
    decisions = score_candidates_with_components(
        candidate_rows=[["look", "open fridge 1"]],
        policy_scores=torch.tensor([[2.0, 0.1]], dtype=torch.float32),
        transition_scores=torch.zeros(1, 2),
        belief_scores=torch.zeros(1, 2),
        stop_logits=torch.full((1, 2), -4.0),
        planner_scores=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        state=ClosedLoopControllerState(action_history=[[]]),
        config=ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop_loop_penalty",
            planner_weight=3.0,
        ),
    )

    assert decisions[0].chosen_action == "open fridge 1"
    assert decisions[0].chosen_reason == "max_final_score"
    scores = {row["action"]: row for row in decisions[0].component_scores}
    assert scores["open fridge 1"]["planner_score"] == 1.0
    assert scores["open fridge 1"]["final_score"] > scores["look"]["final_score"]


def test_structured_candidate_and_component_scorers_keep_qwen_advisory():
    captured = {}

    def base_candidate_scorer(state_texts, candidate_rows):
        captured["state_texts"] = state_texts
        return torch.tensor([[0.5, 0.4]], dtype=torch.float32)

    def base_component_scorer(state_texts, candidate_rows, policy_scores, histories):
        captured["component_state_texts"] = state_texts
        return {
            "transition_scores": torch.zeros_like(policy_scores),
            "belief_scores": torch.zeros_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
        }

    candidate_scorer = StructuredCandidateScorer(base_candidate_scorer, _FakePlanner())
    component_scorer = StructuredComponentScorer(base_component_scorer, candidate_scorer)

    policy_scores = candidate_scorer(["goal: cool apple"], [["look", "open fridge 1"]])
    components = component_scorer(["goal: cool apple"], [["look", "open fridge 1"]], policy_scores, [[]])

    assert "Qwen proposed action: open fridge 1" in captured["state_texts"][0]
    assert "Qwen proposed action: open fridge 1" in captured["component_state_texts"][0]
    assert components["planner_scores"].tolist() == [[0.0, 1.0]]
    assert candidate_scorer.last_metadata[0]["qwen_proposed_action"] == "open fridge 1"
    assert candidate_scorer.last_metadata[0]["qwen_is_final_action"] is False


def test_structured_component_scorer_merges_base_component_metadata():
    def base_candidate_scorer(state_texts, candidate_rows):
        base_candidate_scorer.last_metadata = [{"policy_family": "base"}]
        return torch.tensor([[0.5, 0.4]], dtype=torch.float32)

    def base_component_scorer(state_texts, candidate_rows, policy_scores, histories):
        base_component_scorer.last_metadata = [
            {
                "component_score_source": "trained_clstr_heads",
                "transition_score_source": "trained_clstr_trans_head",
                "belief_score_source": "trained_clstr_skill_head",
            }
        ]
        return {
            "transition_scores": torch.zeros_like(policy_scores),
            "belief_scores": torch.zeros_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
        }

    candidate_scorer = StructuredCandidateScorer(base_candidate_scorer, _FakePlanner())
    component_scorer = StructuredComponentScorer(base_component_scorer, candidate_scorer)

    policy_scores = candidate_scorer(["goal: cool apple"], [["look", "open fridge 1"]])
    component_scorer(["goal: cool apple"], [["look", "open fridge 1"]], policy_scores, [[]])

    assert component_scorer.last_metadata[0]["qwen_proposed_action"] == "open fridge 1"
    assert component_scorer.last_metadata[0]["component_score_source"] == "trained_clstr_heads"
    assert component_scorer.last_metadata[0]["transition_score_source"] == "trained_clstr_trans_head"
