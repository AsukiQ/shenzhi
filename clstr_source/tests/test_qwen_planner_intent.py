import torch

from clstr.qwen_planner_intent import (
    QwenPlannerIntentGenerator,
    build_structured_planner_state_text,
    normalise_action_for_match,
)


class _FakePlannerScorer:
    def __init__(self):
        self.last_metadata = []
        self.seen_state_texts = []

    def __call__(self, state_texts, candidate_rows):
        self.seen_state_texts = list(state_texts)
        assert "Qwen proposed action" not in state_texts[0]
        self.last_metadata = [
            {
                "raw_model_response": "ACTION: open fridge 1",
                "parsed_action": "open fridge 1",
                "parse_status": "exact_match",
                "fallback_used": False,
            }
        ]
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)


def test_structured_planner_state_text_marks_qwen_action_as_advisory():
    text = build_structured_planner_state_text(
        "goal: cool apple\nobservation: kitchen",
        ["look", "open fridge 1"],
        proposed_action="open fridge 1",
    )

    assert "AVAILABLE ACTIONS" in text
    assert "Qwen proposed action: open fridge 1" in text
    assert "advisory" in text.lower()
    assert "CLSTR" in text


def test_qwen_planner_intent_generator_returns_proposal_without_final_decision():
    scorer = _FakePlannerScorer()
    planner = QwenPlannerIntentGenerator(scorer)
    intents = planner.propose(
        ["goal: cool apple\nobservation: kitchen"],
        [["look", "open fridge 1"]],
    )

    assert intents[0].proposed_action == "open fridge 1"
    assert intents[0].parse_status == "exact_match"
    assert intents[0].fallback_used is False
    assert intents[0].is_final_action is False
    assert scorer.seen_state_texts == ["goal: cool apple\nobservation: kitchen"]


def test_normalise_action_for_match_ignores_case_and_trailing_punctuation():
    assert normalise_action_for_match(" ACTION: Open   Fridge 1. ") == "open fridge 1"
