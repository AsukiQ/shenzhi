import torch

from clstr.closed_loop_controller import (
    ClosedLoopControllerConfig,
    ClosedLoopControllerState,
    ClstrTextActionController,
    score_candidates_with_components,
)


def test_score_candidates_records_components_and_loop_penalty_changes_choice():
    policy_scores = torch.tensor([[0.9, 0.8, 0.1]], dtype=torch.float32)
    transition_scores = torch.tensor([[0.0, 0.4, 0.0]], dtype=torch.float32)
    belief_scores = torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float32)
    stop_logits = torch.tensor([[-2.0, -2.0, -2.0]], dtype=torch.float32)
    state = ClosedLoopControllerState(action_history=[["look", "look", "look", "look"]])

    decisions = score_candidates_with_components(
        candidate_rows=[["look", "go to fridge 1", "inventory"]],
        policy_scores=policy_scores,
        transition_scores=transition_scores,
        belief_scores=belief_scores,
        stop_logits=stop_logits,
        state=state,
        config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop_loop_penalty"),
    )

    assert decisions[0].chosen_action == "go to fridge 1"
    assert decisions[0].chosen_reason == "max_final_score"
    assert decisions[0].component_scores[0]["action"] == "look"
    assert decisions[0].component_scores[0]["policy_score"] == 0.9
    assert decisions[0].component_scores[0]["transition_score"] == 0.0
    assert decisions[0].component_scores[0]["belief_score"] == 0.0
    assert decisions[0].component_scores[0]["stop_probability"] < 0.2
    assert decisions[0].component_scores[0]["loop_penalty"] < 0.0
    assert decisions[0].component_scores[1]["final_score"] > decisions[0].component_scores[0]["final_score"]


def test_controller_normalizes_overconfident_policy_scores_before_loop_penalty():
    decisions = score_candidates_with_components(
        candidate_rows=[["look", "go to fridge 1", "open fridge 1"]],
        policy_scores=torch.tensor([[7.8, 0.8, 0.6]], dtype=torch.float32),
        transition_scores=torch.tensor([[0.0, 0.3, 0.2]], dtype=torch.float32),
        belief_scores=torch.tensor([[0.0, 0.2, 0.1]], dtype=torch.float32),
        stop_logits=torch.full((1, 3), -4.0),
        state=ClosedLoopControllerState(action_history=[["look", "look", "look", "look"]]),
        config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop_loop_penalty"),
    )

    assert decisions[0].chosen_action != "look"
    assert decisions[0].component_scores[0]["policy_score"] == 7.8
    assert "policy_score_calibrated" in decisions[0].component_scores[0]
    assert decisions[0].component_scores[0]["final_score"] < decisions[0].component_scores[1]["final_score"]


def test_controller_penalizes_two_action_toggle_cycles_enough_to_escape():
    decisions = score_candidates_with_components(
        candidate_rows=[["open microwave 1", "go to fridge 1"]],
        policy_scores=torch.tensor([[4.8, 0.4]], dtype=torch.float32),
        transition_scores=torch.tensor([[0.03, 0.02]], dtype=torch.float32),
        belief_scores=torch.tensor([[0.03, 0.02]], dtype=torch.float32),
        stop_logits=torch.full((1, 2), -8.0),
        state=ClosedLoopControllerState(action_history=[["go to microwave 1", "open microwave 1", "close microwave 1"]]),
        config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop_loop_penalty"),
    )

    assert decisions[0].chosen_action == "go to fridge 1"
    assert decisions[0].component_scores[0]["loop_penalty"] <= -2.0


def test_controller_penalizes_repeated_long_action_cycles():
    decisions = score_candidates_with_components(
        candidate_rows=[["go to sinkbasin 1", "take egg 1 from countertop 1"]],
        policy_scores=torch.tensor([[2.0, 1.5]], dtype=torch.float32),
        transition_scores=torch.zeros(1, 2),
        belief_scores=torch.zeros(1, 2),
        stop_logits=torch.zeros(1, 2),
        state=ClosedLoopControllerState(
            action_history=[
                [
                    "go to sinkbasin 1",
                    "go to garbagecan 1",
                    "go to stoveburner 5",
                    "go to stoveburner 4",
                    "go to stoveburner 6",
                    "go to sinkbasin 1",
                    "go to garbagecan 1",
                    "go to stoveburner 5",
                    "go to stoveburner 4",
                    "go to stoveburner 6",
                ]
            ]
        ),
        config=ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop_loop_penalty",
            cycle_ngram_penalty=4.0,
        ),
    )

    cycle_row = decisions[0].component_scores[0]
    escape_row = decisions[0].component_scores[1]
    assert cycle_row["loop_penalty"] < escape_row["loop_penalty"]
    assert decisions[0].chosen_action == "take egg 1 from countertop 1"


def test_controller_downranks_help_inventory_and_repeated_look_info_actions():
    decisions = score_candidates_with_components(
        candidate_rows=[["help", "inventory", "look", "go to fridge 1"]],
        policy_scores=torch.tensor([[4.5, 3.5, 7.3, 1.0]], dtype=torch.float32),
        transition_scores=torch.tensor([[0.0, 0.0, 0.0, 0.2]], dtype=torch.float32),
        belief_scores=torch.tensor([[0.0, 0.0, 0.0, 0.1]], dtype=torch.float32),
        stop_logits=torch.full((1, 4), -4.0),
        state=ClosedLoopControllerState(action_history=[["look", "look", "inventory", "help"]]),
        config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop_loop_penalty"),
    )

    assert decisions[0].chosen_action == "go to fridge 1"
    scores = {row["action"]: row for row in decisions[0].component_scores}
    assert scores["help"]["action_prior_penalty"] < 0.0
    assert scores["inventory"]["action_prior_penalty"] < 0.0
    assert scores["look"]["action_prior_penalty"] < 0.0


def test_controller_uses_q_success_scores_when_weighted():
    decisions = score_candidates_with_components(
        candidate_rows=[["look", "take key 1 from table 1"]],
        policy_scores=torch.tensor([[2.0, 0.1]], dtype=torch.float32),
        transition_scores=torch.zeros(1, 2),
        belief_scores=torch.zeros(1, 2),
        stop_logits=torch.zeros(1, 2),
        q_success_scores=torch.tensor([[0.0, 4.0]], dtype=torch.float32),
        state=ClosedLoopControllerState(action_history=[[]]),
        config=ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop_loop_penalty",
            q_success_weight=3.0,
        ),
    )

    assert decisions[0].chosen_action == "take key 1 from table 1"
    rows = {row["action"]: row for row in decisions[0].component_scores}
    assert rows["take key 1 from table 1"]["q_success_score"] == 4.0
    assert rows["take key 1 from table 1"]["q_success_score_calibrated"] > rows["look"]["q_success_score_calibrated"]


def test_clstr_text_action_controller_passes_planner_and_q_success_scores():
    def candidate_scorer(_state_texts, _candidate_rows):
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    def component_scorer(_state_texts, _candidate_rows, policy_scores, _action_histories):
        return {
            "transition_scores": torch.zeros_like(policy_scores),
            "belief_scores": torch.zeros_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
            "planner_scores": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            "q_success_scores": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        }

    controller = ClstrTextActionController(
        candidate_scorer=candidate_scorer,
        component_scorer=component_scorer,
        config=ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop",
            planner_weight=1.0,
            q_success_weight=1.0,
        ),
    )

    assert controller.choose_action("state", ["look", "open drawer 1"]) == "open drawer 1"
    assert controller.last_decision is not None
    chosen = controller.last_decision.component_scores[1]
    assert chosen["planner_score"] == 2.0
    assert chosen["q_success_score"] == 2.0


def test_policy_only_mode_matches_policy_argmax():
    decisions = score_candidates_with_components(
        candidate_rows=[["look", "open fridge 1"]],
        policy_scores=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        transition_scores=torch.tensor([[10.0, -10.0]], dtype=torch.float32),
        belief_scores=torch.tensor([[10.0, -10.0]], dtype=torch.float32),
        stop_logits=torch.zeros(1, 2),
        state=ClosedLoopControllerState(action_history=[["look"]]),
        config=ClosedLoopControllerConfig(mode="policy_only"),
    )

    assert decisions[0].chosen_action == "open fridge 1"
    assert decisions[0].chosen_index == 1
    assert decisions[0].component_scores[0]["transition_score"] == 10.0
    assert decisions[0].component_scores[0]["final_score"] == 0.1


def test_clstr_text_action_controller_uses_component_scorer_when_configured():
    captured = {}

    def candidate_scorer(state_texts, candidate_rows):
        captured["state_texts"] = state_texts
        captured["candidate_rows"] = candidate_rows
        return torch.tensor([[0.9, 0.4]], dtype=torch.float32)

    def component_scorer(state_texts, candidate_rows, policy_scores, action_histories):
        captured["action_histories"] = action_histories
        return {
            "transition_scores": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            "belief_scores": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "stop_logits": torch.zeros_like(policy_scores),
        }

    controller = ClstrTextActionController(
        candidate_scorer=candidate_scorer,
        component_scorer=component_scorer,
        config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop"),
    )

    action = controller.choose_action("observation: start", ["look", "go north"])

    assert action == "go north"
    assert captured["state_texts"] == ["observation: start"]
    assert captured["candidate_rows"] == [["look", "go north"]]
    assert controller.last_decision is not None
    assert controller.last_decision.component_scores[1]["transition_score"] == 2.0
