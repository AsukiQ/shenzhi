from clstr.full_base_data import (
    FULL_BASE_FIELDS,
    FullBaseExample,
    allowed_loss_mask,
    validate_full_base_example,
)


def test_allowed_loss_mask_only_enables_losses_supported_by_evidence():
    mask = allowed_loss_mask(
        allowed_losses=["L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
        admissible_actions=["look", "open drawer 1"],
        expert_action="open drawer 1",
        next_observation_text="The drawer is open.",
        done=False,
        skill_id="alfworld/alfworld-receptacle-opener",
        next_skill_id="alfworld/alfworld-object-placer",
        history_text="look",
    )

    assert mask == {
        "L_policy": True,
        "L_trans": True,
        "L_trans_skill_ce": True,
        "belief": True,
        "STOP": True,
        "routing": True,
    }


def test_allowed_loss_mask_disables_policy_without_expert_in_candidates():
    mask = allowed_loss_mask(
        allowed_losses=["L_policy", "L_trans", "belief", "STOP", "routing"],
        admissible_actions=["look"],
        expert_action="open drawer 1",
        next_observation_text=None,
        done=None,
        skill_id="alfworld/alfworld-receptacle-opener",
        next_skill_id=None,
        history_text="",
    )

    assert mask["L_policy"] is False
    assert mask["L_trans"] is False
    assert mask["L_trans_skill_ce"] is False
    assert mask["belief"] is False
    assert mask["STOP"] is False
    assert mask["routing"] is True


def test_full_base_example_serializes_required_fields_and_validates_policy_label():
    example = FullBaseExample(
        benchmark="alfworld",
        task_id="game-1:0",
        goal_text="put mug in drawer",
        task_text="pick_and_place",
        state_text="goal: put mug in drawer\nobservation: You see a mug.",
        history_text="",
        action_text="take mug 1 from table 1",
        admissible_actions=["take mug 1 from table 1", "look"],
        expert_action="take mug 1 from table 1",
        next_observation_text="You pick up the mug.",
        done=False,
        reward=None,
        skill_id="alfworld/alfworld-object-picker",
        next_skill_id="alfworld/alfworld-receptacle-opener",
        loss_mask={
            "L_policy": True,
            "L_trans": True,
            "L_trans_skill_ce": True,
            "belief": False,
            "STOP": True,
            "routing": True,
        },
        source_quality="official_replay",
        provenance={
            "source_id": "alfworld_official_train_replay",
            "bucket": "official_replay",
            "split": "train",
        },
    )

    row = example.to_record()
    validate_full_base_example(row)

    assert set(FULL_BASE_FIELDS) <= set(row)
    assert row["expert_action"] == row["action_text"]
    assert row["next_skill_id"] == "alfworld/alfworld-receptacle-opener"
    assert row["loss_mask"]["L_policy"] is True
    assert row["loss_mask"]["L_trans_skill_ce"] is True
    assert row["source_quality"] == "official_replay"
    assert row["state_text_current"] == row["state_text"]
    assert row["state_text_full"] == example.state_text


def test_full_base_example_separates_structured_history_from_router_state():
    example = FullBaseExample(
        benchmark="alfworld",
        task_id="game-1:1",
        goal_text="put mug in drawer",
        task_text="pick_and_place",
        state_text=(
            "goal: put mug in drawer\n"
            "observation: drawer is open\n"
            "history: take mug 1 from table 1"
        ),
        history_text="take mug 1 from table 1",
        action_text="put mug 1 in drawer 1",
        admissible_actions=["put mug 1 in drawer 1"],
        expert_action="put mug 1 in drawer 1",
        next_observation_text="The mug is in the drawer.",
        done=True,
        reward=1.0,
        skill_id="alfworld/alfworld-object-placer",
        loss_mask={key: False for key in ("L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing")},
        source_quality="official_replay",
        provenance={"split": "train"},
    )

    row = example.to_record()

    assert row["state_text"] == (
        "goal: put mug in drawer\nobservation: drawer is open"
    )
    assert row["state_text_current"] == row["state_text"]
    assert "history: take mug 1 from table 1" in row["state_text_full"]
