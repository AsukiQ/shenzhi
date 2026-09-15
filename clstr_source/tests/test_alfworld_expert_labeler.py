from clstr.alfworld_expert_labeler import label_official_expert_action


def test_label_official_expert_action_extracts_first_expert_plan_step():
    label = label_official_expert_action(
        infos={"extra.expert_plan": [["open fridge 1", "take apple 1 from fridge 1"]]},
        batch_index=0,
        admissible_commands=["look", "open fridge 1"],
    )

    assert label.official_expert_action == "open fridge 1"
    assert label.expert_action_in_admissible is True
    assert label.usable_for_expert_ce is True
    assert label.skip_reason is None
    assert label.to_record()["official_expert_action_t"] == "open fridge 1"


def test_label_official_expert_action_marks_missing_or_unusable_without_faking_label():
    missing = label_official_expert_action(
        infos={"extra.expert_plan": [[]]},
        batch_index=0,
        admissible_commands=["look"],
    )
    out_of_set = label_official_expert_action(
        infos={"extra.expert_plan": [["open fridge 1"]]},
        batch_index=0,
        admissible_commands=["look"],
    )

    assert missing.official_expert_action is None
    assert missing.usable_for_expert_ce is False
    assert missing.skip_reason == "missing_official_expert_action"
    assert out_of_set.official_expert_action == "open fridge 1"
    assert out_of_set.expert_action_in_admissible is False
    assert out_of_set.usable_for_expert_ce is False
    assert out_of_set.skip_reason == "official_expert_action_not_in_admissible"
