from clstr.alfworld_action_skills import alfworld_action_to_skill_id


def test_alfworld_action_to_skill_id_maps_common_commands():
    known = {
        "alfworld/alfworld-location-navigator",
        "alfworld/alfworld-object-picker",
        "alfworld/alfworld-object-placer",
        "alfworld/alfworld-receptacle-opener",
        "alfworld/alfworld-heat-object-with-appliance",
        "alfworld/alfworld-environment-scanner",
    }

    assert alfworld_action_to_skill_id("go to fridge 1", known) == "alfworld/alfworld-location-navigator"
    assert alfworld_action_to_skill_id("take apple 1 from table 1", known) == "alfworld/alfworld-object-picker"
    assert alfworld_action_to_skill_id("put apple 1 in fridge 1", known) == "alfworld/alfworld-object-placer"
    assert alfworld_action_to_skill_id("open cabinet 2", known) == "alfworld/alfworld-receptacle-opener"
    assert (
        alfworld_action_to_skill_id("heat mug 1 with microwave 1", known)
        == "alfworld/alfworld-heat-object-with-appliance"
    )
    assert alfworld_action_to_skill_id("inventory", known) == "alfworld/alfworld-environment-scanner"


def test_preprocess_and_eval_import_same_alfworld_mapper():
    import clstr.alfworld_eval as eval_mod
    import clstr.full_base_preprocess as preprocess

    assert preprocess.alfworld_action_to_skill_id is eval_mod.alfworld_action_to_skill_id
