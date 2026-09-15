import json
from pathlib import Path

from clstr.skillnet_aux_rebuild import (
    load_skillnet_skills,
    map_action_to_skillnet_skill,
    rebuild_aux_with_skillnet,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_skill(root: Path, environment: str, name: str, description: str) -> None:
    path = root / environment / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n",
        encoding="utf-8",
    )


def make_skillnet_fixture(root: Path) -> Path:
    write_skill(root, "alfworld", "alfworld-location-navigator", "Move to a receptacle or location.")
    write_skill(root, "alfworld", "alfworld-clean-object", "Clean a held object using a sinkbasin.")
    write_skill(root, "alfworld", "alfworld-object-picker", "Pick up an object from a receptacle.")
    write_skill(root, "webshop", "webshop-search-executor", "Execute a search[] action.")
    write_skill(root, "webshop", "webshop-purchase-executor", "Click Buy Now for a verified product.")
    write_skill(root, "scienceworld", "scienceworld-room-navigator", "Teleport to a room.")
    write_skill(root, "scienceworld", "scienceworld-temperature-measurer", "Measure temperature with a thermometer.")
    return root


def test_load_skillnet_skills_parses_frontmatter_by_environment(tmp_path):
    skillnet_root = make_skillnet_fixture(tmp_path / "skills")

    skills = load_skillnet_skills(skillnet_root)

    assert sorted(skills) == ["alfworld", "scienceworld", "webshop"]
    assert skills["alfworld"]["alfworld/alfworld-clean-object"]["name"] == "alfworld-clean-object"
    assert skills["alfworld"]["alfworld/alfworld-clean-object"]["source_path"].endswith(
        "alfworld-clean-object/SKILL.md"
    )
    assert "Clean a held object" in skills["alfworld"]["alfworld/alfworld-clean-object"]["description"]


def test_map_action_to_skillnet_skill_uses_benchmark_specific_rules(tmp_path):
    skills = load_skillnet_skills(make_skillnet_fixture(tmp_path / "skills"))

    assert (
        map_action_to_skillnet_skill("alfworld", "go to fridge 1", skills).skill_id
        == "alfworld/alfworld-location-navigator"
    )
    assert (
        map_action_to_skillnet_skill("alfworld", "clean apple 1 with sinkbasin 1", skills).skill_id
        == "alfworld/alfworld-clean-object"
    )
    assert (
        map_action_to_skillnet_skill("webshop", "search[long clip-in hair extension]", skills).skill_id
        == "webshop/webshop-search-executor"
    )
    assert (
        map_action_to_skillnet_skill("webshop", "click[buy now]", skills).skill_id
        == "webshop/webshop-purchase-executor"
    )
    assert (
        map_action_to_skillnet_skill("scienceworld", "measure temperature of water", skills).skill_id
        == "scienceworld/scienceworld-temperature-measurer"
    )
    assert (
        map_action_to_skillnet_skill("scienceworld", "teleport to kitchen", skills).skill_id
        == "scienceworld/scienceworld-room-navigator"
    )


def test_rebuild_aux_with_skillnet_splits_benchmarks_and_marks_train_only_usable(tmp_path):
    skillnet_root = make_skillnet_fixture(tmp_path / "skills")
    source_root = tmp_path / "aux"
    write_jsonl(
        source_root / "trajectories.jsonl",
        [
            {
                "dataset": "unit",
                "source_path": "train.jsonl",
                "split": "train",
                "environment": "alfworld",
                "task_id": "alf-train",
                "task_type": "pick_and_place",
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "You see a sinkbasin.",
                        "action_text": "clean apple 1 with sinkbasin 1",
                        "next_observation_text": "The apple is clean.",
                        "pseudo_skill_id": "alfworld/text_action",
                        "pseudo_skill_low_confidence": True,
                    }
                ],
            },
            {
                "dataset": "unit",
                "source_path": "valid_seen.jsonl",
                "split": "valid_seen",
                "environment": "webshop",
                "task_id": "web-valid",
                "task_type": "shopping",
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "Search page.",
                        "action_text": "search[green tea]",
                        "next_observation_text": "Results.",
                        "pseudo_skill_id": "webshop/text_action",
                        "pseudo_skill_low_confidence": True,
                    }
                ],
            },
            {
                "dataset": "unit",
                "source_path": "train.jsonl",
                "split": "train",
                "environment": "crafter",
                "task_id": "craft-train",
                "task_type": "crafter",
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "A tree is nearby.",
                        "action_text": "collect wood",
                        "next_observation_text": "You have wood.",
                        "pseudo_skill_id": "crafter/text_action",
                        "pseudo_skill_low_confidence": True,
                    }
                ],
            },
        ],
    )
    (source_root / "manifest.json").write_text(
        json.dumps({"record_counts": {"trajectories": 3, "steps": 3}}) + "\n",
        encoding="utf-8",
    )

    manifest = rebuild_aux_with_skillnet(
        source_data_dir=source_root,
        skillnet_root=skillnet_root,
        output_dir=tmp_path / "rebuilt",
        report_dir=tmp_path / "reports",
    )

    assert manifest["status"] == "ok"
    assert manifest["excluded_environments"] == {"crafter": "missing_skillnet_skill_pool"}
    assert manifest["split_counts"] == {"train": 1, "valid_seen": 1}
    assert manifest["split_counts_by_environment"] == {
        "alfworld": {"train": 1},
        "webshop": {"valid_seen": 1},
    }
    assert manifest["training_usable_step_count"] == 1
    assert manifest["uses_valid_or_test_for_training"] is False
    assert manifest["quality_caveats"]["not_equivalent_to_official_replay"] is True
    assert (tmp_path / "rebuilt" / "alfworld" / "trajectories.jsonl").exists()
    assert (tmp_path / "rebuilt" / "webshop" / "trajectories.jsonl").exists()
    assert not (tmp_path / "rebuilt" / "crafter" / "trajectories.jsonl").exists()
    assert (tmp_path / "rebuilt" / "skills.jsonl").exists()
    assert (tmp_path / "rebuilt" / "pseudo_skills.jsonl").exists()
    assert (tmp_path / "reports" / "rebuild_report.json").exists()

    alf_row = json.loads((tmp_path / "rebuilt" / "alfworld" / "trajectories.jsonl").read_text(encoding="utf-8"))
    web_row = json.loads((tmp_path / "rebuilt" / "webshop" / "trajectories.jsonl").read_text(encoding="utf-8"))
    pseudo_skills = [
        json.loads(line)
        for line in (tmp_path / "rebuilt" / "pseudo_skills.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    alf_step = alf_row["steps"][0]
    web_step = web_row["steps"][0]
    assert {row["skill_id"] for row in pseudo_skills} == {
        "alfworld/alfworld-clean-object",
        "alfworld/alfworld-location-navigator",
        "alfworld/alfworld-object-picker",
        "scienceworld/scienceworld-room-navigator",
        "scienceworld/scienceworld-temperature-measurer",
        "webshop/webshop-purchase-executor",
        "webshop/webshop-search-executor",
    }
    assert alf_step["original_pseudo_skill_id"] == "alfworld/text_action"
    assert alf_step["pseudo_skill_id"] == "alfworld/alfworld-clean-object"
    assert alf_step["pseudo_skill_low_confidence"] is False
    assert alf_step["skillnet_skill_id"] == "alfworld/alfworld-clean-object"
    assert alf_step["skillnet_mapping_confidence"] == "high"
    assert alf_step["usable_for_skillnet_training"] is True
    assert web_step["pseudo_skill_id"] == "webshop/webshop-search-executor"
    assert web_step["skillnet_skill_id"] == "webshop/webshop-search-executor"
    assert web_step["usable_for_skillnet_training"] is False
    assert web_step["unusable_reason"] == "non_train_split"
