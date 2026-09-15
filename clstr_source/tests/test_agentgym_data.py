import json
from pathlib import Path

from clstr.agentgym_data import (
    audit_agentgym_agenttraj_file,
    convert_agentgym_alfworld_conversations,
    parse_agentgym_action,
    parse_available_actions,
)


def test_parse_agentgym_available_actions_and_action():
    human = "Observation text.\nYour task is to: heat apple.\nAVAILABLE ACTIONS: look,go to fridge 1,open fridge 1"
    gpt = "Thought: I need the fridge.\nAction: go to fridge 1 "

    assert parse_available_actions(human) == ["look", "go to fridge 1", "open fridge 1"]
    assert parse_agentgym_action(gpt) == "go to fridge 1"


def test_convert_agentgym_conversations_enables_l_policy_only_for_exact_available_action():
    rows = [
        {
            "item_id": "task-1",
            "conversations": [
                {"from": "human", "loss": None, "value": "Instruction."},
                {"from": "gpt", "loss": False, "value": "OK."},
                {
                    "from": "human",
                    "loss": None,
                    "value": "Room.\nYour task is to: heat apple.\nAVAILABLE ACTIONS: look,go to fridge 1",
                },
                {"from": "gpt", "loss": True, "value": "Thought: move.\nAction: go to fridge 1 "},
                {"from": "human", "loss": None, "value": "At the fridge."},
                {"from": "gpt", "loss": True, "value": "Action: open fridge 1 "},
            ],
        }
    ]

    converted, report = convert_agentgym_alfworld_conversations(rows)

    assert report["source_quality"] == "weak_policy"
    assert report["trajectory_count"] == 1
    assert report["step_count"] == 2
    assert report["l_policy_enabled_count"] == 1
    assert converted[0]["loss_mask"]["L_policy"] is True
    assert converted[0]["admissible_actions"] == ["look", "go to fridge 1"]
    assert converted[0]["expert_action"] == "go to fridge 1"
    assert converted[1]["loss_mask"]["L_policy"] is False
    assert converted[1]["loss_mask"]["L_trans"] is False
    assert converted[1]["loss_mask"]["STOP"] is True


def test_convert_agentgym_conversations_uses_canonical_alfworld_skill_mapper():
    rows = [
        {
            "item_id": "task-heat",
            "conversations": [
                {
                    "from": "human",
                    "loss": None,
                    "value": "Room.\nYour task is to: heat mug.\nAVAILABLE ACTIONS: heat mug 1 with microwave 1",
                },
                {"from": "gpt", "loss": True, "value": "Action: heat mug 1 with microwave 1"},
            ],
        }
    ]

    converted, _report = convert_agentgym_alfworld_conversations(rows)

    assert converted[0]["skill_id"] == "alfworld/alfworld-heat-object-with-appliance"


def test_audit_agentgym_agenttraj_file_writes_manifest_and_jsonl(tmp_path):
    source = tmp_path / "alfworld_train.json"
    source.write_text(
        json.dumps(
            [
                {
                    "item_id": "task-1",
                    "conversations": [
                        {
                            "from": "human",
                            "loss": None,
                            "value": "Room.\nYour task is to: heat apple.\nAVAILABLE ACTIONS: look,go to fridge 1",
                        },
                        {"from": "gpt", "loss": True, "value": "Action: go to fridge 1"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = audit_agentgym_agenttraj_file(
        source_path=source,
        output_dir=tmp_path / "out",
        benchmark="alfworld",
    )

    assert report["status"] == "ok"
    assert report["train_allowed"] is True
    assert report["bucket"] == "weak_policy"
    assert Path(report["converted_path"]).exists()
    assert Path(report["manifest_path"]).exists()
