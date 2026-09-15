import json
from pathlib import Path

from clstr.structured_train import convert_teacher_rollout_to_full_base_rows


def test_convert_teacher_rollout_to_full_base_rows_uses_verified_train_only(tmp_path):
    rollout = tmp_path / "train_rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "split": "train",
                "bucket": "verified_teacher_rollout",
                "gamefile": "/data/json_2.1.1/train/pick_cool_then_place_in_recep-Apple-None-Fridge-1/trial/game.tw-pddl",
                "episode_index": 0,
                "step_index": 0,
                "task_type": "pick_cool_then_place_in_recep",
                "goal_text": "cool apple and put it in fridge",
                "observation_t": "kitchen",
                "history_t": [],
                "admissible_commands_t": ["look", "open fridge 1"],
                "teacher_action_t": "open fridge 1",
                "expert_action_t": "open fridge 1",
                "expert_action_source": "qwen3_teacher_verified",
                "expert_action_in_admissible": True,
                "usable_for_policy": True,
                "parse_status": "exact_match",
                "fallback_used": False,
                "next_observation_t": "fridge is open",
                "done_t": False,
                "reward_t": 0.0,
                "won_t": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "split": "valid_seen",
                "bucket": "verified_teacher_rollout",
                "usable_for_policy": True,
                "admissible_commands_t": ["look"],
                "expert_action_t": "look",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    skills = [{"skill_id": "alfworld/alfworld-receptacle-opener", "name": "open"}]

    rows, report = convert_teacher_rollout_to_full_base_rows(rollout, skills)

    assert report["source_split"] == "train"
    assert report["valid_or_test_used_for_training"] is False
    assert report["converted_count"] == 1
    assert rows[0]["source_quality"] == "verified_teacher_rollout"
    assert rows[0]["expert_action"] == "open fridge 1"
    assert rows[0]["planner_proposed_action"] == "open fridge 1"
    assert "Qwen proposed action: open fridge 1" in rows[0]["state_text"]
    assert rows[0]["loss_mask"]["L_policy"] is True
    assert rows[0]["loss_mask"]["STOP"] is True
    assert rows[0]["provenance"]["split"] == "train"
