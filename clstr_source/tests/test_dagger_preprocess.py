import json
from pathlib import Path

from clstr.dagger_preprocess import build_dagger_train_data, convert_expert_corrected_rollout_to_full_base_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_convert_expert_corrected_rollout_uses_official_expert_as_positive_and_qwen_as_negative(tmp_path):
    rollout = tmp_path / "train_rollout.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "split": "train",
                "bucket": "dagger_expert_corrected_rollout",
                "gamefile": "/data/json_2.1.1/train/pick_and_place_simple-Key-None-Drawer-1/trial/game.tw-pddl",
                "episode_index": 0,
                "step_index": 0,
                "task_type": "pick_and_place_simple",
                "goal_text": "put key in drawer",
                "observation_t": "room",
                "history_t": [],
                "admissible_commands_t": ["look", "go to drawer 1"],
                "qwen_action_t": "look",
                "official_expert_action_t": "go to drawer 1",
                "expert_action_t": "go to drawer 1",
                "expert_action_in_admissible": True,
                "qwen_action_in_admissible": True,
                "qwen_action_matches_expert": False,
                "usable_for_expert_ce": True,
                "next_observation_t": "looked around",
                "done_t": False,
                "won_t": False,
                "reward_t": 0.0,
            },
            {
                "split": "valid_seen",
                "bucket": "dagger_expert_corrected_rollout",
                "usable_for_expert_ce": True,
                "admissible_commands_t": ["look"],
                "official_expert_action_t": "look",
            },
        ],
    )
    skills = [{"skill_id": "alfworld/alfworld-navigation", "name": "go"}]

    rows, report = convert_expert_corrected_rollout_to_full_base_rows(rollout, skills)

    assert report["source_split"] == "train"
    assert report["bucket"] == "dagger_expert_corrected_rollout"
    assert report["converted_count"] == 1
    assert report["excluded_counts"]["excluded_split_valid_seen"] == 1
    row = rows[0]
    assert row["expert_action"] == "go to drawer 1"
    assert row["action_text"] == "go to drawer 1"
    assert row["planner_proposed_action"] == "look"
    assert row["hard_negative_action"] == "look"
    assert row["q_success_labels"] == [0.0, 1.0]
    assert row["q_success_label_weights"] == [0.5, 1.0]
    assert row["loss_mask"]["L_policy"] is True
    assert row["loss_mask"]["hard_negative_margin"] is True
    assert row["loss_mask"]["Q_success"] is True
    assert row["source_quality"] == "dagger_expert_corrected_rollout"
    assert row["provenance"]["bucket"] == "dagger_expert_corrected_rollout"


def test_build_dagger_train_data_writes_manifest_and_report(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    skills_path = tmp_path / "skills.jsonl"
    _write_jsonl(
        rollout,
        [
            {
                "split": "train",
                "bucket": "dagger_expert_corrected_rollout",
                "gamefile": "/data/json_2.1.1/train/pick_clean_then_place_in_recep-Apple-None-Sink-1/trial/game.tw-pddl",
                "episode_index": 0,
                "step_index": 0,
                "task_type": "pick_clean_then_place_in_recep",
                "goal_text": "clean apple",
                "observation_t": "sink",
                "history_t": [],
                "admissible_commands_t": ["look", "clean apple 1 with sinkbasin 1"],
                "qwen_action_t": "clean apple 1 with sinkbasin 1",
                "official_expert_action_t": "clean apple 1 with sinkbasin 1",
                "expert_action_t": "clean apple 1 with sinkbasin 1",
                "expert_action_in_admissible": True,
                "qwen_action_in_admissible": True,
                "qwen_action_matches_expert": True,
                "usable_for_expert_ce": True,
                "next_observation_t": "clean",
                "done_t": True,
                "won_t": True,
                "reward_t": 1.0,
            }
        ],
    )
    _write_jsonl(skills_path, [{"skill_id": "alfworld/alfworld-cleaning", "name": "clean"}])

    report = build_dagger_train_data(
        rollout_path=rollout,
        skills_path=skills_path,
        output_dir=tmp_path / "data" / "clstr_dagger_expert_corrected_train",
        report_output_dir=tmp_path / "outputs" / "clstr_dagger_expert_corrected_train",
    )

    train_path = Path(report["train_path"])
    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    assert manifest["bucket"] == "dagger_expert_corrected_rollout"
    assert manifest["valid_or_test_used_for_training"] is False
    assert report["loss_activation_counts"]["Q_success"] == 1
    assert rows[0]["loss_mask"]["Q_success"] is True
