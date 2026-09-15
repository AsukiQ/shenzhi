import json
from pathlib import Path

from clstr.alfworld_replay import (
    build_alfworld_train_config,
    extract_alfworld_policy_replay,
    read_replay_rows,
)


class _FakeReplayEnv:
    def __init__(self, invalid_expert: bool = False):
        self.invalid_expert = invalid_expert
        self.num_games = 1
        self.step_no = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.step_no = 0
        expert = "take key 1 from table 1" if self.invalid_expert else "look"
        return [
            "You are in a room.\n\nYour task is to: put a key in the drawer."
        ], {
            "admissible_commands": [["look", "go to drawer 1"]],
            "extra.expert_plan": [[expert]],
            "extra.gamefile": [
                "/data/json_2.1.1/train/pick_and_place_simple-Key-None-Drawer-1/trial_1/game.tw-pddl"
            ],
            "won": [False],
            "goal_condition_success_rate": [0.0],
        }

    def step(self, actions):
        self.step_no += 1
        if self.invalid_expert:
            return ["invalid branch"], [0.0], [False], {
                "admissible_commands": [["look"]],
                "extra.expert_plan": [[]],
                "extra.gamefile": [None],
                "won": [False],
                "goal_condition_success_rate": [0.0],
            }
        if self.step_no == 1:
            assert actions == ["look"]
            return ["You see a drawer 1."], [0.0], [False], {
                "admissible_commands": [["open drawer 1", "look"]],
                "extra.expert_plan": [["open drawer 1"]],
                "extra.gamefile": [None],
                "won": [False],
                "goal_condition_success_rate": [0.5],
            }
        assert actions == ["open drawer 1"]
        return ["The drawer is open."], [1.0], [True], {
            "admissible_commands": [["look"]],
            "extra.expert_plan": [[]],
            "extra.gamefile": [None],
            "won": [True],
            "goal_condition_success_rate": [1.0],
        }

    def close(self):
        self.closed = True


class _FakeReplayWrapper:
    def __init__(self, config, train_eval, invalid_expert: bool = False):
        self.config = config
        self.train_eval = train_eval
        self.num_games = 1
        self.invalid_expert = invalid_expert

    def init_env(self, batch_size):
        assert batch_size == 1
        assert self.train_eval == "train"
        return _FakeReplayEnv(invalid_expert=self.invalid_expert)


def test_build_alfworld_train_config_uses_train_split_only(tmp_path):
    config = build_alfworld_train_config(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        max_games=7,
        max_steps=11,
    )

    assert config["dataset"]["data_path"].endswith("json_2.1.1/train")
    assert config["dataset"]["num_train_games"] == 7
    assert config["dataset"]["eval_id_data_path"] is None
    assert config["dataset"]["eval_ood_data_path"] is None
    assert config["general"]["training_method"] == "dagger"
    assert config["dagger"]["training"]["max_nb_steps_per_episode"] == 11


def test_extract_alfworld_policy_replay_writes_train_rows_and_manifest(tmp_path):
    report = extract_alfworld_policy_replay(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        data_output_dir=tmp_path / "data" / "alfworld_policy_replay",
        report_output_dir=tmp_path / "outputs" / "alfworld_policy_replay",
        max_games=1,
        max_steps=5,
        batch_size=1,
        env_factory=lambda cfg, train_eval: _FakeReplayWrapper(cfg, train_eval),
    )

    replay_path = tmp_path / "data" / "alfworld_policy_replay" / "train_replay.jsonl"
    manifest_path = tmp_path / "data" / "alfworld_policy_replay" / "manifest.json"
    extraction_report_path = tmp_path / "outputs" / "alfworld_policy_replay" / "extraction_report.json"

    assert report["split"] == "train"
    assert report["uses_alfworld_valid_or_test_for_training"] is False
    assert report["train_games_attempted"] == 1
    assert report["train_games_successful_replayed"] == 1
    assert report["extracted_step_count"] == 2
    assert report["usable_policy_step_count"] == 2
    assert report["expert_action_in_admissible_rate"] == 1.0
    assert replay_path.exists()
    assert manifest_path.exists()
    assert extraction_report_path.exists()

    rows = read_replay_rows(replay_path)
    assert rows[0]["split"] == "train"
    assert rows[0]["goal_text"] == "put a key in the drawer."
    assert rows[0]["initial_observation"].startswith("You are in a room.")
    assert rows[0]["observation_t"].startswith("You are in a room.")
    assert rows[0]["history_t"] == []
    assert rows[0]["admissible_commands_t"] == ["look", "go to drawer 1"]
    assert rows[0]["expert_action_t"] == "look"
    assert rows[0]["expert_action_in_admissible"] is True
    assert rows[0]["done_t"] is False
    assert rows[0]["won_t"] is False
    assert rows[0]["goal_condition_success_rate_t"] == 0.5
    assert rows[0]["next_observation_t"] == "You see a drawer 1."
    assert rows[1]["history_t"] == ["look"]
    assert rows[1]["done_t"] is True
    assert rows[1]["won_t"] is True
    assert rows[1]["goal_condition_success_rate_t"] == 1.0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"] == "train"
    assert manifest["valid_or_test_used_for_training"] is False
    assert manifest["replay_path"] == str(replay_path)


def test_extract_alfworld_policy_replay_marks_unusable_when_expert_not_admissible(tmp_path):
    report = extract_alfworld_policy_replay(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        data_output_dir=tmp_path / "data" / "alfworld_policy_replay",
        report_output_dir=tmp_path / "outputs" / "alfworld_policy_replay",
        max_games=1,
        max_steps=5,
        batch_size=1,
        env_factory=lambda cfg, train_eval: _FakeReplayWrapper(cfg, train_eval, invalid_expert=True),
    )

    rows = read_replay_rows(tmp_path / "data" / "alfworld_policy_replay" / "train_replay.jsonl")
    assert len(rows) == 1
    assert rows[0]["expert_action_t"] == "take key 1 from table 1"
    assert rows[0]["expert_action_in_admissible"] is False
    assert rows[0]["usable_for_policy"] is False
    assert rows[0]["skip_reason"] == "expert_action_not_in_admissible"
    assert report["usable_policy_step_count"] == 0
    assert report["expert_action_in_admissible_rate"] == 0.0
    assert report["skipped_reasons"]["expert_action_not_in_admissible"] == 1
