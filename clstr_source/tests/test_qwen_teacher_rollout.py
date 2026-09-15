import json
from pathlib import Path

import torch

from clstr.qwen_teacher_rollout import extract_qwen_teacher_rollout


class _FakeTeacherPlanner:
    def __init__(self):
        self.calls = []
        self.last_metadata = []

    def __call__(self, state_texts, candidate_rows):
        self.calls.append((state_texts, candidate_rows))
        rows = []
        metadata = []
        for candidates in candidate_rows:
            action = "look" if "look" in candidates else candidates[0]
            rows.append([1.0 if item == action else 0.0 for item in candidates])
            metadata.append(
                {
                    "raw_model_response": f"ACTION: {action}",
                    "parsed_action": action,
                    "parse_status": "exact_match",
                    "fallback_used": False,
                    "policy_family": "qwen_teacher_admissible",
                }
            )
        self.last_metadata = metadata
        return torch.tensor(rows, dtype=torch.float32)


class _FakeRolloutEnv:
    def __init__(self):
        self.step_no = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.step_no = 0
        return [
            "You are in a room.\n\nYour task is to: put a key in the drawer."
        ], {
            "admissible_commands": [["look", "go to drawer 1"]],
            "extra.expert_plan": [["go to drawer 1", "open drawer 1"]],
            "extra.gamefile": [
                "/data/json_2.1.1/train/pick_and_place_simple-Key-None-Drawer-1/trial_1/game.tw-pddl"
            ],
            "won": [False],
            "goal_condition_success_rate": [0.0],
        }

    def step(self, actions):
        self.step_no += 1
        assert actions == ["look"]
        return ["You see drawer 1."], [0.25], [True], {
            "admissible_commands": [["open drawer 1", "look"]],
            "extra.expert_plan": [["open drawer 1"]],
            "extra.gamefile": [None],
            "won": [True],
            "goal_condition_success_rate": [1.0],
        }

    def close(self):
        self.closed = True


class _FakeWrapper:
    num_games = 1

    def __init__(self, config, train_eval):
        self.config = config
        self.train_eval = train_eval

    def init_env(self, batch_size):
        assert batch_size == 1
        assert self.train_eval == "train"
        return _FakeRolloutEnv()


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_qwen_teacher_rollout_writes_train_only_expert_corrected_rows(tmp_path):
    report = extract_qwen_teacher_rollout(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        data_output_dir=tmp_path / "data" / "alfworld_qwen3_expert_corrected_rollout",
        report_output_dir=tmp_path / "outputs" / "alfworld_qwen3_expert_corrected_rollout",
        max_games=1,
        max_steps=3,
        batch_size=1,
        planner=_FakeTeacherPlanner(),
        env_factory=lambda cfg, train_eval: _FakeWrapper(cfg, train_eval),
    )

    replay_path = tmp_path / "data" / "alfworld_qwen3_expert_corrected_rollout" / "train_rollout.jsonl"
    manifest_path = tmp_path / "data" / "alfworld_qwen3_expert_corrected_rollout" / "manifest.json"
    rows = _read_jsonl(replay_path)

    assert report["split"] == "train"
    assert report["bucket"] == "dagger_expert_corrected_rollout"
    assert report["official_replay"] is False
    assert report["verified_env_replay"] is True
    assert report["uses_alfworld_valid_or_test_for_training"] is False
    assert report["train_games_attempted"] == 1
    assert report["extracted_step_count"] == 1
    assert report["usable_policy_step_count"] == 1
    assert report["expert_action_in_admissible_rate"] == 1.0
    assert report["qwen_action_matches_expert_rate"] == 0.0
    assert rows[0]["split"] == "train"
    assert rows[0]["bucket"] == "dagger_expert_corrected_rollout"
    assert rows[0]["qwen_action_t"] == "look"
    assert rows[0]["official_expert_action_t"] == "go to drawer 1"
    assert rows[0]["expert_action_t"] == "go to drawer 1"
    assert rows[0]["teacher_action_t"] == "go to drawer 1"
    assert rows[0]["expert_action_source"] == "official_alfworld_extra_expert_plan"
    assert rows[0]["expert_action_in_admissible"] is True
    assert rows[0]["usable_for_expert_ce"] is True
    assert rows[0]["usable_for_policy"] is True
    assert rows[0]["qwen_action_in_admissible"] is True
    assert rows[0]["qwen_action_matches_expert"] is False
    assert rows[0]["qwen_action_parse_status"] == "exact_match"
    assert rows[0]["hard_negative_action_t"] == "look"
    assert rows[0]["first_mismatch_step"] == 0
    assert rows[0]["reward_t"] == 0.25
    assert rows[0]["done_t"] is True
    assert rows[0]["won_t"] is True
    assert rows[0]["next_observation_t"] == "You see drawer 1."
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split"] == "train"
    assert manifest["bucket"] == "dagger_expert_corrected_rollout"
    assert manifest["official_replay"] is False
