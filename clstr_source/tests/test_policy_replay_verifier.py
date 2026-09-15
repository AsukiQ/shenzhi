import json
from pathlib import Path

from clstr.envs.base import EnvStep
from clstr.policy_replay_verifier import (
    MissingReplayHarness,
    verify_policy_replay,
    write_policy_replay_blocker,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _ScriptedHarness:
    def __init__(self, candidate_script: list[list[str]]):
        self.candidate_script = list(candidate_script)
        self.step_index = 0
        self.actions = []
        self.observation = "initial observation"

    def reset(self, task_id=None):
        self.task_id = task_id
        self.step_index = 0
        self.actions = []
        self.observation = "initial observation"
        return self.observation

    def candidate_actions(self):
        if self.step_index < len(self.candidate_script):
            return list(self.candidate_script[self.step_index])
        return []

    def step(self, action_text):
        self.actions.append(action_text)
        self.step_index += 1
        self.observation = f"obs after {action_text}"
        return EnvStep(
            observation_text=self.observation,
            reward=1.0 if self.step_index >= len(self.candidate_script) else 0.0,
            done=self.step_index >= len(self.candidate_script),
            success=self.step_index >= len(self.candidate_script),
            valid_actions=self.candidate_actions(),
        )


def test_verify_policy_replay_aborts_trajectory_after_first_candidate_mismatch(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "split": "train",
                "environment": "scienceworld",
                "task_id": "task-1",
                "task_type": "boil",
                "goal_text": "boil lead",
                "steps": [
                    {"t": 0, "observation_text": "initial observation", "action_text": "look around"},
                    {"t": 1, "observation_text": "obs after look around", "action_text": "invalid expert"},
                    {"t": 2, "observation_text": "obs after invalid expert", "action_text": "open door"},
                ],
            },
            {
                "split": "valid_seen",
                "environment": "scienceworld",
                "task_id": "leaky-valid",
                "steps": [{"t": 0, "observation_text": "valid obs", "action_text": "look around"}],
            },
        ],
    )

    report = verify_policy_replay(
        source_path=source_path,
        output_path=tmp_path / "verified.jsonl",
        manifest_path=tmp_path / "manifest.json",
        benchmark="scienceworld",
        source_id="mock_scienceworld",
        source_dataset="mock/source",
        harness_factory=lambda _trajectory: _ScriptedHarness(
            [
                ["look around", "open door"],
                ["open door"],
                ["open door"],
            ]
        ),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "verified.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert report["status"] == "ok"
    assert report["split"] == "train"
    assert report["valid_or_test_used_for_training"] is False
    assert report["trajectories_attempted"] == 1
    assert report["trajectories_skipped_by_split"] == 1
    assert report["extracted_step_count"] == 2
    assert report["usable_policy_step_count"] == 1
    assert report["expert_action_in_admissible_rate"] == 1 / 2
    assert report["trajectories_aborted_after_desync"] == 1
    assert report["skipped_reasons"]["expert_action_not_in_admissible"] == 1
    assert report["skipped_reasons"]["steps_skipped_after_desync"] == 1
    assert [row["expert_action_t"] for row in rows] == ["look around"]
    assert rows[0]["admissible_commands_t"] == ["look around", "open door"]
    assert rows[0]["bucket"] == "verified_replay"
    assert rows[0]["usable_for_policy"] is True
    assert rows[0]["provenance"]["source_id"] == "mock_scienceworld"
    assert rows[0]["history_t"] == []


def test_verify_policy_replay_streams_source_until_max_trajectories(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    first_trajectory = {
        "split": "train",
        "environment": "scienceworld",
        "task_id": "task-1",
        "task_type": "boil",
        "goal_text": "boil lead",
        "steps": [{"t": 0, "observation_text": "initial observation", "action_text": "look around"}],
    }
    source_path.write_text(json.dumps(first_trajectory) + "\n{not valid json}\n", encoding="utf-8")

    report = verify_policy_replay(
        source_path=source_path,
        output_path=tmp_path / "verified.jsonl",
        manifest_path=tmp_path / "manifest.json",
        benchmark="scienceworld",
        source_id="mock_scienceworld",
        source_dataset="mock/source",
        harness_factory=lambda _trajectory: _ScriptedHarness([["look around"]]),
        max_trajectories=1,
    )

    assert report["status"] == "ok"
    assert report["trajectories_attempted"] == 1
    assert report["train_rows_written"] == 1


def test_policy_replay_blocker_is_written_when_official_harness_is_missing(tmp_path):
    report = write_policy_replay_blocker(
        output_path=tmp_path / "blocker.json",
        benchmark="webshop",
        source_id="webshop_aux",
        source_dataset="data/aux_skillnet_rebuilt/webshop/trajectories.jsonl",
        error=MissingReplayHarness("official WebShop text env is unavailable"),
        repro_command="python scripts/verify_policy_replay_candidates.py --benchmark webshop",
    )

    loaded = json.loads((tmp_path / "blocker.json").read_text(encoding="utf-8"))
    assert report == loaded
    assert report["status"] == "blocked_missing_replay_harness"
    assert report["train_rows_written"] == 0
    assert report["can_upgrade_to_verified_replay"] is False
    assert report["repro_command"].startswith("python scripts/verify_policy_replay_candidates.py")
