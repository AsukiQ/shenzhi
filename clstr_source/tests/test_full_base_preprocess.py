import json
from pathlib import Path

from clstr.dataset_registry import SourceSpec, build_clstr_full_base_registry
from clstr.full_base_preprocess import build_clstr_full_base_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_full_base_data_converts_official_replay_and_aux_skillnet_rows(tmp_path):
    replay_path = tmp_path / "data" / "alfworld_policy_replay" / "train_replay.jsonl"
    _write_jsonl(
        replay_path,
        [
            {
                "split": "train",
                "episode_index": 0,
                "step_index": 0,
                "gamefile": "/train/game.tw-pddl",
                "task_type": "pick_and_place",
                "goal_text": "put mug in drawer",
                "observation_t": "You see a mug.",
                "history_t": [],
                "admissible_commands_t": ["take mug 1 from table 1", "look"],
                "expert_action_t": "take mug 1 from table 1",
                "expert_action_in_admissible": True,
                "next_observation_t": "You pick up the mug.",
                "done_t": False,
                "won_t": False,
                "goal_condition_success_rate_t": 0.0,
            }
        ],
    )
    aux_root = tmp_path / "data" / "aux_skillnet_rebuilt"
    _write_jsonl(
        aux_root / "pseudo_skills.jsonl",
        [
            {
                "skill_id": "alfworld/alfworld-object-picker",
                "name": "alfworld-object-picker",
                "description": "pick objects",
                "environment": "alfworld",
            },
            {
                "skill_id": "scienceworld/scienceworld-room-navigator",
                "name": "scienceworld-room-navigator",
                "description": "navigate rooms",
                "environment": "scienceworld",
            },
        ],
    )
    _write_jsonl(
        aux_root / "trajectories.jsonl",
        [
            {
                "split": "train",
                "environment": "scienceworld",
                "task_id": "sci-1",
                "task_type": "chemistry",
                "steps": [
                    {
                        "t": 0,
                        "observation_text": "You are in the foundry.",
                        "action_text": "teleport to kitchen",
                        "next_observation_text": "You teleport to the kitchen.",
                        "done": False,
                        "reward": 0.0,
                        "pseudo_skill_id": "scienceworld/scienceworld-room-navigator",
                        "usable_for_skillnet_training": True,
                    }
                ],
            }
        ],
    )
    registry_dir = tmp_path / "registry"
    build_clstr_full_base_registry(
        repo_root=tmp_path,
        output_dir=registry_dir,
        report_dir=tmp_path / "registry_reports",
        source_specs=[
            SourceSpec(
                source_id="alfworld_official_train_replay",
                benchmark="alfworld",
                source_dataset="data/alfworld_policy_replay/train_replay.jsonl",
                split="train",
                bucket="official_replay",
                allowed_losses=["L_policy", "L_trans", "STOP", "routing"],
                path="data/alfworld_policy_replay/train_replay.jsonl",
                has_admissible_actions=True,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=False,
                leakage_risk="none",
                train_allowed=True,
                reason_if_excluded=None,
            ),
            SourceSpec(
                source_id="aux_skillnet_rebuilt",
                benchmark="multi",
                source_dataset="data/aux_skillnet_rebuilt",
                split="train",
                bucket="aux_only",
                allowed_losses=["L_trans", "belief", "STOP", "routing"],
                path="data/aux_skillnet_rebuilt",
                has_admissible_actions=False,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=True,
                leakage_risk="none",
                train_allowed=True,
                reason_if_excluded=None,
            ),
        ],
    )

    manifest = build_clstr_full_base_data(
        registry_path=registry_dir / "sources.jsonl",
        output_dir=tmp_path / "full_base_train",
        report_dir=tmp_path / "full_base_reports",
        repo_root=tmp_path,
    )

    assert manifest["status"] == "ok"
    assert manifest["record_count"] == 2
    assert manifest["benchmark_counts"] == {"alfworld": 1, "scienceworld": 1}
    assert manifest["loss_activation_counts"]["L_policy"] == 1
    assert manifest["loss_activation_counts"]["L_trans"] == 2
    assert manifest["loss_activation_counts"]["L_trans_skill_ce"] == 0
    assert manifest["loss_activation_counts"]["STOP"] == 2
    assert manifest["loss_activation_counts"]["routing"] == 2
    assert manifest["uses_valid_or_test_for_training"] is False
    assert (tmp_path / "full_base_train" / "train.jsonl").exists()
    assert (tmp_path / "full_base_train" / "skills.jsonl").exists()
    assert (tmp_path / "full_base_reports" / "preprocess_report.json").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "full_base_train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows[0]["loss_mask"]["L_policy"] is True
    assert rows[0]["admissible_actions"] == ["take mug 1 from table 1", "look"]
    assert rows[1]["loss_mask"]["L_policy"] is False
    assert rows[1]["loss_mask"]["belief"] is True
    assert rows[1]["skill_id"] == "scienceworld/scienceworld-room-navigator"


def test_official_replay_conversion_preserves_full_replay_prefix_and_offline_regime(tmp_path):
    replay_path = tmp_path / "data" / "alfworld_policy_replay" / "train_replay.jsonl"
    _write_jsonl(
        replay_path,
        [
            {
                "split": "train",
                "episode_index": 0,
                "step_index": 0,
                "gamefile": "/train/game.tw-pddl",
                "task_type": "pick_and_place",
                "goal_text": "put mug in drawer",
                "observation_t": "You see a mug.",
                "history_t": [],
                "admissible_commands_t": ["take mug 1 from table 1", "look"],
                "expert_action_t": "take mug 1 from table 1",
                "expert_action_in_admissible": True,
                "next_observation_t": "You pick up the mug.",
                "done_t": False,
            },
            {
                "split": "train",
                "episode_index": 0,
                "step_index": 1,
                "gamefile": "/train/game.tw-pddl",
                "task_type": "pick_and_place",
                "goal_text": "put mug in drawer",
                "observation_t": "You are holding a mug.",
                "history_t": ["take mug 1 from table 1"],
                "admissible_commands_t": ["open drawer 1", "look"],
                "expert_action_t": "open drawer 1",
                "expert_action_in_admissible": True,
                "next_observation_t": "The drawer is open.",
                "done_t": False,
            },
        ],
    )
    aux_root = tmp_path / "data" / "aux_skillnet_rebuilt"
    _write_jsonl(
        aux_root / "pseudo_skills.jsonl",
        [
            {
                "skill_id": "alfworld/alfworld-object-picker",
                "name": "alfworld-object-picker",
                "description": "pick objects",
                "environment": "alfworld",
            },
            {
                "skill_id": "alfworld/alfworld-receptacle-opener",
                "name": "alfworld-receptacle-opener",
                "description": "open receptacles",
                "environment": "alfworld",
            },
        ],
    )
    registry_dir = tmp_path / "registry"
    build_clstr_full_base_registry(
        repo_root=tmp_path,
        output_dir=registry_dir,
        report_dir=tmp_path / "registry_reports",
        source_specs=[
            SourceSpec(
                source_id="alfworld_official_train_replay",
                benchmark="alfworld",
                source_dataset="data/alfworld_policy_replay/train_replay.jsonl",
                split="train",
                bucket="official_replay",
                allowed_losses=["L_policy", "L_trans", "L_trans_skill_ce", "STOP", "routing"],
                path="data/alfworld_policy_replay/train_replay.jsonl",
                has_admissible_actions=True,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=False,
                leakage_risk="none",
                train_allowed=True,
                reason_if_excluded=None,
            ),
        ],
    )

    manifest = build_clstr_full_base_data(
        registry_path=registry_dir / "sources.jsonl",
        output_dir=tmp_path / "full_base_train",
        report_dir=tmp_path / "full_base_reports",
        repo_root=tmp_path,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "full_base_train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert manifest["supports_offline_replay_prefix"] is True
    assert manifest["supports_on_policy_rollout"] is False
    assert rows[0]["trajectory_id"] == "/train/game.tw-pddl"
    assert rows[0]["step_index"] == 0
    assert rows[0]["trajectory_step_count"] == 2
    assert rows[0]["replay_prefix"] == []
    assert rows[0]["next_action_text"] == "open drawer 1"
    assert rows[0]["next_skill_id"] == "alfworld/alfworld-receptacle-opener"
    assert rows[0]["loss_mask"]["L_trans_skill_ce"] is True
    assert rows[0]["candidate_source"] == "alfworld_admissible_commands_t"
    assert rows[0]["on_policy_rollout"] is False
    assert rows[0]["m_t_source"] == "offline_full_replay_prefix_or_observation"
    assert rows[1]["replay_prefix"] == [
        {
            "step_index": 0,
            "observation_text": "You see a mug.",
            "action_text": "take mug 1 from table 1",
            "next_observation_text": "You pick up the mug.",
            "skill_id": "alfworld/alfworld-object-picker",
        }
    ]
    assert rows[1]["next_action_text"] is None
    assert rows[1]["next_skill_id"] is None
    assert rows[1]["loss_mask"]["L_trans_skill_ce"] is False


def test_build_full_base_data_accepts_generic_verified_replay_for_l_policy(tmp_path):
    replay_path = tmp_path / "data" / "verified_policy_replay" / "scienceworld_train_replay.jsonl"
    _write_jsonl(
        replay_path,
        [
            {
                "split": "train",
                "benchmark": "scienceworld",
                "bucket": "verified_replay",
                "trajectory_id": "science-task-1",
                "step_index": 0,
                "trajectory_step_count": 1,
                "task_type": "boil",
                "goal_text": "boil lead",
                "observation_t": "You are in the kitchen.",
                "history_t": [],
                "admissible_commands_t": ["look around", "open door"],
                "expert_action_t": "open door",
                "expert_action_in_admissible": True,
                "done_t": False,
                "reward_t": 0.0,
                "next_observation_t": "The door is open.",
                "replay_prefix": [],
                "candidate_source": "official_harness_valid_actions",
                "provenance": {"source_id": "scienceworld_verified_replay"},
            },
            {
                "split": "valid_seen",
                "benchmark": "scienceworld",
                "trajectory_id": "leaky-valid",
                "step_index": 0,
                "admissible_commands_t": ["look around"],
                "expert_action_t": "look around",
                "expert_action_in_admissible": True,
            },
        ],
    )
    aux_root = tmp_path / "data" / "aux_skillnet_rebuilt"
    _write_jsonl(
        aux_root / "pseudo_skills.jsonl",
        [
            {
                "skill_id": "scienceworld/scienceworld-room-scanner",
                "name": "scienceworld-room-scanner",
                "description": "inspect rooms",
                "environment": "scienceworld",
            },
        ],
    )
    registry_dir = tmp_path / "registry"
    build_clstr_full_base_registry(
        repo_root=tmp_path,
        output_dir=registry_dir,
        report_dir=tmp_path / "registry_reports",
        source_specs=[
            SourceSpec(
                source_id="scienceworld_verified_replay",
                benchmark="scienceworld",
                source_dataset="data/verified_policy_replay/scienceworld_train_replay.jsonl",
                split="train",
                bucket="verified_replay",
                allowed_losses=["L_policy", "L_trans", "STOP", "routing"],
                path="data/verified_policy_replay/scienceworld_train_replay.jsonl",
                has_admissible_actions=True,
                has_expert_action=True,
                has_next_observation=True,
                has_done=True,
                has_reward=True,
                leakage_risk="none",
                train_allowed=True,
                reason_if_excluded=None,
            )
        ],
    )

    manifest = build_clstr_full_base_data(
        registry_path=registry_dir / "sources.jsonl",
        output_dir=tmp_path / "full_base_train",
        report_dir=tmp_path / "full_base_reports",
        repo_root=tmp_path,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "full_base_train" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert manifest["record_count"] == 1
    assert manifest["bucket_counts"] == {"verified_replay": 1}
    assert manifest["loss_activation_counts"]["L_policy"] == 1
    assert manifest["uses_valid_or_test_for_training"] is False
    assert rows[0]["benchmark"] == "scienceworld"
    assert rows[0]["source_quality"] == "verified_replay"
    assert rows[0]["candidate_source"] == "official_harness_valid_actions"
    assert rows[0]["loss_mask"]["L_policy"] is True
    assert rows[0]["admissible_actions"] == ["look around", "open door"]
