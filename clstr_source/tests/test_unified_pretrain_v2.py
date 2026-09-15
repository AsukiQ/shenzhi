from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from clstr.traject_split import assign_traject_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_builder():
    path = Path("scripts/build_clstr_unified_pretrain.py")
    spec = importlib.util.spec_from_file_location("build_clstr_unified_pretrain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unified_pretrain_v2_builds_skill_pool_inventory_and_leakage_audit(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [{"task_id": "dev-task"}])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [{"task_id": "test-task"}])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [{"task_id": "challenge-task"}])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "train-task",
                "trajectory_id": "traj-1",
                "step_index": 0,
                "state_text": "goal: train",
                "action_text": "go",
                "next_observation_text": "arrived",
                "skill_id": "traj/skill",
                "next_skill_id": "traj/next",
                "loss_mask": {"L_policy": True, "L_trans": True},
            },
            {
                "benchmark": "appworld",
                "task_id": "dev-task",
                "trajectory_id": "leaky",
                "step_index": 0,
                "state_text": "must be excluded",
                "action_text": "bad",
                "skill_id": "leaky/skill",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "traj/skill", "name": "trajectory skill", "description": "Used by train trajectory."},
            {"skill_id": "traj/next", "name": "next skill", "description": "Used as next skill."},
            {"skill_id": "unused/skill", "name": "unused", "description": "Not used."},
        ],
    )
    _write_jsonl(
        tmp_path / "data/skillret/qrels.jsonl",
        [
            {"query_id": "q-train", "skill_id": "skill-train", "relevance": 1, "split": "train"},
            {"query_id": "q-test", "skill_id": "skill-test", "relevance": 1, "split": "test"},
        ],
    )
    _write_jsonl(
        tmp_path / "data/skillret/queries.jsonl",
        [
            {"query_id": "q-train", "query": "train query", "split": "train"},
            {"query_id": "q-test", "query": "test query", "split": "test"},
        ],
    )
    _write_jsonl(
        tmp_path / "data/skillret/skills.jsonl",
        [
            {"skill_id": "skill-train", "name": "train skill", "description": "Allowed train skill."},
            {"skill_id": "skill-test", "name": "test skill", "description": "Must stay out of train pool."},
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"

    assert manifest["schema_version"] == "v2"
    assert manifest["files"]["skill_pool"] == "skill_pool.jsonl"
    assert manifest["files"]["source_inventory"] == "source_inventory.jsonl"
    assert manifest["files"]["leakage_audit"] == "leakage_audit.json"
    assert manifest["files"]["history_channel_audit"] == "history_channel_audit.json"
    assert manifest["history_channel_audit"]["status"] == "ok"
    assert manifest["trajectory_stream"]["total_rows"] == 1
    assert manifest["trajectory_stream"]["leakage_filtered"] == 1
    assert manifest["retrieval_stream"]["total_pairs"] == 3
    assert manifest["retrieval_stream"]["leakage_filtered"] == 1

    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    audit = json.loads((out_dir / "leakage_audit.json").read_text())

    assert [row["task_id"] for row in trajectories] == ["train-task"]
    assert [row["query_id"] for row in retrieval] == [
        "q-train",
        "trajectory-derived::alfworld::traj-1::0::current",
        "trajectory-derived::alfworld::traj-1::0::next",
    ]
    skill_ids = {row["skill_id"] for row in skill_pool}
    assert {"traj/skill", "traj/next", "skill-train"} <= skill_ids
    assert "skill-test" not in skill_ids
    assert "unused/skill" not in skill_ids
    assert audit["status"] == "ok"
    assert audit["leaked_appworld_task_ids_in_trajectories"] == []
    assert audit["leaked_skillret_query_ids_in_retrieval"] == []
    assert any(row["source_id"] == "traject_bench" and not row["available"] for row in inventory)
    assert any(row["source_id"] == "toolbench_g3" and not row["available"] for row in inventory)
    assert any(row["source_id"] == "toolret_training" and not row["available"] for row in inventory)


def test_unified_pretrain_can_append_extra_retrieval_rows_without_appworld_leakage(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [{"task_id": "dev-task"}])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [{"query_id": "skillret-test", "skill_id": "skill-test", "relevance": 1, "split": "test"}])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/appworld_skill_pool/skill_pool.jsonl",
        [
            {"skill_id": "skill/write", "name": "write skill", "description": "Execute a write action."},
            {"skill_id": "skill/leaky", "name": "leaky skill", "description": "Should not enter."},
        ],
    )
    extra_path = tmp_path / "extra_retrieval.jsonl"
    _write_jsonl(
        extra_path,
        [
            {
                "source": "stage0_high_confidence_correction",
                "query_id": "train-extra",
                "query_text": "create the playlist",
                "positive_skill_id": "skill/write",
                "negative_skill_ids": ["skill/leaky"],
                "provenance": {"source_dataset": "unit_test", "task_id": "train-task"},
            },
            {
                "source": "stage0_high_confidence_correction",
                "query_id": "dev-task",
                "query_text": "must be filtered",
                "positive_skill_id": "skill/leaky",
                "negative_skill_ids": [],
                "provenance": {"source_dataset": "unit_test", "task_id": "dev-task"},
            },
            {
                "source": "stage0_high_confidence_correction",
                "query_id": "skillret-test",
                "query_text": "must also be filtered",
                "positive_skill_id": "skill-leaky",
                "negative_skill_ids": [],
                "provenance": {"source_dataset": "unit_test"},
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_extra",
        schema_version="v2",
        extra_retrieval_paths=[extra_path],
    )

    out_dir = tmp_path / "data/clstr_unified_pretrain_extra"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    assert [row["query_id"] for row in retrieval] == ["train-extra"]
    assert retrieval[0]["source"] == "stage0_high_confidence_correction"
    assert {row["skill_id"] for row in skill_pool} == {"skill/write", "skill/leaky"}
    assert manifest["retrieval_stream"]["extra_retrieval"]["total_pairs"] == 1
    assert manifest["retrieval_stream"]["extra_retrieval"]["leakage_filtered"] == 2


def test_unified_pretrain_v4_1_filters_aux_full_base_and_adds_high_reward_webshop(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "alf-official-0",
                "trajectory_id": "alf-official",
                "step_index": 0,
                "goal_text": "examine the key",
                "state_text": "room observation",
                "action_text": "look",
                "next_observation_text": "desk visible",
                "skill_id": "alfworld/look",
                "next_skill_id": "alfworld/go",
                "source_quality": "official_replay",
                "provenance": {"source_id": "alfworld_official_train_replay"},
            },
            {
                "benchmark": "scienceworld",
                "task_id": "sci-aux-0",
                "trajectory_id": "sci-aux",
                "step_index": 0,
                "goal_text": "boil chocolate",
                "state_text": "failed repeated lab state",
                "action_text": "wait",
                "next_observation_text": "still waiting",
                "skill_id": "scienceworld/wait",
                "next_skill_id": "scienceworld/wait",
                "source_quality": "aux_only",
                "provenance": {"source_id": "aux_skillnet_rebuilt"},
            },
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "alfworld/look", "name": "look", "description": "Inspect the current room."},
            {"skill_id": "alfworld/go", "name": "go", "description": "Navigate to a target."},
            {"skill_id": "scienceworld/wait", "name": "wait", "description": "Wait in ScienceWorld."},
        ],
    )
    webshop_rows = [
        {
            "info": {"reward": 1.0, "llm_name": "gpt-4-0613", "agent_arch": "react"},
            "id": 1500,
            "question": "find a white king bed under 370 dollars",
            "conversations": [
                {"role": "user", "content": "Task: find a white king bed under 370 dollars"},
                {"role": "assistant", "content": "Action:Think[{\"response\": \"Search first.\"}]\n"},
                {"role": "user", "content": "Observation: OK"},
                {"role": "assistant", "content": "Action:search[{\"product\": \"white king bed\"}]\n"},
                {"role": "user", "content": "Observation: results page"},
                {"role": "assistant", "content": "Action:click[{\"button\": \"B096\"}]\n"},
                {"role": "user", "content": "Observation: product page"},
                {"role": "assistant", "content": "Action:Finish[{\"response\": \"done\"}]\n"},
            ],
        },
        {
            "info": {"reward": 0.25, "llm_name": "gpt-4-0613", "agent_arch": "react"},
            "id": 1501,
            "question": "low quality row should stay out",
            "conversations": [
                {"role": "user", "content": "Task: low quality"},
                {"role": "assistant", "content": "Action:search[{\"product\": \"bad\"}]\n"},
            ],
        },
    ]
    (tmp_path / "data/webshop_expert_trajectories").mkdir(parents=True)
    (tmp_path / "data/webshop_expert_trajectories/webshop_gpt4_0613.json").write_text(
        json.dumps(webshop_rows),
        encoding="utf-8",
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v4_1",
        schema_version="v4.1",
        dataset_recipe="v4_1",
        webshop_min_reward=0.8,
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v4_1"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    by_source = {row["source_id"]: row for row in inventory}

    assert manifest["dataset_recipe"] == "v4_1"
    assert manifest["trajectory_stream"]["quality_filters"]["filtered_rows_by_reason"] == {
        "v4_1_aux_only_full_base_filtered:scienceworld": 1
    }
    assert manifest["trajectory_stream"]["counts"]["benchmark:alfworld"] == 1
    assert manifest["trajectory_stream"]["counts"]["benchmark:webshop"] == 4
    assert "benchmark:scienceworld" not in manifest["trajectory_stream"]["counts"]
    assert [row["benchmark"] for row in trajectories] == ["alfworld", "webshop", "webshop", "webshop", "webshop"]
    assert trajectories[1]["skill_id"] == "webshop/webshop-reasoning-planner"
    assert trajectories[1]["next_skill_id"] == "webshop/webshop-search-executor"
    assert trajectories[-1]["done"] is True
    assert trajectories[-1]["source_quality"] == "webshop_hf_gpt4_reward_ge_0.8"
    skill_ids = {row["skill_id"] for row in skill_pool}
    assert "scienceworld/wait" not in skill_ids
    assert {
        "webshop/webshop-reasoning-planner",
        "webshop/webshop-search-executor",
        "webshop/webshop-click-executor",
        "webshop/webshop-finish-executor",
    } <= skill_ids
    assert by_source["webshop_expert_trajectories"]["available"] is True


def test_unified_pretrain_v4_1b_adds_only_non_overlapping_clean_alfworld(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "official::0",
                "trajectory_id": "official-traj",
                "step_index": 0,
                "goal_text": "find mug",
                "state_text": "start",
                "action_text": "look",
                "next_observation_text": "room",
                "skill_id": "alfworld/look",
                "next_skill_id": "alfworld/go",
                "source_quality": "official_replay",
                "provenance": {"source_id": "alfworld_official_train_replay"},
            },
            {
                "benchmark": "scienceworld",
                "task_id": "sci-aux::0",
                "trajectory_id": "sci-aux",
                "step_index": 0,
                "goal_text": "bad science row",
                "state_text": "failed",
                "action_text": "wait",
                "skill_id": "scienceworld/wait",
                "next_skill_id": "scienceworld/wait",
                "source_quality": "aux_only",
                "provenance": {"source_id": "aux_skillnet_rebuilt"},
            },
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "alfworld/look", "name": "look", "description": "Inspect room."},
            {"skill_id": "alfworld/go", "name": "go", "description": "Navigate."},
            {"skill_id": "alfworld/take", "name": "take", "description": "Pick an object."},
            {"skill_id": "scienceworld/wait", "name": "wait", "description": "Wait."},
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_dagger_expert_corrected_train_enriched/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "official::0-conflict",
                "trajectory_id": "official-traj",
                "step_index": 0,
                "goal_text": "find mug",
                "state_text": "start conflict",
                "action_text": "go",
                "next_observation_text": "other room",
                "skill_id": "alfworld/go",
                "next_skill_id": "alfworld/go",
                "source_quality": "dagger_expert_corrected_rollout",
                "provenance": {"source_dataset": "qwen3_expert_corrected_rollout"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "dagger::1",
                "trajectory_id": "dagger-traj",
                "step_index": 1,
                "goal_text": "find mug",
                "state_text": "near mug",
                "action_text": "take mug",
                "next_observation_text": "holding mug",
                "skill_id": "alfworld/take",
                "next_skill_id": "alfworld/take",
                "source_quality": "dagger_expert_corrected_rollout",
                "provenance": {"source_dataset": "qwen3_expert_corrected_rollout"},
            },
        ],
    )
    webshop_rows = [
        {
            "info": {"reward": 1.0},
            "id": 1,
            "question": "buy mug",
            "conversations": [
                {"role": "user", "content": "Task: buy mug"},
                {"role": "assistant", "content": "Action:search[{\"product\": \"mug\"}]\n"},
            ],
        }
    ]
    (tmp_path / "data/webshop_expert_trajectories").mkdir(parents=True)
    (tmp_path / "data/webshop_expert_trajectories/webshop_gpt4_0613.json").write_text(
        json.dumps(webshop_rows),
        encoding="utf-8",
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v4_1b",
        schema_version="v4.1b",
        dataset_recipe="v4_1b",
        webshop_min_reward=0.8,
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v4_1b"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    by_source = {row["source_id"]: row for row in inventory}

    assert manifest["dataset_recipe"] == "v4_1b"
    assert manifest["trajectory_stream"]["quality_filters"]["filtered_rows_by_reason"] == {
        "v4_1_aux_only_full_base_filtered:scienceworld": 1
    }
    assert manifest["trajectory_stream"]["clean_alfworld"]["total_rows"] == 1
    assert manifest["trajectory_stream"]["clean_alfworld"]["skipped_overlap_count"] == 1
    assert [row["source_quality"] for row in trajectories] == [
        "official_replay",
        "dagger_expert_corrected_rollout",
        "webshop_hf_gpt4_reward_ge_0.8",
    ]
    assert [row["task_id"] for row in trajectories] == ["official::0", "dagger::1", "webshop_hf_gpt4::1::0"]
    assert by_source["clstr_dagger_expert_corrected_train_enriched"]["available"] is True


def test_unified_pretrain_v4_2_adds_hf_alfworld_admissible_success_before_weak_rows(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "official::0",
                "trajectory_id": "official-traj",
                "step_index": 0,
                "goal_text": "put a clean mug in cabinet",
                "state_text": "start",
                "action_text": "look",
                "next_observation_text": "room",
                "skill_id": "alfworld/alfworld-environment-scanner",
                "next_skill_id": "alfworld/alfworld-object-locator",
                "source_quality": "official_replay",
                "provenance": {"source_id": "alfworld_official_train_replay"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/agentgym_agenttraj_l/alfworld/converted_train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "weak::0",
                "trajectory_id": "weak-traj",
                "step_index": 0,
                "goal_text": "weak row",
                "state_text": "weak state",
                "action_text": "go to table 1",
                "next_observation_text": "weak next",
                "skill_id": "alfworld/alfworld-receptacle-navigator",
                "next_skill_id": "",
                "source_quality": "weak_policy",
                "provenance": {"source_dataset": "AgentGym/AgentTraj-L", "split": "train"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {
                "skill_id": "alfworld/alfworld-environment-scanner",
                "name": "scanner",
                "description": "Scan the current room.",
            },
            {
                "skill_id": "alfworld/alfworld-object-locator",
                "name": "locator",
                "description": "Find an object.",
            },
            {
                "skill_id": "alfworld/alfworld-object-picker",
                "name": "picker",
                "description": "Pick up an object.",
            },
            {
                "skill_id": "alfworld/alfworld-object-placer",
                "name": "placer",
                "description": "Place an object.",
            },
            {
                "skill_id": "alfworld/alfworld-receptacle-navigator",
                "name": "navigator",
                "description": "Navigate to a receptacle.",
            },
        ],
    )
    hf_dir = tmp_path / "data/hf_alfworld_admissible_success"
    hf_dir.mkdir(parents=True)
    success_messages = [
        {"role": "system", "content": "You are a household task-solving agent."},
        {
            "role": "user",
            "content": (
                "You are in the kitchen.\n\nYour task is to: put a clean mug in cabinet.\n"
                "Admissible actions: [go to table 1, look, inventory]"
            ),
        },
        {"role": "assistant", "content": "Thought: I should find the mug.\nAction: go to table 1"},
        {
            "role": "user",
            "content": (
                "Observation: On the table 1, you see a mug 1.\n"
                "Admissible actions: [take mug 1 from table 1, look, inventory]"
            ),
        },
        {"role": "assistant", "content": "Thought: I found the mug.\nAction: take mug 1 from table 1"},
        {
            "role": "user",
            "content": (
                "Observation: You pick up the mug 1.\n"
                "Admissible actions: [go to cabinet 1, inventory]"
            ),
        },
    ]
    failure_messages = [
        {"role": "user", "content": "Your task is to: fail.\nAdmissible actions: [look]"},
        {"role": "assistant", "content": "Action: look"},
    ]
    pd.DataFrame(
        [
            {
                "messages": success_messages,
                "metadata": {
                    "trajectory_outcome": "success",
                    "has_admissible": True,
                    "dataset_version": "v5",
                    "task_type": "pick_and_place",
                    "description": "put a clean mug in cabinet.",
                },
            },
            {
                "messages": failure_messages,
                "metadata": {
                    "trajectory_outcome": "failure",
                    "has_admissible": True,
                    "dataset_version": "v5",
                    "description": "fail.",
                },
            },
        ]
    ).to_parquet(hf_dir / "train-00000-of-00001.parquet")

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v4_2_alfworld_hf_quality",
        schema_version="v4.2",
        dataset_recipe="v4_2_alfworld_hf_quality",
        trajectory_retrieval_caps={"alfworld": 2},
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v4_2_alfworld_hf_quality"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]

    assert manifest["dataset_recipe"] == "v4_2_alfworld_hf_quality"
    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["total_rows"] == 2
    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["retained_trajectory_count"] == 1
    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["filtered_non_success_count"] == 1
    assert [row["source_quality"] for row in trajectories] == [
        "official_replay",
        "hf_alfworld_admissible_success",
        "hf_alfworld_admissible_success",
        "weak_policy",
    ]
    assert trajectories[1]["skill_id"] == "alfworld/alfworld-receptacle-navigator"
    assert trajectories[1]["next_skill_id"] == "alfworld/alfworld-object-picker"
    assert trajectories[1]["admissible_actions"] == ["go to table 1", "look", "inventory"]
    assert trajectories[1]["loss_mask"]["L_policy"] is True
    assert trajectories[1]["loss_mask"]["L_trans_skill_ce"] is True
    assert trajectories[1]["provenance"]["source_dataset"] == "kuririrn/sft_alfworld_trajectory_dataset_v3to5_admissible_success"
    assert manifest["retrieval_stream"]["trajectory_derived"]["retained_rows_by_benchmark"] == {"alfworld": 2}
    assert manifest["retrieval_stream"]["trajectory_derived"]["skipped_rows_by_benchmark"] == {"alfworld": 2}


def test_unified_pretrain_v4_2_progressive_adds_filtered_weak_policy_cap(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(tmp_path / "data/clstr_full_base_train/train.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "alfworld/alfworld-receptacle-navigator", "name": "navigator"},
            {"skill_id": "alfworld/alfworld-object-picker", "name": "picker"},
            {"skill_id": "alfworld/alfworld-object-placer", "name": "placer"},
        ],
    )
    _write_jsonl(
        tmp_path / "data/agentgym_agenttraj_l/alfworld/converted_train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "weak-policy",
                "trajectory_id": "weak-policy-traj",
                "step_index": 0,
                "state_text": "state policy",
                "action_text": "go to table 1",
                "next_observation_text": "next policy",
                "skill_id": "alfworld/alfworld-receptacle-navigator",
                "next_skill_id": "",
                "source_quality": "weak_policy",
                "admissible_actions": ["go to table 1", "look"],
                "loss_mask": {
                    "L_policy": True,
                    "L_trans": True,
                    "L_trans_skill_ce": False,
                    "belief": True,
                    "STOP": True,
                    "routing": True,
                },
                "provenance": {"source_dataset": "AgentGym/AgentTraj-L", "split": "train"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "weak-belief",
                "trajectory_id": "weak-belief-traj",
                "step_index": 1,
                "state_text": "state belief",
                "action_text": "take mug 1 from table 1",
                "next_observation_text": "next belief",
                "skill_id": "alfworld/alfworld-object-picker",
                "next_skill_id": "",
                "source_quality": "weak_policy",
                "admissible_actions": [],
                "loss_mask": {
                    "L_policy": False,
                    "L_trans": True,
                    "L_trans_skill_ce": False,
                    "belief": True,
                    "STOP": True,
                    "routing": True,
                },
                "provenance": {"source_dataset": "AgentGym/AgentTraj-L", "split": "train"},
            },
            {
                "benchmark": "alfworld",
                "task_id": "weak-bad",
                "trajectory_id": "weak-bad-traj",
                "step_index": 2,
                "state_text": "",
                "action_text": "put mug 1 in cabinet 1",
                "next_observation_text": "next bad",
                "skill_id": "alfworld/alfworld-object-placer",
                "next_skill_id": "alfworld/alfworld-object-placer",
                "source_quality": "weak_policy",
                "loss_mask": {"L_policy": True, "L_trans_skill_ce": True},
                "provenance": {"source_dataset": "AgentGym/AgentTraj-L", "split": "train"},
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v4_2_progressive_final",
        schema_version="v4.2_progressive_final",
        dataset_recipe="v4_2_progressive_final",
        progressive_filtered_weak_policy_cap=1,
        trajectory_retrieval_caps={"alfworld": -1},
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v4_2_progressive_final"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    weak_rows = [row for row in trajectories if row["source_quality"] == "weak_policy_filtered"]

    assert manifest["dataset_recipe"] == "v4_2_progressive_final"
    assert manifest["trajectory_stream"]["filtered_weak_policy"]["cap"] == 1
    assert manifest["trajectory_stream"]["filtered_weak_policy"]["retained_rows"] == 1
    assert manifest["trajectory_stream"]["filtered_weak_policy"]["skipped_rows"] == 2
    assert [row["task_id"] for row in weak_rows] == ["weak-policy"]
    assert weak_rows[0]["loss_mask"]["L_policy"] is True
    assert weak_rows[0]["loss_mask"]["L_trans_skill_ce"] is False
    assert weak_rows[0]["provenance"]["augmentation"] == "filtered_weak_policy_v4_2_progressive_final"


def test_unified_pretrain_v4_2_reads_hf_snapshot_nested_parquet_and_top_level_success(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(tmp_path / "data/clstr_full_base_train/train.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "alfworld/alfworld-receptacle-navigator", "name": "navigator"},
            {"skill_id": "alfworld/alfworld-object-picker", "name": "picker"},
        ],
    )
    nested_dir = tmp_path / "data/hf_alfworld_admissible_success/data"
    nested_dir.mkdir(parents=True)
    messages = [
        {
            "role": "user",
            "content": (
                "Your task is to: find mug.\n"
                "Admissible actions: [go to table 1, look]"
            ),
        },
        {"role": "assistant", "content": "Act: go to table 1"},
        {
            "role": "user",
            "content": (
                "Observation: On the table 1, you see a mug 1.\n"
                "Admissible actions: [take mug 1 from table 1]"
            ),
        },
        {"role": "assistant", "content": "Act: take mug 1 from table 1"},
    ]
    pd.DataFrame(
        [
            {
                "messages": messages,
                "metadata": {"has_admissible": True, "description": "find mug."},
                "trajectory_outcome": "success",
                "dataset_version": "v5",
            },
            {
                "messages": messages,
                "metadata": {"has_admissible": True, "description": "find mug."},
                "trajectory_outcome": "failure",
                "dataset_version": "v5",
            },
        ]
    ).to_parquet(nested_dir / "train-00000-of-00001.parquet")

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v4_2_alfworld_hf_quality",
        schema_version="v4.2",
        dataset_recipe="v4_2_alfworld_hf_quality",
        trajectory_retrieval_caps={"alfworld": -1},
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v4_2_alfworld_hf_quality"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    inventory_by_id = {row["source_id"]: row for row in inventory}

    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["paths"] == [
        str(nested_dir / "train-00000-of-00001.parquet")
    ]
    assert inventory_by_id["hf_alfworld_admissible_success"]["available"] is True
    assert inventory_by_id["hf_alfworld_admissible_success"]["reason_if_missing"] is None
    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["retained_trajectory_count"] == 1
    assert manifest["trajectory_stream"]["hf_alfworld_admissible_success"]["filtered_non_success_count"] == 1
    assert len(trajectories) == 2
    assert {row["provenance"]["trajectory_outcome"] for row in trajectories} == {"success"}


def test_unified_pretrain_can_filter_trajectory_source_quality_before_retrieval_caps(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "official::0",
                "trajectory_id": "official",
                "step_index": 0,
                "state_text": "official",
                "action_text": "look",
                "skill_id": "alfworld/alfworld-environment-scanner",
                "next_skill_id": "",
                "source_quality": "official_replay",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/agentgym_agenttraj_l/alfworld/converted_train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": "weak::0",
                "trajectory_id": "weak",
                "step_index": 0,
                "state_text": "weak",
                "action_text": "go to table 1",
                "skill_id": "alfworld/alfworld-receptacle-navigator",
                "next_skill_id": "",
                "source_quality": "weak_policy",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": "alfworld/alfworld-environment-scanner", "name": "scanner"},
            {"skill_id": "alfworld/alfworld-receptacle-navigator", "name": "navigator"},
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_quality_filtered",
        schema_version="quality-filtered",
        trajectory_source_quality_allowlist="official_replay",
        trajectory_retrieval_caps={"alfworld": -1},
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_quality_filtered"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]

    assert [row["source_quality"] for row in trajectories] == ["official_replay"]
    assert manifest["trajectory_stream"]["source_quality_filter"] == {
        "enabled": True,
        "allowlist": ["official_replay"],
        "source_rows": 2,
        "retained_rows": 1,
        "skipped_rows": 1,
        "skipped_by_source_quality": {"weak_policy": 1},
    }
    assert [row["positive_skill_id"] for row in retrieval] == ["alfworld/alfworld-environment-scanner"]


def test_unified_pretrain_v2_caps_auxiliary_trajectory_derived_retrieval(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/train.jsonl",
        [
            {
                "benchmark": "alfworld",
                "task_id": f"alf-task-{idx}",
                "trajectory_id": f"alf-traj-{idx}",
                "step_index": idx,
                "goal_text": "clean the room",
                "history_text": "look",
                "state_text": f"observation {idx}",
                "action_text": f"take object {idx}",
                "next_observation_text": f"after {idx}",
                "skill_id": f"alf/skill/{idx}",
                "next_skill_id": f"alf/next/{idx}",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            }
            for idx in range(3)
        ]
        + [
            {
                "benchmark": "scienceworld",
                "task_id": f"sci-task-{idx}",
                "trajectory_id": f"sci-traj-{idx}",
                "step_index": idx,
                "goal_text": "finish experiment",
                "state_text": f"lab observation {idx}",
                "action_text": f"mix item {idx}",
                "next_observation_text": f"reaction {idx}",
                "skill_id": f"sci/skill/{idx}",
                "next_skill_id": f"sci/next/{idx}",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            }
            for idx in range(2)
        ],
    )
    _write_jsonl(
        tmp_path / "data/clstr_full_base_train/skills.jsonl",
        [
            {"skill_id": f"alf/skill/{idx}", "name": f"alf skill {idx}"}
            for idx in range(3)
        ]
        + [
            {"skill_id": f"alf/next/{idx}", "name": f"alf next {idx}"}
            for idx in range(3)
        ]
        + [
            {"skill_id": f"sci/skill/{idx}", "name": f"sci skill {idx}"}
            for idx in range(2)
        ]
        + [
            {"skill_id": f"sci/next/{idx}", "name": f"sci next {idx}"}
            for idx in range(2)
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
        trajectory_retrieval_caps={"alfworld": 2, "scienceworld": 1},
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]

    assert manifest["retrieval_stream"]["trajectory_derived"]["row_caps"] == {
        "alfworld": 2,
        "scienceworld": 1,
    }
    assert manifest["retrieval_stream"]["trajectory_derived"]["retained_rows_by_benchmark"] == {
        "alfworld": 2,
        "scienceworld": 1,
    }
    assert manifest["retrieval_stream"]["trajectory_derived"]["skipped_rows_by_benchmark"] == {
        "alfworld": 1,
        "scienceworld": 1,
    }
    assert manifest["retrieval_stream"]["counts"]["trajectory_derived_alfworld_current_pair"] == 2
    assert manifest["retrieval_stream"]["counts"]["trajectory_derived_alfworld_next_pair"] == 2
    assert manifest["retrieval_stream"]["counts"]["trajectory_derived_scienceworld_current_pair"] == 1
    assert manifest["retrieval_stream"]["counts"]["trajectory_derived_scienceworld_next_pair"] == 1
    assert [row["source"] for row in retrieval] == [
        "trajectory_derived_alfworld",
        "trajectory_derived_alfworld",
        "trajectory_derived_alfworld",
        "trajectory_derived_alfworld",
        "trajectory_derived_scienceworld",
        "trajectory_derived_scienceworld",
    ]
    assert retrieval[0]["query_id"] == "trajectory-derived::alfworld::alf-traj-0::0::current"
    assert "goal: clean the room" in retrieval[0]["query_text"]
    assert "history: look" not in retrieval[0]["query_text"]
    assert "observation: observation 0" in retrieval[0]["query_text"]
    assert retrieval[1]["query_id"] == "trajectory-derived::alfworld::alf-traj-0::0::next"
    assert "previous_action: take object 0" in retrieval[1]["query_text"]
    assert "next_observation: after 0" in retrieval[1]["query_text"]


def test_unified_pretrain_v2_sbatch_entrypoint_uses_repo_local_paths():
    script = Path("scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh").read_text(encoding="utf-8")

    assert "scripts/build_clstr_unified_pretrain.py" in script
    assert "OUTPUT_DIR" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final" in script
    assert "--schema_version" in script
    assert "TRAJECTORY_RETRIEVAL_CAPS" in script
    assert "v3_trajectory_retrieval_caps" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_unified_pretrain_v2_cli_help_runs_from_repo_root():
    proc = subprocess.run(
        [sys.executable, "scripts/build_clstr_unified_pretrain.py", "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--output_dir" in proc.stdout
    assert "--schema_version" in proc.stdout
    assert "--trajectory_retrieval_caps" in proc.stdout


def test_unified_pretrain_v2_converts_traject_bench_steps_and_tools(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    public_data = tmp_path / "data/traject_bench/public_data"
    tools_dir = public_data / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "all_tools.json").write_text(
        json.dumps(
            [
                {
                    "tool name": "Weather API: Forecast",
                    "tool description": "Gets a forecast.",
                    "parent tool name": "Weather API",
                    "API name": "Forecast",
                    "domain name": "Weather",
                    "required_parameters": [{"name": "city", "type": "STRING"}],
                    "optional_parameters": [],
                },
                {
                    "tool name": "Map API: Route",
                    "tool description": "Plans a route.",
                    "parent tool name": "Map API",
                    "API name": "Route",
                    "domain name": "Mapping",
                    "required_parameters": [{"name": "origin", "type": "STRING"}],
                    "optional_parameters": [{"name": "avoid", "type": "STRING"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    query_dir = public_data / "sequential/Travel"
    query_dir.mkdir(parents=True)
    (query_dir / "traj_query.json").write_text(
        json.dumps(
            [
                {
                    "query": "Check weather, then plan a route.",
                    "trajectory_type": "sequential",
                    "task_name": "weather-aware route",
                    "task_description": "Use weather before route planning.",
                    "tool list": [
                        {
                            "tool name": "Weather API: Forecast",
                            "tool description": "Gets a forecast.",
                            "required parameters": [{"name": "city", "value": "Paris"}],
                            "optional parameters": [],
                            "executed_output": "rain",
                            "parent tool name": "Weather API",
                            "API name": "Forecast",
                            "domain name": "Weather",
                        },
                        {
                            "tool name": "Map API: Route",
                            "tool description": "Plans a route.",
                            "required parameters": [{"name": "origin", "value": "hotel"}],
                            "optional parameters": [{"name": "avoid", "value": "rain"}],
                            "executed_output": "route",
                            "parent tool name": "Map API",
                            "API name": "Route",
                            "domain name": "Mapping",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]

    assert manifest["trajectory_stream"]["counts"]["benchmark:traject_bench"] == 2
    assert manifest["trajectory_stream"]["counts"]["traject_type:sequential"] == 2
    assert manifest["retrieval_stream"]["counts"]["traject_bench_pair"] == 2
    assert [row["step_index"] for row in trajectories] == [0, 1]
    assert trajectories[0]["skill_id"] == "traject/weather-api/forecast"
    assert trajectories[0]["next_skill_id"] == "traject/map-api/route"
    assert trajectories[0]["tool_inventory_skill_ids"] == [
        "traject/weather-api/forecast",
        "traject/map-api/route",
    ]
    assert trajectories[1]["tool_inventory_skill_ids"] == [
        "traject/weather-api/forecast",
        "traject/map-api/route",
    ]
    assert trajectories[0]["visible_inventory_skill_ids"] == [
        "traject/map-api/route",
        "traject/weather-api/forecast",
    ]
    assert trajectories[1]["visible_inventory_skill_ids"] == [
        "traject/map-api/route",
        "traject/weather-api/forecast",
    ]
    assert trajectories[1]["done"] is True
    assert "previous_tools:" not in trajectories[1]["state_text"]
    assert trajectories[1]["state_text_current"] == trajectories[1]["state_text"]
    assert "previous_tools: Weather API: Forecast" in trajectories[1]["state_text_full"]
    assert [row["source"] for row in retrieval[:2]] == ["traject_bench", "traject_bench"]
    assert [row["source"] for row in retrieval[2:]] == [
        "trajectory_derived_traject_bench",
        "trajectory_derived_traject_bench",
        "trajectory_derived_traject_bench",
    ]
    assert retrieval[0]["query_id"] == "traject::sequential::Travel::traj_query::0::0"
    assert retrieval[0]["positive_skill_id"] == "traject/weather-api/forecast"
    assert retrieval[0]["negative_skill_ids"] == ["traject/map-api/route"]
    assert "previous_tools:" not in retrieval[0]["query_text"]
    assert retrieval[1]["positive_skill_id"] == "traject/map-api/route"
    assert retrieval[1]["negative_skill_ids"] == ["traject/weather-api/forecast"]
    assert "previous_tools:" not in retrieval[1]["query_text"]
    skill_ids = {row["skill_id"] for row in skill_pool}
    assert {"traject/weather-api/forecast", "traject/map-api/route"} <= skill_ids
    skills_by_id = {row["skill_id"]: row for row in skill_pool}
    assert skills_by_id["traject/weather-api/forecast"]["description"] == "Gets a forecast."
    assert skills_by_id["traject/weather-api/forecast"]["provenance"]["dedup_method"] == "traject_tool_identity_v2"
    assert any(row["source_id"] == "traject_bench" and row["available"] for row in inventory)


def test_unified_pretrain_v2_only_uses_train_partition_of_traject_public_data(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    public_data = tmp_path / "data/traject_bench/public_data"
    tools_dir = public_data / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "all_tools.json").write_text(
        json.dumps(
            [
                {
                    "tool name": "Weather API: Forecast",
                    "tool description": "Gets a forecast.",
                    "parent tool name": "Weather API",
                    "API name": "Forecast",
                }
            ]
        ),
        encoding="utf-8",
    )
    query_dir = public_data / "sequential/Travel"
    query_dir.mkdir(parents=True)
    rows = [
        {
            "query": f"Check weather for trip {idx}.",
            "tool list": [
                {
                    "tool name": "Weather API: Forecast",
                    "tool description": "Gets a forecast.",
                    "parent tool name": "Weather API",
                    "API name": "Forecast",
                }
            ],
        }
        for idx in range(80)
    ]
    (query_dir / "traj_query.json").write_text(json.dumps(rows), encoding="utf-8")
    expected_train_task_ids = {
        f"traject::sequential::Travel::traj_query::{idx}::0"
        for idx in range(len(rows))
        if assign_traject_split(f"traject::sequential::Travel::traj_query::{idx}") == "train"
    }
    expected_heldout_count = len(rows) - len(expected_train_task_ids)

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]

    assert expected_train_task_ids
    assert expected_heldout_count > 0
    assert {row["task_id"] for row in trajectories} == expected_train_task_ids
    assert {json.loads(row["provenance"])["split"] for row in retrieval} == {"train"}
    assert {row["provenance"]["split"] for row in trajectories} == {"train"}
    assert {row["provenance"]["split_policy"] for row in trajectories} == {
        "deterministic_hash_trajectory_level_v1"
    }
    assert (
        manifest["trajectory_stream"]["counts"]["traject_split_skipped:dev"]
        + manifest["trajectory_stream"]["counts"]["traject_split_skipped:test"]
        == expected_heldout_count
    )
    assert (
        manifest["retrieval_stream"]["counts"]["traject_bench_pair_skipped:dev"]
        + manifest["retrieval_stream"]["counts"]["traject_bench_pair_skipped:test"]
        == expected_heldout_count
    )
    assert manifest["retrieval_stream"]["total_pairs"] == len(retrieval)


def test_unified_pretrain_v2_keeps_traject_query_only_unknown_tools_distinct(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    public_data = tmp_path / "data/traject_bench/public_data"
    tools_dir = public_data / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "all_tools.json").write_text("[]", encoding="utf-8")
    query_dir = public_data / "sequential/Education"
    query_dir.mkdir(parents=True)
    (query_dir / "traj_query.json").write_text(
        json.dumps(
            [
                {
                    "query": "Use two education helper tools.",
                    "tool list": [
                        {
                            "tool name": "data_visualisation_: getting data",
                            "parent tool name": "unknown",
                            "API name": "unknown",
                            "tool description": "Gets source data for a visualization.",
                        },
                        {
                            "tool name": "Question generator: Blank fields",
                            "parent tool name": "unknown",
                            "API name": "unknown",
                            "tool description": "Generates fill-in-the-blank questions.",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    builder = _load_builder()
    builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]

    expected_ids = [
        "traject/data-visualisation-getting-data",
        "traject/question-generator-blank-fields",
    ]
    assert [row["skill_id"] for row in trajectories] == expected_ids
    assert [row["positive_skill_id"] for row in retrieval[:2]] == expected_ids
    assert [row["source"] for row in retrieval[2:]] == [
        "trajectory_derived_traject_bench",
        "trajectory_derived_traject_bench",
        "trajectory_derived_traject_bench",
    ]
    skills_by_id = {row["skill_id"]: row for row in skill_pool}
    assert set(expected_ids) <= set(skills_by_id)
    assert skills_by_id[expected_ids[0]]["description"] == "Gets source data for a visualization."
    assert skills_by_id[expected_ids[1]]["description"] == "Generates fill-in-the-blank questions."
    assert skills_by_id[expected_ids[0]]["provenance"]["dedup_method"] == "traject_query_tool_identity_v2"
    assert skills_by_id[expected_ids[1]]["provenance"]["dedup_method"] == "traject_query_tool_identity_v2"
    assert len(skills_by_id) == 2


def test_unified_pretrain_v2_does_not_mark_code_only_toolbench_toolret_as_available(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    (tmp_path / "ToolBench").mkdir()
    (tmp_path / "ToolBench/README.md").write_text("ToolBench code only", encoding="utf-8")
    (tmp_path / "tool-retrieval-benchmark").mkdir()
    (tmp_path / "tool-retrieval-benchmark/README.md").write_text("ToolRet code only", encoding="utf-8")

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path / "clstr",
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    inventory_path = tmp_path / "clstr/data/clstr_unified_pretrain_v2/source_inventory.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert by_id["toolbench_g3"]["available"] is False
    assert by_id["toolret_training"]["available"] is False
    assert "toolbench_g3" in manifest["source_inventory"]["missing_required_target_sources"]
    assert "toolret_training" in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_does_not_mark_toolbench_static_as_official_g3(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(tmp_path / "ToolBench/data/toolbench_static/in_domain.json", [{"tools": [], "query": "q"}])
    _write_jsonl(tmp_path / "ToolBench/data/toolbench_static/out_of_domain.json", [{"tools": [], "query": "q"}])

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path / "clstr",
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    inventory_path = tmp_path / "clstr/data/clstr_unified_pretrain_v2/source_inventory.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert by_id["toolbench_g3"]["available"] is False
    assert Path(by_id["toolbench_g3"]["first_existing_path"]).resolve() == (tmp_path / "ToolBench").resolve()
    assert by_id["toolbench_g3"]["reason_if_missing"] == "source_exists_but_required_training_files_missing"
    assert "toolbench_g3" in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_does_not_mark_raw_toolbench_g3_as_training_ready_before_import(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    (tmp_path / "ToolBench/data/instruction").mkdir(parents=True)
    (tmp_path / "ToolBench/data/answer/G3_answer").mkdir(parents=True)
    (tmp_path / "ToolBench/data/instruction/G3_query.json").write_text(
        json.dumps([{"query_id": 1, "query": "q"}]),
        encoding="utf-8",
    )
    (tmp_path / "ToolBench/data/answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json").write_text(
        json.dumps({"win": True}),
        encoding="utf-8",
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path / "clstr",
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    inventory_path = tmp_path / "clstr/data/clstr_unified_pretrain_v2/source_inventory.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert by_id["toolbench_g3"]["available"] is False
    assert Path(by_id["toolbench_g3"]["first_existing_path"]).resolve() == (tmp_path / "ToolBench").resolve()
    assert by_id["toolbench_g3"]["reason_if_missing"] == "source_exists_but_required_training_files_missing"
    assert "toolbench_g3" in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_includes_normalized_toolret_training_pairs_and_skills(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolret_training/retrieval.jsonl",
        [
            {
                "query_id": "toolret-train-0",
                "query_text": "Is this URL available in the Wayback Machine?",
                "positive_skill_id": "toolret/availability",
                "negative_skill_ids": ["toolret/top-grossing-mac-apps"],
                "provenance": {"source_dataset": "mangopy/ToolRet-Training-20w", "split": "train"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolret_training/skills.jsonl",
        [
            {
                "skill_id": "toolret/availability",
                "name": "availability",
                "description": "Checks if a URL is archived and currently accessible.",
                "input_schema": {"url": {"type": "str"}},
            },
            {
                "skill_id": "toolret/top-grossing-mac-apps",
                "name": "top_grossing_mac_apps",
                "description": "Fetches top-grossing Mac apps.",
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert manifest["retrieval_stream"]["counts"]["toolret_training_pair"] == 1
    assert retrieval == [
        {
            "source": "toolret_training",
            "query_id": "toolret-train-0",
            "query_text": "Is this URL available in the Wayback Machine?",
            "positive_skill_id": "toolret/availability",
            "negative_skill_ids": ["toolret/top-grossing-mac-apps"],
            "provenance": json.dumps(
                {"source_dataset": "mangopy/ToolRet-Training-20w", "split": "train"},
                ensure_ascii=False,
            ),
        }
    ]
    skill_ids = {row["skill_id"] for row in skill_pool}
    assert {"toolret/availability", "toolret/top-grossing-mac-apps"} <= skill_ids
    assert by_id["toolret_training"]["available"] is True
    assert "toolret_training" not in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_includes_normalized_toolbench_g3_trajectories_retrieval_and_skills(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolbench_g3/trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-7::0",
                "trajectory_id": "toolbench-g3-7",
                "step_index": 0,
                "goal_text": "Find cocktails and then search birthday news.",
                "state_text": "goal: Find cocktails\nprevious_tools: <empty>",
                "action_text": "list_of_cocktails_for_the_cocktail_db: {}",
                "expert_action": "list_of_cocktails_for_the_cocktail_db: {}",
                "next_observation_text": "{\"response\": \"martini\"}",
                "skill_id": "toolbench-g3/the-cocktail-db/list-of-cocktails",
                "next_skill_id": "toolbench-g3/web-search/newssearch",
                "done": False,
                "loss_mask": {"L_policy": True, "L_trans": True, "routing": True},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-7",
                "query_text": "Find cocktails and then search birthday news.",
                "positive_skill_id": "toolbench-g3/the-cocktail-db/list-of-cocktails",
                "negative_skill_ids": ["toolbench-g3/web-search/newssearch"],
                "provenance": {"source_dataset": "ToolBench-G3", "split": "train"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/skills.jsonl",
        [
            {
                "skill_id": "toolbench-g3/the-cocktail-db/list-of-cocktails",
                "name": "List of Cocktails",
                "description": "Returns a list of cocktails.",
            },
            {
                "skill_id": "toolbench-g3/web-search/newssearch",
                "name": "newsSearch",
                "description": "Gets news articles for a query.",
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out_dir / "source_inventory.jsonl").read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert manifest["trajectory_stream"]["counts"]["benchmark:toolbench_g3"] == 1
    assert manifest["retrieval_stream"]["counts"]["toolbench_g3_pair"] == 1
    assert trajectories[0]["skill_id"] == "toolbench-g3/the-cocktail-db/list-of-cocktails"
    assert retrieval[0]["source"] == "toolbench_g3"
    skill_ids = {row["skill_id"] for row in skill_pool}
    assert {"toolbench-g3/the-cocktail-db/list-of-cocktails", "toolbench-g3/web-search/newssearch"} <= skill_ids
    assert by_id["toolbench_g3"]["available"] is True
    assert "toolbench_g3" not in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_fills_blank_toolbench_g3_skill_description_from_api_schema(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolbench_g3/trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-search::0",
                "trajectory_id": "toolbench-g3-search",
                "step_index": 0,
                "goal_text": "Search product listings.",
                "state_text": "goal: Search product listings.",
                "action_text": "SearchProducts: {\"query\": \"watch\"}",
                "expert_action": "SearchProducts: {\"query\": \"watch\"}",
                "next_observation_text": "{\"items\": []}",
                "skill_id": "toolbench-g3/example-api/searchproducts",
                "next_skill_id": "",
                "done": True,
                "loss_mask": {"L_policy": True, "L_trans": False, "routing": True},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-search",
                "query_text": "Search product listings.",
                "positive_skill_id": "toolbench-g3/example-api/searchproducts",
                "negative_skill_ids": [],
                "provenance": {"source_dataset": "ToolBench-G3", "split": "train"},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/skills.jsonl",
        [
            {
                "skill_id": "toolbench-g3/example-api/searchproducts",
                "name": "SearchProducts",
                "description": " ",
                "input_schema": {
                    "required_parameters": [{"name": "query", "type": "STRING", "description": ""}],
                    "optional_parameters": [{"name": "page", "type": "NUMBER", "description": ""}],
                    "method": "GET",
                },
            }
        ],
    )

    builder = _load_builder()
    builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    skill = next(row for row in skill_pool if row["skill_id"] == "toolbench-g3/example-api/searchproducts")

    assert skill["description"].strip()
    assert "SearchProducts" in skill["description"]
    assert "query" in skill["description"]
    assert "page" in skill["description"]


def test_unified_pretrain_v2_requires_complete_normalized_toolbench_g3_files(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolbench_g3/retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-g3-7",
                "query_text": "partial normalized source",
                "positive_skill_id": "toolbench-g3/a/b",
                "negative_skill_ids": [],
            }
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    inventory_path = tmp_path / "data/clstr_unified_pretrain_v2/source_inventory.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_text().splitlines()]
    by_id = {row["source_id"]: row for row in inventory}

    assert by_id["toolbench_g3"]["available"] is False
    assert by_id["toolbench_g3"]["reason_if_missing"] == "source_exists_but_required_training_files_missing"
    assert "toolbench_g3" in manifest["source_inventory"]["missing_required_target_sources"]


def test_unified_pretrain_v2_partial_dedup_remaps_duplicate_tool_skills(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolret_training/retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": "toolret-q",
                "query_text": "Need weather by city.",
                "positive_skill_id": "toolret/get-weather",
                "negative_skill_ids": ["toolret/news-search"],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolret_training/skills.jsonl",
        [
            {
                "skill_id": "toolret/get-weather",
                "name": "Get Weather",
                "description": "Returns current weather for a city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/news-search",
                "name": "News Search",
                "description": "Searches news.",
                "input_schema": {"query": {"type": "str"}},
            },
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": "toolbench-q",
                "query_text": "Find current weather.",
                "positive_skill_id": "toolbench-g3/weather/get-weather",
                "negative_skill_ids": ["toolret/news-search"],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-q::0",
                "trajectory_id": "toolbench-q",
                "step_index": 0,
                "goal_text": "Find current weather.",
                "state_text": "goal: Find current weather.",
                "action_text": "get_weather(city=Paris)",
                "skill_id": "toolbench-g3/weather/get-weather",
                "done": True,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolbench_g3/skills.jsonl",
        [
            {
                "skill_id": "toolbench-g3/weather/get-weather",
                "name": "Get Weather",
                "description": "Fetches live weather by city name.",
                "input_schema": {"city": {"type": "str"}},
            }
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    aliases = [json.loads(line) for line in (out_dir / "skill_aliases.jsonl").read_text().splitlines()]

    canonical_weather = "toolbench-g3/weather/get-weather"
    assert manifest["files"]["skill_aliases"] == "skill_aliases.jsonl"
    assert manifest["skill_pool"]["dedup_method"] == "partial_rule_name_param_v2"
    assert manifest["skill_pool"]["raw_skill_count"] == 3
    assert manifest["skill_pool"]["canonical_skill_count"] == 2
    assert manifest["skill_pool"]["merged_group_count"] == 1
    assert manifest["skill_pool"]["stream_remap"]["retrieval_rewritten_rows"] == 3
    assert manifest["skill_pool"]["stream_remap"]["trajectory_rewritten_rows"] == 1
    assert {row["skill_id"] for row in skill_pool} == {canonical_weather, "toolret/news-search"}
    weather_record = next(row for row in skill_pool if row["skill_id"] == canonical_weather)
    assert weather_record["alias_skill_ids"] == ["toolbench-g3/weather/get-weather", "toolret/get-weather"]
    assert "Returns current weather for a city." in weather_record["alternate_descriptions"]
    assert retrieval[0]["positive_skill_id"] == canonical_weather
    assert retrieval[1]["positive_skill_id"] == canonical_weather
    assert trajectories[0]["skill_id"] == canonical_weather
    assert {row["raw_skill_id"]: row["canonical_skill_id"] for row in aliases} == {
        "toolbench-g3/weather/get-weather": canonical_weather,
        "toolret/get-weather": canonical_weather,
        "toolret/news-search": "toolret/news-search",
    }


def test_unified_pretrain_v2_partial_dedup_keeps_same_name_with_different_params_separate(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolret_training/retrieval.jsonl",
        [
            {
                "query_id": "q-city",
                "query_text": "Weather by city.",
                "positive_skill_id": "toolret/get-weather-city",
                "negative_skill_ids": ["toolret/get-weather-zip"],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolret_training/skills.jsonl",
        [
            {
                "skill_id": "toolret/get-weather-city",
                "name": "Get Weather",
                "description": "Weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/get-weather-zip",
                "name": "Get Weather",
                "description": "Weather by ZIP code.",
                "input_schema": {"zip": {"type": "str"}},
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    aliases = [json.loads(line) for line in (out_dir / "skill_aliases.jsonl").read_text().splitlines()]

    assert manifest["skill_pool"]["raw_skill_count"] == 2
    assert manifest["skill_pool"]["canonical_skill_count"] == 2
    assert manifest["skill_pool"]["merged_group_count"] == 0
    assert {row["skill_id"] for row in skill_pool} == {"toolret/get-weather-city", "toolret/get-weather-zip"}
    assert {row["raw_skill_id"]: row["canonical_skill_id"] for row in aliases} == {
        "toolret/get-weather-city": "toolret/get-weather-city",
        "toolret/get-weather-zip": "toolret/get-weather-zip",
    }


def test_unified_pretrain_v2_writes_borderline_dedup_review_candidates_without_auto_merging(tmp_path):
    _write_jsonl(tmp_path / "data/appworld_routing/dev_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_normal_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/appworld_routing/test_challenge_tasks.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/qrels.jsonl", [])
    _write_jsonl(tmp_path / "data/skillret/queries.jsonl", [])
    _write_jsonl(
        tmp_path / "data/toolret_training/retrieval.jsonl",
        [
            {
                "query_id": "q-city",
                "query_text": "Weather by city.",
                "positive_skill_id": "toolret/get-weather-city",
                "negative_skill_ids": ["toolret/get-current-weather-city"],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "data/toolret_training/skills.jsonl",
        [
            {
                "skill_id": "toolret/get-weather-city",
                "name": "Get Weather",
                "description": "Weather by city.",
                "input_schema": {"city": {"type": "str"}, "units": {"type": "str"}},
            },
            {
                "skill_id": "toolret/get-current-weather-city",
                "name": "Get Current Weather",
                "description": "Current weather by city name.",
                "input_schema": {"city": {"type": "str"}},
            },
        ],
    )

    builder = _load_builder()
    manifest = builder.build_unified_pretrain(
        repo_root=tmp_path,
        output_dir="data/clstr_unified_pretrain_v2",
        schema_version="v2",
    )
    out_dir = tmp_path / "data/clstr_unified_pretrain_v2"
    skill_pool = [json.loads(line) for line in (out_dir / "skill_pool.jsonl").read_text().splitlines()]
    candidates = [
        json.loads(line)
        for line in (out_dir / "skill_dedup_borderline_candidates.jsonl").read_text().splitlines()
    ]

    assert manifest["files"]["skill_dedup_borderline_candidates"] == "skill_dedup_borderline_candidates.jsonl"
    assert manifest["skill_pool"]["borderline_review"]["status"] == "pending_manual_or_llm_review"
    assert manifest["skill_pool"]["borderline_review"]["candidate_count"] == 1
    assert {row["skill_id"] for row in skill_pool} == {
        "toolret/get-weather-city",
        "toolret/get-current-weather-city",
    }
    assert candidates == [
        {
            "left_skill_id": "toolret/get-current-weather-city",
            "right_skill_id": "toolret/get-weather-city",
            "left_name": "Get Current Weather",
            "right_name": "Get Weather",
            "name_similarity": candidates[0]["name_similarity"],
            "param_overlap": candidates[0]["param_overlap"],
            "reasons": ["fuzzy_name_similarity", "param_signature_partial_overlap"],
            "review_status": "pending_manual_or_llm",
        }
    ]
    assert 0.6 <= candidates[0]["name_similarity"] < 0.95
    assert candidates[0]["param_overlap"] == 0.5


def test_skill_dedup_borderline_audit_cli_generates_candidates_for_existing_skill_pool(tmp_path):
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    output_path = tmp_path / "skill_dedup_borderline_candidates.jsonl"
    report_path = tmp_path / "skill_dedup_borderline_report.json"
    review_output_path = tmp_path / "skill_dedup_borderline_candidates.reviewed.jsonl"
    review_report_path = tmp_path / "skill_dedup_borderline_review_report.json"
    _write_jsonl(
        skill_pool_path,
        [
            {
                "skill_id": "toolret/get-weather-city",
                "name": "Get Weather",
                "description": "Weather by city.",
                "input_schema": {"city": {"type": "str"}, "units": {"type": "str"}},
            },
            {
                "skill_id": "toolret/get-current-weather-city",
                "name": "Get Current Weather",
                "description": "Current weather by city name.",
                "input_schema": {"city": {"type": "str"}},
            },
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_skill_dedup_borderline.py",
            "--skill_pool_path",
            str(skill_pool_path),
            "--output_path",
            str(output_path),
            "--report_path",
            str(report_path),
            "--review_output_path",
            str(review_output_path),
            "--review_report_path",
            str(review_report_path),
            "--max_candidates",
            "10",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_report = json.loads(review_report_path.read_text(encoding="utf-8"))
    candidates = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    reviewed = [json.loads(line) for line in review_output_path.read_text(encoding="utf-8").splitlines()]
    assert report["status"] == "pending_manual_or_llm_review"
    assert report["skill_count"] == 2
    assert report["candidate_count"] == 1
    assert report["conservative_review"]["status"] == "complete"
    assert review_report["status"] == "complete"
    assert review_report["reviewed_candidate_count"] == 1
    assert candidates[0]["left_skill_id"] == "toolret/get-current-weather-city"
    assert candidates[0]["right_skill_id"] == "toolret/get-weather-city"
    assert candidates[0]["review_status"] == "pending_manual_or_llm"
    assert reviewed[0]["review_status"] == "reviewed_keep_separate"
