from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clstr.logged_online_trajectory import (
    audit_logged_online_coverage,
    iter_logged_online_steps,
    load_stage0_handoff_summary,
    load_skill_id_set,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_iter_logged_online_steps_normalizes_unified_step_rows(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "task_id": "traj-1::0",
                "step_index": 0,
                "goal_text": "Find beach cocktails.",
                "state_text": "goal: Find beach cocktails.\nprevious_tools: <empty>",
                "history_text": "",
                "action_text": "cocktails.list: {}",
                "next_observation_text": "returned cocktails",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next", "skill/other"],
                "provenance": {"split": "train_or_released_g3"},
                "reward": 1.0,
            }
        ],
    )

    steps = list(iter_logged_online_steps(trajectories, {"skill/current", "skill/next", "skill/other"}))

    assert len(steps) == 1
    assert steps[0]["trajectory_id"] == "traj-1"
    assert steps[0]["benchmark"] == "toolbench_g3"
    assert steps[0]["split"] == "train_or_released_g3"
    assert steps[0]["step_idx"] == 0
    assert steps[0]["goal"] == "Find beach cocktails."
    assert steps[0]["history_text"] == ""
    assert steps[0]["observation_text"].startswith("goal: Find beach cocktails")
    assert steps[0]["action_text"] == "cocktails.list: {}"
    assert steps[0]["next_observation_text"] == "returned cocktails"
    assert steps[0]["gt_skill_ids"] == ["skill/current"]
    assert steps[0]["gt_next_skill_ids"] == ["skill/next"]
    assert steps[0]["candidate_skill_ids"] == ["skill/current", "skill/next", "skill/other"]
    assert steps[0]["reward_type"] == "logged_gt"
    assert steps[0]["reward"] == 1.0


def test_audit_logged_online_coverage_reports_mapping_and_candidate_coverage(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next"],
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 1,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/missing",
                "candidate_next_skill_ids": ["skill/current"],
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "alfworld",
                "trajectory_id": "traj-2",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current"],
                "provenance": {"split": "dev"},
            },
        ],
    )
    skill_pool = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool,
        [
            {"skill_id": "skill/current", "name": "current"},
            {"skill_id": "skill/next", "name": "next"},
        ],
    )

    report = audit_logged_online_coverage(
        trajectories_path=trajectories,
        skill_pool_path=skill_pool,
        include_splits={"train"},
    )

    assert report["source_rows"] == 3
    assert report["emitted_steps"] == 2
    assert report["skipped_reasons"] == {"split_not_included": 1}
    assert report["overall"]["current_skill_mapped"] == 2
    assert report["overall"]["next_skill_mapped"] == 1
    assert report["overall"]["candidate_rows"] == 2
    assert report["overall"]["current_gt_in_candidates"] == 2
    assert report["overall"]["next_gt_in_candidates"] == 1
    assert report["overall"]["label_updateable_steps"] == 1
    assert report["overall"]["stage4_updateable_steps"] == 1
    assert report["by_benchmark"]["toolbench_g3"]["steps"] == 2
    assert report["by_benchmark"]["toolbench_g3"]["label_updateable_steps"] == 1
    assert report["by_benchmark"]["toolbench_g3"]["stage4_updateable_steps"] == 1


def test_audit_logged_online_coverage_respects_max_source_rows_without_overcounting(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next"],
                "provenance": {"split": "train"},
            },
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 1,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next"],
                "provenance": {"split": "train"},
            },
        ],
    )
    skill_pool = tmp_path / "skill_pool.jsonl"
    _write_jsonl(skill_pool, [{"skill_id": "skill/current"}, {"skill_id": "skill/next"}])

    report = audit_logged_online_coverage(
        trajectories_path=trajectories,
        skill_pool_path=skill_pool,
        include_splits={"train"},
        max_source_rows=1,
    )

    assert report["source_rows"] == 1
    assert report["emitted_steps"] == 1


def test_load_skill_id_set_accepts_skill_id_and_id_fields(tmp_path):
    skill_pool = tmp_path / "skills.jsonl"
    _write_jsonl(skill_pool, [{"skill_id": "skill/a"}, {"id": "skill/b"}, {"skill_id": ""}])

    assert load_skill_id_set(skill_pool) == {"skill/a", "skill/b"}


def test_load_stage0_handoff_summary_keeps_topm_coverage_fields(tmp_path):
    report_path = tmp_path / "stage0_candidate_handoff.json"
    report_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "top_m": 350,
                "positive_missing_policy": "skip",
                "current_positive_required_rows": 10,
                "current_positive_covered_rows": 9,
                "current_positive_coverage@M": 0.9,
                "next_positive_required_rows": 8,
                "next_positive_covered_rows": 7,
                "next_positive_coverage@M": 0.875,
                "skipped_reasons": {"routing_positive_missing_from_stage0_topm": 1},
            }
        ),
        encoding="utf-8",
    )

    summary = load_stage0_handoff_summary(report_path)

    assert summary == {
        "enabled": True,
        "top_m": 350,
        "positive_missing_policy": "skip",
        "current_positive_required_rows": 10,
        "current_positive_covered_rows": 9,
        "current_positive_coverage@M": 0.9,
        "next_positive_required_rows": 8,
        "next_positive_covered_rows": 7,
        "next_positive_coverage@M": 0.875,
        "skipped_reasons": {"routing_positive_missing_from_stage0_topm": 1},
    }


def test_audit_logged_online_stage4_data_cli_writes_report(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    skill_pool = tmp_path / "skill_pool.jsonl"
    output_path = tmp_path / "audit.json"
    steps_output_path = tmp_path / "steps.jsonl"
    _write_jsonl(
        trajectories,
        [
            {
                "benchmark": "toolbench_g3",
                "trajectory_id": "traj-1",
                "step_index": 0,
                "goal_text": "goal",
                "state_text": "state",
                "skill_id": "skill/current",
                "next_skill_id": "skill/next",
                "candidate_next_skill_ids": ["skill/current", "skill/next"],
                "provenance": {"split": "train"},
            }
        ],
    )
    _write_jsonl(skill_pool, [{"skill_id": "skill/current"}, {"skill_id": "skill/next"}])

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_logged_online_stage4_data.py",
            "--trajectories_path",
            str(trajectories),
            "--skill_pool_path",
            str(skill_pool),
            "--output_path",
            str(output_path),
            "--steps_output_path",
            str(steps_output_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["emitted_steps"] == 1
    assert report["overall"]["stage4_updateable_steps"] == 1
    steps = [json.loads(line) for line in steps_output_path.read_text(encoding="utf-8").splitlines()]
    assert steps[0]["gt_next_skill_ids"] == ["skill/next"]
