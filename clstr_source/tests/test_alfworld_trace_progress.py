import json
import sys
from pathlib import Path

import pytest

import clstr.alfworld_trace_progress as trace_progress
from clstr.alfworld_trace_progress import (
    build_trace_progress_report,
    compare_trace_progress_reports,
    resolve_alfworld_goal_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_resolve_goal_text_prefers_traj_data_pddl_params(tmp_path):
    trial_dir = tmp_path / "valid_seen" / "pick_cool_then_place_in_recep-Apple-None-Fridge-1" / "trial_T1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "traj_data.json").write_text(
        json.dumps(
            {
                "task_type": "pick_cool_then_place_in_recep",
                "pddl_params": {
                    "object_target": "Apple",
                    "parent_target": "Fridge",
                    "mrecep_target": "",
                    "toggle_target": "",
                },
                "turk_annotations": {"anns": [{"task_desc": "Cool an apple and place it in the fridge."}]},
            }
        ),
        encoding="utf-8",
    )

    goal = resolve_alfworld_goal_text({"gamefile": str(trial_dir / "game.tw-pddl")})

    assert "find apple" in goal
    assert "cool apple" in goal
    assert "fridge" in goal


def test_trace_progress_report_rewards_goal_actions_over_navigation(tmp_path):
    run_path = tmp_path / "run.jsonl"
    _write_jsonl(
        run_path,
        [
            {
                "goal_text": "find apple and cool apple with fridge and put it in fridge",
                "success": False,
                "points": 0.0,
                "goal_condition_points": 0.0,
                "steps": 3,
                "action_trace": [
                    "take apple 1 from countertop 1",
                    "cool apple 1 with fridge 1",
                    "move apple 1 to fridge 1",
                ],
            },
            {
                "goal_text": "find apple and cool apple with fridge and put it in fridge",
                "success": False,
                "points": 0.0,
                "goal_condition_points": 0.0,
                "steps": 3,
                "action_trace": ["go to countertop 1", "go to fridge 1", "examine fridge 1"],
            },
        ],
    )

    report = build_trace_progress_report(run_path, method_name="candidate")

    assert report["episode_count"] == 2
    assert report["mean_goal_interaction_points"] > 0.0
    assert report["mean_placed_target_count"] == pytest.approx(0.5)
    assert report["mean_progress_reward"] > 0.0
    assert report["rows"][0]["progress_reward"] > report["rows"][1]["progress_reward"]


def test_compare_trace_progress_reports_detects_candidate_uplift(tmp_path):
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    goal = "find apple and cool apple with fridge and put it in fridge"
    _write_jsonl(
        reference_path,
        [
            {
                "goal_text": goal,
                "success": False,
                "points": 0.0,
                "goal_condition_points": 0.0,
                "steps": 3,
                "action_trace": ["go to fridge 1", "go to countertop 1", "go to fridge 1"],
            }
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {
                "goal_text": goal,
                "success": False,
                "points": 0.0,
                "goal_condition_points": 0.0,
                "steps": 3,
                "action_trace": [
                    "take apple 1 from countertop 1",
                    "cool apple 1 with fridge 1",
                    "move apple 1 to fridge 1",
                ],
            }
        ],
    )
    reference = build_trace_progress_report(reference_path, method_name="reference")
    candidate = build_trace_progress_report(candidate_path, method_name="candidate")

    comparison = compare_trace_progress_reports(reference, candidate)

    assert comparison["status"] == "ok"
    assert comparison["mean_progress_reward_delta"] > 0.0
    assert comparison["mean_goal_interaction_points_delta"] > 0.0
    assert comparison["recommendation"] == "trace_progress_uplift_observed"


def test_trace_progress_cli_writes_output_path_from_string_argv(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output_path = tmp_path / "trace_gate.json"
    row = {
        "goal_text": "find apple and put it in fridge",
        "success": False,
        "points": 0.0,
        "goal_condition_points": 0.0,
        "steps": 1,
        "action_trace": ["go to fridge 1"],
    }
    _write_jsonl(reference_path, [row])
    _write_jsonl(candidate_path, [row])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alfworld_trace_progress.py",
            "--reference_run",
            str(reference_path),
            "--candidate_run",
            str(candidate_path),
            "--output_path",
            str(output_path),
        ],
    )

    trace_progress.main()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["reference_episode_count"] == 1
