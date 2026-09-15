import json
from pathlib import Path

from clstr.appworld_label_audit import audit_appworld_task_positive_labels


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_label_audit_flags_task_positive_when_semantic_best_is_better(tmp_path):
    tasks_path = tmp_path / "dev_tasks.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task-played",
                "query_id": "task-played",
                "instruction_text": "Give me the top most played song titles.",
                "positive_skill_ids": ["skillx/appworld/spotify-duration", "skillx/appworld/spotify-auth"],
            }
        ],
    )
    _write_jsonl(
        skills_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-duration",
                "name": "spotify analyze playlist durations",
                "description": "Calculate playlist duration from songs.",
                "appworld_executor_compatible": True,
            },
            {
                "skill_id": "skillx/appworld/spotify-auth",
                "name": "spotify authenticate with stored credentials",
                "description": "Login to Spotify.",
                "appworld_executor_compatible": True,
            },
            {
                "skill_id": "skillx/appworld/spotify-play-count",
                "name": "spotify find songs based on play count",
                "description": "Find songs sorted by most played count.",
                "appworld_executor_compatible": True,
            },
        ],
    )

    report = audit_appworld_task_positive_labels(
        tasks_path=tasks_path,
        skill_pool_path=skills_path,
        top_k=1,
    )

    assert report["task_count"] == 1
    assert report["suspicious_task_count"] == 1
    task = report["suspicious_tasks"][0]
    assert task["task_id"] == "task-played"
    assert task["semantic_best_skill_ids"] == ["skillx/appworld/spotify-play-count"]
    assert task["semantic_best_positive_overlap"] is False
    assert "semantic_best_misses_positive" in task["flags"]


def test_label_audit_compares_executor_selected_skills_against_task_positives(tmp_path):
    tasks_path = tmp_path / "dev_tasks.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task-played",
                "query_id": "task-played",
                "instruction_text": "Give me the top most played song titles.",
                "positive_skill_ids": ["skillx/appworld/spotify-duration"],
            }
        ],
    )
    _write_jsonl(
        skills_path,
        [
            {
                "skill_id": "skillx/appworld/spotify-duration",
                "name": "spotify analyze playlist durations",
                "description": "Calculate playlist duration from songs.",
                "appworld_executor_compatible": True,
            },
            {
                "skill_id": "skillx/appworld/spotify-play-count",
                "name": "spotify find songs based on play count",
                "description": "Find most played songs.",
                "appworld_executor_compatible": True,
            },
        ],
    )
    _write_jsonl(
        runs_path,
        [
            {
                "task_id": "task-played",
                "user_goal": "Give me the top most played song titles.",
                "steps": [
                    {
                        "step_idx": 0,
                        "selected_skill_ids": ["skillx/appworld/spotify-play-count"],
                        "positive_skill_ids": ["skillx/appworld/spotify-duration"],
                    }
                ],
            }
        ],
    )

    report = audit_appworld_task_positive_labels(
        tasks_path=tasks_path,
        skill_pool_path=skills_path,
        runs_path=runs_path,
        top_k=1,
    )

    assert report["executor_run_alignment"]["step_count"] == 1
    assert report["executor_run_alignment"]["selected_better_than_positive_count"] == 1
    assert report["executor_run_alignment"]["flagged_steps"][0]["selected_skill_ids"] == ["skillx/appworld/spotify-play-count"]
