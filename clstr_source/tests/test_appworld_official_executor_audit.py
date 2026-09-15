import json

from scripts.audit_appworld_official_executor_runs import audit_official_executor_runs


def test_audit_official_executor_runs_flags_invalid_api_and_complete_task_positional(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    docs_dir = appworld_root / "data" / "api_docs" / "standard"
    docs_dir.mkdir(parents=True)
    (docs_dir / "spotify.json").write_text(
        json.dumps(
            {
                "login": {"description": "Login.", "parameters": []},
                "search_songs": {"description": "Search songs.", "parameters": []},
            }
        ),
        encoding="utf-8",
    )

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "instruction_text": "Give me a song title.",
                "required_apps": ["spotify"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "success": False,
                "task_completed": False,
                "evaluation_success": False,
                "steps": [
                    {
                        "code": "apis.spotify.set_access_token('token')",
                        "preflight_ok": True,
                        "execution_attempted": True,
                        "execute_output": "Execution failed. No API named set_access_token.",
                        "skill_evidence": {
                            "selected_skill_ids": ["skillx/appworld/spotify-find-songs-based-on-play-count-59"]
                        },
                    },
                    {
                        "code": "result = 'A Song'\napis.supervisor.complete_task(result)",
                        "preflight_ok": True,
                        "execution_attempted": True,
                        "execute_output": "TypeError: positional argument",
                        "skill_evidence": {"selected_skill_ids": []},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_official_executor_runs(
        runs_path=runs_path,
        tasks_path=tasks_path,
        appworld_root=appworld_root,
    )

    assert report["task_count"] == 1
    assert report["step_count"] == 2
    assert report["invalid_api_ref_count"] == 1
    assert report["invalid_api_refs"]["apis.spotify.set_access_token"] == 1
    assert report["complete_task_positional_count"] == 1
    assert report["selected_skill_counts"]["skillx/appworld/spotify-find-songs-based-on-play-count-59"] == 1
