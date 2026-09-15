import json
from pathlib import Path

from scripts.audit_appworld_verified_handoff import audit_verified_handoff


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_verified_handoff_audit_summarizes_suppressed_polluting_skill(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    _write_json(
        appworld_root / "data" / "api_docs" / "standard" / "spotify.json",
        {
            "show_song_library": {"description": "List songs."},
            "show_song": {"description": "Show song."},
        },
    )
    skill_pool = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        skill_pool,
        [
            {
                "skill_id": "skillx/appworld/spotify-find-songs-based-on-play-count-59",
                "name": "spotify find songs based on play count",
                "description": "Find songs and rank them by play count.",
                "executor_desc": "apis.spotify.show_song_library, apis.spotify.show_song",
            }
        ],
    )
    runs_path = tmp_path / "runs.jsonl"
    _write_jsonl(
        runs_path,
        [
            {
                "query_id": "task_1",
                "user_goal": "How many unique songs are there across my Spotify song library?",
                "steps": [
                    {
                        "step_idx": 0,
                        "state_text": "[User Goal]\nHow many unique songs are there across my Spotify song library?\n\nRequired apps: spotify",
                        "selected_skill_ids": ["skillx/appworld/spotify-find-songs-based-on-play-count-59"],
                    }
                ],
            }
        ],
    )

    report = audit_verified_handoff(
        runs_path=runs_path,
        skill_pool_path=skill_pool,
        appworld_root=appworld_root,
        output_dir=tmp_path / "audit",
    )

    assert report["step_count"] == 1
    assert report["decision_counts"]["suppress"] == 1
    rows = [json.loads(line) for line in Path(report["rows_path"]).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["handoff_decision_counts"]["suppress"] == 1
    assert rows[0]["handoff_decisions"][0]["decision"] == "suppress"
    assert "unique_count_goal_vs_ranking_skill" in rows[0]["handoff_decisions"][0]["reasons"]
