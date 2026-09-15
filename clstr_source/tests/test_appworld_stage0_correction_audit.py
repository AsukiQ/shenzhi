import json
from pathlib import Path

from clstr.appworld_stage0_correction_audit import (
    build_high_confidence_stage0_correction_subset,
    build_train_stage0_correction_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_train_stage0_correction_audit_without_candidates_only_reports_pool_matches(tmp_path):
    tasks_path = tmp_path / "train_tasks.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "split": "train",
                "train_allowed": True,
                "instruction_text": "Create a playlist and add songs.",
                "query": "Instruction: Create a playlist and add songs.",
                "api_refs": ["spotify.create_playlist", "spotify.add_song_to_playlist"],
            }
        ],
    )
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/create", "executor_desc": "apis.spotify.create_playlist"},
            {"skill_id": "skill/add", "executor_desc": "apis.spotify.add_song_to_playlist"},
            {"skill_id": "skill/read", "executor_desc": "apis.spotify.show_song"},
        ],
    )

    report = build_train_stage0_correction_audit(
        train_tasks_path=tasks_path,
        skill_pool_path=skills_path,
        output_dir=output_dir,
        max_targets=3,
    )

    assert report["candidate_audit_enabled"] is False
    assert report["train_task_count"] == 1
    assert report["pool_match_task_count"] == 1
    assert report["no_skill_pool_match_count"] == 0
    assert report["stage0_retrieval_correction_count"] == 0
    pool_rows = [json.loads(line) for line in (output_dir / "stage0_pool_positive_rows.jsonl").read_text().splitlines()]
    assert {row["positive_skill_id"] for row in pool_rows} == {"skill/create", "skill/add"}


def test_train_stage0_correction_audit_writes_corrections_only_when_candidate_misses_positive(tmp_path):
    tasks_path = tmp_path / "train_tasks.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    candidates_path = tmp_path / "candidate_topm.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_1",
                "query_id": "task_1",
                "split": "train",
                "train_allowed": True,
                "instruction_text": "Create a playlist and add songs.",
                "query": "Instruction: Create a playlist and add songs.",
                "api_refs": ["spotify.create_playlist", "spotify.add_song_to_playlist"],
            },
            {
                "task_id": "task_2",
                "query_id": "task_2",
                "split": "train",
                "train_allowed": True,
                "instruction_text": "Create a playlist.",
                "query": "Instruction: Create a playlist.",
                "api_refs": ["spotify.create_playlist"],
            },
        ],
    )
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "skill/create", "executor_desc": "apis.spotify.create_playlist"},
            {"skill_id": "skill/add", "executor_desc": "apis.spotify.add_song_to_playlist"},
            {"skill_id": "skill/read", "executor_desc": "apis.spotify.show_song"},
        ],
    )
    _write_jsonl(
        candidates_path,
        [
            {"query_id": "task_1", "candidate_skill_ids": ["skill/read"]},
            {"query_id": "task_2", "ranked_skill_ids": ["skill/create", "skill/read"]},
        ],
    )

    report = build_train_stage0_correction_audit(
        train_tasks_path=tasks_path,
        skill_pool_path=skills_path,
        output_dir=output_dir,
        candidate_rows_path=candidates_path,
        max_targets=3,
    )

    assert report["candidate_audit_enabled"] is True
    assert report["candidate_positive_missing_task_count"] == 1
    assert report["candidate_positive_covered_task_count"] == 1
    assert report["stage0_retrieval_correction_count"] == 2
    correction_rows = [
        json.loads(line) for line in (output_dir / "stage0_api_retrieval_corrections.jsonl").read_text().splitlines()
    ]
    assert {row["positive_skill_id"] for row in correction_rows} == {"skill/create", "skill/add"}
    assert all(row["negative_skill_ids"] == ["skill/read"] for row in correction_rows)


def test_train_stage0_correction_audit_does_not_train_auth_or_supervisor_as_stage0_targets(tmp_path):
    tasks_path = tmp_path / "train_tasks.jsonl"
    skills_path = tmp_path / "skill_pool.jsonl"
    candidates_path = tmp_path / "candidate_topm.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "task_read_song",
                "query_id": "task_read_song",
                "split": "train",
                "train_allowed": True,
                "instruction_text": "Find the least-played song in my Spotify library.",
                "query": "Instruction: Find the least-played song in my Spotify library.",
                "api_refs": [
                    "supervisor.show_account_passwords",
                    "supervisor.show_profile",
                    "spotify.login",
                    "spotify.show_song",
                    "spotify.show_song_library",
                ],
            }
        ],
    )
    _write_jsonl(
        skills_path,
        [
            {
                "skill_id": "skill/auth",
                "executor_desc": "apis.supervisor.show_account_passwords; apis.supervisor.show_profile; apis.spotify.login",
            },
            {"skill_id": "skill/show-song", "executor_desc": "apis.spotify.show_song"},
            {"skill_id": "skill/show-library", "executor_desc": "apis.spotify.show_song_library"},
        ],
    )
    _write_jsonl(candidates_path, [{"query_id": "task_read_song", "ranked_skill_ids": ["skill/auth"]}])

    report = build_train_stage0_correction_audit(
        train_tasks_path=tasks_path,
        skill_pool_path=skills_path,
        output_dir=output_dir,
        candidate_rows_path=candidates_path,
        max_targets=5,
    )

    assert report["pool_positive_row_count"] == 2
    assert report["stage0_retrieval_correction_count"] == 2
    correction_rows = [
        json.loads(line) for line in (output_dir / "stage0_api_retrieval_corrections.jsonl").read_text().splitlines()
    ]
    assert {row["positive_skill_id"] for row in correction_rows} == {"skill/show-song", "skill/show-library"}
    assert "skill/auth" not in {row["positive_skill_id"] for row in correction_rows}


def test_high_confidence_stage0_correction_subset_keeps_write_overlap_and_reports_skips(tmp_path):
    input_path = tmp_path / "stage0_api_retrieval_corrections.jsonl"
    output_dir = tmp_path / "high_confidence"
    _write_jsonl(
        input_path,
        [
            {
                "query_id": "task_write",
                "positive_skill_id": "skill/create",
                "negative_skill_ids": ["skill/read"],
                "provenance": {
                    "task_id": "task_write",
                    "solution_api_refs": ["apis.spotify.create_playlist", "apis.spotify.add_song_to_playlist"],
                    "target_match_detail": {
                        "exact_api_overlap": ["apis.spotify.create_playlist"],
                        "write_api_overlap": ["apis.spotify.create_playlist"],
                    },
                },
            },
            {
                "query_id": "task_read",
                "positive_skill_id": "skill/show",
                "negative_skill_ids": ["skill/auth"],
                "provenance": {
                    "task_id": "task_read",
                    "solution_api_refs": ["apis.spotify.show_song", "apis.spotify.show_song_library"],
                    "target_match_detail": {
                        "exact_api_overlap": ["apis.spotify.show_song"],
                        "write_api_overlap": [],
                    },
                },
            },
        ],
    )

    report = build_high_confidence_stage0_correction_subset(
        input_path=input_path,
        output_dir=output_dir,
        policy="write_only",
    )

    assert report["input_count"] == 2
    assert report["kept_count"] == 1
    assert report["skipped_read_only_count"] == 1
    kept_rows = [json.loads(line) for line in (output_dir / "stage0_high_confidence_corrections.jsonl").read_text().splitlines()]
    assert [row["positive_skill_id"] for row in kept_rows] == ["skill/create"]
    assert kept_rows[0]["source"] == "appworld_train_stage0_high_confidence_api_correction"
    assert kept_rows[0]["provenance"]["high_confidence_policy"] == "write_only"
