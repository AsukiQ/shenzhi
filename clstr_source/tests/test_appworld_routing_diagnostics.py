import json
from pathlib import Path

from clstr.appworld_routing_diagnostics import (
    build_duplicate_aware_eval_report,
    build_executor_failure_topk_report,
    build_train_dev_overlap_report,
    canonical_skill_key,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_duplicate_aware_eval_counts_same_named_skill_as_canonical_hit(tmp_path):
    skills_path = tmp_path / "skills.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        skills_path,
        [
            {"skill_id": "auth-a", "name": "Spotify authenticate", "executor_desc": "apis.spotify.login"},
            {"skill_id": "auth-b", "name": "spotify_authenticate", "executor_desc": "apis.spotify.login"},
            {"skill_id": "song", "name": "Spotify show song", "executor_desc": "apis.spotify.show_song"},
        ],
    )
    _write_jsonl(qrels_path, [{"query_id": "q1", "skill_id": "auth-a", "relevance": 1}])
    _write_jsonl(predictions_path, [{"query_id": "q1", "ranked_skill_ids": ["auth-b", "song"]}])

    report = build_duplicate_aware_eval_report(
        skill_pool_path=skills_path,
        qrels_path=qrels_path,
        predictions_path=predictions_path,
        output_dir=tmp_path / "out",
        method="fake",
        ks=(1, 2),
    )

    assert canonical_skill_key({"name": "spotify_authenticate", "executor_desc": "apis.spotify.login"}) == canonical_skill_key(
        {"name": "Spotify authenticate", "executor_desc": "apis.spotify.login"}
    )
    assert report["id_metrics"]["recall@1"] == 0.0
    assert report["canonical_metrics"]["recall@1"] == 1.0
    assert report["duplicate_cluster_count"] == 1
    assert report["miss_rows"][0]["canonical_hit@1"] is True
    assert (tmp_path / "out" / "report.json").exists()


def test_train_dev_overlap_report_counts_positive_set_reuse_and_nearest_text(tmp_path):
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "task_id": "train_a",
                "instruction_text": "Find Spotify songs from this year.",
                "positive_skill_ids": ["auth", "song"],
            },
            {
                "task_id": "train_b",
                "instruction_text": "Send one Gmail message.",
                "positive_skill_ids": ["gmail"],
            },
        ],
    )
    _write_jsonl(
        dev_path,
        [
            {
                "task_id": "dev_a",
                "instruction_text": "Find Spotify songs from last year.",
                "positive_skill_ids": ["auth", "song"],
            },
            {
                "task_id": "dev_b",
                "instruction_text": "Create a todo item.",
                "positive_skill_ids": ["todo"],
            },
        ],
    )

    report = build_train_dev_overlap_report(
        train_tasks_path=train_path,
        dev_tasks_path=dev_path,
        output_dir=tmp_path / "overlap",
    )

    assert report["dev_task_count"] == 2
    assert report["exact_positive_set_reuse_count"] == 1
    assert report["dev_positive_skills_seen_in_train"] == 2
    assert report["nearest_instruction_jaccard"]["max"] > 0.5
    assert (tmp_path / "overlap" / "report.json").exists()


def test_executor_failure_topk_report_flags_focus_failures_with_reference_success(tmp_path):
    skills_path = tmp_path / "skills.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    clstr_predictions = tmp_path / "clstr_predictions.jsonl"
    skillrouter_predictions = tmp_path / "skillrouter_predictions.jsonl"
    clstr_runs = tmp_path / "clstr_runs.jsonl"
    skillrouter_runs = tmp_path / "skillrouter_runs.jsonl"

    _write_jsonl(
        skills_path,
        [
            {
                "skill_id": "spotify-auth",
                "name": "Spotify authenticate",
                "executor_desc": "apis.spotify.login",
            },
            {
                "skill_id": "phone-auth",
                "name": "Phone authenticate",
                "executor_desc": "apis.phone.login",
            },
            {
                "skill_id": "playlist",
                "name": "Spotify playlist song analysis",
                "executor_desc": "apis.spotify.show_playlist_library apis.spotify.show_song",
            },
            {
                "skill_id": "playlist-copy",
                "name": "Spotify playlist song analysis",
                "executor_desc": "apis.spotify.show_playlist_library apis.spotify.show_song",
            },
        ],
    )
    _write_jsonl(
        tasks_path,
        [
            {
                "task_id": "t1",
                "query_id": "t1",
                "instruction_text": "Find Spotify playlist songs.",
                "required_apps": ["spotify"],
            }
        ],
    )
    _write_jsonl(qrels_path, [{"query_id": "t1", "skill_id": "playlist", "relevance": 1}])
    _write_jsonl(clstr_predictions, [{"query_id": "t1", "ranked_skill_ids": ["spotify-auth", "phone-auth", "playlist-copy"]}])
    _write_jsonl(skillrouter_predictions, [{"query_id": "t1", "ranked_skill_ids": ["playlist", "spotify-auth"]}])
    _write_jsonl(
        clstr_runs,
        [
            {
                "query_id": "t1",
                "method": "clstr_base",
                "success": False,
                "evaluation_success": False,
                "execution_ok": False,
                "task_completed": False,
                "selected_skill_ids": ["spotify-auth", "phone-auth", "playlist-copy"],
            }
        ],
    )
    _write_jsonl(
        skillrouter_runs,
        [
            {
                "query_id": "t1",
                "method": "skillrouter_base",
                "success": True,
                "evaluation_success": True,
                "execution_ok": True,
                "task_completed": True,
                "selected_skill_ids": ["playlist", "spotify-auth"],
            }
        ],
    )

    report = build_executor_failure_topk_report(
        skill_pool_path=skills_path,
        tasks_path=tasks_path,
        qrels_path=qrels_path,
        run_paths={"clstr_base": clstr_runs, "skillrouter_base": skillrouter_runs},
        prediction_paths={"clstr_base": clstr_predictions, "skillrouter_base": skillrouter_predictions},
        output_dir=tmp_path / "report",
        focus_method="clstr_base",
        reference_methods=["skillrouter_base"],
        top_k=3,
    )

    assert report["method_summaries"]["clstr_base"]["success_count"] == 0
    assert report["method_summaries"]["skillrouter_base"]["success_count"] == 1
    assert report["focus_failure_reference_success_count"] == 1
    row = report["focus_failure_reference_success_rows"][0]
    assert row["query_id"] == "t1"
    assert row["focus_topk_stats"]["auth_like_count"] == 2
    assert row["focus_topk_stats"]["off_app_skill_count"] == 1
    assert row["focus_id_hit_rank"] is None
    assert row["focus_canonical_hit_rank"] == 3
    assert row["reference_methods_success"] == ["skillrouter_base"]
    assert (tmp_path / "report" / "report.json").exists()
