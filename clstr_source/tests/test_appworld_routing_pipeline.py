import json
from pathlib import Path

from clstr.appworld_eval import run_appworld_skillrouter_baseline
from clstr.appworld_routing import build_appworld_routing_corpus


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_appworld_root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld_root"
    (root / "data" / "datasets").mkdir(parents=True)
    (root / "data" / "datasets" / "train.txt").write_text("task_spotify_1\n", encoding="utf-8")
    (root / "data" / "datasets" / "dev.txt").write_text("task_spotify_2\n", encoding="utf-8")
    _write_json(
        root / "data" / "api_docs" / "standard" / "spotify.json",
        {
            "login": {"app_name": "spotify", "api_name": "login", "method": "POST", "path": "/spotify/auth/token"},
            "show_song": {
                "app_name": "spotify",
                "api_name": "show_song",
                "method": "GET",
                "path": "/spotify/songs/{song_id}",
            },
        },
    )
    for task_id, instruction in [
        ("task_spotify_1", "Find the most liked Spotify song."),
        ("task_spotify_2", "Report the Spotify song title."),
    ]:
        task_dir = root / "data" / "tasks" / task_id
        _write_json(task_dir / "specs.json", {"instruction": instruction})
        _write_json(task_dir / "ground_truth" / "required_apps.json", ["spotify"])
        _write_json(
            task_dir / "ground_truth" / "api_calls.json",
            [
                {"method": "POST", "url": "/spotify/auth/token", "data": {}},
                {"method": "GET", "url": "/spotify/songs/259", "data": {}},
            ],
        )
    return root


def _fake_skill_pool(tmp_path: Path) -> Path:
    path = tmp_path / "skill_pool.jsonl"
    _write_jsonl(
        path,
        [
            {
                "skill_id": "skillx/appworld/spotify-show-song",
                "name": "spotify show song",
                "description": "Fetch Spotify song metadata and title.",
                "executor_desc": "apis.spotify.show_song",
                "input_schema": {},
                "output_schema": {},
                "failure_modes": [],
                "body": "Use apis.spotify.show_song to inspect a Spotify song.",
            },
            {
                "skill_id": "skillx/appworld/gmail-send-message",
                "name": "gmail send message",
                "description": "Send an email message.",
                "executor_desc": "apis.gmail.send_email",
                "input_schema": {},
                "output_schema": {},
                "failure_modes": [],
                "body": "Use Gmail APIs.",
            },
        ],
    )
    return path


def test_build_appworld_routing_corpus_links_tasks_to_skillx_skills(tmp_path):
    appworld_root = _fake_appworld_root(tmp_path)
    skill_pool = _fake_skill_pool(tmp_path)
    output_dir = tmp_path / "routing"

    report = build_appworld_routing_corpus(
        appworld_root=appworld_root,
        skill_pool_path=skill_pool,
        output_dir=output_dir,
        splits=("train", "dev"),
        positives_per_task=1,
    )

    assert report["status"] == "ok"
    assert report["splits"]["train"]["task_count"] == 1
    train_rows = [json.loads(line) for line in (output_dir / "train_tasks.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (output_dir / "train_qrels.jsonl").read_text().splitlines()]
    replay = [json.loads(line) for line in (output_dir / "train_replay.jsonl").read_text().splitlines()]
    assert train_rows[0]["positive_skill_ids"] == ["skillx/appworld/spotify-show-song"]
    assert train_rows[0]["query"].startswith("Instruction: Find the most liked Spotify song.")
    assert qrels == [
        {
            "query_id": "task_spotify_1",
            "skill_id": "skillx/appworld/spotify-show-song",
            "relevance": 1,
            "split": "train",
        }
    ]
    assert replay[0]["steps"][0]["skill_name"] == "spotify show song"


def test_skillrouter_baseline_runs_on_appworld_routing_qrels(tmp_path):
    appworld_root = _fake_appworld_root(tmp_path)
    skill_pool = _fake_skill_pool(tmp_path)
    data_dir = tmp_path / "routing"
    build_appworld_routing_corpus(
        appworld_root=appworld_root,
        skill_pool_path=skill_pool,
        output_dir=data_dir,
        splits=("dev",),
        positives_per_task=1,
    )

    report = run_appworld_skillrouter_baseline(
        tasks_path=data_dir / "dev_tasks.jsonl",
        qrels_path=data_dir / "dev_qrels.jsonl",
        skill_pool_path=skill_pool,
        output_dir=tmp_path / "baseline",
        top_k=2,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_serialization_lexical"
    assert report["query_count"] == 1
    assert report["skill_count"] == 2
    assert report["metrics"]["recall@1"] == 1.0
    predictions = [json.loads(line) for line in Path(report["predictions_path"]).read_text().splitlines()]
    assert predictions[0]["ranked_skill_ids"][0] == "skillx/appworld/spotify-show-song"
