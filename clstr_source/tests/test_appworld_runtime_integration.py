import json
from pathlib import Path

from clstr.bridges.skillx.appworld_adapter import (
    build_appworld_embedding_inputs,
    summarize_appworld_skill_pool,
)
from clstr.envs.appworld_env import AppWorldEnvAdapter, run_appworld_adapter_smoke


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_appworld_root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld_root"
    data = root / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "datasets" / "train.txt").write_text("task_a_1\n", encoding="utf-8")
    task_dir = data / "tasks" / "task_a_1"
    _write_json(
        task_dir / "specs.json",
        {
            "instruction": "Find the most liked song in my Spotify playlists.",
            "supervisor": {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "email": "ada@example.com",
                "phone_number": "5550101",
            },
            "datetime": "2023-05-18T12:00:00",
            "db_version": "0.1.0",
        },
    )
    (task_dir / "dbs").mkdir(parents=True)
    (task_dir / "dbs" / "spotify.jsonl").write_text("{}\n", encoding="utf-8")
    (task_dir / "dbs" / "supervisor.jsonl").write_text("{}\n", encoding="utf-8")
    ground_truth = task_dir / "ground_truth"
    _write_json(ground_truth / "required_apps.json", ["spotify"])
    _write_json(ground_truth / "metadata.json", {"difficulty": 1, "num_apps": 1, "num_apis": 2})
    (ground_truth / "evaluation.py").write_text("def evaluate(*args, **kwargs):\n    return None\n", encoding="utf-8")
    _write_json(ground_truth / "answer.json", "Song title")
    api_docs = data / "api_docs" / "standard"
    _write_json(api_docs / "spotify.json", {"paths": {"/spotify/songs": {}}})
    _write_json(api_docs / "supervisor.json", {"paths": {"/supervisor/task": {}}})
    return root


class _FakeWorld:
    def __init__(self, task_id, experiment_name, **kwargs):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self.closed = False

    def execute(self, code):
        return f"executed: {code}"

    def task_completed(self):
        return False

    def close(self):
        self.closed = True


def test_appworld_adapter_smoke_loads_task_metadata_without_reset(tmp_path):
    root = _make_appworld_root(tmp_path)

    report = run_appworld_adapter_smoke(
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        attempt_reset=False,
    )

    assert report["status"] == "ok"
    assert report["task_id"] == "task_a_1"
    assert report["instruction_nonempty"] is True
    assert report["required_apps"] == ["spotify"]
    assert report["allowed_apps_count"] >= 1
    assert report["api_docs_available"] is True
    assert report["ground_truth_available"] is True
    assert report["verifier_available"] is True
    assert report["db_available"] is True
    assert report["reset_attempted"] is False
    assert (tmp_path / "outputs" / "report.json").exists()


def test_appworld_adapter_reset_and_step_use_execution_entry(tmp_path):
    root = _make_appworld_root(tmp_path)
    adapter = AppWorldEnvAdapter(
        appworld_root=root,
        appworld_cache=tmp_path / "cache",
        task_id="task_a_1",
        world_factory=_FakeWorld,
    )

    observation = adapter.reset()
    step = adapter.step("print('hello')")

    assert "Find the most liked song" in observation
    assert step.observation_text == "executed: print('hello')"
    assert step.done is False
    assert adapter.execution_entry_available() is True
    assert adapter.reward_done_available() is True
    adapter.close()
    assert adapter._world.closed is True


def test_skill_pool_smoke_report_summarizes_candidates(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    pool_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "skill_id": "skillx/appworld/spotify",
                        "name": "spotify find song",
                        "description": "Find a song.",
                        "body": "Use apis.spotify.show_song.",
                        "executor_desc": "apis.spotify.show_song",
                        "failure_modes": [],
                        "input_schema": {},
                        "output_schema": {},
                    }
                ),
                json.dumps(
                    {
                        "skill_id": "skillx/appworld/gmail",
                        "name": "gmail send message",
                        "description": "Send mail.",
                        "body": "Use apis.gmail.send_email.",
                        "executor_desc": "apis.gmail.send_email",
                        "failure_modes": [],
                        "input_schema": {},
                        "output_schema": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_appworld_skill_pool(pool_path)

    assert report["status"] == "ok"
    assert report["skill_count"] == 2
    assert report["field_coverage"]["body"] == 1.0
    assert report["field_coverage"]["executor_desc"] == 1.0
    assert len(report["sample_skills"]) == 2


def test_build_embedding_inputs_writes_one_row_per_skill(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    pool_path.write_text(
        json.dumps(
            {
                "skill_id": "skillx/appworld/spotify",
                "name": "spotify find song",
                "description": "Find a song.",
                "body": "Use apis.spotify.show_song.",
                "executor_desc": "apis.spotify.show_song",
                "failure_modes": [],
                "input_schema": {},
                "output_schema": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_appworld_embedding_inputs(
        pool_path=pool_path,
        output_path=tmp_path / "embedding_inputs.jsonl",
        manifest_path=tmp_path / "embedding_manifest.json",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "embedding_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["status"] == "ok"
    assert manifest["skill_count"] == 1
    assert rows[0]["skill_id"] == "skillx/appworld/spotify"
    assert "Skill Name: spotify find song" in rows[0]["embedding_text"]
