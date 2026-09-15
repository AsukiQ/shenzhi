import json
from pathlib import Path

from clstr.appworld_eval_subset import build_stratified_task_subset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_build_stratified_task_subset_round_robins_required_app_groups(tmp_path):
    rows = [
        {"task_id": "spotify_1", "required_apps": ["spotify"], "split": "dev"},
        {"task_id": "spotify_2", "required_apps": ["spotify"], "split": "dev"},
        {"task_id": "spotify_3", "required_apps": ["spotify"], "split": "dev"},
        {"task_id": "phone_1", "required_apps": ["phone", "venmo"], "split": "dev"},
        {"task_id": "phone_2", "required_apps": ["phone", "venmo"], "split": "dev"},
        {"task_id": "file_1", "required_apps": ["file_system"], "split": "dev"},
        {"task_id": "note_1", "required_apps": ["file_system", "simple_note"], "split": "dev"},
    ]
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "subset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(input_path, rows)

    manifest = build_stratified_task_subset(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        max_tasks=5,
    )

    selected = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    selected_ids = [row["task_id"] for row in selected]
    assert selected_ids == ["spotify_1", "phone_1", "file_1", "note_1", "spotify_2"]
    assert manifest["input_task_count"] == 7
    assert manifest["selected_task_count"] == 5
    assert manifest["group_counts"] == {
        "file_system": 1,
        "file_system+simple_note": 1,
        "phone+venmo": 2,
        "spotify": 3,
    }
    assert manifest["selected_group_counts"] == {
        "file_system": 1,
        "file_system+simple_note": 1,
        "phone+venmo": 1,
        "spotify": 2,
    }


def test_build_stratified_task_subset_excludes_supervisor_from_group_key(tmp_path):
    input_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "subset.jsonl"
    _write_jsonl(
        input_path,
        [
            {"task_id": "a", "required_apps": ["spotify", "supervisor"]},
            {"task_id": "b", "required_apps": ["supervisor"]},
        ],
    )

    manifest = build_stratified_task_subset(
        input_path=input_path,
        output_path=output_path,
        max_tasks=2,
    )

    assert manifest["group_counts"] == {"spotify": 1, "unknown": 1}
