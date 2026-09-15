import json
from pathlib import Path

from clstr.external_data import (
    build_clean_router_manifest,
    check_alfworld_root,
)


def _make_skillsbench_tasks(root: Path, task_ids: list[str]) -> None:
    for task_id in task_ids:
        task_dir = root / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text(f"# {task_id}\n", encoding="utf-8")
        (task_dir / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")


def test_build_clean_router_manifest_refuses_to_emit_training_files_when_inputs_missing(tmp_path):
    skillsbench_root = tmp_path / "skillsbench"
    clean_root = tmp_path / "clean_router"
    audit_dir = tmp_path / "audit"
    _make_skillsbench_tasks(skillsbench_root, ["task-a", "task-b"])
    (skillsbench_root / "splits").mkdir()
    (skillsbench_root / "splits" / "train_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "train", "task_ids": ["task-a"]}),
        encoding="utf-8",
    )
    (audit_dir).mkdir()
    (audit_dir / "excluded_positive_skill_ids.json").write_text(json.dumps(["gt/a"]), encoding="utf-8")

    manifest = build_clean_router_manifest(
        skillsbench_root=skillsbench_root,
        audit_dir=audit_dir,
        output_dir=clean_root,
    )

    assert manifest["status"] == "missing_inputs"
    assert "skill_pool" in manifest["missing_inputs"]
    assert "successful_trajectories" in manifest["missing_inputs"]
    assert "harness_results" in manifest["missing_inputs"]
    assert (clean_root / "manifest.json").exists()
    assert not (clean_root / "train.jsonl").exists()


def test_build_clean_router_manifest_emits_train_jsonl_only_from_successful_train_tasks(tmp_path):
    skillsbench_root = tmp_path / "skillsbench"
    clean_root = tmp_path / "clean_router"
    audit_dir = tmp_path / "audit"
    _make_skillsbench_tasks(skillsbench_root, ["task-a", "task-b"])
    (skillsbench_root / "splits").mkdir()
    (skillsbench_root / "splits" / "train_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "train", "task_ids": ["task-a"]}),
        encoding="utf-8",
    )
    (audit_dir).mkdir()
    (audit_dir / "excluded_positive_skill_ids.json").write_text(json.dumps(["eval/skill"]), encoding="utf-8")
    (skillsbench_root / "skill_pool.jsonl").write_text(
        json.dumps({"skill_id": "task-a/skill-one", "task_id": "task-a", "name": "Skill One"}) + "\n",
        encoding="utf-8",
    )
    (skillsbench_root / "successful_trajectories.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-a",
                        "success": True,
                        "steps": [{"skill_id": "task-a/skill-one", "observation": "done"}],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-b",
                        "success": True,
                        "steps": [{"skill_id": "task-b/skill-two", "observation": "not train"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (skillsbench_root / "harness_results.jsonl").write_text(
        json.dumps({"task_id": "task-a", "success": True, "status": "success"}) + "\n",
        encoding="utf-8",
    )
    (skillsbench_root / "historical_import_manifest.json").write_text(
        json.dumps(
            {
                "filter": {
                    "dataset": "benchflow/skillsbench-trajectories-apr2026",
                    "kept_train_successful_with_skills": 1,
                    "split_excluded": {"train": 2, "dev": 0, "test": 0, "unknown": 0},
                    "split_exclusion_reasons": {"train": {"without_skills": 2}},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_clean_router_manifest(
        skillsbench_root=skillsbench_root,
        audit_dir=audit_dir,
        output_dir=clean_root,
    )

    assert manifest["status"] == "ready"
    assert manifest["record_counts"]["train"] == 1
    train_records = [
        json.loads(line)
        for line in (clean_root / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["task_id"] for record in train_records] == ["task-a"]
    assert train_records[0]["skill_ids"] == ["task-a/skill-one"]
    skills = [
        json.loads(line)
        for line in (clean_root / "skills.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    train_tasks = [
        json.loads(line)
        for line in (clean_root / "train_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [skill["skill_id"] for skill in skills] == ["task-a/skill-one"]
    assert train_tasks == [{"task_id": "task-a", "instruction_text": "# task-a\n"}]
    assert manifest["historical_import"]["dataset"] == "benchflow/skillsbench-trajectories-apr2026"
    assert manifest["historical_import"]["split_excluded"]["train"] == 2
    assert manifest["historical_import"]["split_exclusion_reasons"]["train"] == {"without_skills": 2}
    assert manifest["historical_import"]["kept_train_successful_with_skills"] == 1


def test_build_clean_router_manifest_treats_failure_only_harness_as_missing_success(tmp_path):
    skillsbench_root = tmp_path / "skillsbench"
    clean_root = tmp_path / "clean_router"
    audit_dir = tmp_path / "audit"
    _make_skillsbench_tasks(skillsbench_root, ["task-a"])
    (skillsbench_root / "splits").mkdir()
    (skillsbench_root / "splits" / "train_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "train", "task_ids": ["task-a"]}),
        encoding="utf-8",
    )
    (audit_dir).mkdir()
    (audit_dir / "excluded_positive_skill_ids.json").write_text(json.dumps([]), encoding="utf-8")
    (skillsbench_root / "skill_pool.jsonl").write_text(
        json.dumps({"skill_id": "task-a/skill-one", "task_id": "task-a"}) + "\n",
        encoding="utf-8",
    )
    (skillsbench_root / "successful_trajectories.jsonl").write_text("", encoding="utf-8")
    (skillsbench_root / "harness_results.jsonl").write_text(
        json.dumps({"task_id": "task-a", "success": False, "status": "blocked"}) + "\n",
        encoding="utf-8",
    )

    manifest = build_clean_router_manifest(
        skillsbench_root=skillsbench_root,
        audit_dir=audit_dir,
        output_dir=clean_root,
    )

    assert manifest["status"] == "missing_inputs"
    assert "successful_trajectories" in manifest["missing_inputs"]
    assert not (clean_root / "train.jsonl").exists()


def test_check_alfworld_root_reports_missing_without_import_side_effects(tmp_path):
    report = check_alfworld_root(tmp_path / "missing_alfworld")

    assert report["status"] == "missing"
    assert report["path"] == str(tmp_path / "missing_alfworld")
    assert "adapter_requirements" in report
