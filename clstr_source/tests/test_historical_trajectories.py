import json
from pathlib import Path

from clstr.data import load_verified_pairs
from clstr.historical_trajectories import (
    build_verified_pairs_from_successful_trajectories,
    import_historical_skillsbench_trajectories,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run(
    root: Path,
    rel_dir: str,
    task_id: str,
    success: bool,
    with_skills: bool,
    trajectory: list[dict],
    model: str = "claude-opus-4-7",
) -> None:
    run_dir = root / rel_dir / f"{task_id}__trial-1"
    write_json(
        run_dir / "result.json",
        {
            "task_name": task_id,
            "agent": "claude-agent-acp",
            "model": model,
            "rewards": [1 if success else 0],
            "error": None if success else "failed verifier",
        },
    )
    write_json(
        run_dir / "config.json",
        {
            "task_path": f"tasks/{task_id}",
            "agent": "claude-agent-acp",
            "model": model,
            "skills_dir": f"tasks/{task_id}/environment/skills" if with_skills else None,
        },
    )
    write_jsonl(run_dir / "trajectory" / "acp_trajectory.jsonl", trajectory)
    (run_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (run_dir / "verifier" / "reward.txt").write_text("1\n" if success else "0\n", encoding="utf-8")


def make_audit_dir(root: Path) -> None:
    write_json(root / "skillrouter_eval_query_ids.json", ["eval-overlap"])
    write_json(root / "skillsbench_train_task_ids.json", {"status": "ok", "split": "train", "task_ids": ["safe-train", "eval-overlap"]})
    write_json(root / "skillsbench_dev_task_ids.json", {"status": "ok", "split": "dev", "task_ids": ["safe-dev"]})
    write_json(root / "skillsbench_test_task_ids.json", {"status": "ok", "split": "test", "task_ids": ["safe-test"]})


def make_skillsbench_root(root: Path) -> None:
    for task_id in ["safe-train", "eval-overlap", "safe-dev", "safe-test"]:
        task_dir = root / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "instruction.md").write_text(f"# {task_id}\n", encoding="utf-8")
    write_jsonl(
        root / "skill_pool.jsonl",
        [
            {"skill_id": "safe-train/skill-a", "task_id": "safe-train", "name": "skill-a"},
            {"skill_id": "safe-train/skill-b", "task_id": "safe-train", "name": "skill-b"},
        ],
    )


def test_import_historical_trajectories_filters_to_train_successful_with_skills(tmp_path):
    dataset_root = tmp_path / "hf"
    skillsbench_root = tmp_path / "skillsbench"
    audit_dir = tmp_path / "audit"
    output_dir = tmp_path / "outputs"
    make_audit_dir(audit_dir)
    make_skillsbench_root(skillsbench_root)
    make_run(
        dataset_root,
        "opus-with-skills",
        "safe-train",
        True,
        True,
        [
            {
                "type": "tool_call",
                "title": "Skill",
                "content": [{"content": {"text": "Launching skill: skill-a"}}],
            },
            {"skill_id": "safe-train/skill-b"},
        ],
    )
    make_run(
        dataset_root,
        "opus-without-skills",
        "safe-train",
        True,
        False,
        [{"skill_id": "safe-train/skill-a"}],
    )
    make_run(
        dataset_root,
        "opus-with-skills",
        "safe-dev",
        True,
        True,
        [{"skill_id": "safe-train/skill-a"}],
    )
    make_run(
        dataset_root,
        "opus-with-skills",
        "eval-overlap",
        True,
        True,
        [{"skill_id": "safe-train/skill-a"}],
    )
    make_run(
        dataset_root,
        "sonnet-with-skills",
        "safe-train",
        False,
        True,
        [{"skill_id": "safe-train/skill-a"}],
        model="claude-sonnet-4-6",
    )

    report = import_historical_skillsbench_trajectories(
        dataset_root=dataset_root,
        skillsbench_root=skillsbench_root,
        audit_dir=audit_dir,
        output_dir=output_dir,
        harness_results_path=skillsbench_root / "harness_results.jsonl",
        successful_trajectories_path=skillsbench_root / "successful_trajectories.jsonl",
        clstr_harness_output_dir=tmp_path / "clstr_harness",
    )

    assert report["filter"]["kept_train_successful_with_skills"] == 1
    assert report["filter"]["excluded"]["not_train_split"] == 1
    assert report["filter"]["excluded"]["skillrouter_eval_overlap"] == 1
    assert report["filter"]["excluded"]["without_skills"] == 1
    assert report["filter"]["excluded"]["not_successful"] == 1
    assert report["filter"]["split_excluded"] == {
        "train": 3,
        "dev": 1,
        "test": 0,
        "unknown": 0,
    }
    assert report["filter"]["split_exclusion_reasons"]["train"] == {
        "not_successful": 1,
        "skillrouter_eval_overlap": 1,
        "without_skills": 1,
    }
    assert report["filter"]["split_exclusion_reasons"]["dev"] == {"not_train_split": 1}
    assert (output_dir / "schema_report.json").exists()
    assert (output_dir / "filter_report.json").exists()
    harness_rows = [
        json.loads(line)
        for line in (skillsbench_root / "harness_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    success_rows = [
        json.loads(line)
        for line in (skillsbench_root / "successful_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["task_id"] for row in harness_rows] == ["safe-train"]
    assert success_rows == [
        {
            "task_id": "safe-train",
            "success": True,
            "steps": [{"skill_id": "safe-train/skill-a"}, {"skill_id": "safe-train/skill-b"}],
            "source": "skillsbench_historical_agent_trajectory",
            "status": "completed",
            "agent": "claude-agent-acp",
            "model": "claude-opus-4-7",
            "trajectory_source_path": "opus-with-skills/safe-train__trial-1",
        }
    ]
    copied = tmp_path / "clstr_harness" / "successful_trajectories.jsonl"
    assert copied.exists()


def test_import_historical_trajectories_reports_missing_skill_mapping_without_fake_success(tmp_path):
    dataset_root = tmp_path / "hf"
    skillsbench_root = tmp_path / "skillsbench"
    audit_dir = tmp_path / "audit"
    output_dir = tmp_path / "outputs"
    make_audit_dir(audit_dir)
    make_skillsbench_root(skillsbench_root)
    make_run(
        dataset_root,
        "opus-with-skills",
        "safe-train",
        True,
        True,
        [{"skill_name": "not-in-pool"}],
    )

    report = import_historical_skillsbench_trajectories(
        dataset_root=dataset_root,
        skillsbench_root=skillsbench_root,
        audit_dir=audit_dir,
        output_dir=output_dir,
        harness_results_path=skillsbench_root / "harness_results.jsonl",
        successful_trajectories_path=skillsbench_root / "successful_trajectories.jsonl",
        clstr_harness_output_dir=tmp_path / "clstr_harness",
    )

    assert report["filter"]["kept_train_successful_with_skills"] == 0
    assert report["filter"]["excluded"]["missing_skill_mapping"] == 1
    assert not (skillsbench_root / "successful_trajectories.jsonl").exists()


def test_build_verified_pairs_requires_explicit_raw_candidates(tmp_path):
    skill_pool = tmp_path / "skill_pool.jsonl"
    successful = tmp_path / "successful_trajectories.jsonl"
    output = tmp_path / "verified.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(
        skill_pool,
        [
            {"skill_id": "task/skill-a", "name": "skill-a"},
            {"skill_id": "task/skill-b", "name": "skill-b"},
        ],
    )
    write_jsonl(
        successful,
        [
            {
                "task_id": "task",
                "success": True,
                "steps": [{"skill_id": "task/skill-a"}, {"skill_id": "task/skill-b"}],
            }
        ],
    )

    report = build_verified_pairs_from_successful_trajectories(
        successful_trajectories_path=successful,
        skill_pool_path=skill_pool,
        output_jsonl=output,
        manifest_path=manifest,
    )

    assert report["status"] == "missing_raw_candidates"
    assert report["pair_candidate_count"] == 1
    assert report["emitted_pair_count"] == 0
    assert not output.exists()


def test_build_verified_pairs_uses_raw_candidates_without_polluting_raw_recall(tmp_path):
    skill_pool = tmp_path / "skill_pool.jsonl"
    successful = tmp_path / "successful_trajectories.jsonl"
    output = tmp_path / "verified.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(
        skill_pool,
        [
            {"skill_id": "task/skill-a", "name": "skill-a"},
            {"skill_id": "task/skill-b", "name": "skill-b"},
            {"skill_id": "task/skill-c", "name": "skill-c"},
        ],
    )
    write_jsonl(
        successful,
        [
            {
                "task_id": "task",
                "success": True,
                "query": "do the task",
                "steps": [
                    {"skill_id": "task/skill-a", "observation": "a done"},
                    {"skill_id": "task/skill-b", "raw_candidates": ["task/skill-c"], "observation": "b done"},
                ],
            }
        ],
    )

    report = build_verified_pairs_from_successful_trajectories(
        successful_trajectories_path=successful,
        skill_pool_path=skill_pool,
        output_jsonl=output,
        manifest_path=manifest,
    )

    assert report["status"] == "ok"
    assert report["emitted_pair_count"] == 1
    pair = load_verified_pairs(output)[0]
    assert pair.a_next_plus == 1
    assert pair.was_in_raw_topk is False
    assert pair.candidates_next == [2, 1]
