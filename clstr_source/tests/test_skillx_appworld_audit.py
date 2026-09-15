import json
import os
import subprocess
import sys

from clstr.bridges.skillx.appworld_adapter import (
    audit_skillx_appworld,
    audit_skillx_appworld_leakage,
    build_appworld_skill_pool,
)


def test_audit_skillx_appworld_reports_missing_root(tmp_path):
    report = audit_skillx_appworld(tmp_path / "missing-skillx")

    assert report["status"] == "blocked"
    assert report["blocker"] == "skillx_root_missing"
    assert report["skill_count"] == 0


def test_audit_skillx_appworld_reads_skill_md_and_builds_pool(tmp_path):
    skill_dir = tmp_path / "SkillX" / "appworld" / "skills" / "send_email"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: send-email-with-attachment\n"
        "description: Send an email with an attachment in AppWorld.\n"
        "---\n\n"
        "# Send Email With Attachment\n\n"
        "Use Gmail and FileSystem APIs to locate a file, attach it, and send a message.\n\n"
        "Failure modes: missing file, invalid recipient, authentication failure.\n",
        encoding="utf-8",
    )

    report = audit_skillx_appworld(tmp_path / "SkillX", min_skill_count=1)
    pool_report = build_appworld_skill_pool(tmp_path / "SkillX", tmp_path / "pool", min_skill_count=1)

    assert report["status"] == "ok"
    assert report["skill_count"] == 1
    assert report["field_coverage"]["name"] == 1.0
    assert report["field_coverage"]["description"] == 1.0
    assert report["field_coverage"]["body"] == 1.0
    assert pool_report["status"] == "ok"

    rows = [
        json.loads(line)
        for line in (tmp_path / "pool" / "skill_pool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["name"] == "send-email-with-attachment"
    assert rows[0]["description"] == "Send an email with an attachment in AppWorld."
    assert "Gmail" in rows[0]["body"]
    assert rows[0]["source_dataset"] == "SkillX-AppWorld"


def test_audit_skillx_appworld_reads_skillx_nested_json_schema(tmp_path):
    skill_file = tmp_path / "SkillX" / "skillx_db" / "appworld" / "vanilla-iter1" / "func_atomic_skills.json"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        json.dumps(
            [
                {
                    "option": "add",
                    "skill": {
                        "name": "spotify find oldest song by release date",
                        "document": "Find the oldest released song from candidate song ids.",
                        "content": "for song_id in song_ids:\n    song = apis.spotify.show_song(song_id=song_id)",
                        "tools": ["apis.spotify.show_song"],
                    },
                    "filter_result": True,
                    "embedding_text": "spotify find oldest song by release date",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = audit_skillx_appworld(tmp_path / "SkillX", min_skill_count=1)
    pool_report = build_appworld_skill_pool(tmp_path / "SkillX", tmp_path / "pool", min_skill_count=1)

    assert report["status"] == "ok"
    assert report["skill_count"] == 1
    assert report["field_coverage"]["executor_desc"] == 1.0
    assert pool_report["status"] == "ok"

    row = json.loads((tmp_path / "pool" / "skill_pool.jsonl").read_text(encoding="utf-8").strip())
    assert row["name"] == "spotify find oldest song by release date"
    assert row["description"] == "Find the oldest released song from candidate song ids."
    assert "apis.spotify.show_song" in row["body"]
    assert row["executor_desc"] == "apis.spotify.show_song"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_skillx_appworld_leakage_flags_dev_test_task_id_and_instruction(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    routing_dir = tmp_path / "routing"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        pool_path,
        [
            {
                "skill_id": "skillx/appworld/leaky",
                "name": "leaky skill",
                "description": "Uses task dev123_1 directly.",
                "body": "Exact solution for dev123_1: Reset friends on venmo to be the same as my friends in my phone.",
                "executor_desc": "apis.venmo.show_friends",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "dev_tasks.jsonl",
        [
            {
                "task_id": "dev123_1",
                "instruction_text": "Reset friends on venmo to be the same as my friends in my phone.",
                "split": "dev",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "test_normal_tasks.jsonl",
        [
            {
                "task_id": "test999_1",
                "instruction_text": "Archive every file system receipt from April into the tax folder.",
                "split": "test_normal",
            }
        ],
    )

    report = audit_skillx_appworld_leakage(pool_path, routing_dir, output_dir=output_dir)

    assert report["status"] == "failed"
    assert report["risk_level"] == "high"
    assert report["dev_test_contamination_count"] == 2
    assert report["contamination"]["task_id_hits"]["dev"][0]["task_id"] == "dev123_1"
    assert report["contamination"]["instruction_hits"]["dev"][0]["task_id"] == "dev123_1"
    assert (output_dir / "report.json").exists()
    assert "dev/test contamination" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_audit_skillx_appworld_leakage_allows_train_only_matches(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    routing_dir = tmp_path / "routing"
    _write_jsonl(
        pool_path,
        [
            {
                "skill_id": "skillx/appworld/train-sourced",
                "name": "train sourced skill",
                "description": "Derived from train123_1.",
                "body": "Find the title of the most-liked song in my Spotify playlists.",
                "executor_desc": "apis.spotify.show_playlist_library",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "train_tasks.jsonl",
        [
            {
                "task_id": "train123_1",
                "instruction_text": "Find the title of the most-liked song in my Spotify playlists.",
                "split": "train",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "dev_tasks.jsonl",
        [
            {
                "task_id": "dev456_1",
                "instruction_text": "Send a payment reminder to my roommate.",
                "split": "dev",
            }
        ],
    )

    report = audit_skillx_appworld_leakage(pool_path, routing_dir)

    assert report["status"] == "ok"
    assert report["risk_level"] == "low"
    assert report["train_overlap_count"] == 2
    assert report["dev_test_contamination_count"] == 0


def test_audit_skillx_appworld_leakage_flags_skillx_plan_test_source(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    routing_dir = tmp_path / "routing"
    skillx_root = tmp_path / "SkillX"
    _write_jsonl(
        pool_path,
        [
            {
                "skill_id": "skillx/appworld/generic",
                "name": "generic skill",
                "description": "Generic Venmo helper.",
                "body": "Authenticate and inspect Venmo transactions.",
                "executor_desc": "apis.venmo.show_transactions",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "test_normal_tasks.jsonl",
        [
            {
                "task_id": "test999_1",
                "instruction_text": "Reset friends on venmo to be the same as my friends in my phone.",
                "split": "test_normal",
            }
        ],
    )
    plan_path = skillx_root / "skillx_db" / "appworld" / "vanilla-iter1" / "plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "train_epoch": 1,
                "plan": {
                    "Reset friends on venmo to be the same as my friends in my phone.": "# step 1: inspect friends"
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_skillx_appworld_leakage(pool_path, routing_dir, skillx_root=skillx_root)

    assert report["status"] == "failed"
    assert report["source_plan_task_count"] == 1
    assert report["contamination"]["source_plan_instruction_overlaps"]["test_normal"][0]["task_id"] == "test999_1"


def test_audit_skillx_appworld_cli_runs_leakage_audit(tmp_path):
    pool_path = tmp_path / "skill_pool.jsonl"
    routing_dir = tmp_path / "routing"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        pool_path,
        [
            {
                "skill_id": "skillx/appworld/clean",
                "name": "clean skill",
                "description": "Generic file organization helper.",
                "body": "List files, filter names, and move matching files into a destination directory.",
                "executor_desc": "apis.file_system.show_directory",
            }
        ],
    )
    _write_jsonl(
        routing_dir / "dev_tasks.jsonl",
        [
            {
                "task_id": "dev456_1",
                "instruction_text": "Send a payment reminder to my roommate.",
                "split": "dev",
            }
        ],
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_skillx_appworld.py",
            "--leakage_audit",
            "--skill_pool_path",
            str(pool_path),
            "--routing_dir",
            str(routing_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"status": "ok"' in result.stdout
    assert json.loads((output_dir / "report.json").read_text(encoding="utf-8"))["status"] == "ok"
