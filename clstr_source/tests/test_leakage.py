import json
import subprocess
import sys
from pathlib import Path

from clstr.leakage import run_leakage_audit


def make_skillrouter_eval_core(root: Path) -> None:
    root.mkdir()
    (root / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_id": "q1", "instruction_text": "task one"}),
                json.dumps({"task_id": "q2", "instruction_text": "task two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "relevance.json").write_text(
        json.dumps(
            {
                "q1": {"gt_skill_ids": ["gt/a"], "core_gt_ids": ["gt/a"]},
                "q2": {"gt_skill_ids": ["gt/b", "gt/c"], "core_gt_ids": ["gt/b"]},
            }
        ),
        encoding="utf-8",
    )


def test_run_leakage_audit_extracts_skillrouter_eval_ids_and_marks_missing_splits(tmp_path):
    eval_root = tmp_path / "skillrouter_eval_core"
    audit_dir = tmp_path / "audit"
    make_skillrouter_eval_core(eval_root)

    report = run_leakage_audit(
        skillrouter_eval_root=eval_root,
        skillsbench_root=tmp_path / "missing_skillsbench",
        output_dir=audit_dir,
    )

    assert report["skillrouter_eval"]["status"] == "ok"
    assert report["skillsbench"]["status"] == "missing"
    assert json.loads((audit_dir / "skillrouter_eval_query_ids.json").read_text()) == ["q1", "q2"]
    assert json.loads((audit_dir / "skillrouter_eval_gt_skill_ids.json").read_text()) == {
        "q1": ["gt/a"],
        "q2": ["gt/b", "gt/c"],
    }
    assert json.loads((audit_dir / "skillsbench_train_task_ids.json").read_text()) == {
        "status": "missing",
        "split": "train",
        "task_ids": [],
    }
    assert (audit_dir / "excluded_positive_skill_ids.json").exists()


def test_run_leakage_audit_cli_writes_summary(tmp_path):
    eval_root = tmp_path / "skillrouter_eval_core"
    audit_dir = tmp_path / "audit"
    make_skillrouter_eval_core(eval_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_leakage_audit.py",
            "--skillrouter_eval_root",
            str(eval_root),
            "--skillsbench_root",
            str(tmp_path / "missing_skillsbench"),
            "--output_dir",
            str(audit_dir),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["skillsbench"]["status"] == "missing"
    assert (audit_dir / "leakage_audit_summary.json").exists()


def test_run_leakage_audit_reports_skillsbench_train_overlap_with_skillrouter_eval_queries(tmp_path):
    eval_root = tmp_path / "skillrouter_eval_core"
    skillsbench_root = tmp_path / "skillsbench"
    audit_dir = tmp_path / "audit"
    make_skillrouter_eval_core(eval_root)
    (skillsbench_root / "splits").mkdir(parents=True)
    (skillsbench_root / "splits" / "train_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "train", "task_ids": ["q1", "safe-train"]}),
        encoding="utf-8",
    )
    (skillsbench_root / "splits" / "dev_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "dev", "task_ids": ["safe-dev"]}),
        encoding="utf-8",
    )
    (skillsbench_root / "splits" / "test_task_ids.json").write_text(
        json.dumps({"status": "ok", "split": "test", "task_ids": ["safe-test"]}),
        encoding="utf-8",
    )

    report = run_leakage_audit(
        skillrouter_eval_root=eval_root,
        skillsbench_root=skillsbench_root,
        output_dir=audit_dir,
    )

    assert report["leakage_checks"]["status"] == "failed"
    assert report["leakage_checks"]["skillsbench_train_overlaps_skillrouter_eval_queries"] == ["q1"]
