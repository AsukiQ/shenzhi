import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_pass_rate import audit_stabletoolbench_pass_rate


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stub\n", encoding="utf-8")


def make_fake_stabletoolbench(root: Path) -> None:
    touch(root / "toolbench/tooleval/eval_pass_rate.py")
    touch(root / "toolbench/tooleval/convert_to_answer_format.py")
    touch(root / "toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/config.yaml")
    write_json(root / "solvable_queries/test_query_ids/G3_instruction.json", {"455": 0})


def test_audit_stabletoolbench_pass_rate_requires_valid_key_and_converted_answers(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_json(stable_root / "openai_key.json", [{"api_key": None, "api_base": None}])

    report = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=stable_root,
        converted_answer_path=tmp_path / "converted",
        api_pool_file=stable_root / "openai_key.json",
        candidate_model="clstr",
        test_set="G3_instruction",
    )

    assert report["status"] == "action_required"
    assert report["can_run_pass_rate"] is False
    assert "missing_converted_answer_file" in report["blockers"]
    assert "api_pool_has_no_valid_api_key" in report["blockers"]
    assert report["required_files"]["eval_pass_rate.py"]["exists"] is True
    assert report["metric_scope"] == "StableToolBench Solvable Pass Rate; not static routing recall."


def test_audit_stabletoolbench_pass_rate_ok_when_all_required_inputs_exist(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_json(stable_root / "openai_key.json", [{"api_key": "sk-test", "api_base": "https://example.test/v1"}])
    write_json(tmp_path / "converted/clstr/G3_instruction.json", {"455": {"query": "q", "available_tools": [], "answer": {}}})

    report = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=stable_root,
        converted_answer_path=tmp_path / "converted",
        api_pool_file=stable_root / "openai_key.json",
        candidate_model="clstr",
        test_set="G3_instruction",
    )

    assert report["status"] == "ok"
    assert report["can_run_pass_rate"] is True
    assert report["blockers"] == []
    assert report["converted_answer_file"] == str(tmp_path / "converted/clstr/G3_instruction.json")
    assert "eval_pass_rate.py" in " ".join(report["reproducible_command"])


def test_audit_stabletoolbench_pass_rate_reproducible_command_uses_absolute_clstr_paths(tmp_path, monkeypatch):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_json(stable_root / "openai_key.json", [{"api_key": "sk-test"}])
    write_json(tmp_path / "converted/clstr/G3_instruction.json", {"455": {"query": "q", "available_tools": [], "answer": {}}})
    monkeypatch.chdir(tmp_path)

    report = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=stable_root,
        converted_answer_path=Path("converted"),
        save_path=Path("save"),
        api_pool_file=stable_root / "openai_key.json",
        candidate_model="clstr",
        test_set="G3_instruction",
    )

    command = " ".join(report["reproducible_command"])
    assert f"--converted_answer_path {tmp_path / 'converted'}" in command
    assert f"--save_path {tmp_path / 'save' / 'clstr'}" in command


def test_audit_stabletoolbench_pass_rate_reproducible_command_can_use_logical_command_root(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_json(stable_root / "openai_key.json", [{"api_key": "sk-test"}])
    write_json(tmp_path / "logical/converted/clstr/G3_instruction.json", {"455": {"query": "q", "available_tools": [], "answer": {}}})

    report = audit_stabletoolbench_pass_rate(
        stabletoolbench_root=stable_root,
        converted_answer_path=Path("converted"),
        save_path=Path("save"),
        api_pool_file=stable_root / "openai_key.json",
        candidate_model="clstr",
        test_set="G3_instruction",
        command_root=tmp_path / "logical",
    )

    command = " ".join(report["reproducible_command"])
    assert f"--converted_answer_path {tmp_path / 'logical' / 'converted'}" in command
    assert f"--save_path {tmp_path / 'logical' / 'save' / 'clstr'}" in command


def test_audit_stabletoolbench_pass_rate_cli_writes_report_and_fails_on_action_required(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_json(stable_root / "openai_key.json", [{"api_key": None}])
    output_path = tmp_path / "audit.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_stabletoolbench_pass_rate.py",
            "--stabletoolbench_root",
            str(stable_root),
            "--converted_answer_path",
            str(tmp_path / "converted"),
            "--api_pool_file",
            str(stable_root / "openai_key.json"),
            "--candidate_model",
            "clstr",
            "--test_set",
            "G3_instruction",
            "--output_path",
            str(output_path),
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
    assert "missing_converted_answer_file" in report["blockers"]
