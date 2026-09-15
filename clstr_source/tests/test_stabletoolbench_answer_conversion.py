import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_answer_conversion import (
    _normalize_openai_tool_call_messages,
    audit_stabletoolbench_answer_conversion,
    convert_stabletoolbench_answers,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_stabletoolbench(root: Path) -> None:
    write_text(root / "toolbench/tooleval/convert_to_answer_format.py", "print('fake convert')\n")


def test_audit_stabletoolbench_answer_conversion_requires_raw_answers(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)

    report = audit_stabletoolbench_answer_conversion(
        stabletoolbench_root=stable_root,
        raw_answer_path=tmp_path / "raw",
        converted_answer_path=tmp_path / "converted",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CLSTR@1",
    )

    assert report["status"] == "action_required"
    assert report["can_convert_answers"] is False
    assert "missing_raw_answer_dir" in report["blockers"]
    assert "no_raw_answer_files_matching_method" in report["blockers"]
    assert report["raw_answer_file_count"] == 0
    assert report["metric_scope"] == "StableToolBench answer format conversion; prerequisite for SoPR, not pass-rate."


def test_audit_stabletoolbench_answer_conversion_ok_when_method_files_exist(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    write_text(tmp_path / "raw/clstr_toolbench_g3/G3_instruction/455_CLSTR@1.json", "{}\n")
    write_text(tmp_path / "raw/clstr_toolbench_g3/G3_instruction/999_other.json", "{}\n")

    report = audit_stabletoolbench_answer_conversion(
        stabletoolbench_root=stable_root,
        raw_answer_path=tmp_path / "raw",
        converted_answer_path=tmp_path / "converted",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CLSTR@1",
    )

    assert report["status"] == "ok"
    assert report["can_convert_answers"] is True
    assert report["blockers"] == []
    assert report["answer_dir"] == str(tmp_path / "raw/clstr_toolbench_g3/G3_instruction")
    assert report["raw_answer_file_count"] == 2
    assert report["matching_raw_answer_file_count"] == 1
    command = " ".join(report["reproducible_command"])
    assert "convert_to_answer_format.py" in command
    assert f"--answer_dir {tmp_path / 'raw/clstr_toolbench_g3/G3_instruction'}" in command
    assert f"--output {tmp_path / 'converted/clstr_toolbench_g3/G3_instruction.json'}" in command


def test_convert_stabletoolbench_answers_cli_writes_report_and_fails_on_action_required(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    output_path = tmp_path / "audit.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/convert_stabletoolbench_answers.py",
            "--stabletoolbench_root",
            str(stable_root),
            "--raw_answer_path",
            str(tmp_path / "raw"),
            "--converted_answer_path",
            str(tmp_path / "converted"),
            "--candidate_model",
            "clstr_toolbench_g3",
            "--test_set",
            "G3_instruction",
            "--method",
            "CLSTR@1",
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
    assert "missing_raw_answer_dir" in report["blockers"]


def test_normalize_openai_tool_calls_removes_all_inert_call_fields():
    payload = {
        "answer_generation": {
            "train_messages": [
                {
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": [{"function": {"name": "search"}}],
                },
                {"role": "assistant", "function_call": None},
                {
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": [],
                },
                {
                    "role": "assistant",
                    "function_call": {"name": "legacy_search"},
                    "tool_calls": [{"function": {"name": "search"}}],
                },
            ]
        }
    }

    normalized, changed = _normalize_openai_tool_call_messages(payload)

    messages = normalized["answer_generation"]["train_messages"]
    assert changed == 5
    assert "function_call" not in messages[0]
    assert messages[0]["tool_calls"] == [{"function": {"name": "search"}}]
    assert "function_call" not in messages[1]
    assert "function_call" not in messages[2]
    assert "tool_calls" not in messages[2]
    assert messages[3]["function_call"] == {"name": "legacy_search"}
    assert payload["answer_generation"]["train_messages"][0]["function_call"] is None


def test_convert_materializes_normalized_copy_without_mutating_raw_answers(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    converter = stable_root / "toolbench/tooleval/convert_to_answer_format.py"
    write_text(
        converter,
        """import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--answer_dir', required=True)
parser.add_argument('--method', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source = next(Path(args.answer_dir).glob(f'*_{args.method}.json'))
payload = json.loads(source.read_text())
message = payload['answer_generation']['train_messages'][0]
assert 'function_call' not in message
assert message['tool_calls']
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'converted': True}) + '\\n')
print('Converted 1 answers')
""",
    )
    raw_file = tmp_path / "raw/clstr_toolbench_g3/G3_instruction/455_CoT@1.json"
    raw_payload = {
        "answer_generation": {
            "train_messages": [
                {
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": [{"function": {"name": "search"}}],
                }
            ]
        }
    }
    write_text(raw_file, json.dumps(raw_payload) + "\n")

    report = convert_stabletoolbench_answers(
        stabletoolbench_root=stable_root,
        raw_answer_path=tmp_path / "raw",
        converted_answer_path=tmp_path / "converted",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CoT@1",
        command_root=tmp_path,
        run_convert=True,
    )

    assert report["status"] == "ok"
    assert report["conversion_returncode"] == 0
    assert report["normalized_file_count"] == 1
    assert report["normalized_message_count"] == 1
    assert report["converted_answer_exists_after"] is True
    assert json.loads(raw_file.read_text(encoding="utf-8")) == raw_payload
    normalized_file = Path(report["normalized_answer_dir"]) / raw_file.name
    normalized = json.loads(normalized_file.read_text(encoding="utf-8"))
    assert "function_call" not in normalized["answer_generation"]["train_messages"][0]
