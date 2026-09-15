import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_raw_generation import audit_stabletoolbench_raw_generation


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    write_text(path, json.dumps(payload))


def make_fake_stabletoolbench(root: Path) -> None:
    write_text(root / "toolbench/inference/qa_pipeline_multithread.py", "# fake official runner\n")
    write_json(root / "solvable_queries/test_instruction/G3_instruction.json", [{"query": "q", "api_list": []}])


def make_fake_tool_root(root: Path) -> None:
    write_json(
        root / "Media/vimeo.json",
        {
            "tool_name": "Vimeo",
            "tool_description": "video tools",
            "api_list": [
                {
                    "name": "SearchVideos",
                    "description": "Search for videos.",
                    "required_parameters": [],
                    "optional_parameters": [],
                }
            ],
        },
    )


def test_audit_stabletoolbench_raw_generation_requires_tool_root_and_llm_credentials(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)

    report = audit_stabletoolbench_raw_generation(
        stabletoolbench_root=stable_root,
        tool_root_dir=tmp_path / "missing_tools",
        raw_answer_path=tmp_path / "raw_answers",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CLSTR@1",
        backbone_model="chatgpt_function",
        openai_key="",
    )

    assert report["status"] == "action_required"
    assert report["can_generate_raw_answers"] is False
    assert "missing_tool_root_dir" in report["blockers"]
    assert "missing_openai_key_for_chatgpt_function" in report["blockers"]
    assert report["metric_scope"] == "StableToolBench raw answer generation; prerequisite for answer conversion and SoPR."


def test_audit_stabletoolbench_raw_generation_ok_for_chatgpt_function_with_tools_and_key(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    tool_root = tmp_path / "toolenv/tools"
    make_fake_tool_root(tool_root)

    report = audit_stabletoolbench_raw_generation(
        stabletoolbench_root=stable_root,
        tool_root_dir=tool_root,
        raw_answer_path=tmp_path / "raw_answers",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CLSTR@1",
        backbone_model="chatgpt_function",
        openai_key="sk-test",
        service_url="http://127.0.0.1:8080/virtual",
    )

    assert report["status"] == "ok"
    assert report["can_generate_raw_answers"] is True
    assert report["blockers"] == []
    assert report["input_query_file"] == str(stable_root / "solvable_queries/test_instruction/G3_instruction.json")
    assert report["output_answer_file"] == str(tmp_path / "raw_answers/clstr_toolbench_g3/G3_instruction")
    assert report["tool_json_file_count"] == 1
    assert report["openai_key_provided"] is True
    command = " ".join(report["reproducible_command"])
    assert "qa_pipeline_multithread.py" in command
    assert f"--tool_root_dir {tool_root}" in command
    assert f"--output_answer_file {tmp_path / 'raw_answers/clstr_toolbench_g3/G3_instruction'}" in command
    assert "SERVICE_URL=http://127.0.0.1:8080/virtual" in command
    assert "sk-test" not in command


def test_audit_stabletoolbench_raw_generation_accepts_derived_input_query_file(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    tool_root = tmp_path / "toolenv/tools"
    make_fake_tool_root(tool_root)
    derived_query_file = tmp_path / "outputs/toolbench_g3/stabletoolbench_clstr_top50/G3_instruction.json"
    write_json(derived_query_file, [{"query_id": 455, "query": "q", "api_list": []}])

    report = audit_stabletoolbench_raw_generation(
        stabletoolbench_root=stable_root,
        tool_root_dir=tool_root,
        raw_answer_path=tmp_path / "raw_answers",
        candidate_model="clstr_toolbench_g3_top50",
        test_set="G3_instruction",
        input_query_file=derived_query_file,
        method="CLSTR@1",
        backbone_model="chatgpt_function",
        openai_key="sk-test",
        service_url="http://127.0.0.1:8080/virtual",
    )

    assert report["status"] == "ok"
    assert report["input_query_file"] == str(derived_query_file)
    assert report["input_query_source"] == "derived"
    assert report["uses_official_full_api_list"] is False
    command = " ".join(report["reproducible_command"])
    assert f"--input_query_file {derived_query_file}" in command


def test_audit_stabletoolbench_raw_generation_rejects_data_example_tool_root(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    tool_root = tmp_path / "ToolBench/data_example/toolenv/tools"
    make_fake_tool_root(tool_root)

    report = audit_stabletoolbench_raw_generation(
        stabletoolbench_root=stable_root,
        tool_root_dir=tool_root,
        raw_answer_path=tmp_path / "raw_answers",
        candidate_model="clstr_toolbench_g3",
        test_set="G3_instruction",
        method="CLSTR@1",
        backbone_model="chatgpt_function",
        openai_key="sk-test",
        service_url="http://127.0.0.1:8080/virtual",
    )

    assert report["status"] == "action_required"
    assert "tool_root_dir_is_data_example_not_official" in report["blockers"]


def test_audit_stabletoolbench_raw_generation_cli_writes_report_and_fails_on_action_required(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_stabletoolbench(stable_root)
    output_path = tmp_path / "raw_generation_readiness.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_stabletoolbench_raw_generation.py",
            "--stabletoolbench_root",
            str(stable_root),
            "--tool_root_dir",
            str(tmp_path / "missing_tools"),
            "--raw_answer_path",
            str(tmp_path / "raw_answers"),
            "--candidate_model",
            "clstr_toolbench_g3",
            "--test_set",
            "G3_instruction",
            "--method",
            "CLSTR@1",
            "--backbone_model",
            "chatgpt_function",
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
    assert "missing_tool_root_dir" in report["blockers"]
