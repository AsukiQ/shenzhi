import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_virtual_api import audit_stabletoolbench_virtual_api


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    write_text(path, json.dumps(payload))


def make_fake_tool_root(root: Path) -> None:
    write_json(
        root / "Media/vimeo.json",
        {
            "tool_name": "vimeo",
            "tool_description": "video tools",
            "api_list": [{"name": "search", "description": "Search videos."}],
        },
    )


def make_fake_server(root: Path, *, mode: str = "mirrorapi", config_text: str) -> None:
    server_dir = root / "server"
    script_name = {
        "mirrorapi": "main_mirrorapi.py",
        "mirrorapi_cache": "main_mirrorapi_cache.py",
        "gpt_cache": "main.py",
    }[mode]
    write_text(server_dir / script_name, "# fake server\n")
    config_name = {
        "mirrorapi": "config_mirrorapi.yml",
        "mirrorapi_cache": "config_mirrorapi_cache.yml",
        "gpt_cache": "config.yml",
    }[mode]
    write_text(server_dir / config_name, config_text)


def test_audit_stabletoolbench_virtual_api_rejects_root_config_tool_path(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    make_fake_server(
        stable_root,
        config_text=(
            'api_key: "EMPTY"\n'
            'api_base: "http://127.0.0.1:12345/v1"\n'
            "temperature: 0.1\n"
            'tools_folder: "/root/gzc/StableToolBench/toolenv/tools"\n'
            "port: 12001\n"
            "model: simulation-250123-qwen25-mixed\n"
        ),
    )

    report = audit_stabletoolbench_virtual_api(
        stabletoolbench_root=stable_root,
        mode="mirrorapi",
        autodl_tmp_root=tmp_path,
    )

    assert report["status"] == "action_required"
    assert report["can_start_virtual_api"] is False
    assert "tools_folder_outside_autodl_tmp" in report["blockers"]
    assert "missing_tools_folder" in report["blockers"]
    assert report["service_url"] == "http://127.0.0.1:12001/virtual"
    assert report["metric_scope"] == "StableToolBench virtual API readiness for official ToolBench-G3 raw generation."


def test_audit_stabletoolbench_virtual_api_ok_for_local_mirrorapi_config(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    tool_root = tmp_path / "StableToolBench/toolenv/tools"
    make_fake_tool_root(tool_root)
    make_fake_server(
        stable_root,
        config_text=(
            'api_key: "EMPTY"\n'
            'api_base: "http://127.0.0.1:12345/v1"\n'
            "temperature: 0.1\n"
            f'tools_folder: "{tool_root}"\n'
            "port: 12001\n"
            "model: simulation-250123-qwen25-mixed\n"
        ),
    )

    report = audit_stabletoolbench_virtual_api(
        stabletoolbench_root=stable_root,
        mode="mirrorapi",
        autodl_tmp_root=tmp_path,
    )

    assert report["status"] == "ok"
    assert report["can_start_virtual_api"] is True
    assert report["blockers"] == []
    assert report["service_url"] == "http://127.0.0.1:12001/virtual"
    assert report["tool_json_file_count"] == 1
    command = " ".join(report["reproducible_server_command"])
    assert f"cd {stable_root / 'server'}" in command
    assert "python main_mirrorapi.py" in command
    assert "uvicorn" not in command


def test_audit_stabletoolbench_virtual_api_rejects_data_example_tools(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    tool_root = tmp_path / "ToolBench/data_example/toolenv/tools"
    make_fake_tool_root(tool_root)
    make_fake_server(
        stable_root,
        config_text=(
            'api_key: "EMPTY"\n'
            'api_base: "http://127.0.0.1:12345/v1"\n'
            "temperature: 0.1\n"
            f'tools_folder: "{tool_root}"\n'
            "port: 12001\n"
            "model: simulation-250123-qwen25-mixed\n"
        ),
    )

    report = audit_stabletoolbench_virtual_api(
        stabletoolbench_root=stable_root,
        mode="mirrorapi",
        autodl_tmp_root=tmp_path,
    )

    assert report["status"] == "action_required"
    assert "tools_folder_is_data_example_not_official" in report["blockers"]


def test_audit_stabletoolbench_virtual_api_gpt_cache_requires_key_and_toolbench_url(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    tool_root = tmp_path / "StableToolBench/server/tools"
    make_fake_tool_root(tool_root)
    make_fake_server(
        stable_root,
        mode="gpt_cache",
        config_text=(
            "api_key:\n"
            "api_base:\n"
            "model: gpt-4-turbo\n"
            "temperature: 0\n"
            "toolbench_url:\n"
            'tools_folder: "./tools"\n'
            'cache_folder: "./tool_response_cache"\n'
            "is_save: true\n"
            "port: 8080\n"
            'log_file: "./server.log"\n'
        ),
    )

    report = audit_stabletoolbench_virtual_api(
        stabletoolbench_root=stable_root,
        mode="gpt_cache",
        autodl_tmp_root=tmp_path,
    )

    assert report["status"] == "action_required"
    assert "missing_api_key_for_gpt_cache" in report["blockers"]
    assert "missing_toolbench_url_for_gpt_cache" in report["blockers"]
    assert report["resolved_config"]["tools_folder"] == str(tool_root)


def test_audit_stabletoolbench_virtual_api_cli_writes_report_and_fails_on_action_required(tmp_path):
    stable_root = tmp_path / "StableToolBench"
    output_path = tmp_path / "virtual_api_readiness.json"
    make_fake_server(
        stable_root,
        config_text=(
            'api_key: "EMPTY"\n'
            'api_base: "http://127.0.0.1:12345/v1"\n'
            "temperature: 0.1\n"
            'tools_folder: "/root/gzc/StableToolBench/toolenv/tools"\n'
            "port: 12001\n"
            "model: simulation-250123-qwen25-mixed\n"
        ),
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_stabletoolbench_virtual_api.py",
            "--stabletoolbench_root",
            str(stable_root),
            "--mode",
            "mirrorapi",
            "--autodl_tmp_root",
            str(tmp_path),
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
    assert "tools_folder_outside_autodl_tmp" in report["blockers"]
