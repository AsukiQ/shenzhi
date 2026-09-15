import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_toolenv import build_stabletoolbench_toolenv_from_queries


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_stabletoolbench_toolenv_from_official_queries(tmp_path):
    query_file = tmp_path / "StableToolBench/solvable_queries/test_instruction/G3_instruction.json"
    write_json(
        query_file,
        [
            {
                "query_id": 455,
                "query": "find videos",
                "api_list": [
                    {
                        "category_name": "Media",
                        "tool_name": "Vimeo",
                        "api_name": "SearchVideos",
                        "api_description": "Search for videos.",
                        "required_parameters": [{"name": "query", "type": "STRING", "description": "term"}],
                        "optional_parameters": [{"name": "page", "type": "NUMBER", "description": "page"}],
                        "method": "GET",
                    },
                    {
                        "category_name": "Media",
                        "tool_name": "Vimeo",
                        "api_name": "SearchVideos",
                        "api_description": "duplicate should dedupe",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                    {
                        "category_name": "Tools",
                        "tool_name": "YTStream - Download YouTube Videos",
                        "api_name": "Download/Stream",
                        "api_description": "Stream or download info.",
                        "required_parameters": [{"name": "id", "type": "STRING", "description": "video id"}],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                ],
            }
        ],
    )

    output_root = tmp_path / "StableToolBench/toolenv/tools"
    report = build_stabletoolbench_toolenv_from_queries(query_file=query_file, output_root=output_root)

    assert report["status"] == "ok"
    assert report["tool_count"] == 2
    assert report["api_count"] == 2
    assert report["written_tool_files"] == 2
    vimeo = read_json(output_root / "Media/vimeo.json")
    assert vimeo["tool_name"] == "Vimeo"
    assert vimeo["tool_description"] == "Search for videos."
    assert [api["name"] for api in vimeo["api_list"]] == ["SearchVideos"]
    assert vimeo["api_list"][0]["required_parameters"][0]["name"] == "query"
    yt = read_json(output_root / "Tools/ytstream_download_youtube_videos.json")
    assert yt["api_list"][0]["name"] == "Download/Stream"


def test_build_stabletoolbench_toolenv_rejects_example_sources(tmp_path):
    query_file = tmp_path / "StableToolBench/solvable_queries_example/test_instruction/G3_instruction.json"
    write_json(query_file, [{"query": "example", "api_list": []}])

    report = build_stabletoolbench_toolenv_from_queries(
        query_file=query_file,
        output_root=tmp_path / "StableToolBench/toolenv/tools",
    )

    assert report["status"] == "action_required"
    assert "query_file_is_example_not_official" in report["blockers"]
    assert report["written_tool_files"] == 0


def test_build_stabletoolbench_toolenv_cli_writes_manifest(tmp_path):
    query_file = tmp_path / "StableToolBench/solvable_queries/test_instruction/G3_instruction.json"
    output_root = tmp_path / "StableToolBench/toolenv/tools"
    manifest = tmp_path / "manifest.json"
    write_json(
        query_file,
        [
            {
                "query_id": 1,
                "query": "shorten url",
                "api_list": [
                    {
                        "category_name": "Tools",
                        "tool_name": "bitly",
                        "api_name": "shorten",
                        "api_description": "Shorten URL.",
                        "required_parameters": [],
                        "optional_parameters": [],
                    }
                ],
            }
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_stabletoolbench_toolenv.py",
            "--query_file",
            str(query_file),
            "--output_root",
            str(output_root),
            "--manifest_path",
            str(manifest),
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    assert report == read_json(manifest)
    assert (output_root / "Tools/bitly.json").is_file()
