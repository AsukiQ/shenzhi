import json
import subprocess
import sys
from pathlib import Path

from clstr.stabletoolbench_clstr_queries import (
    build_stabletoolbench_queries_from_clstr_run,
    export_stabletoolbench_retrieval_queries,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_stabletoolbench_retrieval_queries_preserves_official_query_ids(tmp_path):
    query_file = tmp_path / "StableToolBench/solvable_queries/test_instruction/G3_instruction.json"
    write_json(
        query_file,
        [
            {"query_id": 455, "query": "find related channels", "api_list": [{"tool_name": "Vimeo"}]},
            {"query_id": "456", "query": "search videos", "api_list": []},
        ],
    )
    output_path = tmp_path / "data/stabletoolbench_g3_queries.jsonl"

    report = export_stabletoolbench_retrieval_queries(query_file=query_file, output_path=output_path)

    assert report["status"] == "ok"
    assert report["query_count"] == 2
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {
                "query_id": "455",
                "query_text": "find related channels",
                "source": "stabletoolbench_g3",
                "split": "stabletoolbench_solvable_test",
            }
        ),
        json.dumps(
            {
                "query_id": "456",
                "query_text": "search videos",
                "source": "stabletoolbench_g3",
                "split": "stabletoolbench_solvable_test",
            }
        ),
    ]


def test_build_stabletoolbench_queries_from_clstr_run_replaces_api_list_with_topk_predictions(tmp_path):
    original_query_file = tmp_path / "StableToolBench/solvable_queries/test_instruction/G3_instruction.json"
    write_json(
        original_query_file,
        [
            {
                "query_id": 455,
                "query": "find videos",
                "relevant APIs": [["Vimeo", "SearchVideos"]],
                "api_list": [
                    {
                        "category_name": "Media",
                        "tool_name": "OfficialOnly",
                        "api_name": "NotPredicted",
                        "api_description": "Should be replaced.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    }
                ],
            },
            {
                "query_id": 456,
                "query": "missing predictions should stay in denominator",
                "api_list": [{"category_name": "Media", "tool_name": "Vimeo", "api_name": "SearchVideos"}],
            },
        ],
    )
    skills_path = tmp_path / "data/toolbench_g3/skills.jsonl"
    write_jsonl(
        skills_path,
        [
            {
                "skill_id": "toolbench-g3/vimeo/searchvideos",
                "name": "SearchVideos",
                "description": "Search for videos.",
                "environment": "Media",
                "input_schema": {
                    "required_parameters": [{"name": "query", "type": "STRING"}],
                    "optional_parameters": [{"name": "page", "type": "NUMBER"}],
                    "method": "GET",
                },
                "provenance": {"tool_name": "Vimeo", "api_name": "SearchVideos"},
            },
            {
                "skill_id": "toolbench-g3/ytstream/download-stream",
                "name": "Download/Stream",
                "description": "Download stream metadata.",
                "environment": "Tools",
                "input_schema": {"required_parameters": [], "optional_parameters": [], "method": "POST"},
                "output_schema": {"ok": True},
                "provenance": {"tool_name": "YTStream", "api_name": "Download/Stream"},
            },
        ],
    )
    run_path = tmp_path / "run.tsv"
    run_path.write_text(
        "\n".join(
            [
                "toolbench-g3-455\tQ0\ttoolbench-g3/missing/not-found\t1\t9.00000000\tclstr",
                "toolbench-g3-455\tQ0\ttoolbench-g3/vimeo/searchvideos\t2\t8.00000000\tclstr",
                "toolbench-g3-455\tQ0\ttoolbench-g3/ytstream/download-stream\t3\t7.00000000\tclstr",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_query_file = tmp_path / "outputs/toolbench_g3/stabletoolbench_clstr_topk/G3_instruction.json"
    report_path = tmp_path / "outputs/toolbench_g3/stabletoolbench_clstr_topk/report.json"

    report = build_stabletoolbench_queries_from_clstr_run(
        original_query_file=original_query_file,
        skills_path=skills_path,
        run_path=run_path,
        output_query_file=output_query_file,
        report_path=report_path,
        top_k=2,
    )

    rows = read_json(output_query_file)
    assert report["status"] == "action_required"
    assert report["query_count"] == 2
    assert report["retained_query_count"] == 1
    assert report["missing_run_query_count"] == 1
    assert report["empty_api_list_query_count"] == 1
    assert report["missing_skill_id_count"] == 1
    assert report == read_json(report_path)
    assert rows[0]["query_id"] == 455
    assert rows[0]["relevant APIs"] == [["Vimeo", "SearchVideos"]]
    assert [api["tool_name"] for api in rows[0]["api_list"]] == ["Vimeo"]
    assert rows[0]["api_list"][0]["api_name"] == "SearchVideos"
    assert rows[0]["api_list"][0]["api_description"] == "Search for videos."
    assert rows[0]["api_list"][0]["required_parameters"] == [{"name": "query", "type": "STRING"}]
    assert rows[1]["api_list"] == []


def test_build_stabletoolbench_queries_cli_writes_outputs(tmp_path):
    query_file = tmp_path / "StableToolBench/solvable_queries/test_instruction/G3_instruction.json"
    skills_path = tmp_path / "skills.jsonl"
    run_path = tmp_path / "run.tsv"
    output_query_file = tmp_path / "out/G3_instruction_clstr_top1.json"
    report_path = tmp_path / "out/report.json"
    write_json(query_file, [{"query_id": "1", "query": "shorten url", "api_list": []}])
    write_jsonl(
        skills_path,
        [
            {
                "skill_id": "toolbench-g3/bitly/shorten",
                "description": "Shorten URL.",
                "environment": "Tools",
                "input_schema": {},
                "provenance": {"tool_name": "bitly", "api_name": "shorten"},
            }
        ],
    )
    run_path.write_text("1\tQ0\ttoolbench-g3/bitly/shorten\t1\t1.0\tclstr\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_stabletoolbench_clstr_queries.py",
            "--original_query_file",
            str(query_file),
            "--skills_path",
            str(skills_path),
            "--run_path",
            str(run_path),
            "--output_query_file",
            str(output_query_file),
            "--report_path",
            str(report_path),
            "--top_k",
            "1",
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    assert read_json(output_query_file)[0]["api_list"][0]["tool_name"] == "bitly"
    assert report == read_json(report_path)
