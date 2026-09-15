import json
import subprocess
import sys
import types
from pathlib import Path

import clstr.toolret_eval_import as toolret_eval_import
from clstr.toolret_eval_import import import_toolret_eval
from clstr.toolret_eval_audit import audit_toolret_eval_data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_import_toolret_eval_normalizes_local_queries_tools_and_qrels(tmp_path):
    queries_path = tmp_path / "raw/queries.jsonl"
    tools_path = tmp_path / "raw/tools.jsonl"
    output_dir = tmp_path / "data/toolret_eval"
    write_jsonl(
        queries_path,
        [
            {
                "id": "apigen_query_5",
                "query": "Predict bacterial growth.",
                "instruction": "Given a bacterial population prediction task, retrieve growth tools.",
                "task": "apigen",
                "labels": json.dumps(
                    [
                        {
                            "id": "apigen_tool_272",
                            "relevance": 1,
                            "doc": {
                                "name": "bacterial_growth",
                                "description": "Calculates a bacterial population.",
                                "parameters": {"initial_population": {"type": "int"}},
                            },
                        }
                    ]
                ),
            }
        ],
    )
    write_jsonl(
        tools_path,
        [
            {
                "id": "apigen_tool_272",
                "documentation": "bacterial_growth(initial_population, growth_rate, time)",
                "doc": {
                    "name": "bacterial_growth",
                    "description": "Calculates a bacterial population.",
                    "parameters": {"initial_population": {"type": "int"}},
                },
                "category": "web",
            }
        ],
    )

    manifest = import_toolret_eval(
        query_source=queries_path,
        tool_source=tools_path,
        output_dir=output_dir,
    )

    assert manifest["status"] == "ok"
    assert manifest["query_count"] == 1
    assert manifest["skill_count"] == 1
    assert manifest["qrel_count"] == 1
    assert manifest["missing_qrel_skill_ids"] == []
    assert read_jsonl(output_dir / "queries.jsonl") == [
        {
            "query_id": "apigen_query_5",
            "query_text": "Given a bacterial population prediction task, retrieve growth tools.\n"
            "Task: Predict bacterial growth.",
            "query": "Predict bacterial growth.",
            "instruction": "Given a bacterial population prediction task, retrieve growth tools.",
            "task": "apigen",
            "source": "ToolRet-Queries",
            "split": "test",
        }
    ]
    assert read_jsonl(output_dir / "qrels.jsonl") == [
        {
            "query_id": "apigen_query_5",
            "skill_id": "apigen_tool_272",
            "relevance": 1,
            "task": "apigen",
            "source": "ToolRet",
            "split": "test",
        }
    ]
    assert read_jsonl(output_dir / "skills.jsonl") == [
        {
            "skill_id": "apigen_tool_272",
            "canonical_skill_id": "apigen_tool_272",
            "name": "bacterial_growth",
            "description": "Calculates a bacterial population.",
            "source": "ToolRet-Tools",
            "environment": "web",
            "executor_desc": "Calculates a bacterial population.",
            "input_schema": {"initial_population": {"type": "int"}},
            "output_schema": {},
            "body": "bacterial_growth(initial_population, growth_rate, time)",
            "provenance": {
                "source_dataset": "mangopy/ToolRet-Tools",
                "raw_tool_id": "apigen_tool_272",
            },
        }
    ]


def test_import_toolret_eval_reports_qrels_missing_from_tool_pool(tmp_path):
    queries_path = tmp_path / "raw/queries.jsonl"
    tools_path = tmp_path / "raw/tools.jsonl"
    output_dir = tmp_path / "data/toolret_eval"
    write_jsonl(
        queries_path,
        [
            {
                "id": "q1",
                "query": "Need tool A.",
                "task": "toolbench",
                "labels": [{"id": "missing_tool", "relevance": 2}],
            }
        ],
    )
    write_jsonl(tools_path, [{"id": "other_tool", "documentation": "Other tool."}])

    manifest = import_toolret_eval(
        query_source=queries_path,
        tool_source=tools_path,
        output_dir=output_dir,
    )

    assert manifest["qrel_count"] == 1
    assert manifest["missing_qrel_skill_ids"] == ["missing_tool"]
    assert read_jsonl(output_dir / "qrels.jsonl")[0]["relevance"] == 2


def test_import_toolret_eval_cli_uses_local_sources(tmp_path):
    queries_path = tmp_path / "raw/queries.jsonl"
    tools_path = tmp_path / "raw/tools.jsonl"
    output_dir = tmp_path / "data/toolret_eval"
    write_jsonl(
        queries_path,
        [{"id": "q1", "query": "Need weather.", "labels": [{"id": "weather_api", "relevance": 1}]}],
    )
    write_jsonl(
        tools_path,
        [{"id": "weather_api", "documentation": "Weather API", "doc": {"name": "weather_api"}}],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/import_toolret_eval.py",
            "--query_source",
            str(queries_path),
            "--tool_source",
            str(tools_path),
            "--output_dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    manifest = json.loads(proc.stdout)
    assert manifest["output_dir"] == str(output_dir)
    assert (output_dir / "manifest.json").exists()
    assert read_jsonl(output_dir / "qrels.jsonl")[0]["skill_id"] == "weather_api"


def test_toolret_eval_hf_defaults_expand_all_into_official_configs(monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_load_dataset(repo, config, split):
        calls.append((repo, config, split))
        return [{"id": f"{config}-row"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    query_rows = list(toolret_eval_import._iter_hf_queries(tasks=None))
    tool_rows = list(toolret_eval_import._iter_hf_tools(categories=None))

    assert query_rows[0]["id"].endswith("-row")
    assert tool_rows[0]["id"].endswith("-row")
    assert ("mangopy/ToolRet-Queries", "all", "queries") not in calls
    assert ("mangopy/ToolRet-Tools", "all", "tools") not in calls
    assert ("mangopy/ToolRet-Queries", "apigen", "queries") in calls
    assert ("mangopy/ToolRet-Tools", "web", "tools") in calls


def test_download_toolret_eval_dry_run_reports_official_configs_and_local_outputs(tmp_path):
    path = Path("scripts/download_toolret_eval_data.py")
    spec = __import__("importlib.util").util.spec_from_file_location("download_toolret_eval_data", path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest = module.download_toolret_eval_data(
        output_dir=tmp_path / "raw/toolret_eval",
        dry_run=True,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["source_dataset"]["queries"] == "mangopy/ToolRet-Queries"
    assert manifest["source_dataset"]["tools"] == "mangopy/ToolRet-Tools"
    assert "apigen" in manifest["query_configs"]
    assert manifest["tool_configs"] == ["code", "customized", "web"]
    assert manifest["files"]["queries"] == "queries.jsonl"
    assert manifest["files"]["tools"] == "tools.jsonl"
    assert manifest["hf_endpoint"] == "https://hf-mirror.com"


def test_audit_toolret_eval_data_blocks_missing_positive_qrels(tmp_path):
    data_dir = tmp_path / "data/toolret_eval"
    write_jsonl(data_dir / "queries.jsonl", [{"query_id": "q1"}, {"query_id": "q2"}])
    write_jsonl(
        data_dir / "skills.jsonl",
        [
            {"skill_id": "tool_a", "name": "tool_a"},
            {"skill_id": "tool_a", "name": "duplicate_tool_a"},
        ],
    )
    write_jsonl(
        data_dir / "qrels.jsonl",
        [
            {"query_id": "q1", "skill_id": "tool_a", "relevance": 1},
            {"query_id": "q2", "skill_id": "missing_tool", "relevance": 1},
            {"query_id": "q2", "skill_id": "negative_tool", "relevance": 0},
        ],
    )

    report = audit_toolret_eval_data(data_dir=data_dir, output_path=data_dir / "audit.json")

    assert report["status"] == "action_required"
    assert report["query_count"] == 2
    assert report["skill_count"] == 2
    assert report["unique_skill_count"] == 1
    assert report["positive_qrel_count"] == 2
    assert report["missing_positive_qrel_skill_ids"] == ["missing_tool"]
    assert report["duplicate_skill_ids"] == ["tool_a"]
    assert json.loads((data_dir / "audit.json").read_text(encoding="utf-8")) == report


def test_audit_toolret_eval_data_ok_when_all_positive_qrels_are_in_pool(tmp_path):
    data_dir = tmp_path / "data/toolret_eval"
    write_jsonl(data_dir / "queries.jsonl", [{"query_id": "q1"}])
    write_jsonl(data_dir / "skills.jsonl", [{"skill_id": "tool_a", "name": "tool_a"}])
    write_jsonl(data_dir / "qrels.jsonl", [{"query_id": "q1", "skill_id": "tool_a", "relevance": 1}])

    report = audit_toolret_eval_data(data_dir=data_dir)

    assert report["status"] == "ok"
    assert report["blockers"] == []


def test_audit_toolret_eval_cli_writes_report(tmp_path):
    data_dir = tmp_path / "data/toolret_eval"
    output_path = tmp_path / "audit/report.json"
    write_jsonl(data_dir / "queries.jsonl", [{"query_id": "q1"}])
    write_jsonl(data_dir / "skills.jsonl", [{"skill_id": "tool_a", "name": "tool_a"}])
    write_jsonl(data_dir / "qrels.jsonl", [{"query_id": "q1", "skill_id": "tool_a", "relevance": 1}])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_toolret_eval_data.py",
            "--data_dir",
            str(data_dir),
            "--output_path",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
