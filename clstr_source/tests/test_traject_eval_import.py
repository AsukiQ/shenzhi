from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clstr.traject_eval_audit import audit_traject_eval_data
from clstr.traject_eval_import import (
    import_traject_eval,
    load_trajectbench_route_corpus,
)
from clstr.traject_split import assign_traject_split


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_tiny_traject_public_data(public_data: Path) -> None:
    _write_json(
        public_data / "tools/all_tools.json",
        [
            {
                "parent tool name": "Weather API",
                "parent tool description": "Weather API parent.",
                "API name": "Forecast",
                "tool name": "Weather API: Forecast",
                "tool description": "Gets a forecast.",
                "domain name": "Weather",
                "required_parameters": [{"name": "city", "type": "STRING"}],
                "optional_parameters": [{"name": "units", "type": "STRING"}],
                "output_info": {"summary": "forecast"},
                "code": "forecast(city)",
            },
            {
                "parent tool name": "Map API",
                "parent tool description": "Map API parent.",
                "API name": "Route",
                "tool name": "Map API: Route",
                "tool description": "Builds a route.",
                "domain name": "Mapping",
                "required_parameters": [{"name": "destination", "type": "STRING"}],
                "optional_parameters": [],
                "output_info": {"summary": "route"},
                "code": "route(destination)",
            },
        ],
    )
    _write_json(
        public_data / "sequential/Travel/traj_query.json",
        [
            {
                "query": "Find the weather in Paris, then route to the museum.",
                "task_name": "travel planning",
                "task_description": "Sequential weather and map task.",
                "tool list": [
                    {
                        "tool name": "Weather API: Forecast",
                        "tool description": "Gets a forecast.",
                        "required parameters": [{"name": "city", "value": "Paris"}],
                        "optional parameters": [{"name": "units", "value": "metric"}],
                        "executed_output": "{'temperature': 20}",
                    },
                    {
                        "tool name": "Map API: Route",
                        "tool description": "Builds a route.",
                        "required parameters": [{"name": "destination", "value": "museum"}],
                        "optional parameters": [],
                        "executed_output": "{'distance': '2km'}",
                    },
                ],
            }
        ],
    )


def test_import_traject_eval_normalizes_public_data_queries_skills_and_qrels(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_tiny_traject_public_data(public_data)

    manifest = import_traject_eval(public_data=public_data, output_dir=output_dir, split="test")

    assert manifest["status"] == "ok"
    assert manifest["query_count"] == 2
    assert manifest["skill_count"] == 2
    assert manifest["qrel_count"] == 2
    assert manifest["missing_qrel_skill_ids"] == []
    queries = _read_jsonl(output_dir / "queries.jsonl")
    assert queries[0]["query_id"] == "traject::sequential::Travel::traj_query::0::0"
    assert queries[0]["query_text"].startswith("goal: Find the weather in Paris")
    assert "previous_tools: <empty>" in queries[0]["query_text"]
    assert queries[1]["query_id"] == "traject::sequential::Travel::traj_query::0::1"
    assert "previous_tools: Weather API: Forecast" in queries[1]["query_text"]
    assert queries[1]["trajectory_type"] == "sequential"
    assert queries[1]["domain"] == "Travel"
    qrels = _read_jsonl(output_dir / "qrels.jsonl")
    assert [
        {key: row[key] for key in ("query_id", "skill_id", "relevance", "task", "source", "split", "trajectory_type", "step_index")}
        for row in qrels
    ] == [
        {
            "query_id": "traject::sequential::Travel::traj_query::0::0",
            "skill_id": "traject/weather-api/forecast",
            "relevance": 1,
            "task": "Travel",
            "source": "TRAJECT-Bench",
            "split": "test",
            "trajectory_type": "sequential",
            "step_index": 0,
        },
        {
            "query_id": "traject::sequential::Travel::traj_query::0::1",
            "skill_id": "traject/map-api/route",
            "relevance": 1,
            "task": "Travel",
            "source": "TRAJECT-Bench",
            "split": "test",
            "trajectory_type": "sequential",
            "step_index": 1,
        },
    ]
    assert {row["assigned_traject_split"] for row in qrels} == {
        assign_traject_split("traject::sequential::Travel::traj_query::0")
    }
    assert {row["split_policy"] for row in qrels} == {"deterministic_hash_trajectory_level_v1"}
    skills = {row["skill_id"]: row for row in _read_jsonl(output_dir / "skills.jsonl")}
    assert skills["traject/weather-api/forecast"]["name"] == "Weather API: Forecast"
    assert skills["traject/weather-api/forecast"]["description"] == "Gets a forecast."
    assert skills["traject/weather-api/forecast"]["input_schema"]["required_parameters"] == [
        {"name": "city", "type": "STRING"}
    ]
    assert skills["traject/map-api/route"]["body"] == "route(destination)"


def test_import_traject_eval_maps_case_variant_tool_names_to_all_tools_canonical_id(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_json(
        public_data / "tools/all_tools.json",
        [
            {
                "parent tool name": "Lost Ark",
                "API name": "Get all island with dropped items",
                "tool name": "Lost Ark: Get all island with dropped items",
                "tool description": "Lists Lost Ark islands and dropped items.",
            }
        ],
    )
    _write_json(
        public_data / "parallel/Gaming/simple_ver.json",
        [
            {
                "query": "Find islands with dropped items.",
                "tool list": [
                    {
                        "tool name": "Lost Ark: get all island with dropped items",
                        "tool description": "Lists Lost Ark islands and dropped items.",
                    }
                ],
            }
        ],
    )

    manifest = import_traject_eval(public_data=public_data, output_dir=output_dir)

    assert manifest["missing_qrel_skill_ids"] == []
    assert _read_jsonl(output_dir / "qrels.jsonl")[0]["skill_id"] == (
        "traject/lost-ark/get-all-island-with-dropped-items"
    )
    skills = {row["skill_id"]: row for row in _read_jsonl(output_dir / "skills.jsonl")}
    assert "traject/lost-ark/get-all-island-with-dropped-items" in skills


def test_import_traject_eval_adds_query_only_tools_to_eval_skill_pool(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_json(public_data / "tools/all_tools.json", [])
    _write_json(
        public_data / "parallel/Travel/simple_ver.json",
        [
            {
                "query": "Count Airbnb listings near a coordinate.",
                "tool list": [
                    {
                        "tool name": "Airbnb listings: Count listings by lat lng",
                        "tool description": "Counts Airbnb listings near a latitude/longitude.",
                        "required parameters": [{"name": "lat", "value": "37.6"}],
                        "optional parameters": [{"name": "lng", "value": "-122.3"}],
                    }
                ],
            }
        ],
    )

    manifest = import_traject_eval(public_data=public_data, output_dir=output_dir)

    query_only_sid = "traject/airbnb-listings-count-listings-by-lat-lng"
    assert manifest["missing_qrel_skill_ids"] == []
    assert manifest["skill_count"] == 1
    assert _read_jsonl(output_dir / "qrels.jsonl")[0]["skill_id"] == query_only_sid
    skills = {row["skill_id"]: row for row in _read_jsonl(output_dir / "skills.jsonl")}
    assert skills[query_only_sid]["name"] == "Airbnb listings: Count listings by lat lng"
    assert skills[query_only_sid]["provenance"]["dedup_method"] == "traject_query_tool_identity_v2"


def test_import_traject_eval_does_not_collapse_unknown_parent_api_tools(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_json(public_data / "tools/all_tools.json", [])
    _write_json(
        public_data / "sequential/Education/traj_query.json",
        [
            {
                "query": "Use two education helper tools.",
                "tool list": [
                    {
                        "tool name": "data_visualisation_: getting data",
                        "parent tool name": "unknown",
                        "API name": "unknown",
                        "tool description": "Gets source data for a visualization.",
                    },
                    {
                        "tool name": "Question generator: Blank fields",
                        "parent tool name": "unknown",
                        "API name": "unknown",
                        "tool description": "Generates fill-in-the-blank questions.",
                    },
                ],
            }
        ],
    )

    manifest = import_traject_eval(public_data=public_data, output_dir=output_dir)

    assert manifest["missing_qrel_skill_ids"] == []
    assert manifest["skill_count"] == 2
    qrel_ids = [row["skill_id"] for row in _read_jsonl(output_dir / "qrels.jsonl")]
    assert qrel_ids == [
        "traject/data-visualisation-getting-data",
        "traject/question-generator-blank-fields",
    ]
    assert len(set(qrel_ids)) == 2


def test_import_traject_eval_cli_runs_from_repo_root(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_tiny_traject_public_data(public_data)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/import_traject_eval.py",
            "--public_data",
            str(public_data),
            "--output_dir",
            str(output_dir),
            "--split",
            "dev",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    manifest = json.loads(proc.stdout)
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["split"] == "dev"
    assert (output_dir / "manifest.json").exists()
    assert _read_jsonl(output_dir / "qrels.jsonl")[0]["split"] == "dev"


def test_import_traject_eval_can_filter_deterministic_heldout_split(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    output_dir = tmp_path / "data/traject_eval"
    _write_json(
        public_data / "tools/all_tools.json",
        [
            {
                "parent tool name": "Weather API",
                "API name": "Forecast",
                "tool name": "Weather API: Forecast",
                "tool description": "Gets a forecast.",
            }
        ],
    )
    rows = [
        {
            "query": f"Weather query {idx}",
            "tool list": [{"tool name": "Weather API: Forecast"}],
        }
        for idx in range(80)
    ]
    _write_json(public_data / "sequential/Travel/traj_query.json", rows)
    expected_test_query_ids = {
        f"traject::sequential::Travel::traj_query::{idx}::0"
        for idx in range(len(rows))
        if assign_traject_split(f"traject::sequential::Travel::traj_query::{idx}") == "test"
    }

    manifest = import_traject_eval(
        public_data=public_data,
        output_dir=output_dir,
        split="test",
        split_partition="test",
    )

    queries = _read_jsonl(output_dir / "queries.jsonl")
    qrels = _read_jsonl(output_dir / "qrels.jsonl")
    assert manifest["filters"]["split_partition"] == "test"
    assert expected_test_query_ids
    assert {row["query_id"] for row in queries} == expected_test_query_ids
    assert {row["assigned_traject_split"] for row in queries} == {"test"}
    assert {row["assigned_traject_split"] for row in qrels} == {"test"}


def test_load_trajectbench_route_corpus_uses_heldout_global_visible_pool(tmp_path):
    public_data = tmp_path / "TRAJECT-Bench/public_data"
    _write_json(
        public_data / "tools/all_tools.json",
        [
            {
                "parent tool name": "Weather API",
                "API name": "Forecast",
                "tool name": "Weather API: Forecast",
                "tool description": "Gets a forecast.",
            },
            {
                "parent tool name": "Map API",
                "API name": "Route",
                "tool name": "Map API: Route",
                "tool description": "Builds a route.",
            },
        ],
    )
    test_index = next(
        index
        for index in range(200)
        if assign_traject_split(
            f"traject::parallel::Travel::traj_query::{index}"
        )
        == "test"
    )
    payload = [{} for _ in range(test_index + 1)]
    payload[test_index] = {
        "query": "Check weather and make a route in either order.",
        "task_name": "parallel travel",
        "task_description": "Both tools are required without a causal order.",
        "tool list": [
            {
                "tool name": "Weather API: Forecast",
                "required parameters": [{"name": "city", "value": "Paris"}],
                "executed_output": {"temperature": 20},
            },
            {
                "tool name": "Map API: Route",
                "required parameters": [{"name": "destination", "value": "museum"}],
                "executed_output": {"distance": "2km"},
            },
        ],
    }
    _write_json(public_data / "parallel/Travel/traj_query.json", payload)

    corpus = load_trajectbench_route_corpus(
        public_data,
        split_partition="test",
    )

    assert corpus.report["split_partition"] == "test"
    assert corpus.report["candidate_source"] == "global_public_traject_inventory"
    assert corpus.report["trajectory_count"] == 1
    assert corpus.report["multi_positive_row_count"] == 1
    assert len(corpus.source_rows) == 2
    first, second = corpus.source_rows
    assert first["next_skill_id"] == "traject/weather-api/forecast"
    assert first["equivalent_next_skill_ids"] == ["traject/map-api/route"]
    assert second["equivalent_next_skill_ids"] == []
    assert first["candidate_next_skill_ids"] == second["candidate_next_skill_ids"]
    assert set(first["candidate_next_skill_ids"]) == {
        "traject/weather-api/forecast",
        "traject/map-api/route",
    }
    assert "previous_tools" not in first["state_text_current"]
    assert first["actual_result_executed"] is True
    assert first["actual_result_skill_id"] == first["next_skill_id"]
    assert json.loads(first["actual_result_text"]) == {"temperature": 20}
    assert first["provenance"]["parallel_positive_semantics"] == (
        "remaining_required_tool_set_v1"
    )


def test_audit_traject_eval_data_blocks_missing_positive_qrels(tmp_path):
    data_dir = tmp_path / "data/traject_eval"
    _write_jsonl(data_dir / "queries.jsonl", [{"query_id": "q1"}, {"query_id": "q2"}])
    _write_jsonl(
        data_dir / "skills.jsonl",
        [
            {"skill_id": "tool/a", "name": "a"},
            {"skill_id": "tool/a", "name": "duplicate a"},
        ],
    )
    _write_jsonl(
        data_dir / "qrels.jsonl",
        [
            {"query_id": "q1", "skill_id": "tool/a", "relevance": 1},
            {"query_id": "q2", "skill_id": "missing", "relevance": 1},
        ],
    )

    report = audit_traject_eval_data(data_dir=data_dir, output_path=data_dir / "audit.json")

    assert report["status"] == "action_required"
    assert report["query_count"] == 2
    assert report["skill_count"] == 2
    assert report["unique_skill_count"] == 1
    assert report["missing_positive_qrel_skill_ids"] == ["missing"]
    assert report["duplicate_skill_ids"] == ["tool/a"]
    assert json.loads((data_dir / "audit.json").read_text(encoding="utf-8")) == report


def test_audit_traject_eval_data_ok_when_all_positive_qrels_are_in_pool(tmp_path):
    data_dir = tmp_path / "data/traject_eval"
    _write_jsonl(data_dir / "queries.jsonl", [{"query_id": "q1"}])
    _write_jsonl(data_dir / "skills.jsonl", [{"skill_id": "tool/a", "name": "a"}])
    _write_jsonl(data_dir / "qrels.jsonl", [{"query_id": "q1", "skill_id": "tool/a", "relevance": 1}])

    report = audit_traject_eval_data(data_dir=data_dir)

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["positive_qrel_count"] == 1


def test_audit_traject_eval_data_reports_native_metric_readiness_boundaries(tmp_path):
    data_dir = tmp_path / "data/traject_eval"
    _write_jsonl(
        data_dir / "queries.jsonl",
        [
            {
                "query_id": "traj::0::0",
                "trajectory_id": "traj::0",
                "trajectory_type": "sequential",
                "step_index": 0,
            },
            {
                "query_id": "traj::0::1",
                "trajectory_id": "traj::0",
                "trajectory_type": "sequential",
                "step_index": 1,
            },
        ],
    )
    _write_jsonl(
        data_dir / "skills.jsonl",
        [
            {"skill_id": "tool/weather", "name": "Weather"},
            {"skill_id": "tool/map", "name": "Map"},
        ],
    )
    _write_jsonl(
        data_dir / "qrels.jsonl",
        [
            {"query_id": "traj::0::0", "skill_id": "tool/weather", "relevance": 1},
            {"query_id": "traj::0::1", "skill_id": "tool/map", "relevance": 1},
        ],
    )

    report = audit_traject_eval_data(data_dir=data_dir)

    readiness = report["paper_metric_readiness"]
    assert readiness["routing_eval"]["status"] == "ready"
    assert readiness["trajectory_sequence_proxy"]["status"] == "ready"
    assert readiness["official_em_inclusion"]["status"] == "partial_proxy_ready"
    assert readiness["official_usage"]["status"] == "blocked"
    assert readiness["official_traj_satisfy"]["status"] == "blocked"
    assert readiness["official_acc"]["status"] == "blocked"
    assert report["trajectory_reconstruction"]["trajectory_count"] == 1
    assert report["trajectory_reconstruction"]["trajectory_type_counts"] == {"sequential": 1}
