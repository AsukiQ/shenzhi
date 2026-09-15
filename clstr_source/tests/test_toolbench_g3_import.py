from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from clstr.toolbench_g3_audit import audit_toolbench_g3_data


def _load_importer():
    path = Path("scripts/import_toolbench_g3.py")
    spec = importlib.util.spec_from_file_location("import_toolbench_g3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_downloader():
    path = Path("scripts/download_toolbench_data.py")
    spec = importlib.util.spec_from_file_location("download_toolbench_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_hf_g3_downloader():
    path = Path("scripts/download_toolbench_g3_from_hf.py")
    spec = importlib.util.spec_from_file_location("download_toolbench_g3_from_hf", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_import_toolbench_g3_normalizes_instruction_retrieval_and_tree_trajectory(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 7,
                "query": "Find cocktails and then search birthday news.",
                "relevant APIs": [["The Cocktail DB", "List of Cocktails"], ["Web Search", "newsSearch"]],
                "api_list": [
                    {
                        "category_name": "Food",
                        "tool_name": "The Cocktail DB",
                        "api_name": "List of Cocktails",
                        "api_description": "Returns a list of cocktails.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                    {
                        "category_name": "Data",
                        "tool_name": "Web Search",
                        "api_name": "newsSearch",
                        "api_description": "Gets news articles for a query.",
                        "required_parameters": [{"name": "q", "type": "STRING"}],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "list_of_cocktails_for_the_cocktail_db",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\": \"martini\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "newssearch_for_web_search",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{\"q\":\"birthday\"}",
                                                    "observation": "{\"response\": \"birthday news\"}",
                                                    "children": [],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    out_dir = tmp_path / "data/toolbench_g3"
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").read_text().splitlines()]
    trajectories = [json.loads(line) for line in (out_dir / "trajectories.jsonl").read_text().splitlines()]
    skills = [json.loads(line) for line in (out_dir / "skills.jsonl").read_text().splitlines()]

    assert manifest["status"] == "ok"
    assert manifest["queries"] == 1
    assert manifest["retrieval_pairs"] == 2
    assert manifest["trajectory_rows"] == 2
    assert manifest["skills"] == 2
    assert retrieval[0]["query_id"] == "toolbench-g3-7"
    assert retrieval[0]["positive_skill_id"] == "toolbench-g3/the-cocktail-db/list-of-cocktails"
    assert retrieval[0]["negative_skill_ids"] == ["toolbench-g3/web-search/newssearch"]
    assert trajectories[0]["skill_id"] == "toolbench-g3/the-cocktail-db/list-of-cocktails"
    assert trajectories[0]["next_skill_id"] == "toolbench-g3/web-search/newssearch"
    assert trajectories[0]["next_observation_text"] == "{\"response\": \"martini\"}"
    assert trajectories[1]["done"] is True
    assert "previous_tools:" not in trajectories[1]["state_text"]
    assert trajectories[1]["state_text_current"] == trajectories[1]["state_text"]
    assert "previous_tools: list_of_cocktails_for_the_cocktail_db" in trajectories[1]["state_text_full"]
    skills_by_id = {row["skill_id"]: row for row in skills}
    assert skills_by_id["toolbench-g3/web-search/newssearch"]["description"] == "Gets news articles for a query."


def test_import_toolbench_g3_treats_finish_as_terminal_not_unknown_skill(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 1000,
                "query": "Calculate carbon footprint.",
                "relevant APIs": [["CarbonFootprint", "CleanHydroToCarbonFootprint"]],
                "api_list": [
                    {
                        "category_name": "Environment",
                        "tool_name": "CarbonFootprint",
                        "api_name": "CleanHydroToCarbonFootprint",
                        "api_description": "Converts hydro energy use into carbon footprint.",
                        "required_parameters": [{"name": "energy", "type": "STRING"}],
                        "optional_parameters": [],
                        "method": "GET",
                    }
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/1000_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "cleanhydrotocarbonfootprint_for_carbonfootprint",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{\"energy\":\"HydroElectric\"}",
                                    "observation": "{\"response\":\"ok\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "Finish",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{\"return_type\":\"give_answer\"}",
                                                    "observation": "{\"response\":\"success\"}",
                                                    "children": [],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text().splitlines()
    ]

    assert manifest["trajectory_rows"] == 1
    assert trajectories[0]["skill_id"] == "toolbench-g3/carbonfootprint/cleanhydrotocarbonfootprint"
    assert trajectories[0]["next_skill_id"] == ""
    assert trajectories[0]["done"] is True
    assert "toolbench-g3/unknown/finish" not in json.dumps(trajectories)


def test_import_toolbench_g3_resolves_common_action_alias_variants(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 1012,
                "query": "Find coupons and then check a polygon balance.",
                "relevant APIs": [
                    ["27coupons", "Latest Coupons"],
                    ["Cryptocurrency balance", "Get Polygon Balance From Specific Network"],
                ],
                "api_list": [
                    {
                        "category_name": "Shopping",
                        "tool_name": "27coupons",
                        "api_name": "Latest Coupons",
                        "api_description": "Fetches latest coupons.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                    {
                        "category_name": "Finance",
                        "tool_name": "Cryptocurrency balance",
                        "api_name": "Get Polygon Balance From Specific Network",
                        "api_description": "Gets a Polygon balance.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/1012_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "latest_coupons_for_get_27coupons",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"coupon\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "polygon_balance_from_specific_network_for_cryptocurrency_balance",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{}",
                                                    "observation": "{\"response\":\"balance\"}",
                                                    "children": [],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text().splitlines()
    ]

    assert manifest["trajectory_rows"] == 2
    assert manifest["unresolved_action_count"] == 0
    assert trajectories[0]["skill_id"] == "toolbench-g3/27coupons/latest-coupons"
    assert trajectories[0]["next_skill_id"] == "toolbench-g3/cryptocurrency-balance/get-polygon-balance-from-specific-network"
    assert trajectories[1]["skill_id"] == "toolbench-g3/cryptocurrency-balance/get-polygon-balance-from-specific-network"


def test_import_toolbench_g3_prefers_successful_finish_branch_over_first_failed_branch(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 73,
                "query": "Verify an email.",
                "relevant APIs": [["Blaze Verify", "Heartbeat"], ["Blaze Verify", "Verify an email"]],
                "api_list": [
                    {
                        "category_name": "Email",
                        "tool_name": "Blaze Verify",
                        "api_name": "Heartbeat",
                        "api_description": "Checks service health.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                    {
                        "category_name": "Email",
                        "tool_name": "Blaze Verify",
                        "api_name": "Verify an email",
                        "api_description": "Verifies an email address.",
                        "required_parameters": [{"name": "email", "type": "STRING"}],
                        "optional_parameters": [],
                        "method": "GET",
                    },
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/73_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "not_in_api_list",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"dead branch\"}",
                                    "children": [],
                                }
                            ],
                        },
                        {
                            "node_type": "Action",
                            "description": "heartbeat_for_blaze_verify",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"ok\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "verify_an_email_for_blaze_verify",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{\"email\":\"a@example.com\"}",
                                                    "observation": "{\"response\":\"valid\"}",
                                                    "children": [
                                                        {
                                                            "node_type": "Action",
                                                            "description": "Finish",
                                                            "children": [
                                                                {
                                                                    "node_type": "Action Input",
                                                                    "description": "{\"return_type\":\"give_answer\"}",
                                                                    "observation": "{\"response\":\"success\"}",
                                                                    "children": [],
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text().splitlines()
    ]

    assert manifest["trajectory_rows"] == 2
    assert manifest["skipped_unresolved_answer_files"] == 0
    assert manifest["unresolved_action_count"] == 0
    assert [row["skill_id"] for row in trajectories] == [
        "toolbench-g3/blaze-verify/heartbeat",
        "toolbench-g3/blaze-verify/verify-an-email",
    ]
    assert trajectories[0]["next_skill_id"] == "toolbench-g3/blaze-verify/verify-an-email"
    assert trajectories[1]["done"] is True


def test_import_toolbench_g3_prefers_resolved_success_branch_over_unresolved_success_branch(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 74,
                "query": "Verify an email.",
                "relevant APIs": [["Blaze Verify", "Heartbeat"]],
                "api_list": [
                    {
                        "category_name": "Email",
                        "tool_name": "Blaze Verify",
                        "api_name": "Heartbeat",
                        "api_description": "Checks service health.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    }
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/74_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "unknown_api",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"unknown\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "another_unknown_api",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{}",
                                                    "observation": "{\"response\":\"unknown\"}",
                                                    "children": [
                                                        {
                                                            "node_type": "Action",
                                                            "description": "Finish",
                                                            "children": [
                                                                {
                                                                    "node_type": "Action Input",
                                                                    "description": "{\"return_type\":\"give_answer\"}",
                                                                    "observation": "{\"response\":\"success\"}",
                                                                    "children": [],
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "node_type": "Action",
                            "description": "heartbeat_for_blaze_verify",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"ok\"}",
                                    "children": [
                                        {
                                            "node_type": "Action",
                                            "description": "Finish",
                                            "children": [
                                                {
                                                    "node_type": "Action Input",
                                                    "description": "{\"return_type\":\"give_answer\"}",
                                                    "observation": "{\"response\":\"success\"}",
                                                    "children": [],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text().splitlines()
    ]

    assert manifest["trajectory_rows"] == 1
    assert manifest["skipped_unresolved_answer_files"] == 0
    assert manifest["unresolved_action_count"] == 0
    assert trajectories[0]["skill_id"] == "toolbench-g3/blaze-verify/heartbeat"
    assert trajectories[0]["done"] is True


def test_import_toolbench_g3_skips_successful_tree_with_unresolved_actions(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 42,
                "query": "Use a known API.",
                "relevant APIs": [["Weather", "Forecast"]],
                "api_list": [
                    {
                        "category_name": "Weather",
                        "tool_name": "Weather",
                        "api_name": "Forecast",
                        "api_description": "Gets a forecast.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    }
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/42_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": True,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "not_in_api_list",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"bad\"}",
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    trajectories_text = (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text(encoding="utf-8")

    assert manifest["successful_answer_files"] == 1
    assert manifest["skipped_unresolved_answer_files"] == 1
    assert manifest["unresolved_action_count"] == 1
    assert manifest["unresolved_action_samples"] == [{"query_id": "42", "action_name": "not_in_api_list"}]
    assert manifest["trajectory_rows"] == 0
    assert trajectories_text == ""


def test_import_toolbench_g3_skips_failed_answer_trees_for_supervised_trajectories(tmp_path):
    source_root = tmp_path / "ToolBench/data"
    _write_json(
        source_root / "instruction/G3_query.json",
        [
            {
                "query_id": 1006,
                "query": "Find coupons.",
                "relevant APIs": [["27coupons", "Latest Coupons"]],
                "api_list": [
                    {
                        "category_name": "Shopping",
                        "tool_name": "27coupons",
                        "api_name": "Latest Coupons",
                        "api_description": "Fetches latest coupons.",
                        "required_parameters": [],
                        "optional_parameters": [],
                        "method": "GET",
                    }
                ],
            }
        ],
    )
    _write_json(
        source_root / "answer/G3_answer/1006_ChatGPT_DFS_woFilter_w2.json",
        {
            "win": False,
            "tree": {
                "tree": {
                    "node_type": "Action Input",
                    "description": "",
                    "children": [
                        {
                            "node_type": "Action",
                            "description": "latest_coupons_for_get_27coupons",
                            "children": [
                                {
                                    "node_type": "Action Input",
                                    "description": "{}",
                                    "observation": "{\"response\":\"bad\"}",
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    importer = _load_importer()
    manifest = importer.import_toolbench_g3(source_root, tmp_path / "data/toolbench_g3")
    retrieval = [
        json.loads(line)
        for line in (tmp_path / "data/toolbench_g3/retrieval.jsonl").read_text().splitlines()
    ]
    trajectories_text = (tmp_path / "data/toolbench_g3/trajectories.jsonl").read_text(encoding="utf-8")

    assert manifest["retrieval_pairs"] == 1
    assert manifest["answer_files"] == 1
    assert manifest["successful_answer_files"] == 0
    assert manifest["failed_answer_files"] == 1
    assert manifest["skipped_failed_answer_files"] == 1
    assert manifest["trajectory_rows"] == 0
    assert trajectories_text == ""
    assert retrieval[0]["positive_skill_id"] == "toolbench-g3/27coupons/latest-coupons"


def test_toolbench_g3_import_sbatch_entrypoint_uses_local_source_path():
    script = Path("scripts/sbatch/run_import_toolbench_g3.sh").read_text(encoding="utf-8")

    assert "scripts/download_toolbench_data.py" in script
    assert "--verify_only" in script
    assert "RAW_VERIFY_EXTRACT_ROOT" in script
    assert 'if [[ -f "${SOURCE_ROOT}/instruction/G3_query.json" ]]' in script
    assert "EXPECTED_G3_ANSWER_FILES" in script
    assert "--expected_answer_files" in script
    assert script.count("--expected_answer_files") == 2
    assert "scripts/import_toolbench_g3.py" in script
    assert "scripts/audit_toolbench_g3_data.py" in script
    assert "SOURCE_ROOT is required" in script
    assert "data/toolbench_g3" in script
    assert "outputs/toolbench_g3/toolbench_g3_audit.json" in script
    assert "--source_root" in script
    assert "--fail_on_action_required" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_toolbench_g3_hf_downloader_defaults_to_hf_mirror_and_autodl_tmp_paths():
    script = Path("scripts/download_toolbench_g3_from_hf.py").read_text(encoding="utf-8")

    assert 'SOURCE_DATASET = "Adorg/ToolBench"' in script
    assert 'DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"' in script
    assert "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data" in script
    assert "/data/home/scyb713/run/xzf/AAAI/autodl-tmp" in script
    assert "--allowed_root" in script
    assert "--api_manifest_path" in script


def test_audit_toolbench_g3_data_ok_for_complete_normalized_export(tmp_path):
    data_dir = tmp_path / "data/toolbench_g3"
    _write_jsonl(
        data_dir / "skills.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/forecast", "name": "forecast"},
            {"skill_id": "toolbench-g3/maps/route", "name": "route"},
        ],
    )
    _write_jsonl(
        data_dir / "retrieval.jsonl",
        [
            {
                "query_id": "toolbench-g3-1",
                "query_text": "q",
                "positive_skill_id": "toolbench-g3/weather/forecast",
                "negative_skill_ids": ["toolbench-g3/maps/route"],
            }
        ],
    )
    _write_jsonl(
        data_dir / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "state_text": "state",
                "expert_action": "call",
                "skill_id": "toolbench-g3/weather/forecast",
                "next_skill_id": "toolbench-g3/maps/route",
                "provenance": {"split": "train_or_released_g3"},
            }
        ],
    )

    report = audit_toolbench_g3_data(data_dir=data_dir, output_path=data_dir / "audit.json")

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["skill_count"] == 2
    assert report["retrieval_pair_count"] == 1
    assert report["trajectory_row_count"] == 1
    assert json.loads((data_dir / "audit.json").read_text(encoding="utf-8")) == report


def test_audit_toolbench_g3_data_rejects_incomplete_raw_answer_count(tmp_path):
    data_dir = tmp_path / "data/toolbench_g3"
    source_root = tmp_path / "ToolBench/data"
    _write_json(source_root / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(source_root / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})
    _write_jsonl(
        data_dir / "skills.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/forecast", "name": "forecast"},
            {"skill_id": "toolbench-g3/maps/route", "name": "route"},
        ],
    )
    _write_jsonl(
        data_dir / "retrieval.jsonl",
        [
            {
                "query_id": "toolbench-g3-1",
                "query_text": "q",
                "positive_skill_id": "toolbench-g3/weather/forecast",
                "negative_skill_ids": ["toolbench-g3/maps/route"],
            }
        ],
    )
    _write_jsonl(
        data_dir / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "state_text": "state",
                "expert_action": "call",
                "skill_id": "toolbench-g3/weather/forecast",
                "next_skill_id": "toolbench-g3/maps/route",
            }
        ],
    )

    report = audit_toolbench_g3_data(
        data_dir=data_dir,
        source_root=source_root,
        expected_answer_files=2,
    )

    assert report["status"] == "action_required"
    assert "source_root_incomplete_g3_answer_files" in report["blockers"]
    assert report["official_source"]["status"] == "incomplete_answer_files"
    assert report["official_source"]["expected_g3_answer_files"] == 2
    assert report["official_source"]["g3_answer_files"] == 1
    assert report["may_use_for_paper_main_table"] is False


def test_audit_toolbench_g3_data_rejects_missing_normalized_and_data_example(tmp_path):
    source_root = tmp_path / "ToolBench/data_example"
    _write_json(source_root / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(source_root / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})

    report = audit_toolbench_g3_data(
        data_dir=tmp_path / "data/toolbench_g3",
        source_root=source_root,
    )

    assert report["status"] == "action_required"
    assert "missing_normalized_toolbench_g3_files" in report["blockers"]
    assert "source_root_is_data_example_not_official_g3" in report["blockers"]
    assert report["official_source"]["status"] == "data_example_not_allowed"
    assert report["may_use_for_paper_main_table"] is False


def test_audit_toolbench_g3_data_reports_missing_skill_references(tmp_path):
    data_dir = tmp_path / "data/toolbench_g3"
    _write_jsonl(data_dir / "skills.jsonl", [{"skill_id": "toolbench-g3/weather/forecast", "name": "forecast"}])
    _write_jsonl(
        data_dir / "retrieval.jsonl",
        [
            {
                "query_id": "toolbench-g3-1",
                "query_text": "q",
                "positive_skill_id": "missing",
                "negative_skill_ids": ["toolbench-g3/weather/forecast"],
            }
        ],
    )
    _write_jsonl(
        data_dir / "trajectories.jsonl",
        [
            {
                "benchmark": "toolbench_g3",
                "state_text": "state",
                "expert_action": "call",
                "skill_id": "toolbench-g3/weather/forecast",
                "next_skill_id": "missing-next",
            }
        ],
    )

    report = audit_toolbench_g3_data(data_dir=data_dir)

    assert report["status"] == "action_required"
    assert "missing_retrieval_positive_skill_ids" in report["blockers"]
    assert "missing_trajectory_skill_ids" in report["blockers"]
    assert report["missing_retrieval_positive_skill_ids"] == ["missing"]
    assert report["missing_trajectory_skill_ids"] == ["missing-next"]


def test_audit_toolbench_g3_cli_writes_report_and_fails_on_action_required(tmp_path):
    data_dir = tmp_path / "data/toolbench_g3"
    output_path = tmp_path / "outputs/toolbench_g3/audit.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_toolbench_g3_data.py",
            "--data_dir",
            str(data_dir),
            "--source_root",
            str(tmp_path / "ToolBench/data_example"),
            "--expected_answer_files",
            "5000",
            "--output_path",
            str(output_path),
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "action_required"
    assert "missing_normalized_toolbench_g3_files" in report["blockers"]
    assert json.loads(proc.stdout)["status"] == "action_required"


def test_download_toolbench_data_dry_run_records_official_sources_and_local_paths(tmp_path):
    downloader = _load_downloader()

    manifest = downloader.prepare_toolbench_data(
        zip_path=tmp_path / "ToolBench/data.zip",
        extract_root=tmp_path / "ToolBench",
        dry_run=True,
        source="tsinghua",
        allowed_root=tmp_path,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["source_dataset"] == "OpenBMB/ToolBench data.zip"
    assert manifest["zip_path"] == str(tmp_path / "ToolBench/data.zip")
    assert manifest["extract_root"] == str(tmp_path / "ToolBench")
    assert manifest["data_root"] == str(tmp_path / "ToolBench/data")
    assert manifest["official_sources"]["google_drive_file_id"] == "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
    assert "cloud.tsinghua.edu.cn" in manifest["official_sources"]["tsinghua_cloud_url"]
    assert manifest["download_source"] == "tsinghua"
    assert manifest["download_url"] == downloader.TSINGHUA_DOWNLOAD_URL
    assert manifest["manual_fallback"]["reason"] == "google_drive_or_tsinghua_cloud_unavailable"
    assert manifest["static_smoke_fallback"]["url"] == downloader.MODELSCOPE_TOOLBENCH_STATIC_URL
    assert manifest["static_smoke_fallback"]["paper_role"] == "smoke_only_not_official_g3"
    assert manifest["static_smoke_fallback"]["may_replace_official_toolbench_g3"] is False


def test_download_toolbench_g3_from_hf_dry_run_selects_only_g3_required_files(tmp_path):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json"},
                {"rfilename": "answer/G2_answer/2_ChatGPT_DFS_woFilter_w2.json"},
                {"rfilename": "toolenv/tools/Food/cocktails.json"},
            ]
        },
    )

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        dry_run=True,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["source_dataset"] == "Adorg/ToolBench"
    assert manifest["paper_role"] == "official_g3_mirror"
    assert manifest["output_root"] == str(tmp_path / "ToolBench/data")
    assert manifest["selected_file_count"] == 2
    assert manifest["selected_files"] == [
        "instruction/G3_query.json",
        "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json",
    ]
    assert manifest["downloaded_file_count"] == 0


def test_download_toolbench_g3_from_hf_downloads_selected_files_and_verifies(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )

    def fake_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_file == "instruction/G3_query.json":
            target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
        else:
            target_path.write_text(json.dumps({"win": True}), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", fake_download)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        manifest_path=tmp_path / "ToolBench/hf_g3_download_manifest.json",
    )

    assert manifest["status"] == "ok"
    assert manifest["downloaded_file_count"] == 2
    assert manifest["verification"]["status"] == "ok"
    assert manifest["verification"]["g3_answer_files"] == 1
    assert (tmp_path / "ToolBench/data/instruction/G3_query.json").exists()
    assert (tmp_path / "ToolBench/data/answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json").exists()


def test_download_toolbench_g3_from_hf_requires_all_selected_answer_files(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
                {"rfilename": "answer/G3_answer/8_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )

    def fake_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_file == "instruction/G3_query.json":
            target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
            return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}
        if remote_file.endswith("/7_ChatGPT_DFS_woFilter_w2.json"):
            target_path.write_text(json.dumps({"win": True}), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size if target_path.exists() else 0, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", fake_download)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
    )

    assert manifest["status"] == "incomplete_answer_files"
    assert manifest["verification"]["expected_g3_answer_files"] == 2
    assert manifest["verification"]["g3_answer_files"] == 1
    assert "answer/G3_answer/*.json: expected 2, found 1" in manifest["verification"]["missing"]


def test_download_toolbench_g3_from_hf_workers_download_all_selected_files(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
                {"rfilename": "answer/G3_answer/8_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )
    seen = []

    def fake_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        seen.append(remote_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_file == "instruction/G3_query.json":
            target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
        else:
            target_path.write_text(json.dumps({"win": True}), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", fake_download)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        workers=2,
    )

    assert manifest["status"] == "ok"
    assert manifest["workers"] == 2
    assert set(seen) == {
        "instruction/G3_query.json",
        "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json",
        "answer/G3_answer/8_ChatGPT_DFS_woFilter_w2.json",
    }
    assert manifest["downloaded_file_count"] == 3
    assert manifest["verification"]["g3_answer_files"] == 2


def test_download_toolbench_g3_from_hf_retries_transient_file_failures(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )
    attempts = {}

    def flaky_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        attempts[remote_file] = attempts.get(remote_file, 0) + 1
        if remote_file.startswith("answer/") and attempts[remote_file] == 1:
            raise RuntimeError("temporary mirror reset")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_file == "instruction/G3_query.json":
            target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
        else:
            target_path.write_text(json.dumps({"win": True}), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", flaky_download)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        max_retries=2,
    )

    assert manifest["status"] == "ok"
    assert attempts["answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"] == 2
    assert manifest["failed_download_count"] == 0
    answer_download = next(item for item in manifest["downloads"] if item["remote_file"].startswith("answer/"))
    assert answer_download["attempts"] == 2


def test_download_toolbench_g3_from_hf_backs_off_between_retry_attempts(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )
    attempts = {}
    sleeps = []

    def flaky_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        attempts[remote_file] = attempts.get(remote_file, 0) + 1
        if remote_file.startswith("answer/") and attempts[remote_file] < 3:
            raise RuntimeError("429 Too Many Requests")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if remote_file == "instruction/G3_query.json":
            target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
        else:
            target_path.write_text(json.dumps({"win": True}), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", flaky_download)
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        max_retries=2,
        retry_backoff_seconds=1.5,
        retry_backoff_multiplier=2.0,
    )

    assert manifest["status"] == "ok"
    assert attempts["answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"] == 3
    assert sleeps == [1.5, 3.0]
    assert manifest["retry_backoff_seconds"] == 1.5
    assert manifest["retry_backoff_multiplier"] == 2.0


def test_download_toolbench_g3_from_hf_records_persistent_file_failures_without_crashing(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    api_manifest = tmp_path / "adorg_toolbench_api.json"
    _write_json(
        api_manifest,
        {
            "siblings": [
                {"rfilename": "instruction/G3_query.json"},
                {"rfilename": "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
            ]
        },
    )

    def failing_download(remote_file, target_path, *, hf_endpoint, repo_id, timeout):
        if remote_file.startswith("answer/"):
            raise RuntimeError("mirror unavailable")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps([{"query_id": 7, "query": "q"}]), encoding="utf-8")
        return {"remote_file": remote_file, "bytes": target_path.stat().st_size, "skipped": False}

    monkeypatch.setattr(downloader, "_download_file", failing_download)

    manifest = downloader.download_toolbench_g3_from_hf(
        output_root=tmp_path / "ToolBench/data",
        api_manifest_path=api_manifest,
        max_retries=1,
        workers=2,
    )

    assert manifest["status"] == "download_failed"
    assert manifest["failed_download_count"] == 1
    assert manifest["failed_downloads"][0]["remote_file"] == "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"
    assert manifest["failed_downloads"][0]["attempts"] == 2
    assert "mirror unavailable" in manifest["failed_downloads"][0]["error"]
    assert manifest["verification"]["status"] == "missing_required_files"


def test_download_toolbench_g3_from_hf_tolerates_move_race_when_target_exists(tmp_path, monkeypatch):
    downloader = _load_hf_g3_downloader()
    target = tmp_path / "ToolBench/data/answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b'{"win": true}'

    def fake_get(url, stream, timeout):
        calls.append((url, stream, timeout))
        return FakeResponse()

    def fake_move(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes())
        Path(src).unlink()
        raise FileNotFoundError(src)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(downloader.shutil, "move", fake_move)

    result = downloader._download_file(
        "answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json",
        target,
        hf_endpoint="https://hf-mirror.com",
        repo_id="Adorg/ToolBench",
        timeout=10,
    )

    assert result["skipped"] is False
    assert result["bytes"] == target.stat().st_size
    assert target.read_text(encoding="utf-8") == '{"win": true}'
    assert calls


def test_verify_toolbench_data_rejects_code_only_or_data_example(tmp_path):
    downloader = _load_downloader()
    toolbench_root = tmp_path / "ToolBench"
    _write_json(toolbench_root / "data_example/instruction/G3_query.json", [])

    manifest = downloader.verify_toolbench_data(toolbench_root / "data_example")

    assert manifest["status"] == "missing_required_files"
    assert manifest["has_instruction_g3"] is True
    assert manifest["has_answer_g3"] is False
    assert "answer/G3_answer/*.json" in manifest["missing"]


def test_verify_toolbench_data_identifies_static_smoke_assets_without_accepting_as_g3(tmp_path):
    downloader = _load_downloader()
    static_root = tmp_path / "ToolBench/data"
    _write_json(static_root / "toolbench_static/in_domain.json", [{"tools": [], "query": "q"}])
    _write_json(static_root / "toolbench_static/out_of_domain.json", [{"tools": [], "query": "q"}])

    manifest = downloader.verify_toolbench_data(static_root)

    assert manifest["status"] == "missing_required_files"
    assert manifest["static_smoke_assets"]["available"] is True
    assert manifest["static_smoke_assets"]["paper_role"] == "smoke_only_not_official_g3"
    assert manifest["static_smoke_assets"]["may_replace_official_toolbench_g3"] is False
    assert "instruction/G3_query.json" in manifest["missing"]


def test_verify_toolbench_data_accepts_full_g3_assets(tmp_path):
    downloader = _load_downloader()
    data_root = tmp_path / "ToolBench/data"
    _write_json(data_root / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(data_root / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})

    manifest = downloader.verify_toolbench_data(data_root)

    assert manifest["status"] == "ok"
    assert manifest["has_instruction_g3"] is True
    assert manifest["has_answer_g3"] is True
    assert manifest["g3_answer_files"] == 1


def test_verify_toolbench_data_rejects_incomplete_answer_count_when_expected(tmp_path):
    downloader = _load_downloader()
    data_root = tmp_path / "ToolBench/data"
    _write_json(data_root / "instruction/G3_query.json", [{"query_id": 1, "query": "q"}])
    _write_json(data_root / "answer/G3_answer/1_ChatGPT_DFS_woFilter_w2.json", {"win": True})

    manifest = downloader.verify_toolbench_data(data_root, expected_answer_files=2)

    assert manifest["status"] == "incomplete_answer_files"
    assert manifest["expected_g3_answer_files"] == 2
    assert manifest["g3_answer_files"] == 1
    assert "answer/G3_answer/*.json: expected 2, found 1" in manifest["missing"]


def test_prepare_toolbench_data_rejects_zip_path_traversal(tmp_path):
    downloader = _load_downloader()
    zip_path = tmp_path / "ToolBench/data.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../outside.txt", "bad")

    with pytest.raises(ValueError, match="unsafe zip member"):
        downloader.prepare_toolbench_data(
            zip_path=zip_path,
            extract_root=tmp_path / "ToolBench",
            allowed_root=tmp_path,
        )


def test_prepare_toolbench_data_reports_invalid_zip_without_crashing(tmp_path):
    downloader = _load_downloader()
    zip_path = tmp_path / "ToolBench/data.zip"
    manifest_path = tmp_path / "ToolBench/download_manifest.json"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_text("<html>not a zip</html>", encoding="utf-8")

    manifest = downloader.prepare_toolbench_data(
        zip_path=zip_path,
        extract_root=tmp_path / "ToolBench",
        allowed_root=tmp_path,
        manifest_path=manifest_path,
    )

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "invalid_zip"
    assert written["status"] == "invalid_zip"
    assert "not a zip file" in manifest["error"].lower()


def test_download_helpers_write_to_part_file_before_final_path(tmp_path, monkeypatch):
    downloader = _load_downloader()
    target = tmp_path / "ToolBench/data.zip"
    calls = []

    def fake_stream(url, part_path):
        calls.append((url, part_path))
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"zip-bytes")
        return {"downloaded_bytes": 9, "download_url": url}

    monkeypatch.setattr(downloader, "_stream_url_to_file", fake_stream)

    manifest = downloader._download_plain_url(downloader.TSINGHUA_DOWNLOAD_URL, target)

    assert calls == [(downloader.TSINGHUA_DOWNLOAD_URL, target.with_suffix(".zip.part"))]
    assert target.read_bytes() == b"zip-bytes"
    assert not target.with_suffix(".zip.part").exists()
    assert manifest["downloaded_bytes"] == 9


def test_prepare_toolbench_data_moves_downloaded_invalid_zip_aside(tmp_path, monkeypatch):
    downloader = _load_downloader()
    zip_path = tmp_path / "ToolBench/data.zip"

    def fake_download(url, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<html>not a zip</html>", encoding="utf-8")
        return {"downloaded_bytes": output_path.stat().st_size, "download_url": url}

    monkeypatch.setattr(downloader, "_download_plain_url", fake_download)

    manifest = downloader.prepare_toolbench_data(
        zip_path=zip_path,
        extract_root=tmp_path / "ToolBench",
        allowed_root=tmp_path,
        download=True,
        source="tsinghua",
    )

    assert manifest["status"] == "invalid_zip"
    assert not zip_path.exists()
    assert Path(manifest["invalid_zip_path"]).is_file()
    assert Path(manifest["invalid_zip_path"]).name == "data.zip.invalid"
