from __future__ import annotations

import json
from pathlib import Path

from clstr.function_schema_augmentation import build_function_schema_augmentation, merge_function_schema_augmentation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_function_schema_augmentation_converts_unitool_multistep_rows(tmp_path):
    unitool_root = tmp_path / "unitool"
    train_dir = unitool_root / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "train_converted_APIGen.json").write_text(
        json.dumps(
            [
                {
                    "tools": json.dumps(
                        [
                            {
                                "name": "search_user",
                                "description": "Find a user by name.",
                                "inputSchema": {"type": "object"},
                            },
                            {
                                "name": "send_email",
                                "description": "Send an email to a user.",
                                "inputSchema": {"type": "object"},
                            },
                        ]
                    ),
                    "conversations": [
                        {"from": "human", "value": "Find Ada and email her."},
                        {
                            "from": "function_call",
                            "value": json.dumps({"name": "search_user", "arguments": {"name": "Ada"}}),
                        },
                        {"from": "observation", "value": '{"id": 7}'},
                        {
                            "from": "function_call",
                            "value": json.dumps({"name": "send_email", "arguments": {"user_id": 7}}),
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    report = build_function_schema_augmentation(
        unitool_root=unitool_root,
        bfcl_root=None,
        output_dir=output_dir,
        include_unitool_apigen=True,
        include_unitool_bfcl=False,
        include_official_bfcl=False,
    )

    skills = [json.loads(line) for line in (output_dir / "skill_pool.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (output_dir / "retrieval.jsonl").read_text().splitlines()]

    assert report["status"] == "ok"
    assert report["skill_count"] == 2
    assert report["retrieval_row_count"] == 2
    assert {row["name"] for row in skills} == {"search_user", "send_email"}
    assert rows[0]["source"] == "unitoolcall_apigen"
    assert rows[0]["positive_skill_id"].startswith("unitoolcall/apigen/search-user/")
    assert "Find Ada and email her." in rows[0]["query_text"]
    assert rows[1]["positive_skill_id"].startswith("unitoolcall/apigen/send-email/")
    assert "previous_function_calls:" in rows[1]["query_text"]
    assert rows[1]["negative_skill_ids"]


def test_build_function_schema_augmentation_adds_apibank_train_only_rows(tmp_path):
    apibank_root = tmp_path / "apibank"
    train_dir = apibank_root / "training-data"
    test_dir = apibank_root / "test-data"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (train_dir / "lv1-api-train.json").write_text(
        json.dumps(
            [
                {
                    "instruction": "Generate API request.\nAPI descriptions:",
                    "input": (
                        '{"apiCode": "AddMeeting", "description": "add a meeting", "parameters": {}}\n'
                        '{"apiCode": "CancelMeeting", "description": "cancel a meeting", "parameters": {}}\n'
                        "User: Add a meeting with Ada.\nGenerate API Request:"
                    ),
                    "output": "API-Request: [AddMeeting(person='Ada')]",
                }
            ]
        ),
        encoding="utf-8",
    )
    (test_dir / "level-1-api.json").write_text(
        json.dumps(
            [
                {
                    "instruction": "Generate API request.\nAPI descriptions:",
                    "input": (
                        '{"apiCode": "LeakTool", "description": "held out test API", "parameters": {}}\n'
                        "User: Use the held-out tool.\nGenerate API Request:"
                    ),
                    "expected_output": "API-Request: [LeakTool()]",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = build_function_schema_augmentation(
        unitool_root=None,
        bfcl_root=None,
        apibank_root=apibank_root,
        output_dir=tmp_path / "out",
        include_unitool_apigen=False,
        include_unitool_bfcl=False,
        include_official_bfcl=False,
        include_official_apibank_train=True,
    )

    skills = [json.loads(line) for line in (tmp_path / "out" / "skill_pool.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (tmp_path / "out" / "retrieval.jsonl").read_text().splitlines()]

    assert report["status"] == "ok"
    assert report["retrieval_row_count"] == 1
    assert report["sources"][0]["source"] == "official_apibank_function_schema"
    assert report["sources"][0]["split_policy"] == "train_only"
    assert report["sources"][0]["retrieval_rows"] == 1
    assert {row["name"] for row in skills} >= {"AddMeeting", "CancelMeeting"}
    assert "LeakTool" not in {row["name"] for row in skills}
    assert rows[0]["source"] == "official_apibank_function_schema"
    assert rows[0]["positive_skill_id"] == "apibank/AddMeeting"
    assert rows[0]["negative_skill_ids"] == ["apibank/CancelMeeting"]
    assert "Add a meeting with Ada" in rows[0]["query_text"]
    assert json.loads(rows[0]["provenance"])["split"] == "train_augmentation"


def test_build_function_schema_augmentation_rejects_apibank_test_data_for_train_aug(tmp_path):
    apibank_root = tmp_path / "apibank"
    (apibank_root / "test-data").mkdir(parents=True)
    (apibank_root / "test-data" / "level-1-api.json").write_text("[]", encoding="utf-8")

    try:
        build_function_schema_augmentation(
            unitool_root=None,
            bfcl_root=None,
            apibank_root=apibank_root,
            output_dir=tmp_path / "out",
            include_unitool_apigen=False,
            include_unitool_bfcl=False,
            include_official_bfcl=False,
            include_official_apibank_train=True,
            apibank_train_files=["test-data/level-1-api.json"],
        )
    except ValueError as exc:
        assert "APIBank train augmentation only accepts training-data files" in str(exc)
    else:
        raise AssertionError("expected APIBank test-data path to be rejected")


def test_merge_function_schema_augmentation_merges_sources_and_skips_missing_positive(tmp_path):
    base = tmp_path / "base"
    aug = tmp_path / "aug"
    out = tmp_path / "merged"
    _write_jsonl(base / "skill_pool.jsonl", [{"skill_id": "base/a", "name": "BaseA"}])
    _write_jsonl(
        base / "retrieval.jsonl",
        [{"source": "skillret", "query_id": "q1", "query_text": "base query", "positive_skill_id": "base/a"}],
    )
    _write_jsonl(
        base / "source_inventory.jsonl",
        [{"source_id": "skillret", "benchmark": "SkillRET", "available": True, "train_allowed": True}],
    )
    (base / "manifest.json").write_text(json.dumps({"status": "ok", "files": {}}), encoding="utf-8")
    (base / "trajectories.jsonl").write_text("", encoding="utf-8")

    _write_jsonl(aug / "skill_pool.jsonl", [{"skill_id": "aug/b", "name": "AugB"}])
    _write_jsonl(
        aug / "retrieval.jsonl",
        [
            {"source": "official_apibank_function_schema", "query_id": "a1", "query_text": "aug query", "positive_skill_id": "aug/b"},
            {
                "source": "official_apibank_function_schema",
                "query_id": "a2",
                "query_text": "bad aug query",
                "positive_skill_id": "aug/missing",
            },
        ],
    )
    (aug / "manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sources": [
                    {
                        "source": "official_apibank_function_schema",
                        "split_policy": "train_only",
                        "retrieval_rows": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = merge_function_schema_augmentation(
        base_data_root=base,
        function_schema_augmentation_root=aug,
        output_dir=out,
    )

    merged_skills = [json.loads(line) for line in (out / "skill_pool.jsonl").read_text().splitlines()]
    merged_rows = [json.loads(line) for line in (out / "retrieval.jsonl").read_text().splitlines()]
    inventory = [json.loads(line) for line in (out / "source_inventory.jsonl").read_text().splitlines()]

    assert report["status"] == "ok"
    assert report["skill_count"] == 2
    assert report["retrieval_row_count"] == 2
    assert report["retrieval_rows_skipped_missing_positive"] == 1
    assert {row["skill_id"] for row in merged_skills} == {"base/a", "aug/b"}
    assert {row["query_id"] for row in merged_rows} == {"q1", "a1"}
    assert report["retrieval_by_source"]["official_apibank_function_schema"] == 1
    assert any(row["source_id"] == "official_apibank_function_schema" for row in inventory)
