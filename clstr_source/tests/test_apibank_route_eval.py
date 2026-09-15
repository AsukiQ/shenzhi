from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.apibank_route_eval import (
    _build_apibank_clstr_report,
    apibank_api_skill_id,
    apibank_start_skill_id,
    load_apibank_route_corpus,
    parse_apibank_call_names,
    run_apibank_skillrouter_frozen_eval,
)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _fixture(root: Path) -> Path:
    _write_json(
        root / "test-data" / "level-1-api.json",
        [
            {
                "id": 0,
                "file": "calendar-level-1.jsonl",
                "instruction": (
                    "API descriptions:\n"
                    '{"name": "AddMeeting", "description": "add calendar meeting", "input_parameters": {}}\n'
                    '{"name": "CancelMeeting", "description": "cancel calendar meeting", "input_parameters": {}}'
                ),
                "input": "User: Add a meeting with Ann tomorrow.\nGenerate API Request:\n",
                "expected_output": "API-Request: [AddMeeting(person='Ann', title='planning')]",
            },
            {
                "id": 1,
                "file": "trivial-level-1.jsonl",
                "instruction": (
                    "API descriptions:\n"
                    '{"name": "OnlyTool", "description": "the only candidate", "input_parameters": {}}'
                ),
                "input": "User: Use the only tool.\nGenerate API Request:\n",
                "expected_output": "API-Request: [OnlyTool()]",
            },
        ],
    )
    _write_json(
        root / "test-data" / "level-2-api.json",
        [
            {
                "id": 0,
                "file": "calendar-level-2.jsonl",
                "instruction": (
                    "API descriptions:\n"
                    '{"name": "ToolSearcher", "description": "find tools", "input_parameters": {}}'
                ),
                "input": (
                    "User: Query John's meeting.\n"
                    "API-Request: [ToolSearcher(keywords='meeting')]->"
                    "{'name': 'QueryMeeting', 'description': 'query meeting details', 'input_parameters': {}}\n"
                    "Generate API Request:\n"
                ),
                "expected_output": "API-Request: [QueryMeeting(user_name='John')]",
            },
            {
                "id": 1,
                "file": "missing-level-2.jsonl",
                "instruction": (
                    "API descriptions:\n"
                    '{"name": "ToolSearcher", "description": "find tools", "input_parameters": {}}'
                ),
                "input": "User: Need a token.\nGenerate API Request:\n",
                "expected_output": "API-Request: [GetUserToken(username='u')]",
            },
        ],
    )
    return root


def test_parse_apibank_call_names_ignores_parentheses_inside_arguments():
    output = "API-Request: [medication_lookup(medication_name='Flu (Influenza)')]"

    assert parse_apibank_call_names(output) == ["medication_lookup"]


def test_load_apibank_route_corpus_builds_visible_nontrivial_rows(tmp_path):
    data_root = _fixture(tmp_path)

    corpus = load_apibank_route_corpus(data_root, files=["test-data/level-1-api.json", "test-data/level-2-api.json"])

    assert corpus.report["status"] == "ok"
    assert corpus.report["source_row_count"] == 2
    assert corpus.report["file_counts"]["test-data/level-1-api.json"]["source_rows"] == 1
    assert corpus.report["file_counts"]["test-data/level-1-api.json"]["skipped_trivial_candidate_rows"] == 1
    assert corpus.report["file_counts"]["test-data/level-2-api.json"]["missing_gt_candidate_rows"] == 1

    first = corpus.source_rows[0]
    assert first["benchmark"] == "apibank"
    assert first["skill_id"] == apibank_start_skill_id()
    assert first["next_skill_id"] == apibank_api_skill_id("AddMeeting")
    assert first["next_skill_id"] in first["candidate_next_skill_ids"]
    assert {apibank_api_skill_id("AddMeeting"), apibank_api_skill_id("CancelMeeting")} == set(
        first["candidate_next_skill_ids"]
    )
    assert "Add a meeting" in first["state_text"]

    second = corpus.source_rows[1]
    assert second["skill_id"] == apibank_api_skill_id("ToolSearcher")
    assert second["next_skill_id"] == apibank_api_skill_id("QueryMeeting")
    assert apibank_api_skill_id("QueryMeeting") in second["candidate_next_skill_ids"]
    assert "ToolSearcher" in second["history_text"]
    assert "query meeting details" in second["state_text"]


def test_apibank_skillrouter_frozen_eval_uses_same_visible_candidates(tmp_path, monkeypatch):
    data_root = _fixture(tmp_path / "apibank")

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            if "QueryMeeting" in text or "query meeting" in text:
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.apibank_route_eval._encode_skillrouter_texts", fake_encode)

    report = run_apibank_skillrouter_frozen_eval(
        data_root=data_root,
        output_dir=tmp_path / "out",
        model_name_or_path="fake",
        files=["test-data/level-2-api.json"],
        batch_size=4,
    )

    assert report["status"] == "ok"
    assert report["source_eval_rows"] == 1
    assert report["metrics"]["next_tool_recall@1"] == pytest.approx(1.0)
    assert report["metrics_by_file"]["test-data/level-2-api.json"]["next_tool_mrr"] == pytest.approx(1.0)


def test_apibank_skillrouter_eval_can_apply_finetuned_adapter(tmp_path, monkeypatch):
    data_root = _fixture(tmp_path / "apibank")

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            normalized = text.lower()
            if "querymeeting" in normalized or "query meeting" in normalized:
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.apibank_route_eval._encode_skillrouter_texts", fake_encode)
    adapter_path = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter_state_dict": {
                "q_proj.weight": torch.eye(2),
                "d_proj.weight": torch.eye(2),
            },
            "config": {"train_doc_projection": False},
            "uses_clstr_heads": False,
        },
        adapter_path,
    )

    report = run_apibank_skillrouter_frozen_eval(
        data_root=data_root,
        output_dir=tmp_path / "out_adapter",
        model_name_or_path="fake",
        adapter_checkpoint_path=adapter_path,
        files=["test-data/level-2-api.json"],
        batch_size=4,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_finetuned_biencoder_adapter"
    assert report["adapter_checkpoint_path"] == str(adapter_path)
    assert report["metrics"]["next_tool_recall@1"] == pytest.approx(1.0)


def test_apibank_clstr_report_records_stage4_checkpoint():
    report = _build_apibank_clstr_report(
        output_dir="outputs/apibank",
        data_root="data/apibank",
        stage0_checkpoint_path="stage0.pt",
        stage2_checkpoint_path="stage2.pt",
        stage4_checkpoint_path="stage4.pt",
        skills_path="skills.jsonl",
        source_eval_rows=1,
        retained_eval_rows=1,
        corpus_report={"status": "ok"},
        route_data_report={"positive_injected_rows": 0},
        memory_report={},
        stage0_prior_eval={"stage4_next_skill_recall@1": 0.0},
        base_eval={"stage4_next_skill_recall@1": 0.0},
        stage4_eval={"stage4_next_skill_recall@1": 1.0},
        config={},
        stage0_prior_report={},
        base_eval_by_benchmark={},
        stage4_eval_by_benchmark={},
        model_load={},
        candidate_recall={
            "candidate_recall_applicable": False,
            "candidate_recall_applicability_reason": "benchmark_local_diagnostic_only",
            "candidate_recall_saturated": False,
        },
    )

    assert report["stage4_checkpoint_path"] == "stage4.pt"
    assert report["candidate_recall"]["candidate_recall_applicable"] is False
    assert report["candidate_recall"]["candidate_recall_applicability_reason"] == (
        "benchmark_local_diagnostic_only"
    )


def test_apibank_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, "scripts/run_apibank_full_clstr_route_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "apibank" in result.stdout.lower()
    assert "--stage2_checkpoint_path" in result.stdout
    assert "--stage4_checkpoint_path" in result.stdout
