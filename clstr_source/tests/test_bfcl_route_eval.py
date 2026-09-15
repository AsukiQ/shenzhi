from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from clstr.bfcl_route_eval import _build_bfcl_clstr_report, load_bfcl_route_corpus, run_bfcl_skillrouter_frozen_eval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _function(name: str, description: str = "tool desc") -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "dict", "properties": {}, "required": []},
    }


def _official_bfcl_fixture(root: Path) -> Path:
    _write_jsonl(
        root / "BFCL_v3_multiple.json",
        [
            {
                "id": "multiple_0",
                "question": [[{"role": "user", "content": "Book a hotel, not a flight."}]],
                "function": [_function("book_hotel", "book hotel"), _function("book_flight", "book flight")],
            }
        ],
    )
    _write_jsonl(
        root / "possible_answer" / "BFCL_v3_multiple.json",
        [{"id": "multiple_0", "ground_truth": [{"book_hotel": {"city": ["Paris"]}}]}],
    )
    _write_jsonl(
        root / "BFCL_v3_parallel_multiple.json",
        [
            {
                "id": "parallel_multiple_0",
                "question": [[{"role": "user", "content": "Book a hotel and reserve a car."}]],
                "function": [
                    _function("book_hotel", "book hotel"),
                    _function("reserve_car", "reserve car"),
                    _function("book_flight", "book flight"),
                ],
            }
        ],
    )
    _write_jsonl(
        root / "possible_answer" / "BFCL_v3_parallel_multiple.json",
        [
            {
                "id": "parallel_multiple_0",
                "ground_truth": [{"book_hotel": {"city": ["Paris"]}}, {"reserve_car": {"city": ["Paris"]}}],
            }
        ],
    )
    _write_jsonl(
        root / "multi_turn_func_doc" / "gorilla_file_system.json",
        [_function("cd", "change directory"), _function("mkdir", "make directory"), _function("mv", "move file")],
    )
    _write_jsonl(
        root / "BFCL_v3_multi_turn_base.json",
        [
            {
                "id": "multi_turn_base_0",
                "question": [
                    [{"role": "user", "content": "Go to docs and create temp."}],
                    [{"role": "user", "content": "Move report into temp."}],
                ],
                "initial_config": {},
                "path": ["cd", "mkdir", "mv"],
                "involved_classes": ["GorillaFileSystem"],
            }
        ],
    )
    _write_jsonl(
        root / "possible_answer" / "BFCL_v3_multi_turn_base.json",
        [
            {
                "id": "multi_turn_base_0",
                "ground_truth": [
                    ["cd(folder='docs')", "mkdir(dir_name='temp')"],
                    ["mv(source='report.pdf', destination='temp')"],
                ],
            }
        ],
    )
    return root


def test_load_bfcl_route_corpus_builds_official_single_and_multiturn_rows(tmp_path):
    data_root = _official_bfcl_fixture(tmp_path)

    corpus = load_bfcl_route_corpus(
        data_root,
        categories=["BFCL_v3_multiple", "BFCL_v3_parallel_multiple", "BFCL_v3_multi_turn_base"],
    )

    assert corpus.report["status"] == "ok"
    assert corpus.report["source_row_count"] == 6
    assert corpus.report["category_counts"]["BFCL_v3_multiple"]["source_rows"] == 1
    assert corpus.report["category_counts"]["BFCL_v3_parallel_multiple"]["source_rows"] == 2
    assert corpus.report["category_counts"]["BFCL_v3_multi_turn_base"]["source_rows"] == 3
    assert any(skill["skill_id"] == "bfcl/__start__" for skill in corpus.skills)

    first = next(row for row in corpus.source_rows if row["bfcl_category"] == "BFCL_v3_multiple")
    assert first["next_skill_name"] == "book_hotel"
    assert len(first["candidate_next_skill_ids"]) == 2
    assert first["next_skill_id"] in first["candidate_next_skill_ids"]
    assert first["skill_id"] == "bfcl/__start__"
    assert "Book a hotel" in first["state_text"]

    multiturn_second_call = next(
        row
        for row in corpus.source_rows
        if row["bfcl_category"] == "BFCL_v3_multi_turn_base" and row["turn_index"] == 0 and row["call_index"] == 1
    )
    assert multiturn_second_call["skill_id"].endswith("/cd")
    assert multiturn_second_call["next_skill_name"] == "mkdir"
    assert "previous_calls" in multiturn_second_call["state_text"]

    multiturn_next_turn = next(
        row for row in corpus.source_rows if row["bfcl_category"] == "BFCL_v3_multi_turn_base" and row["turn_index"] == 1
    )
    assert multiturn_next_turn["skill_id"].endswith("/mkdir")
    assert multiturn_next_turn["next_skill_name"] == "mv"
    assert "Go to docs" in multiturn_next_turn["state_text"]
    assert "Move report" in multiturn_next_turn["state_text"]


def test_bfcl_skillrouter_frozen_eval_ranks_with_fake_embeddings(tmp_path, monkeypatch):
    data_root = _official_bfcl_fixture(tmp_path / "bfcl")

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            if "reserve car" in text or "reserve_car" in text:
                rows.append(torch.tensor([0.0, 1.0, 0.0]))
            elif "book flight" in text or "book_flight" in text:
                rows.append(torch.tensor([0.0, 0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.bfcl_route_eval._encode_skillrouter_texts", fake_encode)

    report = run_bfcl_skillrouter_frozen_eval(
        data_root=data_root,
        output_dir=tmp_path / "out",
        model_name_or_path="fake",
        categories=["BFCL_v3_multiple"],
        batch_size=4,
    )

    assert report["status"] == "ok"
    assert report["source_eval_rows"] == 1
    assert report["metrics"]["next_tool_recall@1"] == pytest.approx(1.0)
    assert report["metrics_by_category"]["BFCL_v3_multiple"]["next_tool_mrr"] == pytest.approx(1.0)


def test_bfcl_skillrouter_eval_can_apply_finetuned_adapter(tmp_path, monkeypatch):
    data_root = _official_bfcl_fixture(tmp_path / "bfcl")

    def fake_encode(*, model_name_or_path, texts, batch_size, max_length):
        rows = []
        for text in texts:
            normalized = text.lower()
            if "book_hotel" in normalized or "book hotel" in normalized or "book a hotel" in normalized:
                rows.append(torch.tensor([1.0, 0.0]))
            else:
                rows.append(torch.tensor([0.0, 1.0]))
        return torch.stack(rows)

    monkeypatch.setattr("clstr.bfcl_route_eval._encode_skillrouter_texts", fake_encode)
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

    report = run_bfcl_skillrouter_frozen_eval(
        data_root=data_root,
        output_dir=tmp_path / "out_adapter",
        model_name_or_path="fake",
        adapter_checkpoint_path=adapter_path,
        categories=["BFCL_v3_multiple"],
        batch_size=4,
    )

    assert report["status"] == "ok"
    assert report["method"] == "skillrouter_finetuned_biencoder_adapter"
    assert report["adapter_checkpoint_path"] == str(adapter_path)
    assert report["metrics"]["next_tool_recall@1"] == pytest.approx(1.0)


def test_bfcl_clstr_report_records_stage4_checkpoint():
    report = _build_bfcl_clstr_report(
        output_dir="outputs/bfcl",
        data_root="data/bfcl",
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
            "candidate_recall_applicability_reason": "nonsequential_without_causal_update",
            "candidate_recall_saturated": False,
        },
    )

    assert report["stage4_checkpoint_path"] == "stage4.pt"
    assert report["candidate_recall"]["candidate_recall_applicable"] is False
    assert report["candidate_recall"]["candidate_recall_applicability_reason"] == (
        "nonsequential_without_causal_update"
    )


def test_bfcl_converter_excludes_trivial_default_categories(tmp_path):
    _write_jsonl(
        tmp_path / "BFCL_v3_simple.json",
        [
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "Compute area."}]],
                "function": [_function("area", "area")],
            }
        ],
    )
    _write_jsonl(
        tmp_path / "possible_answer" / "BFCL_v3_simple.json",
        [{"id": "simple_0", "ground_truth": [{"area": {}}]}],
    )

    corpus = load_bfcl_route_corpus(tmp_path, categories=["BFCL_v3_simple"], include_trivial=False)

    assert corpus.report["source_row_count"] == 0
    assert corpus.report["category_counts"]["BFCL_v3_simple"]["skipped_trivial_candidate_rows"] == 1
