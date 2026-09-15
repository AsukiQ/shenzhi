import pytest

from clstr.toolbench_qwen_reranker import (
    build_qwen3_reranker_document,
    format_qwen3_reranker_instruction,
    rank_candidate_skill_ids_by_scores,
)
from clstr.toolbench_qwen_rerank import ranking_metrics_from_ranked_skill_ids, strict_metrics


def test_format_qwen3_reranker_instruction_uses_task_specific_template():
    text = format_qwen3_reranker_instruction(
        instruction="Rank next ToolBench skills.",
        query="goal: search weather\nhistory: get_city",
        document="skill: weather/current",
    )

    assert "<Instruct>: Rank next ToolBench skills." in text
    assert "<Query>: goal: search weather" in text
    assert "<Document>: skill: weather/current" in text


def test_build_qwen3_reranker_document_contains_skill_fields_and_truncates_body():
    doc = build_qwen3_reranker_document(
        {
            "skill_id": "toolbench/weather/current",
            "name": "Current Weather",
            "description": "Read current weather.",
            "body": "A" * 200,
        },
        max_skill_chars=80,
    )

    assert "skill_id: toolbench/weather/current" in doc
    assert "name: Current Weather" in doc
    assert "description: Read current weather." in doc
    assert "[truncated]" in doc
    assert len(doc) < 220


def test_rank_candidate_skill_ids_by_scores_sorts_descending_and_is_stable_on_ties():
    ranked = rank_candidate_skill_ids_by_scores(
        candidate_skill_ids=["a", "b", "c", "d"],
        scores=[0.1, 0.5, 0.5, -1.0],
    )

    assert ranked == ["b", "c", "a", "d"]


def test_qwen3_reranker_ranked_ids_work_with_existing_strict_metrics():
    ranked = rank_candidate_skill_ids_by_scores(
        candidate_skill_ids=["a", "b", "c"],
        scores=[0.1, 0.9, 0.2],
    )
    retained = ranking_metrics_from_ranked_skill_ids(
        ranked_skill_ids=ranked,
        candidate_skill_ids=["a", "b", "c"],
        positive_skill_id="c",
    )

    assert retained["recall@1"] == 0.0
    assert retained["recall@5"] == 1.0
    assert retained["mrr"] == pytest.approx(0.5)
    strict = strict_metrics(retained, retained_rows=10, source_rows=20)
    assert strict["strict_mrr"] == pytest.approx(0.25)
