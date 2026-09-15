import pytest

from clstr.toolbench_qwen_rerank import (
    build_qwen_skill_rerank_prompt,
    parse_qwen_ranked_labels,
    ranking_metrics_from_ranked_skill_ids,
    strict_metrics,
)


def test_parse_qwen_ranked_labels_accepts_json_and_deduplicates():
    labels = ["C001", "C002", "C003", "C004"]

    parsed = parse_qwen_ranked_labels('{"ranking": ["C003", "C001", "C003", "C004"]}', labels)

    assert parsed == ["C003", "C001", "C004"]


def test_parse_qwen_ranked_labels_accepts_numbered_text():
    labels = ["C001", "C002", "C003"]

    parsed = parse_qwen_ranked_labels("1. C002\n2. C001\n3. C003", labels)

    assert parsed == ["C002", "C001", "C003"]


def test_ranking_metrics_appends_unmentioned_candidates_after_model_order():
    metrics = ranking_metrics_from_ranked_skill_ids(
        ranked_skill_ids=["skill-b"],
        candidate_skill_ids=["skill-a", "skill-b", "skill-c"],
        positive_skill_id="skill-a",
        source_row_count=1,
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["generated_rank_count"] == 1.0


def test_strict_metrics_scales_retained_rows_to_source_denominator():
    retained = {"recall@1": 0.5, "recall@5": 1.0, "mrr": 0.75}

    metrics = strict_metrics(retained, retained_rows=4, source_rows=8)

    assert metrics["strict_recall@1"] == 0.25
    assert metrics["strict_recall@5"] == 0.5
    assert metrics["strict_mrr"] == 0.375


def test_prompt_contains_state_and_candidate_full_text_preview():
    prompt = build_qwen_skill_rerank_prompt(
        state_text="goal: check weather",
        candidates=[
            {
                "label": "C001",
                "skill_id": "weather/current",
                "name": "Current Weather",
                "description": "Get weather",
                "body": "Use this API for forecast and alerts.",
            }
        ],
        max_skill_chars=64,
    )

    assert "goal: check weather" in prompt
    assert "C001" in prompt
    assert "weather/current" in prompt
    assert "Use this API" in prompt
    assert "Return JSON only" in prompt
