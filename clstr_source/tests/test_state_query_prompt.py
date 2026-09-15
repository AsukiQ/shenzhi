from __future__ import annotations

import pytest

from clstr.state_query_prompt import (
    CLSTR_CAUSAL_STATE_V1,
    HEAD_TAIL_V1,
    format_state_query,
    resolve_state_query_prompt_contract,
)


def test_clstr_causal_state_prompt_renders_exact_qwen_query_contract():
    text = format_state_query(
        "goal: inspect invoice",
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=2000,
        truncation=HEAD_TAIL_V1,
    )

    assert text == (
        "Instruct: Given an agent task or current execution state and interaction history, "
        "retrieve the skill or tool document most useful for the next action.\n"
        "Query:goal: inspect invoice"
    )


def test_head_tail_truncation_preserves_goal_and_latest_context():
    raw = "goal:" + ("a" * 90) + "latest:" + ("z" * 90)

    text = format_state_query(
        raw,
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=80,
        truncation=HEAD_TAIL_V1,
    )

    query = text.split("\nQuery:", 1)[1]
    assert len(query) == 80
    assert query.startswith("goal:")
    assert query.endswith("z" * 20)
    assert "...[state truncated]..." in query


def test_prompted_state_rejects_preformatted_instruct_text():
    with pytest.raises(ValueError, match="raw state text"):
        format_state_query(
            "Instruct: stale prompt\nQuery:goal",
            prompt_version=CLSTR_CAUSAL_STATE_V1,
            max_chars=2000,
            truncation=HEAD_TAIL_V1,
        )


def test_causal_contract_records_canonical_instruction_and_length_policy():
    contract = resolve_state_query_prompt_contract(
        prompt_version=CLSTR_CAUSAL_STATE_V1,
        max_chars=2000,
        truncation=HEAD_TAIL_V1,
    )

    assert contract["state_query_prompt_version"] == CLSTR_CAUSAL_STATE_V1
    assert contract["state_query_max_chars"] == 2000
    assert contract["state_query_truncation"] == HEAD_TAIL_V1
    assert "next action" in contract["state_query_instruction"]
