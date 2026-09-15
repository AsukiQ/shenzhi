from __future__ import annotations

from clstr.stage0_handoff_query_ablation import (
    build_handoff_ablation_query,
    sanitize_observation_for_handoff,
)


def test_sanitize_toolbench_empty_error_keeps_response_without_error_marker() -> None:
    text = '{"error": "", "response": "{\\"result\\": [1, 2, 3]}"}'

    sanitized = sanitize_observation_for_handoff(text, max_chars=80)

    assert "tool_error" not in sanitized
    assert "result" in sanitized


def test_sanitize_toolbench_nonempty_error_keeps_compact_error_marker() -> None:
    text = '{"error": "Forbidden", "response": "<html><body>Forbidden</body></html>"}'

    sanitized = sanitize_observation_for_handoff(text, max_chars=80)

    assert sanitized.startswith("tool_error: Forbidden")
    assert "<html>" not in sanitized


def test_query_variants_do_not_leak_next_labels() -> None:
    row = {
        "goal_text": "Find weather and then stock prices.",
        "state_text": "goal: Find weather and then stock prices.\nprevious_tools: weather_api",
        "history_text": "weather_api",
        "action_text": "weather_api: {}",
        "next_observation_text": '{"error": "", "response": "sunny"}',
        "next_action_text": "stock_price_api",
        "skill_id": "current/skill",
        "next_skill_id": "secret/gold-skill",
    }

    for variant in ("goal_only", "state_no_observation", "state_sanitized_observation", "goal_history_action"):
        query = build_handoff_ablation_query(row, target="next", variant=variant)
        assert "secret/gold-skill" not in query
        assert "stock_price_api" not in query
