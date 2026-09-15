from __future__ import annotations

from clstr.history_channel import (
    CAUSAL_STATE_CONTRACT,
    CURRENT_STATE_CONTRACT,
    actual_causal_observation,
    actual_causal_replay_step,
    audit_history_channel_rows,
    materialize_history_free_state,
    materialize_structured_causal_state,
    materialize_structured_current_state,
    materialize_structured_retrieval_state,
    router_state_text,
    serialize_causal_state,
    serialize_current_state_components,
    strip_history_sections,
)


def test_structured_causal_state_has_exact_empty_prefix_and_verified_events() -> None:
    current = materialize_structured_current_state({"goal_text": "book a flight"})
    empty = materialize_structured_causal_state(current, [])
    assert empty["causal_state_contract"] == CAUSAL_STATE_CONTRACT
    assert empty["state_text_causal"] == empty["state_text_current"]
    assert empty["zero_history_causal_equals_current"] is True

    causal = materialize_structured_causal_state(
        current,
        [
            {
                "step_index": 0,
                "skill_id": "airline/search",
                "action_text": "search(origin=SFO)",
                "result_text": "two flights",
                "result_executed": True,
            }
        ],
    )
    assert causal["zero_history_causal_equals_current"] is False
    assert "event[0].skill_id: airline/search" in causal["state_text_causal"]
    assert "event[0].result: two flights" in causal["state_text_causal"]
    report = audit_history_channel_rows(
        [causal],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    assert report["status"] == "ok"
    assert report["structured_causal_state_rows"] == 1


def test_structured_causal_audit_rejects_future_event_and_digest_tamper() -> None:
    row = materialize_structured_causal_state(
        {"goal_text": "g"},
        [
            {
                "step_index": 1,
                "skill_id": "tool/a",
                "action_text": "a()",
            }
        ],
    )
    row["decision_step_index"] = 1
    row["causal_state_sha256"] = "tampered"
    report = audit_history_channel_rows(
        [row],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    assert report["status"] == "action_required"
    assert report["causal_state_reasons"] == {
        "causal_event_not_before_decision": 1,
        "causal_state_digest_mismatch": 1,
    }


def test_strip_history_sections_removes_only_structured_execution_history():
    text = (
        "goal: explain the history of aviation\n"
        "observation: current page\n"
        "previous_actions: search[aviation]\n"
        "click[result]"
    )

    assert strip_history_sections(text) == (
        "goal: explain the history of aviation\n"
        "observation: current page"
    )


def test_materialize_history_free_state_preserves_full_context_for_non_router_consumers():
    row = {
        "state_text": "goal: g\nobservation: now\nhistory:\n1. tool({})",
        "history_text": "1. tool({})",
    }

    prepared = materialize_history_free_state(row, replace_state_text=True)

    assert prepared["state_text"] == "goal: g\nobservation: now"
    assert prepared["state_text_current"] == "goal: g\nobservation: now"
    assert prepared["state_text_full"] == row["state_text"]
    assert router_state_text(prepared) == "goal: g\nobservation: now"


def test_router_state_text_prefers_explicit_current_state():
    row = {
        "state_text": "full\nhistory:\nold",
        "state_text_current": "current only",
    }

    assert router_state_text(row) == "current only"


def test_history_channel_audit_rejects_marker_and_exact_history_leakage():
    report = audit_history_channel_rows(
        [
            {
                "trajectory_id": "t",
                "step_index": 1,
                "state_text_current": "goal: g\nhistory:\nold",
                "history_text": "old",
                "replay_prefix": [],
            }
        ],
        require_explicit_current=True,
    )

    assert report["status"] == "action_required"
    assert report["leaked_row_count"] == 1
    assert report["leakage_reasons"]["structured_history_marker"] == 1
    assert report["leakage_reasons"]["history_text_exact_substring"] == 1


def test_actual_causal_observation_rejects_oracle_arguments_and_weak_next_state():
    assert not actual_causal_observation(
        {
            "next_observation_text": "oracle_next_action_arguments: {}",
        }
    )
    assert not actual_causal_observation(
        {
            "next_observation_text": "next state text",
            "observation_source": "next_state_without_tool_result",
        }
    )
    assert actual_causal_observation(
        {
            "next_observation_text": "tool_result: reservation Q69X3R",
            "observation_source": "actual_tool_result",
        }
    )
    assert actual_causal_replay_step(
        {
            "action_text": "lookup_reservation()",
            "skill_id": "tau2/airline/lookup_reservation",
            "next_observation_text": "tool_result: reservation Q69X3R",
            "observation_source": "actual_tool_result",
        }
    )
    assert not actual_causal_replay_step(
        {
            "next_observation_text": "tool_result: reservation Q69X3R",
            "observation_source": "actual_tool_result",
        }
    )


def test_history_channel_audit_can_fail_closed_on_noncausal_replay_observations():
    report = audit_history_channel_rows(
        [
            {
                "trajectory_id": "t",
                "step_index": 1,
                "state_text_current": "goal: g\nobservation: now",
                "history_text": "old",
                "replay_prefix": [
                    {
                        "action_text": "tool()",
                        "next_observation_text": "oracle_next_tool_arguments: {}",
                    }
                ],
            }
        ],
        require_explicit_current=True,
        require_actual_replay_observation=True,
    )

    assert report["status"] == "action_required"
    assert report["invalid_replay_observation_steps"] == 1


def test_structured_current_state_rebuilds_labeled_observation_without_replay():
    row = {
        "goal_text": "find the mug",
        "task_text": "pick_and_place",
        "state_text": (
            "goal: find the mug\n"
            "task_type: pick_and_place\n"
            "observation: You are in the kitchen.\nA mug is on the table.\n"
            "history: look; go kitchen"
        ),
    }

    prepared = materialize_structured_current_state(row, replace_state_text=True)

    assert prepared["current_state_contract"] == CURRENT_STATE_CONTRACT
    assert prepared["current_state_materialization_source"] == "labeled_observation_migration"
    assert prepared["current_state_components"] == {
        "goal_text": "find the mug",
        "task_text": "pick_and_place",
        "current_observation_text": "You are in the kitchen.\nA mug is on the table.",
    }
    assert prepared["state_text_current"] == (
        "goal: find the mug\n"
        "task: pick_and_place\n"
        "observation: You are in the kitchen.\nA mug is on the table."
    )
    assert "history:" not in prepared["state_text_current"]
    assert prepared["state_text_full"] == row["state_text"]


def test_structured_current_state_uses_goal_only_for_metadata_only_sources():
    prepared = materialize_structured_current_state(
        {
            "goal_text": "look up a stock price",
            "state_text": "goal: look up a stock price\nprevious_tools: search",
        },
        replace_state_text=True,
    )
    assert prepared["state_text_current"] == "goal: look up a stock price"
    assert prepared["current_state_materialization_source"] == "explicit_structured_fields"


def test_structured_retrieval_state_supports_query_alias_and_removes_history_suffix():
    prepared = materialize_structured_retrieval_state(
        {
            "query": "goal: compare rates\nprevious_tools: quote",
            "query_id": "q",
        },
        replace_state_text=True,
    )

    assert prepared["query_id"] == "q"
    assert prepared["state_text"] == "goal: compare rates"
    assert prepared["state_text_current"] == "goal: compare rates"
    assert prepared["state_text_full"] == (
        "goal: compare rates\nprevious_tools: quote"
    )
    assert (
        prepared["retrieval_query_materialization_source"]
        == "legacy_history_section_removed"
    )


def test_structured_current_state_fails_without_reconstructable_components():
    try:
        materialize_structured_current_state(
            {"state_text": "unlabeled text\nhistory: old"}
        )
    except ValueError as exc:
        assert "cannot reconstruct current state" in str(exc)
    else:
        raise AssertionError("unstructured legacy state must fail closed")


def test_structured_current_audit_requires_matching_components():
    row = materialize_structured_current_state(
        {"goal_text": "g", "current_observation_text": "now"}
    )
    report = audit_history_channel_rows(
        [row],
        require_explicit_current=True,
        require_structured_current=True,
    )
    assert report["status"] == "ok"
    assert report["structured_current_state_rows"] == 1

    broken = dict(row)
    broken["state_text_current"] = "goal: changed"
    report = audit_history_channel_rows(
        [broken],
        require_explicit_current=True,
        require_structured_current=True,
    )
    assert report["status"] == "action_required"
    assert report["leakage_reasons"]["structured_current_serialization_mismatch"] == 1


def test_structured_current_audit_allows_goal_to_name_prior_or_parallel_tool():
    row = materialize_structured_current_state(
        {
            "goal_text": "In parallel, run quote and exchange_rate",
            "current_observation_text": "The request is ready.",
        }
    )
    row["history_text"] = "quote"

    report = audit_history_channel_rows(
        [row],
        require_explicit_current=True,
        require_structured_current=True,
    )

    assert report["status"] == "ok"
    assert report["leaked_row_count"] == 0


def test_structured_current_audit_rejects_history_inside_current_observation():
    row = materialize_structured_current_state(
        {
            "goal_text": "finish the task",
            "current_observation_text": "prior calls: quote -> exchange_rate",
        }
    )
    row["history_text"] = "quote -> exchange_rate"

    report = audit_history_channel_rows(
        [row],
        require_explicit_current=True,
        require_structured_current=True,
    )

    assert report["status"] == "action_required"
    assert report["leakage_reasons"]["history_text_exact_substring"] == 1


def test_current_component_serialization_does_not_duplicate_goal_as_task():
    assert serialize_current_state_components(
        {
            "goal_text": "same",
            "task_text": "same",
            "current_observation_text": "now",
        }
    ) == "goal: same\nobservation: now"


def test_history_channel_audit_persists_bounded_leakage_evidence():
    report = audit_history_channel_rows(
        [
            {
                "trajectory_id": "t",
                "task_id": "t::1",
                "step_index": 1,
                "state_text_current": "goal: x\nhistory evidence: tool-a",
                "history_text": "tool-a",
            }
        ],
        require_explicit_current=True,
    )
    assert report["leaked_row_count"] == 1
    assert report["leakage_examples"] == [
        {
            "trajectory_id": "t",
            "task_id": "t::1",
            "query_id": "",
            "step_index": 1,
            "reasons": ["history_text_exact_substring"],
            "state_text_current": "goal: x\nhistory evidence: tool-a",
            "history_text": "tool-a",
        }
    ]
