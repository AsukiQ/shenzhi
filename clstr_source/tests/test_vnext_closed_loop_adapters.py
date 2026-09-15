from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


pytest.importorskip("tool_sandbox")

from clstr.vnext_toolsandbox_agent import (
    CLSTRVNextToolSandboxUser,
    CLSTRVNextToolSandboxAgent,
    _agent_tool_names_in_selected_order,
    _deterministic_completion_kwargs,
    _deterministic_user_completion_kwargs,
    _dual_evidence_guidance_order,
    _messages_with_ranked_tool_guidance,
    _provider_tool_calls,
    _provider_tool_names,
    _ranked_tool_guidance,
    _retained_tools_in_selection_order,
    _selection_order_diagnostics,
)
from scripts.run_clstr_vnext_toolsandbox_success import _selection_diagnostics


def test_tau2_tools_preserve_clstr_rank_order() -> None:
    pytest.importorskip("tau2")
    from clstr.vnext_tau2_agent import _tools_in_selection_order

    tool_by_skill_id = {"skill/a": "A", "skill/b": "B", "skill/c": "C"}
    selected = [
        {"skill_id": "skill/c"},
        {"skill_id": "skill/a"},
    ]

    assert _tools_in_selection_order(selected, tool_by_skill_id) == ["C", "A"]


def test_toolsandbox_retained_tools_preserve_rank_then_passthrough() -> None:
    available = {"A": object(), "B": object(), "C": object(), "system": object()}
    retained, filtered = _retained_tools_in_selection_order(
        available=available,
        candidate_by_agent_name={
            "A": "skill/a",
            "B": "skill/b",
            "C": "skill/c",
        },
        passthrough_names=["system"],
        selected=[{"skill_id": "skill/c"}, {"skill_id": "skill/a"}],
    )

    assert list(retained) == ["C", "A", "system"]
    assert filtered == 1


def test_toolsandbox_ranked_guidance_preserves_messages_and_exact_tool_names() -> None:
    messages = [
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Find the message."},
    ]
    guidance = _ranked_tool_guidance(["search_messages", "get_current_timestamp"])

    guided = _messages_with_ranked_tool_guidance(messages, guidance)

    assert messages[0]["content"] == "Be accurate."
    assert guided[0]["content"].startswith("Be accurate.")
    assert "soft learned preference" in guided[0]["content"]
    assert "1. search_messages" in guided[0]["content"]
    assert "2. get_current_timestamp" in guided[0]["content"]
    assert "zero, one, or multiple legal tools" in guided[0]["content"]
    assert guided[1] == messages[1]


def test_toolsandbox_ranked_agent_names_and_provider_parallel_calls_are_preserved() -> None:
    selected = [{"skill_id": "skill/b"}, {"skill_id": "skill/a"}]
    assert _agent_tool_names_in_selected_order(
        candidate_by_agent_name={"A": "skill/a", "B": "skill/b"},
        selected=selected,
    ) == ["B", "A"]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id="call-b",
                            function=SimpleNamespace(name="B"),
                        ),
                        SimpleNamespace(
                            id="call-a",
                            function=SimpleNamespace(name="A"),
                        ),
                    ]
                )
            )
        ]
    )
    assert _provider_tool_names(response) == ["B", "A"]
    assert _provider_tool_calls(response) == [
        {"tool_call_id": "call-b", "tool_name": "B"},
        {"tool_call_id": "call-a", "tool_name": "A"},
    ]


def test_toolsandbox_dual_evidence_guidance_preserves_static_anchor() -> None:
    candidate_by_agent_name = {
        "search_holiday": "skill/holiday",
        "get_wifi_status": "skill/wifi",
        "timestamp_diff": "skill/time",
    }
    shared_top1 = _dual_evidence_guidance_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=[
            {
                "skill_id": "skill/holiday",
                "score": 3.0,
                "static_score": 3.0,
                "dynamic_score": 4.0,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/time",
                "score": 2.0,
                "static_score": 1.0,
                "dynamic_score": 2.0,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/wifi",
                "score": 1.0,
                "static_score": 2.0,
                "dynamic_score": 1.0,
                "selected_expert": "dynamic",
            },
        ],
        route_mode="adaptive",
        history_depth=1,
        guidance_top_k=2,
    )
    assert shared_top1["guided_tool_names"] == [
        "search_holiday",
        "get_wifi_status",
    ]
    assert shared_top1["dual_expert_guidance_applied"] is True

    distinct_top1 = _dual_evidence_guidance_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=[
            {
                "skill_id": "skill/holiday",
                "score": 3.0,
                "static_score": 2.0,
                "dynamic_score": 3.0,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/wifi",
                "score": 2.0,
                "static_score": 3.0,
                "dynamic_score": 2.0,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/time",
                "score": 1.0,
                "static_score": 1.0,
                "dynamic_score": 1.0,
                "selected_expert": "dynamic",
            },
        ],
        route_mode="adaptive",
        history_depth=2,
        guidance_top_k=2,
    )
    assert distinct_top1["guided_tool_names"] == [
        "get_wifi_status",
        "search_holiday",
    ]


def test_toolsandbox_dual_evidence_guidance_leaves_static_route_unchanged() -> None:
    report = _dual_evidence_guidance_order(
        candidate_by_agent_name={"A": "skill/a", "B": "skill/b"},
        selected=[
            {
                "skill_id": "skill/a",
                "score": 2.0,
                "static_score": 2.0,
                "dynamic_score": 1.0,
                "selected_expert": "static",
            },
            {
                "skill_id": "skill/b",
                "score": 1.0,
                "static_score": 1.0,
                "dynamic_score": 2.0,
                "selected_expert": "static",
            },
        ],
        route_mode="static",
        history_depth=3,
        guidance_top_k=2,
    )
    assert report["guided_tool_names"] == ["A", "B"]
    assert report["guidance_policy"] == "selected_expert_topk"
    assert report["dual_expert_guidance_applied"] is False


def test_toolsandbox_finalize_consumes_terminal_result() -> None:
    agent = object.__new__(CLSTRVNextToolSandboxAgent)
    messages = [object(), object()]
    consumed: list[object] = []
    agent.get_messages = lambda: messages
    agent.filter_messages = lambda rows: rows
    agent._consume_new_results = lambda rows: consumed.extend(rows)

    agent.finalize_scenario()

    assert consumed == messages


def test_toolsandbox_ranked_guidance_reaches_provider_without_schema_mutation() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            id="call-search",
                            function=SimpleNamespace(name="search_messages"),
                        )
                    ]
                )
            )
        ]
    )
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return response

    agent = object.__new__(CLSTRVNextToolSandboxAgent)
    agent.model_name = "qwen"
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent.executor_temperature = 0.0
    agent.executor_max_tokens = 1024
    agent.executor_enable_thinking = False
    agent.intervention_mode = "ranked_guidance"
    agent._latest_guidance_context = {
        "guidance": _ranked_tool_guidance(["search_messages", "lookup_contact"]),
        "guided_tool_names": ["search_messages", "lookup_contact"],
        "history_depth": 2,
        "selected_expert": "dynamic",
        "guidance_policy": "static_anchor_dynamic_supplement_v1",
        "dual_expert_guidance_applied": True,
        "static_guidance_tool_names": ["search_messages", "lookup_contact"],
        "dynamic_guidance_tool_names": ["lookup_contact", "search_messages"],
    }
    logs: list[dict[str, object]] = []
    agent._append_log = logs.append
    messages = [{"role": "system", "content": "Be accurate."}]
    tools = [
        {"type": "function", "function": {"name": "lookup_contact"}},
        {"type": "function", "function": {"name": "search_messages"}},
    ]

    returned = agent.model_inference(messages, tools)

    assert returned is response
    assert captured["tools"] is tools
    assert messages == [{"role": "system", "content": "Be accurate."}]
    assert "1. search_messages" in captured["messages"][0]["content"]
    assert logs == [
        {
            "record_type": "provider_decision",
            "intervention_mode": "ranked_guidance",
            "history_depth": 2,
            "selected_expert": "dynamic",
            "guidance_policy": "static_anchor_dynamic_supplement_v1",
            "dual_expert_guidance_applied": True,
            "static_guidance_tool_names": [
                "search_messages",
                "lookup_contact",
            ],
            "dynamic_guidance_tool_names": [
                "lookup_contact",
                "search_messages",
            ],
            "provider_guidance_injected": True,
            "guided_tool_names": ["search_messages", "lookup_contact"],
            "provider_tool_calls": [
                {
                    "tool_call_id": "call-search",
                    "tool_name": "search_messages",
                }
            ],
            "provider_tool_names": ["search_messages"],
            "pre_execution_tool_names": ["search_messages"],
            "decision_surface": "official_provider_response_before_execution",
            "provider_tool_call_count": 1,
            "provider_no_tool": False,
            "provider_parallel_tool_call": False,
            "provider_guidance_adopted": True,
            "provider_top1_guidance_adopted": True,
            "posthoc_tool_substitution": False,
        }
    ]


def test_toolsandbox_executor_sampling_bounds_are_explicit() -> None:
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    kwargs = _deterministic_completion_kwargs(
        model_name="executor",
        openai_messages=messages,
        openai_tools=tools,
        temperature=0.0,
        max_tokens=1024,
        enable_thinking=False,
    )

    assert kwargs == {
        "model": "executor",
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 1024,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_toolsandbox_user_sampling_is_explicit() -> None:
    messages = [{"role": "user", "content": "continue"}]
    tools = [{"type": "function", "function": {"name": "end_conversation"}}]
    kwargs = _deterministic_user_completion_kwargs(
        model_name="gpt-4o-2024-05-13",
        openai_messages=messages,
        openai_tools=tools,
        temperature=0.0,
        max_tokens=1024,
    )

    assert kwargs == {
        "model": "gpt-4o-2024-05-13",
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    captured: dict[str, object] = {}
    response = object()

    def create(**payload):
        captured.update(payload)
        return response

    user = object.__new__(CLSTRVNextToolSandboxUser)
    user.model_name = "gpt-4o-2024-05-13"
    user.temperature = 0.0
    user.max_tokens = 1024
    user.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    assert user.model_inference(messages, tools) is response
    assert captured == kwargs


def test_toolsandbox_order_diagnostics_expose_memory_reranking_without_filtering() -> None:
    report = _selection_order_diagnostics(
        candidate_skill_ids=["skill/a", "skill/b", "skill/c"],
        selected=[
            {
                "skill_id": "skill/b",
                "static_score": 0.2,
                "dynamic_score": 0.8,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/a",
                "static_score": 0.9,
                "dynamic_score": 0.7,
                "selected_expert": "dynamic",
            },
            {
                "skill_id": "skill/c",
                "static_score": 0.1,
                "dynamic_score": 0.2,
                "selected_expert": "dynamic",
            },
        ],
        history_depth=2,
    )

    assert report["rank_order_changed"] is True
    assert report["static_ranked_skill_ids"] == [
        "skill/a",
        "skill/b",
        "skill/c",
    ]
    assert report["dynamic_ranked_skill_ids"] == [
        "skill/b",
        "skill/a",
        "skill/c",
    ]
    assert report["rank_comparison_complete"] is True
    assert report["memory_ranking_applied"] is True
    assert report["memory_ranking_changed"] is True
    assert report["memory_top1_changed"] is True


def test_toolsandbox_selection_diagnostics_expose_noop_filtering(tmp_path) -> None:
    path = tmp_path / "selection.jsonl"
    rows = [
        {
            "record_type": "selection",
            "filter_was_noop": True,
            "filtered_tracked_count": 0,
        },
        {
            "record_type": "selection",
            "filter_was_noop": False,
            "filtered_tracked_count": 2,
        },
        {"record_type": "memory_update"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = _selection_diagnostics(path)

    assert report == {
        "selection_step_count": 2,
        "filter_noop_step_count": 1,
        "filter_noop_step_rate": 0.5,
        "filtered_tracked_tool_total": 2,
        "rank_order_changed_step_count": 0,
        "rank_order_changed_step_rate": 0.0,
        "memory_ranking_applied_step_count": 0,
        "memory_ranking_changed_step_count": 0,
        "memory_ranking_changed_step_rate": 0.0,
        "memory_top1_changed_step_count": 0,
        "memory_top1_changed_step_rate": 0.0,
        "schema_order_preserved_step_count": 0,
        "schema_order_preserved_step_rate": 0.0,
        "clstr_rank_differs_from_input_step_count": 0,
        "clstr_rank_differs_from_input_step_rate": 0.0,
        "guidance_policies": [],
        "dual_expert_guidance_step_count": 0,
        "dual_expert_guidance_step_rate": 0.0,
        "provider_decision_step_count": 0,
        "provider_guidance_step_count": 0,
        "provider_guidance_step_rate": 0.0,
        "provider_guidance_adopted_step_count": 0,
        "provider_guidance_adopted_step_rate": 0.0,
        "provider_top1_guidance_adopted_step_count": 0,
        "provider_posthoc_tool_substitution_step_count": 0,
        "provider_proposed_tool_call_count": 0,
        "provider_observed_result_count": 0,
        "provider_result_coverage_rate": 1.0,
        "selector_intervention_modes": ["candidate_set_filter"],
        "selector_intervention_effective": True,
    }


def test_toolsandbox_selection_diagnostics_links_provider_calls_to_results(
    tmp_path,
) -> None:
    path = tmp_path / "selection.jsonl"
    rows = [
        {
            "record_type": "selection",
            "filter_was_noop": True,
            "filtered_tracked_count": 0,
            "schema_order_preserved": True,
            "clstr_rank_differs_from_input": True,
        },
        {
            "record_type": "provider_decision",
            "provider_guidance_injected": True,
            "provider_guidance_adopted": True,
            "provider_top1_guidance_adopted": True,
            "posthoc_tool_substitution": False,
            "provider_tool_calls": [
                {"tool_call_id": "call-1", "tool_name": "lookup"}
            ],
        },
        {"record_type": "memory_update", "tool_call_id": "call-1"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = _selection_diagnostics(path)

    assert report["selector_intervention_modes"] == [
        "provider_ranked_tool_guidance"
    ]
    assert report["provider_decision_step_count"] == 1
    assert report["schema_order_preserved_step_count"] == 1
    assert report["clstr_rank_differs_from_input_step_count"] == 1
    assert report["provider_guidance_step_count"] == 1
    assert report["provider_guidance_adopted_step_count"] == 1
    assert report["provider_top1_guidance_adopted_step_count"] == 1
    assert report["provider_proposed_tool_call_count"] == 1
    assert report["provider_observed_result_count"] == 1
    assert report["provider_result_coverage_rate"] == 1.0
