from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Optional
import uuid

from clstr.vnext_online_selector import VNEXT_ROUTE_MODES, VNextOnlineSelector
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


_SELECTION_LOG_LOCK = threading.Lock()


def _tools_in_selection_order(
    selected: list[dict[str, Any]],
    tool_by_skill_id: dict[str, Tool],
) -> list[Tool]:
    return [tool_by_skill_id[str(item["skill_id"])] for item in selected]


def _message_observation(message: Any) -> str:
    role = str(getattr(message, "role", "message"))
    content = str(getattr(message, "content", "") or "").strip()
    if content:
        return f"{role}: {content}"
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if tool_calls:
        calls = ", ".join(
            f"{call.name}({json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)})"
            for call in tool_calls
        )
        return f"{role}: tool_calls: {calls}"
    return role


def tau2_current_state(messages: list[Any], *, domain: str) -> str:
    if not messages:
        raise ValueError("Tau2 selection has no agent-visible messages")
    user_messages = [
        str(getattr(message, "content", "") or "").strip()
        for message in messages
        if getattr(message, "role", None) == "user"
        and str(getattr(message, "content", "") or "").strip()
    ]
    goal = user_messages[0] if user_messages else _message_observation(messages[0])
    return (
        f"goal: {goal}\n"
        f"task: Tau2 customer-service domain: {domain}\n"
        f"observation: {_message_observation(messages[-1])}"
    )


def _infer_domain(selector: VNextOnlineSelector, tools: list[Tool]) -> str:
    tool_names = {str(tool.name) for tool in tools}
    overlap_by_domain: dict[str, int] = {}
    for skill_id in selector.skill_ids:
        parts = str(skill_id).split("/")
        if len(parts) != 3 or parts[0] != "tau2" or parts[2] == "__start__":
            continue
        domain, action = parts[1], parts[2]
        if action in tool_names:
            overlap_by_domain[domain] = overlap_by_domain.get(domain, 0) + 1
    if not overlap_by_domain:
        raise ValueError("CLSTR Tau2 agent cannot infer a domain from available tools")
    ranked = sorted(overlap_by_domain.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError(
            "CLSTR Tau2 domain inference is ambiguous: "
            + json.dumps(overlap_by_domain, sort_keys=True)
        )
    return ranked[0][0]


class CLSTRVNextTau2Agent(LLMAgent):
    """Official Tau2 LLMAgent with recurrent per-turn CLSTR tool routing."""

    def __init__(
        self,
        *,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict],
        selector: VNextOnlineSelector,
        top_k: int,
        selection_log_path: str | Path,
        route_mode: str = "adaptive",
        task_id: str | None = None,
    ) -> None:
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.selector = selector
        self.session = selector.new_session()
        self.top_k = max(1, int(top_k))
        self.selection_log_path = Path(selection_log_path)
        self.route_mode = str(route_mode).strip().lower()
        if self.route_mode not in VNEXT_ROUTE_MODES:
            raise ValueError(f"unsupported vNext route mode: {route_mode}")
        self.task_id = None if task_id is None else str(task_id)
        self.session_id = uuid.uuid4().hex
        self.domain = _infer_domain(selector, tools)
        self.skill_id_by_tool_name = {
            str(tool.name): f"tau2/{self.domain}/{tool.name}" for tool in tools
        }
        self.tool_by_skill_id = {
            self.skill_id_by_tool_name[str(tool.name)]: tool for tool in tools
        }
        missing = [
            skill_id
            for skill_id in self.skill_id_by_tool_name.values()
            if skill_id not in selector.skill_index
        ]
        if missing:
            raise ValueError(
                "Tau2 environment tools are absent from the CLSTR release: "
                + ", ".join(missing[:10])
            )
        self._pending_calls: dict[str, dict[str, str]] = {}

    def _append_log(self, payload: dict[str, Any]) -> None:
        self.selection_log_path.parent.mkdir(parents=True, exist_ok=True)
        attributed = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "domain": self.domain,
            **payload,
        }
        with _SELECTION_LOG_LOCK, self.selection_log_path.open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(attributed, ensure_ascii=False) + "\n")

    def _consume_tool_results(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
    ) -> None:
        if isinstance(message, MultiToolMessage):
            tool_messages = list(message.tool_messages)
        elif isinstance(message, ToolMessage):
            tool_messages = [message]
        else:
            return
        for tool_message in tool_messages:
            pending = self._pending_calls.pop(str(tool_message.id), None)
            if pending is None:
                raise RuntimeError(
                    "Tau2 tool result has no CLSTR-aligned pending call: "
                    f"{tool_message.id}"
                )
            update = self.session.observe(
                state_text_before=pending["state_text_before"],
                skill_id=pending["skill_id"],
                action_text=pending["action_text"],
                result_text=str(tool_message.content or "").strip(),
            )
            self._append_log(
                {
                    "record_type": "memory_update",
                    "tool_call_id": str(tool_message.id),
                    "tool_error": bool(tool_message.error),
                    **update,
                }
            )

    def _generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: LLMAgentState,
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio in Tau2 text evaluation")
        self._consume_tool_results(message)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        current_state = tau2_current_state(state.messages, domain=self.domain)
        candidate_skill_ids = list(self.skill_id_by_tool_name.values())
        selected = self.session.select(
            current_state,
            candidate_skill_ids=candidate_skill_ids,
            top_k=min(self.top_k, len(candidate_skill_ids)),
            route_mode=self.route_mode,
        )
        selected_ids = {str(item["skill_id"]) for item in selected}
        selected_tools = _tools_in_selection_order(selected, self.tool_by_skill_id)
        self._append_log(
            {
                "record_type": "selection",
                "method": self.selector.method,
                "route_mode": self.route_mode,
                "message_count": len(state.messages),
                "history_depth": self.session.history_depth,
                "candidate_skill_ids": candidate_skill_ids,
                "selected": [self._selection_log_item(item) for item in selected],
            }
        )
        assistant_message = generate(
            model=self.llm,
            tools=selected_tools,
            messages=state.system_messages + state.messages,
            call_name="clstr_vnext_agent_response",
            **self.llm_args,
        )
        for call in list(assistant_message.tool_calls or []):
            tool_name = str(call.name)
            skill_id = self.skill_id_by_tool_name.get(tool_name)
            if skill_id is None or skill_id not in selected_ids:
                raise RuntimeError(
                    "Tau2 executor emitted a tool outside the CLSTR-selected set: "
                    f"{tool_name}"
                )
            call_id = str(call.id or "").strip()
            if not call_id:
                raise RuntimeError("Tau2 executor emitted a tool call without an ID")
            self._pending_calls[call_id] = {
                "state_text_before": current_state,
                "skill_id": skill_id,
                "action_text": (
                    f"{tool_name}("
                    f"{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
                    ")"
                ),
            }
        return assistant_message

    @staticmethod
    def _selection_log_item(item: dict[str, Any]) -> dict[str, Any]:
        required = (
            "rank",
            "skill_id",
            "score",
            "static_score",
            "dynamic_score",
            "selector_probability",
            "mixture_probability",
            "adaptive_mixture_probability",
            "selected_expert",
        )
        payload = {key: item[key] for key in required}
        for key in (
            "encoder_kind",
            "candidate_count",
            "natural_support_count",
            "static_support_count",
            "static_support_sha256",
            "history_window_depth",
        ):
            if key in item:
                payload[key] = item[key]
        return payload
