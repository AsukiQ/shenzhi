from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import uuid

from openai import OpenAI
from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from clstr.vnext_online_selector import VNEXT_ROUTE_MODES, VNextOnlineSelector
from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser
from tool_sandbox.common.utils import all_logging_disabled


TOOLSANDBOX_INTERVENTION_MODES = frozenset({"schema_order", "ranked_guidance"})
DUAL_EVIDENCE_GUIDANCE_POLICY = "static_anchor_dynamic_supplement_v1"


def _agent_tool_names_in_selected_order(
    *,
    candidate_by_agent_name: dict[str, str],
    selected: list[dict[str, Any]],
) -> list[str]:
    agent_names_by_skill_id: dict[str, list[str]] = {}
    for agent_name, skill_id in candidate_by_agent_name.items():
        agent_names_by_skill_id.setdefault(skill_id, []).append(agent_name)
    return [
        agent_name
        for item in selected
        for agent_name in agent_names_by_skill_id[str(item["skill_id"])]
    ]


def _agent_tool_names_in_score_order(
    *,
    candidate_by_agent_name: dict[str, str],
    selected: list[dict[str, Any]],
    score_key: str,
) -> list[str]:
    ranked = sorted(
        enumerate(selected),
        key=lambda pair: (
            -float(pair[1].get(score_key) or 0.0),
            pair[0],
        ),
    )
    return _agent_tool_names_in_selected_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=[item for _, item in ranked],
    )


def _dual_evidence_guidance_order(
    *,
    candidate_by_agent_name: dict[str, str],
    selected: list[dict[str, Any]],
    route_mode: str,
    history_depth: int,
    guidance_top_k: int,
) -> dict[str, Any]:
    """Preserve current-state evidence while adding one recurrent proposal."""

    selected_names = _agent_tool_names_in_selected_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=selected,
    )
    static_names = _agent_tool_names_in_score_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=selected,
        score_key="static_score",
    )
    dynamic_names = _agent_tool_names_in_score_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=selected,
        score_key="dynamic_score",
    )
    selected_expert = (
        str(selected[0].get("selected_expert") or "static")
        if selected
        else "static"
    )
    apply_dual_evidence = bool(
        str(route_mode).strip().lower() == "adaptive"
        and int(history_depth) > 0
        and selected_expert == "dynamic"
        and static_names
        and dynamic_names
    )
    if apply_dual_evidence:
        # The static top-1 is a current-state anchor.  A distinct recurrent
        # top-1 may supplement it, but cannot remove it.  When both experts
        # agree, fill the remaining budget from the static order so identical
        # evidence does not collapse a top-k prompt to one item.
        evidence_order = [static_names[0], dynamic_names[0], *static_names[1:]]
        evidence_order.extend(dynamic_names[1:])
    else:
        evidence_order = selected_names
    guided_names = list(dict.fromkeys(evidence_order))[
        : max(1, int(guidance_top_k))
    ]
    return {
        "guided_tool_names": guided_names,
        "static_guidance_tool_names": static_names[: max(1, int(guidance_top_k))],
        "dynamic_guidance_tool_names": dynamic_names[
            : max(1, int(guidance_top_k))
        ],
        "guidance_policy": (
            DUAL_EVIDENCE_GUIDANCE_POLICY
            if apply_dual_evidence
            else "selected_expert_topk"
        ),
        "dual_expert_guidance_applied": apply_dual_evidence,
    }


def _ranked_tool_guidance(tool_names: list[str]) -> str:
    unique_names = list(
        dict.fromkeys(
            str(name).strip() for name in tool_names if str(name).strip()
        )
    )
    if not unique_names:
        return ""
    ranked = "\n".join(
        f"{index}. {name}" for index, name in enumerate(unique_names, start=1)
    )
    return (
        "[Trajectory-conditioned tool guidance]\n"
        "This is a soft learned preference, not an instruction to call a tool.\n"
        "Use zero, one, or multiple legal tools as required by the conversation, "
        "and infer every argument from visible evidence.\n"
        "Preferred next tools:\n"
        f"{ranked}"
    )


def _messages_with_ranked_tool_guidance(
    openai_messages: list[dict[str, Any]],
    guidance: str,
) -> list[dict[str, Any]]:
    copied = [dict(message) for message in openai_messages]
    normalized = str(guidance or "").strip()
    if not normalized:
        return copied
    for index, message in enumerate(copied):
        if str(message.get("role") or "") != "system":
            continue
        content = str(message.get("content") or "").strip()
        copied[index]["content"] = (
            f"{content}\n\n{normalized}" if content else normalized
        )
        return copied
    return [{"role": "system", "content": normalized}, *copied]


def _provider_tool_calls(response: Any) -> list[dict[str, str]]:
    choices = list(getattr(response, "choices", None) or [])
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    calls = list(getattr(message, "tool_calls", None) or [])
    return [
        {
            "tool_call_id": str(getattr(call, "id", "") or "").strip(),
            "tool_name": str(
                getattr(getattr(call, "function", None), "name", "") or ""
            ).strip(),
        }
        for call in calls
        if str(getattr(getattr(call, "function", None), "name", "") or "").strip()
    ]


def _provider_tool_names(response: Any) -> list[str]:
    return [item["tool_name"] for item in _provider_tool_calls(response)]


def toolsandbox_current_state(messages: list[Message]) -> str:
    user_messages = [
        str(message.content or "").strip()
        for message in messages
        if message.sender == RoleType.USER and str(message.content or "").strip()
    ]
    if not user_messages:
        visible = [
            str(message.content or "").strip()
            for message in messages
            if str(message.content or "").strip()
        ]
        if not visible:
            raise ValueError("ToolSandbox selection has no agent-visible state")
        user_messages = [visible[0]]
    lines = ["goal: user_messages:", *user_messages]
    observations = [
        str(message.content or "").strip()
        for message in messages
        if message.sender == RoleType.EXECUTION_ENVIRONMENT
        and message.recipient == RoleType.AGENT
        and str(message.content or "").strip()
    ]
    if observations:
        lines.extend(("observation: execution_environment:", observations[-1]))
    return "\n".join(lines).strip()


def _retained_tools_in_selection_order(
    *,
    available: dict[str, Callable[..., Any]],
    candidate_by_agent_name: dict[str, str],
    passthrough_names: list[str],
    selected: list[dict[str, Any]],
) -> tuple[dict[str, Callable[..., Any]], int]:
    ranked_tracked_names = _agent_tool_names_in_selected_order(
        candidate_by_agent_name=candidate_by_agent_name,
        selected=selected,
    )
    retained_names = ranked_tracked_names + passthrough_names
    retained = {agent_name: available[agent_name] for agent_name in retained_names}
    return retained, len(candidate_by_agent_name) - len(ranked_tracked_names)


def _selection_order_diagnostics(
    *,
    candidate_skill_ids: list[str],
    selected: list[dict[str, Any]],
    history_depth: int,
) -> dict[str, Any]:
    """Describe rank intervention without requiring unsafe tool deletion."""

    input_ids = [str(item) for item in candidate_skill_ids]
    selected_ids = [str(item.get("skill_id") or "") for item in selected]
    selected_set = set(selected_ids)
    input_retained_ids = [item for item in input_ids if item in selected_set]
    input_positions = {skill_id: index for index, skill_id in enumerate(input_ids)}

    def ranked_ids(score_key: str) -> list[str]:
        indexed = list(enumerate(selected))
        ranked = sorted(
            indexed,
            key=lambda pair: (
                -float(pair[1].get(score_key) or 0.0),
                input_positions.get(
                    str(pair[1].get("skill_id") or ""),
                    len(input_positions) + pair[0],
                ),
            ),
        )
        return [str(item.get("skill_id") or "") for _, item in ranked]

    static_order = ranked_ids("static_score")
    dynamic_order = ranked_ids("dynamic_score")
    selected_expert = (
        str(selected[0].get("selected_expert") or "static")
        if selected
        else "static"
    )
    memory_applied = int(history_depth) > 0 and selected_expert == "dynamic"
    rank_comparison_complete = (
        len(selected_ids) == len(input_ids)
        and selected_set == set(input_ids)
    )
    static_dynamic_order_changed = static_order != dynamic_order
    static_dynamic_top1_changed = bool(
        static_order
        and dynamic_order
        and static_order[0] != dynamic_order[0]
    )
    return {
        "input_candidate_skill_ids": input_ids,
        "retained_candidate_skill_ids": selected_ids,
        "rank_order_changed": selected_ids != input_retained_ids,
        "static_ranked_skill_ids": static_order,
        "dynamic_ranked_skill_ids": dynamic_order,
        "rank_comparison_complete": rank_comparison_complete,
        "memory_ranking_applied": memory_applied,
        "memory_ranking_changed": bool(
            memory_applied
            and rank_comparison_complete
            and static_dynamic_order_changed
        ),
        "memory_top1_changed": bool(
            memory_applied
            and rank_comparison_complete
            and static_dynamic_top1_changed
        ),
    }


def _deterministic_completion_kwargs(
    *,
    model_name: str,
    openai_messages: list[dict[str, Any]],
    openai_tools: Any,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": openai_messages,
        "tools": openai_tools,
        "temperature": float(temperature),
        "max_tokens": max(1, int(max_tokens)),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": bool(enable_thinking),
            }
        },
    }


def _deterministic_user_completion_kwargs(
    *,
    model_name: str,
    openai_messages: list[dict[str, Any]],
    openai_tools: Any,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": openai_messages,
        "tools": openai_tools,
        "temperature": float(temperature),
        "max_tokens": max(1, int(max_tokens)),
    }


class CLSTRVNextToolSandboxUser(OpenAIAPIUser):
    """Official OpenAI user role with explicit matched sampling controls."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.max_tokens = max(1, int(max_tokens))

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def model_inference(self, openai_messages, openai_tools):
        kwargs = _deterministic_user_completion_kwargs(
            model_name=self.model_name,
            openai_messages=openai_messages,
            openai_tools=openai_tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(**kwargs)


class CLSTRVNextToolSandboxAgent(OpenAIAPIAgent):
    """Official ToolSandbox agent with recurrent CLSTR provider guidance."""

    def __init__(
        self,
        *,
        selector: VNextOnlineSelector,
        model_name: str,
        base_url: str,
        api_key: str,
        top_k: int,
        selection_log_path: str | Path,
        route_mode: str = "adaptive",
        executor_temperature: float = 0.0,
        executor_max_tokens: int = 1024,
        executor_enable_thinking: bool = False,
        intervention_mode: str = "ranked_guidance",
        guidance_top_k: int = 2,
    ) -> None:
        self.selector = selector
        self.session = selector.new_session()
        self.model_name = model_name
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key)
        self.executor_temperature = float(executor_temperature)
        self.executor_max_tokens = max(1, int(executor_max_tokens))
        self.executor_enable_thinking = bool(executor_enable_thinking)
        self.top_k = max(1, int(top_k))
        self.selection_log_path = Path(selection_log_path)
        self.route_mode = str(route_mode).strip().lower()
        if self.route_mode not in VNEXT_ROUTE_MODES:
            raise ValueError(f"unsupported vNext route mode: {route_mode}")
        self.intervention_mode = str(intervention_mode).strip().lower()
        if self.intervention_mode not in TOOLSANDBOX_INTERVENTION_MODES:
            raise ValueError(
                f"unsupported ToolSandbox intervention mode: {intervention_mode}"
            )
        self.guidance_top_k = max(1, int(guidance_top_k))
        self.scenario_name: str | None = None
        self.session_id = uuid.uuid4().hex
        self._consumed_tool_call_ids: set[str] = set()
        self._latest_guidance_context: dict[str, Any] = {}

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(self, openai_messages, openai_tools):
        """Match the official agent call while making executor sampling explicit."""

        guidance = str(self._latest_guidance_context.get("guidance") or "")
        guided_messages = _messages_with_ranked_tool_guidance(
            openai_messages,
            guidance if self.intervention_mode == "ranked_guidance" else "",
        )
        kwargs = _deterministic_completion_kwargs(
            model_name=self.model_name,
            openai_messages=guided_messages,
            openai_tools=openai_tools,
            temperature=self.executor_temperature,
            max_tokens=self.executor_max_tokens,
            enable_thinking=self.executor_enable_thinking,
        )
        with all_logging_disabled():
            response = self.openai_client.chat.completions.create(**kwargs)
        provider_tool_calls = _provider_tool_calls(response)
        provider_tool_names = _provider_tool_names(response)
        guided_tool_names = list(
            self._latest_guidance_context.get("guided_tool_names") or []
        )
        guided_set = set(guided_tool_names)
        self._append_log(
            {
                "record_type": "provider_decision",
                "intervention_mode": self.intervention_mode,
                "history_depth": int(
                    self._latest_guidance_context.get("history_depth") or 0
                ),
                "selected_expert": self._latest_guidance_context.get(
                    "selected_expert"
                ),
                "guidance_policy": self._latest_guidance_context.get(
                    "guidance_policy"
                ),
                "dual_expert_guidance_applied": bool(
                    self._latest_guidance_context.get(
                        "dual_expert_guidance_applied"
                    )
                ),
                "static_guidance_tool_names": list(
                    self._latest_guidance_context.get(
                        "static_guidance_tool_names"
                    )
                    or []
                ),
                "dynamic_guidance_tool_names": list(
                    self._latest_guidance_context.get(
                        "dynamic_guidance_tool_names"
                    )
                    or []
                ),
                "provider_guidance_injected": bool(
                    self.intervention_mode == "ranked_guidance" and guidance
                ),
                "guided_tool_names": guided_tool_names,
                "provider_tool_calls": provider_tool_calls,
                "provider_tool_names": provider_tool_names,
                "pre_execution_tool_names": provider_tool_names,
                "decision_surface": "official_provider_response_before_execution",
                "provider_tool_call_count": len(provider_tool_names),
                "provider_no_tool": not provider_tool_names,
                "provider_parallel_tool_call": len(provider_tool_names) > 1,
                "provider_guidance_adopted": bool(
                    guided_set.intersection(provider_tool_names)
                ),
                "provider_top1_guidance_adopted": bool(
                    guided_tool_names
                    and guided_tool_names[0] in set(provider_tool_names)
                ),
                "posthoc_tool_substitution": False,
            }
        )
        return response

    def _append_log(self, payload: dict[str, Any]) -> None:
        self.selection_log_path.parent.mkdir(parents=True, exist_ok=True)
        attributed = {
            "session_id": self.session_id,
            "scenario_name": self.scenario_name,
            **payload,
        }
        with self.selection_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attributed, ensure_ascii=False) + "\n")

    def begin_scenario(self, scenario_name: str) -> None:
        if self.session.history_depth or self._consumed_tool_call_ids:
            raise RuntimeError("ToolSandbox scenario began before the agent was reset")
        self.scenario_name = str(scenario_name)
        self.session_id = uuid.uuid4().hex
        self._latest_guidance_context = {}

    def finalize_scenario(self) -> None:
        """Consume a terminal tool result that did not trigger another agent turn."""

        messages = self.filter_messages(self.get_messages())
        self._consume_new_results(messages)

    def _consume_new_results(self, messages: list[Message]) -> None:
        calls: dict[str, tuple[int, Message]] = {}
        for index, message in enumerate(messages):
            call_id = str(message.openai_tool_call_id or "").strip()
            if (
                call_id
                and message.sender == RoleType.AGENT
                and message.recipient == RoleType.EXECUTION_ENVIRONMENT
            ):
                calls[call_id] = (index, message)
        context = get_current_context()
        for result in messages:
            call_id = str(result.openai_tool_call_id or "").strip()
            if (
                not call_id
                or call_id in self._consumed_tool_call_ids
                or result.sender != RoleType.EXECUTION_ENVIRONMENT
                or result.recipient != RoleType.AGENT
            ):
                continue
            call_record = calls.get(call_id)
            if call_record is None:
                continue
            call_index, call = call_record
            agent_name = str(call.openai_function_name or "").strip()
            execution_name = context.get_execution_facing_tool_name(agent_name)
            skill_id = f"toolsandbox/{execution_name}"
            if skill_id in self.selector.skill_index:
                update = self.session.observe(
                    state_text_before=toolsandbox_current_state(messages[:call_index]),
                    skill_id=skill_id,
                    action_text=str(call.content or agent_name).strip(),
                    result_text=str(result.content or "").strip(),
                )
                self._append_log(
                    {
                        "record_type": "memory_update",
                        "tool_call_id": call_id,
                        "agent_tool_name": agent_name,
                        "execution_tool_name": execution_name,
                        **update,
                    }
                )
            else:
                self._append_log(
                    {
                        "record_type": "untracked_tool_result",
                        "tool_call_id": call_id,
                        "agent_tool_name": agent_name,
                        "execution_tool_name": execution_name,
                    }
                )
            self._consumed_tool_call_ids.add(call_id)

    def get_available_tools(self) -> dict[str, Callable[..., Any]]:
        available = super().get_available_tools()
        if not available:
            self._latest_guidance_context = {}
            return available
        messages = self.filter_messages(self.get_messages())
        self._consume_new_results(messages)
        context = get_current_context()
        candidate_by_agent_name: dict[str, str] = {}
        passthrough_names: list[str] = []
        for agent_name in available:
            execution_name = context.get_execution_facing_tool_name(agent_name)
            skill_id = f"toolsandbox/{execution_name}"
            if skill_id in self.selector.skill_index:
                candidate_by_agent_name[agent_name] = skill_id
            else:
                passthrough_names.append(agent_name)
        if not candidate_by_agent_name:
            self._latest_guidance_context = {}
            return available
        current_state = toolsandbox_current_state(messages)
        candidate_skill_ids = list(dict.fromkeys(candidate_by_agent_name.values()))
        selected = self.session.select(
            current_state,
            candidate_skill_ids=candidate_skill_ids,
            top_k=(
                len(candidate_skill_ids)
                if self.intervention_mode == "ranked_guidance"
                else min(self.top_k, len(candidate_skill_ids))
            ),
            route_mode=self.route_mode,
        )
        if self.intervention_mode == "ranked_guidance":
            retained = dict(available)
            filtered_tracked_count = 0
        else:
            retained, filtered_tracked_count = _retained_tools_in_selection_order(
                available=available,
                candidate_by_agent_name=candidate_by_agent_name,
                passthrough_names=passthrough_names,
                selected=selected,
            )
        selected_expert = (
            str(selected[0].get("selected_expert") or "static")
            if selected
            else "static"
        )
        guidance_context = _dual_evidence_guidance_order(
            candidate_by_agent_name=candidate_by_agent_name,
            selected=selected,
            route_mode=self.route_mode,
            history_depth=self.session.history_depth,
            guidance_top_k=self.guidance_top_k,
        )
        guided_tool_names = list(guidance_context["guided_tool_names"])
        guidance = (
            _ranked_tool_guidance(guided_tool_names)
            if self.intervention_mode == "ranked_guidance"
            else ""
        )
        self._latest_guidance_context = {
            "guidance": guidance,
            "history_depth": self.session.history_depth,
            "selected_expert": selected_expert,
            **guidance_context,
        }
        order_diagnostics = _selection_order_diagnostics(
            candidate_skill_ids=candidate_skill_ids,
            selected=selected,
            history_depth=self.session.history_depth,
        )
        order_diagnostics["clstr_rank_differs_from_input"] = bool(
            order_diagnostics["rank_order_changed"]
        )
        order_diagnostics["schema_order_preserved"] = bool(
            self.intervention_mode == "ranked_guidance"
        )
        if self.intervention_mode == "ranked_guidance":
            order_diagnostics["rank_order_changed"] = False
            order_diagnostics["retained_candidate_skill_ids"] = list(
                candidate_skill_ids
            )
        self._append_log(
            {
                "record_type": "selection",
                "method": self.selector.method,
                "route_mode": self.route_mode,
                "intervention_mode": self.intervention_mode,
                "message_count": len(messages),
                "history_depth": self.session.history_depth,
                "candidate_skill_ids": candidate_skill_ids,
                "tracked_candidate_count": len(candidate_by_agent_name),
                "selected_tracked_count": (
                    len(candidate_by_agent_name) - filtered_tracked_count
                ),
                "filtered_tracked_count": filtered_tracked_count,
                "filter_was_noop": filtered_tracked_count == 0,
                "passthrough_tool_names": passthrough_names,
                "provider_guidance_injected": bool(guidance),
                "guided_tool_names": guided_tool_names,
                "guidance_top_k": self.guidance_top_k,
                "guidance_policy": guidance_context["guidance_policy"],
                "dual_expert_guidance_applied": guidance_context[
                    "dual_expert_guidance_applied"
                ],
                "static_guidance_tool_names": guidance_context[
                    "static_guidance_tool_names"
                ],
                "dynamic_guidance_tool_names": guidance_context[
                    "dynamic_guidance_tool_names"
                ],
                "selected": [
                    {
                        key: item[key]
                        for key in (
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
                    }
                    for item in selected
                ],
                "retained_agent_tool_names": list(retained),
                **order_diagnostics,
            }
        )
        return retained

    def reset(self) -> None:
        self.session.reset()
        self._consumed_tool_call_ids.clear()
        self._latest_guidance_context = {}
        self.scenario_name = None
        self.session_id = uuid.uuid4().hex
        super().reset()
