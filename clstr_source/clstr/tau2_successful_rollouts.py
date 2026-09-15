from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from clstr.history_channel import materialize_structured_current_state


TAU2_ROLLOUT_DOMAINS = ("airline", "retail", "telecom")
TAU2_DEFAULT_AGENT_MODELS = (
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16",
    "claude-3-7-sonnet-20250219",
)
TAU2_SUCCESSFUL_ROLLOUT_SOURCE_ID = "tau2_official_successful_rollout_v1"
TAU2_SUCCESSFUL_ROLLOUT_OBSERVATION_SOURCE = (
    "tau2_official_successful_rollout_tool_result"
)


@dataclass(frozen=True)
class Tau2SuccessfulRolloutCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_with_sha256(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_domain(value: Any) -> str:
    return _clean_text(value).lower()


def _action_skill_id(domain: str, action_name: str) -> str:
    return f"tau2/{_clean_domain(domain)}/{_clean_text(action_name)}"


def _start_skill_id(domain: str) -> str:
    return f"tau2/{_clean_domain(domain)}/__start__"


def _domain_dir(data_root: Path, domain: str) -> Path:
    return data_root / "domains" / _clean_domain(domain)


def _domain_policy(domain_dir: Path) -> str:
    for filename in ("policy.md", "main_policy.md", "policy_solo.md"):
        path = domain_dir / filename
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return ""


def _official_split_ids(data_root: Path, domain: str, task_split: str) -> set[str]:
    split_path = _domain_dir(data_root, domain) / "split_tasks.json"
    payload = _read_json(split_path)
    if not isinstance(payload, dict) or not isinstance(payload.get(task_split), list):
        raise ValueError(f"Tau2 {domain} split is missing: {task_split}")
    values = {_clean_text(item) for item in payload[task_split] if _clean_text(item)}
    if len(values) != len(payload[task_split]):
        raise ValueError(f"Tau2 {domain} {task_split} split has invalid/duplicate IDs")
    return values


def _task_action_names(data_root: Path, domain: str) -> set[str]:
    tasks = _read_json(_domain_dir(data_root, domain) / "tasks.json")
    if not isinstance(tasks, list):
        raise ValueError(f"Tau2 {domain} tasks.json must contain a list")
    names: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        criteria = task.get("evaluation_criteria")
        criteria = criteria if isinstance(criteria, dict) else {}
        actions = criteria.get("actions")
        actions = actions if isinstance(actions, list) else []
        for action in actions:
            if isinstance(action, dict) and _clean_text(action.get("name")):
                names.add(_clean_text(action["name"]))
    return names


def _skill_row(domain: str, action_name: str | None, *, policy_text: str) -> dict[str, Any]:
    domain = _clean_domain(domain)
    if action_name is None:
        description = f"Initial Tau2 {domain} state before an executable tool call."
        return {
            "skill_id": _start_skill_id(domain),
            "name": f"{domain}.START",
            "description": description,
            "executor_desc": description,
            "body": description,
            "skill_md": description,
            "source_benchmark": "tau2",
            "domain": domain,
        }
    action_name = _clean_text(action_name)
    description = f"Use Tau2 {domain} API action `{action_name}` when the visible dialogue state requires it."
    body = (
        f"Action name: {action_name}\nDomain: {domain}\n"
        f"Agent-visible policy excerpt:\n{policy_text[:2000]}"
    )
    return {
        "skill_id": _action_skill_id(domain, action_name),
        "name": f"{domain}.{action_name}",
        "description": description,
        "executor_desc": description,
        "body": body,
        "skill_md": body,
        "source_benchmark": "tau2",
        "domain": domain,
    }


def load_tau2_skill_catalog(
    data_root: str | Path,
    *,
    domains: Iterable[str] = TAU2_ROLLOUT_DOMAINS,
) -> list[dict[str, Any]]:
    """Build the skill catalog only; reference actions never become route labels."""

    root = Path(data_root).resolve()
    rows: list[dict[str, Any]] = []
    for raw_domain in domains:
        domain = _clean_domain(raw_domain)
        policy = _domain_policy(_domain_dir(root, domain))
        rows.append(_skill_row(domain, None, policy_text=policy))
        rows.extend(
            _skill_row(domain, action_name, policy_text=policy)
            for action_name in sorted(_task_action_names(root, domain))
        )
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        skill_id = _clean_text(row.get("skill_id"))
        if skill_id and skill_id not in seen:
            seen.add(skill_id)
            output.append(row)
    return output


def _result_file(
    results_root: Path,
    *,
    agent_model: str,
    domain: str,
) -> Path:
    matches = sorted(
        results_root.glob(f"{agent_model}_{domain}_default_*_4trials.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one Tau2 default result for {agent_model}/{domain}, found {len(matches)}"
        )
    return matches[0]


def _reward(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info")
    reward_info = reward_info if isinstance(reward_info, dict) else {}
    try:
        return float(reward_info.get("reward") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _message_content(message: dict[str, Any]) -> str:
    value = message.get("content")
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _initial_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if _clean_text(message.get("role")) == "user" and _message_content(message):
            return _message_content(message)
    return ""


def _latest_visible_user_message(
    messages: list[dict[str, Any]],
) -> str:
    """Return the latest user turn without copying tool results into ``c_t``.

    Tau2 agents can observe tool results, but CLSTR deliberately assigns that
    evidence to recurrent-memory correction.  Reusing the latest generic
    external message here would place the same raw tool result in both the
    common static/dynamic route query and ``m_t``, invalidating the intended
    memory-only result channel.
    """

    for message in reversed(messages):
        role = _clean_text(message.get("role"))
        content = _message_content(message)
        if role != "user" or not content:
            continue
        return content
    return ""


def _tool_results_by_id(
    messages: list[dict[str, Any]],
) -> dict[str, tuple[int, dict[str, Any]]]:
    output: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, message in enumerate(messages):
        if _clean_text(message.get("role")) != "tool":
            continue
        result_id = _clean_text(message.get("id"))
        if not result_id:
            continue
        if result_id in output:
            raise ValueError(f"Tau2 rollout contains duplicate tool result ID: {result_id}")
        output[result_id] = (index, message)
    return output


def _extract_simulation_events(
    simulation: dict[str, Any],
    *,
    domain: str,
) -> tuple[list[dict[str, Any]], str | None]:
    raw_messages = simulation.get("messages")
    if not isinstance(raw_messages, list):
        return [], "messages_not_list"
    messages = [message for message in raw_messages if isinstance(message, dict)]
    initial_user = _initial_user_message(messages)
    if not initial_user:
        return [], "missing_visible_user_request"
    try:
        result_by_id = _tool_results_by_id(messages)
    except ValueError:
        return [], "duplicate_tool_result_id"
    events: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if _clean_text(message.get("role")) != "assistant":
            continue
        calls = message.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        calls = [call for call in calls if isinstance(call, dict)]
        if not calls:
            continue
        if len(calls) != 1:
            return [], "parallel_tool_calls_unsupported"
        call = calls[0]
        call_id = _clean_text(call.get("id"))
        action_name = _clean_text(call.get("name"))
        arguments = call.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        if not call_id or not action_name:
            return [], "invalid_tool_call"
        matched = result_by_id.get(call_id)
        if matched is None or matched[0] <= message_index:
            return [], "missing_causal_tool_result"
        result_index, result_message = matched
        result_text = _message_content(result_message)
        if not result_text:
            return [], "empty_causal_tool_result"
        current_observation = _latest_visible_user_message(
            messages[:message_index]
        )
        if not current_observation:
            return [], "missing_visible_user_state"
        structured = materialize_structured_current_state(
            {
                "current_state_components": {
                    "goal_text": initial_user,
                    "task_text": f"Tau2 customer-service domain: {domain}",
                    "current_observation_text": f"user: {current_observation}",
                }
            },
            replace_state_text=True,
        )
        events.append(
            {
                "action_name": action_name,
                "action_arguments": arguments,
                "action_text": (
                    f"tool: {action_name}\narguments: "
                    + json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                ),
                "result_text": result_text,
                "result_error": bool(result_message.get("error")),
                "call_id": call_id,
                "result_message_index": result_index,
                "state_event_id": "",
                "state_text_current": str(structured["state_text_current"]),
                "current_state_components": dict(
                    structured["current_state_components"]
                ),
                "initial_user_message": initial_user,
            }
        )
    if not events:
        return [], "successful_no_tool_trajectory"
    return events, None


def _trajectory_identity(
    *,
    domain: str,
    task_id: str,
    agent_model: str,
    simulation: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "domain": domain,
            "task_id": task_id,
            "agent_model": agent_model,
            "simulation_id": simulation.get("id"),
            "trial": simulation.get("trial"),
            "seed": simulation.get("seed"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"tau2/{domain}/{task_id}/successful/{suffix}"


def load_tau2_successful_rollout_corpus(
    tau2_data_root: str | Path,
    *,
    results_root: str | Path | None = None,
    domains: Iterable[str] = TAU2_ROLLOUT_DOMAINS,
    task_split: str = "train",
    agent_models: Iterable[str] = TAU2_DEFAULT_AGENT_MODELS,
    minimum_reward: float = 1.0,
) -> Tau2SuccessfulRolloutCorpus:
    """Build positive next-tool supervision from successful, agent-visible Tau2 logs.

    Only official split IDs, visible transcript messages, actual assistant tool
    calls, and matching tool results are read.  `user_scenario` and task
    instructions are never copied into model inputs.  Multiple successful paths
    sharing one task/tool prefix become multi-positive next-skill supervision.
    """

    data_root = Path(tau2_data_root).resolve()
    resolved_results_root = (
        Path(results_root).resolve()
        if results_root is not None
        else data_root / "results" / "final"
    )
    task_split = _clean_text(task_split).lower()
    if task_split not in {"train", "test"}:
        raise ValueError("successful Tau2 rollout supervision supports train/test only")
    resolved_domains = tuple(_clean_domain(domain) for domain in domains)
    resolved_models = tuple(_clean_text(model) for model in agent_models)
    skills = load_tau2_skill_catalog(data_root, domains=resolved_domains)
    skills_by_id = {_clean_text(row.get("skill_id")): row for row in skills}
    accepted: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    per_domain: dict[str, Counter[str]] = {
        domain: Counter() for domain in resolved_domains
    }
    result_files: list[dict[str, Any]] = []
    official_task_count = 0
    official_split_task_ids_by_domain: dict[str, list[str]] = {}
    for domain in resolved_domains:
        split_ids = _official_split_ids(data_root, domain, task_split)
        official_split_task_ids_by_domain[domain] = sorted(split_ids)
        official_task_count += len(split_ids)
        for agent_model in resolved_models:
            path = _result_file(
                resolved_results_root,
                agent_model=agent_model,
                domain=domain,
            )
            payload, result_sha256 = _read_json_with_sha256(path)
            simulations = payload.get("simulations") if isinstance(payload, dict) else None
            if not isinstance(simulations, list):
                raise ValueError(f"Tau2 result file lacks simulations: {path}")
            result_files.append(
                {
                    "path": str(path),
                    "sha256": result_sha256,
                    "domain": domain,
                    "agent_model": agent_model,
                }
            )
            for simulation in simulations:
                if not isinstance(simulation, dict):
                    skipped["simulation_not_object"] += 1
                    continue
                task_id = _clean_text(simulation.get("task_id"))
                if task_id not in split_ids:
                    continue
                per_domain[domain]["split_simulations"] += 1
                if _reward(simulation) < float(minimum_reward):
                    skipped["reward_below_threshold"] += 1
                    per_domain[domain]["reward_below_threshold"] += 1
                    continue
                events, blocker = _extract_simulation_events(
                    simulation,
                    domain=domain,
                )
                if blocker is not None:
                    skipped[blocker] += 1
                    per_domain[domain][blocker] += 1
                    continue
                trajectory_id = _trajectory_identity(
                    domain=domain,
                    task_id=task_id,
                    agent_model=agent_model,
                    simulation=simulation,
                )
                accepted.append(
                    {
                        "domain": domain,
                        "task_id": task_id,
                        "agent_model": agent_model,
                        "simulation": simulation,
                        "trajectory_id": trajectory_id,
                        "events": events,
                        "result_file": str(path),
                    }
                )
                per_domain[domain]["successful_tool_trajectories"] += 1
                per_domain[domain]["successful_tool_events"] += len(events)

    observed_names_by_domain: dict[str, set[str]] = defaultdict(set)
    positive_names_by_prefix: dict[tuple[str, str, tuple[str, ...]], set[str]] = (
        defaultdict(set)
    )
    for trajectory in accepted:
        prefix: list[str] = []
        for event in trajectory["events"]:
            action_name = _clean_text(event["action_name"])
            observed_names_by_domain[trajectory["domain"]].add(action_name)
            positive_names_by_prefix[
                (trajectory["domain"], trajectory["task_id"], tuple(prefix))
            ].add(action_name)
            prefix.append(action_name)
    for domain, names in sorted(observed_names_by_domain.items()):
        policy = _domain_policy(_domain_dir(data_root, domain))
        for action_name in sorted(names):
            skill_id = _action_skill_id(domain, action_name)
            if skill_id not in skills_by_id:
                row = _skill_row(domain, action_name, policy_text=policy)
                skills.append(row)
                skills_by_id[skill_id] = row

    candidate_ids_by_domain = {
        domain: sorted(
            skill_id
            for skill_id in skills_by_id
            if skill_id.startswith(f"tau2/{domain}/")
            and not skill_id.endswith("/__start__")
        )
        for domain in resolved_domains
    }
    source_rows: list[dict[str, Any]] = []
    multi_positive_rows = 0
    tasks_with_rows: set[tuple[str, str]] = set()
    distinct_paths_by_task: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for trajectory in accepted:
        domain = trajectory["domain"]
        raw_task_id = trajectory["task_id"]
        events = trajectory["events"]
        trajectory_id = trajectory["trajectory_id"]
        path = tuple(_clean_text(event["action_name"]) for event in events)
        distinct_paths_by_task[(domain, raw_task_id)].add(path)
        tasks_with_rows.add((domain, raw_task_id))
        prefix: list[str] = []
        previous_skill_id = _start_skill_id(domain)
        simulation = trajectory["simulation"]
        for step_index, event in enumerate(events):
            action_name = _clean_text(event["action_name"])
            target_skill_id = _action_skill_id(domain, action_name)
            equivalent = sorted(
                _action_skill_id(domain, name)
                for name in positive_names_by_prefix[
                    (domain, raw_task_id, tuple(prefix))
                ]
            )
            multi_positive_rows += int(len(equivalent) > 1)
            source_rows.append(
                {
                    "benchmark": "tau2",
                    "source_benchmark": "tau2",
                    "split": task_split,
                    "domain": domain,
                    "task_id": f"{trajectory_id}::{step_index}",
                    "trajectory_id": trajectory_id,
                    "step_index": step_index,
                    "state_text": event["state_text_current"],
                    "state_text_current": event["state_text_current"],
                    "state_text_full": event["state_text_current"],
                    "current_state_components": event["current_state_components"],
                    "history_text": "",
                    "action_text": (
                        "previous_tool: START"
                        if step_index == 0
                        else f"previous_tool: {prefix[-1]}"
                    ),
                    "target_action_text": event["action_text"],
                    "split_semantic_text": event["initial_user_message"],
                    "next_observation_text": event["result_text"],
                    "observation_source": TAU2_SUCCESSFUL_ROLLOUT_OBSERVATION_SOURCE,
                    "actual_result_event_id": event["call_id"],
                    "state_event_id": event["state_event_id"],
                    "skill_id": previous_skill_id,
                    "next_skill_id": target_skill_id,
                    "equivalent_next_skill_ids": equivalent,
                    "candidate_next_skill_ids": list(candidate_ids_by_domain[domain]),
                    "route_target": "TOOL",
                    "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                    "supervision_semantics": (
                        "observed_successful_action_multi_positive_by_task_prefix_v1"
                    ),
                    "provenance": {
                        "source": "tau2_official_successful_rollout",
                        "source_id": TAU2_SUCCESSFUL_ROLLOUT_SOURCE_ID,
                        "domain": domain,
                        "raw_task_id": raw_task_id,
                        "agent_model": trajectory["agent_model"],
                        "simulation_id": _clean_text(simulation.get("id")),
                        "trial": simulation.get("trial"),
                        "seed": simulation.get("seed"),
                        "result_file": trajectory["result_file"],
                        "reward": _reward(simulation),
                        "agent_visible_fields_only": True,
                        "user_scenario_fields_read": False,
                    },
                }
            )
            prefix.append(action_name)
            previous_skill_id = target_skill_id

    report = {
        "schema_version": "tau2_official_successful_rollout_corpus_v1",
        "status": "ok" if source_rows else "action_required",
        "task_split": task_split,
        "data_root": str(data_root),
        "results_root": str(resolved_results_root),
        "domains": list(resolved_domains),
        "agent_models": list(resolved_models),
        "minimum_reward": float(minimum_reward),
        "official_split_task_count": int(official_task_count),
        "official_split_task_ids_by_domain": official_split_task_ids_by_domain,
        "task_count_with_successful_tool_rows": len(tasks_with_rows),
        "successful_tool_trajectory_count": len(accepted),
        "successful_tool_event_count": len(source_rows),
        "multi_positive_row_count": int(multi_positive_rows),
        "distinct_successful_action_path_count": sum(
            len(paths) for paths in distinct_paths_by_task.values()
        ),
        "task_count_with_multiple_successful_paths": sum(
            int(len(paths) > 1) for paths in distinct_paths_by_task.values()
        ),
        "skill_count": len(skills),
        "per_domain": {
            domain: dict(sorted(counts.items()))
            for domain, counts in sorted(per_domain.items())
        },
        "skipped_reasons": dict(sorted(skipped.items())),
        "result_files": result_files,
        "input_visibility_contract": {
            "agent_visible_messages_only": True,
            "task_user_scenario_excluded": True,
            "task_description_excluded": True,
            "evaluation_reference_actions_excluded_as_labels": True,
            "actual_assistant_tool_calls_required": True,
            "matching_actual_tool_results_required": True,
            "route_state_external_message_roles": ["user"],
            "raw_tool_results_excluded_from_state_text_current": True,
            "actual_tool_results_preserved_for_memory_correction": True,
        },
    }
    return Tau2SuccessfulRolloutCorpus(
        skills=skills,
        source_rows=source_rows,
        report=report,
    )
