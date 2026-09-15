from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path
import sys
import types
from typing import Any, Iterable


TAU2_REPLAY_DOMAINS = ("airline", "retail", "telecom")


def _load_domain_api(tau2_repo_root: Path, domain: str) -> tuple[Any, Any, Any]:
    source_root = tau2_repo_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Tau2 source directory is missing: {source_root}")
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    # Importing ``tau2`` normally eagerly imports the full LLM runner stack.
    # Native environment replay needs only data models and domain tools, so use
    # the repository source tree as a namespace package and avoid unrelated
    # litellm/voice dependencies.
    if "tau2" not in sys.modules:
        package = types.ModuleType("tau2")
        package.__path__ = [str(source_root / "tau2")]
        package.__package__ = "tau2"
        sys.modules["tau2"] = package
    module = importlib.import_module(f"tau2.domains.{domain}.environment")
    messages = importlib.import_module("tau2.data_model.message")
    return module.get_environment, module.get_tasks, messages.ToolCall


def replay_tau2_golden_tool_results(
    tau2_repo_root: str | Path,
    *,
    domains: Iterable[str] = TAU2_REPLAY_DOMAINS,
    task_split: str = "train",
    max_tasks_per_domain: int | None = None,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    """Execute official Tau2 golden actions from each task's initial state.

    This consumes only the repository's signed task split and native domain
    environment.  Returned keys are ``(domain, task_id, step_index)`` so route
    data construction can attach each real tool result to the action that
    produced it without serializing benchmark/source identity into model input.
    """

    repo_root = Path(tau2_repo_root).resolve()
    if not (repo_root / "data" / "tau2").is_dir():
        raise FileNotFoundError(f"Tau2 data directory is missing under {repo_root}")
    task_split = str(task_split or "").strip().lower()
    if task_split != "train":
        raise ValueError("tool-result collection is restricted to official Tau2 train")
    resolved_domains = tuple(str(domain) for domain in domains)
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    per_domain: dict[str, dict[str, int]] = {}
    for raw_domain in resolved_domains:
        domain = str(raw_domain or "").strip().lower()
        if domain not in TAU2_REPLAY_DOMAINS:
            raise ValueError(f"unsupported Tau2 replay domain: {domain}")
        get_environment, get_tasks, tool_call_cls = _load_domain_api(
            repo_root,
            domain,
        )
        tasks = list(get_tasks(task_split))
        if max_tasks_per_domain is not None:
            tasks = tasks[: max(0, int(max_tasks_per_domain))]
        domain_counts: Counter[str] = Counter()
        for task in tasks:
            task_id = str(task.id)
            initial = task.initial_state
            environment = get_environment()
            environment.set_state(
                initialization_data=(
                    initial.initialization_data if initial is not None else None
                ),
                initialization_actions=(
                    initial.initialization_actions if initial is not None else None
                ),
                message_history=(
                    list(initial.message_history or []) if initial is not None else []
                ),
            )
            actions = list(task.evaluation_criteria.actions or [])
            domain_counts["tasks"] += 1
            domain_counts["actions"] += len(actions)
            for step_index, action in enumerate(actions):
                call = tool_call_cls(
                    id=str(action.action_id),
                    name=str(action.name),
                    arguments=dict(action.arguments),
                    requestor=action.requestor,
                )
                response = environment.get_response(call)
                key = (domain, task_id, int(step_index))
                if key in results:
                    raise ValueError(f"duplicate Tau2 replay key: {key}")
                results[key] = {
                    "action_id": str(action.action_id),
                    "action_name": str(action.name),
                    "action_arguments": dict(action.arguments),
                    "requestor": str(action.requestor),
                    "result_text": str(response.content or ""),
                    "error": bool(response.error),
                }
                domain_counts["errors"] += int(bool(response.error))
        per_domain[domain] = dict(sorted(domain_counts.items()))
        counts.update(domain_counts)
    report = {
        "protocol": "tau2_official_train_native_environment_golden_replay_v1",
        "task_split": task_split,
        "repo_root": str(repo_root),
        "domains": list(resolved_domains),
        "task_count": int(counts["tasks"]),
        "action_count": int(counts["actions"]),
        "error_count": int(counts["errors"]),
        "per_domain": per_domain,
    }
    return results, report
