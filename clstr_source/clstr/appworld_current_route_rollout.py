from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "current_route_rollout.v1"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _skill_evidence(step: dict[str, Any]) -> dict[str, Any]:
    value = step.get("skill_evidence")
    return value if isinstance(value, dict) else {}


def _controller(step: dict[str, Any]) -> dict[str, Any]:
    value = _skill_evidence(step).get("controller")
    return value if isinstance(value, dict) else {}


def _selected_skill_ids(step: dict[str, Any]) -> list[str]:
    evidence = _skill_evidence(step)
    gate = evidence.get("evidence_gate") if isinstance(evidence.get("evidence_gate"), dict) else {}
    values = evidence.get("selected_skill_ids")
    if not values:
        values = gate.get("selected_skill_ids")
    return [str(item) for item in _as_list(values) if str(item)]


def _candidate_skill_ids(step: dict[str, Any]) -> list[str]:
    values = _controller(step).get("candidate_skill_ids")
    return [str(item) for item in _as_list(values) if str(item)]


def _has_policy_logprobs(step: dict[str, Any]) -> bool:
    evidence = _skill_evidence(step)
    controller = _controller(step)
    for source in (evidence, controller):
        for key in (
            "selected_log_prob",
            "selected_log_probs",
            "policy_logprob",
            "policy_logprobs",
            "policy_log_prob",
            "policy_log_probs",
        ):
            if source.get(key) is not None:
                return True
    return False


def label_current_route_step(step: dict[str, Any]) -> str:
    if bool(step.get("task_completed")) and bool(step.get("evaluation_success")):
        return "success"
    if bool(step.get("task_completed")) and not bool(step.get("evaluation_success")):
        return "wrong_completion"
    precheck = step.get("completion_precheck")
    if isinstance(precheck, dict) and precheck.get("ok") is False:
        return "completion_precheck_block"
    if step.get("preflight_ok") is False:
        return "safety_preflight_block"
    if step.get("execution_attempted") is False:
        return "execution_not_attempted"
    if step.get("execution_ok") is False:
        return "execution_failure"
    return "in_progress"


def _routing_miss_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    for step in row.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        evidence = _skill_evidence(step)
        controller = _controller(step)
        sources = [evidence, controller]
        gate = evidence.get("evidence_gate")
        if isinstance(gate, dict):
            sources.append(gate)
        for source in sources:
            if source.get("routing_miss") is True:
                return {
                    "step_idx": int(step.get("step_idx", 0) or 0),
                    "reason": "routing_miss_true",
                }
            for key in (
                "positive_in_candidates",
                "current_positive_in_candidates",
                "next_positive_in_candidates",
                "gold_positive_in_candidates",
            ):
                if key in source and source.get(key) is False:
                    return {
                        "step_idx": int(step.get("step_idx", 0) or 0),
                        "reason": f"{key}_false",
                    }
    return None


def label_current_route_rollout(row: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in row.get("steps", []) or [] if isinstance(step, dict)]
    step_labels = [label_current_route_step(step) for step in steps]
    routing_miss = _routing_miss_evidence(row)
    if bool(row.get("success")) or bool(row.get("evaluation_success")):
        return {
            "label": "success",
            "reward": 1.0,
            "policy_signal": "positive",
            "reason": "official_evaluation_success",
        }
    if row.get("error") and not steps:
        return {
            "label": "setup_error",
            "reward": 0.0,
            "policy_signal": "no_signal",
            "reason": "runner_error_before_decision",
        }
    if routing_miss is not None:
        return {
            "label": "routing_miss",
            "reward": -1.0,
            "policy_signal": "negative",
            "reason": "diagnostics_confirm_routing_miss",
            "routing_miss_evidence": routing_miss,
        }
    if "wrong_completion" in step_labels or (
        bool(row.get("task_completed")) and not bool(row.get("evaluation_success"))
    ):
        return {
            "label": "wrong_completion",
            "reward": -0.5,
            "policy_signal": "ambiguous_do_not_penalize_clstr",
            "reason": "task_completed_but_official_evaluation_failed",
        }
    if any(label in {"execution_failure", "safety_preflight_block"} for label in step_labels):
        return {
            "label": "execution_failure",
            "reward": -0.25,
            "policy_signal": "executor_negative_do_not_penalize_clstr",
            "reason": "code_or_api_execution_failed",
        }
    if not steps:
        return {
            "label": "no_steps",
            "reward": 0.0,
            "policy_signal": "no_signal",
            "reason": "no_executor_steps_recorded",
        }
    return {
        "label": "incomplete",
        "reward": 0.0,
        "policy_signal": "ambiguous_do_not_penalize_clstr",
        "reason": "rollout_stopped_without_success_or_confirmed_routing_miss",
    }


def _decision_from_step(
    step: dict[str, Any],
    *,
    previous_step: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    selected_skill_ids = _selected_skill_ids(step)
    candidate_skill_ids = _candidate_skill_ids(step)
    evidence = _skill_evidence(step)
    controller = _controller(step)
    if not selected_skill_ids and not candidate_skill_ids and not evidence:
        return None
    training_blockers: list[str] = []
    if not selected_skill_ids:
        training_blockers.append("missing_selected_skill_ids")
    if not candidate_skill_ids:
        training_blockers.append("missing_candidate_skill_ids")
    if not _has_policy_logprobs(step):
        training_blockers.append("missing_policy_logprobs")
    decision = {
        "step_idx": int(step.get("step_idx", 0) or 0),
        "step_label": label_current_route_step(step),
        "state_text": str(evidence.get("state_text") or ""),
        "selected_skill_ids": selected_skill_ids,
        "candidate_skill_ids": candidate_skill_ids,
        "selected_candidate_local_indices": _as_list(controller.get("selected_candidate_local_indices")),
        "selected_scores": _as_list(controller.get("selected_scores")),
        "selected_log_probs": _as_list(controller.get("selected_log_probs")),
        "policy_log_probs": _as_list(controller.get("policy_log_probs")),
        "candidate_policy_logits": _as_list(controller.get("candidate_policy_logits")),
        "ranking_mode": controller.get("ranking_mode"),
        "candidate_source": controller.get("candidate_source"),
        "evidence_gate_decision": evidence.get("gate_decision")
        or (
            evidence.get("evidence_gate", {}).get("decision")
            if isinstance(evidence.get("evidence_gate"), dict)
            else None
        ),
        "prompt_style": evidence.get("prompt_style"),
        "visible_skill_limit": evidence.get("visible_skill_limit"),
        "visible_skill_ids": [str(item) for item in _as_list(evidence.get("visible_skill_ids")) if str(item)],
        "rescued_action_skill_ids": [
            str(item) for item in _as_list(evidence.get("rescued_action_skill_ids")) if str(item)
        ],
        "useful_apis": [str(item) for item in _as_list(evidence.get("useful_apis")) if str(item)],
        "state_changing_action_apis": [
            str(item) for item in _as_list(evidence.get("state_changing_action_apis")) if str(item)
        ],
        "filtered_duplicate_skills": int(controller.get("filtered_duplicate_skills") or 0),
        "has_policy_logprobs": _has_policy_logprobs(step),
        "training_ready": not training_blockers,
        "training_blockers": training_blockers,
        "code": str(step.get("code", "")),
        "execute_output": str(step.get("execute_output", "")),
        "task_completed": bool(step.get("task_completed")),
        "evaluation_success": bool(step.get("evaluation_success")),
    }
    if previous_step is not None:
        decision["previous_selected_skill_ids"] = _selected_skill_ids(previous_step)
        decision["previous_execute_output"] = str(previous_step.get("execute_output", ""))
    return decision


def _executor_summary(row: dict[str, Any]) -> dict[str, Any]:
    steps = [step for step in row.get("steps", []) or [] if isinstance(step, dict)]
    labels = [label_current_route_step(step) for step in steps]
    return {
        "step_count": len(steps),
        "execution_failure_count": sum(1 for label in labels if label == "execution_failure"),
        "wrong_completion_count": sum(1 for label in labels if label == "wrong_completion"),
        "preflight_block_count": sum(1 for label in labels if label == "safety_preflight_block"),
        "completion_precheck_block_count": sum(1 for label in labels if label == "completion_precheck_block"),
    }


def build_current_route_rollout_trace(row: dict[str, Any]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    previous_step: dict[str, Any] | None = None
    for step in row.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        decision = _decision_from_step(step, previous_step=previous_step)
        if decision is not None:
            decisions.append(decision)
        previous_step = step
    return {
        "schema_version": SCHEMA_VERSION,
        "method": str(row.get("method", "")),
        "task_id": str(row.get("task_id") or row.get("query_id") or ""),
        "query_id": str(row.get("query_id") or row.get("task_id") or ""),
        "split": row.get("split"),
        "user_goal": str(row.get("user_goal") or row.get("instruction") or ""),
        "success": bool(row.get("success")),
        "task_completed": bool(row.get("task_completed")),
        "evaluation_success": bool(row.get("evaluation_success")),
        "outcome": label_current_route_rollout(row),
        "executor_summary": _executor_summary(row),
        "decisions": decisions,
        "training_ready": bool(decisions)
        and all(bool(decision.get("training_ready")) for decision in decisions),
        "training_blockers": sorted(
            {
                blocker
                for decision in decisions
                for blocker in decision.get("training_blockers", [])
            }
        ),
    }


class CurrentRouteRolloutLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        trace = build_current_route_rollout_trace(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            fp.flush()
        return trace


def write_current_route_rollout_traces(rows: list[dict[str, Any]], path: str | Path) -> list[dict[str, Any]]:
    output_path = Path(path)
    traces = [build_current_route_rollout_trace(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for trace in traces:
            fp.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
        fp.flush()
    return traces
