from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class CurrentRoutePreferenceDataset:
    samples: list[dict[str, Any]]
    report: dict[str, Any]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _outcome_label(rollout: dict[str, Any]) -> str:
    outcome = rollout.get("outcome")
    if isinstance(outcome, dict):
        return str(outcome.get("label") or "")
    return ""


def _policy_signal(rollout: dict[str, Any]) -> str:
    outcome = rollout.get("outcome")
    if isinstance(outcome, dict):
        return str(outcome.get("policy_signal") or "")
    return ""


def _reward(rollout: dict[str, Any]) -> float:
    outcome = rollout.get("outcome")
    if isinstance(outcome, dict):
        try:
            return float(outcome.get("reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _aligned_targets(decision: dict[str, Any]) -> tuple[list[str], list[int]]:
    candidates = [str(item) for item in _as_list(decision.get("candidate_skill_ids"))]
    selected = [str(item) for item in _as_list(decision.get("selected_skill_ids"))]
    proposed_local_indices: list[int] = []
    for item in _as_list(decision.get("selected_candidate_local_indices")):
        try:
            proposed_local_indices.append(int(item))
        except (TypeError, ValueError):
            continue

    target_skill_ids: list[str] = []
    target_local_indices: list[int] = []
    for selected_idx, skill_id in enumerate(selected):
        local_idx: int | None = None
        if selected_idx < len(proposed_local_indices):
            candidate_idx = proposed_local_indices[selected_idx]
            if 0 <= candidate_idx < len(candidates) and candidates[candidate_idx] == skill_id:
                local_idx = int(candidate_idx)
        if local_idx is None:
            try:
                local_idx = candidates.index(skill_id)
            except ValueError:
                local_idx = None
        if local_idx is None:
            continue
        target_skill_ids.append(skill_id)
        target_local_indices.append(int(local_idx))
    return target_skill_ids, target_local_indices


def _success_sample(
    *,
    rollout: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    target_skill_ids, target_local_indices = _aligned_targets(decision)
    if not target_local_indices:
        return None
    task_id = str(rollout.get("task_id") or rollout.get("query_id") or "")
    step_idx = int(decision.get("step_idx", 0) or 0)
    return {
        "schema_version": "current_route_preference.v1",
        "sample_type": "success_imitation",
        "task_id": task_id,
        "query_id": str(rollout.get("query_id") or task_id),
        "decision_id": f"{task_id}::{step_idx}",
        "step_idx": step_idx,
        "state_text": str(decision.get("state_text") or ""),
        "user_goal": str(rollout.get("user_goal") or ""),
        "candidate_skill_ids": [str(item) for item in _as_list(decision.get("candidate_skill_ids"))],
        "target_skill_ids": target_skill_ids,
        "target_local_indices": target_local_indices,
        "selected_log_probs": [float(item) for item in _as_list(decision.get("selected_log_probs"))],
        "policy_log_probs": [float(item) for item in _as_list(decision.get("policy_log_probs"))],
        "candidate_policy_logits": [float(item) for item in _as_list(decision.get("candidate_policy_logits"))],
        "previous_selected_skill_ids": [str(item) for item in _as_list(decision.get("previous_selected_skill_ids"))],
        "previous_execute_output": str(decision.get("previous_execute_output") or ""),
        "ranking_mode": decision.get("ranking_mode"),
        "candidate_source": decision.get("candidate_source"),
        "outcome_label": _outcome_label(rollout),
        "policy_signal": _policy_signal(rollout),
        "reward": _reward(rollout),
        "weight": 1.0,
    }


def _suppress_sample(
    *,
    rollout: dict[str, Any],
    decision: dict[str, Any],
    sample_type: str,
    weight: float,
    qwen_success: bool,
    reference_success: bool,
) -> dict[str, Any] | None:
    avoid_skill_ids, avoid_local_indices = _aligned_targets(decision)
    if not avoid_local_indices:
        return None
    task_id = str(rollout.get("task_id") or rollout.get("query_id") or "")
    step_idx = int(decision.get("step_idx", 0) or 0)
    return {
        "schema_version": "current_route_preference.v1",
        "sample_type": str(sample_type),
        "task_id": task_id,
        "query_id": str(rollout.get("query_id") or task_id),
        "decision_id": f"{task_id}::{step_idx}",
        "step_idx": step_idx,
        "state_text": str(decision.get("state_text") or ""),
        "user_goal": str(rollout.get("user_goal") or ""),
        "candidate_skill_ids": [str(item) for item in _as_list(decision.get("candidate_skill_ids"))],
        "target_skill_ids": [],
        "target_local_indices": [],
        "avoid_skill_ids": avoid_skill_ids,
        "avoid_local_indices": avoid_local_indices,
        "selected_log_probs": [float(item) for item in _as_list(decision.get("selected_log_probs"))],
        "policy_log_probs": [float(item) for item in _as_list(decision.get("policy_log_probs"))],
        "candidate_policy_logits": [float(item) for item in _as_list(decision.get("candidate_policy_logits"))],
        "previous_selected_skill_ids": [str(item) for item in _as_list(decision.get("previous_selected_skill_ids"))],
        "previous_execute_output": str(decision.get("previous_execute_output") or ""),
        "ranking_mode": decision.get("ranking_mode"),
        "candidate_source": decision.get("candidate_source"),
        "outcome_label": _outcome_label(rollout),
        "policy_signal": "negative_route_disagreement",
        "reward": _reward(rollout),
        "weight": float(weight),
        "qwen_success": bool(qwen_success),
        "reference_success": bool(reference_success),
    }


def _run_key(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("query_id") or "")


def _run_success_map(rows: Iterable[dict[str, Any]] | None) -> dict[str, bool]:
    success: dict[str, bool] = {}
    for row in rows or []:
        key = _run_key(row)
        if not key:
            continue
        success[key] = bool(row.get("success") or row.get("evaluation_success") or row.get("final_success"))
    return success


def _rollout_success(rollout: dict[str, Any], current_success_by_task: dict[str, bool]) -> bool:
    key = _run_key(rollout)
    if key in current_success_by_task:
        return bool(current_success_by_task[key])
    return bool(rollout.get("success") or rollout.get("evaluation_success"))


def build_current_route_disagreement_preference_dataset(
    rollouts: Iterable[dict[str, Any]],
    *,
    current_runs: Iterable[dict[str, Any]] | None = None,
    qwen_runs: Iterable[dict[str, Any]] | None = None,
    reference_runs: Iterable[dict[str, Any]] | None = None,
    negative_weight: float = 0.2,
    allow_ambiguous_disagreement: bool = False,
) -> CurrentRoutePreferenceDataset:
    """Build success-imitation plus low-weight route-disagreement suppress samples.

    Suppress samples are deliberately weak: they say the current selected skills should
    receive less mass only when another route solved the same task and the current route
    did not. They are not treated as hard labels for the true next skill.
    """

    current_success_by_task = _run_success_map(current_runs)
    qwen_success_by_task = _run_success_map(qwen_runs)
    reference_success_by_task = _run_success_map(reference_runs)
    samples: list[dict[str, Any]] = []
    route_split_counts = {
        "current_success": 0,
        "current_only_success": 0,
        "qwen_success": 0,
        "qwen_only_success": 0,
        "reference_success": 0,
        "reference_only_success": 0,
        "all_failed": 0,
    }
    skipped_blocked_decision_count = 0
    skipped_unaligned_decision_count = 0
    skipped_no_route_signal_count = 0
    skipped_ambiguous_disagreement_count = 0
    rollout_count = 0

    for rollout in rollouts:
        rollout_count += 1
        task_id = _run_key(rollout)
        current_success = _rollout_success(rollout, current_success_by_task)
        qwen_success = bool(qwen_success_by_task.get(task_id, False))
        reference_success = bool(reference_success_by_task.get(task_id, False))
        if current_success:
            route_split_counts["current_success"] += 1
        if current_success and not qwen_success and not reference_success:
            route_split_counts["current_only_success"] += 1
        if qwen_success:
            route_split_counts["qwen_success"] += 1
        if qwen_success and not current_success:
            route_split_counts["qwen_only_success"] += 1
        if reference_success:
            route_split_counts["reference_success"] += 1
        if reference_success and not current_success and not qwen_success:
            route_split_counts["reference_only_success"] += 1
        if not current_success and not qwen_success and not reference_success:
            route_split_counts["all_failed"] += 1
            skipped_no_route_signal_count += 1
            continue

        if current_success:
            sample_type = "success_imitation"
        elif qwen_success:
            sample_type = "fallback_to_qwen_suppress_current"
        elif reference_success:
            sample_type = "prefer_reference_suppress_current"
        else:
            skipped_no_route_signal_count += 1
            continue
        if sample_type != "success_imitation" and not bool(allow_ambiguous_disagreement):
            if _policy_signal(rollout) != "negative":
                skipped_ambiguous_disagreement_count += 1
                continue

        for decision in _as_list(rollout.get("decisions")):
            if not isinstance(decision, dict):
                continue
            if not bool(decision.get("training_ready")):
                skipped_blocked_decision_count += 1
                continue
            if sample_type == "success_imitation":
                sample = _success_sample(rollout=rollout, decision=decision)
                if sample is not None:
                    sample["qwen_success"] = bool(qwen_success)
                    sample["reference_success"] = bool(reference_success)
            else:
                sample = _suppress_sample(
                    rollout=rollout,
                    decision=decision,
                    sample_type=sample_type,
                    weight=float(negative_weight),
                    qwen_success=qwen_success,
                    reference_success=reference_success,
                )
            if sample is None:
                skipped_unaligned_decision_count += 1
                continue
            samples.append(sample)

    report = {
        "schema_version": "current_route_disagreement_preference_report.v1",
        "rollout_count": rollout_count,
        "sample_count": len(samples),
        "positive_sample_count": sum(1 for sample in samples if sample.get("sample_type") == "success_imitation"),
        "suppress_sample_count": sum(1 for sample in samples if _as_list(sample.get("avoid_local_indices"))),
        "negative_weight": float(negative_weight),
        "allow_ambiguous_disagreement": bool(allow_ambiguous_disagreement),
        "route_split_counts": route_split_counts,
        "skipped_blocked_decision_count": skipped_blocked_decision_count,
        "skipped_ambiguous_disagreement_count": skipped_ambiguous_disagreement_count,
        "skipped_unaligned_decision_count": skipped_unaligned_decision_count,
        "skipped_no_route_signal_count": skipped_no_route_signal_count,
    }
    return CurrentRoutePreferenceDataset(samples=samples, report=report)


def build_current_route_preference_dataset(
    rollouts: Iterable[dict[str, Any]],
) -> CurrentRoutePreferenceDataset:
    samples: list[dict[str, Any]] = []
    skipped_by_outcome: dict[str, int] = {}
    rollout_count = 0
    positive_rollout_count = 0
    skipped_ambiguous_rollout_count = 0
    skipped_blocked_decision_count = 0
    skipped_unaligned_decision_count = 0
    negative_rollout_count = 0

    for rollout in rollouts:
        rollout_count += 1
        label = _outcome_label(rollout)
        signal = _policy_signal(rollout)
        if signal != "positive":
            skipped_by_outcome[label] = skipped_by_outcome.get(label, 0) + 1
            if signal == "negative":
                negative_rollout_count += 1
            else:
                skipped_ambiguous_rollout_count += 1
            continue
        positive_rollout_count += 1
        for decision in _as_list(rollout.get("decisions")):
            if not isinstance(decision, dict):
                continue
            if not bool(decision.get("training_ready")):
                skipped_blocked_decision_count += 1
                continue
            sample = _success_sample(rollout=rollout, decision=decision)
            if sample is None:
                skipped_unaligned_decision_count += 1
                continue
            samples.append(sample)

    report = {
        "schema_version": "current_route_preference_report.v1",
        "rollout_count": rollout_count,
        "positive_rollout_count": positive_rollout_count,
        "negative_rollout_count": negative_rollout_count,
        "sample_count": len(samples),
        "positive_sample_count": sum(1 for sample in samples if sample.get("sample_type") == "success_imitation"),
        "skipped_ambiguous_rollout_count": skipped_ambiguous_rollout_count,
        "skipped_blocked_decision_count": skipped_blocked_decision_count,
        "skipped_unaligned_decision_count": skipped_unaligned_decision_count,
        "skipped_by_outcome": skipped_by_outcome,
    }
    return CurrentRoutePreferenceDataset(samples=samples, report=report)


def _logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    if not math.isfinite(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def summarize_current_route_preference_signal(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sample_rows = list(samples)
    losses: list[float] = []
    candidate_counts: list[int] = []
    target_counts: list[int] = []
    skipped_missing_distribution = 0
    for sample in sample_rows:
        policy_log_probs = [float(item) for item in _as_list(sample.get("policy_log_probs"))]
        target_indices: list[int] = []
        for item in _as_list(sample.get("target_local_indices")):
            try:
                target_indices.append(int(item))
            except (TypeError, ValueError):
                continue
        candidate_counts.append(len(_as_list(sample.get("candidate_skill_ids"))))
        target_counts.append(len(target_indices))
        target_log_probs = [
            policy_log_probs[idx]
            for idx in target_indices
            if 0 <= int(idx) < len(policy_log_probs)
        ]
        if not target_log_probs:
            skipped_missing_distribution += 1
            continue
        loss = -_logsumexp(target_log_probs)
        if math.isfinite(loss):
            losses.append(float(loss))
    return {
        "schema_version": "current_route_preference_signal_report.v1",
        "sample_count": len(sample_rows),
        "finite_loss_sample_count": len(losses),
        "skipped_missing_distribution_count": skipped_missing_distribution,
        "mean_logged_multi_positive_nll": sum(losses) / len(losses) if losses else 0.0,
        "min_logged_multi_positive_nll": min(losses) if losses else 0.0,
        "max_logged_multi_positive_nll": max(losses) if losses else 0.0,
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "mean_target_count": sum(target_counts) / len(target_counts) if target_counts else 0.0,
    }


def write_current_route_preference_dataset(
    rollouts: Iterable[dict[str, Any]],
    *,
    output_jsonl_path: str | Path,
    report_json_path: str | Path,
) -> CurrentRoutePreferenceDataset:
    dataset = build_current_route_preference_dataset(rollouts)
    output_path = Path(output_jsonl_path)
    report_path = Path(report_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for sample in dataset.samples:
            fp.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    report_path.write_text(
        json.dumps(dataset.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset
