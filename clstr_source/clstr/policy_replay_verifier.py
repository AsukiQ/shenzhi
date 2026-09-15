from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from clstr.envs.base import EnvAdapter
from clstr.external_data import write_json, write_jsonl


VALID_TRAIN_SPLITS = {"train"}
FORBIDDEN_TRAIN_SPLITS = {"valid", "valid_seen", "valid_unseen", "test"}


class MissingReplayHarness(RuntimeError):
    """Raised when an official closed-loop replay harness is unavailable."""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _history_text(history: list[str]) -> list[str]:
    return [str(item) for item in history if str(item).strip()]


def _trajectory_goal_text(trajectory: dict[str, Any]) -> str:
    return str(trajectory.get("goal_text") or trajectory.get("task") or trajectory.get("task_description") or "")


def _trajectory_task_type(trajectory: dict[str, Any]) -> str:
    return str(trajectory.get("task_type") or trajectory.get("task_name") or trajectory.get("environment") or "")


def _step_action(step: dict[str, Any]) -> str:
    return str(step.get("action_text") or step.get("action") or step.get("expert_action") or "").strip()


def _step_observation(step: dict[str, Any], fallback: str) -> str:
    return str(step.get("observation_text") or step.get("observation") or fallback or "")


def _step_index(step: dict[str, Any], default: int) -> int:
    try:
        return int(step.get("t", step.get("step_index", default)))
    except (TypeError, ValueError):
        return default


def _candidate_actions(harness: EnvAdapter) -> list[str]:
    candidates = harness.candidate_actions()
    return [str(action) for action in candidates or []]


def _verified_record(
    *,
    trajectory: dict[str, Any],
    step: dict[str, Any],
    benchmark: str,
    source_id: str,
    source_dataset: str,
    trajectory_id: str,
    trajectory_step_count: int,
    observation: str,
    next_observation: str,
    history: list[str],
    admissible_commands: list[str],
    expert_action: str,
    done: bool | None,
    reward: float | None,
    success: bool | None,
    replay_prefix: list[dict[str, Any]],
    step_pos: int,
) -> dict[str, Any]:
    step_index = _step_index(step, step_pos)
    return {
        "split": "train",
        "benchmark": benchmark,
        "bucket": "verified_replay",
        "source_dataset": source_dataset,
        "trajectory_id": trajectory_id,
        "step_index": step_index,
        "trajectory_step_count": trajectory_step_count,
        "task_type": _trajectory_task_type(trajectory),
        "goal_text": _trajectory_goal_text(trajectory),
        "initial_observation": str(trajectory.get("initial_observation") or ""),
        "observation_t": observation,
        "history_t": _history_text(history),
        "admissible_commands_t": list(admissible_commands),
        "expert_action_t": expert_action,
        "expert_action_in_admissible": True,
        "done_t": done,
        "reward_t": reward,
        "won_t": success,
        "next_observation_t": next_observation,
        "replay_prefix": list(replay_prefix),
        "candidate_source": "official_harness_valid_actions",
        "usable_for_policy": True,
        "verification_status": "verified_replay",
        "provenance": {
            "source_id": source_id,
            "source_dataset": source_dataset,
            "source_split": trajectory.get("split"),
            "trajectory_task_id": trajectory.get("task_id"),
        },
    }


def verify_policy_replay(
    *,
    source_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    benchmark: str,
    source_id: str,
    source_dataset: str,
    harness_factory: Callable[[dict[str, Any]], EnvAdapter],
    max_trajectories: int | None = None,
    max_steps_per_trajectory: int | None = None,
) -> dict[str, Any]:
    verified_rows: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    trajectories_attempted = 0
    trajectories_successful_replayed = 0
    trajectories_with_verified_steps = 0
    trajectories_aborted_after_desync = 0
    trajectories_skipped_by_split = 0
    extracted_step_count = 0
    expert_in_admissible_count = 0

    for trajectory in _iter_jsonl(source_path):
        split = str(trajectory.get("split") or "unknown").lower()
        if split not in VALID_TRAIN_SPLITS:
            trajectories_skipped_by_split += 1
            skipped_reasons[f"non_train_split:{split}"] += 1
            continue
        if max_trajectories is not None and trajectories_attempted >= max_trajectories:
            break
        steps = [step for step in trajectory.get("steps") or [] if isinstance(step, dict)]
        if not steps:
            skipped_reasons["empty_trajectory"] += 1
            continue
        trajectories_attempted += 1
        harness = harness_factory(trajectory)
        observation = str(harness.reset(str(trajectory.get("task_id") or "")))
        history: list[str] = []
        replay_prefix: list[dict[str, Any]] = []
        trajectory_used = False
        trajectory_aborted = False
        step_limit = len(steps) if max_steps_per_trajectory is None else min(len(steps), max_steps_per_trajectory)
        for step_pos, step in enumerate(steps[:step_limit]):
            expert_action = _step_action(step)
            if not expert_action:
                skipped_reasons["missing_expert_action"] += 1
                remaining_steps = step_limit - step_pos - 1
                if remaining_steps:
                    skipped_reasons["steps_skipped_after_desync"] += remaining_steps
                trajectories_aborted_after_desync += 1
                trajectory_aborted = True
                break
            observation_t = _step_observation(step, observation)
            candidates = _candidate_actions(harness)
            extracted_step_count += 1
            if expert_action not in candidates:
                skipped_reasons["expert_action_not_in_admissible"] += 1
                remaining_steps = step_limit - step_pos - 1
                if remaining_steps:
                    skipped_reasons["steps_skipped_after_desync"] += remaining_steps
                trajectories_aborted_after_desync += 1
                trajectory_aborted = True
                break
            expert_in_admissible_count += 1
            result = harness.step(expert_action)
            next_observation = result.observation_text
            reward = result.reward
            done = bool(result.done)
            success = result.success
            verified_rows.append(
                _verified_record(
                    trajectory=trajectory,
                    step=step,
                    benchmark=benchmark,
                    source_id=source_id,
                    source_dataset=source_dataset,
                    trajectory_id=str(trajectory.get("task_id") or f"{source_id}:{trajectories_attempted - 1}"),
                    trajectory_step_count=len(steps),
                    observation=observation_t,
                    next_observation=next_observation,
                    history=history,
                    admissible_commands=candidates,
                    expert_action=expert_action,
                    done=done,
                    reward=reward,
                    success=success,
                    replay_prefix=replay_prefix,
                    step_pos=step_pos,
                )
            )
            trajectory_used = True
            observation = next_observation
            history.append(expert_action)
            replay_prefix.append(
                {
                    "step_index": _step_index(step, step_pos),
                    "observation_text": observation_t,
                    "action_text": expert_action,
                    "next_observation_text": next_observation,
                }
            )
            if done:
                break
        if trajectory_used:
            trajectories_with_verified_steps += 1
            if not trajectory_aborted:
                trajectories_successful_replayed += 1
        if max_trajectories is not None and trajectories_attempted >= max_trajectories:
            break

    valid_or_test_used = any(
        str(row.get("provenance", {}).get("source_split", "")).lower() in FORBIDDEN_TRAIN_SPLITS
        for row in verified_rows
    )
    report = {
        "status": "ok" if verified_rows else "missing_verified_steps",
        "benchmark": benchmark,
        "source_id": source_id,
        "source_dataset": source_dataset,
        "split": "train",
        "bucket": "verified_replay",
        "allowed_losses": ["L_policy", "L_trans", "belief", "STOP", "routing"],
        "trajectories_attempted": trajectories_attempted,
        "trajectories_successful_replayed": trajectories_successful_replayed,
        "trajectories_with_verified_steps": trajectories_with_verified_steps,
        "trajectories_aborted_after_desync": trajectories_aborted_after_desync,
        "trajectories_skipped_by_split": trajectories_skipped_by_split,
        "extracted_step_count": extracted_step_count,
        "usable_policy_step_count": len(verified_rows),
        "expert_action_in_admissible_rate": expert_in_admissible_count / max(extracted_step_count, 1),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "valid_or_test_used_for_training": valid_or_test_used,
        "train_rows_written": len(verified_rows),
        "train_path": str(output_path),
        "manifest_path": str(manifest_path),
        "candidate_source": "official_harness_valid_actions",
        "desync_handling": "abort_trajectory_after_first_unverified_transition",
        "can_upgrade_to_verified_replay": bool(verified_rows) and not valid_or_test_used,
    }
    write_jsonl(output_path, verified_rows)
    write_json(manifest_path, report)
    return report


def write_policy_replay_blocker(
    *,
    output_path: str | Path,
    benchmark: str,
    source_id: str,
    source_dataset: str,
    error: Exception,
    repro_command: str,
) -> dict[str, Any]:
    report = {
        "status": "blocked_missing_replay_harness",
        "benchmark": benchmark,
        "source_id": source_id,
        "source_dataset": source_dataset,
        "train_rows_written": 0,
        "can_upgrade_to_verified_replay": False,
        "reason": str(error),
        "repro_command": repro_command,
        "required_evidence": [
            "official harness reset/step must run",
            "per-step admissible/valid actions must come from the harness",
            "expert_action must be observed in the current valid action set",
            "valid/test split data must not be used for training",
        ],
    }
    write_json(output_path, report)
    return report
