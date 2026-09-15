from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from clstr.alfworld_action_skills import alfworld_action_to_skill_id
from clstr.external_data import write_json, write_jsonl


_AVAILABLE_RE = re.compile(r"AVAILABLE ACTIONS\s*:\s*(.+?)(?:\n[A-Z][A-Z _-]{2,}:|\Z)", re.IGNORECASE | re.DOTALL)
_TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_available_actions(text: str) -> list[str]:
    match = _AVAILABLE_RE.search(str(text or ""))
    if not match:
        return []
    payload = match.group(1).strip()
    first_line = payload.splitlines()[0] if "\n" in payload else payload
    return [item.strip() for item in first_line.split(",") if item.strip()]


def parse_agentgym_action(text: str) -> str:
    matches = _ACTION_RE.findall(str(text or ""))
    if not matches:
        return ""
    action = matches[-1].strip().strip("`'\"")
    action = re.sub(r"[.。；;]+$", "", action).strip()
    return " ".join(action.split())


def _parse_goal(text: str) -> str:
    match = _TASK_RE.search(str(text or ""))
    return " ".join(match.group(1).strip().split()) if match else ""


def _history_text(actions: list[str]) -> str:
    return " | ".join(actions[-6:]) if actions else "<empty>"


def _state_text(goal: str, observation: str, history: list[str]) -> str:
    return "\n".join(
        [
            f"goal: {goal}",
            "task_type: alfworld_agentgym",
            f"observation: {observation}",
            f"history: {_history_text(history)}",
        ]
    )


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("from") or message.get("role") or "").lower()


def _message_value(message: dict[str, Any]) -> str:
    return str(message.get("value") or message.get("content") or "")


def _loss_enabled(message: dict[str, Any]) -> bool:
    return bool(message.get("loss") is True)


def _next_human_value(messages: list[dict[str, Any]], start_idx: int) -> str | None:
    for message in messages[start_idx + 1 :]:
        if _message_role(message) == "human":
            return _message_value(message)
    return None


def convert_agentgym_alfworld_conversations(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in rows:
        messages = [msg for msg in row.get("conversations") or [] if isinstance(msg, dict)]
        task_id = str(row.get("item_id") or f"agentgym-{counters['trajectory_count']}")
        history: list[str] = []
        latest_human = ""
        latest_goal = ""
        trajectory_steps: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            role = _message_role(message)
            value = _message_value(message)
            if role == "human":
                latest_human = value
                parsed_goal = _parse_goal(value)
                if parsed_goal:
                    latest_goal = parsed_goal
                continue
            if role != "gpt" or not _loss_enabled(message):
                continue
            action = parse_agentgym_action(value)
            if not action:
                counters["missing_action"] += 1
                continue
            candidates = parse_available_actions(latest_human)
            next_observation = _next_human_value(messages, idx)
            is_last_step = next_observation is None
            l_policy = bool(candidates and action in candidates)
            skill_id = alfworld_action_to_skill_id(action)
            loss_mask = {
                "L_policy": l_policy,
                "L_trans": bool(next_observation),
                "L_trans_skill_ce": False,
                "belief": bool(next_observation),
                "STOP": True,
                "routing": skill_id is not None,
            }
            step_index = len(trajectory_steps)
            record = {
                "benchmark": "alfworld",
                "task_id": f"{task_id}:{step_index}",
                "trajectory_id": task_id,
                "step_index": step_index,
                "goal_text": latest_goal,
                "task_text": latest_goal,
                "state_text": _state_text(latest_goal, latest_human, history),
                "history_text": _history_text(history),
                "action_text": action,
                "admissible_actions": candidates,
                "expert_action": action if l_policy else None,
                "next_observation_text": next_observation,
                "done": is_last_step,
                "reward": None,
                "skill_id": skill_id,
                "loss_mask": loss_mask,
                "source_quality": "weak_policy",
                "candidate_source": "agentgym_available_actions" if candidates else "missing",
                "provenance": {
                    "source_dataset": "AgentGym/AgentTraj-L",
                    "source_file": "alfworld_train.json",
                    "split": "train",
                    "bucket": "weak_policy",
                    "leakage_risk": "external_agent_trajectory_train_only_unverified_env_replay",
                    "item_id": task_id,
                },
            }
            trajectory_steps.append(record)
            history.append(action)
        if trajectory_steps:
            counters["trajectory_count"] += 1
            converted.extend(trajectory_steps)
    counters["step_count"] = len(converted)
    counters["l_policy_enabled_count"] = sum(1 for row in converted if row["loss_mask"].get("L_policy"))
    counters["has_available_actions_count"] = sum(1 for row in converted if row.get("admissible_actions"))
    counters["has_next_observation_count"] = sum(1 for row in converted if row.get("next_observation_text"))
    counters["routing_enabled_count"] = sum(1 for row in converted if row["loss_mask"].get("routing"))
    report = {
        "status": "ok",
        "source_dataset": "AgentGym/AgentTraj-L",
        "source_file": "alfworld_train.json",
        "benchmark": "alfworld",
        "split": "train",
        "bucket": "weak_policy",
        "source_quality": "weak_policy",
        "train_allowed": True,
        "official_replay": False,
        "verified_env_replay": False,
        "valid_or_test_used_for_training": False,
        "allowed_losses": ["L_policy", "L_trans", "belief", "STOP", "routing"],
        "leakage_risk": "external_agent_trajectory_train_only_unverified_env_replay",
        **dict(counters),
    }
    return converted, report


def audit_agentgym_agenttraj_file(
    source_path: str | Path,
    output_dir: str | Path,
    benchmark: str = "alfworld",
) -> dict[str, Any]:
    if benchmark != "alfworld":
        raise ValueError("only alfworld AgentGym conversion is implemented")
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    converted, report = convert_agentgym_alfworld_conversations(rows)
    converted_path = output_dir / "converted_train.jsonl"
    manifest_path = output_dir / "manifest.json"
    audit_path = output_dir / "audit_report.json"
    write_jsonl(converted_path, converted)
    manifest = {
        **report,
        "source_path": str(source_path),
        "converted_path": str(converted_path),
        "manifest_path": str(manifest_path),
        "audit_report_path": str(audit_path),
    }
    write_json(manifest_path, manifest)
    write_json(audit_path, manifest)
    return manifest
