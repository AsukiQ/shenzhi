from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clstr.history_channel import materialize_history_free_state


FULL_BASE_FIELDS = (
    "benchmark",
    "task_id",
    "trajectory_id",
    "step_index",
    "trajectory_step_count",
    "goal_text",
    "task_text",
    "state_text",
    "state_text_current",
    "state_text_full",
    "history_text",
    "replay_prefix",
    "action_text",
    "admissible_actions",
    "expert_action",
    "next_action_text",
    "next_skill_id",
    "next_observation_text",
    "observation_source",
    "done",
    "reward",
    "skill_id",
    "loss_mask",
    "source_quality",
    "candidate_source",
    "on_policy_rollout",
    "m_t_source",
    "provenance",
)

LOSS_KEYS = ("L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing")


def allowed_loss_mask(
    allowed_losses: list[str],
    admissible_actions: list[str],
    expert_action: str | None,
    next_observation_text: str | None,
    done: bool | None,
    skill_id: str | None,
    history_text: str | None,
    next_skill_id: str | None = None,
) -> dict[str, bool]:
    allowed = set(allowed_losses or [])
    candidates = [str(action) for action in admissible_actions or []]
    expert = str(expert_action or "").strip()
    skill = str(skill_id or "").strip()
    next_skill = str(next_skill_id or "").strip()
    history = str(history_text or "").strip()
    return {
        "L_policy": "L_policy" in allowed and bool(expert) and expert in candidates,
        "L_trans": "L_trans" in allowed and bool(next_observation_text),
        "L_trans_skill_ce": "L_trans_skill_ce" in allowed and bool(next_skill),
        "belief": "belief" in allowed and bool(next_observation_text) and bool(history or skill),
        "STOP": "STOP" in allowed and done is not None,
        "routing": "routing" in allowed and bool(skill),
    }


@dataclass(frozen=True)
class FullBaseExample:
    benchmark: str
    task_id: str
    goal_text: str
    task_text: str
    state_text: str
    history_text: str
    action_text: str
    admissible_actions: list[str]
    expert_action: str
    next_observation_text: str | None
    done: bool | None
    reward: float | None
    skill_id: str | None
    loss_mask: dict[str, bool]
    source_quality: str
    provenance: dict[str, Any]
    trajectory_id: str | None = None
    step_index: int | None = None
    trajectory_step_count: int | None = None
    replay_prefix: list[dict[str, Any]] | None = None
    next_action_text: str | None = None
    next_skill_id: str | None = None
    candidate_source: str = "unknown"
    on_policy_rollout: bool = False
    m_t_source: str = "offline_observation"
    observation_source: str = "recorded_environment_observation"

    def to_record(self) -> dict[str, Any]:
        record = {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id or self.task_id.rsplit(":", 1)[0],
            "step_index": self.step_index,
            "trajectory_step_count": self.trajectory_step_count,
            "goal_text": self.goal_text,
            "task_text": self.task_text,
            "state_text": self.state_text,
            "history_text": self.history_text,
            "replay_prefix": list(self.replay_prefix or []),
            "action_text": self.action_text,
            "admissible_actions": list(self.admissible_actions),
            "expert_action": self.expert_action,
            "next_action_text": self.next_action_text,
            "next_skill_id": self.next_skill_id,
            "next_observation_text": self.next_observation_text,
            "observation_source": (
                self.observation_source if self.next_observation_text else ""
            ),
            "done": self.done,
            "reward": self.reward,
            "skill_id": self.skill_id,
            "loss_mask": {key: bool(self.loss_mask.get(key, False)) for key in LOSS_KEYS},
            "source_quality": self.source_quality,
            "candidate_source": self.candidate_source,
            "on_policy_rollout": bool(self.on_policy_rollout),
            "m_t_source": self.m_t_source,
            "provenance": dict(self.provenance),
        }
        return materialize_history_free_state(record, replace_state_text=True)


def validate_full_base_example(record: dict[str, Any]) -> None:
    missing = [field for field in FULL_BASE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"full-base example missing fields: {missing}")
    if not isinstance(record["admissible_actions"], list):
        raise ValueError("admissible_actions must be a list")
    if not isinstance(record["replay_prefix"], list):
        raise ValueError("replay_prefix must be a list")
    if not isinstance(record["loss_mask"], dict):
        raise ValueError("loss_mask must be a dict")
    for key in LOSS_KEYS:
        if key not in record["loss_mask"]:
            raise ValueError(f"loss_mask missing {key}")
    if record["loss_mask"].get("L_policy"):
        expert = str(record.get("expert_action") or "")
        candidates = [str(action) for action in record.get("admissible_actions") or []]
        if not expert or expert not in candidates:
            raise ValueError("L_policy requires expert_action in admissible_actions")
    if record["loss_mask"].get("L_trans") and not record.get("next_observation_text"):
        raise ValueError("L_trans requires next_observation_text")
    if record["loss_mask"].get("L_trans_skill_ce") and not record.get("next_skill_id"):
        raise ValueError("L_trans_skill_ce requires next_skill_id")
    if record["loss_mask"].get("STOP") and record.get("done") is None:
        raise ValueError("STOP requires done label")
    if record["loss_mask"].get("routing") and not record.get("skill_id"):
        raise ValueError("routing requires skill_id")
