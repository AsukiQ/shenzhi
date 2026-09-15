from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvStep:
    observation_text: str
    reward: float | None
    done: bool
    success: bool | None
    valid_actions: list[str]


class EnvAdapter:
    """Non-trainable closed-loop environment adapter interface."""

    is_trainable = False

    def reset(self, task_id: str | None = None) -> str:
        raise NotImplementedError

    def step(self, action_text: str) -> EnvStep:
        raise NotImplementedError

    def observation_text(self) -> str:
        raise NotImplementedError

    def candidate_actions(self) -> list[str]:
        raise NotImplementedError

    def valid_actions(self) -> list[str]:
        return self.candidate_actions()
