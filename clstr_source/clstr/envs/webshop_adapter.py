from __future__ import annotations

from clstr.envs.base import EnvAdapter, EnvStep


class WebShopEnvAdapter(EnvAdapter):
    """Skeleton wrapper for future WebShop closed-loop evaluation."""

    def __init__(self, env_id: str = "webshop"):
        self.env_id = env_id
        self._observation = ""
        self._actions = ["search", "click", "back", "add to cart", "buy now"]

    def reset(self, task_id: str | None = None) -> str:
        suffix = f" task={task_id}" if task_id else ""
        self._observation = f"WebShop environment {self.env_id} reset.{suffix}"
        return self._observation

    def step(self, action_text: str) -> EnvStep:
        self._observation = f"WebShop action executed: {action_text}"
        return EnvStep(
            observation_text=self._observation,
            reward=None,
            done=False,
            success=None,
            valid_actions=self._actions,
        )

    def observation_text(self) -> str:
        return self._observation

    def candidate_actions(self) -> list[str]:
        return list(self._actions)
