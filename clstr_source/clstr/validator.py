from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

from clstr.candidate_utils import inject_positive_candidate
from clstr.data import ExecutionState, ReplayStep, VerifiedPair


def render_validator_prompt(traj: dict[str, Any]) -> str:
    lines = [
        "You are an LLM-based validator for tool-use trajectories.",
        "",
        f"Task: {traj['query']}",
    ]
    for idx, step in enumerate(traj.get("steps", []), start=1):
        lines.append(
            f"Step {idx}: Call {step['tool']}({step.get('args', {})}) -> {step['result']}"
        )
    lines.append(f"Final answer: {traj.get('answer', '')}")
    lines.append('Output strict JSON: {"essential_steps": [1], "reason": "short"}')
    return "\n".join(lines)


def parse_validator_output(payload: str) -> dict[str, Any]:
    obj = json.loads(payload)
    if "essential_steps" not in obj or not isinstance(obj["essential_steps"], list):
        raise ValueError("essential_steps missing")
    return obj


class CachedLLMValidator:
    def __init__(
        self,
        cache_path: str | Path,
        client: Callable[[str], str],
        max_retries: int = 3,
        retry_sleep: float = 0.5,
    ):
        self.cache_path = Path(cache_path)
        self.client = client
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.cache[row["key"]] = row["output"]

    @staticmethod
    def cache_key(traj: dict[str, Any]) -> str:
        stable = json.dumps(
            {
                "task_id": traj.get("task_id"),
                "query": traj.get("query"),
                "steps": traj.get("steps", []),
                "answer": traj.get("answer", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def validate(self, traj: dict[str, Any]) -> dict[str, Any]:
        key = self.cache_key(traj)
        if key in self.cache:
            return self.cache[key]
        prompt = render_validator_prompt(traj)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                output = parse_validator_output(self.client(prompt))
                self.cache[key] = output
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "output": output}, ensure_ascii=False) + "\n")
                return output
            except Exception as exc:  # pragma: no cover - exact API exceptions vary
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep * (2 ** attempt))
        assert last_error is not None
        raise last_error


def openai_validator_client(model: str = "gpt-4o"):
    def _call(prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    return _call


def reconstruct_state(traj: dict[str, Any], step_idx: int) -> ExecutionState:
    history = []
    steps = traj.get("steps", [])
    for i in range(max(step_idx, 0)):
        step = steps[i]
        history.append((step["tool"], step["result"]))
    observation = steps[step_idx - 1]["result"] if step_idx > 0 and step_idx - 1 < len(steps) else ""
    return ExecutionState(
        query=traj["query"],
        history=history,
        observation=observation,
        artifact=traj.get("artifact", {}),
        error=traj.get("error"),
    )


def reconstruct_essential_state(
    traj: dict[str, Any],
    essential_steps_1based: list[int],
    local_pos: int,
) -> ExecutionState:
    """State on the target closed-loop path formed by essential steps only."""
    history = []
    steps = traj.get("steps", [])
    for step_num in essential_steps_1based[: max(local_pos, 0)]:
        step = steps[step_num - 1]
        history.append((step["tool"], step["result"]))
    return ExecutionState(
        query=traj["query"],
        history=history,
        observation=history[-1][1] if history else "",
        artifact=traj.get("artifact", {}),
        error=None,
    )


def _step_candidates(step: dict[str, Any]) -> list[int] | None:
    for field in ("raw_topk", "raw_candidates"):
        if field in step and step[field] is not None:
            return [int(idx) for idx in step[field]]
    return None


def build_verified_templates(
    trajectories: list[dict[str, Any]],
    topk_recall_fn: Callable[[ExecutionState, int], list[int]],
    K: int,
) -> list[VerifiedPair]:
    templates: list[VerifiedPair] = []
    for traj in trajectories:
        parsed = parse_validator_output(traj["validator_output"])
        essential_idx = parsed["essential_steps"]
        if len(essential_idx) < 2:
            continue

        full_prefix = [
            ReplayStep(
                x=reconstruct_essential_state(traj, essential_idx, pos),
                skill_idx=traj["steps"][essential_step - 1]["tool_idx"],
                obs=traj["steps"][essential_step - 1]["result"],
            )
            for pos, essential_step in enumerate(essential_idx)
        ]
        essential_steps = [traj["steps"][idx - 1] for idx in essential_idx]

        for pos in range(len(essential_steps) - 1):
            state_before = reconstruct_essential_state(traj, essential_idx, pos)
            next_state = reconstruct_essential_state(traj, essential_idx, pos + 1)
            next_step = essential_steps[pos + 1]
            a_next_plus = essential_steps[pos + 1]["tool_idx"]
            raw_candidates = _step_candidates(next_step)
            if raw_candidates is None:
                raw_candidates = topk_recall_fn(next_state, K)
            was_in_raw_topk = a_next_plus in raw_candidates
            templates.append(
                VerifiedPair(
                    task_id=traj.get("task_id"),
                    step_idx=pos,
                    state_before=state_before,
                    action_at_t=essential_steps[pos]["tool_idx"],
                    obs_at_t=essential_steps[pos]["result"],
                    candidates_next=inject_positive_candidate(raw_candidates, a_next_plus, K),
                    a_next_plus=a_next_plus,
                    was_in_raw_topk=was_in_raw_topk,
                    # Match rollout's local closed-loop step t. The replay path is
                    # the LLM-validated essential subsequence, not the raw noisy
                    # trajectory, so m_t is a target belief for CLSTR-act.
                    replay_prefix_is_essential_subsequence=True,
                    replay_prefix=full_prefix[:pos],
                )
            )
    return templates
