from __future__ import annotations

from typing import Any


RAW_STATE_V1 = "raw_state_v1"
SR_TASK_DESCRIPTION_V1 = "sr_task_description_v1"
CLSTR_CAUSAL_STATE_V1 = "clstr_causal_state_v1"

NO_TRUNCATION = "none"
HEAD_V1 = "head_v1"
HEAD_TAIL_V1 = "head_tail_v1"
STATE_TRUNCATION_MARKER = "\n...[state truncated]...\n"

_INSTRUCTIONS = {
    RAW_STATE_V1: None,
    SR_TASK_DESCRIPTION_V1: (
        "Given a task description, retrieve the most relevant skill document "
        "that would help an agent complete the task"
    ),
    CLSTR_CAUSAL_STATE_V1: (
        "Given an agent task or current execution state and interaction history, "
        "retrieve the skill or tool document most useful for the next action."
    ),
}

_DEFAULTS = {
    RAW_STATE_V1: {"max_chars": None, "truncation": NO_TRUNCATION},
    SR_TASK_DESCRIPTION_V1: {"max_chars": 1500, "truncation": HEAD_V1},
    CLSTR_CAUSAL_STATE_V1: {"max_chars": 2000, "truncation": HEAD_TAIL_V1},
}


def _truncate_state(text: str, *, max_chars: int | None, truncation: str) -> str:
    if max_chars is None or len(text) <= int(max_chars):
        return text
    budget = int(max_chars)
    if budget <= 0:
        raise ValueError("state_query_max_chars must be positive")
    if truncation == HEAD_V1:
        return text[:budget]
    if truncation == HEAD_TAIL_V1:
        marker = STATE_TRUNCATION_MARKER
        if budget <= len(marker) + 1:
            raise ValueError("state_query_max_chars is too small for head_tail_v1")
        remaining = budget - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return text[:head] + marker + text[-tail:]
    if truncation == NO_TRUNCATION:
        raise ValueError("over-length state cannot use truncation=none")
    raise ValueError(f"unsupported state_query_truncation: {truncation}")


def resolve_state_query_prompt_contract(
    *,
    prompt_version: str,
    max_chars: int | None = None,
    truncation: str | None = None,
    recorded_instruction: str | None = None,
) -> dict[str, Any]:
    version = str(prompt_version or "").strip()
    if version not in _INSTRUCTIONS:
        raise ValueError(f"unsupported state_query_prompt_version: {prompt_version}")
    instruction = _INSTRUCTIONS[version]
    if recorded_instruction is not None and str(recorded_instruction) != str(instruction or ""):
        raise ValueError("state query instruction does not match prompt version")
    resolved_max = _DEFAULTS[version]["max_chars"] if max_chars is None else int(max_chars)
    if resolved_max is not None and int(resolved_max) <= 0:
        raise ValueError("state_query_max_chars must be positive")
    resolved_truncation = str(truncation or _DEFAULTS[version]["truncation"])
    if resolved_truncation not in {NO_TRUNCATION, HEAD_V1, HEAD_TAIL_V1}:
        raise ValueError(f"unsupported state_query_truncation: {resolved_truncation}")
    return {
        "state_query_prompt_version": version,
        "state_query_instruction": str(instruction or ""),
        "state_query_max_chars": resolved_max,
        "state_query_truncation": resolved_truncation,
    }


def format_state_query(
    text: str,
    *,
    prompt_version: str,
    max_chars: int | None = None,
    truncation: str | None = None,
) -> str:
    contract = resolve_state_query_prompt_contract(
        prompt_version=prompt_version,
        max_chars=max_chars,
        truncation=truncation,
    )
    raw = str(text or "")
    if contract["state_query_instruction"] and raw.lstrip().startswith("Instruct:"):
        raise ValueError(
            "state query formatter requires raw state text, not preformatted Instruct text"
        )
    raw = _truncate_state(
        raw,
        max_chars=contract["state_query_max_chars"],
        truncation=contract["state_query_truncation"],
    )
    instruction = contract["state_query_instruction"]
    return raw if not instruction else f"Instruct: {instruction}\nQuery:{raw}"
