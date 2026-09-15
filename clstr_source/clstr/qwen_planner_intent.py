from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from clstr.qwen_direct_policy import QwenDirectAdmissibleActionScorer, QwenDirectPolicyConfig


_ACTION_PREFIX_RE = re.compile(r"^\s*(?:action|answer|chosen action)\s*:\s*", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PlannerIntent:
    proposed_action: str
    raw_response: str = ""
    parse_status: str = ""
    fallback_used: bool = False
    invalid_response: str | None = None
    is_final_action: bool = False


def normalise_action_for_match(text: str) -> str:
    value = _ACTION_PREFIX_RE.sub("", str(text or "").strip())
    value = value.strip().strip("`'\"")
    value = re.sub(r"[.。；;]+$", "", value).strip()
    return " ".join(value.lower().split())


def build_structured_planner_state_text(
    state_text: str,
    admissible_actions: list[str],
    proposed_action: str | None = None,
) -> str:
    action_lines = "\n".join(f"{idx}. {action}" for idx, action in enumerate(admissible_actions, start=1))
    proposed = str(proposed_action or "<none>").strip() or "<none>"
    return "\n".join(
        [
            str(state_text),
            "",
            "AVAILABLE ACTIONS:",
            action_lines,
            "",
            f"Qwen proposed action: {proposed}",
            "Planner note: the Qwen proposed action is advisory only.",
            "CLSTR must verify and rerank admissible actions with skill routing, transition, belief, STOP, and loop penalties.",
        ]
    )


class QwenPlannerIntentGenerator:
    """Frozen Qwen planner wrapper that proposes actions but never decides final CLSTR actions."""

    def __init__(self, scorer: Any | None = None, config: QwenDirectPolicyConfig | None = None):
        self.scorer = scorer or QwenDirectAdmissibleActionScorer(config or QwenDirectPolicyConfig())
        self.last_metadata: list[dict[str, Any]] = []

    def propose(self, state_texts: list[str], candidate_rows: list[list[str]]) -> list[PlannerIntent]:
        scores = self.scorer(state_texts, candidate_rows)
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        metadata = getattr(self.scorer, "last_metadata", None)
        intents: list[PlannerIntent] = []
        self.last_metadata = []
        for row_idx, candidates in enumerate(candidate_rows):
            item = metadata[row_idx] if isinstance(metadata, list) and row_idx < len(metadata) else {}
            proposed = str(item.get("parsed_action") or "")
            if not proposed and candidates:
                chosen_idx = int(torch.argmax(scores[row_idx, : len(candidates)]).item())
                proposed = str(candidates[chosen_idx])
            intent = PlannerIntent(
                proposed_action=proposed,
                raw_response=str(item.get("raw_model_response", "")),
                parse_status=str(item.get("parse_status", "")),
                fallback_used=bool(item.get("fallback_used", False)),
                invalid_response=item.get("invalid_response"),
                is_final_action=False,
            )
            intents.append(intent)
            self.last_metadata.append(
                {
                    "qwen_proposed_action": intent.proposed_action,
                    "qwen_raw_model_response": intent.raw_response,
                    "qwen_parse_status": intent.parse_status,
                    "qwen_fallback_used": intent.fallback_used,
                    "qwen_invalid_response": intent.invalid_response,
                    "qwen_is_final_action": False,
                    "planner_intent_mode": "qwen_action_proposal_advisory",
                }
            )
        return intents
