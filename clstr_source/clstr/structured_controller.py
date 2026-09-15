from __future__ import annotations

from typing import Any

import torch

from clstr.qwen_planner_intent import (
    PlannerIntent,
    build_structured_planner_state_text,
    normalise_action_for_match,
)


def planner_match_scores(
    candidate_rows: list[list[str]],
    proposed_actions: list[str],
    exact_score: float = 1.0,
) -> torch.Tensor:
    width = max((len(row) for row in candidate_rows), default=0)
    if width <= 0:
        return torch.empty(len(candidate_rows), 0)
    rows: list[list[float]] = []
    for row, proposed in zip(candidate_rows, proposed_actions):
        target = normalise_action_for_match(proposed)
        scores = [
            float(exact_score) if target and normalise_action_for_match(candidate) == target else 0.0
            for candidate in row
        ]
        if len(scores) < width:
            scores.extend([torch.finfo(torch.float32).min] * (width - len(scores)))
        rows.append(scores)
    return torch.tensor(rows, dtype=torch.float32)


class StructuredCandidateScorer:
    """Adds Qwen planner proposal to the state, then delegates final policy scoring to CLSTR."""

    def __init__(self, base_candidate_scorer: Any, planner: Any):
        self.base_candidate_scorer = base_candidate_scorer
        self.planner = planner
        self.last_intents: list[PlannerIntent] = []
        self.last_structured_state_texts: list[str] = []
        self.last_metadata: list[dict[str, Any]] = []

    def __call__(self, state_texts: list[str], candidate_rows: list[list[str]]) -> torch.Tensor:
        intents = self.planner.propose(state_texts, candidate_rows)
        self.last_intents = intents
        self.last_structured_state_texts = [
            build_structured_planner_state_text(state_text, list(candidates), intent.proposed_action)
            for state_text, candidates, intent in zip(state_texts, candidate_rows, intents)
        ]
        scores = self.base_candidate_scorer(self.last_structured_state_texts, candidate_rows)
        base_metadata = getattr(self.base_candidate_scorer, "last_metadata", None)
        planner_metadata = getattr(self.planner, "last_metadata", None)
        metadata: list[dict[str, Any]] = []
        for idx, intent in enumerate(intents):
            item: dict[str, Any] = {}
            if isinstance(base_metadata, list) and idx < len(base_metadata) and isinstance(base_metadata[idx], dict):
                item.update(base_metadata[idx])
            if isinstance(planner_metadata, list) and idx < len(planner_metadata) and isinstance(planner_metadata[idx], dict):
                item.update(planner_metadata[idx])
            item.update(
                {
                    "policy_family": "clstr_qwen_structured_controller",
                    "qwen_proposed_action": intent.proposed_action,
                    "qwen_raw_model_response": intent.raw_response,
                    "qwen_parse_status": intent.parse_status,
                    "qwen_fallback_used": intent.fallback_used,
                    "qwen_is_final_action": False,
                    "qwen_direct_generator": False,
                    "qwen_direct_generator_in_clstr": False,
                    "planner_intent_mode": "qwen_action_proposal_advisory",
                }
            )
            metadata.append(item)
        self.last_metadata = metadata
        return scores


class StructuredComponentScorer:
    """Adds planner match scores to existing CLSTR transition/belief/STOP component scores."""

    def __init__(self, base_component_scorer: Any, structured_candidate_scorer: StructuredCandidateScorer):
        self.base_component_scorer = base_component_scorer
        self.structured_candidate_scorer = structured_candidate_scorer
        self.last_metadata: list[dict[str, Any]] = []

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
        policy_scores: torch.Tensor,
        action_histories: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        structured_state_texts = self.structured_candidate_scorer.last_structured_state_texts or state_texts
        components = dict(
            self.base_component_scorer(
                structured_state_texts,
                candidate_rows,
                policy_scores,
                action_histories,
            )
        )
        proposed = [intent.proposed_action for intent in self.structured_candidate_scorer.last_intents]
        components["planner_scores"] = planner_match_scores(candidate_rows, proposed).to(policy_scores.device)
        structured_metadata = list(self.structured_candidate_scorer.last_metadata)
        base_metadata = getattr(self.base_component_scorer, "last_metadata", None)
        merged: list[dict[str, Any]] = []
        for idx, structured_item in enumerate(structured_metadata):
            item: dict[str, Any] = {}
            if isinstance(base_metadata, list) and idx < len(base_metadata) and isinstance(base_metadata[idx], dict):
                item.update(base_metadata[idx])
            if isinstance(structured_item, dict):
                item.update(structured_item)
            merged.append(item)
        self.last_metadata = merged
        return components
