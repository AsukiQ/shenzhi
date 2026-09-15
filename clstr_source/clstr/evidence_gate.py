from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_GATE_DECISIONS = {
    "no_evidence_fallback",
    "schema_only_evidence",
    "workflow_hint_evidence",
}


@dataclass(frozen=True)
class EvidenceGateDecision:
    decision: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    selected_scores: list[float] = field(default_factory=list)
    score_margin: float | None = None
    transition_scores_available: bool = False
    exact_base_executor_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGateResult:
    decision: EvidenceGateDecision
    prompt_block: str
    diagnostics: dict[str, Any]


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    scores: list[float] = []
    for item in value:
        try:
            scores.append(float(item))
        except (TypeError, ValueError):
            continue
    return scores


def _selected_scores(controller_diagnostics: dict[str, Any] | None) -> list[float]:
    diagnostics = controller_diagnostics if isinstance(controller_diagnostics, dict) else {}
    for key in ("selected_scores", "scores", "top_scores", "candidate_scores"):
        scores = _as_float_list(diagnostics.get(key))
        if scores:
            return scores
    return []


def _score_margin(scores: list[float]) -> float | None:
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _transition_available(controller_diagnostics: dict[str, Any] | None) -> bool:
    diagnostics = controller_diagnostics if isinstance(controller_diagnostics, dict) else {}
    for key in ("transition_scores_available", "transition_active", "has_transition_scores"):
        if key in diagnostics:
            return bool(diagnostics.get(key))
    mode = str(diagnostics.get("ranking_mode", ""))
    return "transition" in mode


def _unique_text(items: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(str(item) for item in items if str(item).strip())]


def _format_prompt(
    *,
    decision: EvidenceGateDecision,
    useful_apis: list[str],
    state_changing_action_apis: list[str],
    constraint_hints: list[str],
    workflow_hints: list[str],
    max_chars: int,
) -> str:
    if decision.decision == "no_evidence_fallback":
        return ""
    lines = [
        "[Optional Retrieved Evidence]",
        f"decision: {decision.decision}",
        f"confidence: {decision.confidence}",
        "valid_tools_or_apis:",
    ]
    visible_apis = _unique_text(useful_apis)
    if visible_apis:
        lines.extend(f"- {api}" for api in visible_apis)
    else:
        lines.append("- (none)")
    visible_action_apis = _unique_text(state_changing_action_apis)
    if visible_action_apis:
        lines.append("state_changing_apis:")
        lines.extend(f"- {api}" for api in visible_action_apis)
    visible_constraints = _unique_text(constraint_hints)
    if visible_constraints:
        lines.append("constraint_checks:")
        lines.extend(f"- {hint}" for hint in visible_constraints)
    if decision.decision == "workflow_hint_evidence":
        visible_hints = _unique_text(workflow_hints)
        if visible_hints:
            lines.append("workflow_hints:")
            lines.extend(f"- {hint}" for hint in visible_hints)
    lines.extend(
        [
            "constraint_guard:",
            "- preserve user entity/date/source/count/ranking/action constraints",
            "raw_skill_text_omitted: true",
        ]
    )
    return "\n".join(lines)[: int(max_chars)]


def gate_verified_skill_handoff(
    *,
    selected_skill_ids: list[str],
    handoff_decisions: list[dict[str, Any]],
    useful_apis: list[str],
    state_changing_action_apis: list[str],
    workflow_hints: list[str],
    constraint_hints: list[str] | None = None,
    controller_diagnostics: dict[str, Any] | None = None,
    max_chars: int = 1600,
) -> EvidenceGateResult:
    counts = {"inject_hint": 0, "schema_only": 0, "suppress": 0}
    for row in handoff_decisions:
        value = str(row.get("decision", ""))
        if value in counts:
            counts[value] += 1

    reasons: list[str] = []
    schema_count = len(_unique_text(useful_apis))
    if not handoff_decisions:
        reasons.append("no_selected_skills")
    if handoff_decisions and counts["suppress"] == len(handoff_decisions):
        reasons.append("all_selected_evidence_suppressed")
    if schema_count == 0:
        reasons.append("no_schema_valid_evidence")

    scores = _selected_scores(controller_diagnostics)
    margin = _score_margin(scores)
    transition_available = _transition_available(controller_diagnostics)

    if "all_selected_evidence_suppressed" in reasons or "no_schema_valid_evidence" in reasons:
        public_decision = "no_evidence_fallback"
        confidence = "low"
    elif counts["inject_hint"] > 0 and workflow_hints and (transition_available or (scores and scores[0] >= 0.85)):
        public_decision = "workflow_hint_evidence"
        confidence = "high" if transition_available or (margin is not None and margin >= 0.03) else "medium"
        reasons.append("schema_supported_workflow_hints")
    else:
        public_decision = "schema_only_evidence"
        confidence = "medium" if schema_count else "low"
        reasons.append("schema_grounded_evidence_without_high_confidence_workflow")

    reasons = _unique_text(reasons)
    decision = EvidenceGateDecision(
        decision=public_decision,
        confidence=confidence,
        reasons=reasons,
        selected_skill_ids=_unique_text(selected_skill_ids),
        selected_scores=scores,
        score_margin=margin,
        transition_scores_available=transition_available,
        exact_base_executor_fallback=public_decision == "no_evidence_fallback",
    )
    prompt = _format_prompt(
        decision=decision,
        useful_apis=useful_apis,
        state_changing_action_apis=state_changing_action_apis,
        constraint_hints=constraint_hints or [],
        workflow_hints=workflow_hints,
        max_chars=max_chars,
    )
    return EvidenceGateResult(
        decision=decision,
        prompt_block=prompt,
        diagnostics={
            "evidence_gate": decision.to_dict(),
            "gate_decision": decision.decision,
            "gate_confidence": decision.confidence,
            "gate_reasons": decision.reasons,
            "exact_base_executor_fallback": decision.exact_base_executor_fallback,
            "selected_scores": decision.selected_scores,
            "score_margin": decision.score_margin,
            "transition_scores_available": decision.transition_scores_available,
            "gate_input_counts": counts,
            "prompt_visible_evidence_text": prompt,
        },
    )
