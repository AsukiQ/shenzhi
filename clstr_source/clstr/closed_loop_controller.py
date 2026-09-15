from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


CONTROLLER_MODES = {
    "policy_only",
    "policy_plus_transition",
    "policy_plus_transition_belief",
    "policy_plus_transition_belief_stop",
    "policy_plus_transition_belief_stop_loop_penalty",
}


@dataclass(frozen=True)
class ClosedLoopControllerConfig:
    mode: str = "policy_plus_transition_belief_stop_loop_penalty"
    policy_weight: float = 1.0
    transition_weight: float = 1.0
    belief_weight: float = 0.5
    stop_weight: float = 0.05
    normalize_component_scores: bool = True
    calibrated_score_clip: float = 3.0
    repeat_penalty: float = 1.0
    observation_action_penalty: float = 0.25
    two_cycle_penalty: float = 2.0
    reversible_action_penalty: float = 4.0
    repeat_window: int = 4
    planner_weight: float = 0.0
    help_action_penalty: float = 5.0
    inventory_action_penalty: float = 3.0
    repeated_look_action_penalty: float = 4.0
    cycle_ngram_penalty: float = 3.0
    cycle_ngram_min: int = 3
    cycle_ngram_max: int = 8
    q_success_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported controller mode: {self.mode}")


@dataclass
class ClosedLoopControllerState:
    action_history: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class ClosedLoopDecision:
    chosen_action: str
    chosen_index: int
    chosen_reason: str
    component_scores: list[dict[str, Any]]


class ClstrTextActionController:
    """Text-env controller wrapper around CLSTR candidate scorers."""

    def __init__(
        self,
        candidate_scorer: Any,
        component_scorer: Any | None = None,
        config: ClosedLoopControllerConfig | None = None,
    ) -> None:
        self.candidate_scorer = candidate_scorer
        self.component_scorer = component_scorer
        self.config = config or ClosedLoopControllerConfig()
        self.action_history: list[str] = []
        self.last_decision: ClosedLoopDecision | None = None

    def reset(self) -> None:
        self.action_history = []
        self.last_decision = None

    def choose_action(self, state_text: str, admissible_actions: list[str]) -> str:
        candidate_rows = [list(admissible_actions)]
        state_texts = [str(state_text)]
        policy_scores = self.candidate_scorer(state_texts, candidate_rows)
        if not isinstance(policy_scores, torch.Tensor):
            policy_scores = torch.as_tensor(policy_scores, dtype=torch.float32)

        components: dict[str, torch.Tensor] = {}
        if self.component_scorer is not None:
            result = self.component_scorer(
                state_texts,
                candidate_rows,
                policy_scores,
                [list(self.action_history)],
            )
            if result:
                components = dict(result)

        decisions = score_candidates_with_components(
            candidate_rows=candidate_rows,
            policy_scores=policy_scores,
            transition_scores=components.get("transition_scores"),
            belief_scores=components.get("belief_scores"),
            stop_logits=components.get("stop_logits"),
            state=ClosedLoopControllerState(action_history=[list(self.action_history)]),
            config=self.config,
            planner_scores=components.get("planner_scores"),
            q_success_scores=components.get("q_success_scores"),
        )
        self.last_decision = decisions[0]
        if self.last_decision.chosen_action:
            self.action_history.append(self.last_decision.chosen_action)
        return self.last_decision.chosen_action


def _active_weights(config: ClosedLoopControllerConfig) -> dict[str, float]:
    weights = {
        "policy": float(config.policy_weight),
        "transition": 0.0,
        "belief": 0.0,
        "stop": 0.0,
        "loop": 0.0,
        "planner": float(config.planner_weight),
        "q_success": float(config.q_success_weight),
    }
    if config.mode in {
        "policy_plus_transition",
        "policy_plus_transition_belief",
        "policy_plus_transition_belief_stop",
        "policy_plus_transition_belief_stop_loop_penalty",
    }:
        weights["transition"] = float(config.transition_weight)
    if config.mode in {
        "policy_plus_transition_belief",
        "policy_plus_transition_belief_stop",
        "policy_plus_transition_belief_stop_loop_penalty",
    }:
        weights["belief"] = float(config.belief_weight)
    if config.mode in {
        "policy_plus_transition_belief_stop",
        "policy_plus_transition_belief_stop_loop_penalty",
    }:
        weights["stop"] = float(config.stop_weight)
    if config.mode == "policy_plus_transition_belief_stop_loop_penalty":
        weights["loop"] = 1.0
    return weights


def _loop_penalty(action: str, history: list[str], config: ClosedLoopControllerConfig) -> float:
    if not history:
        return 0.0
    action_text = str(action)
    normalized = action_text.strip().lower()
    recent = [str(item) for item in history[-max(1, int(config.repeat_window)) :]]
    penalty = 0.0
    repeat_count = sum(1 for item in recent if item == action_text)
    if repeat_count:
        penalty -= float(config.repeat_penalty) * repeat_count
    if normalized in {"look", "inventory", "help"} and recent:
        penalty -= float(config.observation_action_penalty) * len(recent)
    if len(history) >= 2 and action_text == history[-2] and action_text != history[-1]:
        penalty -= float(config.two_cycle_penalty)
    if history and _is_reversible_toggle(action_text, str(history[-1])):
        penalty -= float(config.reversible_action_penalty)
    penalty -= _repeated_cycle_penalty(action_text, history, config)
    return penalty


def _repeated_cycle_penalty(action: str, history: list[str], config: ClosedLoopControllerConfig) -> float:
    proposed = [str(item) for item in history] + [str(action)]
    max_period = min(max(int(config.cycle_ngram_max), 0), len(proposed) // 2)
    min_period = max(1, int(config.cycle_ngram_min))
    if max_period < min_period:
        return 0.0
    hits = 0
    for period in range(min_period, max_period + 1):
        if proposed[-period:] == proposed[-2 * period : -period]:
            hits += 1
    return float(config.cycle_ngram_penalty) * hits


def _action_prior_penalty(action: str, history: list[str], config: ClosedLoopControllerConfig) -> float:
    normalized = str(action).strip().lower()
    if normalized == "help":
        return -float(config.help_action_penalty)
    if normalized == "inventory":
        return -float(config.inventory_action_penalty)
    if normalized == "look" and history:
        return -float(config.repeated_look_action_penalty)
    return 0.0


def _verb_object(action: str) -> tuple[str, str] | None:
    parts = str(action).strip().lower().split()
    if len(parts) < 2:
        return None
    verb = parts[0]
    obj = " ".join(parts[1:])
    return verb, obj


def _is_reversible_toggle(action: str, previous: str) -> bool:
    current_pair = _verb_object(action)
    previous_pair = _verb_object(previous)
    if current_pair is None or previous_pair is None:
        return False
    current_verb, current_obj = current_pair
    previous_verb, previous_obj = previous_pair
    inverse_pairs = {
        ("open", "close"),
        ("close", "open"),
        ("turnon", "turnoff"),
        ("turnoff", "turnon"),
        ("turn", "turn"),
    }
    if current_obj != previous_obj:
        return False
    return (current_verb, previous_verb) in inverse_pairs


def _safe_item(tensor: torch.Tensor, row: int, col: int) -> float:
    return round(float(tensor[row, col].detach().float().cpu().item()), 6)


def _calibrate_scores(scores: torch.Tensor, width: int, config: ClosedLoopControllerConfig) -> torch.Tensor:
    if width <= 1 or not bool(config.normalize_component_scores):
        return scores
    values = scores[:width].float()
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values)
    clean = torch.where(finite, values, torch.zeros_like(values))
    mean = clean[finite].mean()
    std = clean[finite].std(unbiased=False)
    if float(std.detach().cpu().item()) < 1.0e-6:
        calibrated = torch.zeros_like(values)
    else:
        calibrated = (values - mean) / std.clamp_min(1.0e-6)
    clip = float(config.calibrated_score_clip)
    if clip > 0:
        calibrated = calibrated.clamp(min=-clip, max=clip)
    return calibrated


def _candidate_width(candidate_rows: list[list[str]], scores: torch.Tensor, row_idx: int) -> int:
    return min(len(candidate_rows[row_idx]), int(scores.size(1)))


def score_candidates_with_components(
    candidate_rows: list[list[str]],
    policy_scores: torch.Tensor,
    transition_scores: torch.Tensor | None,
    belief_scores: torch.Tensor | None,
    stop_logits: torch.Tensor | None,
    state: ClosedLoopControllerState,
    config: ClosedLoopControllerConfig | None = None,
    planner_scores: torch.Tensor | None = None,
    q_success_scores: torch.Tensor | None = None,
) -> list[ClosedLoopDecision]:
    config = config or ClosedLoopControllerConfig()
    weights = _active_weights(config)
    if policy_scores.ndim != 2:
        raise ValueError("policy_scores must be a [batch, candidates] tensor")
    batch_n = len(candidate_rows)
    if policy_scores.size(0) != batch_n:
        raise ValueError("policy_scores batch dimension must match candidate rows")
    if transition_scores is None:
        transition_scores = torch.zeros_like(policy_scores)
    if belief_scores is None:
        belief_scores = torch.zeros_like(policy_scores)
    if stop_logits is None:
        stop_logits = torch.zeros_like(policy_scores)
    if stop_logits.ndim == 1:
        stop_logits = stop_logits.unsqueeze(-1).expand_as(policy_scores)
    if planner_scores is None:
        planner_scores = torch.zeros_like(policy_scores)
    if q_success_scores is None:
        q_success_scores = torch.zeros_like(policy_scores)

    decisions: list[ClosedLoopDecision] = []
    for row_idx, candidates in enumerate(candidate_rows):
        width = _candidate_width(candidate_rows, policy_scores, row_idx)
        if width <= 0:
            decisions.append(
                ClosedLoopDecision(
                    chosen_action="",
                    chosen_index=0,
                    chosen_reason="no_candidates",
                    component_scores=[],
                )
            )
            continue
        history = state.action_history[row_idx] if row_idx < len(state.action_history) else []
        use_calibrated = config.mode != "policy_only"
        if use_calibrated:
            policy_calibrated = _calibrate_scores(policy_scores[row_idx], width, config)
            transition_calibrated = _calibrate_scores(transition_scores[row_idx], width, config)
            belief_calibrated = _calibrate_scores(belief_scores[row_idx], width, config)
            stop_calibrated = _calibrate_scores(-torch.sigmoid(stop_logits[row_idx].float()), width, config)
            planner_calibrated = _calibrate_scores(planner_scores[row_idx], width, config)
            q_success_calibrated = _calibrate_scores(q_success_scores[row_idx], width, config)
        else:
            policy_calibrated = policy_scores[row_idx, :width].float()
            transition_calibrated = transition_scores[row_idx, :width].float()
            belief_calibrated = belief_scores[row_idx, :width].float()
            stop_calibrated = -torch.sigmoid(stop_logits[row_idx, :width].float())
            planner_calibrated = planner_scores[row_idx, :width].float()
            q_success_calibrated = q_success_scores[row_idx, :width].float()
        component_rows: list[dict[str, Any]] = []
        for col_idx, action in enumerate(candidates[:width]):
            policy = _safe_item(policy_scores, row_idx, col_idx)
            transition = _safe_item(transition_scores, row_idx, col_idx)
            belief = _safe_item(belief_scores, row_idx, col_idx)
            stop_logit = _safe_item(stop_logits, row_idx, col_idx)
            stop_probability = round(float(torch.sigmoid(torch.tensor(stop_logit)).item()), 6)
            planner = _safe_item(planner_scores, row_idx, col_idx)
            q_success = _safe_item(q_success_scores, row_idx, col_idx)
            loop = _loop_penalty(action, history, config) if weights["loop"] else 0.0
            action_prior = _action_prior_penalty(action, history, config) if weights["loop"] else 0.0
            stop_score = _safe_item(stop_calibrated.unsqueeze(0), 0, col_idx)
            planner_for_final = _safe_item(planner_calibrated.unsqueeze(0), 0, col_idx)
            q_success_for_final = _safe_item(q_success_calibrated.unsqueeze(0), 0, col_idx)
            policy_for_final = _safe_item(policy_calibrated.unsqueeze(0), 0, col_idx)
            transition_for_final = _safe_item(transition_calibrated.unsqueeze(0), 0, col_idx)
            belief_for_final = _safe_item(belief_calibrated.unsqueeze(0), 0, col_idx)
            final_score = (
                weights["policy"] * policy_for_final
                + weights["transition"] * transition_for_final
                + weights["belief"] * belief_for_final
                + weights["stop"] * stop_score
                + weights["planner"] * planner_for_final
                + weights["q_success"] * q_success_for_final
                + loop
                + action_prior
            )
            component_rows.append(
                {
                    "action": str(action),
                    "candidate_index": col_idx,
                    "policy_score": policy,
                    "policy_score_calibrated": policy_for_final,
                    "transition_score": transition,
                    "transition_score_calibrated": transition_for_final,
                    "belief_score": belief,
                    "belief_score_calibrated": belief_for_final,
                    "stop_logit": stop_logit,
                    "stop_probability": stop_probability,
                    "stop_score": round(float(stop_score), 6),
                    "planner_score": planner,
                    "planner_score_calibrated": planner_for_final,
                    "q_success_score": q_success,
                    "q_success_score_calibrated": q_success_for_final,
                    "loop_penalty": round(float(loop), 6),
                    "action_prior_penalty": round(float(action_prior), 6),
                    "final_score": round(float(final_score), 6),
                    "controller_mode": config.mode,
                }
            )
        chosen = max(component_rows, key=lambda item: float(item["final_score"]))
        decisions.append(
            ClosedLoopDecision(
                chosen_action=str(chosen["action"]),
                chosen_index=int(chosen["candidate_index"]),
                chosen_reason="max_final_score",
                component_scores=component_rows,
            )
        )
    return decisions
