from __future__ import annotations

from typing import Any

import torch


def _as_float_scores(scores: Any, batch_size: int, target_width: int) -> torch.Tensor:
    tensor = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores, dtype=torch.float32)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim != 2:
        raise ValueError("action scorer must return a [batch, candidates] tensor")
    if int(tensor.size(0)) != batch_size:
        raise ValueError("action scorer batch dimension must match candidate rows")
    min_value = torch.finfo(torch.float32).min
    if int(tensor.size(1)) < target_width:
        pad = torch.full((batch_size, target_width - int(tensor.size(1))), min_value, dtype=torch.float32)
        tensor = torch.cat([tensor, pad], dim=1)
    elif int(tensor.size(1)) > target_width:
        tensor = tensor[:, :target_width]
    return tensor


def _row_zscore(scores: torch.Tensor, candidate_rows: list[list[str]]) -> torch.Tensor:
    min_value = torch.finfo(torch.float32).min
    calibrated = torch.full_like(scores.float(), min_value)
    for row_idx, candidates in enumerate(candidate_rows):
        width = min(len(candidates), int(scores.size(1)))
        if width <= 0:
            continue
        values = scores[row_idx, :width].float()
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            calibrated[row_idx, :width] = 0.0
            continue
        clean = torch.where(finite, values, torch.zeros_like(values))
        active = clean[finite]
        if width <= 1 or active.numel() <= 1:
            calibrated[row_idx, :width] = 0.0
            continue
        std = active.std(unbiased=False)
        if float(std.item()) < 1.0e-6:
            calibrated[row_idx, :width] = 0.0
            continue
        mean = active.mean()
        row_scores = (values - mean) / std.clamp_min(1.0e-6)
        row_scores = torch.where(finite, row_scores, torch.zeros_like(row_scores))
        calibrated[row_idx, :width] = row_scores
    return calibrated


def _argmax_action(scores: torch.Tensor, candidates: list[str]) -> tuple[int, str, float]:
    width = min(len(candidates), int(scores.size(0)))
    if width <= 0:
        return 0, "", 0.0
    idx = int(torch.argmax(scores[:width].float()).item())
    return idx, str(candidates[idx]), float(scores[idx].float().item())


def _proposal_bonus_scores(scores: torch.Tensor, candidate_rows: list[list[str]]) -> torch.Tensor:
    min_value = torch.finfo(torch.float32).min
    proposal = torch.full_like(scores.float(), min_value)
    for row_idx, candidates in enumerate(candidate_rows):
        width = min(len(candidates), int(scores.size(1)))
        if width <= 0:
            continue
        chosen_idx = int(torch.argmax(scores[row_idx, :width].float()).item())
        proposal[row_idx, :width] = 0.0
        proposal[row_idx, chosen_idx] = 1.0
    return proposal


def _is_reliable_qwen_choice(qwen_item: dict[str, Any], qwen_action: str, candidates: list[str]) -> bool:
    if not qwen_action or qwen_action not in candidates:
        return False
    if bool(qwen_item.get("fallback_used", False)):
        return False
    parse_status = str(qwen_item.get("parse_status", "") or "").strip().lower()
    if not parse_status:
        return False
    return parse_status not in {"fallback", "no_candidates"}


def _is_coarse_unified_memory_prior(clstr_item: dict[str, Any]) -> bool:
    return str(clstr_item.get("policy_family") or "") == "clstr_unified_memory_admissible_action_scorer"


def append_abstract_skill_guidance(state_text: str, guidance: str) -> str:
    """Append optional planning guidance while preserving the original state."""

    original = str(state_text)
    hint = str(guidance or "").strip()
    if not hint:
        return original
    separator = "\n" if original.endswith("\n") else "\n\n"
    return original + separator + hint


class QwenClstrSkillPromptActionScorer:
    """Let CLSTR propose abstract skills and Qwen ground one exact legal action."""

    def __init__(
        self,
        qwen_scorer: Any,
        clstr_retriever: Any,
        *,
        guidance_top_k: int = 2,
    ) -> None:
        self.qwen_scorer = qwen_scorer
        self.clstr_retriever = clstr_retriever
        self.guidance_top_k = max(1, int(guidance_top_k))
        self.last_metadata: list[dict[str, Any]] = []
        self.last_qwen_scores: torch.Tensor | None = None
        self.last_guided_state_texts: list[str] = []

    @property
    def causal_update_count(self) -> Any:
        return getattr(self.clstr_retriever, "causal_update_count", None)

    def reset_episode_batch(self, state_texts: list[str]) -> None:
        reset = getattr(self.clstr_retriever, "reset_episode_batch", None)
        if callable(reset):
            reset(state_texts)

    def observe_transitions(self, **kwargs: Any) -> None:
        observe = getattr(self.clstr_retriever, "observe_transitions", None)
        if callable(observe):
            observe(**kwargs)

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
    ) -> torch.Tensor:
        if len(state_texts) != len(candidate_rows):
            raise ValueError("Qwen/CLSTR skill-prompt batch sizes differ")
        retrieve = getattr(self.clstr_retriever, "retrieve_skill_guidance", None)
        if not callable(retrieve):
            raise TypeError("CLSTR skill-prompt interface requires retrieve_skill_guidance")
        guidance_rows = retrieve(
            state_texts,
            candidate_rows,
            top_k=self.guidance_top_k,
        )
        if len(guidance_rows) != len(state_texts):
            raise ValueError("CLSTR skill-guidance batch size drift")
        guided_states = [
            append_abstract_skill_guidance(state_text, guidance)
            for state_text, guidance in zip(state_texts, guidance_rows)
        ]
        self.last_guided_state_texts = guided_states
        batch_size = len(candidate_rows)
        target_width = max((len(row) for row in candidate_rows), default=0)
        qwen_scores = _as_float_scores(
            self.qwen_scorer(guided_states, candidate_rows),
            batch_size,
            target_width,
        )
        self.last_qwen_scores = qwen_scores

        qwen_metadata = getattr(self.qwen_scorer, "last_metadata", None)
        clstr_metadata = getattr(self.clstr_retriever, "last_guidance_metadata", None)
        metadata: list[dict[str, Any]] = []
        for row_index, candidates in enumerate(candidate_rows):
            candidate_list = [str(item) for item in candidates]
            chosen_index, chosen_action, chosen_score = _argmax_action(
                qwen_scores[row_index], candidate_list
            )
            qwen_item = (
                qwen_metadata[row_index]
                if isinstance(qwen_metadata, list)
                and row_index < len(qwen_metadata)
                and isinstance(qwen_metadata[row_index], dict)
                else {}
            )
            clstr_item = (
                clstr_metadata[row_index]
                if isinstance(clstr_metadata, list)
                and row_index < len(clstr_metadata)
                and isinstance(clstr_metadata[row_index], dict)
                else {}
            )
            guidance = str(guidance_rows[row_index] or "")
            metadata.append(
                {
                    **qwen_item,
                    "policy_family": "qwen_clstr_abstract_skill_prompt_executor",
                    "executor_interface": "skill_prompt",
                    "action_selection": "qwen_exact_action_from_abstract_skill_guidance",
                    "qwen_policy_family": qwen_item.get("policy_family"),
                    "qwen_chosen_index": chosen_index,
                    "qwen_chosen_action": chosen_action,
                    "qwen_chosen_score": round(chosen_score, 6),
                    "executor_chosen_index": chosen_index,
                    "executor_chosen_action": chosen_action,
                    "raw_model_response": qwen_item.get("raw_model_response", ""),
                    "parsed_action": qwen_item.get("parsed_action", chosen_action),
                    "parse_status": qwen_item.get("parse_status", ""),
                    "fallback_used": bool(qwen_item.get("fallback_used", False)),
                    "uses_clstr_skill_guidance": bool(guidance),
                    "uses_clstr_prior": False,
                    "score_fusion": False,
                    "skill_guidance": guidance,
                    "skill_guidance_top_k": self.guidance_top_k,
                    "original_state_preserved": True,
                    "candidate_actions_unchanged": True,
                    "clstr_policy_family": clstr_item.get("policy_family"),
                    "candidate_skill_schema": clstr_item.get(
                        "candidate_skill_schema"
                    ),
                    "mapped_abstract_skill_count": int(
                        clstr_item.get("mapped_abstract_skill_count", 0) or 0
                    ),
                    "mapped_abstract_skill_ids": list(
                        clstr_item.get("mapped_abstract_skill_ids") or []
                    ),
                    "selected_abstract_skill_ids": list(
                        clstr_item.get("selected_abstract_skill_ids") or []
                    ),
                    "uses_recurrent_m_t": bool(
                        clstr_item.get("uses_recurrent_m_t", False)
                    ),
                    "clstr_memory_source": clstr_item.get("clstr_memory_source"),
                    "clstr_history_depth": int(
                        clstr_item.get("history_depth", 0) or 0
                    ),
                    "clstr_selector_probability": clstr_item.get(
                        "selector_probability"
                    ),
                    "clstr_mixture_probability": clstr_item.get(
                        "mixture_probability"
                    ),
                    "clstr_route_mode": clstr_item.get("route_mode"),
                    "clstr_selected_expert": clstr_item.get("selected_expert"),
                    "clstr_post_action_result_correction": bool(
                        clstr_item.get("post_action_result_correction", False)
                    ),
                    "runtime_appended_skill_count": int(
                        clstr_item.get("runtime_appended_skill_count", 0) or 0
                    ),
                    "clstr_final_action": False,
                    "executor_gate": False,
                }
            )
        self.last_metadata = metadata
        return qwen_scores


class QwenClstrAbstractGroundedActionScorer:
    """Use an abstract-skill gate before scoring exact legal actions.

    The reliability decision uses only checkpoint-known abstract skills. Qwen
    supplies either a dense likelihood surface or one exact generated-action
    proposal over unchanged legal commands. The chosen static or recurrent
    expert then scores separately encoded legal actions through the configured
    grounding head without adding runtime skill rows.
    """

    def __init__(
        self,
        qwen_scorer: Any,
        clstr_retriever: Any,
        *,
        guidance_top_k: int = 2,
        qwen_weight: float = 1.0,
        clstr_weight: float = 0.25,
        qwen_score_mode: str = "dense_likelihood",
        inject_skill_guidance: bool = True,
        executor_interface: str = "abstract_grounded",
        policy_family: str = "qwen_clstr_abstract_grounded_executor",
    ) -> None:
        self.qwen_scorer = qwen_scorer
        self.clstr_retriever = clstr_retriever
        self.guidance_top_k = max(1, int(guidance_top_k))
        self.qwen_weight = float(qwen_weight)
        self.clstr_weight = float(clstr_weight)
        self.qwen_score_mode = str(qwen_score_mode or "dense_likelihood")
        if self.qwen_score_mode not in {"dense_likelihood", "proposal_bonus"}:
            raise ValueError(
                f"unsupported abstract-grounded Qwen score mode: {qwen_score_mode}"
            )
        self.inject_skill_guidance = bool(inject_skill_guidance)
        self.executor_interface = str(executor_interface).strip()
        if self.executor_interface not in {
            "abstract_grounded",
            "guided_exact_prior",
            "guided_static_exact_prior",
        }:
            raise ValueError(
                "unsupported abstract-grounded executor interface: "
                f"{executor_interface}"
            )
        self.policy_family = str(policy_family).strip()
        if not self.policy_family:
            raise ValueError("abstract-grounded policy family must be nonempty")
        self.last_metadata: list[dict[str, Any]] = []
        self.last_qwen_scores: torch.Tensor | None = None
        self.last_clstr_scores: torch.Tensor | None = None
        self.last_guided_state_texts: list[str] = []

    @property
    def causal_update_count(self) -> Any:
        return getattr(self.clstr_retriever, "causal_update_count", None)

    def reset_episode_batch(self, state_texts: list[str]) -> None:
        reset = getattr(self.clstr_retriever, "reset_episode_batch", None)
        if callable(reset):
            reset(state_texts)

    def observe_transitions(self, **kwargs: Any) -> None:
        observe = getattr(self.clstr_retriever, "observe_transitions", None)
        if callable(observe):
            observe(**kwargs)

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
    ) -> torch.Tensor:
        if len(state_texts) != len(candidate_rows):
            raise ValueError("Qwen/CLSTR abstract-grounding batch sizes differ")
        retrieve = getattr(
            self.clstr_retriever,
            "retrieve_grounded_action_prior",
            None,
        )
        if not callable(retrieve):
            raise TypeError(
                "abstract-grounded interface requires retrieve_grounded_action_prior"
            )
        guidance_rows, abstract_action_prior = retrieve(
            state_texts,
            candidate_rows,
            guidance_top_k=self.guidance_top_k,
        )
        if len(guidance_rows) != len(state_texts):
            raise ValueError("CLSTR abstract-grounding guidance batch size drift")
        guided_states = (
            [
                append_abstract_skill_guidance(state_text, guidance)
                for state_text, guidance in zip(state_texts, guidance_rows)
            ]
            if self.inject_skill_guidance
            else [str(state_text) for state_text in state_texts]
        )
        self.last_guided_state_texts = guided_states
        batch_size = len(candidate_rows)
        target_width = max((len(row) for row in candidate_rows), default=0)
        qwen_raw = _as_float_scores(
            self.qwen_scorer(guided_states, candidate_rows),
            batch_size,
            target_width,
        )
        clstr_raw = _as_float_scores(
            abstract_action_prior,
            batch_size,
            target_width,
        )
        if self.qwen_score_mode == "proposal_bonus":
            qwen_grounding = _proposal_bonus_scores(qwen_raw, candidate_rows)
            qwen_normalization = "proposal_bonus"
        else:
            qwen_grounding = _row_zscore(qwen_raw, candidate_rows)
            qwen_normalization = "row_zscore"
        combined = (
            self.qwen_weight * qwen_grounding
            + self.clstr_weight * clstr_raw
        )
        floor = torch.finfo(torch.float32).min
        for row_index, candidates in enumerate(candidate_rows):
            width = len(candidates)
            if width < target_width:
                combined[row_index, width:] = floor
        self.last_qwen_scores = qwen_raw
        self.last_clstr_scores = clstr_raw

        qwen_metadata = getattr(self.qwen_scorer, "last_metadata", None)
        clstr_metadata = getattr(
            self.clstr_retriever,
            "last_grounding_metadata",
            None,
        )
        metadata: list[dict[str, Any]] = []
        for row_index, candidates in enumerate(candidate_rows):
            candidate_list = [str(item) for item in candidates]
            qwen_index, qwen_action, qwen_score = _argmax_action(
                qwen_raw[row_index], candidate_list
            )
            clstr_index, clstr_action, clstr_score = _argmax_action(
                clstr_raw[row_index], candidate_list
            )
            chosen_index, chosen_action, chosen_score = _argmax_action(
                combined[row_index], candidate_list
            )
            qwen_item = (
                qwen_metadata[row_index]
                if isinstance(qwen_metadata, list)
                and row_index < len(qwen_metadata)
                and isinstance(qwen_metadata[row_index], dict)
                else {}
            )
            clstr_item = (
                clstr_metadata[row_index]
                if isinstance(clstr_metadata, list)
                and row_index < len(clstr_metadata)
                and isinstance(clstr_metadata[row_index], dict)
                else {}
            )
            guidance = str(guidance_rows[row_index] or "")
            clstr_normalization = str(
                clstr_item.get("grounding_score_normalization")
                or "legal_action_row_zscore"
            )
            metadata.append(
                {
                    **qwen_item,
                    "policy_family": self.policy_family,
                    "executor_interface": self.executor_interface,
                    "action_selection": (
                        f"qwen_{self.qwen_score_mode}_"
                        f"{'guided_' if self.inject_skill_guidance else ''}"
                        "plus_abstract_route_prior_"
                        "over_exact_legal_actions"
                    ),
                    "qwen_policy_family": qwen_item.get("policy_family"),
                    "qwen_chosen_index": qwen_index,
                    "qwen_chosen_action": qwen_action,
                    "qwen_chosen_score": round(qwen_score, 6),
                    "clstr_chosen_index": clstr_index,
                    "clstr_chosen_action": clstr_action,
                    "clstr_chosen_score": round(clstr_score, 6),
                    "hybrid_chosen_index": chosen_index,
                    "hybrid_chosen_action": chosen_action,
                    "hybrid_chosen_score": round(chosen_score, 6),
                    "executor_chosen_index": chosen_index,
                    "executor_chosen_action": chosen_action,
                    "qwen_weight": self.qwen_weight,
                    "clstr_weight": self.clstr_weight,
                    "qwen_score_mode": self.qwen_score_mode,
                    "qwen_score_normalization": qwen_normalization,
                    "clstr_score_normalization": clstr_normalization,
                    "score_normalization": (
                        f"qwen:{qwen_normalization};clstr:{clstr_normalization}"
                    ),
                    "uses_clstr_skill_guidance": bool(
                        guidance and self.inject_skill_guidance
                    ),
                    "skill_guidance_injected": bool(
                        guidance and self.inject_skill_guidance
                    ),
                    "abstract_route_guidance": guidance,
                    "uses_clstr_prior": bool(self.clstr_weight != 0.0),
                    "score_fusion": True,
                    "skill_guidance": guidance if self.inject_skill_guidance else "",
                    "skill_guidance_top_k": self.guidance_top_k,
                    "original_state_preserved": True,
                    "candidate_actions_unchanged": True,
                    "candidate_skill_schema": clstr_item.get(
                        "candidate_skill_schema"
                    ),
                    "grounding_candidate_schema": clstr_item.get(
                        "grounding_candidate_schema"
                    ),
                    "mapped_action_count": int(
                        clstr_item.get("mapped_action_count", 0) or 0
                    ),
                    "unmapped_action_count": int(
                        clstr_item.get("unmapped_action_count", 0) or 0
                    ),
                    "mapped_abstract_skill_count": int(
                        clstr_item.get("mapped_abstract_skill_count", 0) or 0
                    ),
                    "abstract_gate_candidate_count": int(
                        clstr_item.get("abstract_gate_candidate_count", 0) or 0
                    ),
                    "grounding_score_source": clstr_item.get(
                        "grounding_score_source"
                    ),
                    "grounding_score_mode": clstr_item.get(
                        "grounding_score_mode"
                    ),
                    "grounding_expert_mode": clstr_item.get(
                        "grounding_expert_mode"
                    ),
                    "abstract_selected_expert": clstr_item.get(
                        "abstract_selected_expert"
                    ),
                    "grounding_selected_expert": clstr_item.get(
                        "grounding_selected_expert"
                    ),
                    "grounding_static_top_action": clstr_item.get(
                        "grounding_static_top_action"
                    ),
                    "grounding_dynamic_top_action": clstr_item.get(
                        "grounding_dynamic_top_action"
                    ),
                    "grounding_memory_top1_changed": bool(
                        clstr_item.get("grounding_memory_top1_changed", False)
                    ),
                    "grounding_route_residual_l2": float(
                        clstr_item.get("grounding_route_residual_l2", 0.0) or 0.0
                    ),
                    "grounding_route_residual_max_abs": float(
                        clstr_item.get(
                            "grounding_route_residual_max_abs", 0.0
                        )
                        or 0.0
                    ),
                    "mapped_abstract_skill_ids": list(
                        clstr_item.get("mapped_abstract_skill_ids") or []
                    ),
                    "selected_abstract_skill_ids": list(
                        clstr_item.get("selected_abstract_skill_ids") or []
                    ),
                    "uses_recurrent_m_t": bool(
                        clstr_item.get("uses_recurrent_m_t", False)
                    ),
                    "clstr_memory_source": clstr_item.get("clstr_memory_source"),
                    "clstr_history_depth": int(
                        clstr_item.get("history_depth", 0) or 0
                    ),
                    "clstr_selector_probability": clstr_item.get(
                        "selector_probability"
                    ),
                    "clstr_mixture_probability": clstr_item.get(
                        "mixture_probability"
                    ),
                    "clstr_route_mode": clstr_item.get("route_mode"),
                    "clstr_selected_expert": clstr_item.get("selected_expert"),
                    "clstr_post_action_result_correction": bool(
                        clstr_item.get("post_action_result_correction", False)
                    ),
                    "runtime_appended_skill_count": int(
                        clstr_item.get("runtime_appended_skill_count", 0) or 0
                    ),
                    "runtime_pseudo_skills_allowed": False,
                    "clstr_final_action": False,
                    "executor_gate": True,
                }
            )
        self.last_metadata = metadata
        return combined


class QwenClstrHybridActionScorer:
    """Fuse frozen Qwen admissible-action scores with a CLSTR action prior.

    The scorer is intentionally evaluation-only. It does not make CLSTR the direct
    executor; it keeps Qwen as the action scorer and tests whether a trained CLSTR
    selector prior improves the same admissible-action decision surface.
    """

    def __init__(
        self,
        qwen_scorer: Any,
        clstr_scorer: Any,
        qwen_weight: float = 1.0,
        clstr_weight: float = 0.5,
        normalize_scores: bool = True,
        qwen_score_mode: str = "raw",
        prior_name: str = "clstr",
        policy_family: str = "qwen_clstr_hybrid_executor_gate",
        executor_interface: str = "exact_prior",
        inject_skill_guidance: bool = False,
        guidance_top_k: int = 2,
    ) -> None:
        if qwen_score_mode not in {"raw", "proposal_bonus"}:
            raise ValueError(f"unsupported qwen_score_mode: {qwen_score_mode}")
        self.qwen_scorer = qwen_scorer
        self.clstr_scorer = clstr_scorer
        self.qwen_weight = float(qwen_weight)
        self.clstr_weight = float(clstr_weight)
        self.normalize_scores = bool(normalize_scores)
        self.qwen_score_mode = qwen_score_mode
        self.prior_name = str(prior_name).strip().lower() or "clstr"
        self.policy_family = str(policy_family).strip() or "qwen_clstr_hybrid_executor_gate"
        self.executor_interface = str(executor_interface).strip() or "exact_prior"
        if self.executor_interface not in {
            "exact_prior",
            "legacy_guided_exact_prior",
        }:
            raise ValueError(
                f"unsupported hybrid executor interface: {executor_interface}"
            )
        self.inject_skill_guidance = bool(inject_skill_guidance)
        self.guidance_top_k = max(1, int(guidance_top_k))
        self.last_metadata: list[dict[str, Any]] = []
        self.last_qwen_scores: torch.Tensor | None = None
        self.last_clstr_scores: torch.Tensor | None = None
        self.last_guided_state_texts: list[str] = []

    @property
    def causal_update_count(self) -> Any:
        return getattr(self.clstr_scorer, "causal_update_count", None)

    def reset_episode_batch(self, state_texts: list[str]) -> None:
        reset = getattr(self.clstr_scorer, "reset_episode_batch", None)
        if callable(reset):
            reset(state_texts)

    def observe_transitions(self, **kwargs: Any) -> None:
        observe = getattr(self.clstr_scorer, "observe_transitions", None)
        if callable(observe):
            observe(**kwargs)

    def __call__(self, state_texts: list[str], candidate_rows: list[list[str]]) -> torch.Tensor:
        batch_size = len(candidate_rows)
        target_width = max((len(row) for row in candidate_rows), default=0)
        if target_width <= 0:
            self.last_metadata = []
            return torch.empty(batch_size, 0, dtype=torch.float32)

        guidance_rows = ["" for _ in state_texts]
        guidance_metadata: list[dict[str, Any]] | None = None
        if self.inject_skill_guidance:
            # Preserve legacy exact-prior scoring and m_0 initialization by
            # scoring exact legal-action candidates before the independent
            # abstract guidance lookup. The lookup does not update memory.
            clstr_raw = _as_float_scores(
                self.clstr_scorer(state_texts, candidate_rows),
                batch_size,
                target_width,
            )
            retrieve = getattr(self.clstr_scorer, "retrieve_skill_guidance", None)
            if not callable(retrieve):
                raise TypeError(
                    "guided exact-prior interface requires retrieve_skill_guidance"
                )
            guidance_rows = retrieve(
                state_texts,
                candidate_rows,
                top_k=self.guidance_top_k,
            )
            if len(guidance_rows) != len(state_texts):
                raise ValueError("guided exact-prior guidance batch size drift")
            guided_states = [
                append_abstract_skill_guidance(state_text, guidance)
                for state_text, guidance in zip(state_texts, guidance_rows)
            ]
            guidance_metadata = getattr(
                self.clstr_scorer,
                "last_guidance_metadata",
                None,
            )
            qwen_raw = _as_float_scores(
                self.qwen_scorer(guided_states, candidate_rows),
                batch_size,
                target_width,
            )
        else:
            guided_states = [str(state_text) for state_text in state_texts]
            qwen_raw = _as_float_scores(
                self.qwen_scorer(state_texts, candidate_rows),
                batch_size,
                target_width,
            )
            clstr_raw = _as_float_scores(
                self.clstr_scorer(state_texts, candidate_rows),
                batch_size,
                target_width,
            )
        self.last_guided_state_texts = guided_states
        self.last_qwen_scores = qwen_raw
        self.last_clstr_scores = clstr_raw

        if self.qwen_score_mode == "proposal_bonus":
            qwen_for_mix = _proposal_bonus_scores(qwen_raw, candidate_rows)
            qwen_normalization = "proposal_bonus"
        elif self.normalize_scores:
            qwen_for_mix = _row_zscore(qwen_raw, candidate_rows)
            qwen_normalization = "row_zscore"
        else:
            qwen_for_mix = qwen_raw
            qwen_normalization = "none"

        if self.normalize_scores:
            clstr_for_mix = _row_zscore(clstr_raw, candidate_rows)
            clstr_normalization = "row_zscore"
        else:
            clstr_for_mix = clstr_raw
            clstr_normalization = "none"

        min_value = torch.finfo(torch.float32).min
        combined = self.qwen_weight * qwen_for_mix + self.clstr_weight * clstr_for_mix
        for row_idx, candidates in enumerate(candidate_rows):
            width = min(len(candidates), target_width)
            if width < target_width:
                combined[row_idx, width:] = min_value

        qwen_metadata = getattr(self.qwen_scorer, "last_metadata", None)
        clstr_metadata = getattr(self.clstr_scorer, "last_metadata", None)
        metadata: list[dict[str, Any]] = []
        for row_idx, candidates in enumerate(candidate_rows):
            candidate_list = [str(item) for item in candidates]
            qwen_idx, qwen_action, qwen_score = _argmax_action(qwen_raw[row_idx], candidate_list)
            clstr_idx, clstr_action, clstr_score = _argmax_action(clstr_raw[row_idx], candidate_list)
            hybrid_idx, hybrid_action, hybrid_score = _argmax_action(combined[row_idx], candidate_list)
            qwen_item = (
                qwen_metadata[row_idx]
                if isinstance(qwen_metadata, list)
                and row_idx < len(qwen_metadata)
                and isinstance(qwen_metadata[row_idx], dict)
                else {}
            )
            clstr_item = (
                clstr_metadata[row_idx]
                if isinstance(clstr_metadata, list)
                and row_idx < len(clstr_metadata)
                and isinstance(clstr_metadata[row_idx], dict)
                else {}
            )
            guidance_item = (
                guidance_metadata[row_idx]
                if isinstance(guidance_metadata, list)
                and row_idx < len(guidance_metadata)
                and isinstance(guidance_metadata[row_idx], dict)
                else {}
            )
            guidance = str(guidance_rows[row_idx] or "")
            qwen_reliable = _is_reliable_qwen_choice(qwen_item, qwen_action, candidate_list)
            coarse_unified_memory_prior = _is_coarse_unified_memory_prior(clstr_item)
            clstr_override_allowed = True
            clstr_override_reason = (
                "allowed_legacy_coarse_prior"
                if coarse_unified_memory_prior
                else "allowed_noncoarse_prior"
            )
            if coarse_unified_memory_prior and qwen_reliable and clstr_action and clstr_action != qwen_action:
                width = min(len(candidate_list), target_width)
                if width > 0:
                    combined[row_idx, :width] = qwen_for_mix[row_idx, :width]
                    combined[row_idx, qwen_idx] = torch.max(combined[row_idx, :width].float()) + 1.0e-4
                    hybrid_idx, hybrid_action, hybrid_score = _argmax_action(combined[row_idx], candidate_list)
                clstr_override_allowed = False
                clstr_override_reason = "blocked_coarse_prior_reliable_qwen"
            elif coarse_unified_memory_prior and not qwen_reliable:
                clstr_override_reason = "allowed_qwen_fallback"
            elif coarse_unified_memory_prior and clstr_action == qwen_action:
                clstr_override_reason = "allowed_same_action"
            prior_name = self.prior_name
            metadata_item = {
                    "policy_family": self.policy_family,
                    "executor_interface": self.executor_interface,
                    f"uses_{prior_name}_prior": bool(self.clstr_weight != 0.0),
                    "uses_clstr_prior": bool(self.clstr_weight != 0.0) if prior_name == "clstr" else False,
                    "qwen_weight": self.qwen_weight,
                    "clstr_weight": self.clstr_weight,
                    f"{prior_name}_weight": self.clstr_weight,
                    "qwen_score_mode": self.qwen_score_mode,
                    "qwen_score_normalization": qwen_normalization,
                    "clstr_score_normalization": clstr_normalization if prior_name == "clstr" else "",
                    f"{prior_name}_score_normalization": clstr_normalization,
                    "score_normalization": (
                        qwen_normalization
                        if qwen_normalization == clstr_normalization
                        else f"qwen:{qwen_normalization};{prior_name}:{clstr_normalization}"
                    ),
                    "candidate_count": len(candidate_list),
                    "uses_clstr_skill_guidance": bool(
                        self.inject_skill_guidance and guidance
                    ),
                    "guidance_conditions_qwen": bool(
                        self.inject_skill_guidance and guidance
                    ),
                    "score_fusion": True,
                    "skill_guidance_injected": bool(
                        self.inject_skill_guidance and guidance
                    ),
                    "skill_guidance": (
                        guidance if self.inject_skill_guidance else ""
                    ),
                    "skill_guidance_top_k": self.guidance_top_k,
                    "selected_abstract_skill_ids": list(
                        guidance_item.get("selected_abstract_skill_ids") or []
                    ),
                    "abstract_selected_expert": guidance_item.get(
                        "selected_expert"
                    ),
                    "abstract_route_mode": guidance_item.get("route_mode"),
                    "qwen_policy_family": qwen_item.get("policy_family"),
                    "clstr_policy_family": clstr_item.get("policy_family") if prior_name == "clstr" else None,
                    f"{prior_name}_policy_family": clstr_item.get("policy_family"),
                    "qwen_chosen_index": qwen_idx,
                    "qwen_chosen_action": qwen_action,
                    "qwen_chosen_score": round(qwen_score, 6),
                    "clstr_chosen_index": clstr_idx if prior_name == "clstr" else -1,
                    "clstr_chosen_action": clstr_action if prior_name == "clstr" else "",
                    "clstr_chosen_score": round(clstr_score, 6) if prior_name == "clstr" else 0.0,
                    "clstr_memory_source": clstr_item.get("clstr_memory_source") if prior_name == "clstr" else None,
                    "clstr_replay_prefix_len": clstr_item.get("clstr_replay_prefix_len", 0) if prior_name == "clstr" else 0,
                    "clstr_replay_prefix_used_count": (
                        clstr_item.get("clstr_replay_prefix_used_count", 0) if prior_name == "clstr" else 0
                    ),
                    "uses_recurrent_m_t": bool(clstr_item.get("uses_recurrent_m_t", False))
                    if prior_name == "clstr"
                    else False,
                    "clstr_history_depth": int(clstr_item.get("history_depth", 0) or 0)
                    if prior_name == "clstr"
                    else 0,
                    "clstr_selector_probability": clstr_item.get("selector_probability")
                    if prior_name == "clstr"
                    else None,
                    "clstr_mixture_probability": clstr_item.get("mixture_probability")
                    if prior_name == "clstr"
                    else None,
                    "clstr_post_action_result_correction": bool(
                        clstr_item.get("post_action_result_correction", False)
                    )
                    if prior_name == "clstr"
                    else False,
                    "candidate_skill_schema": clstr_item.get("candidate_skill_schema")
                    if prior_name == "clstr"
                    else None,
                    "runtime_appended_skill_count": int(
                        clstr_item.get("runtime_appended_skill_count", 0) or 0
                    )
                    if prior_name == "clstr"
                    else 0,
                    "runtime_pseudo_skills_allowed": True,
                    "clstr_route_scorer": clstr_item.get("route_scorer") if prior_name == "clstr" else None,
                    f"{prior_name}_chosen_index": clstr_idx,
                    f"{prior_name}_chosen_action": clstr_action,
                    f"{prior_name}_chosen_score": round(clstr_score, 6),
                    "hybrid_chosen_index": hybrid_idx,
                    "hybrid_chosen_action": hybrid_action,
                    "hybrid_chosen_score": round(hybrid_score, 6),
                    "qwen_raw_model_response": qwen_item.get("raw_model_response", ""),
                    "qwen_parsed_action": qwen_item.get("parsed_action", ""),
                    "qwen_parse_status": qwen_item.get("parse_status", ""),
                    "qwen_fallback_used": bool(qwen_item.get("fallback_used", False)),
                    "raw_model_response": qwen_item.get("raw_model_response", ""),
                    "parsed_action": qwen_item.get("parsed_action", ""),
                    "parse_status": qwen_item.get("parse_status", ""),
                    "fallback_used": bool(qwen_item.get("fallback_used", False)),
                    "qwen_choice_reliable": bool(qwen_reliable),
                    "coarse_unified_memory_prior": bool(coarse_unified_memory_prior),
                    "legacy_coarse_safeguard_applicable": bool(
                        coarse_unified_memory_prior
                    ),
                    "clstr_override_allowed": bool(clstr_override_allowed),
                    "clstr_override_reason": clstr_override_reason,
                    "qwen_direct_generator": bool(qwen_item.get("qwen_direct_generator", False)),
                    "qwen_direct_generator_in_clstr": False,
                    "clstr_final_action": False,
                    "executor_gate": True,
                    "action_selection": f"qwen_scores_plus_{prior_name}_prior_over_admissible_actions",
            }
            metadata.append(metadata_item)
        self.last_metadata = metadata
        return combined
