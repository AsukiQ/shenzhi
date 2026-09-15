from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from clstr.appworld_executor import (
    PredictionFileSkillProvider,
    SkillProvider,
    _call_generator,
    _canonical_skill_key,
    _default_world_factory,
    _evaluation_succeeded,
    _execution_succeeded,
    _is_auth_like_skill,
    _json_safe,
    build_appworld_code_repair_prompt,
    build_api_docs_context,
    build_appworld_executor_prompt,
    enrich_task_with_appworld_specs,
    extract_python_code,
    load_appworld_api_refs,
    preflight_appworld_code,
    sanitize_appworld_code,
)
from clstr.appworld_routing import read_json, read_jsonl, write_json, write_jsonl
from clstr.belief import subspace_obs
from clstr.envs.appworld_env import configure_appworld_paths
from clstr.full_base_train import (
    TRANSITION_SCORING_MODES,
    V4_1B_TRANSITION_SCORING_MODE,
    _transition_candidate_logits_for_mode,
)
from clstr.multistep_stop import MultiStepStopPolicy


def _skill_id(skill: Any, fallback: int | str) -> str:
    if isinstance(skill, dict):
        return str(skill.get("skill_id") or skill.get("id") or fallback)
    return str(getattr(skill, "skill_id", None) or getattr(skill, "id", None) or fallback)


def _skill_payload(skill: Any) -> dict[str, Any]:
    if isinstance(skill, dict):
        return dict(skill)
    return {
        "skill_id": getattr(skill, "skill_id", None),
        "name": getattr(skill, "name", ""),
        "description": getattr(skill, "description", ""),
        "input_schema": getattr(skill, "input_schema", {}),
        "output_schema": getattr(skill, "output_schema", {}),
        "executor_desc": getattr(skill, "executor_desc", ""),
        "failure_modes": getattr(skill, "failure_modes", []),
    }


def _appworld_executor_compatible(skill: dict[str, Any]) -> bool:
    skill_id = str(skill.get("skill_id") or skill.get("id") or "")
    return bool(
        skill.get("appworld_executor_compatible") is True
        or str(skill.get("executor_domain") or "").lower() == "appworld"
        or skill_id.startswith("skillx/appworld/")
    )


def _standardize_tensor(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float()
    if int(values.numel()) <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1.0e-8)


def _score_range(scores: torch.Tensor | None) -> float | None:
    if scores is None:
        return None
    values = scores.float()
    if int(values.numel()) <= 1:
        return 0.0
    return float((values.max() - values.min()).detach().cpu().item())


def _gated_standardize_tensor(
    scores: torch.Tensor | None,
    *,
    min_range: float,
) -> tuple[torch.Tensor | None, bool, float | None]:
    if scores is None:
        return None, False, None
    raw_range = _score_range(scores)
    threshold = max(float(min_range), 0.0)
    enabled = bool(raw_range is not None and raw_range > 1.0e-8 and raw_range >= threshold)
    if not enabled:
        return torch.zeros_like(scores.float()), False, raw_range
    return _standardize_tensor(scores), True, raw_range


def _filtered_ranked_indices(
    *,
    candidates: list[int],
    scores: torch.Tensor,
    skill_payloads: list[dict[str, Any]],
    top_k: int,
    dedupe_canonical_skills: bool = False,
    max_auth_like_skills: int | None = None,
    appworld_executor_compatible_only: bool = False,
) -> tuple[list[int], list[float], dict[str, int | None]]:
    if not candidates:
        return [], [], {
            "filtered_duplicate_skills": 0,
            "filtered_auth_like_skills": 0,
            "filtered_executor_incompatible_skills": 0,
            "max_auth_like_skills": max_auth_like_skills,
        }
    values, local_indices = torch.topk(scores, k=len(candidates))
    selected: list[int] = []
    selected_scores: list[float] = []
    seen_canonical: set[str] = set()
    auth_like_count = 0
    filtered_duplicate = 0
    filtered_auth = 0
    filtered_executor_incompatible = 0
    for value, local_idx in zip(values.detach().cpu().tolist(), local_indices.detach().cpu().tolist()):
        skill_idx = int(candidates[int(local_idx)])
        skill = skill_payloads[skill_idx]
        if appworld_executor_compatible_only and not _appworld_executor_compatible(skill):
            filtered_executor_incompatible += 1
            continue
        if dedupe_canonical_skills:
            canonical = _canonical_skill_key(skill)
            if canonical in seen_canonical:
                filtered_duplicate += 1
                continue
            seen_canonical.add(canonical)
        if max_auth_like_skills is not None and _is_auth_like_skill(skill):
            if auth_like_count >= int(max_auth_like_skills):
                filtered_auth += 1
                continue
            auth_like_count += 1
        selected.append(skill_idx)
        selected_scores.append(float(value))
        if len(selected) >= int(top_k):
            break
    return selected, selected_scores, {
        "filtered_duplicate_skills": filtered_duplicate,
        "filtered_auth_like_skills": filtered_auth,
        "filtered_executor_incompatible_skills": filtered_executor_incompatible,
        "max_auth_like_skills": max_auth_like_skills,
    }


@dataclass
class MultiStepSelection:
    skills: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_skill_ids(self) -> list[str]:
        return [str(skill.get("skill_id")) for skill in self.skills if skill.get("skill_id") is not None]


class MultiStepController(Protocol):
    def reset(self, task: dict[str, Any]) -> None:
        ...

    def select(
        self,
        *,
        task: dict[str, Any],
        state_text: str,
        steps: list[dict[str, Any]],
        top_k: int,
    ) -> MultiStepSelection:
        ...

    def observe(
        self,
        *,
        selection: MultiStepSelection,
        code: str,
        execute_output: str,
        step: dict[str, Any],
    ) -> None:
        ...


@dataclass
class StaticSkillProviderStepController:
    """Multi-step controller that reuses a normal top-k skill provider every step."""

    provider: SkillProvider

    @classmethod
    def from_prediction_file(
        cls,
        *,
        skill_pool_path: str | Path,
        predictions_path: str | Path,
        dedupe_canonical_skills: bool = False,
        max_auth_like_skills: int | None = None,
    ) -> "StaticSkillProviderStepController":
        return cls(
            PredictionFileSkillProvider(
                skill_pool_path=skill_pool_path,
                predictions_path=predictions_path,
                dedupe_canonical_skills=dedupe_canonical_skills,
                max_auth_like_skills=max_auth_like_skills,
            )
        )

    def reset(self, task: dict[str, Any]) -> None:
        return None

    def select(
        self,
        *,
        task: dict[str, Any],
        state_text: str,
        steps: list[dict[str, Any]],
        top_k: int,
    ) -> MultiStepSelection:
        return MultiStepSelection(
            skills=self.provider.get_skills(task, top_k=top_k),
            diagnostics={"controller": "static_skill_provider"},
        )

    def observe(
        self,
        *,
        selection: MultiStepSelection,
        code: str,
        execute_output: str,
        step: dict[str, Any],
    ) -> None:
        return None


@dataclass
class CLSTRMultiStepController:
    """CLSTR step controller that keeps recurrent belief state across AppWorld steps."""

    model: Any
    skills: list[Any]
    ranking_mode: str = "policy_blend"
    candidate_top_k: int | None = None
    candidate_source: str = "routing"
    policy_blend_alpha: float = 0.5
    learned_component_min_range: float = 0.1
    learned_component_trust_top_k: int | None = 80
    transition_scoring_mode: str = V4_1B_TRANSITION_SCORING_MODE
    transition_residual_lambda: float = 0.0
    allow_legacy_policy_skill_router: bool = False
    update_belief: bool = True
    recurrent_belief: bool = True
    dedupe_canonical_skills: bool = False
    max_auth_like_skills: int | None = None
    appworld_executor_compatible_only: bool = False

    def __post_init__(self) -> None:
        self.model.eval()
        self.skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(self.skills)]
        self.skill_payloads = [_skill_payload(skill) for skill in self.skills]
        self._skill_index_by_id = {skill_id: idx for idx, skill_id in enumerate(self.skill_ids)}
        self.m_t: torch.Tensor | None = None
        self.belief_updates = 0
        self.last_action_idx: int | None = None
        self.last_observation_text: str | None = None
        mode = str(self.ranking_mode)
        if mode not in {
            "skill_table",
            "policy_head",
            "policy_blend",
            "transition_head",
            "transition_blend",
            "policy_transition_blend",
        }:
            raise ValueError(f"unsupported CLSTR multi-step ranking_mode: {mode}")
        if mode in {"policy_head", "policy_blend", "policy_transition_blend"} and not bool(
            self.allow_legacy_policy_skill_router
        ):
            raise ValueError(
                "ranking_mode policy_head/policy_blend/policy_transition_blend is a "
                "legacy_policy_skill_router path. It uses the policy head over skill "
                "candidate embeddings and is not the current CLSTR mainline contract. "
                "Pass allow_legacy_policy_skill_router=True only for legacy AppWorld "
                "multistep reproduction."
            )
        source = str(self.candidate_source)
        if source not in {"routing", "routing_belief_union", "routing_belief_union_after_update"}:
            raise ValueError(f"unsupported CLSTR multi-step candidate_source: {source}")
        if str(self.transition_scoring_mode) not in TRANSITION_SCORING_MODES:
            raise ValueError(f"unsupported transition_scoring_mode: {self.transition_scoring_mode}")

    def reset(self, task: dict[str, Any]) -> None:
        self.m_t = None
        self.belief_updates = 0
        self.last_action_idx = None
        self.last_observation_text = None

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self.model, "device", "cpu"))

    def _candidate_embeddings(self, state_text: str, candidates: list[int]) -> torch.Tensor:
        if callable(getattr(self.model, "batch_cross_encode", None)):
            try:
                return self.model.batch_cross_encode([state_text], [candidates])
            except RuntimeError:
                pass
        skill_embs = self.model.skill_table.E.index_select(0, torch.tensor(candidates, device=self.device, dtype=torch.long))
        return skill_embs.unsqueeze(0)

    def _transition_candidate_scores(
        self,
        *,
        state_text: str,
        h_t: torch.Tensor,
        candidates: list[int],
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.last_action_idx is None:
            return None, {
                "transition_scores_available": False,
                "transition_fallback_reason": "missing_previous_action",
            }
        if not candidates:
            return None, {
                "transition_scores_available": False,
                "transition_fallback_reason": "empty_candidates",
            }
        skill_table = getattr(self.model, "skill_table", None)
        skill_embs = getattr(skill_table, "E", None)
        skill_count = int(skill_embs.size(0)) if isinstance(skill_embs, torch.Tensor) else len(self.skill_ids)
        current_labels = torch.tensor([int(self.last_action_idx)], device=self.device, dtype=torch.long)
        transition_m_obs = subspace_obs(self.model.skill_table, h_t)
        obs_text = str(self.last_observation_text or state_text or "")
        try:
            obs_emb = self.model.encode_observations([obs_text])
        except (AttributeError, RuntimeError, TypeError, ValueError):
            obs_emb = h_t
        action_emb = None
        if isinstance(skill_embs, torch.Tensor):
            try:
                action_emb = skill_embs.index_select(0, current_labels.to(skill_embs.device)).to(
                    device=h_t.device,
                    dtype=h_t.dtype,
                )
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                action_emb = None
        candidate_ids = torch.tensor([candidates], device=self.device, dtype=torch.long)
        logits, head_type, _prior_logits, _residual_logits = _transition_candidate_logits_for_mode(
            self.model,
            h_t,
            transition_m_obs,
            current_labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            residual_lambda=float(self.transition_residual_lambda),
            scoring_mode=str(self.transition_scoring_mode),
        )
        return logits.squeeze(0), {
            "transition_scores_available": True,
            "transition_fallback_reason": None,
            "transition_head_type": head_type,
            "transition_scoring_mode": str(self.transition_scoring_mode),
            "transition_residual_lambda": float(self.transition_residual_lambda),
            "transition_current_action_idx": int(self.last_action_idx),
        }

    def _topk_from_candidate_pool(
        self,
        scores: torch.Tensor,
        *,
        k: int,
        candidate_pool: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        if not candidate_pool:
            return torch.empty(0, device=scores.device, dtype=scores.dtype), []
        pool_idx = torch.tensor(candidate_pool, device=scores.device, dtype=torch.long)
        pool_scores = scores.index_select(0, pool_idx)
        output_k = min(max(1, int(k)), int(pool_scores.numel()))
        values, local_indices = torch.topk(pool_scores, k=output_k)
        global_indices = pool_idx.index_select(0, local_indices).detach().cpu().tolist()
        return values, [int(idx) for idx in global_indices]

    def select(
        self,
        *,
        task: dict[str, Any],
        state_text: str,
        steps: list[dict[str, Any]],
        top_k: int,
    ) -> MultiStepSelection:
        del task, steps
        with torch.no_grad():
            h_t = self.model.encode_states([state_text])
            if self.m_t is None or not self.recurrent_belief:
                m_current = subspace_obs(self.model.skill_table, h_t)
                if self.recurrent_belief:
                    self.m_t = m_current
            else:
                m_current = self.m_t
            routing_logits = self.model.skill_table.retrieval_logits(h_t).squeeze(0)
            if routing_logits.ndim != 1:
                raise ValueError("CLSTR multi-step routing logits must be rank-1 after squeeze")
            output_k = min(max(1, int(top_k)), int(routing_logits.numel()))
            raw_candidate_k = self.candidate_top_k
            if raw_candidate_k is None:
                raw_candidate_k = max(output_k, int(getattr(self.model, "K", output_k)))
            candidate_k = min(max(output_k, int(raw_candidate_k)), int(routing_logits.numel()))
            full_candidate_pool = list(range(int(routing_logits.numel())))
            available_candidate_mask_enabled = False
            available_candidate_count: int | None = None
            available_filtered_executor_incompatible_skills: int | None = None
            candidate_pool = full_candidate_pool
            if self.appworld_executor_compatible_only:
                compatible_pool = [
                    idx for idx in full_candidate_pool
                    if idx < len(self.skill_payloads) and _appworld_executor_compatible(self.skill_payloads[idx])
                ]
                available_candidate_mask_enabled = bool(compatible_pool)
                available_candidate_count = len(compatible_pool)
                available_filtered_executor_incompatible_skills = len(full_candidate_pool) - len(compatible_pool)
                if compatible_pool:
                    candidate_pool = compatible_pool
                    candidate_k = min(candidate_k, len(candidate_pool))
            _base_values, routing_candidates = self._topk_from_candidate_pool(
                routing_logits,
                k=candidate_k,
                candidate_pool=candidate_pool,
            )
            candidates = list(routing_candidates)
            candidate_source = str(self.candidate_source)
            belief_candidates: list[int] = []
            belief_candidate_enabled = bool(
                self.recurrent_belief
                and (
                    candidate_source == "routing_belief_union"
                    or (candidate_source == "routing_belief_union_after_update" and self.belief_updates > 0)
                )
            )
            if belief_candidate_enabled:
                belief_scores = (m_current.float() @ self.model.skill_table.E.float().t()).squeeze(0)
                _belief_values, belief_candidates = self._topk_from_candidate_pool(
                    belief_scores,
                    k=candidate_k,
                    candidate_pool=candidate_pool,
                )
                seen = set(candidates)
                for skill_idx in belief_candidates:
                    if skill_idx not in seen:
                        candidates.append(skill_idx)
                        seen.add(skill_idx)
            candidate_idx = torch.tensor(candidates, device=routing_logits.device, dtype=torch.long)
            base_values = routing_logits.index_select(0, candidate_idx)
            candidate_embs = self._candidate_embeddings(state_text, candidates)
            full_policy_output = self.model.policy_forward(
                h_t,
                m_current,
                candidate_embs,
                routing_logits=base_values,
            ).squeeze(0)
            candidate_count = len(candidates)
            policy_logits = full_policy_output[:candidate_count]
            stop_head_logit = (
                float(full_policy_output[candidate_count].detach().cpu().item())
                if int(full_policy_output.numel()) > candidate_count
                else None
            )
            mode = str(self.ranking_mode)
            transition_diag: dict[str, Any] = {}
            transition_scores: torch.Tensor | None = None
            if mode in {"transition_head", "transition_blend", "policy_transition_blend"}:
                transition_scores, transition_diag = self._transition_candidate_scores(
                    state_text=state_text,
                    h_t=h_t,
                    candidates=candidates,
                )
            base_component = _standardize_tensor(base_values)
            policy_component, policy_component_enabled, policy_component_raw_range = _gated_standardize_tensor(
                policy_logits,
                min_range=float(self.learned_component_min_range),
            )
            transition_component, transition_component_enabled, transition_component_raw_range = _gated_standardize_tensor(
                transition_scores,
                min_range=float(self.learned_component_min_range),
            )
            learned_component_enabled = False
            if mode == "skill_table":
                ranking_scores = base_values
            elif mode == "policy_head":
                ranking_scores = policy_logits
            elif mode == "policy_blend":
                alpha = min(max(float(self.policy_blend_alpha), 0.0), 1.0)
                learned_component_enabled = bool(policy_component_enabled)
                ranking_scores = (
                    (1.0 - alpha) * base_component + alpha * policy_component
                    if learned_component_enabled and policy_component is not None
                    else base_values
                )
            elif mode == "transition_head":
                ranking_scores = transition_scores if transition_scores is not None else base_values
            elif mode == "transition_blend":
                alpha = min(max(float(self.policy_blend_alpha), 0.0), 1.0)
                learned_component_enabled = bool(transition_component_enabled)
                ranking_scores = (
                    (1.0 - alpha) * base_component + alpha * transition_component
                    if learned_component_enabled and transition_component is not None
                    else base_values
                )
            else:
                alpha = min(max(float(self.policy_blend_alpha), 0.0), 1.0)
                learned_parts = []
                if policy_component_enabled and policy_component is not None:
                    learned_parts.append(policy_component)
                if transition_component_enabled and transition_component is not None:
                    learned_parts.append(transition_component)
                learned_component_enabled = bool(learned_parts)
                if learned_parts:
                    learned_component = torch.stack(learned_parts, dim=0).mean(dim=0)
                    ranking_scores = (1.0 - alpha) * base_component + alpha * learned_component
                else:
                    ranking_scores = base_values
            rank_candidates = candidates
            rank_scores = ranking_scores
            trust_region_active = False
            trusted_candidate_count = len(candidates)
            outside_trust_candidate_count = 0
            trust_top_k = self.learned_component_trust_top_k
            if learned_component_enabled and trust_top_k is not None and int(trust_top_k) > 0:
                trusted_candidate_count = min(len(candidates), max(output_k, int(trust_top_k)))
                outside_trust_candidate_count = max(0, len(candidates) - trusted_candidate_count)
                trust_region_active = outside_trust_candidate_count > 0
                if trust_region_active:
                    rank_candidates = candidates[:trusted_candidate_count]
                    rank_scores = ranking_scores[:trusted_candidate_count]
            selected_indices, selected_scores, filter_diag = _filtered_ranked_indices(
                candidates=rank_candidates,
                scores=rank_scores,
                skill_payloads=self.skill_payloads,
                top_k=output_k,
                dedupe_canonical_skills=self.dedupe_canonical_skills,
                max_auth_like_skills=self.max_auth_like_skills,
                appworld_executor_compatible_only=self.appworld_executor_compatible_only,
            )
            local_index_by_global = {int(skill_idx): local_idx for local_idx, skill_idx in enumerate(candidates)}
            selected_candidate_local_indices = [
                int(local_index_by_global[idx])
                for idx in selected_indices
                if int(idx) in local_index_by_global
            ]
            policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
            selected_log_probs = [
                float(policy_log_probs[int(local_idx)].detach().cpu().item())
                for local_idx in selected_candidate_local_indices
            ]
            selected_skills = [self.skill_payloads[idx] for idx in selected_indices]
            return MultiStepSelection(
                skills=selected_skills,
                diagnostics={
                    "controller": "clstr_multistep",
                    "ranking_mode": mode,
                    "legacy_policy_skill_router": bool(mode in {"policy_head", "policy_blend", "policy_transition_blend"}),
                    "allow_legacy_policy_skill_router": bool(self.allow_legacy_policy_skill_router),
                    "candidate_source": candidate_source,
                    "belief_candidate_enabled": belief_candidate_enabled,
                    "routing_candidate_skill_ids": [self.skill_ids[idx] for idx in routing_candidates],
                    "belief_candidate_skill_ids": [self.skill_ids[idx] for idx in belief_candidates],
                    "candidate_top_k": candidate_k,
                    "candidate_skill_ids": [self.skill_ids[idx] for idx in candidates],
                    "selected_action_indices": selected_indices,
                    "selected_candidate_local_indices": selected_candidate_local_indices,
                    "selected_scores": selected_scores,
                    "selected_log_probs": selected_log_probs,
                    "policy_log_probs": [float(item) for item in policy_log_probs.detach().cpu().tolist()],
                    "candidate_policy_logits": [float(item) for item in policy_logits.detach().cpu().tolist()],
                    "learned_component_min_range": float(self.learned_component_min_range),
                    "learned_component_enabled": bool(learned_component_enabled),
                    "learned_component_trust_top_k": (
                        int(self.learned_component_trust_top_k)
                        if self.learned_component_trust_top_k is not None
                        else None
                    ),
                    "learned_component_trust_region_active": bool(trust_region_active),
                    "learned_component_trusted_candidate_count": int(trusted_candidate_count),
                    "learned_component_outside_trust_candidate_count": int(outside_trust_candidate_count),
                    "policy_component_enabled": bool(policy_component_enabled),
                    "policy_component_raw_range": policy_component_raw_range,
                    "transition_component_enabled": bool(transition_component_enabled),
                    "transition_component_raw_range": transition_component_raw_range,
                    "belief_updates": int(self.belief_updates),
                    "belief_norm": float(m_current.norm(dim=-1).mean().detach().cpu().item()),
                    "recurrent_belief": bool(self.recurrent_belief),
                    "stop_head_logit": stop_head_logit,
                    "executor_compatible_only": bool(self.appworld_executor_compatible_only),
                    "available_candidate_mask_enabled": bool(available_candidate_mask_enabled),
                    "available_candidate_count": available_candidate_count,
                    "candidate_pool_before_available_filter": int(routing_logits.numel()),
                    "available_filtered_executor_incompatible_skills": available_filtered_executor_incompatible_skills,
                    **transition_diag,
                    **filter_diag,
                },
            )

    def observe(
        self,
        *,
        selection: MultiStepSelection,
        code: str,
        execute_output: str,
        step: dict[str, Any],
    ) -> None:
        del code
        selected_ids = selection.selected_skill_ids
        if selected_ids:
            action_idx = self._skill_index_by_id.get(selected_ids[0], int(getattr(self.model, "stop_idx", len(self.skills))))
        else:
            action_idx = int(getattr(self.model, "stop_idx", len(self.skills)))
        self.last_action_idx = action_idx if 0 <= int(action_idx) < len(self.skills) else None
        self.last_observation_text = str(execute_output or step.get("execute_output") or "")
        if not self.recurrent_belief or not self.update_belief or self.m_t is None:
            return
        with torch.no_grad():
            a_t = torch.tensor([int(action_idx)], device=self.device, dtype=torch.long)
            obs_text = self.last_observation_text
            o_t_emb = self.model.encode_observations([obs_text])
            next_state_text = str(step.get("next_state_text") or obs_text)
            self.m_t, _m_hat, _m_obs = self.model.step_update(self.m_t, a_t, o_t_emb, [next_state_text])
            self.belief_updates += 1


@dataclass
class LiveEmbeddingStepController:
    """Per-step embedding retriever that reranks skills from the current state text."""

    skills: list[dict[str, Any]]
    encode_texts: Callable[[list[str]], torch.Tensor]
    query_text_fn: Callable[[str], str]
    skill_text_fn: Callable[[dict[str, Any]], str]
    scorer: Any | None = None
    normalize_embeddings: bool = True
    controller_name: str = "live_embedding_step"
    dedupe_canonical_skills: bool = False
    max_auth_like_skills: int | None = None
    appworld_executor_compatible_only: bool = False

    def __post_init__(self) -> None:
        self.skill_ids = [_skill_id(skill, idx) for idx, skill in enumerate(self.skills)]
        self.skill_payloads = [_skill_payload(skill) for skill in self.skills]
        skill_texts = [self.skill_text_fn(skill) for skill in self.skills]
        self.skill_embs = self._maybe_normalize(self.encode_texts(skill_texts).float())
        if self.scorer is not None and callable(getattr(self.scorer, "eval", None)):
            self.scorer.eval()

    def _maybe_normalize(self, embs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_embeddings:
            return embs
        return torch.nn.functional.normalize(embs.float(), p=2, dim=-1)

    def reset(self, task: dict[str, Any]) -> None:
        return None

    def rank_state(self, state_text: str, top_k: int) -> tuple[list[int], torch.Tensor]:
        query_text = self.query_text_fn(state_text)
        with torch.no_grad():
            query_emb = self._maybe_normalize(self.encode_texts([query_text]).float())
            skill_embs = self.skill_embs
            if self.scorer is not None:
                scores = self.scorer.score(query_emb, skill_embs).squeeze(0)
            else:
                scores = (query_emb @ skill_embs.t()).squeeze(0)
            output_k = min(max(1, int(top_k)), len(self.skill_ids))
            values, indices = torch.topk(scores, k=output_k)
        return [int(idx) for idx in indices.detach().cpu().tolist()], values.detach().cpu()

    def select(
        self,
        *,
        task: dict[str, Any],
        state_text: str,
        steps: list[dict[str, Any]],
        top_k: int,
    ) -> MultiStepSelection:
        del task, steps
        output_k = min(max(1, int(top_k)), len(self.skill_ids))
        candidates, scores = self.rank_state(state_text, top_k=len(self.skill_ids))
        selected_indices, selected_scores, filter_diag = _filtered_ranked_indices(
            candidates=candidates,
            scores=scores,
            skill_payloads=self.skill_payloads,
            top_k=output_k,
            dedupe_canonical_skills=self.dedupe_canonical_skills,
            max_auth_like_skills=self.max_auth_like_skills,
            appworld_executor_compatible_only=self.appworld_executor_compatible_only,
        )
        return MultiStepSelection(
            skills=[self.skill_payloads[idx] for idx in selected_indices],
            diagnostics={
                "controller": self.controller_name,
                "candidate_source": "current_state_text",
                "selected_scores": selected_scores,
                "selected_action_indices": selected_indices,
                "candidate_skill_ids": [self.skill_ids[idx] for idx in selected_indices],
                "executor_compatible_only": bool(self.appworld_executor_compatible_only),
                **filter_diag,
            },
        )

    def observe(
        self,
        *,
        selection: MultiStepSelection,
        code: str,
        execute_output: str,
        step: dict[str, Any],
    ) -> None:
        return None


class HybridCLSTRSkillRouterController:
    """Use SkillRouter live retrieval as candidate prior, then rerank with CLSTR belief policy."""

    def __init__(
        self,
        *,
        model: Any,
        skills: list[Any],
        retriever: Any,
        clstr_alpha: float = 0.25,
        candidate_top_k: int | None = None,
        update_belief: bool = True,
        recurrent_belief: bool = True,
        dedupe_canonical_skills: bool = False,
        max_auth_like_skills: int | None = None,
        appworld_executor_compatible_only: bool = False,
    ) -> None:
        self.retriever = retriever
        self.clstr_alpha = min(max(float(clstr_alpha), 0.0), 1.0)
        self.candidate_top_k = candidate_top_k
        self.clstr = CLSTRMultiStepController(
            model=model,
            skills=skills,
            ranking_mode="policy_head",
            candidate_top_k=candidate_top_k,
            allow_legacy_policy_skill_router=True,
            update_belief=update_belief,
            recurrent_belief=recurrent_belief,
            dedupe_canonical_skills=dedupe_canonical_skills,
            max_auth_like_skills=max_auth_like_skills,
            appworld_executor_compatible_only=appworld_executor_compatible_only,
        )

    def reset(self, task: dict[str, Any]) -> None:
        if callable(getattr(self.retriever, "reset", None)):
            self.retriever.reset(task)
        self.clstr.reset(task)

    def select(
        self,
        *,
        task: dict[str, Any],
        state_text: str,
        steps: list[dict[str, Any]],
        top_k: int,
    ) -> MultiStepSelection:
        del task, steps
        model = self.clstr.model
        with torch.no_grad():
            output_k = min(max(1, int(top_k)), len(self.clstr.skill_ids))
            raw_candidate_k = self.candidate_top_k or max(output_k, int(getattr(model, "K", output_k)))
            candidate_k = min(max(output_k, int(raw_candidate_k)), len(self.clstr.skill_ids))
            candidates, skillrouter_scores = self.retriever.rank_state(state_text, top_k=candidate_k)
            h_t = model.encode_states([state_text])
            if self.clstr.m_t is None or not self.clstr.recurrent_belief:
                m_current = subspace_obs(model.skill_table, h_t)
                if self.clstr.recurrent_belief:
                    self.clstr.m_t = m_current
            else:
                m_current = self.clstr.m_t
            candidate_embs = self.clstr._candidate_embeddings(state_text, candidates)
            full_policy_output = model.policy_forward(
                h_t,
                m_current,
                candidate_embs,
                routing_logits=skillrouter_scores.to(h_t.device),
            ).squeeze(0)
            candidate_count = len(candidates)
            policy_logits = full_policy_output[:candidate_count]
            stop_head_logit = (
                float(full_policy_output[candidate_count].detach().cpu().item())
                if int(full_policy_output.numel()) > candidate_count
                else None
            )
            alpha = self.clstr_alpha
            ranking_scores = (1.0 - alpha) * _standardize_tensor(skillrouter_scores.to(policy_logits.device)) + alpha * _standardize_tensor(policy_logits)
            selected_indices, selected_scores, filter_diag = _filtered_ranked_indices(
                candidates=candidates,
                scores=ranking_scores,
                skill_payloads=self.clstr.skill_payloads,
                top_k=output_k,
                dedupe_canonical_skills=self.clstr.dedupe_canonical_skills,
                max_auth_like_skills=self.clstr.max_auth_like_skills,
                appworld_executor_compatible_only=self.clstr.appworld_executor_compatible_only,
            )
            local_index_by_global = {int(skill_idx): local_idx for local_idx, skill_idx in enumerate(candidates)}
            selected_candidate_local_indices = [
                int(local_index_by_global[idx])
                for idx in selected_indices
                if int(idx) in local_index_by_global
            ]
            policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
            selected_log_probs = [
                float(policy_log_probs[int(local_idx)].detach().cpu().item())
                for local_idx in selected_candidate_local_indices
            ]
        return MultiStepSelection(
            skills=[self.clstr.skill_payloads[idx] for idx in selected_indices],
            diagnostics={
                "controller": "hybrid_clstr_skillrouter",
                "candidate_source": "skillrouter_live",
                "clstr_alpha": alpha,
                "candidate_top_k": candidate_k,
                "candidate_skill_ids": [self.clstr.skill_ids[idx] for idx in candidates],
                "selected_action_indices": selected_indices,
                "selected_candidate_local_indices": selected_candidate_local_indices,
                "selected_scores": selected_scores,
                "selected_log_probs": selected_log_probs,
                "policy_log_probs": [float(item) for item in policy_log_probs.detach().cpu().tolist()],
                "candidate_policy_logits": [float(item) for item in policy_logits.detach().cpu().tolist()],
                "belief_updates": int(self.clstr.belief_updates),
                "belief_norm": float(m_current.norm(dim=-1).mean().detach().cpu().item()),
                "recurrent_belief": bool(self.clstr.recurrent_belief),
                "stop_head_logit": stop_head_logit,
                "executor_compatible_only": bool(self.clstr.appworld_executor_compatible_only),
                **filter_diag,
            },
        )

    def observe(
        self,
        *,
        selection: MultiStepSelection,
        code: str,
        execute_output: str,
        step: dict[str, Any],
    ) -> None:
        self.clstr.observe(selection=selection, code=code, execute_output=execute_output, step=step)


def _task_instruction(task: dict[str, Any]) -> str:
    return str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")


def _compact(value: Any, max_chars: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= int(max_chars):
        return text
    return text[: max(0, int(max_chars) - 3)] + "..."


def _history_lines(
    steps: list[dict[str, Any]],
    max_step_chars: int = 900,
    include_selected_skill_ids: bool = True,
) -> list[str]:
    if not steps:
        return ["No previous execution steps."]
    lines: list[str] = []
    for step in steps:
        selected = ", ".join(str(item) for item in step.get("selected_skill_ids", [])) or "none"
        code = _compact(step.get("qwen_code") or step.get("code") or "", max_step_chars // 2)
        output = _compact(step.get("execute_output") or "", max_step_chars)
        lines.append(f"Step {int(step.get('step_idx', len(lines)))}")
        if include_selected_skill_ids:
            lines.append(f"- selected_skill_ids: {selected}")
        lines.extend(
            [
                f"- execution_ok: {bool(step.get('execution_ok', False))}",
                f"- task_completed: {bool(step.get('task_completed', False))}",
                f"- code: {code}",
                f"- environment_output: {output}",
            ]
        )
    return lines


def build_multistep_state_text(
    *,
    task: dict[str, Any],
    steps: list[dict[str, Any]],
    max_chars: int = 6000,
    include_selected_skill_ids: bool = True,
) -> str:
    required_apps = ", ".join(str(app) for app in task.get("required_apps", []) if str(app) != "supervisor") or "unknown"
    sections = [
        "[User Goal]",
        _task_instruction(task),
        "",
        f"Required apps: {required_apps}",
        "",
        "[Execution History]",
        *_history_lines(steps, include_selected_skill_ids=include_selected_skill_ids),
        "",
        "[Controller Instruction]",
        "Need choose the next useful SkillX skill/context and write only the next executable Python code.",
    ]
    return "\n".join(sections)[: int(max_chars)]


def build_multistep_executor_prompt(
    *,
    task: dict[str, Any],
    state_text: str,
    skills: list[dict[str, Any]],
    api_docs_context: str,
    skill_context_mode: str,
    valid_api_refs: set[tuple[str, str]] | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> str:
    prompt_task = dict(task)
    prompt_task["instruction_text"] = state_text
    return build_appworld_executor_prompt(
        task=prompt_task,
        skills=skills,
        api_docs_context=api_docs_context,
        skill_context_mode=skill_context_mode,
        valid_api_refs=valid_api_refs,
        diagnostics_out=diagnostics_out,
    )


def _positive_skill_ids_for_step(task: dict[str, Any], step_idx: int) -> list[str]:
    step_labels = task.get("step_positive_skill_ids")
    if isinstance(step_labels, list) and step_idx < len(step_labels):
        row = step_labels[step_idx]
        if isinstance(row, list):
            return [str(item) for item in row]
    return [str(item) for item in task.get("positive_skill_ids", [])]


def _step_hit(selected_skill_ids: list[str], positive_skill_ids: list[str]) -> bool | None:
    if not positive_skill_ids:
        return None
    return bool(set(selected_skill_ids) & set(positive_skill_ids))


def run_appworld_multistep_executor_eval(
    *,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_dir: str | Path,
    appworld_root: str | Path,
    appworld_cache: str | Path,
    method: str,
    controller: MultiStepController,
    generator: Any,
    world_factory: Callable[..., Any] | None = None,
    max_tasks: int | None = None,
    max_steps: int = 5,
    top_k: int = 5,
    max_apis_per_app: int = 12,
    skill_context_mode: str = "safe_metadata",
    experiment_name: str = "clstr_appworld_multistep",
    timeout_seconds: int = 60,
    max_interactions: int = 10,
    use_stop_head: bool = False,
    stop_policy: MultiStepStopPolicy | None = None,
    preflight_max_repairs: int = 2,
) -> dict[str, Any]:
    root, cache = configure_appworld_paths(appworld_root, appworld_cache)
    os.environ.setdefault("IPYTHONDIR", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython")
    tasks = read_jsonl(tasks_path)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    factory = world_factory or _default_world_factory()
    output_dir = Path(output_dir)
    runs_path = output_dir / "runs.jsonl"
    run_rows: list[dict[str, Any]] = []
    stop_policy = stop_policy or MultiStepStopPolicy(use_stop_head=bool(use_stop_head))

    for raw_task in tasks:
        task = enrich_task_with_appworld_specs(raw_task, root)
        task_id = str(task.get("task_id") or task.get("query_id"))
        query_id = str(task.get("query_id") or task_id)
        controller.reset(task)
        world = None
        steps: list[dict[str, Any]] = []
        row: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "query_id": query_id,
            "split": task.get("split"),
            "user_goal": _task_instruction(task),
            "reset_ok": False,
            "success": False,
            "final_success": False,
            "task_completed": False,
            "evaluation_success": False,
            "steps": steps,
        }
        try:
            world = factory(
                task_id,
                experiment_name=f"{experiment_name}_{method}",
                max_interactions=max_interactions,
                timeout_seconds=timeout_seconds,
                load_ground_truth=True,
                ground_truth_mode="minimal",
                show_api_response_schemas=False,
            )
            row["reset_ok"] = True
            api_docs_context = build_api_docs_context(
                appworld_root=root,
                required_apps=[str(item) for item in task.get("required_apps", [])],
                api_refs=[str(item) for item in task.get("api_refs", [])],
                max_apis_per_app=max_apis_per_app,
            )
            valid_api_refs = load_appworld_api_refs(
                appworld_root=root,
                required_apps=[str(item) for item in task.get("required_apps", [])],
                api_refs=[str(item) for item in task.get("api_refs", [])],
            )
            for step_idx in range(max(1, int(max_steps))):
                state_text = build_multistep_state_text(task=task, steps=steps)
                selection = controller.select(task=task, state_text=state_text, steps=steps, top_k=top_k)
                positive_skill_ids = _positive_skill_ids_for_step(task, step_idx)
                prompt_state_text = build_multistep_state_text(
                    task=task,
                    steps=steps,
                    include_selected_skill_ids=False,
                )
                skill_handoff: dict[str, Any] = {}
                prompt = build_multistep_executor_prompt(
                    task=task,
                    state_text=prompt_state_text,
                    skills=selection.skills,
                    api_docs_context=api_docs_context,
                    skill_context_mode=skill_context_mode,
                    valid_api_refs=valid_api_refs,
                    diagnostics_out=skill_handoff,
                )
                step: dict[str, Any] = {
                    "step_idx": step_idx,
                    "state_text": state_text,
                    "selected_skill_ids": selection.selected_skill_ids,
                    "positive_skill_ids": positive_skill_ids,
                    "selection": selection.diagnostics,
                    "skill_handoff": skill_handoff,
                    "prompt_chars": len(prompt),
                    "generation_ok": False,
                    "execution_ok": False,
                    "task_completed": False,
                    "evaluation_success": False,
                }
                try:
                    response = _call_generator(generator, prompt)
                    code = extract_python_code(response)
                    step.update({"generation_ok": True, "raw_response": response, "qwen_code": code, "code": code})
                    sanitizer = sanitize_appworld_code(code)
                    step["code_sanitizer"] = {key: value for key, value in sanitizer.items() if key != "code"}
                    if bool(sanitizer.get("changed")):
                        code = str(sanitizer.get("code") or code)
                        step["code"] = code
                    preflight_errors: list[dict[str, Any]] = []
                    repair_responses: list[str] = []
                    repair_sanitizers: list[dict[str, Any]] = []
                    preflight = preflight_appworld_code(
                        code,
                        api_docs_context,
                        valid_api_refs=valid_api_refs,
                    )
                    repair_count = 0
                    while not bool(preflight.get("ok")) and repair_count < int(preflight_max_repairs):
                        preflight_errors.append(dict(preflight))
                        repair_prompt = build_appworld_code_repair_prompt(
                            original_prompt=prompt,
                            code=code,
                            preflight=preflight,
                        )
                        repair_response = _call_generator(generator, repair_prompt)
                        repair_responses.append(repair_response)
                        code = extract_python_code(repair_response)
                        repair_sanitizer = sanitize_appworld_code(code)
                        repair_sanitizers.append(
                            {key: value for key, value in repair_sanitizer.items() if key != "code"}
                        )
                        if bool(repair_sanitizer.get("changed")):
                            code = str(repair_sanitizer.get("code") or code)
                        repair_count += 1
                        preflight = preflight_appworld_code(
                            code,
                            api_docs_context,
                            valid_api_refs=valid_api_refs,
                        )
                    step["preflight_ok"] = bool(preflight.get("ok"))
                    step["preflight"] = dict(preflight)
                    step["preflight_errors"] = preflight_errors
                    step["preflight_repair_count"] = repair_count
                    if repair_responses:
                        step["repair_raw_responses"] = repair_responses
                        step["repair_code_sanitizers"] = repair_sanitizers
                        step["code"] = code
                        step["qwen_code"] = code
                    if bool(preflight.get("ok")):
                        execute_output = str(world.execute(code))
                        step["execute_output"] = execute_output
                        step["execution_ok"] = _execution_succeeded(execute_output)
                        try:
                            step["task_completed"] = bool(world.task_completed())
                        except Exception as exc:
                            step["task_completed_error"] = repr(exc)
                        try:
                            step["evaluate"] = _json_safe(world.evaluate(suppress_errors=True))
                            step["evaluation_success"] = _evaluation_succeeded(step["evaluate"])
                        except Exception as exc:
                            step["evaluate_error"] = repr(exc)
                            step["evaluation_success"] = False
                    else:
                        execute_output = f"Preflight failed before execution: {preflight.get('error', '')}"
                        step["execute_output"] = execute_output
                        step["execution_ok"] = False
                    step["next_state_text"] = build_multistep_state_text(task=task, steps=[*steps, step])
                    controller.observe(selection=selection, code=code, execute_output=execute_output, step=step)
                except Exception as exc:
                    step["error"] = repr(exc)
                steps.append(step)
                selection_diag = getattr(selection, "diagnostics", {}) if selection is not None else {}
                stop_head_logit = selection_diag.get("stop_head_logit") if isinstance(selection_diag, dict) else None
                stop_decision = stop_policy.decide(
                    done=bool(step.get("task_completed", False)),
                    success=bool(step.get("evaluation_success", False)),
                    stop_head_logit=stop_head_logit,
                )
                step["stop_decision"] = {
                    "should_stop": bool(stop_decision.should_stop),
                    "reason": stop_decision.reason,
                }
                if bool(stop_decision.should_stop):
                    break
            if steps:
                last = steps[-1]
                row["task_completed"] = bool(last.get("task_completed", False))
                row["evaluation_success"] = bool(last.get("evaluation_success", False))
            row["success"] = bool(row["task_completed"] and row["evaluation_success"])
            row["final_success"] = bool(row["success"])
        except Exception as exc:
            row["error"] = repr(exc)
        finally:
            if world is not None and callable(getattr(world, "close", None)):
                world.close()
        run_rows.append(row)

    write_jsonl(runs_path, run_rows)
    total_steps = sum(len(row.get("steps", [])) for row in run_rows)
    labeled_hits = 0
    labeled_steps = 0
    execution_failures = 0
    generation_failures = 0
    preflight_repair_count = 0
    preflight_failures = 0
    for row in run_rows:
        for step in row.get("steps", []):
            generation_failures += int(not step.get("generation_ok", False))
            execution_failures += int(step.get("generation_ok", False) and not step.get("execution_ok", False))
            preflight_repair_count += int(step.get("preflight_repair_count", 0) or 0)
            preflight_failures += int(step.get("generation_ok", False) and not step.get("preflight_ok", True))
            hit = _step_hit(list(step.get("selected_skill_ids", [])), list(step.get("positive_skill_ids", [])))
            if hit is not None:
                labeled_steps += 1
                labeled_hits += int(hit)
    success_count = sum(1 for row in run_rows if row.get("success"))
    task_completed_count = sum(1 for row in run_rows if row.get("task_completed"))
    evaluate_success_count = sum(1 for row in run_rows if row.get("evaluation_success"))
    report = {
        "status": "ok",
        "method": method,
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "appworld_root": str(root),
        "appworld_cache": str(cache),
        "runs_path": str(runs_path),
        "task_count": len(run_rows),
        "success_count": success_count,
        "success_rate": round(success_count / max(len(run_rows), 1), 6),
        "generation_failures": generation_failures,
        "execution_failures": execution_failures,
        "preflight_repair_count": preflight_repair_count,
        "preflight_failures": preflight_failures,
        "task_completed_count": task_completed_count,
        "evaluate_success_count": evaluate_success_count,
        "average_steps": round(total_steps / max(len(run_rows), 1), 6),
        "step_positive_skill_hit_rate": (
            round(labeled_hits / labeled_steps, 6) if labeled_steps else None
        ),
        "step_positive_labeled_count": labeled_steps,
        "stop_accuracy": None,
        "stop_accuracy_note": "not_available_until_step_stop_labels_are_provided",
        "top_k": int(top_k),
        "max_steps": int(max_steps),
        "skill_context_mode": str(skill_context_mode),
        "stop_policy": {
            "stop_on_done": bool(stop_policy.stop_on_done),
            "stop_on_success": bool(stop_policy.stop_on_success),
            "use_stop_head": bool(stop_policy.use_stop_head),
            "stop_head_threshold": float(stop_policy.stop_head_threshold),
        },
        "caveat": "Multi-step AppWorld runtime loop; final success is official task_completed/evaluate.",
    }
    write_json(output_dir / "report.json", report)
    return report


def _load_multistep_report(path: str | Path) -> dict[str, Any]:
    return read_json(path, default={}) or {}


def build_multistep_executor_comparison(report_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        report = _load_multistep_report(path)
        rows.append(
            {
                "path": str(path),
                "status": report.get("status", "missing"),
                "method": report.get("method", Path(path).parent.name),
                "task_count": report.get("task_count"),
                "success_count": report.get("success_count"),
                "success_rate": report.get("success_rate"),
                "generation_failures": report.get("generation_failures"),
                "execution_failures": report.get("execution_failures"),
                "task_completed_count": report.get("task_completed_count"),
                "evaluate_success_count": report.get("evaluate_success_count"),
                "average_steps": report.get("average_steps"),
                "step_positive_skill_hit_rate": report.get("step_positive_skill_hit_rate"),
                "step_positive_labeled_count": report.get("step_positive_labeled_count"),
                "stop_accuracy": report.get("stop_accuracy"),
                "top_k": report.get("top_k"),
                "max_steps": report.get("max_steps"),
                "skill_context_mode": report.get("skill_context_mode"),
                "runs_path": report.get("runs_path"),
            }
        )
    payload = {
        "status": "ok" if rows else "blocked",
        "comparison_role": "AppWorld multi-step executor comparison with official task_completed/evaluate success.",
        "rows": rows,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "comparison.json", payload)
    lines = [
        "# AppWorld Multi-Step Executor Comparison",
        "",
        "| method | status | task_count | success_count | success_rate | task_completed | evaluate_success | avg_steps | step_hit | gen_fail | exec_fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {task_count} | {success_count} | {success_rate} | {task_completed_count} | {evaluate_success_count} | {average_steps} | {step_positive_skill_hit_rate} | {generation_failures} | {execution_failures} |".format(
                method=row["method"],
                status=row["status"],
                task_count=row["task_count"],
                success_count=row["success_count"],
                success_rate=row["success_rate"],
                task_completed_count=row["task_completed_count"],
                evaluate_success_count=row["evaluate_success_count"],
                average_steps=row["average_steps"],
                step_positive_skill_hit_rate=row["step_positive_skill_hit_rate"],
                generation_failures=row["generation_failures"],
                execution_failures=row["execution_failures"],
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _trajectory_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_idx": int(step.get("step_idx", 0)),
        "state_text": str(step.get("state_text") or ""),
        "selected_skill_ids": [str(item) for item in step.get("selected_skill_ids", [])],
        "positive_skill_ids": [str(item) for item in step.get("positive_skill_ids", [])],
        "qwen_code": str(step.get("qwen_code") or step.get("code") or ""),
        "execute_output": str(step.get("execute_output") or ""),
        "execution_ok": bool(step.get("execution_ok", False)),
        "task_completed": bool(step.get("task_completed", False)),
        "evaluate": _json_safe(step.get("evaluate", {})),
    }


def build_train_multistep_trajectories_from_runs(
    *,
    runs_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    allowed_split: str = "train",
    require_success: bool = True,
    supervision_type: str = "train_success_rollout",
) -> dict[str, Any]:
    rows = read_jsonl(runs_path)
    output_rows: list[dict[str, Any]] = []
    skipped_by_split = 0
    skipped_unsuccessful = 0
    for row in rows:
        split = str(row.get("split") or "")
        if split != str(allowed_split):
            skipped_by_split += 1
            continue
        final_success = bool(row.get("final_success", row.get("success", False)))
        if require_success and not final_success:
            skipped_unsuccessful += 1
            continue
        output_rows.append(
            {
                "task_id": str(row.get("task_id") or row.get("query_id")),
                "query_id": str(row.get("query_id") or row.get("task_id")),
                "split": split,
                "user_goal": str(row.get("user_goal") or ""),
                "supervision_type": str(supervision_type),
                "leakage_guard": f"{allowed_split}_split_only",
                "final_success": final_success,
                "steps": [_trajectory_step(step) for step in row.get("steps", [])],
            }
        )
    written = write_jsonl(output_path, output_rows)
    manifest = {
        "status": "ok" if written else "empty",
        "runs_path": str(runs_path),
        "output_path": str(output_path),
        "allowed_split": str(allowed_split),
        "require_success": bool(require_success),
        "supervision_type": str(supervision_type),
        "input_run_count": len(rows),
        "trajectory_count": written,
        "skipped_by_split": skipped_by_split,
        "skipped_unsuccessful": skipped_unsuccessful,
        "leakage_guard": f"{allowed_split}_split_only",
        "caveat": "Use only train split successful AppWorld multi-step rollouts for training data export.",
    }
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def build_appworld_oracle_multistep_trajectories(
    *,
    appworld_root: str | Path,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    allowed_split: str = "train",
    max_tasks: int | None = None,
    compress_consecutive: bool = True,
) -> dict[str, Any]:
    from clstr.appworld_act_verified_pairs import _build_ref_index, _map_api_calls_to_skill_steps
    from clstr.appworld_dynamic_routing_audit import _read_jsonl as _read_dynamic_jsonl

    tasks = read_jsonl(tasks_path)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    skills = _read_dynamic_jsonl(skill_pool_path)
    ref_index = _build_ref_index(skills)
    output_rows: list[dict[str, Any]] = []
    skipped_by_split = 0
    skipped_without_steps = 0
    skipped_reasons: dict[str, int] = {}

    for task in tasks:
        split = str(task.get("split") or "")
        if split != str(allowed_split):
            skipped_by_split += 1
            continue
        mapped_steps, skipped = _map_api_calls_to_skill_steps(
            appworld_root=Path(appworld_root),
            task=task,
            skills=skills,
            ref_index=ref_index,
            compress_consecutive=compress_consecutive,
        )
        for key, value in skipped.items():
            skipped_reasons[key] = skipped_reasons.get(key, 0) + int(value)
        if not mapped_steps:
            skipped_without_steps += 1
            continue

        history_steps: list[dict[str, Any]] = []
        trajectory_steps: list[dict[str, Any]] = []
        for step_idx, mapped in enumerate(mapped_steps):
            state_text = build_multistep_state_text(task=task, steps=history_steps)
            step = {
                "step_idx": int(step_idx),
                "state_text": state_text,
                "selected_skill_ids": [mapped.skill_id],
                "positive_skill_ids": [mapped.skill_id],
                "qwen_code": "",
                "execute_output": mapped.observation,
                "execution_ok": True,
                "task_completed": step_idx == len(mapped_steps) - 1,
                "evaluate": {
                    "success": step_idx == len(mapped_steps) - 1,
                    "oracle_api_trace": True,
                },
                "api_ref": mapped.api_ref,
                "method": mapped.method,
                "url": mapped.url,
            }
            trajectory_steps.append(step)
            history_steps.append(step)

        output_rows.append(
            {
                "task_id": str(task.get("task_id") or task.get("query_id")),
                "query_id": str(task.get("query_id") or task.get("task_id")),
                "split": split,
                "user_goal": _task_instruction(task),
                "supervision_type": "train_oracle_api_trace",
                "execution_source": "ground_truth_api_calls_not_qwen_execution",
                "leakage_guard": f"{allowed_split}_split_only",
                "final_success": True,
                "steps": trajectory_steps,
            }
        )

    written = write_jsonl(output_path, output_rows)
    manifest = {
        "status": "ok" if written else "empty",
        "appworld_root": str(appworld_root),
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "output_path": str(output_path),
        "allowed_split": str(allowed_split),
        "task_count": len(tasks),
        "trajectory_count": written,
        "skipped_by_split": skipped_by_split,
        "skipped_without_steps": skipped_without_steps,
        "skipped_reasons": skipped_reasons,
        "supervision_type": "train_oracle_api_trace",
        "execution_source": "ground_truth_api_calls_not_qwen_execution",
        "leakage_guard": f"{allowed_split}_split_only",
        "caveat": "Oracle API trace gives train-only step skill labels, but it is not a Qwen-generated AppWorld execution trajectory.",
    }
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def build_appworld_eval_step_label_tasks(
    *,
    appworld_root: str | Path,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    allowed_split: str = "dev",
    max_tasks: int | None = None,
    compress_consecutive: bool = True,
    appworld_executor_compatible_only: bool = True,
) -> dict[str, Any]:
    """Augment eval tasks with oracle API-trace step labels without creating train data."""

    if str(allowed_split) == "train":
        raise ValueError(
            "build_appworld_eval_step_label_tasks is eval-only; use "
            "build_appworld_oracle_multistep_trajectories for train split trajectories."
        )

    from clstr.appworld_act_verified_pairs import _build_ref_index, _map_api_calls_to_skill_steps
    from clstr.appworld_dynamic_routing_audit import _read_jsonl as _read_dynamic_jsonl

    tasks = read_jsonl(tasks_path)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    raw_skills = _read_dynamic_jsonl(skill_pool_path)
    skills = (
        [skill for skill in raw_skills if _appworld_executor_compatible(skill)]
        if appworld_executor_compatible_only
        else raw_skills
    )
    ref_index = _build_ref_index(skills)
    output_rows: list[dict[str, Any]] = []
    skipped_by_split = 0
    skipped_without_steps = 0
    skipped_reasons: dict[str, int] = {}
    label_set_sizes: list[int] = []
    split = str(allowed_split)
    leakage_guard = f"{split}_eval_only_no_training"
    step_label_source = f"appworld_{split}_ground_truth_api_calls_mapped_to_skillx_eval_only"

    def _labels_for_api_ref(api_ref: str) -> tuple[list[str], list[str]]:
        ids: list[str] = []
        names: list[str] = []
        seen: set[str] = set()
        for _idx, skill in ref_index.get(api_ref, []):
            skill_id = str(skill.get("skill_id") or skill.get("id") or "")
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            ids.append(skill_id)
            names.append(str(skill.get("name") or skill_id))
        return ids, names

    for task in tasks:
        task_split = str(task.get("split") or "")
        if task_split != split:
            skipped_by_split += 1
            continue
        mapped_steps, skipped = _map_api_calls_to_skill_steps(
            appworld_root=Path(appworld_root),
            task=task,
            skills=skills,
            ref_index=ref_index,
            compress_consecutive=compress_consecutive,
        )
        for key, value in skipped.items():
            skipped_reasons[key] = skipped_reasons.get(key, 0) + int(value)
        if not mapped_steps:
            skipped_without_steps += 1
            continue

        row = dict(task)
        step_positive_skill_ids: list[list[str]] = []
        step_positive_skill_names: list[list[str]] = []
        for mapped in mapped_steps:
            ids, names = _labels_for_api_ref(mapped.api_ref)
            if not ids:
                ids = [mapped.skill_id]
                names = [mapped.skill_name]
            step_positive_skill_ids.append(ids)
            step_positive_skill_names.append(names)
            label_set_sizes.append(len(ids))

        row["step_positive_skill_ids"] = step_positive_skill_ids
        row["step_positive_skill_names"] = step_positive_skill_names
        row["step_positive_api_refs"] = [[mapped.api_ref] for mapped in mapped_steps]
        row["step_label_source"] = step_label_source
        row["step_label_execution_source"] = "ground_truth_api_calls_not_qwen_execution"
        row["step_label_split"] = split
        row["step_label_eval_only"] = True
        row["step_label_train_allowed"] = False
        row["leakage_guard"] = leakage_guard
        output_rows.append(row)

    written = write_jsonl(output_path, output_rows)
    manifest = {
        "status": "ok" if written else "empty",
        "appworld_root": str(appworld_root),
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "output_path": str(output_path),
        "allowed_split": split,
        "task_count": len(tasks),
        "skill_count": len(raw_skills),
        "label_skill_count": len(skills),
        "appworld_executor_compatible_only": bool(appworld_executor_compatible_only),
        "labeled_task_count": written,
        "skipped_by_split": skipped_by_split,
        "skipped_without_steps": skipped_without_steps,
        "skipped_reasons": skipped_reasons,
        "compress_consecutive": bool(compress_consecutive),
        "step_label_positive_policy": "all_appworld_executor_compatible_skills_sharing_oracle_api_ref",
        "avg_step_label_set_size": round(sum(label_set_sizes) / max(len(label_set_sizes), 1), 6),
        "max_step_label_set_size": max(label_set_sizes) if label_set_sizes else 0,
        "step_label_source": step_label_source,
        "execution_source": "ground_truth_api_calls_not_qwen_execution",
        "eval_only": True,
        "dev_test_used_for_training": False,
        "leakage_guard": leakage_guard,
        "caveat": (
            "These labels are for executor/routing evaluation only. Do not use dev/test "
            "ground-truth API traces for model training or prompt construction."
        ),
    }
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def _runs_by_task(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("task_id") or row.get("query_id")): row
        for row in read_jsonl(path)
    }


def _execution_failure_count(row: dict[str, Any]) -> int:
    return sum(
        1
        for step in row.get("steps", []) or []
        if bool(step.get("generation_ok", True)) and not bool(step.get("execution_ok", False))
    )


def _first_failure_output(row: dict[str, Any], max_chars: int = 500) -> str:
    for step in row.get("steps", []) or []:
        if bool(step.get("execution_ok", False)):
            continue
        text = " ".join(str(step.get("execute_output") or step.get("error") or "").split())
        if text:
            return text[: int(max_chars)]
    return ""


def build_multistep_failure_diagnostics(
    *,
    focus_runs_path: str | Path,
    reference_runs_path: str | Path,
    output_dir: str | Path,
    focus_name: str = "focus",
    reference_name: str = "reference",
) -> dict[str, Any]:
    focus = _runs_by_task(focus_runs_path)
    reference = _runs_by_task(reference_runs_path)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(focus) & set(reference)):
        focus_row = focus[task_id]
        reference_row = reference[task_id]
        if not bool(reference_row.get("success", False)) or bool(focus_row.get("success", False)):
            continue
        first_step = (focus_row.get("steps") or [{}])[0]
        rows.append(
            {
                "task_id": task_id,
                "focus_name": focus_name,
                "reference_name": reference_name,
                "focus_success": bool(focus_row.get("success", False)),
                "reference_success": bool(reference_row.get("success", False)),
                "focus_task_completed": bool(focus_row.get("task_completed", False)),
                "focus_evaluation_success": bool(focus_row.get("evaluation_success", False)),
                "focus_step_count": len(focus_row.get("steps", []) or []),
                "reference_step_count": len(reference_row.get("steps", []) or []),
                "focus_execution_failures": _execution_failure_count(focus_row),
                "reference_execution_failures": _execution_failure_count(reference_row),
                "first_focus_selected_skill_ids": [str(item) for item in first_step.get("selected_skill_ids", [])],
                "first_focus_failure_output": _first_failure_output(focus_row),
            }
        )
    payload = {
        "status": "ok",
        "focus_runs_path": str(focus_runs_path),
        "reference_runs_path": str(reference_runs_path),
        "focus_name": str(focus_name),
        "reference_name": str(reference_name),
        "focus_task_count": len(focus),
        "reference_task_count": len(reference),
        "reference_success_focus_failure_count": len(rows),
        "rows": rows,
        "caveat": "Diagnostics only; inspect traces before attributing failures to routing, context, or executor.",
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "failure_diagnostics.json", payload)
    lines = [
        "# AppWorld Multi-Step Failure Diagnostics",
        "",
        "| task_id | focus_exec_fail | focus_steps | reference_steps | first_focus_selected_skill_ids | first_failure |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {focus_execution_failures} | {focus_step_count} | {reference_step_count} | {skills} | {failure} |".format(
                task_id=row["task_id"],
                focus_execution_failures=row["focus_execution_failures"],
                focus_step_count=row["focus_step_count"],
                reference_step_count=row["reference_step_count"],
                skills=", ".join(row["first_focus_selected_skill_ids"][:5]),
                failure=str(row["first_focus_failure_output"]).replace("|", "/"),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
