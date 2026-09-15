from __future__ import annotations

import re
from typing import Any

import torch

from clstr.alfworld_action_skills import map_alfworld_action_to_skill_id
from clstr.alfworld_eval import _alfworld_router_state_text
from clstr.vnext_online_selector import VNEXT_ROUTE_MODES, VNextOnlineSelector


_ACTION_SKILL_PREFIX = "alfworld_action/"
_ABSTRACT_SKILL_PREFIX = "alfworld/"
_SKILL_GUIDANCE_HEADER = "[Trajectory-conditioned skill guidance]"


def alfworld_action_skill_id(action: str) -> str:
    action_text = str(action or "").strip()
    if not action_text:
        raise ValueError("ALFWorld action skill requires nonempty action text")
    return f"{_ACTION_SKILL_PREFIX}{action_text}"


def build_alfworld_action_skill(action: str) -> dict[str, Any]:
    action_text = str(action or "").strip()
    skill_id = alfworld_action_skill_id(action_text)
    description = f"Execute the admissible ALFWorld action exactly: {action_text}"
    return {
        "skill_id": skill_id,
        "canonical_skill_id": skill_id,
        "name": action_text,
        "description": description,
        "executor_desc": description,
        "body": f"Official admissible action text: {action_text}",
        "skill_md": f"Official admissible action text: {action_text}",
        "source": "alfworld",
        "source_benchmark": "alfworld",
        "domain": "alfworld",
        "train_allowed": False,
        "is_appended_after_checkpoint": True,
    }


def _selector_skill_ids(selector: Any) -> list[str]:
    skill_index = getattr(selector, "skill_id_to_idx", None)
    if isinstance(skill_index, dict):
        return [str(item) for item in skill_index]
    return [str(item) for item in getattr(selector, "skill_ids", [])]


def _abstract_skill_ids(selector: Any) -> list[str]:
    return [
        skill_id
        for skill_id in _selector_skill_ids(selector)
        if skill_id and not skill_id.startswith(_ACTION_SKILL_PREFIX)
    ]


def _humanize_abstract_skill_name(skill: dict[str, Any], skill_id: str) -> str:
    name = str(skill.get("name") or skill_id.rsplit("/", 1)[-1]).strip()
    if name.lower().startswith("alfworld-"):
        name = name[len("alfworld-") :]
    return " ".join(name.replace("_", "-").split("-")).strip()


def _first_semantic_sentence(text: str) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return ""
    parts = re.split(r"(?<=[A-Za-z)])\.\s+(?=[A-Z])", normalized, maxsplit=1)
    sentence = parts[0].rstrip(". ") + "."
    if len(sentence) <= 200:
        return sentence
    shortened = sentence[:197].rsplit(" ", 1)[0].rstrip(".,;: ")
    return shortened + "..."


def _format_abstract_skill_guidance(
    ranked: list[dict[str, Any]],
    candidate_actions: list[str],
) -> str:
    if not ranked:
        return ""
    normalized_actions = [
        " ".join(str(action).strip().lower().split())
        for action in candidate_actions
        if str(action).strip()
    ]
    lines = [
        _SKILL_GUIDANCE_HEADER,
        "Use this only as high-level planning guidance; do not execute it verbatim.",
        "Preserve objects, receptacles, locations, and other arguments from the task and state.",
        "Choose one exact command from AVAILABLE ACTIONS.",
    ]
    for rank, item in enumerate(ranked, start=1):
        skill_id = str(item.get("skill_id") or "").strip()
        skill = item.get("skill") if isinstance(item.get("skill"), dict) else {}
        name = _humanize_abstract_skill_name(skill, skill_id)
        summary = _first_semantic_sentence(
            str(skill.get("description") or skill.get("executor_desc") or "")
        )
        lowered_summary = " ".join(summary.lower().split())
        if not summary or any(action in lowered_summary for action in normalized_actions):
            summary = (
                f"Use the abstract skill '{name}' and infer its exact arguments "
                "from the task and current state."
            )
        lines.append(f"{rank}. {summary}")
    return "\n".join(lines)


class VNextAlfworldAdmissibleActionScorer:
    """Batched official-action adapter over independent vNext sessions."""

    def __init__(
        self,
        selector: VNextOnlineSelector,
        *,
        route_mode: str = "adaptive",
        allow_runtime_action_skills: bool = True,
        grounding_score_mode: str = "route_query",
        grounding_expert_mode: str = "selected",
        memory_update_skill_mode: str = "mapped_abstract",
    ) -> None:
        normalized_route_mode = str(route_mode or "adaptive").strip().lower()
        if normalized_route_mode not in VNEXT_ROUTE_MODES:
            raise ValueError(f"unsupported ALFWorld vNext route mode: {route_mode}")
        self.selector = selector
        self.route_mode = normalized_route_mode
        self.allow_runtime_action_skills = bool(allow_runtime_action_skills)
        self.memory_update_skill_mode = str(
            memory_update_skill_mode or "mapped_abstract"
        ).strip().lower()
        if self.memory_update_skill_mode not in {
            "mapped_abstract",
            "exact_action",
        }:
            raise ValueError(
                "unsupported ALFWorld memory-update skill mode: "
                f"{memory_update_skill_mode}"
            )
        self.grounding_score_mode = str(
            grounding_score_mode or "route_query"
        ).strip().lower()
        if self.grounding_score_mode not in {
            "route_query",
            "memory_skill_head",
        }:
            raise ValueError(
                "unsupported ALFWorld grounding score mode: "
                f"{grounding_score_mode}"
            )
        self.grounding_expert_mode = str(
            grounding_expert_mode or "selected"
        ).strip().lower()
        if self.grounding_expert_mode not in {"selected", "static"}:
            raise ValueError(
                "unsupported ALFWorld grounding expert mode: "
                f"{grounding_expert_mode}"
            )
        self.sessions: list[Any] = []
        self.last_state_texts: list[str] = []
        self.last_metadata: list[dict[str, Any]] = []
        self.last_guidance_metadata: list[dict[str, Any]] = []
        self.last_grounding_metadata: list[dict[str, Any]] = []
        self.last_append_report: dict[str, Any] = {}
        self.grounding_embedding_cache: dict[str, torch.Tensor] = {}
        self.causal_update_count = torch.empty(0, dtype=torch.long)

    def reset_episode_batch(self, state_texts: list[str]) -> None:
        if not state_texts:
            raise ValueError("ALFWorld vNext scorer requires a nonempty episode batch")
        self.sessions = [self.selector.new_session() for _ in state_texts]
        self.last_state_texts = [
            _alfworld_router_state_text(state_text) for state_text in state_texts
        ]
        self.last_metadata = []
        self.last_guidance_metadata = []
        self.last_grounding_metadata = []
        self.causal_update_count = torch.zeros(len(state_texts), dtype=torch.long)

    def _ensure_batch(self, state_texts: list[str], candidate_rows: list[list[str]]) -> None:
        if len(state_texts) != len(candidate_rows):
            raise ValueError("ALFWorld state and candidate batch sizes differ")
        if len(self.sessions) != len(state_texts):
            raise ValueError("ALFWorld scorer requires reset_episode_batch before scoring")

    def _grounding_action_embeddings(self, actions: list[str]) -> torch.Tensor:
        normalized_actions = [str(action) for action in actions]
        missing = list(
            dict.fromkeys(
                action
                for action in normalized_actions
                if action not in self.grounding_embedding_cache
            )
        )
        if missing:
            encode = getattr(self.selector, "encode_external_candidate_texts", None)
            if not callable(encode):
                raise TypeError(
                    "ALFWorld grounding requires external candidate encoding"
                )
            encoded = encode(missing)
            for action, embedding in zip(missing, encoded):
                self.grounding_embedding_cache[action] = (
                    embedding.detach().float().cpu()
                )
        return torch.stack(
            [self.grounding_embedding_cache[action] for action in normalized_actions]
        )

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
    ) -> torch.Tensor:
        self._ensure_batch(state_texts, candidate_rows)
        action_skills: list[dict[str, Any]] = []
        for candidates in candidate_rows:
            if not candidates:
                raise ValueError("official ALFWorld candidate row is empty")
            if len(candidates) != len(set(str(item) for item in candidates)):
                raise ValueError("official ALFWorld candidate row contains duplicates")
            action_skills.extend(
                build_alfworld_action_skill(action) for action in candidates
            )
        self.last_append_report = self.selector.ensure_skills(action_skills)

        width = max(len(row) for row in candidate_rows)
        floor = torch.finfo(torch.float32).min
        scores = torch.full((len(candidate_rows), width), floor, dtype=torch.float32)
        metadata: list[dict[str, Any]] = []
        stripped_states = [
            _alfworld_router_state_text(state_text) for state_text in state_texts
        ]
        for row_index, (session, state_text, candidates) in enumerate(
            zip(self.sessions, stripped_states, candidate_rows)
        ):
            candidate_skill_ids = [
                alfworld_action_skill_id(action) for action in candidates
            ]
            ranked = session.select(
                state_text,
                candidate_skill_ids=candidate_skill_ids,
                top_k=len(candidate_skill_ids),
                coarse_k=len(candidate_skill_ids),
                dynamic_extra_k=len(candidate_skill_ids),
                route_mode=self.route_mode,
            )
            score_by_id = {
                str(item["skill_id"]): float(item["score"]) for item in ranked
            }
            missing = [
                skill_id for skill_id in candidate_skill_ids if skill_id not in score_by_id
            ]
            if missing:
                raise RuntimeError(
                    "vNext natural support omitted legal ALFWorld actions: "
                    + ", ".join(missing[:5])
                )
            for column, skill_id in enumerate(candidate_skill_ids):
                scores[row_index, column] = score_by_id[skill_id]
            chosen = max(ranked, key=lambda item: float(item["score"]))
            metadata.append(
                {
                    "policy_family": "clstr_vnext_recurrent_concrete_action",
                    "route_scorer": "vnext_safe_candidate_route_scores",
                    "candidate_skill_schema": "alfworld_exact_admissible_action_v1",
                    "candidate_count": len(candidates),
                    "history_depth": int(session.history_depth),
                    "uses_recurrent_m_t": bool(session.history_depth > 0),
                    "clstr_memory_source": (
                        "factual_recurrent_m_t"
                        if session.history_depth > 0
                        else "release_initial_belief_m_0"
                    ),
                    "selector_probability": chosen.get("selector_probability"),
                    "mixture_probability": chosen.get("mixture_probability"),
                    "adaptive_mixture_probability": chosen.get(
                        "adaptive_mixture_probability"
                    ),
                    "route_mode": chosen.get("route_mode", self.route_mode),
                    "selected_expert": chosen.get("selected_expert"),
                    "selected_skill_id": chosen.get("skill_id"),
                    "memory_update_skill_mode": self.memory_update_skill_mode,
                    "runtime_appended_skill_count": int(
                        self.selector.checkpoint_binding.get(
                            "runtime_appended_skill_count", 0
                        )
                    ),
                    "history_removed_from_current_state": True,
                    "post_action_result_correction": bool(
                        session.history_depth > 0
                    ),
                }
            )
        self.last_state_texts = stripped_states
        self.last_metadata = metadata
        return scores

    def retrieve_skill_guidance(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
        *,
        top_k: int = 2,
    ) -> list[str]:
        """Retrieve abstract recurrent skill hints without scoring exact actions."""

        self._ensure_batch(state_texts, candidate_rows)
        requested_top_k = max(1, int(top_k))
        known_abstract_ids = _abstract_skill_ids(self.selector)
        known_abstract_set = set(known_abstract_ids)
        initialization_ids = [
            skill_id
            for skill_id in known_abstract_ids
            if skill_id.startswith(_ABSTRACT_SKILL_PREFIX)
        ] or known_abstract_ids[:1]
        stripped_states = [
            _alfworld_router_state_text(state_text) for state_text in state_texts
        ]
        guidance_rows: list[str] = []
        metadata: list[dict[str, Any]] = []
        for session, state_text, raw_candidates in zip(
            self.sessions, stripped_states, candidate_rows
        ):
            candidates = [str(item) for item in raw_candidates]
            if not candidates:
                raise ValueError("official ALFWorld candidate row is empty")
            if len(candidates) != len(set(candidates)):
                raise ValueError("official ALFWorld candidate row contains duplicates")
            mappings = [
                map_alfworld_action_to_skill_id(action, known_abstract_set)
                for action in candidates
            ]
            mapped_skill_ids = list(
                dict.fromkeys(
                    mapping.skill_id
                    for mapping in mappings
                    if mapping.skill_id is not None
                )
            )
            selection_ids = mapped_skill_ids
            initialization_only = False
            if not selection_ids:
                initialization_only = True
                selection_ids = list(initialization_ids)
                if not selection_ids:
                    fallback_rows = [build_alfworld_action_skill(action) for action in candidates]
                    self.last_append_report = self.selector.ensure_skills(fallback_rows)
                    selection_ids = [
                        alfworld_action_skill_id(action) for action in candidates
                    ]
            ranked = session.select(
                state_text,
                candidate_skill_ids=selection_ids,
                top_k=(
                    1
                    if initialization_only
                    else min(requested_top_k, len(selection_ids))
                ),
                coarse_k=len(selection_ids),
                dynamic_extra_k=len(selection_ids),
                route_mode=self.route_mode,
            )
            selected = [] if initialization_only else ranked
            guidance = _format_abstract_skill_guidance(selected, candidates)
            guidance_rows.append(guidance)
            route_item = ranked[0] if ranked else {}
            metadata.append(
                {
                    "policy_family": "clstr_vnext_recurrent_abstract_skill_guidance",
                    "route_scorer": "vnext_safe_candidate_route_scores",
                    "candidate_skill_schema": "alfworld_mapped_abstract_skill_v1",
                    "candidate_count": len(candidates),
                    "mapped_abstract_skill_count": len(mapped_skill_ids),
                    "mapped_abstract_skill_ids": mapped_skill_ids,
                    "selected_abstract_skill_ids": [
                        str(item.get("skill_id") or "") for item in selected
                    ],
                    "selected_abstract_skills": [
                        dict(item.get("skill") or {}) for item in selected
                    ],
                    "skill_guidance": guidance,
                    "skill_guidance_empty": not bool(guidance),
                    "initialization_only": initialization_only,
                    "history_depth": int(session.history_depth),
                    "uses_recurrent_m_t": bool(session.history_depth > 0),
                    "clstr_memory_source": (
                        "factual_recurrent_m_t"
                        if session.history_depth > 0
                        else "release_initial_belief_m_0"
                    ),
                    "selector_probability": route_item.get("selector_probability"),
                    "mixture_probability": route_item.get("mixture_probability"),
                    "adaptive_mixture_probability": route_item.get(
                        "adaptive_mixture_probability"
                    ),
                    "route_mode": route_item.get("route_mode", self.route_mode),
                    "selected_expert": route_item.get("selected_expert"),
                    "history_removed_from_current_state": True,
                    "post_action_result_correction": bool(session.history_depth > 0),
                    "runtime_appended_skill_count": int(
                        self.selector.checkpoint_binding.get(
                            "runtime_appended_skill_count", 0
                        )
                    ),
                }
            )
        self.last_state_texts = stripped_states
        self.last_guidance_metadata = metadata
        return guidance_rows

    def retrieve_grounded_action_prior(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
        *,
        guidance_top_k: int = 2,
    ) -> tuple[list[str], torch.Tensor]:
        """Choose an expert in abstract space, then score encoded legal actions.

        The current candidate distribution can affect the concrete score row,
        but it cannot affect the static/dynamic decision: that choice is made
        first on unique checkpoint-known abstract skills.  Exact legal actions
        are temporary encoded candidates and never enter the skill table.
        """

        self._ensure_batch(state_texts, candidate_rows)
        requested_top_k = max(1, int(guidance_top_k))
        known_abstract_ids = [
            skill_id
            for skill_id in _abstract_skill_ids(self.selector)
            if skill_id.startswith(_ABSTRACT_SKILL_PREFIX)
        ]
        known_abstract_set = set(known_abstract_ids)
        if not known_abstract_ids:
            raise ValueError("ALFWorld abstract grounding requires trained abstract skills")
        stripped_states = [
            _alfworld_router_state_text(state_text) for state_text in state_texts
        ]
        width = max((len(row) for row in candidate_rows), default=0)
        floor = torch.finfo(torch.float32).min
        action_priors = torch.full(
            (len(candidate_rows), width),
            floor,
            dtype=torch.float32,
        )
        guidance_rows: list[str] = []
        metadata: list[dict[str, Any]] = []
        for row_index, (session, state_text, raw_candidates) in enumerate(
            zip(self.sessions, stripped_states, candidate_rows)
        ):
            candidates = [str(item) for item in raw_candidates]
            if not candidates:
                raise ValueError("official ALFWorld candidate row is empty")
            if len(candidates) != len(set(candidates)):
                raise ValueError("official ALFWorld candidate row contains duplicates")
            mappings = [
                map_alfworld_action_to_skill_id(action, known_abstract_set)
                for action in candidates
            ]
            mapped_skill_ids = list(
                dict.fromkeys(
                    mapping.skill_id
                    for mapping in mappings
                    if mapping.skill_id is not None
                )
            )
            # The reliability decision lives on one fixed checkpoint-known
            # ALFWorld inventory. Legal-action availability must not change the
            # gate features or the definition of m_0 from one environment step
            # to the next.
            initialization_only = False
            selection_ids = known_abstract_ids
            ranked = session.select(
                state_text,
                candidate_skill_ids=selection_ids,
                top_k=len(selection_ids),
                coarse_k=len(selection_ids),
                dynamic_extra_k=len(selection_ids),
                route_mode=self.route_mode,
            )
            selected = ranked[:requested_top_k]
            guidance = _format_abstract_skill_guidance(selected, candidates)
            guidance_rows.append(guidance)
            route_item = ranked[0] if ranked else {}
            abstract_selected_expert = str(
                route_item.get("selected_expert") or "static"
            )
            grounding_requested_expert = (
                "static"
                if self.grounding_expert_mode == "static"
                else abstract_selected_expert
            )
            grounding = session.score_external_candidate_embeddings(
                state_text,
                self._grounding_action_embeddings(candidates),
                selected_expert=grounding_requested_expert,
                route_mode=self.route_mode,
                scoring_head=self.grounding_score_mode,
            )
            raw_action_scores = grounding["scores"]
            if not isinstance(raw_action_scores, torch.Tensor):
                raise TypeError("external action grounder must return tensor scores")
            if int(raw_action_scores.numel()) != len(candidates):
                raise ValueError("external action score width differs from legal actions")
            static_action_scores = grounding.get("static_scores")
            dynamic_action_scores = grounding.get("dynamic_scores")
            route_residual = grounding.get("route_residual")
            for score_name, score_values in (
                ("static_scores", static_action_scores),
                ("dynamic_scores", dynamic_action_scores),
                ("route_residual", route_residual),
            ):
                if not isinstance(score_values, torch.Tensor) or int(
                    score_values.numel()
                ) != len(candidates):
                    raise ValueError(
                        f"external action grounder {score_name} width differs "
                        "from legal actions"
                    )
            static_top_index = int(torch.argmax(static_action_scores).item())
            dynamic_top_index = int(torch.argmax(dynamic_action_scores).item())
            if raw_action_scores.numel() <= 1 or float(
                raw_action_scores.std(unbiased=False).item()
            ) < 1.0e-6:
                normalized_action_scores = torch.zeros_like(raw_action_scores)
            else:
                normalized_action_scores = (
                    raw_action_scores - raw_action_scores.mean()
                ) / raw_action_scores.std(unbiased=False).clamp_min(1.0e-6)
            action_priors[row_index, : len(candidates)] = normalized_action_scores
            metadata.append(
                {
                    "policy_family": "clstr_vnext_abstract_grounded_action_prior",
                    "route_scorer": (
                        "abstract_gate_then_external_candidate_memory_scores"
                        if self.grounding_score_mode == "memory_skill_head"
                        else "abstract_gate_then_external_candidate_route_scores"
                    ),
                    "candidate_skill_schema": "alfworld_mapped_abstract_skill_v1",
                    "grounding_candidate_schema": "alfworld_exact_legal_action_v1",
                    "concrete_action_text_scoring": True,
                    "candidate_count": len(candidates),
                    "mapped_action_count": sum(
                        mapping.skill_id is not None for mapping in mappings
                    ),
                    "unmapped_action_count": sum(
                        mapping.skill_id is None for mapping in mappings
                    ),
                    "mapped_abstract_skill_count": len(mapped_skill_ids),
                    "mapped_abstract_skill_ids": mapped_skill_ids,
                    "selected_abstract_skill_ids": [
                        str(item.get("skill_id") or "") for item in selected
                    ],
                    "selected_abstract_skills": [
                        dict(item.get("skill") or {}) for item in selected
                    ],
                    "skill_guidance": guidance,
                    "skill_guidance_empty": not bool(guidance),
                    "initialization_only": initialization_only,
                    "abstract_gate_candidate_count": len(selection_ids),
                    "grounding_score_source": (
                        "memory_skill_head_over_encoded_legal_action"
                        if self.grounding_score_mode == "memory_skill_head"
                        else "stage2_route_query_dot_encoded_legal_action"
                    ),
                    "grounding_score_mode": grounding.get("scoring_head"),
                    "grounding_expert_mode": self.grounding_expert_mode,
                    "abstract_selected_expert": abstract_selected_expert,
                    "grounding_score_normalization": (
                        f"{self.grounding_score_mode}_legal_action_row_zscore"
                    ),
                    "grounding_selected_expert": grounding["selected_expert"],
                    "grounding_static_top_action": candidates[static_top_index],
                    "grounding_dynamic_top_action": candidates[dynamic_top_index],
                    "grounding_memory_top1_changed": bool(
                        static_top_index != dynamic_top_index
                    ),
                    "grounding_route_residual_l2": float(
                        torch.linalg.vector_norm(route_residual.float()).item()
                    ),
                    "grounding_route_residual_max_abs": float(
                        route_residual.float().abs().max().item()
                    ),
                    "history_depth": int(session.history_depth),
                    "uses_recurrent_m_t": bool(session.history_depth > 0),
                    "clstr_memory_source": (
                        "factual_recurrent_m_t"
                        if session.history_depth > 0
                        else "release_initial_belief_m_0"
                    ),
                    "selector_probability": route_item.get("selector_probability"),
                    "mixture_probability": route_item.get("mixture_probability"),
                    "adaptive_mixture_probability": route_item.get(
                        "adaptive_mixture_probability"
                    ),
                    "route_mode": route_item.get("route_mode", self.route_mode),
                    "selected_expert": route_item.get("selected_expert"),
                    "history_removed_from_current_state": True,
                    "post_action_result_correction": bool(session.history_depth > 0),
                    "runtime_appended_skill_count": int(
                        self.selector.checkpoint_binding.get(
                            "runtime_appended_skill_count", 0
                        )
                    ),
                    "runtime_pseudo_skills_allowed": False,
                }
            )
        self.last_state_texts = stripped_states
        self.last_grounding_metadata = metadata
        return guidance_rows, action_priors

    def observe_transitions(
        self,
        *,
        chosen_actions: list[str],
        next_observation_texts: list[str],
        next_state_texts: list[str],
        active_mask: list[bool],
    ) -> None:
        del next_state_texts
        batch_size = len(self.sessions)
        if not all(
            len(values) == batch_size
            for values in (chosen_actions, next_observation_texts, active_mask)
        ):
            raise ValueError("ALFWorld transition batch size drift")
        if len(self.last_state_texts) != batch_size:
            raise ValueError("ALFWorld transition requires a preceding score call")
        for index, active in enumerate(active_mask):
            if not bool(active):
                continue
            action = str(chosen_actions[index])
            known_abstract_ids = set(_abstract_skill_ids(self.selector))
            if self.memory_update_skill_mode == "exact_action":
                if not self.allow_runtime_action_skills:
                    raise RuntimeError(
                        "exact-action memory updates require runtime action skills"
                    )
                update_skill_id = alfworld_action_skill_id(action)
                if update_skill_id not in set(_selector_skill_ids(self.selector)):
                    self.last_append_report = self.selector.ensure_skills(
                        [build_alfworld_action_skill(action)]
                    )
            else:
                mapped_skill_id = map_alfworld_action_to_skill_id(
                    action,
                    known_abstract_ids,
                ).skill_id
                update_skill_id = mapped_skill_id
                if update_skill_id is None:
                    if self.allow_runtime_action_skills:
                        update_skill_id = alfworld_action_skill_id(action)
                        if update_skill_id not in set(
                            _selector_skill_ids(self.selector)
                        ):
                            self.last_append_report = self.selector.ensure_skills(
                                [build_alfworld_action_skill(action)]
                            )
                    else:
                        fallback_ids = [
                            "alfworld/alfworld-device-operator",
                            "alfworld/alfworld-environment-scanner",
                        ]
                        update_skill_id = next(
                            (
                                skill_id
                                for skill_id in fallback_ids
                                if skill_id in known_abstract_ids
                            ),
                            sorted(known_abstract_ids)[0],
                        )
            self.sessions[index].observe(
                state_text_before=self.last_state_texts[index],
                skill_id=update_skill_id,
                action_text=action,
                result_text=str(next_observation_texts[index]),
            )
        self.causal_update_count = torch.tensor(
            [session.history_depth for session in self.sessions],
            dtype=torch.long,
        )
