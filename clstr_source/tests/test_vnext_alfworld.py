from __future__ import annotations

from contextlib import nullcontext
import threading

import torch

from clstr.alfworld_qwen_clstr_gate import (
    QwenClstrAbstractGroundedActionScorer,
    QwenClstrHybridActionScorer,
    QwenClstrSkillPromptActionScorer,
)
from clstr.vnext_alfworld import (
    VNextAlfworldAdmissibleActionScorer,
    alfworld_action_skill_id,
)
from clstr.vnext_online_selector import VNextOnlineSession


class _FakeSession:
    def __init__(self) -> None:
        self.history_depth = 0
        self.states: list[str] = []
        self.observations: list[dict] = []
        self.select_kwargs: list[dict] = []
        self.candidate_skill_rows: list[list[str]] = []
        self.external_scoring_heads: list[str] = []

    def select(self, state_text, *, candidate_skill_ids, top_k, **kwargs):
        del top_k
        self.select_kwargs.append(dict(kwargs))
        self.candidate_skill_rows.append(list(candidate_skill_ids))
        self.states.append(state_text)
        return [
            {
                "skill_id": skill_id,
                "score": float(index),
                "selector_probability": 0.75,
                "mixture_probability": 0.5,
                "adaptive_mixture_probability": 0.5,
                "route_mode": kwargs.get("route_mode", "adaptive"),
                "selected_expert": "dynamic" if self.history_depth else "static",
                "skill": {},
            }
            for index, skill_id in enumerate(candidate_skill_ids)
        ]

    def observe(self, **kwargs):
        self.observations.append(kwargs)
        self.history_depth += 1
        return {"history_depth": self.history_depth}

    def score_external_candidate_embeddings(
        self,
        state_text,
        candidate_embeddings,
        *,
        selected_expert,
        route_mode,
        scoring_head="route_query",
    ):
        del state_text, route_mode
        self.external_scoring_heads.append(str(scoring_head))
        width = int(candidate_embeddings.size(0))
        static = torch.arange(width, dtype=torch.float32)
        dynamic = torch.flip(static, dims=(0,))
        use_dynamic = self.history_depth > 0 and selected_expert == "dynamic"
        return {
            "scores": dynamic if use_dynamic else static,
            "static_scores": static,
            "dynamic_scores": dynamic,
            "route_residual": dynamic - static,
            "selected_expert": "dynamic" if use_dynamic else "static",
            "scoring_head": str(scoring_head),
        }


class _FakeSelector:
    method = "fake_release"

    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []
        self.appended: list[dict] = []
        self.checkpoint_binding = {"runtime_appended_skill_count": 0}
        self.skill_ids = [
            "alfworld/alfworld-location-navigator",
            "alfworld/alfworld-object-picker",
            "alfworld/alfworld-object-state-inspector",
        ]
        self.skill_id_to_idx = {
            skill_id: index for index, skill_id in enumerate(self.skill_ids)
        }
        self.skill_by_id = {
            "alfworld/alfworld-location-navigator": {
                "skill_id": "alfworld/alfworld-location-navigator",
                "name": "alfworld-location-navigator",
                "description": "Move to a task-relevant location before interacting with it.",
            },
            "alfworld/alfworld-object-picker": {
                "skill_id": "alfworld/alfworld-object-picker",
                "name": "alfworld-object-picker",
                "description": "Pick up the task-relevant object from its current receptacle.",
            },
            "alfworld/alfworld-object-state-inspector": {
                "skill_id": "alfworld/alfworld-object-state-inspector",
                "name": "alfworld-object-state-inspector",
                "description": "Inspect the relevant receptacle and use the resulting observation.",
            },
        }

    def new_session(self):
        session = _FakeSession()
        self.sessions.append(session)
        return session

    def encode_external_candidate_texts(self, texts):
        return torch.tensor(
            [[float(index), 1.0] for index, _text in enumerate(texts)],
            dtype=torch.float32,
        )

    def ensure_skills(self, rows):
        by_id = {row["skill_id"]: dict(row) for row in rows}
        self.appended.extend(by_id.values())
        self.checkpoint_binding["runtime_appended_skill_count"] = len(
            {row["skill_id"] for row in self.appended}
        )
        for row in by_id.values():
            if row["skill_id"] not in self.skill_id_to_idx:
                self.skill_id_to_idx[row["skill_id"]] = len(self.skill_ids)
                self.skill_ids.append(row["skill_id"])
                self.skill_by_id[row["skill_id"]] = row
        return {"appended_count": len(by_id)}


class _DirectMemoryHeadModel:
    def encode_states(self, states):
        assert len(states) == 1
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    def vnext_initial_belief(self, current_state, legal, top_k=None):
        del legal, top_k
        return current_state

    def skill_head(self, candidates, memory):
        return torch.einsum("bcd,bd->bc", candidates, memory)


class _DirectMemoryHeadSelector:
    def __init__(self):
        self.device = torch.device("cpu")
        self.skills = [{"skill_id": "skill/a"}]
        self.skill_id_to_idx = {"skill/a": 0}
        self.belief_top_k = None
        self.model = _DirectMemoryHeadModel()
        self._model_lock = threading.RLock()

    @staticmethod
    def _autocast():
        return nullcontext()


def test_external_memory_skill_head_scores_static_m0_and_recurrent_mt_directly():
    session = VNextOnlineSession(_DirectMemoryHeadSelector())
    session.memory = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    session.events = [{"action_text": "previous action"}]
    session.last_candidate_skill_ids = ["skill/a"]
    candidates = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )

    result = session.score_external_candidate_embeddings(
        "goal: test\nobservation: current",
        candidates,
        selected_expert="dynamic",
        route_mode="adaptive",
        scoring_head="memory_skill_head",
    )

    assert result["scoring_head"] == "memory_skill_head"
    assert result["selected_expert"] == "dynamic"
    assert torch.equal(result["static_scores"], torch.tensor([0.0, 1.0]))
    assert torch.equal(result["dynamic_scores"], torch.tensor([1.0, 0.0]))
    assert torch.equal(result["scores"], torch.tensor([1.0, 0.0]))


def test_vnext_alfworld_scores_exact_actions_and_updates_from_real_result():
    selector = _FakeSelector()
    scorer = VNextAlfworldAdmissibleActionScorer(selector)
    state = "goal: cool apple\nobservation: kitchen\nhistory: look | go north"
    scorer.reset_episode_batch([state])
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scores = scorer([state], candidates)

    assert scores.tolist() == [[0.0, 1.0]]
    assert selector.sessions[0].states == [
        "goal: cool apple\nobservation: kitchen"
    ]
    assert {row["skill_id"] for row in selector.appended} == {
        alfworld_action_skill_id(action) for action in candidates[0]
    }
    assert selector.sessions[0].select_kwargs[0]["coarse_k"] == 2
    assert scorer.last_metadata[0]["candidate_skill_schema"] == (
        "alfworld_exact_admissible_action_v1"
    )
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is False
    assert selector.sessions[0].select_kwargs[0]["route_mode"] == "adaptive"

    scorer.observe_transitions(
        chosen_actions=[candidates[0][1]],
        next_observation_texts=["You arrive at the fridge."],
        next_state_texts=["unused"],
        active_mask=[True],
    )
    assert scorer.causal_update_count.tolist() == [1]
    assert selector.sessions[0].observations[0]["result_text"] == (
        "You arrive at the fridge."
    )
    assert selector.sessions[0].observations[0]["skill_id"] == (
        "alfworld/alfworld-location-navigator"
    )

    scorer(["goal: cool apple\nobservation: fridge\nhistory: go to fridge 1"], candidates)
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is True


def test_vnext_alfworld_can_force_static_route_mode():
    selector = _FakeSelector()
    scorer = VNextAlfworldAdmissibleActionScorer(selector, route_mode="static")
    scorer.reset_episode_batch(["goal: cool apple\nobservation: kitchen"])

    scorer(
        ["goal: cool apple\nobservation: kitchen"],
        [["look", "go to fridge 1"]],
    )

    assert selector.sessions[0].select_kwargs[0]["route_mode"] == "static"
    assert scorer.last_metadata[0]["route_mode"] == "static"


def test_vnext_alfworld_retrieves_abstract_guidance_without_appending_actions():
    selector = _FakeSelector()
    scorer = VNextAlfworldAdmissibleActionScorer(selector, route_mode="dynamic")
    state = "goal: cool apple\nobservation: kitchen\nhistory: look"
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scorer.reset_episode_batch([state])

    guidance = scorer.retrieve_skill_guidance([state], candidates, top_k=2)

    assert "[Trajectory-conditioned skill guidance]" in guidance[0]
    assert "take apple 1 from table 1" not in guidance[0].lower()
    assert "go to fridge 1" not in guidance[0].lower()
    assert selector.appended == []
    assert selector.sessions[0].states == [
        "goal: cool apple\nobservation: kitchen"
    ]
    assert selector.sessions[0].select_kwargs[0]["route_mode"] == "dynamic"
    assert scorer.last_guidance_metadata[0]["mapped_abstract_skill_count"] == 2
    assert scorer.last_guidance_metadata[0]["candidate_skill_schema"] == (
        "alfworld_mapped_abstract_skill_v1"
    )


def test_vnext_alfworld_empty_guidance_initializes_memory_and_observes_unmapped_action():
    selector = _FakeSelector()
    scorer = VNextAlfworldAdmissibleActionScorer(selector)
    state = "goal: slice apple\nobservation: kitchen"
    action = "slice apple 1 with knife 1"
    scorer.reset_episode_batch([state])

    guidance = scorer.retrieve_skill_guidance([state], [[action]], top_k=2)

    assert guidance == [""]
    assert scorer.last_guidance_metadata[0]["initialization_only"] is True
    assert selector.sessions[0].states == [state]
    scorer.observe_transitions(
        chosen_actions=[action],
        next_observation_texts=["The apple is sliced."],
        next_state_texts=["unused"],
        active_mask=[True],
    )
    assert selector.sessions[0].observations[0]["skill_id"] == (
        alfworld_action_skill_id(action)
    )
    assert selector.sessions[0].observations[0]["action_text"] == action
    assert selector.sessions[0].observations[0]["result_text"] == (
        "The apple is sliced."
    )


class _RecordingQwenScorer:
    def __init__(self) -> None:
        self.calls = []
        self.last_metadata = []

    def __call__(self, state_texts, candidate_rows):
        self.calls.append((list(state_texts), [list(row) for row in candidate_rows]))
        self.last_metadata = [
            {
                "policy_family": "qwen_direct_admissible",
                "parsed_action": row[1],
                "parse_status": "exact_match",
                "fallback_used": False,
            }
            for row in candidate_rows
        ]
        return torch.tensor([[0.0, 1.0] for _ in candidate_rows])


def test_skill_prompt_executor_preserves_state_and_exact_action_surface():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(selector, route_mode="static")
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrSkillPromptActionScorer(
        qwen_scorer=qwen,
        clstr_retriever=retriever,
        guidance_top_k=1,
    )
    state = "goal: cool apple\nobservation: kitchen\nhistory: look"
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scorer.reset_episode_batch([state])

    scores = scorer([state], candidates)

    qwen_states, qwen_candidates = qwen.calls[0]
    assert qwen_states[0].startswith(state + "\n\n")
    assert "[Trajectory-conditioned skill guidance]" in qwen_states[0]
    assert qwen_candidates == candidates
    assert scores.tolist() == [[0.0, 1.0]]
    assert scorer.last_metadata[0]["executor_interface"] == "skill_prompt"
    assert scorer.last_metadata[0]["uses_clstr_skill_guidance"] is True
    assert scorer.last_metadata[0]["uses_clstr_prior"] is False
    assert scorer.last_metadata[0]["score_fusion"] is False
    assert scorer.last_metadata[0]["qwen_chosen_action"] == "go to fridge 1"
    scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["You arrive at the fridge."],
        next_state_texts=["next state"],
        active_mask=[True],
    )
    assert selector.sessions[0].observations[0] == {
        "state_text_before": "goal: cool apple\nobservation: kitchen",
        "skill_id": "alfworld/alfworld-location-navigator",
        "action_text": "go to fridge 1",
        "result_text": "You arrive at the fridge.",
    }
    assert scorer.causal_update_count.tolist() == [1]


def test_skill_prompt_executor_empty_guidance_is_exact_qwen_fallback():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(selector)
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrSkillPromptActionScorer(qwen, retriever)
    state = "goal: slice apple\nobservation: kitchen"
    candidates = [["slice apple 1 with knife 1", "wait"]]
    scorer.reset_episode_batch([state])

    scorer([state], candidates)

    assert qwen.calls[0][0] == [state]
    assert qwen.calls[0][1] == candidates
    assert scorer.last_metadata[0]["uses_clstr_skill_guidance"] is False


def test_abstract_grounded_executor_scores_exact_actions_without_pseudo_rows():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode="adaptive",
        allow_runtime_action_skills=False,
    )
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrAbstractGroundedActionScorer(
        qwen_scorer=qwen,
        clstr_retriever=retriever,
        guidance_top_k=2,
    )
    state = "goal: cool apple\nobservation: kitchen\nhistory: look"
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scorer.reset_episode_batch([state])

    scores = scorer([state], candidates)

    assert scores.shape == (1, 2)
    assert selector.appended == []
    assert selector.checkpoint_binding["runtime_appended_skill_count"] == 0
    assert qwen.calls[0][1] == candidates
    assert "[Trajectory-conditioned skill guidance]" in qwen.calls[0][0][0]
    metadata = scorer.last_metadata[0]
    assert metadata["executor_interface"] == "abstract_grounded"
    assert metadata["candidate_skill_schema"] == (
        "alfworld_mapped_abstract_skill_v1"
    )
    assert metadata["grounding_candidate_schema"] == (
        "alfworld_exact_legal_action_v1"
    )
    assert metadata["abstract_gate_candidate_count"] == 3
    assert selector.sessions[0].candidate_skill_rows[0] == selector.skill_ids
    assert metadata["runtime_pseudo_skills_allowed"] is False
    assert metadata["runtime_appended_skill_count"] == 0
    assert metadata["clstr_selected_expert"] == "static"
    assert torch.allclose(
        scorer.last_clstr_scores,
        torch.tensor([[-1.0, 1.0]]),
    )

    scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["You arrive at the fridge."],
        next_state_texts=["next state"],
        active_mask=[True],
    )
    scorer(["goal: cool apple\nobservation: fridge"], candidates)

    assert selector.appended == []
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is True
    assert scorer.last_metadata[0]["clstr_selected_expert"] == "dynamic"
    assert torch.allclose(
        scorer.last_clstr_scores,
        torch.tensor([[1.0, -1.0]]),
    )


def test_abstract_grounded_generate_mode_uses_proposal_bonus_without_prompt_injection():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode="adaptive",
        allow_runtime_action_skills=False,
        grounding_score_mode="memory_skill_head",
    )
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrAbstractGroundedActionScorer(
        qwen_scorer=qwen,
        clstr_retriever=retriever,
        guidance_top_k=2,
        qwen_weight=1.0,
        clstr_weight=1.0,
        qwen_score_mode="proposal_bonus",
        inject_skill_guidance=False,
    )
    state = "goal: cool apple\nobservation: kitchen"
    candidates = [[
        "take apple 1 from table 1",
        "go to fridge 1",
        "open fridge 1",
    ]]
    scorer.reset_episode_batch([state])

    scores = scorer([state], candidates)

    assert qwen.calls[0][0] == [state]
    assert torch.argmax(scores[0]).item() == 2
    metadata = scorer.last_metadata[0]
    assert metadata["qwen_score_mode"] == "proposal_bonus"
    assert metadata["qwen_score_normalization"] == "proposal_bonus"
    assert metadata["uses_clstr_skill_guidance"] is False
    assert metadata["skill_guidance_injected"] is False
    assert metadata["skill_guidance"] == ""
    assert "[Trajectory-conditioned skill guidance]" in metadata[
        "abstract_route_guidance"
    ]
    assert metadata["runtime_appended_skill_count"] == 0
    assert metadata["grounding_score_mode"] == "memory_skill_head"
    assert metadata["clstr_score_normalization"] == (
        "memory_skill_head_legal_action_row_zscore"
    )
    assert metadata["grounding_static_top_action"] == "open fridge 1"
    assert metadata["grounding_dynamic_top_action"] == "take apple 1 from table 1"
    assert metadata["grounding_memory_top1_changed"] is True
    assert metadata["grounding_route_residual_l2"] > 0.0
    assert selector.sessions[0].external_scoring_heads == ["memory_skill_head"]


def test_guided_exact_prior_shares_one_adaptive_expert_for_prompt_and_fusion():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode="adaptive",
        allow_runtime_action_skills=False,
        grounding_score_mode="memory_skill_head",
    )
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrAbstractGroundedActionScorer(
        qwen_scorer=qwen,
        clstr_retriever=retriever,
        guidance_top_k=2,
        qwen_weight=1.0,
        clstr_weight=0.25,
        qwen_score_mode="proposal_bonus",
        inject_skill_guidance=True,
        executor_interface="guided_exact_prior",
        policy_family="qwen_clstr_guided_exact_prior_executor",
    )
    state = "goal: cool apple\nobservation: kitchen"
    candidates = [[
        "take apple 1 from table 1",
        "go to fridge 1",
        "open fridge 1",
    ]]
    scorer.reset_episode_batch([state])

    scores = scorer([state], candidates)

    assert scores.shape == (1, 3)
    assert len(selector.sessions[0].candidate_skill_rows) == 1
    assert selector.sessions[0].candidate_skill_rows[0] == selector.skill_ids
    assert selector.appended == []
    assert "[Trajectory-conditioned skill guidance]" in qwen.calls[0][0][0]
    metadata = scorer.last_metadata[0]
    assert metadata["executor_interface"] == "guided_exact_prior"
    assert metadata["policy_family"] == (
        "qwen_clstr_guided_exact_prior_executor"
    )
    assert metadata["uses_clstr_skill_guidance"] is True
    assert metadata["skill_guidance_injected"] is True
    assert metadata["uses_clstr_prior"] is True
    assert metadata["score_fusion"] is True
    assert metadata["runtime_pseudo_skills_allowed"] is False
    assert metadata["runtime_appended_skill_count"] == 0
    assert metadata["clstr_selected_expert"] == "static"
    assert metadata["grounding_selected_expert"] == "static"


def test_guided_static_exact_prior_keeps_memory_in_abstract_prompt_only():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode="adaptive",
        allow_runtime_action_skills=False,
        grounding_score_mode="memory_skill_head",
        grounding_expert_mode="static",
    )
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrAbstractGroundedActionScorer(
        qwen_scorer=qwen,
        clstr_retriever=retriever,
        guidance_top_k=2,
        qwen_score_mode="proposal_bonus",
        inject_skill_guidance=True,
        executor_interface="guided_static_exact_prior",
        policy_family="qwen_clstr_guided_static_exact_prior_executor",
    )
    first_state = "goal: cool apple\nobservation: kitchen"
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scorer.reset_episode_batch([first_state])
    scorer([first_state], candidates)
    scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["You arrive at the fridge."],
        next_state_texts=["unused"],
        active_mask=[True],
    )

    scorer(["goal: cool apple\nobservation: fridge"], candidates)

    metadata = scorer.last_metadata[0]
    assert metadata["executor_interface"] == "guided_static_exact_prior"
    assert metadata["clstr_selected_expert"] == "dynamic"
    assert metadata["abstract_selected_expert"] == "dynamic"
    assert metadata["grounding_expert_mode"] == "static"
    assert metadata["grounding_selected_expert"] == "static"
    assert metadata["uses_recurrent_m_t"] is True
    assert metadata["uses_clstr_skill_guidance"] is True
    assert metadata["score_fusion"] is True
    assert metadata["runtime_appended_skill_count"] == 0
    assert selector.appended == []


class _ConstantScorer:
    def __init__(self, scores):
        self.scores = scores
        self.last_metadata = [{}]

    def __call__(self, state_texts, candidate_rows):
        del state_texts, candidate_rows
        return torch.tensor([self.scores], dtype=torch.float32)


class _LifecycleScorer(_ConstantScorer):
    def __init__(self):
        super().__init__([0.0, 1.0])
        self.reset_rows = None
        self.observed = None
        self.causal_update_count = torch.tensor([3])

    def reset_episode_batch(self, state_texts):
        self.reset_rows = list(state_texts)

    def observe_transitions(self, **kwargs):
        self.observed = kwargs


def test_qwen_clstr_hybrid_forwards_recurrent_lifecycle():
    prior = _LifecycleScorer()
    hybrid = QwenClstrHybridActionScorer(
        qwen_scorer=_ConstantScorer([1.0, 0.0]),
        clstr_scorer=prior,
        qwen_score_mode="proposal_bonus",
    )
    hybrid.reset_episode_batch(["state"])
    hybrid.observe_transitions(
        chosen_actions=["go north"],
        next_observation_texts=["hallway"],
        next_state_texts=["next"],
        active_mask=[True],
    )

    assert prior.reset_rows == ["state"]
    assert prior.observed["next_observation_texts"] == ["hallway"]
    assert hybrid.causal_update_count.tolist() == [3]


def test_legacy_guided_exact_prior_preserves_prior_and_conditions_qwen():
    selector = _FakeSelector()
    retriever = VNextAlfworldAdmissibleActionScorer(
        selector,
        route_mode="adaptive",
        allow_runtime_action_skills=True,
        memory_update_skill_mode="exact_action",
    )
    qwen = _RecordingQwenScorer()
    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=qwen,
        clstr_scorer=retriever,
        qwen_weight=1.0,
        clstr_weight=0.25,
        normalize_scores=True,
        qwen_score_mode="proposal_bonus",
        executor_interface="legacy_guided_exact_prior",
        inject_skill_guidance=True,
        guidance_top_k=2,
    )
    state = "goal: cool apple\nobservation: kitchen"
    candidates = [["take apple 1 from table 1", "go to fridge 1"]]
    scorer.reset_episode_batch([state])

    scores = scorer([state], candidates)

    assert scores.shape == (1, 2)
    assert "[Trajectory-conditioned skill guidance]" in qwen.calls[0][0][0]
    assert selector.sessions[0].candidate_skill_rows[0] == [
        alfworld_action_skill_id(action) for action in candidates[0]
    ]
    assert selector.sessions[0].candidate_skill_rows[1] == [
        "alfworld/alfworld-object-picker",
        "alfworld/alfworld-location-navigator",
    ]
    metadata = scorer.last_metadata[0]
    assert metadata["executor_interface"] == "legacy_guided_exact_prior"
    assert metadata["uses_clstr_skill_guidance"] is True
    assert metadata["guidance_conditions_qwen"] is True
    assert metadata["uses_clstr_prior"] is True
    assert metadata["score_fusion"] is True
    assert metadata["runtime_pseudo_skills_allowed"] is True
    assert metadata["runtime_appended_skill_count"] == 2
    assert metadata["selected_abstract_skill_ids"]

    scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["You arrive at the fridge."],
        next_state_texts=["unused"],
        active_mask=[True],
    )
    assert len(selector.sessions[0].observations) == 1
    assert selector.sessions[0].observations[0]["skill_id"] == (
        alfworld_action_skill_id("go to fridge 1")
    )
