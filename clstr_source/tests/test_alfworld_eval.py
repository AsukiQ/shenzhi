import json
import sys
import types
from pathlib import Path

import torch
import pytest

from clstr.closed_loop_controller import ClosedLoopControllerConfig
from clstr.alfworld_eval import (
    build_alfworld_policy_state_text,
    build_alfworld_policy_diagnostic_report,
    build_alfworld_comparison_table,
    build_available_actions_planner_state_text,
    build_alfworld_protocol_report,
    ClstrUnifiedMemoryAdmissibleActionScorer,
    ClstrUnifiedMemoryConcreteActionScorer,
    SkillRouterAdmissibleActionScorer,
    _skill_memory,
    make_candidate_scorer,
    make_controller_component_scorer,
    run_alfworld_closed_loop_eval,
)
from clstr.alfworld_qwen_clstr_gate import QwenClstrHybridActionScorer


class _FakeBatchEnv:
    def __init__(self):
        self.num_games = 1
        self._done = False
        self._step = 0

    def reset(self):
        self._done = False
        self._step = 0
        return ["open the fridge"], {"admissible_commands": [["look around", "open fridge"]], "extra.gamefile": ["task/foo/bar"]}

    def step(self, actions):
        self._step += 1
        success = actions[0] == "open fridge"
        self._done = success
        score = 1.0 if success else 0.0
        done = [self._done]
        infos = {
            "won": [score],
            "goal_condition_success_rate": [score],
            "admissible_commands": [["restart"]],
            "extra.gamefile": ["task/foo/bar"],
        }
        return ["done"], [score], done, infos

    def close(self):
        pass


class _FakeEnvWrapper:
    def __init__(self, config, train_eval):
        self.config = config
        self.train_eval = train_eval
        self.num_games = 1

    def init_env(self, batch_size):
        return _FakeBatchEnv()


class _SequentialBatchEnv:
    def __init__(self):
        self.num_games = 3
        self._episode = -1

    def reset(self):
        self._episode += 1
        target = f"open fridge {self._episode}"
        return [target], {"admissible_commands": [["look around", target]], "extra.gamefile": [f"task/game-{self._episode}"]}

    def step(self, actions):
        success = actions[0] == f"open fridge {self._episode}"
        score = 1.0 if success else 0.0
        return ["done"], [score], [True], {
            "won": [score],
            "goal_condition_success_rate": [score],
            "admissible_commands": [["restart"]],
            "extra.gamefile": [f"task/game-{self._episode}"],
        }

    def close(self):
        pass


class _SequentialEnvWrapper:
    def __init__(self, config, train_eval):
        self.config = config
        self.train_eval = train_eval
        self.num_games = 3

    def init_env(self, batch_size):
        assert batch_size == 1
        return _SequentialBatchEnv()


class _VectorSequentialBatchEnv:
    def __init__(self, batch_size: int, *, drift_gamefiles: bool = False):
        self.batch_size = int(batch_size)
        self.drift_gamefiles = bool(drift_gamefiles)
        self._cursor = 0
        self._active_indices: list[int] = []

    def seed(self, _seed):
        return None

    def reset(self):
        self._active_indices = list(range(self._cursor, self._cursor + self.batch_size))
        self._cursor += self.batch_size
        targets = [f"open fridge {index}" for index in self._active_indices]
        prefix = "drift/game" if self.drift_gamefiles else "task/game"
        return targets, {
            "admissible_commands": [["look around", target] for target in targets],
            "extra.gamefile": [f"{prefix}-{index}" for index in self._active_indices],
        }

    def step(self, actions):
        success = [
            action == f"open fridge {index}"
            for action, index in zip(actions, self._active_indices, strict=True)
        ]
        scores = [1.0 if item else 0.0 for item in success]
        return ["done"] * self.batch_size, scores, [True] * self.batch_size, {
            "won": scores,
            "goal_condition_success_rate": scores,
            "admissible_commands": [["restart"] for _ in self._active_indices],
            "extra.gamefile": [f"task/game-{index}" for index in self._active_indices],
        }

    def close(self):
        pass


class _VectorSequentialEnvWrapper:
    def __init__(self, config, train_eval):
        self.config = config
        self.train_eval = train_eval
        self.num_games = 8

    def init_env(self, batch_size):
        return _VectorSequentialBatchEnv(
            batch_size,
            drift_gamefiles=bool(self.config.get("drift_gamefiles")),
        )


def test_qwen_clstr_hybrid_blocks_coarse_clstr_override_of_reliable_qwen_action():
    class _QwenScorer:
        def __init__(self):
            self.last_metadata = []

        def __call__(self, state_texts, candidate_rows):
            del state_texts
            self.last_metadata = [
                {
                    "policy_family": "qwen_direct_admissible",
                    "parsed_action": "open fridge 1",
                    "parse_status": "exact_match",
                    "fallback_used": False,
                    "chosen_index": 0,
                }
            ]
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    class _CoarseClstrScorer:
        def __init__(self):
            self.last_metadata = []

        def __call__(self, state_texts, candidate_rows):
            del state_texts
            self.last_metadata = [
                {
                    "policy_family": "clstr_unified_memory_admissible_action_scorer",
                    "route_scorer": "unified_memory",
                    "uses_recurrent_m_t": True,
                    "candidate_skill_mappings": [
                        {"action": "open fridge 1", "mapped_skill_id": "alfworld/alfworld-receptacle-opener"},
                        {"action": "open cabinet 2", "mapped_skill_id": "alfworld/alfworld-receptacle-opener"},
                    ],
                }
            ]
            return torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=_QwenScorer(),
        clstr_scorer=_CoarseClstrScorer(),
        qwen_weight=1.0,
        clstr_weight=1.0,
        normalize_scores=False,
    )

    scores = scorer(["goal: cool apple\nobservation: near fridge"], [["open fridge 1", "open cabinet 2"]])

    assert int(torch.argmax(scores[0]).item()) == 0
    assert scorer.last_metadata[0]["hybrid_chosen_action"] == "open fridge 1"
    assert scorer.last_metadata[0]["clstr_override_allowed"] is False
    assert scorer.last_metadata[0]["clstr_override_reason"] == "blocked_coarse_prior_reliable_qwen"


def test_qwen_clstr_hybrid_allows_coarse_clstr_when_qwen_used_fallback():
    class _QwenFallbackScorer:
        def __init__(self):
            self.last_metadata = []

        def __call__(self, state_texts, candidate_rows):
            del state_texts
            self.last_metadata = [
                {
                    "policy_family": "qwen_direct_admissible",
                    "parsed_action": "look",
                    "parse_status": "fallback",
                    "fallback_used": True,
                    "chosen_index": 0,
                }
            ]
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    class _CoarseClstrScorer:
        def __init__(self):
            self.last_metadata = []

        def __call__(self, state_texts, candidate_rows):
            del state_texts
            self.last_metadata = [
                {
                    "policy_family": "clstr_unified_memory_admissible_action_scorer",
                    "route_scorer": "unified_memory",
                    "uses_recurrent_m_t": True,
                    "candidate_skill_mappings": [
                        {"action": "look", "mapped_skill_id": "alfworld/alfworld-object-state-inspector"},
                        {"action": "open fridge 1", "mapped_skill_id": "alfworld/alfworld-receptacle-opener"},
                    ],
                }
            ]
            return torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    scorer = QwenClstrHybridActionScorer(
        qwen_scorer=_QwenFallbackScorer(),
        clstr_scorer=_CoarseClstrScorer(),
        qwen_weight=1.0,
        clstr_weight=1.0,
        normalize_scores=False,
    )

    scores = scorer(["goal: cool apple\nobservation: near fridge"], [["look", "open fridge 1"]])

    assert int(torch.argmax(scores[0]).item()) == 1
    assert scorer.last_metadata[0]["hybrid_chosen_action"] == "open fridge 1"
    assert scorer.last_metadata[0]["clstr_override_allowed"] is True
    assert scorer.last_metadata[0]["clstr_override_reason"] == "allowed_qwen_fallback"


def test_alfworld_skill_memory_uses_sparse_belief_path_not_dense_retrieval():
    class _SkillTable:
        def __init__(self):
            self.E = torch.eye(3)

        def logits(self, h):
            return self.retrieval_logits(h)

        def retrieval_logits(self, h):
            return torch.tensor([[9.0, 0.0, 0.0]], dtype=h.dtype, device=h.device)

        def belief_logits(self, h):
            return torch.tensor([[0.0, 9.0, 0.0]], dtype=h.dtype, device=h.device)

    class _Model:
        def __init__(self):
            self.skill_table = _SkillTable()

    memory = _skill_memory(_Model(), torch.zeros(1, 3))

    assert int(torch.argmax(memory, dim=-1).item()) == 1


def test_alfworld_action_mapper_prefers_exact_concrete_action_skill_id():
    from clstr.alfworld_action_skills import map_alfworld_action_to_skill_id

    mapping = map_alfworld_action_to_skill_id(
        "  Go   to Fridge 1 ",
        {"alfworld_action/go to fridge 1"},
    )

    assert mapping.skill_id == "alfworld_action/go to fridge 1"
    assert mapping.confidence == "exact"
    assert mapping.reason == "exact_concrete_action_skill"


def test_unified_memory_admissible_action_scorer_uses_real_post_action_state(monkeypatch):
    from clstr import alfworld_eval

    captured = {}

    class _Residual(torch.nn.Module):
        def forward(self, h, memory_delta):
            del memory_delta
            return torch.tensor([[0.0, 5.0]], dtype=h.dtype).expand(len(h), -1)

    class _InternalGate(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(features.size(0), dtype=features.dtype)

    class _CandidateAdmissionHead(torch.nn.Module):
        def forward(self, h, memory_delta, candidate_embeddings, scalar_features):
            del h, memory_delta, scalar_features
            admission = candidate_embeddings.new_full(
                candidate_embeddings.shape[:2],
                4.0,
            )
            residual = 4.0 * candidate_embeddings[..., 1]
            return admission, residual

    class _FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [
                {"skill_id": "alfworld/alfworld-location-navigator"},
                {"skill_id": "alfworld/alfworld-object-state-inspector"},
            ]
            self.skill_table = type("_SkillTable", (), {"E": torch.eye(2)})()
            self.route_memory_residual_adapter = _Residual()
            self.route_memory_candidate_utility_gate = _InternalGate()
            self.route_memory_candidate_admission_residual = (
                _CandidateAdmissionHead()
            )
            self.encoded_states = []
            self.encoded_transition_texts = []

        def eval(self):
            return self

        def encode_observations(self, texts):
            self.encoded_transition_texts.extend(str(text) for text in texts)
            return torch.ones(len(texts), 2)

        def encode_states(self, texts):
            self.encoded_states.extend(str(text) for text in texts)
            return torch.ones(len(texts), 2)

        def initial_belief(self, h):
            return torch.zeros_like(h)

        def unified_route_logits(self, h, m, candidate_rows=None):
            captured.setdefault("memories", []).append(m.detach().clone())
            static = torch.tensor([[2.0, 0.1]], dtype=torch.float32).expand(len(candidate_rows), -1)
            dynamic = torch.tensor([[0.1, 2.0]], dtype=torch.float32).expand(len(candidate_rows), -1)
            full = torch.where((m[:, :1] > 0), dynamic, static)
            return full.gather(1, torch.tensor(candidate_rows, dtype=torch.long))

        def route_memory_alpha(self, h, static_memory, dynamic_memory, causal_update_count):
            captured["causal_gate_inputs"] = {
                "h": h.detach().clone(),
                "static_memory": static_memory.detach().clone(),
                "dynamic_memory": dynamic_memory.detach().clone(),
                "causal_update_count": causal_update_count.detach().clone(),
            }
            return torch.ones(len(h), dtype=h.dtype, device=h.device)

    def fake_post_action_memory(model, *, m_t, current_skill_labels, action_embeddings, observation_embeddings, h_next):
        captured["action_text_embedding"] = action_embeddings.detach().clone()
        captured["observation_embedding"] = observation_embeddings.detach().clone()
        captured["next_state_embedding"] = h_next.detach().clone()
        return m_t, m_t, torch.ones_like(m_t)

    monkeypatch.setattr(alfworld_eval, "_post_action_memory", fake_post_action_memory, raising=False)

    model = _FakeModel()
    state_t = "goal: inspect room\nobservation: kitchen\nhistory: <empty>"
    state_t_plus_1 = "goal: inspect room\nobservation: fridge opened\nhistory: go to fridge 1"
    candidates = [["go to fridge 1", "look"]]

    static_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(model, reliability_mode="static")
    dynamic_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(model, reliability_mode="dynamic")
    fixed_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model, reliability_mode="fixed_alpha", fixed_alpha=0.5
    )
    learned_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model,
        reliability_mode="learned",
        memory_utility_gate=lambda features: torch.ones(features.size(0)),
    )
    causal_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model,
        reliability_mode="causal_gate",
        safe_memory_residual_bound=0.1,
    )
    cmc_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model,
        reliability_mode="cmc",
        memory_utility_gate=lambda features: torch.ones(features.size(0)),
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=2.0,
    )
    provenance_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model,
        reliability_mode="cmc_candidate_provenance",
        feature_update_count_cap=4.0,
        feature_candidate_count_cap=2.0,
    )
    admission_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
        model,
        reliability_mode="candidate_admission_residual",
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )
    for scorer in (
        static_scorer,
        dynamic_scorer,
        fixed_scorer,
        learned_scorer,
        causal_scorer,
        cmc_scorer,
        provenance_scorer,
        admission_scorer,
    ):
        scorer.reset_episode_batch([state_t])

    static_scores = static_scorer([state_t], candidates)
    zero_history_scores = learned_scorer([state_t], candidates)
    admission_zero_history_scores = admission_scorer([state_t], candidates)
    dynamic_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    fixed_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    causal_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    cmc_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    provenance_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    admission_scorer.observe_transitions(
        chosen_actions=["go to fridge 1"],
        next_observation_texts=["real next observation"],
        next_state_texts=[state_t_plus_1],
        active_mask=[True],
    )
    dynamic_scores = dynamic_scorer([state_t_plus_1], candidates)
    fixed_scores = fixed_scorer([state_t_plus_1], candidates)
    causal_scores = causal_scorer([state_t_plus_1], candidates)
    cmc_scores = cmc_scorer([state_t_plus_1], candidates)
    provenance_scores = provenance_scorer([state_t_plus_1], candidates)
    admission_scores = admission_scorer([state_t_plus_1], candidates)

    assert static_scores.argmax(dim=-1).item() == 0
    assert dynamic_scores.argmax(dim=-1).item() == 1
    assert causal_scores.argmax(dim=-1).item() == 0
    assert cmc_scores.argmax(dim=-1).item() == 1
    assert cmc_scorer.last_metadata[0]["memory_utility_alpha"] == pytest.approx(1.0)
    assert cmc_scorer.last_metadata[0]["memory_utility_external_gate_loaded"] is True
    assert torch.equal(provenance_scores, static_scores)
    assert (
        provenance_scorer.last_metadata[0]["memory_utility_reliability_mode"]
        == "cmc_candidate_provenance"
    )
    assert provenance_scorer.last_metadata[0]["memory_utility_external_gate_loaded"] is False
    assert torch.equal(zero_history_scores, static_scores)
    assert torch.equal(admission_zero_history_scores, static_scores)
    assert admission_scores.argmax(dim=-1).item() == 1
    assert admission_scorer.last_metadata[0][
        "memory_utility_reliability_mode"
    ] == "candidate_admission_residual"
    assert not torch.equal(fixed_scores, static_scores)
    assert captured["causal_gate_inputs"]["causal_update_count"].tolist() == [1.0]
    assert causal_scorer.last_metadata[0]["memory_utility_reliability_mode"] == "causal_gate"
    assert causal_scorer.last_metadata[0]["safe_memory_residual_bound"] == 0.1
    assert dynamic_scorer.last_transition_inputs == {
        "action_text": "go to fridge 1",
        "next_observation_text": "real next observation",
        "next_state_text": state_t_plus_1,
    }
    assert dynamic_scorer.last_metadata[0]["causal_update_count"] == 1
    assert dynamic_scorer.last_metadata[0]["memory_protocol"] == "stateful_post_action_v1"


def test_unified_memory_concrete_action_scorer_scores_action_text_with_replayed_mt(monkeypatch):
    from clstr import alfworld_eval

    captured = {}

    class _SkillHead:
        def __call__(self, candidate_embs, belief_embs):
            captured["belief_embs"] = belief_embs.detach().clone()
            return torch.einsum("bcd,bd->bc", candidate_embs.float(), belief_embs.float())

    class _FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = type("_SkillTable", (), {"E": torch.eye(2)})()
            self.skill_head = _SkillHead()
            self.encoded_states = []
            self.encoded_transition_texts = []

        def eval(self):
            return self

        def encode_observations(self, texts):
            self.encoded_transition_texts.extend(str(text) for text in texts)
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "open fridge" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                elif "open cabinet" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
            return torch.stack(rows)

        def encode_states(self, texts):
            self.encoded_states.extend(str(text) for text in texts)
            return torch.tensor([[1.0, 0.0] for _text in texts], dtype=torch.float32)

        def initial_belief(self, h):
            return torch.zeros_like(h)

    def fake_apply_replay_prefix_beliefs(model, batch, m_obs, skill_id_to_idx, skill_count, device, trainable=False):
        captured["batch"] = batch
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32), 1

    monkeypatch.setattr(alfworld_eval, "_apply_replay_prefix_beliefs", fake_apply_replay_prefix_beliefs)

    model = _FakeModel()
    scorer = ClstrUnifiedMemoryConcreteActionScorer(model)
    state_text = (
        "goal: cool apple\n"
        "task_type: pick_and_place\n"
        "observation: You are in the kitchen.\n"
        "history: go to fridge 1"
    )
    scores = scorer(
        [state_text],
        [["open cabinet 1", "open fridge 1"]],
    )

    assert int(torch.argmax(scores[0]).item()) == 1
    assert torch.allclose(captured["belief_embs"], torch.tensor([[0.0, 1.0]]))
    assert captured["batch"][0]["replay_prefix"]
    assert captured["batch"][0]["replay_prefix"][0]["observation_source"] == (
        "action_only_no_tool_result"
    )
    assert model.encoded_states == [
        "goal: cool apple\ntask_type: pick_and_place\nobservation: You are in the kitchen."
    ]
    assert set(model.encoded_transition_texts) == {"open cabinet 1", "open fridge 1"}
    assert scorer.last_metadata[0]["policy_family"] == "clstr_unified_memory_concrete_action_scorer"
    assert scorer.last_metadata[0]["uses_recurrent_m_t"] is True


def test_unified_memory_concrete_action_scorer_uses_checkpoint_native_cmc(monkeypatch):
    from clstr import alfworld_eval

    class _SkillHead:
        def __call__(self, candidate_embs, belief_embs):
            return torch.einsum("bcd,bd->bc", candidate_embs.float(), belief_embs.float())

    class _Residual(torch.nn.Module):
        def forward(self, h, memory_delta):
            del memory_delta
            return torch.tensor([[5.0, 0.0]], dtype=h.dtype).expand(len(h), -1)

    class _Gate(torch.nn.Module):
        def forward(self, features):
            return torch.zeros(features.size(0), dtype=features.dtype)

    class _FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = type("_SkillTable", (), {"E": torch.eye(2)})()
            self.skill_head = _SkillHead()
            self.route_memory_residual_adapter = _Residual()
            self.route_memory_candidate_utility_gate = _Gate()

        def eval(self):
            return self

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                rows.append(
                    torch.tensor([1.0, 0.0])
                    if "cabinet" in str(text).lower()
                    else torch.tensor([0.0, 1.0])
                )
            return torch.stack(rows)

        def encode_states(self, texts):
            return torch.ones(len(texts), 2)

        def initial_belief(self, h):
            return torch.zeros_like(h)

    def fake_apply_replay_prefix_beliefs(
        model,
        batch,
        m_obs,
        skill_id_to_idx,
        skill_count,
        device,
        trainable=False,
    ):
        del model, skill_id_to_idx, skill_count, device, trainable
        if batch[0]["replay_prefix"]:
            return torch.tensor([[0.0, 1.0]]), 1
        return m_obs, 0

    monkeypatch.setattr(
        alfworld_eval,
        "_apply_replay_prefix_beliefs",
        fake_apply_replay_prefix_beliefs,
    )
    scorer = ClstrUnifiedMemoryConcreteActionScorer(
        _FakeModel(),
        reliability_mode="cmc",
        memory_utility_gate=lambda features: torch.ones(features.size(0)),
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )
    candidates = [["open cabinet 1", "open fridge 1"]]
    dynamic_scores = scorer(
        ["goal: cool apple\nobservation: kitchen\nhistory: go to fridge 1"],
        candidates,
    )
    assert dynamic_scores.argmax(dim=-1).item() == 0
    assert scorer.last_metadata[0]["memory_utility_reliability_mode"] == "cmc"
    assert scorer.last_metadata[0]["memory_utility_alpha"] == pytest.approx(1.0)
    assert scorer.last_metadata[0]["memory_utility_external_gate_loaded"] is True

    static_scores = scorer(
        ["goal: cool apple\nobservation: kitchen\nhistory: <empty>"],
        candidates,
    )
    assert torch.equal(static_scores, torch.zeros_like(static_scores))
    assert scorer.last_metadata[0]["memory_utility_alpha"] == pytest.approx(0.0)

    provenance_scorer = ClstrUnifiedMemoryConcreteActionScorer(
        _FakeModel(),
        reliability_mode="cmc_candidate_provenance",
        feature_update_count_cap=16.0,
        feature_candidate_count_cap=256.0,
    )
    provenance_scores = provenance_scorer(
        ["goal: cool apple\nobservation: kitchen\nhistory: go to fridge 1"],
        candidates,
    )
    assert torch.equal(provenance_scores, torch.zeros_like(provenance_scores))
    assert (
        provenance_scorer.last_metadata[0]["memory_utility_reliability_mode"]
        == "cmc_candidate_provenance"
    )
    assert provenance_scorer.last_metadata[0]["memory_utility_external_gate_loaded"] is False


@pytest.mark.parametrize(
    "scorer_cls",
    [ClstrUnifiedMemoryAdmissibleActionScorer, ClstrUnifiedMemoryConcreteActionScorer],
)
def test_unified_memory_scorers_require_initial_belief(scorer_cls):
    class _MissingInitializerModel:
        device = torch.device("cpu")
        skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]

        def eval(self):
            return self

    with pytest.raises(ValueError, match="unified memory.*model.initial_belief"):
        scorer_cls(_MissingInitializerModel())


def test_alfworld_scorer_factory_selects_unified_memory_admissible_action_mode():
    from clstr.alfworld_eval import make_alfworld_clstr_candidate_scorer

    class _Model:
        device = torch.device("cpu")
        skills = [{"skill_id": "alfworld_action/go to fridge 1"}]

        def initial_belief(self, h):
            return h

        def eval(self):
            return self

    scorer = make_alfworld_clstr_candidate_scorer(
        _Model(),
        None,
        scorer_mode="unified_memory_admissible_action",
        replay_prefix_max_steps=4,
    )

    assert isinstance(scorer, ClstrUnifiedMemoryAdmissibleActionScorer)
    assert scorer.replay_prefix_max_steps == 4


def test_build_alfworld_policy_state_text_accepts_optional_progress_memory():
    default_state = build_alfworld_policy_state_text(
        "put tissuebox in sidetable",
        "pick_and_place",
        "You are near the bed.",
        ["go to bed 1"],
    )
    memory_state = build_alfworld_policy_state_text(
        "put tissuebox in sidetable",
        "pick_and_place",
        "You are near the bed.",
        ["go to bed 1"],
        progress_memory="placed_target_count=1/2; reverted_target_count=0",
    )

    assert "progress_memory:" not in default_state
    assert "progress_memory: placed_target_count=1/2; reverted_target_count=0" in memory_state


def test_build_alfworld_protocol_report_reads_official_entrypoints(tmp_path):
    official_repo = tmp_path / "alfworld_repo"
    (official_repo / "scripts").mkdir(parents=True)
    (official_repo / "configs").mkdir(parents=True)
    (official_repo / "README.md").write_text(
        "\n".join(
            [
                "pip install alfworld[full]",
                "alfworld-download",
                "python scripts/run_eval.py configs/eval_config.yaml",
            ]
        ),
        encoding="utf-8",
    )
    (official_repo / "scripts" / "run_eval.py").write_text("print('eval')\n", encoding="utf-8")
    (official_repo / "configs" / "eval_config.yaml").write_text(
        "general:\n  evaluate:\n    eval_paths:\n      - valid_seen\n      - valid_unseen\n",
        encoding="utf-8",
    )

    report = build_alfworld_protocol_report(
        official_repo=official_repo,
        output_path=tmp_path / "protocol_report.json",
        data_dir=tmp_path / "alfworld_data",
        use_network_turbo=False,
    )

    assert report["status"] == "ok"
    assert report["official_eval_entrypoint"].endswith("scripts/run_eval.py")
    assert report["metrics"] == ["average_points", "average_goal_condition_points", "average_steps"]
    assert "valid_seen" in report["eval_splits"]
    assert "textworld" in report["dependencies"]
    assert (tmp_path / "protocol_report.json").exists()


def test_run_alfworld_closed_loop_eval_uses_candidate_scorer_and_writes_metrics(tmp_path):
    captured_state_texts = []

    def candidate_scorer(state_texts, candidate_texts):
        captured_state_texts.extend(state_texts)
        scores = []
        for row in candidate_texts:
            scores.append([0.0 if text == "look around" else 1.0 for text in row])
        return __import__("torch").tensor(scores)

    report = run_alfworld_closed_loop_eval(
        env_factory=_FakeEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=candidate_scorer,
        output_dir=tmp_path / "run",
        max_episodes=1,
        max_steps=3,
        batch_size=1,
        run_name="fake_run",
    )

    assert report["status"] == "ok"
    assert report["metrics"]["success_rate"] == 1.0
    assert report["metrics"]["average_episode_steps"] == 1.0
    assert (tmp_path / "run" / "run.jsonl").exists()
    saved = [json.loads(line) for line in (tmp_path / "run" / "run.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert saved[0]["success"] is True
    assert saved[0]["candidate_trace"][0] == ["look around", "open fridge"]
    assert saved[0]["chosen_indices"] == [1]
    assert saved[0]["score_trace_top_actions"][0][0]["action"] == "open fridge"
    assert "goal:" in captured_state_texts[0]
    assert "observation:" in captured_state_texts[0]
    assert "history:" in captured_state_texts[0]


def test_run_alfworld_closed_loop_eval_forwards_real_post_action_transition(tmp_path):
    class _StatefulScorer:
        def __init__(self):
            self.reset_states = None
            self.transition = None

        def reset_episode_batch(self, state_texts):
            self.reset_states = list(state_texts)

        def __call__(self, _state_texts, candidate_texts):
            return torch.tensor(
                [[0.0 if action == "look around" else 1.0 for action in row] for row in candidate_texts]
            )

        def observe_transitions(
            self,
            *,
            chosen_actions,
            next_observation_texts,
            next_state_texts,
            active_mask,
        ):
            self.transition = {
                "chosen_actions": list(chosen_actions),
                "next_observation_texts": list(next_observation_texts),
                "next_state_texts": list(next_state_texts),
                "active_mask": list(active_mask),
            }

    scorer = _StatefulScorer()
    run_alfworld_closed_loop_eval(
        env_factory=_FakeEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=tmp_path / "run",
        max_episodes=1,
        max_steps=1,
        batch_size=1,
        run_name="stateful",
    )

    assert scorer.reset_states and "history: <empty>" in scorer.reset_states[0]
    assert scorer.transition["chosen_actions"] == ["open fridge"]
    assert scorer.transition["active_mask"] == [True]
    assert scorer.transition["next_observation_texts"]
    assert "history: open fridge" in scorer.transition["next_state_texts"][0]


def test_run_alfworld_closed_loop_eval_resumes_existing_run_jsonl_without_restarting_completed_games(tmp_path):
    def candidate_scorer(state_texts, candidate_texts):
        del state_texts
        return torch.tensor([[0.0, 1.0] for _ in candidate_texts])

    output_dir = tmp_path / "resumable_run"
    first = run_alfworld_closed_loop_eval(
        env_factory=_SequentialEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=candidate_scorer,
        output_dir=output_dir,
        max_episodes=1,
        max_steps=3,
        batch_size=1,
        run_name="resume_fake_run",
    )
    existing_rows = [
        json.loads(line)
        for line in (output_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_rows[0]["resume_marker"] = "preserved"
    (output_dir / "run.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in existing_rows) + "\n",
        encoding="utf-8",
    )
    second = run_alfworld_closed_loop_eval(
        env_factory=_SequentialEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=candidate_scorer,
        output_dir=output_dir,
        max_episodes=2,
        max_steps=3,
        batch_size=1,
        run_name="resume_fake_run",
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert first["metrics"]["episode_count"] == 1
    assert second["metrics"]["episode_count"] == 2
    assert [row["gamefile"] for row in rows] == ["task/game-0", "task/game-1"]
    assert [row["episode_index"] for row in rows] == [0, 1]
    assert rows[0]["resume_marker"] == "preserved"


def _vector_candidate_scorer(calls):
    def score(state_texts, candidate_texts):
        calls.append((list(state_texts), [list(row) for row in candidate_texts]))
        return torch.tensor([[0.0, 1.0] for _ in candidate_texts])

    return score


def test_run_alfworld_closed_loop_eval_scores_batch_four_in_one_vectorized_step(tmp_path):
    calls = []

    report = run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=_vector_candidate_scorer(calls),
        output_dir=tmp_path / "batch4",
        max_episodes=4,
        max_steps=3,
        batch_size=4,
        run_name="vector_run",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "batch4" / "run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["metrics"]["episode_count"] == 4
    assert len(calls) == 1
    assert [row["gamefile"] for row in rows] == [f"task/game-{index}" for index in range(4)]


def test_run_alfworld_closed_loop_eval_resumes_aligned_vector_batches(tmp_path):
    calls = []
    output_dir = tmp_path / "batch_resume"
    scorer = _vector_candidate_scorer(calls)
    run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=4,
        max_steps=3,
        batch_size=2,
        run_name="vector_resume",
    )
    existing = [json.loads(line) for line in (output_dir / "run.jsonl").read_text().splitlines()]
    existing[0]["resume_marker"] = "preserved"
    (output_dir / "run.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in existing),
        encoding="utf-8",
    )
    calls.clear()

    report = run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=6,
        max_steps=3,
        batch_size=2,
        run_name="vector_resume",
    )

    rows = [json.loads(line) for line in (output_dir / "run.jsonl").read_text().splitlines()]
    assert report["metrics"]["episode_count"] == 6
    assert len(calls) == 1
    assert rows[0]["resume_marker"] == "preserved"
    assert [row["gamefile"] for row in rows] == [f"task/game-{index}" for index in range(6)]


def test_run_alfworld_closed_loop_eval_rejects_incomplete_vector_batch_resume(tmp_path):
    output_dir = tmp_path / "mid_batch"
    scorer = _vector_candidate_scorer([])
    run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=2,
        batch_size=2,
        run_name="vector_resume",
    )
    first = (output_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (output_dir / "run.jsonl").write_text(first + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="complete batches"):
        run_alfworld_closed_loop_eval(
            env_factory=_VectorSequentialEnvWrapper,
            config={},
            split="valid_seen",
            candidate_scorer=scorer,
            output_dir=output_dir,
            max_episodes=4,
            batch_size=2,
            run_name="vector_resume",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("episode_index", 3), ("split", "valid_unseen"), ("method", "other")],
)
def test_run_alfworld_closed_loop_eval_rejects_resume_identity_drift(
    tmp_path,
    field,
    value,
):
    output_dir = tmp_path / f"identity_{field}"
    scorer = _vector_candidate_scorer([])
    run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=2,
        batch_size=2,
        run_name="vector_resume",
    )
    rows = [json.loads(line) for line in (output_dir / "run.jsonl").read_text().splitlines()]
    rows[0][field] = value
    (output_dir / "run.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resume identity"):
        run_alfworld_closed_loop_eval(
            env_factory=_VectorSequentialEnvWrapper,
            config={},
            split="valid_seen",
            candidate_scorer=scorer,
            output_dir=output_dir,
            max_episodes=4,
            batch_size=2,
            run_name="vector_resume",
        )


def test_run_alfworld_closed_loop_eval_rejects_resume_gamefile_order_drift(tmp_path):
    output_dir = tmp_path / "gamefile_drift"
    scorer = _vector_candidate_scorer([])
    run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=2,
        batch_size=2,
        run_name="vector_resume",
    )

    with pytest.raises(ValueError, match="gamefile order"):
        run_alfworld_closed_loop_eval(
            env_factory=_VectorSequentialEnvWrapper,
            config={"drift_gamefiles": True},
            split="valid_seen",
            candidate_scorer=scorer,
            output_dir=output_dir,
            max_episodes=4,
            batch_size=2,
            run_name="vector_resume",
        )


def test_run_alfworld_closed_loop_eval_reopens_complete_partial_final_batch(tmp_path):
    output_dir = tmp_path / "partial_final"
    calls = []
    scorer = _vector_candidate_scorer(calls)
    first = run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=3,
        batch_size=2,
        run_name="vector_resume",
    )
    calls.clear()

    second = run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=3,
        batch_size=2,
        run_name="vector_resume",
    )

    assert first["metrics"]["episode_count"] == 3
    assert second["metrics"] == first["metrics"]
    assert calls == []


def test_run_alfworld_closed_loop_eval_keeps_last_committed_batch_on_atomic_replace_failure(
    tmp_path,
    monkeypatch,
):
    from clstr import alfworld_eval

    output_dir = tmp_path / "atomic_failure"
    scorer = _vector_candidate_scorer([])
    run_alfworld_closed_loop_eval(
        env_factory=_VectorSequentialEnvWrapper,
        config={},
        split="valid_seen",
        candidate_scorer=scorer,
        output_dir=output_dir,
        max_episodes=2,
        batch_size=2,
        run_name="vector_resume",
    )
    before = (output_dir / "run.jsonl").read_text(encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(alfworld_eval.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        run_alfworld_closed_loop_eval(
            env_factory=_VectorSequentialEnvWrapper,
            config={},
            split="valid_seen",
            candidate_scorer=scorer,
            output_dir=output_dir,
            max_episodes=4,
            batch_size=2,
            run_name="vector_resume",
        )

    assert (output_dir / "run.jsonl").read_text(encoding="utf-8") == before
    assert not list(output_dir.glob(".run.jsonl.*.tmp"))


def test_run_alfworld_closed_loop_eval_writes_optional_policy_metadata_trace(tmp_path):
    class _MetadataScorer:
        def __init__(self):
            self.last_metadata = []

        def __call__(self, state_texts, candidate_texts):
            del state_texts
            self.last_metadata = [
                {
                    "policy_family": "qwen_direct_admissible",
                    "raw_model_response": "Action: open fridge",
                    "parsed_action": "open fridge",
                    "parse_status": "exact_match",
                    "fallback_used": False,
                    "model_name_or_path": "fake-qwen",
                }
            ]
            return torch.tensor([[0.0 if text == "look around" else 1.0 for text in candidate_texts[0]]])

    report = run_alfworld_closed_loop_eval(
        env_factory=_FakeEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=_MetadataScorer(),
        output_dir=tmp_path / "metadata_run",
        max_episodes=1,
        max_steps=3,
        batch_size=1,
        run_name="qwen_direct_fake_run",
    )

    assert report["metrics"]["success_rate"] == 1.0
    row = json.loads((tmp_path / "metadata_run" / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["policy_family_trace"] == ["qwen_direct_admissible"]
    assert row["raw_model_response_trace"] == ["Action: open fridge"]
    assert row["parsed_action_trace"] == ["open fridge"]
    assert row["parse_status_trace"] == ["exact_match"]
    assert row["fallback_trace"] == [False]


def test_run_alfworld_closed_loop_eval_derives_replay_memory_update_count(tmp_path):
    class _TwoStepBatchEnv(_FakeBatchEnv):
        def step(self, actions):
            self._step += 1
            success = self._step >= 2 and actions[0] == "open fridge"
            score = 1.0 if success else 0.0
            return ["continue" if not success else "done"], [score], [success], {
                "won": [score],
                "goal_condition_success_rate": [score],
                "admissible_commands": [["look around", "open fridge"]],
                "extra.gamefile": ["task/foo/bar"],
            }

    class _TwoStepEnvWrapper(_FakeEnvWrapper):
        def init_env(self, batch_size):
            assert batch_size == 1
            return _TwoStepBatchEnv()

    class _ReplayMetadataScorer:
        def __init__(self):
            self.calls = 0
            self.last_metadata = []

        def __call__(self, state_texts, candidate_texts):
            del state_texts
            replay_len = self.calls
            self.last_metadata = [
                {
                    "clstr_memory_source": (
                        "replay_prefix" if replay_len else "initial_belief"
                    ),
                    "clstr_replay_prefix_len": replay_len,
                    "clstr_replay_prefix_used_count": int(replay_len > 0),
                    "uses_recurrent_m_t": bool(replay_len),
                }
            ]
            self.calls += 1
            return torch.tensor(
                [[1.0, 0.0] if replay_len == 0 else [0.0, 1.0]],
                dtype=torch.float32,
            )

    run_alfworld_closed_loop_eval(
        env_factory=_TwoStepEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=_ReplayMetadataScorer(),
        output_dir=tmp_path / "replay_memory_count",
        max_episodes=1,
        max_steps=3,
        batch_size=1,
        run_name="replay_memory_count",
    )

    row = json.loads(
        (tmp_path / "replay_memory_count" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["causal_update_count_final"] == 1
    assert row["memory_protocol"] == "stateful_post_action_v1"


def test_skillrouter_admissible_action_scorer_ranks_same_candidate_set(monkeypatch):
    def fake_encode(self, texts):
        del self
        rows = []
        for text in texts:
            if "fridge" in text and "look around" not in text:
                rows.append(torch.tensor([0.0, 1.0]))
            else:
                rows.append(torch.tensor([1.0, 0.0]))
        return torch.stack(rows)

    monkeypatch.setattr(SkillRouterAdmissibleActionScorer, "_encode", fake_encode)
    scorer = SkillRouterAdmissibleActionScorer(
        model_name_or_path="fake",
        batch_size=4,
        max_length=128,
    )

    scores = scorer(["goal: open the fridge\nobservation: near fridge"], [["look around", "open fridge"]])

    assert scores.shape == (1, 2)
    assert int(scores.argmax(dim=-1).item()) == 1
    assert scorer.last_metadata[0]["policy_family"] == "skillrouter_frozen_admissible_action"


def test_alfworld_skillrouter_cli_help_is_available():
    result = __import__("subprocess").run(
        [sys.executable, "scripts/run_alfworld_skillrouter_eval.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "skillrouter" in result.stdout.lower()
    assert "--model_name_or_path" in result.stdout


def test_evaluate_alfworld_qwen_direct_can_use_likelihood_ranker(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    captured = {}

    class _FakeLikelihoodScorer:
        def __init__(self, config):
            captured["config"] = config
            self.config = config
            self.last_metadata = []

        def __call__(self, state_texts, candidate_texts):
            self.last_metadata = [
                {
                    "policy_family": "qwen_direct_likelihood_admissible",
                    "action_selection": "direct_loglikelihood_ranking_over_admissible_actions",
                    "parsed_action": "open fridge",
                    "parse_status": "likelihood_ranked_exact_action",
                    "fallback_used": False,
                }
            ]
            return torch.tensor([[0.0 if text == "look around" else 1.0 for text in candidate_texts[0]]])

    monkeypatch.setattr(alfworld_eval, "QwenDirectLikelihoodActionScorer", _FakeLikelihoodScorer)
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda official_repo, data_dir, split: ({"env": {"type": "AlfredTWEnv"}}, "eval_in_distribution"),
    )
    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda env_type: _FakeEnvWrapper
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_qwen_direct(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        output_dir=tmp_path / "qwen_likelihood",
        model_name_or_path="fake-qwen",
        split="valid_seen",
        max_episodes=1,
        max_steps=3,
        scoring_method="likelihood",
    )

    assert report["metrics"]["success_rate"] == 1.0
    assert report["metrics"]["scoring_method"] == "likelihood"
    assert report["metrics"]["qwen_direct_generator"] is False
    assert report["metrics"]["qwen_direct_likelihood_ranker"] is True


def test_qwen_direct_cli_treats_zero_max_episodes_as_full_split(tmp_path, monkeypatch):
    from scripts import run_qwen3_alfworld_direct_eval

    captured = {}

    monkeypatch.setattr(run_qwen3_alfworld_direct_eval, "build_alfworld_protocol_report", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(run_qwen3_alfworld_direct_eval, "build_alfworld_env_report", lambda *args, **kwargs: {"status": "ok"})

    def fake_evaluate_alfworld_qwen_direct(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.jsonl").write_text("", encoding="utf-8")
        return {
            "status": "ok",
            "metrics": {
                "status": "ok",
                "episode_count": 0,
                "success_rate": 0.0,
                "average_reward": 0.0,
                "average_goal_condition_points": 0.0,
                "average_episode_steps": 0.0,
            },
        }

    monkeypatch.setattr(run_qwen3_alfworld_direct_eval, "evaluate_alfworld_qwen_direct", fake_evaluate_alfworld_qwen_direct)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen3_alfworld_direct_eval.py",
            "--data_dir",
            str(tmp_path / "alfworld_data"),
            "--official_repo",
            str(tmp_path / "alfworld_repo"),
            "--output_dir",
            str(tmp_path / "qwen_full"),
            "--splits",
            "valid_seen",
            "--max_episodes",
            "0",
            "--scoring_method",
            "likelihood",
        ],
    )

    run_qwen3_alfworld_direct_eval.main()

    assert captured["max_episodes"] is None
    assert captured["scoring_method"] == "likelihood"


def test_run_alfworld_closed_loop_eval_uses_controller_components_and_writes_trace(tmp_path):
    def candidate_scorer(state_texts, candidate_texts):
        del state_texts
        return torch.tensor([[1.0 if text == "look around" else 0.4 for text in candidate_texts[0]]])

    def component_scorer(state_texts, candidate_texts, policy_scores, action_histories):
        del state_texts, policy_scores, action_histories
        transition = torch.tensor([[0.0 if text == "look around" else 3.0 for text in candidate_texts[0]]])
        belief = torch.tensor([[0.0 if text == "look around" else 1.0 for text in candidate_texts[0]]])
        stop_logits = torch.full_like(transition, -2.0)
        return {
            "transition_scores": transition,
            "belief_scores": belief,
            "stop_logits": stop_logits,
        }

    report = run_alfworld_closed_loop_eval(
        env_factory=_FakeEnvWrapper,
        config={"env": {"type": "AlfredTWEnv"}},
        split="valid_seen",
        candidate_scorer=candidate_scorer,
        output_dir=tmp_path / "controller_run",
        max_episodes=1,
        max_steps=3,
        batch_size=1,
        run_name="controller_fake_run",
        component_scorer=component_scorer,
        controller_config=ClosedLoopControllerConfig(mode="policy_plus_transition_belief_stop"),
    )

    assert report["metrics"]["success_rate"] == 1.0
    row = json.loads((tmp_path / "controller_run" / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["action_trace"] == ["open fridge"]
    assert row["chosen_action_trace"] == ["open fridge"]
    assert row["chosen_reason_trace"] == ["max_final_score"]
    assert row["component_score_trace"][0][0]["policy_score"] == 1.0
    assert row["component_score_trace"][0][1]["transition_score"] == 3.0
    assert row["component_score_trace"][0][1]["final_score"] > row["component_score_trace"][0][0]["final_score"]
    diagnostic = json.loads((tmp_path / "controller_run" / "controller_diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["component_trace_logged"] is True
    assert diagnostic["controller_mode_counts"]["policy_plus_transition_belief_stop"] == 2


def test_candidate_scorer_prefers_native_clstr_skill_head_over_legacy_adapter():
    class _NativePolicyModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skill_head = self
            self.skill_table = None

        def eval(self):
            return self

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "open fridge" in lowered:
                    rows.append(torch.tensor([2.0, 0.0], dtype=torch.float32))
                elif "look" in lowered:
                    rows.append(torch.tensor([0.0, 2.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.0, 0.0], dtype=torch.float32))
            return torch.stack(rows)

        def __call__(self, candidate_embs, belief_embs):
            del belief_embs
            return candidate_embs[:, :, 0] * 4.0

    class _FailingAdapter:
        def eval(self):
            return self

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("legacy UniversalActionAdapter should not be used when native skill_head exists")

    scorer = make_candidate_scorer(_NativePolicyModel(), action_adapter=_FailingAdapter())
    scores = scorer(
        ["goal: put lettuce in fridge\nobservation: kitchen\nhistory: <empty>"],
        [["look", "open fridge"]],
    )

    assert int(torch.argmax(scores, dim=-1).item()) == 1


def test_available_actions_planner_state_text_exposes_candidates_without_direct_choice():
    text = build_available_actions_planner_state_text(
        "goal: heat apple\nobservation: kitchen\nhistory: <empty>",
        ["look", "open fridge 1"],
    )

    assert "AVAILABLE ACTIONS" in text
    assert "1. look" in text
    assert "2. open fridge 1" in text
    assert "CLSTR skill-routing query" in text
    assert "Do not directly execute" in text


def test_candidate_scorer_can_encode_available_actions_as_latent_planner_context():
    class _PlannerRecordingModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skill_head = self
            self.skill_table = None
            self.encoded_texts = []

        def encode_observations(self, texts):
            self.encoded_texts.extend(str(text) for text in texts)
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "available actions" in lowered and "open fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "open fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "look" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.0, 0.0], dtype=torch.float32))
            return torch.stack(rows)

        def __call__(self, candidate_embs, belief_embs):
            return torch.einsum("bcd,bd->bc", candidate_embs, belief_embs)

    model = _PlannerRecordingModel()
    scorer = make_candidate_scorer(model, include_available_actions_in_state=True)
    scores = scorer(
        ["goal: heat apple\nobservation: kitchen\nhistory: <empty>"],
        [["look", "open fridge 1"]],
    )

    assert int(scores.argmax(dim=-1).item()) == 1
    assert any("AVAILABLE ACTIONS" in text for text in model.encoded_texts)
    assert scorer.last_metadata[0]["policy_family"] == "clstr_qwen_available_actions_planner"
    assert scorer.last_metadata[0]["qwen_direct_generator"] is False


def test_candidate_scorer_reuses_candidate_embedding_cache_between_steps():
    class _CountingModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skill_head = None
            self.skill_table = None
            self.encoded_texts = []

        def encode_observations(self, texts):
            self.encoded_texts.extend(str(text) for text in texts)
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "open fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "look" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
            return torch.stack(rows)

    model = _CountingModel()
    scorer = make_candidate_scorer(model)
    scorer(["goal: open fridge"], [["look", "open fridge"]])
    model.encoded_texts.clear()

    scorer(["goal: open fridge"], [["look", "open fridge"]])

    assert "goal: open fridge" in model.encoded_texts
    assert "look" not in model.encoded_texts
    assert "open fridge" not in model.encoded_texts


def test_candidate_scorer_prompts_state_but_keeps_candidate_actions_unprompted():
    class _RoleAwareModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skill_head = None
            self.skill_table = None
            self.encoded_states = []
            self.encoded_transition_texts = []

        def encode_states(self, texts):
            self.encoded_states.extend(str(text) for text in texts)
            return torch.tensor([[1.0, 0.0] for _text in texts], dtype=torch.float32)

        def encode_observations(self, texts):
            self.encoded_transition_texts.extend(str(text) for text in texts)
            rows = []
            for text in texts:
                if "open fridge" in str(text).lower():
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
            return torch.stack(rows)

    model = _RoleAwareModel()
    scorer = make_candidate_scorer(model)
    state_text = "goal: open fridge\nobservation: kitchen\nhistory: <empty>"

    scores = scorer([state_text], [["look", "open fridge"]])

    assert int(scores.argmax(dim=-1).item()) == 1
    assert model.encoded_states == ["goal: open fridge\nobservation: kitchen"]
    assert set(model.encoded_transition_texts) == {"look", "open fridge"}


def test_controller_component_scorer_does_not_use_proxy_transition_without_next_observation():
    class _RecordingTransition:
        def __init__(self):
            self.calls = 0

        def __call__(self, m_obs, labels, action_embs):
            del m_obs, labels, action_embs
            self.calls += 1
            return torch.zeros(2, 2)

    class _NoProxyObservationModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = None
            self.stop_head = None
            self.transition = _RecordingTransition()
            self.gate = None

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "cool lettuce" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                elif "go to fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "open microwave" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.0, 0.1], dtype=torch.float32))
            return torch.stack(rows)

    model = _NoProxyObservationModel()
    scorer = make_controller_component_scorer(model)
    scores = scorer(
        ["goal: cool lettuce\nobservation: You are in the kitchen.\nhistory: <empty>"],
        [["go to fridge 1", "open microwave 1"]],
        torch.zeros(1, 2),
        [[]],
    )

    assert model.transition.calls == 0
    assert torch.equal(scores["transition_scores"], torch.zeros(1, 2))
    assert torch.equal(scores["belief_scores"], torch.zeros(1, 2))
    assert scorer.last_metadata[0]["transition_score_source"] == "disabled_no_post_action_observation"
    assert scorer.last_metadata[0]["belief_score_source"] == "disabled_no_post_action_observation"


def test_controller_component_scorer_returns_components_on_policy_score_device():
    class _CpuModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = None
            self.stop_head = None
            self.q_success_head = None

        def encode_observations(self, texts):
            return torch.zeros(len(texts), 2, dtype=torch.float32)

    scorer = make_controller_component_scorer(_CpuModel())
    scores = scorer(
        ["goal: inspect room\nobservation: kitchen\nhistory: <empty>"],
        [["look", "inventory"]],
        torch.zeros(1, 2, device="meta"),
        [[]],
    )

    assert all(value.device.type == "meta" for value in scores.values())


def test_controller_component_scorer_does_not_call_trained_heads_as_proxy_transition():
    class _RecordingTransHead:
        def __init__(self):
            self.calls = []

        def __call__(self, pred, candidate_embs):
            self.calls.append((pred.detach().clone(), candidate_embs.detach().clone()))
            return torch.tensor([[1.5], [-2.0]], dtype=torch.float32, device=pred.device)

    class _RecordingSkillHead:
        def __init__(self):
            self.calls = []

        def __call__(self, candidate_embs, belief_embs):
            self.calls.append((candidate_embs.detach().clone(), belief_embs.detach().clone()))
            return torch.tensor([[0.25], [3.5]], dtype=torch.float32, device=candidate_embs.device)

    class _TrainedHeadModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = None
            self.transition = self
            self.gate = None
            self.stop_head = None
            self.trans_head = _RecordingTransHead()
            self.skill_head = _RecordingSkillHead()

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "go to fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "look" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.5, 0.5], dtype=torch.float32))
            return torch.stack(rows)

        def __call__(self, m_obs, labels, action_embs):
            del labels
            return m_obs + action_embs

    model = _TrainedHeadModel()
    scorer = make_controller_component_scorer(model)
    scores = scorer(
        ["goal: cool lettuce\nobservation: You are in the kitchen.\nhistory: <empty>"],
        [["go to fridge 1", "look"]],
        torch.zeros(1, 2),
        [[]],
    )

    assert model.trans_head.calls == []
    assert model.skill_head.calls == []
    assert torch.equal(scores["transition_scores"], torch.zeros(1, 2))
    assert torch.equal(scores["belief_scores"], torch.zeros(1, 2))
    assert scorer.last_metadata[0]["component_score_source"] == "policy_q_success_stop_only_no_proxy_observation"


def test_controller_component_scorer_does_not_use_action_embedding_for_proxy_transition_head():
    class _RecordingTransHead:
        def __init__(self):
            self.calls = []

        def __call__(self, pred, candidate_embs):
            self.calls.append((pred.detach().clone(), candidate_embs.detach().clone()))
            return torch.tensor([[2.0], [-1.0]], dtype=torch.float32, device=pred.device)

    class _ActionEmbedding:
        def __init__(self):
            self.calls = 0

        def __call__(self, labels):
            self.calls += 1
            values = labels.detach().float()
            return torch.stack([values, values + 10.0, values + 20.0], dim=-1)

    class _TransitionHeadModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [
                {"skill_id": "alfworld/alfworld-location-navigator"},
                {"skill_id": "alfworld/alfworld-object-state-inspector"},
            ]
            self.skill_table = None
            self.transition = self
            self.gate = None
            self.stop_head = None
            self.skill_head = None
            self.trans_head = _RecordingTransHead()
            self.action_emb = _ActionEmbedding()

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "go to fridge" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "look" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.5, 0.5], dtype=torch.float32))
            return torch.stack(rows)

        def __call__(self, m_obs, labels, action_embs):
            del labels, action_embs
            return m_obs

    model = _TransitionHeadModel()
    scorer = make_controller_component_scorer(model)
    scores = scorer(
        ["goal: cool lettuce\nobservation: start\nhistory: <empty>"],
        [["go to fridge 1", "look"]],
        torch.zeros(1, 2),
        [[]],
    )

    assert model.trans_head.calls == []
    assert model.action_emb.calls == 0
    assert torch.equal(scores["transition_scores"], torch.zeros(1, 2))
    assert scorer.last_metadata[0]["transition_candidate_embedding_source"] == "not_used"


def test_controller_component_scorer_uses_current_state_stop_logits_without_proxy_transition():
    class _CandidateStopModel:
        device = torch.device("cpu")

        def __init__(self):
            self.skills = [{"skill_id": "alfworld/alfworld-location-navigator"}]
            self.skill_table = None
            self.transition_calls = 0
            self.transition = self
            self.gate = None
            self.stop_head = self

        def encode_observations(self, texts):
            rows = []
            for text in texts:
                lowered = str(text).lower()
                if "finish action" in lowered:
                    rows.append(torch.tensor([1.0, 0.0], dtype=torch.float32))
                elif "continue action" in lowered:
                    rows.append(torch.tensor([0.0, 1.0], dtype=torch.float32))
                else:
                    rows.append(torch.tensor([0.0, 0.0], dtype=torch.float32))
            return torch.stack(rows)

        def __call__(self, *args):
            if len(args) == 2:
                _h, m_obs = args
                return m_obs[:, 0]
            _m_obs, _labels, action_embs = args
            self.transition_calls += 1
            return action_embs

    model = _CandidateStopModel()
    scorer = make_controller_component_scorer(model)
    scores = scorer(
        ["goal: complete task\nobservation: start\nhistory: <empty>"],
        [["finish action", "continue action"]],
        torch.zeros(1, 2),
        [[]],
    )

    assert model.transition_calls == 0
    assert scores["stop_logits"].shape == (1, 2)
    assert torch.allclose(scores["stop_logits"][0, 0], scores["stop_logits"][0, 1])


def test_evaluate_alfworld_clstr_env_factory_accepts_train_eval_keyword(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    captured_loader_args = {}

    def fake_load_model(
        routing_init_manifest,
        checkpoint_path,
        output_dir,
        data_root="data/aux_trajectories",
        stage4_checkpoint_path=None,
        skill_rows_path_override=None,
    ):
        captured_loader_args["data_root"] = data_root
        captured_loader_args["stage4_checkpoint_path"] = stage4_checkpoint_path
        captured_loader_args["skill_rows_path_override"] = skill_rows_path_override
        return (
            object(),
            None,
            {"routing_init_manifest": str(routing_init_manifest)},
        )

    monkeypatch.setattr(
        alfworld_eval,
        "_load_clstr_alfworld_model",
        fake_load_model,
    )
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda official_repo, data_dir, split: ({"env": {"type": "AlfredTWEnv"}}, "eval_in_distribution"),
    )

    def fake_run_alfworld_closed_loop_eval(env_factory, config, split, candidate_scorer, output_dir, **kwargs):
        wrapper = env_factory(config, train_eval=split)
        assert wrapper.train_eval == "eval_in_distribution"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"status": "ok", "method": "test", "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 0.0}
        (output_dir / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return {"status": "ok", "metrics": metrics, "result": wrapper}

    monkeypatch.setattr(alfworld_eval, "run_alfworld_closed_loop_eval", fake_run_alfworld_closed_loop_eval)
    monkeypatch.setattr(
        alfworld_eval,
        "make_candidate_scorer",
        lambda model, action_adapter=None: (lambda state_texts, candidate_texts: __import__("torch").zeros((len(candidate_texts), len(candidate_texts[0])))),
    )

    class _FakeEnv:
        def __init__(self, cfg, train_eval):
            self.cfg = cfg
            self.train_eval = train_eval

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda env_type: (lambda cfg, train_eval: _FakeEnv(cfg, train_eval))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_clstr(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        output_dir=tmp_path / "out",
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=None,
        stage4_checkpoint_path=tmp_path / "stage4.pt",
        split="valid_seen",
        run_name="test",
        max_episodes=1,
        max_steps=1,
        batch_size=1,
        aux_data_root=tmp_path / "full_base_train",
    )

    assert report["status"] == "ok"
    assert report["metrics"]["trajectory_trained"] is False
    assert captured_loader_args["data_root"] == tmp_path / "full_base_train"
    assert captured_loader_args["stage4_checkpoint_path"] == tmp_path / "stage4.pt"
    assert report["metrics"]["stage4_checkpoint_path"] == str(tmp_path / "stage4.pt")
    assert report["metrics"]["aux_data_root"] == str(tmp_path / "full_base_train")


def test_evaluate_alfworld_clstr_final_chain_selects_unified_memory_scorer(
    tmp_path,
    monkeypatch,
):
    from clstr import alfworld_eval

    captured = {}
    model = object()

    def _fake_load_model(**kwargs):
        captured["loader"] = kwargs
        return model, None, {
            "final_chain_adapter": {
                "checkpoint_load_order": ["stage0", "stage2", "stage4"]
            }
        }

    def _fake_make_scorer(loaded_model, action_adapter, **kwargs):
        captured["scorer"] = {
            "model": loaded_model,
            "action_adapter": action_adapter,
            **kwargs,
        }

        def _score(_state_texts, candidate_texts):
            return torch.zeros((len(candidate_texts), len(candidate_texts[0])))

        return _score

    def _fake_run(**kwargs):
        captured["run"] = kwargs
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "status": "ok",
            "method": "test",
            "success_rate": 0.0,
            "average_reward": 0.0,
            "average_episode_steps": 0.0,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics) + "\n",
            encoding="utf-8",
        )
        return {"status": "ok", "metrics": metrics}

    monkeypatch.setattr(alfworld_eval, "_load_clstr_alfworld_model", _fake_load_model)
    monkeypatch.setattr(
        alfworld_eval,
        "make_alfworld_clstr_candidate_scorer",
        _fake_make_scorer,
    )
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda *_args, **_kwargs: (
            {"env": {"type": "AlfredTWEnv"}},
            "eval_in_distribution",
        ),
    )
    monkeypatch.setattr(alfworld_eval, "run_alfworld_closed_loop_eval", _fake_run)
    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda _env_type: object
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_clstr(
        official_repo=tmp_path / "repo",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        routing_init_manifest=tmp_path / "final_chain.json",
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        checkpoint_path=tmp_path / "stage2.pt",
        stage4_checkpoint_path=tmp_path / "stage4.pt",
        skill_rows_path_override=tmp_path / "training_skills.jsonl",
        benchmark_skill_rows_path=tmp_path / "alfworld_skills.jsonl",
        scorer_mode="unified_memory_admissible_action",
        replay_prefix_max_steps=5,
        split="valid_seen",
        run_name="test",
        max_episodes=1,
    )

    assert captured["loader"]["stage0_checkpoint_path"] == tmp_path / "stage0.pt"
    assert captured["loader"]["benchmark_skill_rows_path"] == tmp_path / "alfworld_skills.jsonl"
    assert captured["scorer"]["model"] is model
    assert captured["scorer"]["scorer_mode"] == "unified_memory_admissible_action"
    assert captured["scorer"]["replay_prefix_max_steps"] == 5
    assert report["metrics"]["candidate_scorer_mode"] == "unified_memory_admissible_action"
    assert report["metrics"]["replay_prefix_max_steps"] == 5


def test_evaluate_alfworld_clstr_passes_controller_component_scorer(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    captured = {}

    monkeypatch.setattr(
        alfworld_eval,
        "_load_clstr_alfworld_model",
        lambda routing_init_manifest, checkpoint_path, output_dir, data_root="data/aux_trajectories", stage4_checkpoint_path=None, skill_rows_path_override=None: (
            object(),
            None,
            {"routing_init_manifest": str(routing_init_manifest)},
        ),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda official_repo, data_dir, split: ({"env": {"type": "AlfredTWEnv"}}, "eval_in_distribution"),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "make_candidate_scorer",
        lambda model, action_adapter=None: (lambda state_texts, candidate_texts: torch.zeros((len(candidate_texts), len(candidate_texts[0])))),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "make_controller_component_scorer",
        lambda model: (lambda state_texts, candidate_texts, policy_scores, action_histories: {
            "transition_scores": torch.ones_like(policy_scores),
            "belief_scores": torch.ones_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
        }),
    )

    def fake_run_alfworld_closed_loop_eval(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"status": "ok", "method": "test", "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 0.0}
        (output_dir / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return {"status": "ok", "metrics": metrics}

    monkeypatch.setattr(alfworld_eval, "run_alfworld_closed_loop_eval", fake_run_alfworld_closed_loop_eval)

    class _FakeEnv:
        def __init__(self, cfg, train_eval):
            self.cfg = cfg
            self.train_eval = train_eval

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda env_type: (lambda cfg, train_eval: _FakeEnv(cfg, train_eval))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_clstr(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        output_dir=tmp_path / "out",
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint.pt",
        split="valid_seen",
        run_name="controller_test",
        max_episodes=1,
        max_steps=1,
        batch_size=1,
        controller_mode="policy_plus_transition_belief_stop",
    )

    assert captured["component_scorer"] is not None
    assert captured["controller_config"].mode == "policy_plus_transition_belief_stop"
    assert report["metrics"]["controller_mode"] == "policy_plus_transition_belief_stop"
    assert report["metrics"]["transition_belief_stop_participate_in_action_selection"] is True


def test_evaluate_alfworld_clstr_reports_disabled_transition_belief_components(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    monkeypatch.setattr(
        alfworld_eval,
        "_load_clstr_alfworld_model",
        lambda routing_init_manifest, checkpoint_path, output_dir, data_root="data/aux_trajectories", stage4_checkpoint_path=None, skill_rows_path_override=None: (
            object(),
            None,
            {"routing_init_manifest": str(routing_init_manifest)},
        ),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda official_repo, data_dir, split: ({"env": {"type": "AlfredTWEnv"}}, "eval_in_distribution"),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "make_candidate_scorer",
        lambda model, action_adapter=None: (lambda state_texts, candidate_texts: torch.zeros((len(candidate_texts), len(candidate_texts[0])))),
    )

    def disabled_component_scorer(*args, **kwargs):
        policy_scores = args[2]
        return {
            "transition_scores": torch.zeros_like(policy_scores),
            "belief_scores": torch.zeros_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
        }

    disabled_component_scorer.supports_transition_scores = False
    disabled_component_scorer.supports_belief_scores = False
    monkeypatch.setattr(alfworld_eval, "make_controller_component_scorer", lambda model: disabled_component_scorer)

    def fake_run_alfworld_closed_loop_eval(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"status": "ok", "method": "test", "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 0.0}
        (output_dir / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return {"status": "ok", "metrics": metrics}

    monkeypatch.setattr(alfworld_eval, "run_alfworld_closed_loop_eval", fake_run_alfworld_closed_loop_eval)

    class _FakeEnv:
        def __init__(self, cfg, train_eval):
            self.cfg = cfg
            self.train_eval = train_eval

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda env_type: (lambda cfg, train_eval: _FakeEnv(cfg, train_eval))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_clstr(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        output_dir=tmp_path / "out",
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint.pt",
        split="valid_seen",
        run_name="controller_test",
        max_episodes=1,
        max_steps=1,
        batch_size=1,
        controller_mode="policy_plus_transition_belief_stop",
    )

    assert report["metrics"]["controller_mode"] == "policy_plus_transition_belief_stop"
    assert report["metrics"]["transition_participates_in_action_selection"] is False
    assert report["metrics"]["belief_participates_in_action_selection"] is False
    assert report["metrics"]["transition_belief_stop_participate_in_action_selection"] is False


def test_evaluate_alfworld_clstr_passes_q_success_weight_to_controller(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    captured = {}

    monkeypatch.setattr(
        alfworld_eval,
        "_load_clstr_alfworld_model",
        lambda routing_init_manifest, checkpoint_path, output_dir, data_root="data/aux_trajectories", stage4_checkpoint_path=None, skill_rows_path_override=None: (
            object(),
            None,
            {"routing_init_manifest": str(routing_init_manifest)},
        ),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "_make_env_config",
        lambda official_repo, data_dir, split: ({"env": {"type": "AlfredTWEnv"}}, "eval_in_distribution"),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "make_candidate_scorer",
        lambda model, action_adapter=None: (lambda state_texts, candidate_texts: torch.zeros((len(candidate_texts), len(candidate_texts[0])))),
    )
    monkeypatch.setattr(
        alfworld_eval,
        "make_controller_component_scorer",
        lambda model: (lambda state_texts, candidate_texts, policy_scores, action_histories: {
            "transition_scores": torch.ones_like(policy_scores),
            "belief_scores": torch.ones_like(policy_scores),
            "stop_logits": torch.zeros_like(policy_scores),
            "q_success_scores": torch.ones_like(policy_scores) * 2.0,
        }),
    )

    def fake_run_alfworld_closed_loop_eval(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"status": "ok", "method": "test", "success_rate": 0.0, "average_reward": 0.0, "average_episode_steps": 0.0}
        (output_dir / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return {"status": "ok", "metrics": metrics}

    monkeypatch.setattr(alfworld_eval, "run_alfworld_closed_loop_eval", fake_run_alfworld_closed_loop_eval)

    class _FakeEnv:
        def __init__(self, cfg, train_eval):
            self.cfg = cfg
            self.train_eval = train_eval

    fake_environment_module = types.ModuleType("alfworld.agents.environment")
    fake_environment_module.get_environment = lambda env_type: (lambda cfg, train_eval: _FakeEnv(cfg, train_eval))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment_module)

    report = alfworld_eval.evaluate_alfworld_clstr(
        official_repo=tmp_path / "alfworld_repo",
        data_dir=tmp_path / "alfworld_data",
        output_dir=tmp_path / "out",
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=tmp_path / "checkpoint.pt",
        split="valid_seen",
        run_name="controller_test",
        max_episodes=1,
        max_steps=1,
        batch_size=1,
        controller_mode="policy_plus_transition_belief_stop_loop_penalty",
        q_success_weight=1.75,
    )

    assert captured["controller_config"].q_success_weight == 1.75
    assert report["metrics"]["q_success_weight"] == 1.75
    assert report["metrics"]["q_success_participates_in_action_selection"] is True


def test_load_clstr_alfworld_model_accepts_full_base_skills_jsonl(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    full_base_root = tmp_path / "full_base_train"
    full_base_root.mkdir()
    (full_base_root / "skills.jsonl").write_text(
        json.dumps(
            {
                "skill_id": "alfworld/alfworld-object-picker",
                "name": "object picker",
                "description": "pick up objects",
                "environment": "alfworld",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    def fake_build_model(routing_init_manifest, skill_rows, output_dir):
        captured["skill_rows"] = skill_rows
        return _FakeModel(), {"d": 8}, {"routing_init_manifest": str(routing_init_manifest)}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_routing_init", fake_build_model)

    model, action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=None,
        output_dir=tmp_path / "cache",
        data_root=full_base_root,
    )

    assert model is not None
    assert action_adapter is None
    assert report["skill_source_path"] == str(full_base_root / "skills.jsonl")
    assert captured["skill_rows"][0]["skill_id"] == "alfworld/alfworld-object-picker"


def test_load_clstr_alfworld_model_uses_checkpoint_skills_path_over_aux_root(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    aux_root = tmp_path / "aux_root"
    aux_root.mkdir()
    (aux_root / "pseudo_skills.jsonl").write_text(
        json.dumps({"skill_id": "aux/legacy-skill", "description": "legacy aux skill"}) + "\n",
        encoding="utf-8",
    )
    full_base_root = tmp_path / "full_base_train"
    full_base_root.mkdir()
    skills_path = full_base_root / "skills.jsonl"
    skills_path.write_text(
        "\n".join(
            [
                json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}),
                json.dumps({"skill_id": "scienceworld/heat-object", "description": "heat objects"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}, "skills_path": str(skills_path)}, checkpoint_path)
    captured = {}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = dict(state_dict)
            captured["strict"] = strict
            return None

        def eval(self):
            return self

    def fake_build_model(routing_init_manifest, skill_rows, output_dir):
        captured["skill_rows"] = skill_rows
        return _FakeModel(), {"d": 8}, {"routing_init_manifest": str(routing_init_manifest)}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_routing_init", fake_build_model)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=aux_root,
    )

    assert report["skill_source_path"] == str(skills_path)
    assert report["skill_count"] == 2
    assert [row["skill_id"] for row in captured["skill_rows"]] == [
        "alfworld/alfworld-object-picker",
        "scienceworld/heat-object",
    ]
    assert captured["strict"] is False


def test_load_clstr_alfworld_model_skill_rows_override_takes_priority(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    aux_root = tmp_path / "aux_root"
    aux_root.mkdir()
    (aux_root / "pseudo_skills.jsonl").write_text(
        json.dumps({"skill_id": "aux/legacy-skill", "description": "legacy aux skill"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_skills = tmp_path / "checkpoint_skills.jsonl"
    checkpoint_skills.write_text(
        json.dumps({"skill_id": "checkpoint/global-skill", "description": "global skill"}) + "\n",
        encoding="utf-8",
    )
    override_skills = tmp_path / "alfworld_local_skills.jsonl"
    override_skills.write_text(
        json.dumps({"skill_id": "alfworld/local-skill", "description": "local alfworld skill"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}, "skills_path": str(checkpoint_skills)}, checkpoint_path)
    captured = {}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = dict(state_dict)
            captured["strict"] = strict
            return None

        def eval(self):
            return self

    def fake_build_model(routing_init_manifest, skill_rows, output_dir):
        captured["skill_rows"] = skill_rows
        return _FakeModel(), {"d": 8}, {"routing_init_manifest": str(routing_init_manifest)}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_routing_init", fake_build_model)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=aux_root,
        skill_rows_path_override=override_skills,
    )

    assert report["skill_source_path"] == str(override_skills)
    assert report["skill_source_override"] == str(override_skills)
    assert report["skill_source_from_checkpoint"] is True
    assert report["skill_count"] == 1
    assert [row["skill_id"] for row in captured["skill_rows"]] == ["alfworld/local-skill"]
    assert captured["strict"] is False


def test_load_clstr_alfworld_model_final_chain_restores_stage0_then_appends_actions(
    tmp_path,
    monkeypatch,
):
    from clstr import alfworld_eval

    training_skills = tmp_path / "training_skills.jsonl"
    training_skills.write_text(
        json.dumps({"skill_id": "training/base", "description": "base"}) + "\n",
        encoding="utf-8",
    )
    benchmark_skills = tmp_path / "alfworld_skills.jsonl"
    benchmark_rows = [
        {
            "skill_id": "alfworld_action/go to fridge 1",
            "description": "go to fridge 1",
        }
    ]
    benchmark_skills.write_text(
        "".join(json.dumps(row) + "\n" for row in benchmark_rows),
        encoding="utf-8",
    )
    calls = []
    model = object()

    def _fake_restore(**kwargs):
        calls.append(kwargs)
        return (
            model,
            {"freeze_backbone": True, "d": 8},
            {"training/base": 0, "alfworld_action/go to fridge 1": 1},
            {
                "checkpoint_load_order": [
                    "stage0",
                    "stage2",
                    "stage4",
                    "append_benchmark_skills",
                ],
                "stage0": {"stage0_loaded": True},
                "stage2": {"loaded": True},
                "stage4": {"loaded": True},
                "skill_append": {
                    "appended_count": 1,
                    "appended_skill_ids": ["alfworld_action/go to fridge 1"],
                },
            },
        )

    monkeypatch.setattr(
        alfworld_eval,
        "restore_native_benchmark_checkpoint_chain",
        _fake_restore,
        raising=False,
    )

    restored, action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "final_chain.json",
        checkpoint_path=tmp_path / "stage2.pt",
        stage4_checkpoint_path=tmp_path / "stage4.pt",
        stage0_checkpoint_path=tmp_path / "stage0.pt",
        skill_rows_path_override=training_skills,
        benchmark_skill_rows_path=benchmark_skills,
        output_dir=tmp_path / "cache",
    )

    assert restored is model
    assert action_adapter is None
    assert len(calls) == 1
    assert calls[0]["training_skills_path"] == training_skills
    assert calls[0]["benchmark_skills"] == benchmark_rows
    assert calls[0]["require_safe_memory_delta"] is True
    assert report["final_chain_adapter"]["checkpoint_load_order"] == [
        "stage0",
        "stage2",
        "stage4",
        "append_benchmark_skills",
    ]
    assert report["skill_count"] == 2
    assert report["benchmark_skill_source_path"] == str(benchmark_skills)


def test_load_clstr_alfworld_model_uses_checkpoint_routing_skill_source_path(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    aux_root = tmp_path / "aux_root"
    aux_root.mkdir()
    full_base_root = tmp_path / "full_base_train"
    full_base_root.mkdir()
    skills_path = full_base_root / "skill_pool.jsonl"
    skills_path.write_text(
        "\n".join(
            [
                json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}),
                json.dumps({"skill_id": "traject/tool", "description": "use tool"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "legacy_controller.pt"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"d": 8, "defer_skill_table_init": True},
            "routing_init": {"skill_source_path": str(skills_path)},
        },
        checkpoint_path,
    )
    captured = {}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = dict(state_dict)
            captured["strict"] = strict
            return None

        def eval(self):
            return self

    def fake_build_model_from_checkpoint_config(config, skill_rows, output_dir):
        captured["skill_rows"] = skill_rows
        return _FakeModel(), {"d": 8}, {"checkpoint_config_init": True}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_checkpoint_config", fake_build_model_from_checkpoint_config)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=aux_root,
    )

    assert report["skill_source_path"] == str(skills_path)
    assert report["skill_count"] == 2
    assert [row["skill_id"] for row in captured["skill_rows"]] == [
        "alfworld/alfworld-object-picker",
        "traject/tool",
    ]


def test_alfworld_jsonl_reader_preserves_unicode_next_line_inside_json_strings(tmp_path):
    from clstr import alfworld_eval

    path = tmp_path / "skill_pool.jsonl"
    row = {
        "skill_id": "skill/contains-next-line-control",
        "description": "first\u0085second",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    # str.splitlines() treats U+0085 as a line separator even when it is
    # inside a valid JSON string. JSONL readers must split only on physical
    # newline bytes by iterating over the file handle.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    loaded = alfworld_eval._read_jsonl(path)

    assert loaded == [row]


def test_load_clstr_alfworld_model_can_initialize_from_checkpoint_config_without_routing_manifest(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    skills_path = tmp_path / "skill_pool.jsonl"
    skills_path.write_text(
        json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "model_state_dict": {},
            "skills_path": str(skills_path),
            "config": {
                "d": 8,
                "base_model_name": "dummy-base",
                "top_k": 1,
            },
        },
        checkpoint_path,
    )
    captured = {}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_state_dict"] = dict(state_dict)
            captured["strict"] = strict
            return None

        def eval(self):
            return self

    def fake_build_from_checkpoint_config(config, skill_rows, output_dir):
        captured["config"] = dict(config)
        captured["skill_rows"] = list(skill_rows)
        captured["output_dir"] = output_dir
        return _FakeModel(), {"d": int(config["d"])}, {"checkpoint_config_init": True}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_checkpoint_config", fake_build_from_checkpoint_config)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "missing-routing-manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=tmp_path / "unused_aux",
    )

    assert captured["config"]["base_model_name"] == "dummy-base"
    assert captured["skill_rows"][0]["skill_id"] == "alfworld/alfworld-object-picker"
    assert report["checkpoint_config_init"] is True
    assert report["skill_source_from_checkpoint"] is True
    assert report["skill_count"] == 1
    assert captured["strict"] is False


def test_load_clstr_alfworld_model_applies_stage4_overlay_after_base_checkpoint(tmp_path, monkeypatch):
    from clstr import alfworld_eval

    skills_path = tmp_path / "skill_pool.jsonl"
    skills_path.write_text(
        json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "stage2.pt"
    stage4_checkpoint_path = tmp_path / "stage4.pt"
    torch.save(
        {
            "model_state_dict": {"base.weight": torch.ones(1)},
            "skills_path": str(skills_path),
            "config": {"d": 8, "base_model_name": "dummy-base", "top_k": 1},
            "stage": "stage2",
        },
        checkpoint_path,
    )
    torch.save(
        {
            "model_state_dict": {
                "q_success_head.weight": torch.ones(1) * 2,
                "not_in_model.weight": torch.ones(1),
            },
            "stage": "stage4_overlay",
        },
        stage4_checkpoint_path,
    )
    captured = {"load_calls": []}

    class _FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def state_dict(self):
            return {
                "base.weight": torch.zeros(1),
                "q_success_head.weight": torch.zeros(1),
            }

        def load_state_dict(self, state_dict, strict=False):
            captured["load_calls"].append((sorted(state_dict), strict))
            return None

        def eval(self):
            return self

    def fake_build_from_checkpoint_config(config, skill_rows, output_dir):
        del config, skill_rows, output_dir
        return _FakeModel(), {"d": 8}, {"checkpoint_config_init": True}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_checkpoint_config", fake_build_from_checkpoint_config)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "missing-routing-manifest.json",
        checkpoint_path=checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=tmp_path / "unused_aux",
    )

    assert captured["load_calls"] == [(["base.weight"], False), (["q_success_head.weight"], False)]
    assert report["stage4_checkpoint"]["checkpoint_path"] == str(stage4_checkpoint_path)
    assert report["stage4_checkpoint"]["stage"] == "stage4_overlay"
    assert report["stage4_checkpoint"]["loaded_key_count"] == 1


def test_load_clstr_alfworld_model_does_not_rebuild_skill_table_when_checkpoint_has_embeddings(tmp_path, monkeypatch):
    from clstr import alfworld_eval
    from clstr.model import CLSTRConfig

    skills_path = tmp_path / "skill_pool.jsonl"
    skills_path.write_text(
        json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "stage2.pt"
    torch.save(
        {
            "model_state_dict": {"skill_table.E": torch.zeros(1, 4)},
            "skills_path": str(skills_path),
            "config": {
                "d": 4,
                "base_model_name": "dummy-base",
                "top_k": 1,
                "defer_skill_table_init": True,
            },
        },
        checkpoint_path,
    )
    captured = {"rebuild_count": 0}

    class _FakeModel:
        device = "cpu"
        config = CLSTRConfig(defer_skill_table_init=True, d=4)

        def to(self, device):
            self.device = device
            return self

        def state_dict(self):
            return {"skill_table.E": torch.zeros(1, 4)}

        def load_state_dict(self, state_dict, strict=False):
            captured["loaded_keys"] = sorted(state_dict)
            captured["strict"] = strict
            return None

        def rebuild_skill_table(self):
            captured["rebuild_count"] += 1

        def eval(self):
            return self

    def fake_build_from_checkpoint_config(config, skill_rows, output_dir):
        del config, skill_rows, output_dir
        return _FakeModel(), {"d": 4}, {"checkpoint_config_init": True}

    monkeypatch.setattr(alfworld_eval, "_build_model_from_checkpoint_config", fake_build_from_checkpoint_config)

    _model, _action_adapter, report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "missing-routing-manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=tmp_path / "unused_aux",
    )

    assert report["checkpoint_skill_table_embeddings_loaded"] is True
    assert captured["rebuild_count"] == 0
    assert captured["loaded_keys"] == ["skill_table.E"]
    assert captured["strict"] is False


def test_load_clstr_alfworld_model_rebuilds_deferred_qwen_skill_table(tmp_path, monkeypatch):
    from clstr import alfworld_eval
    from clstr.model import CLSTRConfig

    aux_root = tmp_path / "aux_root"
    aux_root.mkdir()
    skills_path = aux_root / "skills.jsonl"
    skills_path.write_text(
        json.dumps({"skill_id": "alfworld/alfworld-object-picker", "description": "pick objects"}) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "qwen_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {},
            "skills_path": str(skills_path),
            "qwen_external_encoder": True,
            "qwen_model_name_or_path": "models/Qwen3-8B",
            "config": {"d": 4, "max_length": 32, "torch_dtype": "bfloat16", "hf_cache_dir": ".cache/huggingface"},
        },
        checkpoint_path,
    )

    class _FakeQwenModel:
        def __init__(self):
            self.config = CLSTRConfig(defer_skill_table_init=True, d=4)
            self.device = "cpu"
            self.rebuild_seen_device = None

        def to(self, device):
            self.device = device
            return self

        def rebuild_skill_table(self):
            self.rebuild_seen_device = self.device

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict, strict=False):
            return None

        def eval(self):
            return self

    fake_model = _FakeQwenModel()

    def fake_build_qwen_external_clstr_model(*args, **kwargs):
        return fake_model, {"d": 4}, {"qwen_external_encoder": True}

    monkeypatch.setattr(alfworld_eval, "build_qwen_external_clstr_model", fake_build_qwen_external_clstr_model)

    model, _action_adapter, _report = alfworld_eval._load_clstr_alfworld_model(
        routing_init_manifest=tmp_path / "manifest.json",
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "cache",
        data_root=aux_root,
    )

    assert model is fake_model
    assert fake_model.rebuild_seen_device == fake_model.device


def test_alfworld_eval_cli_passes_aux_data_root(tmp_path, monkeypatch):
    from scripts import run_alfworld_clstr_eval

    captured = {}

    monkeypatch.setattr(
        run_alfworld_clstr_eval,
        "build_alfworld_protocol_report",
        lambda **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        run_alfworld_clstr_eval,
        "build_alfworld_env_report",
        lambda **kwargs: {"status": "ok"},
    )

    def fake_evaluate_alfworld_clstr(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.jsonl").write_text("", encoding="utf-8")
        return {
            "status": "ok",
            "metrics": {"status": "ok", "episode_count": 0},
            "result": {"status": "ok"},
        }

    monkeypatch.setattr(run_alfworld_clstr_eval, "evaluate_alfworld_clstr", fake_evaluate_alfworld_clstr)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_alfworld_clstr_eval.py",
            "eval",
            "--data_dir",
            str(tmp_path / "alfworld_data"),
            "--official_repo",
            str(tmp_path / "alfworld_repo"),
            "--output_dir",
            str(tmp_path / "out"),
            "--aux_data_root",
            str(tmp_path / "full_base_train"),
            "--skill_rows_path_override",
            str(tmp_path / "training_skills.jsonl"),
            "--stage0_checkpoint_path",
            str(tmp_path / "stage0.pt"),
            "--benchmark_skill_rows_path",
            str(tmp_path / "alfworld_skills.jsonl"),
            "--scorer_mode",
            "unified_memory_admissible_action",
            "--replay_prefix_max_steps",
            "5",
            "--controller_mode",
            "policy_plus_transition_belief_stop_loop_penalty",
            "--max_episodes",
            "1",
        ],
    )

    run_alfworld_clstr_eval.main()

    assert captured["aux_data_root"] == tmp_path / "full_base_train"
    assert captured["skill_rows_path_override"] == tmp_path / "training_skills.jsonl"
    assert captured["stage0_checkpoint_path"] == tmp_path / "stage0.pt"
    assert captured["benchmark_skill_rows_path"] == tmp_path / "alfworld_skills.jsonl"
    assert captured["scorer_mode"] == "unified_memory_admissible_action"
    assert captured["replay_prefix_max_steps"] == 5
    assert captured["controller_mode"] == "policy_plus_transition_belief_stop_loop_penalty"


def test_alfworld_eval_cli_aggregates_checkpoint_training_metadata_from_splits(tmp_path, monkeypatch):
    from scripts import run_alfworld_clstr_eval

    monkeypatch.setattr(run_alfworld_clstr_eval, "build_alfworld_protocol_report", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(run_alfworld_clstr_eval, "build_alfworld_env_report", lambda **kwargs: {"status": "ok"})

    def fake_evaluate_alfworld_clstr(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.jsonl").write_text(json.dumps({"split": kwargs["split"]}) + "\n", encoding="utf-8")
        return {
            "status": "ok",
            "metrics": {
                "status": "ok",
                "episode_count": 2,
                "success_rate": 0.25,
                "average_reward": 0.25,
                "average_goal_condition_points": 0.5,
                "average_episode_steps": 10.0,
                "training_data": "CLSTR full-base registry train_allowed rows",
                "training_objective": "component_complete_masked_multi_loss",
                "qdoc_adapter_used": False,
                "skill_source_path": "/tmp/full-base/skills.jsonl",
                "skill_count": 121,
                "controller_mode": "policy_plus_transition_belief_stop_loop_penalty",
            },
        }

    monkeypatch.setattr(run_alfworld_clstr_eval, "evaluate_alfworld_clstr", fake_evaluate_alfworld_clstr)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_alfworld_clstr_eval.py",
            "eval",
            "--data_dir",
            str(tmp_path / "alfworld_data"),
            "--official_repo",
            str(tmp_path / "alfworld_repo"),
            "--output_dir",
            str(tmp_path / "out"),
            "--checkpoint_path",
            str(tmp_path / "checkpoint.pt"),
            "--splits",
            "valid_seen",
            "valid_unseen",
        ],
    )

    run_alfworld_clstr_eval.main()

    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_data"] == "CLSTR full-base registry train_allowed rows"
    assert metrics["training_objective"] == "component_complete_masked_multi_loss"
    assert metrics["skill_source_path"] == "/tmp/full-base/skills.jsonl"
    assert metrics["skill_count"] == 121


def test_build_alfworld_comparison_table_includes_success_rate(tmp_path):
    eval_root = tmp_path / "eval"
    (eval_root / "baseline").mkdir(parents=True)
    (eval_root / "trained").mkdir(parents=True)
    (eval_root / "baseline" / "metrics.json").write_text(
        json.dumps({"method": "baseline", "success_rate": 0.1, "average_episode_steps": 10.0, "average_reward": 0.1})
        + "\n",
        encoding="utf-8",
    )
    (eval_root / "trained" / "metrics.json").write_text(
        json.dumps({"method": "trained", "success_rate": 0.2, "average_episode_steps": 8.0, "average_reward": 0.2})
        + "\n",
        encoding="utf-8",
    )

    summary = build_alfworld_comparison_table(
        eval_root=eval_root,
        run_names=["baseline", "trained"],
        output_table_path=tmp_path / "comparison.md",
        output_summary_path=tmp_path / "summary.json",
    )

    assert summary["row_count"] == 2
    table = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "success_rate" in table.lower()
    assert "training data" in table.lower()
    assert "L_policy source" in table
    assert "baseline" in table
    assert "trained" in table
    assert "ALFWorld is a closed-loop benchmark" in table


def test_build_alfworld_policy_diagnostic_report_summarizes_traces_and_policy_path(tmp_path):
    eval_root = tmp_path / "alfworld_eval"
    for method, actions in {
        "clstr_routing_init_baseline": ["look"] * 6 + ["inventory"] * 2,
        "clstr_aux_full_obs_cosine": ["go to drawer 1", "open drawer 1"] * 4,
    }.items():
        run_dir = eval_root / method / "valid_seen"
        run_dir.mkdir(parents=True)
        (run_dir / "run.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "split": "valid_seen",
                    "gamefile": "game.tw-pddl",
                    "success": False,
                    "points": 0.0,
                    "goal_condition_points": 0.0,
                    "steps": len(actions),
                    "action_trace": actions,
                    "method": method,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (eval_root / method / "metrics.json").write_text(
            json.dumps({"method": method, "success_rate": 0.0, "average_reward": 0.0, "episode_count": 1})
            + "\n",
            encoding="utf-8",
        )

    report = build_alfworld_policy_diagnostic_report(
        eval_root=eval_root,
        output_path=tmp_path / "diagnostic_report.json",
        methods=["clstr_routing_init_baseline", "clstr_aux_full_obs_cosine"],
    )

    assert report["current_policy"]["uses_action_scorer_only"] is True
    assert report["current_policy"]["transition_belief_stop_participate_in_action_selection"] is False
    assert "cannot be equivalent" in report["auxiliary_action_pool_ce_vs_l_policy"]["conclusion"]
    assert report["replay_extraction_necessity"]["required"] is True
    assert report["candidate_recall"] == {
        "candidate_recall_applicable": False,
        "candidate_recall_applicability_reason": "environment_admissible_actions",
        "candidate_recall_saturated": False,
        "pool_protocol": "environment_candidates",
        "candidate_source": "environment_admissible_actions",
    }
    baseline = report["action_trace_diagnostic"]["methods"]["clstr_routing_init_baseline"]
    assert baseline["top_actions"][0] == ["look", 6]
    assert baseline["stuck_patterns"]["consecutive_repeat_ge_5_episodes"] == 1
    assert (tmp_path / "diagnostic_report.json").exists()
