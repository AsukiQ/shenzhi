from __future__ import annotations

from contextlib import nullcontext
import hashlib
import threading

import pytest
import torch

from clstr.matched_history_online import (
    ControlledMatchedHistorySelector,
    ControlledMatchedHistorySession,
    canonical_tau2_action_text,
)


class _FakeStaticQuery:
    @staticmethod
    def temperature() -> torch.Tensor:
        return torch.ones(())

    @staticmethod
    def static_query_components(
        state: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del memory
        return state, torch.zeros_like(state), state


class _FakeVNext:
    def __init__(self, d: int) -> None:
        self.d = int(d)
        self.static_query = _FakeStaticQuery()

    def static_query_components(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.static_query.static_query_components(state, memory)


class _FakeFoundation:
    def __init__(self, skill_embeddings: torch.Tensor) -> None:
        self._skills = torch.nn.functional.normalize(skill_embeddings.float(), dim=-1)
        self.vnext = _FakeVNext(int(skill_embeddings.size(1)))

    def _encode(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(list(digest[: self.vnext.d]), dtype=torch.float32)
            rows.append((values - 127.5) / 127.5)
        return torch.stack(rows)

    def encode_states(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    def encode_observations(self, texts: list[str]) -> torch.Tensor:
        return self._encode(texts)

    @staticmethod
    def vnext_initial_belief(
        state: torch.Tensor,
        legal: torch.Tensor,
        *,
        top_k: int,
    ) -> torch.Tensor:
        del legal, top_k
        return state

    def vnext_normalized_skill_embeddings(self, *, dtype: torch.dtype) -> torch.Tensor:
        return self._skills.to(dtype=dtype)

    def vnext_full_pool_logits(self, query: torch.Tensor, *, head: str) -> torch.Tensor:
        assert head == "recall"
        return torch.einsum("bd,sd->bs", query, self._skills.to(query.dtype))


class _CapturingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_time = -1
        self.last_result_mask: torch.Tensor | None = None

    def forward(
        self,
        initial_memory: torch.Tensor,
        event_states: torch.Tensor,
        event_skills: torch.Tensor,
        event_actions: torch.Tensor,
        event_results: torch.Tensor,
        event_mask: torch.Tensor,
        result_mask: torch.Tensor,
        *,
        serialized_history_embedding=None,
    ) -> torch.Tensor:
        del event_states, event_skills, event_results, serialized_history_embedding
        self.last_time = int(event_mask.size(1))
        self.last_result_mask = result_mask.detach().clone()
        return initial_memory + event_actions.sum(dim=1)


class _FakeScorer(torch.nn.Module):
    @staticmethod
    def forward(
        base_query: torch.Tensor,
        current_state: torch.Tensor,
        history_memory: torch.Tensor,
        reference_memory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        del current_state
        candidate = base_query + 0.1 * (history_memory - reference_memory)
        return torch.where(history_mask.unsqueeze(-1), candidate, base_query)


def _selector(method: str) -> ControlledMatchedHistorySelector:
    torch.manual_seed(7)
    skill_rows = [
        {"skill_id": f"tau2/test/tool{index}"}
        for index in range(12)
    ]
    selector = ControlledMatchedHistorySelector.__new__(ControlledMatchedHistorySelector)
    selector.foundation = _FakeFoundation(torch.randn(12, 4))
    selector.skills = skill_rows
    selector.skill_ids = [row["skill_id"] for row in skill_rows]
    selector.skill_index = {value: index for index, value in enumerate(selector.skill_ids)}
    selector.skill_id_to_idx = selector.skill_index
    selector.device = torch.device("cpu")
    selector.encoder_kind = method
    selector.max_horizon = 16
    selector.support_k = 500
    selector.belief_top_k = 64
    selector.method = f"matched_history_e3_{method}"
    selector._model_lock = threading.RLock()
    selector._autocast = nullcontext
    selector.encoder = None if method == "static" else _CapturingEncoder()
    selector.scorer = None if method == "static" else _FakeScorer()
    return selector


def test_tau2_action_is_canonicalized_to_the_e1_channel() -> None:
    assert canonical_tau2_action_text(
        "tau2/airline/get_user_details",
        'get_user_details({"user_id": "abc"})',
    ) == 'tool: get_user_details arguments: {"user_id": "abc"}'
    assert canonical_tau2_action_text(
        "tau2/airline/get_user_details",
        'tool: get_user_details arguments: {"user_id": "abc"}',
    ) == 'tool: get_user_details arguments: {"user_id": "abc"}'
    with pytest.raises(ValueError, match="does not match"):
        canonical_tau2_action_text(
            "tau2/airline/get_user_details",
            'other_tool({"user_id": "abc"})',
        )


def test_zero_history_is_exactly_static_and_top8() -> None:
    static_session = ControlledMatchedHistorySession(_selector("static"))
    transformer_session = ControlledMatchedHistorySession(_selector("transformer"))
    candidates = static_session.selector.skill_ids
    static = static_session.select(
        "state zero",
        candidate_skill_ids=candidates,
        top_k=8,
        route_mode="static",
    )
    transformer = transformer_session.select(
        "state zero",
        candidate_skill_ids=candidates,
        top_k=8,
        route_mode="dynamic",
    )
    assert len(static) == len(transformer) == 8
    assert [row["skill_id"] for row in static] == [
        row["skill_id"] for row in transformer
    ]
    assert [row["score"] for row in static] == [row["score"] for row in transformer]
    assert static[0]["static_support_sha256"] == transformer[0][
        "static_support_sha256"
    ]
    assert transformer[0]["selected_expert"] == "static"


def test_history_is_causal_bounded_and_support_preserving() -> None:
    selector = _selector("transformer")
    session = ControlledMatchedHistorySession(selector)
    candidates = selector.skill_ids
    support_hashes = set()
    for index in range(18):
        state = f"state {index}"
        selected = session.select(
            state,
            candidate_skill_ids=candidates,
            top_k=8,
            route_mode="dynamic",
        )
        support_hashes.add(selected[0]["static_support_sha256"])
        skill_id = selected[0]["skill_id"]
        tool_name = skill_id.rsplit("/", 1)[-1]
        session.observe(
            state_text_before=state,
            skill_id=skill_id,
            action_text=f'{tool_name}({{"index": {index}}})',
            result_text=f"result-{index}",
        )
    selected = session.select(
        "state 18",
        candidate_skill_ids=candidates,
        top_k=8,
        route_mode="dynamic",
    )
    encoder = selector.encoder
    assert isinstance(encoder, _CapturingEncoder)
    assert session.history_depth == 18
    assert encoder.last_time == 16
    assert encoder.last_result_mask is not None
    assert bool(encoder.last_result_mask.all().item())
    assert selected[0]["history_window_depth"] == 16
    assert selected[0]["static_support_count"] == len(candidates)
    assert len(support_hashes) >= 1


def test_observe_requires_its_immediately_preceding_selection() -> None:
    selector = _selector("transformer")
    session = ControlledMatchedHistorySession(selector)
    with pytest.raises(RuntimeError, match="preceding selection"):
        session.observe(
            state_text_before="state",
            skill_id=selector.skill_ids[0],
            action_text='tool0({"x": 1})',
            result_text="ok",
        )
    session.select(
        "state",
        candidate_skill_ids=selector.skill_ids,
        top_k=8,
        route_mode="dynamic",
    )
    with pytest.raises(ValueError, match="state differs"):
        session.observe(
            state_text_before="future state",
            skill_id=selector.skill_ids[0],
            action_text='tool0({"x": 1})',
            result_text="ok",
        )


def test_one_selection_can_align_multiple_tool_results() -> None:
    selector = _selector("transformer")
    session = ControlledMatchedHistorySession(selector)
    session.select(
        "shared pre-action state",
        candidate_skill_ids=selector.skill_ids,
        top_k=8,
        route_mode="dynamic",
    )
    for skill_id in selector.skill_ids[:2]:
        tool_name = skill_id.rsplit("/", 1)[-1]
        session.observe(
            state_text_before="shared pre-action state",
            skill_id=skill_id,
            action_text=f'{tool_name}({{"x": 1}})',
            result_text=f"result-{tool_name}",
        )
    assert session.history_depth == 2
    assert [event["state_text"] for event in session.events] == [
        "shared pre-action state",
        "shared pre-action state",
    ]
