import pytest
import torch
from torch import nn

from clstr.belief import InitialBelief
from clstr.candidate_admission_residual import CandidateAdmissionResidualHead
from clstr.model import (
    CLSTRConfig,
    CLSTRModel,
    RouteMemoryUtilityGate,
    UnifiedMemoryRetriever,
)
from clstr.counterfactual_memory_calibration import RouteMemoryResidualAdapter
from clstr.memory_utility_gate import MemoryUtilityGate


class FakeEncoder:
    def __call__(self, texts):
        rows = []
        for idx, text in enumerate(texts):
            rows.append(torch.tensor([float(len(text)), float(idx + 1), 1.0]))
        return torch.stack(rows)


class FakeCrossEncoder:
    def batch_forward(self, state_texts, candidate_texts):
        batch = len(state_texts)
        k = len(candidate_texts[0])
        out = torch.zeros(batch, k, 3)
        for b in range(batch):
            for i in range(k):
                out[b, i] = torch.tensor([float(b + 1), float(i + 1), 1.0])
        return out


class FakeSkillTable:
    def __init__(self):
        self.E = torch.eye(3)

    def logits(self, h):
        return h

    def retrieval_logits(self, h):
        return self.logits(h)

    def belief_logits(self, h):
        return h


class FakeSkillHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_candidate_embs = None
        self.last_context = None

    def forward(self, u, m):
        self.last_candidate_embs = u.detach().clone()
        self.last_context = m.detach().clone()
        return u.sum(dim=-1) + m.sum(dim=-1, keepdim=True)


class FakeStopHead(nn.Module):
    def forward(self, h, m):
        return (h + m).sum(dim=-1, keepdim=True)


class IdentityUnifiedRetriever(nn.Module):
    def forward(self, h_t, m_t):
        del m_t
        return h_t


class CountingUnifiedRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)
        self.calls = 0
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(3))

    def forward(self, h_t, m_t):
        del m_t
        self.calls += 1
        return self.proj(h_t)


def test_policy_forward_uses_cross_encoded_candidates_and_appends_stop():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_head = FakeSkillHead()
    model.stop_head = FakeStopHead()
    model.action_emb = nn.Embedding(5, 2)
    model.trans_head = nn.Identity()

    h_t = torch.tensor([[1.0, 2.0, 3.0]])
    m_t = torch.tensor([[0.5, 0.5, 0.5]])
    candidate_embs = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]]]
    )

    logits = CLSTRModel.policy_forward(model, h_t, m_t, candidate_embs)

    assert logits.shape == (1, 4)
    assert torch.equal(logits[:, :-1], torch.tensor([[2.5, 2.5, 7.5]]))


def test_mt_fusion_policy_forward_uses_state_hidden_and_belief_state():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=3, mt_fusion_mode="h_plus_m")
    model.skill_head = FakeSkillHead()
    model.stop_head = FakeStopHead()
    model.mt_fusion_proj = nn.Linear(3, 3, bias=False)
    model.mt_fusion_norm = nn.Identity()
    model.mt_skill_head = nn.Linear(12, 1, bias=False)
    with torch.no_grad():
        model.mt_fusion_proj.weight.copy_(torch.eye(3))
        model.mt_skill_head.weight.zero_()
        # The head reads h_cond[0], so changing h_t should change skill logits
        # even when candidate embeddings and m_t are unchanged.
        model.mt_skill_head.weight[0, 6] = 1.0

    m_t = torch.tensor([[0.5, 0.0, 0.0]])
    candidate_embs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    low_h = torch.tensor([[1.0, 0.0, 0.0]])
    high_h = torch.tensor([[3.0, 0.0, 0.0]])

    low_logits = CLSTRModel.policy_forward(model, low_h, m_t, candidate_embs)
    high_logits = CLSTRModel.policy_forward(model, high_h, m_t, candidate_embs)

    assert torch.allclose(low_logits[:, :-1], torch.tensor([[1.5, 1.5]]))
    assert torch.allclose(high_logits[:, :-1], torch.tensor([[3.5, 3.5]]))
    assert not torch.allclose(low_logits[:, :-1], high_logits[:, :-1])


def test_gated_residual_mt_fusion_uses_belief_gate_to_condition_state_hidden():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=2, mt_fusion_mode="gated_residual")
    model.skill_head = FakeSkillHead()
    model.stop_head = FakeStopHead()
    model.mt_fusion_proj = nn.Linear(2, 2, bias=False)
    model.mt_fusion_gate = nn.Linear(4, 2)
    model.mt_fusion_norm = nn.Identity()
    model.mt_skill_head = nn.Linear(8, 1, bias=False)
    with torch.no_grad():
        model.mt_fusion_proj.weight.copy_(torch.eye(2))
        model.mt_fusion_gate.weight.zero_()
        model.mt_skill_head.weight.zero_()
        model.mt_skill_head.weight[0, 4] = 1.0

    h_t = torch.tensor([[1.0, 0.0]])
    m_t = torch.tensor([[2.0, 0.0]])
    candidate_embs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    with torch.no_grad():
        model.mt_fusion_gate.bias.fill_(-20.0)
    closed_logits = CLSTRModel.policy_forward(model, h_t, m_t, candidate_embs)

    with torch.no_grad():
        model.mt_fusion_gate.bias.fill_(20.0)
    open_logits = CLSTRModel.policy_forward(model, h_t, m_t, candidate_embs)

    assert torch.allclose(closed_logits[:, :-1], torch.tensor([[1.0, 1.0]]), atol=1.0e-4)
    assert torch.allclose(open_logits[:, :-1], torch.tensor([[3.0, 3.0]]), atol=1.0e-4)
    assert not torch.allclose(closed_logits[:, :-1], open_logits[:, :-1])


def test_prior_residual_policy_mode_anchors_skill_logits_to_routing_prior():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(
        d=3,
        mt_fusion_mode="none",
        policy_skill_mode="prior_residual",
        routing_prior_strength=1.0,
        policy_residual_scale=0.25,
    )
    model.skill_head = FakeSkillHead()
    model.stop_head = FakeStopHead()

    h_t = torch.tensor([[1.0, 2.0, 3.0]])
    m_t = torch.tensor([[0.5, 0.5, 0.5]])
    candidate_embs = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]]]
    )
    routing_logits = torch.tensor([[3.0, 1.0, -1.0]])

    logits = CLSTRModel.policy_forward(model, h_t, m_t, candidate_embs, routing_logits=routing_logits)

    expected_prior = torch.tensor([[1.2247449, 0.0, -1.2247449]])
    expected_residual = torch.tensor([[2.5, 2.5, 7.5]]) * 0.25
    assert torch.allclose(logits[:, :-1], expected_prior + expected_residual, atol=1.0e-5)


def test_batch_cross_encode_serializes_skill_candidates():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.cross_encoder = FakeCrossEncoder()
    model.skills = [
        {
            "name": "weather.lookup",
            "description": "desc",
            "input_schema": {},
            "output_schema": {},
            "executor_desc": "",
            "failure_modes": [],
        },
        {
            "name": "calendar.search",
            "description": "desc",
            "input_schema": {},
            "output_schema": {},
            "executor_desc": "",
            "failure_modes": [],
        },
    ]

    out = CLSTRModel.batch_cross_encode(
        model,
        ["query:weather"],
        [[0, 1]],
    )

    assert out.shape == (1, 2, 3)


def test_clstr_model_does_not_expose_untrained_skill_candidate_policy_interface():
    assert not hasattr(CLSTRModel, "policy_forward_from_candidates")
    assert not hasattr(CLSTRModel, "policy_candidate_embeddings")


def test_route_logits_from_candidates_uses_stage0_routing_not_policy_head():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_table = FakeSkillTable()
    model.stop_head = FakeStopHead()

    class FailingSkillHead(nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("route_logits_from_candidates must not call skill_head")

    model.skill_head = FailingSkillHead()

    def encode_states(states):
        del states
        return torch.tensor([[1.0, 9.0, 3.0]])

    model.encode_states = encode_states

    logits = CLSTRModel.route_logits_from_candidates(
        model,
        ["state"],
        torch.tensor([[0.5, 0.5, 0.5]]),
        [[2, 0]],
    )

    assert torch.equal(logits[:, :2], torch.tensor([[3.0, 1.0]]))
    assert logits.shape == (1, 3)


def test_unified_memory_retriever_returns_one_vector_per_state():
    retriever = UnifiedMemoryRetriever(d=3)
    h_t = torch.randn(2, 3)
    m_t = torch.randn(2, 3)

    z_t = retriever(h_t, m_t)

    assert z_t.shape == (2, 3)


def test_clstr_model_owns_separate_cmc_delta_modules() -> None:
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)

    CLSTRModel._init_counterfactual_memory_calibration(model, 4)

    assert isinstance(model.route_memory_residual_adapter, RouteMemoryResidualAdapter)
    assert isinstance(model.route_memory_candidate_utility_gate, MemoryUtilityGate)
    assert isinstance(
        model.route_memory_candidate_admission_residual,
        CandidateAdmissionResidualHead,
    )
    assert {
        name.split(".", 1)[0]
        for name in model.state_dict()
        if name.startswith("route_memory_")
    } == {
        "route_memory_residual_adapter",
        "route_memory_candidate_utility_gate",
        "route_memory_candidate_admission_residual",
    }


def test_unified_route_logits_can_degenerate_to_h_only_full_pool_retrieval():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_table = FakeSkillTable()
    model.unified_retriever = IdentityUnifiedRetriever()

    h_t = torch.tensor([[1.0, 3.0, 2.0]])
    m_t = torch.tensor([[9.0, 9.0, 9.0]])

    logits = CLSTRModel.unified_route_logits(model, h_t, m_t)

    assert torch.equal(logits, torch.tensor([[1.0, 3.0, 2.0]]))


def test_unified_route_logits_supports_padded_candidate_subsets():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_table = FakeSkillTable()
    model.unified_retriever = IdentityUnifiedRetriever()

    h_t = torch.tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]])
    m_t = torch.zeros_like(h_t)

    logits = CLSTRModel.unified_route_logits(model, h_t, m_t, candidate_rows=[[2, 0], [1]])

    assert logits.shape == (2, 2)
    assert torch.equal(logits[0], torch.tensor([2.0, 1.0]))
    assert logits[1, 0] == 5.0
    assert logits[1, 1] < -1.0e20


def test_unified_route_full_logits_and_gather_match_public_subset_api():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_table = FakeSkillTable()
    model.unified_retriever = CountingUnifiedRetriever()
    h_t = torch.tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]])
    m_t = torch.zeros_like(h_t)
    candidate_rows = [[2, 0], [1]]

    full_logits = CLSTRModel.unified_route_full_logits(model, h_t, m_t)
    calls_after_full = model.unified_retriever.calls
    gathered = CLSTRModel.gather_unified_route_logits(model, full_logits, candidate_rows)
    public = CLSTRModel.unified_route_logits(model, h_t, m_t, candidate_rows=candidate_rows)

    assert model.unified_retriever.calls == calls_after_full + 1
    assert torch.equal(gathered, public)
    assert gathered.shape == (2, 2)
    assert gathered[1, 1] == torch.finfo(gathered.dtype).min


def test_gather_unified_route_logits_expands_one_full_row_without_retrieval():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.unified_retriever = CountingUnifiedRetriever()
    full_logits = torch.tensor([[1.0, 2.0, 3.0]])

    gathered = CLSTRModel.gather_unified_route_logits(model, full_logits, [[2, 0], [1]])

    assert model.unified_retriever.calls == 0
    assert torch.equal(gathered[0], torch.tensor([3.0, 1.0]))
    assert gathered[1, 0] == 2.0
    assert gathered[1, 1] == torch.finfo(gathered.dtype).min


def test_gather_unified_route_logits_rejects_out_of_range_skill_indices():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)

    with pytest.raises(ValueError, match="outside full_logits"):
        CLSTRModel.gather_unified_route_logits(model, torch.ones(1, 3), [[3]])


def test_gathered_unified_route_logits_backpropagate_only_selected_entries():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.skill_table = FakeSkillTable()
    model.unified_retriever = CountingUnifiedRetriever()
    h_t = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    m_t = torch.zeros_like(h_t)

    full_logits = CLSTRModel.unified_route_full_logits(model, h_t, m_t)
    gathered = CLSTRModel.gather_unified_route_logits(model, full_logits, [[2]])
    gathered.sum().backward()

    assert model.unified_retriever.proj.weight.grad is not None
    assert torch.equal(h_t.grad, torch.tensor([[0.0, 0.0, 1.0]]))


def test_initial_belief_uses_configured_sparse_topk_for_m0():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=3, initial_belief_top_k=1)
    model.skill_table = FakeSkillTable()
    model.initial_belief_head = InitialBelief(3)
    model.initial_belief_head.norm = nn.Identity()
    with torch.no_grad():
        model.initial_belief_head.state_proj.weight.copy_(torch.eye(3))
        model.initial_belief_head.belief_proj.weight.copy_(torch.eye(3))

    h_t = torch.tensor([[0.0, 5.0, 1.0]])

    m_0 = CLSTRModel.initial_belief(model, h_t)

    assert torch.equal(m_0, torch.tensor([[0.0, 6.0, 1.0]]))


def test_appworld_skillrouter_text_formats_match_skillrouter_baseline_templates():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(
        skill_text_format="appworld_skillrouter_base",
        state_text_format="appworld_skillrouter_query",
    )
    model.skills = [
        {
            "skill_id": "skillx/appworld/spotify-song",
            "name": "spotify inspect song",
            "description": "Inspect Spotify songs.",
            "executor_desc": "apis.spotify.show_song",
            "body": "Call show_song.",
        }
    ]

    assert model.candidate_texts([[0]]) == [["spotify inspect song | Inspect Spotify songs. | apis.spotify.show_song | Call show_song."]]
    assert model._serialize_state_for_encoder("Find Spotify songs.").startswith(
        "Instruct: Given a task description, retrieve the most relevant skill document"
    )
    assert model._serialize_state_for_encoder("Find Spotify songs.").endswith("Query:Find Spotify songs.")


def test_model_prompts_states_but_not_transition_observations():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(
        d=3,
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
    )

    class CaptureEncoder:
        def __init__(self):
            self.calls = []

        def __call__(self, texts):
            self.calls.append(list(texts))
            return torch.zeros(len(texts), 3)

    model.encoder = CaptureEncoder()

    model.encode_states(["goal: find weather"])
    model.encode_observations(["weather result"])

    assert model.encoder.calls[0][0].startswith("Instruct:")
    assert model.encoder.calls[0][0].endswith("Query:goal: find weather")
    assert model.encoder.calls[1] == ["weather result"]


def test_route_memory_utility_gate_starts_near_static_and_is_causal() -> None:
    torch.manual_seed(7)
    gate = RouteMemoryUtilityGate(3, initial_alpha=0.01)
    h_t = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    static_memory = torch.zeros_like(h_t)
    dynamic_memory = torch.tensor([[0.5, -0.5, 1.0], [2.0, 0.0, -1.0]])

    alpha = gate(h_t, static_memory, dynamic_memory)

    assert alpha.shape == (2,)
    assert bool(((alpha >= 0.0) & (alpha <= 1.0)).all())
    assert float(alpha.max().detach()) < 0.02
    assert not torch.equal(alpha[0], alpha[1])


def test_route_memory_alpha_forces_zero_without_causal_updates_and_backpropagates() -> None:
    torch.manual_seed(11)
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.route_memory_utility_gate = RouteMemoryUtilityGate(2, initial_alpha=0.01)
    h_t = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    static_memory = torch.zeros_like(h_t)
    dynamic_memory = torch.tensor([[1.0, -1.0], [-1.0, 2.0]], requires_grad=True)

    alpha = CLSTRModel.route_memory_alpha(
        model,
        h_t,
        static_memory,
        dynamic_memory,
        torch.tensor([0.0, 2.0]),
    )

    assert alpha[0].item() == 0.0
    assert 0.0 < alpha[1].item() < 0.02
    alpha[1].backward()
    assert dynamic_memory.grad is not None
    assert float(dynamic_memory.grad[1].abs().sum()) > 0.0
    assert h_t.grad is not None
    assert float(h_t.grad[1].abs().sum()) > 0.0
    assert any(parameter.grad is not None for parameter in model.route_memory_utility_gate.parameters())
