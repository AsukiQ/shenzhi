import torch

from clstr.belief import BeliefGate, InitialBelief, TransitionPredictor, subspace_obs


def test_subspace_obs_shape():
    class DummyTable:
        def __init__(self):
            self.E = torch.eye(2)

        def belief_logits(self, h):
            return h

    obs = subspace_obs(DummyTable(), torch.tensor([[2.0, 0.0]]))
    assert obs.shape == (1, 2)


def test_subspace_obs_uses_sparse_topk_belief_when_configured():
    class DummyTable:
        def __init__(self):
            self.E = torch.eye(5)
            self.belief_top_k = 2

        def belief_logits(self, h):
            del h
            return torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])

    obs = subspace_obs(DummyTable(), torch.zeros(1, 5))
    expected = torch.tensor([[0.7310586, 0.2689414, 0.0, 0.0, 0.0]])

    assert torch.allclose(obs, expected, atol=1.0e-6)


def test_subspace_obs_defaults_to_full_support():
    class DummyTable:
        def __init__(self):
            self.E = torch.eye(128)

        def belief_logits(self, h):
            del h
            values = torch.linspace(1.0, -1.0, steps=128)
            return values.unsqueeze(0)

    obs = subspace_obs(DummyTable(), torch.zeros(1, 128))
    expected = torch.softmax(torch.linspace(1.0, -1.0, steps=128), dim=0).unsqueeze(0)

    assert torch.allclose(obs, expected, atol=1.0e-6)


def test_initial_belief_fuses_state_and_sparse_belief_vectors():
    init = InitialBelief(d=3)
    init.norm = torch.nn.Identity()
    with torch.no_grad():
        init.state_proj.weight.copy_(torch.eye(3))
        init.belief_proj.weight.copy_(torch.eye(3))

    h_t = torch.tensor([[2.0, 0.0, 1.0]])
    b_t = torch.tensor([[0.0, 3.0, 1.0]])

    m_0 = init(h_t, b_t)

    assert torch.equal(m_0, torch.tensor([[2.0, 3.0, 2.0]]))


def test_bayes_scalar_gate_is_in_unit_interval():
    gate = BeliefGate(d=4, mode="bayes_scalar")
    gamma = gate(torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 4))
    assert gamma.shape == (1, 1)
    assert torch.all((gamma >= 0.0) & (gamma <= 1.0))


def test_transition_predictor_output_shape():
    predictor = TransitionPredictor(d=4, d_a=3, n_actions=5)
    out = predictor(
        m_t=torch.zeros(2, 4),
        a_t=torch.tensor([0, 3]),
        o_t_emb=torch.ones(2, 4),
    )
    assert out.shape == (2, 4)


def test_bayes_diag_gate_matches_hidden_dimension():
    gate = BeliefGate(d=4, mode="bayes_diag")
    gamma = gate(torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 4))
    assert gamma.shape == (2, 4)
    assert torch.all((gamma >= 0.0) & (gamma <= 1.0))


def test_fixed_half_gate_returns_half_tensor():
    gate = BeliefGate(d=4, mode="fixed_half")
    gamma = gate(torch.zeros(1, 4), torch.ones(1, 4), torch.zeros(1, 4))
    assert torch.allclose(gamma, torch.full((1, 4), 0.5))


def test_belief_gate_rejects_unknown_mode():
    try:
        BeliefGate(d=4, mode="unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown gate mode")
