"""The causal transition, gate, and scoring heads support end-to-end gradients."""

from __future__ import annotations

import torch

from clstr.belief import BeliefGate, TransitionPredictor
from clstr.heads import TransHead


def test_causal_modules_receive_gradient_through_transition_and_gate_update():
    d = 8
    d_a = 8
    n_actions = 4
    transition = TransitionPredictor(d=d, d_a=d_a, n_actions=n_actions)
    gate = BeliefGate(d=d, mode="learned_elementwise")
    trans_head = TransHead(d=d, d_a=d_a, use_sn=False)
    # action embedding lives on TransitionPredictor; reuse it
    a = torch.tensor([0], dtype=torch.long)
    m_t = torch.zeros(1, d)
    obs = torch.randn(1, d)
    cand_emb = torch.randn(1, d_a)
    # Surrogate step_update: prediction -> observation -> gate fusion
    m_hat = transition(m_t, a, obs)
    # Use a deterministic m_obs surrogate (real model uses subspace_obs on encoder output)
    m_obs = torch.randn(1, d)
    gamma = gate(m_hat, m_obs, obs)
    m_next = gamma * m_obs + (1.0 - gamma) * m_hat
    # Action-level transition score (paper section 8) feeds into final policy
    score = trans_head(m_next, cand_emb)
    loss = score.sum()
    loss.backward()
    has_grad = lambda module: any(
        p.grad is not None and float(p.grad.detach().abs().sum().cpu().item()) > 0.0
        for p in module.parameters()
    )
    assert has_grad(transition), "TransitionPredictor received no gradient"
    assert has_grad(gate), "BeliefGate received no gradient"
    assert has_grad(trans_head), "TransHead received no gradient"
