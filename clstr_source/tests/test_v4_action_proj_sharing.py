from __future__ import annotations

import torch
from torch import nn

from clstr.belief import TransitionPredictor
from clstr.full_base_train import _freeze_for_full_base
from clstr.full_base_train import _transition_prediction as full_base_transition_prediction
from clstr.model import CLSTRConfig, CLSTRModel


def test_transition_predictor_accepts_projected_action_embeddings():
    predictor = TransitionPredictor(d=4, d_a=4, n_actions=None)
    m_t = torch.zeros(2, 4)
    action_emb = torch.randn(2, 4)
    obs_emb = torch.ones(2, 4)

    out = predictor(m_t, action_emb, obs_emb)

    assert out.shape == (2, 4)
    assert not hasattr(predictor, "action_emb")


def test_model_action_embeddings_are_soft_shared_from_skill_table_E():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=4, d_a=4)
    model.skill_table = nn.Module()
    model.skill_table.E = nn.Parameter(torch.eye(3, 4))
    model.action_proj = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.action_proj.weight.copy_(2.0 * torch.eye(4))

    action_ids = torch.tensor([0, 2], dtype=torch.long)
    action_emb = CLSTRModel.action_embeddings(model, action_ids)

    assert torch.allclose(action_emb, torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]))

    loss = action_emb.sum()
    loss.backward()
    assert model.action_proj.weight.grad is not None
    assert model.skill_table.E.grad is not None


def test_full_base_transition_prediction_uses_soft_shared_action_embeddings():
    class _RecordingTransition(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_action_emb = None

        def forward(self, m_obs, action_emb, obs_emb):
            self.seen_action_emb = action_emb.detach().clone()
            return m_obs + action_emb + obs_emb

    model = nn.Module()
    model.skill_table = nn.Module()
    model.skill_table.E = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    model.action_proj = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.action_proj.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 5.0]]))
    model.action_embeddings = CLSTRModel.action_embeddings.__get__(model, nn.Module)
    model.transition = _RecordingTransition()

    labels = torch.tensor([1], dtype=torch.long)
    fallback = torch.tensor([[9.0, 9.0]])
    out = full_base_transition_prediction(
        model,
        h=torch.zeros(1, 2),
        m_obs=torch.ones(1, 2),
        labels=labels,
        obs_emb=fallback,
    )

    assert torch.allclose(model.transition.seen_action_emb, torch.tensor([[0.0, 10.0]]))
    assert torch.allclose(out, torch.tensor([[10.0, 20.0]]))


def test_full_base_freeze_marks_action_proj_trainable():
    model = nn.Module()
    model.transition = nn.Linear(2, 2)
    model.gate = nn.Linear(2, 2)
    model.stop_head = nn.Linear(2, 1)
    model.skill_head = nn.Linear(2, 1)
    model.trans_head = nn.Linear(2, 1)
    model.action_proj = nn.Linear(2, 2)
    model.q_success_head = nn.Linear(2, 1)

    report = _freeze_for_full_base(model)

    assert "action_proj" in report["trainable_modules"]
    assert any(param.requires_grad for param in model.action_proj.parameters())
