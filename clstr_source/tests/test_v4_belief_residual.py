from __future__ import annotations

import torch
from torch import nn

from clstr.model import CLSTRConfig, CLSTRModel


class _RecordingSkillHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = None

    def forward(self, candidate_embs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self.context = context.detach().clone()
        return candidate_embs.sum(dim=-1) + context.sum(dim=-1, keepdim=True)


class _ZeroStopHead(nn.Module):
    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.size(0), 1, device=h.device)


def test_belief_residual_skill_head_context_uses_m_t_minus_h_t():
    model = CLSTRModel.__new__(CLSTRModel)
    nn.Module.__init__(model)
    model.config = CLSTRConfig(d=2, skill_head_context="belief_residual")
    model.skill_head = _RecordingSkillHead()
    model.stop_head = _ZeroStopHead()

    h_t = torch.tensor([[1.0, 10.0]])
    m_t = torch.tensor([[4.0, 12.0]])
    candidate_embs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    logits = CLSTRModel.policy_forward(model, h_t, m_t, candidate_embs)

    assert torch.allclose(model.skill_head.context, torch.tensor([[3.0, 2.0]]))
    assert torch.allclose(logits[:, :-1], torch.tensor([[6.0, 6.0]]))
