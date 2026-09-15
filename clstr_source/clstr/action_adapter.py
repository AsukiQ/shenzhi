from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalActionAdapter(nn.Module):
    """Scores free-form action candidates against a state embedding."""

    loss_taxonomy = "L_act_pretraining_proxy"

    def __init__(self, d: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or d)
        self.scorer = nn.Sequential(
            nn.Linear(d * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state_embs: torch.Tensor,
        action_embs: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_embs.ndim != 2:
            raise ValueError("state_embs must have shape [batch, d]")
        if action_embs.ndim != 3:
            raise ValueError("action_embs must have shape [batch, candidates, d]")
        if state_embs.size(0) != action_embs.size(0) or state_embs.size(1) != action_embs.size(2):
            raise ValueError("state/action embedding shapes are incompatible")
        state = state_embs.unsqueeze(1).expand(-1, action_embs.size(1), -1)
        features = torch.cat([state, action_embs, state * action_embs], dim=-1)
        scores = self.scorer(features).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.to(torch.bool), torch.finfo(scores.dtype).min)
        return scores

    @staticmethod
    def ranking_loss(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, candidates]")
        labels = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
        return F.cross_entropy(scores, labels)
