from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SkillHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if u.ndim == 2:
            return self.net(torch.cat([u, m, u * m], dim=-1))
        m_exp = m.unsqueeze(1).expand_as(u)
        feat = torch.cat([u, m_exp, u * m_exp], dim=-1)
        return self.net(feat).squeeze(-1)


class StopHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, m], dim=-1))


class TransHead(nn.Module):
    def __init__(self, d: int, d_a: int | None = None, use_sn: bool = True):
        super().__init__()
        self.d_a = d if d_a is None else d_a
        linear_1 = nn.Linear(d + self.d_a, d)
        linear_2 = nn.Linear(d, 1)
        if use_sn:
            linear_1 = spectral_norm(linear_1)
            linear_2 = spectral_norm(linear_2)
        self.net = nn.Sequential(linear_1, nn.GELU(), linear_2)

    def forward(self, m_hat: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        if candidate_emb.ndim == 2:
            return self.net(torch.cat([m_hat, candidate_emb], dim=-1))
        m_exp = m_hat.unsqueeze(1).expand(-1, candidate_emb.size(1), -1)
        return self.net(torch.cat([m_exp, candidate_emb], dim=-1)).squeeze(-1)
