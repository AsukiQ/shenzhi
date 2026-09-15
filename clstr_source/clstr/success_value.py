from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class QSuccessHead(nn.Module):
    """Estimate action-level eventual success logits for admissible candidates."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, m: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 2:
            return self.net(torch.cat([h, m, u, u * m], dim=-1)).squeeze(-1)
        h_exp = h.unsqueeze(1).expand(-1, u.size(1), -1)
        m_exp = m.unsqueeze(1).expand(-1, u.size(1), -1)
        return self.net(torch.cat([h_exp, m_exp, u, u * m_exp], dim=-1)).squeeze(-1)


def q_success_scores(
    head: QSuccessHead,
    h: torch.Tensor,
    m: torch.Tensor,
    candidate_embs: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = head(h, m, candidate_embs)
    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask.to(scores.device).to(torch.bool), torch.finfo(scores.dtype).min)
    return scores


def compute_q_success_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if logits.shape != labels.shape:
        raise ValueError("q_success logits and labels must have the same shape")
    mask = torch.ones_like(labels, dtype=torch.bool) if candidate_mask is None else candidate_mask.to(labels.device).to(torch.bool)
    if not bool(mask.any()):
        loss = logits.sum() * 0.0
        return loss, {"q_success_bce_loss": 0.0, "q_success_sample_count": 0}
    values = logits[mask].float()
    targets = labels.to(logits.device).float()[mask]
    weight_values = torch.ones_like(targets) if weights is None else weights.to(logits.device).float()[mask]
    raw = F.binary_cross_entropy_with_logits(values, targets, reduction="none")
    loss = (raw * weight_values).sum() / weight_values.sum().clamp_min(1.0e-6)
    with torch.no_grad():
        probs = torch.sigmoid(values)
        preds = probs >= 0.5
        binary_targets = targets >= 0.5
        acc = (preds == binary_targets).float().mean() if targets.numel() else torch.tensor(0.0)
    return loss, {
        "q_success_bce_loss": float(loss.detach().cpu().item()),
        "q_success_sample_count": int(targets.numel()),
        "q_success_accuracy@0.5": float(acc.detach().cpu().item()),
    }
