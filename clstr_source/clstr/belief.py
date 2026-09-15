from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_GATE_MODES = {
    "learned_elementwise",
    "bayes_scalar",
    "bayes_diag",
    "fixed_half",
}

DEFAULT_BELIEF_TOP_K = None


def resolve_belief_top_k(skill_table, skill_count: int, top_k: int | None = None) -> int:
    raw = top_k
    if raw is None:
        raw = getattr(skill_table, "belief_top_k", DEFAULT_BELIEF_TOP_K)
    if raw is None:
        return int(skill_count)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_BELIEF_TOP_K
    if value <= 0 or value >= int(skill_count):
        return int(skill_count)
    return max(1, value)


def subspace_obs(
    skill_table,
    h: torch.Tensor,
    top_k: int | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute belief state m_t via sparse softmax over belief logits.

    Uses skill_table.belief_logits(h) which applies learnable logit_scale_belief
    and skill_bias_belief (F2-v2). The old fixed tau_s parameter has been removed.
    """
    logits = skill_table.belief_logits(h)
    skill_count = int(logits.size(-1))
    if valid_mask is not None:
        if valid_mask.shape != logits.shape:
            raise ValueError("belief valid_mask must match belief logits")
        mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        if bool((~mask.any(dim=-1)).any().item()):
            raise ValueError("every belief row must contain at least one runtime-visible skill")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    k = resolve_belief_top_k(skill_table, skill_count, top_k=top_k)
    E = skill_table.E.to(device=logits.device, dtype=logits.dtype)
    if k >= skill_count:
        probs = F.softmax(logits, dim=-1)
        return probs @ E
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_values, dim=-1)
    selected = E.index_select(0, top_indices.reshape(-1)).view(logits.size(0), k, -1)
    return torch.bmm(probs.unsqueeze(1), selected).squeeze(1)


class InitialBelief(nn.Module):
    """Initialize recurrent belief from current state and sparse skill belief."""

    def __init__(self, d: int):
        super().__init__()
        self.state_proj = nn.Linear(d, d, bias=False)
        self.belief_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        with torch.no_grad():
            self.state_proj.weight.copy_(torch.eye(d))
            self.belief_proj.weight.copy_(torch.eye(d))

    def forward(self, h_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return self.norm(self.state_proj(h_t) + self.belief_proj(b_t))


class TransitionPredictor(nn.Module):
    def __init__(self, d: int, d_a: int, n_actions: int | None):
        super().__init__()
        if n_actions is not None:
            self.action_emb = nn.Embedding(n_actions, d_a)
        self.obs_proj = nn.Linear(d, d)
        self.cell = nn.GRUCell(d_a + d, d)

    def forward(
        self, m_t: torch.Tensor, a_t: torch.Tensor, o_t_emb: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self, "action_emb") and a_t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            action_emb = self.action_emb(a_t)
        else:
            action_emb = a_t
        inp = torch.cat([action_emb, self.obs_proj(o_t_emb)], dim=-1)
        return self.cell(inp, m_t)


class BeliefGate(nn.Module):
    def __init__(self, d: int, mode: str = "learned_elementwise"):
        super().__init__()
        if mode not in VALID_GATE_MODES:
            raise ValueError(f"unsupported belief gate mode: {mode}")
        self.mode = mode
        self.linear = nn.Linear(d * 3, d)
        self.log_sigma_t = nn.Parameter(torch.zeros(1))
        self.log_sigma_o = nn.Parameter(torch.zeros(1))
        if mode == "bayes_diag":
            self.log_sigma_t = nn.Parameter(torch.zeros(d))
            self.log_sigma_o = nn.Parameter(torch.zeros(d))

    def forward(
        self, m_hat: torch.Tensor, m_obs: torch.Tensor, obs_emb: torch.Tensor
    ) -> torch.Tensor:
        if self.mode == "learned_elementwise":
            return torch.sigmoid(self.linear(torch.cat([m_hat, m_obs, obs_emb], dim=-1)))
        if self.mode == "fixed_half":
            return torch.full_like(m_hat, 0.5)
        sigma_t2 = self.log_sigma_t.exp().pow(2)
        sigma_o2 = self.log_sigma_o.exp().pow(2)
        gamma = sigma_t2 / (sigma_t2 + sigma_o2)
        if self.mode == "bayes_scalar":
            return gamma.view(1, 1).expand(m_hat.size(0), 1)
        return gamma.view(1, -1).expand(m_hat.size(0), m_hat.size(1))
