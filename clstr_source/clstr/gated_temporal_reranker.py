from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatedTemporalConfig:
    dim: int
    query_dim: int | None = None
    lambda_max: float = 0.5
    context_top_k: int = 64
    gate_feature_dim: int = 4
    hidden_dim: int | None = None


@dataclass
class GatedTemporalOutput:
    final_logits: torch.Tensor
    prior_logits: torch.Tensor
    residual_logits: torch.Tensor
    lambda_t: torch.Tensor
    candidate_context: torch.Tensor
    gate_features: dict[str, torch.Tensor]


def _masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is None:
        return logits
    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    masked = _masked_logits(logits, valid_mask)
    probs = torch.softmax(masked, dim=-1)
    if valid_mask is None:
        return probs
    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    return torch.where(mask, probs, torch.zeros_like(probs))


def _safe_log_softmax(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    masked = _masked_logits(logits, valid_mask)
    log_probs = torch.log_softmax(masked, dim=-1)
    if valid_mask is None:
        return log_probs
    mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    return torch.where(mask, log_probs, torch.zeros_like(log_probs))


def _best_positive_rank(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if valid_mask is not None:
        mask = mask & valid_mask.to(device=logits.device, dtype=torch.bool)
    order = torch.argsort(_masked_logits(logits, valid_mask), dim=-1, descending=True)
    sorted_positive = mask.gather(1, order)
    ranks = torch.arange(1, logits.size(1) + 1, device=logits.device, dtype=logits.dtype).unsqueeze(0)
    missing_rank = torch.full((logits.size(0),), float(logits.size(1) + 1), device=logits.device, dtype=logits.dtype)
    ranked = torch.where(sorted_positive, ranks.expand_as(sorted_positive).to(logits.dtype), torch.full_like(ranks.expand_as(sorted_positive), float(logits.size(1) + 1)))
    best_rank = ranked.min(dim=-1).values
    has_positive = mask.any(dim=-1)
    return torch.where(has_positive, best_rank, missing_rank)


def prior_preserving_loss(
    *,
    final_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    rank_drop_k: int = 5,
    rank_drop_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    valid = valid_mask.to(device=final_logits.device, dtype=torch.bool) if valid_mask is not None else torch.ones_like(final_logits, dtype=torch.bool)
    positives = positive_mask.to(device=final_logits.device, dtype=torch.bool) & valid
    final_log_probs = _safe_log_softmax(final_logits, valid)
    prior_log_probs = _safe_log_softmax(prior_logits, valid)
    final_probs = torch.where(valid, final_log_probs.exp(), torch.zeros_like(final_logits))
    kl_per_row = (final_probs * (final_log_probs - prior_log_probs)).sum(dim=-1)
    valid_rows = valid.any(dim=-1)
    kl_loss = kl_per_row[valid_rows].mean() if valid_rows.any() else final_logits.sum() * 0.0

    prior_rank = _best_positive_rank(prior_logits.detach(), positives, valid)
    applies = (prior_rank <= max(1, int(rank_drop_k))) & positives.any(dim=-1)
    if applies.any():
        masked_final = _masked_logits(final_logits, valid)
        k = min(max(1, int(rank_drop_k)), final_logits.size(1))
        kth_scores = torch.topk(masked_final, k=k, dim=-1).values[:, -1]
        positive_scores = final_logits.masked_fill(~positives, torch.finfo(final_logits.dtype).min).max(dim=-1).values
        rank_drop = F.relu(kth_scores + float(rank_drop_margin) - positive_scores)
        rank_drop_loss = rank_drop[applies].mean()
    else:
        rank_drop_loss = final_logits.sum() * 0.0

    return {
        "kl_loss": kl_loss,
        "rank_drop_loss": rank_drop_loss,
    }


class GatedTemporalReranker(nn.Module):
    def __init__(self, config: GatedTemporalConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim or max(config.dim, 32))
        query_dim = int(config.query_dim or config.dim)
        self.query_proj = nn.Identity() if query_dim == int(config.dim) else nn.Linear(query_dim, config.dim)
        self.gate = nn.Linear(int(config.gate_feature_dim), 1)
        self.residual_adapter = nn.Sequential(
            nn.Linear(config.dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if isinstance(self.query_proj, nn.Linear):
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            # Higher Stage0 margin should trust the prior more; higher entropy
            # should allow more temporal correction.
            self.gate.weight[0, 0] = -1.0
            self.gate.weight[0, 1] = 1.0

    def gate_lambda(
        self,
        prior_logits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        current_labels: torch.Tensor | None = None,
        candidate_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masked = _masked_logits(prior_logits, valid_mask)
        width = int(masked.size(-1))
        if width <= 1:
            margin = torch.zeros(masked.size(0), device=masked.device, dtype=masked.dtype)
        else:
            top2 = torch.topk(masked, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        probs = _masked_softmax(prior_logits, valid_mask)
        entropy = -(probs * torch.where(probs > 0, probs.log(), torch.zeros_like(probs))).sum(dim=-1)
        if valid_mask is None:
            denom = torch.full_like(entropy, float(max(width, 1))).log().clamp_min(1.0e-6)
        else:
            valid_count = valid_mask.to(device=prior_logits.device, dtype=prior_logits.dtype).sum(dim=-1).clamp_min(1.0)
            denom = valid_count.log().clamp_min(1.0e-6)
        normalized_entropy = (entropy / denom).clamp(min=0.0, max=1.0)

        current_present = torch.zeros_like(margin)
        top1_is_current = torch.zeros_like(margin)
        if current_labels is not None and candidate_ids is not None:
            ids = candidate_ids.to(device=prior_logits.device)
            labels = current_labels.to(device=prior_logits.device).view(-1, 1)
            matches = ids == labels
            if valid_mask is not None:
                matches = matches & valid_mask.to(device=prior_logits.device, dtype=torch.bool)
            current_present = matches.any(dim=-1).to(dtype=prior_logits.dtype)
            top1 = torch.argmax(masked, dim=-1, keepdim=True)
            top1_is_current = matches.gather(1, top1).squeeze(1).to(dtype=prior_logits.dtype)

        feature_tensor = torch.stack([margin, normalized_entropy, current_present, top1_is_current], dim=-1)
        lambda_t = torch.sigmoid(self.gate(feature_tensor)) * float(self.config.lambda_max)
        return lambda_t, {
            "stage0_margin": margin.detach(),
            "stage0_entropy": normalized_entropy.detach(),
            "current_skill_present": current_present.detach(),
            "stage0_top1_is_current": top1_is_current.detach(),
        }

    def candidate_context_pool(
        self,
        query: torch.Tensor,
        candidate_embs: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = max(float(query.size(-1)), 1.0) ** 0.5
        scores = (candidate_embs * query.unsqueeze(1)).sum(dim=-1) / scale
        valid = valid_mask.to(device=scores.device, dtype=torch.bool) if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        k = min(max(1, int(self.config.context_top_k)), int(scores.size(-1)))
        top_scores, top_idx = torch.topk(masked_scores, k=k, dim=-1)
        gathered_mask = valid.gather(1, top_idx)
        top_weights = torch.softmax(top_scores, dim=-1)
        top_weights = torch.where(gathered_mask, top_weights, torch.zeros_like(top_weights))
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, candidate_embs.size(-1))
        top_embs = candidate_embs.gather(1, gather_idx)
        return (top_weights.unsqueeze(-1) * top_embs).sum(dim=1)

    def forward(
        self,
        query: torch.Tensor,
        candidate_embs: torch.Tensor,
        prior_logits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        current_labels: torch.Tensor | None = None,
        candidate_ids: torch.Tensor | None = None,
    ) -> GatedTemporalOutput:
        query = self.query_proj(query)
        context = self.candidate_context_pool(query, candidate_embs, valid_mask)
        context_exp = context.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)
        query_exp = query.unsqueeze(1).expand_as(candidate_embs)
        residual_input = torch.cat([query_exp, candidate_embs, context_exp, prior_logits.unsqueeze(-1)], dim=-1)
        residual_logits = self.residual_adapter(residual_input).squeeze(-1)
        if valid_mask is not None:
            residual_logits = residual_logits.masked_fill(
                ~valid_mask.to(device=residual_logits.device, dtype=torch.bool),
                torch.finfo(residual_logits.dtype).min,
            )
        lambda_t, gate_features = self.gate_lambda(
            prior_logits,
            valid_mask,
            current_labels=current_labels,
            candidate_ids=candidate_ids,
        )
        final_logits = prior_logits + lambda_t * residual_logits
        if valid_mask is not None:
            final_logits = final_logits.masked_fill(
                ~valid_mask.to(device=final_logits.device, dtype=torch.bool),
                torch.finfo(final_logits.dtype).min,
            )
        return GatedTemporalOutput(
            final_logits=final_logits,
            prior_logits=prior_logits,
            residual_logits=residual_logits,
            lambda_t=lambda_t,
            candidate_context=context,
            gate_features=gate_features,
        )
