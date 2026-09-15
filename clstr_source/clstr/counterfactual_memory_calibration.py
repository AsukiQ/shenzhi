from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from clstr.memory_utility_gate import (
    effective_memory_alpha,
    fuse_route_scores,
    memory_utility_features,
)


CMC_FEATURE_UPDATE_COUNT_CAP = 16.0
CMC_FEATURE_CANDIDATE_COUNT_CAP = 256.0


class RouteMemoryResidualAdapter(torch.nn.Module):
    """Learn a small correction to the frozen Stage2 dynamic route vector."""

    def __init__(self, d: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        d = int(d)
        hidden_dim = int(hidden_dim)
        if d <= 0 or hidden_dim <= 0:
            raise ValueError("CMC adapter dimensions must be positive")
        self.d = d
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(3 * d),
            torch.nn.Linear(3 * d, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, d),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        memory_delta: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 2 or h.shape != memory_delta.shape or int(h.size(-1)) != self.d:
            raise ValueError("CMC adapter inputs must have matching rank-2 model dimensions")
        if not h.is_floating_point() or not memory_delta.is_floating_point():
            raise ValueError("CMC adapter inputs must use floating dtypes")
        if h.dtype != memory_delta.dtype or h.device != memory_delta.device:
            raise ValueError("CMC adapter inputs must share dtype and device")
        return self.net(torch.cat((h, memory_delta, h * memory_delta), dim=-1))


@dataclass(frozen=True)
class CMCCandidateScoringOutput:
    static_logits: torch.Tensor
    raw_dynamic_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    fused_logits: torch.Tensor
    route_residual: torch.Tensor
    features: torch.Tensor
    raw_alpha: torch.Tensor
    effective_alpha: torch.Tensor


def _candidate_residual_logits(
    route_residual: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    embeddings = candidate_embeddings.detach().to(
        device=route_residual.device,
        dtype=route_residual.dtype,
    )
    if embeddings.ndim == 2:
        if int(embeddings.size(1)) != int(route_residual.size(1)):
            raise ValueError("CMC candidate embeddings must match the model dimension")
        return route_residual @ embeddings.t()
    if embeddings.ndim == 3:
        if (
            int(embeddings.size(0)) != int(route_residual.size(0))
            or int(embeddings.size(2)) != int(route_residual.size(1))
        ):
            raise ValueError("CMC batched candidate embeddings must match batch and model dimensions")
        return torch.einsum("bd,bcd->bc", route_residual, embeddings)
    raise ValueError("CMC candidate embeddings must be rank 2 or rank 3")


def score_cmc_candidates(
    model: Any,
    *,
    h: torch.Tensor,
    static_memory: torch.Tensor,
    dynamic_memory: torch.Tensor,
    static_logits: torch.Tensor,
    raw_dynamic_logits: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
    feature_update_count_cap: float,
    feature_candidate_count_cap: float,
) -> CMCCandidateScoringOutput:
    """Apply the checkpoint-native CMC adapter and gate to one legal candidate set."""

    if (
        h.ndim != 2
        or h.shape != static_memory.shape
        or h.shape != dynamic_memory.shape
    ):
        raise ValueError("CMC state and memory tensors must have matching rank-2 shapes")
    if static_logits.ndim != 2 or static_logits.shape != raw_dynamic_logits.shape:
        raise ValueError("CMC static and raw dynamic logits must have matching rank-2 shapes")
    if int(static_logits.size(0)) != int(h.size(0)):
        raise ValueError("CMC logits and state tensors must share a batch size")
    adapter = getattr(model, "route_memory_residual_adapter", None)
    candidate_gate = getattr(model, "route_memory_candidate_utility_gate", None)
    if not callable(adapter) or not callable(candidate_gate):
        raise ValueError("CMC scoring requires residual adapter and candidate utility gate")

    route_residual = adapter(h, dynamic_memory - static_memory)
    correction = _candidate_residual_logits(route_residual, candidate_embeddings).to(
        device=raw_dynamic_logits.device,
        dtype=raw_dynamic_logits.dtype,
    )
    if correction.shape != raw_dynamic_logits.shape:
        raise ValueError("CMC candidate correction must match raw dynamic logits")
    dynamic_logits = raw_dynamic_logits + correction
    features = memory_utility_features(
        static_logits,
        dynamic_logits,
        valid_mask,
        static_memory,
        dynamic_memory,
        causal_update_count,
        update_count_cap=feature_update_count_cap,
        candidate_count_cap=feature_candidate_count_cap,
    ).detach()
    raw_alpha = candidate_gate(features)
    if not isinstance(raw_alpha, torch.Tensor):
        raise ValueError("CMC candidate utility gate must return a tensor")
    raw_alpha = raw_alpha.to(device=dynamic_logits.device, dtype=dynamic_logits.dtype)
    effective_alpha = effective_memory_alpha(raw_alpha, causal_update_count)
    fused_logits = fuse_route_scores(
        static_logits,
        dynamic_logits,
        effective_alpha,
        valid_mask,
    )
    return CMCCandidateScoringOutput(
        static_logits=static_logits,
        raw_dynamic_logits=raw_dynamic_logits,
        dynamic_logits=dynamic_logits,
        fused_logits=fused_logits,
        route_residual=route_residual,
        features=features,
        raw_alpha=raw_alpha,
        effective_alpha=effective_alpha,
    )


@dataclass(frozen=True)
class CounterfactualMemoryCalibrationOutput:
    loss: torch.Tensor
    dynamic_loss: torch.Tensor
    fused_loss: torch.Tensor
    no_regret_loss: torch.Tensor
    static_utility: torch.Tensor
    dynamic_utility: torch.Tensor
    fused_utility: torch.Tensor
    static_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    fused_logits: torch.Tensor
    alpha_zero_logits: torch.Tensor
    alpha_one_logits: torch.Tensor


def _validate_cmc_inputs(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    alpha: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        static_logits.ndim != 2
        or static_logits.shape != dynamic_logits.shape
        or static_logits.shape != positive_mask.shape
        or static_logits.shape != valid_mask.shape
    ):
        raise ValueError("CMC logits and masks must have matching rank-2 shapes")
    if not static_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError("CMC logits must use floating dtypes")
    if static_logits.dtype != dynamic_logits.dtype or static_logits.device != dynamic_logits.device:
        raise ValueError("CMC logits must share dtype and device")
    if (
        not alpha.is_floating_point()
        or alpha.ndim != 1
        or int(alpha.numel()) != int(static_logits.size(0))
        or alpha.device != static_logits.device
    ):
        raise ValueError("CMC alpha must provide one floating scalar per route row")
    if not torch.isfinite(alpha).all() or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("CMC alpha must be finite and in [0, 1]")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=static_logits.device, dtype=torch.bool) & valid
    if bool((valid.sum(dim=-1) <= 0).any()):
        raise ValueError("CMC requires at least one valid candidate per row")
    if bool((positive.sum(dim=-1) <= 0).any()):
        raise ValueError("CMC requires at least one valid positive per row")
    if (
        not torch.isfinite(static_logits[valid]).all()
        or not torch.isfinite(dynamic_logits[valid]).all()
    ):
        raise ValueError("CMC valid logits must be finite")
    return positive, valid


def _masked_log_utility(
    logits: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    valid_logits = logits.masked_fill(~valid, -torch.inf)
    return torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(
        valid_logits,
        dim=-1,
    )


def counterfactual_memory_calibration_loss(
    *,
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    alpha: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> CounterfactualMemoryCalibrationOutput:
    """Train raw dynamic and deployed fused rankings with a static no-regret floor."""

    positive, valid = _validate_cmc_inputs(
        static_logits,
        dynamic_logits,
        alpha,
        positive_mask,
        valid_mask,
    )
    alpha_zero_logits = fuse_route_scores(
        static_logits,
        dynamic_logits,
        torch.zeros_like(alpha),
        valid,
    )
    alpha_one_logits = fuse_route_scores(
        static_logits,
        dynamic_logits,
        torch.ones_like(alpha),
        valid,
    )
    fused_logits = fuse_route_scores(
        static_logits,
        dynamic_logits,
        alpha,
        valid,
    )
    static_utility = _masked_log_utility(alpha_zero_logits, positive, valid)
    dynamic_utility = _masked_log_utility(alpha_one_logits, positive, valid)
    fused_utility = _masked_log_utility(fused_logits, positive, valid)
    dynamic_loss = -dynamic_utility.mean()
    fused_loss = -fused_utility.mean()
    no_regret_loss = torch.relu(static_utility.detach() - fused_utility).mean()
    loss = dynamic_loss + fused_loss + no_regret_loss
    return CounterfactualMemoryCalibrationOutput(
        loss=loss,
        dynamic_loss=dynamic_loss,
        fused_loss=fused_loss,
        no_regret_loss=no_regret_loss,
        static_utility=static_utility,
        dynamic_utility=dynamic_utility,
        fused_utility=fused_utility,
        static_logits=alpha_zero_logits,
        dynamic_logits=alpha_one_logits,
        fused_logits=fused_logits,
        alpha_zero_logits=alpha_zero_logits,
        alpha_one_logits=alpha_one_logits,
    )
