from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


CANDIDATE_ADMISSION_RESIDUAL_V1 = "candidate_admission_residual_v1"
CANDIDATE_ADMISSION_RESIDUAL_BOUND = 2.0
CANDIDATE_ADMISSION_UPDATE_COUNT_CAP = 16.0
CANDIDATE_ADMISSION_CANDIDATE_COUNT_CAP = 256.0
CANDIDATE_ADMISSION_SCALAR_FEATURES = (
    "static_standardized_logit",
    "dynamic_standardized_logit",
    "centered_dynamic_minus_static",
    "normalized_static_rank",
    "normalized_dynamic_rank",
    "is_dynamic_extra",
    "log_normalized_update_count",
    "log_normalized_candidate_count",
)


@dataclass(frozen=True)
class CandidateAdmissionScoringOutput:
    final_logits: torch.Tensor
    static_logits: torch.Tensor
    dynamic_logits: torch.Tensor
    admission_logits: torch.Tensor
    admission_probability: torch.Tensor
    bounded_residual: torch.Tensor


@dataclass(frozen=True)
class CandidateAdmissionResidualObjective:
    loss: torch.Tensor
    listwise_loss: torch.Tensor
    admission_loss: torch.Tensor
    no_regret_loss: torch.Tensor
    safety_loss: torch.Tensor
    gain_loss: torch.Tensor
    admission_positive_count: int
    admission_negative_count: int


class CandidateAdmissionResidualHead(torch.nn.Module):
    """Candidate-wise admission and signed residual heads over frozen features."""

    def __init__(
        self,
        model_dim: int,
        scalar_dim: int = len(CANDIDATE_ADMISSION_SCALAR_FEATURES),
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        model_dim = int(model_dim)
        scalar_dim = int(scalar_dim)
        hidden_dim = int(hidden_dim)
        if model_dim <= 0 or scalar_dim <= 0 or hidden_dim <= 0:
            raise ValueError("candidate admission dimensions must be positive")
        self.model_dim = model_dim
        self.scalar_dim = scalar_dim
        input_dim = 5 * model_dim + scalar_dim
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.admission_head = torch.nn.Linear(hidden_dim, 1)
        self.residual_head = torch.nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        torch.nn.init.zeros_(self.admission_head.weight)
        torch.nn.init.constant_(self.admission_head.bias, -4.0)
        torch.nn.init.zeros_(self.residual_head.weight)
        torch.nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        h: torch.Tensor,
        memory_delta: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 2 or h.shape != memory_delta.shape:
            raise ValueError("candidate admission state inputs must match at rank 2")
        if candidate_embeddings.ndim != 3 or scalar_features.ndim != 3:
            raise ValueError("candidate admission candidate inputs must be rank 3")
        if (
            int(candidate_embeddings.size(0)) != int(h.size(0))
            or int(candidate_embeddings.size(2)) != self.model_dim
            or tuple(scalar_features.shape[:2])
            != tuple(candidate_embeddings.shape[:2])
            or int(scalar_features.size(2)) != self.scalar_dim
        ):
            raise ValueError("candidate admission feature shapes are incompatible")
        width = int(candidate_embeddings.size(1))
        h_rows = h.unsqueeze(1).expand(-1, width, -1)
        memory_rows = memory_delta.unsqueeze(1).expand(-1, width, -1)
        candidate_embeddings = candidate_embeddings.to(
            device=h.device,
            dtype=h.dtype,
        )
        scalar_features = scalar_features.to(device=h.device, dtype=h.dtype)
        features = torch.cat(
            (
                h_rows,
                memory_rows,
                candidate_embeddings,
                h_rows * candidate_embeddings,
                memory_rows * candidate_embeddings,
                scalar_features,
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        return (
            self.admission_head(hidden).squeeze(-1),
            self.residual_head(hidden).squeeze(-1),
        )


def _validate_rank2_contract(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        static_logits.ndim != 2
        or static_logits.shape != dynamic_logits.shape
        or static_logits.shape != dynamic_extra_mask.shape
        or static_logits.shape != valid_mask.shape
    ):
        raise ValueError("candidate admission logits and masks must share rank-2 shapes")
    if not static_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError("candidate admission logits must be floating tensors")
    if (
        static_logits.dtype != dynamic_logits.dtype
        or static_logits.device != dynamic_logits.device
    ):
        raise ValueError("candidate admission logits must share dtype and device")
    if (
        causal_update_count.ndim != 1
        or int(causal_update_count.numel()) != int(static_logits.size(0))
    ):
        raise ValueError("candidate admission update count must provide one value per row")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    extras = dynamic_extra_mask.to(device=static_logits.device, dtype=torch.bool) & valid
    if not torch.isfinite(static_logits[valid]).all() or not torch.isfinite(
        dynamic_logits[valid]
    ).all():
        raise ValueError("valid candidate admission logits must be finite")
    counts = causal_update_count.to(
        device=static_logits.device,
        dtype=static_logits.dtype,
    )
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("candidate admission update counts must be finite and nonnegative")
    return valid, extras


def _standardized(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = values.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / count.to(
        values.dtype
    )
    centered = values - mean
    variance = centered.square().masked_fill(~valid, 0.0).sum(
        dim=-1,
        keepdim=True,
    ) / count.to(values.dtype)
    return centered / variance.sqrt().clamp_min(1.0e-6)


def _normalized_rank(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(values.dtype).min
    order = torch.argsort(
        values.masked_fill(~valid, floor),
        dim=-1,
        descending=True,
        stable=True,
    )
    ranks = torch.empty_like(order)
    positions = torch.arange(values.size(1), device=values.device).expand_as(order)
    ranks.scatter_(1, order, positions)
    denominator = (valid.sum(dim=-1, keepdim=True) - 1).clamp_min(1)
    return ranks.to(values.dtype) / denominator.to(values.dtype)


def candidate_admission_scalar_features(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed label-free candidate feature contract."""

    valid, extras = _validate_rank2_contract(
        static_logits,
        dynamic_logits,
        dynamic_extra_mask,
        valid_mask,
        causal_update_count,
    )
    static_z = _standardized(static_logits, valid)
    dynamic_z = _standardized(dynamic_logits, valid)
    residual_z = _standardized(dynamic_logits - static_logits, valid)
    update = causal_update_count.to(
        device=static_logits.device,
        dtype=static_logits.dtype,
    ).clamp_max(CANDIDATE_ADMISSION_UPDATE_COUNT_CAP)
    update_feature = (
        torch.log1p(update)
        / static_logits.new_tensor(CANDIDATE_ADMISSION_UPDATE_COUNT_CAP).log1p()
    ).unsqueeze(-1).expand_as(static_logits)
    candidate_count = valid.sum(dim=-1, keepdim=True).to(static_logits.dtype)
    candidate_count = candidate_count.clamp_max(
        CANDIDATE_ADMISSION_CANDIDATE_COUNT_CAP
    )
    count_feature = torch.log1p(candidate_count) / static_logits.new_tensor(
        CANDIDATE_ADMISSION_CANDIDATE_COUNT_CAP
    ).log1p()
    count_feature = count_feature.expand_as(static_logits)
    output = torch.stack(
        (
            static_z,
            dynamic_z,
            residual_z,
            _normalized_rank(static_logits, valid),
            _normalized_rank(dynamic_logits, valid),
            extras.to(static_logits.dtype),
            update_feature,
            count_feature,
        ),
        dim=-1,
    )
    return output.masked_fill(~valid.unsqueeze(-1), 0.0)


def score_candidate_admission_residual(
    head: torch.nn.Module,
    *,
    h: torch.Tensor,
    memory_delta: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
    residual_bound: float = CANDIDATE_ADMISSION_RESIDUAL_BOUND,
) -> CandidateAdmissionScoringOutput:
    """Apply candidate admission and bounded signed residual ranking."""

    bound = float(residual_bound)
    if bound != CANDIDATE_ADMISSION_RESIDUAL_BOUND:
        raise ValueError("candidate admission residual bound must equal 2.0")
    valid, extras = _validate_rank2_contract(
        static_logits,
        dynamic_logits,
        dynamic_extra_mask,
        valid_mask,
        causal_update_count,
    )
    if (
        h.ndim != 2
        or h.shape != memory_delta.shape
        or candidate_embeddings.ndim != 3
        or tuple(candidate_embeddings.shape[:2]) != tuple(static_logits.shape)
        or int(candidate_embeddings.size(2)) != int(h.size(1))
        or int(h.size(0)) != int(static_logits.size(0))
    ):
        raise ValueError("candidate admission state and candidate shapes are incompatible")
    scalar_features = candidate_admission_scalar_features(
        static_logits,
        dynamic_logits,
        extras,
        valid,
        causal_update_count,
    )
    admission_logits, raw_residual = head(
        h.detach(),
        memory_delta.detach(),
        candidate_embeddings.detach(),
        scalar_features.detach(),
    )
    if (
        not isinstance(admission_logits, torch.Tensor)
        or not isinstance(raw_residual, torch.Tensor)
        or admission_logits.shape != static_logits.shape
        or raw_residual.shape != static_logits.shape
    ):
        raise ValueError("candidate admission head outputs must match candidate logits")
    admission_logits = admission_logits.to(
        device=static_logits.device,
        dtype=static_logits.dtype,
    )
    raw_residual = raw_residual.to(
        device=static_logits.device,
        dtype=static_logits.dtype,
    )
    if not torch.isfinite(admission_logits[valid]).all() or not torch.isfinite(
        raw_residual[valid]
    ).all():
        raise ValueError("valid candidate admission outputs must be finite")
    admission_probability = torch.sigmoid(admission_logits)
    bounded_residual = bound * torch.tanh(raw_residual / bound)
    scale = torch.where(
        extras,
        admission_probability,
        torch.ones_like(admission_probability),
    )
    active = (
        causal_update_count.to(device=static_logits.device) > 0
    ).unsqueeze(-1) & valid
    final_logits = torch.where(
        active,
        static_logits + scale * bounded_residual,
        static_logits,
    )
    final_logits = final_logits.masked_fill(~valid, torch.finfo(final_logits.dtype).min)
    return CandidateAdmissionScoringOutput(
        final_logits=final_logits,
        static_logits=static_logits,
        dynamic_logits=dynamic_logits,
        admission_logits=admission_logits,
        admission_probability=admission_probability,
        bounded_residual=bounded_residual,
    )


def _multi_positive_listwise_loss(
    logits: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    negative = valid & ~positive
    eligible = positive.any(dim=-1) & negative.any(dim=-1)
    if not bool(eligible.any()):
        return logits.new_zeros(())
    selected_logits = logits[eligible]
    selected_valid = valid[eligible]
    selected_positive = positive[eligible]
    floor = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(
        selected_logits.masked_fill(~selected_valid, floor),
        dim=-1,
    )
    numerator = torch.logsumexp(
        selected_logits.masked_fill(~selected_positive, floor),
        dim=-1,
    )
    return (denominator - numerator).mean()


def _positive_negative_margin(
    logits: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    negative = valid & ~positive
    eligible = positive.any(dim=-1) & negative.any(dim=-1)
    floor = torch.finfo(logits.dtype).min
    positive_score = logits.masked_fill(~positive, floor).max(dim=-1).values
    negative_score = logits.masked_fill(~negative, floor).max(dim=-1).values
    return positive_score - negative_score, eligible


def candidate_admission_residual_loss(
    *,
    scoring: CandidateAdmissionScoringOutput,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    admission_weight: float,
    counterfactual_weight: float,
    gain_margin: float,
) -> CandidateAdmissionResidualObjective:
    """Optimize final ranking, extra admission, and counterfactual no regret."""

    shape = scoring.final_logits.shape
    if (
        scoring.final_logits.ndim != 2
        or positive_mask.shape != shape
        or valid_mask.shape != shape
        or dynamic_extra_mask.shape != shape
    ):
        raise ValueError("candidate admission objective tensors must share rank-2 shapes")
    for name, value in (
        ("admission_weight", admission_weight),
        ("counterfactual_weight", counterfactual_weight),
        ("gain_margin", gain_margin),
    ):
        if not torch.isfinite(torch.tensor(float(value))) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    valid = valid_mask.to(device=scoring.final_logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=scoring.final_logits.device, dtype=torch.bool) & valid
    extras = dynamic_extra_mask.to(
        device=scoring.final_logits.device,
        dtype=torch.bool,
    ) & valid
    listwise_loss = _multi_positive_listwise_loss(
        scoring.final_logits,
        positive,
        valid,
    )

    admission_positive = extras & positive
    admission_negative = extras & ~positive
    admission_positive_count = int(admission_positive.sum().detach().cpu().item())
    admission_negative_count = int(admission_negative.sum().detach().cpu().item())
    admission_eligible = admission_positive | admission_negative
    if bool(admission_eligible.any()):
        positive_weight = max(
            1.0,
            min(
                64.0,
                admission_negative_count / max(1, admission_positive_count),
            ),
        )
        admission_loss = F.binary_cross_entropy_with_logits(
            scoring.admission_logits[admission_eligible],
            admission_positive[admission_eligible].to(
                scoring.admission_logits.dtype
            ),
            pos_weight=scoring.admission_logits.new_tensor(positive_weight),
        )
    else:
        admission_loss = scoring.final_logits.new_zeros(())

    static_margin, margin_eligible = _positive_negative_margin(
        scoring.static_logits,
        positive,
        valid,
    )
    final_margin, _ = _positive_negative_margin(
        scoring.final_logits,
        positive,
        valid,
    )
    safety_loss = (
        torch.relu(static_margin[margin_eligible].detach() - final_margin[margin_eligible]).mean()
        if bool(margin_eligible.any())
        else scoring.final_logits.new_zeros(())
    )
    gain_rows = admission_positive.any(dim=-1) & margin_eligible
    gain_loss = (
        torch.relu(float(gain_margin) - final_margin[gain_rows]).mean()
        if bool(gain_rows.any())
        else scoring.final_logits.new_zeros(())
    )
    no_regret_loss = safety_loss + gain_loss
    loss = (
        listwise_loss
        + float(admission_weight) * admission_loss
        + float(counterfactual_weight) * no_regret_loss
    )
    return CandidateAdmissionResidualObjective(
        loss=loss,
        listwise_loss=listwise_loss,
        admission_loss=admission_loss,
        no_regret_loss=no_regret_loss,
        safety_loss=safety_loss,
        gain_loss=gain_loss,
        admission_positive_count=admission_positive_count,
        admission_negative_count=admission_negative_count,
    )
