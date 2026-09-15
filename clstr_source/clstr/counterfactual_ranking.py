from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CounterfactualUtilityOutput:
    loss: torch.Tensor
    gain_loss: torch.Tensor
    safety_loss: torch.Tensor
    weak_count: int
    strong_count: int
    gain_violation_count: int
    safety_violation_count: int


@dataclass
class FullPoolCausalRouteOutput:
    main_loss: torch.Tensor
    counterfactual: CounterfactualUtilityOutput
    eligible_mask: torch.Tensor
    exclusion_counts: dict[str, int]


def multi_positive_log_utility(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != positive_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, positive_mask, and valid_mask must be matching rank-2 tensors")
    if not logits.is_floating_point():
        raise ValueError("utility logits must use a floating dtype")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positives = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    floor = torch.finfo(logits.dtype).min
    masked_all = logits.masked_fill(~valid, floor)
    masked_pos = logits.masked_fill(~positives, floor)
    return torch.logsumexp(masked_pos.float(), dim=-1) - torch.logsumexp(masked_all.float(), dim=-1)


def shuffled_history_utility_loss(
    *,
    true_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if true_logits.ndim != 2 or true_logits.shape != shuffled_logits.shape:
        raise ValueError("true and shuffled logits must have matching rank-2 shapes")
    if true_logits.shape != positive_mask.shape or true_logits.shape != valid_mask.shape:
        raise ValueError("logits and masks must have matching shapes")
    if true_logits.device != shuffled_logits.device or true_logits.dtype != shuffled_logits.dtype:
        raise ValueError("true and shuffled logits must share dtype and device")
    if margin < 0:
        raise ValueError("shuffled-history margin must be nonnegative")

    valid = valid_mask.to(device=true_logits.device, dtype=torch.bool)
    positives = positive_mask.to(device=true_logits.device, dtype=torch.bool) & valid
    finite = torch.where(
        valid,
        torch.isfinite(true_logits) & torch.isfinite(shuffled_logits),
        True,
    ).all(dim=-1)
    eligible = positives.any(dim=-1) & (valid & ~positives).any(dim=-1) & finite
    if not bool(eligible.any()):
        return true_logits.masked_fill(~valid, 0.0).sum() * 0.0

    true_utility = multi_positive_log_utility(
        true_logits[eligible],
        positives[eligible],
        valid[eligible],
    )
    shuffled_utility = multi_positive_log_utility(
        shuffled_logits[eligible],
        positives[eligible],
        valid[eligible],
    )
    return F.relu(float(margin) + shuffled_utility - true_utility).mean()


def counterfactual_utility_loss(
    *,
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    gain_margin: float,
    safety_tolerance: float,
    gain_weight: float = 1.0,
    safety_weight: float = 1.0,
) -> CounterfactualUtilityOutput:
    if dynamic_logits.ndim != 2 or dynamic_logits.shape != static_logits.shape:
        raise ValueError("dynamic and static logits must have matching rank-2 shapes")
    if dynamic_logits.shape != positive_mask.shape or dynamic_logits.shape != valid_mask.shape:
        raise ValueError("logits and masks must have matching shapes")
    if dynamic_logits.device != static_logits.device or dynamic_logits.dtype != static_logits.dtype:
        raise ValueError("dynamic and static logits must share dtype and device")
    if gain_margin < 0 or safety_tolerance < 0 or gain_weight < 0 or safety_weight < 0:
        raise ValueError("counterfactual margins, tolerance, and branch weights must be nonnegative")
    if int(dynamic_logits.size(1)) == 0:
        raise ValueError("counterfactual logits must contain at least one skill")

    valid = valid_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    positives = positive_mask.to(device=dynamic_logits.device, dtype=torch.bool) & valid
    static = static_logits.detach()
    u_static = multi_positive_log_utility(static, positives, valid).detach()
    u_dynamic = multi_positive_log_utility(dynamic_logits, positives, valid)
    gain = u_dynamic - u_static
    floor = torch.finfo(static.dtype).min
    static_top1 = static.masked_fill(~valid, floor).argmax(dim=-1)
    static_correct = positives.gather(1, static_top1.unsqueeze(1)).squeeze(1)
    has_positive = positives.any(dim=-1)
    has_negative = (valid & ~positives).any(dim=-1)
    finite = torch.where(valid, torch.isfinite(dynamic_logits) & torch.isfinite(static), True).all(dim=-1)
    eligible = has_positive & has_negative & finite
    weak = eligible & ~static_correct
    strong = eligible & static_correct
    gain_terms = F.relu(float(gain_margin) - gain[weak])
    safety_terms = F.relu(-float(safety_tolerance) - gain[strong])
    zero = dynamic_logits.new_zeros(())
    gain_loss = gain_terms.mean() if gain_terms.numel() else zero
    safety_loss = safety_terms.mean() if safety_terms.numel() else zero
    total = float(gain_weight) * gain_loss + float(safety_weight) * safety_loss
    return CounterfactualUtilityOutput(
        loss=total,
        gain_loss=gain_loss,
        safety_loss=safety_loss,
        weak_count=int(weak.sum().detach().cpu().item()),
        strong_count=int(strong.sum().detach().cpu().item()),
        gain_violation_count=int((gain_terms > 0).sum().detach().cpu().item()),
        safety_violation_count=int((safety_terms > 0).sum().detach().cpu().item()),
    )


def full_pool_causal_route_objective(
    *,
    dynamic_logits: torch.Tensor,
    static_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    row_weights: torch.Tensor,
    gain_margin: float,
    safety_tolerance: float,
    gain_weight: float = 1.0,
    safety_weight: float = 1.0,
    compute_counterfactual: bool = True,
) -> FullPoolCausalRouteOutput:
    if dynamic_logits.ndim != 2 or dynamic_logits.shape != static_logits.shape:
        raise ValueError("full-pool branch logits must have matching rank-2 shapes")
    if dynamic_logits.shape != positive_mask.shape or dynamic_logits.shape != valid_mask.shape:
        raise ValueError("full-pool logits and masks must have matching shapes")
    if row_weights.ndim != 1 or int(row_weights.numel()) != int(dynamic_logits.size(0)):
        raise ValueError("row_weights must have one value per row")
    if dynamic_logits.device != static_logits.device or dynamic_logits.dtype != static_logits.dtype:
        raise ValueError("full-pool branch logits must share dtype and device")

    valid = valid_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    known_positive = positive_mask.to(device=dynamic_logits.device, dtype=torch.bool)
    legal_positive = known_positive & valid
    has_legal_positive = legal_positive.any(dim=-1)
    has_legal_negative = (valid & ~legal_positive).any(dim=-1)
    finite = torch.where(
        valid,
        torch.isfinite(dynamic_logits) & torch.isfinite(static_logits),
        True,
    ).all(dim=-1)
    eligible = has_legal_positive & has_legal_negative & finite
    weights = row_weights.to(device=dynamic_logits.device, dtype=torch.float32)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("row_weights must be finite and nonnegative")

    zero = dynamic_logits.new_zeros(())
    if bool(eligible.any()):
        eligible_weights = weights[eligible]
        if not bool((eligible_weights.sum() > 0).item()):
            raise ValueError("eligible full-pool rows have zero total weight")
        row_nll = -multi_positive_log_utility(
            dynamic_logits[eligible],
            legal_positive[eligible],
            valid[eligible],
        )
        main_loss = (row_nll * eligible_weights).sum() / eligible_weights.sum()
    else:
        main_loss = zero
    if compute_counterfactual:
        counterfactual = counterfactual_utility_loss(
            dynamic_logits=dynamic_logits[eligible],
            static_logits=static_logits[eligible],
            positive_mask=legal_positive[eligible],
            valid_mask=valid[eligible],
            gain_margin=gain_margin,
            safety_tolerance=safety_tolerance,
            gain_weight=gain_weight,
            safety_weight=safety_weight,
        )
    else:
        counterfactual = CounterfactualUtilityOutput(
            loss=zero,
            gain_loss=zero,
            safety_loss=zero,
            weak_count=0,
            strong_count=0,
            gain_violation_count=0,
            safety_violation_count=0,
        )
    return FullPoolCausalRouteOutput(
        main_loss=main_loss,
        counterfactual=counterfactual,
        eligible_mask=eligible,
        exclusion_counts={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "no_valid_positive_rows": int((~has_legal_positive).sum().detach().cpu().item()),
            "no_valid_negative_rows": int((~has_legal_negative).sum().detach().cpu().item()),
            "nonfinite_logit_rows": int((~finite).sum().detach().cpu().item()),
        },
    )
