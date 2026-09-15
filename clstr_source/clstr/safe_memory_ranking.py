from __future__ import annotations

import math
from dataclasses import dataclass

import torch


CANDIDATE_PROVENANCE_RESIDUAL_BOUND = 2.0


@dataclass
class LocalCandidateMasks:
    masks: torch.Tensor
    source_row_indices: torch.Tensor
    requested_sizes: torch.Tensor


@dataclass
class SafeLocalRouteObjective:
    loss: torch.Tensor
    nll_loss: torch.Tensor
    gain_loss: torch.Tensor
    safety_loss: torch.Tensor
    eligible_count: int
    weak_count: int
    strong_count: int
    gain_violation_count: int
    safety_violation_count: int


def bounded_memory_fusion(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    alpha: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    residual_bound: float,
) -> torch.Tensor:
    """Fuse a centered, bounded memory residual into frozen static logits."""

    if (
        static_logits.ndim != 2
        or static_logits.shape != dynamic_logits.shape
        or static_logits.shape != valid_mask.shape
    ):
        raise ValueError(
            "static_logits, dynamic_logits, and valid_mask must have matching rank-2 shapes"
        )
    if not static_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError("route logits must use floating dtypes")
    if (
        static_logits.dtype != dynamic_logits.dtype
        or static_logits.device != dynamic_logits.device
    ):
        raise ValueError("static and dynamic logits must share dtype and device")
    if not alpha.is_floating_point() or alpha.ndim != 1:
        raise ValueError("alpha must be a rank-1 floating tensor")
    if int(alpha.numel()) != int(static_logits.size(0)):
        raise ValueError("alpha must have one value per route row")
    if alpha.device != static_logits.device:
        raise ValueError("alpha and route logits must share a device")
    if not torch.isfinite(alpha).all() or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha must be finite and in [0, 1]")
    bound = float(residual_bound)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("residual_bound must be finite and positive")

    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    if not torch.isfinite(static_logits[valid]).all() or not torch.isfinite(
        dynamic_logits[valid]
    ).all():
        raise ValueError("valid route logits must be finite")

    raw_residual = dynamic_logits - static_logits
    valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    centered_residual = raw_residual - (
        raw_residual.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
        / valid_count.to(dtype=raw_residual.dtype)
    )
    bounded_residual = bound * torch.tanh(centered_residual / bound)
    expanded_alpha = alpha.to(dtype=static_logits.dtype).unsqueeze(-1)
    candidate = static_logits + expanded_alpha * bounded_residual
    candidate = torch.where(expanded_alpha == 0, static_logits, candidate)
    return candidate.masked_fill(~valid, torch.finfo(static_logits.dtype).min)


def candidate_provenance_positive_residual_fusion(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    dynamic_extra_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    causal_update_count: torch.Tensor,
    *,
    residual_bound: float,
) -> torch.Tensor:
    """Boost only dynamic-only candidates with a bounded positive residual."""

    if (
        static_logits.ndim != 2
        or static_logits.shape != dynamic_logits.shape
        or static_logits.shape != dynamic_extra_mask.shape
        or static_logits.shape != valid_mask.shape
    ):
        raise ValueError(
            "provenance fusion logits and masks must have matching rank-2 shapes"
        )
    if not static_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError("provenance fusion logits must use floating dtypes")
    if (
        static_logits.dtype != dynamic_logits.dtype
        or static_logits.device != dynamic_logits.device
    ):
        raise ValueError("provenance fusion logits must share dtype and device")
    if (
        causal_update_count.ndim != 1
        or int(causal_update_count.numel()) != int(static_logits.size(0))
    ):
        raise ValueError("causal update count must provide one value per row")
    bound = float(residual_bound)
    if bound != CANDIDATE_PROVENANCE_RESIDUAL_BOUND:
        raise ValueError("candidate provenance residual bound must equal 2.0")
    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    extras = dynamic_extra_mask.to(device=static_logits.device, dtype=torch.bool)
    counts = causal_update_count.to(
        device=static_logits.device,
        dtype=static_logits.dtype,
    )
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal update count must be finite and nonnegative")
    if not torch.isfinite(static_logits[valid]).all() or not torch.isfinite(
        dynamic_logits[valid]
    ).all():
        raise ValueError("valid provenance fusion logits must be finite")

    raw_residual = dynamic_logits - static_logits
    valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    centered_residual = raw_residual - (
        raw_residual.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
        / valid_count.to(dtype=raw_residual.dtype)
    )
    bounded_positive = bound * torch.tanh(
        torch.relu(centered_residual) / bound
    )
    active = extras & valid & (counts > 0).unsqueeze(-1)
    fused = torch.where(
        active,
        static_logits + bounded_positive,
        static_logits,
    )
    return fused.masked_fill(~valid, torch.finfo(static_logits.dtype).min)


def _stable_top_indices(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    limit: int,
) -> list[int]:
    valid_count = int(mask.sum().detach().cpu().item())
    selected_count = min(max(0, int(limit)), valid_count)
    if selected_count == 0:
        return []
    floor = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, floor)
    threshold = torch.topk(
        masked,
        k=selected_count,
        largest=True,
        sorted=False,
    ).values.min()
    strict = (mask & (values > threshold)).nonzero(as_tuple=False).view(-1)
    if int(strict.numel()):
        strict_order = torch.argsort(
            values.index_select(0, strict),
            descending=True,
            stable=True,
        )
        strict = strict.index_select(0, strict_order)
    boundary = (mask & (values == threshold)).nonzero(as_tuple=False).view(-1)
    fill = selected_count - int(strict.numel())
    chosen = torch.cat((strict, boundary[:fill]), dim=0)
    return [int(item) for item in chosen.detach().cpu().tolist()]


def build_local_candidate_masks(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    explicit_row_mask: torch.Tensor,
    *,
    candidate_sizes: tuple[int, ...] = (2, 3, 4, 5, 8, 10),
) -> LocalCandidateMasks:
    if (
        static_logits.ndim != 2
        or static_logits.shape != dynamic_logits.shape
        or static_logits.shape != positive_mask.shape
        or static_logits.shape != valid_mask.shape
    ):
        raise ValueError("local candidate tensors must have matching rank-2 shapes")
    if static_logits.dtype != dynamic_logits.dtype or static_logits.device != dynamic_logits.device:
        raise ValueError("static and dynamic logits must share dtype and device")
    if not static_logits.is_floating_point():
        raise ValueError("local candidate logits must use floating dtypes")
    if explicit_row_mask.ndim != 1 or int(explicit_row_mask.numel()) != int(static_logits.size(0)):
        raise ValueError("explicit_row_mask must have one value per route row")
    sizes = tuple(int(size) for size in candidate_sizes)
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("candidate_sizes must contain integers of at least two")

    valid = valid_mask.to(device=static_logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=static_logits.device, dtype=torch.bool) & valid
    explicit = explicit_row_mask.to(device=static_logits.device, dtype=torch.bool)
    if not torch.isfinite(static_logits[valid]).all() or not torch.isfinite(dynamic_logits[valid]).all():
        raise ValueError("valid local candidate logits must be finite")

    masks: list[torch.Tensor] = []
    source_rows: list[int] = []
    requested: list[int] = []
    for row_idx in range(int(static_logits.size(0))):
        row_valid = valid[row_idx]
        row_positive = positive[row_idx]
        row_negative = row_valid & ~row_positive
        valid_count = int(row_valid.sum().detach().cpu().item())
        if not bool(row_positive.any()) or not bool(row_negative.any()):
            continue
        if bool(explicit[row_idx]):
            masks.append(row_valid.clone())
            source_rows.append(row_idx)
            requested.append(valid_count)
            continue

        selection_limit = max(1, min(max(sizes), valid_count) - 1)
        positive_order = _stable_top_indices(
            static_logits[row_idx],
            row_positive,
            limit=selection_limit,
        )
        static_negative_order = _stable_top_indices(
            static_logits[row_idx],
            row_negative,
            limit=selection_limit,
        )
        disagreement_order = _stable_top_indices(
            dynamic_logits[row_idx] - static_logits[row_idx],
            row_negative,
            limit=selection_limit,
        )
        effective_sizes: list[int] = []
        for size in sizes:
            effective = min(size, valid_count)
            if effective not in effective_sizes:
                effective_sizes.append(effective)
        for effective in effective_sizes:
            positive_count = min(len(positive_order), effective - 1)
            chosen = list(positive_order[:positive_count])
            negative_slots = effective - len(chosen)
            alternating: list[int] = []
            for offset in range(max(len(static_negative_order), len(disagreement_order))):
                if offset < len(static_negative_order):
                    alternating.append(static_negative_order[offset])
                if offset < len(disagreement_order):
                    alternating.append(disagreement_order[offset])
            for index in alternating:
                if index in chosen:
                    continue
                chosen.append(index)
                if len(chosen) >= effective:
                    break
            if len(chosen) != effective or negative_slots <= 0:
                raise ValueError("failed to build a positive-and-negative local candidate mask")
            mask = torch.zeros_like(row_valid)
            mask[torch.tensor(chosen, dtype=torch.long, device=mask.device)] = True
            masks.append(mask)
            source_rows.append(row_idx)
            requested.append(effective)

    if masks:
        stacked = torch.stack(masks, dim=0)
    else:
        stacked = torch.zeros(
            (0, int(static_logits.size(1))),
            dtype=torch.bool,
            device=static_logits.device,
        )
    return LocalCandidateMasks(
        masks=stacked,
        source_row_indices=torch.tensor(source_rows, dtype=torch.long, device=static_logits.device),
        requested_sizes=torch.tensor(requested, dtype=torch.long, device=static_logits.device),
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) == 0:
        return weights.new_zeros(())
    selected_weights = weights.to(device=values.device, dtype=values.dtype)
    denominator = selected_weights.sum()
    if not bool((denominator > 0).item()):
        raise ValueError("eligible local candidate masks have zero total weight")
    return (values * selected_weights).sum() / denominator


def safe_local_route_objective(
    *,
    fused_logits: torch.Tensor,
    static_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_masks: torch.Tensor,
    source_row_indices: torch.Tensor,
    row_weights: torch.Tensor,
    gain_margin: float,
    safety_tolerance: float,
) -> SafeLocalRouteObjective:
    if fused_logits.ndim != 2 or fused_logits.shape != static_logits.shape:
        raise ValueError("fused and static logits must have matching rank-2 shapes")
    if positive_mask.shape != fused_logits.shape:
        raise ValueError("positive_mask must match route logits")
    if candidate_masks.ndim != 2 or int(candidate_masks.size(1)) != int(fused_logits.size(1)):
        raise ValueError("candidate_masks must match the route skill dimension")
    if source_row_indices.ndim != 1 or int(source_row_indices.numel()) != int(candidate_masks.size(0)):
        raise ValueError("source_row_indices must have one value per candidate mask")
    if row_weights.ndim != 1 or int(row_weights.numel()) != int(fused_logits.size(0)):
        raise ValueError("row_weights must have one value per route row")
    if gain_margin < 0.0 or safety_tolerance < 0.0:
        raise ValueError("local gain margin and safety tolerance must be nonnegative")
    if not torch.isfinite(row_weights).all() or bool((row_weights < 0).any()):
        raise ValueError("row_weights must be finite and nonnegative")
    if int(source_row_indices.numel()) and (
        bool((source_row_indices < 0).any())
        or bool((source_row_indices >= int(fused_logits.size(0))).any())
    ):
        raise ValueError("source_row_indices contain an out-of-range row")

    source = source_row_indices.to(device=fused_logits.device, dtype=torch.long)
    masks = candidate_masks.to(device=fused_logits.device, dtype=torch.bool)
    fused = fused_logits.index_select(0, source)
    static = static_logits.detach().index_select(0, source)
    positive = positive_mask.to(device=fused_logits.device, dtype=torch.bool).index_select(0, source) & masks
    negative = masks & ~positive
    finite = torch.where(
        masks,
        torch.isfinite(fused) & torch.isfinite(static),
        True,
    ).all(dim=-1)
    eligible = positive.any(dim=-1) & negative.any(dim=-1) & finite
    zero = fused_logits.sum() * 0.0
    if not bool(eligible.any()):
        return SafeLocalRouteObjective(
            loss=zero,
            nll_loss=zero,
            gain_loss=zero,
            safety_loss=zero,
            eligible_count=0,
            weak_count=0,
            strong_count=0,
            gain_violation_count=0,
            safety_violation_count=0,
        )

    masks = masks[eligible]
    positive = positive[eligible]
    negative = negative[eligible]
    fused = fused[eligible]
    static = static[eligible]
    expanded_weights = row_weights.to(device=fused_logits.device, dtype=fused_logits.dtype).index_select(
        0, source[eligible]
    )
    floor = torch.finfo(fused.dtype).min
    fused_all = fused.masked_fill(~masks, floor)
    fused_positive = fused.masked_fill(~positive, floor)
    row_nll = torch.logsumexp(fused_all.float(), dim=-1) - torch.logsumexp(
        fused_positive.float(), dim=-1
    )
    nll_loss = _weighted_mean(row_nll, expanded_weights)

    fused_margin = fused.masked_fill(~positive, floor).max(dim=-1).values - fused.masked_fill(
        ~negative, floor
    ).max(dim=-1).values
    static_margin = static.masked_fill(~positive, floor).max(dim=-1).values - static.masked_fill(
        ~negative, floor
    ).max(dim=-1).values
    strong = static_margin >= 0.0
    weak = ~strong
    gain_terms = torch.relu(static_margin[weak] + float(gain_margin) - fused_margin[weak])
    safety_terms = torch.relu(
        static_margin[strong] - float(safety_tolerance) - fused_margin[strong]
    )
    gain_loss = _weighted_mean(gain_terms, expanded_weights[weak]) if bool(weak.any()) else zero
    safety_loss = (
        _weighted_mean(safety_terms, expanded_weights[strong]) if bool(strong.any()) else zero
    )
    return SafeLocalRouteObjective(
        loss=nll_loss + gain_loss + safety_loss,
        nll_loss=nll_loss,
        gain_loss=gain_loss,
        safety_loss=safety_loss,
        eligible_count=int(eligible.sum().detach().cpu().item()),
        weak_count=int(weak.sum().detach().cpu().item()),
        strong_count=int(strong.sum().detach().cpu().item()),
        gain_violation_count=int((gain_terms > 0).sum().detach().cpu().item()),
        safety_violation_count=int((safety_terms > 0).sum().detach().cpu().item()),
    )
