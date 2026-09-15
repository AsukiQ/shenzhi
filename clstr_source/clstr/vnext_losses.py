from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FullPoolLossOutput:
    loss: torch.Tensor
    row_nll: torch.Tensor
    eligible_mask: torch.Tensor
    report: dict[str, Any]


@dataclass(frozen=True)
class CandidateListwiseLossOutput:
    loss: torch.Tensor
    row_nll: torch.Tensor
    eligible_mask: torch.Tensor
    report: dict[str, Any]


@dataclass(frozen=True)
class HardNegativeLossOutput:
    loss: torch.Tensor
    eligible_mask: torch.Tensor
    report: dict[str, Any]


@dataclass(frozen=True)
class DenseFullPoolObjectiveOutput:
    nll_loss: torch.Tensor
    hard_negative_loss: torch.Tensor
    eligible_mask: torch.Tensor
    report: dict[str, Any]


@dataclass(frozen=True)
class QueryExpertMixtureLossOutput:
    loss: torch.Tensor
    target: torch.Tensor
    eligible_mask: torch.Tensor
    report: dict[str, Any]


def dense_full_pool_nll_and_hard_negative(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    hard_negative_top_k: int = 32,
    hard_negative_margin: float = 0.1,
    hard_negative_positive_anchor: str = "set_logsumexp_v1",
) -> DenseFullPoolObjectiveOutput:
    """Compute Stage0 set NLL plus configurable positive coverage pressure.

    Retrieval rows can anchor the margin on the weakest required tool so one
    easy member cannot hide a future dependency. Static-route rows preserve set
    semantics because their positives may be alternative next decisions.
    """

    if logits.ndim != 2:
        raise ValueError("full-pool logits must be rank-2")
    positive, legal = _validate_masks(
        positive_mask,
        legal_mask,
        batch_size=int(logits.size(0)),
        skill_count=int(logits.size(1)),
    )
    positive = positive.to(device=logits.device)
    legal = legal.to(device=logits.device)
    top_k = int(hard_negative_top_k)
    margin = float(hard_negative_margin)
    if top_k <= 0:
        raise ValueError("hard-negative top_k must be positive")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("hard-negative margin must be finite and nonnegative")
    anchor_protocol = str(hard_negative_positive_anchor)
    if anchor_protocol not in {"set_logsumexp_v1", "weakest_legal_positive_v1"}:
        raise ValueError("unsupported hard-negative positive anchor")
    floor = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~legal, floor), dim=-1)
    positive_score = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    ceiling = torch.finfo(logits.dtype).max
    weakest_positive_score = logits.masked_fill(~positive, ceiling).amin(dim=-1)
    hard_negative_positive_score = (
        weakest_positive_score
        if anchor_protocol == "weakest_legal_positive_v1"
        else positive_score
    )
    negative = legal & ~positive
    eligible = (
        positive.any(dim=-1)
        & negative.any(dim=-1)
        & torch.isfinite(denominator)
        & torch.isfinite(positive_score)
        & torch.isfinite(weakest_positive_score)
    )
    row_nll = denominator.float() - positive_score.float()
    nll_loss = (
        row_nll[eligible].mean().to(dtype=logits.dtype)
        if bool(eligible.any().item())
        else logits.sum() * 0.0
    )
    k = min(top_k, int(logits.size(1)))
    mined = torch.topk(logits.masked_fill(~negative, floor), k=k, dim=-1).values
    valid_pairs = mined.ne(floor) & torch.isfinite(mined)
    pair_loss = F.relu(
        margin
        + mined.float()
        - hard_negative_positive_score.float().unsqueeze(-1)
    )
    pair_loss = torch.where(valid_pairs, pair_loss, torch.zeros_like(pair_loss))
    row_hard = pair_loss.sum(dim=-1) / valid_pairs.sum(dim=-1).clamp_min(1)
    hard_loss = (
        row_hard[eligible].mean().to(dtype=logits.dtype)
        if bool(eligible.any().item())
        else logits.sum() * 0.0
    )
    return DenseFullPoolObjectiveOutput(
        nll_loss=nll_loss,
        hard_negative_loss=hard_loss,
        eligible_mask=eligible,
        report={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "hard_negative_pair_count": int(
                valid_pairs[eligible].sum().detach().cpu().item()
            ),
            "hard_negative_top_k": top_k,
            "hard_negative_margin": margin,
            "hard_negative_positive_anchor": anchor_protocol,
            "positive_injection_count": 0,
        },
    )


def _validate_masks(
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    batch_size: int,
    skill_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (int(batch_size), int(skill_count))
    if tuple(positive_mask.shape) != expected or tuple(legal_mask.shape) != expected:
        raise ValueError("positive and legal masks must match [batch, skills]")
    legal = legal_mask.to(dtype=torch.bool)
    positive = positive_mask.to(dtype=torch.bool) & legal
    return positive, legal


def chunked_full_pool_multi_positive_nll(
    query: torch.Tensor,
    skill_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    temperature: torch.Tensor | float,
    chunk_size: int = 8192,
    row_weights: torch.Tensor | None = None,
    skill_embeddings_are_normalized: bool = False,
    normalize_query: bool = True,
) -> FullPoolLossOutput:
    if query.ndim != 2 or skill_embeddings.ndim != 2:
        raise ValueError("query and skill embeddings must be rank-2")
    if int(query.size(1)) != int(skill_embeddings.size(1)):
        raise ValueError("query and skill embedding dimensions must match")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("full-pool chunk_size must be positive")
    batch_size = int(query.size(0))
    skill_count = int(skill_embeddings.size(0))
    positive, legal = _validate_masks(
        positive_mask,
        legal_mask,
        batch_size=batch_size,
        skill_count=skill_count,
    )
    positive = positive.to(device=query.device)
    legal = legal.to(device=query.device)
    skills = (
        skill_embeddings.to(device=query.device, dtype=query.dtype)
        if skill_embeddings_are_normalized
        else F.normalize(skill_embeddings.float(), p=2, dim=-1).to(
            device=query.device,
            dtype=query.dtype,
        )
    )
    score_query = (
        F.normalize(query.float(), p=2, dim=-1).to(query.dtype)
        if normalize_query
        else query
    )
    scale = torch.as_tensor(temperature, device=query.device, dtype=query.dtype)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all().item()) or not bool((scale > 0).all().item()):
        raise ValueError("full-pool temperature must be one finite positive value")
    denominator: torch.Tensor | None = None
    numerator: torch.Tensor | None = None
    for start in range(0, skill_count, chunk_size):
        end = min(skill_count, start + chunk_size)
        logits = scale * score_query @ skills[start:end].t()
        legal_chunk = legal[:, start:end]
        positive_chunk = positive[:, start:end]
        floor = torch.finfo(logits.dtype).min
        denominator_chunk = torch.logsumexp(logits.masked_fill(~legal_chunk, floor), dim=-1)
        numerator_chunk = torch.logsumexp(logits.masked_fill(~positive_chunk, floor), dim=-1)
        denominator = (
            denominator_chunk
            if denominator is None
            else torch.logaddexp(denominator, denominator_chunk)
        )
        numerator = (
            numerator_chunk
            if numerator is None
            else torch.logaddexp(numerator, numerator_chunk)
        )
    if denominator is None or numerator is None:
        raise ValueError("full-pool skill table must be non-empty")
    has_positive = positive.any(dim=-1)
    has_negative = (legal & ~positive).any(dim=-1)
    finite = torch.isfinite(denominator) & torch.isfinite(numerator)
    eligible = has_positive & has_negative & finite
    row_nll = denominator - numerator
    if row_weights is None:
        weights = torch.ones(batch_size, device=query.device, dtype=torch.float32)
    else:
        if row_weights.ndim != 1 or int(row_weights.numel()) != batch_size:
            raise ValueError("row_weights must have one value per query")
        weights = row_weights.to(device=query.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or bool((weights < 0).any().item()):
            raise ValueError("row_weights must be finite and nonnegative")
    if bool(eligible.any().item()):
        eligible_weights = weights[eligible]
        if not bool((eligible_weights.sum() > 0).item()):
            raise ValueError("eligible full-pool rows have zero total weight")
        loss = (row_nll[eligible].float() * eligible_weights).sum() / eligible_weights.sum()
        loss = loss.to(dtype=query.dtype)
    else:
        loss = query.sum() * 0.0
    return FullPoolLossOutput(
        loss=loss,
        row_nll=row_nll,
        eligible_mask=eligible,
        report={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "no_positive_rows": int((~has_positive).sum().detach().cpu().item()),
            "no_negative_rows": int((~has_negative).sum().detach().cpu().item()),
            "nonfinite_rows": int((~finite).sum().detach().cpu().item()),
            "chunk_size": chunk_size,
            "skill_count": skill_count,
            "normalize_query": bool(normalize_query),
        },
    )


def chunked_full_pool_hard_negative_margin(
    query: torch.Tensor,
    skill_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    temperature: torch.Tensor | float,
    top_k: int = 32,
    margin: float = 0.1,
    chunk_size: int = 8192,
    skill_embeddings_are_normalized: bool = False,
    normalize_query: bool = True,
) -> HardNegativeLossOutput:
    """Mine legal non-positive TopK logits without materializing the full pool."""

    if query.ndim != 2 or skill_embeddings.ndim != 2:
        raise ValueError("query and skill embeddings must be rank-2")
    if int(query.size(1)) != int(skill_embeddings.size(1)):
        raise ValueError("query and skill embedding dimensions must match")
    top_k = int(top_k)
    chunk_size = int(chunk_size)
    margin = float(margin)
    if top_k <= 0 or chunk_size <= 0:
        raise ValueError("hard-negative top_k and chunk_size must be positive")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("hard-negative margin must be finite and nonnegative")
    batch_size = int(query.size(0))
    skill_count = int(skill_embeddings.size(0))
    positive, legal = _validate_masks(
        positive_mask,
        legal_mask,
        batch_size=batch_size,
        skill_count=skill_count,
    )
    positive = positive.to(device=query.device)
    legal = legal.to(device=query.device)
    skills = (
        skill_embeddings.to(device=query.device, dtype=query.dtype)
        if skill_embeddings_are_normalized
        else F.normalize(skill_embeddings.float(), p=2, dim=-1).to(
            device=query.device,
            dtype=query.dtype,
        )
    )
    score_query = (
        F.normalize(query.float(), p=2, dim=-1).to(query.dtype)
        if normalize_query
        else query
    )
    scale = torch.as_tensor(temperature, device=query.device, dtype=query.dtype)
    if (
        scale.numel() != 1
        or not bool(torch.isfinite(scale).all().item())
        or not bool((scale > 0).all().item())
    ):
        raise ValueError("hard-negative temperature must be finite and positive")
    positive_score: torch.Tensor | None = None
    mined = query.new_empty((batch_size, 0))
    floor = torch.finfo(query.dtype).min
    for start in range(0, skill_count, chunk_size):
        end = min(skill_count, start + chunk_size)
        logits = scale * score_query @ skills[start:end].t()
        positive_chunk = positive[:, start:end]
        negative_chunk = legal[:, start:end] & ~positive_chunk
        chunk_positive = torch.logsumexp(
            logits.masked_fill(~positive_chunk, floor),
            dim=-1,
        )
        positive_score = (
            chunk_positive
            if positive_score is None
            else torch.logaddexp(positive_score, chunk_positive)
        )
        width = min(top_k, end - start)
        chunk_negative = torch.topk(
            logits.masked_fill(~negative_chunk, floor),
            k=width,
            dim=-1,
        ).values
        combined = torch.cat((mined, chunk_negative), dim=-1)
        mined = torch.topk(
            combined,
            k=min(top_k, int(combined.size(1))),
            dim=-1,
        ).values
    if positive_score is None:
        raise ValueError("full-pool skill table must be non-empty")
    has_positive = positive.any(dim=-1)
    has_negative = (legal & ~positive).any(dim=-1)
    valid_pairs = mined.ne(floor) & torch.isfinite(mined)
    eligible = has_positive & has_negative & torch.isfinite(positive_score) & valid_pairs.any(dim=-1)
    pair_loss = F.relu(
        margin + mined.float() - positive_score.float().unsqueeze(-1)
    )
    pair_loss = torch.where(valid_pairs, pair_loss, torch.zeros_like(pair_loss))
    row_loss = pair_loss.sum(dim=-1) / valid_pairs.sum(dim=-1).clamp_min(1)
    loss = (
        row_loss[eligible].mean().to(dtype=query.dtype)
        if bool(eligible.any().item())
        else query.sum() * 0.0
    )
    return HardNegativeLossOutput(
        loss=loss,
        eligible_mask=eligible,
        report={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "pair_count": int(valid_pairs[eligible].sum().detach().cpu().item()),
            "top_k": top_k,
            "margin": margin,
            "chunk_size": chunk_size,
            "skill_count": skill_count,
            "normalize_query": bool(normalize_query),
        },
    )


def natural_candidate_multi_positive_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> CandidateListwiseLossOutput:
    """Listwise loss on immutable natural support; never changes membership."""

    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("candidate loss tensors must match [batch, candidates]")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    negative = valid & ~positive
    eligible = positive.any(dim=-1) & negative.any(dim=-1)
    floor = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~valid, floor), dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    row_nll = denominator - numerator
    loss = (
        row_nll[eligible].float().mean().to(dtype=logits.dtype)
        if bool(eligible.any().item())
        else logits.masked_fill(~valid, 0.0).sum() * 0.0
    )
    return CandidateListwiseLossOutput(
        loss=loss,
        row_nll=row_nll,
        eligible_mask=eligible,
        report={
            "row_count": int(logits.size(0)),
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "natural_miss_rows": int((~positive.any(dim=-1)).sum().detach().cpu().item()),
            "no_negative_rows": int((~negative.any(dim=-1)).sum().detach().cpu().item()),
            "positive_injection_count": 0,
        },
    )


def natural_candidate_topk_coverage_loss(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    margin: float = 0.05,
    recoverable_weight: float = 4.0,
    listwise_weight: float = 0.05,
) -> CandidateListwiseLossOutput:
    """Optimize natural-support Top-K coverage while retaining weak rank pressure."""

    if (
        logits.ndim != 2
        or base_logits.shape != logits.shape
        or positive_mask.shape != logits.shape
        or valid_mask.shape != logits.shape
    ):
        raise ValueError("candidate coverage tensors must match [batch, candidates]")
    width = int(logits.size(1))
    top_k = min(int(k), width)
    if top_k <= 0:
        raise ValueError("candidate coverage k must be positive")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("candidate coverage margin must be finite and nonnegative")
    if (
        not math.isfinite(float(recoverable_weight))
        or float(recoverable_weight) < 1.0
    ):
        raise ValueError("recoverable candidate weight must be finite and at least one")
    if (
        not math.isfinite(float(listwise_weight))
        or float(listwise_weight) < 0.0
    ):
        raise ValueError("candidate listwise weight must be finite and nonnegative")

    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    negative = valid & ~positive
    eligible = positive.any(dim=-1) & negative.any(dim=-1)
    floor = torch.finfo(logits.dtype).min

    direct_positions = torch.topk(
        base_logits.to(device=logits.device, dtype=logits.dtype).masked_fill(
            ~valid,
            floor,
        ),
        k=top_k,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    direct_hit = positive.gather(1, direct_positions).any(dim=-1)
    recoverable = eligible & ~direct_hit

    best_positive = logits.masked_fill(~positive, floor).amax(dim=-1)
    negative_boundary = torch.topk(
        logits.masked_fill(~negative, floor),
        k=top_k,
        dim=-1,
        largest=True,
        sorted=True,
    ).values[:, -1]
    boundary_loss = F.softplus(
        negative_boundary.float() + float(margin) - best_positive.float()
    )

    denominator = torch.logsumexp(logits.masked_fill(~valid, floor), dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    listwise_nll = (denominator - numerator).float()
    row_loss = boundary_loss + float(listwise_weight) * listwise_nll
    row_weights = torch.where(
        recoverable,
        torch.full_like(row_loss, float(recoverable_weight)),
        torch.ones_like(row_loss),
    )
    if bool(eligible.any().item()):
        loss = (
            (row_loss[eligible] * row_weights[eligible]).sum()
            / row_weights[eligible].sum().clamp_min(1.0)
        ).to(dtype=logits.dtype)
    else:
        loss = logits.masked_fill(~valid, 0.0).sum() * 0.0
    return CandidateListwiseLossOutput(
        loss=loss,
        row_nll=row_loss,
        eligible_mask=eligible,
        report={
            "row_count": int(logits.size(0)),
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "natural_miss_rows": int((~positive.any(dim=-1)).sum().detach().cpu().item()),
            "no_negative_rows": int((~negative.any(dim=-1)).sum().detach().cpu().item()),
            "direct_topk_hit_rows": int(
                (eligible & direct_hit).sum().detach().cpu().item()
            ),
            "recoverable_rows": int(recoverable.sum().detach().cpu().item()),
            "mean_boundary_loss": float(
                boundary_loss[eligible].mean().detach().cpu().item()
                if bool(eligible.any().item())
                else 0.0
            ),
            "mean_listwise_nll": float(
                listwise_nll[eligible].mean().detach().cpu().item()
                if bool(eligible.any().item())
                else 0.0
            ),
            "top_k": int(top_k),
            "margin": float(margin),
            "recoverable_weight": float(recoverable_weight),
            "listwise_weight": float(listwise_weight),
            "positive_injection_count": 0,
        },
    )


def ranking_utility(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != positive_mask.shape or logits.shape != legal_mask.shape:
        raise ValueError("utility tensors must have matching [batch, candidates] shapes")
    legal = legal_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & legal
    negative = legal & ~positive
    if bool((~positive.any(dim=-1)).any().item()):
        raise ValueError("every utility row requires a legal positive")
    if bool((~negative.any(dim=-1)).any().item()):
        raise ValueError("every utility row requires a legal negative")
    floor = torch.finfo(logits.dtype).min
    return torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1) - torch.logsumexp(
        logits.masked_fill(~negative, floor),
        dim=-1,
    )


def paired_counterfactual_margin_loss(
    own_utility: torch.Tensor,
    swapped_utility: torch.Tensor,
    *,
    margin: float,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if own_utility.shape != swapped_utility.shape or own_utility.ndim != 1:
        raise ValueError("paired utilities must be matching rank-1 tensors")
    eligible = (
        torch.ones_like(own_utility, dtype=torch.bool)
        if eligible_mask is None
        else eligible_mask.to(device=own_utility.device, dtype=torch.bool)
    )
    if eligible.shape != own_utility.shape:
        raise ValueError("counterfactual eligibility must match utilities")
    if not bool(eligible.any().item()):
        return own_utility.sum() * 0.0
    return F.softplus(float(margin) - own_utility[eligible] + swapped_utility[eligible]).mean()


def head_local_counterfactual_loss(
    recall_own: torch.Tensor,
    recall_swapped: torch.Tensor,
    route_own: torch.Tensor,
    route_swapped: torch.Tensor,
    *,
    margin: float,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    recall = paired_counterfactual_margin_loss(
        recall_own,
        recall_swapped,
        margin=margin,
        eligible_mask=eligible_mask,
    )
    route = paired_counterfactual_margin_loss(
        route_own,
        route_swapped,
        margin=margin,
        eligible_mask=eligible_mask,
    )
    return 0.5 * (recall + route)


def no_regret_loss(
    static_utility: torch.Tensor,
    dynamic_utility: torch.Tensor,
    *,
    tolerance: float,
    eligible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if static_utility.shape != dynamic_utility.shape or static_utility.ndim != 1:
        raise ValueError("no-regret utilities must be matching rank-1 tensors")
    eligible = (
        torch.ones_like(static_utility, dtype=torch.bool)
        if eligible_mask is None
        else eligible_mask.to(device=static_utility.device, dtype=torch.bool)
    )
    if eligible.shape != static_utility.shape:
        raise ValueError("no-regret eligibility must match utilities")
    if not bool(eligible.any().item()):
        return static_utility.sum() * 0.0
    return F.relu(
        static_utility[eligible] - float(tolerance) - dynamic_utility[eligible]
    ).mean()


def query_expert_mixture_target_loss(
    mixture_probability: torch.Tensor,
    static_utility: torch.Tensor,
    raw_dynamic_utility: torch.Tensor,
    *,
    static_supported_mask: torch.Tensor,
    eligible_mask: torch.Tensor | None = None,
    utility_scale: float = 0.5,
) -> QueryExpertMixtureLossOutput:
    """Calibrate a query-level static/dynamic selector from detached benefit.

    The target rises smoothly from zero only when the raw dynamic expert
    actually improves over the static expert. A static-support miss is reported
    but never forced to target one: candidate extension must demonstrate route
    utility before deployment. Positive-
    and non-positive benefit rows are averaged separately so a mostly-static
    batch cannot collapse the selector by class imbalance.
    """

    if (
        mixture_probability.ndim != 1
        or static_utility.shape != mixture_probability.shape
        or raw_dynamic_utility.shape != mixture_probability.shape
    ):
        raise ValueError("mixture probabilities and utilities must be matching vectors")
    if static_supported_mask.shape != mixture_probability.shape:
        raise ValueError("static-supported mask must match mixture rows")
    scale = float(utility_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mixture utility scale must be finite and positive")
    eligible = (
        torch.ones_like(mixture_probability, dtype=torch.bool)
        if eligible_mask is None
        else eligible_mask.to(device=mixture_probability.device, dtype=torch.bool)
    )
    if eligible.shape != mixture_probability.shape:
        raise ValueError("mixture eligibility must match probabilities")
    static_supported = static_supported_mask.to(
        device=mixture_probability.device,
        dtype=torch.bool,
    )
    benefit = (raw_dynamic_utility.detach() - static_utility.detach()).float()
    smooth_positive = 1.0 - torch.exp(-F.relu(benefit) / scale)
    target = smooth_positive.clamp(0.0, 1.0)
    probability = mixture_probability.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    # Probability-form BCE is rejected by CUDA autocast even after an explicit
    # FP32 cast. Compute the identical stable expression directly in FP32; the
    # selector still receives gradients through ``probability``.
    row_loss = -(
        target * torch.log(probability)
        + (1.0 - target) * torch.log1p(-probability)
    )
    positive_benefit = eligible & target.gt(0.0)
    static_preferred = eligible & ~target.gt(0.0)
    terms: list[torch.Tensor] = []
    if bool(positive_benefit.any().item()):
        terms.append(row_loss[positive_benefit].mean())
    if bool(static_preferred.any().item()):
        terms.append(row_loss[static_preferred].mean())
    loss = (
        torch.stack(terms).mean().to(dtype=mixture_probability.dtype)
        if terms
        else mixture_probability.sum() * 0.0
    )
    return QueryExpertMixtureLossOutput(
        loss=loss,
        target=target.detach(),
        eligible_mask=eligible.detach(),
        report={
            "eligible_rows": int(eligible.sum().detach().cpu().item()),
            "positive_benefit_rows": int(
                positive_benefit.sum().detach().cpu().item()
            ),
            "static_preferred_rows": int(
                static_preferred.sum().detach().cpu().item()
            ),
            "static_support_miss_rows": int(
                (eligible & ~static_supported).sum().detach().cpu().item()
            ),
            "mean_target": float(
                target[eligible].mean().detach().cpu().item()
                if bool(eligible.any().item())
                else 0.0
            ),
            "utility_scale": scale,
        },
    )
