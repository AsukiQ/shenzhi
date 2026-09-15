from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


UNIFIED_EXPERT_NAMES = ("foundation", "adapted_static", "recurrent_memory")
UNIFIED_ROUTER_SCHEMA = "clstr_vnext_unified_three_expert_router_v6"
UNIFIED_ROUTING_MODE_SOFT = "soft_probability_mixture"
UNIFIED_ROUTING_MODE_SPARSE = "sparse_top1_expert"
UNIFIED_ROUTING_MODES = (
    UNIFIED_ROUTING_MODE_SOFT,
    UNIFIED_ROUTING_MODE_SPARSE,
)
SUPPORT_AWARE_ANCHOR_PROTOCOL = (
    "incomplete_support_dynamic_complete_support_top4_foundation_equal_rrf_tail_v2"
)
SUPPORT_AWARE_FOUNDATION_PREFIX_K = 4
UNIFIED_ROUTER_FEATURE_NAMES = (
    "foundation_margin",
    "foundation_confidence",
    "adapted_static_margin",
    "adapted_static_confidence",
    "recurrent_margin",
    "recurrent_confidence",
    "foundation_static_js",
    "foundation_recurrent_js",
    "static_recurrent_js",
    "foundation_static_top1_agreement",
    "foundation_recurrent_top1_agreement",
    "static_recurrent_top1_agreement",
    "recurrent_static_residual_rms",
    "observation_correction_fraction",
    "log_history_depth",
    "log_legal_pool_size",
    "natural_support_fraction",
)


@dataclass(frozen=True)
class UnifiedThreeExpertScores:
    logits: torch.Tensor
    expert_weights: torch.Tensor
    router_probabilities: torch.Tensor
    features: torch.Tensor
    foundation_logits: torch.Tensor
    adapted_static_logits: torch.Tensor
    recurrent_memory_logits: torch.Tensor


class UnifiedThreeExpertRouter(nn.Module):
    """Sample-level semantic/static/memory router with no source identity input."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        feature_dim = len(UNIFIED_ROUTER_FEATURE_NAMES)
        self.norm = nn.LayerNorm(feature_dim)
        self.hidden = nn.Linear(feature_dim, int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), len(UNIFIED_EXPERT_NAMES))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.hidden.reset_parameters()
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        # Preserve the immutable semantic foundation by default. Adapted-static
        # and recurrent experts must earn sparse override from calibrated
        # observable evidence.
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor((1.0, -1.0, -1.0)))

    def forward(
        self,
        features: torch.Tensor,
        *,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.size(-1)) != len(
            UNIFIED_ROUTER_FEATURE_NAMES
        ):
            raise ValueError(
                "unified router features must have shape "
                f"[batch, {len(UNIFIED_ROUTER_FEATURE_NAMES)}]"
            )
        history = history_mask.to(device=features.device, dtype=torch.bool)
        if history.ndim != 1 or int(history.numel()) != int(features.size(0)):
            raise ValueError("unified router history mask must have one value per row")
        hidden = F.gelu(self.hidden(self.norm(features)))
        expert_logits = self.output(hidden).float()
        # A row without an executed prefix cannot use the recurrent expert.
        expert_logits = torch.cat(
            (
                expert_logits[:, :2],
                expert_logits[:, 2:3].masked_fill(~history.unsqueeze(-1), -1.0e4),
            ),
            dim=-1,
        )
        return torch.softmax(expert_logits, dim=-1).to(dtype=features.dtype)


def _masked_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.to(device=logits.device, dtype=torch.bool)
    if logits.ndim != 2 or valid.shape != logits.shape:
        raise ValueError("expert logits and masks must match [batch, candidates]")
    if bool((~valid.any(dim=-1)).any().item()):
        raise ValueError("every unified expert requires at least one valid candidate")
    masked = logits.float().masked_fill(~valid, float("-inf"))
    probability = torch.softmax(masked, dim=-1).masked_fill(~valid, 0.0)
    log_probability = torch.log(probability.clamp_min(1.0e-12))
    entropy = -(probability * log_probability).sum(dim=-1)
    count = valid.sum(dim=-1).float().clamp_min(1.0)
    normalized_entropy = entropy / count.log().clamp_min(1.0)
    top_width = min(2, int(logits.size(1)))
    top = torch.topk(masked, k=top_width, dim=-1).values
    margin = (
        torch.where(
            count.ge(2),
            top[:, 0] - top[:, 1],
            torch.zeros_like(top[:, 0]),
        )
        if top_width == 2
        else torch.zeros_like(top[:, 0])
    )
    finite = torch.where(valid, logits.float(), torch.zeros_like(logits.float()))
    mean = finite.sum(dim=-1) / count
    variance = (
        torch.where(valid, (logits.float() - mean.unsqueeze(-1)).pow(2), 0.0).sum(
            dim=-1
        )
        / count
    )
    scale = variance.sqrt().clamp_min(1.0e-4)
    standardized_margin = margin / scale
    top_index = masked.argmax(dim=-1)
    return probability, standardized_margin, 1.0 - normalized_entropy, top_index


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    mixture = 0.5 * (left + right)
    left_kl = (
        left
        * (torch.log(left.clamp_min(1.0e-12)) - torch.log(mixture.clamp_min(1.0e-12)))
    ).sum(dim=-1)
    right_kl = (
        right
        * (torch.log(right.clamp_min(1.0e-12)) - torch.log(mixture.clamp_min(1.0e-12)))
    ).sum(dim=-1)
    return 0.5 * (left_kl + right_kl) / math.log(2.0)


def unified_router_features(
    foundation_logits: torch.Tensor,
    adapted_static_logits: torch.Tensor,
    recurrent_memory_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    static_support_mask: torch.Tensor | None,
    history_depth: torch.Tensor,
    observation_correction_count: torch.Tensor,
    legal_pool_size: torch.Tensor,
) -> torch.Tensor:
    if not (
        foundation_logits.shape
        == adapted_static_logits.shape
        == recurrent_memory_logits.shape
        == valid_mask.shape
    ):
        raise ValueError("unified expert logits and validity must share one shape")
    valid = valid_mask.to(device=foundation_logits.device, dtype=torch.bool)
    static_valid = (
        valid
        if static_support_mask is None
        else static_support_mask.to(device=foundation_logits.device, dtype=torch.bool)
        & valid
    )
    foundation_p, foundation_margin, foundation_conf, foundation_top = (
        _masked_distribution(foundation_logits, valid)
    )
    static_p, static_margin, static_conf, static_top = _masked_distribution(
        adapted_static_logits,
        static_valid,
    )
    recurrent_p, recurrent_margin, recurrent_conf, recurrent_top = (
        _masked_distribution(recurrent_memory_logits, valid)
    )
    depth = history_depth.to(device=foundation_logits.device, dtype=torch.float32)
    correction_count = observation_correction_count.to(
        device=foundation_logits.device,
        dtype=torch.float32,
    )
    pool = legal_pool_size.to(device=foundation_logits.device, dtype=torch.float32)
    if not (
        depth.ndim == correction_count.ndim == pool.ndim == 1
        and depth.shape == correction_count.shape == pool.shape
    ):
        raise ValueError(
            "history depth, observation corrections, and legal pool size "
            "must be matching vectors"
        )
    if int(depth.numel()) != int(foundation_logits.size(0)):
        raise ValueError("unified route metadata must have one value per row")
    if (
        bool((depth < 0).any().item())
        or bool((correction_count < 0).any().item())
        or bool((pool <= 0).any().item())
    ):
        raise ValueError(
            "unified route depth/corrections must be nonnegative and pool positive"
        )
    if not (
        bool(torch.isfinite(depth).all().item())
        and bool(torch.isfinite(correction_count).all().item())
        and bool(torch.isfinite(pool).all().item())
    ):
        raise ValueError("unified route metadata must be finite")
    if bool((correction_count > depth).any().item()):
        raise ValueError(
            "observation correction count cannot exceed replay history depth"
        )
    support_count = valid.sum(dim=-1).float()
    shared_static_count = static_valid.sum(dim=-1).float().clamp_min(1.0)
    # Adapted-static logits are intentionally filled with the dtype minimum on
    # memory-only candidate extras.  Comparing recurrent logits against that
    # sentinel over the full union overflows when squared.  Reliability must
    # measure the recurrent residual only where both experts have real scores.
    recurrent_static_residual = torch.where(
        static_valid,
        recurrent_memory_logits.float() - adapted_static_logits.float(),
        0.0,
    )
    recurrent_static_rms = (
        recurrent_static_residual.pow(2).sum(dim=-1)
        / shared_static_count
    ).sqrt()
    recurrent_scale = torch.where(
        static_valid,
        recurrent_memory_logits.float(),
        0.0,
    ).pow(2).sum(dim=-1).div(shared_static_count).sqrt().clamp_min(1.0e-4)
    features = torch.stack(
        (
            foundation_margin,
            foundation_conf,
            static_margin,
            static_conf,
            recurrent_margin,
            recurrent_conf,
            _js_divergence(foundation_p, static_p),
            _js_divergence(foundation_p, recurrent_p),
            _js_divergence(static_p, recurrent_p),
            foundation_top.eq(static_top).float(),
            foundation_top.eq(recurrent_top).float(),
            static_top.eq(recurrent_top).float(),
            recurrent_static_rms / recurrent_scale,
            (
                correction_count / depth.clamp_min(1.0)
            ).clamp(min=0.0, max=1.0),
            (torch.log1p(depth) / math.log1p(16.0)).clamp(max=2.0),
            (torch.log1p(pool) / math.log1p(67557.0)).clamp(max=2.0),
            (support_count / pool).clamp(min=0.0, max=1.0),
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("unified router produced non-finite reliability features")
    return features


def route_with_unified_three_experts(
    router: UnifiedThreeExpertRouter,
    foundation_logits: torch.Tensor,
    adapted_static_logits: torch.Tensor,
    recurrent_memory_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    static_support_mask: torch.Tensor | None,
    history_depth: torch.Tensor,
    observation_correction_count: torch.Tensor,
    legal_pool_size: torch.Tensor,
    routing_mode: str = UNIFIED_ROUTING_MODE_SOFT,
) -> UnifiedThreeExpertScores:
    if routing_mode not in UNIFIED_ROUTING_MODES:
        raise ValueError(f"unsupported unified routing mode: {routing_mode}")
    valid = valid_mask.to(device=foundation_logits.device, dtype=torch.bool)
    static_valid = (
        valid
        if static_support_mask is None
        else static_support_mask.to(device=foundation_logits.device, dtype=torch.bool)
        & valid
    )
    features = unified_router_features(
        foundation_logits,
        adapted_static_logits,
        recurrent_memory_logits,
        valid,
        static_support_mask=static_valid,
        history_depth=history_depth,
        observation_correction_count=observation_correction_count,
        legal_pool_size=legal_pool_size,
    )
    probabilities = router(features, history_mask=history_depth.gt(0))
    if routing_mode == UNIFIED_ROUTING_MODE_SPARSE:
        selected = probabilities.argmax(dim=-1)
        weights = F.one_hot(
            selected,
            num_classes=len(UNIFIED_EXPERT_NAMES),
        ).to(dtype=probabilities.dtype)
        score_floor = torch.finfo(torch.float32).min
        expert_logits = torch.stack(
            (
                foundation_logits.float().masked_fill(~valid, score_floor),
                adapted_static_logits.float().masked_fill(
                    ~static_valid, score_floor
                ),
                recurrent_memory_logits.float().masked_fill(~valid, score_floor),
            ),
            dim=1,
        )
        logits = expert_logits.gather(
            1,
            selected.view(-1, 1, 1).expand(-1, 1, expert_logits.size(-1)),
        ).squeeze(1)
    else:
        weights = probabilities
        foundation_p = torch.softmax(
            foundation_logits.float().masked_fill(~valid, float("-inf")), dim=-1
        ).masked_fill(~valid, 0.0)
        static_p = torch.softmax(
            adapted_static_logits.float().masked_fill(~static_valid, float("-inf")),
            dim=-1,
        ).masked_fill(~static_valid, 0.0)
        recurrent_p = torch.softmax(
            recurrent_memory_logits.float().masked_fill(~valid, float("-inf")), dim=-1
        ).masked_fill(~valid, 0.0)
        mixture = (
            weights[:, 0:1].float() * foundation_p
            + weights[:, 1:2].float() * static_p
            + weights[:, 2:3].float() * recurrent_p
        )
        logits = torch.log(mixture.clamp_min(1.0e-12)).masked_fill(
            ~valid,
            torch.finfo(torch.float32).min,
        )
    return UnifiedThreeExpertScores(
        logits=logits,
        expert_weights=weights,
        router_probabilities=probabilities,
        features=features,
        foundation_logits=foundation_logits,
        adapted_static_logits=adapted_static_logits,
        recurrent_memory_logits=recurrent_memory_logits,
    )


def _reciprocal_rank_scores(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    masked = logits.float().masked_fill(~valid, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    positions = torch.arange(
        1,
        int(logits.size(-1)) + 1,
        device=logits.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand_as(masked)
    ranks = torch.empty_like(masked).scatter(1, order, positions)
    return ranks.reciprocal().masked_fill(~valid, 0.0)


def route_with_support_aware_anchor(
    foundation_logits: torch.Tensor,
    adapted_static_logits: torch.Tensor,
    recurrent_memory_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    static_support_mask: torch.Tensor | None,
    history_depth: torch.Tensor,
    observation_correction_count: torch.Tensor,
    legal_pool_size: torch.Tensor,
) -> UnifiedThreeExpertScores:
    """Use memory for incomplete support and a semantic anchor for complete support.

    Candidate-support completeness is observable for every sample and does not
    encode benchmark, source, or interface identity.  Complete-support rows
    preserve the foundation Top-4 exactly, then rank the remaining candidates
    by equal reciprocal-rank evidence from all three experts.  Incomplete-
    support rows use recurrent memory after an executed prefix and the adapted
    static expert otherwise.
    """

    if not (
        foundation_logits.shape
        == adapted_static_logits.shape
        == recurrent_memory_logits.shape
        == valid_mask.shape
    ):
        raise ValueError("support-aware expert logits and validity must share one shape")
    valid = valid_mask.to(device=foundation_logits.device, dtype=torch.bool)
    static_valid = (
        valid
        if static_support_mask is None
        else static_support_mask.to(device=foundation_logits.device, dtype=torch.bool)
        & valid
    )
    depth = history_depth.to(device=foundation_logits.device, dtype=torch.float32)
    has_history = depth.gt(0)
    effective_recurrent_logits = torch.where(
        has_history.unsqueeze(-1),
        recurrent_memory_logits,
        adapted_static_logits,
    )
    features = unified_router_features(
        foundation_logits,
        adapted_static_logits,
        effective_recurrent_logits,
        valid,
        static_support_mask=static_valid,
        history_depth=history_depth,
        observation_correction_count=observation_correction_count,
        legal_pool_size=legal_pool_size,
    )
    pool = legal_pool_size.to(device=foundation_logits.device, dtype=torch.float32)
    support_count = valid.sum(dim=-1).to(dtype=torch.float32)
    if bool((support_count > pool).any().item()):
        raise ValueError("natural candidate support cannot exceed the legal pool")
    complete_support = support_count.eq(pool)
    score_floor = torch.finfo(torch.float32).min
    output = torch.full_like(foundation_logits.float(), score_floor)
    expert_weights = torch.zeros(
        (int(valid.size(0)), len(UNIFIED_EXPERT_NAMES)),
        device=foundation_logits.device,
        dtype=torch.float32,
    )

    incomplete = ~complete_support
    incomplete_recurrent = incomplete & has_history
    incomplete_static = incomplete & ~has_history
    if bool(incomplete_recurrent.any().item()):
        output[incomplete_recurrent] = recurrent_memory_logits.float()[
            incomplete_recurrent
        ].masked_fill(~valid[incomplete_recurrent], score_floor)
        expert_weights[incomplete_recurrent, 2] = 1.0
    if bool(incomplete_static.any().item()):
        output[incomplete_static] = adapted_static_logits.float()[
            incomplete_static
        ].masked_fill(~static_valid[incomplete_static], score_floor)
        expert_weights[incomplete_static, 1] = 1.0

    if bool(complete_support.any().item()):
        foundation_rr = _reciprocal_rank_scores(foundation_logits, valid)
        static_rr = _reciprocal_rank_scores(adapted_static_logits, static_valid)
        recurrent_rr = torch.where(
            has_history.unsqueeze(-1),
            _reciprocal_rank_scores(recurrent_memory_logits, valid),
            static_rr,
        )
        tail_score = foundation_rr + static_rr + recurrent_rr
        complete_indices = torch.nonzero(complete_support, as_tuple=False).flatten()
        for row_index in complete_indices:
            row_valid = valid[row_index]
            valid_indices = torch.nonzero(row_valid, as_tuple=False).flatten()
            foundation_order = valid_indices.index_select(
                0,
                torch.argsort(
                    foundation_logits[row_index].float().index_select(
                        0,
                        valid_indices,
                    ),
                    descending=True,
                    stable=True,
                ),
            )
            prefix_width = min(
                SUPPORT_AWARE_FOUNDATION_PREFIX_K,
                int(valid_indices.numel()),
            )
            prefix = foundation_order[:prefix_width]
            remaining_mask = row_valid.clone()
            remaining_mask[prefix] = False
            remaining = torch.nonzero(
                remaining_mask,
                as_tuple=False,
            ).flatten()
            remaining = remaining.index_select(
                0,
                torch.argsort(
                    tail_score[row_index].index_select(0, remaining),
                    descending=True,
                    stable=True,
                ),
            )
            final_order = torch.cat((prefix, remaining), dim=0)
            ranking_score = torch.arange(
                int(final_order.numel()),
                0,
                -1,
                device=foundation_logits.device,
                dtype=torch.float32,
            )
            output[row_index].scatter_(0, final_order, ranking_score)
        complete_with_history = complete_support & has_history
        complete_without_history = complete_support & ~has_history
        expert_weights[complete_with_history] = 1.0 / len(UNIFIED_EXPERT_NAMES)
        # The recurrent branch is causally unavailable without an executed
        # prefix. Its RRF term is therefore an exact adapted-static fallback,
        # and the contribution report must not claim recurrent-memory use.
        expert_weights[complete_without_history, 0] = 1.0 / 3.0
        expert_weights[complete_without_history, 1] = 2.0 / 3.0

    output = output.masked_fill(~valid, score_floor)
    return UnifiedThreeExpertScores(
        logits=output,
        expert_weights=expert_weights,
        router_probabilities=expert_weights,
        features=features,
        foundation_logits=foundation_logits,
        adapted_static_logits=adapted_static_logits,
        recurrent_memory_logits=recurrent_memory_logits,
    )
