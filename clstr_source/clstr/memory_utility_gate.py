from __future__ import annotations

import math

import torch


RELIABILITY_MODES = {
    "candidate_admission_residual",
    "cmc",
    "cmc_candidate_provenance",
    "static",
    "dynamic",
    "fixed_alpha",
    "heuristic",
    "learned",
    "causal_gate",
}
RELIABILITY_FEATURE_SCHEMA_VERSION = "memory_utility_features_v1"
HEURISTIC_ALPHA_VERSION = "memory_utility_heuristic_v1"
MEMORY_UTILITY_FEATURE_NAMES = (
    "static_normalized_entropy",
    "dynamic_normalized_entropy",
    "static_standardized_gap",
    "dynamic_standardized_gap",
    "top1_agreement",
    "js_divergence",
    "mean_abs_standardized_logit_difference",
    "memory_cosine_distance",
    "memory_relative_l2_difference",
    "log_normalized_causal_update_count",
    "log_normalized_valid_candidate_count",
)


class MemoryUtilityGate(torch.nn.Module):
    def __init__(
        self,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        width = len(MEMORY_UTILITY_FEATURE_NAMES)
        mean = (
            torch.zeros(width, dtype=torch.float32)
            if feature_mean is None
            else feature_mean.float()
        )
        scale = (
            torch.ones(width, dtype=torch.float32)
            if feature_scale is None
            else feature_scale.float()
        )
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("memory utility normalization must match feature schema")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or bool((scale <= 0).any())
        ):
            raise ValueError(
                "memory utility normalization must be finite with positive scale"
            )
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.size(1)) != len(
            MEMORY_UTILITY_FEATURE_NAMES
        ):
            raise ValueError("memory utility gate features do not match schema")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.net(normalized).squeeze(-1)


class AnchoredMemoryUtilityGate(torch.nn.Module):
    """Convert harmful-memory probability into a baseline-preserving alpha."""

    def __init__(
        self,
        harm_gate: MemoryUtilityGate,
        *,
        alpha_base: float,
    ) -> None:
        super().__init__()
        value = float(alpha_base)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("anchored memory alpha_base must be in (0, 1]")
        self.harm_gate = harm_gate
        self.register_buffer(
            "alpha_base",
            torch.tensor(value, dtype=torch.float32),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        harm_probability = self.harm_gate(features)
        return self.alpha_base.to(
            device=harm_probability.device,
            dtype=harm_probability.dtype,
        ) * (1.0 - harm_probability)


def _validate_route_score_contract(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if static.ndim != 2 or static.shape != dynamic.shape or static.shape != valid_mask.shape:
        raise ValueError("static, dynamic, and valid_mask must have matching rank-2 shapes")
    if not static.is_floating_point() or not dynamic.is_floating_point():
        raise ValueError("route scores must use floating dtypes")
    if static.dtype != dynamic.dtype or static.device != dynamic.device:
        raise ValueError("static and dynamic route scores must share dtype and device")
    valid = valid_mask.to(device=static.device, dtype=torch.bool)
    if not torch.isfinite(static[valid]).all() or not torch.isfinite(dynamic[valid]).all():
        raise ValueError("valid route scores must be finite")
    return valid


def fuse_route_scores(
    static: torch.Tensor,
    dynamic: torch.Tensor,
    alpha: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = _validate_route_score_contract(static, dynamic, valid_mask)
    if not alpha.is_floating_point():
        raise ValueError("alpha must use a floating dtype")
    if alpha.ndim != 1 or int(alpha.numel()) != int(static.size(0)):
        raise ValueError("alpha must have one scalar per route row")
    if alpha.device != static.device:
        raise ValueError("alpha and route scores must share a device")
    if not torch.isfinite(alpha).all() or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("alpha must be finite and in [0, 1]")

    output = torch.full_like(static, torch.finfo(static.dtype).min)
    expanded_alpha = alpha.to(dtype=static.dtype).view(-1, 1).expand_as(static)
    static_valid = static[valid]
    dynamic_valid = dynamic[valid]
    alpha_valid = expanded_alpha[valid]
    interpolated = torch.lerp(static_valid, dynamic_valid, alpha_valid)
    output[valid] = torch.where(
        alpha_valid == 0,
        static_valid,
        torch.where(alpha_valid == 1, dynamic_valid, interpolated),
    )
    return output


def effective_memory_alpha(
    raw_alpha: torch.Tensor,
    causal_update_count: torch.Tensor,
) -> torch.Tensor:
    if not raw_alpha.is_floating_point():
        raise ValueError("raw_alpha must use a floating dtype")
    if raw_alpha.shape != causal_update_count.shape:
        raise ValueError("raw_alpha and causal_update_count must have matching shapes")
    if not torch.isfinite(raw_alpha).all() or bool(((raw_alpha < 0) | (raw_alpha > 1)).any()):
        raise ValueError("raw_alpha must be finite and in [0, 1]")
    counts = causal_update_count.to(device=raw_alpha.device, dtype=raw_alpha.dtype)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")
    return torch.where(counts > 0, raw_alpha, torch.zeros_like(raw_alpha))


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    count = int(probabilities.numel())
    if count <= 1:
        return probabilities.new_zeros(())
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return entropy / math.log(count)


def _standardized(values: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) <= 1:
        return torch.zeros_like(values)
    scale = values.var(unbiased=False).sqrt().clamp_min(1.0e-6)
    return (values - values.mean()) / scale


def _standardized_gap(values: torch.Tensor) -> torch.Tensor:
    if int(values.numel()) <= 1:
        return values.new_zeros(())
    top = torch.topk(values, k=2, largest=True, sorted=True).values
    return top[0] - top[1]


def _memory_distances(
    static_memory: torch.Tensor,
    dynamic_memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_norm = torch.linalg.vector_norm(static_memory)
    dynamic_norm = torch.linalg.vector_norm(dynamic_memory)
    epsilon = static_memory.new_tensor(1.0e-6)
    if bool((static_norm <= epsilon) & (dynamic_norm <= epsilon)):
        cosine_distance = static_memory.new_zeros(())
    elif bool((static_norm <= epsilon) | (dynamic_norm <= epsilon)):
        cosine_distance = static_memory.new_ones(())
    else:
        similarity = torch.dot(static_memory, dynamic_memory) / (static_norm * dynamic_norm)
        cosine_distance = (1.0 - similarity.clamp(-1.0, 1.0)).clamp(0.0, 2.0)
    relative_scale = torch.maximum(static_norm, dynamic_norm).clamp_min(epsilon)
    relative_l2 = torch.linalg.vector_norm(dynamic_memory - static_memory) / relative_scale
    return cosine_distance, relative_l2


def memory_utility_features(
    static_logits: torch.Tensor,
    dynamic_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    static_memory: torch.Tensor,
    dynamic_memory: torch.Tensor,
    causal_update_count: torch.Tensor,
    *,
    update_count_cap: float,
    candidate_count_cap: float,
) -> torch.Tensor:
    valid = _validate_route_score_contract(static_logits, dynamic_logits, valid_mask)
    if (
        static_memory.ndim != 2
        or static_memory.shape != dynamic_memory.shape
        or int(static_memory.size(0)) != int(static_logits.size(0))
    ):
        raise ValueError("static and dynamic memory must have matching rank-2 batch shapes")
    if not static_memory.is_floating_point() or not dynamic_memory.is_floating_point():
        raise ValueError("memory tensors must use floating dtypes")
    if static_memory.dtype != dynamic_memory.dtype or static_memory.device != dynamic_memory.device:
        raise ValueError("static and dynamic memory must share dtype and device")
    if static_memory.device != static_logits.device:
        raise ValueError("memory and route scores must share a device")
    if causal_update_count.ndim != 1 or int(causal_update_count.numel()) != int(static_logits.size(0)):
        raise ValueError("causal_update_count must have one value per route row")
    update_cap = float(update_count_cap)
    candidate_cap = float(candidate_count_cap)
    if not math.isfinite(update_cap) or not math.isfinite(candidate_cap) or update_cap <= 0 or candidate_cap <= 0:
        raise ValueError("feature count caps must be finite and positive")
    counts = causal_update_count.to(device=static_logits.device, dtype=static_logits.dtype)
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("causal_update_count must be finite and nonnegative")

    with torch.no_grad():
        rows: list[torch.Tensor] = []
        for row_idx in range(int(static_logits.size(0))):
            row_valid = valid[row_idx]
            candidate_count = int(row_valid.sum().item())
            if candidate_count == 0:
                rows.append(static_logits.new_zeros(len(MEMORY_UTILITY_FEATURE_NAMES)))
                continue
            static_values = static_logits[row_idx, row_valid]
            dynamic_values = dynamic_logits[row_idx, row_valid]
            static_z = _standardized(static_values)
            dynamic_z = _standardized(dynamic_values)
            static_prob = torch.softmax(static_values, dim=0)
            dynamic_prob = torch.softmax(dynamic_values, dim=0)
            mixture = 0.5 * (static_prob + dynamic_prob)
            js_divergence = 0.5 * (
                (
                    static_prob
                    * (static_prob.clamp_min(1.0e-12).log() - mixture.clamp_min(1.0e-12).log())
                ).sum()
                + (
                    dynamic_prob
                    * (dynamic_prob.clamp_min(1.0e-12).log() - mixture.clamp_min(1.0e-12).log())
                ).sum()
            )
            cosine_distance, relative_l2 = _memory_distances(
                static_memory[row_idx].to(dtype=static_logits.dtype),
                dynamic_memory[row_idx].to(dtype=static_logits.dtype),
            )
            normalized_update_count = torch.log1p(counts[row_idx].clamp(max=update_cap)) / math.log1p(
                update_cap
            )
            normalized_candidate_count = static_logits.new_tensor(
                math.log1p(min(candidate_count, candidate_cap)) / math.log1p(candidate_cap)
            )
            rows.append(
                torch.stack(
                    (
                        _normalized_entropy(static_prob),
                        _normalized_entropy(dynamic_prob),
                        _standardized_gap(static_z),
                        _standardized_gap(dynamic_z),
                        (static_values.argmax() == dynamic_values.argmax()).to(static_logits.dtype),
                        js_divergence,
                        (static_z - dynamic_z).abs().mean(),
                        cosine_distance,
                        relative_l2,
                        normalized_update_count,
                        normalized_candidate_count,
                    )
                )
            )
        if not rows:
            return static_logits.new_zeros((0, len(MEMORY_UTILITY_FEATURE_NAMES))).detach()
        return torch.stack(rows, dim=0).detach()


def heuristic_memory_alpha(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or int(features.size(1)) != len(MEMORY_UTILITY_FEATURE_NAMES):
        raise ValueError("features must match the versioned memory utility feature schema")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("memory utility features must be finite floating values")
    js = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("js_divergence")]
    agreement = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("top1_agreement")]
    dynamic_gap = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("dynamic_standardized_gap")]
    static_gap = features[:, MEMORY_UTILITY_FEATURE_NAMES.index("static_standardized_gap")]
    return torch.sigmoid(dynamic_gap - static_gap - js + agreement - 0.5).detach()


def positive_rank_and_utility(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = _validate_route_score_contract(logits, logits, valid_mask)
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must match route logits")
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    eligible = positive.any(dim=-1)
    floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~valid, floor)
    best_positive = masked.masked_fill(~positive, floor).max(dim=-1).values
    ranks = ((masked > best_positive.unsqueeze(1)) & valid).sum(dim=-1) + 1
    ranks = torch.where(eligible, ranks, torch.zeros_like(ranks))
    utilities = logits.new_zeros(int(logits.size(0)))
    if bool(eligible.any()):
        eligible_logits = masked[eligible]
        eligible_positive = positive[eligible]
        positive_logits = eligible_logits.masked_fill(~eligible_positive, floor)
        utilities[eligible] = torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(
            eligible_logits,
            dim=-1,
        )
    return ranks, utilities, eligible
