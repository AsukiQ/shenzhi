from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonAffineRMSNorm(nn.Module):
    def __init__(self, d: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        if int(d) <= 0:
            raise ValueError("RMSNorm dimension must be positive")
        self.d = int(d)
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or int(value.size(-1)) != self.d:
            raise ValueError("memory norm expects [batch, d]")
        scale = value.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * scale.to(device=value.device, dtype=value.dtype)


class BoundedLayerScale(nn.Module):
    def __init__(self, *, initial: float = 0.01, maximum: float = 1.0) -> None:
        super().__init__()
        initial = float(initial)
        maximum = float(maximum)
        if not math.isfinite(initial) or not math.isfinite(maximum):
            raise ValueError("LayerScale bounds must be finite")
        if not 0.0 < initial < maximum:
            raise ValueError("LayerScale requires 0 < initial < maximum")
        ratio = initial / maximum
        self.maximum = maximum
        self.raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) * self.maximum

    def reset_to(self, initial: float) -> None:
        initial = float(initial)
        if not math.isfinite(initial) or not 0.0 < initial < self.maximum:
            raise ValueError("LayerScale reset requires 0 < initial < maximum")
        ratio = initial / self.maximum
        with torch.no_grad():
            self.raw.fill_(math.log(ratio / (1.0 - ratio)))


class LatentTraceSynchronization(nn.Module):
    """Low-rank decayed pair features over recent valid latent states."""

    def __init__(
        self,
        d: int,
        *,
        pair_dim: int = 96,
        trace_length: int = 8,
        initial_decay: float = 0.5,
        scale_initial: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.pair_dim = int(pair_dim)
        self.trace_length = int(trace_length)
        self.initial_decay = float(initial_decay)
        if min(self.d, self.pair_dim, self.trace_length) <= 0:
            raise ValueError(
                "synchronization dimensions and trace length must be positive"
            )
        if not math.isfinite(self.initial_decay) or self.initial_decay <= 0.0:
            raise ValueError("synchronization decay must be positive and finite")
        self.left_projection = nn.Linear(self.d, self.pair_dim, bias=False)
        self.right_projection = nn.Linear(self.d, self.pair_dim, bias=False)
        inverse_softplus = self.initial_decay + math.log(
            -math.expm1(-self.initial_decay)
        )
        self.raw_decay = nn.Parameter(
            torch.full((self.pair_dim,), inverse_softplus)
        )
        self.output_projection = nn.Linear(self.pair_dim, self.d, bias=False)
        self.scale = BoundedLayerScale(initial=float(scale_initial), maximum=1.0)
        self.memory_norm = NonAffineRMSNorm(self.d)
        self.reset_parameters(scale_initial=float(scale_initial))

    def reset_parameters(self, *, scale_initial: float = 0.05) -> None:
        self.left_projection.reset_parameters()
        self.right_projection.reset_parameters()
        self.output_projection.reset_parameters()
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1.0e-3)
        inverse_softplus = self.initial_decay + math.log(
            -math.expm1(-self.initial_decay)
        )
        with torch.no_grad():
            self.raw_decay.fill_(inverse_softplus)
        self.scale.reset_to(float(scale_initial))

    def decay(self) -> torch.Tensor:
        return F.softplus(self.raw_decay)

    def forward(
        self,
        memory: torch.Tensor,
        trace: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 2 or int(memory.size(-1)) != self.d:
            raise ValueError("synchronization memory must have shape [batch, d]")
        if (
            trace.ndim != 3
            or int(trace.size(0)) != int(memory.size(0))
            or int(trace.size(-1)) != self.d
        ):
            raise ValueError("synchronization trace must have shape [batch, time, d]")
        if tuple(valid_mask.shape) != tuple(trace.shape[:2]):
            raise ValueError("synchronization mask must have shape [batch, time]")
        if trace.device != memory.device or valid_mask.device != memory.device:
            raise ValueError("synchronization inputs must share one device")
        if int(trace.size(1)) == 0:
            return memory

        valid = valid_mask.to(dtype=torch.bool)
        valid_rank = valid.to(dtype=torch.long).cumsum(dim=1)
        age = valid_rank[:, -1:].sub(valid_rank)
        recent = valid & age.lt(self.trace_length)
        projection_input = torch.where(
            recent.unsqueeze(-1),
            trace,
            torch.zeros_like(trace),
        ).to(dtype=self.left_projection.weight.dtype)
        left = self.left_projection(projection_input)
        right = self.right_projection(projection_input)
        decay = self.decay().float().view(1, 1, self.pair_dim)
        weights = torch.exp(-age.float().unsqueeze(-1) * decay)
        weights = weights * recent.unsqueeze(-1).float()
        pair_features = left.float() * right.float()
        normalizer = weights.sum(dim=1).clamp_min(1.0).sqrt()
        synchronized = (weights * pair_features).sum(dim=1) / normalizer
        delta = self.output_projection(
            synchronized.to(dtype=self.output_projection.weight.dtype)
        )
        candidate = self.memory_norm(
            memory + self.scale().to(dtype=delta.dtype) * torch.tanh(delta)
        ).to(dtype=memory.dtype)
        history = valid.any(dim=1)
        return torch.where(history.unsqueeze(-1), candidate, memory)


class TwoLayerAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = int(hidden_dim)
        if min(input_dim, output_dim, hidden_dim) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.input.reset_parameters()
        self.output.reset_parameters()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or int(value.size(-1)) != self.input_dim:
            raise ValueError("adapter input must have shape [batch, input_dim]")
        return self.output(F.gelu(self.input(self.norm(value))))


class UnifiedStaticQuery(nn.Module):
    """Legacy-compatible static scorer used identically for recall and route."""

    def __init__(self, d: int) -> None:
        super().__init__()
        d = int(d)
        self.d = d
        # Keep this layout byte-compatible with UnifiedMemoryRetriever.fuse so
        # the verified legacy Stage0 checkpoint can be transplanted exactly.
        self.fuse = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.register_buffer("unit_temperature", torch.ones(()), persistent=False)

    def forward(self, h_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        _validate_model_vectors(h_t, b_t, d=self.d)
        value = self.fuse(torch.cat((h_t, b_t, h_t * b_t), dim=-1))
        # LayerNorm is an autocast FP32 op.  Rejoin the recurrent dtype so
        # Stage2 never mixes FP32 static queries with BF16 dynamic deltas.
        return value.to(dtype=h_t.dtype)

    def temperature(self) -> torch.Tensor:
        return self.unit_temperature


class StaticRouteQueryDelta(nn.Module):
    """Identity-initialized route-only query residual over the frozen proposal."""

    def __init__(
        self,
        d: int,
        *,
        hidden_dim: int,
        scale_initial: float = 0.1,
        scale_maximum: float = 4.0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.adapter = TwoLayerAdapter(
            3 * self.d,
            self.d,
            hidden_dim=hidden_dim,
        )
        self.scale = BoundedLayerScale(
            initial=float(scale_initial),
            maximum=float(scale_maximum),
        )
        nn.init.zeros_(self.adapter.output.weight)
        nn.init.zeros_(self.adapter.output.bias)

    def forward(self, h_t: torch.Tensor, static_memory: torch.Tensor) -> torch.Tensor:
        _validate_model_vectors(h_t, static_memory, d=self.d)
        raw = self.adapter(
            torch.cat((h_t, static_memory, h_t * static_memory), dim=-1)
        )
        return self.scale().to(dtype=raw.dtype) * torch.tanh(raw)


class MemoryQueryDelta(nn.Module):
    def __init__(self, d: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.d = int(d)
        self.query = TwoLayerAdapter(
            3 * self.d,
            self.d,
            hidden_dim=hidden_dim,
        )

    def reset_parameters(self, *, zero_output: bool = False) -> None:
        self.query.reset_parameters()
        if zero_output:
            nn.init.zeros_(self.query.output.weight)
            nn.init.zeros_(self.query.output.bias)

    def _encode(self, h_t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        _validate_model_vectors(h_t, memory, d=self.d)
        return self.query(torch.cat((h_t, memory, h_t * memory), dim=-1))

    def forward(
        self,
        h_t: torch.Tensor,
        dynamic_memory: torch.Tensor,
        static_memory: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode(h_t, dynamic_memory) - self._encode(h_t, static_memory)


class TransitionDelta(nn.Module):
    def __init__(self, d: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.d = int(d)
        self.adapter = TwoLayerAdapter(
            4 * self.d,
            self.d,
            hidden_dim=hidden_dim,
        )

    def reset_parameters(self) -> None:
        self.adapter.reset_parameters()

    def forward(
        self,
        memory: torch.Tensor,
        state: torch.Tensor,
        skill: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        _validate_model_vectors(memory, state, skill, action, d=self.d)
        return self.adapter(torch.cat((memory, state, skill, action), dim=-1))


class CorrectionDelta(nn.Module):
    def __init__(self, d: int, *, hidden_dim: int) -> None:
        super().__init__()
        self.d = int(d)
        self.adapter = TwoLayerAdapter(
            5 * self.d,
            self.d,
            hidden_dim=hidden_dim,
        )

    def reset_parameters(self) -> None:
        self.adapter.reset_parameters()

    def forward(
        self,
        predicted_memory: torch.Tensor,
        state: torch.Tensor,
        skill: torch.Tensor,
        action: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor:
        _validate_model_vectors(
            predicted_memory,
            state,
            skill,
            action,
            result,
            d=self.d,
        )
        return self.adapter(
            torch.cat((predicted_memory, state, skill, action, result), dim=-1)
        )


class ScalarCorrectionGate(nn.Module):
    def __init__(self, d: int, *, hidden_dim: int, initial_beta: float = 0.5) -> None:
        super().__init__()
        d = int(d)
        initial_beta = float(initial_beta)
        if not 0.0 < initial_beta < 1.0:
            raise ValueError("initial correction beta must be in (0, 1)")
        self.d = d
        self.initial_beta = initial_beta
        self.norm = nn.LayerNorm(3 * d)
        self.hidden = nn.Linear(3 * d, int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.hidden.reset_parameters()
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(
            self.output.bias,
            math.log(self.initial_beta / (1.0 - self.initial_beta)),
        )

    def forward(
        self,
        predicted_memory: torch.Tensor,
        result: torch.Tensor,
        correction_delta: torch.Tensor,
    ) -> torch.Tensor:
        _validate_model_vectors(
            predicted_memory,
            result,
            correction_delta,
            d=self.d,
        )
        value = torch.cat((predicted_memory, result, correction_delta), dim=-1)
        hidden = F.gelu(self.hidden(self.norm(value)))
        return torch.sigmoid(self.output(hidden))


class QueryLevelExpertMixture(nn.Module):
    """Predict one static/dynamic mixture coefficient per decision step.

    Confidence statistics alone cannot tell whether a trajectory contains the
    semantic evidence required for the next decision.  The deployed selector
    therefore optionally augments those statistics with learned low-dimensional
    projections of the current route state, structured current state, and the
    recurrent-memory delta.  It remains one shared per-query selector: source
    and benchmark identities never enter this module.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        semantic_dim: int | None = None,
        semantic_projection_dim: int = 16,
        hidden_dim: int = 32,
        initial_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("query expert mixture feature dimension must be positive")
        if not 0.0 < float(initial_probability) < 0.5:
            raise ValueError("candidate gate initial probability must be in (0, 0.5)")
        self.feature_dim = int(feature_dim)
        self.semantic_dim = None if semantic_dim is None else int(semantic_dim)
        if self.semantic_dim is not None and self.semantic_dim <= 0:
            raise ValueError("query expert mixture semantic dimension must be positive")
        self.semantic_projection_dim = (
            0
            if self.semantic_dim is None
            else min(int(semantic_projection_dim), self.semantic_dim)
        )
        if self.semantic_dim is not None and self.semantic_projection_dim <= 0:
            raise ValueError("semantic projection dimension must be positive")
        self.initial_probability = float(initial_probability)
        if self.semantic_dim is None:
            self.route_state_projection = None
            self.current_state_projection = None
            self.memory_delta_projection = None
            self.static_expert_projection = None
            self.dynamic_expert_projection = None
            self.expert_delta_projection = None
            selector_dim = self.feature_dim
        else:
            self.route_state_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            self.current_state_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            self.memory_delta_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            self.static_expert_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            self.dynamic_expert_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            self.expert_delta_projection = nn.Linear(
                self.semantic_dim,
                self.semantic_projection_dim,
                bias=False,
            )
            selector_dim = self.feature_dim + 6 * self.semantic_projection_dim + 6
        self.selector_dim = int(selector_dim)
        self.norm = nn.LayerNorm(self.selector_dim)
        self.hidden = nn.Linear(self.selector_dim, int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.hidden.reset_parameters()
        for projection in (
            self.route_state_projection,
            self.current_state_projection,
            self.memory_delta_projection,
            self.static_expert_projection,
            self.dynamic_expert_projection,
            self.expert_delta_projection,
        ):
            if projection is not None:
                projection.reset_parameters()
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(
            self.output.bias,
            math.log(
                self.initial_probability / (1.0 - self.initial_probability)
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        route_state: torch.Tensor | None = None,
        current_state: torch.Tensor | None = None,
        memory_delta: torch.Tensor | None = None,
        static_expert_state: torch.Tensor | None = None,
        dynamic_expert_state: torch.Tensor | None = None,
        expert_delta_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2 or int(features.size(-1)) != self.feature_dim:
            raise ValueError(
                "query expert mixture expects "
                f"[batch, {self.feature_dim}]"
            )
        selector_features = features
        if self.semantic_dim is not None:
            semantic_values = (
                route_state,
                current_state,
                memory_delta,
                static_expert_state,
                dynamic_expert_state,
                expert_delta_state,
            )
            if any(value is None for value in semantic_values):
                raise ValueError("semantic query mixture requires all semantic states")
            assert route_state is not None
            assert current_state is not None
            assert memory_delta is not None
            assert static_expert_state is not None
            assert dynamic_expert_state is not None
            assert expert_delta_state is not None
            _validate_model_vectors(
                route_state,
                current_state,
                memory_delta,
                static_expert_state,
                dynamic_expert_state,
                expert_delta_state,
                d=self.semantic_dim,
            )
            if int(route_state.size(0)) != int(features.size(0)):
                raise ValueError("semantic query mixture batch size mismatch")
            semantic_dtype = features.dtype
            normalized_route = F.normalize(route_state.float(), p=2, dim=-1).to(
                dtype=semantic_dtype
            )
            normalized_current = F.normalize(current_state.float(), p=2, dim=-1).to(
                dtype=semantic_dtype
            )
            normalized_delta = F.normalize(memory_delta.float(), p=2, dim=-1).to(
                dtype=semantic_dtype
            )
            normalized_static_expert = F.normalize(
                static_expert_state.float(), p=2, dim=-1
            ).to(dtype=semantic_dtype)
            normalized_dynamic_expert = F.normalize(
                dynamic_expert_state.float(), p=2, dim=-1
            ).to(dtype=semantic_dtype)
            normalized_expert_delta = F.normalize(
                expert_delta_state.float(), p=2, dim=-1
            ).to(dtype=semantic_dtype)
            assert self.route_state_projection is not None
            assert self.current_state_projection is not None
            assert self.memory_delta_projection is not None
            assert self.static_expert_projection is not None
            assert self.dynamic_expert_projection is not None
            assert self.expert_delta_projection is not None
            projected = (
                torch.tanh(self.route_state_projection(normalized_route)),
                torch.tanh(self.current_state_projection(normalized_current)),
                torch.tanh(self.memory_delta_projection(normalized_delta)),
                torch.tanh(
                    self.static_expert_projection(normalized_static_expert)
                ),
                torch.tanh(
                    self.dynamic_expert_projection(normalized_dynamic_expert)
                ),
                torch.tanh(
                    self.expert_delta_projection(normalized_expert_delta)
                ),
            )
            relations = torch.stack(
                (
                    (normalized_route * normalized_current).sum(dim=-1),
                    (normalized_route * normalized_delta).sum(dim=-1),
                    (normalized_current * normalized_delta).sum(dim=-1),
                    (
                        normalized_static_expert * normalized_dynamic_expert
                    ).sum(dim=-1),
                    (normalized_route * normalized_expert_delta).sum(dim=-1),
                    (normalized_delta * normalized_expert_delta).sum(dim=-1),
                ),
                dim=-1,
            )
            selector_features = torch.cat(
                (features, *projected, relations),
                dim=-1,
            )
        hidden = F.gelu(self.hidden(self.norm(selector_features)))
        return torch.sigmoid(self.output(hidden)).squeeze(-1)


class CandidateCompressor(nn.Module):
    """Query-conditioned residual compressor over natural coarse candidates."""

    def __init__(
        self,
        d: int,
        *,
        hidden_dim: int = 128,
        scale_initial: float = 0.1,
        scale_maximum: float = 4.0,
    ) -> None:
        super().__init__()
        self.d = int(d)
        hidden_dim = min(int(hidden_dim), self.d)
        if min(self.d, hidden_dim) <= 0:
            raise ValueError("candidate compressor dimensions must be positive")
        self.norm = nn.LayerNorm(5 * self.d + 1)
        self.hidden = nn.Linear(5 * self.d + 1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.scale = BoundedLayerScale(
            initial=float(scale_initial),
            maximum=float(scale_maximum),
        )
        # Preserve the Stage0 ordering exactly before compressor training.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        causal_query: torch.Tensor,
        current_state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        base_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> "VNextCandidateCompressionScores":
        if causal_query.ndim != 2 or current_state.shape != causal_query.shape:
            raise ValueError("compressor states must match [batch, d]")
        if int(causal_query.size(-1)) != self.d:
            raise ValueError("compressor state dimension mismatch")
        if (
            candidate_embeddings.ndim != 3
            or int(candidate_embeddings.size(0)) != int(causal_query.size(0))
            or int(candidate_embeddings.size(-1)) != self.d
        ):
            raise ValueError("compressor candidates must have shape [batch, candidates, d]")
        if base_logits.shape != candidate_embeddings.shape[:2]:
            raise ValueError("compressor base logits must match [batch, candidates]")
        if valid_mask.shape != base_logits.shape:
            raise ValueError("compressor validity must match base logits")
        valid = valid_mask.to(device=causal_query.device, dtype=torch.bool)
        if bool((~valid.any(dim=-1)).any().item()):
            raise ValueError("compressor requires at least one natural candidate per row")

        dtype = causal_query.dtype
        causal = F.normalize(causal_query.float(), p=2, dim=-1).to(dtype=dtype)
        current = F.normalize(current_state.float(), p=2, dim=-1).to(
            device=causal_query.device,
            dtype=dtype,
        )
        candidates = F.normalize(candidate_embeddings.float(), p=2, dim=-1).to(
            device=causal_query.device,
            dtype=dtype,
        )
        width = int(candidates.size(1))
        causal_expanded = causal.unsqueeze(1).expand(-1, width, -1)
        current_expanded = current.unsqueeze(1).expand(-1, width, -1)

        base = base_logits.to(device=causal_query.device, dtype=dtype)
        valid_float = valid.to(dtype=base.dtype)
        count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (torch.where(valid, base, torch.zeros_like(base)) * valid_float).sum(
            dim=-1,
            keepdim=True,
        ) / count
        centered = torch.where(valid, base - mean, torch.zeros_like(base))
        variance = (centered.float().pow(2) * valid_float.float()).sum(
            dim=-1,
            keepdim=True,
        ) / count.float()
        standardized_base = centered / variance.add(1.0e-6).sqrt().to(dtype=base.dtype)

        features = torch.cat(
            (
                causal_expanded,
                current_expanded,
                candidates,
                causal_expanded * candidates,
                current_expanded * candidates,
                standardized_base.unsqueeze(-1),
            ),
            dim=-1,
        )
        raw_delta = self.output(F.gelu(self.hidden(self.norm(features)))).squeeze(-1)
        bounded_delta = self.scale().to(dtype=raw_delta.dtype) * torch.tanh(raw_delta)
        logits = base + bounded_delta
        floor = torch.finfo(logits.dtype).min
        return VNextCandidateCompressionScores(
            base_logits=base.masked_fill(~valid, floor),
            logits=logits.masked_fill(~valid, floor),
            bounded_delta=bounded_delta.masked_fill(~valid, 0.0),
        )


@dataclass(frozen=True)
class VNextQueries:
    static_recall: torch.Tensor
    static_route_delta: torch.Tensor
    static_route: torch.Tensor
    dynamic_recall: torch.Tensor
    dynamic_route: torch.Tensor
    recall_delta: torch.Tensor
    route_delta: torch.Tensor


@dataclass(frozen=True)
class VNextMemoryUpdate:
    predicted_memory: torch.Tensor
    effective_memory: torch.Tensor
    transition_delta: torch.Tensor
    correction_delta: torch.Tensor | None
    correction_beta: torch.Tensor | None
    latent_trace: torch.Tensor | None = None


@dataclass(frozen=True)
class VNextCandidateRouteScores:
    static_logits: torch.Tensor
    raw_dynamic_logits: torch.Tensor
    mixed_logits: torch.Tensor
    selector_probability: torch.Tensor
    mixture_probability: torch.Tensor
    static_support_mask: torch.Tensor
    raw_residual: torch.Tensor


@dataclass(frozen=True)
class VNextCandidateCompressionScores:
    base_logits: torch.Tensor
    logits: torch.Tensor
    bounded_delta: torch.Tensor


class CLSTRVNextCore(nn.Module):
    def __init__(
        self,
        d: int,
        *,
        hidden_dim: int = 256,
        scale_initial: float = 0.01,
        scale_maximum: float = 1.0,
        temperature_initial: float = 10.0,
        synchronization_enabled: bool = False,
        synchronization_pair_dim: int = 96,
        synchronization_trace_length: int = 8,
        synchronization_scale_initial: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = int(d)
        hidden_dim = min(int(hidden_dim), self.d)
        self.synchronization_enabled = bool(synchronization_enabled)
        self.synchronization_pair_dim = int(synchronization_pair_dim)
        self.synchronization_trace_length = int(synchronization_trace_length)
        self.synchronization_scale_initial = float(synchronization_scale_initial)
        self.memory_norm = NonAffineRMSNorm(self.d)
        # Proposal and the accepted static route retain the Stage0 geometry.
        # Stage2 learns a separate full-capacity memory residual without
        # rewriting this frozen endpoint.
        self.static_query = UnifiedStaticQuery(self.d)
        # Stage2 copies the accepted Stage0 query into this independent legacy-
        # layout router. Its difference between m_t and m_0 is the unbounded
        # memory residual added to the frozen static route.
        self.unified_route_query = UnifiedStaticQuery(self.d)
        self.route_skill_adapter = nn.Linear(self.d, self.d, bias=False)
        nn.init.zeros_(self.route_skill_adapter.weight)
        self.static_route_query_delta = StaticRouteQueryDelta(
            self.d,
            hidden_dim=hidden_dim,
        )
        # Recall and route have different optimization geometry: recall must
        # move a target across a full-pool Top-K boundary, whereas route only
        # orders an already compressed support.  Independent recurrent heads
        # prevent those gradients from cancelling in one shared query delta.
        self.memory_recall_query = MemoryQueryDelta(self.d, hidden_dim=hidden_dim)
        self.memory_route_query = MemoryQueryDelta(self.d, hidden_dim=hidden_dim)
        # Retain the historical attribute name for checkpoint-key compatibility;
        # the module now predicts one query-level expert mixture coefficient,
        # never independent candidate-local gates.
        self.candidate_route_gate = QueryLevelExpertMixture(
            feature_dim=5,
            hidden_dim=min(32, hidden_dim),
            initial_probability=0.1,
        )
        self.route_expert_mixture = QueryLevelExpertMixture(
            feature_dim=6,
            semantic_dim=self.d,
            semantic_projection_dim=min(16, self.d),
            hidden_dim=min(32, hidden_dim),
            initial_probability=0.1,
        )
        self.route_expert_selection_threshold = 0.55
        self.candidate_compressor = CandidateCompressor(
            self.d,
            hidden_dim=min(128, hidden_dim),
        )
        self.action_adapter = nn.Linear(self.d, self.d, bias=False)
        self.result_adapter = nn.Linear(self.d, self.d, bias=False)
        nn.init.eye_(self.action_adapter.weight)
        nn.init.eye_(self.result_adapter.weight)
        self.transition_delta = TransitionDelta(self.d, hidden_dim=hidden_dim)
        self.correction_delta = CorrectionDelta(self.d, hidden_dim=hidden_dim)
        self.correction_gate = ScalarCorrectionGate(self.d, hidden_dim=min(64, hidden_dim))
        self.transition_scale = BoundedLayerScale(
            initial=scale_initial,
            maximum=scale_maximum,
        )
        self.result_scale = BoundedLayerScale(
            initial=scale_initial,
            maximum=scale_maximum,
        )
        self.recall_scale = BoundedLayerScale(
            initial=scale_initial,
            maximum=scale_maximum,
        )
        self.route_scale = BoundedLayerScale(
            initial=scale_initial,
            maximum=scale_maximum,
        )
        self.synchronization = (
            LatentTraceSynchronization(
                self.d,
                pair_dim=self.synchronization_pair_dim,
                trace_length=self.synchronization_trace_length,
                scale_initial=self.synchronization_scale_initial,
            )
            if self.synchronization_enabled
            else None
        )

    def reset_stage2_recurrent_parameters(
        self,
        *,
        seed: int,
        transition_scale_initial: float = 0.05,
        result_scale_initial: float = 0.05,
        recall_scale_initial: float = 0.10,
        synchronization_scale_initial: float | None = None,
    ) -> dict[str, float | int | bool]:
        """Intentionally initialize Stage2-only recurrent state.

        Stage0 and the static reranker freeze these parameters, so loading
        their checkpoint values would otherwise inherit arbitrary unused
        random tensors.  Query output projections start at zero to make the
        history-conditioned scorer exactly static at step zero.  Transition
        and correction adapters remain nonzero so factual and counterfactual
        memories differ and the zero-output query heads receive a gradient.
        """

        first = next(self.parameters())
        devices = (
            [int(first.device.index)]
            if first.is_cuda and first.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if first.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
            self.memory_recall_query.reset_parameters(zero_output=True)
            self.memory_route_query.reset_parameters(zero_output=True)
            # Preserve the exact historical RNG stream used to initialize the
            # raw recurrent expert.  The legacy five-feature module remains
            # frozen, but resetting it here consumes the same draws as the
            # accepted pre-depth implementation.  Reset the new selector only
            # after every raw-expert module has been initialized.
            self.candidate_route_gate.reset_parameters()
            self.unified_route_query.load_state_dict(self.static_query.state_dict())
            with torch.no_grad():
                self.route_skill_adapter.weight.zero_()
            self.transition_delta.reset_parameters()
            self.correction_delta.reset_parameters()
            self.correction_gate.reset_parameters()
            self.route_expert_mixture.reset_parameters()
            with torch.no_grad():
                self.action_adapter.weight.copy_(
                    torch.eye(
                        self.d,
                        device=self.action_adapter.weight.device,
                        dtype=self.action_adapter.weight.dtype,
                    )
                )
                self.result_adapter.weight.copy_(
                    torch.eye(
                        self.d,
                        device=self.result_adapter.weight.device,
                        dtype=self.result_adapter.weight.dtype,
                    )
                )
            self.transition_scale.reset_to(float(transition_scale_initial))
            self.result_scale.reset_to(float(result_scale_initial))
            self.recall_scale.reset_to(float(recall_scale_initial))
            if self.synchronization is not None:
                self.synchronization.reset_parameters(
                    scale_initial=(
                        self.synchronization_scale_initial
                        if synchronization_scale_initial is None
                        else float(synchronization_scale_initial)
                    )
                )
        return {
            "seed": int(seed),
            "transition_scale_initial": float(transition_scale_initial),
            "result_scale_initial": float(result_scale_initial),
            "recall_scale_initial": float(recall_scale_initial),
            "zero_initialized_memory_recall_output": True,
            "zero_history_exact_static": True,
            "history_rows_use_unbounded_unified_residual_at_step0": True,
            "unified_route_initialized_from_static_query": True,
            "route_skill_adapter_zero_residual": True,
            "route_expert_mixture_initial_probability": float(
                self.route_expert_mixture.initial_probability
            ),
            "route_expert_mixture_feature_dim": int(
                self.route_expert_mixture.feature_dim
            ),
            "route_expert_mixture_semantic_dim": int(
                self.route_expert_mixture.semantic_dim or 0
            ),
            "route_expert_mixture_semantic_projection_dim": int(
                self.route_expert_mixture.semantic_projection_dim
            ),
            "semantic_per_sample_route_selector": True,
            "candidate_semantic_expert_selector": True,
            "query_level_expert_mixture_enabled": True,
            "candidate_local_gate_enabled": False,
            "legacy_candidate_route_gate_frozen": True,
            "legacy_candidate_route_gate_reset_for_rng_compatibility": True,
            "depth_selector_reset_after_raw_expert": True,
            "route_expert_selection_threshold": float(
                self.route_expert_selection_threshold
            ),
            "synchronization_enabled": bool(self.synchronization_enabled),
            "synchronization_pair_dim": int(self.synchronization_pair_dim),
            "synchronization_trace_length": int(self.synchronization_trace_length),
            "synchronization_scale_initial": float(
                self.synchronization_scale_initial
                if synchronization_scale_initial is None
                else synchronization_scale_initial
            ),
        }

    def append_latent_trace(
        self,
        memory: torch.Tensor,
        latent_trace: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Append one post-action memory for the optional native trace."""

        if not self.synchronization_enabled:
            return latent_trace
        if memory.ndim != 2 or int(memory.size(-1)) != self.d:
            raise ValueError("latent-trace memory must have shape [batch, d]")
        if latent_trace is None:
            latent_trace = memory.new_zeros(
                (
                    int(memory.size(0)),
                    self.synchronization_trace_length,
                    self.d,
                )
            )
        if (
            latent_trace.ndim != 3
            or tuple(latent_trace.shape[::2]) != (int(memory.size(0)), self.d)
            or latent_trace.device != memory.device
        ):
            raise ValueError("latent trace must have shape [batch, time, d]")
        if int(latent_trace.size(1)) == 0:
            latent_trace = memory.new_zeros(
                (
                    int(memory.size(0)),
                    self.synchronization_trace_length,
                    self.d,
                )
            )
        elif int(latent_trace.size(1)) != self.synchronization_trace_length:
            if int(latent_trace.size(1)) > self.synchronization_trace_length:
                latent_trace = latent_trace[:, -self.synchronization_trace_length :]
            else:
                padding = latent_trace.new_zeros(
                    (
                        int(latent_trace.size(0)),
                        self.synchronization_trace_length - int(latent_trace.size(1)),
                        self.d,
                    )
                )
                latent_trace = torch.cat((padding, latent_trace), dim=1)
        updated = torch.cat((latent_trace, memory.unsqueeze(1)), dim=1)
        return updated[:, -self.synchronization_trace_length :]

    def synchronized_memory(
        self,
        memory: torch.Tensor,
        latent_trace: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.synchronization_enabled or self.synchronization is None:
            return memory
        if latent_trace is None or int(latent_trace.size(1)) == 0:
            return memory
        # Prefix replay uses a fixed-width trace with left zero padding so
        # batched segments can have different history lengths.  Zero rows are
        # structural padding, not observed latent states.
        valid = latent_trace.ne(0).any(dim=-1)
        return self.synchronization(memory, latent_trace, valid)

    def normalize_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return self.memory_norm(memory)

    def static_query_components(
        self,
        h_t: torch.Tensor,
        b_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t, b_t = _align_model_vector_dtypes(b_t, h_t, b_t, d=self.d)
        unified = self.static_query(h_t, b_t)
        route_delta = self.static_route_query_delta(h_t, b_t)
        route = unified + route_delta
        return unified, route_delta, route

    def static_queries(
        self,
        h_t: torch.Tensor,
        b_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unified, _route_delta, route = self.static_query_components(h_t, b_t)
        return unified, route

    def queries(
        self,
        h_t: torch.Tensor,
        dynamic_memory: torch.Tensor,
        static_memory: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        memory_state: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
    ) -> VNextQueries:
        memory_state = h_t if memory_state is None else memory_state
        h_t, memory_state, dynamic_memory, static_memory = _align_model_vector_dtypes(
            dynamic_memory,
            h_t,
            memory_state,
            dynamic_memory,
            static_memory,
            d=self.d,
        )
        if history_mask.ndim != 1 or int(history_mask.numel()) != int(h_t.size(0)):
            raise ValueError("history_mask must have one value per route row")
        static_recall, static_route_delta, static_route = (
            self.static_query_components(h_t, static_memory)
        )
        retrieval_memory = self.synchronized_memory(dynamic_memory, latent_trace)
        raw_recall = self.memory_recall_query(
            memory_state,
            retrieval_memory,
            static_memory,
        )
        recall_delta = self.recall_scale().to(raw_recall.dtype) * torch.tanh(raw_recall)
        mask = history_mask.to(device=h_t.device, dtype=torch.bool).unsqueeze(-1)
        route_reference = self.unified_route_query(h_t, static_memory)
        route_candidate = self.unified_route_query(h_t, retrieval_memory)
        route_delta = route_candidate - route_reference
        # The explicit mask preserves a bit-exact static fallback for rows with
        # no usable prefix. History-bearing rows receive the full unbounded
        # legacy-capacity memory residual; no Gate or clamp suppresses it.
        dynamic_recall = static_recall + recall_delta
        dynamic_route = static_route + torch.where(
            mask,
            route_delta,
            torch.zeros_like(route_delta),
        )
        return VNextQueries(
            static_recall=static_recall,
            static_route_delta=static_route_delta,
            static_route=static_route,
            dynamic_recall=torch.where(mask, dynamic_recall, static_recall),
            dynamic_route=dynamic_route,
            recall_delta=torch.where(mask, recall_delta, torch.zeros_like(recall_delta)),
            route_delta=torch.where(mask, route_delta, torch.zeros_like(route_delta)),
        )

    def full_pool_logits(
        self,
        query: torch.Tensor,
        skill_embeddings: torch.Tensor,
        *,
        head: str,
        skill_embeddings_are_normalized: bool = False,
    ) -> torch.Tensor:
        if query.ndim != 2 or int(query.size(-1)) != self.d:
            raise ValueError("full-pool query must have shape [batch, d]")
        if skill_embeddings.ndim != 2 or int(skill_embeddings.size(-1)) != self.d:
            raise ValueError("skill embeddings must have shape [skills, d]")
        skills = (
            skill_embeddings.to(device=query.device, dtype=query.dtype)
            if skill_embeddings_are_normalized
            else F.normalize(skill_embeddings.float(), p=2, dim=-1).to(
                device=query.device,
                dtype=query.dtype,
            )
        )
        if head not in {"recall", "route"}:
            raise ValueError(f"unsupported vNext head: {head}")
        temperature = self.static_query.temperature()
        return temperature.to(device=query.device, dtype=query.dtype) * query @ skills.t()

    def candidate_compression_scores(
        self,
        causal_query: torch.Tensor,
        current_state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        base_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> VNextCandidateCompressionScores:
        return self.candidate_compressor(
            causal_query,
            current_state,
            candidate_embeddings,
            base_logits,
            valid_mask,
        )

    def candidate_route_scores(
        self,
        h_t: torch.Tensor,
        dynamic_memory: torch.Tensor,
        static_memory: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        static_support_mask: torch.Tensor | None = None,
        history_depth: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
        temperature: torch.Tensor,
        hard_fallback: bool,
        residual_bound: float = 2.0,
    ) -> VNextCandidateRouteScores:
        memory_state = h_t if memory_state is None else memory_state
        h_t, memory_state, dynamic_memory, static_memory = _align_model_vector_dtypes(
            h_t,
            h_t,
            memory_state,
            dynamic_memory,
            static_memory,
            d=self.d,
        )
        queries = self.queries(
            h_t,
            dynamic_memory,
            static_memory,
            history_mask,
            memory_state=memory_state,
            latent_trace=latent_trace,
        )
        if candidate_embeddings.ndim != 3 or int(candidate_embeddings.size(0)) != int(
            h_t.size(0)
        ) or int(candidate_embeddings.size(-1)) != self.d:
            raise ValueError("candidate embeddings must have shape [batch, candidates, d]")
        if tuple(valid_mask.shape) != tuple(candidate_embeddings.shape[:2]):
            raise ValueError("candidate validity must match [batch, candidates]")
        candidates = F.normalize(candidate_embeddings.float(), p=2, dim=-1).to(
            device=h_t.device,
            dtype=h_t.dtype,
        )
        adapted_candidates = candidates + self.route_skill_adapter(candidates)
        static_base_logits = temperature.to(h_t.dtype) * torch.einsum(
            "bd,bcd->bc", queries.static_recall, candidates
        )
        static_route_delta_logits = temperature.to(h_t.dtype) * torch.einsum(
            "bd,bcd->bc", queries.static_route_delta, candidates
        )
        static_logits = static_base_logits + static_route_delta_logits
        route_residual = temperature.to(h_t.dtype) * torch.einsum(
            "bd,bcd->bc", queries.route_delta, adapted_candidates
        )
        raw_dynamic_logits = static_logits + route_residual
        history = history_mask.to(device=h_t.device, dtype=torch.bool).unsqueeze(-1)
        depth = (
            history_mask.to(device=h_t.device, dtype=torch.float32)
            if history_depth is None
            else history_depth.to(device=h_t.device, dtype=torch.float32)
        )
        if depth.ndim != 1 or int(depth.numel()) != int(h_t.size(0)):
            raise ValueError("history depth must have one value per route row")
        if not bool(torch.isfinite(depth).all().item()) or bool((depth < 0).any().item()):
            raise ValueError("history depth must be finite and nonnegative")
        valid = valid_mask.to(device=h_t.device, dtype=torch.bool)
        static_support = (
            valid
            if static_support_mask is None
            else static_support_mask.to(device=h_t.device, dtype=torch.bool) & valid
        )
        if tuple(static_support.shape) != tuple(valid.shape):
            raise ValueError("static support membership must match candidate validity")
        if bool((~static_support.any(dim=-1)).any().item()):
            raise ValueError("every route row requires an anchored static candidate")
        static_floor = torch.finfo(static_logits.dtype).min
        anchored_static_logits = static_logits.masked_fill(
            ~static_support,
            static_floor,
        )

        def _expert_summary(
            logits: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            floor = torch.finfo(logits.dtype).min
            masked = logits.masked_fill(~mask, floor)
            log_partition = torch.logsumexp(masked.float(), dim=-1)
            log_probability = masked.float() - log_partition.unsqueeze(-1)
            probability = torch.exp(log_probability).masked_fill(~mask, 0.0)
            entropy = -(probability * log_probability.masked_fill(~mask, 0.0)).sum(
                dim=-1
            )
            count = mask.sum(dim=-1).float().clamp_min(1.0)
            normalized_entropy = entropy / count.log().clamp_min(1.0)
            top_width = min(2, int(logits.size(1)))
            top = torch.topk(masked.float(), k=top_width, dim=-1).values
            margin = (
                torch.where(
                    mask.sum(dim=-1).ge(2),
                    top[:, 0] - top[:, 1],
                    torch.zeros_like(top[:, 0]),
                )
                if top_width == 2
                else torch.zeros_like(top[:, 0])
            )
            return log_partition, normalized_entropy, margin

        static_partition, static_entropy, static_margin = _expert_summary(
            anchored_static_logits,
            static_support,
        )
        dynamic_partition, dynamic_entropy, dynamic_margin = _expert_summary(
            raw_dynamic_logits,
            valid,
        )
        temperature_value = temperature.detach().float().clamp_min(1.0e-6)
        residual_rms = route_residual.float().pow(2).mean(dim=-1).sqrt()
        normalized_depth = (
            torch.log1p(depth) / math.log1p(16.0)
        ).clamp(max=2.0)
        static_probability = torch.softmax(
            anchored_static_logits.float(),
            dim=-1,
        ).masked_fill(~static_support, 0.0)
        dynamic_probability = torch.softmax(
            raw_dynamic_logits.float().masked_fill(~valid, float("-inf")),
            dim=-1,
        ).masked_fill(~valid, 0.0)
        static_expert_state = torch.einsum(
            "bc,bcd->bd",
            static_probability.to(dtype=candidates.dtype),
            candidates,
        ).to(dtype=h_t.dtype)
        dynamic_expert_state = torch.einsum(
            "bc,bcd->bd",
            dynamic_probability.to(dtype=candidates.dtype),
            candidates,
        ).to(dtype=h_t.dtype)
        expert_delta_state = dynamic_expert_state - static_expert_state
        features = torch.stack(
            (
                static_margin / temperature_value,
                1.0 - static_entropy,
                dynamic_margin / temperature_value,
                1.0 - dynamic_entropy,
                residual_rms / temperature_value,
                normalized_depth,
            ),
            dim=-1,
        ).detach()
        probability = self.route_expert_mixture(
            features.to(dtype=h_t.dtype),
            route_state=h_t.detach(),
            current_state=memory_state.detach(),
            memory_delta=(dynamic_memory - static_memory).detach(),
            static_expert_state=static_expert_state.detach(),
            dynamic_expert_state=dynamic_expert_state.detach(),
            expert_delta_state=expert_delta_state.detach(),
        )
        history_row = history.squeeze(-1)
        effective_probability = probability * history_row.to(probability.dtype)
        mixture_probability = (
            effective_probability.ge(self.route_expert_selection_threshold).to(
                effective_probability.dtype
            )
            if hard_fallback
            else effective_probability
        )

        alpha = mixture_probability.float()
        mixed_probability_distribution = (
            (1.0 - alpha).unsqueeze(-1) * static_probability
            + alpha.unsqueeze(-1) * dynamic_probability
        )
        mixed = (
            torch.log(mixed_probability_distribution.clamp_min(1.0e-12))
            + static_partition.unsqueeze(-1)
        ).to(dtype=static_logits.dtype)
        exact_static = (~history_row) | mixture_probability.eq(0.0) | (
            route_residual.eq(0).all(dim=-1) & valid.eq(static_support).all(dim=-1)
        )
        mixed = torch.where(
            exact_static.unsqueeze(-1),
            anchored_static_logits,
            mixed,
        )
        mixed = torch.where(
            mixture_probability.eq(1.0).unsqueeze(-1),
            raw_dynamic_logits,
            mixed,
        )
        mixed = mixed.masked_fill(~valid, static_floor)
        # One query-level probability controls the complete unbounded dynamic
        # expert. No benchmark/source label enters this inference path.
        del residual_bound
        floor = torch.finfo(mixed.dtype).min
        return VNextCandidateRouteScores(
            static_logits=anchored_static_logits.masked_fill(~valid, floor),
            raw_dynamic_logits=raw_dynamic_logits.masked_fill(~valid, floor),
            mixed_logits=mixed,
            selector_probability=effective_probability,
            mixture_probability=mixture_probability,
            static_support_mask=static_support,
            raw_residual=route_residual.masked_fill(~valid, 0.0),
        )

    def update_memory(
        self,
        memory: torch.Tensor,
        state: torch.Tensor,
        skill_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        *,
        result_embedding: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
    ) -> VNextMemoryUpdate:
        predicted, transition_delta, action = self.predict_memory(
            memory,
            state,
            skill_embedding,
            action_embedding,
        )
        if result_embedding is None:
            return VNextMemoryUpdate(
                predicted_memory=predicted,
                effective_memory=predicted,
                transition_delta=transition_delta,
                correction_delta=None,
                correction_beta=None,
                latent_trace=self.append_latent_trace(predicted, latent_trace),
            )
        effective, correction_delta, beta = self.correct_memory(
            predicted,
            state,
            skill_embedding,
            action,
            result_embedding,
            action_is_adapted=True,
        )
        return VNextMemoryUpdate(
            predicted_memory=predicted,
            effective_memory=effective,
            transition_delta=transition_delta,
            correction_delta=correction_delta,
            correction_beta=beta,
            latent_trace=self.append_latent_trace(effective, latent_trace),
        )

    def predict_memory(
        self,
        memory: torch.Tensor,
        state: torch.Tensor,
        skill_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_model_vector_layout(
            memory,
            state,
            skill_embedding,
            action_embedding,
            d=self.d,
        )
        action = self.action_adapter(action_embedding)
        memory, state, skill_embedding, action = _align_model_vector_dtypes(
            action,
            memory,
            state,
            skill_embedding,
            action,
            d=self.d,
        )
        transition_delta = self.transition_delta(memory, state, skill_embedding, action)
        predicted = self.memory_norm(
            memory
            + self.transition_scale().to(transition_delta.dtype)
            * torch.tanh(transition_delta)
        )
        return predicted, transition_delta, action

    def correct_memory(
        self,
        predicted_memory: torch.Tensor,
        state: torch.Tensor,
        skill_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        result_embedding: torch.Tensor,
        *,
        action_is_adapted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_model_vector_layout(
            predicted_memory,
            state,
            skill_embedding,
            action_embedding,
            result_embedding,
            d=self.d,
        )
        action = action_embedding if action_is_adapted else self.action_adapter(action_embedding)
        result = self.result_adapter(result_embedding)
        (
            predicted_memory,
            state,
            skill_embedding,
            action,
            result,
        ) = _align_model_vector_dtypes(
            result,
            predicted_memory,
            state,
            skill_embedding,
            action,
            result,
            d=self.d,
        )
        correction_delta = self.correction_delta(
            predicted_memory,
            state,
            skill_embedding,
            action,
            result,
        )
        beta = self.correction_gate(predicted_memory, result, correction_delta)
        effective = self.memory_norm(
            predicted_memory
            + beta
            * self.result_scale().to(correction_delta.dtype)
            * torch.tanh(correction_delta)
        )
        return effective, correction_delta, beta


def _validate_model_vector_layout(*values: torch.Tensor, d: int) -> None:
    if not values:
        return
    first = values[0]
    if first.ndim != 2 or int(first.size(-1)) != int(d):
        raise ValueError("vNext vectors must have shape [batch, d]")
    for value in values[1:]:
        if value.shape != first.shape:
            raise ValueError("vNext vectors must share shape")
        if value.device != first.device:
            raise ValueError("vNext vectors must share device")


def _align_model_vector_dtypes(
    reference: torch.Tensor,
    *values: torch.Tensor,
    d: int,
) -> tuple[torch.Tensor, ...]:
    """Align frozen-cache features to the recurrent autocast dtype.

    Frozen Qwen caches intentionally persist float32 vectors, while CUDA
    autocast may produce bf16 recurrent/static memories.  Device disagreement
    remains an error; dtype conversion is differentiable and happens once at
    the public vNext composition boundary before the strict inner modules.
    """

    _validate_model_vector_layout(reference, *values, d=d)
    return tuple(
        value
        if value.dtype == reference.dtype
        else value.to(dtype=reference.dtype)
        for value in values
    )


def _validate_model_vectors(*values: torch.Tensor, d: int) -> None:
    _validate_model_vector_layout(*values, d=d)
    if not values:
        return
    first = values[0]
    for value in values[1:]:
        if value.dtype != first.dtype:
            raise ValueError("vNext vectors must share dtype")
