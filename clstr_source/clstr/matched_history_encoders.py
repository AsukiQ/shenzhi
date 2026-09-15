from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from clstr.vnext_core import (
    BoundedLayerScale,
    CorrectionDelta,
    NonAffineRMSNorm,
    ScalarCorrectionGate,
    TransitionDelta,
    TwoLayerAdapter,
)


HistoryEncoderKind = Literal["serialized", "gru", "transformer", "lstr"]


def _validate_inputs(
    initial_memory: torch.Tensor,
    event_states: torch.Tensor,
    event_skills: torch.Tensor,
    event_actions: torch.Tensor,
    event_results: torch.Tensor,
    event_mask: torch.Tensor,
    result_mask: torch.Tensor,
    *,
    d: int,
) -> tuple[int, int]:
    if initial_memory.ndim != 2 or int(initial_memory.size(-1)) != int(d):
        raise ValueError("initial memory must have shape [batch, d]")
    batch = int(initial_memory.size(0))
    sequence_shape = tuple(event_states.shape)
    if len(sequence_shape) != 3 or sequence_shape[0] != batch or sequence_shape[2] != int(d):
        raise ValueError("event states must have shape [batch, time, d]")
    for name, value in (
        ("event skills", event_skills),
        ("event actions", event_actions),
        ("event results", event_results),
    ):
        if tuple(value.shape) != sequence_shape:
            raise ValueError(f"{name} must match event-state shape")
        if value.device != initial_memory.device:
            raise ValueError(f"{name} must share the initial-memory device")
    mask_shape = sequence_shape[:2]
    if tuple(event_mask.shape) != mask_shape or tuple(result_mask.shape) != mask_shape:
        raise ValueError("event and result masks must have shape [batch, time]")
    if event_mask.device != initial_memory.device or result_mask.device != initial_memory.device:
        raise ValueError("event and result masks must share the initial-memory device")
    event_mask = event_mask.to(dtype=torch.bool)
    result_mask = result_mask.to(dtype=torch.bool)
    if bool((result_mask & ~event_mask).any().item()):
        raise ValueError("result mask cannot enable a padded event")
    return batch, int(sequence_shape[1])


def _event_features(
    event_states: torch.Tensor,
    event_skills: torch.Tensor,
    event_actions: torch.Tensor,
    event_results: torch.Tensor,
    result_mask: torch.Tensor,
) -> torch.Tensor:
    visible_results = torch.where(
        result_mask.to(dtype=torch.bool).unsqueeze(-1),
        event_results,
        torch.zeros_like(event_results),
    )
    presence = result_mask.to(dtype=event_states.dtype).unsqueeze(-1)
    return torch.cat(
        (event_states, event_skills, event_actions, visible_results, presence),
        dim=-1,
    )


class SerializedHistoryEncoder(nn.Module):
    """Adapt one frozen embedding of the deterministic factual-history text."""

    def __init__(self, d: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.d = int(d)
        hidden = max(1, 5 * self.d // 4) if hidden_dim is None else int(hidden_dim)
        self.adapter = TwoLayerAdapter(3 * self.d, self.d, hidden_dim=hidden)
        self.scale = BoundedLayerScale(initial=0.1, maximum=4.0)
        self.memory_norm = NonAffineRMSNorm(self.d)

    def forward(
        self,
        initial_memory: torch.Tensor,
        event_states: torch.Tensor,
        event_skills: torch.Tensor,
        event_actions: torch.Tensor,
        event_results: torch.Tensor,
        event_mask: torch.Tensor,
        result_mask: torch.Tensor,
        *,
        serialized_history_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, _time = _validate_inputs(
            initial_memory,
            event_states,
            event_skills,
            event_actions,
            event_results,
            event_mask,
            result_mask,
            d=self.d,
        )
        history = event_mask.to(dtype=torch.bool).any(dim=-1)
        if serialized_history_embedding is None:
            if bool(history.any().item()):
                raise ValueError("serialized encoder requires factual-history embeddings")
            return initial_memory
        if tuple(serialized_history_embedding.shape) != (batch, self.d):
            raise ValueError("serialized history embedding must have shape [batch, d]")
        if serialized_history_embedding.device != initial_memory.device:
            raise ValueError("serialized history embedding must share the memory device")
        serialized = serialized_history_embedding.to(dtype=initial_memory.dtype)
        delta = self.adapter(
            torch.cat((initial_memory, serialized, initial_memory * serialized), dim=-1)
        )
        encoded = self.memory_norm(
            initial_memory + self.scale().to(dtype=delta.dtype) * torch.tanh(delta)
        )
        return torch.where(history.unsqueeze(-1), encoded, initial_memory)


class GRUHistoryEncoder(nn.Module):
    """Generic recurrent encoder over the shared factual event vectors."""

    def __init__(self, d: int, *, input_dim: int = 256) -> None:
        super().__init__()
        self.d = int(d)
        self.input_dim = int(input_dim)
        if min(self.d, self.input_dim) <= 0:
            raise ValueError("GRU history dimensions must be positive")
        self.event_projection = nn.Sequential(
            nn.LayerNorm(4 * self.d + 1),
            nn.Linear(4 * self.d + 1, self.input_dim),
            nn.GELU(),
        )
        self.cell = nn.GRUCell(self.input_dim, self.d)
        self.memory_norm = NonAffineRMSNorm(self.d)

    def forward(
        self,
        initial_memory: torch.Tensor,
        event_states: torch.Tensor,
        event_skills: torch.Tensor,
        event_actions: torch.Tensor,
        event_results: torch.Tensor,
        event_mask: torch.Tensor,
        result_mask: torch.Tensor,
        *,
        serialized_history_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del serialized_history_embedding
        _batch, time = _validate_inputs(
            initial_memory,
            event_states,
            event_skills,
            event_actions,
            event_results,
            event_mask,
            result_mask,
            d=self.d,
        )
        features = self.event_projection(
            _event_features(
                event_states,
                event_skills,
                event_actions,
                event_results,
                result_mask,
            )
        )
        memory = initial_memory
        mask = event_mask.to(dtype=torch.bool)
        for index in range(time):
            candidate = self.memory_norm(self.cell(features[:, index], memory))
            memory = torch.where(mask[:, index].unsqueeze(-1), candidate, memory)
        return memory


class CausalTransformerHistoryEncoder(nn.Module):
    """Causal self-attention over the complete bounded factual prefix."""

    def __init__(
        self,
        d: int,
        *,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        max_horizon: int = 16,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.model_dim = int(model_dim)
        self.max_horizon = int(max_horizon)
        if self.max_horizon <= 0:
            raise ValueError("Transformer history horizon must be positive")
        if self.model_dim <= 0 or self.model_dim % int(num_heads) != 0:
            raise ValueError("Transformer model dimension must be divisible by heads")
        self.initial_projection = nn.Linear(self.d, self.model_dim)
        self.event_projection = nn.Sequential(
            nn.LayerNorm(4 * self.d + 1),
            nn.Linear(4 * self.d + 1, self.model_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(self.max_horizon + 1, self.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(self.model_dim, self.d)
        self.memory_norm = NonAffineRMSNorm(self.d)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        initial_memory: torch.Tensor,
        event_states: torch.Tensor,
        event_skills: torch.Tensor,
        event_actions: torch.Tensor,
        event_results: torch.Tensor,
        event_mask: torch.Tensor,
        result_mask: torch.Tensor,
        *,
        serialized_history_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del serialized_history_embedding
        batch, time = _validate_inputs(
            initial_memory,
            event_states,
            event_skills,
            event_actions,
            event_results,
            event_mask,
            result_mask,
            d=self.d,
        )
        if time > self.max_horizon:
            raise ValueError("event sequence exceeds the configured Transformer horizon")
        events = self.event_projection(
            _event_features(
                event_states,
                event_skills,
                event_actions,
                event_results,
                result_mask,
            )
        )
        tokens = torch.cat((self.initial_projection(initial_memory).unsqueeze(1), events), dim=1)
        tokens = tokens + self.position_embedding[: time + 1].to(dtype=tokens.dtype)
        event_valid = event_mask.to(dtype=torch.bool)
        padding_mask = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=event_valid.device),
                ~event_valid,
            ),
            dim=1,
        )
        causal_mask = torch.triu(
            torch.ones(time + 1, time + 1, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        encoded = self.encoder(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        lengths = event_valid.sum(dim=-1, dtype=torch.long)
        gather = lengths.view(batch, 1, 1).expand(-1, 1, self.model_dim)
        final_token = encoded.gather(1, gather).squeeze(1)
        candidate = self.memory_norm(initial_memory + self.output_projection(final_token))
        history = lengths.gt(0)
        return torch.where(history.unsqueeze(-1), candidate, initial_memory)


class LSTRHistoryEncoder(nn.Module):
    """The production LSTR transition/correction architecture without route heads."""

    def __init__(
        self,
        d: int,
        *,
        hidden_dim: int = 256,
        transition_scale_initial: float = 0.05,
        result_scale_initial: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = int(d)
        hidden = min(int(hidden_dim), self.d)
        self.memory_norm = NonAffineRMSNorm(self.d)
        self.action_adapter = nn.Linear(self.d, self.d, bias=False)
        self.result_adapter = nn.Linear(self.d, self.d, bias=False)
        nn.init.eye_(self.action_adapter.weight)
        nn.init.eye_(self.result_adapter.weight)
        self.transition_delta = TransitionDelta(self.d, hidden_dim=hidden)
        self.correction_delta = CorrectionDelta(self.d, hidden_dim=hidden)
        self.correction_gate = ScalarCorrectionGate(
            self.d,
            hidden_dim=min(64, hidden),
        )
        self.transition_scale = BoundedLayerScale(
            initial=float(transition_scale_initial),
            maximum=1.0,
        )
        self.result_scale = BoundedLayerScale(
            initial=float(result_scale_initial),
            maximum=1.0,
        )

    def _predict(
        self,
        memory: torch.Tensor,
        state: torch.Tensor,
        skill: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.action_adapter(action_embedding)
        memory, state, skill = (
            value.to(device=action.device, dtype=action.dtype)
            for value in (memory, state, skill)
        )
        delta = self.transition_delta(memory, state, skill, action)
        predicted = self.memory_norm(
            memory + self.transition_scale().to(dtype=delta.dtype) * torch.tanh(delta)
        )
        return predicted, action

    def _correct(
        self,
        predicted: torch.Tensor,
        state: torch.Tensor,
        skill: torch.Tensor,
        action: torch.Tensor,
        result_embedding: torch.Tensor,
    ) -> torch.Tensor:
        result = self.result_adapter(result_embedding)
        predicted, state, skill, action = (
            value.to(device=result.device, dtype=result.dtype)
            for value in (predicted, state, skill, action)
        )
        delta = self.correction_delta(predicted, state, skill, action, result)
        beta = self.correction_gate(predicted, result, delta)
        return self.memory_norm(
            predicted
            + beta * self.result_scale().to(dtype=delta.dtype) * torch.tanh(delta)
        )

    def forward(
        self,
        initial_memory: torch.Tensor,
        event_states: torch.Tensor,
        event_skills: torch.Tensor,
        event_actions: torch.Tensor,
        event_results: torch.Tensor,
        event_mask: torch.Tensor,
        result_mask: torch.Tensor,
        *,
        serialized_history_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del serialized_history_embedding
        _batch, time = _validate_inputs(
            initial_memory,
            event_states,
            event_skills,
            event_actions,
            event_results,
            event_mask,
            result_mask,
            d=self.d,
        )
        valid = event_mask.to(dtype=torch.bool)
        has_result = result_mask.to(dtype=torch.bool)
        memory = initial_memory
        for index in range(time):
            predicted, action = self._predict(
                memory,
                event_states[:, index],
                event_skills[:, index],
                event_actions[:, index],
            )
            corrected = predicted
            result_rows = valid[:, index] & has_result[:, index]
            if bool(result_rows.any().item()):
                selected = result_rows.nonzero(as_tuple=False).view(-1)
                selected_corrected = self._correct(
                    predicted.index_select(0, selected),
                    event_states[:, index].index_select(0, selected),
                    event_skills[:, index].index_select(0, selected),
                    action.index_select(0, selected),
                    event_results[:, index].index_select(0, selected),
                )
                corrected = predicted.index_copy(0, selected, selected_corrected)
            memory = torch.where(valid[:, index].unsqueeze(-1), corrected, memory)
        return memory


def build_matched_history_encoder(
    kind: HistoryEncoderKind,
    d: int,
    *,
    max_horizon: int = 16,
) -> nn.Module:
    resolved = str(kind)
    if resolved == "serialized":
        return SerializedHistoryEncoder(d)
    if resolved == "gru":
        return GRUHistoryEncoder(d)
    if resolved == "transformer":
        return CausalTransformerHistoryEncoder(d, max_horizon=max_horizon)
    if resolved == "lstr":
        return LSTRHistoryEncoder(d)
    raise ValueError(f"unsupported matched history encoder: {resolved}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)
