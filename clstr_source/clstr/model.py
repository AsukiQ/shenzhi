from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.belief import BeliefGate, InitialBelief, TransitionPredictor, subspace_obs
from clstr.bridges.skillrouter.serialization import serialize_skill_text
from clstr.candidate_admission_residual import CandidateAdmissionResidualHead
from clstr.counterfactual_memory_calibration import RouteMemoryResidualAdapter
from clstr.data import ExecutionState, serialize_execution_state
from clstr.encoders import CrossEncoder, SkillTable, StateEncoder
from clstr.gated_temporal_reranker import GatedTemporalConfig, GatedTemporalReranker
from clstr.heads import SkillHead, StopHead, TransHead
from clstr.memory_utility_gate import MemoryUtilityGate
from clstr.state_query_prompt import (
    HEAD_V1,
    RAW_STATE_V1,
    SR_TASK_DESCRIPTION_V1,
    format_state_query,
    resolve_state_query_prompt_contract,
)
from clstr.success_value import QSuccessHead
from clstr.vnext_candidates import (
    NaturalCandidatePath,
    NaturalCandidateUnionPath,
    candidate_membership_mask,
    masked_topk_tensor,
    natural_compressed_support,
    static_plus_dynamic_extra_support,
)
from clstr.vnext_core import (
    CLSTRVNextCore,
    VNextCandidateCompressionScores,
    VNextCandidateRouteScores,
    VNextMemoryUpdate,
    VNextQueries,
)


@dataclass
class CLSTRConfig:
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    d: int = 256
    d_a: int | None = None
    top_k: int = 16
    gate_mode: str = "learned_elementwise"
    use_sn: bool = True
    encoder_pooling: str = "masked_mean"
    cross_encoder_pooling: str = "masked_mean"
    tokenizer_padding_side: str | None = None
    torch_dtype: str | None = None
    freeze_backbone: bool = False
    trust_remote_code: bool = True
    max_length: int | None = None
    projection_init: str = "default"
    normalize_embeddings: bool = False
    hf_cache_dir: str | None = None
    local_files_only: bool = False
    defer_skill_table_init: bool = False
    skill_text_format: str = "clstr"
    state_text_format: str = "raw"
    state_query_prompt_version: str | None = None
    state_query_max_chars: int | None = None
    state_query_truncation: str | None = None
    skill_table_batch_size: int = 32
    skill_table_adapter_init: str = "default"
    use_cross_encoder: bool = True
    mt_fusion_mode: str = "none"
    policy_skill_mode: str = "head"
    routing_prior_strength: float = 1.0
    policy_residual_scale: float = 1.0
    logit_scale_retr_init: float = 0.0
    logit_scale_belief_init: float = math.log(0.2)
    candidate_widen_factor: int = 4
    candidate_sample_temperature: float = 1.0
    skill_head_context: str = "belief"
    policy_candidate_encoder: str = "bi_encoder"
    initial_belief_top_k: int | None = 64
    vnext_hidden_dim: int = 256
    vnext_scale_initial: float = 0.01
    vnext_scale_maximum: float = 1.0
    vnext_temperature_initial: float = 10.0
    vnext_synchronization_enabled: bool = False
    vnext_synchronization_pair_dim: int = 96
    vnext_synchronization_trace_length: int = 8
    vnext_synchronization_scale_initial: float = 0.05
    frozen_backbone_snapshot_digest: str | None = None
    frozen_backbone_snapshot_manifest_path: str | None = None

    def __post_init__(self) -> None:
        if self.d_a is None:
            self.d_a = self.d
        if self.state_query_prompt_version is None:
            if self.state_text_format == "raw":
                self.state_query_prompt_version = RAW_STATE_V1
            elif self.state_text_format == "appworld_skillrouter_query":
                self.state_query_prompt_version = SR_TASK_DESCRIPTION_V1
            else:
                raise ValueError(f"unsupported state_text_format: {self.state_text_format}")
        contract = resolve_state_query_prompt_contract(
            prompt_version=self.state_query_prompt_version,
            max_chars=self.state_query_max_chars,
            truncation=self.state_query_truncation,
        )
        self.state_query_prompt_version = contract["state_query_prompt_version"]
        self.state_query_max_chars = contract["state_query_max_chars"]
        self.state_query_truncation = contract["state_query_truncation"]


def _skillret_official_skill_text(skill: dict) -> str:
    name = str(skill.get("name", "")).strip()
    desc = str(skill.get("description", "")).strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    return f"{name} | {desc} | {body}"


def _appworld_skillrouter_base_skill_text(skill: dict) -> str:
    name = str(skill.get("name", "")).strip()
    desc = str(skill.get("description", "")).strip()
    executor = str(skill.get("executor_desc", "")).strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    return f"{name} | {desc} | {executor} | {body}"


def _clstr_enriched_skill_text(skill: dict) -> str:
    return "\n".join(
        [
            f"name:{skill.get('name', '')}",
            f"desc:{skill.get('description', '')}",
            f"body:{skill.get('body', skill.get('skill_md', ''))}",
            f"action_templates:{skill.get('action_templates', [])}",
            f"positive_action_examples:{skill.get('positive_action_examples', [])}",
            f"negative_action_examples:{skill.get('negative_action_examples', [])}",
            f"preconditions:{skill.get('preconditions', [])}",
            f"effects:{skill.get('effects', [])}",
            f"in:{skill.get('input_schema', {})}",
            f"out:{skill.get('output_schema', {})}",
            f"fail:{skill.get('failure_modes', [])}",
        ]
    )


def _skill_text_serializer(format_name: str):
    if format_name == "clstr":
        return serialize_skill_text
    if format_name == "skillret_official":
        return _skillret_official_skill_text
    if format_name == "appworld_skillrouter_base":
        return _appworld_skillrouter_base_skill_text
    if format_name == "clstr_enriched":
        return _clstr_enriched_skill_text
    raise ValueError(f"unsupported skill_text_format: {format_name}")


def _appworld_skillrouter_query_text(text: str) -> str:
    return format_state_query(
        text,
        prompt_version=SR_TASK_DESCRIPTION_V1,
        max_chars=1500,
        truncation=HEAD_V1,
    )


def _standardize_last_dim(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float()
    if values.ndim == 0:
        return torch.zeros_like(values)
    if values.ndim == 1:
        if int(values.numel()) <= 1:
            return torch.zeros_like(values)
        std = values.std(unbiased=False)
        if float(std.detach().cpu().item()) <= 1.0e-8:
            return torch.zeros_like(values)
        return (values - values.mean()) / std.clamp_min(1.0e-8)
    if int(values.size(-1)) <= 1:
        return torch.zeros_like(values)
    std = values.std(dim=-1, unbiased=False, keepdim=True)
    mean = values.mean(dim=-1, keepdim=True)
    return torch.where(std > 1.0e-8, (values - mean) / std.clamp_min(1.0e-8), torch.zeros_like(values))


class UnifiedMemoryRetriever(nn.Module):
    """Fuse current state and belief memory into one retrieval vector."""

    def __init__(self, d: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    def forward(self, h_t: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([h_t, m_t, h_t * m_t], dim=-1))


class RouteMemoryUtilityGate(nn.Module):
    """Predict how much causal memory residual to use for one route row."""

    def __init__(self, d: int, *, initial_alpha: float = 0.01):
        super().__init__()
        if int(d) <= 0:
            raise ValueError("route memory utility gate dimension must be positive")
        initial = float(initial_alpha)
        if not math.isfinite(initial) or not 0.0 < initial < 1.0:
            raise ValueError("initial_alpha must be finite and strictly between zero and one")
        self.input_norm = nn.LayerNorm(int(d) * 4)
        self.hidden = nn.Linear(int(d) * 4, int(d))
        self.output = nn.Linear(int(d), 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.output.bias, math.log(initial / (1.0 - initial)))

    def forward(
        self,
        h_t: torch.Tensor,
        static_memory: torch.Tensor,
        dynamic_memory: torch.Tensor,
    ) -> torch.Tensor:
        if (
            h_t.ndim != 2
            or h_t.shape != static_memory.shape
            or h_t.shape != dynamic_memory.shape
        ):
            raise ValueError("route utility state and memory tensors must have matching rank-2 shapes")
        if not h_t.is_floating_point():
            raise ValueError("route utility inputs must use floating dtypes")
        if (
            h_t.dtype != static_memory.dtype
            or h_t.dtype != dynamic_memory.dtype
            or h_t.device != static_memory.device
            or h_t.device != dynamic_memory.device
        ):
            raise ValueError("route utility inputs must share dtype and device")
        delta = dynamic_memory - static_memory
        features = torch.cat((h_t, static_memory, delta, h_t * delta), dim=-1)
        hidden = torch.nn.functional.gelu(self.hidden(self.input_norm(features)))
        return torch.sigmoid(self.output(hidden).squeeze(-1))


class CLSTRModel(nn.Module):
    def __init__(self, config: CLSTRConfig, skills):
        super().__init__()
        self.config = config
        self.K = config.top_k
        self.skills = list(skills)
        encoder_kwargs = {
            "tokenizer_padding_side": config.tokenizer_padding_side,
            "torch_dtype": config.torch_dtype,
            "freeze_backbone": config.freeze_backbone,
            "trust_remote_code": config.trust_remote_code,
            "max_length": config.max_length,
            "projection_init": config.projection_init,
            "normalize_embeddings": config.normalize_embeddings,
            "hf_cache_dir": config.hf_cache_dir,
            "local_files_only": config.local_files_only,
        }
        self.encoder = StateEncoder(
            config.base_model_name,
            config.d,
            pooling=config.encoder_pooling,
            **encoder_kwargs,
        )
        self.cross_encoder = (
            CrossEncoder(
                config.base_model_name,
                config.d,
                pooling=config.cross_encoder_pooling,
                **encoder_kwargs,
            )
            if config.use_cross_encoder
            else None
        )
        self.skill_table = SkillTable(
            self.skills,
            self.encoder,
            config.d,
            trainable=False,
            initialize_embeddings=not config.defer_skill_table_init,
            skill_text_fn=_skill_text_serializer(config.skill_text_format),
            embedding_batch_size=config.skill_table_batch_size,
            adapter_init=config.skill_table_adapter_init,
            logit_scale_retr_init=config.logit_scale_retr_init,
            logit_scale_belief_init=config.logit_scale_belief_init,
        )
        self._vnext_skill_embedding_cache: dict[tuple[str, str, int, int], torch.Tensor] = {}
        self.initial_belief_head = InitialBelief(config.d)
        self.vnext = CLSTRVNextCore(
            config.d,
            hidden_dim=config.vnext_hidden_dim,
            scale_initial=config.vnext_scale_initial,
            scale_maximum=config.vnext_scale_maximum,
            temperature_initial=config.vnext_temperature_initial,
            synchronization_enabled=config.vnext_synchronization_enabled,
            synchronization_pair_dim=config.vnext_synchronization_pair_dim,
            synchronization_trace_length=config.vnext_synchronization_trace_length,
            synchronization_scale_initial=config.vnext_synchronization_scale_initial,
        )
        self.unified_retriever = UnifiedMemoryRetriever(config.d)
        self.route_memory_utility_gate = RouteMemoryUtilityGate(config.d)
        self._init_counterfactual_memory_calibration(config.d)
        self.action_proj = nn.Linear(config.d, config.d_a, bias=False)
        if config.d == config.d_a:
            with torch.no_grad():
                self.action_proj.weight.copy_(torch.eye(config.d))
        self.transition = TransitionPredictor(config.d, config.d_a, None)
        self.gate = BeliefGate(config.d, mode=config.gate_mode)
        self.skill_head = SkillHead(config.d)
        self.stop_head = StopHead(config.d)
        self.trans_head = TransHead(config.d, d_a=config.d_a, use_sn=config.use_sn)
        self.q_success_head = QSuccessHead(config.d)
        self.gated_temporal_reranker = GatedTemporalReranker(GatedTemporalConfig(dim=config.d))
        if str(config.mt_fusion_mode) in {"h_plus_m", "gated_residual"}:
            self.mt_fusion_proj = nn.Linear(config.d, config.d, bias=False)
            self.mt_fusion_norm = nn.LayerNorm(config.d)
            self.mt_skill_head = nn.Sequential(nn.Linear(config.d * 4, config.d), nn.GELU(), nn.Linear(config.d, 1))
            nn.init.eye_(self.mt_fusion_proj.weight)
            if str(config.mt_fusion_mode) == "gated_residual":
                self.mt_fusion_gate = nn.Linear(config.d * 2, config.d)
                nn.init.zeros_(self.mt_fusion_gate.weight)
                nn.init.zeros_(self.mt_fusion_gate.bias)
        elif str(config.mt_fusion_mode) != "none":
            raise ValueError(f"unsupported mt_fusion_mode: {config.mt_fusion_mode}")
        self._cross_encoder_cache: dict[tuple[str, int], torch.Tensor] | None = None

    def _init_counterfactual_memory_calibration(self, d: int) -> None:
        self.route_memory_residual_adapter = RouteMemoryResidualAdapter(int(d))
        self.route_memory_candidate_utility_gate = MemoryUtilityGate()
        self.route_memory_candidate_admission_residual = (
            CandidateAdmissionResidualHead(int(d))
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def stop_idx(self) -> int:
        return len(self.skills)

    def rebuild_skill_table(self) -> None:
        with torch.no_grad():
            self.skill_table.E.copy_(self.skill_table.rebuild_embeddings())
        self._vnext_skill_embedding_cache.clear()

    def vnext_normalized_skill_embeddings(
        self,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        embeddings = self.skill_table.E
        key = (
            str(embeddings.device),
            str(dtype),
            int(getattr(embeddings, "_version", 0)),
            int(embeddings.data_ptr()),
        )
        cached = self._vnext_skill_embedding_cache.get(key)
        if cached is None:
            cached = F.normalize(embeddings.detach().float(), p=2, dim=-1).to(
                device=embeddings.device,
                dtype=dtype,
            ).contiguous()
            self._vnext_skill_embedding_cache = {key: cached}
        return cached

    @staticmethod
    def serialize_state(state: ExecutionState | str) -> str:
        if isinstance(state, str):
            return state
        return serialize_execution_state(state)

    def _serialize_state_for_encoder(self, state: ExecutionState | str) -> str:
        text = self.serialize_state(state)
        config = getattr(self, "config", None)
        return format_state_query(
            text,
            prompt_version=getattr(config, "state_query_prompt_version", RAW_STATE_V1),
            max_chars=getattr(config, "state_query_max_chars", None),
            truncation=getattr(config, "state_query_truncation", None),
        )

    def encode_states(self, states: list[ExecutionState | str]) -> torch.Tensor:
        return self.encoder([self._serialize_state_for_encoder(state) for state in states])

    def encode_observations(self, observations: list[str]) -> torch.Tensor:
        return self.encoder(observations)

    def candidate_texts(self, candidate_rows: list[list[int]]) -> list[list[str]]:
        payloads: list[list[str]] = []
        skill_text_format = getattr(getattr(self, "config", None), "skill_text_format", "clstr")
        serializer = _skill_text_serializer(skill_text_format)
        for row in candidate_rows:
            payloads.append(
                [serializer(SkillTable._skill_payload(self.skills[idx])) for idx in row]
            )
        return payloads

    def batch_cross_encode(
        self,
        states: list[ExecutionState | str],
        candidate_rows: list[list[int]],
    ) -> torch.Tensor:
        if self.cross_encoder is None:
            raise RuntimeError("cross encoder is disabled for this CLSTRModel")
        state_texts = [self._serialize_state_for_encoder(state) for state in states]
        candidate_texts = self.candidate_texts(candidate_rows)
        cache = getattr(self, "_cross_encoder_cache", None)
        if cache is None:
            return self.cross_encoder.batch_forward(state_texts, candidate_texts)

        result_rows: list[list[torch.Tensor | None]] = [
            [None for _ in row] for row in candidate_rows
        ]
        missing_state_texts: list[str] = []
        missing_candidate_texts: list[list[str]] = []
        missing_positions: list[tuple[int, int, tuple[str, int]]] = []
        for row_idx, (state_text, row, text_row) in enumerate(zip(state_texts, candidate_rows, candidate_texts)):
            for col_idx, (skill_idx, candidate_text) in enumerate(zip(row, text_row)):
                key = (state_text, int(skill_idx))
                cached = cache.get(key)
                if cached is not None:
                    result_rows[row_idx][col_idx] = cached
                else:
                    missing_state_texts.append(state_text)
                    missing_candidate_texts.append([candidate_text])
                    missing_positions.append((row_idx, col_idx, key))

        if missing_positions:
            encoded = self.cross_encoder.batch_forward(missing_state_texts, missing_candidate_texts)
            for encoded_row, (row_idx, col_idx, key) in zip(encoded, missing_positions):
                emb = encoded_row[0]
                cache[key] = emb
                result_rows[row_idx][col_idx] = emb

        rows = []
        for row in result_rows:
            if any(item is None for item in row):
                raise RuntimeError("cross-encoder cache assembly failed")
            rows.append(torch.stack([item for item in row if item is not None], dim=0))
        return torch.stack(rows, dim=0)

    def reset_task_cache(self) -> None:
        self._cross_encoder_cache = {}

    def append_skills(self, new_skills: Sequence[Mapping[str, Any] | Any], **kwargs) -> dict[str, Any]:
        report = self.skill_table.append_skills(new_skills, **kwargs)
        if int(report.get("appended_count", 0)) > 0:
            self.skills = list(self.skill_table.skills)
            self._cross_encoder_cache = None
            self._vnext_skill_embedding_cache.clear()
        return report

    def sample_candidates(
        self,
        routing_logits: torch.Tensor,
        *,
        train: bool,
        k: int | None = None,
        widen_factor: int | None = None,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[list[list[int]], torch.Tensor]:
        logits = routing_logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2:
            raise ValueError("routing_logits must be rank-1 or rank-2")

        width = min(int(k or self.K), int(logits.size(-1)))
        if width <= 0:
            return [[] for _ in range(int(logits.size(0)))], torch.empty(logits.size(0), 0, device=logits.device)
        if not train:
            values, indices = torch.topk(logits, k=width, dim=-1)
            return [[int(idx) for idx in row.detach().cpu().tolist()] for row in indices], values

        temp = float(temperature if temperature is not None else getattr(self.config, "candidate_sample_temperature", 1.0))
        if temp <= 1.0e-6:
            values, indices = torch.topk(logits, k=width, dim=-1)
            return [[int(idx) for idx in row.detach().cpu().tolist()] for row in indices], values

        factor = max(1, int(widen_factor if widen_factor is not None else getattr(self.config, "candidate_widen_factor", 4)))
        top_m = min(int(logits.size(-1)), max(width, width * factor))
        top_values, top_indices = torch.topk(logits, k=top_m, dim=-1)
        sampled_rows: list[list[int]] = []
        sampled_values: list[torch.Tensor] = []
        for row_values, row_indices in zip(top_values, top_indices):
            probs = torch.softmax(row_values.float() / temp, dim=-1)
            local = torch.multinomial(probs, num_samples=width, replacement=False, generator=generator)
            sampled_idx = row_indices.index_select(0, local)
            sampled_rows.append([int(idx) for idx in sampled_idx.detach().cpu().tolist()])
            sampled_values.append(logits[0 if logits.size(0) == 1 else len(sampled_rows) - 1].index_select(0, sampled_idx))
        return sampled_rows, torch.stack(sampled_values, dim=0)

    def action_embeddings(self, action_ids: torch.Tensor) -> torch.Tensor:
        ids = action_ids.to(device=self.skill_table.E.device, dtype=torch.long)
        flat = ids.reshape(-1)
        action_features = torch.zeros(
            flat.size(0),
            self.skill_table.E.size(-1),
            device=self.skill_table.E.device,
            dtype=self.skill_table.E.dtype,
        )
        valid = (flat >= 0) & (flat < self.skill_table.E.size(0))
        if bool(valid.any().item()):
            action_features[valid] = self.skill_table.E.index_select(0, flat[valid])
        projected = self.action_proj(action_features)
        return projected.view(*ids.shape, -1)

    def policy_forward(
        self,
        h_t: torch.Tensor,
        m_t: torch.Tensor,
        candidate_embs: torch.Tensor,
        routing_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skill_logits = self._policy_skill_logits(h_t, m_t, candidate_embs)
        skill_logits = self._apply_policy_skill_mode(skill_logits, routing_logits)
        stop_logits = self.stop_head(h_t, m_t)
        if stop_logits.ndim == 2 and stop_logits.size(-1) == 1:
            stop_logits = stop_logits.squeeze(-1)
        return torch.cat([skill_logits, stop_logits.unsqueeze(-1)], dim=-1)

    def route_logits_from_candidates(
        self,
        states: list[ExecutionState | str],
        m_t: torch.Tensor,
        candidate_rows: list[list[int]],
    ) -> torch.Tensor:
        h_t = self.encode_states(states)
        skill_table = getattr(self, "skill_table", None)
        if skill_table is None:
            raise RuntimeError("route_logits_from_candidates requires skill_table")
        if hasattr(skill_table, "retrieval_logits"):
            full_logits = skill_table.retrieval_logits(h_t)
        elif hasattr(skill_table, "logits"):
            full_logits = skill_table.logits(h_t)
        else:
            raise RuntimeError("route_logits_from_candidates requires skill_table.retrieval_logits")
        if full_logits.ndim != 2:
            raise ValueError("route_logits_from_candidates expects rank-2 retrieval logits")
        batch_size = len(candidate_rows)
        if int(full_logits.size(0)) == 1 and batch_size > 1:
            full_logits = full_logits.expand(batch_size, -1)
        if int(full_logits.size(0)) != batch_size:
            raise ValueError("candidate_rows batch size must match retrieval logits batch size")
        max_width = max((len(row) for row in candidate_rows), default=0)
        if max_width <= 0:
            candidate_logits = full_logits[:, :0]
        else:
            ids = torch.zeros(batch_size, max_width, dtype=torch.long, device=full_logits.device)
            valid = torch.zeros(batch_size, max_width, dtype=torch.bool, device=full_logits.device)
            for row_idx, row in enumerate(candidate_rows):
                if not row:
                    continue
                row_ids = torch.tensor([int(idx) for idx in row], dtype=torch.long, device=full_logits.device)
                width = int(row_ids.numel())
                ids[row_idx, :width] = row_ids
                valid[row_idx, :width] = True
            candidate_logits = full_logits.gather(1, ids).masked_fill(
                ~valid,
                torch.finfo(full_logits.dtype).min,
            )
        stop_logits = self.stop_head(h_t, m_t)
        if stop_logits.ndim == 2 and stop_logits.size(-1) == 1:
            stop_logits = stop_logits.squeeze(-1)
        return torch.cat([candidate_logits, stop_logits.to(candidate_logits.device).unsqueeze(-1)], dim=-1)

    def unified_route_full_logits(self, h_t: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        skill_table = getattr(self, "skill_table", None)
        if skill_table is None or not hasattr(skill_table, "E"):
            raise RuntimeError("unified_route_full_logits requires skill_table.E")
        z_t = self.unified_retriever(h_t, m_t)
        skill_embs = skill_table.E.to(device=z_t.device, dtype=z_t.dtype)
        return z_t @ skill_embs.t()

    def route_memory_alpha(
        self,
        h_t: torch.Tensor,
        static_memory: torch.Tensor,
        dynamic_memory: torch.Tensor,
        causal_update_count: torch.Tensor,
    ) -> torch.Tensor:
        raw_alpha = self.route_memory_utility_gate(h_t, static_memory, dynamic_memory)
        if (
            causal_update_count.ndim != 1
            or int(causal_update_count.numel()) != int(raw_alpha.numel())
        ):
            raise ValueError("causal_update_count must have one value per route row")
        counts = causal_update_count.to(device=raw_alpha.device, dtype=raw_alpha.dtype)
        if not torch.isfinite(counts).all() or bool((counts < 0).any()):
            raise ValueError("causal_update_count must be finite and nonnegative")
        return torch.where(counts > 0, raw_alpha, torch.zeros_like(raw_alpha))

    def gather_unified_route_logits(
        self,
        full_logits: torch.Tensor,
        candidate_rows: list[list[int]],
    ) -> torch.Tensor:
        if full_logits.ndim != 2:
            raise ValueError("full_logits must have shape [batch, skills]")
        batch_size = len(candidate_rows)
        if int(full_logits.size(0)) == 1 and batch_size > 1:
            full_logits = full_logits.expand(batch_size, -1)
        if int(full_logits.size(0)) != batch_size:
            raise ValueError("candidate_rows batch size must match full_logits")
        max_width = max((len(row) for row in candidate_rows), default=0)
        if max_width <= 0:
            return full_logits[:, :0]
        ids = torch.zeros(batch_size, max_width, dtype=torch.long, device=full_logits.device)
        valid = torch.zeros(batch_size, max_width, dtype=torch.bool, device=full_logits.device)
        for row_idx, row in enumerate(candidate_rows):
            if any(int(idx) < 0 or int(idx) >= int(full_logits.size(1)) for idx in row):
                raise ValueError("candidate skill index is outside full_logits")
            if not row:
                continue
            row_ids = torch.tensor([int(idx) for idx in row], dtype=torch.long, device=full_logits.device)
            width = int(row_ids.numel())
            ids[row_idx, :width] = row_ids
            valid[row_idx, :width] = True
        return full_logits.gather(1, ids).masked_fill(~valid, torch.finfo(full_logits.dtype).min)

    def unified_route_logits(
        self,
        h_t: torch.Tensor,
        m_t: torch.Tensor,
        candidate_rows: list[list[int]] | None = None,
    ) -> torch.Tensor:
        full_logits = self.unified_route_full_logits(h_t, m_t)
        if candidate_rows is None:
            return full_logits
        return self.gather_unified_route_logits(full_logits, candidate_rows)

    def initial_belief(
        self,
        h_t: torch.Tensor,
        top_k: int | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        resolved_top_k = top_k
        if resolved_top_k is None:
            resolved_top_k = getattr(getattr(self, "config", None), "initial_belief_top_k", 64)
        sparse_belief = subspace_obs(
            self.skill_table,
            h_t,
            top_k=resolved_top_k,
            valid_mask=valid_mask,
        )
        return self.initial_belief_head(h_t, sparse_belief)

    def vnext_initial_belief(
        self,
        h_t: torch.Tensor,
        runtime_visible_mask: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> torch.Tensor:
        # InitialBelief already ends in LayerNorm.  Keep its output byte-aligned
        # with the verified legacy unified scorer; recurrent updates normalize
        # subsequent memories inside the vNext transition/correction core.
        return self.initial_belief(
            h_t,
            top_k=top_k,
            valid_mask=runtime_visible_mask,
        )

    def vnext_queries(
        self,
        h_t: torch.Tensor,
        dynamic_memory: torch.Tensor,
        static_memory: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        memory_state: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
    ) -> VNextQueries:
        return self.vnext.queries(
            h_t,
            dynamic_memory,
            static_memory,
            history_mask,
            memory_state=memory_state,
            latent_trace=latent_trace,
        )

    def vnext_full_pool_logits(
        self,
        query: torch.Tensor,
        *,
        head: str,
    ) -> torch.Tensor:
        return self.vnext.full_pool_logits(
            query,
            self.vnext_normalized_skill_embeddings(dtype=query.dtype),
            head=head,
            skill_embeddings_are_normalized=True,
        )

    def vnext_candidate_logits(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        head: str = "route",
    ) -> torch.Tensor:
        if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
            raise ValueError("vNext candidate ids and validity must match [batch, candidates]")
        if int(candidate_ids.size(0)) != int(query.size(0)):
            raise ValueError("vNext candidate rows must match query rows")
        embeddings = self.vnext_normalized_skill_embeddings(dtype=query.dtype)
        selected = embeddings.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0),
            candidate_ids.size(1),
            -1,
        )
        if head == "route":
            temperature = self.vnext.static_query.temperature()
        elif head == "recall":
            temperature = self.vnext.static_query.temperature()
        else:
            raise ValueError(f"unsupported vNext candidate head: {head}")
        logits = temperature.to(device=query.device, dtype=query.dtype) * torch.einsum(
            "bd,bcd->bc",
            query,
            selected,
        )
        return logits.masked_fill(
            ~valid_mask.to(device=query.device, dtype=torch.bool),
            torch.finfo(logits.dtype).min,
        )

    def vnext_candidate_compression_scores(
        self,
        causal_query: torch.Tensor,
        current_state: torch.Tensor,
        candidate_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> VNextCandidateCompressionScores:
        if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
            raise ValueError("vNext compressor ids and validity must match [batch, candidates]")
        if base_logits.shape != candidate_ids.shape:
            raise ValueError("vNext compressor base logits must match candidate ids")
        embeddings = self.vnext_normalized_skill_embeddings(dtype=causal_query.dtype)
        selected = embeddings.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0),
            candidate_ids.size(1),
            -1,
        )
        return self.vnext.candidate_compression_scores(
            causal_query,
            current_state,
            selected,
            base_logits,
            valid_mask,
        )

    def vnext_natural_candidate_path(
        self,
        query: torch.Tensor,
        current_state: torch.Tensor,
        legal_mask: torch.Tensor,
        *,
        coarse_k: int,
        compressed_m: int,
    ) -> NaturalCandidatePath:
        """Build one label-free full-pool→coarse→compressed candidate path."""

        if query.ndim != 2 or current_state.shape != query.shape:
            raise ValueError("vNext candidate path states must match [batch, d]")
        if legal_mask.ndim != 2 or int(legal_mask.size(0)) != int(query.size(0)):
            raise ValueError("vNext candidate path legal mask must match the batch")
        recall_logits = self.vnext_full_pool_logits(query, head="recall")
        if int(recall_logits.size(1)) != int(legal_mask.size(1)):
            raise ValueError("vNext candidate path legal mask must match the skill pool")
        coarse_ids, coarse_valid = masked_topk_tensor(
            recall_logits,
            legal_mask,
            k=coarse_k,
        )
        coarse_logits = self.vnext_candidate_logits(
            query,
            coarse_ids,
            coarse_valid,
            head="recall",
        )
        compression = self.vnext_candidate_compression_scores(
            query,
            current_state,
            coarse_ids,
            coarse_valid,
            coarse_logits,
        )
        support = natural_compressed_support(
            coarse_ids,
            coarse_valid,
            compression.logits,
            compressed_m=compressed_m,
        )
        return NaturalCandidatePath(
            query=query,
            recall_logits=recall_logits,
            coarse_candidate_ids=coarse_ids,
            coarse_valid_mask=coarse_valid,
            coarse_base_logits=coarse_logits,
            compression_logits=compression.logits,
            support=support,
        )

    def vnext_natural_candidate_union_path(
        self,
        static_query: torch.Tensor,
        dynamic_query: torch.Tensor,
        current_state: torch.Tensor,
        legal_mask: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        coarse_k: int,
        dynamic_extra_k: int,
    ) -> NaturalCandidateUnionPath:
        """Build static Top-K plus label-free memory-only candidate extras."""

        if static_query.shape != dynamic_query.shape or current_state.shape != static_query.shape:
            raise ValueError("vNext union candidate queries must match [batch, d]")
        static_logits = self.vnext_full_pool_logits(static_query, head="recall")
        dynamic_logits = self.vnext_full_pool_logits(dynamic_query, head="recall")
        if legal_mask.shape != static_logits.shape:
            raise ValueError("vNext union legal mask must match the full skill pool")
        coarse_ids, coarse_valid = masked_topk_tensor(
            static_logits,
            legal_mask,
            k=coarse_k,
        )
        coarse_base_logits = self.vnext_candidate_logits(
            static_query,
            coarse_ids,
            coarse_valid,
            head="recall",
        )
        # Auxiliary Top500->Top64 coverage is supervised by the dynamic recall
        # query, but the final route never discards the remaining static Top500.
        compression_logits = self.vnext_candidate_logits(
            dynamic_query,
            coarse_ids,
            coarse_valid,
            head="recall",
        )
        proposal_changed = (
            history_mask.to(device=dynamic_query.device, dtype=torch.bool)
            & dynamic_query.ne(static_query).any(dim=-1)
        )
        support, extra_ids, extra_valid = static_plus_dynamic_extra_support(
            coarse_ids,
            coarse_valid,
            dynamic_logits,
            legal_mask,
            proposal_changed,
            dynamic_extra_k=dynamic_extra_k,
        )
        return NaturalCandidateUnionPath(
            static_query=static_query,
            dynamic_query=dynamic_query,
            static_recall_logits=static_logits,
            recall_logits=dynamic_logits,
            coarse_candidate_ids=coarse_ids,
            coarse_valid_mask=coarse_valid,
            coarse_base_logits=coarse_base_logits,
            compression_logits=compression_logits,
            dynamic_extra_candidate_ids=extra_ids,
            dynamic_extra_valid_mask=extra_valid,
            support=support,
        )

    def vnext_safe_candidate_route_scores(
        self,
        h_t: torch.Tensor,
        dynamic_memory: torch.Tensor,
        static_memory: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        static_candidate_ids: torch.Tensor | None = None,
        static_valid_mask: torch.Tensor | None = None,
        history_depth: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
        hard_fallback: bool,
        residual_bound: float = 2.0,
    ) -> VNextCandidateRouteScores:
        if candidate_ids.ndim != 2 or valid_mask.shape != candidate_ids.shape:
            raise ValueError("vNext candidate ids and validity must match [batch, candidates]")
        if (static_candidate_ids is None) != (static_valid_mask is None):
            raise ValueError("static candidate ids and validity must be provided together")
        static_support_mask = (
            valid_mask.to(device=candidate_ids.device, dtype=torch.bool)
            if static_candidate_ids is None
            else candidate_membership_mask(
                candidate_ids,
                valid_mask,
                static_candidate_ids,
                static_valid_mask,
            )
        )
        embeddings = self.vnext_normalized_skill_embeddings(dtype=h_t.dtype)
        selected = embeddings.index_select(0, candidate_ids.reshape(-1)).view(
            candidate_ids.size(0),
            candidate_ids.size(1),
            -1,
        )
        return self.vnext.candidate_route_scores(
            h_t,
            dynamic_memory,
            static_memory,
            history_mask,
            selected,
            valid_mask,
            static_support_mask=static_support_mask,
            history_depth=history_depth,
            memory_state=memory_state,
            latent_trace=latent_trace,
            temperature=self.vnext.static_query.temperature(),
            hard_fallback=bool(hard_fallback),
            residual_bound=float(residual_bound),
        )

    def vnext_update_memory(
        self,
        memory: torch.Tensor,
        state: torch.Tensor,
        skill_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        *,
        result_embedding: torch.Tensor | None = None,
        latent_trace: torch.Tensor | None = None,
    ) -> VNextMemoryUpdate:
        return self.vnext.update_memory(
            memory,
            state,
            skill_embedding,
            action_embedding,
            result_embedding=result_embedding,
            latent_trace=latent_trace,
        )

    def _apply_policy_skill_mode(
        self,
        skill_logits: torch.Tensor,
        routing_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        mode = str(getattr(getattr(self, "config", None), "policy_skill_mode", "head"))
        if mode == "head":
            return skill_logits
        if mode != "prior_residual":
            raise ValueError(f"unsupported policy_skill_mode: {mode}")
        if routing_logits is None:
            prior = torch.zeros_like(skill_logits)
        else:
            prior = routing_logits.to(device=skill_logits.device, dtype=torch.float32)
            if prior.ndim == 1 and skill_logits.ndim == 2:
                prior = prior.unsqueeze(0)
            if prior.shape != skill_logits.shape:
                if prior.ndim == skill_logits.ndim and prior.size(0) == 1 and skill_logits.ndim == 2:
                    prior = prior.expand_as(skill_logits)
                else:
                    prior = prior.reshape(skill_logits.shape)
            prior = _standardize_last_dim(prior)
        residual_scale = float(getattr(getattr(self, "config", None), "policy_residual_scale", 1.0))
        prior_strength = float(getattr(getattr(self, "config", None), "routing_prior_strength", 1.0))
        return prior_strength * prior + residual_scale * skill_logits

    def _policy_skill_logits(
        self,
        h_t: torch.Tensor,
        m_t: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(getattr(getattr(self, "config", None), "mt_fusion_mode", "none"))
        if mode == "none":
            context_mode = str(getattr(getattr(self, "config", None), "skill_head_context", "belief"))
            if context_mode in {"belief", "m_t"}:
                context = m_t
            elif context_mode == "belief_residual":
                context = m_t - h_t
            else:
                raise ValueError(f"unsupported skill_head_context: {context_mode}")
            return self.skill_head(candidate_embs, context)
        if mode == "h_plus_m":
            h_cond = self.mt_fusion_norm(h_t + self.mt_fusion_proj(m_t))
        elif mode == "gated_residual":
            gate = torch.sigmoid(self.mt_fusion_gate(torch.cat([h_t, m_t], dim=-1)))
            h_cond = self.mt_fusion_norm(h_t + gate * self.mt_fusion_proj(m_t))
        else:
            raise ValueError(f"unsupported mt_fusion_mode: {mode}")
        if candidate_embs.ndim == 2:
            feat = torch.cat([candidate_embs, m_t, h_cond, candidate_embs * h_cond], dim=-1)
        else:
            m_exp = m_t.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)
            h_exp = h_cond.unsqueeze(1).expand(-1, candidate_embs.size(1), -1)
            feat = torch.cat([candidate_embs, m_exp, h_exp, candidate_embs * h_exp], dim=-1)
        return self.mt_skill_head(feat).squeeze(-1)

    def transition_logits(
        self,
        m_hat: torch.Tensor,
        candidate_rows: list[list[int]],
    ) -> torch.Tensor:
        idx = torch.tensor(candidate_rows, device=self.device, dtype=torch.long)
        cand_emb = self.action_embeddings(idx)
        return self.trans_head(m_hat, cand_emb)

    def step_update(
        self,
        m_t: torch.Tensor,
        a_t: torch.Tensor,
        o_t_emb: torch.Tensor,
        x_next_text: list[str],
    ):
        action_emb = self.action_embeddings(a_t)
        m_hat = self.transition(m_t, action_emb, o_t_emb)
        h_next = self.encoder(x_next_text)
        m_obs = subspace_obs(self.skill_table, h_next)
        gamma = self.gate(m_hat, m_obs, o_t_emb)
        m_next = gamma * m_obs + (1.0 - gamma) * m_hat
        return m_next, m_hat, m_obs
