from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.bridges.skillrouter.serialization import serialize_skill_text
from clstr.data import serialize_execution_state

try:
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - exercised via StateEncoder instantiation only
    AutoModel = None
    AutoTokenizer = None

VALID_POOLING = {"masked_mean", "last_token", "cls"}


def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> str | torch.dtype | None:
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"unsupported torch_dtype: {torch_dtype}")
    return mapping[torch_dtype]


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "masked_mean":
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    if pooling == "last_token":
        if mask[:, -1].sum() == mask.shape[0]:
            return hidden[:, -1]
        seq_lens = mask.sum(dim=1).to(torch.long) - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_index, seq_lens]
    if pooling == "cls":
        return hidden[:, 0]
    raise ValueError(f"unsupported pooling mode: {pooling}")


class StateEncoder(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        d: int,
        pooling: str = "masked_mean",
        tokenizer_padding_side: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        freeze_backbone: bool = False,
        trust_remote_code: bool = True,
        max_length: int | None = None,
        projection_init: str = "default",
        normalize_embeddings: bool = False,
        hf_cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        if pooling not in VALID_POOLING:
            raise ValueError(f"unsupported pooling mode: {pooling}")
        if AutoModel is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required to instantiate StateEncoder")
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": hf_cache_dir,
            "local_files_only": local_files_only,
        }
        if tokenizer_padding_side is not None:
            tokenizer_kwargs["padding_side"] = tokenizer_padding_side
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tokenizer_kwargs)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": hf_cache_dir,
            "local_files_only": local_files_only,
        }
        resolved_dtype = _resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        self.backbone = AutoModel.from_pretrained(base_model_name, **model_kwargs)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.proj = nn.Linear(self.backbone.config.hidden_size, d)
        if projection_init == "identity":
            if self.backbone.config.hidden_size != d:
                raise ValueError(
                    "identity projection_init requires d to match backbone hidden_size "
                    f"({d} != {self.backbone.config.hidden_size})"
                )
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(d))
                self.proj.bias.zero_()
        elif projection_init != "default":
            raise ValueError(f"unsupported projection_init: {projection_init}")
        self.pooling = pooling
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone
        self.projection_init = projection_init
        self.normalize_embeddings = normalize_embeddings
        self.hf_cache_dir = hf_cache_dir
        self.local_files_only = bool(local_files_only)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def tokenize(self, text_batch: list[str]) -> dict[str, torch.Tensor]:
        max_length = self.max_length or getattr(self.backbone.config, "max_position_embeddings", None)
        tok = self.tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.backbone.device) for k, v in tok.items()}

    def encode_tokenized(self, tok: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(**tok)
        pooled = pool_hidden(out.last_hidden_state, tok["attention_mask"], self.pooling)
        return self.project_pooled(pooled)

    def encode_backbone_pooled(self, text_batch: list[str]) -> torch.Tensor:
        tok = self.tokenize(text_batch)
        out = self.backbone(**tok)
        return pool_hidden(out.last_hidden_state, tok["attention_mask"], self.pooling)

    def project_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = pooled.to(
            device=self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )
        projected = self.proj(pooled)
        if self.normalize_embeddings:
            projected = F.normalize(projected, p=2, dim=1)
        return projected

    def forward(self, text_batch: list[str]) -> torch.Tensor:
        return self.project_pooled(self.encode_backbone_pooled(text_batch))


class CrossEncoder(StateEncoder):
    def pair_text(self, state_text: str, candidate_text: str) -> str:
        return f"[STATE]\n{state_text}\n[CANDIDATE]\n{candidate_text}"

    def forward(self, state_texts: list[str], candidate_texts: list[str]) -> torch.Tensor:
        if len(state_texts) != len(candidate_texts):
            raise ValueError("state_texts and candidate_texts must have the same length")
        batch = [
            self.pair_text(state_text, candidate_text)
            for state_text, candidate_text in zip(state_texts, candidate_texts)
        ]
        return super().forward(batch)

    def batch_forward(
        self,
        state_texts: list[str],
        candidate_texts: list[list[str]],
    ) -> torch.Tensor:
        if len(state_texts) != len(candidate_texts):
            raise ValueError("batch size mismatch between states and candidates")
        if not state_texts:
            return torch.empty(
                0,
                0,
                self.proj.out_features,
                device=self.proj.weight.device,
            )

        k = len(candidate_texts[0])
        if any(len(row) != k for row in candidate_texts):
            raise ValueError("all candidate rows must have the same width")

        flat_states: list[str] = []
        flat_candidates: list[str] = []
        for state_text, row in zip(state_texts, candidate_texts):
            for candidate_text in row:
                flat_states.append(state_text)
                flat_candidates.append(candidate_text)

        encoded = self.forward(flat_states, flat_candidates)
        return encoded.view(len(state_texts), k, -1)


class SkillTable(nn.Module):
    def __init__(
        self,
        skills: Sequence[Mapping[str, Any] | Any],
        encoder_fn: Callable[[list[str]], torch.Tensor],
        d: int,
        trainable: bool = False,
        initialize_embeddings: bool = True,
        skill_text_fn: Callable[[Mapping[str, Any]], str] | None = None,
        embedding_batch_size: int = 32,
        embedding_progress_callback: Callable[[dict[str, int]], None] | None = None,
        adapter_init: str = "default",
        logit_scale_retr_init: float = 0.0,
        logit_scale_belief_init: float = math.log(0.2),
    ):
        super().__init__()
        self.skills = list(skills)
        self.encoder_fn = encoder_fn
        self.skill_text_fn = skill_text_fn or serialize_skill_text
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.embedding_progress_callback = embedding_progress_callback
        self.W = nn.Linear(d, d, bias=False)
        if adapter_init == "identity":
            with torch.no_grad():
                self.W.weight.copy_(torch.eye(d))
        elif adapter_init != "default":
            raise ValueError(f"unsupported adapter_init: {adapter_init}")
        if initialize_embeddings:
            embeds = self.build_embeddings()
        else:
            embeds = torch.zeros(len(self.skills), d)
        self.E = nn.Parameter(embeds, requires_grad=trainable)
        self.logit_scale_retr = nn.Parameter(torch.tensor([float(logit_scale_retr_init)]))
        self.skill_bias_retr = nn.Parameter(torch.zeros(len(self.skills)))
        self.logit_scale_belief = nn.Parameter(torch.tensor([float(logit_scale_belief_init)]))
        self.skill_bias_belief = nn.Parameter(torch.zeros(len(self.skills)))

    @staticmethod
    def _skill_payload(skill: Mapping[str, Any] | Any) -> Mapping[str, Any]:
        if isinstance(skill, Mapping):
            return skill
        try:
            return vars(skill)
        except TypeError as exc:
            raise TypeError(
                "skill must be a mapping or an object with a __dict__"
            ) from exc

    def build_embeddings(self) -> torch.Tensor:
        texts = [self.skill_text_fn(self._skill_payload(skill)) for skill in self.skills]
        if not texts:
            return torch.empty(0, self.W.in_features)
        batches: list[torch.Tensor] = []
        batch_count = math.ceil(len(texts) / self.embedding_batch_size)
        for batch_index, start in enumerate(range(0, len(texts), self.embedding_batch_size), start=1):
            end = min(start + self.embedding_batch_size, len(texts))
            batches.append(self.encoder_fn(texts[start:end]))
            if self.embedding_progress_callback is not None:
                self.embedding_progress_callback(
                    {
                        "batch_index": batch_index,
                        "batch_count": batch_count,
                        "start": start,
                        "end": end,
                        "total": len(texts),
                    }
                )
        return torch.cat(batches, dim=0)

    def rebuild_embeddings(self) -> torch.Tensor:
        embeds = self.build_embeddings()
        with torch.no_grad():
            self.E.copy_(embeds)
        return self.E

    @staticmethod
    def _skill_id_for_append(skill: Mapping[str, Any] | Any, fallback: str) -> str:
        payload = SkillTable._skill_payload(skill)
        return str(
            payload.get("skill_id")
            or payload.get("canonical_skill_id")
            or payload.get("id")
            or fallback
        )

    def append_skills(
        self,
        new_skills: Sequence[Mapping[str, Any] | Any],
        *,
        new_retrieval_bias: float = 0.0,
        new_belief_bias: float = 0.0,
    ) -> dict[str, Any]:
        """Append previously unseen skills and expand skill-count-bound params.

        This supports dynamic skill inventories after loading a checkpoint. New
        rows are initialized from text embeddings and marked as low-coverage so
        downstream scorers can avoid over-trusting transition/belief heads.
        """
        old_count = len(self.skills)
        seen = {
            self._skill_id_for_append(skill, str(idx))
            for idx, skill in enumerate(self.skills)
        }
        appended: list[Mapping[str, Any]] = []
        appended_ids: list[str] = []
        skipped_duplicates: list[str] = []
        for offset, skill in enumerate(new_skills):
            skill_id = self._skill_id_for_append(skill, str(old_count + offset))
            if skill_id in seen:
                skipped_duplicates.append(skill_id)
                continue
            seen.add(skill_id)
            payload = dict(self._skill_payload(skill))
            payload["skill_id"] = skill_id
            payload.setdefault("canonical_skill_id", skill_id)
            payload.setdefault("is_appended_after_checkpoint", True)
            payload.setdefault("retrieval_seen_count", 0)
            payload.setdefault("transition_seen_count", 0)
            payload.setdefault("act_seen_count", 0)
            appended.append(payload)
            appended_ids.append(skill_id)

        if not appended:
            return {
                "old_count": old_count,
                "new_count": old_count,
                "appended_count": 0,
                "appended_skill_ids": [],
                "skipped_duplicate_skill_ids": skipped_duplicates,
            }

        texts = [self.skill_text_fn(self._skill_payload(skill)) for skill in appended]
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            end = min(start + self.embedding_batch_size, len(texts))
            batches.append(self.encoder_fn(texts[start:end]))
        new_embeddings = torch.cat(batches, dim=0).to(device=self.E.device, dtype=self.E.dtype)

        old_e_requires_grad = self.E.requires_grad
        old_retr_bias_requires_grad = self.skill_bias_retr.requires_grad
        old_belief_bias_requires_grad = self.skill_bias_belief.requires_grad
        self.E = nn.Parameter(
            torch.cat([self.E.detach(), new_embeddings.detach()], dim=0),
            requires_grad=old_e_requires_grad,
        )
        self.skill_bias_retr = nn.Parameter(
            torch.cat(
                [
                    self.skill_bias_retr.detach(),
                    self.skill_bias_retr.detach().new_full((len(appended),), float(new_retrieval_bias)),
                ],
                dim=0,
            ),
            requires_grad=old_retr_bias_requires_grad,
        )
        self.skill_bias_belief = nn.Parameter(
            torch.cat(
                [
                    self.skill_bias_belief.detach(),
                    self.skill_bias_belief.detach().new_full((len(appended),), float(new_belief_bias)),
                ],
                dim=0,
            ),
            requires_grad=old_belief_bias_requires_grad,
        )
        self.skills.extend(appended)
        return {
            "old_count": old_count,
            "new_count": len(self.skills),
            "appended_count": len(appended),
            "appended_skill_ids": appended_ids,
            "skipped_duplicate_skill_ids": skipped_duplicates,
        }

    def _cosine_scores(self, h: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.W(h), p=2, dim=-1)
        e = F.normalize(self.E, p=2, dim=-1)
        return q @ e.t()

    @staticmethod
    def _scale_value(logit_scale: torch.Tensor) -> torch.Tensor:
        return logit_scale.exp().clamp(max=100.0)

    def retrieval_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self._scale_value(self.logit_scale_retr) * self._cosine_scores(h) + self.skill_bias_retr.unsqueeze(0)

    def belief_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self._scale_value(self.logit_scale_belief) * self._cosine_scores(h) + self.skill_bias_belief.unsqueeze(0)

    def set_retrieval_trainable(self, flag: bool) -> "SkillTable":
        """Control trainability of the entire retrieval path.

        Affects: W, E, logit_scale_retr, skill_bias_retr (4 params).
        Use case: freeze retrieval during explicit head-only ablations.
        """
        self.W.weight.requires_grad_(flag)
        self.E.requires_grad_(flag)
        self.logit_scale_retr.requires_grad_(flag)
        self.skill_bias_retr.requires_grad_(flag)
        return self

    def set_belief_scale_trainable(self, flag: bool) -> "SkillTable":
        """Control trainability of belief-specific scale and bias only.

        Affects: logit_scale_belief, skill_bias_belief (2 params).
        Does NOT affect: W, E (shared params remain trainable).
        Use case: Fine-tune belief temperature without retraining embeddings.
        """
        self.logit_scale_belief.requires_grad_(flag)
        self.skill_bias_belief.requires_grad_(flag)
        return self

    def set_trainable(self, flag: bool) -> "SkillTable":
        """Control trainability of all SkillTable parameters (legacy full freeze)."""
        for param in self.parameters():
            param.requires_grad_(flag)
        return self


def serialize_state_text(state) -> str:
    if isinstance(state, str):
        return state
    return serialize_execution_state(state)
