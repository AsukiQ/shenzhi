from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.encoders import pool_hidden
from clstr.model import CLSTRConfig, CLSTRModel

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None


@dataclass(frozen=True)
class QwenExternalEncoderConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    output_dim: int | None = None
    pooling: str = "last_token"
    torch_dtype: str = "bfloat16"
    cache_dir: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = True
    max_length: int | None = 4096
    freeze_backbone: bool = True
    projection_trainable: bool = False
    normalize_embeddings: bool = True


def _resolve_dtype(value: str | torch.dtype | None) -> torch.dtype | str | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if value == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported torch_dtype: {value}")
    return mapping[value]


def qwen_external_metadata(
    model_name_or_path: str,
    quantization_mode: str = "none",
    output_dim: int | None = None,
) -> dict[str, Any]:
    return {
        "qwen_model": "Qwen/Qwen3-8B",
        "qwen_model_name_or_path": str(model_name_or_path),
        "qwen_external_encoder": True,
        "qwen_frozen": True,
        "qwen_quantization_mode": str(quantization_mode),
        "qwen_direct_generator": False,
        "native_clstr_heads_used": True,
        "external_encoder_type": "qwen3_8b_frozen",
        "encoder_output_dim": output_dim,
    }


class FrozenQwenExternalEncoder(nn.Module):
    def __init__(self, config: QwenExternalEncoderConfig):
        super().__init__()
        if AutoTokenizer is None or AutoModel is None:
            raise ModuleNotFoundError("transformers is required for FrozenQwenExternalEncoder")
        self.config = config
        loader_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": config.cache_dir,
            "local_files_only": config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **loader_kwargs)
        self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = dict(loader_kwargs)
        dtype = _resolve_dtype(config.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.backbone = AutoModel.from_pretrained(config.model_name_or_path, **model_kwargs)
        hidden_size = int(self.backbone.config.hidden_size)
        output_dim = int(config.output_dim or hidden_size)
        self.proj = nn.Linear(hidden_size, output_dim)
        if output_dim == hidden_size:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(output_dim))
                self.proj.bias.zero_()
        for param in self.backbone.parameters():
            param.requires_grad_(not config.freeze_backbone)
        for param in self.proj.parameters():
            param.requires_grad_(bool(config.projection_trainable))

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        max_length = self.config.max_length or getattr(self.backbone.config, "max_position_embeddings", None)
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.backbone.device) if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()}

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokenized = self.tokenize(texts)
        if self.config.freeze_backbone:
            with torch.no_grad():
                output = self.backbone(**tokenized)
        else:
            output = self.backbone(**tokenized)
        pooled = pool_hidden(output.last_hidden_state, tokenized["attention_mask"], self.config.pooling)
        pooled = pooled.to(self.proj.weight.dtype)
        projected = self.proj(pooled)
        if self.config.normalize_embeddings:
            projected = F.normalize(projected.float(), p=2, dim=-1).to(projected.dtype)
        return projected

    def encode_observations(self, texts: list[str]) -> torch.Tensor:
        return self.forward(texts)


def _hidden_size_for_model(model_name_or_path: str, cache_dir: str | None = None, local_files_only: bool = False) -> int:
    if AutoConfig is None:
        raise ModuleNotFoundError("transformers is required to inspect Qwen config")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return int(config.hidden_size)


def build_qwen_external_clstr_config(
    model_name_or_path: str = "Qwen/Qwen3-8B",
    model_dim: int | None = None,
    max_length: int | None = 4096,
    torch_dtype: str = "bfloat16",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> CLSTRConfig:
    hidden_size = _hidden_size_for_model(model_name_or_path, cache_dir=cache_dir, local_files_only=local_files_only)
    d = int(model_dim or hidden_size)
    return CLSTRConfig(
        base_model_name=str(model_name_or_path),
        d=d,
        d_a=d,
        top_k=50,
        encoder_pooling="last_token",
        cross_encoder_pooling="last_token",
        tokenizer_padding_side="left",
        torch_dtype=torch_dtype,
        freeze_backbone=True,
        trust_remote_code=True,
        max_length=max_length,
        projection_init="identity" if d == hidden_size else "default",
        normalize_embeddings=True,
        defer_skill_table_init=True,
        skill_text_format="skillret_official",
        skill_table_batch_size=1,
        skill_table_adapter_init="identity" if d == hidden_size else "default",
        use_cross_encoder=False,
        skill_head_context="belief_residual",
        candidate_widen_factor=4,
        candidate_sample_temperature=1.0,
        hf_cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def build_qwen_external_clstr_model(
    skills: list[dict[str, Any]],
    model_name_or_path: str = "Qwen/Qwen3-8B",
    model_dim: int | None = None,
    max_length: int | None = 4096,
    torch_dtype: str = "bfloat16",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any]]:
    config = build_qwen_external_clstr_config(
        model_name_or_path=model_name_or_path,
        model_dim=model_dim,
        max_length=max_length,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = CLSTRModel(config, skills)
    metadata = qwen_external_metadata(
        model_name_or_path=model_name_or_path,
        quantization_mode="none",
        output_dim=config.d,
    )
    setattr(model, "qwen_external_metadata", metadata)
    report = {
        "manifest": None,
        "base_clstr_checkpoint": None,
        "base_clstr_loaded": False,
        "qdoc_adapter_used": False,
        "downstream_init_for": "clstr_qwen3_8b_external_encoder",
        **metadata,
    }
    return model, vars(config), report


def write_qwen_init_manifest(
    output_dir: str | Path,
    model_name_or_path: str,
    cache_dir: str | None,
    quantization_mode: str = "none",
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "model_name_or_path": str(model_name_or_path),
        "cache_dir": str(cache_dir) if cache_dir else None,
        **qwen_external_metadata(model_name_or_path, quantization_mode=quantization_mode),
    }
    if error:
        report["error"] = str(error)
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
