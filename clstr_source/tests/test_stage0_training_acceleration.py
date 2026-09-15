from __future__ import annotations

import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import clstr.model as model_module
import clstr.retrieval_warmup as retrieval_warmup
from clstr.retrieval_warmup import run_skillret_retrieval_warmup


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_tiny_unified_data(data_root):
    _write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": f"skill/{index}",
                "name": f"Skill {index}",
                "description": f"Handles family {index}",
            }
            for index in range(4)
        ],
    )
    _write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "source": "toolret_training",
                "query_id": f"q-{index}",
                "query_text": f"Need skill {index}",
                "positive_skill_id": f"skill/{index}",
            }
            for index in range(4)
        ],
    )


class _CacheableFixedEncoder(nn.Module):
    pooled_batches: list[list[str]] = []

    def __init__(
        self,
        _base_model_name,
        d,
        *,
        freeze_backbone=False,
        projection_init="default",
        normalize_embeddings=False,
        **_kwargs,
    ):
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not freeze_backbone)
        self.proj = nn.Linear(d, d)
        if projection_init == "identity":
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(d))
                self.proj.bias.zero_()
        self.normalize_embeddings = bool(normalize_embeddings)
        self.d = int(d)

    def encode_backbone_pooled(self, texts):
        type(self).pooled_batches.append(list(texts))
        rows = []
        for text in texts:
            matches = re.findall(r"\d+", text)
            index = int(matches[-1]) % self.d if matches else 0
            row = torch.zeros(self.d, device=self.proj.weight.device)
            row[index] = 1.0
            rows.append(row)
        return torch.stack(rows)

    def project_pooled(self, pooled):
        projected = self.proj(
            pooled.to(
                device=self.proj.weight.device,
                dtype=self.proj.weight.dtype,
            )
        )
        return (
            F.normalize(projected, p=2, dim=-1)
            if self.normalize_embeddings
            else projected
        )

    def forward(self, texts):
        return self.project_pooled(self.encode_backbone_pooled(texts))


def _run_kwargs(data_root, output_dir):
    return {
        "data_root": data_root,
        "base_model_name": "unused",
        "output_dir": output_dir,
        "max_steps": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "model_dim": 4,
        "top_k": 4,
        "max_skills": 4,
        "max_queries": 4,
        "learning_rate": 1.0e-2,
        "data_format": "unified_v2",
        "use_cross_encoder": False,
        "freeze_backbone": True,
        "projection_init": "identity",
        "normalize_embeddings": True,
        "skill_table_adapter_init": "identity",
        "skill_table_batch_size": 2,
        "query_text_format": "raw",
        "state_query_prompt_version": "raw_state_v1",
        "retrieval_loss_mode": "multi_positive_nll",
        "sampling_strategy": "batch_stride",
        "shuffle_queries": False,
        "train_skill_embeddings": False,
        "train_skill_bias": False,
        "train_encoder_backbone": False,
        "train_encoder_projection": True,
        "train_skill_adapter": True,
        "train_retrieval_scale": True,
        "route_scorer": "unified_memory",
        "belief_top_k": 2,
    }


def test_stage0_schedule_cache_trains_projection_after_cached_pooling(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_tiny_unified_data(data_root)
    _CacheableFixedEncoder.pooled_batches = []
    monkeypatch.setattr(model_module, "StateEncoder", _CacheableFixedEncoder)
    monkeypatch.setattr(model_module, "CrossEncoder", _CacheableFixedEncoder)

    report = run_skillret_retrieval_warmup(
        **_run_kwargs(data_root, output_dir),
        frozen_backbone_cache_mode="schedule",
        frozen_backbone_cache_batch_size=2,
    )

    assert report["frozen_backbone_cache"]["enabled"] is True
    assert report["frozen_backbone_cache"]["unique_row_count"] == 4
    assert report["frozen_backbone_cache"]["lookup_count"] == 4
    step0 = torch.load(
        output_dir / "checkpoints" / "clstr_unified_retrieval_v2-step0.pt",
        map_location="cpu",
    )
    final = torch.load(report["checkpoint"], map_location="cpu")
    assert not torch.equal(
        step0["model_state_dict"]["encoder.proj.weight"],
        final["model_state_dict"]["encoder.proj.weight"],
    )


def test_stage0_verified_resume_skips_skill_table_rebuild(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_tiny_unified_data(data_root)
    monkeypatch.setattr(model_module, "StateEncoder", _CacheableFixedEncoder)
    monkeypatch.setattr(model_module, "CrossEncoder", _CacheableFixedEncoder)
    first_kwargs = _run_kwargs(data_root, output_dir)
    first_kwargs["max_steps"] = 1
    first = run_skillret_retrieval_warmup(**first_kwargs)

    def forbidden_rebuild(_self):
        raise AssertionError("verified resume must not rebuild the skill table")

    monkeypatch.setattr(
        retrieval_warmup.CLSTRModel,
        "rebuild_skill_table",
        forbidden_rebuild,
    )
    second = run_skillret_retrieval_warmup(
        **_run_kwargs(data_root, output_dir),
        resume_checkpoint_path=first["checkpoint"],
        resume_skill_table_mode="verified_checkpoint",
    )

    assert second["skill_table_resume"]["rebuild_skipped"] is True
    assert second["resume"]["checkpoint_step"] == 1
