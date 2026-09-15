from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _read_jsonl_count(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            count += 1
    return count


def _skill_pool_quality(path: str | Path) -> dict[str, Any]:
    count = 0
    missing_source_record_count = 0
    missing_samples: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            count += 1
            provenance = row.get("provenance") if isinstance(row, dict) else {}
            if isinstance(provenance, str) and provenance.strip():
                try:
                    provenance = json.loads(provenance)
                except json.JSONDecodeError:
                    provenance = {}
            if not isinstance(provenance, dict):
                provenance = {}
            if provenance.get("missing_source_record"):
                missing_source_record_count += 1
                if len(missing_samples) < 20:
                    missing_samples.append(str(row.get("skill_id") or f"line:{line_no}"))
    return {
        "skill_count": count,
        "missing_source_record_count": missing_source_record_count,
        "missing_source_record_samples": missing_samples,
    }


def _checkpoint_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint model_state_dict must be a dict, got {type(state).__name__}")
    return state


def _validate_unified_train_safety(payload: dict[str, Any]) -> dict[str, Any]:
    train_safety = payload.get("train_safety")
    if not isinstance(train_safety, dict):
        raise ValueError("routing checkpoint missing train-safe retrieval metadata")
    if train_safety.get("unified_v2_train_safe_retrieval") is not True:
        raise ValueError("routing checkpoint missing train-safe retrieval metadata")
    excluded = {str(item) for item in train_safety.get("excluded_retrieval_splits") or []}
    if "public_train_or_eval_unlabeled" not in excluded:
        raise ValueError("routing checkpoint missing train-safe retrieval metadata")
    return train_safety


def validate_stage2_routing_checkpoint(
    checkpoint_path: str | Path,
    skills_path: str | Path,
    model_dim: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    skills_path = Path(skills_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"routing checkpoint not found: {checkpoint_path}")
    if not skills_path.exists():
        raise FileNotFoundError(f"skills file not found: {skills_path}")

    skill_pool_quality = _skill_pool_quality(skills_path)
    skill_count = int(skill_pool_quality["skill_count"])
    if int(skill_pool_quality["missing_source_record_count"]) > 0:
        raise ValueError(
            "skill_pool has placeholder records: "
            f"{skill_pool_quality['missing_source_record_count']} missing source records, "
            f"samples={skill_pool_quality['missing_source_record_samples']}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = _checkpoint_state(payload)
    if "skill_table.E" not in state:
        raise ValueError("routing checkpoint missing skill_table.E")
    embeddings = state["skill_table.E"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"skill_table.E must be rank-2, got shape {list(embeddings.shape)}")
    if int(embeddings.shape[0]) != int(skill_count):
        raise ValueError(
            "skill count mismatch: "
            f"checkpoint skill_table.E has {int(embeddings.shape[0])} rows, "
            f"skills file has {int(skill_count)} rows"
        )
    if model_dim is not None and int(embeddings.shape[1]) != int(model_dim):
        raise ValueError(
            "model dimension mismatch: "
            f"checkpoint skill_table.E has dim {int(embeddings.shape[1])}, "
            f"expected {int(model_dim)}"
        )

    bias = state.get("skill_table.skill_bias_retr")
    if bias is not None:
        bias_tensor = bias if isinstance(bias, torch.Tensor) else torch.as_tensor(bias)
        if int(bias_tensor.numel()) != int(skill_count):
            raise ValueError(
                "retrieval bias mismatch: "
                f"skill_table.skill_bias_retr has {int(bias_tensor.numel())} values, "
                f"skills file has {int(skill_count)} rows"
            )

    checkpoint_stage = payload.get("stage") if isinstance(payload, dict) else None
    train_safety: dict[str, Any] = {}
    if checkpoint_stage == "clstr_unified_retrieval_v2":
        train_safety = _validate_unified_train_safety(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("train_safety"), dict):
        train_safety = payload["train_safety"]

    report = {
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
        "skills_path": str(skills_path),
        "checkpoint_stage": checkpoint_stage,
        "train_safety": train_safety,
        "skill_count": int(skill_count),
        "skill_pool_quality": skill_pool_quality,
        "embedding_shape": [int(dim) for dim in embeddings.shape],
        "model_dim": int(model_dim) if model_dim is not None else int(embeddings.shape[1]),
        "has_retrieval_adapter": "skill_table.W.weight" in state,
        "has_retrieval_scale_and_bias": (
            "skill_table.logit_scale_retr" in state
            and "skill_table.skill_bias_retr" in state
        ),
        "checkpoint_key_count": len(state),
    }
    return report
