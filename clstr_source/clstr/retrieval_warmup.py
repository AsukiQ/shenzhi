from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.history_channel import strip_history_sections
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.skillret import load_skillret_training_rows
from clstr.stage0_frozen_backbone_cache import (
    Stage0FrozenBackboneCache,
    assign_stage0_query_indices,
    build_stage0_frozen_backbone_cache,
    plan_stage0_scheduled_rows,
)
from clstr.stage0_skill_pool_identity import (
    ordered_skill_pool_identity,
    validate_verified_resume_skill_table,
)
from clstr.state_query_prompt import (
    RAW_STATE_V1,
    SR_TASK_DESCRIPTION_V1,
    resolve_state_query_prompt_contract,
)
from clstr.stage_checkpoint_init import load_compatible_state_dict
from clstr.training_monitor import TrainingMonitor


UNSAFE_RETRIEVAL_SPLITS = {
    "dev",
    "eval",
    "evaluation",
    "public_train_or_eval_unlabeled",
    "test",
    "valid",
    "validation",
}
ROUTER_STATE_CONTRACT = "history_free_current_state_v1"

STAGE0_BIENCODER_DEFAULTS: dict[str, Any] = {
    "encoder_pooling": "last_token",
    "cross_encoder_pooling": "last_token",
    "tokenizer_padding_side": "left",
    "projection_init": "identity",
    "normalize_embeddings": True,
    "skill_text_format": "skillret_official",
    "skill_table_adapter_init": "identity",
    "use_cross_encoder": False,
    "query_text_format": "skillrouter",
    "state_query_prompt_version": None,
    "state_query_max_chars": None,
    "state_query_truncation": None,
    "data_format": "unified_v2",
    "expand_alias_positives": True,
    "shuffle_queries": True,
    "sampling_strategy": "handoff_balanced",
    "retrieval_loss_mode": "multi_positive_nll",
    "train_skill_embeddings": False,
    "train_skill_bias": False,
    "train_encoder_backbone": False,
    "train_encoder_projection": False,
    "train_skill_adapter": True,
    "train_retrieval_scale": True,
    "explicit_negative_loss_weight": 0.0,
    "explicit_negative_margin": 0.1,
    "mined_hard_negative_loss_weight": 0.0,
    "mined_hard_negative_margin": 0.1,
    "mined_hard_negative_top_k": 32,
    "route_scorer": "legacy_biencoder",
    "belief_top_k": 64,
    "frozen_backbone_cache_mode": "off",
    "frozen_backbone_cache_batch_size": 128,
    "resume_skill_table_mode": "rebuild",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _stage0_raw_query_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [strip_history_sections(str(row["query"])) for row in rows]


def _resolve_stage0_state_query_contract(
    *,
    query_text_format: str,
    state_query_prompt_version: str | None,
    state_query_max_chars: int | None,
    state_query_truncation: str | None,
) -> dict[str, Any]:
    version = state_query_prompt_version
    if version is None:
        if query_text_format == "skillrouter":
            version = SR_TASK_DESCRIPTION_V1
        elif query_text_format == "raw":
            version = RAW_STATE_V1
        else:
            raise ValueError(f"unsupported query_text_format: {query_text_format}")
    return resolve_state_query_prompt_contract(
        prompt_version=version,
        max_chars=state_query_max_chars,
        truncation=state_query_truncation,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {line_no}: invalid JSON: {exc}") from exc
    return rows


def _flatten_skill_id_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        output: list[str] = []
        for key in ("skill_id", "canonical_skill_id", "id"):
            if value.get(key):
                output.append(str(value[key]))
        return output
    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_skill_id_values(item))
        return output
    text = str(value).strip()
    return [text] if text else []


def _warmup_checkpoint_state_dict(
    model: Any,
    *,
    include_encoder_backbone: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = dict(model.state_dict())
    excluded_prefixes = [
        "cross_encoder.backbone.",
        "skill_table.encoder_fn.",
    ]
    if not include_encoder_backbone:
        excluded_prefixes.append("encoder.backbone.")
    filtered = {
        key: value
        for key, value in state.items()
        if not any(str(key).startswith(prefix) for prefix in excluded_prefixes)
    }
    return filtered, {
        "checkpoint_includes_encoder_backbone": bool(include_encoder_backbone),
        "checkpoint_excludes_frozen_backbone": not bool(include_encoder_backbone),
        "checkpoint_state_key_count": len(filtered),
        "excluded_state_key_prefixes": excluded_prefixes,
    }


def _save_named_stage0_checkpoint(
    *,
    checkpoint_dir: str | Path,
    stage_name: str,
    step: int,
    payload: dict[str, Any],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{stage_name}-step{int(step)}.pt"
    torch.save(payload, path)
    return path


def _skill_ids_from_skill_pool(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    skill_ids: list[str] = []
    for row in _read_jsonl(Path(path)):
        skill_id = str(row.get("skill_id") or row.get("id") or "").strip()
        if skill_id:
            skill_ids.append(skill_id)
    return skill_ids


def _transplant_skill_table_state_by_skill_id(
    model: Any,
    state_dict: dict[str, Any],
    *,
    init_checkpoint_skills_path: str | Path | None,
    current_skills_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if init_checkpoint_skills_path is None or current_skills_path is None:
        return state_dict, {
            "enabled": False,
            "reason": "missing_skill_pool_paths",
            "matched_skill_count": 0,
            "transplanted_keys": [],
        }
    old_skill_ids = _skill_ids_from_skill_pool(init_checkpoint_skills_path)
    current_skill_ids = _skill_ids_from_skill_pool(current_skills_path)
    if not old_skill_ids or not current_skill_ids:
        return state_dict, {
            "enabled": False,
            "reason": "empty_skill_pool",
            "old_skill_count": len(old_skill_ids),
            "current_skill_count": len(current_skill_ids),
            "matched_skill_count": 0,
            "transplanted_keys": [],
        }

    old_index = {skill_id: idx for idx, skill_id in enumerate(old_skill_ids)}
    matched_pairs = [(new_idx, old_index[skill_id]) for new_idx, skill_id in enumerate(current_skill_ids) if skill_id in old_index]
    current_state = model.state_dict() if hasattr(model, "state_dict") else {}
    updated = dict(state_dict)
    transplanted_keys: list[str] = []
    skipped_keys: dict[str, str] = {}
    for key in ("skill_table.E", "skill_table.skill_bias_retr", "skill_table.skill_bias_belief"):
        if key not in state_dict or key not in current_state:
            continue
        source = state_dict[key] if isinstance(state_dict[key], torch.Tensor) else torch.as_tensor(state_dict[key])
        target = current_state[key].detach().clone()
        if source.ndim != target.ndim:
            skipped_keys[key] = "rank_mismatch"
            continue
        if source.shape[0] != len(old_skill_ids):
            skipped_keys[key] = "checkpoint_skill_count_mismatch"
            continue
        if target.shape[0] != len(current_skill_ids):
            skipped_keys[key] = "current_skill_count_mismatch"
            continue
        if tuple(source.shape[1:]) != tuple(target.shape[1:]):
            skipped_keys[key] = "tail_shape_mismatch"
            continue
        for new_idx, old_idx in matched_pairs:
            target[new_idx] = source[old_idx].to(device=target.device, dtype=target.dtype)
        updated[key] = target
        transplanted_keys.append(key)
    return updated, {
        "enabled": True,
        "init_checkpoint_skills_path": str(init_checkpoint_skills_path),
        "current_skills_path": str(current_skills_path),
        "old_skill_count": len(old_skill_ids),
        "current_skill_count": len(current_skill_ids),
        "matched_skill_count": len(matched_pairs),
        "new_only_skill_count": len(current_skill_ids) - len(matched_pairs),
        "transplanted_keys": transplanted_keys,
        "skipped_keys": skipped_keys,
    }


def _load_warmup_init_checkpoint(
    model: Any,
    checkpoint_path: str | Path | None,
    *,
    init_checkpoint_skills_path: str | Path | None = None,
    current_skills_path: str | Path | None = None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"loaded": False, "path": None}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        state_dict = payload["model_state_dict"]
        checkpoint_step = payload.get("step")
    elif isinstance(payload, dict):
        state_dict = payload
        checkpoint_step = None
    else:
        raise ValueError(f"unsupported init checkpoint payload type: {type(payload).__name__}")
    state_dict, skill_table_transplant_report = _transplant_skill_table_state_by_skill_id(
        model,
        state_dict,
        init_checkpoint_skills_path=init_checkpoint_skills_path,
        current_skills_path=current_skills_path,
    )
    load_report = load_compatible_state_dict(
        model,
        state_dict,
        partial_load_mode="warmup_init_checkpoint_compatible_state",
    )
    return {
        "loaded": True,
        "path": str(path),
        "checkpoint_step": checkpoint_step,
        "loaded_state_key_count": int(load_report["loaded_key_count"]),
        "loaded_state_keys": list(load_report["loaded_keys"]),
        "missing_state_keys": list(load_report["missing_keys"]),
        "unexpected_state_keys": list(load_report["unexpected_keys"]),
        "skipped_state_keys": list(load_report["skipped_keys"]),
        "shape_mismatched": dict(load_report["shape_mismatched"]),
        "partial_load_mode": load_report["partial_load_mode"],
        "skill_table_id_aligned_transplant": skill_table_transplant_report,
    }


def _load_stage0_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if path.is_dir() and (path / "checkpoints" / "latest.pt").exists():
        path = path / "checkpoints" / "latest.pt"
    elif path.is_dir() and (path / "latest.pt").exists():
        path = path / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"stage0 resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported stage0 resume checkpoint payload type: {type(payload).__name__}")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"stage0 resume checkpoint lacks model_state_dict: {path}")
    selected_skills_path = (
        path.parent.parent / "selected_skills.jsonl"
        if path.parent.name == "checkpoints"
        else path.parent / "selected_skills.jsonl"
    )
    selected_skill_rows = (
        _read_jsonl(selected_skills_path)
        if selected_skills_path.is_file()
        else None
    )
    return {
        **payload,
        "checkpoint_path": str(path),
        "step": int(payload.get("step") or 0),
        "model_state_dict": state_dict,
        "optimizer_state_dict": payload.get("optimizer_state_dict"),
        "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        "resume_selected_skills_path": (
            str(selected_skills_path) if selected_skills_path.is_file() else None
        ),
        "resume_selected_skill_rows": selected_skill_rows,
    }


def _stage0_resume_expected_config(model: Any) -> dict[str, Any]:
    config = model.config
    return {
        "base_model_name": config.base_model_name,
        "d": int(config.d),
        "encoder_pooling": config.encoder_pooling,
        "tokenizer_padding_side": config.tokenizer_padding_side,
        "torch_dtype": config.torch_dtype,
        "freeze_backbone": bool(config.freeze_backbone),
        "max_length": config.max_length,
        "projection_init": config.projection_init,
        "normalize_embeddings": bool(config.normalize_embeddings),
        "skill_text_format": config.skill_text_format,
    }


def _load_monitor_history(monitor: TrainingMonitor) -> None:
    if not monitor.metrics_path.exists():
        return
    history: list[dict[str, Any]] = []
    with monitor.metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    monitor.history = history
    monitor._write_loss_curve()
    monitor._write_diagnostic_curves()


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _snapshot_trainable_parameters(
    model: Any,
    parameter_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and (parameter_names is None or name in parameter_names)
    }


def _parameter_anchor_loss(
    model: Any,
    anchors: dict[str, torch.Tensor],
    weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    device = None
    for param in model.parameters():
        device = param.device
        break
    if device is None:
        device = torch.device("cpu")
    zero = torch.zeros((), device=device)
    if not anchors or float(weight) <= 0.0:
        return zero, {
            "anchor_parameter_count": 0,
            "anchor_unweighted_loss": 0.0,
        }

    losses: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        anchor = anchors.get(name)
        if anchor is None:
            continue
        losses.append((param.float() - anchor.to(device=param.device, dtype=param.dtype).float()).pow(2).mean())
    if not losses:
        return zero, {
            "anchor_parameter_count": 0,
            "anchor_unweighted_loss": 0.0,
        }
    unweighted = torch.stack(losses).mean()
    return unweighted * float(weight), {
        "anchor_parameter_count": len(losses),
        "anchor_unweighted_loss": float(unweighted.detach().cpu().item()),
    }


def _filter_queries_to_skill_pool(
    queries: list[dict[str, Any]],
    positives_by_query: dict[str, list[str]],
    skill_id_to_idx: dict[str, int],
    max_queries: int | None,
    alias_positive_ids_by_skill_id: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    usable = []
    alias_positive_ids_by_skill_id = alias_positive_ids_by_skill_id or {}
    for query in queries:
        qid = str(query["query_id"])
        raw_positive_ids = [str(skill_id) for skill_id in positives_by_query.get(qid, query.get("positive_skill_ids", [])) if skill_id]
        raw_negative_ids = _flatten_skill_id_values(query.get("negative_skill_ids"))
        resolved_positive_ids: list[str] = []
        alias_resolved_positive_ids: list[str] = []
        positive_indices: list[int] = []
        positive_index_set: set[int] = set()
        for raw_skill_id in raw_positive_ids:
            for candidate_skill_id in [raw_skill_id, *alias_positive_ids_by_skill_id.get(raw_skill_id, [])]:
                idx = skill_id_to_idx.get(candidate_skill_id)
                if idx is None or idx in positive_index_set:
                    continue
                positive_index_set.add(int(idx))
                positive_indices.append(int(idx))
                resolved_positive_ids.append(candidate_skill_id)
                if candidate_skill_id != raw_skill_id:
                    alias_resolved_positive_ids.append(candidate_skill_id)
        if positive_indices:
            negative_indices: list[int] = []
            negative_index_set: set[int] = set()
            resolved_negative_ids: list[str] = []
            for raw_skill_id in raw_negative_ids:
                idx = skill_id_to_idx.get(raw_skill_id)
                if idx is None or idx in positive_index_set or idx in negative_index_set:
                    continue
                negative_index_set.add(int(idx))
                negative_indices.append(int(idx))
                resolved_negative_ids.append(raw_skill_id)
            copied = dict(query)
            copied["positive_indices"] = positive_indices
            copied["raw_positive_skill_ids"] = raw_positive_ids
            copied["resolved_positive_skill_ids"] = resolved_positive_ids
            copied["alias_resolved_positive_skill_ids"] = alias_resolved_positive_ids
            copied["raw_negative_skill_ids"] = raw_negative_ids
            copied["negative_indices"] = negative_indices
            copied["resolved_negative_skill_ids"] = resolved_negative_ids
            copied["stage0_alias_positive_resolved"] = bool(alias_resolved_positive_ids)
            copied["stage0_multi_positive_expanded"] = len(resolved_positive_ids) > len(raw_positive_ids)
            copied["stage0_alias_positive_expanded"] = bool(alias_resolved_positive_ids) or copied["stage0_multi_positive_expanded"]
            usable.append(copied)
        if max_queries is not None and len(usable) >= max_queries:
            break
    return usable


def _skill_pool_alias_positive_ids_by_skill_id(skill_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    canonical_to_skill_ids: dict[str, set[str]] = defaultdict(set)
    skill_id_to_canonical: dict[str, str] = {}
    for row in skill_rows:
        skill_id = str(row.get("skill_id") or row.get("id") or "")
        if not skill_id:
            continue
        canonical_id = str(row.get("canonical_skill_id") or skill_id)
        canonical_to_skill_ids[canonical_id].add(skill_id)
        skill_id_to_canonical[skill_id] = canonical_id
        aliases = _flatten_skill_id_values(row.get("alias_skill_ids"))
        for alias in aliases:
            alias = str(alias)
            if not alias:
                continue
            canonical_to_skill_ids[canonical_id].add(alias)
            skill_id_to_canonical[alias] = canonical_id
    output: dict[str, list[str]] = {}
    for skill_id, canonical_id in skill_id_to_canonical.items():
        equivalents = canonical_to_skill_ids.get(canonical_id, {skill_id})
        output[skill_id] = sorted(alias for alias in equivalents if alias and alias != skill_id)
    return output


def _load_split_query_ids(split_path: str | Path | None, split_name: str) -> set[str] | None:
    if split_path is None:
        return None
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return {str(item) for item in payload["splits"][split_name]["query_ids"]}


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _retrieval_split_name(row: dict[str, Any]) -> str:
    provenance = _provenance(row)
    return str(row.get("split") or provenance.get("split") or "").strip().lower()


def _is_safe_retrieval_train_row(row: dict[str, Any]) -> bool:
    return _retrieval_split_name(row) not in UNSAFE_RETRIEVAL_SPLITS


def load_unified_v2_retrieval_rows(
    data_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    """Load CLSTR unified v2 retrieval stream.

    Expected files:
      - skill_pool.jsonl: canonical skill records
      - retrieval.jsonl: query -> positive_skill_id rows already leakage-filtered
    """
    root = Path(data_root)
    skill_rows = _read_jsonl(root / "skill_pool.jsonl")
    if not skill_rows:
        raise ValueError(f"no unified v2 skill_pool.jsonl rows found under {root}")
    query_by_id: dict[str, dict[str, Any]] = {}
    query_key_by_qid_source: dict[tuple[str, str], str] = {}
    source_count_by_qid: dict[str, int] = defaultdict(int)
    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)
    for row in _read_jsonl(root / "retrieval.jsonl"):
        if not _is_safe_retrieval_train_row(row):
            continue
        qid = str(row.get("query_id") or "")
        sid = str(row.get("positive_skill_id") or row.get("skill_id") or "")
        if not qid or not sid:
            continue
        source = str(row.get("source") or "")
        source_key = (qid, source)
        if source_key not in query_key_by_qid_source:
            if source_count_by_qid[qid] == 0:
                query_key = qid
            else:
                safe_source = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source) or "unknown"
                query_key = f"{qid}::source::{safe_source}"
                while query_key in query_by_id:
                    query_key = f"{qid}::source::{safe_source}::{source_count_by_qid[qid]}"
            query_key_by_qid_source[source_key] = query_key
            source_count_by_qid[qid] += 1
        query_key = query_key_by_qid_source[source_key]
        if query_key not in query_by_id:
            metadata = {
                "source": row.get("source"),
                "provenance": row.get("provenance"),
            }
            if query_key != qid:
                metadata["original_query_id"] = qid
            query_by_id[query_key] = {
                "query_id": query_key,
                "id": query_key,
                "query": str(row.get("query_text") or row.get("query") or ""),
                "positive_skill_ids": [],
                "negative_skill_ids": [],
                "source_dataset": "clstr_unified_pretrain_v2",
                "metadata": metadata,
            }
        positives[query_key].append(sid)
        query_by_id[query_key]["positive_skill_ids"].append(sid)
        for negative_skill_id in _flatten_skill_id_values(row.get("negative_skill_ids")):
            if negative_skill_id and negative_skill_id not in negatives[query_key]:
                negatives[query_key].append(negative_skill_id)
                query_by_id[query_key]["negative_skill_ids"].append(negative_skill_id)
    query_rows = list(query_by_id.values())
    if not query_rows:
        raise ValueError(f"no usable unified v2 retrieval.jsonl rows found under {root}")
    return skill_rows, query_rows, dict(positives)


def load_retrieval_warmup_rows(
    data_root: str | Path,
    data_format: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    if data_format == "skillret":
        return load_skillret_training_rows(data_root, split="train")
    if data_format == "unified_v2":
        return load_unified_v2_retrieval_rows(data_root)
    raise ValueError(f"unsupported retrieval warmup data_format: {data_format}")


def _recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(-1))
    top = torch.topk(logits, k=k, dim=-1).indices
    return float((top == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())


def _multi_positive_nll(logits: torch.Tensor, positive_indices: list[list[int]]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row_idx, positives in enumerate(positive_indices):
        valid = sorted({int(idx) for idx in positives if 0 <= int(idx) < logits.size(1)})
        if not valid:
            continue
        row = logits[row_idx]
        pos = torch.tensor(valid, device=logits.device, dtype=torch.long)
        losses.append(torch.logsumexp(row, dim=0) - torch.logsumexp(row.index_select(0, pos), dim=0))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _explicit_negative_margin_loss(
    logits: torch.Tensor,
    positive_indices: list[list[int]],
    negative_indices: list[list[int]],
    margin: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    margin = max(0.0, float(margin))
    losses: list[torch.Tensor] = []
    row_count = 0
    pair_count = 0
    for row_idx, positives in enumerate(positive_indices):
        valid_pos = sorted({int(idx) for idx in positives if 0 <= int(idx) < logits.size(1)})
        row_negatives = negative_indices[row_idx] if row_idx < len(negative_indices) else []
        valid_neg = sorted(
            {
                int(idx)
                for idx in row_negatives
                if 0 <= int(idx) < logits.size(1) and int(idx) not in valid_pos
            }
        )
        if not valid_pos or not valid_neg:
            continue
        row = logits[row_idx]
        pos = torch.tensor(valid_pos, device=logits.device, dtype=torch.long)
        neg = torch.tensor(valid_neg, device=logits.device, dtype=torch.long)
        positive_score = torch.logsumexp(row.index_select(0, pos), dim=0)
        losses.append(F.relu(margin + row.index_select(0, neg) - positive_score).mean())
        row_count += 1
        pair_count += len(valid_neg)
    if not losses:
        return logits.sum() * 0.0, {
            "explicit_negative_rows": 0,
            "explicit_negative_pair_count": 0,
        }
    return torch.stack(losses).mean(), {
        "explicit_negative_rows": row_count,
        "explicit_negative_pair_count": pair_count,
    }


def _mined_hard_negative_margin_loss(
    logits: torch.Tensor,
    positive_indices: list[list[int]],
    top_k: int,
    margin: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    top_k = max(0, int(top_k))
    margin = max(0.0, float(margin))
    losses: list[torch.Tensor] = []
    row_count = 0
    pair_count = 0
    for row_idx, positives in enumerate(positive_indices):
        valid_pos = sorted({int(idx) for idx in positives if 0 <= int(idx) < logits.size(1)})
        if not valid_pos or top_k <= 0:
            continue
        row = logits[row_idx]
        candidate_mask = torch.ones(logits.size(1), device=logits.device, dtype=torch.bool)
        candidate_mask[torch.tensor(valid_pos, device=logits.device, dtype=torch.long)] = False
        candidate_count = int(candidate_mask.sum().detach().cpu().item())
        if candidate_count <= 0:
            continue
        k = min(top_k, candidate_count)
        masked_row = row.masked_fill(~candidate_mask, torch.finfo(row.dtype).min)
        mined_scores = torch.topk(masked_row, k=k, dim=0).values
        pos = torch.tensor(valid_pos, device=logits.device, dtype=torch.long)
        positive_score = torch.logsumexp(row.index_select(0, pos), dim=0)
        losses.append(F.relu(margin + mined_scores - positive_score).mean())
        row_count += 1
        pair_count += k
    if not losses:
        return logits.sum() * 0.0, {
            "mined_hard_negative_rows": 0,
            "mined_hard_negative_pair_count": 0,
        }
    return torch.stack(losses).mean(), {
        "mined_hard_negative_rows": row_count,
        "mined_hard_negative_pair_count": pair_count,
    }


def _recall_at_k_multi(logits: torch.Tensor, positive_indices: list[list[int]], k: int) -> float:
    if not positive_indices:
        return 0.0
    k = min(k, logits.size(-1))
    top = torch.topk(logits, k=k, dim=-1).indices.detach().cpu().tolist()
    hits = 0
    usable = 0
    for row_top, positives in zip(top, positive_indices):
        positive_set = {int(idx) for idx in positives}
        if not positive_set:
            continue
        usable += 1
        hits += int(any(int(idx) in positive_set for idx in row_top))
    return float(hits / max(usable, 1))


def stage0_biencoder_scores(query_embs: torch.Tensor, skill_embs: torch.Tensor) -> torch.Tensor:
    queries = F.normalize(query_embs.float(), p=2, dim=-1)
    skills = F.normalize(skill_embs.float(), p=2, dim=-1)
    return queries @ skills.t()


def _unified_static_warmup_logits(
    model: Any,
    query_texts: list[str],
    *,
    belief_top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_t = model.encode_states(query_texts)
    m_0 = model.initial_belief(h_t, top_k=belief_top_k)
    logits = model.unified_route_logits(h_t, m_0)
    return h_t, m_0, logits


def compute_unified_static_warmup_loss(
    model: Any,
    query_texts: list[str],
    positive_indices: list[list[int]],
    *,
    retrieval_loss_mode: str = "multi_positive_nll",
    belief_top_k: int | None = None,
    recall_ks: tuple[int, ...] | list[int] = (1, 5, 20),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the t=0 unified-memory retrieval objective.

    This is the Stage0 replacement objective for the unified route: it trains
    the same `z_t dot E_skill` scorer used by sequential stages, with `m_t`
    initialized from the current state only.
    """
    if retrieval_loss_mode not in {"single_positive_ce", "multi_positive_nll"}:
        raise ValueError(f"unsupported retrieval_loss_mode: {retrieval_loss_mode}")
    if not query_texts:
        try:
            param = next(model.parameters())
            zero = param.sum() * 0.0
        except StopIteration:
            zero = torch.zeros((), requires_grad=True)
        return zero, {
            "route_scorer": "unified_memory",
            "unified_static_query_count": 0,
            "unified_static_positive_count": 0,
        }
    _h_t, _m_0, logits = _unified_static_warmup_logits(model, query_texts, belief_top_k=belief_top_k)
    if retrieval_loss_mode == "multi_positive_nll":
        loss = _multi_positive_nll(logits, positive_indices)
    else:
        labels = torch.tensor(
            [int(row[0]) for row in positive_indices],
            dtype=torch.long,
            device=logits.device,
        )
        loss = F.cross_entropy(logits, labels)
    metrics: dict[str, Any] = {
        "route_scorer": "unified_memory",
        "unified_static_query_count": int(len(query_texts)),
        "unified_static_positive_count": int(sum(len(row) for row in positive_indices)),
        "unified_static_loss": float(loss.detach().cpu().item()),
        "uses_h_only_prior_at_inference": False,
    }
    for k in recall_ks:
        metrics[f"unified_static_recall_at_{int(k)}"] = _recall_at_k_multi(logits, positive_indices, int(k))
    return loss, metrics


def _query_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("source") or row.get("source_dataset") or "unknown")


def _query_training_bucket(row: dict[str, Any]) -> str:
    source = _query_source(row)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = metadata.get("provenance")
    if isinstance(provenance, str) and provenance.strip():
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    target = str(provenance.get("target") or "")
    if source.startswith("trajectory_derived_") and target:
        return f"{source}:{target}"
    return source


def _build_training_buckets(
    queries: list[dict[str, Any]],
    *,
    sampling_strategy: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in queries:
        key = (
            _query_training_bucket(row)
            if sampling_strategy in {"handoff_balanced", "handoff_tempered"}
            else _query_source(row)
        )
        buckets.setdefault(key, []).append(row)
    return buckets


def _training_bucket_counts(buckets: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {str(key): len(rows) for key, rows in sorted(buckets.items())}


def _build_source_buckets(queries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return _build_training_buckets(queries, sampling_strategy="source_balanced")


def _is_stage0_correction_bucket(name: str) -> bool:
    return "stage0" in str(name) and "missing_positive" in str(name)


def _sample_balanced_bucket_rows(
    buckets: dict[str, list[dict[str, Any]]],
    bucket_names: list[str],
    *,
    step: int,
    slots: int,
) -> list[dict[str, Any]]:
    if slots <= 0 or not bucket_names:
        return []
    start = ((max(1, int(step)) - 1) * max(1, int(slots))) % len(bucket_names)
    batch: list[dict[str, Any]] = []
    for offset in range(max(1, int(slots))):
        source = bucket_names[(start + offset) % len(bucket_names)]
        source_rows = buckets[source]
        row_index = ((max(1, int(step)) - 1) + offset // len(bucket_names)) % len(source_rows)
        batch.append(source_rows[row_index])
    return batch


def _batch_queries_for_step(
    queries: list[dict[str, Any]],
    step: int,
    batch_size: int,
    sampling_strategy: str = "batch_stride",
    source_buckets: dict[str, list[dict[str, Any]]] | None = None,
    tempered_correction_fraction: float = 0.2,
) -> list[dict[str, Any]]:
    if not queries:
        raise ValueError("queries must be non-empty")
    if sampling_strategy in {"source_balanced", "handoff_balanced"}:
        buckets = source_buckets or _build_training_buckets(queries, sampling_strategy=sampling_strategy)
        source_names = [source for source, rows in buckets.items() if rows]
        if not source_names:
            raise ValueError(f"{sampling_strategy} sampling requires at least one non-empty source bucket")
        return _sample_balanced_bucket_rows(
            buckets,
            source_names,
            step=step,
            slots=max(1, int(batch_size)),
        )
    if sampling_strategy == "handoff_tempered":
        buckets = source_buckets or _build_training_buckets(queries, sampling_strategy=sampling_strategy)
        source_names = [source for source, rows in buckets.items() if rows]
        if not source_names:
            raise ValueError("handoff_tempered sampling requires at least one non-empty source bucket")
        correction_names = [source for source in source_names if _is_stage0_correction_bucket(source)]
        base_names = [source for source in source_names if source not in correction_names]
        if not correction_names or not base_names:
            return _sample_balanced_bucket_rows(
                buckets,
                source_names,
                step=step,
                slots=max(1, int(batch_size)),
            )
        fraction = min(max(float(tempered_correction_fraction), 0.0), 1.0)
        correction_slots = int(round(max(1, int(batch_size)) * fraction))
        if fraction > 0.0:
            correction_slots = max(1, correction_slots)
        correction_slots = min(max(1, int(batch_size)), correction_slots)
        base_slots = max(1, int(batch_size)) - correction_slots
        base_batch = _sample_balanced_bucket_rows(
            buckets,
            base_names,
            step=step,
            slots=base_slots,
        )
        correction_batch = _sample_balanced_bucket_rows(
            buckets,
            correction_names,
            step=step,
            slots=correction_slots,
        )
        return base_batch + correction_batch
    if sampling_strategy != "batch_stride":
        raise ValueError(f"unsupported retrieval warmup sampling_strategy: {sampling_strategy}")
    start = ((max(1, int(step)) - 1) * max(1, int(batch_size))) % len(queries)
    return [queries[(start + offset) % len(queries)] for offset in range(max(1, int(batch_size)))]


def _normalize_metric_recall_ks(top_k: int, metric_recall_ks: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    values = [1, int(top_k)]
    if metric_recall_ks is not None:
        values.extend(int(value) for value in metric_recall_ks)
    return tuple(sorted({max(1, value) for value in values}))


def _optimizer_parameter_names(model: Any, params: list[torch.nn.Parameter]) -> list[str]:
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    return [id_to_name.get(id(param), "<unnamed>") for param in params]


def run_skillret_retrieval_warmup(
    data_root: str | Path,
    base_model_name: str,
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 2,
    model_dim: int = 16,
    top_k: int = 8,
    max_skills: int | None = 512,
    max_queries: int | None = 256,
    learning_rate: float = 1.0e-4,
    seed: int = 13,
    split_path: str | Path | None = None,
    train_split: str = "train",
    encoder_pooling: str = "masked_mean",
    cross_encoder_pooling: str = "masked_mean",
    tokenizer_padding_side: str | None = None,
    torch_dtype: str | None = None,
    freeze_backbone: bool = False,
    max_length: int | None = None,
    projection_init: str = "default",
    normalize_embeddings: bool = False,
    defer_skill_table_init: bool = False,
    skill_text_format: str = "clstr",
    skill_table_batch_size: int = 32,
    skill_table_adapter_init: str = "default",
    use_cross_encoder: bool = True,
    query_text_format: str = "raw",
    state_query_prompt_version: str | None = None,
    state_query_max_chars: int | None = None,
    state_query_truncation: str | None = None,
    data_format: str = "skillret",
    expand_alias_positives: bool = True,
    shuffle_queries: bool = True,
    sampling_strategy: str = "batch_stride",
    metric_recall_ks: tuple[int, ...] | list[int] | None = None,
    gradient_accumulation_steps: int = 1,
    retrieval_loss_mode: str = "single_positive_ce",
    train_skill_embeddings: bool = True,
    train_skill_bias: bool = True,
    train_encoder_backbone: bool = False,
    encoder_backbone_learning_rate: float | None = None,
    train_encoder_projection: bool = True,
    train_skill_adapter: bool = True,
    train_retrieval_scale: bool = True,
    init_checkpoint_path: str | Path | None = None,
    init_checkpoint_skills_path: str | Path | None = None,
    tempered_correction_fraction: float = 0.2,
    preservation_anchor_weight: float = 0.0,
    explicit_negative_loss_weight: float = 0.0,
    explicit_negative_margin: float = 0.1,
    mined_hard_negative_loss_weight: float = 0.0,
    mined_hard_negative_margin: float = 0.1,
    mined_hard_negative_top_k: int = 32,
    route_scorer: str = "legacy_biencoder",
    belief_top_k: int | None = None,
    resume_checkpoint_path: str | Path | None = None,
    frozen_backbone_cache_mode: str = "off",
    frozen_backbone_cache_batch_size: int = 128,
    resume_skill_table_mode: str = "rebuild",
    protocol_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_state = _load_stage0_resume_state(resume_checkpoint_path) if resume_checkpoint_path is not None else None
    monitor = TrainingMonitor(output_dir, checkpoint_dir=checkpoint_dir, reset=resume_state is None)
    if resume_state is not None:
        _load_monitor_history(monitor)
    setup_status_path = output_dir / "setup_status.jsonl"
    if resume_state is None:
        setup_status_path.write_text("", encoding="utf-8")
    random.seed(seed)
    torch.manual_seed(seed)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    if retrieval_loss_mode not in {"single_positive_ce", "multi_positive_nll"}:
        raise ValueError(f"unsupported retrieval_loss_mode: {retrieval_loss_mode}")
    route_scorer = str(route_scorer or "legacy_biencoder")
    if route_scorer not in {"legacy_biencoder", "unified_memory"}:
        raise ValueError(f"unsupported retrieval route_scorer: {route_scorer}")
    frozen_backbone_cache_mode = str(frozen_backbone_cache_mode or "off")
    if frozen_backbone_cache_mode not in {"off", "schedule"}:
        raise ValueError(
            "unsupported frozen_backbone_cache_mode: "
            f"{frozen_backbone_cache_mode}"
        )
    frozen_backbone_cache_batch_size = int(frozen_backbone_cache_batch_size)
    if frozen_backbone_cache_batch_size <= 0:
        raise ValueError("frozen_backbone_cache_batch_size must be positive")
    resume_skill_table_mode = str(resume_skill_table_mode or "rebuild")
    if resume_skill_table_mode not in {"rebuild", "verified_checkpoint"}:
        raise ValueError(
            f"unsupported resume_skill_table_mode: {resume_skill_table_mode}"
        )
    if frozen_backbone_cache_mode == "schedule" and train_encoder_backbone:
        raise ValueError(
            "Stage0 frozen-backbone cache cannot train the encoder backbone"
        )
    if resume_skill_table_mode == "verified_checkpoint" and resume_state is None:
        raise ValueError(
            "verified_checkpoint resume_skill_table_mode requires a resume checkpoint"
        )
    if sampling_strategy not in {"batch_stride", "source_balanced", "handoff_balanced", "handoff_tempered"}:
        raise ValueError(f"unsupported retrieval warmup sampling_strategy: {sampling_strategy}")
    tempered_correction_fraction = min(max(float(tempered_correction_fraction), 0.0), 1.0)
    preservation_anchor_weight = max(0.0, float(preservation_anchor_weight))
    explicit_negative_loss_weight = max(0.0, float(explicit_negative_loss_weight))
    explicit_negative_margin = max(0.0, float(explicit_negative_margin))
    mined_hard_negative_loss_weight = max(0.0, float(mined_hard_negative_loss_weight))
    mined_hard_negative_margin = max(0.0, float(mined_hard_negative_margin))
    mined_hard_negative_top_k = max(0, int(mined_hard_negative_top_k))
    protocol_metadata = dict(protocol_metadata or {})
    state_query_contract = _resolve_stage0_state_query_contract(
        query_text_format=query_text_format,
        state_query_prompt_version=state_query_prompt_version,
        state_query_max_chars=state_query_max_chars,
        state_query_truncation=state_query_truncation,
    )

    skill_rows, query_rows, positives_by_query = load_retrieval_warmup_rows(data_root, data_format=data_format)
    _append_jsonl(
        setup_status_path,
        {
            "phase": "data_loaded",
            "data_root": str(data_root),
            "data_format": data_format,
            "raw_skill_count": len(skill_rows),
            "raw_query_count": len(query_rows),
        },
    )
    protocol_query_ids = _load_split_query_ids(split_path, train_split)
    if protocol_query_ids is not None:
        query_rows = [row for row in query_rows if str(row["query_id"]) in protocol_query_ids]
        positives_by_query = {
            query_id: positives
            for query_id, positives in positives_by_query.items()
            if query_id in protocol_query_ids
        }
    query_count_before_max_queries = len(query_rows)
    if max_skills is not None:
        skill_rows = skill_rows[:max_skills]
    if not skill_rows:
        raise ValueError(f"no retrieval warmup skills found under {data_root}")
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skill_rows)}
    alias_positive_ids_by_skill_id = (
        _skill_pool_alias_positive_ids_by_skill_id(skill_rows)
        if expand_alias_positives
        else {}
    )
    queries = _filter_queries_to_skill_pool(
        query_rows,
        positives_by_query,
        skill_id_to_idx,
        max_queries,
        alias_positive_ids_by_skill_id=alias_positive_ids_by_skill_id,
    )
    if not queries:
        raise ValueError("no retrieval warmup queries have positives inside the selected skill pool")
    if shuffle_queries:
        random.Random(seed).shuffle(queries)
    assign_stage0_query_indices(queries)
    source_buckets = _build_training_buckets(queries, sampling_strategy=sampling_strategy)
    training_bucket_counts = _training_bucket_counts(source_buckets)
    positive_label_counts = [len(row.get("positive_indices") or []) for row in queries]
    explicit_negative_counts = [len(row.get("negative_indices") or []) for row in queries]
    explicit_negative_report = {
        "enabled": explicit_negative_loss_weight > 0.0,
        "loss_weight": explicit_negative_loss_weight,
        "margin": explicit_negative_margin,
        "query_count": len(queries),
        "query_with_negative_count": sum(1 for count in explicit_negative_counts if count > 0),
        "query_with_negative_fraction": sum(1 for count in explicit_negative_counts if count > 0) / max(len(queries), 1),
        "mean_negative_count": sum(explicit_negative_counts) / max(len(explicit_negative_counts), 1),
        "max_negative_count": max(explicit_negative_counts) if explicit_negative_counts else 0,
    }
    mined_hard_negative_report = {
        "enabled": mined_hard_negative_loss_weight > 0.0,
        "loss_weight": mined_hard_negative_loss_weight,
        "margin": mined_hard_negative_margin,
        "top_k": mined_hard_negative_top_k,
        "query_count": len(queries),
        "note": "online hard negatives are mined from current full-pool non-positive logits per batch",
    }
    alias_positive_expanded_query_count = sum(1 for row in queries if row.get("stage0_alias_positive_expanded"))
    alias_resolved_query_count = sum(1 for row in queries if row.get("stage0_alias_positive_resolved"))
    multi_positive_expanded_query_count = sum(1 for row in queries if row.get("stage0_multi_positive_expanded"))
    alias_positive_report = {
        "enabled": bool(expand_alias_positives),
        "query_count": len(queries),
        "expanded_query_count": int(alias_positive_expanded_query_count),
        "expanded_query_fraction": alias_positive_expanded_query_count / max(len(queries), 1),
        "alias_resolved_query_count": int(alias_resolved_query_count),
        "alias_resolved_query_fraction": alias_resolved_query_count / max(len(queries), 1),
        "multi_positive_expanded_query_count": int(multi_positive_expanded_query_count),
        "multi_positive_expanded_query_fraction": multi_positive_expanded_query_count / max(len(queries), 1),
        "mean_positive_count": sum(positive_label_counts) / max(len(positive_label_counts), 1),
        "max_positive_count": max(positive_label_counts) if positive_label_counts else 0,
    }

    skills_path = output_dir / "selected_skills.jsonl"
    skills_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skill_rows),
        encoding="utf-8",
    )
    _append_jsonl(
        setup_status_path,
        {
            "phase": "selected_skills_written",
            "selected_skills_path": str(skills_path),
            "selected_skill_count": len(skill_rows),
            "usable_query_count": len(queries),
            "query_count_before_max_queries": query_count_before_max_queries,
            "sampling_strategy": sampling_strategy,
            "tempered_correction_fraction": tempered_correction_fraction,
            "training_bucket_count": len(training_bucket_counts),
            "training_bucket_counts": training_bucket_counts,
            "shuffle_queries": bool(shuffle_queries),
            "seed": int(seed),
            "alias_positive_expansion": alias_positive_report,
            "explicit_negative_supervision": explicit_negative_report,
            "mined_hard_negative_supervision": mined_hard_negative_report,
        },
    )
    skills = load_eval_pool(skills_path)
    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=base_model_name,
            d=model_dim,
            d_a=max(4, model_dim // 4),
            top_k=top_k,
            encoder_pooling=encoder_pooling,
            cross_encoder_pooling=cross_encoder_pooling,
            tokenizer_padding_side=tokenizer_padding_side,
            torch_dtype=torch_dtype,
            freeze_backbone=freeze_backbone,
            max_length=max_length,
            projection_init=projection_init,
            normalize_embeddings=normalize_embeddings,
            defer_skill_table_init=True,
            skill_text_format=skill_text_format,
            state_query_prompt_version=state_query_contract["state_query_prompt_version"],
            state_query_max_chars=state_query_contract["state_query_max_chars"],
            state_query_truncation=state_query_contract["state_query_truncation"],
            skill_table_batch_size=skill_table_batch_size,
            skill_table_adapter_init=skill_table_adapter_init,
            use_cross_encoder=use_cross_encoder,
            initial_belief_top_k=belief_top_k,
        ),
        skills,
    )
    current_skill_pool_identity = ordered_skill_pool_identity(
        skill_rows,
        skill_text_fn=model.skill_table.skill_text_fn,
    )
    _append_jsonl(
        setup_status_path,
        {
            "phase": "model_initialized",
            "base_model_name": base_model_name,
            "requested_defer_skill_table_init": bool(defer_skill_table_init),
            "effective_defer_skill_table_init": True,
            "use_cross_encoder": bool(use_cross_encoder),
            "route_scorer": route_scorer,
            "belief_top_k": belief_top_k,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    recall_ks = _normalize_metric_recall_ks(top_k, metric_recall_ks)
    metrics: dict[str, float] = {
        "loss": 0.0,
        "retriever_loss": 0.0,
        "reranker_loss": 0.0,
        "reranker_accuracy": 0.0,
        "preservation_anchor_loss": 0.0,
        "preservation_anchor_unweighted_loss": 0.0,
        "preservation_anchor_parameter_count": 0.0,
        "preservation_anchor_weight": preservation_anchor_weight,
        "explicit_negative_loss": 0.0,
        "weighted_explicit_negative_loss": 0.0,
        "explicit_negative_rows": 0.0,
        "explicit_negative_pair_count": 0.0,
        "explicit_negative_loss_weight": explicit_negative_loss_weight,
        "explicit_negative_margin": explicit_negative_margin,
        "mined_hard_negative_loss": 0.0,
        "weighted_mined_hard_negative_loss": 0.0,
        "mined_hard_negative_rows": 0.0,
        "mined_hard_negative_pair_count": 0.0,
        "mined_hard_negative_loss_weight": mined_hard_negative_loss_weight,
        "mined_hard_negative_margin": mined_hard_negative_margin,
        "mined_hard_negative_top_k": mined_hard_negative_top_k,
        "route_scorer": route_scorer,
        "uses_h_only_prior_at_inference": route_scorer != "unified_memory",
    }
    metrics.update({f"recall_at_{k}": 0.0 for k in recall_ks})
    stage_name = "clstr_unified_retrieval_v2" if data_format == "unified_v2" else "skillret_retrieval_warmup"
    train_safety = {
        "unified_v2_train_safe_retrieval": data_format == "unified_v2",
        "excluded_retrieval_splits": sorted(UNSAFE_RETRIEVAL_SPLITS) if data_format == "unified_v2" else [],
        "public_train_or_eval_unlabeled_excluded": data_format == "unified_v2",
        "train_split_filter_applied": data_format == "unified_v2",
    }
    checkpoint_config = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "d_a": max(4, model_dim // 4),
        "top_k": top_k,
        "encoder_pooling": encoder_pooling,
        "cross_encoder_pooling": cross_encoder_pooling,
        "tokenizer_padding_side": tokenizer_padding_side,
        "torch_dtype": torch_dtype,
        "freeze_backbone": freeze_backbone,
        "max_length": max_length,
        "projection_init": projection_init,
        "normalize_embeddings": normalize_embeddings,
        "defer_skill_table_init": True,
        "skill_text_format": skill_text_format,
        "skill_table_batch_size": skill_table_batch_size,
        "skill_table_adapter_init": skill_table_adapter_init,
        "use_cross_encoder": use_cross_encoder,
        "query_text_format": query_text_format,
        "router_state_contract": ROUTER_STATE_CONTRACT,
        **state_query_contract,
        "data_format": data_format,
        "expand_alias_positives": bool(expand_alias_positives),
        "metric_recall_ks": list(recall_ks),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "retrieval_loss_mode": retrieval_loss_mode,
        "route_scorer": route_scorer,
        "belief_top_k": belief_top_k,
        "uses_h_only_prior_at_inference": route_scorer != "unified_memory",
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": tempered_correction_fraction,
        "train_skill_embeddings": bool(train_skill_embeddings),
        "train_skill_bias": bool(train_skill_bias),
        "train_encoder_backbone": bool(train_encoder_backbone),
        "encoder_backbone_learning_rate": encoder_backbone_learning_rate,
        "train_encoder_projection": bool(train_encoder_projection),
        "train_skill_adapter": bool(train_skill_adapter),
        "train_retrieval_scale": bool(train_retrieval_scale),
        "preservation_anchor_weight": preservation_anchor_weight,
        "explicit_negative_loss_weight": explicit_negative_loss_weight,
        "explicit_negative_margin": explicit_negative_margin,
        "mined_hard_negative_loss_weight": mined_hard_negative_loss_weight,
        "mined_hard_negative_margin": mined_hard_negative_margin,
        "mined_hard_negative_top_k": mined_hard_negative_top_k,
        "init_checkpoint_path": None if init_checkpoint_path is None else str(init_checkpoint_path),
        "init_checkpoint_skills_path": None
        if init_checkpoint_skills_path is None
        else str(init_checkpoint_skills_path),
        "frozen_backbone_cache_mode": frozen_backbone_cache_mode,
        "frozen_backbone_cache_batch_size": frozen_backbone_cache_batch_size,
        "resume_skill_table_mode": resume_skill_table_mode,
        "skill_pool_identity": current_skill_pool_identity,
    }
    if resume_state is None:
        setup_checkpoint_state, setup_checkpoint_state_report = _warmup_checkpoint_state_dict(
            model,
            include_encoder_backbone=bool(train_encoder_backbone),
        )
        monitor.save_latest(
            {
                "stage": stage_name,
                "step": 0,
                "setup_phase": "model_initialized_before_skill_table_rebuild",
                "config": checkpoint_config,
                "model_state_dict": setup_checkpoint_state,
                "metrics": metrics,
                "train_safety": train_safety,
                "setup_status_path": str(setup_status_path),
                "skill_count": len(skill_rows),
                "query_count": len(queries),
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "retrieval_loss_mode": retrieval_loss_mode,
                "route_scorer": route_scorer,
                "belief_top_k": belief_top_k,
                "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
                "protocol_metadata": protocol_metadata,
                "sampling_strategy": sampling_strategy,
                "tempered_correction_fraction": tempered_correction_fraction,
                "training_bucket_count": len(training_bucket_counts),
                "training_bucket_counts": training_bucket_counts,
                "shuffle_queries": bool(shuffle_queries),
                "alias_positive_expansion": alias_positive_report,
                "explicit_negative_supervision": explicit_negative_report,
                "mined_hard_negative_supervision": mined_hard_negative_report,
                **setup_checkpoint_state_report,
            }
        )
    skill_table_resume_report: dict[str, Any] = {
        "mode": resume_skill_table_mode,
        "resume_checkpoint": resume_state is not None,
        "rebuild_skipped": False,
        "skill_pool_identity": current_skill_pool_identity,
    }
    if resume_state is not None and resume_skill_table_mode == "verified_checkpoint":
        verification = validate_verified_resume_skill_table(
            checkpoint_state=resume_state["model_state_dict"],
            checkpoint_config=dict(resume_state.get("config") or {}),
            checkpoint_identity=(
                resume_state.get("skill_pool_identity")
                or dict(resume_state.get("config") or {}).get("skill_pool_identity")
            ),
            resume_skill_rows=resume_state.get("resume_selected_skill_rows"),
            current_skill_rows=skill_rows,
            expected_shape=tuple(model.skill_table.E.shape),
            expected_config=_stage0_resume_expected_config(model),
            skill_text_fn=model.skill_table.skill_text_fn,
        )
        skill_table_resume_report = {
            **skill_table_resume_report,
            **verification,
            "rebuild_skipped": True,
            "resume_checkpoint_path": resume_state["checkpoint_path"],
        }
        _append_jsonl(
            setup_status_path,
            {
                "phase": "skill_table_rebuild_skipped_verified_resume",
                **skill_table_resume_report,
            },
        )
    else:
        _append_jsonl(
            setup_status_path,
            {
                "phase": "skill_table_rebuild_started",
                "skill_count": len(skill_rows),
                "skill_table_batch_size": int(skill_table_batch_size),
            },
        )
        previous_progress_callback = getattr(
            model.skill_table,
            "embedding_progress_callback",
            None,
        )

        def _record_skill_table_rebuild_progress(progress: dict[str, int]) -> None:
            _append_jsonl(
                setup_status_path,
                {
                    "phase": "skill_table_rebuild_progress",
                    "skill_count": len(skill_rows),
                    "skill_table_batch_size": int(skill_table_batch_size),
                    "batch_index": int(progress["batch_index"]),
                    "batch_count": int(progress["batch_count"]),
                    "encoded_skill_count": int(progress["end"]),
                    "total_skill_count": int(progress["total"]),
                },
            )

        model.skill_table.embedding_progress_callback = (
            _record_skill_table_rebuild_progress
        )
        try:
            model.rebuild_skill_table()
        finally:
            model.skill_table.embedding_progress_callback = (
                previous_progress_callback
            )
        _append_jsonl(
            setup_status_path,
            {
                "phase": "skill_table_rebuilt",
                "skill_count": len(skill_rows),
                "skill_embedding_shape": list(model.skill_table.E.shape),
            },
        )
    if resume_state is not None:
        resume_load_report = load_compatible_state_dict(
            model,
            resume_state["model_state_dict"],
            partial_load_mode="stage0_resume_checkpoint_compatible_state",
        )
        init_checkpoint_report = {
            "loaded": True,
            "path": resume_state["checkpoint_path"],
            "checkpoint_step": int(resume_state["step"]),
            "loaded_state_key_count": int(resume_load_report["loaded_key_count"]),
            "loaded_state_keys": list(resume_load_report["loaded_keys"]),
            "missing_state_keys": list(resume_load_report["missing_keys"]),
            "resume_checkpoint": True,
        }
    else:
        init_checkpoint_report = _load_warmup_init_checkpoint(
            model,
            init_checkpoint_path,
            init_checkpoint_skills_path=init_checkpoint_skills_path,
            current_skills_path=data_root / "skill_pool.jsonl" if data_format == "unified_v2" else None,
        )
    if init_checkpoint_report["loaded"]:
        _append_jsonl(
            setup_status_path,
            {
                "phase": "init_checkpoint_loaded",
                **init_checkpoint_report,
            },
        )
    if hasattr(model.skill_table, "set_retrieval_trainable"):
        model.skill_table.set_retrieval_trainable(False)
    encoder_backbone = getattr(model.encoder, "backbone", None)
    encoder_backbone_params = (
        list(encoder_backbone.parameters())
        if encoder_backbone is not None
        else []
    )
    if train_encoder_backbone and not encoder_backbone_params:
        raise ValueError(
            "train_encoder_backbone requires encoder.backbone parameters"
        )
    for param in encoder_backbone_params:
        param.requires_grad_(bool(train_encoder_backbone))
    for param in model.encoder.proj.parameters():
        param.requires_grad_(bool(train_encoder_projection))
    for param in model.skill_table.W.parameters():
        param.requires_grad_(bool(train_skill_adapter))
    model.skill_table.E.requires_grad_(bool(train_skill_embeddings))
    model.skill_table.logit_scale_retr.requires_grad_(bool(train_retrieval_scale))
    model.skill_table.skill_bias_retr.requires_grad_(bool(train_skill_bias))
    if hasattr(model.skill_table, "set_belief_scale_trainable"):
        model.skill_table.set_belief_scale_trainable(route_scorer == "unified_memory")

    optimizer_params: list[dict[str, Any]] = []
    if train_encoder_backbone:
        optimizer_params.append(
            {
                "params": encoder_backbone_params,
                "lr": float(encoder_backbone_learning_rate if encoder_backbone_learning_rate is not None else learning_rate),
            }
        )
    if train_encoder_projection:
        optimizer_params.append({"params": list(model.encoder.proj.parameters()), "lr": learning_rate})
    if train_skill_adapter:
        optimizer_params.append({"params": list(model.skill_table.W.parameters()), "lr": learning_rate})
    if train_skill_embeddings:
        optimizer_params.append({"params": [model.skill_table.E], "lr": learning_rate})
    if train_retrieval_scale:
        optimizer_params.append({"params": [model.skill_table.logit_scale_retr], "lr": learning_rate})
    if train_skill_bias:
        optimizer_params.append({"params": [model.skill_table.skill_bias_retr], "lr": learning_rate})
    if route_scorer == "unified_memory":
        optimizer_params.append({"params": list(model.initial_belief_head.parameters()), "lr": learning_rate})
        optimizer_params.append({"params": list(model.unified_retriever.parameters()), "lr": learning_rate})
        optimizer_params.append({"params": [model.skill_table.logit_scale_belief], "lr": learning_rate})
        optimizer_params.append({"params": [model.skill_table.skill_bias_belief], "lr": learning_rate})
    if model.cross_encoder is not None:
        optimizer_params.append({"params": list(model.cross_encoder.proj.parameters()), "lr": learning_rate})
        optimizer_params.append({"params": list(model.skill_head.parameters()), "lr": learning_rate})
    flat_optimizer_params = [
        param
        for group in optimizer_params
        for param in group["params"]
    ]
    optimizer_parameter_names = _optimizer_parameter_names(model, flat_optimizer_params)
    anchor_snapshot = (
        _snapshot_trainable_parameters(model, parameter_names=set(optimizer_parameter_names))
        if preservation_anchor_weight > 0.0
        else {}
    )
    anchor_setup_report = {
        "preservation_anchor_weight": preservation_anchor_weight,
        "preservation_anchor_parameter_count": len(anchor_snapshot),
        "preservation_anchor_enabled": bool(anchor_snapshot),
        "preservation_anchor_parameter_names": sorted(anchor_snapshot),
    }
    if not flat_optimizer_params:
        raise ValueError("no trainable retrieval warmup parameters selected")
    optimizer = torch.optim.AdamW(optimizer_params)
    start_step = 1
    if resume_state is not None:
        optimizer_state = resume_state.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, device)
        start_step = int(resume_state["step"]) + 1
    frozen_backbone_cache: Stage0FrozenBackboneCache | None = None
    frozen_backbone_cache_report: dict[str, Any] = {
        "enabled": False,
        "mode": frozen_backbone_cache_mode,
        "batch_size": frozen_backbone_cache_batch_size,
    }
    if start_step > int(max_steps):
        report = {
            "status": "ok",
            "training_data_source": stage_name if data_format == "unified_v2" else "skillret_retrieval",
            "data_format": data_format,
            "data_root": str(data_root),
            "resume": {
                "enabled": True,
                "checkpoint_path": resume_state["checkpoint_path"] if resume_state is not None else None,
                "checkpoint_step": int(resume_state["step"]) if resume_state is not None else None,
            },
            "message": "resume checkpoint is already at or beyond max_steps",
            "steps": max_steps,
            "metrics": resume_state.get("metrics", {}) if resume_state is not None else {},
            "skill_pool_identity": current_skill_pool_identity,
            "skill_table_resume": skill_table_resume_report,
            "frozen_backbone_cache": frozen_backbone_cache_report,
            **monitor.paths_report(),
        }
        _write_json(output_dir / "train_report.json", report)
        return report

    if frozen_backbone_cache_mode == "schedule":
        if encoder_backbone is None:
            raise ValueError(
                "Stage0 frozen-backbone cache requires encoder.backbone"
            )
        if any(param.requires_grad for param in encoder_backbone_params):
            raise ValueError(
                "Stage0 frozen-backbone cache requires a fully frozen backbone"
            )
        scheduled_rows = plan_stage0_scheduled_rows(
            queries=queries,
            start_step=start_step,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_size=batch_size,
            sampling_strategy=sampling_strategy,
            source_buckets=source_buckets,
            tempered_correction_fraction=tempered_correction_fraction,
            batch_sampler=_batch_queries_for_step,
        )
        cache_identity = {
            "router_state_contract": ROUTER_STATE_CONTRACT,
            "base_model_name": base_model_name,
            "encoder_pooling": encoder_pooling,
            "tokenizer_padding_side": tokenizer_padding_side,
            "torch_dtype": torch_dtype,
            "max_length": max_length,
            "state_query_prompt_version": state_query_contract[
                "state_query_prompt_version"
            ],
            "state_query_max_chars": state_query_contract[
                "state_query_max_chars"
            ],
            "state_query_truncation": state_query_contract[
                "state_query_truncation"
            ],
            "sampling_strategy": sampling_strategy,
            "seed": int(seed),
            "start_step": int(start_step),
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(
                gradient_accumulation_steps
            ),
        }
        _append_jsonl(
            setup_status_path,
            {
                "phase": "frozen_backbone_cache_build_started",
                "mode": frozen_backbone_cache_mode,
                "scheduled_unique_row_count": len(scheduled_rows),
                "batch_size": frozen_backbone_cache_batch_size,
                "identity": cache_identity,
            },
        )
        frozen_backbone_cache = build_stage0_frozen_backbone_cache(
            model,
            scheduled_rows,
            batch_size=frozen_backbone_cache_batch_size,
            identity=cache_identity,
        )
        frozen_backbone_cache_report = frozen_backbone_cache.report()
        _append_jsonl(
            setup_status_path,
            {
                "phase": "frozen_backbone_cache_built",
                **frozen_backbone_cache_report,
            },
        )

    initial_checkpoint_state, initial_checkpoint_state_report = _warmup_checkpoint_state_dict(
        model,
        include_encoder_backbone=bool(train_encoder_backbone),
    )
    initial_checkpoint_payload = {
        "stage": stage_name,
        "step": int(start_step - 1),
        "setup_phase": "training_started",
        "config": checkpoint_config,
        "model_state_dict": initial_checkpoint_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "train_safety": train_safety,
        "setup_status_path": str(setup_status_path),
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "retrieval_loss_mode": retrieval_loss_mode,
        "route_scorer": route_scorer,
        "belief_top_k": belief_top_k,
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
        },
        "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
        "protocol_metadata": protocol_metadata,
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": tempered_correction_fraction,
        "training_bucket_count": len(training_bucket_counts),
        "training_bucket_counts": training_bucket_counts,
        "shuffle_queries": bool(shuffle_queries),
        "alias_positive_expansion": alias_positive_report,
        "explicit_negative_supervision": explicit_negative_report,
        "mined_hard_negative_supervision": mined_hard_negative_report,
        "init_checkpoint": init_checkpoint_report,
        "preservation_anchor": anchor_setup_report,
        "skill_pool_identity": current_skill_pool_identity,
        "skill_table_resume": skill_table_resume_report,
        "frozen_backbone_cache": frozen_backbone_cache_report,
        "trainable_parameter_policy": {
            "trains_encoder_backbone": bool(
                any(param.requires_grad for param in encoder_backbone_params)
            ),
            "optimizer_parameter_names": optimizer_parameter_names,
        },
        **initial_checkpoint_state_report,
    }
    initial_checkpoint_path = None
    if resume_state is None and start_step == 1:
        initial_checkpoint_path = _save_named_stage0_checkpoint(
            checkpoint_dir=checkpoint_dir,
            stage_name=stage_name,
            step=0,
            payload=initial_checkpoint_payload,
        )
    monitor.save_latest(initial_checkpoint_payload)
    _append_jsonl(
        setup_status_path,
        {
            "phase": "training_started",
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
            "retrieval_loss_mode": retrieval_loss_mode,
            "route_scorer": route_scorer,
            "belief_top_k": belief_top_k,
            "sampling_strategy": sampling_strategy,
            "tempered_correction_fraction": tempered_correction_fraction,
            "training_bucket_count": len(training_bucket_counts),
            "training_bucket_counts": training_bucket_counts,
            "shuffle_queries": bool(shuffle_queries),
            "seed": int(seed),
            "alias_positive_expansion": alias_positive_report,
            "explicit_negative_supervision": explicit_negative_report,
            "mined_hard_negative_supervision": mined_hard_negative_report,
            "preservation_anchor": anchor_setup_report,
            "skill_pool_identity": current_skill_pool_identity,
            "skill_table_resume": skill_table_resume_report,
            "frozen_backbone_cache": frozen_backbone_cache_report,
            "latest_checkpoint": str(monitor.latest_checkpoint_path),
            "training_metrics_path": str(monitor.metrics_path),
            "loss_curve_path": str(monitor.loss_curve_path),
        },
    )
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, max_steps + 1):
        step_batches: list[dict[str, Any]] = []
        step_positive_indices: list[list[int]] = []
        step_logits: list[torch.Tensor] = []
        micro_losses: list[torch.Tensor] = []
        retriever_loss_values: list[float] = []
        explicit_negative_loss_values: list[float] = []
        weighted_explicit_negative_loss_values: list[float] = []
        explicit_negative_row_counts: list[int] = []
        explicit_negative_pair_counts: list[int] = []
        mined_hard_negative_loss_values: list[float] = []
        weighted_mined_hard_negative_loss_values: list[float] = []
        mined_hard_negative_row_counts: list[int] = []
        mined_hard_negative_pair_counts: list[int] = []
        reranker_loss_values: list[float] = []
        reranker_accuracy_values: list[float] = []
        anchor_loss_values: list[float] = []
        anchor_unweighted_loss_values: list[float] = []
        anchor_parameter_count = len(anchor_snapshot)
        for micro_step in range(gradient_accumulation_steps):
            micro_index = (step - 1) * gradient_accumulation_steps + micro_step + 1
            batch = _batch_queries_for_step(
                queries,
                micro_index,
                batch_size,
                sampling_strategy=sampling_strategy,
                source_buckets=source_buckets,
                tempered_correction_fraction=tempered_correction_fraction,
            )
            labels = torch.tensor(
                [row["positive_indices"][0] for row in batch],
                device=device,
                dtype=torch.long,
            )
            positive_indices = [
                [int(idx) for idx in row["positive_indices"]]
                for row in batch
            ]
            negative_indices = [
                [int(idx) for idx in row.get("negative_indices", [])]
                for row in batch
            ]
            query_texts = _stage0_raw_query_texts(batch)
            if frozen_backbone_cache is None:
                h = model.encode_states(query_texts)
            else:
                h = frozen_backbone_cache.project_batch(
                    batch,
                    projection_fn=model.encoder.project_pooled,
                    device=device,
                )
            if route_scorer == "unified_memory":
                m_0 = model.initial_belief(h, top_k=belief_top_k)
                logits = model.unified_route_logits(h, m_0)
                if retrieval_loss_mode == "multi_positive_nll":
                    retriever_loss = _multi_positive_nll(logits, positive_indices)
                else:
                    retriever_loss = F.cross_entropy(logits, labels)
            else:
                logits = model.skill_table.retrieval_logits(h)
                if retrieval_loss_mode == "multi_positive_nll":
                    retriever_loss = _multi_positive_nll(logits, positive_indices)
                else:
                    retriever_loss = F.cross_entropy(logits, labels)
            if explicit_negative_loss_weight > 0.0:
                explicit_negative_loss, explicit_negative_loss_report = _explicit_negative_margin_loss(
                    logits,
                    positive_indices,
                    negative_indices,
                    margin=explicit_negative_margin,
                )
            else:
                explicit_negative_loss = torch.zeros((), device=device)
                explicit_negative_loss_report = {
                    "explicit_negative_rows": 0,
                    "explicit_negative_pair_count": 0,
                }
            weighted_explicit_negative_loss = explicit_negative_loss * explicit_negative_loss_weight
            if mined_hard_negative_loss_weight > 0.0:
                mined_hard_negative_loss, mined_hard_negative_loss_report = _mined_hard_negative_margin_loss(
                    logits,
                    positive_indices,
                    top_k=mined_hard_negative_top_k,
                    margin=mined_hard_negative_margin,
                )
            else:
                mined_hard_negative_loss = torch.zeros((), device=device)
                mined_hard_negative_loss_report = {
                    "mined_hard_negative_rows": 0,
                    "mined_hard_negative_pair_count": 0,
                }
            weighted_mined_hard_negative_loss = mined_hard_negative_loss * mined_hard_negative_loss_weight
            if model.cross_encoder is not None:
                rerank_candidate_rows = []
                for label in labels.detach().cpu().tolist():
                    negative = (label + 1) % len(skill_rows)
                    if negative == label and len(skill_rows) > 1:
                        negative = (label + 2) % len(skill_rows)
                    row = [label] if negative == label else [label, negative]
                    rerank_candidate_rows.append(row)
                rerank_embs = model.batch_cross_encode(query_texts, rerank_candidate_rows)
                rerank_logits = model.skill_head(rerank_embs, h)
                rerank_labels = torch.zeros(len(batch), device=device, dtype=torch.long)
                reranker_loss = F.cross_entropy(rerank_logits, rerank_labels)
                reranker_accuracy = float((rerank_logits.argmax(dim=-1) == rerank_labels).float().mean().detach().cpu().item())
            else:
                reranker_loss = torch.zeros((), device=device)
                reranker_accuracy = 0.0
            anchor_loss, anchor_report = _parameter_anchor_loss(
                model,
                anchor_snapshot,
                weight=preservation_anchor_weight,
            )
            micro_loss = (
                retriever_loss
                + weighted_explicit_negative_loss
                + weighted_mined_hard_negative_loss
                + reranker_loss
                + anchor_loss
            )
            (micro_loss / gradient_accumulation_steps).backward()
            micro_losses.append(micro_loss.detach())
            retriever_loss_values.append(float(retriever_loss.detach().cpu().item()))
            explicit_negative_loss_values.append(float(explicit_negative_loss.detach().cpu().item()))
            weighted_explicit_negative_loss_values.append(float(weighted_explicit_negative_loss.detach().cpu().item()))
            explicit_negative_row_counts.append(int(explicit_negative_loss_report["explicit_negative_rows"]))
            explicit_negative_pair_counts.append(int(explicit_negative_loss_report["explicit_negative_pair_count"]))
            mined_hard_negative_loss_values.append(float(mined_hard_negative_loss.detach().cpu().item()))
            weighted_mined_hard_negative_loss_values.append(float(weighted_mined_hard_negative_loss.detach().cpu().item()))
            mined_hard_negative_row_counts.append(int(mined_hard_negative_loss_report["mined_hard_negative_rows"]))
            mined_hard_negative_pair_counts.append(int(mined_hard_negative_loss_report["mined_hard_negative_pair_count"]))
            reranker_loss_values.append(float(reranker_loss.detach().cpu().item()))
            reranker_accuracy_values.append(float(reranker_accuracy))
            anchor_loss_values.append(float(anchor_loss.detach().cpu().item()))
            anchor_unweighted_loss_values.append(float(anchor_report["anchor_unweighted_loss"]))
            anchor_parameter_count = int(anchor_report["anchor_parameter_count"])
            step_batches.extend(batch)
            step_positive_indices.extend(positive_indices)
            step_logits.append(logits.detach())
        torch.nn.utils.clip_grad_norm_(flat_optimizer_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logits_for_metrics = torch.cat(step_logits, dim=0)
        labels_for_metrics = torch.tensor(
            [row["positive_indices"][0] for row in step_batches],
            device=device,
            dtype=torch.long,
        )
        loss_value = sum(float(value.cpu().item()) for value in micro_losses) / max(len(micro_losses), 1)
        retriever_loss_value = sum(retriever_loss_values) / max(len(retriever_loss_values), 1)
        explicit_negative_loss_value = sum(explicit_negative_loss_values) / max(len(explicit_negative_loss_values), 1)
        weighted_explicit_negative_loss_value = sum(weighted_explicit_negative_loss_values) / max(len(weighted_explicit_negative_loss_values), 1)
        explicit_negative_row_count = sum(explicit_negative_row_counts)
        explicit_negative_pair_count = sum(explicit_negative_pair_counts)
        mined_hard_negative_loss_value = sum(mined_hard_negative_loss_values) / max(len(mined_hard_negative_loss_values), 1)
        weighted_mined_hard_negative_loss_value = sum(weighted_mined_hard_negative_loss_values) / max(len(weighted_mined_hard_negative_loss_values), 1)
        mined_hard_negative_row_count = sum(mined_hard_negative_row_counts)
        mined_hard_negative_pair_count = sum(mined_hard_negative_pair_counts)
        reranker_loss_value = sum(reranker_loss_values) / max(len(reranker_loss_values), 1)
        reranker_accuracy = sum(reranker_accuracy_values) / max(len(reranker_accuracy_values), 1)
        anchor_loss_value = sum(anchor_loss_values) / max(len(anchor_loss_values), 1)
        anchor_unweighted_loss_value = sum(anchor_unweighted_loss_values) / max(len(anchor_unweighted_loss_values), 1)
        if retrieval_loss_mode == "multi_positive_nll":
            recall_values = {
                f"recall_at_{k}": _recall_at_k_multi(logits_for_metrics, step_positive_indices, k)
                for k in recall_ks
            }
        else:
            recall_values = {
                f"recall_at_{k}": _recall_at_k(logits_for_metrics, labels_for_metrics, k)
                for k in recall_ks
            }
        metrics = {
            "step": step,
            "optimizer_step": step,
            "loss": loss_value,
            "retriever_loss": retriever_loss_value,
            "explicit_negative_loss": explicit_negative_loss_value,
            "weighted_explicit_negative_loss": weighted_explicit_negative_loss_value,
            "explicit_negative_rows": explicit_negative_row_count,
            "explicit_negative_pair_count": explicit_negative_pair_count,
            "explicit_negative_loss_weight": explicit_negative_loss_weight,
            "explicit_negative_margin": explicit_negative_margin,
            "mined_hard_negative_loss": mined_hard_negative_loss_value,
            "weighted_mined_hard_negative_loss": weighted_mined_hard_negative_loss_value,
            "mined_hard_negative_rows": mined_hard_negative_row_count,
            "mined_hard_negative_pair_count": mined_hard_negative_pair_count,
            "mined_hard_negative_loss_weight": mined_hard_negative_loss_weight,
            "mined_hard_negative_margin": mined_hard_negative_margin,
            "mined_hard_negative_top_k": mined_hard_negative_top_k,
            "reranker_loss": reranker_loss_value,
            "reranker_accuracy": reranker_accuracy,
            "preservation_anchor_loss": anchor_loss_value,
            "preservation_anchor_unweighted_loss": anchor_unweighted_loss_value,
            "preservation_anchor_parameter_count": anchor_parameter_count,
            "preservation_anchor_weight": preservation_anchor_weight,
            "retrieval_loss_mode": retrieval_loss_mode,
            "route_scorer": route_scorer,
            "uses_h_only_prior_at_inference": route_scorer != "unified_memory",
            "micro_batch_count": gradient_accumulation_steps,
            "effective_batch_size": len(step_batches),
            "positive_label_counts": [len(row) for row in step_positive_indices],
            **recall_values,
            "query_ids": [str(row.get("query_id", "")) for row in step_batches],
            "query_sources": [str(row.get("metadata", {}).get("source") or row.get("source_dataset", "")) for row in step_batches],
        }
        step_checkpoint_state, step_checkpoint_state_report = _warmup_checkpoint_state_dict(
            model,
            include_encoder_backbone=bool(train_encoder_backbone),
        )
        current_cache_report = (
            frozen_backbone_cache.report()
            if frozen_backbone_cache is not None
            else frozen_backbone_cache_report
        )
        monitor.record(
            metrics,
            {
                "stage": stage_name,
                "step": step,
                "config": checkpoint_config,
                "model_state_dict": step_checkpoint_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "train_safety": train_safety,
                "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
                "protocol_metadata": protocol_metadata,
                "setup_status_path": str(setup_status_path),
                "skill_count": len(skill_rows),
                "query_count": len(queries),
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "retrieval_loss_mode": retrieval_loss_mode,
                "route_scorer": route_scorer,
                "belief_top_k": belief_top_k,
                "sampling_strategy": sampling_strategy,
                "tempered_correction_fraction": tempered_correction_fraction,
                "training_bucket_count": len(training_bucket_counts),
                "training_bucket_counts": training_bucket_counts,
                "shuffle_queries": bool(shuffle_queries),
                "alias_positive_expansion": alias_positive_report,
                "explicit_negative_supervision": explicit_negative_report,
                "mined_hard_negative_supervision": mined_hard_negative_report,
                "init_checkpoint": init_checkpoint_report,
                "preservation_anchor": anchor_setup_report,
                "skill_pool_identity": current_skill_pool_identity,
                "skill_table_resume": skill_table_resume_report,
                "frozen_backbone_cache": current_cache_report,
                **step_checkpoint_state_report,
            },
        )

    checkpoint_path = checkpoint_dir / f"{stage_name}-step{max_steps}.pt"
    checkpoint_state, checkpoint_state_report = _warmup_checkpoint_state_dict(
        model,
        include_encoder_backbone=bool(train_encoder_backbone),
    )
    final_cache_report = (
        frozen_backbone_cache.report()
        if frozen_backbone_cache is not None
        else frozen_backbone_cache_report
    )
    final_checkpoint_payload = {
        "stage": stage_name,
        "step": max_steps,
        "config": checkpoint_config,
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "train_safety": train_safety,
        "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
        "protocol_metadata": protocol_metadata,
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
        "retrieval_loss_mode": retrieval_loss_mode,
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": tempered_correction_fraction,
        "training_bucket_count": len(training_bucket_counts),
        "training_bucket_counts": training_bucket_counts,
        "shuffle_queries": bool(shuffle_queries),
        "alias_positive_expansion": alias_positive_report,
        "explicit_negative_supervision": explicit_negative_report,
        "mined_hard_negative_supervision": mined_hard_negative_report,
        "setup_status_path": str(setup_status_path),
        "init_checkpoint": init_checkpoint_report,
        "preservation_anchor": anchor_setup_report,
        "skill_pool_identity": current_skill_pool_identity,
        "skill_table_resume": skill_table_resume_report,
        "frozen_backbone_cache": final_cache_report,
        **monitor.paths_report(),
        **checkpoint_state_report,
    }
    torch.save(final_checkpoint_payload, checkpoint_path)
    monitor.save_latest(final_checkpoint_payload)
    training_objective = (
        "L_retr_full_pool_multi_positive_nll"
        if retrieval_loss_mode == "multi_positive_nll"
        else "L_retr_full_pool_cross_entropy"
    )
    if explicit_negative_loss_weight > 0.0:
        training_objective = f"{training_objective}_plus_explicit_negative_margin"
    if mined_hard_negative_loss_weight > 0.0:
        training_objective = f"{training_objective}_plus_mined_hard_negative_margin"
    report = {
        "status": "ok",
        "training_data_source": stage_name if data_format == "unified_v2" else "skillret_retrieval",
        "data_format": data_format,
        "training_objective": training_objective,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "initial_checkpoint_path": (
            None if initial_checkpoint_path is None else str(initial_checkpoint_path)
        ),
        **monitor.paths_report(),
        "setup_status_path": str(setup_status_path),
        "steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
        "retrieval_loss_mode": retrieval_loss_mode,
        "router_state_contract": ROUTER_STATE_CONTRACT,
        **state_query_contract,
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": tempered_correction_fraction,
        "training_bucket_count": len(training_bucket_counts),
        "training_bucket_counts": training_bucket_counts,
        "shuffle_queries": bool(shuffle_queries),
        "alias_positive_expansion": alias_positive_report,
        "explicit_negative_supervision": explicit_negative_report,
        "mined_hard_negative_supervision": mined_hard_negative_report,
        "train_safety": train_safety,
        "stage0_protocol": protocol_metadata if protocol_metadata.get("role") else {},
        "protocol_metadata": protocol_metadata,
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "checkpoint_step": (
                None if resume_state is None else int(resume_state["step"])
            ),
            "start_step": int(start_step),
        },
        "init_checkpoint": init_checkpoint_report,
        "preservation_anchor": anchor_setup_report,
        "skill_pool_identity": current_skill_pool_identity,
        "skill_table_resume": skill_table_resume_report,
        "frozen_backbone_cache": final_cache_report,
        "split_protocol": {
            "split_path": None if split_path is None else str(split_path),
            "train_split": train_split,
            "query_count_before_max_queries": query_count_before_max_queries,
        },
        "metrics": metrics,
        "metric_recall_ks": list(recall_ks),
        "trainable_parameter_policy": {
            "skill_table_retrieval_path_trainable": bool(model.skill_table.W.weight.requires_grad),
            "trains_encoder_backbone": bool(
                any(param.requires_grad for param in encoder_backbone_params)
            ),
            "trains_skill_table_E": bool(model.skill_table.E.requires_grad),
            "trains_skill_embeddings": bool(model.skill_table.E.requires_grad),
            "trains_skill_bias": bool(model.skill_table.skill_bias_retr.requires_grad),
            "trains_encoder_projection": bool(any(param.requires_grad for param in model.encoder.proj.parameters())),
            "trains_skill_adapter": bool(model.skill_table.W.weight.requires_grad),
            "trains_retrieval_scale": bool(model.skill_table.logit_scale_retr.requires_grad),
            "trains_retrieval_scale_and_bias": bool(
                model.skill_table.logit_scale_retr.requires_grad
                and model.skill_table.skill_bias_retr.requires_grad
            ),
            "use_cross_encoder": bool(model.cross_encoder is not None),
            "optimizer_parameter_names": optimizer_parameter_names,
        },
        **checkpoint_state_report,
        "model_config": {
            "base_model_name": base_model_name,
            "d": model_dim,
            "top_k": top_k,
            "encoder_pooling": encoder_pooling,
            "cross_encoder_pooling": cross_encoder_pooling,
            "tokenizer_padding_side": tokenizer_padding_side,
            "torch_dtype": torch_dtype,
            "freeze_backbone": freeze_backbone,
            "max_length": max_length,
            "projection_init": projection_init,
            "normalize_embeddings": normalize_embeddings,
            "defer_skill_table_init": defer_skill_table_init,
            "effective_defer_skill_table_init": True,
            "skill_text_format": skill_text_format,
            "skill_table_batch_size": skill_table_batch_size,
            "skill_table_adapter_init": skill_table_adapter_init,
            "use_cross_encoder": use_cross_encoder,
            "query_text_format": query_text_format,
            "router_state_contract": ROUTER_STATE_CONTRACT,
            **state_query_contract,
            "data_format": data_format,
            "expand_alias_positives": bool(expand_alias_positives),
            "shuffle_queries": bool(shuffle_queries),
            "metric_recall_ks": list(recall_ks),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "retrieval_loss_mode": retrieval_loss_mode,
            "sampling_strategy": sampling_strategy,
            "tempered_correction_fraction": tempered_correction_fraction,
            "train_skill_embeddings": bool(train_skill_embeddings),
            "train_skill_bias": bool(train_skill_bias),
            "train_encoder_backbone": bool(train_encoder_backbone),
            "encoder_backbone_learning_rate": encoder_backbone_learning_rate,
            "train_encoder_projection": bool(train_encoder_projection),
            "train_skill_adapter": bool(train_skill_adapter),
            "train_retrieval_scale": bool(train_retrieval_scale),
            "preservation_anchor_weight": preservation_anchor_weight,
            "explicit_negative_loss_weight": explicit_negative_loss_weight,
            "explicit_negative_margin": explicit_negative_margin,
            "mined_hard_negative_loss_weight": mined_hard_negative_loss_weight,
            "mined_hard_negative_margin": mined_hard_negative_margin,
            "mined_hard_negative_top_k": mined_hard_negative_top_k,
            "init_checkpoint_path": None if init_checkpoint_path is None else str(init_checkpoint_path),
            "init_checkpoint_skills_path": None
            if init_checkpoint_skills_path is None
            else str(init_checkpoint_skills_path),
            "frozen_backbone_cache_mode": frozen_backbone_cache_mode,
            "frozen_backbone_cache_batch_size": frozen_backbone_cache_batch_size,
            "resume_skill_table_mode": resume_skill_table_mode,
            "skill_pool_identity": current_skill_pool_identity,
        },
        "note": (
            "retrieval warmup uses CLSTR unified v2 retrieval stream and canonical skill_pool"
            if data_format == "unified_v2"
            else "retrieval warmup uses SKILLRET query-skill qrels; it does not use SkillRouter eval labels or SkillsBench trajectories"
        ),
    }
    _write_json(output_dir / "train_report.json", report)
    return report


def run_stage0_biencoder_train(
    *,
    data_root: str | Path,
    base_model_name: str,
    output_dir: str | Path,
    max_steps: int = 5000,
    batch_size: int = 128,
    gradient_accumulation_steps: int = 2,
    model_dim: int = 1024,
    top_k: int = 100,
    max_skills: int | None = None,
    max_queries: int | None = None,
    learning_rate: float = 2.0e-5,
    seed: int = 13,
    split_path: str | Path | None = None,
    train_split: str = "train",
    torch_dtype: str | None = "bfloat16",
    freeze_backbone: bool = True,
    max_length: int | None = 2048,
    skill_table_batch_size: int = 32,
    encoder_pooling: str = STAGE0_BIENCODER_DEFAULTS["encoder_pooling"],
    cross_encoder_pooling: str = STAGE0_BIENCODER_DEFAULTS["cross_encoder_pooling"],
    tokenizer_padding_side: str | None = STAGE0_BIENCODER_DEFAULTS["tokenizer_padding_side"],
    query_text_format: str = STAGE0_BIENCODER_DEFAULTS["query_text_format"],
    state_query_prompt_version: str | None = STAGE0_BIENCODER_DEFAULTS["state_query_prompt_version"],
    state_query_max_chars: int | None = STAGE0_BIENCODER_DEFAULTS["state_query_max_chars"],
    state_query_truncation: str | None = STAGE0_BIENCODER_DEFAULTS["state_query_truncation"],
    metric_recall_ks: tuple[int, ...] | list[int] = (20, 50, 100),
    retrieval_loss_mode: str = STAGE0_BIENCODER_DEFAULTS["retrieval_loss_mode"],
    sampling_strategy: str = STAGE0_BIENCODER_DEFAULTS["sampling_strategy"],
    tempered_correction_fraction: float = 0.2,
    train_skill_embeddings: bool = STAGE0_BIENCODER_DEFAULTS["train_skill_embeddings"],
    train_skill_bias: bool = STAGE0_BIENCODER_DEFAULTS["train_skill_bias"],
    train_encoder_backbone: bool = STAGE0_BIENCODER_DEFAULTS["train_encoder_backbone"],
    encoder_backbone_learning_rate: float | None = None,
    train_encoder_projection: bool = STAGE0_BIENCODER_DEFAULTS["train_encoder_projection"],
    train_skill_adapter: bool = STAGE0_BIENCODER_DEFAULTS["train_skill_adapter"],
    train_retrieval_scale: bool = STAGE0_BIENCODER_DEFAULTS["train_retrieval_scale"],
    expand_alias_positives: bool = STAGE0_BIENCODER_DEFAULTS["expand_alias_positives"],
    init_checkpoint_path: str | Path | None = None,
    init_checkpoint_skills_path: str | Path | None = None,
    preservation_anchor_weight: float = 0.0,
    explicit_negative_loss_weight: float = STAGE0_BIENCODER_DEFAULTS["explicit_negative_loss_weight"],
    explicit_negative_margin: float = STAGE0_BIENCODER_DEFAULTS["explicit_negative_margin"],
    mined_hard_negative_loss_weight: float = STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_loss_weight"],
    mined_hard_negative_margin: float = STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_margin"],
    mined_hard_negative_top_k: int = STAGE0_BIENCODER_DEFAULTS["mined_hard_negative_top_k"],
    route_scorer: str = STAGE0_BIENCODER_DEFAULTS["route_scorer"],
    belief_top_k: int | None = STAGE0_BIENCODER_DEFAULTS["belief_top_k"],
    resume_checkpoint_path: str | Path | None = None,
    frozen_backbone_cache_mode: str = STAGE0_BIENCODER_DEFAULTS[
        "frozen_backbone_cache_mode"
    ],
    frozen_backbone_cache_batch_size: int = STAGE0_BIENCODER_DEFAULTS[
        "frozen_backbone_cache_batch_size"
    ],
    resume_skill_table_mode: str = STAGE0_BIENCODER_DEFAULTS[
        "resume_skill_table_mode"
    ],
    **overrides: Any,
) -> dict[str, Any]:
    ignored_override_keys = sorted(set(overrides) & set(STAGE0_BIENCODER_DEFAULTS))
    state_query_contract = _resolve_stage0_state_query_contract(
        query_text_format=query_text_format,
        state_query_prompt_version=state_query_prompt_version,
        state_query_max_chars=state_query_max_chars,
        state_query_truncation=state_query_truncation,
    )
    stage0_protocol = {
        **STAGE0_BIENCODER_DEFAULTS,
        "retrieval_loss_mode": retrieval_loss_mode,
        "sampling_strategy": sampling_strategy,
        "tempered_correction_fraction": float(tempered_correction_fraction),
        "train_skill_embeddings": bool(train_skill_embeddings),
        "train_skill_bias": bool(train_skill_bias),
        "train_encoder_backbone": bool(train_encoder_backbone),
        "encoder_backbone_learning_rate": encoder_backbone_learning_rate,
        "train_encoder_projection": bool(train_encoder_projection),
        "train_skill_adapter": bool(train_skill_adapter),
        "train_retrieval_scale": bool(train_retrieval_scale),
        "resume_checkpoint_path": None if resume_checkpoint_path is None else str(resume_checkpoint_path),
        "expand_alias_positives": bool(expand_alias_positives),
        "encoder_pooling": str(encoder_pooling),
        "cross_encoder_pooling": str(cross_encoder_pooling),
        "tokenizer_padding_side": tokenizer_padding_side,
        "query_text_format": str(query_text_format),
        "router_state_contract": ROUTER_STATE_CONTRACT,
        **state_query_contract,
        "preservation_anchor_weight": float(preservation_anchor_weight),
        "explicit_negative_loss_weight": float(explicit_negative_loss_weight),
        "explicit_negative_margin": float(explicit_negative_margin),
        "mined_hard_negative_loss_weight": float(mined_hard_negative_loss_weight),
        "mined_hard_negative_margin": float(mined_hard_negative_margin),
        "mined_hard_negative_top_k": int(mined_hard_negative_top_k),
        "route_scorer": str(route_scorer),
        "belief_top_k": belief_top_k,
        "frozen_backbone_cache_mode": str(frozen_backbone_cache_mode),
        "frozen_backbone_cache_batch_size": int(
            frozen_backbone_cache_batch_size
        ),
        "resume_skill_table_mode": str(resume_skill_table_mode),
        "uses_h_only_prior_at_inference": str(route_scorer) != "unified_memory",
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "ignored_override_keys": ignored_override_keys,
        "role": (
            "Unified memory-conditioned static retriever warm-up"
            if str(route_scorer) == "unified_memory"
            else "SkillRouter-compatible bi-encoder coarse retriever"
        ),
        "excludes_cross_encoder_or_listwise_reranker": True,
        "safety_note": "Freeze frozen SkillRouter skill embeddings by default; train only lightweight CLSTR retrieval adapter/scale unless explicitly overridden.",
    }
    report = run_skillret_retrieval_warmup(
        data_root=data_root,
        base_model_name=base_model_name,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_dim=model_dim,
        top_k=top_k,
        max_skills=max_skills,
        max_queries=max_queries,
        learning_rate=learning_rate,
        seed=seed,
        split_path=split_path,
        train_split=train_split,
        torch_dtype=torch_dtype,
        freeze_backbone=freeze_backbone,
        max_length=max_length,
        skill_table_batch_size=skill_table_batch_size,
        encoder_pooling=encoder_pooling,
        cross_encoder_pooling=cross_encoder_pooling,
        tokenizer_padding_side=tokenizer_padding_side,
        projection_init=STAGE0_BIENCODER_DEFAULTS["projection_init"],
        normalize_embeddings=STAGE0_BIENCODER_DEFAULTS["normalize_embeddings"],
        skill_text_format=STAGE0_BIENCODER_DEFAULTS["skill_text_format"],
        skill_table_adapter_init=STAGE0_BIENCODER_DEFAULTS["skill_table_adapter_init"],
        use_cross_encoder=STAGE0_BIENCODER_DEFAULTS["use_cross_encoder"],
        query_text_format=query_text_format,
        state_query_prompt_version=state_query_contract["state_query_prompt_version"],
        state_query_max_chars=state_query_contract["state_query_max_chars"],
        state_query_truncation=state_query_contract["state_query_truncation"],
        data_format=STAGE0_BIENCODER_DEFAULTS["data_format"],
        expand_alias_positives=expand_alias_positives,
        shuffle_queries=STAGE0_BIENCODER_DEFAULTS["shuffle_queries"],
        sampling_strategy=sampling_strategy,
        tempered_correction_fraction=tempered_correction_fraction,
        metric_recall_ks=tuple(metric_recall_ks),
        retrieval_loss_mode=retrieval_loss_mode,
        train_skill_embeddings=train_skill_embeddings,
        train_skill_bias=train_skill_bias,
        train_encoder_backbone=train_encoder_backbone,
        encoder_backbone_learning_rate=encoder_backbone_learning_rate,
        train_encoder_projection=train_encoder_projection,
        train_skill_adapter=train_skill_adapter,
        train_retrieval_scale=train_retrieval_scale,
        init_checkpoint_path=init_checkpoint_path,
        init_checkpoint_skills_path=init_checkpoint_skills_path,
        preservation_anchor_weight=preservation_anchor_weight,
        explicit_negative_loss_weight=explicit_negative_loss_weight,
        explicit_negative_margin=explicit_negative_margin,
        mined_hard_negative_loss_weight=mined_hard_negative_loss_weight,
        mined_hard_negative_margin=mined_hard_negative_margin,
        mined_hard_negative_top_k=mined_hard_negative_top_k,
        route_scorer=route_scorer,
        belief_top_k=belief_top_k,
        resume_checkpoint_path=resume_checkpoint_path,
        frozen_backbone_cache_mode=frozen_backbone_cache_mode,
        frozen_backbone_cache_batch_size=frozen_backbone_cache_batch_size,
        resume_skill_table_mode=resume_skill_table_mode,
        protocol_metadata=stage0_protocol,
    )
    report["stage0_protocol"] = stage0_protocol
    _write_json(Path(output_dir) / "train_report.json", report)
    return report
