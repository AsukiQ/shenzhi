from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from clstr.gated_temporal_reranker import GatedTemporalConfig, GatedTemporalReranker
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.state_query_prompt import (
    RAW_STATE_V1,
    SR_TASK_DESCRIPTION_V1,
    resolve_state_query_prompt_contract,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def checkpoint_payload(path: str | Path, label: str = "checkpoint") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{label} checkpoint must be a dict payload, got {type(payload).__name__}")
    return payload


def checkpoint_state(payload: dict[str, Any], label: str = "checkpoint") -> dict[str, Any]:
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"{label} checkpoint model_state_dict must be a dict, got {type(state).__name__}")
    return state


def _stage0_config_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload.get("config") or {})
    d = int(raw.get("d", 256))
    legacy_query_format = str(raw.get("query_text_format") or "skillrouter")
    prompt_version = raw.get("state_query_prompt_version")
    if prompt_version is None:
        prompt_version = (
            SR_TASK_DESCRIPTION_V1
            if legacy_query_format == "skillrouter"
            else RAW_STATE_V1
        )
    prompt_contract = resolve_state_query_prompt_contract(
        prompt_version=str(prompt_version),
        max_chars=raw.get("state_query_max_chars"),
        truncation=raw.get("state_query_truncation"),
        recorded_instruction=raw.get("state_query_instruction"),
    )
    config = {
        "base_model_name": str(raw.get("base_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        "d": d,
        "d_a": int(raw.get("d_a", max(4, d // 4))),
        "top_k": int(raw.get("top_k", 100)),
        "encoder_pooling": str(raw.get("encoder_pooling", "last_token")),
        "cross_encoder_pooling": str(raw.get("cross_encoder_pooling", raw.get("encoder_pooling", "last_token"))),
        "tokenizer_padding_side": raw.get("tokenizer_padding_side"),
        "torch_dtype": raw.get("torch_dtype"),
        "freeze_backbone": bool(raw.get("freeze_backbone", True)),
        "max_length": raw.get("max_length"),
        "projection_init": str(raw.get("projection_init", "identity")),
        "normalize_embeddings": bool(raw.get("normalize_embeddings", True)),
        "hf_cache_dir": raw.get("hf_cache_dir"),
        "local_files_only": bool(raw.get("local_files_only", False)),
        "defer_skill_table_init": True,
        "skill_text_format": str(raw.get("skill_text_format", "skillret_official")),
        "skill_table_batch_size": int(raw.get("skill_table_batch_size", 32)),
        "skill_table_adapter_init": str(raw.get("skill_table_adapter_init", "identity")),
        "use_cross_encoder": bool(raw.get("use_cross_encoder", False)),
        "query_text_format": legacy_query_format,
        "data_format": raw.get("data_format"),
        "metric_recall_ks": list(raw.get("metric_recall_ks") or []),
        **prompt_contract,
    }
    return config


_SKILL_TABLE_PREFIX_EXPAND_KEYS = {
    "skill_table.E",
    "skill_table.skill_bias_retr",
    "skill_table.skill_bias_belief",
}
_ROUTING_FOUNDATION_HEAD_ALLOWLIST = {
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
}
_DEFAULT_LOGIT_SCALE_RETR = 0.0
_DEFAULT_LOGIT_SCALE_BELIEF = math.log(0.2)


def _is_protected_routing_foundation_key(key: str, protected_prefixes: tuple[str, ...]) -> bool:
    key = str(key)
    if key in _ROUTING_FOUNDATION_HEAD_ALLOWLIST:
        return False
    return any(key.startswith(prefix) for prefix in protected_prefixes)


def _prefix_expand_skill_table_state(
    model: Any,
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    current = model.state_dict() if hasattr(model, "state_dict") else {}
    expanded = dict(state)
    expanded_keys: list[str] = []
    for key in sorted(_SKILL_TABLE_PREFIX_EXPAND_KEYS):
        if key not in state or key not in current:
            continue
        source = state[key] if isinstance(state[key], torch.Tensor) else torch.as_tensor(state[key])
        target = current[key]
        if source.ndim != target.ndim:
            continue
        if source.shape[0] > target.shape[0]:
            continue
        if tuple(source.shape[1:]) != tuple(target.shape[1:]):
            continue
        if tuple(source.shape) == tuple(target.shape):
            continue
        merged = target.detach().clone()
        merged[: source.shape[0]] = source.to(device=merged.device, dtype=merged.dtype)
        expanded[key] = merged
        expanded_keys.append(key)
    return expanded, expanded_keys


def _compatible_state(model: Any, state: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, dict[str, list[int]]]]:
    current = model.state_dict() if hasattr(model, "state_dict") else {}
    compatible: dict[str, Any] = {}
    skipped: list[str] = []
    shape_mismatched: dict[str, dict[str, list[int]]] = {}
    for key, value in state.items():
        key = str(key)
        if key not in current:
            skipped.append(key)
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tuple(current[key].shape) != tuple(tensor.shape):
            shape_mismatched[key] = {"checkpoint": list(tensor.shape), "model": list(current[key].shape)}
            skipped.append(key)
            continue
        compatible[key] = tensor
    return compatible, sorted(skipped), shape_mismatched


def _tensor_from_state(state: dict[str, Any], key: str) -> torch.Tensor | None:
    value = state.get(key)
    if value is None:
        return None
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _allclose_scalar(value: torch.Tensor, target: float, *, atol: float = 1.0e-6) -> bool:
    if value.numel() != 1:
        return False
    target_tensor = torch.tensor([float(target)], device=value.device, dtype=value.dtype)
    return bool(torch.allclose(value.reshape(1), target_tensor, atol=float(atol), rtol=0.0))


def _seed_default_belief_calibration_from_retrieval(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "skill_table.logit_scale_retr",
        "skill_table.skill_bias_retr",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    }
    missing = sorted(key for key in required if key not in state)
    if missing:
        return {"applied": False, "reason": "missing_keys", "missing_keys": missing}, state

    retr_scale = _tensor_from_state(state, "skill_table.logit_scale_retr")
    retr_bias = _tensor_from_state(state, "skill_table.skill_bias_retr")
    belief_scale = _tensor_from_state(state, "skill_table.logit_scale_belief")
    belief_bias = _tensor_from_state(state, "skill_table.skill_bias_belief")
    if retr_scale is None or retr_bias is None or belief_scale is None or belief_bias is None:
        return {"applied": False, "reason": "invalid_tensors"}, state
    if tuple(retr_scale.shape) != tuple(belief_scale.shape) or tuple(retr_bias.shape) != tuple(belief_bias.shape):
        return {
            "applied": False,
            "reason": "shape_mismatch",
            "retrieval_scale_shape": list(retr_scale.shape),
            "belief_scale_shape": list(belief_scale.shape),
            "retrieval_bias_shape": list(retr_bias.shape),
            "belief_bias_shape": list(belief_bias.shape),
        }, state

    belief_scale_default = _allclose_scalar(belief_scale.detach(), _DEFAULT_LOGIT_SCALE_BELIEF)
    belief_bias_default = bool(torch.allclose(belief_bias.detach().float(), torch.zeros_like(belief_bias.detach().float()), atol=1.0e-8, rtol=0.0))
    if not (belief_scale_default and belief_bias_default):
        return {"applied": False, "reason": "belief_not_default"}, state

    retr_scale_default = _allclose_scalar(retr_scale.detach(), _DEFAULT_LOGIT_SCALE_RETR)
    retr_bias_default = bool(torch.allclose(retr_bias.detach().float(), torch.zeros_like(retr_bias.detach().float()), atol=1.0e-8, rtol=0.0))
    if retr_scale_default and retr_bias_default:
        return {"applied": False, "reason": "retrieval_not_trained"}, state

    seeded = dict(state)
    seeded["skill_table.logit_scale_belief"] = retr_scale.detach().clone()
    seeded["skill_table.skill_bias_belief"] = retr_bias.detach().clone()
    return {
        "applied": True,
        "reason": "belief_was_default",
        "source": "skill_table.retrieval_calibration",
        "target": "skill_table.belief_calibration",
    }, seeded


def _first_module_device_dtype(module: torch.nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        return tensor.device, tensor.dtype if tensor.is_floating_point() else None
    return None, None


def _infer_gated_temporal_config_from_state(state: dict[str, Any], existing: Any) -> tuple[GatedTemporalConfig | None, str]:
    prefix = "gated_temporal_reranker."
    if not any(str(key).startswith(prefix) for key in state):
        return None, "no_gated_temporal_keys"
    residual_key = prefix + "residual_adapter.0.weight"
    residual = state.get(residual_key)
    if residual is None:
        return None, "missing_residual_adapter_0_weight"
    residual_tensor = residual if isinstance(residual, torch.Tensor) else torch.as_tensor(residual)
    if residual_tensor.ndim != 2:
        return None, "invalid_residual_adapter_0_weight_rank"
    hidden_dim = int(residual_tensor.shape[0])
    residual_input_dim = int(residual_tensor.shape[1])
    if residual_input_dim <= 1 or (residual_input_dim - 1) % 3 != 0:
        return None, "invalid_residual_adapter_input_dim"
    dim = (residual_input_dim - 1) // 3

    query_weight = state.get(prefix + "query_proj.weight")
    if query_weight is None:
        query_dim = dim
    else:
        query_tensor = query_weight if isinstance(query_weight, torch.Tensor) else torch.as_tensor(query_weight)
        if query_tensor.ndim != 2 or int(query_tensor.shape[0]) != dim:
            return None, "invalid_query_proj_weight_shape"
        query_dim = int(query_tensor.shape[1])

    gate_weight = state.get(prefix + "gate.weight")
    if gate_weight is None:
        gate_feature_dim = int(getattr(getattr(existing, "config", None), "gate_feature_dim", 4) or 4)
    else:
        gate_tensor = gate_weight if isinstance(gate_weight, torch.Tensor) else torch.as_tensor(gate_weight)
        if gate_tensor.ndim != 2 or int(gate_tensor.shape[0]) != 1:
            return None, "invalid_gate_weight_shape"
        gate_feature_dim = int(gate_tensor.shape[1])

    old_config = getattr(existing, "config", None)
    return (
        GatedTemporalConfig(
            dim=int(dim),
            query_dim=int(query_dim),
            lambda_max=float(getattr(old_config, "lambda_max", 0.5) or 0.5),
            context_top_k=int(getattr(old_config, "context_top_k", 64) or 64),
            gate_feature_dim=int(gate_feature_dim),
            hidden_dim=int(hidden_dim),
        ),
        "ok",
    )


def _maybe_reconfigure_gated_temporal_reranker(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    existing = getattr(model, "gated_temporal_reranker", None)
    config, reason = _infer_gated_temporal_config_from_state(state, existing)
    if config is None:
        return {"reconfigured": False, "reason": reason}
    if existing is None:
        return {"reconfigured": False, "reason": "model_has_no_gated_temporal_reranker"}

    existing_config = getattr(existing, "config", None)
    existing_query_dim = int(
        getattr(existing_config, "query_dim", 0) or getattr(existing_config, "dim", 0) or 0
    )
    already_matches = (
        int(getattr(existing_config, "dim", 0) or 0) == int(config.dim)
        and existing_query_dim == int(config.query_dim or config.dim)
        and int(getattr(existing_config, "hidden_dim", 0) or 0) == int(config.hidden_dim or config.dim)
        and int(getattr(existing_config, "gate_feature_dim", 0) or 0) == int(config.gate_feature_dim)
    )
    if already_matches:
        return {
            "reconfigured": False,
            "reason": "already_matches_checkpoint_shapes",
            "dim": int(config.dim),
            "query_dim": int(config.query_dim or config.dim),
            "hidden_dim": int(config.hidden_dim or config.dim),
            "gate_feature_dim": int(config.gate_feature_dim),
        }

    device, dtype = _first_module_device_dtype(existing)
    reranker = GatedTemporalReranker(config)
    if device is not None:
        if dtype is not None:
            reranker = reranker.to(device=device, dtype=dtype)
        else:
            reranker = reranker.to(device=device)
    setattr(model, "gated_temporal_reranker", reranker)
    return {
        "reconfigured": True,
        "reason": "checkpoint_shapes",
        "dim": int(config.dim),
        "query_dim": int(config.query_dim or config.dim),
        "hidden_dim": int(config.hidden_dim or config.dim),
        "gate_feature_dim": int(config.gate_feature_dim),
    }


def _maybe_reconfigure_stage4_score_calibrator(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    weight = state.get("stage4_score_calibrator.weight")
    if not isinstance(weight, torch.Tensor):
        return {"reconfigured": False, "reason": "missing_checkpoint_weight"}
    if weight.ndim != 2 or int(weight.size(0)) != 1 or int(weight.size(1)) <= 0:
        return {
            "reconfigured": False,
            "reason": "invalid_checkpoint_weight_shape",
            "shape": list(weight.shape),
        }
    existing = getattr(model, "stage4_score_calibrator", None)
    if isinstance(existing, torch.nn.Linear) and int(existing.in_features) == int(weight.size(1)) and int(existing.out_features) == 1:
        return {
            "reconfigured": False,
            "reason": "already_compatible",
            "feature_count": int(weight.size(1)),
        }
    device, dtype = _first_module_device_dtype(model)
    calibrator = torch.nn.Linear(int(weight.size(1)), 1, bias=False)
    if device is not None:
        if dtype is not None and dtype.is_floating_point:
            calibrator = calibrator.to(device=device, dtype=dtype)
        else:
            calibrator = calibrator.to(device=device)
    setattr(model, "stage4_score_calibrator", calibrator)
    return {
        "reconfigured": True,
        "reason": "checkpoint_weight_shape",
        "feature_count": int(weight.size(1)),
    }


def load_compatible_state_dict(
    model: Any,
    state: dict[str, Any],
    *,
    partial_load_mode: str,
    allow_skill_table_prefix_expansion: bool = False,
) -> dict[str, Any]:
    expanded_keys: list[str] = []
    if allow_skill_table_prefix_expansion:
        state, expanded_keys = _prefix_expand_skill_table_state(model, state)
    belief_seed_report, state = _seed_default_belief_calibration_from_retrieval(state)
    gated_temporal_report = _maybe_reconfigure_gated_temporal_reranker(model, state)
    stage4_score_calibrator_report = _maybe_reconfigure_stage4_score_calibrator(model, state)
    compatible, skipped, shape_mismatched = _compatible_state(model, state)
    result = model.load_state_dict(compatible, strict=False)
    missing_keys = sorted(getattr(result, "missing_keys", []) or [])
    gate_prefix = "route_memory_utility_gate."
    model_gate_keys = sorted(
        key for key in model.state_dict() if str(key).startswith(gate_prefix)
    )
    loaded_gate_keys = sorted(
        key for key in compatible if str(key).startswith(gate_prefix)
    )
    missing_gate_keys = sorted(
        key for key in missing_keys if str(key).startswith(gate_prefix)
    )
    if not model_gate_keys:
        gate_status = "not_present_in_model"
    elif len(loaded_gate_keys) == len(model_gate_keys):
        gate_status = "loaded"
    elif not loaded_gate_keys and missing_gate_keys == model_gate_keys:
        gate_status = "initialized_default"
    else:
        gate_status = "partial"
    return {
        "partial_load_mode": str(partial_load_mode),
        "allow_skill_table_prefix_expansion": bool(allow_skill_table_prefix_expansion),
        "skill_table_prefix_expanded_keys": expanded_keys,
        "belief_calibration_seeded_from_retrieval": belief_seed_report,
        "gated_temporal_reranker_reconfigured": gated_temporal_report,
        "stage4_score_calibrator_reconfigured": stage4_score_calibrator_report,
        "loaded": bool(compatible),
        "loaded_keys": sorted(compatible),
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(getattr(result, "unexpected_keys", []) or []),
        "skipped_keys": skipped,
        "shape_mismatched": shape_mismatched,
        "loaded_key_count": len(compatible),
        "route_memory_utility_gate_compatibility": {
            "status": gate_status,
            "loaded_keys": loaded_gate_keys,
            "missing_keys": missing_gate_keys,
            "model_keys": model_gate_keys,
        },
    }


def build_clstr_model_from_stage0_checkpoint(
    checkpoint_path: str | Path,
    skills_path: str | Path,
    model_cache_dir: str | Path | None = None,
    *,
    allow_skill_table_prefix_expansion: bool = False,
    base_model_name_override: str | Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    payload = checkpoint_payload(checkpoint_path, "stage0")
    config = _stage0_config_from_checkpoint(payload)
    checkpoint_base_model_name = str(config["base_model_name"])
    if base_model_name_override is not None:
        resolved_override = str(base_model_name_override).strip()
        if not resolved_override:
            raise ValueError("Stage0 base model override must be nonempty")
        config["base_model_name"] = resolved_override
        config["local_files_only"] = True
    skills = _read_jsonl(skills_path)
    selected_skills_path = None
    if model_cache_dir is not None:
        selected_skills_path = _write_jsonl(Path(model_cache_dir) / "selected_stage0_skills.jsonl", skills)
    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(config["base_model_name"]),
            d=int(config["d"]),
            d_a=int(config["d_a"]),
            top_k=int(config["top_k"]),
            encoder_pooling=str(config["encoder_pooling"]),
            cross_encoder_pooling=str(config["cross_encoder_pooling"]),
            tokenizer_padding_side=config["tokenizer_padding_side"],
            torch_dtype=config["torch_dtype"],
            freeze_backbone=bool(config["freeze_backbone"]),
            max_length=config["max_length"],
            projection_init=str(config["projection_init"]),
            normalize_embeddings=bool(config["normalize_embeddings"]),
            hf_cache_dir=config.get("hf_cache_dir"),
            local_files_only=bool(config.get("local_files_only", False)),
            defer_skill_table_init=True,
            skill_text_format=str(config["skill_text_format"]),
            state_query_prompt_version=str(config["state_query_prompt_version"]),
            state_query_max_chars=config["state_query_max_chars"],
            state_query_truncation=str(config["state_query_truncation"]),
            skill_table_batch_size=int(config["skill_table_batch_size"]),
            skill_table_adapter_init=str(config["skill_table_adapter_init"]),
            use_cross_encoder=bool(config["use_cross_encoder"]),
        ),
        skills,
    )
    load_report = load_compatible_state_dict(
        model,
        checkpoint_state(payload, "stage0"),
        partial_load_mode="stage0_checkpoint_compatible_state",
        allow_skill_table_prefix_expansion=allow_skill_table_prefix_expansion,
    )
    local_skill_table_rebuilt = False
    local_skill_table_rebuild_reason = ""
    if "skill_table.E" not in set(load_report["loaded_keys"]) and hasattr(model, "rebuild_skill_table"):
        model.rebuild_skill_table()
        local_skill_table_rebuilt = True
        local_skill_table_rebuild_reason = "skill_table.E_not_loaded"
    report = {
        "stage0_checkpoint_path": str(checkpoint_path),
        "stage0_checkpoint_stage": payload.get("stage"),
        "stage0_checkpoint_base_model_name": checkpoint_base_model_name,
        "stage0_effective_base_model_name": str(config["base_model_name"]),
        "stage0_base_model_name_overridden": bool(base_model_name_override is not None),
        "stage0_loaded": bool(load_report["loaded"]),
        "stage0_loaded_keys": list(load_report["loaded_keys"]),
        "stage0_skipped_keys": list(load_report["skipped_keys"]),
        "stage0_shape_mismatched": dict(load_report["shape_mismatched"]),
        "stage0_skill_table_prefix_expanded_keys": list(load_report.get("skill_table_prefix_expanded_keys") or []),
        "stage0_belief_calibration_seeded_from_retrieval": dict(
            load_report.get("belief_calibration_seeded_from_retrieval") or {}
        ),
        "selected_skills_path": None if selected_skills_path is None else str(selected_skills_path),
        "skill_count": len(skills),
        "stage0_config": config,
        "local_skill_table_rebuilt": local_skill_table_rebuilt,
        "local_skill_table_rebuild_reason": local_skill_table_rebuild_reason,
    }
    return model, config, report


def merge_routing_and_head_state(
    routing_payload: dict[str, Any],
    head_payload: dict[str, Any],
    *,
    protect_routing_foundation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    routing_state = checkpoint_state(routing_payload, "routing")
    head_state = checkpoint_state(head_payload, "head")
    protected_prefixes = ("encoder.", "cross_encoder.", "skill_table.") if protect_routing_foundation else ()
    skipped_head_routing_foundation_keys = sorted(
        key for key in head_state if _is_protected_routing_foundation_key(str(key), protected_prefixes)
    )
    mergeable_head_state = {
        key: value
        for key, value in head_state.items()
        if key not in skipped_head_routing_foundation_keys
    }
    merged: dict[str, Any] = {}
    merged.update(routing_state)
    merged.update(mergeable_head_state)
    return merged, {
        "routing_checkpoint_stage": routing_payload.get("stage"),
        "head_checkpoint_stage": head_payload.get("stage"),
        "routing_key_count": len(routing_state),
        "head_key_count": len(head_state),
        "merged_key_count": len(merged),
        "head_overrides_routing_keys": sorted(set(routing_state) & set(mergeable_head_state)),
        "protect_routing_foundation": bool(protect_routing_foundation),
        "protected_routing_foundation_prefixes": list(protected_prefixes),
        "routing_foundation_head_allowlist": sorted(_ROUTING_FOUNDATION_HEAD_ALLOWLIST),
        "skipped_head_routing_foundation_keys": skipped_head_routing_foundation_keys,
    }


def load_routing_and_head_checkpoints(
    model: Any,
    routing_checkpoint_path: str | Path,
    head_checkpoint_path: str | Path,
    partial_load_mode: str,
    *,
    protect_routing_foundation: bool = False,
    allow_skill_table_prefix_expansion: bool = False,
) -> dict[str, Any]:
    routing_payload = checkpoint_payload(routing_checkpoint_path, "routing")
    head_payload = checkpoint_payload(head_checkpoint_path, "head")
    merged_state, merge_report = merge_routing_and_head_state(
        routing_payload,
        head_payload,
        protect_routing_foundation=protect_routing_foundation,
    )
    load_report = load_compatible_state_dict(
        model,
        merged_state,
        partial_load_mode=partial_load_mode,
        allow_skill_table_prefix_expansion=allow_skill_table_prefix_expansion,
    )
    return {
        "routing_checkpoint_path": str(routing_checkpoint_path),
        "head_checkpoint_path": str(head_checkpoint_path),
        **merge_report,
        **load_report,
    }


def load_head_checkpoint_into_model(
    model: Any,
    head_checkpoint_path: str | Path,
    partial_load_mode: str = "stage1_checkpoint_compatible_head_state",
    *,
    protect_routing_foundation: bool = True,
    allow_skill_table_prefix_expansion: bool = False,
) -> dict[str, Any]:
    head_payload = checkpoint_payload(head_checkpoint_path, "head")
    head_state = checkpoint_state(head_payload, "head")
    protected_prefixes = ("encoder.", "cross_encoder.", "skill_table.") if protect_routing_foundation else ()
    skipped_head_routing_foundation_keys = sorted(
        key for key in head_state if _is_protected_routing_foundation_key(str(key), protected_prefixes)
    )
    head_state = {
        key: value
        for key, value in head_state.items()
        if key not in skipped_head_routing_foundation_keys
    }
    load_report = load_compatible_state_dict(
        model,
        head_state,
        partial_load_mode=partial_load_mode,
        allow_skill_table_prefix_expansion=allow_skill_table_prefix_expansion,
    )
    return {
        "head_checkpoint_path": str(head_checkpoint_path),
        "head_checkpoint_stage": head_payload.get("stage"),
        "head_checkpoint_step": head_payload.get("step"),
        "protect_routing_foundation": bool(protect_routing_foundation),
        "protected_routing_foundation_prefixes": list(protected_prefixes),
        "routing_foundation_head_allowlist": sorted(_ROUTING_FOUNDATION_HEAD_ALLOWLIST),
        "skipped_head_routing_foundation_keys": skipped_head_routing_foundation_keys,
        **load_report,
    }
