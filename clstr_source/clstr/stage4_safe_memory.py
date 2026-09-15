from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from clstr.candidate_admission_residual import CANDIDATE_ADMISSION_RESIDUAL_V1


IMMUTABLE_ROUTER_PREFIXES = (
    "encoder.",
    "skill_table.",
    "initial_belief_head.",
    "unified_retriever.",
)
FAST_ROUTER_PREFIXES = (
    "encoder.proj.",
    "initial_belief_head.",
    "unified_retriever.",
)
FAST_ROUTER_EXACT_KEYS = {
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
}
STAGE4_DELTA_PREFIXES = (
    "transition.",
    "gate.",
    "action_proj.",
    "route_memory_utility_gate.",
)
STAGE4_TRAINABLE_MODULES = (
    "transition",
    "gate",
    "action_proj",
    "route_memory_utility_gate",
)
CMC_STAGE4_DELTA_PREFIXES = (
    "route_memory_residual_adapter.",
    "route_memory_candidate_utility_gate.",
)
CMC_STAGE4_TRAINABLE_MODULES = (
    "route_memory_residual_adapter",
    "route_memory_candidate_utility_gate",
)
CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES = (
    "route_memory_residual_adapter.",
    "route_memory_candidate_admission_residual.",
)
CANDIDATE_ADMISSION_STAGE4_TRAINABLE_MODULES = (
    "route_memory_candidate_admission_residual",
)


def _router_tensor_selected(name: str, *, scope: str) -> bool:
    if scope not in {"full", "fast"}:
        raise ValueError("router digest scope must be full or fast")
    if scope == "full":
        return any(name.startswith(prefix) for prefix in IMMUTABLE_ROUTER_PREFIXES)
    return (
        any(name.startswith(prefix) for prefix in FAST_ROUTER_PREFIXES)
        or name in FAST_ROUTER_EXACT_KEYS
    )


def router_state_digest(model: Any, *, scope: str = "full") -> str:
    digest = hashlib.sha256()
    selected_count = 0
    for name, tensor in sorted(model.state_dict().items()):
        if not _router_tensor_selected(name, scope=scope):
            continue
        selected_count += 1
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        byte_view = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    if selected_count == 0:
        raise ValueError("immutable router digest selected no tensors")
    return digest.hexdigest()


def set_stage4_safe_training_mode(model: Any) -> None:
    model.eval()
    for name in STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"safe-memory Stage4 requires model.{name}")
        module.train()


def freeze_stage4_safe_memory(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules: list[str] = []
    for name in STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"safe-memory Stage4 requires model.{name}")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        trainable_modules.append(name)
    set_stage4_safe_training_mode(model)
    optimizer_parameter_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return {
        "frozen_routing_foundation": True,
        "immutable_stage2_router": True,
        "immutable_router_eval_mode": True,
        "frozen_belief_gate": False,
        "train_transition": True,
        "trainable_modules": trainable_modules,
        "optimizer_parameter_names": optimizer_parameter_names,
        "route_scorer": "unified_memory",
        "full_router_digest": router_state_digest(model, scope="full"),
        "fast_router_digest": router_state_digest(model, scope="fast"),
    }


def set_stage4_cmc_training_mode(model: Any) -> None:
    model.eval()
    for name in CMC_STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"CMC Stage4 requires model.{name}")
        module.train()


def _reset_stage4_cmc_delta(model: Any, *, initial_alpha: float = 0.01) -> None:
    adapter = getattr(model, "route_memory_residual_adapter", None)
    gate = getattr(model, "route_memory_candidate_utility_gate", None)
    if adapter is None or gate is None:
        raise ValueError("CMC Stage4 delta modules are missing")
    reset_adapter = getattr(adapter, "reset_parameters", None)
    if callable(reset_adapter):
        reset_adapter()
    linear = None
    net = getattr(gate, "net", None)
    if isinstance(net, torch.nn.Sequential) and len(net) > 0:
        linear = net[0]
    if not isinstance(linear, torch.nn.Linear) or int(linear.out_features) != 1:
        raise ValueError("CMC candidate utility gate has an unsupported architecture")
    initial_alpha = float(initial_alpha)
    if not 0.0 < initial_alpha < 1.0:
        raise ValueError("CMC initial alpha must be strictly between zero and one")
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.constant_(
        linear.bias,
        float(torch.logit(torch.tensor(initial_alpha)).item()),
    )
    feature_mean = getattr(gate, "feature_mean", None)
    feature_scale = getattr(gate, "feature_scale", None)
    if isinstance(feature_mean, torch.Tensor):
        feature_mean.zero_()
    if isinstance(feature_scale, torch.Tensor):
        feature_scale.fill_(1.0)


def freeze_stage4_cmc(
    model: Any,
    *,
    reset_delta: bool = True,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if reset_delta:
        _reset_stage4_cmc_delta(model)
    trainable_modules: list[str] = []
    for name in CMC_STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"CMC Stage4 requires model.{name}")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        trainable_modules.append(name)
    set_stage4_cmc_training_mode(model)
    optimizer_parameter_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    full_digest = router_state_digest(model, scope="full")
    fast_digest = router_state_digest(model, scope="fast")
    return {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "frozen_routing_foundation": True,
        "immutable_stage2_router": True,
        "immutable_router_eval_mode": True,
        "frozen_belief_gate": True,
        "train_transition": False,
        "trainable_modules": trainable_modules,
        "optimizer_parameter_names": optimizer_parameter_names,
        "route_scorer": "unified_memory",
        "full_router_digest": full_digest,
        "fast_router_digest": fast_digest,
        "parent_stage2_full_router_digest": full_digest,
        "parent_stage2_fast_router_digest": fast_digest,
    }


def set_stage4_candidate_admission_training_mode(model: Any) -> None:
    model.eval()
    for name in CANDIDATE_ADMISSION_STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"candidate-admission Stage4 requires model.{name}")
        module.train()


def freeze_stage4_candidate_admission(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules: list[str] = []
    for name in CANDIDATE_ADMISSION_STAGE4_TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"candidate-admission Stage4 requires model.{name}")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        trainable_modules.append(name)
    set_stage4_candidate_admission_training_mode(model)
    optimizer_parameter_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    full_digest = router_state_digest(model, scope="full")
    fast_digest = router_state_digest(model, scope="fast")
    return {
        "stage4_method": CANDIDATE_ADMISSION_RESIDUAL_V1,
        "frozen_routing_foundation": True,
        "immutable_stage2_router": True,
        "immutable_router_eval_mode": True,
        "frozen_belief_gate": True,
        "frozen_cmc_residual_adapter": True,
        "train_transition": False,
        "trainable_modules": trainable_modules,
        "optimizer_parameter_names": optimizer_parameter_names,
        "route_scorer": "candidate_admission_residual",
        "full_router_digest": full_digest,
        "fast_router_digest": fast_digest,
        "parent_stage2_full_router_digest": full_digest,
        "parent_stage2_fast_router_digest": fast_digest,
    }


def stage4_delta_state_dict(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(STAGE4_DELTA_PREFIXES)
    }
    validate_stage4_delta_state_dict(state)
    return state


def stage4_cmc_delta_state_dict(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(CMC_STAGE4_DELTA_PREFIXES)
    }
    validate_stage4_cmc_delta_state_dict(state)
    return state


def stage4_candidate_admission_delta_state_dict(
    model: Any,
) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith(CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES)
    }
    validate_stage4_candidate_admission_delta_state_dict(state)
    return state


def validate_stage4_candidate_admission_delta_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError("candidate-admission delta state must be a nonempty dict")
    forbidden = sorted(
        name
        for name in state
        if not str(name).startswith(CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES)
    )
    if forbidden:
        raise ValueError(
            f"forbidden candidate-admission delta key: {forbidden[0]}"
        )
    missing = [
        prefix
        for prefix in CANDIDATE_ADMISSION_STAGE4_DELTA_PREFIXES
        if not any(str(name).startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(
            f"candidate-admission delta missing prefix: {missing[0]}"
        )
    return {
        "status": "ok",
        "stage4_method": CANDIDATE_ADMISSION_RESIDUAL_V1,
        "state_key_count": len(state),
        "state_keys": sorted(state),
    }


def validate_stage4_cmc_delta_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError("CMC delta state must be a nonempty dict")
    forbidden = sorted(
        name
        for name in state
        if not str(name).startswith(CMC_STAGE4_DELTA_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"forbidden CMC delta key: {forbidden[0]}")
    missing = [
        prefix
        for prefix in CMC_STAGE4_DELTA_PREFIXES
        if not any(str(name).startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(f"CMC delta state missing prefix: {missing[0]}")
    return {
        "status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "state_key_count": len(state),
        "state_keys": sorted(state),
    }


def validate_stage4_checkpoint_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate a compact Stage4 checkpoint using its declared method."""

    if not isinstance(payload, dict):
        raise TypeError("Stage4 checkpoint payload must be a dict")
    state = payload.get("model_state_dict") or {}
    if not isinstance(state, dict):
        raise TypeError("Stage4 checkpoint model_state_dict must be a dict")
    if payload.get("stage4_method") == "counterfactual_memory_calibration_v1":
        return validate_stage4_cmc_delta_state_dict(state)
    if payload.get("stage4_method") == CANDIDATE_ADMISSION_RESIDUAL_V1:
        for field in (
            "base_cmc_checkpoint_sha256",
            "parent_stage2_full_router_digest",
            "parent_stage2_fast_router_digest",
        ):
            if not str(payload.get(field) or ""):
                raise ValueError(
                    f"candidate-admission checkpoint missing {field}"
                )
        return validate_stage4_candidate_admission_delta_state_dict(state)
    return validate_stage4_delta_state_dict(state)


def validate_stage4_candidate_admission_parent_cmc(
    payload: dict[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("candidate-admission checkpoint payload must be a dict")
    recorded = str(payload.get("base_cmc_checkpoint_sha256") or "")
    if recorded != str(expected_sha256):
        raise ValueError("candidate-admission base CMC identity mismatch")
    return {
        "status": "ok",
        "base_cmc_checkpoint_sha256": recorded,
    }


def validate_stage4_cmc_parent_router(
    payload: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("CMC checkpoint payload must be a dict")
    expected_full = router_state_digest(model, scope="full")
    expected_fast = router_state_digest(model, scope="fast")
    recorded_full = str(payload.get("parent_stage2_full_router_digest") or "")
    recorded_fast = str(payload.get("parent_stage2_fast_router_digest") or "")
    if recorded_full != expected_full or recorded_fast != expected_fast:
        raise ValueError("CMC parent Stage2 router digest mismatch")
    return {
        "status": "ok",
        "parent_stage2_full_router_digest": expected_full,
        "parent_stage2_fast_router_digest": expected_fast,
    }


def validate_stage4_delta_state_dict(
    state: dict[str, Any],
    *,
    require_route_memory_utility_gate: bool = False,
) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError("Stage4 delta state must be a nonempty dict")
    forbidden = sorted(
        name
        for name in state
        if not str(name).startswith(STAGE4_DELTA_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"forbidden Stage4 delta key: {forbidden[0]}")
    required_prefixes = (
        STAGE4_DELTA_PREFIXES
        if require_route_memory_utility_gate
        else STAGE4_DELTA_PREFIXES[:-1]
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(str(name).startswith(prefix) for name in state)
    ]
    if missing:
        raise ValueError(f"Stage4 delta state missing prefix: {missing[0]}")
    return {
        "status": "ok",
        "state_key_count": len(state),
        "state_keys": sorted(state),
        "route_memory_utility_gate_present": any(
            str(name).startswith("route_memory_utility_gate.") for name in state
        ),
    }
