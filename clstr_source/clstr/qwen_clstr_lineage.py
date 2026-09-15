from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from clstr.stage4_safe_memory import (
    validate_stage4_candidate_admission_delta_state_dict,
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_delta_state_dict,
)
from clstr.state_query_prompt import resolve_state_query_prompt_contract


SCHEMA_VERSION = "qwen06_clstr_lineage_v1"
QWEN_MODEL_BASENAME = "Qwen3-Embedding-0.6B"
DERIVED_ALLOWED_ROUTING_CALIBRATION_KEYS = {
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
}
CMC_STAGE4_METHOD = "counterfactual_memory_calibration_v1"
CANDIDATE_ADMISSION_STAGE4_METHOD = "candidate_admission_residual_v1"
CMC_FEATURE_SCHEMA = "memory_utility_features_v1"
CMC_FEATURE_UPDATE_COUNT_CAP = 16.0
CMC_FEATURE_CANDIDATE_COUNT_CAP = 256.0


def sha256_path(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    files = (
        [resolved]
        if resolved.is_file()
        else sorted(item for item in resolved.rglob("*") if item.is_file())
    )
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.name if resolved.is_file() else str(item.relative_to(resolved))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size": size,
        "file_count": len(files),
    }


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _require_qwen_model_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.name != QWEN_MODEL_BASENAME:
        raise ValueError("Qwen backbone identity mismatch")
    return resolved


def _expected_prompt_contract() -> dict[str, Any]:
    return resolve_state_query_prompt_contract(
        prompt_version="clstr_causal_state_v1",
        max_chars=2000,
        truncation="head_tail_v1",
    )


def validate_qwen_clstr_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_stage: str,
    expected_model_path: str | Path,
    require_unified_memory: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
    config = dict(payload.get("config") or payload.get("model_config") or {})
    protocol = dict(payload.get("stage0_protocol") or payload.get("protocol_metadata") or {})
    expected_model_path = _require_qwen_model_path(expected_model_path)
    model_path = str(config.get("base_model_name") or "")
    if not model_path or Path(model_path).resolve() != expected_model_path:
        raise ValueError("Qwen backbone identity mismatch")
    if config.get("freeze_backbone") is not True:
        raise ValueError("Qwen backbone must remain frozen")
    trainable = dict(payload.get("trainable_parameter_policy") or {})
    if trainable.get("trains_encoder_backbone") is True:
        raise ValueError("Qwen backbone must remain frozen")
    route_scorer = str(
        payload.get("route_scorer")
        or config.get("route_scorer")
        or protocol.get("route_scorer")
        or ""
    )
    if require_unified_memory and route_scorer != "unified_memory":
        raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
    state = payload.get("model_state_dict") or {}
    if not isinstance(state, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    if any(str(key).startswith("encoder.backbone.") for key in state):
        raise ValueError("frozen encoder backbone unexpectedly stored in checkpoint")
    if str(payload.get("stage") or "") != str(expected_stage):
        raise ValueError("checkpoint stage mismatch")
    prompt_contract = resolve_state_query_prompt_contract(
        prompt_version=str(config.get("state_query_prompt_version") or ""),
        max_chars=config.get("state_query_max_chars"),
        truncation=config.get("state_query_truncation"),
        recorded_instruction=config.get("state_query_instruction"),
    )
    if prompt_contract != _expected_prompt_contract():
        raise ValueError("state query prompt contract mismatch")
    return {
        "stage": str(expected_stage),
        "route_scorer": route_scorer,
        "backbone_frozen": True,
        **prompt_contract,
        "checkpoint": sha256_path(checkpoint_path),
    }


def validate_qwen_clstr_derived_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_stage: str,
    require_unified_memory: bool = True,
    allow_selected_intermediate: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
    if str(payload.get("stage") or "") != str(expected_stage):
        raise ValueError("checkpoint stage mismatch")
    train_report = payload.get("train_report") or {}
    if not isinstance(train_report, dict):
        raise TypeError("derived checkpoint train_report must be a dict")
    train_status = str(train_report.get("status") or "")
    selected_intermediate_ok = (
        allow_selected_intermediate
        and expected_stage == "clstr_stage4_transition_conditioned_next_skill"
        and train_status == "running"
    )
    if train_status != "ok" and not selected_intermediate_ok:
        raise ValueError("derived checkpoint train report must be ok")
    if train_report.get("valid_or_test_used_for_training") is not False:
        raise ValueError("derived checkpoint must use training-only data")
    route_scorer = str(payload.get("route_scorer") or train_report.get("route_scorer") or "")
    if require_unified_memory and route_scorer != "unified_memory":
        raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
    if payload.get("checkpoint_excludes_frozen_qwen_backbone") is not True:
        raise ValueError("derived checkpoint must exclude frozen Qwen backbone")
    if payload.get("checkpoint_excludes_frozen_routing_foundation") is not True:
        raise ValueError("derived checkpoint must exclude frozen routing foundation")
    freeze_report = train_report.get("freeze_report") or {}
    if not isinstance(freeze_report, dict) or freeze_report.get("frozen_routing_foundation") is not True:
        raise ValueError("derived checkpoint must record frozen routing foundation")
    state = payload.get("model_state_dict") or {}
    if not isinstance(state, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    stage_identity: dict[str, Any] = {}
    if expected_stage == "clstr_stage4_transition_conditioned_next_skill":
        stage4_method = str(
            payload.get("stage4_method")
            or train_report.get("stage4_method")
            or ""
        )
        if stage4_method == CANDIDATE_ADMISSION_STAGE4_METHOD:
            if train_report.get("validation_used_for_checkpoint_selection") is not True:
                raise ValueError(
                    "candidate-admission checkpoint must be validation selected"
                )
            if payload.get("stage4_method") != CANDIDATE_ADMISSION_STAGE4_METHOD:
                raise ValueError("candidate-admission checkpoint method mismatch")
            if train_report.get("stage4_method") != CANDIDATE_ADMISSION_STAGE4_METHOD:
                raise ValueError("candidate-admission train method mismatch")
            base_cmc_sha = str(
                payload.get("base_cmc_checkpoint_sha256") or ""
            )
            if len(base_cmc_sha) != 64 or base_cmc_sha != str(
                train_report.get("base_cmc_checkpoint_sha256") or ""
            ):
                raise ValueError("candidate-admission base CMC identity mismatch")
            parent_full = str(
                payload.get("parent_stage2_full_router_digest") or ""
            )
            parent_fast = str(
                payload.get("parent_stage2_fast_router_digest") or ""
            )
            if (
                freeze_report.get("parent_stage2_full_router_digest")
                != parent_full
                or freeze_report.get("parent_stage2_fast_router_digest")
                != parent_fast
            ):
                raise ValueError(
                    "candidate-admission Stage2 parent router digest mismatch"
                )
            if freeze_report.get("trainable_modules") != [
                "route_memory_candidate_admission_residual"
            ] or freeze_report.get("frozen_cmc_residual_adapter") is not True:
                raise ValueError("candidate-admission freeze contract mismatch")
            validate_stage4_candidate_admission_delta_state_dict(state)
            stage_identity = {
                "stage4_method": CANDIDATE_ADMISSION_STAGE4_METHOD,
                "base_cmc_checkpoint_sha256": base_cmc_sha,
                "parent_stage2_full_router_digest": parent_full,
                "parent_stage2_fast_router_digest": parent_fast,
            }
        elif stage4_method == CMC_STAGE4_METHOD:
            if (
                train_report.get("validation_used_for_checkpoint_selection")
                is not True
            ):
                raise ValueError(
                    "Stage4 derived checkpoint must be validation selected"
                )
            if payload.get("stage4_method") != CMC_STAGE4_METHOD:
                raise ValueError("CMC checkpoint stage4 method mismatch")
            if train_report.get("stage4_method") != CMC_STAGE4_METHOD:
                raise ValueError("CMC train report stage4 method mismatch")
            if train_report.get("feature_schema") != CMC_FEATURE_SCHEMA:
                raise ValueError("CMC feature schema mismatch")
            if (
                train_report.get("feature_update_count_cap")
                != CMC_FEATURE_UPDATE_COUNT_CAP
            ):
                raise ValueError("CMC update-count feature cap mismatch")
            if (
                train_report.get("feature_candidate_count_cap")
                != CMC_FEATURE_CANDIDATE_COUNT_CAP
            ):
                raise ValueError("CMC candidate-count feature cap mismatch")
            parent_full = str(
                payload.get("parent_stage2_full_router_digest") or ""
            )
            parent_fast = str(
                payload.get("parent_stage2_fast_router_digest") or ""
            )
            if not parent_full or not parent_fast:
                raise ValueError("CMC checkpoint lacks Stage2 parent router digests")
            if (
                freeze_report.get("parent_stage2_full_router_digest")
                != parent_full
            ):
                raise ValueError("CMC full Stage2 parent router digest mismatch")
            if (
                freeze_report.get("parent_stage2_fast_router_digest")
                != parent_fast
            ):
                raise ValueError("CMC fast Stage2 parent router digest mismatch")
            validate_stage4_cmc_delta_state_dict(state)
            stage_identity = {
                "stage4_method": CMC_STAGE4_METHOD,
                "feature_schema": CMC_FEATURE_SCHEMA,
                "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
                "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
                "parent_stage2_full_router_digest": parent_full,
                "parent_stage2_fast_router_digest": parent_fast,
            }
        elif (
            train_report.get("safe_memory_protocol_version")
            == "stage4_safe_memory_v1"
        ):
            if (
                train_report.get("validation_used_for_checkpoint_selection")
                is not True
            ):
                raise ValueError(
                    "Stage4 derived checkpoint must be validation selected"
                )
            validate_stage4_delta_state_dict(state)
        else:
            raise ValueError("Stage4 derived checkpoint lacks safe-memory protocol")
    else:
        for key in state:
            name = str(key)
            if name.startswith(("encoder.", "cross_encoder.")) or (
                name.startswith("skill_table.")
                and name not in DERIVED_ALLOWED_ROUTING_CALIBRATION_KEYS
            ):
                raise ValueError(
                    "frozen routing foundation unexpectedly stored in checkpoint"
                )
    return {
        "stage": str(expected_stage),
        "route_scorer": route_scorer,
        "backbone_frozen": True,
        "checkpoint": sha256_path(checkpoint_path),
        **stage_identity,
    }


def _lineage_parent_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    loaded = _load_json_object(resolved, label="parent lineage manifest")
    if loaded.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported parent lineage schema version")
    checkpoint = loaded.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ValueError("parent lineage manifest lacks checkpoint digest")
    return {
        "manifest_path": str(resolved),
        "manifest_sha256": sha256_path(resolved)["sha256"],
        "checkpoint_sha256": str(checkpoint["sha256"]),
    }


def create_lineage_manifest(
    *,
    stage: str,
    checkpoint_path: str | Path,
    expected_checkpoint_stage: str,
    model_path: str | Path,
    skill_pool_path: str | Path,
    data_manifest_path: str | Path,
    parent_manifests: dict[str, str | Path],
) -> dict[str, Any]:
    checkpoint_report = validate_qwen_clstr_checkpoint(
        checkpoint_path,
        expected_stage=expected_checkpoint_stage,
        expected_model_path=model_path,
    )
    model_path = _require_qwen_model_path(model_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "route_scorer": checkpoint_report["route_scorer"],
        "state_query": {
            "prompt_version": checkpoint_report["state_query_prompt_version"],
            "instruction": checkpoint_report["state_query_instruction"],
            "max_chars": checkpoint_report["state_query_max_chars"],
            "truncation": checkpoint_report["state_query_truncation"],
        },
        "backbone": {**sha256_path(model_path), "frozen": True},
        "skill_pool": sha256_path(skill_pool_path),
        "data_manifest": sha256_path(data_manifest_path),
        "checkpoint": checkpoint_report["checkpoint"],
        "parents": {
            str(role): _lineage_parent_record(path)
            for role, path in sorted(parent_manifests.items())
        },
    }


def _validated_parent_lineage(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = _load_json_object(resolved, label="parent lineage manifest")
    parents = payload.get("parents") or {}
    if not isinstance(parents, dict):
        raise ValueError("parent lineage parents must be an object")
    expected_parents: dict[str, Path] = {}
    for role, record in parents.items():
        if not isinstance(record, dict) or not record.get("manifest_path"):
            raise ValueError(f"parent lineage record missing: {role}")
        expected_parents[str(role)] = Path(str(record["manifest_path"]))
    validate_lineage_manifest(
        resolved,
        expected_stage=str(payload.get("stage") or ""),
        expected_parent_manifests=expected_parents,
    )
    return payload


def create_derived_lineage_manifest(
    *,
    stage: str,
    checkpoint_path: str | Path,
    expected_checkpoint_stage: str,
    model_path: str | Path,
    skill_pool_path: str | Path,
    data_manifest_path: str | Path,
    parent_manifests: dict[str, str | Path],
    identity_parent_role: str,
    stage_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity_parent_role = str(identity_parent_role)
    if identity_parent_role not in parent_manifests:
        raise ValueError("identity parent role is missing")
    parent_payloads = {
        str(role): _validated_parent_lineage(path)
        for role, path in sorted(parent_manifests.items())
    }
    identity_parent = parent_payloads[identity_parent_role]
    for role, parent in parent_payloads.items():
        for key in ("route_scorer", "state_query", "backbone", "skill_pool", "data_manifest"):
            if parent.get(key) != identity_parent.get(key):
                raise ValueError(f"parent lineage identity mismatch: {role}.{key}")
    if identity_parent.get("route_scorer") != "unified_memory":
        raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
    expected_state_query = {
        "prompt_version": _expected_prompt_contract()["state_query_prompt_version"],
        "instruction": _expected_prompt_contract()["state_query_instruction"],
        "max_chars": _expected_prompt_contract()["state_query_max_chars"],
        "truncation": _expected_prompt_contract()["state_query_truncation"],
    }
    if identity_parent.get("state_query") != expected_state_query:
        raise ValueError("state query prompt contract mismatch")
    model_identity = {**sha256_path(_require_qwen_model_path(model_path)), "frozen": True}
    skill_pool_identity = sha256_path(skill_pool_path)
    data_manifest_identity = sha256_path(data_manifest_path)
    if model_identity != identity_parent.get("backbone"):
        raise ValueError("derived backbone identity mismatch")
    if skill_pool_identity != identity_parent.get("skill_pool"):
        raise ValueError("derived skill pool identity mismatch")
    if data_manifest_identity != identity_parent.get("data_manifest"):
        raise ValueError("derived data manifest identity mismatch")
    checkpoint_report = validate_qwen_clstr_derived_checkpoint(
        checkpoint_path,
        expected_stage=expected_checkpoint_stage,
        allow_selected_intermediate=(
            expected_checkpoint_stage
            == "clstr_stage4_transition_conditioned_next_skill"
            and isinstance(stage_metadata, dict)
            and isinstance(stage_metadata.get("stage4_selection"), dict)
        ),
    )
    if stage_metadata is not None and not isinstance(stage_metadata, dict):
        raise TypeError("derived lineage stage_metadata must be a dict")
    resolved_stage_metadata = dict(stage_metadata or {})
    if checkpoint_report.get("stage4_method") == CMC_STAGE4_METHOD:
        cmc_identity = {
            key: checkpoint_report[key]
            for key in (
                "stage4_method",
                "feature_schema",
                "feature_update_count_cap",
                "feature_candidate_count_cap",
                "parent_stage2_full_router_digest",
                "parent_stage2_fast_router_digest",
            )
        }
        recorded_cmc = resolved_stage_metadata.get("cmc")
        if recorded_cmc is not None and recorded_cmc != cmc_identity:
            raise ValueError("derived lineage CMC identity mismatch")
        resolved_stage_metadata["cmc"] = cmc_identity
    if (
        checkpoint_report.get("stage4_method")
        == CANDIDATE_ADMISSION_STAGE4_METHOD
    ):
        candidate_identity = {
            key: checkpoint_report[key]
            for key in (
                "stage4_method",
                "base_cmc_checkpoint_sha256",
                "parent_stage2_full_router_digest",
                "parent_stage2_fast_router_digest",
            )
        }
        recorded = resolved_stage_metadata.get("candidate_admission")
        if recorded is not None and recorded != candidate_identity:
            raise ValueError("derived lineage candidate-admission identity mismatch")
        resolved_stage_metadata["candidate_admission"] = candidate_identity
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "route_scorer": checkpoint_report["route_scorer"],
        "state_query": expected_state_query,
        "backbone": model_identity,
        "skill_pool": skill_pool_identity,
        "data_manifest": data_manifest_identity,
        "checkpoint": checkpoint_report["checkpoint"],
        "parents": {
            str(role): _lineage_parent_record(path)
            for role, path in sorted(parent_manifests.items())
        },
        "identity_parent_role": identity_parent_role,
    }
    if resolved_stage_metadata:
        manifest["stage_metadata"] = resolved_stage_metadata
    return manifest


def _validate_recorded_identity(
    recorded: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(recorded, dict) or not recorded.get("path") or not recorded.get("sha256"):
        raise ValueError(f"{label} identity is missing")
    actual = sha256_path(recorded["path"])
    if actual["sha256"] != str(recorded["sha256"]):
        raise ValueError(f"{label} digest mismatch")
    if int(actual["size"]) != int(recorded.get("size", -1)):
        raise ValueError(f"{label} size mismatch")
    if int(actual["file_count"]) != int(recorded.get("file_count", -1)):
        raise ValueError(f"{label} file count mismatch")
    return actual


def validate_lineage_manifest(
    manifest_path: str | Path,
    *,
    expected_stage: str,
    expected_parent_manifests: dict[str, str | Path],
    expected_model_path: str | Path | None = None,
    expected_skill_pool_path: str | Path | None = None,
    expected_data_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json_object(manifest_path, label="lineage manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported lineage schema version")
    if str(manifest.get("stage") or "") != str(expected_stage):
        raise ValueError("lineage stage mismatch")
    if manifest.get("route_scorer") != "unified_memory":
        raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
    state_query = dict(manifest.get("state_query") or {})
    expected_prompt = _expected_prompt_contract()
    recorded_prompt = {
        "state_query_prompt_version": state_query.get("prompt_version"),
        "state_query_instruction": state_query.get("instruction"),
        "state_query_max_chars": state_query.get("max_chars"),
        "state_query_truncation": state_query.get("truncation"),
    }
    if recorded_prompt != expected_prompt:
        raise ValueError("state query prompt contract mismatch")
    backbone = dict(manifest.get("backbone") or {})
    if backbone.get("frozen") is not True:
        raise ValueError("Qwen backbone must remain frozen")
    _require_qwen_model_path(backbone.get("path") or "")
    _validate_recorded_identity(backbone, label="backbone")
    _validate_recorded_identity(manifest.get("skill_pool"), label="skill pool")
    _validate_recorded_identity(manifest.get("data_manifest"), label="data manifest")
    expected_paths = {
        "backbone": expected_model_path,
        "skill pool": expected_skill_pool_path,
        "data manifest": expected_data_manifest_path,
    }
    recorded_paths = {
        "backbone": backbone.get("path"),
        "skill pool": dict(manifest.get("skill_pool") or {}).get("path"),
        "data manifest": dict(manifest.get("data_manifest") or {}).get("path"),
    }
    for label, expected_path in expected_paths.items():
        if expected_path is None:
            continue
        if Path(str(recorded_paths[label] or "")).resolve() != Path(expected_path).resolve():
            raise ValueError(f"expected {label} path mismatch")
    checkpoint = _validate_recorded_identity(manifest.get("checkpoint"), label="checkpoint")

    parents = manifest.get("parents") or {}
    if not isinstance(parents, dict):
        raise ValueError("lineage parents must be an object")
    expected_roles = {str(role) for role in expected_parent_manifests}
    if set(parents) != expected_roles:
        raise ValueError("parent lineage roles mismatch")
    for role, expected_path in expected_parent_manifests.items():
        role = str(role)
        expected_path = Path(expected_path).resolve()
        recorded = parents.get(role)
        if not isinstance(recorded, dict):
            raise ValueError(f"parent lineage record missing: {role}")
        if Path(str(recorded.get("manifest_path") or "")).resolve() != expected_path:
            raise ValueError(f"parent lineage path mismatch: {role}")
        manifest_identity = sha256_path(expected_path)
        if manifest_identity["sha256"] != str(recorded.get("manifest_sha256") or ""):
            raise ValueError(f"parent manifest digest mismatch: {role}")
        loaded_parent = _load_json_object(expected_path, label=f"parent lineage manifest {role}")
        if loaded_parent.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported parent lineage schema version")
        parent_checkpoint = dict(loaded_parent.get("checkpoint") or {})
        if str(parent_checkpoint.get("sha256") or "") != str(recorded.get("checkpoint_sha256") or ""):
            raise ValueError(f"parent checkpoint digest mismatch: {role}")
        if not parent_checkpoint.get("path"):
            raise ValueError(f"parent checkpoint identity missing: {role}")
        actual_parent_checkpoint = sha256_path(parent_checkpoint["path"])
        if actual_parent_checkpoint["sha256"] != str(parent_checkpoint.get("sha256") or ""):
            raise ValueError(f"parent checkpoint file digest mismatch: {role}")
    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "stage": str(expected_stage),
        "checkpoint_sha256": checkpoint["sha256"],
        "parent_roles": sorted(expected_roles),
    }
