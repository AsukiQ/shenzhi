from __future__ import annotations

import bisect
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import torch


STAGE0_TRAINABLE_PREFIXES = (
    "encoder.proj.",
    "initial_belief_head.",
    "skill_table.W.",
    "skill_table.logit_scale_belief",
    "skill_table.skill_bias_belief",
    "vnext.static_query.",
)

STAGE2_MEMORY_TRAINABLE_PREFIXES = (
    "vnext.action_adapter.",
    "vnext.result_adapter.",
    "vnext.transition_delta.",
    "vnext.correction_delta.",
    "vnext.correction_gate.",
    "vnext.transition_scale.",
    "vnext.result_scale.",
    "vnext.synchronization.",
)

STAGE2_CANDIDATE_TRAINABLE_PREFIXES = (
    "vnext.memory_recall_query.",
    "vnext.recall_scale.",
)

STAGE2_STATIC_ROUTE_TRAINABLE_PREFIXES = (
    "vnext.unified_route_query.",
    "vnext.route_skill_adapter.",
    "vnext.route_expert_mixture.",
)

STAGE2_SHARED_TRAINABLE_PREFIXES = (
    *STAGE2_CANDIDATE_TRAINABLE_PREFIXES,
    *STAGE2_STATIC_ROUTE_TRAINABLE_PREFIXES,
)

STAGE2_TRAINABLE_PREFIXES = (
    *STAGE2_MEMORY_TRAINABLE_PREFIXES,
    *STAGE2_SHARED_TRAINABLE_PREFIXES,
)

COMPRESSOR_TRAINABLE_PREFIXES = (
    "vnext.candidate_compressor.",
)

STATIC_ROUTE_ADAPTER_TRAINABLE_PREFIXES = (
    "vnext.static_route_query_delta.",
)

VNEXT_CHECKPOINT_PREFIXES = (
    "encoder.proj.",
    "initial_belief_head.",
    "vnext.",
)

VNEXT_CHECKPOINT_SKILL_TABLE_KEYS = frozenset(
    {
        "skill_table.E",
        "skill_table.logit_scale_retr",
        "skill_table.skill_bias_retr",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
    }
)


def is_vnext_checkpoint_state_key(name: str) -> bool:
    name = str(name)
    return (
        name in VNEXT_CHECKPOINT_SKILL_TABLE_KEYS
        or name.startswith("skill_table.W.")
        or name.startswith(VNEXT_CHECKPOINT_PREFIXES)
    )


def require_canonical_vnext_checkpoint_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("vNext checkpoint model_state_dict must be a mapping")
    unexpected = sorted(
        str(name) for name in state if not is_vnext_checkpoint_state_key(str(name))
    )
    if unexpected:
        raise ValueError(
            "vNext checkpoint contains noncanonical state keys: "
            f"{unexpected[:8]}"
        )
    return {
        "status": "ok",
        "state_key_count": len(state),
        "contains_encoder_backbone": any("backbone." in str(name) for name in state),
    }


LEGACY_STAGE0_DIRECT_KEYS = frozenset(
    {
        "encoder.proj.weight",
        "encoder.proj.bias",
        "skill_table.E",
        "skill_table.W.weight",
        "skill_table.logit_scale_retr",
        "skill_table.skill_bias_retr",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
        "initial_belief_head.state_proj.weight",
        "initial_belief_head.belief_proj.weight",
        "initial_belief_head.norm.weight",
        "initial_belief_head.norm.bias",
    }
)


def mapped_legacy_stage0_state(
    model: Any,
    checkpoint_state: dict[str, Any],
    *,
    legacy_skill_ids: list[str] | None = None,
    current_skill_ids: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Map only the verified legacy static foundation into canonical vNext."""

    if not isinstance(checkpoint_state, dict):
        raise ValueError("legacy Stage0 checkpoint state must be a mapping")
    current = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    source_by_target: dict[str, str] = {}
    skill_row_keys = {
        "skill_table.E",
        "skill_table.skill_bias_retr",
        "skill_table.skill_bias_belief",
    }
    skill_index: torch.Tensor | None = None
    if legacy_skill_ids is not None or current_skill_ids is not None:
        if legacy_skill_ids is None or current_skill_ids is None:
            raise ValueError("legacy Stage0 ID mapping requires both skill ID lists")
        if len(legacy_skill_ids) != len(set(legacy_skill_ids)):
            raise ValueError("legacy Stage0 skills contain duplicate IDs")
        if len(current_skill_ids) != len(set(current_skill_ids)):
            raise ValueError("current Stage0 skills contain duplicate IDs")
        legacy_index = {skill_id: index for index, skill_id in enumerate(legacy_skill_ids)}
        missing_skill_ids = [
            skill_id for skill_id in current_skill_ids if skill_id not in legacy_index
        ]
        if missing_skill_ids:
            raise ValueError(
                "legacy Stage0 checkpoint lacks current skills: "
                f"{missing_skill_ids[:4]}"
            )
        skill_index = torch.tensor(
            [legacy_index[skill_id] for skill_id in current_skill_ids],
            dtype=torch.long,
        )
    for key in sorted(LEGACY_STAGE0_DIRECT_KEYS):
        if key not in checkpoint_state or key not in current:
            continue
        value = checkpoint_state[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if key in skill_row_keys and skill_index is not None:
            if int(tensor.size(0)) != len(legacy_skill_ids or []):
                raise ValueError(f"legacy Stage0 skill tensor row mismatch for {key}")
            tensor = tensor.index_select(0, skill_index)
        if tuple(tensor.shape) != tuple(current[key].shape):
            raise ValueError(f"legacy Stage0 tensor shape mismatch for {key}")
        mapped[key] = tensor
        source_by_target[key] = key
    for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
        source = f"unified_retriever.fuse.{suffix}"
        target = f"vnext.static_query.fuse.{suffix}"
        if source not in checkpoint_state or target not in current:
            raise ValueError(f"legacy Stage0 unified scorer key is missing: {source}")
        value = checkpoint_state[source]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tuple(tensor.shape) != tuple(current[target].shape):
            raise ValueError(f"legacy unified scorer shape mismatch for {source}")
        mapped[target] = tensor
        source_by_target[target] = source
    required_targets = {
        "encoder.proj.weight",
        "encoder.proj.bias",
        "skill_table.E",
        "skill_table.W.weight",
        "skill_table.logit_scale_belief",
        "skill_table.skill_bias_belief",
        "initial_belief_head.state_proj.weight",
        "initial_belief_head.belief_proj.weight",
        "initial_belief_head.norm.weight",
        "initial_belief_head.norm.bias",
        "vnext.static_query.fuse.0.weight",
        "vnext.static_query.fuse.0.bias",
        "vnext.static_query.fuse.2.weight",
        "vnext.static_query.fuse.2.bias",
    }
    missing = sorted(required_targets - set(mapped))
    if missing:
        raise ValueError(f"legacy Stage0 foundation mapping is incomplete: {missing}")
    return mapped, {
        "status": "ok",
        "mapped_key_count": len(mapped),
        "source_by_target": dict(sorted(source_by_target.items())),
        "required_targets": sorted(required_targets),
        "skill_id_alignment": {
            "enabled": skill_index is not None,
            "legacy_skill_count": (
                None if legacy_skill_ids is None else len(legacy_skill_ids)
            ),
            "current_skill_count": (
                None if current_skill_ids is None else len(current_skill_ids)
            ),
            "dropped_legacy_skill_count": (
                0
                if legacy_skill_ids is None or current_skill_ids is None
                else len(legacy_skill_ids) - len(current_skill_ids)
            ),
        },
    }


def semantic_source_id(row: dict[str, Any]) -> str:
    """Return one source identity without changing semantic loss eligibility."""

    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    value = str(
        provenance.get("source_id")
        or row.get("source")
        or row.get("benchmark")
        or ""
    ).strip()
    if not value:
        raise ValueError("source-balanced sampling requires a persisted source identity")
    return value


def semantic_task_id(row: dict[str, Any]) -> str:
    """Return a stable task/trajectory grouping identity for dev stratification."""

    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for value in (
        row.get("split_group_identity"),
        row.get("task_group_identity"),
        row.get("task_id"),
        provenance.get("task_id"),
        row.get("trajectory_id"),
        provenance.get("trajectory_id"),
        row.get("state_equivalence_identity"),
    ):
        resolved = str(value or "").strip()
        if resolved:
            return resolved
    payload = json.dumps(
        {
            "source": semantic_source_id(row),
            "goal": row.get("goal_text") or row.get("task_text") or "",
            "state": row.get("state_text_current") or "",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_stratified_cap_rows(
    rows: list[dict[str, Any]],
    limit: int | None,
    *,
    extra_stratum: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically round-robin sources/strata, then task clusters."""

    if not rows:
        return [], {
            "protocol": "source_stratum_task_round_robin_v1",
            "input_row_count": 0,
            "selected_row_count": 0,
            "selected_by_stratum": {},
        }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        source = semantic_source_id(row)
        suffix = str(extra_stratum(row) if extra_stratum is not None else "all")
        stratum = f"{source}|{suffix}"
        task_id = semantic_task_id(row)
        grouped.setdefault(stratum, {}).setdefault(task_id, []).append(row)
    for task_groups in grouped.values():
        for task_id, task_rows in list(task_groups.items()):
            task_groups[task_id] = sorted(
                task_rows,
                key=lambda row: hashlib.sha256(
                    json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            )
    target = len(rows) if limit is None or int(limit) <= 0 else min(len(rows), int(limit))
    task_positions = {
        stratum: {task_id: 0 for task_id in task_groups}
        for stratum, task_groups in grouped.items()
    }
    task_cursors = {stratum: 0 for stratum in grouped}
    selected: list[dict[str, Any]] = []
    selected_by_stratum: dict[str, int] = {stratum: 0 for stratum in grouped}
    while len(selected) < target:
        progressed = False
        for stratum in sorted(grouped):
            task_ids = sorted(grouped[stratum])
            if not task_ids:
                continue
            for offset in range(len(task_ids)):
                task_pos = (task_cursors[stratum] + offset) % len(task_ids)
                task_id = task_ids[task_pos]
                row_pos = task_positions[stratum][task_id]
                if row_pos >= len(grouped[stratum][task_id]):
                    continue
                selected.append(grouped[stratum][task_id][row_pos])
                task_positions[stratum][task_id] += 1
                task_cursors[stratum] = (task_pos + 1) % len(task_ids)
                selected_by_stratum[stratum] += 1
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected, {
        "protocol": "source_stratum_task_round_robin_v1",
        "input_row_count": len(rows),
        "selected_row_count": len(selected),
        "stratum_count": len(grouped),
        "task_cluster_count": sum(len(task_groups) for task_groups in grouped.values()),
        "input_by_stratum": {
            stratum: sum(len(task_rows) for task_rows in task_groups.values())
            for stratum, task_groups in sorted(grouped.items())
        },
        "selected_by_stratum": dict(sorted(selected_by_stratum.items())),
    }


def seed_vnext_run(seed: int) -> dict[str, Any]:
    """Seed model initialization and CUDA execution before any model build."""

    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "python_random_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": cuda_available,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
    }


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def configure_vnext_stage0(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if _matches_prefix(name, STAGE0_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    retrieval_bias = getattr(
        getattr(model, "skill_table", None),
        "skill_bias_retr",
        None,
    )
    if isinstance(retrieval_bias, torch.Tensor):
        # It is not part of unified scoring.  Freeze it without overwriting a
        # resumed/compatibility checkpoint; scratch models already initialize
        # it to zero.
        retrieval_bias.requires_grad_(False)
    return trainability_report(model, stage="vnext_stage0")


def configure_vnext_stage2(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if _matches_prefix(name, STAGE2_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    return trainability_report(model, stage="vnext_stage2")


def configure_vnext_candidate_compressor(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if _matches_prefix(name, COMPRESSOR_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    return trainability_report(model, stage="vnext_candidate_compressor")


def configure_vnext_static_route_adapter(model: Any) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if _matches_prefix(name, STATIC_ROUTE_ADAPTER_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    return trainability_report(model, stage="vnext_static_route_adapter")


def trainability_report(model: Any, *, stage: str) -> dict[str, Any]:
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    frozen = sorted(name for name, parameter in model.named_parameters() if not parameter.requires_grad)
    return {
        "stage": stage,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "legacy_stop_trainable": any(name.startswith("stop_head.") for name in trainable),
        "legacy_route_gate_trainable": any(
            name.startswith("route_memory_utility_gate.")
            or name.startswith("route_memory_candidate_utility_gate.")
            for name in trainable
        ),
        "encoder_backbone_trainable": any(name.startswith("encoder.backbone.") for name in trainable),
        "encoder_projection_trainable": any(name.startswith("encoder.proj.") for name in trainable),
        "skill_embeddings_trainable": "skill_table.E" in trainable,
        "skill_bias_trainable": any("skill_bias_" in name for name in trainable),
        "retrieval_skill_bias_trainable": "skill_table.skill_bias_retr" in trainable,
        "belief_skill_bias_trainable": "skill_table.skill_bias_belief" in trainable,
        "frozen_parameter_names": frozen,
    }


def require_canonical_trainability(report: dict[str, Any]) -> None:
    stage = str(report.get("stage") or "")
    blockers = [
        key
        for key in (
            "legacy_stop_trainable",
            "legacy_route_gate_trainable",
            "encoder_backbone_trainable",
            "skill_embeddings_trainable",
        )
        if bool(report.get(key))
    ]
    if bool(report.get("retrieval_skill_bias_trainable")):
        blockers.append("retrieval_skill_bias_trainable")
    if bool(report.get("belief_skill_bias_trainable")) and stage != "vnext_stage0":
        blockers.append("belief_skill_bias_trainable")
    if bool(report.get("encoder_projection_trainable")) and stage != "vnext_stage0":
        blockers.append("encoder_projection_trainable")
    if stage == "vnext_stage0" and not bool(report.get("encoder_projection_trainable")):
        blockers.append("encoder_projection_not_trainable")
    if blockers:
        raise ValueError(f"noncanonical trainable parameters: {blockers}")


def state_digest(
    model: Any,
    *,
    include: Callable[[str], bool],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not include(name):
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        # ``Tensor.view(dtype)`` rejects zero-dimensional floating tensors.
        # Flatten first so scalar temperatures and LayerScale parameters have
        # the same byte-stable representation as every other state tensor.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def vnext_checkpoint_state(model: Any) -> dict[str, torch.Tensor]:
    """Return only the canonical frozen foundation and vNext modules."""

    return {
        name: tensor.detach().to(device="cpu")
        for name, tensor in model.state_dict().items()
        if is_vnext_checkpoint_state_key(name)
    }


def static_foundation_digest(model: Any) -> str:
    return state_digest(
        model,
        include=lambda name: name.startswith("encoder.proj.")
        or name == "skill_table.E"
        or name.startswith("skill_table.W.")
        or name.startswith("skill_table.logit_scale_")
        or name.startswith("skill_table.skill_bias_")
        or name.startswith("initial_belief_head.")
        or name.startswith("vnext.static_query."),
    )


def static_route_foundation_digest(model: Any) -> str:
    """Bind the accepted proposal plus its frozen static route adapter."""

    return state_digest(
        model,
        include=lambda name: name.startswith("encoder.proj.")
        or name == "skill_table.E"
        or name.startswith("skill_table.W.")
        or name.startswith("skill_table.logit_scale_")
        or name.startswith("skill_table.skill_bias_")
        or name.startswith("initial_belief_head.")
        or name.startswith("vnext.static_query.")
        or name.startswith("vnext.static_route_query_delta."),
    )


def candidate_foundation_digest(model: Any) -> str:
    """Bind Stage0 plus the independently evolving route foundation."""

    return state_digest(
        model,
        include=lambda name: name.startswith("encoder.proj.")
        or name == "skill_table.E"
        or name.startswith("skill_table.W.")
        or name.startswith("skill_table.logit_scale_")
        or name.startswith("skill_table.skill_bias_")
        or name.startswith("initial_belief_head.")
        or name.startswith("vnext.static_query.")
        or name.startswith("vnext.static_route_query_delta.")
        or name.startswith("vnext.unified_route_query.")
        or name.startswith("vnext.route_skill_adapter.")
        or name.startswith("vnext.candidate_route_gate.")
        or name.startswith("vnext.route_expert_mixture.")
        or name.startswith("vnext.candidate_compressor."),
    )


def legacy_candidate_foundation_digest(model: Any) -> str:
    """Reproduce the pre-direct-router static-reranker digest exactly."""

    return state_digest(
        model,
        include=lambda name: name.startswith("encoder.proj.")
        or name == "skill_table.E"
        or name.startswith("skill_table.W.")
        or name.startswith("skill_table.logit_scale_")
        or name.startswith("skill_table.skill_bias_")
        or name.startswith("initial_belief_head.")
        or name.startswith("vnext.static_query.")
        or name.startswith("vnext.static_route_query_delta.")
        or name.startswith("vnext.candidate_compressor."),
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def read_jsonl(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= int(max_rows):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_inventory_catalogs(path: str | Path) -> dict[str, dict[str, Any]]:
    catalogs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        catalog_id = str(row.get("inventory_catalog_id") or "").strip()
        if not catalog_id:
            raise ValueError("inventory catalog row lacks inventory_catalog_id")
        if catalog_id in catalogs:
            raise ValueError(f"duplicate inventory catalog: {catalog_id}")
        catalogs[catalog_id] = row
    return catalogs


def derive_inventory_catalog_subset(
    catalogs: dict[str, dict[str, Any]],
    selected_skill_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Derive digest-correct catalogs for a declared skill-table subset."""

    selected = {str(item) for item in selected_skill_ids if str(item)}
    if not selected:
        raise ValueError("inventory subset requires at least one selected skill")
    output: dict[str, dict[str, Any]] = {}
    mapping: dict[str, str] = {}
    derived = 0
    dropped = 0
    for catalog_id, row in sorted(catalogs.items()):
        original = [
            str(item)
            for item in row.get("runtime_visible_skill_ids") or []
            if str(item)
        ]
        filtered = [item for item in original if item in selected]
        if not filtered:
            dropped += 1
            continue
        if filtered == original:
            copied = dict(row)
            output[catalog_id] = copied
            mapping[catalog_id] = catalog_id
            continue
        payload = json.dumps(
            sorted(set(filtered)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        derived_id = f"{catalog_id}__skill_subset_{digest[:16]}"
        copied = dict(row)
        copied["inventory_catalog_id"] = derived_id
        copied["inventory_catalog_digest"] = digest
        copied["runtime_visible_skill_ids"] = sorted(set(filtered))
        copied["inventory_pool_size"] = len(copied["runtime_visible_skill_ids"])
        copied["inventory_parent_catalog_id"] = catalog_id
        copied["inventory_parent_catalog_digest"] = str(
            row.get("inventory_catalog_digest") or ""
        )
        copied["inventory_subset_protocol"] = "declared_skill_table_subset_v1"
        output[derived_id] = copied
        mapping[catalog_id] = derived_id
        derived += 1
    return output, mapping, {
        "input_catalog_count": len(catalogs),
        "output_catalog_count": len(output),
        "derived_subset_catalog_count": int(derived),
        "dropped_empty_catalog_count": int(dropped),
        "selected_skill_count": len(selected),
    }


def rewrite_inventory_catalog_references(
    rows: Iterable[dict[str, Any]],
    mapping: dict[str, str],
    catalogs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        catalog_id = str(row.get("runtime_visible_catalog_id") or "")
        if catalog_id not in mapping:
            continue
        resolved_id = mapping[catalog_id]
        catalog = catalogs[resolved_id]
        copied["runtime_visible_catalog_id"] = resolved_id
        copied["inventory_catalog_digest"] = str(
            catalog.get("inventory_catalog_digest") or ""
        )
        copied["inventory_pool_size"] = int(
            catalog.get("inventory_pool_size")
            or len(catalog.get("runtime_visible_skill_ids") or [])
        )
        if resolved_id != catalog_id:
            copied["inventory_parent_catalog_id"] = catalog_id
            copied["inventory_parent_catalog_digest"] = str(
                row.get("inventory_catalog_digest") or ""
            )
        output.append(copied)
    return output


@dataclass
class FrozenTextCache:
    # Despite the historical field name, v3 caches store frozen-backbone pooled
    # features.  The trainable encoder projection is applied by ``batch`` so
    # Stage0 can update it without invalidating or bypassing the cache.
    embeddings_cpu: torch.Tensor | None
    text_to_index: dict[str, int]
    role: str
    build_seconds: float
    projection_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    shards_cpu: tuple[torch.Tensor, ...] = ()
    shard_starts: tuple[int, ...] = ()
    cache_row_count: int | None = None
    hash_to_index: dict[str, int] | None = None
    lookup_count: int = 0
    requested_unique_text_count: int | None = None
    persistent_hit_count: int = 0
    persistent_miss_count: int = 0
    cache_identity: str | None = None
    cache_tensor_path: str | None = None

    def pooled_batch(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        resolved_indices: list[int] = []
        missing: list[str] = []
        for text in texts:
            if text in self.text_to_index:
                resolved_indices.append(self.text_to_index[text])
                continue
            hashed_index = (
                None
                if self.hash_to_index is None
                else self.hash_to_index.get(_text_digest(text))
            )
            if hashed_index is None:
                missing.append(text)
            else:
                resolved_indices.append(int(hashed_index))
        if missing:
            raise RuntimeError(f"frozen text cache miss for role={self.role}: {missing[:2]}")
        indices = torch.tensor(
            resolved_indices,
            dtype=torch.long,
        )
        self.lookup_count += len(texts)
        if self.embeddings_cpu is not None:
            pooled = self.embeddings_cpu.index_select(0, indices).to(
                device=device,
                non_blocking=True,
            )
        else:
            if not self.shards_cpu or not self.shard_starts:
                raise RuntimeError("sharded frozen text cache has no loaded shards")
            dimension = int(self.shards_cpu[0].size(1))
            output_cpu = torch.empty(
                len(texts),
                dimension,
                dtype=self.shards_cpu[0].dtype,
                device="cpu",
                pin_memory=device.type == "cuda" and torch.cuda.is_available(),
            )
            positions_by_shard: dict[int, list[tuple[int, int]]] = {}
            for output_index, global_index in enumerate(indices.tolist()):
                shard_index = bisect.bisect_right(self.shard_starts, global_index) - 1
                if shard_index < 0 or shard_index >= len(self.shards_cpu):
                    raise RuntimeError("frozen text cache index is outside loaded shards")
                local_index = global_index - int(self.shard_starts[shard_index])
                positions_by_shard.setdefault(shard_index, []).append(
                    (output_index, local_index)
                )
            for shard_index, positions in positions_by_shard.items():
                output_positions = torch.tensor(
                    [item[0] for item in positions],
                    dtype=torch.long,
                )
                local_positions = torch.tensor(
                    [item[1] for item in positions],
                    dtype=torch.long,
                )
                selected = self.shards_cpu[shard_index].index_select(
                    0,
                    local_positions,
                )
                output_cpu.index_copy_(0, output_positions, selected)
            pooled = output_cpu.to(device=device, non_blocking=True)
        return pooled

    def batch(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        pooled = self.pooled_batch(texts, device=device)
        if self.projection_fn is None:
            raise RuntimeError("frozen text cache lacks an online projection function")
        return self.projection_fn(pooled)

    def batch_with_projection(
        self,
        texts: list[str],
        *,
        device: torch.device,
        projection_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Reuse one frozen-backbone cache under an alternate frozen projection."""

        if not callable(projection_fn):
            raise TypeError("alternate frozen-cache projection must be callable")
        return projection_fn(self.pooled_batch(texts, device=device))

    def report(self) -> dict[str, Any]:
        if self.embeddings_cpu is not None:
            shape = list(self.embeddings_cpu.shape)
            dtype = str(self.embeddings_cpu.dtype)
            byte_count = int(
                self.embeddings_cpu.numel() * self.embeddings_cpu.element_size()
            )
        else:
            row_count = sum(int(shard.size(0)) for shard in self.shards_cpu)
            dimension = int(self.shards_cpu[0].size(1)) if self.shards_cpu else 0
            shape = [row_count, dimension]
            dtype = str(self.shards_cpu[0].dtype) if self.shards_cpu else "unknown"
            byte_count = sum(
                int(shard.numel() * shard.element_size()) for shard in self.shards_cpu
            )
        return {
            "role": self.role,
            "unique_text_count": int(
                self.cache_row_count
                if self.cache_row_count is not None
                else len(self.text_to_index)
            ),
            "requested_text_count": len(self.text_to_index),
            "shape": shape,
            "dtype": dtype,
            "bytes": byte_count,
            "shard_count": len(self.shards_cpu) if self.shards_cpu else 1,
            "build_seconds": float(self.build_seconds),
            "lookup_count": int(self.lookup_count),
            "requested_unique_text_count": int(
                self.requested_unique_text_count
                if self.requested_unique_text_count is not None
                else len(self.text_to_index)
            ),
            "persistent_hit_count": int(self.persistent_hit_count),
            "persistent_miss_count": int(self.persistent_miss_count),
            "persistent_hit_rate": (
                float(self.persistent_hit_count)
                / float(self.persistent_hit_count + self.persistent_miss_count)
                if self.persistent_hit_count + self.persistent_miss_count
                else 0.0
            ),
            "cache_identity": self.cache_identity,
            "cache_tensor_path": self.cache_tensor_path,
            "representation": "frozen_backbone_pooled_pre_projection_v1",
            "online_projection": True,
        }


def _ordered_texts(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(text) for text in texts if str(text)))


def _text_digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_tensor_mmap(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"persistent frozen cache shard is invalid: {path}")
    return value


_FROZEN_BACKBONE_FILE_SUFFIXES = frozenset(
    {
        ".bin",
        ".json",
        ".merges",
        ".model",
        ".pt",
        ".pth",
        ".py",
        ".safetensors",
        ".sentencepiece",
        ".tiktoken",
        ".txt",
        ".vocab",
    }
)


def _frozen_backbone_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in _FROZEN_BACKBONE_FILE_SUFFIXES
            or path.name in {"tokenizer.model", "spiece.model"}
        )
    )
    if not files or not any(
        path.suffix.lower() in {".bin", ".safetensors"} for path in files
    ):
        raise ValueError("frozen backbone snapshot lacks a local weight file")
    return files


def capture_frozen_backbone_snapshot(model_name_or_path: str | Path) -> dict[str, Any]:
    """Hash one immutable local HF snapshot once before cache construction."""

    root = Path(model_name_or_path).resolve()
    if not root.is_dir():
        raise ValueError(
            "canonical frozen-backbone identity requires a local model directory"
        )
    records: list[dict[str, Any]] = []
    stat_hints: dict[str, dict[str, int]] = {}
    for path in _frozen_backbone_files(root):
        relative = str(path.relative_to(root))
        stat = path.stat()
        records.append(
            {
                "path": relative,
                "bytes": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        )
        stat_hints[relative] = {
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    contract = immutable_run_contract(
        {
            "schema_version": "clstr_frozen_backbone_snapshot_v1",
            "model_root": str(root),
            "files": records,
        }
    )
    return {
        "schema_version": "clstr_frozen_backbone_snapshot_manifest_v1",
        "contract": contract,
        "stat_hints": stat_hints,
    }


def load_frozen_backbone_snapshot(
    path: str | Path,
    *,
    expected_model_name_or_path: str | Path,
    verify_all_hashes: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError(f"frozen backbone snapshot manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = payload.get("contract") if isinstance(payload, dict) else None
    if (
        not isinstance(contract, dict)
        or payload.get("schema_version")
        != "clstr_frozen_backbone_snapshot_manifest_v1"
        or contract.get("schema_version") != "clstr_frozen_backbone_snapshot_v1"
    ):
        raise ValueError("frozen backbone snapshot manifest has an invalid schema")
    expected_digest = immutable_run_contract(
        {key: value for key, value in contract.items() if key != "contract_digest"}
    )["contract_digest"]
    if str(contract.get("contract_digest") or "") != str(expected_digest):
        raise ValueError("frozen backbone snapshot contract digest does not reproduce")
    root = Path(expected_model_name_or_path).resolve()
    if str(root) != str(contract.get("model_root") or ""):
        raise ValueError("frozen backbone snapshot model root changed")
    hints = payload.get("stat_hints") if isinstance(payload.get("stat_hints"), dict) else {}
    checked_hash_count = 0
    records = contract.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("frozen backbone snapshot has no file records")
    for raw_record in records:
        record = raw_record if isinstance(raw_record, dict) else {}
        relative = str(record.get("path") or "")
        file_path = (root / relative).resolve()
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("frozen backbone snapshot file escapes model root") from exc
        if not file_path.is_file():
            raise ValueError(f"frozen backbone snapshot file is missing: {relative}")
        stat = file_path.stat()
        if int(stat.st_size) != int(record.get("bytes") or -1):
            raise ValueError(f"frozen backbone snapshot file size changed: {relative}")
        hint = hints.get(relative) if isinstance(hints.get(relative), dict) else {}
        must_hash = bool(
            verify_all_hashes
            or int(hint.get("bytes") or -1) != int(stat.st_size)
            or int(hint.get("mtime_ns") or -1) != int(stat.st_mtime_ns)
        )
        if must_hash:
            checked_hash_count += 1
            if file_sha256(file_path) != str(record.get("sha256") or ""):
                raise ValueError(f"frozen backbone snapshot content changed: {relative}")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "contract": contract,
        "contract_digest": str(contract["contract_digest"]),
        "file_count": len(records),
        "verified_hash_count": checked_hash_count,
        "full_hash_verification": bool(verify_all_hashes),
    }


def verify_frozen_backbone_contract(
    contract: dict[str, Any],
    *,
    expected_model_name_or_path: str | Path,
) -> dict[str, Any]:
    """Verify an embedded checkpoint contract without its original manifest path."""

    if contract.get("schema_version") != "clstr_frozen_backbone_snapshot_v1":
        raise ValueError("embedded frozen backbone contract has an invalid schema")
    expected_digest = immutable_run_contract(
        {key: value for key, value in contract.items() if key != "contract_digest"}
    )["contract_digest"]
    if str(contract.get("contract_digest") or "") != str(expected_digest):
        raise ValueError("embedded frozen backbone contract digest does not reproduce")
    root = Path(expected_model_name_or_path).resolve()
    if str(root) != str(contract.get("model_root") or ""):
        raise ValueError("embedded frozen backbone model root changed")
    records = contract.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("embedded frozen backbone contract has no files")
    for raw_record in records:
        record = raw_record if isinstance(raw_record, dict) else {}
        relative = str(record.get("path") or "")
        file_path = (root / relative).resolve()
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("embedded frozen backbone file escapes model root") from exc
        if (
            not file_path.is_file()
            or int(file_path.stat().st_size) != int(record.get("bytes") or -1)
            or file_sha256(file_path) != str(record.get("sha256") or "")
        ):
            raise ValueError(f"embedded frozen backbone content changed: {relative}")
    return {
        "path": None,
        "sha256": None,
        "contract": contract,
        "contract_digest": str(contract["contract_digest"]),
        "file_count": len(records),
        "verified_hash_count": len(records),
        "full_hash_verification": True,
    }


def load_or_capture_frozen_backbone_snapshot(
    model_name_or_path: str | Path,
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, capture_frozen_backbone_snapshot(model_name_or_path))
    return load_frozen_backbone_snapshot(
        path,
        expected_model_name_or_path=model_name_or_path,
        verify_all_hashes=False,
    )


def frozen_encoder_contract(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    values = {
        "schema_version": "clstr_frozen_encoder_cache_v1",
        "base_model_name": str(getattr(config, "base_model_name", "")),
        "dimension": int(getattr(config, "d", 0) or 0),
        "encoder_pooling": str(getattr(config, "encoder_pooling", "")),
        "torch_dtype": str(getattr(config, "torch_dtype", "")),
        "max_length": int(getattr(config, "max_length", 0) or 0),
        "state_query_prompt_version": str(
            getattr(config, "state_query_prompt_version", "")
        ),
        "state_query_max_chars": getattr(config, "state_query_max_chars", None),
        "state_query_truncation": str(
            getattr(config, "state_query_truncation", "")
        ),
        "frozen_backbone_snapshot_digest": str(
            getattr(config, "frozen_backbone_snapshot_digest", "") or ""
        ),
        "encoder_projection_digest": state_digest(
            model,
            include=lambda name: name.startswith("encoder.proj."),
        ),
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    values["contract_digest"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return values


def frozen_backbone_cache_contract(model: Any) -> dict[str, Any]:
    """Identity for reusable pooled features, deliberately excluding proj."""

    config = getattr(model, "config", None)
    backbone = getattr(getattr(model, "encoder", None), "backbone", None)
    backbone_config = getattr(backbone, "config", None)
    values = {
        "schema_version": "clstr_frozen_backbone_pooled_cache_v1",
        "base_model_name": str(getattr(config, "base_model_name", "")),
        "backbone_hidden_size": int(
            getattr(backbone_config, "hidden_size", 0) or 0
        ),
        "encoder_pooling": str(getattr(config, "encoder_pooling", "")),
        "torch_dtype": str(getattr(config, "torch_dtype", "")),
        "max_length": int(getattr(config, "max_length", 0) or 0),
        "state_query_prompt_version": str(
            getattr(config, "state_query_prompt_version", "")
        ),
        "state_query_max_chars": getattr(config, "state_query_max_chars", None),
        "state_query_truncation": str(
            getattr(config, "state_query_truncation", "")
        ),
        "frozen_backbone_snapshot_digest": str(
            getattr(config, "frozen_backbone_snapshot_digest", "") or ""
        ),
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    values["contract_digest"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return values


def _encode_text_batches(
    model: Any,
    texts: list[str],
    *,
    role: str,
    batch_size: int,
) -> torch.Tensor:
    if role == "state":
        prepare = lambda batch: [model._serialize_state_for_encoder(text) for text in batch]
    elif role in {"action", "result", "matched_history"}:
        prepare = lambda batch: batch
    else:
        raise ValueError(f"unsupported frozen text role: {role}")
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start : start + int(batch_size)]
            pooled = model.encoder.encode_backbone_pooled(prepare(batch))
            if pooled.ndim != 2 or int(pooled.size(0)) != len(batch):
                raise ValueError("frozen backbone output must match requested text batch")
            batches.append(pooled.detach().to(device="cpu").contiguous())
    if not batches:
        raise ValueError(f"frozen text cache requires texts for role={role}")
    return torch.cat(batches, dim=0).contiguous()


def build_frozen_text_cache(
    model: Any,
    texts: Iterable[str],
    *,
    role: str,
    batch_size: int,
) -> FrozenTextCache:
    ordered = _ordered_texts(texts)
    if not ordered:
        raise ValueError(f"frozen text cache requires texts for role={role}")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("frozen text cache batch_size must be positive")
    started = time.perf_counter()
    embeddings_cpu = _encode_text_batches(
        model,
        ordered,
        role=role,
        batch_size=batch_size,
    )
    if torch.cuda.is_available():
        embeddings_cpu = embeddings_cpu.pin_memory()
    return FrozenTextCache(
        embeddings_cpu=embeddings_cpu,
        text_to_index={text: index for index, text in enumerate(ordered)},
        role=role,
        build_seconds=time.perf_counter() - started,
        projection_fn=model.encoder.project_pooled,
        requested_unique_text_count=len(ordered),
        persistent_miss_count=len(ordered),
        hash_to_index={_text_digest(text): index for index, text in enumerate(ordered)},
    )


def load_frozen_text_cache_read_only(
    model: Any,
    texts: Iterable[str],
    *,
    role: str,
    cache_root: str | Path,
    batch_size: int | None = None,
    cache_shard_size: int | None = None,
) -> FrozenTextCache:
    """Load a complete persistent text cache without mutating its directory."""

    requested = _ordered_texts(texts)
    if not requested:
        raise ValueError(f"frozen text cache requires texts for role={role}")
    contract = frozen_backbone_cache_contract(model)
    identity = str(contract["contract_digest"])
    directory = Path(cache_root) / identity
    manifest_path = directory / f"{role}.index.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"read-only frozen text cache is missing its {role} index: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "clstr_frozen_text_cache_v2"
        or manifest.get("encoder_contract") != contract
        or manifest.get("role") != role
    ):
        raise ValueError("read-only frozen text cache contract mismatch")
    text_hashes = [str(item) for item in manifest.get("text_hashes") or []]
    shard_records = [dict(item) for item in manifest.get("shards") or []]
    if len(text_hashes) != len(set(text_hashes)):
        raise ValueError("read-only frozen text cache contains duplicate hashes")
    hash_to_index = {text_hash: index for index, text_hash in enumerate(text_hashes)}
    missing = [text for text in requested if _text_digest(text) not in hash_to_index]
    if missing:
        raise RuntimeError(
            f"read-only frozen text cache miss for role={role}: {missing[:2]}"
        )
    shards: list[torch.Tensor] = []
    shard_starts: list[int] = []
    expected_start = 0
    for record in shard_records:
        shard_path = Path(str(record.get("path") or ""))
        recorded_start = record.get("start")
        if (
            not shard_path.is_file()
            or recorded_start is None
            or int(recorded_start) != expected_start
        ):
            raise ValueError("read-only frozen text cache shard index is not contiguous")
        shard = _load_tensor_mmap(shard_path)
        recorded_row_count = record.get("row_count")
        if recorded_row_count is None or int(shard.size(0)) != int(recorded_row_count):
            raise ValueError("read-only frozen text cache shard row count mismatch")
        shards.append(shard)
        shard_starts.append(expected_start)
        expected_start += int(shard.size(0))
    if expected_start != len(text_hashes):
        raise ValueError("read-only frozen text cache index row count mismatch")
    return FrozenTextCache(
        embeddings_cpu=None,
        shards_cpu=tuple(shards),
        shard_starts=tuple(shard_starts),
        cache_row_count=len(text_hashes),
        text_to_index={text: hash_to_index[_text_digest(text)] for text in requested},
        hash_to_index=hash_to_index,
        role=role,
        build_seconds=0.0,
        projection_fn=model.encoder.project_pooled,
        requested_unique_text_count=len(requested),
        persistent_hit_count=len(requested),
        persistent_miss_count=0,
        cache_identity=identity,
        cache_tensor_path=str(manifest_path.resolve()),
    )


def load_or_build_frozen_text_cache(
    model: Any,
    texts: Iterable[str],
    *,
    role: str,
    batch_size: int,
    cache_root: str | Path,
    cache_shard_size: int = 4096,
) -> FrozenTextCache:
    """Reuse and incrementally extend a role-specific sharded text cache."""

    requested = _ordered_texts(texts)
    if not requested:
        raise ValueError(f"frozen text cache requires texts for role={role}")
    if int(batch_size) <= 0:
        raise ValueError("frozen text cache batch_size must be positive")
    if int(cache_shard_size) <= 0:
        raise ValueError("frozen text cache shard size must be positive")
    contract = frozen_backbone_cache_contract(model)
    identity = str(contract["contract_digest"])
    directory = Path(cache_root) / identity
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f"{role}.index.json"
    progress_path = directory / f"{role}.progress.json"
    shard_directory = directory / f"{role}.shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"{role}.lock"
    started = time.perf_counter()
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        text_hashes: list[str] = []
        shard_records: list[dict[str, Any]] = []
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != "clstr_frozen_text_cache_v2"
                or manifest.get("encoder_contract") != contract
                or manifest.get("role") != role
            ):
                raise ValueError("persistent frozen text cache contract mismatch")
            text_hashes = [str(item) for item in manifest.get("text_hashes") or []]
            shard_records = [dict(item) for item in manifest.get("shards") or []]
            if len(text_hashes) != len(set(text_hashes)):
                raise ValueError("persistent frozen text cache contains duplicate hashes")
        else:
            legacy_tensor_path = directory / f"{role}.pt"
            legacy_manifest_path = directory / f"{role}.json"
            if legacy_tensor_path.is_file() and legacy_manifest_path.is_file():
                legacy_manifest = json.loads(
                    legacy_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    legacy_manifest.get("schema_version")
                    != "clstr_frozen_text_cache_v1"
                    or legacy_manifest.get("encoder_contract") != contract
                    or legacy_manifest.get("role") != role
                ):
                    raise ValueError("legacy frozen text cache contract mismatch")
                legacy_texts = [str(item) for item in legacy_manifest.get("texts") or []]
                legacy_tensor = torch.load(legacy_tensor_path, map_location="cpu")
                if (
                    not isinstance(legacy_tensor, torch.Tensor)
                    or legacy_tensor.ndim != 2
                    or int(legacy_tensor.size(0)) != len(legacy_texts)
                ):
                    raise ValueError("legacy frozen text cache tensor/manifest mismatch")
                for start in range(0, len(legacy_texts), int(cache_shard_size)):
                    end = min(start + int(cache_shard_size), len(legacy_texts))
                    shard_path = shard_directory / f"shard-{len(shard_records):06d}.pt"
                    temporary = shard_path.with_suffix(".pt.tmp")
                    torch.save(legacy_tensor[start:end].contiguous(), temporary)
                    temporary.replace(shard_path)
                    shard_records.append(
                        {
                            "path": str(shard_path.resolve()),
                            "start": start,
                            "end": end,
                            "row_count": end - start,
                            "sha256": file_sha256(shard_path),
                        }
                    )
                text_hashes = [_text_digest(text) for text in legacy_texts]
                _atomic_write_json(
                    manifest_path,
                    {
                        "schema_version": "clstr_frozen_text_cache_v2",
                        "role": role,
                        "encoder_contract": contract,
                        "text_hashes": text_hashes,
                        "shards": shard_records,
                    },
                )
        existing_hashes = set(text_hashes)
        missing = [text for text in requested if _text_digest(text) not in existing_hashes]
        for start in range(0, len(missing), int(cache_shard_size)):
            chunk = missing[start : start + int(cache_shard_size)]
            encoded = _encode_text_batches(
                model,
                chunk,
                role=role,
                batch_size=int(batch_size),
            )
            shard_start = len(text_hashes)
            shard_end = shard_start + len(chunk)
            shard_path = shard_directory / f"shard-{len(shard_records):06d}.pt"
            temporary = shard_path.with_suffix(".pt.tmp")
            torch.save(encoded, temporary)
            temporary.replace(shard_path)
            shard_records.append(
                {
                    "path": str(shard_path.resolve()),
                    "start": shard_start,
                    "end": shard_end,
                    "row_count": len(chunk),
                    "sha256": file_sha256(shard_path),
                }
            )
            text_hashes.extend(_text_digest(text) for text in chunk)
            _atomic_write_json(
                manifest_path,
                {
                    "schema_version": "clstr_frozen_text_cache_v2",
                    "role": role,
                    "encoder_contract": contract,
                    "text_hashes": text_hashes,
                    "shards": shard_records,
                },
            )
            _atomic_write_json(
                progress_path,
                {
                    "schema_version": "clstr_frozen_text_cache_progress_v1",
                    "status": "building" if shard_end < len(existing_hashes) + len(missing) else "complete",
                    "role": role,
                    "encoder_contract_digest": identity,
                    "completed_row_count": len(text_hashes),
                    "shard_count": len(shard_records),
                },
            )
        if not shard_records or not text_hashes:
            raise RuntimeError("persistent frozen text cache failed to materialize")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    shards: list[torch.Tensor] = []
    shard_starts: list[int] = []
    expected_start = 0
    for record in shard_records:
        shard_path = Path(str(record.get("path") or ""))
        recorded_start = record.get("start")
        if (
            not shard_path.is_file()
            or recorded_start is None
            or int(recorded_start) != expected_start
        ):
            raise ValueError("persistent frozen text cache shard index is not contiguous")
        shard = _load_tensor_mmap(shard_path)
        recorded_row_count = record.get("row_count")
        if recorded_row_count is None or int(shard.size(0)) != int(
            recorded_row_count
        ):
            raise ValueError("persistent frozen text cache shard row count mismatch")
        shards.append(shard)
        shard_starts.append(expected_start)
        expected_start += int(shard.size(0))
    if expected_start != len(text_hashes):
        raise ValueError("persistent frozen text cache index row count mismatch")
    hash_to_index = {text_hash: index for index, text_hash in enumerate(text_hashes)}
    text_to_index = {
        text: hash_to_index[_text_digest(text)]
        for text in requested
    }
    requested_hits = len(requested) - len(missing)
    return FrozenTextCache(
        embeddings_cpu=None,
        shards_cpu=tuple(shards),
        shard_starts=tuple(shard_starts),
        cache_row_count=len(text_hashes),
        text_to_index=text_to_index,
        hash_to_index=hash_to_index,
        role=role,
        build_seconds=time.perf_counter() - started,
        projection_fn=model.encoder.project_pooled,
        requested_unique_text_count=len(requested),
        persistent_hit_count=requested_hits,
        persistent_miss_count=len(missing),
        cache_identity=identity,
        cache_tensor_path=str(manifest_path.resolve()),
    )


def load_or_build_skill_embedding_cache(
    model: Any,
    skills: list[dict[str, Any]],
    *,
    cache_root: str | Path,
    batch_size: int,
    cache_shard_size: int = 2048,
) -> dict[str, Any]:
    """Build the frozen skill table as durable atomic shards and restore it."""

    if int(batch_size) <= 0 or int(cache_shard_size) <= 0:
        raise ValueError("skill embedding cache batch and shard sizes must be positive")
    if len(skills) != int(model.skill_table.E.size(0)):
        raise ValueError("skill embedding cache/model table size mismatch")
    texts = [
        model.skill_table.skill_text_fn(model.skill_table._skill_payload(skill))
        for skill in skills
    ]
    skill_digest = hashlib.sha256()
    for skill, text in zip(skills, texts):
        skill_digest.update(
            json.dumps(skill, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        skill_digest.update(b"\0")
        skill_digest.update(text.encode("utf-8"))
        skill_digest.update(b"\n")
    contract = immutable_run_contract(
        {
            "schema_version": "clstr_frozen_skill_cache_v1",
            "encoder_contract": frozen_encoder_contract(model),
            "skill_text_format": str(getattr(model.config, "skill_text_format", "")),
            "skill_count": len(skills),
            "skills_and_text_sha256": skill_digest.hexdigest(),
        }
    )
    identity = str(contract["contract_digest"])
    directory = Path(cache_root) / identity / "skill_embeddings"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "index.json"
    progress_path = directory / "progress.json"
    lock_path = directory / "build.lock"
    started = time.perf_counter()
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        records: list[dict[str, Any]] = []
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != "clstr_frozen_skill_cache_v1"
                or manifest.get("contract") != contract
            ):
                raise ValueError("persistent skill embedding cache contract mismatch")
            records = [dict(item) for item in manifest.get("shards") or []]
        completed = {
            (int(record.get("start") or 0), int(record.get("end") or 0)): record
            for record in records
        }
        for start in range(0, len(texts), int(cache_shard_size)):
            end = min(start + int(cache_shard_size), len(texts))
            text_hash = hashlib.sha256(
                "\n".join(_text_digest(text) for text in texts[start:end]).encode("ascii")
            ).hexdigest()
            record = completed.get((start, end))
            shard_path = directory / f"shard-{start:08d}-{end:08d}.pt"
            if record is not None:
                if (
                    str(record.get("text_hash") or "") != text_hash
                    or Path(str(record.get("path") or "")).resolve()
                    != shard_path.resolve()
                    or not shard_path.is_file()
                ):
                    raise ValueError("persistent skill embedding shard identity mismatch")
                continue
            encoded_chunks: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch_start in range(start, end, int(batch_size)):
                    batch_end = min(batch_start + int(batch_size), end)
                    chunk = model.skill_table.encoder_fn(
                        texts[batch_start:batch_end]
                    )
                    if chunk.ndim != 2 or tuple(chunk.shape) != (
                        batch_end - batch_start,
                        int(model.skill_table.E.size(1)),
                    ):
                        raise ValueError(
                            "skill embedding encoder output shape mismatch"
                        )
                    encoded_chunks.append(
                        chunk.detach().to(device="cpu").contiguous()
                    )
            encoded = torch.cat(encoded_chunks, dim=0).contiguous()
            if tuple(encoded.shape) != (
                end - start,
                int(model.skill_table.E.size(1)),
            ):
                raise ValueError("skill embedding encoder output shape mismatch")
            temporary = shard_path.with_suffix(".pt.tmp")
            torch.save(encoded, temporary)
            temporary.replace(shard_path)
            records.append(
                {
                    "path": str(shard_path.resolve()),
                    "start": start,
                    "end": end,
                    "row_count": end - start,
                    "text_hash": text_hash,
                    "sha256": file_sha256(shard_path),
                }
            )
            records.sort(key=lambda item: int(item["start"]))
            _atomic_write_json(
                manifest_path,
                {
                    "schema_version": "clstr_frozen_skill_cache_v1",
                    "contract": contract,
                    "shards": records,
                },
            )
            _atomic_write_json(
                progress_path,
                {
                    "schema_version": "clstr_frozen_skill_cache_progress_v1",
                    "status": "complete" if end == len(texts) else "building",
                    "completed_skill_count": end,
                    "skill_count": len(texts),
                    "shard_count": len(records),
                },
            )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    expected_start = 0
    with torch.no_grad():
        for record in sorted(records, key=lambda item: int(item["start"])):
            start = int(record["start"])
            end = int(record["end"])
            if start != expected_start or end <= start:
                raise ValueError("persistent skill embedding shards are not contiguous")
            shard = _load_tensor_mmap(Path(str(record["path"])))
            if tuple(shard.shape) != (end - start, int(model.skill_table.E.size(1))):
                raise ValueError("persistent skill embedding shard shape mismatch")
            model.skill_table.E[start:end].copy_(
                shard.to(device=model.skill_table.E.device, dtype=model.skill_table.E.dtype)
            )
            expected_start = end
    if expected_start != len(skills):
        raise ValueError("persistent skill embedding cache is incomplete")
    if hasattr(model, "_vnext_skill_embedding_cache"):
        model._vnext_skill_embedding_cache.clear()
    return {
        "status": "ok",
        "cache_identity": identity,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "skill_count": len(skills),
        "shard_count": len(records),
        "build_seconds": time.perf_counter() - started,
        "contract": contract,
    }


def capture_vnext_source_manifest(
    project_root: str | Path,
    canonical_paths: Iterable[str | Path],
    *,
    require_clean: bool,
) -> dict[str, Any]:
    """Capture code content and invocation provenance without binding job IDs to resume."""

    root = Path(project_root).resolve()
    paths = sorted({str(Path(path)) for path in canonical_paths})
    resolved: list[Path] = []
    for raw_path in paths:
        path = (root / raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("canonical source path escapes project root") from exc
        if not path.is_file():
            raise ValueError(f"canonical source file is missing: {raw_path}")
        resolved.append(path)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    relative_paths = [str(path.relative_to(root)) for path in resolved]
    status_lines = git("status", "--porcelain=v1", "--", *relative_paths).splitlines()
    file_records = [
        {
            "path": relative,
            "sha256": file_sha256(root / relative),
            "bytes": int((root / relative).stat().st_size),
        }
        for relative in relative_paths
    ]
    tracked_diff = git("diff", "--binary", "HEAD", "--", *relative_paths)
    dirty_payload = json.dumps(
        {
            "status": status_lines,
            "tracked_diff": tracked_diff,
            "files": file_records,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    dirty = bool(status_lines)
    if require_clean and dirty:
        raise ValueError(
            "canonical vNext source is dirty; snapshot/commit it before a full run"
        )
    source_contract = immutable_run_contract(
        {
            "schema_version": "clstr_vnext_source_contract_v1",
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "canonical_files": file_records,
            "dirty": dirty,
            "dirty_status": status_lines,
            "dirty_diff_sha256": hashlib.sha256(
                dirty_payload.encode("utf-8")
            ).hexdigest(),
        }
    )
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR",
        "SLURM_CLUSTER_NAME",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_GPUS",
    )
    invocation = {
        "argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
        "captured_unix_time": time.time(),
    }
    return {
        "schema_version": "clstr_vnext_source_manifest_v1",
        "project_root": str(root),
        "source_contract": source_contract,
        "invocation": invocation,
    }


def persist_vnext_source_manifest(
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    path = Path(output_dir) / "source_manifest.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        prior_digest = str(
            ((existing.get("source_contract") or {}).get("contract_digest")) or ""
        )
        current_digest = str(
            ((manifest.get("source_contract") or {}).get("contract_digest")) or ""
        )
        if prior_digest != current_digest:
            raise ValueError("resume output source contract differs from canonical code")
    invocations = list(existing.get("invocations") or [])
    invocations.append(dict(manifest.get("invocation") or {}))
    payload = {
        "schema_version": "clstr_vnext_source_manifest_v1",
        "project_root": manifest.get("project_root"),
        "source_contract": manifest.get("source_contract"),
        "invocations": invocations,
    }
    _atomic_write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "source_contract_digest": str(
            ((payload.get("source_contract") or {}).get("contract_digest")) or ""
        ),
        "dirty": bool((payload.get("source_contract") or {}).get("dirty")),
        "invocation_count": len(invocations),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(Path(path), payload)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_verified_data_contract(
    contract_path: str | Path,
    expected_files: dict[str, str | Path | None],
) -> dict[str, Any]:
    """Bind canonical trainer inputs to one approved release/smoke manifest."""

    path = Path(contract_path)
    if not path.is_file():
        raise ValueError(f"vNext data contract does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("vNext training requires an approved data contract")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("vNext data contract lacks its file manifest")
    verified: dict[str, Any] = {}
    for key, raw_expected_path in sorted(expected_files.items()):
        if raw_expected_path is None:
            continue
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"vNext data contract lacks file entry: {key}")
        recorded_path = Path(str(entry.get("path") or ""))
        recorded_digest = str(entry.get("sha256") or "")
        expected_path = Path(raw_expected_path)
        if not recorded_path.is_file() or not expected_path.is_file() or not recorded_digest:
            raise ValueError(f"vNext data contract has an invalid file entry: {key}")
        observed_digest = file_sha256(expected_path)
        if observed_digest != recorded_digest:
            raise ValueError(f"vNext trainer input digest differs from data contract: {key}")
        verified[key] = {
            "path": str(expected_path.resolve()),
            "manifest_path": str(recorded_path.resolve()),
            "content_alias": recorded_path.resolve() != expected_path.resolve(),
            "sha256": observed_digest,
        }
    return {
        "status": "ok",
        "contract_path": str(path.resolve()),
        "contract_sha256": file_sha256(path),
        "schema_version": str(payload.get("schema_version") or ""),
        "verified_files": verified,
    }


def immutable_run_contract(payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    values.pop("contract_digest", None)
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    values["contract_digest"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return values


def require_matching_run_contract(
    observed: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    if not isinstance(observed, dict):
        raise ValueError("resume checkpoint lacks immutable run contract")
    if str(observed.get("contract_digest") or "") != str(
        expected.get("contract_digest") or ""
    ) or observed != expected:
        raise ValueError("resume checkpoint run contract does not match current invocation")
