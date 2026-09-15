from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from clstr.stage4_safe_memory import (
    router_state_digest,
    validate_stage4_candidate_admission_delta_state_dict,
    validate_stage4_cmc_delta_state_dict,
    validate_stage4_delta_state_dict,
)
from clstr.stage_checkpoint_init import (
    build_clstr_model_from_stage0_checkpoint,
    load_head_checkpoint_into_model,
)


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or "").strip()


def _read_skill_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source.name} line {line_no}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source.name} line {line_no}: skill row must be an object")
            rows.append(row)
    return rows


def _unique_skill_ids(rows: list[dict[str, Any]], *, label: str) -> list[str]:
    skill_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        skill_id = _skill_id(row)
        if not skill_id or skill_id in seen:
            raise ValueError(f"{label} must contain unique non-empty skill IDs")
        seen.add(skill_id)
        skill_ids.append(skill_id)
    return skill_ids


def restore_native_benchmark_checkpoint_chain(
    *,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    training_skills_path: str | Path,
    benchmark_skills: list[dict[str, Any]],
    model_cache_dir: str | Path,
    device: torch.device,
    require_safe_memory_delta: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, int], dict[str, Any]]:
    training_skills_path = Path(training_skills_path)
    training_skills = _read_skill_rows(training_skills_path)
    training_skill_ids = _unique_skill_ids(training_skills, label="training skill pool")
    benchmark_skill_ids = _unique_skill_ids(benchmark_skills, label="benchmark skill pool")
    training_ids = set(training_skill_ids)
    appended_skills = [
        dict(row)
        for row in benchmark_skills
        if _skill_id(row) not in training_ids
    ]
    appended_skill_ids = [_skill_id(row) for row in appended_skills]
    final_skill_ids = [*training_skill_ids, *appended_skill_ids]

    model, model_config, stage0_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=training_skills_path,
        model_cache_dir=Path(model_cache_dir),
        allow_skill_table_prefix_expansion=False,
    )
    if model_config.get("freeze_backbone") is not True:
        raise ValueError("native Qwen benchmark evaluation requires a frozen backbone")
    if stage0_report.get("stage0_loaded") is not True:
        raise ValueError("Stage0 checkpoint did not load for native benchmark evaluation")
    if "skill_table.E" not in set(stage0_report.get("stage0_loaded_keys") or []):
        raise ValueError("Stage0 training skill table was not restored exactly")
    if stage0_report.get("local_skill_table_rebuilt") is True:
        raise ValueError("native benchmark evaluation must not rebuild the Stage0 training skill table")

    stage2_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state_for_native_benchmark",
        protect_routing_foundation=True,
        allow_skill_table_prefix_expansion=False,
    )
    if stage2_report.get("loaded") is not True or stage2_report.get("shape_mismatched"):
        raise ValueError("Stage2 checkpoint overlay failed for native benchmark evaluation")

    stage4_report = None
    stage4_delta_validation = None
    stage2_router_digest = None
    stage4_router_digest = None
    stage2_fast_router_digest = None
    stage4_fast_router_digest = None
    stage4_method = None
    if stage4_checkpoint_path is not None:
        if require_safe_memory_delta:
            stage4_payload = torch.load(stage4_checkpoint_path, map_location="cpu")
            if not isinstance(stage4_payload, dict):
                raise TypeError("Stage4 checkpoint payload must be a dict")
            stage4_method = str(stage4_payload.get("stage4_method") or "")
            stage4_delta_validation = (
                validate_stage4_candidate_admission_delta_state_dict(
                    stage4_payload.get("model_state_dict") or {}
                )
                if stage4_method == "candidate_admission_residual_v1"
                else validate_stage4_cmc_delta_state_dict(
                    stage4_payload.get("model_state_dict") or {}
                )
                if stage4_method == "counterfactual_memory_calibration_v1"
                else validate_stage4_delta_state_dict(
                    stage4_payload.get("model_state_dict") or {}
                )
            )
            stage2_router_digest = router_state_digest(model, scope="full")
            if stage4_method in {
                "counterfactual_memory_calibration_v1",
                "candidate_admission_residual_v1",
            }:
                stage2_fast_router_digest = router_state_digest(
                    model,
                    scope="fast",
                )
                if (
                    str(
                        stage4_payload.get(
                            "parent_stage2_full_router_digest"
                        )
                        or ""
                    )
                    != stage2_router_digest
                    or str(
                        stage4_payload.get(
                            "parent_stage2_fast_router_digest"
                        )
                        or ""
                    )
                    != stage2_fast_router_digest
                ):
                    raise ValueError("CMC parent Stage2 router digest mismatch")
        stage4_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state_for_native_benchmark",
            protect_routing_foundation=True,
            allow_skill_table_prefix_expansion=False,
        )
        if stage4_report.get("loaded") is not True or stage4_report.get("shape_mismatched"):
            raise ValueError("Stage4 checkpoint overlay failed for native benchmark evaluation")
        if require_safe_memory_delta:
            stage4_router_digest = router_state_digest(model, scope="full")
            if stage4_method in {
                "counterfactual_memory_calibration_v1",
                "candidate_admission_residual_v1",
            }:
                stage4_fast_router_digest = router_state_digest(
                    model,
                    scope="fast",
                )
            if stage4_router_digest != stage2_router_digest:
                raise ValueError("Stage4 overlay changed the immutable Stage2 router")
            if (
                stage2_fast_router_digest is not None
                and stage4_fast_router_digest != stage2_fast_router_digest
            ):
                raise ValueError("Stage4 overlay changed the fast Stage2 router")

    if hasattr(model, "to"):
        model.to(device)
    backbone = getattr(getattr(model, "encoder", None), "backbone", None)
    trainable_backbone_parameters = (
        sum(
            int(parameter.numel())
            for parameter in backbone.parameters()
            if parameter.requires_grad
        )
        if backbone is not None
        else 0
    )
    if trainable_backbone_parameters:
        raise ValueError("native Qwen benchmark evaluation found trainable backbone parameters")

    append_fn = getattr(model, "append_skills", None)
    if not callable(append_fn):
        raise ValueError("native benchmark model cannot append unseen skills")
    append_report = append_fn(appended_skills)
    if not isinstance(append_report, dict):
        raise TypeError("native benchmark skill append report must be an object")
    if list(append_report.get("appended_skill_ids") or []) != appended_skill_ids:
        raise ValueError("native benchmark skill append order mismatch")
    if int(append_report.get("appended_count", -1)) != len(appended_skill_ids):
        raise ValueError("native benchmark skill append count mismatch")

    model_skill_ids = [_skill_id(dict(row)) for row in list(getattr(model, "skills", []))]
    if model_skill_ids != final_skill_ids:
        raise ValueError("native benchmark model skill-table order mismatch")
    if hasattr(model, "eval"):
        model.eval()

    skill_id_to_idx = {
        skill_id: index for index, skill_id in enumerate(final_skill_ids)
    }
    return model, model_config, skill_id_to_idx, {
        "checkpoint_load_order": [
            "stage0",
            "stage2",
            *(("stage4",) if stage4_checkpoint_path is not None else ()),
            "append_benchmark_skills",
        ],
        "training_skills_path": str(training_skills_path),
        "trainable_backbone_parameters": trainable_backbone_parameters,
        "skill_merge": {
            "training_skill_count": len(training_skill_ids),
            "benchmark_skill_count": len(benchmark_skill_ids),
            "known_benchmark_skill_count": sum(
                skill_id in training_ids for skill_id in benchmark_skill_ids
            ),
            "appended_benchmark_skill_count": len(appended_skill_ids),
            "final_skill_count": len(final_skill_ids),
        },
        "stage0": stage0_report,
        "stage2": stage2_report,
        "stage4": stage4_report,
        "stage4_delta_validation": stage4_delta_validation,
        "stage2_router_digest": stage2_router_digest,
        "stage4_router_digest": stage4_router_digest,
        "stage2_fast_router_digest": stage2_fast_router_digest,
        "stage4_fast_router_digest": stage4_fast_router_digest,
        "router_digest_unchanged": (
            stage2_router_digest == stage4_router_digest
            and (
                stage2_fast_router_digest == stage4_fast_router_digest
                if stage2_fast_router_digest is not None
                else True
            )
            if require_safe_memory_delta and stage4_checkpoint_path is not None
            else None
        ),
        "skill_append": append_report,
    }
