from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from clstr.history_channel import (
    audit_history_channel_rows,
    compact_causal_state_from_row,
)
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.stage_checkpoint_init import load_compatible_state_dict
from clstr.vnext_candidates import (
    label_natural_support,
    training_only_teacher_retained_support,
)
from clstr.vnext_data import runtime_visible_mask
from clstr.vnext_losses import (
    natural_candidate_topk_coverage_loss,
    no_regret_loss,
    paired_counterfactual_margin_loss,
    query_expert_mixture_target_loss,
    ranking_utility,
)
from clstr.vnext_training import (
    STAGE2_CANDIDATE_TRAINABLE_PREFIXES,
    STAGE2_MEMORY_TRAINABLE_PREFIXES,
    STAGE2_STATIC_ROUTE_TRAINABLE_PREFIXES,
    capture_vnext_source_manifest,
    candidate_foundation_digest,
    configure_vnext_stage2,
    derive_inventory_catalog_subset,
    file_sha256,
    immutable_run_contract,
    load_inventory_catalogs,
    legacy_candidate_foundation_digest,
    load_frozen_backbone_snapshot,
    load_frozen_text_cache_read_only,
    load_or_build_frozen_text_cache,
    move_optimizer_state_to_device,
    persist_vnext_source_manifest,
    read_jsonl,
    require_canonical_trainability,
    require_canonical_vnext_checkpoint_state,
    require_matching_run_contract,
    require_verified_data_contract,
    rewrite_inventory_catalog_references,
    seed_vnext_run,
    semantic_source_id,
    semantic_task_id,
    stable_stratified_cap_rows,
    static_foundation_digest,
    static_route_foundation_digest,
    vnext_checkpoint_state,
    write_json,
    verify_frozen_backbone_contract,
)


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _correction_result_text(row: dict[str, Any]) -> str:
    """Return only result text explicitly eligible for memory correction."""

    text = str(row.get("actual_result_text") or "")
    capabilities = (
        row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
    )
    if not (
        text.strip()
        and bool(row.get("actual_result_executed"))
        and bool(capabilities.get("actual_execution_result"))
    ):
        return ""
    return text


def _checkpoint_state(model: CLSTRModel) -> dict[str, torch.Tensor]:
    return vnext_checkpoint_state(model)


def _require_selected_skill_prefix(
    payload: dict[str, Any],
    skills_path: str | Path,
) -> str:
    inputs = ((payload.get("run_contract") or {}).get("inputs") or {})
    expected_skills_digest = str(
        inputs.get("selected_skills_sha256") or inputs.get("skills_sha256") or ""
    )
    if not expected_skills_digest:
        raise ValueError("vNext checkpoint lacks its selected skill-prefix digest")
    observed_skills_digest = file_sha256(skills_path)
    if observed_skills_digest != expected_skills_digest:
        raise ValueError(
            "skills_path is not the exact selected skill prefix from the checkpoint"
        )
    return observed_skills_digest


def _load_candidate_model(
    checkpoint_path: str | Path,
    skills_path: str | Path,
    *,
    device: torch.device,
    synchronization_enabled: bool | None = None,
    synchronization_pair_dim: int | None = None,
    synchronization_trace_length: int | None = None,
    synchronization_scale_initial: float | None = None,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Restore a quality-gated fixed-Top500 route-query adapter checkpoint."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("stage") != "clstr_vnext_candidate_compressor":
        raise ValueError(
            "vNext Stage2 candidate checkpoint must be a "
            "clstr_vnext_candidate_compressor checkpoint"
        )
    _require_selected_skill_prefix(payload, skills_path)
    require_canonical_vnext_checkpoint_state(payload.get("model_state_dict"))
    run_contract = payload.get("run_contract") or {}
    inputs = (run_contract.get("inputs") or {})
    optimization = (run_contract.get("optimization") or {})
    parent_sha = str(payload.get("parent_stage0_checkpoint_sha256") or "")
    if not parent_sha or parent_sha != str(inputs.get("stage0_checkpoint_sha256") or ""):
        raise ValueError("static reranker parent Stage0 lineage is inconsistent")
    if str(payload.get("objective_mode") or "") != "static_route_query_residual":
        raise ValueError(
            "Stage2 requires a fixed-Top500 static route-query adapter checkpoint"
        )
    if int(payload.get("coarse_k") or 0) != 500 or int(payload.get("compressed_m") or 0) != 500:
        raise ValueError("Stage2 static reranker must preserve the full natural Top500 set")
    if str(inputs.get("training_rows_kind") or "") != "trajectory_successor_route":
        raise ValueError(
            "Stage2 static reranker was not trained on ordered successor-route rows"
        )
    if (
        str(optimization.get("candidate_protocol") or "")
        != "natural_top500_static_route_query_residual_v1"
    ):
        raise ValueError(
            "Stage2 static route-query adapter has an incompatible candidate protocol"
        )
    config_values = dict(payload.get("config") or {})
    config_values["defer_skill_table_init"] = True
    if synchronization_enabled is not None:
        config_values["vnext_synchronization_enabled"] = bool(synchronization_enabled)
    if synchronization_pair_dim is not None:
        config_values["vnext_synchronization_pair_dim"] = int(synchronization_pair_dim)
    if synchronization_trace_length is not None:
        config_values["vnext_synchronization_trace_length"] = int(
            synchronization_trace_length
        )
    if synchronization_scale_initial is not None:
        config_values["vnext_synchronization_scale_initial"] = float(
            synchronization_scale_initial
        )
    config = CLSTRConfig(**config_values)
    snapshot_contract = inputs.get("frozen_backbone_snapshot_contract")
    snapshot_path = str(config.frozen_backbone_snapshot_manifest_path or "")
    if not isinstance(snapshot_contract, dict):
        raise ValueError("static route-query adapter lacks its frozen-backbone contract")
    backbone_snapshot = (
        load_frozen_backbone_snapshot(
            snapshot_path,
            expected_model_name_or_path=config.base_model_name,
            verify_all_hashes=False,
        )
        if snapshot_path and Path(snapshot_path).is_file()
        else verify_frozen_backbone_contract(
            snapshot_contract,
            expected_model_name_or_path=config.base_model_name,
        )
    )
    if (
        backbone_snapshot["contract"] != snapshot_contract
        or str(backbone_snapshot["contract_digest"])
        != str(config.frozen_backbone_snapshot_digest or "")
    ):
        raise ValueError("static route-query adapter frozen-backbone identity changed")
    skills = read_jsonl(skills_path)
    model = CLSTRModel(config, skills).to(device)
    model.eval()
    load_report = load_compatible_state_dict(
        model,
        payload["model_state_dict"],
        partial_load_mode="vnext_stage2_from_static_reranker",
    )
    if "skill_table.E" not in set(load_report.get("loaded_keys") or []):
        raise ValueError("static route-query adapter did not restore skill_table.E")
    expected_static = str(payload.get("static_foundation_digest") or "")
    expected_candidate = str(payload.get("candidate_foundation_digest") or "")
    if not expected_static or static_foundation_digest(model) != expected_static:
        raise ValueError("static route-query adapter changed its Stage0 foundation")
    observed_candidate = candidate_foundation_digest(model)
    observed_legacy_candidate = legacy_candidate_foundation_digest(model)
    if not expected_candidate or expected_candidate not in {
        observed_candidate,
        observed_legacy_candidate,
    }:
        raise ValueError(
            "static route-query adapter foundation changed during restore"
        )
    return model, vars(config), payload, backbone_snapshot


def _load_stage0_model(
    checkpoint_path: str | Path,
    skills_path: str | Path,
    *,
    device: torch.device,
    synchronization_enabled: bool | None = None,
    synchronization_pair_dim: int | None = None,
    synchronization_trace_length: int | None = None,
    synchronization_scale_initial: float | None = None,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("stage") != "clstr_vnext_stage0":
        raise ValueError("vNext Stage2 requires a clstr_vnext_stage0 checkpoint")
    _require_selected_skill_prefix(payload, skills_path)
    require_canonical_vnext_checkpoint_state(payload.get("model_state_dict"))
    config_values = dict(payload.get("config") or {})
    config_values["defer_skill_table_init"] = True
    if synchronization_enabled is not None:
        config_values["vnext_synchronization_enabled"] = bool(synchronization_enabled)
    if synchronization_pair_dim is not None:
        config_values["vnext_synchronization_pair_dim"] = int(synchronization_pair_dim)
    if synchronization_trace_length is not None:
        config_values["vnext_synchronization_trace_length"] = int(
            synchronization_trace_length
        )
    if synchronization_scale_initial is not None:
        config_values["vnext_synchronization_scale_initial"] = float(
            synchronization_scale_initial
        )
    config = CLSTRConfig(**config_values)
    snapshot_contract = (
        ((payload.get("run_contract") or {}).get("inputs") or {}).get(
            "frozen_backbone_snapshot_contract"
        )
    )
    snapshot_path = str(config.frozen_backbone_snapshot_manifest_path or "")
    if not isinstance(snapshot_contract, dict):
        raise ValueError("Stage0 checkpoint lacks its frozen-backbone snapshot contract")
    backbone_snapshot = (
        load_frozen_backbone_snapshot(
            snapshot_path,
            expected_model_name_or_path=config.base_model_name,
            verify_all_hashes=False,
        )
        if snapshot_path and Path(snapshot_path).is_file()
        else verify_frozen_backbone_contract(
            snapshot_contract,
            expected_model_name_or_path=config.base_model_name,
        )
    )
    if (
        backbone_snapshot["contract"] != snapshot_contract
        or str(backbone_snapshot["contract_digest"])
        != str(config.frozen_backbone_snapshot_digest or "")
    ):
        raise ValueError("Stage0 frozen-backbone snapshot identity changed")
    skills = read_jsonl(skills_path)
    model = CLSTRModel(config, skills).to(device)
    model.eval()
    load_report = load_compatible_state_dict(
        model,
        payload["model_state_dict"],
        partial_load_mode="vnext_stage2_from_stage0",
    )
    if "skill_table.E" not in set(load_report.get("loaded_keys") or []):
        raise ValueError("vNext Stage0 checkpoint did not restore skill_table.E")
    return model, vars(config), payload, backbone_snapshot


def _causal_route_state_text(row: dict[str, Any]) -> str:
    return compact_causal_state_from_row(row)


def _required_state_cache_texts(
    trajectory_groups: list[list[dict[str, Any]]],
) -> list[str]:
    """Return the exact state texts consumed by every Stage2 route path."""

    return [
        text
        for rows in trajectory_groups
        for row in rows
        for text in (
            str(row["state_text_current"]),
            _causal_route_state_text(row),
        )
    ]


def _prepare_trajectories(
    rows: list[dict[str, Any]],
    skill_ids: set[str],
    catalogs: dict[str, dict[str, Any]],
    *,
    sequence_role: str = "ordinary",
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if sequence_role not in {"ordinary", "pair_support"}:
        raise ValueError(f"unsupported Stage2 sequence role: {sequence_role}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    quarantined_input_result_text: Counter[str] = Counter()
    for row in rows:
        pair_support_only = bool(row.get("pair_support_only"))
        if sequence_role == "ordinary" and pair_support_only:
            raise ValueError("pair-support row entered ordinary Stage2 sampling")
        if sequence_role == "pair_support" and not pair_support_only:
            raise ValueError("causal pair-support input contains an ordinary trajectory row")
        capabilities = (
            row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
        )
        if sequence_role == "pair_support" and not bool(
            capabilities.get("causal_branch_pair")
        ):
            raise ValueError("causal pair-support row lacks its branch capability")
        result_text = str(row.get("actual_result_text") or "").strip()
        result_executed = bool(row.get("actual_result_executed"))
        result_capability = bool(capabilities.get("actual_execution_result"))
        if result_capability and not (result_executed and result_text):
            raise ValueError(
                "Stage2 executed-result capability lacks an aligned executed result"
            )
        if result_executed and not result_text:
            raise ValueError("Stage2 executed-result flag lacks actual_result_text")
        result_quarantine_reason = ""
        if result_text and not result_capability:
            result_quarantine_reason = (
                "executed_flag_without_actual_execution_capability"
                if result_executed
                else "result_text_without_actual_execution_capability"
            )
            quarantined_input_result_text[result_quarantine_reason] += 1
        target = str(row.get("target_skill_id") or row.get("skill_id") or "").strip()
        if target not in skill_ids:
            skipped["target_not_selected"] += 1
            continue
        catalog_id = str(row.get("runtime_visible_catalog_id") or "")
        if catalog_id not in catalogs:
            skipped["catalog_not_selected"] += 1
            continue
        if not str(row.get("state_text_current") or "").strip():
            skipped["empty_state"] += 1
            continue
        _causal_route_state_text(row)
        if row.get("decision_step_index") in (None, ""):
            raise ValueError("Stage2 row lacks an explicit causal decision index")
        if int(row["decision_step_index"]) != int(row.get("step_index") or 0):
            raise ValueError("Stage2 causal decision index does not match trajectory step")
        trajectory_id = str(row.get("trajectory_id") or row.get("task_id") or "").strip()
        if not trajectory_id:
            skipped["missing_trajectory_id"] += 1
            continue
        copied = dict(row)
        copied["_vnext_result_correction_eligible"] = bool(
            result_text and result_executed and result_capability
        )
        if result_quarantine_reason:
            copied["actual_result_text"] = ""
            copied["actual_result_executed"] = False
            copied["_vnext_result_text_quarantined"] = True
            copied["_vnext_result_text_quarantine_reason"] = result_quarantine_reason
        copied["_vnext_target_skill_id"] = target
        copied["_vnext_positive_skill_ids"] = sorted(
            {
                target,
                *(
                    str(item).strip()
                    for item in row.get("equivalent_next_skill_ids") or []
                    if str(item).strip() in skill_ids
                ),
            }
        )
        legal = {
            str(item)
            for item in catalogs[catalog_id].get("runtime_visible_skill_ids") or []
            if str(item)
        }
        if target not in legal or not set(copied["_vnext_positive_skill_ids"]).issubset(legal):
            raise ValueError("Stage2 target/equivalent positive is outside runtime inventory")
        if not str(row.get("action_text") or "").strip():
            raise ValueError("Stage2 ordered event lacks an executed action")
        grouped[trajectory_id].append(copied)
    trajectories: list[list[dict[str, Any]]] = []
    for trajectory_rows in grouped.values():
        trajectory_rows.sort(key=lambda row: int(row.get("step_index") or 0))
        if len(trajectory_rows) < 2:
            skipped["trajectory_too_short"] += len(trajectory_rows)
            continue
        steps = [int(row.get("step_index") or 0) for row in trajectory_rows]
        if steps[0] != 0 or any(
            current != previous + 1
            for previous, current in zip(steps, steps[1:])
        ):
            skipped["trajectory_not_contiguous_from_initial_decision"] += len(
                trajectory_rows
            )
            continue
        if sequence_role == "ordinary":
            ordered_flags = [
                bool((row.get("capabilities") or {}).get("ordered_next_tool"))
                for row in trajectory_rows
            ]
            if ordered_flags[0] or not all(ordered_flags[1:]):
                skipped["trajectory_not_fully_ordered_after_initial_event"] += len(
                    trajectory_rows
                )
                continue
        trajectories.append(trajectory_rows)
    retained_quarantine: Counter[str] = Counter(
        str(row.get("_vnext_result_text_quarantine_reason") or "")
        for trajectory_rows in trajectories
        for row in trajectory_rows
        if bool(row.get("_vnext_result_text_quarantined"))
    )
    return trajectories, {
        "sequence_role": sequence_role,
        "trajectory_count": len(trajectories),
        "row_count": sum(len(rows) for rows in trajectories),
        "skipped": dict(sorted(skipped.items())),
        "length_min": min((len(rows) for rows in trajectories), default=0),
        "length_max": max((len(rows) for rows in trajectories), default=0),
        "result_text_quarantine": {
            "detected_input_row_count": int(
                sum(quarantined_input_result_text.values())
            ),
            "detected_input_rows_by_reason": dict(
                sorted(quarantined_input_result_text.items())
            ),
            "retained_prepared_row_count": int(sum(retained_quarantine.values())),
            "retained_prepared_rows_by_reason": dict(
                sorted(retained_quarantine.items())
            ),
            "excluded_from_memory_correction": True,
            "route_and_transition_supervision_retained": True,
        },
    }


def _trajectory_index(
    trajectories: list[list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for rows in trajectories:
        trajectory_id = str(rows[0].get("trajectory_id") or rows[0].get("task_id") or "")
        if not trajectory_id or trajectory_id in output:
            raise ValueError("Stage2 trajectories require unique nonempty identities")
        output[trajectory_id] = rows
    return output


def _load_robust_prefix_anchors(
    path: str | Path | None,
    *,
    expected_kind: str,
    trajectories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for view_row in read_jsonl(path):
        if str(view_row.get("robust_prefix_kind") or "") != expected_kind:
            raise ValueError("robust-prefix view kind mismatch")
        if not bool((view_row.get("capabilities") or {}).get("verified_robust_prefix")):
            raise ValueError("robust-prefix view lacks verified capability")
        trajectory_id = str(view_row.get("trajectory_id") or "")
        step_index = int(view_row.get("step_index") or 0)
        identity = (trajectory_id, step_index)
        if not trajectory_id or identity in seen or trajectory_id not in trajectories:
            raise ValueError("robust-prefix view has an invalid or duplicate row identity")
        rows = trajectories[trajectory_id]
        matches = [
            (index, row)
            for index, row in enumerate(rows)
            if int(row.get("step_index") or index) == step_index
        ]
        if len(matches) != 1:
            raise ValueError("robust-prefix view target step is not unique")
        target_index, prepared_row = matches[0]
        if target_index <= 0:
            raise ValueError("robust-prefix supervision requires nonempty executed history")
        if str(view_row.get("target_skill_id") or "") != str(
            prepared_row.get("_vnext_target_skill_id") or ""
        ):
            raise ValueError("robust-prefix view target differs from the canonical trajectory")
        annotation = view_row.get("robust_prefix_annotation") or {}
        error_steps = annotation.get(
            "executed_error_step_indices",
            view_row.get("executed_error_step_indices"),
        )
        if not isinstance(error_steps, list) or not error_steps:
            raise ValueError("robust-prefix view lacks executed error event indices")
        try:
            event_step = min(int(item) for item in error_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("robust-prefix error event indices are invalid") from exc
        event_matches = [
            index
            for index, row in enumerate(rows)
            if int(row.get("step_index") or index) == event_step
        ]
        if len(event_matches) != 1 or event_matches[0] >= target_index:
            raise ValueError("robust-prefix error event is not aligned before its target")
        output.append(
            {
                "rows": rows,
                "event_index": int(event_matches[0]),
                "target_index": int(target_index),
                "kind": expected_kind,
                "source": semantic_source_id(prepared_row),
                "identity": f"{trajectory_id}:{step_index}",
            }
        )
        seen.add(identity)
    return sorted(output, key=lambda row: str(row["identity"]))


def _resolve_pair_row(
    reference: dict[str, Any],
    trajectories: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    trajectory_id = str(reference.get("trajectory_id") or "")
    if trajectory_id not in trajectories:
        raise ValueError(f"causal pair references unknown trajectory: {trajectory_id}")
    rows = trajectories[trajectory_id]
    step_index = int(reference.get("step_index") or 0)
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if int(row.get("step_index") or index) == step_index
    ]
    if len(matches) != 1:
        raise ValueError("causal pair step reference is not unique")
    index, row = matches[0]
    expected_target = str(reference.get("target_skill_id") or "")
    if expected_target and expected_target != str(row.get("_vnext_target_skill_id") or ""):
        raise ValueError("causal pair target does not match trajectory row")
    return rows, index, row


def _prepared_prefix_record(
    rows: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    events = [
        {
            "skill_id": str(
                row.get("_vnext_target_skill_id")
                or row.get("target_skill_id")
                or row.get("skill_id")
                or ""
            ),
            "action_text": str(row.get("action_text") or ""),
            "actual_result_text": _correction_result_text(row),
            "actual_result_executed": bool(_correction_result_text(row)),
        }
        for row in rows[: int(index)]
    ]

    def digest(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "digest": digest(events),
        "event_digests": [digest(event) for event in events],
        "skill_counts": Counter(str(event["skill_id"]) for event in events),
    }


def _load_causal_pairs(
    path: str | Path | None,
    *,
    expected_kind: str,
    trajectories: dict[str, list[dict[str, Any]]],
    catalogs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in read_jsonl(path):
        pair_id = str(pair.get("causal_pair_id") or "")
        if not pair_id or pair_id in seen:
            raise ValueError("causal pairs require unique nonempty causal_pair_id")
        if str(pair.get("causal_pair_kind") or "") != expected_kind:
            raise ValueError(f"causal pair kind mismatch for {pair_id}")
        if not bool(pair.get("causal_label_verified")) or not bool(
            pair.get("trainable_causal_label")
        ):
            raise ValueError("causal training pair lacks a verified trainable label")
        if not str(pair.get("causal_cluster_id") or ""):
            raise ValueError("causal training pair lacks a task/current-state cluster identity")
        verification = pair.get("causal_verification") or {}
        evidence_type = str(verification.get("evidence_type") or "")
        allowed_evidence = {
            "history_branch": {
                "symmetric_target_completion_contrast_v1",
                "executed_branch_annotation_v1",
                "trusted_expert_branch_annotation_v1",
                "verified_unordered_remaining_tool_branch_v1",
            },
            "order_effect": {
                "executed_order_effect_v1",
                "trusted_temporal_order_annotation_v1",
            },
            "result_outcome": {"same_action_different_executed_result_v1"},
        }
        if evidence_type not in allowed_evidence[expected_kind]:
            raise ValueError("causal training pair has unsupported verification evidence")
        if expected_kind == "history_branch":
            history_a = str(pair.get("history_a_digest") or "")
            history_b = str(pair.get("history_b_digest") or "")
            divergence = pair.get("first_history_divergence_index")
            if (
                not history_a
                or not history_b
                or history_a == history_b
                or divergence is None
                or int(divergence) < 0
            ):
                raise ValueError("history-branch pair lacks a distinct audited replay prefix")
        if expected_kind == "result_outcome" and not bool(
            pair.get("same_pre_event_history")
        ):
            raise ValueError("outcome pair does not hold pre-event history fixed")
        rows_a, index_a, row_a = _resolve_pair_row(pair.get("row_a") or {}, trajectories)
        rows_b, index_b, row_b = _resolve_pair_row(pair.get("row_b") or {}, trajectories)
        if rows_a is rows_b or index_a <= 0 or index_a != index_b:
            raise ValueError("causal pair must use different trajectories at one positive prefix length")
        if str(row_a.get("state_text_current") or "") != str(row_b.get("state_text_current") or ""):
            raise ValueError("causal pair current states are not exact aliases")
        if str(row_a.get("goal_text") or row_a.get("task_text") or "") != str(
            row_b.get("goal_text") or row_b.get("task_text") or ""
        ):
            raise ValueError("causal pair does not preserve the task/goal boundary")
        digest_a = str(row_a.get("inventory_catalog_digest") or "")
        digest_b = str(row_b.get("inventory_catalog_digest") or "")
        if not digest_a or digest_a != digest_b:
            raise ValueError("causal pair does not preserve runtime inventory")
        target_a = str(row_a.get("_vnext_target_skill_id") or "")
        target_b = str(row_b.get("_vnext_target_skill_id") or "")
        if not target_a or not target_b or target_a == target_b:
            raise ValueError("causal pair must contain two different verified targets")
        catalog = catalogs.get(str(row_a.get("runtime_visible_catalog_id") or ""))
        legal = {str(item) for item in (catalog or {}).get("runtime_visible_skill_ids") or []}
        if target_a not in legal or target_b not in legal:
            raise ValueError("causal pair targets are not cross-legal")
        equivalents_a = {
            str(item) for item in row_a.get("equivalent_next_skill_ids") or [] if str(item)
        }
        equivalents_b = {
            str(item) for item in row_b.get("equivalent_next_skill_ids") or [] if str(item)
        }
        if target_b in equivalents_a or target_a in equivalents_b:
            raise ValueError("causal pair targets are declared semantic equivalents")
        prefix_a = _prepared_prefix_record(rows_a, index_a)
        prefix_b = _prepared_prefix_record(rows_b, index_b)
        if str(pair.get("history_a_digest") or "") != str(prefix_a["digest"]) or str(
            pair.get("history_b_digest") or ""
        ) != str(prefix_b["digest"]):
            raise ValueError("causal pair replay-prefix digest does not reproduce")
        divergence = next(
            (
                index
                for index, (event_a, event_b) in enumerate(
                    zip(prefix_a["event_digests"], prefix_b["event_digests"])
                )
                if event_a != event_b
            ),
            None,
        )
        if divergence is None or int(pair.get("first_history_divergence_index")) != int(
            divergence
        ):
            raise ValueError("causal pair history divergence does not reproduce")
        if evidence_type in {
            "symmetric_target_completion_contrast_v1",
            "verified_unordered_remaining_tool_branch_v1",
        }:
            counts_a = prefix_a["skill_counts"]
            counts_b = prefix_b["skill_counts"]
            if not (
                int(counts_a[target_a]) < int(counts_b[target_a])
                and int(counts_b[target_b]) < int(counts_a[target_b])
            ):
                raise ValueError("history-branch completion contrast does not reproduce")
        if evidence_type == "verified_unordered_remaining_tool_branch_v1" and not (
            bool(verification.get("unordered_required_set_verified"))
            and bool(verification.get("actual_swapped_event_results"))
            and int(verification.get("shared_prefix_event_count") or 0)
            == len(prefix_a["event_digests"]) - 1
        ):
            raise ValueError("unordered remaining-tool branch provenance does not reproduce")
        pair_source = str(pair.get("source_id") or "")
        row_source_a = semantic_source_id(row_a)
        row_source_b = semantic_source_id(row_b)
        if not pair_source or pair_source != row_source_a or pair_source != row_source_b:
            raise ValueError("causal pair does not preserve one audited source identity")
        prepared = dict(pair)
        prepared["_rows_a"] = rows_a
        prepared["_rows_b"] = rows_b
        prepared["_index"] = index_a
        prepared["_row_a"] = row_a
        prepared["_row_b"] = row_b
        if expected_kind == "result_outcome":
            event_a = pair.get("event_a") or {}
            event_b = pair.get("event_b") or {}
            event_rows_a, event_index_a, event_row_a = _resolve_pair_row(event_a, trajectories)
            event_rows_b, event_index_b, event_row_b = _resolve_pair_row(event_b, trajectories)
            if event_rows_a is not rows_a or event_rows_b is not rows_b:
                raise ValueError("outcome pair event belongs to a different trajectory")
            if event_index_a != event_index_b or event_index_a >= index_a:
                raise ValueError("outcome pair event alignment is invalid")
            if str(event_row_a.get("skill_id") or event_row_a.get("target_skill_id") or "") != str(
                event_row_b.get("skill_id") or event_row_b.get("target_skill_id") or ""
            ) or str(event_row_a.get("action_text") or "") != str(event_row_b.get("action_text") or ""):
                raise ValueError("outcome pair does not hold executed action fixed")
            if (
                str(event_row_a.get("state_text_current") or "")
                != str(event_row_b.get("state_text_current") or "")
                or str(event_row_a.get("goal_text") or event_row_a.get("task_text") or "")
                != str(event_row_b.get("goal_text") or event_row_b.get("task_text") or "")
                or str(event_row_a.get("inventory_catalog_digest") or "")
                != str(event_row_b.get("inventory_catalog_digest") or "")
            ):
                raise ValueError("outcome pair does not hold pre-action state fixed")
            result_a = _correction_result_text(event_row_a)
            result_b = _correction_result_text(event_row_b)
            if not result_a or not result_b or result_a == result_b:
                raise ValueError("outcome pair requires two different executed results")
            if not bool(event_row_a.get("result_novel_for_memory")) or not bool(
                event_row_b.get("result_novel_for_memory")
            ):
                raise ValueError("outcome pair result is not eligible memory innovation")
            event_prefix_a = _prepared_prefix_record(event_rows_a, event_index_a)
            event_prefix_b = _prepared_prefix_record(event_rows_b, event_index_b)
            expected_pre_event_digest = str(
                verification.get("pre_event_history_digest") or ""
            )
            if (
                str(event_prefix_a["digest"]) != str(event_prefix_b["digest"])
                or expected_pre_event_digest != str(event_prefix_a["digest"])
            ):
                raise ValueError("outcome pair pre-event history does not reproduce")
            prepared["_event_index"] = event_index_a
            prepared["_event_row_a"] = event_row_a
            prepared["_event_row_b"] = event_row_b
        seen.add(pair_id)
        output.append(prepared)
    return output


def _require_weighted_causal_views(
    causal_pairs: dict[str, list[dict[str, Any]]],
    causal_dev_pairs: dict[str, list[dict[str, Any]]],
    *,
    weights: dict[str, float],
) -> None:
    for kind, raw_weight in weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"causal loss weight must be finite and nonnegative: {kind}")
        if weight == 0.0:
            continue
        if not causal_pairs.get(kind):
            raise ValueError(
                f"positive causal loss weight requires nonempty verified train view: {kind}"
            )
        if not causal_dev_pairs.get(kind):
            raise ValueError(
                f"positive causal loss weight requires nonempty verified dev view: {kind}"
            )


def _curriculum_horizon_candidates(
    progress: float,
    max_horizon: int,
) -> tuple[int, ...]:
    if progress < 0.25:
        candidates = [2, 3]
    elif progress < 0.60:
        candidates = [3, 4, 5, 6, 7]
    else:
        candidates = [2, 3, 4, 5, 6, 7, 8, 12, 16]
    candidates = [value for value in candidates if value <= int(max_horizon)]
    if not candidates:
        return (max(2, int(max_horizon)),)
    return tuple(candidates)


def _choose_horizon(
    progress: float,
    max_horizon: int,
    rng: random.Random,
) -> int:
    return rng.choice(list(_curriculum_horizon_candidates(progress, max_horizon)))


def _segment_sampling_family(row: dict[str, Any]) -> str:
    benchmark = str(
        row.get("source_benchmark") or row.get("benchmark") or ""
    ).strip().lower()
    if benchmark in {"toolbench", "toolbench_g3"}:
        return "toolbench"
    if benchmark in {"tau2", "toolsandbox"}:
        return benchmark
    return benchmark or semantic_source_id(row)


def _family_horizon_support(
    trajectories: list[list[dict[str, Any]]],
    *,
    max_horizon: int,
) -> dict[str, tuple[int, ...]]:
    """Return every mechanically supported horizon for each family."""

    maximum_by_family: dict[str, int] = {}
    for rows in trajectories:
        if not rows:
            continue
        family = _segment_sampling_family(rows[0])
        maximum_by_family[family] = max(
            int(maximum_by_family.get(family) or 0),
            len(rows),
        )
    return {
        family: tuple(range(2, min(int(max_horizon), maximum) + 1))
        for family, maximum in sorted(maximum_by_family.items())
        if int(maximum) >= 2
    }


def _choose_family_first_horizon(
    progress: float,
    max_horizon: int,
    rng: random.Random,
    *,
    family_horizon_support: dict[str, tuple[int, ...]],
    family_position: int,
) -> tuple[int, str]:
    """Rotate the anchor family before choosing one supported horizon."""

    families = sorted(family_horizon_support)
    if not families:
        raise ValueError("family-first horizon sampling has no eligible family")
    family = families[int(family_position) % len(families)]
    supported = set(family_horizon_support[family])
    candidates = [
        value
        for value in _curriculum_horizon_candidates(progress, max_horizon)
        if int(value) in supported
    ]
    if not candidates:
        raise ValueError(
            f"family-first horizon sampling has no curriculum support: {family}"
        )
    return int(rng.choice(candidates)), family


def _segment_sampling_task(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    for value in (
        provenance.get("raw_task_id"),
        provenance.get("scenario_name"),
        row.get("split_group_identity"),
        row.get("task_group_identity"),
        row.get("trajectory_id"),
    ):
        resolved = str(value or "").strip()
        if resolved:
            return resolved
    return semantic_task_id(row)


def _sample_segments(
    trajectories: list[list[dict[str, Any]]],
    *,
    batch_size: int,
    horizon: int,
    rng: random.Random,
    anchored: bool,
    source_position: int = 0,
    forced_anchors: list[dict[str, Any]] | None = None,
    required_family: str | None = None,
) -> tuple[list[tuple[list[dict[str, Any]], int, int, bool]], dict[str, Any]]:
    required_family = str(required_family or "").strip() or None
    eligible = [
        rows
        for rows in trajectories
        if len(rows) >= horizon
        and (
            required_family is None
            or _segment_sampling_family(rows[0]) == required_family
        )
    ]
    if not eligible:
        suffix = (
            ""
            if required_family is None
            else f" for required family={required_family}"
        )
        raise ValueError(
            f"no Stage2 trajectory supports horizon={horizon}{suffix}"
        )
    anchors: list[dict[str, Any]] = []
    if forced_anchors:
        anchors = [
            anchor
            for anchor in forced_anchors
            if len(anchor["rows"]) >= horizon
            and int(anchor["target_index"]) - int(anchor["event_index"]) + 1
            <= horizon
            and (
                required_family is None
                or _segment_sampling_family(anchor["rows"][0])
                == required_family
            )
        ]
    elif anchored:
        for rows in eligible:
            step_to_index = {int(row.get("step_index") or idx): idx for idx, row in enumerate(rows)}
            for event_index, row in enumerate(rows[:-1]):
                hidden_step = row.get("first_future_step_without_result_visibility")
                if hidden_step is None or not bool(row.get("result_visible_in_next_state")):
                    continue
                target_index = step_to_index.get(int(hidden_step))
                if target_index is None or target_index <= event_index:
                    continue
                if target_index - event_index + 1 <= horizon:
                    anchors.append(
                        {
                            "rows": rows,
                            "event_index": event_index,
                            "target_index": target_index,
                            "kind": "delayed_result",
                        }
                    )
    mode = (
        "robust_prefix"
        if forced_anchors and anchors
        else "delayed_result"
        if anchors
        else "ordinary"
    )
    family_source_task_groups: dict[
        str,
        dict[str, dict[str, list[Any]]],
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for item in anchors if anchors else eligible:
        rows = item["rows"] if anchors else item
        family = _segment_sampling_family(rows[0])
        source = semantic_source_id(rows[0])
        task = _segment_sampling_task(rows[0])
        family_source_task_groups[family][source][task].append(item)
    families = sorted(family_source_task_groups)
    if not families:
        raise ValueError("family-balanced Stage2 segment pool has no source")
    cursor = int(source_position) % len(families)
    segments: list[tuple[list[dict[str, Any]], int, int, bool]] = []
    sampled_families: Counter[str] = Counter()
    sampled_sources: Counter[str] = Counter()
    sampled_anchor_kinds: Counter[str] = Counter()
    for offset in range(int(batch_size)):
        family = families[(cursor + offset) % len(families)]
        sources = sorted(family_source_task_groups[family])
        family_visit = (int(source_position) + offset) // len(families)
        source = sources[family_visit % len(sources)]
        source_visit = family_visit // len(sources)
        tasks = sorted(family_source_task_groups[family][source])
        task = tasks[source_visit % len(tasks)]
        sampled_families[family] += 1
        sampled_sources[source] += 1
        item = rng.choice(family_source_task_groups[family][source][task])
        if anchors:
            rows = item["rows"]
            event_index = int(item["event_index"])
            target_index = int(item["target_index"])
            sampled_anchor_kinds[str(item.get("kind") or mode)] += 1
            start_min = max(0, target_index - horizon + 1)
            start_max = min(event_index, len(rows) - horizon)
            start = rng.randint(start_min, max(start_min, start_max))
            segments.append((rows, start, start + horizon, True))
        else:
            rows = item
            start = rng.randint(0, len(rows) - horizon)
            segments.append((rows, start, start + horizon, False))
    return segments, {
        "protocol": "benchmark_family_source_trajectory_round_robin_v2",
        "mode": mode,
        "horizon": int(horizon),
        "eligible_items_by_family_source": {
            family: {
                source: sum(len(items) for items in task_groups.values())
                for source, task_groups in sorted(source_groups.items())
            }
            for family, source_groups in sorted(
                family_source_task_groups.items()
            )
        },
        "eligible_tasks_by_family_source": {
            family: {
                source: len(task_groups)
                for source, task_groups in sorted(source_groups.items())
            }
            for family, source_groups in sorted(
                family_source_task_groups.items()
            )
        },
        "eligible_items_by_source": {
            source: sum(len(items) for items in task_groups.values())
            for source_groups in family_source_task_groups.values()
            for source, task_groups in sorted(source_groups.items())
        },
        "sampled_families": dict(sorted(sampled_families.items())),
        "sampled_sources": dict(sorted(sampled_sources.items())),
        "sampled_anchor_kinds": dict(sorted(sampled_anchor_kinds.items())),
        "required_family": required_family,
        "required_family_honored": bool(
            required_family is None
            or sampled_families == Counter({required_family: int(batch_size)})
        ),
        "forced_anchor_requested": bool(forced_anchors),
        "forced_anchor_eligible_count": len(anchors) if forced_anchors else None,
    }


def _positive_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(len(rows), len(skill_id_to_idx), dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        indices = [
            skill_id_to_idx[skill_id]
            for skill_id in row.get("_vnext_positive_skill_ids") or []
            if skill_id in skill_id_to_idx
        ]
        if indices:
            mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def _memory_conditioned_natural_support(
    model: CLSTRModel,
    states: torch.Tensor,
    memories: torch.Tensor,
    static_memory: torch.Tensor,
    history_mask: torch.Tensor,
    legal: torch.Tensor,
    positive: torch.Tensor,
    *,
    current_states: torch.Tensor | None = None,
    latent_trace: torch.Tensor | None = None,
    coarse_k: int,
    compressed_m: int,
):
    """Build and label the label-free recurrent proposal/compression path."""

    queries = model.vnext_queries(
        states,
        memories,
        static_memory,
        history_mask,
        memory_state=states if current_states is None else current_states,
        latent_trace=latent_trace,
    )
    path = model.vnext_natural_candidate_union_path(
        queries.static_recall,
        queries.dynamic_recall,
        states if current_states is None else current_states,
        legal,
        history_mask,
        coarse_k=coarse_k,
        dynamic_extra_k=compressed_m,
    )
    coarse_labels = label_natural_support(
        path.coarse_candidate_ids,
        path.coarse_valid_mask,
        positive,
    )
    support_labels = label_natural_support(
        path.support.candidate_ids,
        path.support.valid_mask,
        positive,
    )
    return queries, path, coarse_labels, support_labels


def _dense_multi_positive_nll(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    return (
        torch.logsumexp(logits.masked_fill(~legal, floor), dim=-1)
        - torch.logsumexp(logits.masked_fill(~(positive & legal), floor), dim=-1)
    ).mean()


def _best_positive_rank(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    masked_positive = logits.masked_fill(~(positive & legal), floor)
    best_values, best_indices = masked_positive.max(dim=-1)
    skill_ids = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)
    return 1 + (
        legal
        & (
            (logits > best_values.unsqueeze(-1))
            | (
                logits.eq(best_values.unsqueeze(-1))
                & skill_ids.lt(best_indices.unsqueeze(-1))
            )
        )
    ).sum(dim=-1)


def _row_memory_supervision_eligible(rows: list[dict[str, Any]], absolute_indices: list[int]) -> torch.Tensor:
    values: list[bool] = []
    for trajectory, index in zip(rows, absolute_indices):
        del trajectory
        values.append(index > 0)
    return torch.tensor(values, dtype=torch.bool)


def _batch_memory_at_start(
    model: CLSTRModel,
    segments: list[tuple[list[dict[str, Any]], int, int, bool]],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    with_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if not segments:
        raise ValueError("prefix replay requires at least one segment")
    with torch.no_grad():
        first_rows = [rows[0] for rows, _start, _end, _anchored in segments]
        states = state_cache.batch(
            [_causal_route_state_text(row) for row in first_rows],
            device=device,
        )
        legal = runtime_visible_mask(
            first_rows,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        memories = model.vnext_initial_belief(states, legal, top_k=belief_top_k)
        latent_trace = (
            memories.new_zeros(
                (
                    int(memories.size(0)),
                    int(model.vnext.synchronization_trace_length),
                    int(memories.size(-1)),
                )
            )
            if with_trace
            else None
        )
        starts = [int(start) for _rows, start, _end, _anchored in segments]
        for event_index in range(max(starts, default=0)):
            active_positions = [
                position
                for position, start in enumerate(starts)
                if event_index < start
            ]
            if not active_positions:
                continue
            active_index = torch.tensor(
                active_positions,
                dtype=torch.long,
                device=device,
            )
            active_rows = [segments[position][0][event_index] for position in active_positions]
            updated = _apply_replay_events(
                model,
                memories.index_select(0, active_index),
                active_rows,
                state_cache=state_cache,
                action_cache=action_cache,
                result_cache=result_cache,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                latent_trace=(
                    latent_trace.index_select(0, active_index)
                    if latent_trace is not None
                    else None
                ),
            )
            updated_trace = None
            if latent_trace is not None:
                updated, updated_trace = updated
            if memories.dtype != updated.dtype:
                memories = memories.to(dtype=updated.dtype)
            memories = memories.index_copy(0, active_index, updated)
            if latent_trace is not None and updated_trace is not None:
                latent_trace = latent_trace.index_copy(
                    0,
                    active_index,
                    updated_trace,
                )
    if latent_trace is not None:
        return memories.detach(), latent_trace.detach()
    return memories.detach()


def _apply_replay_events(
    model: CLSTRModel,
    memories: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    latent_trace: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    states = state_cache.batch(
        [str(row["state_text_current"]) for row in rows],
        device=device,
    )
    skill_indices = torch.tensor(
        [skill_id_to_idx[row["_vnext_target_skill_id"]] for row in rows],
        dtype=torch.long,
        device=device,
    )
    skills = model.vnext_normalized_skill_embeddings(dtype=states.dtype).index_select(
        0,
        skill_indices,
    )
    actions = action_cache.batch(
        [str(row.get("action_text") or "") for row in rows],
        device=device,
    )
    predicted, _transition_delta, adapted_action = model.vnext.predict_memory(
        memories,
        states,
        skills,
        actions,
    )
    result_indices = [
        row_index
        for row_index, row in enumerate(rows)
        if _correction_result_text(row)
    ]
    if not result_indices:
        effective = predicted
    else:
        if result_cache is None:
            raise RuntimeError("result-bearing replay row has no frozen result cache")
        index = torch.tensor(result_indices, dtype=torch.long, device=device)
        results = result_cache.batch(
            [_correction_result_text(rows[row_index]) for row_index in result_indices],
            device=device,
        )
        corrected, _correction_delta, _beta = model.vnext.correct_memory(
            predicted.index_select(0, index),
            states.index_select(0, index),
            skills.index_select(0, index),
            adapted_action.index_select(0, index),
            results,
            action_is_adapted=True,
        )
        effective = predicted.index_copy(0, index, corrected)
    if latent_trace is not None:
        return effective, model.vnext.append_latent_trace(effective, latent_trace)
    return effective


def _replay_pair_to_decision(
    model: CLSTRModel,
    pair: dict[str, Any],
    *,
    max_horizon: int,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    decision_index = int(pair["_index"])
    start = max(0, decision_index - int(max_horizon))
    segments = [
        (pair["_rows_a"], start, decision_index + 1, False),
        (pair["_rows_b"], start, decision_index + 1, False),
    ]
    with_trace = bool(model.vnext.synchronization_enabled)
    replayed = _batch_memory_at_start(
        model,
        segments,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
        with_trace=with_trace,
    )
    if with_trace:
        memories, latent_trace = replayed
    else:
        memories = replayed
        latent_trace = None
    for event_index in range(start, decision_index):
        updated = _apply_replay_events(
            model,
            memories,
            [pair["_rows_a"][event_index], pair["_rows_b"][event_index]],
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            latent_trace=latent_trace,
        )
        if latent_trace is None:
            memories = updated
        else:
            memories, latent_trace = updated
    if latent_trace is not None:
        return memories, latent_trace
    return memories


def _result_pair_memories(
    model: CLSTRModel,
    pair: dict[str, Any],
    *,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Return factual and result-swapped memories with event-anchored BPTT."""

    if result_cache is None:
        raise RuntimeError("outcome counterfactual requires a frozen result cache")
    event_index = int(pair["_event_index"])
    decision_index = int(pair["_index"])
    segments = [
        (pair["_rows_a"], event_index, decision_index + 1, True),
        (pair["_rows_b"], event_index, decision_index + 1, True),
    ]
    with_trace = bool(model.vnext.synchronization_enabled)
    replayed = _batch_memory_at_start(
        model,
        segments,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
        with_trace=with_trace,
    )
    if with_trace:
        before_event, before_trace = replayed
    else:
        before_event = replayed
        before_trace = None
    event_rows = [pair["_event_row_a"], pair["_event_row_b"]]
    states = state_cache.batch(
        [str(row["state_text_current"]) for row in event_rows],
        device=device,
    )
    skill_indices = torch.tensor(
        [skill_id_to_idx[row["_vnext_target_skill_id"]] for row in event_rows],
        dtype=torch.long,
        device=device,
    )
    skills = model.vnext_normalized_skill_embeddings(dtype=states.dtype).index_select(
        0,
        skill_indices,
    )
    actions = action_cache.batch(
        [str(row.get("action_text") or "") for row in event_rows],
        device=device,
    )
    predicted, _transition_delta, adapted_actions = model.vnext.predict_memory(
        before_event,
        states,
        skills,
        actions,
    )
    result_texts = [_correction_result_text(row) for row in event_rows]
    factual_results = result_cache.batch(result_texts, device=device)
    swapped_results = factual_results.flip(0)
    repeated_predicted = torch.cat((predicted, predicted), dim=0)
    repeated_states = torch.cat((states, states), dim=0)
    repeated_skills = torch.cat((skills, skills), dim=0)
    repeated_actions = torch.cat((adapted_actions, adapted_actions), dim=0)
    corrected, _correction_delta, _beta = model.vnext.correct_memory(
        repeated_predicted,
        repeated_states,
        repeated_skills,
        repeated_actions,
        torch.cat((factual_results, swapped_results), dim=0),
        action_is_adapted=True,
    )
    memories = corrected
    latent_trace = (
        model.vnext.append_latent_trace(
            memories,
            torch.cat((before_trace, before_trace), dim=0),
        )
        if before_trace is not None
        else None
    )
    for replay_index in range(event_index + 1, decision_index):
        replay_rows = [
            pair["_rows_a"][replay_index],
            pair["_rows_b"][replay_index],
            pair["_rows_a"][replay_index],
            pair["_rows_b"][replay_index],
        ]
        updated = _apply_replay_events(
            model,
            memories,
            replay_rows,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            latent_trace=latent_trace,
        )
        if latent_trace is None:
            memories = updated
        else:
            memories, latent_trace = updated
    if latent_trace is not None:
        return memories[:2], memories[2:], latent_trace[:2], latent_trace[2:]
    return memories[:2], memories[2:]


def _causal_pair_loss(
    model: CLSTRModel,
    pair: dict[str, Any],
    factual_memories: torch.Tensor,
    intervened_memories: torch.Tensor,
    *,
    factual_latent_trace: torch.Tensor | None = None,
    intervened_latent_trace: torch.Tensor | None = None,
    state_cache: Any,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    margin: float,
    hard_fallback: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rows = [pair["_row_a"], pair["_row_b"]]
    states = state_cache.batch(
        [_causal_route_state_text(row) for row in rows],
        device=device,
    )
    current_states = state_cache.batch(
        [str(row["state_text_current"]) for row in rows],
        device=device,
    )
    legal = runtime_visible_mask(
        rows,
        skill_id_to_idx,
        device=device,
        inventory_catalogs=catalogs,
    )
    positive = _positive_mask(rows, skill_id_to_idx, device=device)
    static_memory = model.vnext_initial_belief(states, legal, top_k=belief_top_k)
    history = torch.ones(len(rows), dtype=torch.bool, device=device)
    history_depth = torch.full(
        (len(rows),),
        float(pair["_index"]),
        dtype=torch.float32,
        device=device,
    )
    (
        _factual_queries,
        factual_path,
        _factual_coarse_labels,
        factual_labels,
    ) = _memory_conditioned_natural_support(
        model,
        states,
        factual_memories,
        static_memory,
        history,
        legal,
        positive,
        current_states=current_states,
        latent_trace=factual_latent_trace,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
    )
    (
        _intervened_queries,
        intervened_path,
        _intervened_coarse_labels,
        intervened_labels,
    ) = _memory_conditioned_natural_support(
        model,
        states,
        intervened_memories,
        static_memory,
        history,
        legal,
        positive,
        current_states=current_states,
        latent_trace=intervened_latent_trace,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
    )
    (
        _static_queries,
        static_path,
        _static_coarse_labels,
        static_labels,
    ) = _memory_conditioned_natural_support(
        model,
        states,
        static_memory,
        static_memory,
        torch.zeros_like(history),
        legal,
        positive,
        current_states=current_states,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
    )
    factual_scores = model.vnext_safe_candidate_route_scores(
        states,
        factual_memories,
        static_memory,
        history,
        factual_path.support.candidate_ids,
        factual_path.support.valid_mask,
        static_candidate_ids=factual_path.coarse_candidate_ids,
        static_valid_mask=factual_path.coarse_valid_mask,
        history_depth=history_depth,
        memory_state=current_states,
        latent_trace=factual_latent_trace,
        hard_fallback=hard_fallback,
    )
    intervened_scores = model.vnext_safe_candidate_route_scores(
        states,
        intervened_memories,
        static_memory,
        history,
        intervened_path.support.candidate_ids,
        intervened_path.support.valid_mask,
        static_candidate_ids=intervened_path.coarse_candidate_ids,
        static_valid_mask=intervened_path.coarse_valid_mask,
        history_depth=history_depth,
        memory_state=current_states,
        latent_trace=intervened_latent_trace,
        hard_fallback=hard_fallback,
    )
    static_scores = model.vnext_safe_candidate_route_scores(
        states,
        static_memory,
        static_memory,
        torch.zeros_like(history),
        static_path.support.candidate_ids,
        static_path.support.valid_mask,
        static_candidate_ids=static_path.coarse_candidate_ids,
        static_valid_mask=static_path.coarse_valid_mask,
        history_depth=torch.zeros_like(history_depth),
        memory_state=current_states,
        hard_fallback=hard_fallback,
    )
    recall_factual_utility = ranking_utility(
        factual_path.recall_logits.float(),
        positive,
        legal,
    )
    recall_intervened_utility = ranking_utility(
        intervened_path.recall_logits.float(),
        positive,
        legal,
    )
    recall_static_utility = ranking_utility(
        static_path.recall_logits.float(),
        positive,
        legal,
    )
    recall_loss = paired_counterfactual_margin_loss(
        recall_factual_utility,
        recall_intervened_utility,
        margin=float(margin),
    )
    route_eligible = factual_labels.eligible_mask & intervened_labels.eligible_mask
    raw_route_factual_utility = torch.zeros_like(recall_factual_utility)
    raw_route_intervened_utility = torch.zeros_like(recall_factual_utility)
    fused_route_factual_utility = torch.zeros_like(recall_factual_utility)
    fused_route_intervened_utility = torch.zeros_like(recall_factual_utility)
    if bool(route_eligible.any().item()):
        factual_local = route_eligible.nonzero(as_tuple=False).view(-1)
        raw_factual_local_utility = ranking_utility(
            factual_scores.raw_dynamic_logits.index_select(0, factual_local),
            factual_labels.positive_mask.index_select(0, factual_local),
            factual_path.support.valid_mask.index_select(0, factual_local),
        )
        raw_intervened_local_utility = ranking_utility(
            intervened_scores.raw_dynamic_logits.index_select(0, factual_local),
            intervened_labels.positive_mask.index_select(0, factual_local),
            intervened_path.support.valid_mask.index_select(0, factual_local),
        )
        fused_factual_local_utility = ranking_utility(
            factual_scores.mixed_logits.index_select(0, factual_local),
            factual_labels.positive_mask.index_select(0, factual_local),
            factual_path.support.valid_mask.index_select(0, factual_local),
        )
        fused_intervened_local_utility = ranking_utility(
            intervened_scores.mixed_logits.index_select(0, factual_local),
            intervened_labels.positive_mask.index_select(0, factual_local),
            intervened_path.support.valid_mask.index_select(0, factual_local),
        )
        raw_route_factual_utility = raw_route_factual_utility.index_copy(
            0,
            factual_local,
            raw_factual_local_utility,
        )
        raw_route_intervened_utility = raw_route_intervened_utility.index_copy(
            0,
            factual_local,
            raw_intervened_local_utility,
        )
        fused_route_factual_utility = fused_route_factual_utility.index_copy(
            0,
            factual_local,
            fused_factual_local_utility,
        )
        fused_route_intervened_utility = fused_route_intervened_utility.index_copy(
            0,
            factual_local,
            fused_intervened_local_utility,
        )
        raw_route_loss = paired_counterfactual_margin_loss(
            raw_route_factual_utility,
            raw_route_intervened_utility,
            margin=float(margin),
            eligible_mask=route_eligible,
        )
        fused_route_loss = paired_counterfactual_margin_loss(
            fused_route_factual_utility,
            fused_route_intervened_utility,
            margin=float(margin),
            eligible_mask=route_eligible,
        )
        route_loss = 0.5 * (raw_route_loss + fused_route_loss)
    else:
        route_loss = factual_scores.raw_dynamic_logits.sum() * 0.0
    static_eligible = factual_labels.eligible_mask & static_labels.eligible_mask
    route_static_utility = torch.zeros_like(recall_factual_utility)
    if bool(static_eligible.any().item()):
        static_local = static_eligible.nonzero(as_tuple=False).view(-1)
        static_local_utility = ranking_utility(
            static_scores.static_logits.index_select(0, static_local),
            static_labels.positive_mask.index_select(0, static_local),
            static_path.support.valid_mask.index_select(0, static_local),
        )
        route_static_utility = route_static_utility.index_copy(
            0,
            static_local,
            static_local_utility,
        )
    loss = recall_loss + route_loss
    recall_advantage = recall_factual_utility - recall_intervened_utility
    raw_route_advantage = recall_advantage + torch.where(
        route_eligible,
        raw_route_factual_utility - raw_route_intervened_utility,
        torch.zeros_like(raw_route_factual_utility),
    )
    route_advantage = recall_advantage + torch.where(
        route_eligible,
        fused_route_factual_utility - fused_route_intervened_utility,
        torch.zeros_like(fused_route_factual_utility),
    )
    raw_route_dynamic_static_advantage = (
        recall_factual_utility - recall_static_utility
    ) + torch.where(
        static_eligible,
        raw_route_factual_utility - route_static_utility,
        torch.zeros_like(raw_route_factual_utility),
    )
    route_dynamic_static_advantage = (
        recall_factual_utility - recall_static_utility
    ) + torch.where(
        static_eligible,
        fused_route_factual_utility - route_static_utility,
        torch.zeros_like(fused_route_factual_utility),
    )
    return loss, {
        "route_advantage": route_advantage.detach(),
        "raw_route_advantage": raw_route_advantage.detach(),
        "recall_advantage": recall_advantage.detach(),
        "route_dynamic_static_advantage": route_dynamic_static_advantage.detach(),
        "raw_route_dynamic_static_advantage": (
            raw_route_dynamic_static_advantage.detach()
        ),
        "natural_support_eligible": route_eligible.detach(),
        "candidate_support_memory_invariant": torch.zeros_like(
            fused_route_factual_utility,
            dtype=torch.bool,
        ).detach(),
    }


def _stable_cap_pairs(
    pairs: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    ordered = sorted(pairs, key=lambda pair: str(pair.get("causal_pair_id") or ""))
    if limit is None or int(limit) <= 0 or len(ordered) <= int(limit):
        return ordered
    groups = _causal_pairs_by_source(ordered)
    positions = {source: 0 for source in groups}
    selected: list[dict[str, Any]] = []
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(groups):
            position = positions[source]
            if position >= len(groups[source]):
                continue
            selected.append(groups[source][position])
            positions[source] += 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def _causal_pairs_by_source(
    pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        row = pair.get("_row_a") or {}
        grouped[semantic_source_id(row)].append(pair)
    return {
        source: sorted(rows, key=lambda pair: str(pair.get("causal_pair_id") or ""))
        for source, rows in sorted(grouped.items())
    }


def _paired_mean_ci(
    values: list[float],
    *,
    seed: int,
    samples: int = 1000,
    cluster_ids: list[str] | None = None,
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "cluster_count": 0,
            "mean": 0.0,
            "pair_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    if cluster_ids is not None and len(cluster_ids) != len(values):
        raise ValueError("cluster bootstrap requires one cluster identity per value")
    if cluster_ids is None:
        unit_values = list(values)
    else:
        grouped: dict[str, list[float]] = defaultdict(list)
        for cluster_id, value in zip(cluster_ids, values):
            if not str(cluster_id):
                raise ValueError("cluster bootstrap received an empty cluster identity")
            grouped[str(cluster_id)].append(float(value))
        unit_values = [
            sum(grouped[cluster_id]) / len(grouped[cluster_id])
            for cluster_id in sorted(grouped)
        ]
    mean = sum(unit_values) / len(unit_values)
    pair_mean = sum(values) / len(values)
    if len(unit_values) == 1:
        return {
            "count": len(values),
            "cluster_count": 1,
            "mean": mean,
            "pair_mean": pair_mean,
            "ci_low": mean,
            "ci_high": mean,
        }
    rng = random.Random(int(seed))
    bootstrap = sorted(
        sum(unit_values[rng.randrange(len(unit_values))] for _ in unit_values)
        / len(unit_values)
        for _ in range(int(samples))
    )
    low_index = max(0, int(0.025 * len(bootstrap)))
    high_index = min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))
    return {
        "count": len(values),
        "cluster_count": len(unit_values),
        "mean": float(mean),
        "pair_mean": float(pair_mean),
        "ci_low": float(bootstrap[low_index]),
        "ci_high": float(bootstrap[high_index]),
    }


def _horizon_bucket(index: int) -> str:
    value = int(index)
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    if value <= 16:
        return "8-16"
    if value <= 32:
        return "17-32"
    return "33+"


def _ordinary_benchmark_family(source: str) -> str:
    value = str(source).strip().lower()
    for family, markers in (
        ("alfworld", ("alfworld",)),
        ("toolbench", ("toolbench",)),
        ("webshop", ("webshop",)),
        ("traject", ("traject",)),
        ("agentgym", ("agentgym",)),
    ):
        if any(marker in value for marker in markers):
            return family
    return value or "unknown"


def _build_ordinary_dev_anchors(
    trajectories: list[list[dict[str, Any]]],
    *,
    max_rows: int | None,
    max_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rows in trajectories:
        for target_index in range(1, len(rows)):
            row = rows[target_index]
            source = semantic_source_id(row)
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            if not trajectory_id:
                raise ValueError("ordinary-dev anchor lacks trajectory_id")
            identity = hashlib.sha256(
                f"{source}\0{trajectory_id}\0{target_index}".encode("utf-8")
            ).hexdigest()
            copied = dict(row)
            copied["_ordinary_anchor"] = {
                "identity": identity,
                "rows": rows,
                "target_index": int(target_index),
                "source": source,
                "family": _ordinary_benchmark_family(source),
                "horizon_bucket": _horizon_bucket(target_index),
                "trajectory_cluster_id": f"{source}:{trajectory_id}",
                "pool_size": int(row.get("inventory_pool_size") or 0),
            }
            candidates.append(copied)
    selected_rows, sampling = stable_stratified_cap_rows(
        candidates,
        max_rows,
        extra_stratum=lambda row: str(
            (row.get("_ordinary_anchor") or {}).get("horizon_bucket") or ""
        ),
    )
    anchors = [dict(row["_ordinary_anchor"]) for row in selected_rows]
    sampling["anchor_identity_sha256"] = hashlib.sha256(
        "\n".join(anchor["identity"] for anchor in anchors).encode("ascii")
    ).hexdigest()
    sampling["max_horizon"] = int(max_horizon)
    sampling["differentiable_horizon_limit"] = int(max_horizon)
    sampling["evaluation_prefix_protocol"] = "all_prefixes_with_long_drift_buckets_v1"
    sampling["maximum_evaluated_prefix_index"] = max(
        (int(anchor["target_index"]) for anchor in anchors),
        default=0,
    )
    return anchors, sampling


def _ordinary_metric_report(
    records: list[dict[str, Any]],
    *,
    seed: int,
    recall_ms: tuple[int, ...],
) -> dict[str, Any]:
    clusters = [str(record["cluster_id"]) for record in records]
    static_candidate_hits = [
        float(record["candidate_hit@500"]) for record in records
    ]
    union_candidate_hits = [
        float(record["candidate_hit@union"]) for record in records
    ]
    union_minus_static_hits = [
        union - static
        for union, static in zip(union_candidate_hits, static_candidate_hits)
    ]
    static_candidate_recall = _paired_mean_ci(
        static_candidate_hits,
        seed=int(seed) + 29,
        cluster_ids=clusters,
    )
    report: dict[str, Any] = {
        "row_count": len(records),
        "cluster_count": len(set(clusters)),
        "route_mrr_delta": _paired_mean_ci(
            [float(record["route_mrr_delta"]) for record in records],
            seed=int(seed) + 17,
            cluster_ids=clusters,
        ),
        "raw_route_mrr_delta": _paired_mean_ci(
            [float(record["raw_route_mrr_delta"]) for record in records],
            seed=int(seed) + 23,
            cluster_ids=clusters,
        ),
        "same_support_route_mrr_delta": _paired_mean_ci(
            [float(record["same_support_route_mrr_delta"]) for record in records],
            seed=int(seed) + 24,
            cluster_ids=clusters,
        ),
        "candidate_extension_mrr_delta": _paired_mean_ci(
            [float(record["candidate_extension_mrr_delta"]) for record in records],
            seed=int(seed) + 25,
            cluster_ids=clusters,
        ),
        "static_route_mrr": _paired_mean_ci(
            [float(record["static_route_mrr"]) for record in records],
            seed=int(seed) + 26,
            cluster_ids=clusters,
        ),
        "same_support_dynamic_route_mrr": _paired_mean_ci(
            [float(record["same_support_dynamic_route_mrr"]) for record in records],
            seed=int(seed) + 27,
            cluster_ids=clusters,
        ),
        "dynamic_route_mrr": _paired_mean_ci(
            [float(record["dynamic_route_mrr"]) for record in records],
            seed=int(seed) + 28,
            cluster_ids=clusters,
        ),
        "raw_dynamic_route_mrr": _paired_mean_ci(
            [float(record["raw_dynamic_route_mrr"]) for record in records],
            seed=int(seed) + 29,
            cluster_ids=clusters,
        ),
        "mixture_probability": {
            "mean": (
                sum(float(record["mixture_probability"]) for record in records)
                / len(records)
                if records
                else 0.0
            ),
            "nonzero_rate": (
                sum(
                    int(float(record["mixture_probability"]) > 0.0)
                    for record in records
                )
                / len(records)
                if records
                else 0.0
            ),
            "at_least_half_rate": (
                sum(
                    int(float(record["mixture_probability"]) >= 0.5)
                    for record in records
                )
                / len(records)
                if records
                else 0.0
            ),
        },
        "selector_probability": {
            "mean": (
                sum(
                    float(
                        record.get(
                            "selector_probability",
                            record["mixture_probability"],
                        )
                    )
                    for record in records
                )
                / len(records)
                if records
                else 0.0
            ),
            "at_least_threshold_rate": (
                sum(
                    int(
                        float(
                            record.get(
                                "selector_probability",
                                record["mixture_probability"],
                            )
                        )
                        >= 0.55
                    )
                    for record in records
                )
                / len(records)
                if records
                else 0.0
            ),
        },
        "fixed_causal_recall_mean_rank": (
            sum(float(record["fixed_causal_recall_rank"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "dynamic_recall_mean_rank": (
            sum(float(record["dynamic_recall_rank"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "dynamic_route_mean_rank": (
            sum(float(record["dynamic_route_rank"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "raw_dynamic_route_mean_rank": (
            sum(float(record["raw_dynamic_route_rank"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "same_support_dynamic_route_mean_rank": (
            sum(
                float(record["same_support_dynamic_route_rank"])
                for record in records
            )
            / len(records)
            if records
            else 0.0
        ),
        "static_route_mean_rank": (
            sum(float(record["static_route_rank"]) for record in records)
            / len(records)
            if records
            else 0.0
        ),
        # Compatibility name retained for existing selection artifacts. The
        # coarse IDs are the preserved frozen-static Top500, not the recurrent
        # query's standalone Top500 diagnostic.
        "candidate_recall_at_500": static_candidate_recall,
        "static_candidate_recall_at_500": static_candidate_recall,
        "candidate_union_recall": _paired_mean_ci(
            union_candidate_hits,
            seed=int(seed) + 37,
            cluster_ids=clusters,
        ),
        "candidate_union_minus_static_recall": _paired_mean_ci(
            union_minus_static_hits,
            seed=int(seed) + 41,
            cluster_ids=clusters,
        ),
        "candidate_support_change_rate": _paired_mean_ci(
            [float(record["candidate_support_changed"]) for record in records],
            seed=int(seed) + 43,
            cluster_ids=clusters,
        ),
    }
    # MRR alone over-rewards a small set of rank-1 wins and cannot establish
    # whether a recurrent route actually improves the deployment Top-K
    # boundary.  Persist the exact per-expert R@1/R@5/R@10 evidence used by the
    # open-pool release selector.  These values are derived only from natural
    # support ranks produced by the held-out evaluator.
    for offset, k_value in enumerate((1, 5, 10)):
        static_hits = [
            float(int(record["static_route_rank"]) <= int(k_value))
            for record in records
        ]
        same_support_hits = [
            float(
                int(record["same_support_dynamic_route_rank"])
                <= int(k_value)
            )
            for record in records
        ]
        dynamic_hits = [
            float(int(record["dynamic_route_rank"]) <= int(k_value))
            for record in records
        ]
        raw_dynamic_hits = [
            float(int(record["raw_dynamic_route_rank"]) <= int(k_value))
            for record in records
        ]
        route_deltas = [
            dynamic - static
            for dynamic, static in zip(dynamic_hits, static_hits)
        ]
        raw_route_deltas = [
            dynamic - static
            for dynamic, static in zip(raw_dynamic_hits, static_hits)
        ]
        same_support_deltas = [
            dynamic - static
            for dynamic, static in zip(same_support_hits, static_hits)
        ]
        suffix = f"at_{int(k_value)}"
        report[f"static_route_recall_{suffix}"] = _paired_mean_ci(
            static_hits,
            seed=int(seed) + 801 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"same_support_dynamic_route_recall_{suffix}"] = _paired_mean_ci(
            same_support_hits,
            seed=int(seed) + 802 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"dynamic_route_recall_{suffix}"] = _paired_mean_ci(
            dynamic_hits,
            seed=int(seed) + 803 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"raw_dynamic_route_recall_{suffix}"] = _paired_mean_ci(
            raw_dynamic_hits,
            seed=int(seed) + 804 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"route_recall_{suffix}_delta"] = _paired_mean_ci(
            route_deltas,
            seed=int(seed) + 805 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"raw_route_recall_{suffix}_delta"] = _paired_mean_ci(
            raw_route_deltas,
            seed=int(seed) + 806 + 10 * offset,
            cluster_ids=clusters,
        )
        report[f"same_support_route_recall_{suffix}_delta"] = _paired_mean_ci(
            same_support_deltas,
            seed=int(seed) + 807 + 10 * offset,
            cluster_ids=clusters,
        )
    candidate_recall: dict[str, Any] = {}
    deployment_pool_floor = max(int(value) for value in recall_ms)
    for offset, m_value in enumerate(recall_ms):
        causal_hits = [float(record[f"fixed_hit@{m_value}"]) for record in records]
        dynamic_hits = [float(record[f"dynamic_hit@{m_value}"]) for record in records]
        recall_deltas = [
            dynamic - static
            for dynamic, static in zip(dynamic_hits, causal_hits)
        ]
        nonsaturated_records = [
            record for record in records if int(record.get("pool_size") or 0) > int(m_value)
        ]
        nonsaturated_clusters = [
            str(record["cluster_id"]) for record in nonsaturated_records
        ]
        deployment_pool_records = [
            record
            for record in records
            if int(record.get("pool_size") or 0) > deployment_pool_floor
        ]
        deployment_pool_clusters = [
            str(record["cluster_id"]) for record in deployment_pool_records
        ]
        deployment_union_hits = [
            float(record["candidate_hit@union"])
            for record in deployment_pool_records
        ]
        deployment_fixed_hits = [
            float(record[f"fixed_hit@{m_value}"])
            for record in deployment_pool_records
        ]
        candidate_recall[str(m_value)] = {
            "fixed_causal_hit_rate": _paired_mean_ci(
                causal_hits,
                seed=int(seed) + 51 + offset,
                cluster_ids=clusters,
            ),
            "dynamic_hit_rate": _paired_mean_ci(
                dynamic_hits,
                seed=int(seed) + 151 + offset,
                cluster_ids=clusters,
            ),
            "dynamic_minus_static": _paired_mean_ci(
                recall_deltas,
                seed=int(seed) + 201 + offset,
                cluster_ids=clusters,
            ),
            "nonsaturated": {
                "row_count": len(nonsaturated_records),
                "cluster_count": len(set(nonsaturated_clusters)),
                "fixed_causal_hit_rate": _paired_mean_ci(
                    [
                        float(record[f"fixed_hit@{m_value}"])
                        for record in nonsaturated_records
                    ],
                    seed=int(seed) + 301 + offset,
                    cluster_ids=nonsaturated_clusters,
                ),
                "dynamic_hit_rate": _paired_mean_ci(
                    [
                        float(record[f"dynamic_hit@{m_value}"])
                        for record in nonsaturated_records
                    ],
                    seed=int(seed) + 401 + offset,
                    cluster_ids=nonsaturated_clusters,
                ),
            },
            "deployment_full_pool": {
                "minimum_pool_size_exclusive": int(deployment_pool_floor),
                "row_count": len(deployment_pool_records),
                "cluster_count": len(set(deployment_pool_clusters)),
                "fixed_causal_hit_rate": _paired_mean_ci(
                    [
                        float(record[f"fixed_hit@{m_value}"])
                        for record in deployment_pool_records
                    ],
                    seed=int(seed) + 551 + offset,
                    cluster_ids=deployment_pool_clusters,
                ),
                "dynamic_hit_rate": _paired_mean_ci(
                    [
                        float(record[f"dynamic_hit@{m_value}"])
                        for record in deployment_pool_records
                    ],
                    seed=int(seed) + 651 + offset,
                    cluster_ids=deployment_pool_clusters,
                ),
                "dynamic_minus_static": _paired_mean_ci(
                    [
                        float(record[f"dynamic_hit@{m_value}"])
                        - float(record[f"fixed_hit@{m_value}"])
                        for record in deployment_pool_records
                    ],
                    seed=int(seed) + 751 + offset,
                    cluster_ids=deployment_pool_clusters,
                ),
                "deployed_union_hit_rate": _paired_mean_ci(
                    deployment_union_hits,
                    seed=int(seed) + 851 + offset,
                    cluster_ids=deployment_pool_clusters,
                ),
                "deployed_union_minus_fixed": _paired_mean_ci(
                    [
                        union - fixed
                        for union, fixed in zip(
                            deployment_union_hits,
                            deployment_fixed_hits,
                        )
                    ],
                    seed=int(seed) + 951 + offset,
                    cluster_ids=deployment_pool_clusters,
                ),
            },
        }
    report["candidate_recall"] = candidate_recall
    return report


@torch.no_grad()
def _evaluate_stage2_ordinary_dev(
    model: CLSTRModel,
    *,
    anchors: list[dict[str, Any]],
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    final_k: int,
    batch_size: int,
    seed: int,
    recall_ms: tuple[int, ...] = (100, 500),
) -> dict[str, Any]:
    if not anchors:
        raise ValueError("Stage2 ordinary-dev validation requires fixed anchors")
    if int(batch_size) <= 0:
        raise ValueError("ordinary-dev validation batch size must be positive")
    records: list[dict[str, Any]] = []
    for start in range(0, len(anchors), int(batch_size)):
        batch = anchors[start : start + int(batch_size)]
        segments = [
            (
                anchor["rows"],
                int(anchor["target_index"]),
                int(anchor["target_index"]) + 1,
                False,
            )
            for anchor in batch
        ]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            replayed = _batch_memory_at_start(
                model,
                segments,
                state_cache=state_cache,
                action_cache=action_cache,
                result_cache=result_cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                with_trace=bool(model.vnext.synchronization_enabled),
            )
            if model.vnext.synchronization_enabled:
                memories, latent_trace = replayed
            else:
                memories = replayed
                latent_trace = None
        rows = [anchor["rows"][int(anchor["target_index"])] for anchor in batch]
        states = state_cache.batch(
            [_causal_route_state_text(row) for row in rows],
            device=device,
        )
        current_states = state_cache.batch(
            [str(row["state_text_current"]) for row in rows],
            device=device,
        )
        legal = runtime_visible_mask(
            rows,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        positive = _positive_mask(rows, skill_id_to_idx, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            static_memory = model.vnext_initial_belief(
                states,
                legal,
                top_k=belief_top_k,
            )
            history = torch.ones(len(rows), dtype=torch.bool, device=device)
            history_depth = torch.tensor(
                [float(anchor["target_index"]) for anchor in batch],
                dtype=torch.float32,
                device=device,
            )
            (
                _dynamic_queries,
                dynamic_path,
                _dynamic_coarse_labels,
                dynamic_support_labels,
            ) = _memory_conditioned_natural_support(
                model,
                states,
                memories,
                static_memory,
                history,
                legal,
                positive,
                current_states=current_states,
                latent_trace=latent_trace,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
            )
            (
                _static_queries,
                static_path,
                _static_coarse_labels,
                static_support_labels,
            ) = _memory_conditioned_natural_support(
                model,
                states,
                static_memory,
                static_memory,
                torch.zeros_like(history),
                legal,
                positive,
                current_states=current_states,
                latent_trace=latent_trace,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
            )
            # Identical BF16 queries can otherwise produce different Top-K
            # members at an exact score tie across two independent CUDA
            # kernels. Reuse the same natural path so step-zero equality is
            # measured semantically rather than as a kernel tie-break artifact.
            if torch.equal(
                _dynamic_queries.dynamic_recall,
                _static_queries.static_recall,
            ):
                dynamic_path = static_path
                _dynamic_coarse_labels = _static_coarse_labels
                dynamic_support_labels = static_support_labels
            dynamic_scores = model.vnext_safe_candidate_route_scores(
                states,
                memories,
                static_memory,
                history,
                dynamic_path.support.candidate_ids,
                dynamic_path.support.valid_mask,
                static_candidate_ids=dynamic_path.coarse_candidate_ids,
                static_valid_mask=dynamic_path.coarse_valid_mask,
                history_depth=history_depth,
                memory_state=current_states,
                latent_trace=latent_trace,
                hard_fallback=True,
            )
            static_scores = model.vnext_safe_candidate_route_scores(
                states,
                static_memory,
                static_memory,
                torch.zeros_like(history),
                static_path.support.candidate_ids,
                static_path.support.valid_mask,
                static_candidate_ids=static_path.coarse_candidate_ids,
                static_valid_mask=static_path.coarse_valid_mask,
                history_depth=torch.zeros_like(history_depth),
                memory_state=current_states,
                hard_fallback=False,
            )
            same_support_dynamic_scores = model.vnext_safe_candidate_route_scores(
                states,
                memories,
                static_memory,
                history,
                static_path.support.candidate_ids,
                static_path.support.valid_mask,
                static_candidate_ids=static_path.coarse_candidate_ids,
                static_valid_mask=static_path.coarse_valid_mask,
                history_depth=history_depth,
                memory_state=current_states,
                latent_trace=latent_trace,
                hard_fallback=True,
            )
        static_recall_rank = _best_positive_rank(
            static_path.recall_logits,
            positive,
            legal,
        )
        dynamic_recall_rank = _best_positive_rank(
            dynamic_path.recall_logits,
            positive,
            legal,
        )
        dynamic_route_rank = _best_positive_rank(
            dynamic_scores.mixed_logits,
            dynamic_support_labels.positive_mask,
            dynamic_path.support.valid_mask,
        )
        raw_dynamic_route_rank = _best_positive_rank(
            dynamic_scores.raw_dynamic_logits,
            dynamic_support_labels.positive_mask,
            dynamic_path.support.valid_mask,
        )
        static_route_rank = _best_positive_rank(
            static_scores.static_logits,
            static_support_labels.positive_mask,
            static_path.support.valid_mask,
        )
        same_support_dynamic_route_rank = _best_positive_rank(
            same_support_dynamic_scores.mixed_logits,
            static_support_labels.positive_mask,
            static_path.support.valid_mask,
        )
        for index, anchor in enumerate(batch):
            fixed_recall_value = int(static_recall_rank[index].cpu().item())
            dynamic_recall_value = int(dynamic_recall_rank[index].cpu().item())
            dynamic_route_value = int(dynamic_route_rank[index].cpu().item())
            raw_dynamic_route_value = int(
                raw_dynamic_route_rank[index].cpu().item()
            )
            static_route_value = int(static_route_rank[index].cpu().item())
            same_support_dynamic_route_value = int(
                same_support_dynamic_route_rank[index].cpu().item()
            )
            dynamic_union_hit = bool(
                dynamic_support_labels.positive_mask[index].any().item()
            )
            static_union_hit = bool(
                static_support_labels.positive_mask[index].any().item()
            )
            dynamic_support_hit = dynamic_union_hit and dynamic_route_value <= int(final_k)
            static_support_hit = static_union_hit and static_route_value <= int(final_k)
            raw_dynamic_support_hit = (
                dynamic_union_hit and raw_dynamic_route_value <= int(final_k)
            )
            dynamic_route_mrr = (
                1.0 / dynamic_route_value if dynamic_support_hit else 0.0
            )
            raw_dynamic_route_mrr = (
                1.0 / raw_dynamic_route_value if raw_dynamic_support_hit else 0.0
            )
            static_route_mrr = (
                1.0 / static_route_value if static_support_hit else 0.0
            )
            same_support_dynamic_hit = (
                static_union_hit
                and same_support_dynamic_route_value <= int(final_k)
            )
            same_support_dynamic_route_mrr = (
                1.0 / same_support_dynamic_route_value
                if same_support_dynamic_hit
                else 0.0
            )
            dynamic_members = set(
                dynamic_path.support.candidate_ids[index]
                .masked_select(dynamic_path.support.valid_mask[index])
                .detach()
                .cpu()
                .tolist()
            )
            static_members = set(
                static_path.support.candidate_ids[index]
                .masked_select(static_path.support.valid_mask[index])
                .detach()
                .cpu()
                .tolist()
            )
            record: dict[str, Any] = {
                "source": str(anchor["source"]),
                "family": str(anchor["family"]),
                "horizon_bucket": str(anchor["horizon_bucket"]),
                "cluster_id": str(anchor["trajectory_cluster_id"]),
                "pool_size": int(anchor["pool_size"]),
                "fixed_causal_recall_rank": fixed_recall_value,
                "dynamic_recall_rank": dynamic_recall_value,
                "dynamic_route_rank": dynamic_route_value,
                "raw_dynamic_route_rank": raw_dynamic_route_value,
                "same_support_dynamic_route_rank": (
                    same_support_dynamic_route_value
                ),
                "static_route_rank": static_route_value,
                "route_mrr_delta": dynamic_route_mrr - static_route_mrr,
                "raw_route_mrr_delta": (
                    raw_dynamic_route_mrr - static_route_mrr
                ),
                "same_support_route_mrr_delta": (
                    same_support_dynamic_route_mrr - static_route_mrr
                ),
                "candidate_extension_mrr_delta": (
                    dynamic_route_mrr - same_support_dynamic_route_mrr
                ),
                "static_route_mrr": static_route_mrr,
                "same_support_dynamic_route_mrr": same_support_dynamic_route_mrr,
                "dynamic_route_mrr": dynamic_route_mrr,
                "raw_dynamic_route_mrr": raw_dynamic_route_mrr,
                "mixture_probability": float(
                    dynamic_scores.mixture_probability[index].float().cpu().item()
                ),
                "selector_probability": float(
                    dynamic_scores.selector_probability[index].float().cpu().item()
                ),
                "same_support_mixture_probability": float(
                    same_support_dynamic_scores.mixture_probability[index]
                    .float()
                    .cpu()
                    .item()
                ),
                "same_support_selector_probability": float(
                    same_support_dynamic_scores.selector_probability[index]
                    .float()
                    .cpu()
                    .item()
                ),
                "candidate_support_changed": float(
                    dynamic_members != static_members
                ),
                "candidate_hit@500": float(
                    bool(
                        _dynamic_coarse_labels.positive_mask[index].any().item()
                    )
                ),
                "candidate_hit@union": float(
                    dynamic_union_hit
                ),
                "candidate_hit@final": float(dynamic_support_hit),
            }
            for m_value in recall_ms:
                record[f"fixed_hit@{m_value}"] = float(
                    fixed_recall_value <= int(m_value)
                )
                record[f"dynamic_hit@{m_value}"] = float(
                    dynamic_recall_value <= int(m_value)
                )
            records.append(record)
    return {
        "protocol": "static_top500_dynamic_top500_diff64_direct_unified_finalk_ordinary_dev_v2",
        "coarse_k": int(coarse_k),
        "dynamic_extra_k": int(compressed_m),
        "final_k": int(final_k),
        "candidate_support_memory_invariant": False,
        "recall_ms": [int(value) for value in recall_ms],
        "overall": _ordinary_metric_report(
            records,
            seed=int(seed) + 1001,
            recall_ms=recall_ms,
        ),
        "per_source": {
            source: _ordinary_metric_report(
                [record for record in records if record["source"] == source],
                seed=int(seed) + 2001 + source_index,
                recall_ms=recall_ms,
            )
            for source_index, source in enumerate(
                sorted({str(record["source"]) for record in records})
            )
        },
        "per_family": {
            family: _ordinary_metric_report(
                [record for record in records if record["family"] == family],
                seed=int(seed) + 2501 + family_index,
                recall_ms=recall_ms,
            )
            for family_index, family in enumerate(
                sorted({str(record["family"]) for record in records})
            )
        },
        "per_horizon": {
            bucket: _ordinary_metric_report(
                [record for record in records if record["horizon_bucket"] == bucket],
                seed=int(seed) + 3001 + bucket_index,
                recall_ms=recall_ms,
            )
            for bucket_index, bucket in enumerate(
                sorted({str(record["horizon_bucket"]) for record in records})
            )
        },
    }


def _stable_cap_anchors(
    anchors: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    ordered = sorted(anchors, key=lambda anchor: str(anchor.get("identity") or ""))
    if limit is None or int(limit) <= 0 or len(ordered) <= int(limit):
        return ordered
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in ordered:
        grouped[str(anchor.get("source") or "")].append(anchor)
    positions = {source: 0 for source in grouped}
    selected: list[dict[str, Any]] = []
    while len(selected) < int(limit):
        progressed = False
        for source in sorted(grouped):
            position = positions[source]
            if position >= len(grouped[source]):
                continue
            selected.append(grouped[source][position])
            positions[source] += 1
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


@torch.no_grad()
def _evaluate_robust_prefix_dev(
    model: CLSTRModel,
    *,
    anchors_by_kind: dict[str, list[dict[str, Any]]],
    max_rows_per_kind: int | None,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    final_k: int,
    seed: int,
) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    all_deltas: list[float] = []
    all_raw_deltas: list[float] = []
    all_clusters: list[str] = []
    source_deltas: dict[str, list[float]] = defaultdict(list)
    source_clusters: dict[str, list[str]] = defaultdict(list)
    for kind_index, kind in enumerate(
        ("one_error_prefix", "two_or_more_error_prefix", "recovery_prefix")
    ):
        anchors = _stable_cap_anchors(anchors_by_kind.get(kind) or [], max_rows_per_kind)
        if not anchors:
            by_kind[kind] = {
                "row_count": 0,
                "dynamic_route_mrr": 0.0,
                "raw_dynamic_route_mrr": 0.0,
                "static_route_mrr": 0.0,
                "route_mrr_delta": _paired_mean_ci(
                    [],
                    seed=int(seed) + kind_index,
                    cluster_ids=[],
                ),
                "raw_route_mrr_delta": _paired_mean_ci(
                    [],
                    seed=int(seed) + 1000 + kind_index,
                    cluster_ids=[],
                ),
                "candidate_support_memory_invariant": False,
            }
            continue
        segments = [
            (
                anchor["rows"],
                int(anchor["target_index"]),
                int(anchor["target_index"]) + 1,
                True,
            )
            for anchor in anchors
        ]
        replayed = _batch_memory_at_start(
            model,
            segments,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            catalogs=catalogs,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            with_trace=bool(model.vnext.synchronization_enabled),
        )
        if model.vnext.synchronization_enabled:
            memories, latent_trace = replayed
        else:
            memories = replayed
            latent_trace = None
        rows = [anchor["rows"][int(anchor["target_index"])] for anchor in anchors]
        states = state_cache.batch(
            [_causal_route_state_text(row) for row in rows],
            device=device,
        )
        current_states = state_cache.batch(
            [str(row["state_text_current"]) for row in rows],
            device=device,
        )
        legal = runtime_visible_mask(
            rows,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        positive = torch.zeros_like(legal)
        target_indices = torch.tensor(
            [skill_id_to_idx[row["_vnext_target_skill_id"]] for row in rows],
            dtype=torch.long,
            device=device,
        )
        positive.scatter_(1, target_indices.unsqueeze(-1), True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            static_memory = model.vnext_initial_belief(
                states,
                legal,
                top_k=belief_top_k,
            )
            history = torch.ones(len(rows), dtype=torch.bool, device=device)
            history_depth = torch.tensor(
                [float(anchor["target_index"]) for anchor in anchors],
                dtype=torch.float32,
                device=device,
            )
            (
                _dynamic_queries,
                dynamic_path,
                _dynamic_coarse_labels,
                dynamic_labels,
            ) = _memory_conditioned_natural_support(
                model,
                states,
                memories,
                static_memory,
                history,
                legal,
                positive,
                current_states=current_states,
                latent_trace=latent_trace,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
            )
            (
                _static_queries,
                static_path,
                _static_coarse_labels,
                static_labels,
            ) = _memory_conditioned_natural_support(
                model,
                states,
                static_memory,
                static_memory,
                torch.zeros_like(history),
                legal,
                positive,
                current_states=current_states,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
            )
            if torch.equal(
                _dynamic_queries.dynamic_recall,
                _static_queries.static_recall,
            ):
                dynamic_path = static_path
                _dynamic_coarse_labels = _static_coarse_labels
                dynamic_labels = static_labels
            dynamic_scores = model.vnext_safe_candidate_route_scores(
                states,
                memories,
                static_memory,
                history,
                dynamic_path.support.candidate_ids,
                dynamic_path.support.valid_mask,
                static_candidate_ids=dynamic_path.coarse_candidate_ids,
                static_valid_mask=dynamic_path.coarse_valid_mask,
                history_depth=history_depth,
                memory_state=current_states,
                latent_trace=latent_trace,
                hard_fallback=True,
            )
            static_scores = model.vnext_safe_candidate_route_scores(
                states,
                static_memory,
                static_memory,
                torch.zeros_like(history),
                static_path.support.candidate_ids,
                static_path.support.valid_mask,
                static_candidate_ids=static_path.coarse_candidate_ids,
                static_valid_mask=static_path.coarse_valid_mask,
                history_depth=torch.zeros_like(history_depth),
                memory_state=current_states,
                hard_fallback=False,
            )
        dynamic_route_rank = _best_positive_rank(
            dynamic_scores.mixed_logits,
            dynamic_labels.positive_mask,
            dynamic_path.support.valid_mask,
        )
        raw_dynamic_route_rank = _best_positive_rank(
            dynamic_scores.raw_dynamic_logits,
            dynamic_labels.positive_mask,
            dynamic_path.support.valid_mask,
        )
        static_route_rank = _best_positive_rank(
            static_scores.static_logits,
            static_labels.positive_mask,
            static_path.support.valid_mask,
        )
        dynamic_union_hit = dynamic_labels.positive_mask.any(dim=-1)
        static_union_hit = static_labels.positive_mask.any(dim=-1)
        dynamic_hit = dynamic_union_hit & dynamic_route_rank.le(int(final_k))
        raw_dynamic_hit = dynamic_union_hit & raw_dynamic_route_rank.le(int(final_k))
        static_hit = static_union_hit & static_route_rank.le(int(final_k))
        dynamic_mrr = torch.where(
            raw_dynamic_hit,
            dynamic_route_rank.float().reciprocal(),
            torch.zeros_like(dynamic_route_rank, dtype=torch.float32),
        )
        raw_dynamic_mrr = torch.where(
            dynamic_hit,
            raw_dynamic_route_rank.float().reciprocal(),
            torch.zeros_like(raw_dynamic_route_rank, dtype=torch.float32),
        )
        static_mrr = torch.where(
            static_hit,
            static_route_rank.float().reciprocal(),
            torch.zeros_like(static_route_rank, dtype=torch.float32),
        )
        deltas = (dynamic_mrr - static_mrr).cpu().tolist()
        raw_deltas = (raw_dynamic_mrr - static_mrr).cpu().tolist()
        clusters = [
            str(row.get("split_group_identity") or row.get("trajectory_id") or "")
            for row in rows
        ]
        by_kind[kind] = {
            "row_count": len(rows),
            "dynamic_route_mrr": float(dynamic_mrr.mean().cpu().item()),
            "raw_dynamic_route_mrr": float(
                raw_dynamic_mrr.mean().cpu().item()
            ),
            "static_route_mrr": float(static_mrr.mean().cpu().item()),
            "route_mrr_delta": _paired_mean_ci(
                [float(value) for value in deltas],
                seed=int(seed) + 3001 + kind_index,
                cluster_ids=clusters,
            ),
            "raw_route_mrr_delta": _paired_mean_ci(
                [float(value) for value in raw_deltas],
                seed=int(seed) + 4001 + kind_index,
                cluster_ids=clusters,
            ),
            "candidate_support_memory_invariant": False,
        }
        all_deltas.extend(float(value) for value in deltas)
        all_raw_deltas.extend(float(value) for value in raw_deltas)
        all_clusters.extend(clusters)
        for anchor, delta, cluster in zip(anchors, deltas, clusters):
            source = str(anchor.get("source") or "")
            source_deltas[source].append(float(delta))
            source_clusters[source].append(cluster)
    return {
        "by_kind": by_kind,
        "route_mrr_delta": _paired_mean_ci(
            all_deltas,
            seed=int(seed) + 5001,
            cluster_ids=all_clusters,
        ),
        "raw_route_mrr_delta": _paired_mean_ci(
            all_raw_deltas,
            seed=int(seed) + 5501,
            cluster_ids=all_clusters,
        ),
        "candidate_support_memory_invariant": False,
        "per_source": {
            source: _paired_mean_ci(
                values,
                seed=int(seed) + 6001 + source_index,
                cluster_ids=source_clusters[source],
            )
            for source_index, (source, values) in enumerate(
                sorted(source_deltas.items())
            )
        },
    }


def _gradient_health(model: CLSTRModel) -> dict[str, Any]:
    groups = {
        "transition_delta": "vnext.transition_delta.",
        "correction_delta": "vnext.correction_delta.",
        "memory_recall_query": "vnext.memory_recall_query.",
        "unified_route_query": "vnext.unified_route_query.",
        "route_skill_adapter": "vnext.route_skill_adapter.",
        "route_expert_mixture": "vnext.route_expert_mixture.",
        "correction_gate": "vnext.correction_gate.",
    }
    if bool(model.vnext.synchronization_enabled):
        groups["synchronization"] = "vnext.synchronization."
    report: dict[str, Any] = {}
    for group, prefix in groups.items():
        gradients = [
            parameter.grad.detach().float()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
        norm = (
            float(torch.stack([grad.norm() for grad in gradients]).norm().cpu().item())
            if gradients and finite
            else 0.0
        )
        report[group] = {
            "tensor_count": len(gradients),
            "finite": finite,
            "norm": norm,
        }
    return report


def _empty_gradient_interval(
    start_step: int,
    *,
    synchronization_enabled: bool = False,
) -> dict[str, Any]:
    module_names = [
        "transition_delta",
        "correction_delta",
        "memory_recall_query",
        "unified_route_query",
        "route_skill_adapter",
        "route_expert_mixture",
        "correction_gate",
    ]
    if bool(synchronization_enabled):
        module_names.append("synchronization")
    return {
        "start_step": int(start_step),
        "end_step": None,
        "batch_count": 0,
        "modules": {
            name: {
                "observed_batch_count": 0,
                "max_tensor_count": 0,
                "all_finite": True,
                "max_norm": 0.0,
            }
            for name in module_names
        },
    }


def _update_gradient_interval(
    interval: dict[str, Any],
    observation: dict[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    updated = {
        "start_step": int(interval.get("start_step") or step),
        "end_step": int(step),
        "batch_count": int(interval.get("batch_count") or 0) + 1,
        "modules": {
            name: dict(values)
            for name, values in (interval.get("modules") or {}).items()
        },
    }
    for name, values in updated["modules"].items():
        current = observation.get(name) if isinstance(observation.get(name), dict) else {}
        tensor_count = int(current.get("tensor_count") or 0)
        if tensor_count <= 0:
            continue
        values["observed_batch_count"] = int(values.get("observed_batch_count") or 0) + 1
        values["max_tensor_count"] = max(
            int(values.get("max_tensor_count") or 0),
            tensor_count,
        )
        values["all_finite"] = bool(values.get("all_finite", True)) and bool(
            current.get("finite")
        )
        values["max_norm"] = max(
            float(values.get("max_norm") or 0.0),
            float(current.get("norm") or 0.0),
        )
    return updated


def _finalize_gradient_interval(interval: dict[str, Any], *, end_step: int) -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name, values in (interval.get("modules") or {}).items():
        observed = int(values.get("observed_batch_count") or 0)
        modules[name] = {
            "tensor_count": int(values.get("max_tensor_count") or 0),
            "finite": bool(observed > 0 and values.get("all_finite", True)),
            "norm": float(values.get("max_norm") or 0.0),
            "observed_batch_count": observed,
        }
    return {
        "protocol": "validation_interval_gradient_evidence_v1",
        "step": int(end_step),
        "start_step": int(interval.get("start_step") or end_step),
        "end_step": int(end_step),
        "batch_count": int(interval.get("batch_count") or 0),
        "modules": modules,
    }


@torch.no_grad()
def _evaluate_stage2_causal_dev(
    model: CLSTRModel,
    *,
    causal_pairs: dict[str, list[dict[str, Any]]],
    robust_prefix_anchors: dict[str, list[dict[str, Any]]],
    max_pairs_per_kind: int | None,
    max_robust_rows_per_kind: int | None,
    max_horizon: int,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    final_k: int,
    margin: float,
    seed: int,
    selection_pair_weights: dict[str, float],
    include_robust_in_selection: bool,
) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    all_causal_advantages: list[float] = []
    all_causal_clusters: list[str] = []
    all_routed_causal_advantages: list[float] = []
    all_routed_causal_clusters: list[str] = []
    all_dynamic_static_advantages: list[float] = []
    all_dynamic_static_clusters: list[str] = []
    source_causal_advantages: dict[str, list[float]] = defaultdict(list)
    source_causal_clusters: dict[str, list[str]] = defaultdict(list)
    source_routed_causal_advantages: dict[str, list[float]] = defaultdict(list)
    source_routed_causal_clusters: dict[str, list[str]] = defaultdict(list)
    source_dynamic_static_advantages: dict[str, list[float]] = defaultdict(list)
    source_dynamic_static_clusters: dict[str, list[str]] = defaultdict(list)
    fallback_probe: tuple[dict[str, Any], torch.Tensor] | None = None
    for kind_index, pair_kind in enumerate(
        ("history_branch", "order_effect", "result_outcome")
    ):
        pair_kind_enabled = float(selection_pair_weights.get(pair_kind, 0.0)) > 0.0
        pairs = _stable_cap_pairs(causal_pairs[pair_kind], max_pairs_per_kind)
        diagnostics_by_name: dict[str, list[float]] = {
            "route_advantage": [],
            "raw_route_advantage": [],
            "route_dynamic_static_advantage": [],
            "raw_route_dynamic_static_advantage": [],
            "loss": [],
            "memory_rms": [],
        }
        diagnostic_clusters: dict[str, list[str]] = {
            name: [] for name in diagnostics_by_name
        }
        success_pairs = 0
        direction_count = 0
        for pair in pairs:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if pair_kind == "result_outcome":
                    replayed_pair = _result_pair_memories(
                        model,
                        pair,
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        catalogs=catalogs,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        belief_top_k=belief_top_k,
                    )
                    if model.vnext.synchronization_enabled:
                        (
                            factual_memories,
                            intervened_memories,
                            factual_latent_trace,
                            intervened_latent_trace,
                        ) = replayed_pair
                    else:
                        factual_memories, intervened_memories = replayed_pair
                        factual_latent_trace = None
                        intervened_latent_trace = None
                else:
                    replayed_pair = _replay_pair_to_decision(
                        model,
                        pair,
                        max_horizon=max_horizon,
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        catalogs=catalogs,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        belief_top_k=belief_top_k,
                    )
                    if model.vnext.synchronization_enabled:
                        factual_memories, factual_latent_trace = replayed_pair
                        intervened_memories = factual_memories.flip(0)
                        intervened_latent_trace = factual_latent_trace.flip(0)
                    else:
                        factual_memories = replayed_pair
                        intervened_memories = factual_memories.flip(0)
                        factual_latent_trace = None
                        intervened_latent_trace = None
                loss, pair_diagnostics = _causal_pair_loss(
                    model,
                    pair,
                    factual_memories,
                    intervened_memories,
                    factual_latent_trace=factual_latent_trace,
                    intervened_latent_trace=intervened_latent_trace,
                    state_cache=state_cache,
                    catalogs=catalogs,
                    skill_id_to_idx=skill_id_to_idx,
                    device=device,
                    belief_top_k=belief_top_k,
                    coarse_k=coarse_k,
                    compressed_m=compressed_m,
                    margin=margin,
                    hard_fallback=True,
                )
            cluster_id = str(pair.get("causal_cluster_id") or "")
            diagnostics_by_name["loss"].append(float(loss.float().cpu().item()))
            diagnostic_clusters["loss"].append(cluster_id)
            diagnostics_by_name["memory_rms"].append(
                float(
                    factual_memories.float()
                    .pow(2)
                    .mean(dim=-1)
                    .sqrt()
                    .mean()
                    .cpu()
                    .item()
                )
            )
            diagnostic_clusters["memory_rms"].append(cluster_id)
            route_advantage = pair_diagnostics["route_advantage"].float().cpu().tolist()
            raw_route_advantage = pair_diagnostics[
                "raw_route_advantage"
            ].float().cpu().tolist()
            route_static = pair_diagnostics[
                "route_dynamic_static_advantage"
            ].float().cpu().tolist()
            raw_route_static = pair_diagnostics[
                "raw_route_dynamic_static_advantage"
            ].float().cpu().tolist()
            pair_route_advantage = sum(route_advantage) / len(route_advantage)
            pair_raw_route_advantage = sum(raw_route_advantage) / len(
                raw_route_advantage
            )
            pair_route_static = sum(route_static) / len(route_static)
            pair_raw_route_static = sum(raw_route_static) / len(raw_route_static)
            for name, value in (
                ("route_advantage", pair_route_advantage),
                ("raw_route_advantage", pair_raw_route_advantage),
                ("route_dynamic_static_advantage", pair_route_static),
                (
                    "raw_route_dynamic_static_advantage",
                    pair_raw_route_static,
                ),
            ):
                diagnostics_by_name[name].append(float(value))
                diagnostic_clusters[name].append(cluster_id)
            success_pairs += int(
                all(route > 0.0 for route in route_advantage)
            )
            direction_count += len(route_advantage)
            pair_causal_advantage = pair_raw_route_advantage
            pair_routed_causal_advantage = pair_route_advantage
            pair_dynamic_static_advantage = pair_route_static
            if pair_kind_enabled:
                all_causal_advantages.append(pair_causal_advantage)
                all_causal_clusters.append(cluster_id)
                all_routed_causal_advantages.append(pair_routed_causal_advantage)
                all_routed_causal_clusters.append(cluster_id)
                all_dynamic_static_advantages.append(pair_dynamic_static_advantage)
                all_dynamic_static_clusters.append(cluster_id)
                source = str(pair.get("source_id") or semantic_source_id(pair["_row_a"]))
                source_causal_advantages[source].append(pair_causal_advantage)
                source_causal_clusters[source].append(cluster_id)
                source_routed_causal_advantages[source].append(
                    pair_routed_causal_advantage
                )
                source_routed_causal_clusters[source].append(cluster_id)
                source_dynamic_static_advantages[source].append(
                    pair_dynamic_static_advantage
                )
                source_dynamic_static_clusters[source].append(cluster_id)
            if fallback_probe is None:
                fallback_probe = (pair, factual_memories)
        pair_count = len(diagnostics_by_name["route_advantage"])
        by_kind[pair_kind] = {
            "affects_selection": bool(pair_kind_enabled),
            "pair_count": len(pairs),
            "direction_count": int(direction_count),
            "causal_success_rate": (
                float(success_pairs) / pair_count if pair_count else 0.0
            ),
            "candidate_support_memory_invariant": False,
            "metrics": {
                name: _paired_mean_ci(
                    values,
                    seed=int(seed) + kind_index * 101 + metric_index,
                    cluster_ids=diagnostic_clusters[name],
                )
                for metric_index, (name, values) in enumerate(
                    diagnostics_by_name.items()
                )
            },
        }
    if fallback_probe is None:
        raise ValueError("Stage2 causal-dev validation has no verified pair")
    if not all_causal_advantages:
        raise ValueError("Stage2 causal-dev selection has no enabled verified pair")
    probe_pair, probe_memory = fallback_probe
    probe_rows = [probe_pair["_row_a"], probe_pair["_row_b"]]
    probe_states = state_cache.batch(
        [_causal_route_state_text(row) for row in probe_rows],
        device=device,
    )
    probe_current_states = state_cache.batch(
        [str(row["state_text_current"]) for row in probe_rows],
        device=device,
    )
    probe_legal = runtime_visible_mask(
        probe_rows,
        skill_id_to_idx,
        device=device,
        inventory_catalogs=catalogs,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        probe_static_memory = model.vnext_initial_belief(
            probe_states,
            probe_legal,
            top_k=belief_top_k,
        )
        fallback = model.vnext_queries(
            probe_states,
            probe_memory,
            probe_static_memory,
            torch.zeros(len(probe_rows), dtype=torch.bool, device=device),
        )
        probe_positive = _positive_mask(
            probe_rows,
            skill_id_to_idx,
            device=device,
        )
        (
            _probe_queries,
            probe_path,
            _probe_coarse_labels,
            _probe_labels,
        ) = _memory_conditioned_natural_support(
            model,
            probe_states,
            probe_memory,
            probe_static_memory,
            torch.zeros(len(probe_rows), dtype=torch.bool, device=device),
            probe_legal,
            probe_positive,
            current_states=probe_current_states,
            coarse_k=coarse_k,
            compressed_m=compressed_m,
        )
        probe_candidate_fallback = model.vnext_safe_candidate_route_scores(
            probe_states,
            probe_memory,
            probe_static_memory,
            torch.zeros(len(probe_rows), dtype=torch.bool, device=device),
            probe_path.support.candidate_ids,
            probe_path.support.valid_mask,
            static_candidate_ids=probe_path.coarse_candidate_ids,
            static_valid_mask=probe_path.coarse_valid_mask,
            hard_fallback=True,
        )
    exact_fallback = bool(
        torch.equal(fallback.dynamic_recall, fallback.static_recall)
        and torch.equal(fallback.dynamic_route, fallback.static_route)
        and torch.equal(
            probe_candidate_fallback.raw_dynamic_logits,
            probe_candidate_fallback.static_logits,
        )
        and torch.equal(
            probe_candidate_fallback.mixed_logits,
            probe_candidate_fallback.static_logits,
        )
        and torch.equal(
            probe_candidate_fallback.mixture_probability,
            torch.zeros_like(probe_candidate_fallback.mixture_probability),
        )
    )
    causal_ci = _paired_mean_ci(
        all_causal_advantages,
        seed=int(seed) + 7001,
        cluster_ids=all_causal_clusters,
    )
    routed_causal_ci = _paired_mean_ci(
        all_routed_causal_advantages,
        seed=int(seed) + 8001,
        cluster_ids=all_routed_causal_clusters,
    )
    safety_ci = _paired_mean_ci(
        all_dynamic_static_advantages,
        seed=int(seed) + 9001,
        cluster_ids=all_dynamic_static_clusters,
    )
    robust_prefix = _evaluate_robust_prefix_dev(
        model,
        anchors_by_kind=robust_prefix_anchors,
        max_rows_per_kind=max_robust_rows_per_kind,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
        final_k=final_k,
        seed=int(seed) + 23003,
    )
    robust_delta = float(robust_prefix["route_mrr_delta"]["mean"])
    selection_score = (
        float(causal_ci["mean"])
        - max(0.0, -float(safety_ci["mean"]))
        - (
            max(0.0, -robust_delta)
            if bool(include_robust_in_selection)
            else 0.0
        )
    )
    per_source = {
        source: {
            "causal_advantage": _paired_mean_ci(
                source_causal_advantages[source],
                seed=int(seed) + 17011 + source_index,
                cluster_ids=source_causal_clusters[source],
            ),
            "routed_causal_advantage": _paired_mean_ci(
                source_routed_causal_advantages[source],
                seed=int(seed) + 18011 + source_index,
                cluster_ids=source_routed_causal_clusters[source],
            ),
            "dynamic_static_advantage": _paired_mean_ci(
                source_dynamic_static_advantages[source],
                seed=int(seed) + 19001 + source_index,
                cluster_ids=source_dynamic_static_clusters[source],
            ),
        }
        for source_index, source in enumerate(sorted(source_causal_advantages))
    }
    return {
        "by_kind": by_kind,
        "causal_advantage": causal_ci,
        "routed_causal_advantage": routed_causal_ci,
        "dynamic_static_advantage": safety_ci,
        "candidate_support_memory_invariant": False,
        "exact_zero_history_fallback": exact_fallback,
        "selection_score": selection_score,
        "selection_component": "heldout_verified_raw_dynamic_causal_dev_with_routed_safety",
        "selection_pair_weights": {
            key: float(value) for key, value in sorted(selection_pair_weights.items())
        },
        "robust_prefix_affects_selection": bool(include_robust_in_selection),
        "per_source": per_source,
        "robust_prefix": robust_prefix,
    }


def _gradient_record_gate(
    record: dict[str, Any] | None,
    *,
    minimum_gradient_norm: float,
    synchronization_required: bool = False,
) -> bool:
    if not isinstance(record, dict):
        return False
    modules = record.get("modules")
    if not isinstance(modules, dict):
        return False
    required = [
        "transition_delta",
        "correction_delta",
        "memory_recall_query",
        "unified_route_query",
        "route_skill_adapter",
        "route_expert_mixture",
        "correction_gate",
    ]
    if bool(synchronization_required):
        required.append("synchronization")
    return all(
        isinstance(modules.get(name), dict)
        and bool(modules[name].get("finite"))
        and int(modules[name].get("tensor_count") or 0) > 0
        and float(modules[name].get("norm") or 0.0) > float(minimum_gradient_norm)
        for name in required
    )


def _ordinary_safety_gate(
    report: dict[str, Any],
    *,
    mrr_noninferiority_tolerance: float,
    minimum_clusters_per_stratum: int,
    minimum_stratum_coverage: float,
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    for scope, values_by_name, decisive_scope in (
        ("overall", {"overall": report.get("overall") or {}}, True),
        ("family", report.get("per_family") or {}, True),
        ("horizon", report.get("per_horizon") or {}, True),
        ("source", report.get("per_source") or {}, False),
    ):
        total_rows = sum(int(values.get("row_count") or 0) for values in values_by_name.values())
        testable_rows = 0
        testable_count = 0
        for name, values in sorted(values_by_name.items()):
            cluster_count = int(values.get("cluster_count") or 0)
            cluster_sufficient = cluster_count >= int(minimum_clusters_per_stratum)
            if cluster_sufficient:
                testable_rows += int(values.get("row_count") or 0)
                testable_count += 1
            route_low = float((values.get("route_mrr_delta") or {}).get("ci_low") or 0.0)
            checked.append(
                {
                    "scope": scope,
                    "name": str(name),
                    "decisive": bool(decisive_scope and cluster_sufficient),
                    "cluster_count": cluster_count,
                    "cluster_sufficient": cluster_sufficient,
                    "route_ci_low": route_low,
                    "status": (
                        "ok"
                        if cluster_sufficient
                        and route_low >= -float(mrr_noninferiority_tolerance)
                        else "not_testable"
                        if not cluster_sufficient
                        else "action_required"
                    ),
                    "pass": bool(
                        cluster_sufficient
                        and route_low >= -float(mrr_noninferiority_tolerance)
                    ),
                }
            )
        scope_coverage = float(testable_rows) / float(total_rows) if total_rows else 0.0
        coverage[scope] = {
            "row_count": int(total_rows),
            "testable_row_count": int(testable_rows),
            "testable_row_coverage": scope_coverage,
            "testable_stratum_count": int(testable_count),
            "minimum_stratum_coverage": (
                1.0 if scope == "overall" else float(minimum_stratum_coverage)
            ),
            "pass": bool(
                testable_count > 0
                and scope_coverage
                >= (1.0 if scope == "overall" else float(minimum_stratum_coverage))
            ),
        }
    decisive = [item for item in checked if bool(item["decisive"])]
    decisive_scopes = ("overall", "family", "horizon")
    return {
        "pass": bool(
            decisive
            and all(item["pass"] for item in decisive)
            and all(bool((coverage.get(scope) or {}).get("pass")) for scope in decisive_scopes)
        ),
        "mrr_noninferiority_tolerance": float(mrr_noninferiority_tolerance),
        "metric_scope": "end_to_end_static_vs_recurrent_natural_support",
        "minimum_clusters_per_stratum": int(minimum_clusters_per_stratum),
        "minimum_stratum_coverage": float(minimum_stratum_coverage),
        "coverage": coverage,
        "checked": checked,
    }


def _multi_m_coarse_recall_gate(
    report: dict[str, Any],
    *,
    minimum_full_pool_clusters: int,
) -> dict[str, Any]:
    multi_m = ((report.get("overall") or {}).get("candidate_recall") or {})
    minimum_fixed_causal_candidate_recall_at_500 = 0.5
    minimum_candidate_recall_at_500 = 0.5
    support_by_m: dict[str, Any] = {}
    for m_value in (100, 500):
        values = multi_m.get(str(m_value)) or {}
        full_pool = values.get("deployment_full_pool") or {}
        fixed = full_pool.get("fixed_causal_hit_rate") or {}
        dynamic = full_pool.get("dynamic_hit_rate") or {}
        dynamic_minus_static = full_pool.get("dynamic_minus_static") or {}
        full_pool_clusters = int(fixed.get("cluster_count") or 0)
        support_sufficient = bool(
            full_pool_clusters >= int(minimum_full_pool_clusters)
            and int(fixed.get("count") or 0) > 0
        )
        support_by_m[str(m_value)] = {
            "full_pool_cluster_count": full_pool_clusters,
            "support_sufficient": support_sufficient,
            "fixed_causal_hit_rate": fixed,
            "dynamic_hit_rate": dynamic,
            "dynamic_minus_static": dynamic_minus_static,
        }
    deployment = support_by_m.get("500") or {}
    deployment_fixed = deployment.get("fixed_causal_hit_rate") or {}
    deployment_dynamic = deployment.get("dynamic_hit_rate") or {}
    deployment_delta = deployment.get("dynamic_minus_static") or {}
    candidate_500 = (report.get("overall") or {}).get(
        "static_candidate_recall_at_500"
    ) or (report.get("overall") or {}).get("candidate_recall_at_500") or {}
    candidate_union = (report.get("overall") or {}).get(
        "candidate_union_recall"
    ) or {}
    deployment_union = (
        deployment.get("deployed_union_hit_rate") or candidate_union
    )
    union_minus_static = (report.get("overall") or {}).get(
        "candidate_union_minus_static_recall"
    ) or deployment.get("deployed_union_minus_fixed") or {}
    legacy_union_contract = bool(
        not union_minus_static
        and int(candidate_500.get("count") or 0) > 0
        and int(candidate_union.get("count") or 0)
        == int(candidate_500.get("count") or 0)
        and float(candidate_union.get("mean") or 0.0)
        >= float(candidate_500.get("mean") or 0.0)
    )
    union_preservation_pass = bool(
        (
            int(union_minus_static.get("count") or 0) > 0
            and float(union_minus_static.get("ci_low") or 0.0) >= 0.0
        )
        or legacy_union_contract
    )
    deployment_clusters = int(deployment.get("full_pool_cluster_count") or 0)
    deployment_testable = bool(
        deployment.get("support_sufficient")
    )
    deployment_pass = bool(
        deployment_testable
        and float(deployment_fixed.get("ci_low") or 0.0)
        >= minimum_fixed_causal_candidate_recall_at_500
        and float(deployment_union.get("ci_low") or 0.0)
        >= minimum_candidate_recall_at_500
        and union_preservation_pass
    )
    candidate_500_pass = bool(
        int(candidate_500.get("count") or 0) > 0
        and float(candidate_500.get("ci_low") or 0.0)
        >= minimum_candidate_recall_at_500
    )
    candidate_union_pass = bool(
        int(candidate_union.get("count") or 0) > 0
        and float(candidate_union.get("ci_low") or 0.0) >= 0.5
    )
    support_change = (report.get("overall") or {}).get(
        "candidate_support_change_rate"
    ) or {}
    support_change_pass = bool(
        int(support_change.get("count") or 0) > 0
        and float(support_change.get("mean") or 0.0) > 0.0
    )
    testable = bool(deployment_testable)
    passed = bool(
        testable
        and deployment_pass
        and candidate_500_pass
        and candidate_union_pass
        and support_change_pass
    )
    return {
        "status": "ok" if passed else "not_testable" if not testable else "action_required",
        "pass": passed,
        "support_by_m": support_by_m,
        "minimum_full_pool_clusters": int(minimum_full_pool_clusters),
        "deployment_m": 500,
        "deployment_cluster_count": deployment_clusters,
        "deployment_testable": deployment_testable,
        "deployment_pass": deployment_pass,
        "dynamic_only_deployment_noninferiority_pass": bool(
            deployment_testable
            and float(deployment_dynamic.get("ci_low") or 0.0)
            >= minimum_candidate_recall_at_500
            and float(deployment_delta.get("ci_low") or 0.0) >= -0.01
        ),
        "dynamic_only_deployment_is_diagnostic": True,
        "candidate_recall_at_500": candidate_500,
        "candidate_recall_at_500_pass": candidate_500_pass,
        "candidate_union_recall": candidate_union,
        "candidate_union_recall_pass": candidate_union_pass,
        "candidate_union_minus_static_recall": union_minus_static,
        "candidate_union_preservation_pass": union_preservation_pass,
        "candidate_union_legacy_contract_fallback": legacy_union_contract,
        "minimum_candidate_union_recall": 0.5,
        "candidate_support_change_rate": support_change,
        "candidate_support_change_pass": support_change_pass,
        "minimum_fixed_causal_candidate_recall_at_500": (
            minimum_fixed_causal_candidate_recall_at_500
        ),
        "minimum_candidate_recall_at_500": (
            minimum_candidate_recall_at_500
        ),
        "memory_invariant_support": False,
        "evidence_source": "ordinary_dev_preserved_static_top500_plus_dynamic_diff64_union_v2",
    }


def _stage2_validation_gates(
    validation: dict[str, Any],
    *,
    enabled_kind_weights: dict[str, float],
    minimum_dev_clusters_per_enabled_kind: int,
    minimum_dev_clusters_per_source: int,
    no_regret_tolerance: float,
    ordinary_mrr_noninferiority_tolerance: float,
    minimum_ordinary_dev_clusters_per_stratum: int,
    minimum_ordinary_dev_stratum_coverage: float,
    minimum_full_pool_clusters: int,
    minimum_gradient_norm: float,
    require_robust_prefix_curriculum: bool,
    robust_mrr_noninferiority_tolerance: float,
    synchronization_required: bool = False,
) -> dict[str, Any]:
    causal_kind_passes: dict[str, bool] = {}
    cluster_kind_passes: dict[str, bool] = {}
    causal_lowers: list[float] = []
    for kind, weight in enabled_kind_weights.items():
        if float(weight) <= 0.0:
            continue
        metrics = ((validation.get("by_kind") or {}).get(kind) or {}).get("metrics") or {}
        route = metrics.get("raw_route_advantage") or {}
        route_low = float(route.get("ci_low") or 0.0)
        causal_lowers.append(route_low)
        causal_kind_passes[kind] = bool(route_low > 0.0)
        cluster_kind_passes[kind] = bool(
            int(route.get("cluster_count") or 0)
            >= int(minimum_dev_clusters_per_enabled_kind)
        )
    source_clusters_pass = bool(
        (validation.get("per_source") or {})
        and all(
            int((source_report.get("causal_advantage") or {}).get("cluster_count") or 0)
            >= int(minimum_dev_clusters_per_source)
            for source_report in (validation.get("per_source") or {}).values()
        )
    )
    causal_safety = bool(
        float((validation.get("dynamic_static_advantage") or {}).get("ci_low") or 0.0)
        >= -float(no_regret_tolerance)
        and all(
            float((source_report.get("dynamic_static_advantage") or {}).get("ci_low") or 0.0)
            >= -float(no_regret_tolerance)
            for source_report in (validation.get("per_source") or {}).values()
        )
    )
    ordinary = _ordinary_safety_gate(
        validation.get("ordinary_dev") or {},
        mrr_noninferiority_tolerance=ordinary_mrr_noninferiority_tolerance,
        minimum_clusters_per_stratum=minimum_ordinary_dev_clusters_per_stratum,
        minimum_stratum_coverage=minimum_ordinary_dev_stratum_coverage,
    )
    coarse = _multi_m_coarse_recall_gate(
        validation.get("ordinary_dev") or {},
        minimum_full_pool_clusters=minimum_full_pool_clusters,
    )
    robust_report = validation.get("robust_prefix") or {}
    robust_required_reports = [
        report
        for kind, report in ((robust_report.get("by_kind") or {}).items())
        if kind in {"one_error_prefix", "two_or_more_error_prefix", "recovery_prefix"}
        and int(report.get("row_count") or 0) > 0
    ]
    robust_pass = bool(
        not require_robust_prefix_curriculum
        or (
            robust_required_reports
            and all(
                float((report.get("route_mrr_delta") or {}).get("ci_low") or 0.0)
                >= -float(robust_mrr_noninferiority_tolerance)
                for report in robust_required_reports
            )
        )
    )
    gates = {
        "causal_route_gate": bool(causal_kind_passes and all(causal_kind_passes.values())),
        "causal_route_by_kind": causal_kind_passes,
        "cluster_sufficiency_gate": bool(
            cluster_kind_passes
            and all(cluster_kind_passes.values())
            and source_clusters_pass
        ),
        "cluster_sufficiency_by_kind": cluster_kind_passes,
        "causal_safety_gate": causal_safety,
        "ordinary_safety_gate": bool(ordinary["pass"]),
        "ordinary_safety": ordinary,
        "coarse_recall_gate": bool(coarse["pass"]),
        "coarse_recall": coarse,
        "exact_fallback_gate": bool(validation.get("exact_zero_history_fallback")),
        "gradient_health_gate": _gradient_record_gate(
            validation.get("gradient_health"),
            minimum_gradient_norm=minimum_gradient_norm,
            synchronization_required=synchronization_required,
        ),
        "robust_prefix_dev_gate": robust_pass,
        "worst_causal_route_ci_low": (
            min(causal_lowers) if causal_lowers else float("-inf")
        ),
    }
    gates["pass"] = bool(
        gates["causal_route_gate"]
        and gates["cluster_sufficiency_gate"]
        and gates["causal_safety_gate"]
        and gates["ordinary_safety_gate"]
        and gates["coarse_recall_gate"]
        and gates["exact_fallback_gate"]
        and gates["gradient_health_gate"]
        and gates["robust_prefix_dev_gate"]
    )
    return gates


def _select_stage2_validation(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    if not records:
        raise ValueError("Stage2 checkpoint selector requires validation records")
    passing = [record for record in records if bool((record.get("gates") or {}).get("pass"))]
    if passing:
        # Every eligible checkpoint already satisfies causal utility, static
        # safety, ordinary non-inferiority, exact fallback, support quality,
        # and gradient-health gates.  Select the strongest held-out checkpoint
        # among that safe set; benchmark rows are never part of this score.
        selected = max(
            passing,
            key=lambda record: (
                float(record.get("selection_score", float("-inf"))),
                -int(record.get("step") or 0),
            ),
        )
        return selected, "best_gate_passing_heldout_score"
    selected = max(
        records,
        key=lambda record: (
            float(record.get("selection_score", float("-inf"))),
            -int(record.get("step") or 0),
        ),
    )
    return selected, "diagnostic_no_checkpoint_passed_all_gates"


def _save_checkpoint(
    path: Path,
    *,
    model: CLSTRModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: dict[str, Any],
    trainability: dict[str, Any],
    static_digest: str,
    static_route_digest: str,
    candidate_digest: str,
    skills_path: Path,
    run_contract: dict[str, Any],
    trainer_progress: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "clstr_vnext_stage2",
            "step": int(step),
            "model_state_dict": _checkpoint_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "trainability": trainability,
            "static_foundation_digest": static_digest,
            "static_route_foundation_digest": static_route_digest,
            "candidate_foundation_digest": candidate_foundation_digest(model),
            "parent_candidate_foundation_digest": candidate_digest,
            "skills_path": str(skills_path.resolve()),
            "run_contract": run_contract,
            "trainer_progress": trainer_progress or {},
        },
        temporary,
    )
    temporary.replace(path)


def train_vnext_stage2(
    *,
    stage0_checkpoint_path: str | Path,
    candidate_checkpoint_path: str | Path | None = None,
    skills_path: str | Path,
    trajectory_rows_path: str | Path,
    trajectory_dev_rows_path: str | Path,
    causal_pair_support_rows_path: str | Path,
    causal_pair_support_dev_rows_path: str | Path,
    inventory_catalogs_path: str | Path,
    data_contract_path: str | Path,
    causal_branch_pairs_path: str | Path,
    causal_branch_dev_pairs_path: str | Path,
    causal_order_pairs_path: str | Path | None = None,
    causal_order_dev_pairs_path: str | Path | None = None,
    causal_outcome_pairs_path: str | Path | None = None,
    causal_outcome_dev_pairs_path: str | Path | None = None,
    one_error_prefix_rows_path: str | Path | None = None,
    one_error_prefix_dev_rows_path: str | Path | None = None,
    two_or_more_error_prefix_rows_path: str | Path | None = None,
    two_or_more_error_prefix_dev_rows_path: str | Path | None = None,
    recovery_prefix_rows_path: str | Path | None = None,
    recovery_prefix_dev_rows_path: str | Path | None = None,
    output_dir: str | Path,
    max_steps: int,
    curriculum_total_steps: int | None = None,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1.0e-4,
    candidate_learning_rate_scale: float = 1.0,
    static_route_learning_rate_scale: float = 1.0,
    transition_scale_initial: float = 0.05,
    result_scale_initial: float = 0.05,
    recall_scale_initial: float = 0.10,
    synchronization_enabled: bool = False,
    synchronization_pair_dim: int = 96,
    synchronization_trace_length: int = 8,
    synchronization_scale_initial: float = 0.05,
    max_horizon: int = 16,
    family_first_horizon_sampling: bool = False,
    coarse_k: int = 500,
    compressed_m: int = 64,
    final_k: int = 100,
    belief_top_k: int = 64,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    frozen_cache_dir: str | Path | None = None,
    frozen_cache_read_only: bool = False,
    lambda_recall: float = 0.2,
    lambda_compression: float = 0.2,
    lambda_anchor: float = 1.0e-4,
    teacher_retention_start: float = 1.0,
    teacher_retention_end: float = 0.0,
    lambda_safety: float = 0.05,
    lambda_raw_route: float = 0.5,
    lambda_route_topk: float = 0.0,
    route_topk: int = 5,
    route_topk_margin: float = 0.05,
    route_topk_recoverable_weight: float = 4.0,
    route_topk_listwise_weight: float = 0.05,
    lambda_mixture: float = 0.2,
    mixture_utility_scale: float = 0.5,
    lambda_history: float = 0.2,
    lambda_order: float = 0.0,
    lambda_result: float = 0.0,
    counterfactual_margin: float = 0.2,
    no_regret_tolerance: float = 0.05,
    ordinary_mrr_noninferiority_tolerance: float = 0.01,
    seed: int = 23,
    checkpoint_interval: int = 400,
    validation_interval: int = 400,
    max_dev_pairs_per_kind: int | None = 128,
    max_ordinary_dev_rows: int | None = 1024,
    ordinary_dev_batch_size: int = 64,
    max_robust_dev_rows_per_kind: int | None = 128,
    minimum_dev_score_gain: float = 0.0,
    minimum_gradient_norm: float = 1.0e-10,
    minimum_dev_clusters_per_enabled_kind: int = 20,
    minimum_dev_clusters_per_source: int = 5,
    minimum_ordinary_dev_clusters_per_stratum: int = 10,
    minimum_ordinary_dev_stratum_coverage: float = 0.90,
    minimum_full_pool_clusters: int = 20,
    require_robust_prefix_curriculum: bool = False,
    minimum_robust_prefix_training_exposure_rate: float = 0.15,
    robust_mrr_noninferiority_tolerance: float = 0.01,
    max_rows: int | None = None,
    resume_checkpoint_path: str | Path | None = None,
    warm_start_checkpoint_path: str | Path | None = None,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 2 <= int(max_horizon) <= 16:
        raise ValueError("canonical CLSTR vNext horizon protocol is bounded to 2..16")
    if int(coarse_k) != 500 or not 0 < int(compressed_m) < int(coarse_k):
        raise ValueError("performance Stage2 requires static Top500 plus dynamic extras")
    if not 0 < int(final_k) <= int(coarse_k) + int(compressed_m):
        raise ValueError("performance Stage2 final_k must fit the candidate union")
    if not 0.0 < float(candidate_learning_rate_scale) <= 1.0:
        raise ValueError("candidate learning-rate scale must be in (0, 1]")
    if not 0.0 < float(static_route_learning_rate_scale) <= 1.0:
        raise ValueError("static route learning-rate scale must be in (0, 1]")
    for name, value in (
        ("transition_scale_initial", transition_scale_initial),
        ("result_scale_initial", result_scale_initial),
        ("recall_scale_initial", recall_scale_initial),
        ("synchronization_scale_initial", synchronization_scale_initial),
    ):
        if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1)")
    for name, value in (
        ("lambda_recall", lambda_recall),
        ("lambda_compression", lambda_compression),
        ("lambda_anchor", lambda_anchor),
        ("lambda_safety", lambda_safety),
        ("lambda_raw_route", lambda_raw_route),
        ("lambda_route_topk", lambda_route_topk),
        ("lambda_mixture", lambda_mixture),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < int(route_topk) <= int(final_k):
        raise ValueError("route_topk must be positive and fit final_k")
    if bool(synchronization_enabled) and (
        int(synchronization_pair_dim) <= 0
        or not 1 <= int(synchronization_trace_length) <= int(max_horizon)
    ):
        raise ValueError(
            "native synchronization requires positive pair width and a trace within the horizon"
        )
    if not math.isfinite(float(route_topk_margin)) or float(route_topk_margin) < 0.0:
        raise ValueError("route_topk_margin must be finite and nonnegative")
    if (
        not math.isfinite(float(route_topk_recoverable_weight))
        or float(route_topk_recoverable_weight) < 1.0
    ):
        raise ValueError(
            "route_topk_recoverable_weight must be finite and at least one"
        )
    if (
        not math.isfinite(float(route_topk_listwise_weight))
        or float(route_topk_listwise_weight) < 0.0
    ):
        raise ValueError(
            "route_topk_listwise_weight must be finite and nonnegative"
        )
    if resume_checkpoint_path is not None and warm_start_checkpoint_path is not None:
        raise ValueError("Stage2 exact resume and objective warm start are mutually exclusive")
    if not math.isfinite(float(mixture_utility_scale)) or float(
        mixture_utility_scale
    ) <= 0.0:
        raise ValueError("mixture_utility_scale must be finite and positive")
    if not (
        0.0 <= float(teacher_retention_end)
        <= float(teacher_retention_start)
        <= 1.0
    ):
        raise ValueError("teacher retention must anneal within [0, 1]")
    if int(checkpoint_interval) != int(validation_interval):
        raise ValueError(
            "canonical segmented Stage2 requires checkpoint_interval == validation_interval"
        )
    reproducibility = seed_vnext_run(seed)
    source_manifest = capture_vnext_source_manifest(
        Path(__file__).resolve().parents[1],
        (
            "clstr/model.py",
            "clstr/belief.py",
            "clstr/history_channel.py",
            "clstr/vnext_core.py",
            "clstr/vnext_data.py",
            "clstr/vnext_candidates.py",
            "clstr/vnext_losses.py",
            "clstr/vnext_training.py",
            "clstr/vnext_stage2_train.py",
            "clstr/vnext_eval.py",
            "clstr/vnext_online_selector.py",
            "scripts/run_clstr_vnext_stage2_train.py",
            "scripts/resolve_clstr_vnext_full_inputs.py",
            "scripts/finalize_clstr_vnext_full_chain.py",
            "scripts/sbatch/run_clstr_vnext_full_stage2_segment.sh",
        ),
        require_clean=bool(require_clean_source),
    )
    source_manifest_record = persist_vnext_source_manifest(output_dir, source_manifest)
    resolved_curriculum_total_steps = (
        int(curriculum_total_steps)
        if curriculum_total_steps is not None
        else int(max_steps)
    )
    if resolved_curriculum_total_steps < int(max_steps) or resolved_curriculum_total_steps <= 0:
        raise ValueError("curriculum_total_steps must be positive and cover max_steps")
    if int(minimum_dev_clusters_per_enabled_kind) <= 0:
        raise ValueError("minimum_dev_clusters_per_enabled_kind must be positive")
    if int(minimum_dev_clusters_per_source) <= 0:
        raise ValueError("minimum_dev_clusters_per_source must be positive")
    if int(minimum_ordinary_dev_clusters_per_stratum) <= 0:
        raise ValueError("minimum_ordinary_dev_clusters_per_stratum must be positive")
    if int(minimum_full_pool_clusters) <= 0:
        raise ValueError("minimum_full_pool_clusters must be positive")
    if int(ordinary_dev_batch_size) <= 0:
        raise ValueError("ordinary_dev_batch_size must be positive")
    if not 0.0 <= float(ordinary_mrr_noninferiority_tolerance) <= 1.0:
        raise ValueError("ordinary MRR non-inferiority tolerance must be in [0, 1]")
    if not 0.0 < float(minimum_ordinary_dev_stratum_coverage) <= 1.0:
        raise ValueError("ordinary-dev stratum coverage must be in (0, 1]")
    if not 0.0 <= float(minimum_robust_prefix_training_exposure_rate) <= 1.0:
        raise ValueError("minimum robust-prefix exposure rate must be in [0, 1]")
    if not 0.0 <= float(robust_mrr_noninferiority_tolerance) <= 1.0:
        raise ValueError("robust-prefix MRR tolerance must be in [0, 1]")
    data_contract = require_verified_data_contract(
        data_contract_path,
        {
            "training_skills": skills_path,
            "trajectory_rows": trajectory_rows_path,
            "trajectory_dev_rows": trajectory_dev_rows_path,
            "causal_pair_support_rows": causal_pair_support_rows_path,
            "causal_pair_support_dev_rows": causal_pair_support_dev_rows_path,
            "inventory_catalogs": inventory_catalogs_path,
            "causal_branch_pairs": causal_branch_pairs_path,
            "causal_branch_dev_pairs": causal_branch_dev_pairs_path,
            "causal_order_pairs": causal_order_pairs_path,
            "causal_order_dev_pairs": causal_order_dev_pairs_path,
            "causal_outcome_pairs": causal_outcome_pairs_path,
            "causal_outcome_dev_pairs": causal_outcome_dev_pairs_path,
            "one_error_prefix_rows": one_error_prefix_rows_path,
            "one_error_prefix_dev_rows": one_error_prefix_dev_rows_path,
            "two_or_more_error_prefix_rows": two_or_more_error_prefix_rows_path,
            "two_or_more_error_prefix_dev_rows": two_or_more_error_prefix_dev_rows_path,
            "recovery_prefix_rows": recovery_prefix_rows_path,
            "recovery_prefix_dev_rows": recovery_prefix_dev_rows_path,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("vNext Stage2 training requires a CUDA Slurm node")
    if candidate_checkpoint_path is not None:
        model, model_config, _stage0_payload, backbone_snapshot = (
            _load_candidate_model(
                candidate_checkpoint_path,
                skills_path,
                device=device,
                synchronization_enabled=synchronization_enabled,
                synchronization_pair_dim=synchronization_pair_dim,
                synchronization_trace_length=synchronization_trace_length,
                synchronization_scale_initial=synchronization_scale_initial,
            )
        )
        if file_sha256(stage0_checkpoint_path) != str(
            _stage0_payload.get("parent_stage0_checkpoint_sha256") or ""
        ):
            raise ValueError(
                "Stage2 stage0_checkpoint_path is not the static reranker's parent"
            )
    else:
        model, model_config, _stage0_payload, backbone_snapshot = (
            _load_stage0_model(
                stage0_checkpoint_path,
                skills_path,
                device=device,
                synchronization_enabled=synchronization_enabled,
                synchronization_pair_dim=synchronization_pair_dim,
                synchronization_trace_length=synchronization_trace_length,
                synchronization_scale_initial=synchronization_scale_initial,
            )
        )
    recurrent_initialization_contract = {
        "protocol": "explicit_stage2_direct_unified_router_reset_v3",
        "seed": int(seed) + 200003,
        "transition_scale_initial": float(transition_scale_initial),
        "result_scale_initial": float(result_scale_initial),
        "recall_scale_initial": float(recall_scale_initial),
        "zero_initialized_memory_recall_output": True,
        "zero_history_exact_static": True,
        "history_rows_use_unbounded_unified_residual_at_step0": True,
        "unified_route_initialized_from_static_query": True,
        "route_skill_adapter_zero_residual": True,
        "route_expert_mixture_initial_probability": 0.1,
        "route_expert_mixture_feature_dim": 6,
        "route_expert_mixture_semantic_dim": int(model.vnext.d),
        "route_expert_mixture_semantic_projection_dim": min(16, int(model.vnext.d)),
        "semantic_per_sample_route_selector": True,
        "candidate_semantic_expert_selector": True,
        "query_level_expert_mixture_enabled": True,
        "candidate_local_gate_enabled": False,
        "legacy_candidate_route_gate_frozen": True,
        "legacy_candidate_route_gate_reset_for_rng_compatibility": True,
        "depth_selector_reset_after_raw_expert": True,
        "route_expert_selection_threshold": 0.55,
        "synchronization_enabled": bool(synchronization_enabled),
        "synchronization_pair_dim": int(synchronization_pair_dim),
        "synchronization_trace_length": int(synchronization_trace_length),
        "synchronization_scale_initial": float(synchronization_scale_initial),
    }
    # Initialize the parent route state deterministically before both fresh and
    # resumed runs.  Resume loading then overwrites it with learned state, while
    # the parent digest remains identical to the original segment.
    observed_initialization = model.vnext.reset_stage2_recurrent_parameters(
        seed=int(seed) + 200003,
        transition_scale_initial=float(transition_scale_initial),
        result_scale_initial=float(result_scale_initial),
        recall_scale_initial=float(recall_scale_initial),
        synchronization_scale_initial=float(synchronization_scale_initial),
    )
    if resume_checkpoint_path is None and warm_start_checkpoint_path is None:
        if any(
            observed_initialization.get(key) != value
            for key, value in recurrent_initialization_contract.items()
            if key != "protocol"
        ):
            raise RuntimeError("Stage2 recurrent initialization contract did not reproduce")
        recurrent_initialization_action = "reset_from_static_parent"
    elif resume_checkpoint_path is not None:
        recurrent_initialization_action = (
            "deterministic_parent_then_restored_from_stage2_resume_checkpoint"
        )
    else:
        recurrent_initialization_action = (
            "deterministic_parent_then_restored_from_stage2_objective_warm_start"
        )
    selected_skills = read_jsonl(skills_path)
    skill_id_to_idx = {_skill_id(row): idx for idx, row in enumerate(selected_skills)}
    catalogs, catalog_mapping, catalog_report = derive_inventory_catalog_subset(
        load_inventory_catalogs(inventory_catalogs_path),
        set(skill_id_to_idx),
    )
    raw_trajectory_rows = rewrite_inventory_catalog_references(
        read_jsonl(trajectory_rows_path, max_rows=max_rows),
        catalog_mapping,
        catalogs,
    )
    trajectories, data_report = _prepare_trajectories(
        raw_trajectory_rows,
        set(skill_id_to_idx),
        catalogs,
        sequence_role="ordinary",
    )
    family_horizon_support = _family_horizon_support(
        trajectories,
        max_horizon=max_horizon,
    )
    if bool(family_first_horizon_sampling) and not family_horizon_support:
        raise ValueError(
            "family-first horizon sampling requires eligible trajectory families"
        )
    raw_dev_trajectory_rows = rewrite_inventory_catalog_references(
        read_jsonl(trajectory_dev_rows_path),
        catalog_mapping,
        catalogs,
    )
    dev_trajectories, dev_data_report = _prepare_trajectories(
        raw_dev_trajectory_rows,
        set(skill_id_to_idx),
        catalogs,
        sequence_role="ordinary",
    )
    ordinary_dev_anchors, ordinary_dev_sampling = _build_ordinary_dev_anchors(
        dev_trajectories,
        max_rows=max_ordinary_dev_rows,
        max_horizon=max_horizon,
    )
    if not ordinary_dev_anchors:
        raise ValueError("canonical Stage2 requires ordinary-dev safety anchors")
    raw_pair_support_rows = rewrite_inventory_catalog_references(
        read_jsonl(causal_pair_support_rows_path),
        catalog_mapping,
        catalogs,
    )
    pair_support_trajectories, pair_support_report = _prepare_trajectories(
        raw_pair_support_rows,
        set(skill_id_to_idx),
        catalogs,
        sequence_role="pair_support",
    )
    raw_pair_support_dev_rows = rewrite_inventory_catalog_references(
        read_jsonl(causal_pair_support_dev_rows_path),
        catalog_mapping,
        catalogs,
    )
    history_channel = audit_history_channel_rows(
        [
            *raw_trajectory_rows,
            *raw_dev_trajectory_rows,
            *raw_pair_support_rows,
            *raw_pair_support_dev_rows,
        ],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    if history_channel.get("status") != "ok":
        raise ValueError("Stage2 data contract failed structured current/causal audit")
    pair_support_dev_trajectories, pair_support_dev_report = _prepare_trajectories(
        raw_pair_support_dev_rows,
        set(skill_id_to_idx),
        catalogs,
        sequence_role="pair_support",
    )
    if not any(
        _correction_result_text(row)
        for trajectory in trajectories
        for row in trajectory
    ):
        raise ValueError("canonical Stage2 requires executed-result training rows")
    if not any(
        _correction_result_text(row)
        for trajectory in dev_trajectories
        for row in trajectory
    ):
        raise ValueError("canonical Stage2 requires executed-result dev rows")
    data_report["inventory_catalogs"] = catalog_report
    data_report["data_contract"] = data_contract
    data_report["history_channel"] = history_channel
    data_report["causal_pair_support"] = {
        "train": pair_support_report,
        "dev": pair_support_dev_report,
        "excluded_from_ordinary_stage2_sampling": True,
    }
    trajectories_by_id = _trajectory_index(
        [*trajectories, *pair_support_trajectories]
    )
    dev_trajectories_by_id = _trajectory_index(
        [*dev_trajectories, *pair_support_dev_trajectories]
    )
    robust_prefix_anchors = {
        "one_error_prefix": _load_robust_prefix_anchors(
            one_error_prefix_rows_path,
            expected_kind="one_error_prefix",
            trajectories=trajectories_by_id,
        ),
        "two_or_more_error_prefix": _load_robust_prefix_anchors(
            two_or_more_error_prefix_rows_path,
            expected_kind="two_or_more_error_prefix",
            trajectories=trajectories_by_id,
        ),
        "recovery_prefix": _load_robust_prefix_anchors(
            recovery_prefix_rows_path,
            expected_kind="recovery_prefix",
            trajectories=trajectories_by_id,
        ),
    }
    robust_prefix_dev_anchors = {
        "one_error_prefix": _load_robust_prefix_anchors(
            one_error_prefix_dev_rows_path,
            expected_kind="one_error_prefix",
            trajectories=dev_trajectories_by_id,
        ),
        "two_or_more_error_prefix": _load_robust_prefix_anchors(
            two_or_more_error_prefix_dev_rows_path,
            expected_kind="two_or_more_error_prefix",
            trajectories=dev_trajectories_by_id,
        ),
        "recovery_prefix": _load_robust_prefix_anchors(
            recovery_prefix_dev_rows_path,
            expected_kind="recovery_prefix",
            trajectories=dev_trajectories_by_id,
        ),
    }
    robust_train = [
        anchor for anchors in robust_prefix_anchors.values() for anchor in anchors
    ]
    robust_dev = [
        anchor for anchors in robust_prefix_dev_anchors.values() for anchor in anchors
    ]
    robust_train_in_horizon = [
        anchor
        for anchor in robust_train
        if int(anchor["target_index"]) - int(anchor["event_index"]) + 1
        <= int(max_horizon)
    ]
    robust_dev_in_horizon = [
        anchor
        for anchor in robust_dev
        if int(anchor["target_index"]) - int(anchor["event_index"]) + 1
        <= int(max_horizon)
    ]
    if bool(require_robust_prefix_curriculum):
        if not (
            robust_prefix_anchors["one_error_prefix"]
            or robust_prefix_anchors["two_or_more_error_prefix"]
        ) or not robust_prefix_anchors["recovery_prefix"]:
            raise ValueError("canonical Stage2 requires verified error and recovery train views")
        if not (
            robust_prefix_dev_anchors["one_error_prefix"]
            or robust_prefix_dev_anchors["two_or_more_error_prefix"]
        ) or not robust_prefix_dev_anchors["recovery_prefix"]:
            raise ValueError("canonical Stage2 requires verified error and recovery dev views")
        if not robust_train_in_horizon or not robust_dev_in_horizon:
            raise ValueError("verified robust prefixes fall outside the configured BPTT horizon")
    causal_pairs = {
        "history_branch": _load_causal_pairs(
            causal_branch_pairs_path,
            expected_kind="history_branch",
            trajectories=trajectories_by_id,
            catalogs=catalogs,
        ),
        "order_effect": _load_causal_pairs(
            causal_order_pairs_path,
            expected_kind="order_effect",
            trajectories=trajectories_by_id,
            catalogs=catalogs,
        ),
        "result_outcome": _load_causal_pairs(
            causal_outcome_pairs_path,
            expected_kind="result_outcome",
            trajectories=trajectories_by_id,
            catalogs=catalogs,
        ),
    }
    causal_dev_pairs = {
        "history_branch": _load_causal_pairs(
            causal_branch_dev_pairs_path,
            expected_kind="history_branch",
            trajectories=dev_trajectories_by_id,
            catalogs=catalogs,
        ),
        "order_effect": _load_causal_pairs(
            causal_order_dev_pairs_path,
            expected_kind="order_effect",
            trajectories=dev_trajectories_by_id,
            catalogs=catalogs,
        ),
        "result_outcome": _load_causal_pairs(
            causal_outcome_dev_pairs_path,
            expected_kind="result_outcome",
            trajectories=dev_trajectories_by_id,
            catalogs=catalogs,
        ),
    }
    delayed_outcome_pairs = [
        pair
        for pair in causal_pairs["result_outcome"]
        if int(pair["_index"]) - int(pair["_event_index"]) <= int(max_horizon)
    ]
    dropped_outcome_pairs = len(causal_pairs["result_outcome"]) - len(delayed_outcome_pairs)
    causal_pairs["result_outcome"] = delayed_outcome_pairs
    delayed_dev_outcome_pairs = [
        pair
        for pair in causal_dev_pairs["result_outcome"]
        if int(pair["_index"]) - int(pair["_event_index"]) <= int(max_horizon)
    ]
    dropped_dev_outcome_pairs = len(causal_dev_pairs["result_outcome"]) - len(
        delayed_dev_outcome_pairs
    )
    causal_dev_pairs["result_outcome"] = delayed_dev_outcome_pairs
    _require_weighted_causal_views(
        causal_pairs,
        causal_dev_pairs,
        weights={
            "history_branch": float(lambda_history),
            "order_effect": float(lambda_order),
            "result_outcome": float(lambda_result),
        },
    )
    data_report["causal_pairs"] = {
        "history_branch": len(causal_pairs["history_branch"]),
        "order_effect": len(causal_pairs["order_effect"]),
        "result_outcome": len(causal_pairs["result_outcome"]),
        "result_outcome_boundary_dropped": int(dropped_outcome_pairs),
    }
    data_report["causal_dev"] = {
        **dev_data_report,
        "causal_pairs": {
            "history_branch": len(causal_dev_pairs["history_branch"]),
            "order_effect": len(causal_dev_pairs["order_effect"]),
            "result_outcome": len(causal_dev_pairs["result_outcome"]),
            "result_outcome_boundary_dropped": int(dropped_dev_outcome_pairs),
        },
    }
    data_report["ordinary_dev_anchor"] = ordinary_dev_sampling
    data_report["robust_prefix_curriculum"] = {
        "required": bool(require_robust_prefix_curriculum),
        "schedule": {
            "first_30_percent": 0.0,
            "next_30_percent": 0.25,
            "final_40_percent": 0.5,
        },
        "train_counts": {
            kind: len(anchors) for kind, anchors in sorted(robust_prefix_anchors.items())
        },
        "dev_counts": {
            kind: len(anchors)
            for kind, anchors in sorted(robust_prefix_dev_anchors.items())
        },
        "train_in_max_horizon_count": len(robust_train_in_horizon),
        "dev_in_max_horizon_count": len(robust_dev_in_horizon),
        "train_outside_max_horizon_count": len(robust_train)
        - len(robust_train_in_horizon),
        "dev_outside_max_horizon_count": len(robust_dev)
        - len(robust_dev_in_horizon),
    }
    trainability = configure_vnext_stage2(model)
    require_canonical_trainability(trainability)
    static_digest = static_foundation_digest(model)
    static_route_digest = static_route_foundation_digest(model)
    candidate_digest = candidate_foundation_digest(model)
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _name, parameter in named_trainable]
    memory_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(STAGE2_MEMORY_TRAINABLE_PREFIXES)
    ]
    candidate_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(STAGE2_CANDIDATE_TRAINABLE_PREFIXES)
    ]
    route_residual_parameters = [
        parameter
        for name, parameter in named_trainable
        if name.startswith(STAGE2_STATIC_ROUTE_TRAINABLE_PREFIXES)
    ]
    if (
        len(memory_parameters)
        + len(candidate_parameters)
        + len(route_residual_parameters)
        != len(trainable)
    ):
        raise ValueError("Stage2 optimizer groups do not cover every trainable parameter")
    if not memory_parameters or not candidate_parameters or not route_residual_parameters:
        raise ValueError(
            "Stage2 requires recurrent-memory, memory-proposal, and direct-router groups"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": memory_parameters,
                "lr": float(learning_rate),
                "group_role": "recurrent",
            },
            {
                "params": candidate_parameters,
                "lr": float(learning_rate) * float(candidate_learning_rate_scale),
                "group_role": "memory_candidate_proposal",
            },
            {
                "params": route_residual_parameters,
                "lr": float(learning_rate) * float(static_route_learning_rate_scale),
                "group_role": "unbounded_unified_memory_residual",
            },
        ]
    )
    start_step = 1
    resume_payload: dict[str, Any] | None = None
    warm_start_report: dict[str, Any] | None = None
    if resume_checkpoint_path:
        payload = torch.load(resume_checkpoint_path, map_location="cpu")
        require_canonical_vnext_checkpoint_state(payload.get("model_state_dict"))
        resume_config = dict(payload.get("config") or {})
        synchronization_fields = {
            "vnext_synchronization_enabled": bool(synchronization_enabled),
            "vnext_synchronization_pair_dim": int(synchronization_pair_dim),
            "vnext_synchronization_trace_length": int(synchronization_trace_length),
            "vnext_synchronization_scale_initial": float(
                synchronization_scale_initial
            ),
        }
        for field, expected in synchronization_fields.items():
            default = {
                "vnext_synchronization_enabled": False,
                "vnext_synchronization_pair_dim": 96,
                "vnext_synchronization_trace_length": 8,
                "vnext_synchronization_scale_initial": 0.05,
            }[field]
            observed = resume_config.get(field, default)
            if isinstance(expected, float):
                matches = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1.0e-12)
            else:
                matches = observed == expected
            if not matches:
                raise ValueError(
                    "Stage2 resume synchronization configuration differs: "
                    f"{field} checkpoint={observed!r} requested={expected!r}"
                )
        resume_payload = payload
        resume_load_report = load_compatible_state_dict(
            model,
            payload["model_state_dict"],
            partial_load_mode="vnext_stage2_resume",
        )
        if any(
            str(key).startswith("vnext.route_expert_mixture.")
            for key in resume_load_report.get("missing_keys") or []
        ):
            raise ValueError(
                "Stage2 resume checkpoint predates the depth-aware route mixture"
            )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        start_step = int(payload["step"]) + 1
        if static_foundation_digest(model) != static_digest:
            raise ValueError("Stage2 resume changed the frozen static foundation")
        if (
            str(payload.get("static_route_foundation_digest") or "")
            != static_route_digest
            or static_route_foundation_digest(model) != static_route_digest
        ):
            raise ValueError("Stage2 resume changed the frozen static route foundation")
        if str(payload.get("parent_candidate_foundation_digest") or "") != candidate_digest:
            raise ValueError("Stage2 resume changed its parent candidate foundation")
    elif warm_start_checkpoint_path:
        payload = torch.load(warm_start_checkpoint_path, map_location="cpu")
        if str(payload.get("stage") or "") != "clstr_vnext_stage2":
            raise ValueError("Stage2 objective warm start requires a Stage2 checkpoint")
        if int(payload.get("step") or 0) <= 0:
            raise ValueError("Stage2 objective warm start requires a trained checkpoint")
        require_canonical_vnext_checkpoint_state(payload.get("model_state_dict"))
        _require_selected_skill_prefix(payload, skills_path)
        parent_inputs = ((payload.get("run_contract") or {}).get("inputs") or {})
        if str(parent_inputs.get("stage0_checkpoint_sha256") or "") != file_sha256(
            stage0_checkpoint_path
        ):
            raise ValueError(
                "Stage2 objective warm start does not share the Stage0 parent"
            )
        expected_candidate_sha = (
            file_sha256(candidate_checkpoint_path)
            if candidate_checkpoint_path is not None
            else None
        )
        if parent_inputs.get("candidate_checkpoint_sha256") != expected_candidate_sha:
            raise ValueError(
                "Stage2 objective warm start does not share the candidate parent"
            )
        warm_start_load_report = load_compatible_state_dict(
            model,
            payload["model_state_dict"],
            partial_load_mode="vnext_stage2_objective_warm_start",
        )
        unexpected = list(warm_start_load_report.get("unexpected_keys") or [])
        skipped = list(warm_start_load_report.get("skipped_keys") or [])
        shape_mismatched = list(
            warm_start_load_report.get("shape_mismatched") or []
        )
        loaded = set(warm_start_load_report.get("loaded_keys") or [])
        if (
            loaded != set(payload["model_state_dict"])
            or unexpected
            or skipped
            or shape_mismatched
        ):
            raise ValueError(
                "Stage2 objective warm start requires an exact model-state match"
            )
        if static_foundation_digest(model) != static_digest:
            raise ValueError(
                "Stage2 objective warm start changed the frozen static foundation"
            )
        if (
            str(payload.get("static_route_foundation_digest") or "")
            != static_route_digest
            or static_route_foundation_digest(model) != static_route_digest
        ):
            raise ValueError(
                "Stage2 objective warm start changed the frozen static route foundation"
            )
        if str(payload.get("parent_candidate_foundation_digest") or "") != candidate_digest:
            raise ValueError(
                "Stage2 objective warm start changed its parent candidate foundation"
            )
        warm_start_path = Path(warm_start_checkpoint_path).resolve()
        synchronization_extension_keys = sorted(
            key
            for key in warm_start_load_report.get("missing_keys") or []
            if str(key).startswith("vnext.synchronization.")
        )
        parent_synchronization_enabled = bool(
            (payload.get("config") or {}).get(
                "vnext_synchronization_enabled",
                False,
            )
        )
        if (
            bool(synchronization_enabled)
            and not parent_synchronization_enabled
            and not synchronization_extension_keys
        ):
            raise ValueError(
                "native synchronization warm start unexpectedly restored no new "
                "synchronization parameters"
            )
        warm_start_report = {
            "enabled": True,
            "protocol": "stage2_objective_refinement_reset_optimizer_v1",
            "checkpoint_path": str(warm_start_path),
            "checkpoint_sha256": file_sha256(warm_start_path),
            "parent_step": int(payload.get("step") or 0),
            "optimizer_restored": False,
            "trainer_progress_restored": False,
            "model_state_exact": True,
            "parent_checkpoint_keys_loaded_exactly": True,
            "initialized_extension_keys": synchronization_extension_keys,
            "native_synchronization_initialized_from_current_source": bool(
                synchronization_extension_keys
            ),
            "parent_native_synchronization_enabled": bool(
                parent_synchronization_enabled
            ),
        }
    # The no-drift anchor belongs to the actual starting route state.  For an
    # objective refinement this is the verified parent Stage2 checkpoint, not
    # the deterministic Stage0 reset that was overwritten above.
    shared_anchor = {
        name: parameter.detach().clone()
        for name, parameter in named_trainable
        if name.startswith(STAGE2_STATIC_ROUTE_TRAINABLE_PREFIXES)
    }
    all_trajectories = [
        *trajectories,
        *dev_trajectories,
        *pair_support_trajectories,
        *pair_support_dev_trajectories,
    ]
    state_texts = _required_state_cache_texts(all_trajectories)
    action_texts = [
        str(row.get("action_text") or "")
        for rows in all_trajectories
        for row in rows
    ]
    result_texts = [
        _correction_result_text(row)
        for rows in all_trajectories
        for row in rows
        if _correction_result_text(row)
    ]
    cache_root = (
        Path(frozen_cache_dir)
        if frozen_cache_dir is not None
        else output_dir / "frozen_qwen_cache"
    )
    cache_loader = (
        load_frozen_text_cache_read_only
        if bool(frozen_cache_read_only)
        else load_or_build_frozen_text_cache
    )
    state_cache = cache_loader(
        model,
        state_texts,
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    action_cache = cache_loader(
        model,
        action_texts,
        role="action",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    result_cache = (
        cache_loader(
            model,
            result_texts,
            role="result",
            batch_size=cache_batch_size,
            cache_root=cache_root,
            cache_shard_size=cache_shard_size,
        )
        if result_texts
        else None
    )
    # Recurrent proposal membership changes as memory recall trains, so
    # candidate IDs cannot be frozen or cached. Frozen Qwen state,
    # action, result, and skill embeddings remain cached below.
    catalog_identity = immutable_run_contract(
        {
            "catalogs": [
                {
                    "catalog_id": catalog_id,
                    "digest": str(row.get("inventory_catalog_digest") or ""),
                    "pool_size": int(row.get("inventory_pool_size") or 0),
                }
                for catalog_id, row in sorted(catalogs.items())
            ]
        }
    )["contract_digest"]
    run_contract = immutable_run_contract(
        {
            "schema_version": "clstr_vnext_stage2_direct_unified_route_run_v1",
            "inputs": {
                "stage0_checkpoint_sha256": file_sha256(stage0_checkpoint_path),
                "candidate_checkpoint_sha256": (
                    file_sha256(candidate_checkpoint_path)
                    if candidate_checkpoint_path is not None
                    else None
                ),
                "candidate_parent_stage0_checkpoint_sha256": (
                    str(_stage0_payload.get("parent_stage0_checkpoint_sha256") or "")
                    if candidate_checkpoint_path is not None
                    else None
                ),
                "objective_warm_start_checkpoint_sha256": (
                    file_sha256(warm_start_checkpoint_path)
                    if warm_start_checkpoint_path is not None
                    else None
                ),
                "frozen_backbone_snapshot_contract": backbone_snapshot["contract"],
                "data_contract_sha256": file_sha256(data_contract_path),
                "skills_sha256": file_sha256(skills_path),
                "trajectory_rows_sha256": file_sha256(trajectory_rows_path),
                "trajectory_dev_rows_sha256": file_sha256(trajectory_dev_rows_path),
                "causal_pair_support_rows_sha256": file_sha256(
                    causal_pair_support_rows_path
                ),
                "causal_pair_support_dev_rows_sha256": file_sha256(
                    causal_pair_support_dev_rows_path
                ),
                "inventory_catalogs_sha256": file_sha256(inventory_catalogs_path),
                "causal_branch_pairs_sha256": file_sha256(causal_branch_pairs_path),
                "causal_branch_dev_pairs_sha256": file_sha256(
                    causal_branch_dev_pairs_path
                ),
                "causal_order_pairs_sha256": (
                    file_sha256(causal_order_pairs_path)
                    if causal_order_pairs_path is not None
                    else None
                ),
                "causal_order_dev_pairs_sha256": (
                    file_sha256(causal_order_dev_pairs_path)
                    if causal_order_dev_pairs_path is not None
                    else None
                ),
                "causal_outcome_pairs_sha256": (
                    file_sha256(causal_outcome_pairs_path)
                    if causal_outcome_pairs_path is not None
                    else None
                ),
                "causal_outcome_dev_pairs_sha256": (
                    file_sha256(causal_outcome_dev_pairs_path)
                    if causal_outcome_dev_pairs_path is not None
                    else None
                ),
                "one_error_prefix_rows_sha256": (
                    file_sha256(one_error_prefix_rows_path)
                    if one_error_prefix_rows_path is not None
                    else None
                ),
                "one_error_prefix_dev_rows_sha256": (
                    file_sha256(one_error_prefix_dev_rows_path)
                    if one_error_prefix_dev_rows_path is not None
                    else None
                ),
                "two_or_more_error_prefix_rows_sha256": (
                    file_sha256(two_or_more_error_prefix_rows_path)
                    if two_or_more_error_prefix_rows_path is not None
                    else None
                ),
                "two_or_more_error_prefix_dev_rows_sha256": (
                    file_sha256(two_or_more_error_prefix_dev_rows_path)
                    if two_or_more_error_prefix_dev_rows_path is not None
                    else None
                ),
                "recovery_prefix_rows_sha256": (
                    file_sha256(recovery_prefix_rows_path)
                    if recovery_prefix_rows_path is not None
                    else None
                ),
                "recovery_prefix_dev_rows_sha256": (
                    file_sha256(recovery_prefix_dev_rows_path)
                    if recovery_prefix_dev_rows_path is not None
                    else None
                ),
                "derived_catalog_identity": catalog_identity,
                "static_foundation_digest": static_digest,
                "static_route_foundation_digest": static_route_digest,
                "candidate_foundation_digest": candidate_digest,
                "ordinary_dev_anchor_identity": ordinary_dev_sampling[
                    "anchor_identity_sha256"
                ],
            },
            "model_config": model_config,
            "optimization": {
                "batch_size": int(batch_size),
                "curriculum_total_steps": int(resolved_curriculum_total_steps),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "learning_rate": float(learning_rate),
                "candidate_learning_rate_scale": float(
                    candidate_learning_rate_scale
                ),
                "static_route_learning_rate_scale": float(
                    static_route_learning_rate_scale
                ),
                "recurrent_initialization": recurrent_initialization_contract,
                "max_horizon": int(max_horizon),
                "horizon_protocol": (
                    "family_first_supported_full_bptt_le16_v1"
                    if bool(family_first_horizon_sampling)
                    else "bounded_full_bptt_le16_v1"
                ),
                "family_first_horizon_sampling": bool(
                    family_first_horizon_sampling
                ),
                "family_horizon_support": {
                    family: [int(value) for value in values]
                    for family, values in family_horizon_support.items()
                },
                "coarse_k": int(coarse_k),
                "compressed_m": int(compressed_m),
                "dynamic_extra_k": int(compressed_m),
                "final_k": int(final_k),
                "belief_top_k": int(belief_top_k),
                "lambda_recall": float(lambda_recall),
                "lambda_compression": float(lambda_compression),
                "lambda_anchor": float(lambda_anchor),
                "teacher_retention_start": float(teacher_retention_start),
                "teacher_retention_end": float(teacher_retention_end),
                "lambda_safety": float(lambda_safety),
                "lambda_raw_route": float(lambda_raw_route),
                "lambda_route_topk": float(lambda_route_topk),
                "route_topk": int(route_topk),
                "route_topk_margin": float(route_topk_margin),
                "route_topk_recoverable_weight": float(
                    route_topk_recoverable_weight
                ),
                "route_topk_listwise_weight": float(
                    route_topk_listwise_weight
                ),
                "lambda_mixture": float(lambda_mixture),
                "mixture_utility_scale": float(mixture_utility_scale),
                "lambda_history": float(lambda_history),
                "lambda_order": float(lambda_order),
                "lambda_result": float(lambda_result),
                "counterfactual_margin": float(counterfactual_margin),
                "no_regret_tolerance": float(no_regret_tolerance),
                "ordinary_mrr_noninferiority_tolerance": float(
                    ordinary_mrr_noninferiority_tolerance
                ),
                "seed": int(seed),
                "max_rows": None if max_rows is None else int(max_rows),
                "validation_interval": int(validation_interval),
                "checkpoint_interval": int(checkpoint_interval),
                "max_dev_pairs_per_kind": (
                    None
                    if max_dev_pairs_per_kind is None
                    else int(max_dev_pairs_per_kind)
                ),
                "max_ordinary_dev_rows": (
                    None
                    if max_ordinary_dev_rows is None
                    else int(max_ordinary_dev_rows)
                ),
                "ordinary_dev_batch_size": int(ordinary_dev_batch_size),
                "max_robust_dev_rows_per_kind": (
                    None
                    if max_robust_dev_rows_per_kind is None
                    else int(max_robust_dev_rows_per_kind)
                ),
                "minimum_dev_score_gain": float(minimum_dev_score_gain),
                "minimum_gradient_norm": float(minimum_gradient_norm),
                "minimum_dev_clusters_per_enabled_kind": int(
                    minimum_dev_clusters_per_enabled_kind
                ),
                "minimum_dev_clusters_per_source": int(
                    minimum_dev_clusters_per_source
                ),
                "minimum_ordinary_dev_clusters_per_stratum": int(
                    minimum_ordinary_dev_clusters_per_stratum
                ),
                "minimum_ordinary_dev_stratum_coverage": float(
                    minimum_ordinary_dev_stratum_coverage
                ),
                "minimum_full_pool_clusters": int(minimum_full_pool_clusters),
                "require_robust_prefix_curriculum": bool(
                    require_robust_prefix_curriculum
                ),
                "minimum_robust_prefix_training_exposure_rate": float(
                    minimum_robust_prefix_training_exposure_rate
                ),
                "robust_mrr_noninferiority_tolerance": float(
                    robust_mrr_noninferiority_tolerance
                ),
                "robust_prefix_schedule": "clean_then_25pct_then_50pct_v1",
                "sampling_protocol": "benchmark_family_source_trajectory_round_robin_v2",
                "ordinary_dev_protocol": (
                    "static_top500_plus_dynamic_top500_diff64_union_v2"
                ),
                "coarse_recall_protocol": (
                    "frozen_static_top500_and_recurrent_full_pool_extra_v1"
                ),
                "candidate_protocol": "static_top500_plus_dynamic_top500_diff64_union_v2",
                "candidate_support_cache_protocol": "none_recurrent_support_v1",
                "evaluation_positive_injection_count": 0,
                "training_teacher_retention_protocol": "annealed_route_only_v1",
                "route_topk_protocol": (
                    "immutable_natural_support_current_logits_boundary_v1"
                    if float(lambda_route_topk) > 0.0
                    else "disabled"
                ),
                "route_protocol": "history_depth_aware_soft_train_confidence_abstaining_hard_expert_route_final100_v7",
                "static_context_protocol": "bounded_compact_skill_action_prefix_v1",
                "memory_state_protocol": "current_only_transition_unbounded_route_residual_v2",
                "cache_shard_size": int(cache_shard_size),
                "require_clean_source": bool(require_clean_source),
            },
            "frozen_cache_identities": {
                "state": state_cache.cache_identity,
                "action": action_cache.cache_identity,
                "result": None if result_cache is None else result_cache.cache_identity,
            },
            "source_contract_digest": source_manifest_record[
                "source_contract_digest"
            ],
        }
    )
    if resume_payload is not None:
        require_matching_run_contract(resume_payload.get("run_contract"), run_contract)
    if start_step == 1:
        _save_checkpoint(
            output_dir / "checkpoints" / "clstr_vnext_stage2-step0.pt",
            model=model,
            optimizer=optimizer,
            step=0,
            model_config=model_config,
            trainability=trainability,
            static_digest=static_digest,
            static_route_digest=static_route_digest,
            candidate_digest=candidate_digest,
            skills_path=Path(skills_path),
            run_contract=run_contract,
            trainer_progress=None,
        )
    enabled_kind_weights = {
        "history_branch": float(lambda_history),
        "order_effect": float(lambda_order),
        "result_outcome": float(lambda_result),
    }

    def evaluate_validation(
        gradient_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        validation = _evaluate_stage2_causal_dev(
            model,
            causal_pairs=causal_dev_pairs,
            robust_prefix_anchors=robust_prefix_dev_anchors,
            max_pairs_per_kind=max_dev_pairs_per_kind,
            max_robust_rows_per_kind=max_robust_dev_rows_per_kind,
            max_horizon=max_horizon,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            catalogs=catalogs,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            coarse_k=coarse_k,
            compressed_m=compressed_m,
            final_k=final_k,
            margin=counterfactual_margin,
            seed=seed,
            selection_pair_weights=enabled_kind_weights,
            include_robust_in_selection=bool(require_robust_prefix_curriculum),
        )
        validation["ordinary_dev"] = _evaluate_stage2_ordinary_dev(
            model,
            anchors=ordinary_dev_anchors,
            state_cache=state_cache,
            action_cache=action_cache,
            result_cache=result_cache,
            catalogs=catalogs,
            skill_id_to_idx=skill_id_to_idx,
            device=device,
            belief_top_k=belief_top_k,
            coarse_k=coarse_k,
            compressed_m=compressed_m,
            final_k=final_k,
            batch_size=ordinary_dev_batch_size,
            seed=seed,
        )
        validation["gradient_health"] = gradient_record
        validation["gates"] = _stage2_validation_gates(
            validation,
            enabled_kind_weights=enabled_kind_weights,
            minimum_dev_clusters_per_enabled_kind=(
                minimum_dev_clusters_per_enabled_kind
            ),
            minimum_dev_clusters_per_source=minimum_dev_clusters_per_source,
            no_regret_tolerance=no_regret_tolerance,
            ordinary_mrr_noninferiority_tolerance=(
                ordinary_mrr_noninferiority_tolerance
            ),
            minimum_ordinary_dev_clusters_per_stratum=(
                minimum_ordinary_dev_clusters_per_stratum
            ),
            minimum_ordinary_dev_stratum_coverage=(
                minimum_ordinary_dev_stratum_coverage
            ),
            minimum_full_pool_clusters=minimum_full_pool_clusters,
            minimum_gradient_norm=minimum_gradient_norm,
            require_robust_prefix_curriculum=require_robust_prefix_curriculum,
            robust_mrr_noninferiority_tolerance=(
                robust_mrr_noninferiority_tolerance
            ),
            synchronization_required=bool(synchronization_enabled),
        )
        return validation

    resume_validation = evaluate_validation(None)
    if resume_payload is None:
        validation_records: list[dict[str, Any]] = [
            {"step": int(start_step - 1), **resume_validation}
        ]
        initial_score = float(resume_validation["selection_score"])
        selected_record, selection_reason = _select_stage2_validation(
            validation_records
        )
        best_step = int(selected_record["step"])
        best_score = float(selected_record["selection_score"])
        best_checkpoint = output_dir / "checkpoints" / f"clstr_vnext_stage2-step{best_step}.pt"
        best_validation = dict(selected_record)
    else:
        prior_selection_path = output_dir / "stage2_selection.json"
        if not prior_selection_path.is_file():
            raise ValueError("Stage2 resume requires prior stage2_selection.json in output_dir")
        prior_selection = json.loads(prior_selection_path.read_text(encoding="utf-8"))
        validation_records = list(prior_selection.get("validation_records") or [])
        if not validation_records or int(validation_records[-1].get("step", -1)) != int(
            start_step - 1
        ):
            raise ValueError("Stage2 resume selection does not end at checkpoint step")
        if abs(
            float(validation_records[-1].get("selection_score") or 0.0)
            - float(resume_validation["selection_score"])
        ) > 1.0e-8:
            raise ValueError("Stage2 resume validation does not reproduce prior checkpoint")
        initial_score = float(prior_selection["initial_score"])
        selected_record, selection_reason = _select_stage2_validation(
            validation_records
        )
        best_step = int(selected_record["step"])
        best_score = float(selected_record["selection_score"])
        best_checkpoint = (
            output_dir
            / "checkpoints"
            / f"clstr_vnext_stage2-step{best_step}.pt"
        )
        best_validation = dict(selected_record)
    prior_progress = (
        dict(resume_payload.get("trainer_progress") or {})
        if resume_payload is not None
        else {}
    )
    if resume_payload is not None and int(start_step) > 1:
        if not prior_progress:
            raise ValueError("Stage2 resume checkpoint lacks cumulative trainer_progress")
        if int(prior_progress.get("completed_step", -1)) != int(start_step - 1):
            raise ValueError("Stage2 resume trainer_progress does not end at checkpoint step")
    loss_curve: list[dict[str, Any]] = list(prior_progress.get("loss_curve") or [])
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    prior_elapsed_seconds = float(prior_progress.get("elapsed_seconds") or 0.0)
    autocast_dtype = torch.bfloat16
    anchor_batch_count = int(prior_progress.get("anchor_batch_count") or 0)
    delayed_anchor_rows = int(prior_progress.get("delayed_anchor_rows") or 0)
    robust_prefix_requested_batch_count = int(
        prior_progress.get("robust_prefix_requested_batch_count") or 0
    )
    robust_prefix_batch_count = int(prior_progress.get("robust_prefix_batch_count") or 0)
    robust_prefix_row_count = int(prior_progress.get("robust_prefix_row_count") or 0)
    sampled_segment_rows = int(prior_progress.get("sampled_segment_rows") or 0)
    robust_prefix_kind_exposures: Counter[str] = Counter(
        prior_progress.get("robust_prefix_kind_exposures") or {}
    )
    natural_miss_rows = int(prior_progress.get("natural_miss_rows") or 0)
    coarse_miss_rows = int(prior_progress.get("coarse_miss_rows") or 0)
    naturally_recalled_rows = int(prior_progress.get("naturally_recalled_rows") or 0)
    supervised_route_rows = int(prior_progress.get("supervised_route_rows") or 0)
    teacher_retained_rows = int(prior_progress.get("teacher_retained_rows") or 0)
    compression_eligible_rows = int(
        prior_progress.get("compression_eligible_rows") or 0
    )
    route_topk_eligible_rows = int(
        prior_progress.get("route_topk_eligible_rows") or 0
    )
    route_topk_mixed_recoverable_rows = int(
        prior_progress.get("route_topk_mixed_recoverable_rows") or 0
    )
    route_topk_raw_recoverable_rows = int(
        prior_progress.get("route_topk_raw_recoverable_rows") or 0
    )
    causal_loss_sums: Counter[str] = Counter(prior_progress.get("causal_loss_sums") or {})
    causal_pair_updates: Counter[str] = Counter(
        prior_progress.get("causal_pair_updates") or {}
    )
    causal_advantage_sums: Counter[str] = Counter(
        prior_progress.get("causal_advantage_sums") or {}
    )
    horizon_counts: Counter[int] = Counter(
        {int(key): int(value) for key, value in (prior_progress.get("horizon_counts") or {}).items()}
    )
    family_first_batch_count = int(
        prior_progress.get("family_first_batch_count") or 0
    )
    family_horizon_batch_counts: Counter[str] = Counter(
        prior_progress.get("family_horizon_batch_counts") or {}
    )
    gradient_health_records: list[dict[str, Any]] = list(
        prior_progress.get("gradient_health_records") or []
    )
    gradient_interval: dict[str, Any] = dict(
        prior_progress.get("gradient_interval")
        or _empty_gradient_interval(
            start_step,
            synchronization_enabled=bool(synchronization_enabled),
        )
    )
    if resume_payload is not None and int(gradient_interval.get("batch_count") or 0) != 0:
        raise ValueError("Stage2 completed-boundary resume has pending gradient evidence")
    segment_source_exposures: Counter[str] = Counter(
        prior_progress.get("segment_source_exposures") or {}
    )
    segment_source_eligibility: Counter[str] = Counter(
        prior_progress.get("segment_source_eligibility") or {}
    )
    causal_pair_source_exposures: Counter[str] = Counter(
        prior_progress.get("causal_pair_source_exposures") or {}
    )

    def trainer_progress() -> dict[str, Any]:
        return {
            "completed_step": int(loss_curve[-1]["step"]) if loss_curve else int(start_step - 1),
            "loss_curve": loss_curve,
            "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
            "anchor_batch_count": int(anchor_batch_count),
            "delayed_anchor_rows": int(delayed_anchor_rows),
            "robust_prefix_requested_batch_count": int(robust_prefix_requested_batch_count),
            "robust_prefix_batch_count": int(robust_prefix_batch_count),
            "robust_prefix_row_count": int(robust_prefix_row_count),
            "sampled_segment_rows": int(sampled_segment_rows),
            "robust_prefix_kind_exposures": dict(robust_prefix_kind_exposures),
            "natural_miss_rows": int(natural_miss_rows),
            "coarse_miss_rows": int(coarse_miss_rows),
            "naturally_recalled_rows": int(naturally_recalled_rows),
            "supervised_route_rows": int(supervised_route_rows),
            "teacher_retained_rows": int(teacher_retained_rows),
            "compression_eligible_rows": int(compression_eligible_rows),
            "route_topk_eligible_rows": int(route_topk_eligible_rows),
            "route_topk_mixed_recoverable_rows": int(
                route_topk_mixed_recoverable_rows
            ),
            "route_topk_raw_recoverable_rows": int(
                route_topk_raw_recoverable_rows
            ),
            "causal_loss_sums": dict(causal_loss_sums),
            "causal_pair_updates": dict(causal_pair_updates),
            "causal_advantage_sums": dict(causal_advantage_sums),
            "horizon_counts": {str(key): int(value) for key, value in horizon_counts.items()},
            "family_first_batch_count": int(family_first_batch_count),
            "family_horizon_batch_counts": dict(
                family_horizon_batch_counts
            ),
            "gradient_health_records": gradient_health_records,
            "gradient_interval": gradient_interval,
            "segment_source_exposures": dict(segment_source_exposures),
            "segment_source_eligibility": dict(segment_source_eligibility),
            "causal_pair_source_exposures": dict(causal_pair_source_exposures),
        }
    causal_pair_source_groups = {
        kind: _causal_pairs_by_source(pairs)
        for kind, pairs in causal_pairs.items()
    }
    for step in range(start_step, int(max_steps) + 1):
        micro_losses: list[float] = []
        for micro in range(int(gradient_accumulation_steps)):
            rng = random.Random(int(seed) + step * 1009 + micro)
            global_micro_index = (
                (int(step) - 1) * int(gradient_accumulation_steps) + int(micro)
            )
            progress = float(step - 1) / max(
                int(resolved_curriculum_total_steps) - 1,
                1,
            )
            horizon = _choose_horizon(
                progress,
                int(max_horizon),
                rng,
            )
            request_robust = bool(
                require_robust_prefix_curriculum
                and robust_train_in_horizon
                and rng.random() < (
                    0.0
                    if progress < 0.30
                    else 0.25
                    if progress < 0.60
                    else 0.50
                )
            )
            anchor_family: str | None = None
            family_source_position: int | None = None
            if bool(family_first_horizon_sampling) and not request_robust:
                family_rng = random.Random(
                    int(seed) + 500_009 + int(family_first_batch_count) * 7_919
                )
                horizon, anchor_family = _choose_family_first_horizon(
                    progress,
                    int(max_horizon),
                    family_rng,
                    family_horizon_support=family_horizon_support,
                    family_position=family_first_batch_count,
                )
                family_source_position = (
                    int(family_first_batch_count)
                    // len(family_horizon_support)
                    * int(batch_size)
                )
            horizon_counts[int(horizon)] += 1
            robust_prefix_requested_batch_count += int(request_robust)
            anchored = not request_robust and (step + micro) % 4 == 0
            segments, segment_sampling = _sample_segments(
                trajectories,
                batch_size=batch_size,
                horizon=horizon,
                rng=rng,
                anchored=anchored,
                source_position=(
                    int(family_source_position)
                    if family_source_position is not None
                    else global_micro_index * int(batch_size)
                ),
                forced_anchors=(robust_train_in_horizon if request_robust else None),
                required_family=anchor_family,
            )
            if anchor_family is not None:
                if not bool(segment_sampling["required_family_honored"]):
                    raise RuntimeError(
                        "family-first Stage2 sampler violated its anchor family"
                    )
                family_horizon_batch_counts[
                    f"{anchor_family}:h{int(horizon)}"
                ] += 1
                family_first_batch_count += 1
            sampled_segment_rows += int(batch_size)
            for source, count in segment_sampling["sampled_sources"].items():
                segment_source_exposures[
                    f"{segment_sampling['mode']}:{source}"
                ] += int(count)
            for source in segment_sampling["eligible_items_by_source"]:
                segment_source_eligibility[
                    f"h{int(horizon)}:{segment_sampling['mode']}:{source}"
                ] += 1
            if segment_sampling["mode"] == "robust_prefix":
                robust_prefix_batch_count += 1
                robust_prefix_row_count += int(batch_size)
                for kind, count in segment_sampling["sampled_anchor_kinds"].items():
                    robust_prefix_kind_exposures[kind] += int(count)
            anchor_batch_count += int(any(item[3] for item in segments))
            if segment_sampling["mode"] == "delayed_result":
                delayed_anchor_rows += sum(int(item[3]) for item in segments)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                replayed = _batch_memory_at_start(
                    model,
                    segments,
                    state_cache=state_cache,
                    action_cache=action_cache,
                    result_cache=result_cache,
                    catalogs=catalogs,
                    skill_id_to_idx=skill_id_to_idx,
                    device=device,
                    belief_top_k=belief_top_k,
                    with_trace=bool(model.vnext.synchronization_enabled),
                )
                if model.vnext.synchronization_enabled:
                    memories, latent_trace = replayed
                else:
                    memories = replayed
                    latent_trace = None
            sequence_losses: list[torch.Tensor] = []
            for offset in range(horizon):
                rows = [segment[0][segment[1] + offset] for segment in segments]
                absolute_indices = [segment[1] + offset for segment in segments]
                h_t = state_cache.batch(
                    [_causal_route_state_text(row) for row in rows],
                    device=device,
                )
                memory_states = state_cache.batch(
                    [str(row["state_text_current"]) for row in rows],
                    device=device,
                )
                legal = runtime_visible_mask(
                    rows,
                    skill_id_to_idx,
                    device=device,
                    inventory_catalogs=catalogs,
                )
                target_indices = torch.tensor(
                    [skill_id_to_idx[row["_vnext_target_skill_id"]] for row in rows],
                    dtype=torch.long,
                    device=device,
                )
                positive = _positive_mask(rows, skill_id_to_idx, device=device)
                history_mask = torch.tensor(
                    [index > 0 for index in absolute_indices],
                    dtype=torch.bool,
                    device=device,
                )
                history_depth = torch.tensor(
                    absolute_indices,
                    dtype=torch.float32,
                    device=device,
                )
                # Current-state visibility disables only an immediate result
                # counterfactual.  It never removes ordinary route/recall
                # supervision for the action-conditioned recurrent path.
                supervised = history_mask.clone()
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    b_t = model.vnext_initial_belief(h_t, legal, top_k=belief_top_k)
                    if bool(supervised.any().item()):
                        index = supervised.nonzero(as_tuple=False).view(-1)
                        positive_selected = positive.index_select(0, index)
                        selected_states = h_t.index_select(0, index)
                        selected_current_states = memory_states.index_select(0, index)
                        selected_memories = memories.index_select(0, index)
                        selected_static_memory = b_t.index_select(0, index)
                        selected_history = history_mask.index_select(0, index)
                        selected_history_depth = history_depth.index_select(0, index)
                        selected_legal = legal.index_select(0, index)
                        (
                            _queries,
                            candidate_path,
                            coarse_labels,
                            natural_route_labels,
                        ) = _memory_conditioned_natural_support(
                            model,
                            selected_states,
                            selected_memories,
                            selected_static_memory,
                            selected_history,
                            selected_legal,
                            positive_selected,
                            current_states=selected_current_states,
                            latent_trace=(
                                latent_trace.index_select(0, index)
                                if latent_trace is not None
                                else None
                            ),
                            coarse_k=coarse_k,
                            compressed_m=compressed_m,
                        )
                        recall_loss = _dense_multi_positive_nll(
                            candidate_path.recall_logits.float(),
                            positive_selected,
                            selected_legal,
                        )
                        compression = natural_candidate_topk_coverage_loss(
                            candidate_path.compression_logits,
                            candidate_path.coarse_base_logits.detach(),
                            coarse_labels.positive_mask,
                            candidate_path.coarse_valid_mask,
                            k=compressed_m,
                        )
                        natural_supported = natural_route_labels.positive_mask.any(
                            dim=-1
                        )
                        coarse_supported = coarse_labels.positive_mask.any(dim=-1)
                        supervised_route_rows += int(natural_supported.numel())
                        supported_count = int(
                            natural_supported.sum().detach().cpu().item()
                        )
                        naturally_recalled_rows += supported_count
                        natural_miss_rows += int(natural_supported.numel()) - supported_count
                        coarse_miss_rows += int(
                            (~coarse_supported).sum().detach().cpu().item()
                        )
                        compression_eligible_rows += int(
                            compression.report["eligible_rows"]
                        )
                        schedule_denominator = max(
                            1,
                            int(resolved_curriculum_total_steps) - 1,
                        )
                        schedule_progress = min(
                            1.0,
                            max(0.0, float(step - 1) / float(schedule_denominator)),
                        )
                        retention_probability = float(teacher_retention_start) + (
                            float(teacher_retention_end)
                            - float(teacher_retention_start)
                        ) * schedule_progress
                        retain_rows = torch.tensor(
                            [
                                rng.random() < retention_probability
                                for _ in range(int(index.numel()))
                            ],
                            device=device,
                            dtype=torch.bool,
                        )
                        teacher_support = training_only_teacher_retained_support(
                            candidate_path.support,
                            positive_selected,
                            candidate_path.recall_logits.detach(),
                            retain_rows,
                        )
                        teacher_retained_rows += int(
                            teacher_support.retained_mask.sum().detach().cpu().item()
                        )
                        teacher_labels = label_natural_support(
                            teacher_support.candidate_ids,
                            teacher_support.valid_mask,
                            positive_selected,
                        )
                        sequence_objective = (
                            float(lambda_recall) * recall_loss
                            + float(lambda_compression) * compression.loss
                        )
                        if float(lambda_route_topk) > 0.0:
                            # This route pass is deliberately separate from
                            # teacher retention below.  Candidate membership is
                            # the immutable natural static+dynamic union, and
                            # rows whose positive is absent receive no route
                            # Top-K gradient rather than a label-aware insert.
                            natural_route = model.vnext_safe_candidate_route_scores(
                                selected_states,
                                selected_memories,
                                selected_static_memory,
                                selected_history,
                                candidate_path.support.candidate_ids,
                                candidate_path.support.valid_mask,
                                static_candidate_ids=(
                                    candidate_path.coarse_candidate_ids
                                ),
                                static_valid_mask=candidate_path.coarse_valid_mask,
                                history_depth=selected_history_depth,
                                memory_state=selected_current_states,
                                latent_trace=(
                                    latent_trace.index_select(0, index)
                                    if latent_trace is not None
                                    else None
                                ),
                                hard_fallback=False,
                            )
                            mixed_topk = natural_candidate_topk_coverage_loss(
                                natural_route.mixed_logits,
                                natural_route.mixed_logits.detach(),
                                natural_route_labels.positive_mask,
                                candidate_path.support.valid_mask,
                                k=int(route_topk),
                                margin=float(route_topk_margin),
                                recoverable_weight=float(
                                    route_topk_recoverable_weight
                                ),
                                listwise_weight=float(
                                    route_topk_listwise_weight
                                ),
                            )
                            raw_topk = natural_candidate_topk_coverage_loss(
                                natural_route.raw_dynamic_logits,
                                natural_route.raw_dynamic_logits.detach(),
                                natural_route_labels.positive_mask,
                                candidate_path.support.valid_mask,
                                k=int(route_topk),
                                margin=float(route_topk_margin),
                                recoverable_weight=float(
                                    route_topk_recoverable_weight
                                ),
                                listwise_weight=float(
                                    route_topk_listwise_weight
                                ),
                            )
                            route_topk_eligible_rows += int(
                                raw_topk.report["eligible_rows"]
                            )
                            route_topk_mixed_recoverable_rows += int(
                                mixed_topk.report["recoverable_rows"]
                            )
                            route_topk_raw_recoverable_rows += int(
                                raw_topk.report["recoverable_rows"]
                            )
                            sequence_objective = sequence_objective + float(
                                lambda_route_topk
                            ) * 0.5 * (mixed_topk.loss + raw_topk.loss)
                        if bool(teacher_labels.eligible_mask.any().item()):
                            local = teacher_labels.eligible_mask.nonzero(
                                as_tuple=False
                            ).view(-1)
                            row_index = index.index_select(0, local)
                            candidate_ids = teacher_support.candidate_ids.index_select(
                                0, local
                            )
                            candidate_valid = teacher_support.valid_mask.index_select(
                                0, local
                            )
                            static_candidate_ids = (
                                candidate_path.coarse_candidate_ids.index_select(0, local)
                            )
                            static_candidate_valid = (
                                candidate_path.coarse_valid_mask.index_select(0, local)
                            )
                            route_positive = teacher_labels.positive_mask.index_select(
                                0, local
                            )
                            safe_route = model.vnext_safe_candidate_route_scores(
                                h_t.index_select(0, row_index),
                                memories.index_select(0, row_index),
                                b_t.index_select(0, row_index),
                                history_mask.index_select(0, row_index),
                                candidate_ids,
                                candidate_valid,
                                static_candidate_ids=static_candidate_ids,
                                static_valid_mask=static_candidate_valid,
                                history_depth=selected_history_depth.index_select(
                                    0,
                                    local,
                                ),
                                memory_state=selected_current_states.index_select(
                                    0,
                                    local,
                                ),
                                latent_trace=(
                                    latent_trace.index_select(0, index).index_select(
                                        0,
                                        local,
                                    )
                                    if latent_trace is not None
                                    else None
                                ),
                                hard_fallback=False,
                            )
                            mixed_route_loss = _dense_multi_positive_nll(
                                safe_route.mixed_logits,
                                route_positive,
                                candidate_valid,
                            )
                            raw_route_loss = _dense_multi_positive_nll(
                                safe_route.raw_dynamic_logits,
                                route_positive,
                                candidate_valid,
                            )
                            route_mixed_utility = ranking_utility(
                                safe_route.mixed_logits,
                                route_positive,
                                candidate_valid,
                            )
                            route_raw_utility = ranking_utility(
                                safe_route.raw_dynamic_logits,
                                route_positive,
                                candidate_valid,
                            )
                            static_positive = (
                                route_positive & safe_route.static_support_mask
                            )
                            static_utility_eligible = static_positive.any(dim=-1) & (
                                safe_route.static_support_mask & ~route_positive
                            ).any(dim=-1)
                            route_static_utility = torch.zeros_like(route_raw_utility)
                            if bool(static_utility_eligible.any().item()):
                                static_local = static_utility_eligible.nonzero(
                                    as_tuple=False
                                ).view(-1)
                                static_values = ranking_utility(
                                    safe_route.static_logits.index_select(
                                        0,
                                        static_local,
                                    ),
                                    route_positive.index_select(0, static_local),
                                    safe_route.static_support_mask.index_select(
                                        0,
                                        static_local,
                                    ),
                                )
                                route_static_utility = route_static_utility.index_copy(
                                    0,
                                    static_local,
                                    static_values,
                                )
                            safety = no_regret_loss(
                                route_static_utility,
                                route_mixed_utility,
                                tolerance=no_regret_tolerance,
                                eligible_mask=static_utility_eligible,
                            )
                            mixture_calibration = query_expert_mixture_target_loss(
                                safe_route.mixture_probability,
                                route_static_utility,
                                route_raw_utility,
                                static_supported_mask=static_positive.any(dim=-1),
                                eligible_mask=natural_supported.index_select(0, local),
                                utility_scale=mixture_utility_scale,
                            )
                            sequence_objective = (
                                sequence_objective
                                + mixed_route_loss
                                + float(lambda_raw_route) * raw_route_loss
                                + float(lambda_safety) * safety
                                + float(lambda_mixture) * mixture_calibration.loss
                            )
                        sequence_losses.append(sequence_objective)
                    if offset + 1 < horizon:
                        skill = model.vnext_normalized_skill_embeddings(
                            dtype=h_t.dtype
                        ).index_select(0, target_indices)
                        action = action_cache.batch(
                            [str(row.get("action_text") or "") for row in rows],
                            device=device,
                        )
                        predicted, _transition_delta, adapted_action = model.vnext.predict_memory(
                            memories,
                            memory_states,
                            skill,
                            action,
                        )
                        result_indices = [
                            row_idx
                            for row_idx, row in enumerate(rows)
                            if _correction_result_text(row)
                        ]
                        effective = predicted
                        if result_indices and result_cache is not None:
                            result_index = torch.tensor(
                                result_indices,
                                dtype=torch.long,
                                device=device,
                            )
                            results = result_cache.batch(
                                [_correction_result_text(rows[row_idx]) for row_idx in result_indices],
                                device=device,
                            )
                            corrected, _correction_delta, _beta = model.vnext.correct_memory(
                                predicted.index_select(0, result_index),
                                memory_states.index_select(0, result_index),
                                skill.index_select(0, result_index),
                                adapted_action.index_select(0, result_index),
                                results,
                                action_is_adapted=True,
                            )
                            effective = predicted.index_copy(0, result_index, corrected)
                        memories = effective
                        if latent_trace is not None:
                            latent_trace = model.vnext.append_latent_trace(
                                memories,
                                latent_trace,
                            )
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                base_loss = (
                    torch.stack(sequence_losses).mean()
                    if sequence_losses
                    else sum(parameter.reshape(-1)[0] * 0.0 for parameter in trainable)
                )
                causal_terms: list[torch.Tensor] = []
                pair_specs = (
                    ("history_branch", float(lambda_history)),
                    ("order_effect", float(lambda_order)),
                    ("result_outcome", float(lambda_result)),
                )
                for pair_kind, weight in pair_specs:
                    source_groups = causal_pair_source_groups[pair_kind]
                    if weight <= 0.0 or not source_groups:
                        continue
                    sources = sorted(source_groups)
                    pair_source = sources[global_micro_index % len(sources)]
                    pair = rng.choice(source_groups[pair_source])
                    causal_pair_source_exposures[
                        f"{pair_kind}:{pair_source}"
                    ] += 1
                    if pair_kind == "result_outcome":
                        replayed_pair = _result_pair_memories(
                            model,
                            pair,
                            state_cache=state_cache,
                            action_cache=action_cache,
                            result_cache=result_cache,
                            catalogs=catalogs,
                            skill_id_to_idx=skill_id_to_idx,
                            device=device,
                            belief_top_k=belief_top_k,
                        )
                        if model.vnext.synchronization_enabled:
                            (
                                factual_memories,
                                intervened_memories,
                                factual_latent_trace,
                                intervened_latent_trace,
                            ) = replayed_pair
                        else:
                            factual_memories, intervened_memories = replayed_pair
                            factual_latent_trace = None
                            intervened_latent_trace = None
                    else:
                        replayed_pair = _replay_pair_to_decision(
                            model,
                            pair,
                            max_horizon=max_horizon,
                            state_cache=state_cache,
                            action_cache=action_cache,
                            result_cache=result_cache,
                            catalogs=catalogs,
                            skill_id_to_idx=skill_id_to_idx,
                            device=device,
                            belief_top_k=belief_top_k,
                        )
                        if model.vnext.synchronization_enabled:
                            factual_memories, factual_latent_trace = replayed_pair
                            intervened_memories = factual_memories.flip(0)
                            intervened_latent_trace = factual_latent_trace.flip(0)
                        else:
                            factual_memories = replayed_pair
                            intervened_memories = factual_memories.flip(0)
                            factual_latent_trace = None
                            intervened_latent_trace = None
                    pair_loss, pair_diagnostics = _causal_pair_loss(
                        model,
                        pair,
                        factual_memories,
                        intervened_memories,
                        factual_latent_trace=factual_latent_trace,
                        intervened_latent_trace=intervened_latent_trace,
                        state_cache=state_cache,
                        catalogs=catalogs,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        belief_top_k=belief_top_k,
                        coarse_k=coarse_k,
                        compressed_m=compressed_m,
                        margin=counterfactual_margin,
                        hard_fallback=False,
                    )
                    causal_terms.append(weight * pair_loss)
                    causal_loss_sums[pair_kind] += float(
                        pair_loss.detach().float().cpu().item()
                    )
                    causal_pair_updates[pair_kind] += 1
                    causal_advantage_sums[f"{pair_kind}:route"] += float(
                        pair_diagnostics["route_advantage"].float().mean().cpu().item()
                    )
                objective = base_loss + (
                    torch.stack(causal_terms).sum()
                    if causal_terms
                    else base_loss.new_zeros(())
                )
                anchor_loss = torch.stack(
                    [
                        (parameter.float() - shared_anchor[name].float())
                        .pow(2)
                        .mean()
                        for name, parameter in named_trainable
                        if name in shared_anchor
                    ]
                ).mean()
                objective = objective + float(lambda_anchor) * anchor_loss.to(
                    dtype=objective.dtype
                )
                loss = objective / int(gradient_accumulation_steps)
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"nonfinite vNext Stage2 loss at step {step}")
            loss.backward()
            micro_losses.append(float(loss.detach().float().cpu().item()) * int(gradient_accumulation_steps))
        gradient_interval = _update_gradient_interval(
            gradient_interval,
            _gradient_health(model),
            step=step,
        )
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == int(max_steps) or (
            int(checkpoint_interval) > 0 and step % int(checkpoint_interval) == 0
        ):
            if static_foundation_digest(model) != static_digest:
                raise RuntimeError(
                    f"frozen static foundation changed at Stage2 step {step}"
                )
            if static_route_foundation_digest(model) != static_route_digest:
                raise RuntimeError(
                    f"frozen static route foundation changed at Stage2 step {step}"
                )
        loss_curve.append(
            {
                "step": step,
                "loss": sum(micro_losses) / max(len(micro_losses), 1),
                "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
                "transition_scale": float(model.vnext.transition_scale().detach().cpu().item()),
                "result_scale": float(model.vnext.result_scale().detach().cpu().item()),
                "recall_scale": float(model.vnext.recall_scale().detach().cpu().item()),
            }
        )
        should_validate = bool(
            step == int(max_steps)
            or (int(validation_interval) > 0 and step % int(validation_interval) == 0)
        )
        should_checkpoint = bool(
            should_validate
            or (int(checkpoint_interval) > 0 and step % int(checkpoint_interval) == 0)
        )
        gradient_record: dict[str, Any] | None = None
        if should_validate:
            gradient_record = _finalize_gradient_interval(
                gradient_interval,
                end_step=step,
            )
            gradient_health_records.append(gradient_record)
            gradient_interval = _empty_gradient_interval(
                step + 1,
                synchronization_enabled=bool(synchronization_enabled),
            )
        checkpoint_path = output_dir / "checkpoints" / f"clstr_vnext_stage2-step{step}.pt"
        if should_checkpoint:
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                model_config=model_config,
                trainability=trainability,
                static_digest=static_digest,
                static_route_digest=static_route_digest,
                candidate_digest=candidate_digest,
                skills_path=Path(skills_path),
                run_contract=run_contract,
                trainer_progress=trainer_progress(),
            )
        if should_validate:
            validation = evaluate_validation(gradient_record)
            validation_records.append({"step": int(step), **validation})
            selected_record, selection_reason = _select_stage2_validation(
                validation_records
            )
            best_step = int(selected_record["step"])
            best_score = float(selected_record["selection_score"])
            best_checkpoint = (
                output_dir
                / "checkpoints"
                / f"clstr_vnext_stage2-step{best_step}.pt"
            )
            best_validation = dict(selected_record)
            write_json(
                output_dir / "stage2_selection.json",
                {
                    "status": "training",
                    "selected_step": int(best_step),
                    "selected_checkpoint_path": str(best_checkpoint.resolve()),
                    "selected_score": float(best_score),
                    "initial_score": float(initial_score),
                    "score_gain": float(best_score - initial_score),
                    "selected_validation": best_validation,
                    "selection_reason": selection_reason,
                    "selection_source": (
                        "best_gate_passing_heldout_score_only"
                    ),
                    "validation_records": validation_records,
                },
            )
    final_checkpoint = output_dir / "checkpoints" / f"clstr_vnext_stage2-step{max_steps}.pt"
    _save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        step=max_steps,
        model_config=model_config,
        trainability=trainability,
        static_digest=static_digest,
        static_route_digest=static_route_digest,
        candidate_digest=candidate_digest,
        skills_path=Path(skills_path),
        run_contract=run_contract,
        trainer_progress=trainer_progress(),
    )
    elapsed = prior_elapsed_seconds + time.perf_counter() - started
    score_gain = float(best_score - initial_score)
    selected_gates = dict(best_validation.get("gates") or {})
    causal_gate = bool(selected_gates.get("causal_route_gate"))
    cluster_sufficiency_gate = bool(
        selected_gates.get("cluster_sufficiency_gate")
    )
    safety_gate = bool(selected_gates.get("causal_safety_gate"))
    ordinary_safety_gate = bool(selected_gates.get("ordinary_safety_gate"))
    fallback_gate = bool(selected_gates.get("exact_fallback_gate"))
    coarse_recall_gate = bool(selected_gates.get("coarse_recall_gate"))
    gradient_gate = bool(selected_gates.get("gradient_health_gate"))
    robust_prefix_dev_gate = bool(selected_gates.get("robust_prefix_dev_gate"))
    natural_support_rate = (
        float(naturally_recalled_rows) / float(supervised_route_rows)
        if supervised_route_rows
        else 0.0
    )
    total_sampled_segment_rows = int(sampled_segment_rows)
    robust_prefix_exposure_rate = (
        float(robust_prefix_row_count) / float(total_sampled_segment_rows)
        if total_sampled_segment_rows
        else 0.0
    )
    robust_prefix_exposure_gate = bool(
        not require_robust_prefix_curriculum
        or (
            robust_prefix_exposure_rate
            >= float(minimum_robust_prefix_training_exposure_rate)
            and (
                int(robust_prefix_kind_exposures["one_error_prefix"])
                + int(robust_prefix_kind_exposures["two_or_more_error_prefix"])
                > 0
            )
            and int(robust_prefix_kind_exposures["recovery_prefix"]) > 0
        )
    )
    quality_status = (
        "ok"
        if best_step > 0
        and score_gain >= float(minimum_dev_score_gain)
        and causal_gate
        and cluster_sufficiency_gate
        and safety_gate
        and ordinary_safety_gate
        and fallback_gate
        and coarse_recall_gate
        and gradient_gate
        and robust_prefix_exposure_gate
        and robust_prefix_dev_gate
        else "action_required"
    )
    selection = {
        "status": quality_status,
        "selected_step": int(best_step),
        "selected_checkpoint_path": str(best_checkpoint.resolve()),
        "selected_score": float(best_score),
        "initial_score": float(initial_score),
        "score_gain": score_gain,
        "minimum_score_gain": float(minimum_dev_score_gain),
        "selected_validation": best_validation,
        "causal_route_gate": causal_gate,
        "cluster_sufficiency_gate": cluster_sufficiency_gate,
        "minimum_dev_clusters_per_enabled_kind": int(
            minimum_dev_clusters_per_enabled_kind
        ),
        "minimum_dev_clusters_per_source": int(minimum_dev_clusters_per_source),
        "causal_safety_gate": safety_gate,
        "ordinary_safety_gate": ordinary_safety_gate,
        "minimum_ordinary_dev_clusters_per_stratum": int(
            minimum_ordinary_dev_clusters_per_stratum
        ),
        "ordinary_mrr_noninferiority_tolerance": float(
            ordinary_mrr_noninferiority_tolerance
        ),
        "minimum_ordinary_dev_stratum_coverage": float(
            minimum_ordinary_dev_stratum_coverage
        ),
        "minimum_full_pool_clusters": int(minimum_full_pool_clusters),
        "exact_fallback_gate": fallback_gate,
        "coarse_recall_gate": coarse_recall_gate,
        "coarse_recall": selected_gates.get("coarse_recall"),
        "positive_injection_count": 0,
        "training_teacher_retained_rows": int(teacher_retained_rows),
        "natural_support_rate": natural_support_rate,
        "gradient_health_gate": gradient_gate,
        "robust_prefix_exposure_gate": robust_prefix_exposure_gate,
        "robust_prefix_dev_gate": robust_prefix_dev_gate,
        "robust_mrr_noninferiority_tolerance": float(
            robust_mrr_noninferiority_tolerance
        ),
        "robust_prefix_exposure_rate": robust_prefix_exposure_rate,
        "minimum_robust_prefix_training_exposure_rate": float(
            minimum_robust_prefix_training_exposure_rate
        ),
        "selected_validation_gates": selected_gates,
        "selection_reason": selection_reason,
        "selection_source": (
            "best_gate_passing_heldout_score_only"
        ),
        "validation_records": validation_records,
    }
    write_json(output_dir / "stage2_selection.json", selection)
    write_json(output_dir / "stage2_quality_gate.json", selection)
    report = {
        "status": quality_status,
        "stage": "clstr_vnext_stage2",
        "method_contract": {
            "coarse_recall": "frozen_static_top500_plus_recurrent_extra64",
            "candidate_support": "legal_static_top500_dynamic_top500_diff64_union",
            "static_reranker": (
                "frozen_static_route_anchor_from_static_reranker"
                if candidate_checkpoint_path is not None
                else "frozen_static_route_anchor_from_stage0"
            ),
            "candidate_compression": "static_top500_plus_memory_extra64_with_query_level_expert_mixture_to_final64_and_final100",
            "memory_scope": "full_pool_extra_candidate_recall_and_query_level_dynamic_expert",
            "memory_query_heads": "full_pool_extra_recall_plus_unbounded_dynamic_route_expert",
            "route_fusion": "soft_train_hard_inference_query_level_selection_between_frozen_static_and_unbounded_dynamic_experts",
            "route_reliability_features": "detached_normalized_route_state_current_state_memory_delta_semantics_plus_static_dynamic_margin_confidence_residual_rms_and_log_normalized_absolute_history_depth",
            "route_reliability_source_labels": False,
            "route_selector_reports_soft_probability": True,
            "causal_evidence_route": "raw_dynamic_expert",
            "deployment_safety_route": "hard_selected_expert",
            "raw_memory_supervision": "counterfactual_margin_preserved",
            "final_route_supervision": (
                "mixed_ce_raw_dynamic_ce_natural_support_topk_boundary_"
                "no_regret_utility_calibration_plus_raw_causal"
                if float(lambda_route_topk) > 0.0
                else "mixed_ce_raw_dynamic_ce_no_regret_utility_calibration_plus_raw_causal"
            ),
            "route_topk_supervision": (
                "mixed_and_raw_dynamic_natural_support_only"
                if float(lambda_route_topk) > 0.0
                else "disabled"
            ),
            "recurrent_initialization": recurrent_initialization_contract,
            "recurrent_initialization_action": recurrent_initialization_action,
            "candidate_support_memory_invariant": False,
            "static_route_context": "bounded_compact_skill_action_history",
            "memory_query_state": "structured_current_only",
            "memory_update_state": "structured_current_only",
            "native_synchronization": (
                "enabled_latent_trace_query_readout"
                if bool(synchronization_enabled)
                else "disabled"
            ),
            "native_synchronization_trace_length": int(
                synchronization_trace_length
            ),
            "route_fallback": "history_depth_aware_hard_expert_selection_with_exact_zero_history_static",
            "backbone_frozen": True,
        },
        "step": int(max_steps),
        "curriculum_total_steps": int(resolved_curriculum_total_steps),
        "start_step": int(start_step),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "objective_warm_start": warm_start_report,
        "data": data_report,
        "trainability": trainability,
        "static_foundation_digest": static_digest,
        "static_foundation_digest_final": static_foundation_digest(model),
        "static_route_foundation_digest": static_route_digest,
        "static_route_foundation_digest_final": static_route_foundation_digest(model),
        "candidate_foundation_digest": candidate_digest,
        "candidate_foundation_digest_final": candidate_foundation_digest(model),
        "run_contract": run_contract,
        "source_manifest": source_manifest_record,
        "frozen_backbone_snapshot": backbone_snapshot,
        "state_cache": state_cache.report(),
        "action_cache": action_cache.report(),
        "result_cache": None if result_cache is None else result_cache.report(),
        "frozen_cache_read_only": bool(frozen_cache_read_only),
        "candidate_support_cache": {
            "protocol": "none_recurrent_support_v1",
            "candidate_support_memory_invariant": False,
        },
        "reproducibility": reproducibility,
        "loss_curve": loss_curve,
        "finite_loss": all(math.isfinite(float(row["loss"])) for row in loss_curve),
        "selection": selection,
        "anchor_batch_count": anchor_batch_count,
        "delayed_anchor_row_count": delayed_anchor_rows,
        "robust_prefix_requested_batch_count": robust_prefix_requested_batch_count,
        "robust_prefix_batch_count": robust_prefix_batch_count,
        "robust_prefix_row_count": robust_prefix_row_count,
        "robust_prefix_exposure_rate": robust_prefix_exposure_rate,
        "robust_prefix_kind_exposures": dict(
            sorted(robust_prefix_kind_exposures.items())
        ),
        "positive_injection_count": 0,
        "training_teacher_retained_rows": int(teacher_retained_rows),
        "coarse_natural_miss_rows": int(coarse_miss_rows),
        "compression_eligible_rows": int(compression_eligible_rows),
        "route_topk": {
            "enabled": bool(float(lambda_route_topk) > 0.0),
            "k": int(route_topk),
            "lambda": float(lambda_route_topk),
            "margin": float(route_topk_margin),
            "recoverable_weight": float(route_topk_recoverable_weight),
            "listwise_weight": float(route_topk_listwise_weight),
            "eligible_rows": int(route_topk_eligible_rows),
            "mixed_recoverable_rows": int(
                route_topk_mixed_recoverable_rows
            ),
            "raw_recoverable_rows": int(route_topk_raw_recoverable_rows),
            "positive_injection_count": 0,
            "candidate_membership_protocol": "immutable_natural_support",
        },
        "route_natural_miss_rows": natural_miss_rows,
        "route_naturally_recalled_rows": naturally_recalled_rows,
        "route_supervised_rows": supervised_route_rows,
        "natural_support_rate": natural_support_rate,
        "gradient_health": gradient_health_records,
        "minimum_gradient_norm": float(minimum_gradient_norm),
        "coarse_recall_support_contract": {
            "minimum_full_pool_clusters": int(minimum_full_pool_clusters),
            "memory_invariant_support_required": False,
            "zero_history_static_fallback_required": True,
        },
        "horizon_counts": {
            str(key): int(value) for key, value in sorted(horizon_counts.items())
        },
        "sampling": {
            "protocol": "benchmark_family_source_trajectory_round_robin_v2",
            "horizon_protocol": (
                "family_first_supported_full_bptt_le16_v1"
                if bool(family_first_horizon_sampling)
                else "bounded_full_bptt_le16_v1"
            ),
            "family_first_horizon_sampling": bool(
                family_first_horizon_sampling
            ),
            "family_horizon_support": {
                family: [int(value) for value in values]
                for family, values in family_horizon_support.items()
            },
            "family_first_batch_count": int(family_first_batch_count),
            "family_horizon_batch_counts": dict(
                sorted(family_horizon_batch_counts.items())
            ),
            "segment_source_exposures": dict(sorted(segment_source_exposures.items())),
            "segment_source_eligibility_calls": dict(
                sorted(segment_source_eligibility.items())
            ),
            "causal_pair_input_counts": {
                kind: {
                    source: len(pairs)
                    for source, pairs in sorted(source_groups.items())
                }
                for kind, source_groups in sorted(causal_pair_source_groups.items())
            },
            "causal_pair_source_exposures": dict(
                sorted(causal_pair_source_exposures.items())
            ),
        },
        "causal_pair_updates": dict(sorted(causal_pair_updates.items())),
        "causal_loss_means": {
            key: float(value) / max(int(causal_pair_updates[key]), 1)
            for key, value in sorted(causal_loss_sums.items())
        },
        "causal_advantage_means": {
            key: float(value)
            / max(int(causal_pair_updates[key.split(":", 1)[0]]), 1)
            for key, value in sorted(causal_advantage_sums.items())
        },
        "elapsed_seconds": elapsed,
        "steps_per_second": len(loss_curve) / max(elapsed, 1.0e-9),
    }
    write_json(output_dir / "train_report.json", report)
    return report
