from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from clstr.full_base_train import (
    UNIFIED_MEMORY_ROUTE_SCORER,
    _attach_stage0_topm_candidates,
    _equivalent_skill_ids_by_skill_id,
    _filter_transition_candidates_by_inventory,
    _read_jsonl,
    _skill_id,
)
from clstr.memory_candidate_recall import (
    candidate_recall_protocol_report,
    source_rows_have_causal_sequence,
)
from clstr.memory_utility_gate import RELIABILITY_MODES
from clstr.memory_utility_records import checkpoint_chain_digest
from clstr.memory_utility_gate_train import resolve_reliability_gate
from clstr.logged_online_stage4_train import (
    attach_trajectory_prefix_online_memory_scores,
    evaluate_logged_online_stage4_rows,
)
from clstr.stage2_real_topm_eval import aggregate_eval_metrics
from clstr.stage4_act_train import (
    STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    _as_stage4_handoff_row,
    _build_stage4_next_skill_rows_from_source_rows,
    _candidate_rank_prior_scores,
)
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
from clstr.stage4_safe_memory import (
    router_state_digest,
    validate_stage4_checkpoint_payload,
)


SAFE_PUBLIC_RELIABILITY_MODES = frozenset(RELIABILITY_MODES)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def strict_stage4_metrics(
    metrics: dict[str, Any],
    *,
    retained_rows: int,
    source_rows: int,
) -> dict[str, float]:
    retained_rows = max(0, int(retained_rows))
    source_rows = max(0, int(source_rows))
    scale = retained_rows / source_rows if source_rows > 0 else 0.0
    return {
        "strict_stage4_next_skill_recall@1": float(metrics.get("stage4_next_skill_recall@1") or 0.0) * scale,
        "strict_stage4_next_skill_recall@5": float(metrics.get("stage4_next_skill_recall@5") or 0.0) * scale,
        "strict_stage4_next_skill_mrr": float(metrics.get("stage4_next_skill_mrr") or 0.0) * scale,
        "retained_row_fraction": scale,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
        "retained_stage4_candidate_count": float(metrics.get("stage4_candidate_count") or 0.0),
    }


def strict_metric_contract() -> dict[str, str]:
    return {
        "primary_metric_scope": "strict",
        "retained_metrics_scope": "diagnostic_only",
        "dropped_candidate_handoff_rows": "counted_as_zero_in_strict_metrics",
        "main_table_required_prefix": "strict_",
    }


def apply_transition_inventory_filter_to_stage4_rows(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    transition_inventory_mask_mode: str = "off",
    transition_inventory_min_candidates: int = 0,
    preserve_positive: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(transition_inventory_mask_mode or "off")
    min_candidates = max(0, int(transition_inventory_min_candidates or 0))
    skill_ids_by_idx = {int(idx): str(skill_id) for skill_id, idx in skill_id_to_idx.items()}
    if not rows:
        return [], {
            "enabled": mode != "off",
            "mode": mode,
            "min_candidates": min_candidates,
            "input_rows": 0,
            "output_rows": 0,
            "skipped_rows": 0,
            "positive_removed_rows": 0,
            "preserve_positive": bool(preserve_positive),
            "audit": {},
        }

    output_rows = [dict(row) for row in rows]
    if mode == "off":
        return output_rows, {
            "enabled": False,
            "mode": mode,
            "min_candidates": min_candidates,
            "input_rows": len(rows),
            "output_rows": len(output_rows),
            "skipped_rows": 0,
            "positive_removed_rows": 0,
            "preserve_positive": bool(preserve_positive),
            "audit": {
                "inventory_mask_applied_rows": 0,
                "inventory_mask_missing_rows": 0,
                "inventory_mask_removed_candidates": 0,
                "inventory_mask_positive_missing_rows": 0,
                "inventory_mask_backfilled_rows": 0,
                "inventory_mask_backfilled_candidates": 0,
            },
        }

    filter_rows: list[dict[str, Any]] = []
    candidate_rows: list[list[int]] = []
    labels: list[int] = []
    positions: list[int] = []
    skipped_rows = 0
    for row_idx, row in enumerate(output_rows):
        candidate_indices = [int(idx) for idx in row.get("candidate_next_skill_indices") or []]
        next_skill_id = str(row.get("next_skill_id") or "")
        if "positive_next_skill_idx" in row:
            positive_idx = int(row["positive_next_skill_idx"])
        elif next_skill_id in skill_id_to_idx:
            positive_idx = int(skill_id_to_idx[next_skill_id])
        else:
            skipped_rows += 1
            continue
        if not candidate_indices or positive_idx not in candidate_indices:
            skipped_rows += 1
            continue
        copied_for_filter = dict(row)
        copied_for_filter.setdefault("benchmark", row.get("source_benchmark"))
        filter_rows.append(copied_for_filter)
        candidate_rows.append(candidate_indices)
        labels.append(positive_idx)
        positions.append(row_idx)

    if not filter_rows:
        return output_rows, {
            "enabled": True,
            "mode": mode,
            "min_candidates": min_candidates,
            "input_rows": len(rows),
            "output_rows": len(output_rows),
            "skipped_rows": skipped_rows,
            "positive_removed_rows": 0,
            "preserve_positive": bool(preserve_positive),
            "audit": {},
        }

    filtered_rows, audit = _filter_transition_candidates_by_inventory(
        rows=filter_rows,
        candidate_rows=candidate_rows,
        labels=torch.tensor(labels, dtype=torch.long),
        skill_ids_by_idx=skill_ids_by_idx,
        mode=mode,
        min_candidates=min_candidates,
        preserve_positive=preserve_positive,
    )

    positive_removed_rows = 0
    drop_output_indices: set[int] = set()
    for output_idx, positive_idx, filtered_indices in zip(positions, labels, filtered_rows):
        if positive_idx not in filtered_indices:
            positive_removed_rows += 1
            drop_output_indices.add(int(output_idx))
            continue
        original_indices = [int(idx) for idx in output_rows[output_idx].get("candidate_next_skill_indices") or []]
        original_scores = output_rows[output_idx].get("candidate_next_prior_scores")
        score_map: dict[int, float] = {}
        if isinstance(original_scores, list) and len(original_scores) == len(original_indices):
            for idx, score in zip(original_indices, original_scores):
                try:
                    score_map[int(idx)] = float(score)
                except (TypeError, ValueError):
                    continue
        filtered_ids = [
            skill_ids_by_idx[int(idx)]
            for idx in filtered_indices
            if int(idx) in skill_ids_by_idx
        ]
        output_rows[output_idx]["candidate_next_skill_indices"] = [int(idx) for idx in filtered_indices]
        output_rows[output_idx]["candidate_next_skill_ids"] = filtered_ids
        output_rows[output_idx]["positive_next_skill_idx"] = int(positive_idx)
        output_rows[output_idx]["positive_next_skill_position"] = int(filtered_indices.index(int(positive_idx)))
        rank_scores = _candidate_rank_prior_scores(len(filtered_indices))
        output_rows[output_idx]["candidate_next_prior_scores"] = [
            float(score_map.get(int(idx), rank_scores[pos]))
            for pos, idx in enumerate(filtered_indices)
        ]

    if drop_output_indices:
        output_rows = [row for row_idx, row in enumerate(output_rows) if row_idx not in drop_output_indices]

    return output_rows, {
        "enabled": True,
        "mode": mode,
        "min_candidates": min_candidates,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "skipped_rows": skipped_rows,
        "positive_removed_rows": positive_removed_rows,
        "preserve_positive": bool(preserve_positive),
        "audit": dict(audit),
    }


def _numeric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def build_toolbench_full_clstr_route_report(
    *,
    output_dir: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    train_trajectories_path: str | Path,
    eval_trajectories_path: str | Path,
    skills_path: str | Path,
    source_eval_rows: int,
    retained_eval_rows: int,
    train_feedback_rows: int,
    target_data_report: dict[str, Any],
    train_data_report: dict[str, Any],
    memory_selection: dict[str, Any],
    prior_eval: dict[str, Any],
    stage4_eval: dict[str, Any],
    config: dict[str, Any],
    prior_eval_by_benchmark: dict[str, Any] | None = None,
    stage4_eval_by_benchmark: dict[str, Any] | None = None,
    model_load: dict[str, Any] | None = None,
    candidate_recall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_prior = strict_stage4_metrics(
        prior_eval,
        retained_rows=retained_eval_rows,
        source_rows=source_eval_rows,
    )
    strict_stage4 = strict_stage4_metrics(
        stage4_eval,
        retained_rows=retained_eval_rows,
        source_rows=source_eval_rows,
    )
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    if int(target_data_report.get("positive_injected_rows") or 0) > 0:
        blockers.append("gold_positive_injected")
    for blocker in (candidate_recall or {}).get("protocol_blockers", []):
        if str(blocker) not in blockers:
            blockers.append(str(blocker))
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "stage4_route": "logged_online_memory",
        "route_scorer": str(config.get("route_scorer") or UNIFIED_MEMORY_ROUTE_SCORER),
        "output_dir": str(output_dir),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "train_trajectories_path": str(train_trajectories_path),
        "eval_trajectories_path": str(eval_trajectories_path),
        "skills_path": str(skills_path),
        "source_eval_rows": int(source_eval_rows),
        "retained_eval_rows": int(retained_eval_rows),
        "train_feedback_rows": int(train_feedback_rows),
        "target_data_report": target_data_report,
        "train_data_report": train_data_report,
        "memory_selection": memory_selection,
        "prior_eval": prior_eval,
        "stage4_eval": stage4_eval,
        "prior_eval_by_benchmark": prior_eval_by_benchmark or {},
        "stage4_eval_by_benchmark": stage4_eval_by_benchmark or {},
        "delta_stage4_vs_prior": _numeric_delta(stage4_eval, prior_eval),
        "strict": {
            "prior": strict_prior,
            "stage4": strict_stage4,
        },
        "metric_contract": strict_metric_contract(),
        "strict_delta": _numeric_delta(strict_stage4, strict_prior),
        "candidate_recall": candidate_recall or {},
        "config": config,
        "model_load": model_load or {},
        "paper_scope_note": (
            "This evaluates the full CLSTR route on the fixed ToolBench-G3 clean split. "
            "Strict metrics count rows dropped by Stage0/candidate handoff as zero over the original eval denominator."
        ),
    }


def _stage4_rows_from_source_rows(
    *,
    model: Any,
    source_rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    output_dir: Path,
    manifest_name: str,
    stage0_checkpoint_path: str | Path,
    top_m: int,
    candidate_count: int | None,
    stage0_handoff_query_mode: str,
    stage0_candidate_encode_batch_size: int,
    stage0_candidate_progress_interval_batches: int,
    device: torch.device,
    next_skill_pool_mode: str = "stage0_candidates",
    require_next_state_text: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handoff_rows = [
        _as_stage4_handoff_row(row, next_skill_pool_mode=next_skill_pool_mode)
        for row in source_rows
    ]
    retained_rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        handoff_rows,
        skills,
        skill_id_to_idx,
        top_m=top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=stage0_checkpoint_path,
        manifest_path=output_dir / manifest_name,
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
        next_skill_pool_mode=next_skill_pool_mode,
    )
    stage4_rows, data_report = _build_stage4_next_skill_rows_from_source_rows(
        retained_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
        max_rows=None,
        allowed_benchmarks={"toolbench_g3"},
        stage0_candidate_handoff_report=handoff_report,
        next_skill_pool_mode=next_skill_pool_mode,
        require_next_state_text=require_next_state_text,
    )
    data_report = {
        **data_report,
        "source_rows_before_handoff": len(source_rows),
        "stage0_candidate_handoff": handoff_report,
    }
    return stage4_rows, data_report


def _memory_feedback_rows(
    *,
    feedback_mode: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if feedback_mode == "train_only":
        return list(train_rows)
    if feedback_mode == "eval_prefix":
        return list(eval_rows)
    if feedback_mode == "train_plus_eval_prefix":
        return list(train_rows) + list(eval_rows)
    raise ValueError(f"unsupported feedback_mode: {feedback_mode}")


def run_toolbench_full_clstr_route_eval(
    *,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    train_trajectories_path: str | Path,
    eval_trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    top_m: int = 350,
    dynamic_extra_k: int = 64,
    candidate_count: int | None = None,
    batch_size: int = 8,
    feedback_mode: str = "eval_prefix",
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.25,
    transition_scoring_mode: str = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    expected_memory_utility_gate_checkpoint_sha256: str | None = None,
    expected_memory_utility_gate_audit_sha256: str | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    transition_inventory_mask_mode: str = "off",
    transition_inventory_min_candidates: int = 0,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 100,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
    route_records_path: str | Path | None = None,
    route_record_manifest_path: str | Path | None = None,
    route_record_model_digest: str | None = None,
) -> dict[str, Any]:
    reliability_mode = str(reliability_mode or "dynamic")
    if reliability_mode not in SAFE_PUBLIC_RELIABILITY_MODES:
        raise ValueError(f"unsupported reliability_mode: {reliability_mode}")
    if not math.isfinite(float(fixed_alpha)) or not 0.0 <= float(fixed_alpha) <= 1.0:
        raise ValueError("fixed_alpha must be finite and in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_utility_gate, reliability_gate_report = resolve_reliability_gate(
        reliability_mode=reliability_mode,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
        expected_gate_sha256=expected_memory_utility_gate_checkpoint_sha256,
        expected_audit_sha256=expected_memory_utility_gate_audit_sha256,
        device=device,
    )
    if reliability_mode == "learned":
        feature_update_count_cap = float(reliability_gate_report["feature_update_count_cap"])
        feature_candidate_count_cap = float(reliability_gate_report["feature_candidate_count_cap"])
    if (
        route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        and str(transition_inventory_mask_mode or "off") != "off"
    ):
        raise ValueError(
            "memory candidate recall requires transition_inventory_mask_mode=off; "
            "the declared global pool cannot be inferred from namespace inventory filters"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    equivalent_skill_ids = _equivalent_skill_ids_by_skill_id(skills)
    candidate_recall_mode = (
        "static_plus_dynamic_extra"
        if route_scorer == UNIFIED_MEMORY_ROUTE_SCORER
        else "stage0_candidates"
    )
    if candidate_recall_mode == "static_plus_dynamic_extra" and int(top_m) <= 0:
        raise ValueError("top_m must be positive for memory candidate recall")
    final_k = (
        int(candidate_count)
        if candidate_count is not None
        else max(1, min(int(top_m), len(skills)))
    )
    next_skill_pool_mode = (
        "full_pool"
        if candidate_recall_mode == "static_plus_dynamic_extra"
        else "stage0_candidates"
    )
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=Path(skills_path),
        model_cache_dir=output_dir / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    stage4_load_report = None
    if stage4_checkpoint_path:
        stage4_payload = torch.load(stage4_checkpoint_path, map_location="cpu")
        if not isinstance(stage4_payload, dict):
            raise TypeError("Stage4 checkpoint payload must be a dict")
        stage4_delta_validation = validate_stage4_checkpoint_payload(
            stage4_payload
        )
        stage2_router_digest = router_state_digest(model, scope="full")
        stage4_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state",
        )
        stage4_router_digest = router_state_digest(model, scope="full")
        if stage4_router_digest != stage2_router_digest:
            raise ValueError("Stage4 overlay changed the immutable Stage2 router")
        stage4_load_report = {
            **stage4_load_report,
            "stage4_delta_validation": stage4_delta_validation,
            "router_digest_unchanged": True,
        }
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    needs_train_feedback = feedback_mode in {"train_only", "train_plus_eval_prefix"}
    train_source_rows = _read_jsonl(train_trajectories_path) if needs_train_feedback else []
    eval_source_rows = _read_jsonl(eval_trajectories_path)
    if max_train_rows is not None:
        train_source_rows = train_source_rows[: max(0, int(max_train_rows))]
    if max_eval_rows is not None:
        eval_source_rows = eval_source_rows[: max(0, int(max_eval_rows))]
    if needs_train_feedback:
        train_stage4_rows, train_data_report = _stage4_rows_from_source_rows(
            model=model,
            source_rows=train_source_rows,
            skills=skills,
            skill_id_to_idx=skill_id_to_idx,
            output_dir=output_dir,
            manifest_name="train_stage0_candidate_handoff.json",
            stage0_checkpoint_path=stage0_checkpoint_path,
            top_m=top_m,
            candidate_count=candidate_count,
            stage0_handoff_query_mode=stage0_handoff_query_mode,
            stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
            stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
            device=device,
            next_skill_pool_mode=next_skill_pool_mode,
            require_next_state_text=False,
        )
        train_stage4_rows, train_inventory_report = apply_transition_inventory_filter_to_stage4_rows(
            train_stage4_rows,
            skill_id_to_idx,
            transition_inventory_mask_mode=transition_inventory_mask_mode,
            transition_inventory_min_candidates=transition_inventory_min_candidates,
        )
        train_data_report = {
            **train_data_report,
            "transition_inventory_filter": train_inventory_report,
        }
    else:
        train_stage4_rows = []
        train_data_report = {
            "stage4_rows": 0,
            "skipped_reason": "feedback_mode_does_not_use_train_split",
        }
    eval_stage4_rows, target_data_report = _stage4_rows_from_source_rows(
        model=model,
        source_rows=eval_source_rows,
        skills=skills,
        skill_id_to_idx=skill_id_to_idx,
        output_dir=output_dir,
        manifest_name="eval_stage0_candidate_handoff.json",
        stage0_checkpoint_path=stage0_checkpoint_path,
        top_m=top_m,
        candidate_count=candidate_count,
        stage0_handoff_query_mode=stage0_handoff_query_mode,
        stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
        stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
        device=device,
        next_skill_pool_mode=next_skill_pool_mode,
        require_next_state_text=False,
    )
    eval_stage4_rows, eval_inventory_report = apply_transition_inventory_filter_to_stage4_rows(
        eval_stage4_rows,
        skill_id_to_idx,
        transition_inventory_mask_mode=transition_inventory_mask_mode,
        transition_inventory_min_candidates=transition_inventory_min_candidates,
        preserve_positive=False,
    )
    target_data_report = {
        **target_data_report,
        "transition_inventory_filter": eval_inventory_report,
    }

    feedback_rows = _memory_feedback_rows(
        feedback_mode=feedback_mode,
        train_rows=train_stage4_rows,
        eval_rows=eval_stage4_rows,
    )
    scored_eval_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        eval_stage4_rows,
        feedback_rows=feedback_rows,
        next_skill_bonus=online_memory_next_skill_bonus,
        exact_transition_bonus=online_memory_exact_transition_bonus,
        memory_mode=online_memory_mode,
    )
    candidate_eval_kwargs = {
        "candidate_recall_mode": candidate_recall_mode,
        "skill_id_to_idx": skill_id_to_idx,
        "equivalent_skill_ids_by_skill_id": equivalent_skill_ids,
        "static_k": int(top_m),
        "dynamic_extra_k": int(dynamic_extra_k),
        "final_k": int(final_k),
    }
    route_record_kwargs: dict[str, Any] = {}
    if route_records_path is not None:
        effective_model_digest = str(route_record_model_digest or "").strip() or checkpoint_chain_digest(
            {
                "stage0": stage0_checkpoint_path,
                "stage2": stage2_checkpoint_path,
                "stage4": stage4_checkpoint_path,
            }
        )
        route_record_kwargs = {
            "route_records_path": route_records_path,
            "route_record_manifest_path": route_record_manifest_path,
            "route_record_pool_protocol": "known_global",
            "route_record_model_digest": effective_model_digest,
            "route_record_sequential_benchmarks": (
                {"toolbench_g3"}
                if source_rows_have_causal_sequence(eval_source_rows)
                else set()
            ),
        }
    prior_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_eval_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
        route_scorer=route_scorer,
        reliability_mode="static",
        fixed_alpha=1.0,
        **candidate_eval_kwargs,
    )
    prior_eval_by_benchmark = {"toolbench_g3": dict(prior_eval)}
    stage4_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_eval_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
        route_scorer=route_scorer,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        memory_utility_gate=memory_utility_gate,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
        **route_record_kwargs,
        **candidate_eval_kwargs,
    )
    stage4_eval_by_benchmark = {"toolbench_g3": dict(stage4_eval)}
    target_outside_pool_rows = sum(
        1
        for row in eval_source_rows
        if str(row.get("next_skill_id") or "").strip()
        and str(row.get("next_skill_id") or "").strip() not in skill_id_to_idx
    )
    candidate_recall = candidate_recall_protocol_report(
        stage4_eval,
        pool_protocol="known_global",
        candidate_source=str(
            target_data_report.get("candidate_source") or "declared_legal_full_skill_pool"
        ),
        legal_pool_size=len(skills),
        source_rows=len(eval_source_rows),
        static_k=int(top_m),
        dynamic_extra_k=int(dynamic_extra_k),
        final_k=int(final_k),
        causal_sequential=source_rows_have_causal_sequence(eval_source_rows),
        target_outside_declared_legal_pool_rows=target_outside_pool_rows,
    )
    memory_selection = {
        "feedback_mode": feedback_mode,
        "online_memory_mode": online_memory_mode,
        "online_memory_weight": float(online_memory_weight),
        "reliability_mode": reliability_mode,
        "fixed_alpha": float(fixed_alpha),
        "online_memory_report": memory_report,
        "selected_mode_by_benchmark": {"toolbench_g3": online_memory_mode if online_memory_weight != 0.0 else "none"},
    }
    report = build_toolbench_full_clstr_route_report(
        output_dir=output_dir,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        train_trajectories_path=train_trajectories_path,
        eval_trajectories_path=eval_trajectories_path,
        skills_path=skills_path,
        source_eval_rows=len(eval_source_rows),
        retained_eval_rows=len(eval_stage4_rows),
        train_feedback_rows=len(train_stage4_rows),
        target_data_report=target_data_report,
        train_data_report=train_data_report,
        memory_selection=memory_selection,
        prior_eval=prior_eval,
        stage4_eval=stage4_eval,
        prior_eval_by_benchmark=prior_eval_by_benchmark,
        stage4_eval_by_benchmark=stage4_eval_by_benchmark,
        config={
            "top_m": int(top_m),
            "dynamic_extra_k": int(dynamic_extra_k),
            "candidate_count": candidate_count,
            "candidate_recall_mode": candidate_recall_mode,
            "next_skill_pool_mode": next_skill_pool_mode,
            "final_k": int(final_k),
            "batch_size": int(batch_size),
            "feedback_mode": feedback_mode,
            "online_memory_mode": online_memory_mode,
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
            "route_scorer": str(route_scorer),
            "reliability_mode": reliability_mode,
            "fixed_alpha": float(fixed_alpha),
            "reliability_gate": reliability_gate_report,
            "transition_inventory_mask_mode": str(transition_inventory_mask_mode),
            "transition_inventory_min_candidates": int(transition_inventory_min_candidates or 0),
            "stage0_handoff_query_mode": str(stage0_handoff_query_mode),
            "max_train_rows": max_train_rows,
            "max_eval_rows": max_eval_rows,
            "route_records_path": None if route_records_path is None else str(route_records_path),
            "route_record_manifest_path": (
                None if route_record_manifest_path is None else str(route_record_manifest_path)
            ),
        },
        model_load={
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
        },
        candidate_recall=candidate_recall,
    )
    _write_json(output_dir / "full_clstr_route_eval_report.json", report)
    return report
