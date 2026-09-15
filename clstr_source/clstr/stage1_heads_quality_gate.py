from __future__ import annotations

from pathlib import Path
from typing import Any

from clstr.stage2_quality_gate import (
    UNIFIED_MEMORY_ROUTE_SCORER,
    _max_step,
    _missing_required_loss_terms,
    _route_scorer_safety,
)
from clstr.stage_quality_common import (
    EXPECTED_TRANSITION_SCORING_MODE,
    artifact_exists as _path_exists,
    bounded_window_size,
    read_json_file as _read_json,
    read_jsonl_file as _read_jsonl,
    transition_scoring_safety as _transition_scoring_safety,
    window_mean as _window_mean,
    write_json_report,
)


DEFAULT_REQUIRED_LOSS_TERMS = ["L_policy", "L_trans", "L_trans_skill_ce", "belief"]
STAGE1_EXPECTED_TRANSITION_SCORING_MODE = EXPECTED_TRANSITION_SCORING_MODE
STAGE1_EXPECTED_TRANSITION_RESIDUAL_LAMBDA = 0.25


def _action_aware_transition_input_semantics(train_report: dict[str, Any]) -> bool:
    semantics = train_report.get("transition_input_semantics")
    if not isinstance(semantics, dict):
        return False
    return (
        str(semantics.get("action_channel") or "")
        == "actual_action_text_embedding_via_action_proj_when_available_else_current_skill_embedding"
        and str(semantics.get("observation_channel") or "") == "next_observation_text_embedding"
        and semantics.get("uses_actual_action_text_when_available") is True
        and semantics.get("uses_next_observation_as_observation") is True
        and semantics.get("action_text_used_as_observation") is False
    )


def _read_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _transition_hard_negative_safety(train_report: dict[str, Any]) -> dict[str, Any]:
    loss_weights = train_report.get("loss_weights")
    loss_weight = 0.0
    if isinstance(loss_weights, dict):
        loss_weight = _read_float(loss_weights.get("transition_hard_negative_margin"))
    candidate_training = train_report.get("transition_candidate_training")
    candidate_training_loss_weight = 0.0
    if isinstance(candidate_training, dict):
        candidate_training_loss_weight = _read_float(candidate_training.get("hard_negative_loss_weight"))
    enabled = (
        loss_weight > 0.0
        and candidate_training_loss_weight > 0.0
        and abs(loss_weight - candidate_training_loss_weight) <= 1.0e-12
    )
    configured = loss_weight > 0.0 or candidate_training_loss_weight > 0.0
    consistent = abs(loss_weight - candidate_training_loss_weight) <= 1.0e-12
    return {
        "enabled": enabled,
        "configured": configured,
        "consistent": consistent,
        "loss_weight": loss_weight,
        "candidate_training_loss_weight": candidate_training_loss_weight,
        "hard_negative_loss_weight": candidate_training_loss_weight,
        "transition_candidate_training": candidate_training if isinstance(candidate_training, dict) else {},
    }


def _report_metric(train_report: dict[str, Any], key: str) -> float:
    values: list[float] = []
    for field in ("metrics", "metric_averages", "last_batch_metrics"):
        metrics = train_report.get(field)
        if isinstance(metrics, dict):
            values.append(_read_float(metrics.get(key)))
    return max(values) if values else 0.0


def _stage1_belief_loss_optional(train_report: dict[str, Any], route_scorer: str) -> bool:
    if route_scorer != UNIFIED_MEMORY_ROUTE_SCORER:
        return False
    auto_replay_prefix = train_report.get("auto_replay_prefix")
    if not isinstance(auto_replay_prefix, dict):
        return False
    replay_disabled = (
        auto_replay_prefix.get("enabled") is False
        and int(auto_replay_prefix.get("max_steps") or 0) <= 0
        and int(auto_replay_prefix.get("total_prefix_steps") or 0) <= 0
    )
    trainable_replay_disabled = train_report.get("trainable_replay_prefix") is False
    return bool(replay_disabled and trainable_replay_disabled)


def _effective_stage1_required_loss_terms(
    train_report: dict[str, Any],
    required_loss_terms: list[str],
    route_scorer: str,
) -> tuple[list[str], list[str]]:
    effective = list(required_loss_terms)
    optional_due_to_config: list[str] = []
    if "belief" in effective and _stage1_belief_loss_optional(train_report, route_scorer):
        effective.remove("belief")
        optional_due_to_config.append("belief")
    return effective, optional_due_to_config


def _historical_stage1_topm_metadata_compatible(
    train_report: dict[str, Any],
    rows: list[dict[str, Any]],
    transition_scoring_safety: dict[str, Any],
) -> bool:
    """Accept older Stage1 artifacts that predate explicit scoring metadata."""
    if transition_scoring_safety.get("observed"):
        return False
    handoff = train_report.get("stage0_candidate_handoff")
    if not isinstance(handoff, dict) or handoff.get("candidate_source") != "stage0_topm_online":
        return False
    if str(handoff.get("positive_missing_policy") or "") != "skip":
        return False
    if int(handoff.get("injected_positive_rows") or 0) > 0:
        return False

    candidate_training = train_report.get("transition_candidate_training")
    if isinstance(candidate_training, dict):
        loss_type = str(candidate_training.get("loss_type") or "")
        positive_mode = str(candidate_training.get("positive_mode") or "")
        if loss_type and loss_type != "listwise_nll":
            return False
        if positive_mode and positive_mode != "gold_plus_equivalent":
            return False

    head_types = {
        str(row.get("transition_skill_head_type") or "")
        for row in rows
        if row.get("transition_skill_head_type")
    }
    if head_types != {"native_trans_head_stage0_topm"}:
        return False
    row_loss_types = {
        str(row.get("transition_loss_type") or "")
        for row in rows
        if row.get("transition_loss_type")
    }
    if row_loss_types and row_loss_types != {"listwise_nll"}:
        return False
    row_positive_modes = {
        str(row.get("transition_positive_mode") or "")
        for row in rows
        if row.get("transition_positive_mode")
    }
    if row_positive_modes and row_positive_modes != {"gold_plus_equivalent"}:
        return False
    return True


def audit_stage1_heads_quality(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    train_report_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_path: str | Path | None = None,
    min_steps: int = 3000,
    first_window: int = 100,
    last_window: int = 100,
    min_loss_drop: float = 0.0,
    required_loss_terms: list[str] | None = None,
    max_transition_recall_at_5_drop: float = 0.1,
    max_transition_skill_ce_increase: float = 0.2,
    min_transition_recall_at_1_tail: float = 0.35,
    min_transition_recall_at_5_tail: float = 0.7,
    expected_transition_scoring_mode: str = STAGE1_EXPECTED_TRANSITION_SCORING_MODE,
    expected_transition_residual_lambda: float | None = STAGE1_EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    expected_route_scorer: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_report_path = Path(train_report_path) if train_report_path is not None else output_dir / "train_report.json"
    train_report = _read_json(train_report_path)
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(train_report.get("checkpoint") or output_dir / "checkpoints" / f"clstr_stage1_heads-step{min_steps}.pt")
    )
    metrics_path = Path(metrics_path) if metrics_path is not None else Path(train_report.get("training_metrics_path") or output_dir / "training_metrics.jsonl")
    required_loss_terms = list(required_loss_terms or DEFAULT_REQUIRED_LOSS_TERMS)
    blockers: list[str] = []
    transition_hard_negative_safety = _transition_hard_negative_safety(train_report)

    if not train_report_path.is_file():
        blockers.append("missing_train_report")
    if train_report and train_report.get("status") != "ok":
        blockers.append("stage1_train_report_not_ok")
    if train_report and train_report.get("stage") != "clstr_stage1_heads_init":
        blockers.append("unexpected_stage1_stage")
    if train_report and train_report.get("training_objective") != "stage1_heads_init_topm_supervised":
        blockers.append("unexpected_training_objective")
    if train_report and train_report.get("frozen_routing_foundation") is not True:
        blockers.append("routing_foundation_not_frozen")
    if train_report and train_report.get("checkpoint_excludes_frozen_routing_foundation") is not True:
        blockers.append("stage1_checkpoint_includes_frozen_routing_foundation")
    handoff = train_report.get("stage0_candidate_handoff")
    if train_report and (not isinstance(handoff, dict) or handoff.get("candidate_source") != "stage0_topm_online"):
        blockers.append("stage1_not_using_stage0_topm")
    if isinstance(handoff, dict):
        if str(handoff.get("positive_missing_policy") or "") != "skip":
            blockers.append("stage1_positive_missing_policy_not_skip")
        if int(handoff.get("injected_positive_rows") or 0) > 0:
            blockers.append("stage1_injected_stage0_positives")

    if not _path_exists(checkpoint_path):
        blockers.append("missing_stage1_checkpoint")
    rows = _read_jsonl(metrics_path)
    if not metrics_path.is_file():
        blockers.append("missing_training_metrics")
    transition_scoring_safety = _transition_scoring_safety(
        train_report,
        rows,
        expected_mode=str(expected_transition_scoring_mode or STAGE1_EXPECTED_TRANSITION_SCORING_MODE),
        expected_residual_lambda=expected_transition_residual_lambda,
    )
    route_scorer_safety = _route_scorer_safety(train_report, rows, expected_route_scorer=expected_route_scorer)
    unified_route_scorer = route_scorer_safety.get("expected") == UNIFIED_MEMORY_ROUTE_SCORER
    if train_report and route_scorer_safety["matched"] is not True:
        blockers.append("stage1_route_scorer_mismatch")
    historical_stage1_topm_metadata_compatible = _historical_stage1_topm_metadata_compatible(
        train_report,
        rows,
        transition_scoring_safety,
    )
    action_aware_transition_input = _action_aware_transition_input_semantics(train_report)
    if (
        train_report
        and not unified_route_scorer
        and not (action_aware_transition_input or historical_stage1_topm_metadata_compatible)
    ):
        blockers.append("stage1_transition_input_semantics_not_action_aware")
    if (
        train_report
        and not unified_route_scorer
        and transition_scoring_safety["matched"] is not True
        and not historical_stage1_topm_metadata_compatible
    ):
        blockers.append("stage1_transition_scoring_mismatch")
    max_step = _max_step(rows, train_report)
    if max_step < int(min_steps):
        blockers.append("insufficient_training_steps")

    first_window = bounded_window_size(first_window, len(rows))
    last_window = bounded_window_size(last_window, len(rows))
    first_loss = _window_mean(rows, "loss", first_window, tail=False)
    last_loss = _window_mean(rows, "loss", last_window, tail=True)
    loss_drop = None if first_loss is None or last_loss is None else round(float(first_loss - last_loss), 12)
    if loss_drop is None or loss_drop < float(min_loss_drop):
        blockers.append("insufficient_learning_signal")

    transition_recall_at_1_last = _window_mean(rows, "transition_skill_recall@1", last_window, tail=True)
    transition_recall_at_5_first = _window_mean(rows, "transition_skill_recall@5", first_window, tail=False)
    transition_recall_at_5_last = _window_mean(rows, "transition_skill_recall@5", last_window, tail=True)
    transition_recall_at_5_drop = (
        None
        if transition_recall_at_5_first is None or transition_recall_at_5_last is None
        else round(float(transition_recall_at_5_first - transition_recall_at_5_last), 12)
    )
    transition_ce_first = _window_mean(rows, "transition_skill_ce_loss", first_window, tail=False)
    transition_ce_last = _window_mean(rows, "transition_skill_ce_loss", last_window, tail=True)
    transition_ce_increase = (
        None
        if transition_ce_first is None or transition_ce_last is None
        else round(float(transition_ce_last - transition_ce_first), 12)
    )
    transition_hard_negative_pair_count_last = _window_mean(
        rows,
        "transition_hard_negative_pair_count",
        last_window,
        tail=True,
    )
    transition_hard_negative_pair_count = max(
        float(transition_hard_negative_pair_count_last or 0.0),
        _report_metric(train_report, "transition_hard_negative_pair_count"),
    )
    if (
        transition_recall_at_1_last is None
        or transition_recall_at_5_drop is None
        or transition_ce_increase is None
    ):
        blockers.append("missing_transition_quality_metrics")
    else:
        if transition_recall_at_1_last < float(min_transition_recall_at_1_tail):
            blockers.append("transition_recall_at_1_too_low")
        if transition_recall_at_5_last is not None and transition_recall_at_5_last < float(min_transition_recall_at_5_tail):
            blockers.append("transition_recall_at_5_too_low")
        if transition_recall_at_5_drop > float(max_transition_recall_at_5_drop):
            blockers.append("transition_recall_at_5_regressed")
        if transition_ce_increase > float(max_transition_skill_ce_increase):
            blockers.append("transition_skill_ce_regressed")

    effective_required_loss_terms, optional_loss_terms = _effective_stage1_required_loss_terms(
        train_report,
        required_loss_terms,
        str(route_scorer_safety.get("expected") or ""),
    )
    missing_terms = _missing_required_loss_terms(train_report, effective_required_loss_terms)
    if missing_terms:
        blockers.append("missing_required_loss_terms")
    if train_report and transition_hard_negative_safety["configured"] and transition_hard_negative_safety["consistent"] is not True:
        blockers.append("transition_hard_negative_margin_inconsistent")
    if train_report and transition_hard_negative_safety["enabled"] and transition_hard_negative_pair_count <= 0.0:
        blockers.append("transition_hard_negative_pairs_missing")

    latest_checkpoint = train_report.get("latest_checkpoint") or output_dir / "checkpoints" / "latest.pt"
    loss_curve = train_report.get("loss_curve_path") or output_dir / "loss_curve.svg"
    if not _path_exists(latest_checkpoint):
        blockers.append("missing_latest_checkpoint")
    if not _path_exists(loss_curve):
        blockers.append("missing_loss_curve")

    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": sorted(set(blockers)),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_report_path": str(train_report_path),
        "metrics_path": str(metrics_path),
        "metrics_summary": {
            "metric_count": len(rows),
            "max_step": max_step,
            "first_window": first_window,
            "last_window": last_window,
            "first_loss_mean": first_loss,
            "last_loss_mean": last_loss,
            "loss_drop": loss_drop,
            "min_steps": int(min_steps),
            "min_loss_drop": float(min_loss_drop),
        },
        "transition_quality": {
            "last_transition_skill_recall@1": transition_recall_at_1_last,
            "min_transition_recall_at_1_tail": float(min_transition_recall_at_1_tail),
            "first_transition_skill_recall@5": transition_recall_at_5_first,
            "last_transition_skill_recall@5": transition_recall_at_5_last,
            "transition_skill_recall@5_drop": transition_recall_at_5_drop,
            "max_transition_recall_at_5_drop": float(max_transition_recall_at_5_drop),
            "min_transition_recall_at_5_tail": float(min_transition_recall_at_5_tail),
            "first_transition_skill_ce_loss": transition_ce_first,
            "last_transition_skill_ce_loss": transition_ce_last,
            "transition_skill_ce_increase": transition_ce_increase,
            "max_transition_skill_ce_increase": float(max_transition_skill_ce_increase),
        },
        "required_loss_terms": {
            "required": required_loss_terms,
            "effective_required": effective_required_loss_terms,
            "optional_due_to_config": optional_loss_terms,
            "loss_activation_counts": train_report.get("loss_activation_counts") if isinstance(train_report.get("loss_activation_counts"), dict) else {},
            "sampled_loss_activation_counts": train_report.get("sampled_loss_activation_counts") if isinstance(train_report.get("sampled_loss_activation_counts"), dict) else {},
            "missing_or_unsampled": missing_terms,
        },
        "transition_hard_negative_safety": {
            "required": False,
            "hard_negative_pair_count": transition_hard_negative_pair_count,
            "last_window_hard_negative_pair_count": transition_hard_negative_pair_count_last,
            **transition_hard_negative_safety,
        },
        "checkpoint_safety": {
            "frozen_routing_foundation": train_report.get("frozen_routing_foundation"),
            "checkpoint_excludes_frozen_routing_foundation": train_report.get("checkpoint_excludes_frozen_routing_foundation"),
            "excluded_state_key_prefixes": train_report.get("excluded_state_key_prefixes")
            if isinstance(train_report.get("excluded_state_key_prefixes"), list)
            else [],
        },
        "transition_input_safety": {
            "action_aware_next_observation": action_aware_transition_input,
            "historical_stage1_topm_metadata_compatible": historical_stage1_topm_metadata_compatible,
            "transition_input_semantics": train_report.get("transition_input_semantics")
            if isinstance(train_report.get("transition_input_semantics"), dict)
            else {},
            "transition_scoring_safety": transition_scoring_safety,
            "route_scorer_safety": route_scorer_safety,
        },
        "route_scorer_safety": route_scorer_safety,
        "handoff_safety": {
            "candidate_source": handoff.get("candidate_source") if isinstance(handoff, dict) else None,
            "positive_missing_policy": handoff.get("positive_missing_policy") if isinstance(handoff, dict) else None,
            "injected_positive_rows": handoff.get("injected_positive_rows") if isinstance(handoff, dict) else None,
        },
        "paper_scope_note": (
            "This gate only proves Stage1 heads initialization is ready for "
            "Stage2 continuation. It is not a benchmark result."
        ),
    }
    if output_path is not None:
        write_json_report(output_path, report)
    return report
