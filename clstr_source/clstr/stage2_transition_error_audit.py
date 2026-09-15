from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_METRICS = (
    "loss",
    "transition_skill_ce_loss",
    "transition_skill_recall@1",
    "transition_skill_recall@5",
    "transition_skill_mrr",
    "transition_hard_negative_margin_loss",
    "policy_expert_recall@1",
    "retrieval_contrastive_loss",
)

TRANSITION_EVAL_METRICS = {
    "transition_skill_ce_loss",
    "transition_skill_recall@1",
    "transition_skill_recall@5",
    "transition_skill_mrr",
    "transition_skill_ce_candidate_count",
    "transition_skill_ce_count",
    "transition_hard_negative_margin_loss",
}

METRIC_WEIGHT_LOSSES = {
    "policy_ce_loss": "L_policy",
    "policy_expert_recall@1": "L_policy",
    "transition_cosine_loss": "L_trans",
    "transition_skill_ce_loss": "L_trans_skill_ce",
    "transition_skill_recall@1": "L_trans_skill_ce",
    "transition_skill_recall@5": "L_trans_skill_ce",
    "transition_skill_mrr": "L_trans_skill_ce",
    "transition_skill_ce_candidate_count": "L_trans_skill_ce",
    "transition_skill_ce_count": "L_trans_skill_ce",
    "transition_hard_negative_margin_loss": "L_trans_skill_ce",
    "belief_cosine_loss": "belief",
    "stop_bce_loss": "STOP",
    "retrieval_contrastive_loss": "routing",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = min(max(float(q), 0.0), 1.0)
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_training_trend(
    metrics_path: str | Path,
    *,
    first_window: int = 200,
    tail_window: int = 1000,
    metric_keys: tuple[str, ...] = DEFAULT_TRAIN_METRICS,
) -> dict[str, Any]:
    rows = _read_jsonl(metrics_path)
    first_window = max(1, int(first_window))
    tail_window = max(1, int(tail_window))
    max_step = max((int(row.get("step") or 0) for row in rows), default=0)
    report: dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "metric_count": len(rows),
        "max_step": max_step,
        "first_window": first_window,
        "tail_window": tail_window,
    }

    for key in metric_keys:
        values = _numeric_values(rows, key)
        if not values:
            continue
        first_values = values[: min(first_window, len(values))]
        tail_values = values[-min(tail_window, len(values)) :]
        if len(values) > len(tail_values):
            prev_start = max(0, len(values) - len(tail_values) * 2)
            previous_tail_values = values[prev_start : len(values) - len(tail_values)]
        else:
            previous_tail_values = first_values
        first_mean = _mean(first_values)
        previous_tail_mean = _mean(previous_tail_values)
        tail_mean = _mean(tail_values)
        report[key] = {
            "count": len(values),
            "first_mean": _round(first_mean),
            "previous_tail_mean": _round(previous_tail_mean),
            "tail_mean": _round(tail_mean),
            "first_to_tail_delta": _round(None if tail_mean is None or first_mean is None else tail_mean - first_mean),
            "tail_delta": _round(
                None if tail_mean is None or previous_tail_mean is None else tail_mean - previous_tail_mean
            ),
            "last": _round(values[-1]),
        }

    recall_trend = report.get("transition_skill_recall@5") or {}
    recall_tail_delta = recall_trend.get("tail_delta")
    if recall_tail_delta is None:
        signal = "unknown"
    elif recall_tail_delta >= 0.02:
        signal = "still_improving"
    elif recall_tail_delta >= 0.005:
        signal = "modest_tail_improvement"
    else:
        signal = "weak_tail_improvement"
    report["continue_training_signal"] = signal
    return report


def _metric_weight(metric_key: str, batch_size: int, loss_counts: dict[str, int]) -> int:
    loss_key = METRIC_WEIGHT_LOSSES.get(metric_key)
    if loss_key:
        return int(loss_counts.get(loss_key) or 0)
    return int(batch_size)


def summarize_eval_batches(metrics_path: str | Path) -> dict[str, Any]:
    rows = _read_jsonl(metrics_path)
    weighted_sums: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    batch_values: dict[str, list[float]] = {}
    transition_eval_rows = 0
    batches_with_transition = 0

    for row in rows:
        batch_size = int(row.get("batch_size") or 0)
        loss_counts = {str(k): int(v) for k, v in (row.get("loss_mask_counts") or {}).items()}
        transition_count = int(loss_counts.get("L_trans_skill_ce") or 0)
        transition_eval_rows += transition_count
        if transition_count > 0:
            batches_with_transition += 1
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            if key == "weighted_loss_terms" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            weight = _metric_weight(str(key), batch_size, loss_counts)
            if weight <= 0:
                continue
            weighted_sums[str(key)] += float(value) * weight
            weights[str(key)] += weight
            batch_values.setdefault(str(key), []).append(float(value))

    distribution: dict[str, Any] = {}
    for key, values in batch_values.items():
        sorted_values = sorted(values)
        distribution[key] = {
            "batch_count": len(values),
            "min": _round(sorted_values[0]),
            "p25": _round(_quantile(sorted_values, 0.25)),
            "median": _round(_quantile(sorted_values, 0.50)),
            "p75": _round(_quantile(sorted_values, 0.75)),
            "max": _round(sorted_values[-1]),
        }
        if key == "transition_skill_recall@5":
            distribution[key]["zero_batch_fraction"] = _round(sum(1 for value in values if value <= 0.0) / len(values))
            distribution[key]["perfect_batch_fraction"] = _round(sum(1 for value in values if value >= 1.0) / len(values))

    return {
        "metrics_path": str(metrics_path),
        "batch_count": len(rows),
        "batches_with_transition": batches_with_transition,
        "transition_eval_rows": transition_eval_rows,
        "weighted_metrics": {
            key: _round(weighted_sums[key] / weights[key])
            for key in sorted(weighted_sums)
            if weights[key] > 0
        },
        "metric_weights": {key: int(weights[key]) for key in sorted(weights)},
        "batch_distribution": distribution,
    }


def _aggregate_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = ((report.get("aggregate") or {}).get("metrics") or {})
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }


def _handoff_summary(report: dict[str, Any]) -> dict[str, Any]:
    handoff = report.get("stage0_candidate_handoff") or {}
    keys = [
        "source_rows",
        "retained_rows",
        "skipped_rows",
        "current_positive_coverage@M",
        "next_positive_coverage@M",
        "masked_next_skill_ce_rows",
        "injected_positive_rows",
    ]
    return {key: handoff.get(key) for key in keys if key in handoff}


def _build_markdown(report: dict[str, Any]) -> str:
    train = report.get("training_trend") or {}
    eval_report = report.get("eval_report_summary") or {}
    eval_metrics = ((report.get("eval_batch_summary") or {}).get("weighted_metrics") or {})
    baseline = report.get("baseline_comparison") or {}
    findings = report.get("findings") or []
    recommendations = report.get("recommendations") or []
    lines = [
        "# Stage2 Transition Error Audit",
        "",
        f"- status: `{report.get('status')}`",
        f"- findings: `{', '.join(findings) if findings else 'none'}`",
        f"- train metric rows: `{train.get('metric_count')}`",
        f"- train max step: `{train.get('max_step')}`",
        f"- continue training signal: `{train.get('continue_training_signal')}`",
        "",
        "## No-inject Eval",
        "",
        f"- eval status: `{eval_report.get('status')}`",
        f"- eval blockers: `{', '.join(eval_report.get('blockers') or [])}`",
        f"- transition recall@1: `{eval_metrics.get('transition_skill_recall@1')}`",
        f"- transition recall@5: `{eval_metrics.get('transition_skill_recall@5')}`",
        f"- transition MRR: `{eval_metrics.get('transition_skill_mrr')}`",
        f"- transition CE: `{eval_metrics.get('transition_skill_ce_loss')}`",
        "",
        "## Baseline Comparison",
        "",
        f"- recall@5 delta: `{baseline.get('transition_skill_recall@5_delta')}`",
        f"- recall@1 delta: `{baseline.get('transition_skill_recall@1_delta')}`",
        f"- CE delta: `{baseline.get('transition_skill_ce_loss_delta')}`",
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines) + "\n"


def audit_stage2_transition_outputs(
    *,
    train_metrics_path: str | Path,
    eval_metrics_path: str | Path,
    eval_report_path: str | Path,
    baseline_eval_report_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    first_window: int = 200,
    tail_window: int = 1000,
    min_meaningful_recall5_gain: float = 0.005,
    min_eval_recall5: float = 0.5,
) -> dict[str, Any]:
    training_trend = summarize_training_trend(
        train_metrics_path,
        first_window=first_window,
        tail_window=tail_window,
    )
    eval_batch_summary = summarize_eval_batches(eval_metrics_path)
    eval_report = _read_json(eval_report_path)
    baseline_report = _read_json(baseline_eval_report_path)
    eval_metrics = _aggregate_metrics(eval_report)
    baseline_metrics = _aggregate_metrics(baseline_report)

    baseline_comparison: dict[str, Any] = {}
    for key in ("transition_skill_recall@5", "transition_skill_recall@1", "transition_skill_mrr", "transition_skill_ce_loss"):
        if key in eval_metrics and key in baseline_metrics:
            baseline_comparison[f"{key}_delta"] = _round(eval_metrics[key] - baseline_metrics[key])

    findings: list[str] = []
    eval_recall5 = eval_metrics.get("transition_skill_recall@5")
    if eval_report.get("status") != "ok":
        findings.extend(str(item) for item in eval_report.get("blockers") or [])
    if eval_recall5 is not None and eval_recall5 < float(min_eval_recall5):
        if "transition_recall_at_5_below_threshold" not in findings:
            findings.append("transition_recall_at_5_below_threshold")
    recall_delta = baseline_comparison.get("transition_skill_recall@5_delta")
    if recall_delta is not None and recall_delta < float(min_meaningful_recall5_gain):
        findings.append("no_meaningful_no_inject_recall5_gain")
    if training_trend.get("continue_training_signal") == "weak_tail_improvement":
        findings.append("transition_training_tail_plateau")

    train_recall_tail = (training_trend.get("transition_skill_recall@5") or {}).get("tail_mean")
    if train_recall_tail is not None and eval_recall5 is not None and train_recall_tail - eval_recall5 >= 0.02:
        findings.append("train_eval_transition_gap")

    recommendations = [
        "Do not spend another full job on pure Stage2 v3 loss-weight tuning before row-level diagnosis.",
        "Add row-level transition ranking diagnostics if exact benchmark/source failure attribution is needed.",
        "Prioritize Stage2 v4: multi-positive listwise next-skill reranking over Stage0 top-M candidates.",
        "Audit equivalent next-skill labels and candidate hard negatives before increasing training steps.",
    ]
    if training_trend.get("continue_training_signal") in {"still_improving", "modest_tail_improvement"}:
        recommendations.insert(
            1,
            "A short continuation run may be useful, but gate it with no-inject eval rather than training recall.",
        )

    status = "ok" if not findings else "action_required"
    report = {
        "status": status,
        "findings": sorted(set(findings)),
        "training_trend": training_trend,
        "eval_report_summary": {
            "status": eval_report.get("status"),
            "blockers": eval_report.get("blockers") or [],
            "stage0_handoff": _handoff_summary(eval_report),
        },
        "eval_batch_summary": eval_batch_summary,
        "baseline_comparison": baseline_comparison,
        "recommendations": recommendations,
    }
    _write_json(output_json_path, report)
    if output_markdown_path is not None:
        output_markdown_path = Path(output_markdown_path)
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(_build_markdown(report), encoding="utf-8")
    return report
