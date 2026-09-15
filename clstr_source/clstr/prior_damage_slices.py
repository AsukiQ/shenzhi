from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


_NUMERIC_METRICS = (
    "loss",
    "transition_skill_ce_loss",
    "transition_skill_recall@1",
    "transition_skill_recall@5",
    "transition_skill_mrr",
    "stage0_prior_transition_skill_recall@1",
    "stage0_prior_transition_skill_recall@5",
    "stage0_prior_transition_skill_mrr",
    "transition_delta_vs_stage0_prior_recall@1",
    "transition_delta_vs_stage0_prior_recall@5",
    "transition_delta_vs_stage0_prior_mrr",
    "transition_worse_than_stage0_prior_fraction",
    "transition_improved_vs_stage0_prior_fraction",
    "transition_skill_ce_count",
    "transition_current_skill_switch_rows",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _slice(count: float, denominator: float, *, estimated: bool = False) -> dict[str, Any]:
    key = "estimated_count" if estimated else "count"
    return {
        key: float(count) if estimated else int(round(count)),
        "denominator": int(round(denominator)) if denominator else 0,
        "fraction": _fraction(float(count), float(denominator)),
        "estimated": bool(estimated),
    }


def _rank_bucket_counts(handoff_report: dict[str, Any]) -> dict[str, int]:
    raw = handoff_report.get("next_positive_rank_bucket_counts") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): _int(value) for key, value in raw.items()}


def _summarize_metric_window(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    summary: dict[str, float] = {}
    for key in _NUMERIC_METRICS:
        values = [_number(row.get(key), default=float("nan")) for row in rows if isinstance(row.get(key), (int, float))]
        values = [value for value in values if value == value]
        if values:
            summary[key] = float(mean(values))
    return summary


def _metric_windows(metric_rows: list[dict[str, Any]], window: int) -> dict[str, dict[str, float]]:
    window = max(1, int(window))
    return {
        "first": _summarize_metric_window(metric_rows[:window]),
        "last": _summarize_metric_window(metric_rows[-window:]),
        "all": _summarize_metric_window(metric_rows),
    }


def _row_rank(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _row_transition_relation(row: dict[str, Any]) -> str:
    for key in ("transition_relation", "current_next_relation", "relation"):
        value = row.get(key)
        if value:
            return str(value)
    current = row.get("skill_id")
    nxt = row.get("next_skill_id")
    if current and nxt:
        return "self" if str(current) == str(nxt) else "switch"
    return "unknown"


def _build_row_slices(row_records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    denominator = len(row_records)
    worse_rows: list[dict[str, Any]] = []
    improved_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    stage0_top1_wrong_top5: list[dict[str, Any]] = []
    stage0_top1_wrong_top20: list[dict[str, Any]] = []
    gt_missing: list[dict[str, Any]] = []

    by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in row_records:
        stage0_rank = _row_rank(row, "stage0_gold_rank", "stage0_rank", "stage0_next_skill_rank")
        final_rank = _row_rank(row, "stage2_gold_rank", "clstr_gold_rank", "final_gold_rank")
        benchmark = str(row.get("benchmark") or "unknown")
        missing = bool(row.get("gt_missing_from_topM")) or stage0_rank is None or stage0_rank <= 0
        if missing:
            gt_missing.append(row)
            by_benchmark[benchmark]["gt_missing_from_topM"].append(row)
        if stage0_rank is not None and 1 < stage0_rank <= 5:
            stage0_top1_wrong_top5.append(row)
            by_benchmark[benchmark]["stage0_top1_wrong_top5_contains_gt"].append(row)
        if stage0_rank is not None and 1 < stage0_rank <= 20:
            stage0_top1_wrong_top20.append(row)
            by_benchmark[benchmark]["stage0_top1_wrong_top20_contains_gt"].append(row)
        if final_rank is not None and stage0_rank is not None:
            if final_rank > stage0_rank:
                worse_rows.append(row)
                by_benchmark[benchmark]["clstr_worse_than_stage0_prior"].append(row)
            elif final_rank < stage0_rank:
                improved_rows.append(row)
                by_benchmark[benchmark]["clstr_improved_vs_stage0_prior"].append(row)
        if _row_transition_relation(row) == "switch":
            switch_rows.append(row)
            by_benchmark[benchmark]["transition_switch_rows"].append(row)

    slices = {
        "gt_missing_from_topM": _slice(len(gt_missing), denominator),
        "stage0_top1_wrong_top5_contains_gt": _slice(len(stage0_top1_wrong_top5), denominator),
        "stage0_top1_wrong_top20_contains_gt": _slice(len(stage0_top1_wrong_top20), denominator),
        "clstr_worse_than_stage0_prior": _slice(len(worse_rows), denominator),
        "clstr_improved_vs_stage0_prior": _slice(len(improved_rows), denominator),
        "transition_switch_rows": _slice(len(switch_rows), denominator),
    }
    benchmark_summary: dict[str, Any] = {}
    for benchmark, grouped in sorted(by_benchmark.items()):
        benchmark_denominator = sum(1 for row in row_records if str(row.get("benchmark") or "unknown") == benchmark)
        benchmark_summary[benchmark] = {
            name: _slice(len(items), benchmark_denominator)
            for name, items in sorted(grouped.items())
        }
    return slices, benchmark_summary


def _build_handoff_slices(handoff_report: dict[str, Any]) -> dict[str, Any]:
    required_next = _int(handoff_report.get("next_positive_required_rows"))
    covered_next = _int(handoff_report.get("next_positive_covered_rows"))
    missing_next = _int(handoff_report.get("masked_next_skill_ce_rows"))
    if required_next <= 0 and (covered_next or missing_next):
        required_next = covered_next + missing_next
    bucket_counts = _rank_bucket_counts(handoff_report)
    denominator = required_next or sum(bucket_counts.values())
    missing_count = missing_next or bucket_counts.get("missing", 0)
    top2_to_5 = bucket_counts.get("2-5", 0)
    top6_to_20 = bucket_counts.get("6-20", 0)
    top21_to_100 = bucket_counts.get("21-100", 0)
    over100 = bucket_counts.get(">100", 0)
    return {
        "gt_missing_from_topM": _slice(missing_count, denominator),
        "stage0_top1_wrong_top5_contains_gt": _slice(top2_to_5, denominator),
        "stage0_top1_wrong_top20_contains_gt": _slice(top2_to_5 + top6_to_20, denominator),
        "stage0_top20_wrong_top100_contains_gt": _slice(top21_to_100, denominator),
        "stage0_gt_rank_gt100": _slice(over100, denominator),
    }


def _estimate_metric_tail_slices(metric_rows: list[dict[str, Any]], tail_window: int) -> dict[str, Any]:
    tail = metric_rows[-max(1, int(tail_window)) :]
    if not tail:
        return {}
    ce_count = sum(_number(row.get("transition_skill_ce_count")) for row in tail)
    worse_count = sum(
        _number(row.get("transition_worse_than_stage0_prior_fraction")) * _number(row.get("transition_skill_ce_count"))
        for row in tail
    )
    improved_count = sum(
        _number(row.get("transition_improved_vs_stage0_prior_fraction")) * _number(row.get("transition_skill_ce_count"))
        for row in tail
    )
    switch_count = sum(_number(row.get("transition_current_skill_switch_rows")) for row in tail)
    return {
        "clstr_worse_than_stage0_prior": _slice(worse_count, ce_count, estimated=True),
        "clstr_improved_vs_stage0_prior": _slice(improved_count, ce_count, estimated=True),
        "transition_switch_rows": _slice(switch_count, ce_count, estimated=True),
    }


def _merge_slices(
    *,
    row_slices: dict[str, Any],
    handoff_slices: dict[str, Any],
    metric_tail_slices: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(handoff_slices)
    for key, value in metric_tail_slices.items():
        merged.setdefault(key, value)
    for key, value in row_slices.items():
        # Handoff is the authoritative source for Stage0 coverage/rank buckets
        # because row-level diagnostics usually only contain retained candidates.
        if key in {
            "gt_missing_from_topM",
            "stage0_top1_wrong_top5_contains_gt",
            "stage0_top1_wrong_top20_contains_gt",
        }:
            continue
        # Row-level records are more specific for CLSTR-vs-prior movement.
        if value.get("denominator", 0) > 0:
            merged[key] = value
    return merged


def _decision(slices: dict[str, Any], metric_windows: dict[str, dict[str, float]], row_count: int) -> dict[str, Any]:
    missing = slices.get("gt_missing_from_topM", {})
    worse = slices.get("clstr_worse_than_stage0_prior", {})
    missing_fraction = _number(missing.get("fraction"))
    worse_fraction = _number(worse.get("fraction"))
    last = metric_windows.get("last") or {}
    delta_mrr = _number(last.get("transition_delta_vs_stage0_prior_mrr"))
    recall5 = _number(last.get("transition_skill_recall@5"))
    prior_recall5 = _number(last.get("stage0_prior_transition_skill_recall@5"))

    primary = "mixed_or_unclear"
    reasons: list[str] = []
    if missing_fraction >= 0.2 and missing_fraction >= worse_fraction:
        primary = "candidate_recall"
        reasons.append("gt_missing_from_topM_fraction_high")
    elif row_count >= 20 and worse_fraction >= 0.2:
        primary = "prior_damage_or_gate"
        reasons.append("row_level_worse_than_stage0_fraction_high")
    elif row_count == 0 and worse_fraction >= 0.2:
        primary = "prior_damage_or_gate"
        reasons.append("metric_tail_worse_than_stage0_fraction_high")
    elif delta_mrr < -0.01 or (prior_recall5 - recall5) > 0.05:
        primary = "prior_damage_or_gate"
        reasons.append("tail_metrics_show_stage0_prior_damage")
    else:
        reasons.append("no_single_dominant_failure_slice")

    return {
        "primary_bottleneck": primary,
        "reasons": reasons,
        "missing_fraction": missing_fraction,
        "worse_than_stage0_prior_fraction": worse_fraction,
        "last_transition_delta_vs_stage0_prior_mrr": delta_mrr,
        "last_transition_recall@5": recall5,
        "last_stage0_prior_transition_recall@5": prior_recall5,
    }


def build_prior_damage_slice_report(
    *,
    row_records: list[dict[str, Any]] | None = None,
    handoff_report: dict[str, Any] | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
    tail_window: int = 100,
    run_id: str = "",
) -> dict[str, Any]:
    row_records = list(row_records or [])
    handoff_report = dict(handoff_report or {})
    metric_rows = list(metric_rows or [])
    row_slices, by_benchmark = _build_row_slices(row_records)
    handoff_slices = _build_handoff_slices(handoff_report)
    metric_tail_slices = _estimate_metric_tail_slices(metric_rows, tail_window)
    windows = _metric_windows(metric_rows, tail_window)
    slices = _merge_slices(
        row_slices=row_slices,
        handoff_slices=handoff_slices,
        metric_tail_slices=metric_tail_slices,
    )
    report = {
        "run_id": str(run_id),
        "row_record_count": len(row_records),
        "metric_row_count": len(metric_rows),
        "tail_window": max(1, int(tail_window)),
        "slices": slices,
        "by_benchmark": by_benchmark,
        "metric_windows": windows,
        "handoff_summary": {
            "top_m": handoff_report.get("top_m"),
            "candidate_source": handoff_report.get("candidate_source"),
            "next_positive_required_rows": handoff_report.get("next_positive_required_rows"),
            "next_positive_covered_rows": handoff_report.get("next_positive_covered_rows"),
            "masked_next_skill_ce_rows": handoff_report.get("masked_next_skill_ce_rows"),
            "next_positive_rank_bucket_counts": handoff_report.get("next_positive_rank_bucket_counts"),
        },
        "decision": {},
    }
    report["decision"] = _decision(slices, windows, len(row_records))
    return report


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _candidate_stage_dirs(run_dir: Path) -> list[Path]:
    return [
        run_dir,
        run_dir / "stage1_heads_init",
        run_dir / "stage2_full_base",
        run_dir / "stage2",
    ]


def load_prior_damage_inputs(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    stage_dirs = _candidate_stage_dirs(run_dir)
    metrics_path = _find_first_existing([path / "training_metrics.jsonl" for path in stage_dirs])
    handoff_path = _find_first_existing(
        [
            path / "stage0_candidate_handoff.json"
            for path in stage_dirs
        ]
        + [
            path / "stage0_candidate_handoff_no_inject.json"
            for path in stage_dirs
        ]
    )
    row_records_path = _find_first_existing(
        [
            path / "transition_row_diagnostics.jsonl"
            for path in stage_dirs
        ]
        + [
            path / "row_diagnostics.jsonl"
            for path in stage_dirs
        ]
    )
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path) if metrics_path else "",
        "handoff_path": str(handoff_path) if handoff_path else "",
        "row_records_path": str(row_records_path) if row_records_path else "",
        "metric_rows": _read_jsonl(metrics_path) if metrics_path else [],
        "handoff_report": _read_json(handoff_path) if handoff_path else {},
        "row_records": _read_jsonl(row_records_path) if row_records_path else [],
    }


def write_prior_damage_slice_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
