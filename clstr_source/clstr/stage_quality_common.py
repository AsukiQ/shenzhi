from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_json_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_jsonl_file(path: str | Path, *, missing_ok: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    if missing_ok and not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def artifact_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).is_file() and Path(path).stat().st_size > 0


def finite_float(value: Any, *, allow_bool: bool = False) -> float | None:
    if isinstance(value, bool) and not allow_bool:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def window_mean(rows: list[dict[str, Any]], key: str, window: int, *, tail: bool) -> float | None:
    selected = rows[-window:] if tail else rows[:window]
    values = [value for row in selected if (value := finite_float(row.get(key))) is not None]
    return mean(values)


def bounded_window_size(requested: int, row_count: int) -> int:
    return max(1, min(int(requested), row_count if row_count else 1))


def source_coverage(rows: list[dict[str, Any]]) -> tuple[list[str], int]:
    observed: set[str] = set()
    rows_with_sources = 0
    for row in rows:
        sources = row.get("query_sources")
        if isinstance(sources, list) and sources:
            rows_with_sources += 1
            observed.update(str(source) for source in sources if str(source))
    return sorted(observed), rows_with_sources


def max_counter(
    rows: list[dict[str, Any]],
    train_report: dict[str, Any] | None = None,
    *,
    row_keys: tuple[str, ...] = ("step", "update"),
    report_keys: tuple[str, ...] = ("max_steps", "updates", "step"),
) -> int:
    values: list[int] = []
    for row in rows:
        raw = None
        for key in row_keys:
            if key in row:
                raw = row.get(key)
                break
        if isinstance(raw, int) and not isinstance(raw, bool):
            values.append(raw)
    if train_report:
        raw_report_value = None
        for key in report_keys:
            if key in train_report and train_report.get(key):
                raw_report_value = train_report.get(key)
                break
        if isinstance(raw_report_value, int) and not isinstance(raw_report_value, bool):
            values.append(raw_report_value)
    return max(values or [0])


def max_step_or_update(rows: list[dict[str, Any]], train_report: dict[str, Any] | None = None) -> int:
    return max_counter(rows, train_report)


def benchmark_counts(train_report: dict[str, Any]) -> dict[str, int]:
    data_report = train_report.get("data_report")
    if not isinstance(data_report, dict):
        return {}
    counts = data_report.get("benchmark_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in counts.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


EXPECTED_TRANSITION_SCORING_MODE = "stage0_rank_prior_plus_transition_residual"
EXPECTED_TRANSITION_RESIDUAL_LAMBDA = 0.25


def _parse_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def transition_scoring_safety(
    train_report: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    *,
    expected_mode: str = EXPECTED_TRANSITION_SCORING_MODE,
    expected_residual_lambda: float | None = EXPECTED_TRANSITION_RESIDUAL_LAMBDA,
    lambda_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    omitted_metric_rows = 0

    def add_candidate(source: str, payload: dict[str, Any], mode_key: str, lambda_key: str) -> None:
        nonlocal omitted_metric_rows
        mode = payload.get(mode_key)
        residual_lambda = payload.get(lambda_key)
        if mode is None and residual_lambda is None:
            return
        parsed_lambda = _parse_finite_float(residual_lambda)
        mode_text = str(mode or "")
        mode_ok = mode_text == expected_mode
        lambda_ok = (
            True
            if expected_residual_lambda is None
            else parsed_lambda is not None and abs(parsed_lambda - float(expected_residual_lambda)) <= float(lambda_tolerance)
        )
        candidate = {
            "source": source,
            "mode": mode_text,
            "residual_lambda": parsed_lambda,
            "mode_ok": mode_ok,
            "lambda_ok": lambda_ok,
            "matched": bool(mode_ok and lambda_ok),
        }
        if source.startswith("metric_row_") and len(observed) >= 20:
            omitted_metric_rows += 1
            return
        observed.append(candidate)

    if isinstance(train_report, dict):
        add_candidate("train_report", train_report, "transition_scoring_mode", "transition_residual_lambda")
        for key in ("transition_input_semantics", "transition_skill_ce", "transition_candidate_training"):
            section = train_report.get(key)
            if not isinstance(section, dict):
                continue
            if key == "transition_input_semantics":
                add_candidate(key, section, "transition_scoring_mode", "transition_residual_lambda")
            else:
                add_candidate(key, section, "scoring_mode", "residual_lambda")
            nested = section.get("input_semantics")
            if isinstance(nested, dict):
                add_candidate(f"{key}.input_semantics", nested, "transition_scoring_mode", "transition_residual_lambda")
        for key in ("metrics", "metric_averages", "last_batch_metrics", "last_metrics"):
            section = train_report.get(key)
            if isinstance(section, dict):
                add_candidate(key, section, "transition_scoring_mode", "transition_residual_lambda")
        nested_report = train_report.get("train_report")
        if isinstance(nested_report, dict):
            add_candidate("train_report.train_report", nested_report, "transition_scoring_mode", "transition_residual_lambda")
            nested_last_metrics = nested_report.get("last_metrics")
            if isinstance(nested_last_metrics, dict):
                add_candidate("train_report.train_report.last_metrics", nested_last_metrics, "transition_scoring_mode", "transition_residual_lambda")

    for index, row in enumerate(rows or [], start=1):
        add_candidate(f"metric_row_{index}", row, "transition_scoring_mode", "transition_residual_lambda")

    matched_sources = [item["source"] for item in observed if item.get("matched")]
    return {
        "expected_mode": expected_mode,
        "expected_residual_lambda": expected_residual_lambda,
        "lambda_tolerance": float(lambda_tolerance),
        "matched": bool(matched_sources),
        "matched_sources": matched_sources,
        "observed": observed,
        "omitted_metric_rows": omitted_metric_rows,
    }
