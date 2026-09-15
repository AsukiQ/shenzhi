from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from clstr.memory_utility_records import canonical_digest


SUPPORTED_BATCH_PARITY_KINDS = frozenset({"frozen_tau2", "native_tau2", "alfworld"})
NUMERIC_TOLERANCE = 1.0e-8


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _append_once(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _numeric_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=NUMERIC_TOLERANCE,
        )
    return False


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_semantic_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _semantic_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _numeric_equal(left, right)
    return left == right


def _required_paths(kind: str, root: Path) -> tuple[Path, ...]:
    if kind == "frozen_tau2":
        return (
            root / "frozen_route_predictions.jsonl",
            root / "frozen_route_eval_report.json",
        )
    if kind == "native_tau2":
        return (
            root / "tau2_ranked_rows.jsonl",
            root / "tau2_full_clstr_route_eval_report.json",
        )
    return (root / "run.jsonl", root / "metrics.json")


def _compare_frozen(
    fallback_dir: Path,
    accelerated_dir: Path,
    blockers: list[str],
) -> tuple[int, int]:
    fallback_rows = _read_jsonl(fallback_dir / "frozen_route_predictions.jsonl")
    accelerated_rows = _read_jsonl(accelerated_dir / "frozen_route_predictions.jsonl")
    if len(fallback_rows) != len(accelerated_rows):
        _append_once(blockers, "prediction_row_count_mismatch")
    for fallback, accelerated in zip(fallback_rows, accelerated_rows):
        for key, blocker in (
            ("row_id", "row_id_mismatch"),
            ("positive_skill_id", "positive_skill_id_mismatch"),
            ("positive_rank", "positive_rank_mismatch"),
            ("declared_candidate_count", "candidate_count_mismatch"),
            ("ranked_skill_ids", "ranked_skill_ids_mismatch"),
        ):
            if fallback.get(key) != accelerated.get(key):
                _append_once(blockers, blocker)
    fallback_report = _read_json(fallback_dir / "frozen_route_eval_report.json")
    accelerated_report = _read_json(accelerated_dir / "frozen_route_eval_report.json")
    if not _semantic_equal(
        fallback_report.get("routing_metrics"),
        accelerated_report.get("routing_metrics"),
    ):
        _append_once(blockers, "routing_metrics_mismatch")
    return len(fallback_rows), len(accelerated_rows)


def _native_row_identity(row: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        row.get("row_id"),
        row.get("trajectory_id"),
        row.get("step_index"),
        index,
    )


def _compare_native(
    fallback_dir: Path,
    accelerated_dir: Path,
    blockers: list[str],
) -> tuple[int, int]:
    fallback_rows = _read_jsonl(fallback_dir / "tau2_ranked_rows.jsonl")
    accelerated_rows = _read_jsonl(accelerated_dir / "tau2_ranked_rows.jsonl")
    if len(fallback_rows) != len(accelerated_rows):
        _append_once(blockers, "ranked_row_count_mismatch")
    for index, (fallback, accelerated) in enumerate(zip(fallback_rows, accelerated_rows)):
        if _native_row_identity(fallback, index) != _native_row_identity(accelerated, index):
            _append_once(blockers, "row_id_mismatch")
        if fallback.get("candidate_next_skill_ids") != accelerated.get(
            "candidate_next_skill_ids"
        ):
            _append_once(blockers, "candidate_order_mismatch")
    fallback_report = _read_json(fallback_dir / "tau2_full_clstr_route_eval_report.json")
    accelerated_report = _read_json(
        accelerated_dir / "tau2_full_clstr_route_eval_report.json"
    )
    for key in (
        "source_eval_rows",
        "retained_eval_rows",
        "stage0_prior_report",
        "stage0_prior_eval",
        "base_eval",
        "stage4_eval",
        "strict",
    ):
        if not _semantic_equal(fallback_report.get(key), accelerated_report.get(key)):
            _append_once(blockers, f"{key}_mismatch")
    return len(fallback_rows), len(accelerated_rows)


def _compare_alfworld(
    fallback_dir: Path,
    accelerated_dir: Path,
    blockers: list[str],
) -> tuple[int, int]:
    fallback_rows = _read_jsonl(fallback_dir / "run.jsonl")
    accelerated_rows = _read_jsonl(accelerated_dir / "run.jsonl")
    if len(fallback_rows) != len(accelerated_rows):
        _append_once(blockers, "episode_count_mismatch")
    for fallback, accelerated in zip(fallback_rows, accelerated_rows):
        identity = ("episode_index", "split", "method", "gamefile")
        if any(fallback.get(key) != accelerated.get(key) for key in identity):
            _append_once(blockers, "episode_identity_mismatch")
        actions = ("action_trace", "chosen_action_trace")
        if any(fallback.get(key) != accelerated.get(key) for key in actions):
            _append_once(blockers, "episode_action_trace_mismatch")
        outcomes = ("success", "points", "goal_condition_points", "steps")
        if any(
            not _semantic_equal(fallback.get(key), accelerated.get(key))
            for key in outcomes
        ):
            _append_once(blockers, "episode_outcome_mismatch")
    fallback_metrics = _read_json(fallback_dir / "metrics.json")
    accelerated_metrics = _read_json(accelerated_dir / "metrics.json")
    metric_keys = (
        "success_rate",
        "average_reward",
        "average_goal_condition_points",
        "average_episode_steps",
        "episode_count",
    )
    if any(
        not _semantic_equal(fallback_metrics.get(key), accelerated_metrics.get(key))
        for key in metric_keys
    ):
        _append_once(blockers, "aggregate_metrics_mismatch")
    return len(fallback_rows), len(accelerated_rows)


def compare_qwen_clstr_eval_batches(
    *,
    kind: str,
    fallback_dir: str | Path,
    accelerated_dir: str | Path,
) -> dict[str, Any]:
    kind = str(kind).strip()
    if kind not in SUPPORTED_BATCH_PARITY_KINDS:
        raise ValueError(f"unsupported batch parity kind: {kind}")
    fallback_dir = Path(fallback_dir).resolve()
    accelerated_dir = Path(accelerated_dir).resolve()
    blockers: list[str] = []
    fallback_paths = _required_paths(kind, fallback_dir)
    accelerated_paths = _required_paths(kind, accelerated_dir)
    if any(not path.is_file() for path in fallback_paths):
        _append_once(blockers, "missing_fallback_artifacts")
    if any(not path.is_file() for path in accelerated_paths):
        _append_once(blockers, "missing_accelerated_artifacts")

    fallback_rows = 0
    accelerated_rows = 0
    if not blockers:
        try:
            if kind == "frozen_tau2":
                fallback_rows, accelerated_rows = _compare_frozen(
                    fallback_dir,
                    accelerated_dir,
                    blockers,
                )
            elif kind == "native_tau2":
                fallback_rows, accelerated_rows = _compare_native(
                    fallback_dir,
                    accelerated_dir,
                    blockers,
                )
            else:
                fallback_rows, accelerated_rows = _compare_alfworld(
                    fallback_dir,
                    accelerated_dir,
                    blockers,
                )
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            _append_once(blockers, "invalid_parity_artifacts")

    payload: dict[str, Any] = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "kind": kind,
        "fallback_dir": str(fallback_dir),
        "accelerated_dir": str(accelerated_dir),
        "semantic_rows": {
            "fallback": int(fallback_rows),
            "accelerated": int(accelerated_rows),
        },
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "effective_profile": "accelerated" if not blockers else "fallback",
    }
    payload["report_sha256"] = canonical_digest(payload)
    return payload
