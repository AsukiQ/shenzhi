#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.state_probe_run import E2_HISTORY_METHODS, E2_METHODS, E2_RUN_SCHEMA
from clstr.vnext_training import file_sha256


E2_AUDIT_SCHEMA = "clstr_state_probe_e2_audit_v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("E2 audit cannot aggregate an empty metric")
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.pstdev(numbers),
    }


def _metric(report: dict[str, Any], method: str, target: str, key: str) -> float:
    return float(report["metrics"][method][target][key])


def audit_state_probe_e2(
    *,
    report_paths: Sequence[str | Path],
    expected_seeds: Sequence[int],
    output_path: str | Path,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in report_paths]
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        raise ValueError("E2 audit report paths must be unique existing files")
    reports = [_read_json(path) for path in paths]
    seeds = sorted(int(report.get("e1_seed") or -1) for report in reports)
    if seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError("E2 audit seeds differ from the expected set")
    for report in reports:
        if report.get("schema_version") != E2_RUN_SCHEMA or report.get("status") != "ok":
            raise ValueError("E2 audit received an invalid run report")
        if set(report.get("metrics") or {}) != set((*E2_METHODS, "majority")):
            raise ValueError("E2 audit report has the wrong method set")

    invariants = {
        "source_commit": {str(report.get("source_commit") or "") for report in reports},
        "contract": {_stable_digest(report.get("contract") or {}) for report in reports},
        "aligned_rows_sha256": {
            str((report.get("data") or {}).get("aligned_rows_sha256") or "")
            for report in reports
        },
        "split_assignment": {
            str((((report.get("data") or {}).get("split") or {}).get("assignment_sha256")) or "")
            for report in reports
        },
        "foundation_sha256": {
            str(((report.get("inputs") or {}).get("foundation_checkpoint_sha256")) or "")
            for report in reports
        },
        "skills_sha256": {
            str(((report.get("inputs") or {}).get("skills_sha256")) or "")
            for report in reports
        },
    }
    for name, values in invariants.items():
        if len(values) != 1 or "" in values:
            raise ValueError(f"E2 cross-seed invariant changed: {name}")

    canonical_examples: list[str] | None = None
    canonical_trajectories: list[str] | None = None
    canonical_splits: list[str] | None = None
    canonical_latest: torch.Tensor | None = None
    canonical_cumulative: torch.Tensor | None = None
    canonical_current: torch.Tensor | None = None
    for path, report in zip(paths, reports):
        artifacts = report.get("artifacts") or {}
        representation_path = Path(str(artifacts.get("representations_path") or "")).resolve()
        prediction_path = Path(str(artifacts.get("test_predictions_path") or "")).resolve()
        if not representation_path.is_file() or not prediction_path.is_file():
            raise ValueError("E2 audit is missing a declared artifact")
        if file_sha256(representation_path) != str(artifacts.get("representations_sha256") or ""):
            raise ValueError("E2 representation artifact digest changed")
        if file_sha256(prediction_path) != str(artifacts.get("test_predictions_sha256") or ""):
            raise ValueError("E2 prediction artifact digest changed")
        payload = torch.load(representation_path, map_location="cpu", weights_only=False)
        examples = [str(value) for value in payload.get("example_ids") or []]
        trajectories = [str(value) for value in payload.get("trajectory_ids") or []]
        split_names = [str(value) for value in payload.get("split_names") or []]
        latest = payload.get("latest_result_labels")
        cumulative = payload.get("cumulative_error_labels")
        representations = payload.get("representations") or {}
        current = representations.get("current_only")
        if len(examples) != 1362 or not isinstance(latest, torch.Tensor):
            raise ValueError("E2 representation artifact has the wrong row schema")
        if not isinstance(cumulative, torch.Tensor) or not isinstance(current, torch.Tensor):
            raise ValueError("E2 representation artifact lacks canonical tensors")
        if set(representations) != set(E2_METHODS):
            raise ValueError("E2 representation artifact has the wrong method set")
        if canonical_examples is None:
            canonical_examples = examples
            canonical_trajectories = trajectories
            canonical_splits = split_names
            canonical_latest = latest
            canonical_cumulative = cumulative
            canonical_current = current
        else:
            if examples != canonical_examples or trajectories != canonical_trajectories:
                raise ValueError("E2 example order changed across seeds")
            if split_names != canonical_splits:
                raise ValueError("E2 split order changed across seeds")
            if not torch.equal(latest, canonical_latest) or not torch.equal(
                cumulative, canonical_cumulative
            ):
                raise ValueError("E2 labels changed across seeds")
            if not torch.equal(current, canonical_current):
                raise ValueError("E2 current-only representations changed across seeds")
        checkpoint_reports = ((report.get("inputs") or {}).get("e1_checkpoints") or {})
        if set(checkpoint_reports) != set(E2_HISTORY_METHODS):
            raise ValueError("E2 run lacks a complete E1 checkpoint matrix")
        for method, checkpoint_report in checkpoint_reports.items():
            if str(checkpoint_report.get("method") or "") != method:
                raise ValueError("E2 checkpoint method metadata changed")
            if int(checkpoint_report.get("e1_seed") or -1) != int(report["e1_seed"]):
                raise ValueError("E2 checkpoint seed metadata changed")

    ordered_reports = sorted(reports, key=lambda item: int(item["e1_seed"]))
    aggregate: dict[str, Any] = {}
    for method in (*E2_METHODS, "majority"):
        aggregate[method] = {
            "latest_result": {
                "macro_f1": _mean_std(
                    [_metric(report, method, "latest_result", "test_macro_f1") for report in ordered_reports]
                ),
                "auroc": _mean_std(
                    [_metric(report, method, "latest_result", "test_auroc") for report in ordered_reports]
                ),
            },
            "cumulative_error_count": {
                "macro_f1": _mean_std(
                    [
                        _metric(report, method, "cumulative_error_count", "test_macro_f1")
                        for report in ordered_reports
                    ]
                )
            },
        }
    lstr_minus_transformer = {
        "latest_result_macro_f1": _mean_std(
            [
                _metric(report, "lstr", "latest_result", "test_macro_f1")
                - _metric(report, "transformer", "latest_result", "test_macro_f1")
                for report in ordered_reports
            ]
        ),
        "latest_result_auroc": _mean_std(
            [
                _metric(report, "lstr", "latest_result", "test_auroc")
                - _metric(report, "transformer", "latest_result", "test_auroc")
                for report in ordered_reports
            ]
        ),
        "cumulative_error_macro_f1": _mean_std(
            [
                _metric(report, "lstr", "cumulative_error_count", "test_macro_f1")
                - _metric(report, "transformer", "cumulative_error_count", "test_macro_f1")
                for report in ordered_reports
            ]
        ),
    }
    audit = {
        "schema_version": E2_AUDIT_SCHEMA,
        "status": "ok",
        "seeds": seeds,
        "source_commit": next(iter(invariants["source_commit"])),
        "data": {
            "aligned_rows_sha256": next(iter(invariants["aligned_rows_sha256"])),
            "split_assignment_sha256": next(iter(invariants["split_assignment"])),
            "row_count": 1362,
            "test_row_count": sum(1 for value in canonical_splits or [] if value == "test"),
            "test_trajectory_count": len(
                {
                    trajectory
                    for trajectory, split in zip(
                        canonical_trajectories or [], canonical_splits or []
                    )
                    if split == "test"
                }
            ),
        },
        "aggregate": aggregate,
        "lstr_minus_transformer": lstr_minus_transformer,
        "report_paths": [str(path) for path in paths],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected_seeds", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    expected_seeds = [int(value) for value in str(args.expected_seeds).split(",") if value.strip()]
    audit_state_probe_e2(
        report_paths=args.report,
        expected_seeds=expected_seeds,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
