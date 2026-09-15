#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_test import (
    MATCHED_HISTORY_TEST_METHODS,
    MATCHED_HISTORY_TEST_RUN_SCHEMA,
)
from clstr.vnext_training import file_sha256


TEST_AUDIT_SCHEMA = "clstr_matched_history_e1_test_audit_v1"
DEV_AUDIT_SCHEMA = "clstr_matched_history_e1_audit_v1"
BENCHMARKS = ("toolbench_g3", "tau2")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_predictions(
    report: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    artifacts = report.get("artifacts") or {}
    path = Path(str(artifacts.get("test_predictions_path") or "")).resolve()
    if not path.is_file() or file_sha256(path) != str(
        artifacts.get("test_predictions_sha256") or ""
    ):
        raise ValueError("E1 test prediction artifact is missing or changed")
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("benchmark") or ""),
                str(row.get("trajectory_id") or ""),
                int(row.get("decision_index") or 0),
            )
            if key in output:
                raise ValueError("E1 test predictions contain a duplicate decision")
            output[key] = row
    if len(output) != int(report.get("test_prediction_count") or 0):
        raise ValueError("E1 test prediction count differs from its report")
    return output


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("E1 test aggregate contains a missing/nonfinite metric")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _cluster_bootstrap(
    row_values: dict[tuple[str, int], float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for (trajectory_id, _decision_index), value in row_values.items():
        by_trajectory[trajectory_id].append(float(value))
    clusters = sorted(by_trajectory)
    if len(clusters) < 2:
        raise ValueError("E1 test clustered interval requires two trajectories")
    point_values = [value for cluster in clusters for value in by_trajectory[cluster]]
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(int(draws)):
        sampled = [rng.choice(clusters) for _cluster in clusters]
        values = [value for cluster in sampled for value in by_trajectory[cluster]]
        samples.append(statistics.fmean(values))
    samples.sort()
    return {
        "mean_difference": statistics.fmean(point_values),
        "ci_low": samples[max(0, int(0.025 * len(samples)) - 1)],
        "ci_high": samples[min(len(samples) - 1, int(0.975 * len(samples)))],
        "trajectory_count": len(clusters),
        "decision_count": len(point_values),
        "bootstrap_draws": int(draws),
    }


def audit_test_reports(
    *,
    report_paths: list[Path],
    dev_audit_path: Path,
    expected_seeds: set[int],
    bootstrap_draws: int,
) -> dict[str, Any]:
    dev_audit = _read_json(dev_audit_path)
    if (
        dev_audit.get("schema_version") != DEV_AUDIT_SCHEMA
        or dev_audit.get("status") != "ok"
    ):
        raise ValueError("E1 dev audit is incomplete")
    if set(int(value) for value in dev_audit.get("seeds") or []) != expected_seeds:
        raise ValueError("E1 test seeds differ from the dev audit")
    strongest_generic = str(dev_audit.get("strongest_generic_encoder") or "")
    if strongest_generic not in {"serialized", "gru", "transformer"}:
        raise ValueError("E1 dev audit lacks a valid generic encoder selection")
    selected_train_reports = {
        str(Path(path).resolve()) for path in dev_audit.get("report_paths") or []
    }

    reports: dict[tuple[int, str], dict[str, Any]] = {}
    predictions: dict[
        tuple[int, str], dict[tuple[str, str, int], dict[str, Any]]
    ] = {}
    for path in report_paths:
        report = _read_json(path)
        if (
            report.get("schema_version") != MATCHED_HISTORY_TEST_RUN_SCHEMA
            or report.get("status") != "ok"
            or not bool(report.get("source_worktree_clean"))
        ):
            raise ValueError("E1 test report is incomplete or nonformal")
        seed = int(report.get("seed") or -1)
        method = str(report.get("encoder_kind") or "")
        key = (seed, method)
        if seed not in expected_seeds or method not in MATCHED_HISTORY_TEST_METHODS:
            raise ValueError("E1 test report has an unexpected seed/method")
        if key in reports:
            raise ValueError("E1 test report matrix has a duplicate cell")
        contract = report.get("contract") or {}
        if (
            int(contract.get("optimizer_steps", -1)) != 0
            or contract.get("checkpoint_selection")
            != "preselected_on_disjoint_e1_dev"
            or contract.get("test_access") != "single_frozen_evaluation"
        ):
            raise ValueError("E1 test report used test-time fitting or selection")
        train_report_path = str(
            Path(str((report.get("inputs") or {}).get("train_report_path") or "")).resolve()
        )
        if train_report_path not in selected_train_reports:
            raise ValueError("E1 test checkpoint report is absent from the dev audit")
        reports[key] = report
        predictions[key] = _load_predictions(report)
    expected_keys = {
        (seed, method)
        for seed in expected_seeds
        for method in MATCHED_HISTORY_TEST_METHODS
    }
    if set(reports) != expected_keys:
        raise ValueError("E1 test report matrix is incomplete")

    reference = reports[min(reports)]
    reference_contract = reference.get("contract")
    invariant_input_keys = (
        "foundation_checkpoint_sha256",
        "skills_sha256",
        "test_rows_sha256",
        "test_data_report_sha256",
        "inventory_catalogs_sha256",
    )
    reference_inputs = reference.get("inputs") or {}
    source_commits = {str(report.get("source_commit") or "") for report in reports.values()}
    if len(source_commits) != 1 or not next(iter(source_commits)):
        raise ValueError("E1 test runs do not share one source commit")
    for report in reports.values():
        if report.get("contract") != reference_contract:
            raise ValueError("E1 test runs do not share one contract")
        inputs = report.get("inputs") or {}
        if any(
            str(inputs.get(key) or "") != str(reference_inputs.get(key) or "")
            for key in invariant_input_keys
        ):
            raise ValueError("E1 test runs do not share immutable evaluation inputs")

    reference_predictions = predictions[min(predictions)]
    for current in predictions.values():
        if set(current) != set(reference_predictions):
            raise ValueError("E1 test methods do not evaluate identical decisions")
        for key, reference_row in reference_predictions.items():
            row = current[key]
            if (
                row.get("candidate_support_sha256")
                != reference_row.get("candidate_support_sha256")
                or int(row.get("candidate_support_size") or 0)
                != int(reference_row.get("candidate_support_size") or 0)
                or int(row.get("static_rank") or 0)
                != int(reference_row.get("static_rank") or 0)
            ):
                raise ValueError("E1 test methods do not share support/Static ranks")

    aggregate: dict[str, Any] = {}
    for method in MATCHED_HISTORY_TEST_METHODS:
        aggregate[method] = {
            benchmark: {
                metric: _mean_std(
                    [
                        float(reports[(seed, method)]["metrics"][benchmark][metric])
                        for seed in sorted(expected_seeds)
                    ]
                )
                for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_20")
            }
            for benchmark in BENCHMARKS
        }
        aggregate[method]["macro_mrr"] = _mean_std(
            [
                float(reports[(seed, method)]["metrics"]["macro_mrr"])
                for seed in sorted(expected_seeds)
            ]
        )

    paired_intervals: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        values_by_row: dict[tuple[str, int], list[float]] = defaultdict(list)
        for seed in sorted(expected_seeds):
            lstr_rows = predictions[(seed, "lstr")]
            generic_rows = predictions[(seed, strongest_generic)]
            for (row_benchmark, trajectory_id, decision_index), row in lstr_rows.items():
                if row_benchmark != benchmark:
                    continue
                generic = generic_rows[(row_benchmark, trajectory_id, decision_index)]
                values_by_row[(trajectory_id, decision_index)].append(
                    float(row["reciprocal_rank"])
                    - float(generic["reciprocal_rank"])
                )
        averaged = {
            key: statistics.fmean(values) for key, values in values_by_row.items()
        }
        paired_intervals[benchmark] = _cluster_bootstrap(
            averaged,
            draws=int(bootstrap_draws),
            seed=20260724,
        )

    return {
        "schema_version": TEST_AUDIT_SCHEMA,
        "status": "ok",
        "seeds": sorted(expected_seeds),
        "methods": list(MATCHED_HISTORY_TEST_METHODS),
        "source_commit": next(iter(source_commits)),
        "contract": reference_contract,
        "dev_selected_strongest_generic_encoder": strongest_generic,
        "generic_encoder_selection_source": "disjoint_e1_dev_audit",
        "test_checkpoint_selection": False,
        "aggregate": aggregate,
        "lstr_minus_dev_selected_generic_clustered_mrr": paired_intervals,
        "dev_audit_path": str(dev_audit_path.resolve()),
        "dev_audit_sha256": file_sha256(dev_audit_path),
        "report_paths": sorted(str(path.resolve()) for path in report_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--dev_audit_path", required=True)
    parser.add_argument("--expected_seeds", default="23,31,47")
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    expected_seeds = {
        int(value.strip())
        for value in str(args.expected_seeds).split(",")
        if value.strip()
    }
    if not expected_seeds:
        raise ValueError("E1 test audit requires expected seeds")
    output = audit_test_reports(
        report_paths=[Path(path).resolve() for path in args.report],
        dev_audit_path=Path(args.dev_audit_path).resolve(),
        expected_seeds=expected_seeds,
        bootstrap_draws=int(args.bootstrap_draws),
    )
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
