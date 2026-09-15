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

from clstr.matched_history_train import MATCHED_HISTORY_TRAIN_SCHEMA
from clstr.vnext_training import file_sha256


KINDS = ("serialized", "gru", "transformer", "lstr")
GENERIC_KINDS = ("serialized", "gru", "transformer")
BENCHMARKS = ("toolbench_g3", "tau2")


def _load_predictions(report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = Path(str(report.get("dev_predictions_path") or "")).resolve()
    if not path.is_file() or file_sha256(path) != str(
        report.get("dev_predictions_sha256") or ""
    ):
        raise ValueError("E1 prediction artifact is missing or changed")
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
                raise ValueError("E1 prediction artifact has duplicate decision keys")
            output[key] = row
    if len(output) != int(report.get("dev_prediction_count") or 0):
        raise ValueError("E1 prediction count differs from its report")
    return output


def _same_inputs(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "train_rows_sha256",
        "dev_rows_sha256",
        "inventory_catalogs_sha256",
    )
    if any(str(left.get(key) or "") != str(right.get(key) or "") for key in keys):
        return False
    left_foundation = left.get("foundation") or {}
    right_foundation = right.get("foundation") or {}
    return all(
        str(left_foundation.get(key) or "")
        == str(right_foundation.get(key) or "")
        for key in (
            "checkpoint_sha256",
            "training_skills_sha256",
            "static_foundation_digest",
            "candidate_foundation_digest",
        )
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("E1 aggregate contains a missing or nonfinite metric")
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
        raise ValueError("E1 clustered interval requires at least two trajectories")
    point_values = [value for cluster in clusters for value in by_trajectory[cluster]]
    point = statistics.fmean(point_values)
    rng = random.Random(int(seed))
    samples: list[float] = []
    for _ in range(int(draws)):
        sampled = [rng.choice(clusters) for _cluster in clusters]
        values = [value for cluster in sampled for value in by_trajectory[cluster]]
        samples.append(statistics.fmean(values))
    samples.sort()
    low = samples[max(0, int(0.025 * len(samples)) - 1)]
    high = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {
        "mean_difference": point,
        "ci_low": low,
        "ci_high": high,
        "trajectory_count": len(clusters),
        "decision_count": len(point_values),
        "bootstrap_draws": int(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected_seeds", default="23,31,47")
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--allow_insufficient_clusters", action="store_true")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    expected_seeds = {
        int(value.strip())
        for value in str(args.expected_seeds).split(",")
        if value.strip()
    }
    if not expected_seeds:
        raise ValueError("E1 audit requires at least one expected seed")
    reports: dict[tuple[int, str], dict[str, Any]] = {}
    predictions: dict[tuple[int, str], dict[tuple[str, str, int], dict[str, Any]]] = {}
    for raw_path in args.report:
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != MATCHED_HISTORY_TRAIN_SCHEMA
            or report.get("status") != "ok"
        ):
            raise ValueError("E1 report is not complete")
        seed = int(report.get("seed") or -1)
        kind = str(report.get("encoder_kind") or "")
        key = (seed, kind)
        if seed not in expected_seeds or kind not in KINDS or key in reports:
            raise ValueError("E1 report has an unexpected or duplicate seed/method")
        reports[key] = report
        predictions[key] = _load_predictions(report)
    expected_keys = {(seed, kind) for seed in expected_seeds for kind in KINDS}
    if set(reports) != expected_keys:
        missing = sorted(expected_keys - set(reports))
        extra = sorted(set(reports) - expected_keys)
        raise ValueError(f"E1 report matrix is incomplete: missing={missing} extra={extra}")

    reference = reports[min(reports)]
    reference_contract = reference.get("contract")
    reference_inputs = reference.get("inputs") or {}
    reference_scorer_parameters = int(
        (reference.get("trainable_parameters") or {}).get("shared_route_scorer") or 0
    )
    source_commits = {str(report.get("source_commit") or "") for report in reports.values()}
    if len(source_commits) != 1 or not next(iter(source_commits)):
        raise ValueError("E1 reports do not share one source commit")
    for report in reports.values():
        if report.get("contract") != reference_contract:
            raise ValueError("E1 methods do not share one experiment contract")
        if not _same_inputs(report.get("inputs") or {}, reference_inputs):
            raise ValueError("E1 methods do not share immutable inputs")
        if int((report.get("trainable_parameters") or {}).get("shared_route_scorer") or 0) != reference_scorer_parameters:
            raise ValueError("E1 methods do not share the same scorer capacity")

    for seed in sorted(expected_seeds):
        lstr_parameters = int(
            reports[(seed, "lstr")]["trainable_parameters"]["history_encoder"]
        )
        for kind in KINDS:
            count = int(reports[(seed, kind)]["trainable_parameters"]["history_encoder"])
            if abs(count - lstr_parameters) / float(lstr_parameters) > 0.10:
                raise ValueError("E1 representation parameters exceed the 10 percent contract")
        reference_predictions = predictions[(seed, "lstr")]
        for kind in KINDS:
            current = predictions[(seed, kind)]
            if set(current) != set(reference_predictions):
                raise ValueError("E1 methods do not evaluate identical decision rows")
            for key, lstr_row in reference_predictions.items():
                row = current[key]
                if (
                    row.get("candidate_support_sha256")
                    != lstr_row.get("candidate_support_sha256")
                    or int(row.get("candidate_support_size") or 0)
                    != int(lstr_row.get("candidate_support_size") or 0)
                    or int(row.get("static_rank") or 0)
                    != int(lstr_row.get("static_rank") or 0)
                ):
                    raise ValueError("E1 methods do not share fixed support/static scores")

    aggregate: dict[str, Any] = {}
    for kind in KINDS:
        aggregate[kind] = {
            benchmark: {
                metric: _mean_std(
                    [
                        float(reports[(seed, kind)]["best_metrics"][benchmark][metric])
                        for seed in sorted(expected_seeds)
                    ]
                )
                for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_20")
            }
            for benchmark in BENCHMARKS
        }
        aggregate[kind]["macro_mrr"] = _mean_std(
            [float(reports[(seed, kind)]["best_metrics"]["macro_mrr"]) for seed in sorted(expected_seeds)]
        )
        aggregate[kind]["history_encoder_parameters"] = int(
            reports[(min(expected_seeds), kind)]["trainable_parameters"]["history_encoder"]
        )
    strongest_generic = max(
        GENERIC_KINDS,
        key=lambda kind: float(aggregate[kind]["macro_mrr"]["mean"]),
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
            key: statistics.fmean(values)
            for key, values in values_by_row.items()
        }
        try:
            paired_intervals[benchmark] = _cluster_bootstrap(
                averaged,
                draws=args.bootstrap_draws,
                seed=20260724,
            )
        except ValueError as error:
            if not args.allow_insufficient_clusters:
                raise
            paired_intervals[benchmark] = {
                "status": "insufficient_smoke_clusters",
                "reason": str(error),
                "decision_count": len(averaged),
            }

    output = {
        "schema_version": "clstr_matched_history_e1_audit_v1",
        "status": "ok",
        "seeds": sorted(expected_seeds),
        "methods": list(KINDS),
        "source_commit": next(iter(source_commits)),
        "contract": reference_contract,
        "shared_route_scorer_parameters": reference_scorer_parameters,
        "aggregate": aggregate,
        "strongest_generic_encoder": strongest_generic,
        "lstr_minus_strongest_generic_clustered_mrr": paired_intervals,
        "report_paths": sorted(str(Path(path).resolve()) for path in args.report),
    }
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
