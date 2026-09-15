#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.memory_utility_gate import (
    HEURISTIC_ALPHA_VERSION,
    MEMORY_UTILITY_FEATURE_NAMES,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    positive_rank_and_utility,
)
from clstr.memory_utility_records import (
    ROUTE_MANIFEST_SCHEMA_VERSION,
    ROUTE_RECORD_SCHEMA_VERSION,
    canonical_digest,
)
from clstr.qwen_clstr_lineage import sha256_path


FIXED_ALPHA_GRID = tuple(round(index * 0.05, 2) for index in range(21))
UTILITY_TIE_TOLERANCE = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no}: expected object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_route_records_from_manifests(
    manifest_paths: Iterable[str | Path],
) -> dict[str, Any]:
    identity_keys = (
        "candidate_union_version",
        "candidate_selection_version",
        "pool_protocol",
        "static_k",
        "dynamic_extra_k",
        "final_k",
        "feature_schema",
        "heuristic_alpha_version",
        "model_checkpoint_chain_digest",
        "skill_mapping_digest",
    )
    per_manifest_required_keys = (
        "records_path",
        "records_sha256",
        "record_count",
        "source_rows_digest",
        "feature_update_count_cap",
        "feature_candidate_count_cap",
        "sequential_benchmarks",
    )
    manifests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    expected_identity: dict[str, Any] | None = None
    seen_records_paths: set[Path] = set()
    seen_row_digests: set[str] = set()
    paths = [Path(raw_path) for raw_path in manifest_paths]
    if not paths:
        raise ValueError("at least one route manifest is required")
    for path in paths:
        manifest = _read_json(path)
        if manifest.get("schema_version") != ROUTE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported route manifest schema: {path}")
        missing = [
            key
            for key in (*identity_keys, *per_manifest_required_keys)
            if manifest.get(key) is None or manifest.get(key) == ""
        ]
        if missing:
            raise ValueError(f"route manifest missing required identity field: {missing[0]}")
        if manifest["feature_schema"] != RELIABILITY_FEATURE_SCHEMA_VERSION:
            raise ValueError("route manifest feature_schema is unsupported")
        if manifest["heuristic_alpha_version"] != HEURISTIC_ALPHA_VERSION:
            raise ValueError("route manifest heuristic_alpha_version is unsupported")
        static_k = int(manifest["static_k"])
        dynamic_extra_k = int(manifest["dynamic_extra_k"])
        final_k = int(manifest["final_k"])
        if static_k <= 0 or dynamic_extra_k < 0 or final_k <= 0 or final_k > static_k:
            raise ValueError("route manifest candidate budgets are invalid")
        identity = {key: manifest.get(key) for key in identity_keys}
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError("route manifest identity mismatch")
        records_path = Path(str(manifest.get("records_path") or ""))
        if not records_path.is_absolute():
            records_path = path.parent / records_path
        records_path = records_path.resolve()
        if records_path in seen_records_paths:
            raise ValueError(f"duplicate route records file: {records_path}")
        seen_records_paths.add(records_path)
        if not records_path.is_file():
            raise ValueError(f"route records file is missing: {records_path}")
        actual_records_digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
        if actual_records_digest != str(manifest["records_sha256"]):
            raise ValueError(f"route manifest records_sha256 mismatch: {path}")
        loaded = _read_jsonl(records_path)
        if len(loaded) != int(manifest["record_count"]):
            raise ValueError(f"route manifest record_count mismatch: {path}")
        loaded_row_digests: list[str] = []
        sequential_values = manifest["sequential_benchmarks"]
        if not isinstance(sequential_values, list):
            raise ValueError("route manifest sequential_benchmarks must be a list")
        sequential_benchmarks = {str(item) for item in sequential_values}
        for row in loaded:
            if row.get("schema_version") != ROUTE_RECORD_SCHEMA_VERSION:
                raise ValueError(f"unsupported route record schema in {records_path}")
            row_digest = str(row.get("row_digest") or "")
            if not row_digest:
                raise ValueError(f"route record row_digest is missing in {records_path}")
            if row_digest in seen_row_digests:
                raise ValueError("duplicate route record row_digest across manifests")
            seen_row_digests.add(row_digest)
            loaded_row_digests.append(row_digest)
            benchmark = str(row.get("benchmark") or "unknown")
            if bool(row.get("sequential_benchmark")) != (
                benchmark in sequential_benchmarks
            ):
                raise ValueError(
                    "route record sequential_benchmark disagrees with manifest"
                )
        if manifest.get("source_rows_digest") != canonical_digest(loaded_row_digests):
            raise ValueError(f"route manifest source_rows_digest mismatch: {path}")
        manifests.append({**manifest, "manifest_path": str(path), "records_path": str(records_path)})
        records.extend(loaded)
    return {
        "records": records,
        "manifests": manifests,
        "identity": expected_identity or {},
    }


def split_records_by_trajectory(
    records: list[dict[str, Any]],
    *,
    dev_fraction: float = 0.25,
    seed: int = 17,
) -> dict[str, Any]:
    fraction = float(dev_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    trajectories_by_benchmark: dict[str, set[str]] = defaultdict(set)

    def trajectory_key(row: dict[str, Any]) -> str:
        benchmark = str(row.get("benchmark") or "unknown")
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            raise ValueError("route records require trajectory_id")
        return f"{benchmark}::{trajectory_id}"

    for row in records:
        benchmark = str(row.get("benchmark") or "unknown")
        trajectories_by_benchmark[benchmark].add(trajectory_key(row))

    dev_ids: set[str] = set()
    for benchmark, trajectory_ids in sorted(trajectories_by_benchmark.items()):
        ordered = sorted(
            trajectory_ids,
            key=lambda item: hashlib.sha256(f"{seed}|{benchmark}|{item}".encode("utf-8")).hexdigest(),
        )
        if len(ordered) <= 1:
            continue
        dev_count = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
        dev_ids.update(ordered[:dev_count])
    all_ids = {trajectory_key(row) for row in records}
    train_ids = all_ids - dev_ids
    return {
        "train": [row for row in records if trajectory_key(row) in train_ids],
        "dev": [row for row in records if trajectory_key(row) in dev_ids],
        "train_trajectory_ids": sorted(train_ids),
        "dev_trajectory_ids": sorted(dev_ids),
    }


def macro_mrr(rows: list[dict[str, Any]], *, rank_key: str) -> float:
    reciprocal_by_benchmark: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rank = int(row.get(rank_key) or 0)
        reciprocal_by_benchmark[str(row.get("benchmark") or "unknown")].append(
            1.0 / rank if rank > 0 else 0.0
        )
    if not reciprocal_by_benchmark:
        return 0.0
    benchmark_means = [sum(values) / len(values) for values in reciprocal_by_benchmark.values()]
    return float(sum(benchmark_means) / len(benchmark_means))


def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("schema_version") != ROUTE_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported route record schema")
    row_digest = str(row.get("row_digest") or "")
    if not row_digest:
        raise ValueError("route record row_digest must be nonempty")
    static_values = row.get("static_logits")
    dynamic_values = row.get("dynamic_logits")
    valid_values = row.get("valid_mask")
    positive_values = row.get("positive_mask")
    features = row.get("features")
    if not all(isinstance(value, list) for value in (static_values, dynamic_values, valid_values, positive_values)):
        raise ValueError("route records require list logits and masks")
    width = len(static_values)
    if width == 0 or any(len(value) != width for value in (dynamic_values, valid_values, positive_values)):
        raise ValueError("route record logits and masks must have one nonempty shared width")
    if not isinstance(features, list) or len(features) != len(MEMORY_UTILITY_FEATURE_NAMES):
        raise ValueError("route record features do not match the versioned schema")
    if not all(isinstance(value, bool) for value in (*valid_values, *positive_values)):
        raise ValueError("route record masks must contain booleans")
    if not isinstance(row.get("sequential_benchmark"), bool):
        raise ValueError("route record sequential_benchmark must be boolean")
    static = torch.tensor(static_values, dtype=torch.float32).view(1, -1)
    dynamic = torch.tensor(dynamic_values, dtype=torch.float32).view(1, -1)
    valid = torch.tensor(valid_values, dtype=torch.bool).view(1, -1)
    positive = torch.tensor(positive_values, dtype=torch.bool).view(1, -1)
    static_rank, static_utility, has_positive = positive_rank_and_utility(static, positive, valid)
    dynamic_rank, dynamic_utility, _ = positive_rank_and_utility(dynamic, positive, valid)
    has_negative = bool((valid & ~positive).any().item())
    causal_update_count = float(row.get("causal_update_count") or 0.0)
    if not math.isfinite(causal_update_count) or causal_update_count < 0:
        raise ValueError("route record causal_update_count must be finite and nonnegative")
    normalized = dict(row)
    normalized.update(
        {
            "benchmark": str(row.get("benchmark") or "unknown"),
            "row_digest": row_digest,
            "trajectory_id": str(row.get("trajectory_id") or ""),
            "sequential_benchmark": bool(row["sequential_benchmark"]),
            "causal_update_count": causal_update_count,
            "features": [float(value) for value in features],
            "static_logits": [float(value) for value in static_values],
            "dynamic_logits": [float(value) for value in dynamic_values],
            "valid_mask": [bool(value) for value in valid_values],
            "positive_mask": [bool(value) for value in positive_values],
            "static_rank": int(static_rank.item()),
            "dynamic_rank": int(dynamic_rank.item()),
            "static_utility": float(static_utility.item()),
            "dynamic_utility": float(dynamic_utility.item()),
            "has_positive": bool(has_positive.item()),
            "has_negative": has_negative,
            "valid_candidate_count": int(valid.sum().item()),
            "gate_eligible": bool(has_positive.item()) and has_negative and causal_update_count > 0,
        }
    )
    if not normalized["trajectory_id"]:
        raise ValueError("route record trajectory_id must be nonempty")
    if not all(math.isfinite(value) for value in normalized["features"]):
        raise ValueError("route record features must be finite")
    return normalized


def _apply_train_derived_count_features(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    update_cap = max(
        1.0,
        max((float(row["causal_update_count"]) for row in train_rows), default=0.0),
    )
    candidate_cap = max(
        1.0,
        max((float(row["valid_candidate_count"]) for row in train_rows), default=0.0),
    )
    update_index = MEMORY_UTILITY_FEATURE_NAMES.index("log_normalized_causal_update_count")
    candidate_index = MEMORY_UTILITY_FEATURE_NAMES.index(
        "log_normalized_valid_candidate_count"
    )

    def transformed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            features = list(row["features"])
            features[update_index] = math.log1p(
                min(float(row["causal_update_count"]), update_cap)
            ) / math.log1p(update_cap)
            features[candidate_index] = math.log1p(
                min(float(row["valid_candidate_count"]), candidate_cap)
            ) / math.log1p(candidate_cap)
            output.append({**row, "features": features})
        return output

    return (
        transformed(train_rows),
        transformed(dev_rows),
        {
            "causal_update_count": float(update_cap),
            "valid_candidate_count": float(candidate_cap),
            "derived_from": "train_trajectories_only",
        },
    )


def _fused_rank(row: dict[str, Any], alpha: float) -> int:
    valid = row["valid_mask"]
    positive = row["positive_mask"]
    if not any(keep and is_positive for keep, is_positive in zip(valid, positive)):
        return 0
    scores = [
        float(static) + float(alpha) * (float(dynamic) - float(static))
        for static, dynamic in zip(row["static_logits"], row["dynamic_logits"])
    ]
    best_positive = max(
        score
        for score, keep, is_positive in zip(scores, valid, positive)
        if keep and is_positive
    )
    return 1 + sum(
        1
        for score, keep in zip(scores, valid)
        if keep and score > best_positive
    )


def _fused_rank_and_utility(
    row: dict[str, Any],
    alpha: float,
) -> tuple[int, float]:
    scores = torch.tensor(
        [
            float(static) + float(alpha) * (float(dynamic) - float(static))
            for static, dynamic in zip(row["static_logits"], row["dynamic_logits"])
        ],
        dtype=torch.float32,
    ).view(1, -1)
    valid = torch.tensor(row["valid_mask"], dtype=torch.bool).view(1, -1)
    positive = torch.tensor(row["positive_mask"], dtype=torch.bool).view(1, -1)
    rank, utility, eligible = positive_rank_and_utility(
        scores,
        positive,
        valid,
    )
    if not bool(eligible.item()):
        return 0, 0.0
    return int(rank.item()), float(utility.item())


def _rows_with_fixed_endpoint(
    rows: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        fixed_rank, fixed_utility = _fused_rank_and_utility(row, alpha)
        output.append(
            {
                **row,
                "fixed_alpha": float(alpha),
                "fixed_rank": int(fixed_rank),
                "fixed_utility": float(fixed_utility),
                "dynamic_rank": int(fixed_rank),
                "dynamic_utility": float(fixed_utility),
            }
        )
    return output


def _rows_with_alpha_rank(rows: list[dict[str, Any]], alpha_by_row: list[float]) -> list[dict[str, Any]]:
    return [
        {**row, "fused_rank": _fused_rank(row, alpha)}
        for row, alpha in zip(rows, alpha_by_row)
    ]


def _heuristic_alpha(features: list[float]) -> float:
    names = MEMORY_UTILITY_FEATURE_NAMES
    value = (
        features[names.index("dynamic_standardized_gap")]
        - features[names.index("static_standardized_gap")]
        - features[names.index("js_divergence")]
        + features[names.index("top1_agreement")]
        - 0.5
    )
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _fit_linear_selector(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    *,
    seed: int,
    class_balanced: bool = False,
    calibrate_threshold_on_train: bool = False,
) -> dict[str, Any]:
    train_non_ties = [
        row
        for row in train_rows
        if abs(float(row["dynamic_utility"]) - float(row["static_utility"])) > UTILITY_TIE_TOLERANCE
    ]
    dev_non_ties = [
        row
        for row in dev_rows
        if abs(float(row["dynamic_utility"]) - float(row["static_utility"])) > UTILITY_TIE_TOLERANCE
    ]
    if not train_non_ties or not dev_non_ties:
        return {
            "auroc": 0.5,
            "constant_utility_regret": 0.0,
            "linear_utility_regret": 0.0,
            "utility_regret_improvement_fraction": 0.0,
            "feature_mean": [0.0] * len(MEMORY_UTILITY_FEATURE_NAMES),
            "feature_std": [1.0] * len(MEMORY_UTILITY_FEATURE_NAMES),
            "solver": "torch_regularized_newton_logistic_v1",
            "iterations": 0,
            "fit_status": "insufficient_rows",
            "train_winner_classes": [],
            "dev_row_digests": [],
            "dev_choose_dynamic": [],
            "seed": int(seed),
            "class_balanced": bool(class_balanced),
            "decision_threshold": 0.5,
            "decision_threshold_source": "fixed_0.5",
            "threshold_train_macro_mrr": 0.0,
            "threshold_train_worst_source_regret": 0.0,
        }
    train_x = torch.tensor([row["features"] for row in train_non_ties], dtype=torch.float32)
    train_y = torch.tensor(
        [float(row["dynamic_utility"] > row["static_utility"]) for row in train_non_ties],
        dtype=torch.float32,
    )
    dev_x = torch.tensor([row["features"] for row in dev_non_ties], dtype=torch.float32)
    dev_y = [int(row["dynamic_utility"] > row["static_utility"]) for row in dev_non_ties]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    train_z = (train_x - mean) / std
    dev_z = (dev_x - mean) / std
    train_classes = sorted({int(value) for value in train_y.tolist()})
    solver = "torch_regularized_newton_logistic_v1"
    iterations = 0
    if len(train_classes) < 2:
        fit_status = "single_class_constant"
        train_scores = [float(train_classes[0])] * len(train_non_ties)
        dev_scores = [float(train_classes[0])] * len(dev_non_ties)
    else:
        train_design = torch.cat(
            (
                train_z.to(dtype=torch.float64),
                torch.ones((len(train_non_ties), 1), dtype=torch.float64),
            ),
            dim=1,
        )
        dev_design = torch.cat(
            (
                dev_z.to(dtype=torch.float64),
                torch.ones((len(dev_non_ties), 1), dtype=torch.float64),
            ),
            dim=1,
        )
        targets = train_y.to(dtype=torch.float64)
        if class_balanced:
            positive_count = targets.sum().clamp_min(1.0)
            negative_count = (1.0 - targets).sum().clamp_min(1.0)
            sample_weight = torch.where(
                targets > 0.5,
                0.5 * len(targets) / positive_count,
                0.5 * len(targets) / negative_count,
            )
        else:
            sample_weight = torch.ones_like(targets)
        weight_sum = sample_weight.sum().clamp_min(1.0)
        parameters = torch.zeros(train_design.size(1), dtype=torch.float64)
        regularization = torch.full_like(parameters, 1.0e-4)
        regularization[-1] = 0.0
        fit_status = "max_iterations"
        for iteration in range(1, 51):
            probabilities = torch.sigmoid(train_design @ parameters)
            gradient = (
                train_design.T
                @ (sample_weight * (probabilities - targets))
                / weight_sum
                + regularization * parameters
            )
            curvature = (
                sample_weight
                * (probabilities * (1.0 - probabilities)).clamp_min(1.0e-8)
            )
            hessian = (
                train_design.T @ (train_design * curvature.unsqueeze(1))
                / weight_sum
                + torch.diag(regularization + 1.0e-8)
            )
            step = torch.linalg.solve(hessian, gradient)
            parameters = parameters - step
            iterations = iteration
            if float(torch.linalg.vector_norm(step).item()) <= 1.0e-7:
                fit_status = "converged"
                break
        train_scores = torch.sigmoid(train_design @ parameters).tolist()
        dev_scores = torch.sigmoid(dev_design @ parameters).tolist()
    auroc = _auroc(dev_y, [float(value) for value in dev_scores])
    constant_dynamic = sum(
        float(row["dynamic_utility"]) - float(row["static_utility"])
        for row in train_non_ties
    ) >= 0.0

    def regret(rows: list[dict[str, Any]], choices: list[bool]) -> float:
        if not rows:
            return 0.0
        values = []
        for row, choose_dynamic in zip(rows, choices):
            static_utility = float(row["static_utility"])
            dynamic_utility = float(row["dynamic_utility"])
            chosen = dynamic_utility if choose_dynamic else static_utility
            values.append(max(static_utility, dynamic_utility) - chosen)
        return float(sum(values) / len(values))

    constant_regret = regret(dev_non_ties, [constant_dynamic] * len(dev_non_ties))
    decision_threshold = 0.5
    decision_threshold_source = "fixed_0.5"
    threshold_train_macro_mrr = 0.0
    threshold_train_worst_source_regret = 0.0
    if calibrate_threshold_on_train:
        ordered_scores = sorted({float(score) for score in train_scores})
        thresholds = [0.0, 1.0 + 1.0e-8]
        thresholds.extend(
            0.5 * (left + right)
            for left, right in zip(ordered_scores, ordered_scores[1:])
        )

        def threshold_metrics(threshold: float) -> tuple[float, float]:
            selected = [
                {
                    **row,
                    "threshold_rank": int(
                        row["dynamic_rank"]
                        if float(score) >= threshold
                        else row["static_rank"]
                    ),
                }
                for row, score in zip(train_non_ties, train_scores)
            ]
            selected_macro = macro_mrr(selected, rank_key="threshold_rank")
            source_regrets = []
            for source in sorted({str(row["benchmark"]) for row in train_non_ties}):
                source_rows = [
                    row for row in train_non_ties if row["benchmark"] == source
                ]
                selected_rows = [
                    row for row in selected if row["benchmark"] == source
                ]
                baseline = macro_mrr(
                    [
                        {**row, "baseline_rank": int(row["dynamic_rank"])}
                        for row in source_rows
                    ],
                    rank_key="baseline_rank",
                )
                source_regrets.append(
                    baseline - macro_mrr(selected_rows, rank_key="threshold_rank")
                )
            return selected_macro, max(source_regrets, default=1.0)

        eligible_thresholds = []
        for threshold in thresholds:
            selected_macro, worst_regret = threshold_metrics(threshold)
            if worst_regret <= 0.005:
                eligible_thresholds.append(
                    (selected_macro, -float(threshold), worst_regret, threshold)
                )
        if eligible_thresholds:
            best = max(eligible_thresholds)
            threshold_train_macro_mrr = float(best[0])
            threshold_train_worst_source_regret = float(best[2])
            decision_threshold = float(best[3])
            decision_threshold_source = "train_macro_mrr_no_regret"
    dev_choose_dynamic = [score >= decision_threshold for score in dev_scores]
    linear_regret = regret(dev_non_ties, dev_choose_dynamic)
    improvement = (
        (constant_regret - linear_regret) / constant_regret
        if constant_regret > 1.0e-12
        else 0.0
    )
    return {
        "auroc": auroc,
        "constant_utility_regret": constant_regret,
        "linear_utility_regret": linear_regret,
        "utility_regret_improvement_fraction": improvement,
        "feature_mean": [float(value) for value in mean.tolist()],
        "feature_std": [float(value) for value in std.tolist()],
        "solver": solver,
        "iterations": int(iterations),
        "fit_status": fit_status,
        "train_winner_classes": train_classes,
        "dev_row_digests": [str(row["row_digest"]) for row in dev_non_ties],
        "dev_choose_dynamic": dev_choose_dynamic,
        "seed": int(seed),
        "class_balanced": bool(class_balanced),
        "decision_threshold": float(decision_threshold),
        "decision_threshold_source": decision_threshold_source,
        "threshold_train_macro_mrr": float(threshold_train_macro_mrr),
        "threshold_train_worst_source_regret": float(
            threshold_train_worst_source_regret
        ),
    }


def _selector_mrr_summary(
    rows: list[dict[str, Any]],
    selector: dict[str, Any],
) -> dict[str, Any]:
    choice_by_digest = {
        str(row_digest): bool(choose_dynamic)
        for row_digest, choose_dynamic in zip(
            selector.get("dev_row_digests") or [],
            selector.get("dev_choose_dynamic") or [],
        )
    }
    selected = [
        {
            **row,
            "selector_rank": int(
                row["dynamic_rank"]
                if choice_by_digest.get(str(row["row_digest"]), False)
                else row["static_rank"]
            ),
        }
        for row in rows
    ]
    by_source = {
        source: macro_mrr(
            [row for row in selected if row["benchmark"] == source],
            rank_key="selector_rank",
        )
        for source in sorted({str(row["benchmark"]) for row in selected})
    }
    return {
        "macro_mrr": macro_mrr(selected, rank_key="selector_rank"),
        "by_source_mrr": by_source,
    }


def _trajectory_bootstrap(
    rows: list[dict[str, Any]],
    *,
    rank_key: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        clusters_by_benchmark[str(row["benchmark"])][str(row["trajectory_id"])].append(row)
    if not clusters_by_benchmark or samples <= 0:
        return {
            "resampling_unit": "trajectory_id",
            "stratified_by_benchmark": True,
            "samples": 0,
            "ci95": [0.0, 0.0],
        }
    generator = torch.Generator().manual_seed(int(seed))
    values: list[float] = []
    for _ in range(int(samples)):
        sampled_rows: list[dict[str, Any]] = []
        for benchmark, clusters in sorted(clusters_by_benchmark.items()):
            trajectory_ids = sorted(clusters)
            sampled_indices = torch.randint(
                0,
                len(trajectory_ids),
                (len(trajectory_ids),),
                generator=generator,
            ).tolist()
            for sample_index, trajectory_index in enumerate(sampled_indices):
                trajectory_id = trajectory_ids[int(trajectory_index)]
                sampled_rows.extend(
                    {
                        **row,
                        "trajectory_id": (
                            f"bootstrap/{benchmark}/{sample_index}/{trajectory_id}"
                        ),
                    }
                    for row in clusters[trajectory_id]
                )
        values.append(macro_mrr(sampled_rows, rank_key=rank_key))
    ordered = sorted(values)
    low = ordered[max(0, int(math.floor(0.025 * (len(ordered) - 1))))]
    high = ordered[min(len(ordered) - 1, int(math.ceil(0.975 * (len(ordered) - 1))))]
    return {
        "resampling_unit": "trajectory_id",
        "stratified_by_benchmark": True,
        "samples": len(values),
        "ci95": [float(low), float(high)],
    }


def _run_memory_utility_oracle_audit(
    records: list[dict[str, Any]],
    *,
    minimum_benchmark_rows: int = 50,
    minimum_benchmark_trajectories: int = 10,
    minimum_qualifying_benchmarks: int = 4,
    bootstrap_samples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    normalized = [_normalize_record(row) for row in records]
    exclusions = {
        "zero_history_rows": sum(row["causal_update_count"] <= 0 for row in normalized),
        "missing_positive_rows": sum(not row["has_positive"] for row in normalized),
        "missing_negative_rows": sum(not row["has_negative"] for row in normalized),
    }
    eligible_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if row["gate_eligible"]:
            eligible_by_benchmark[row["benchmark"]].append(row)
    qualification = {}
    qualifying = []
    for benchmark, benchmark_rows in sorted(eligible_by_benchmark.items()):
        sequential_values = {bool(row["sequential_benchmark"]) for row in benchmark_rows}
        if len(sequential_values) != 1:
            raise ValueError(
                f"route records disagree on sequential_benchmark for {benchmark}"
            )
        trajectory_count = len({row["trajectory_id"] for row in benchmark_rows})
        qualifies = (
            len(benchmark_rows) >= int(minimum_benchmark_rows)
            and trajectory_count >= int(minimum_benchmark_trajectories)
        )
        qualification[benchmark] = {
            "eligible_rows": len(benchmark_rows),
            "trajectory_count": trajectory_count,
            "sequential_benchmark": next(iter(sequential_values)),
            "qualifies": qualifies,
        }
        if qualifies:
            qualifying.append(benchmark)
    frozen_rows = [
        row
        for benchmark in qualifying
        for row in eligible_by_benchmark[benchmark]
    ]
    split = split_records_by_trajectory(frozen_rows, dev_fraction=0.25, seed=seed)
    train_rows, dev_rows, count_feature_caps = _apply_train_derived_count_features(
        split["train"],
        split["dev"],
    )

    train_fixed = {
        alpha: macro_mrr(
            _rows_with_alpha_rank(train_rows, [alpha] * len(train_rows)),
            rank_key="fused_rank",
        )
        for alpha in FIXED_ALPHA_GRID
    }
    best_fixed_alpha = max(FIXED_ALPHA_GRID, key=lambda alpha: (train_fixed[alpha], -alpha))
    static_rows = [{**row, "rank": int(row["static_rank"])} for row in dev_rows]
    dynamic_rows = [{**row, "rank": int(row["dynamic_rank"])} for row in dev_rows]
    fixed_rows = _rows_with_alpha_rank(dev_rows, [best_fixed_alpha] * len(dev_rows))
    heuristic_rows = _rows_with_alpha_rank(
        dev_rows,
        [_heuristic_alpha(row["features"]) for row in dev_rows],
    )
    rank_oracle_rows = [
        {**row, "rank_oracle_rank": min(int(row["static_rank"]), int(row["dynamic_rank"]))}
        for row in dev_rows
    ]
    utility_oracle_rows = [
        {
            **row,
            "utility_oracle_rank": (
                int(row["dynamic_rank"])
                if float(row["dynamic_utility"]) > float(row["static_utility"])
                else int(row["static_rank"])
            ),
        }
        for row in dev_rows
    ]
    metrics = {
        "static_macro_mrr": macro_mrr(static_rows, rank_key="rank"),
        "dynamic_macro_mrr": macro_mrr(dynamic_rows, rank_key="rank"),
        "best_fixed_macro_mrr": macro_mrr(fixed_rows, rank_key="fused_rank"),
        "heuristic_macro_mrr": macro_mrr(heuristic_rows, rank_key="fused_rank"),
        "rank_oracle_macro_mrr": macro_mrr(rank_oracle_rows, rank_key="rank_oracle_rank"),
        "utility_oracle_macro_mrr": macro_mrr(
            utility_oracle_rows,
            rank_key="utility_oracle_rank",
        ),
    }
    winner_rows = [
        row
        for row in dev_rows
        if abs(float(row["dynamic_utility"]) - float(row["static_utility"]))
        > UTILITY_TIE_TOLERANCE
    ]
    dynamic_winners = [row for row in winner_rows if row["dynamic_utility"] > row["static_utility"]]
    static_winners = [row for row in winner_rows if row["static_utility"] > row["dynamic_utility"]]
    winner_denominator = max(1, len(winner_rows))
    winner_support = {
        "dynamic_better_rows": len(dynamic_winners),
        "static_better_rows": len(static_winners),
        "utility_tie_rows": len(dev_rows) - len(winner_rows),
        "dynamic_better_fraction": len(dynamic_winners) / winner_denominator,
        "static_better_fraction": len(static_winners) / winner_denominator,
        "dynamic_better_by_benchmark": {
            benchmark: sum(row["benchmark"] == benchmark for row in dynamic_winners)
            for benchmark in qualifying
        },
        "static_better_by_benchmark": {
            benchmark: sum(row["benchmark"] == benchmark for row in static_winners)
            for benchmark in qualifying
        },
    }
    linear = _fit_linear_selector(train_rows, dev_rows, seed=seed)
    linear_summary = _selector_mrr_summary(dev_rows, linear)
    best_endpoint = max(metrics["static_macro_mrr"], metrics["dynamic_macro_mrr"])
    endpoint_headroom = metrics["rank_oracle_macro_mrr"] - best_endpoint
    fixed_headroom = metrics["rank_oracle_macro_mrr"] - metrics["best_fixed_macro_mrr"]
    legacy_recommendation_blockers = []
    if len(qualifying) < int(minimum_qualifying_benchmarks):
        legacy_recommendation_blockers.append("insufficient_qualifying_benchmarks")
    if winner_support["dynamic_better_fraction"] < 0.05 or winner_support["static_better_fraction"] < 0.05:
        legacy_recommendation_blockers.append("insufficient_bidirectional_winner_fraction")
    if len(dynamic_winners) < 20 or len(static_winners) < 20:
        legacy_recommendation_blockers.append("insufficient_held_out_winner_rows")
    sequential_qualifying = [
        benchmark
        for benchmark in qualifying
        if bool(qualification[benchmark]["sequential_benchmark"])
    ]
    dynamic_supported_benchmarks = sum(
        winner_support["dynamic_better_by_benchmark"][benchmark] >= 5
        for benchmark in sequential_qualifying
    )
    static_supported_benchmarks = sum(
        winner_support["static_better_by_benchmark"][benchmark] >= 5
        for benchmark in sequential_qualifying
    )
    if dynamic_supported_benchmarks < 2 or static_supported_benchmarks < 2:
        legacy_recommendation_blockers.append(
            "insufficient_sequential_cross_benchmark_winner_support"
        )
    if endpoint_headroom < 0.01:
        legacy_recommendation_blockers.append("insufficient_rank_oracle_endpoint_headroom")
    if fixed_headroom < 0.005:
        legacy_recommendation_blockers.append("insufficient_rank_oracle_fixed_alpha_headroom")
    if not (
        linear["auroc"] >= 0.60
        or linear["utility_regret_improvement_fraction"] >= 0.10
    ):
        legacy_recommendation_blockers.append("utility_not_predictable")

    static_by_source = {
        source: macro_mrr(
            [row for row in static_rows if row["benchmark"] == source],
            rank_key="rank",
        )
        for source in sorted({str(row["benchmark"]) for row in static_rows})
    }
    oracle_gain = (
        metrics["utility_oracle_macro_mrr"] - metrics["static_macro_mrr"]
    )
    selector_gain = linear_summary["macro_mrr"] - metrics["static_macro_mrr"]
    gain_capture_fraction = (
        selector_gain / oracle_gain if oracle_gain > 1.0e-12 else 0.0
    )
    source_regret = {
        source: static_by_source[source]
        - float(linear_summary["by_source_mrr"][source])
        for source in static_by_source
    }
    worst_source_regret = max(source_regret.values(), default=1.0)
    both_utility_signs = bool(dynamic_winners) and bool(static_winners)
    direct_eligibility = {
        "both_utility_signs_present": both_utility_signs,
        "utility_oracle_gain_over_static": float(oracle_gain),
        "linear_selector_macro_mrr": float(linear_summary["macro_mrr"]),
        "linear_selector_gain_over_static": float(selector_gain),
        "linear_selector_gain_capture_fraction": float(gain_capture_fraction),
        "source_regret_vs_static": {
            source: float(value) for source, value in source_regret.items()
        },
        "worst_source_regret_vs_static": float(worst_source_regret),
        "eligible": False,
    }
    recommendation_blockers = []
    if len(qualifying) < int(minimum_qualifying_benchmarks):
        recommendation_blockers.append("insufficient_qualifying_benchmarks")
    if not both_utility_signs:
        recommendation_blockers.append("missing_bidirectional_utility_rows")
    if oracle_gain < 0.005:
        recommendation_blockers.append("insufficient_utility_oracle_gain")
    if gain_capture_fraction < 0.5:
        recommendation_blockers.append("insufficient_linear_gain_capture")
    if worst_source_regret > 0.005:
        recommendation_blockers.append("excessive_source_regret")
    direct_eligibility["eligible"] = not recommendation_blockers

    anchored_train_rows = _rows_with_fixed_endpoint(
        train_rows,
        best_fixed_alpha,
    )
    anchored_dev_rows = _rows_with_fixed_endpoint(
        dev_rows,
        best_fixed_alpha,
    )
    anchored_linear = _fit_linear_selector(
        anchored_train_rows,
        anchored_dev_rows,
        seed=seed,
        class_balanced=True,
        calibrate_threshold_on_train=True,
    )
    anchored_linear_summary = _selector_mrr_summary(
        anchored_dev_rows,
        anchored_linear,
    )
    anchored_oracle_rows = [
        {
            **row,
            "fixed_or_static_utility_oracle_rank": int(
                row["static_rank"]
                if float(row["static_utility"]) > float(row["fixed_utility"])
                else row["fixed_rank"]
            ),
        }
        for row in anchored_dev_rows
    ]
    anchored_oracle_mrr = macro_mrr(
        anchored_oracle_rows,
        rank_key="fixed_or_static_utility_oracle_rank",
    )
    fixed_macro_mrr = float(metrics["best_fixed_macro_mrr"])
    fixed_by_source = {
        source: macro_mrr(
            [
                {**row, "fixed_source_rank": int(row["fixed_rank"])}
                for row in anchored_dev_rows
                if row["benchmark"] == source
            ],
            rank_key="fixed_source_rank",
        )
        for source in sorted({str(row["benchmark"]) for row in anchored_dev_rows})
    }
    anchored_source_regret = {
        source: fixed_by_source[source]
        - float(anchored_linear_summary["by_source_mrr"][source])
        for source in fixed_by_source
    }
    anchored_worst_source_regret = max(
        anchored_source_regret.values(),
        default=1.0,
    )
    anchored_non_ties = [
        row
        for row in anchored_dev_rows
        if abs(float(row["fixed_utility"]) - float(row["static_utility"]))
        > UTILITY_TIE_TOLERANCE
    ]
    fixed_better_rows = [
        row
        for row in anchored_non_ties
        if float(row["fixed_utility"]) > float(row["static_utility"])
    ]
    harmful_memory_rows = [
        row
        for row in anchored_non_ties
        if float(row["static_utility"]) > float(row["fixed_utility"])
    ]
    anchored_oracle_gain = anchored_oracle_mrr - fixed_macro_mrr
    anchored_selector_gain = (
        float(anchored_linear_summary["macro_mrr"]) - fixed_macro_mrr
    )
    both_harm_signs = bool(fixed_better_rows) and bool(harmful_memory_rows)
    anchored_blockers = []
    if len(qualifying) < int(minimum_qualifying_benchmarks):
        anchored_blockers.append("insufficient_qualifying_benchmarks")
    if not both_harm_signs:
        anchored_blockers.append("missing_bidirectional_harm_rows")
    if anchored_oracle_gain < 0.005:
        anchored_blockers.append("insufficient_fixed_or_static_oracle_gain")
    if anchored_selector_gain < 0.002:
        anchored_blockers.append("insufficient_harm_selector_gain_over_fixed")
    if anchored_worst_source_regret > 0.005:
        anchored_blockers.append("excessive_source_regret_vs_fixed")
    anchored_harm_eligibility = {
        "alpha_base": float(best_fixed_alpha),
        "both_harm_signs_present": both_harm_signs,
        "fixed_better_rows": len(fixed_better_rows),
        "harmful_memory_rows": len(harmful_memory_rows),
        "utility_tie_rows": len(anchored_dev_rows) - len(anchored_non_ties),
        "fixed_macro_mrr": fixed_macro_mrr,
        "fixed_or_static_utility_oracle_macro_mrr": float(
            anchored_oracle_mrr
        ),
        "fixed_or_static_utility_oracle_gain_over_fixed": float(
            anchored_oracle_gain
        ),
        "linear_harm_selector_macro_mrr": float(
            anchored_linear_summary["macro_mrr"]
        ),
        "linear_harm_selector_auroc": float(anchored_linear["auroc"]),
        "linear_harm_selector_gain_over_fixed": float(
            anchored_selector_gain
        ),
        "source_regret_vs_fixed": {
            source: float(value)
            for source, value in anchored_source_regret.items()
        },
        "worst_source_regret_vs_fixed": float(anchored_worst_source_regret),
        "linear_fit": {
            "solver": anchored_linear["solver"],
            "iterations": int(anchored_linear["iterations"]),
            "fit_status": anchored_linear["fit_status"],
            "train_winner_classes": anchored_linear["train_winner_classes"],
            "seed": int(anchored_linear["seed"]),
            "class_balanced": bool(anchored_linear["class_balanced"]),
            "decision_threshold": float(
                anchored_linear["decision_threshold"]
            ),
            "decision_threshold_source": anchored_linear[
                "decision_threshold_source"
            ],
            "threshold_train_macro_mrr": float(
                anchored_linear["threshold_train_macro_mrr"]
            ),
            "threshold_train_worst_source_regret": float(
                anchored_linear["threshold_train_worst_source_regret"]
            ),
        },
        "eligible": not anchored_blockers,
    }

    leave_one_benchmark_out = {}
    for benchmark in qualifying:
        held_out_train = [row for row in train_rows if row["benchmark"] != benchmark]
        held_out_dev = [row for row in dev_rows if row["benchmark"] == benchmark]
        held_out_linear = _fit_linear_selector(
            held_out_train,
            held_out_dev,
            seed=seed,
        )
        held_out_summary = _selector_mrr_summary(held_out_dev, held_out_linear)
        held_out_static = macro_mrr(
            [
                {**row, "held_out_static_rank": int(row["static_rank"])}
                for row in held_out_dev
            ],
            rank_key="held_out_static_rank",
        )
        leave_one_benchmark_out[benchmark] = {
            "train_rows": len(held_out_train),
            "dev_rows": len(held_out_dev),
            "train_benchmarks": sorted({row["benchmark"] for row in held_out_train}),
            "dev_benchmarks": sorted({row["benchmark"] for row in held_out_dev}),
            "held_out_linear_auroc": float(held_out_linear["auroc"]),
            "constant_utility_regret": float(
                held_out_linear["constant_utility_regret"]
            ),
            "linear_utility_regret": float(held_out_linear["linear_utility_regret"]),
            "utility_regret_improvement_fraction": float(
                held_out_linear["utility_regret_improvement_fraction"]
            ),
            "static_macro_mrr": float(held_out_static),
            "linear_selector_macro_mrr": float(held_out_summary["macro_mrr"]),
            "regret_vs_static": float(
                held_out_static - held_out_summary["macro_mrr"]
            ),
            "linear_fit": {
                "solver": held_out_linear["solver"],
                "iterations": int(held_out_linear["iterations"]),
                "fit_status": held_out_linear["fit_status"],
                "train_winner_classes": held_out_linear["train_winner_classes"],
                "seed": int(held_out_linear["seed"]),
            },
        }
    return {
        "status": "ok" if len(qualifying) >= int(minimum_qualifying_benchmarks) else "action_required",
        "record_schema_version": ROUTE_RECORD_SCHEMA_VERSION,
        "feature_schema": "memory_utility_features_v1",
        "heuristic_alpha_version": HEURISTIC_ALPHA_VERSION,
        "qualification_frozen_before_split": True,
        "qualification": qualification,
        "qualifying_benchmarks": qualifying,
        "sequential_qualifying_benchmarks": sequential_qualifying,
        "split": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_trajectory_ids": split["train_trajectory_ids"],
            "dev_trajectory_ids": split["dev_trajectory_ids"],
        },
        "best_fixed_alpha": float(best_fixed_alpha),
        "fixed_alpha_grid": list(FIXED_ALPHA_GRID),
        "metrics": metrics,
        "winner_support": winner_support,
        "direct_utility_eligibility": direct_eligibility,
        "anchored_harm_eligibility": anchored_harm_eligibility,
        "held_out_linear_auroc": float(linear["auroc"]),
        "linear_fit": {
            "solver": linear["solver"],
            "iterations": int(linear["iterations"]),
            "fit_status": linear["fit_status"],
            "train_winner_classes": linear["train_winner_classes"],
            "seed": int(linear["seed"]),
        },
        "constant_utility_regret": float(linear["constant_utility_regret"]),
        "linear_utility_regret": float(linear["linear_utility_regret"]),
        "utility_regret_improvement_fraction": float(
            linear["utility_regret_improvement_fraction"]
        ),
        "normalization": {
            "feature_mean": linear["feature_mean"],
            "feature_std": linear["feature_std"],
        },
        "count_feature_caps": count_feature_caps,
        "rank_oracle_headroom_over_best_endpoint": float(endpoint_headroom),
        "rank_oracle_headroom_over_best_fixed_alpha": float(fixed_headroom),
        "leave_one_benchmark_out": leave_one_benchmark_out,
        "legacy_recommendation_blockers": legacy_recommendation_blockers,
        "bootstrap": _trajectory_bootstrap(
            rank_oracle_rows,
            rank_key="rank_oracle_rank",
            samples=bootstrap_samples,
            seed=seed,
        ),
        "exclusions": exclusions,
        "recommendation_thresholds": {
            "minimum_qualifying_benchmarks": int(minimum_qualifying_benchmarks),
            "minimum_utility_oracle_gain": 0.005,
            "minimum_gain_capture_fraction": 0.5,
            "maximum_source_regret": 0.005,
            "minimum_fixed_or_static_oracle_gain": 0.005,
            "minimum_harm_selector_gain_over_fixed": 0.002,
            "maximum_source_regret_vs_fixed": 0.005,
            "minimum_winner_fraction": 0.05,
            "minimum_winner_rows": 20,
            "minimum_winner_rows_per_benchmark": 5,
            "minimum_supported_benchmarks_per_winner": 2,
            "minimum_endpoint_headroom": 0.01,
            "minimum_fixed_alpha_headroom": 0.005,
            "minimum_linear_auroc": 0.60,
            "minimum_regret_improvement_fraction": 0.10,
        },
        "recommendation_blockers": anchored_blockers,
        "learned_gate_recommended": not anchored_blockers,
    }


def run_memory_utility_oracle_audit(
    records: list[dict[str, Any]],
    *,
    minimum_benchmark_rows: int = 50,
    minimum_benchmark_trajectories: int = 10,
    minimum_qualifying_benchmarks: int = 4,
    bootstrap_samples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    previous_threads = int(torch.get_num_threads())
    changed_threads = previous_threads != 1
    if changed_threads:
        torch.set_num_threads(1)
    try:
        report = _run_memory_utility_oracle_audit(
            records,
            minimum_benchmark_rows=minimum_benchmark_rows,
            minimum_benchmark_trajectories=minimum_benchmark_trajectories,
            minimum_qualifying_benchmarks=minimum_qualifying_benchmarks,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        active_threads = int(torch.get_num_threads())
        report["linear_fit"]["torch_num_threads"] = active_threads
        for result in report["leave_one_benchmark_out"].values():
            result["linear_fit"]["torch_num_threads"] = active_threads
        return report
    finally:
        if changed_threads:
            torch.set_num_threads(previous_threads)


def finalize_memory_utility_oracle_audit_report(
    report: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    route_manifests: list[dict[str, Any]],
    route_manifest_identity: dict[str, Any],
) -> dict[str, Any]:
    normalized_manifests = [
        {
            **manifest,
            "manifest_file_identity": sha256_path(manifest["manifest_path"]),
        }
        for manifest in route_manifests
    ]
    payload = {
        **report,
        "route_manifests": normalized_manifests,
        "route_manifest_identity": route_manifest_identity,
        "route_record_source": {
            "record_count": len(records),
            "row_digest_sha256": canonical_digest(
                [str(row["row_digest"]) for row in records]
            ),
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit causal-memory utility headroom and learnability.")
    parser.add_argument("--record_manifest", action="append", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    loaded = load_route_records_from_manifests(args.record_manifest)
    report = run_memory_utility_oracle_audit(
        loaded["records"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = finalize_memory_utility_oracle_audit_report(
        report,
        records=loaded["records"],
        route_manifests=loaded["manifests"],
        route_manifest_identity=loaded["identity"],
    )
    _write_json(Path(args.output_path), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
