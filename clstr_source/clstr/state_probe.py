from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_PROBE_SCHEMA = "clstr_state_probe_e2_v1"
LATEST_RESULT_LABELS = {"success": 0, "error": 1}
CUMULATIVE_ERROR_LABELS = {"0": 0, "1": 1, "2+": 2}
CANONICAL_ROW_COUNT = 1362
CANONICAL_TRAJECTORY_COUNT = 562
CANONICAL_LATEST_COUNTS = {0: 1037, 1: 325}
CANONICAL_CUMULATIVE_COUNTS = {0: 925, 1: 295, 2: 142}
CANONICAL_SPLIT_SEED = 20260724
CANONICAL_SPLIT_ASSIGNMENT_SHA256 = (
    "a71816fef5305f0f069c96759ae2e2cbd97825de0c5d858786b632aa56297567"
)
CANONICAL_SPLIT_ROWS = {"train": 820, "dev": 279, "test": 263}


@dataclass(frozen=True)
class ProbeExample:
    example_id: str
    trajectory_id: str
    step_index: int
    state_text: str
    skill_id: str
    action_text: str
    result_text: str
    latest_result_label: int
    cumulative_error_label: int

    def event(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "state_text": self.state_text,
            "skill_id": self.skill_id,
            "action_text": self.action_text,
            "result_text": self.result_text,
            "result_executed": True,
        }


def _decode_error_value(result_text: str) -> Any:
    text = str(result_text or "")
    key = '"error"'
    key_position = text.find(key)
    if key_position < 0:
        raise ValueError("aligned result text lacks a structured error field")
    colon_position = text.find(":", key_position + len(key))
    if colon_position < 0:
        raise ValueError("aligned result error field lacks a value")
    suffix = text[colon_position + 1 :].lstrip()
    if not suffix:
        raise ValueError("aligned result error value is empty")
    try:
        value, _end = json.JSONDecoder().raw_decode(suffix)
    except json.JSONDecodeError as exc:
        raise ValueError("aligned result error value is not JSON-decodable") from exc
    return value


def result_has_error(result_text: str) -> bool:
    value = _decode_error_value(result_text)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "none", "null", "false", "0", "no error"}
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return bool(value)


def load_toolbench_probe_trajectories(
    path: str | Path,
    *,
    enforce_canonical: bool = True,
) -> list[list[ProbeExample]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"state-probe source does not exist: {source}")
    trajectories: list[list[ProbeExample]] = []
    current_id = ""
    current_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def finalize(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        error_count = 0
        examples: list[ProbeExample] = []
        for expected_step, row in enumerate(rows):
            step = int(row.get("step_index") if row.get("step_index") is not None else -1)
            if step != expected_step:
                raise ValueError("state-probe trajectory is not contiguous from step zero")
            if str(row.get("benchmark") or "") != "toolbench_g3":
                raise ValueError("state-probe source contains a non-ToolBench row")
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            example_id = str(row.get("task_id") or f"{trajectory_id}::{step}").strip()
            state_text = str(row.get("state_text") or "").strip()
            skill_id = str(row.get("skill_id") or "").strip()
            action_text = str(row.get("action_text") or "").strip()
            result_text = str(row.get("actual_result_text") or "").strip()
            if not all((trajectory_id, example_id, state_text, skill_id, action_text, result_text)):
                raise ValueError("state-probe aligned row lacks a required causal field")
            if not bool(row.get("actual_result_executed")):
                raise ValueError("state-probe requires a verified executed result on every row")
            aligned_skill = str(row.get("actual_result_skill_id") or "").strip()
            event_skill = str(row.get("result_event_skill_id") or "").strip()
            if aligned_skill != skill_id or event_skill != skill_id:
                raise ValueError("state-probe result is not aligned to the executed skill")
            has_error = result_has_error(result_text)
            error_count += int(has_error)
            examples.append(
                ProbeExample(
                    example_id=example_id,
                    trajectory_id=trajectory_id,
                    step_index=step,
                    state_text=state_text,
                    skill_id=skill_id,
                    action_text=action_text,
                    result_text=result_text,
                    latest_result_label=int(has_error),
                    cumulative_error_label=min(error_count, 2),
                )
            )
        trajectories.append(examples)

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"state-probe row {line_number} is not an object")
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            if not trajectory_id:
                raise ValueError(f"state-probe row {line_number} lacks trajectory_id")
            if not current_id:
                current_id = trajectory_id
            if trajectory_id != current_id:
                finalize(current_rows)
                seen.add(current_id)
                if trajectory_id in seen:
                    raise ValueError("state-probe source is not contiguous by trajectory")
                current_id = trajectory_id
                current_rows = []
            current_rows.append(row)
    finalize(current_rows)
    if not trajectories:
        raise ValueError("state-probe source is empty")

    flat = [example for trajectory in trajectories for example in trajectory]
    latest = Counter(example.latest_result_label for example in flat)
    cumulative = Counter(example.cumulative_error_label for example in flat)
    if enforce_canonical:
        if len(flat) != CANONICAL_ROW_COUNT:
            raise ValueError("canonical state-probe row count changed")
        if len(trajectories) != CANONICAL_TRAJECTORY_COUNT:
            raise ValueError("canonical state-probe trajectory count changed")
        if dict(sorted(latest.items())) != CANONICAL_LATEST_COUNTS:
            raise ValueError("canonical latest-result label counts changed")
        if dict(sorted(cumulative.items())) != CANONICAL_CUMULATIVE_COUNTS:
            raise ValueError("canonical cumulative-error label counts changed")
    return trajectories


def post_transition_events(
    trajectory: Sequence[ProbeExample],
    row_index: int,
    *,
    max_horizon: int = 16,
) -> list[dict[str, Any]]:
    if int(max_horizon) <= 0:
        raise ValueError("state-probe history horizon must be positive")
    index = int(row_index)
    if index < 0 or index >= len(trajectory):
        raise IndexError("state-probe row index is outside the trajectory")
    start = max(0, index - int(max_horizon) + 1)
    events = [example.event() for example in trajectory[start : index + 1]]
    if not events or int(events[-1]["step_index"]) != int(trajectory[index].step_index):
        raise RuntimeError("state-probe post-transition prefix is not causally aligned")
    return events


def _largest_remainder_sizes(total: int) -> dict[str, int]:
    ratios = {"train": 0.6, "dev": 0.2, "test": 0.2}
    raw = {name: total * ratio for name, ratio in ratios.items()}
    sizes = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = int(total) - sum(sizes.values())
    tie_order = {"dev": 0, "test": 1, "train": 2}
    order = sorted(
        ratios,
        key=lambda name: (-(raw[name] - sizes[name]), tie_order[name]),
    )
    for name in order[:remaining]:
        sizes[name] += 1
    return sizes


def trajectory_stratified_split(
    trajectories: Sequence[Sequence[ProbeExample]],
    *,
    seed: int = CANONICAL_SPLIT_SEED,
) -> dict[str, str]:
    if not trajectories:
        raise ValueError("state-probe split requires trajectories")
    strata: dict[int, list[str]] = defaultdict(list)
    for trajectory in trajectories:
        if not trajectory:
            raise ValueError("state-probe split contains an empty trajectory")
        trajectory_id = trajectory[0].trajectory_id
        if any(item.trajectory_id != trajectory_id for item in trajectory):
            raise ValueError("state-probe trajectory contains mixed identities")
        strata[int(trajectory[-1].cumulative_error_label)].append(trajectory_id)
    if set(strata) != {0, 1, 2}:
        raise ValueError("state-probe split requires all terminal error-count strata")

    split_names = ("train", "dev", "test")
    ratios = {"train": 0.6, "dev": 0.2, "test": 0.2}
    target_sizes = _largest_remainder_sizes(sum(len(values) for values in strata.values()))
    quotas: dict[int, dict[str, int]] = {}
    fractions: dict[tuple[int, str], float] = {}
    row_remaining: dict[int, int] = {}
    for stratum, values in sorted(strata.items()):
        quotas[stratum] = {}
        for name in split_names:
            raw = len(values) * ratios[name]
            quotas[stratum][name] = int(math.floor(raw))
            fractions[(stratum, name)] = raw - quotas[stratum][name]
        row_remaining[stratum] = len(values) - sum(quotas[stratum].values())
    column_remaining = {
        name: target_sizes[name] - sum(quotas[stratum][name] for stratum in strata)
        for name in split_names
    }
    tie_order = {"dev": 0, "test": 1, "train": 2}
    while sum(row_remaining.values()) > 0:
        eligible = [
            (stratum, name)
            for stratum in sorted(strata)
            for name in split_names
            if row_remaining[stratum] > 0 and column_remaining[name] > 0
        ]
        if not eligible:
            raise RuntimeError("state-probe stratified quota allocation failed")
        stratum, name = max(
            eligible,
            key=lambda item: (
                fractions[item],
                -tie_order[item[1]],
                -item[0],
            ),
        )
        quotas[stratum][name] += 1
        row_remaining[stratum] -= 1
        column_remaining[name] -= 1
        fractions[(stratum, name)] = -1.0
    if any(column_remaining.values()):
        raise RuntimeError("state-probe split sizes do not match global targets")

    assignment: dict[str, str] = {}
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda trajectory_id: hashlib.sha256(
                f"{int(seed)}:{trajectory_id}".encode("utf-8")
            ).hexdigest(),
        )
        offset = 0
        for name in split_names:
            count = quotas[stratum][name]
            for trajectory_id in ordered[offset : offset + count]:
                if trajectory_id in assignment:
                    raise RuntimeError("state-probe trajectory assigned more than once")
                assignment[trajectory_id] = name
            offset += count
        if offset != len(ordered):
            raise RuntimeError("state-probe stratum was not fully assigned")
    if len(assignment) != sum(len(values) for values in strata.values()):
        raise RuntimeError("state-probe split omitted trajectories")
    return assignment


def require_canonical_split(report: dict[str, Any]) -> None:
    if str(report.get("assignment_sha256") or "") != CANONICAL_SPLIT_ASSIGNMENT_SHA256:
        raise ValueError("canonical state-probe trajectory assignment changed")
    for name, expected_rows in CANONICAL_SPLIT_ROWS.items():
        block = report.get(name)
        if not isinstance(block, dict) or int(block.get("row_count") or 0) != expected_rows:
            raise ValueError("canonical state-probe split row counts changed")
        latest = block.get("latest_result_counts") or {}
        cumulative = block.get("cumulative_error_counts") or {}
        if set(int(key) for key in latest) != {0, 1}:
            raise ValueError("canonical state-probe binary split lost a class")
        if set(int(key) for key in cumulative) != {0, 1, 2}:
            raise ValueError("canonical state-probe cumulative split lost a class")


def split_report(
    trajectories: Sequence[Sequence[ProbeExample]],
    assignment: dict[str, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    mapping_rows: list[dict[str, str]] = []
    for name in ("train", "dev", "test"):
        selected = [
            example
            for trajectory in trajectories
            if assignment[trajectory[0].trajectory_id] == name
            for example in trajectory
        ]
        trajectory_ids = sorted(
            trajectory[0].trajectory_id
            for trajectory in trajectories
            if assignment[trajectory[0].trajectory_id] == name
        )
        output[name] = {
            "trajectory_count": len(trajectory_ids),
            "row_count": len(selected),
            "latest_result_counts": dict(
                sorted(Counter(item.latest_result_label for item in selected).items())
            ),
            "cumulative_error_counts": dict(
                sorted(Counter(item.cumulative_error_label for item in selected).items())
            ),
        }
        mapping_rows.extend(
            {"trajectory_id": trajectory_id, "split": name}
            for trajectory_id in trajectory_ids
        )
    mapping_json = json.dumps(
        sorted(mapping_rows, key=lambda item: item["trajectory_id"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    output["assignment_sha256"] = hashlib.sha256(mapping_json.encode("utf-8")).hexdigest()
    return output


def macro_f1(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_classes: int,
) -> float:
    y = labels.detach().cpu().to(torch.long).reshape(-1)
    pred = predictions.detach().cpu().to(torch.long).reshape(-1)
    if y.numel() != pred.numel() or y.numel() <= 0:
        raise ValueError("macro-F1 requires equally sized nonempty vectors")
    values: list[float] = []
    for class_index in range(int(num_classes)):
        true_positive = int(((y == class_index) & (pred == class_index)).sum().item())
        false_positive = int(((y != class_index) & (pred == class_index)).sum().item())
        false_negative = int(((y == class_index) & (pred != class_index)).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator <= 0 else 2.0 * true_positive / denominator)
    return float(sum(values) / len(values))


def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().to(torch.long).reshape(-1)
    values = scores.detach().cpu().to(torch.float64).reshape(-1)
    if y.numel() != values.numel() or y.numel() <= 0:
        raise ValueError("AUROC requires equally sized nonempty vectors")
    positives = int((y == 1).sum().item())
    negatives = int((y == 0).sum().item())
    if positives <= 0 or negatives <= 0:
        raise ValueError("AUROC requires both binary classes")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(sorted_values)
    start = 0
    while start < int(sorted_values.numel()):
        end = start + 1
        while end < int(sorted_values.numel()) and bool(
            sorted_values[end].eq(sorted_values[start]).item()
        ):
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[start:end] = average_rank
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[y == 1].sum().item())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.to(torch.long), minlength=int(num_classes)).float()
    if bool((counts <= 0).any().item()):
        raise ValueError("linear probe training split lacks a class")
    weights = float(labels.numel()) / (float(num_classes) * counts)
    return weights / weights.mean()


def _fit_logistic(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_classes: int,
    l2_strength: float,
    device: torch.device,
    max_iter: int,
) -> nn.Linear:
    layer = nn.Linear(int(train_features.size(1)), int(num_classes), bias=True).to(device)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    x = train_features.to(device=device, dtype=torch.float32)
    y = train_labels.to(device=device, dtype=torch.long)
    weights = _class_weights(y, num_classes).to(device)
    optimizer = torch.optim.LBFGS(
        layer.parameters(),
        lr=1.0,
        max_iter=int(max_iter),
        history_size=25,
        line_search_fn="strong_wolfe",
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = layer(x)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss = loss + 0.5 * float(l2_strength) * layer.weight.square().sum()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("linear probe objective became non-finite")
        loss.backward()
        return loss

    optimizer.step(closure)
    layer.eval()
    return layer


def _standardize(
    train_features: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    train = train_features.to(torch.float32)
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False)
    scale = torch.where(scale >= 1.0e-6, scale, torch.ones_like(scale))
    return tuple((value.to(torch.float32) - mean) / scale for value in (train, *others))


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() <= 0:
        raise ValueError("bootstrap interval is empty")
    return {
        "low": float(torch.quantile(tensor, 0.025).item()),
        "high": float(torch.quantile(tensor, 0.975).item()),
    }


def trajectory_bootstrap_metrics(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    trajectory_ids: Sequence[str],
    num_classes: int,
    draws: int = 2000,
    seed: int = 29,
) -> dict[str, Any]:
    if int(draws) <= 0:
        raise ValueError("state-probe bootstrap draws must be positive")
    if len(trajectory_ids) != int(labels.numel()):
        raise ValueError("state-probe bootstrap identities do not match labels")
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(trajectory_ids):
        clusters[str(trajectory_id)].append(index)
    names = sorted(clusters)
    if len(names) < 2:
        raise ValueError("state-probe bootstrap requires multiple trajectories")
    rng = random.Random(int(seed))
    f1_values: list[float] = []
    auroc_values: list[float] = []
    predictions = probabilities.argmax(dim=-1)
    for _draw in range(int(draws)):
        sampled = [rng.choice(names) for _ in names]
        indices = [index for name in sampled for index in clusters[name]]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        sampled_labels = labels.index_select(0, index_tensor)
        sampled_predictions = predictions.index_select(0, index_tensor)
        f1_values.append(
            macro_f1(sampled_labels, sampled_predictions, num_classes=num_classes)
        )
        if int(num_classes) == 2:
            try:
                auroc_values.append(
                    binary_auroc(
                        sampled_labels,
                        probabilities.index_select(0, index_tensor)[:, 1],
                    )
                )
            except ValueError:
                continue
    report: dict[str, Any] = {
        "draws": int(draws),
        "seed": int(seed),
        "resampling_unit": "trajectory_id",
        "trajectory_count": len(names),
        "macro_f1": _percentile_interval(f1_values),
    }
    if int(num_classes) == 2:
        if not auroc_values:
            raise RuntimeError("state-probe bootstrap produced no valid AUROC draws")
        report["auroc"] = _percentile_interval(auroc_values)
        report["valid_auroc_draws"] = len(auroc_values)
    return report


def fit_regularized_linear_probe(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    split_names: Sequence[str],
    trajectory_ids: Sequence[str],
    num_classes: int,
    device: str | torch.device,
    l2_grid: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0),
    max_iter: int = 100,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 29,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    x = features.detach().cpu().to(torch.float32)
    y = labels.detach().cpu().to(torch.long).reshape(-1)
    if x.ndim != 2 or int(x.size(0)) != int(y.numel()):
        raise ValueError("linear probe features must have shape [rows, dimensions]")
    if len(split_names) != int(y.numel()) or len(trajectory_ids) != int(y.numel()):
        raise ValueError("linear probe metadata length differs from features")
    masks = {
        name: torch.tensor([value == name for value in split_names], dtype=torch.bool)
        for name in ("train", "dev", "test")
    }
    if any(not bool(mask.any().item()) for mask in masks.values()):
        raise ValueError("linear probe split contains an empty partition")
    train_x, dev_x, test_x = _standardize(
        x[masks["train"]],
        x[masks["dev"]],
        x[masks["test"]],
    )
    train_y = y[masks["train"]]
    dev_y = y[masks["dev"]]
    test_y = y[masks["test"]]
    resolved_device = torch.device(device)
    candidates: list[dict[str, Any]] = []
    best: tuple[float, float, float] | None = None
    best_l2 = 0.0
    for l2_strength in l2_grid:
        if float(l2_strength) <= 0:
            raise ValueError("linear probe L2 grid must be positive")
        model = _fit_logistic(
            train_x,
            train_y,
            num_classes=num_classes,
            l2_strength=float(l2_strength),
            device=resolved_device,
            max_iter=max_iter,
        )
        with torch.no_grad():
            dev_logits = model(dev_x.to(resolved_device)).detach().cpu()
            dev_probabilities = dev_logits.softmax(dim=-1)
        dev_f1 = macro_f1(
            dev_y,
            dev_probabilities.argmax(dim=-1),
            num_classes=num_classes,
        )
        dev_nll = float(F.cross_entropy(dev_logits, dev_y).item())
        record: dict[str, Any] = {
            "l2_strength": float(l2_strength),
            "dev_macro_f1": dev_f1,
            "dev_nll": dev_nll,
        }
        if int(num_classes) == 2:
            record["dev_auroc"] = binary_auroc(dev_y, dev_probabilities[:, 1])
        candidates.append(record)
        key = (dev_f1, -dev_nll, float(l2_strength))
        if best is None or key > best:
            best = key
            best_l2 = float(l2_strength)

    combined_mask = masks["train"] | masks["dev"]
    combined_x, final_test_x = _standardize(x[combined_mask], x[masks["test"]])
    combined_y = y[combined_mask]
    final_model = _fit_logistic(
        combined_x,
        combined_y,
        num_classes=num_classes,
        l2_strength=best_l2,
        device=resolved_device,
        max_iter=max_iter,
    )
    with torch.no_grad():
        test_logits = final_model(final_test_x.to(resolved_device)).detach().cpu()
        test_probabilities = test_logits.softmax(dim=-1)
    test_predictions = test_probabilities.argmax(dim=-1)
    test_trajectory_ids = [
        str(trajectory_id)
        for trajectory_id, is_test in zip(trajectory_ids, masks["test"].tolist())
        if is_test
    ]
    metrics: dict[str, Any] = {
        "selected_l2_strength": best_l2,
        "selection_candidates": candidates,
        "train_row_count": int(masks["train"].sum().item()),
        "dev_row_count": int(masks["dev"].sum().item()),
        "test_row_count": int(masks["test"].sum().item()),
        "test_macro_f1": macro_f1(
            test_y,
            test_predictions,
            num_classes=num_classes,
        ),
    }
    if int(num_classes) == 2:
        metrics["test_auroc"] = binary_auroc(test_y, test_probabilities[:, 1])
    metrics["trajectory_bootstrap"] = trajectory_bootstrap_metrics(
        labels=test_y,
        probabilities=test_probabilities,
        trajectory_ids=test_trajectory_ids,
        num_classes=num_classes,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    predictions = {
        "labels": test_y,
        "predictions": test_predictions,
        "probabilities": test_probabilities,
        "row_indices": masks["test"].nonzero(as_tuple=False).flatten(),
    }
    return metrics, predictions


def majority_baseline(
    *,
    labels: torch.Tensor,
    split_names: Sequence[str],
    trajectory_ids: Sequence[str],
    num_classes: int,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 29,
) -> dict[str, Any]:
    y = labels.detach().cpu().to(torch.long).reshape(-1)
    train_mask = torch.tensor([value == "train" for value in split_names], dtype=torch.bool)
    test_mask = torch.tensor([value == "test" for value in split_names], dtype=torch.bool)
    counts = torch.bincount(y[train_mask], minlength=int(num_classes))
    majority_class = int(torch.argmax(counts).item())
    test_y = y[test_mask]
    probabilities = torch.zeros((int(test_y.numel()), int(num_classes)), dtype=torch.float32)
    probabilities[:, majority_class] = 1.0
    predictions = probabilities.argmax(dim=-1)
    test_trajectory_ids = [
        str(trajectory_id)
        for trajectory_id, is_test in zip(trajectory_ids, test_mask.tolist())
        if is_test
    ]
    report: dict[str, Any] = {
        "majority_class": majority_class,
        "train_class_counts": counts.tolist(),
        "test_macro_f1": macro_f1(test_y, predictions, num_classes=num_classes),
    }
    if int(num_classes) == 2:
        report["test_auroc"] = binary_auroc(test_y, probabilities[:, 1])
    report["trajectory_bootstrap"] = trajectory_bootstrap_metrics(
        labels=test_y,
        probabilities=probabilities,
        trajectory_ids=test_trajectory_ids,
        num_classes=num_classes,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    return report


def examples_as_dicts(trajectories: Iterable[Sequence[ProbeExample]]) -> list[dict[str, Any]]:
    return [asdict(example) for trajectory in trajectories for example in trajectory]
