from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from clstr.memory_utility_records import canonical_digest


STAGE4_SPLIT_VERSION = "stage4_trajectory_split_v1"
STAGE4_VALIDATION_ROW_VERSION = "stage4_validation_rows_v1"
STAGE4_GATE_ROW_VERSION = "stage4_gate_rows_v1"


@dataclass(frozen=True)
class Stage4DataProtocol:
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    gate_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def _stable_hash(*parts: object) -> int:
    payload = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _row_id(row: dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or "").strip()
    if not benchmark:
        raise ValueError("safe-memory Stage4 requires benchmark")
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if not trajectory_id:
        raise ValueError("safe-memory Stage4 requires trajectory_id")
    if row.get("step_index") is None:
        raise ValueError("safe-memory Stage4 requires step_index")
    return str(
        row.get("row_id")
        or f"{benchmark}/{trajectory_id}/step-{int(row['step_index'])}"
    )


def _trajectory_key(benchmark: str, trajectory_id: str) -> str:
    return f"{benchmark}::{trajectory_id}"


def _probe_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    seed: int,
    version: str,
    count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _stable_hash(version, seed, benchmark, _row_id(row)),
            _row_id(row),
        ),
    )
    return ordered[: max(0, int(count))]


def build_stage4_data_protocol(
    rows: list[dict[str, Any]],
    *,
    expected_benchmarks: Sequence[str],
    seed: int,
    validation_fraction: float,
    validation_rows_per_benchmark: int,
    minimum_validation_rows_per_benchmark: int,
    gate_rows_per_benchmark: int,
    source_data_identity: dict[str, Any],
    prompt_contract: dict[str, Any],
) -> Stage4DataProtocol:
    benchmarks = tuple(str(item) for item in expected_benchmarks)
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("expected_benchmarks must be unique and nonempty")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if set(source_data_identity) != set(benchmarks):
        raise ValueError("source_data_identity must cover every Stage4 benchmark")
    if not prompt_contract.get("prompt_mode") or not prompt_contract.get(
        "prompt_sha256"
    ):
        raise ValueError("safe-memory Stage4 requires a pinned prompt contract")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        benchmark: {} for benchmark in benchmarks
    }
    for source in rows:
        row = dict(source)
        benchmark = str(row.get("benchmark") or "").strip()
        if benchmark not in grouped:
            raise ValueError(f"unexpected Stage4 benchmark: {benchmark}")
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        row["row_id"] = _row_id(row)
        grouped[benchmark].setdefault(trajectory_id, []).append(row)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    train_trajectory_ids: list[str] = []
    validation_trajectory_ids: list[str] = []
    validation_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    train_trajectory_counts: Counter[str] = Counter()
    validation_trajectory_counts: Counter[str] = Counter()
    threshold = int(round(validation_fraction * 100.0))
    for benchmark in benchmarks:
        benchmark_train: list[dict[str, Any]] = []
        benchmark_validation: list[dict[str, Any]] = []
        for trajectory_id, trajectory_rows in sorted(grouped[benchmark].items()):
            target = (
                benchmark_validation
                if _stable_hash(
                    STAGE4_SPLIT_VERSION,
                    seed,
                    benchmark,
                    trajectory_id,
                )
                % 100
                < threshold
                else benchmark_train
            )
            target.extend(
                sorted(
                    trajectory_rows,
                    key=lambda row: (int(row["step_index"]), row["row_id"]),
                )
            )
            if target is benchmark_validation:
                validation_trajectory_ids.append(
                    _trajectory_key(benchmark, trajectory_id)
                )
                validation_trajectory_counts[benchmark] += 1
            else:
                train_trajectory_ids.append(_trajectory_key(benchmark, trajectory_id))
                train_trajectory_counts[benchmark] += 1
        selected_validation = _probe_rows(
            benchmark_validation,
            benchmark=benchmark,
            seed=seed,
            version=STAGE4_VALIDATION_ROW_VERSION,
            count=validation_rows_per_benchmark,
        )
        if len(selected_validation) < int(minimum_validation_rows_per_benchmark):
            raise ValueError(f"underfilled Stage4 validation benchmark: {benchmark}")
        selected_gate = _probe_rows(
            benchmark_train,
            benchmark=benchmark,
            seed=seed,
            version=STAGE4_GATE_ROW_VERSION,
            count=gate_rows_per_benchmark,
        )
        train_rows.extend(benchmark_train)
        validation_rows.extend(selected_validation)
        gate_rows.extend(selected_gate)
        validation_counts[benchmark] = len(selected_validation)
        gate_counts[benchmark] = len(selected_gate)
        train_counts[benchmark] = len(benchmark_train)

    manifest = {
        "schema_version": "stage4_data_protocol_v1",
        "split_version": STAGE4_SPLIT_VERSION,
        "seed": int(seed),
        "expected_benchmarks": list(benchmarks),
        "source_data_identity": dict(source_data_identity),
        "prompt_contract": dict(prompt_contract),
        "train_trajectory_ids": sorted(train_trajectory_ids),
        "validation_trajectory_ids": sorted(validation_trajectory_ids),
        "train_row_ids": sorted(_row_id(row) for row in train_rows),
        "validation_row_ids": [_row_id(row) for row in validation_rows],
        "gate_row_ids": [_row_id(row) for row in gate_rows],
        "validation_rows_by_benchmark": dict(sorted(validation_counts.items())),
        "gate_rows_by_benchmark": dict(sorted(gate_counts.items())),
        "train_rows_by_benchmark": dict(sorted(train_counts.items())),
        "train_trajectories_by_benchmark": dict(
            sorted(train_trajectory_counts.items())
        ),
        "validation_trajectories_by_benchmark": dict(
            sorted(validation_trajectory_counts.items())
        ),
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return Stage4DataProtocol(
        train_rows,
        validation_rows,
        gate_rows,
        manifest,
    )


class BenchmarkBalancedStage4Batcher:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        benchmarks: Sequence[str],
        batch_size: int,
        seed: int,
    ) -> None:
        self.benchmarks = tuple(str(item) for item in benchmarks)
        if int(batch_size) % len(self.benchmarks) != 0:
            raise ValueError("Stage4 batch_size must be divisible by benchmark count")
        self.quota = int(batch_size) // len(self.benchmarks)
        self.seed = int(seed)
        self.rows_by_benchmark = {
            benchmark: [
                dict(row)
                for row in rows
                if str(row.get("benchmark")) == benchmark
            ]
            for benchmark in self.benchmarks
        }
        missing = [
            benchmark
            for benchmark, items in self.rows_by_benchmark.items()
            if not items
        ]
        if missing:
            raise ValueError(f"Stage4 training benchmark has no rows: {missing[0]}")
        self._epoch_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def _ordered_epoch(
        self,
        benchmark: str,
        epoch: int,
    ) -> list[dict[str, Any]]:
        key = (benchmark, int(epoch))
        cached = self._epoch_cache.get(key)
        if cached is not None:
            return cached
        rows = list(self.rows_by_benchmark[benchmark])
        random.Random(
            _stable_hash("stage4_batch", self.seed, benchmark, epoch)
        ).shuffle(rows)
        self._epoch_cache[key] = rows
        return rows

    def batch_for_step(self, step: int) -> list[dict[str, Any]]:
        if int(step) <= 0:
            raise ValueError("Stage4 step must be positive")
        batch: list[dict[str, Any]] = []
        for benchmark in self.benchmarks:
            source = self.rows_by_benchmark[benchmark]
            absolute_start = (int(step) - 1) * self.quota
            for offset in range(self.quota):
                absolute_index = absolute_start + offset
                epoch, index = divmod(absolute_index, len(source))
                batch.append(self._ordered_epoch(benchmark, epoch)[index])
        return batch
