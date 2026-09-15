from __future__ import annotations

import random

import pytest

from clstr.stage4_data_protocol import (
    BenchmarkBalancedStage4Batcher,
    build_stage4_data_protocol,
)


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")
SOURCE_DATA_IDENTITY = {
    benchmark: {"sha256": f"{benchmark}-source-sha256"}
    for benchmark in BENCHMARKS
}
PROMPT_CONTRACT = {
    "prompt_mode": "clstr_state_query_v1",
    "prompt_sha256": "prompt-contract-sha256",
}


def _rows() -> list[dict]:
    rows: list[dict] = []
    for benchmark in BENCHMARKS:
        for trajectory_idx in range(100):
            trajectory_id = f"trajectory-{trajectory_idx:03d}"
            for step_index in range(2):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "trajectory_id": trajectory_id,
                        "step_index": step_index,
                        "row_id": (
                            f"{benchmark}/{trajectory_id}/step-{step_index}"
                        ),
                    }
                )
    return rows


def test_stage4_split_is_trajectory_disjoint_and_input_order_invariant() -> None:
    rows = _rows()
    shuffled = list(rows)
    random.Random(991).shuffle(shuffled)

    first = build_stage4_data_protocol(
        rows,
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )
    second = build_stage4_data_protocol(
        shuffled,
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )

    assert first.manifest == second.manifest
    assert [row["row_id"] for row in first.validation_rows] == [
        row["row_id"] for row in second.validation_rows
    ]
    train_ids = {
        f'{row["benchmark"]}::{row["trajectory_id"]}'
        for row in first.train_rows
    }
    validation_ids = {
        f'{row["benchmark"]}::{row["trajectory_id"]}'
        for row in first.validation_rows
    }
    assert train_ids.isdisjoint(validation_ids)
    assert "toolbench_g3::trajectory-000" in set(
        first.manifest["train_trajectory_ids"]
        + first.manifest["validation_trajectory_ids"]
    )
    assert "webshop::trajectory-000" in set(
        first.manifest["train_trajectory_ids"]
        + first.manifest["validation_trajectory_ids"]
    )
    assert first.manifest["validation_rows_by_benchmark"] == {
        benchmark: 8 for benchmark in BENCHMARKS
    }
    assert first.manifest["source_data_identity"] == SOURCE_DATA_IDENTITY
    assert first.manifest["prompt_contract"] == PROMPT_CONTRACT


def test_stage4_balanced_batch_contains_four_rows_per_benchmark() -> None:
    protocol = build_stage4_data_protocol(
        _rows(),
        expected_benchmarks=BENCHMARKS,
        seed=17,
        validation_fraction=0.10,
        validation_rows_per_benchmark=8,
        minimum_validation_rows_per_benchmark=4,
        gate_rows_per_benchmark=12,
        source_data_identity=SOURCE_DATA_IDENTITY,
        prompt_contract=PROMPT_CONTRACT,
    )
    batcher = BenchmarkBalancedStage4Batcher(
        protocol.train_rows,
        benchmarks=BENCHMARKS,
        batch_size=16,
        seed=17,
    )

    batch = batcher.batch_for_step(1)
    assert {
        benchmark: sum(
            row["benchmark"] == benchmark
            for row in batch
        )
        for benchmark in BENCHMARKS
    } == {benchmark: 4 for benchmark in BENCHMARKS}
    assert [row["row_id"] for row in batch] == [
        row["row_id"]
        for row in BenchmarkBalancedStage4Batcher(
            protocol.train_rows,
            benchmarks=BENCHMARKS,
            batch_size=16,
            seed=17,
        ).batch_for_step(1)
    ]


@pytest.mark.parametrize("missing_field", ["trajectory_id", "step_index"])
def test_stage4_protocol_rejects_missing_trajectory_fields(
    missing_field: str,
) -> None:
    rows = _rows()
    rows[0] = dict(rows[0])
    rows[0].pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        build_stage4_data_protocol(
            rows,
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=8,
            minimum_validation_rows_per_benchmark=4,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )


def test_stage4_protocol_rejects_unknown_benchmark() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "benchmark": "unknown"}
    with pytest.raises(ValueError, match="unexpected Stage4 benchmark"):
        build_stage4_data_protocol(
            rows,
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=8,
            minimum_validation_rows_per_benchmark=4,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )


def test_stage4_batcher_rejects_nondivisible_batch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BenchmarkBalancedStage4Batcher(
            _rows(),
            benchmarks=BENCHMARKS,
            batch_size=15,
            seed=17,
        )


def test_stage4_protocol_rejects_underfilled_validation() -> None:
    with pytest.raises(
        ValueError,
        match="underfilled Stage4 validation benchmark",
    ):
        build_stage4_data_protocol(
            _rows(),
            expected_benchmarks=BENCHMARKS,
            seed=17,
            validation_fraction=0.10,
            validation_rows_per_benchmark=256,
            minimum_validation_rows_per_benchmark=128,
            gate_rows_per_benchmark=12,
            source_data_identity=SOURCE_DATA_IDENTITY,
            prompt_contract=PROMPT_CONTRACT,
        )
