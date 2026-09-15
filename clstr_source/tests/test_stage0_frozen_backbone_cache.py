from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from clstr.stage0_frozen_backbone_cache import (
    STAGE0_QUERY_INDEX_KEY,
    Stage0FrozenBackboneCache,
    assign_stage0_query_indices,
    build_stage0_frozen_backbone_cache,
    plan_stage0_scheduled_rows,
)


def test_schedule_plan_reuses_training_sampler_and_preserves_first_use_order():
    rows = [{"query": f"q{index}"} for index in range(5)]
    assign_stage0_query_indices(rows)
    calls: list[tuple[int, int, str, float]] = []

    def sampler(
        _queries,
        micro_index,
        batch_size,
        *,
        sampling_strategy,
        source_buckets,
        tempered_correction_fraction,
    ):
        assert source_buckets == {"bucket": rows}
        calls.append(
            (
                micro_index,
                batch_size,
                sampling_strategy,
                tempered_correction_fraction,
            )
        )
        return {
            1: [rows[2], rows[1]],
            2: [rows[1], rows[4]],
        }[micro_index]

    planned = plan_stage0_scheduled_rows(
        queries=rows,
        start_step=1,
        max_steps=1,
        gradient_accumulation_steps=2,
        batch_size=2,
        sampling_strategy="handoff_balanced",
        source_buckets={"bucket": rows},
        tempered_correction_fraction=0.2,
        batch_sampler=sampler,
    )

    assert calls == [
        (1, 2, "handoff_balanced", 0.2),
        (2, 2, "handoff_balanced", 0.2),
    ]
    assert [row["query"] for row in planned] == ["q2", "q1", "q4"]


def test_schedule_plan_uses_resume_step_micro_indices():
    rows = [{"query": "q"}]
    assign_stage0_query_indices(rows)
    calls: list[int] = []

    def sampler(_queries, micro_index, _batch_size, **_kwargs):
        calls.append(micro_index)
        return rows

    plan_stage0_scheduled_rows(
        queries=rows,
        start_step=3,
        max_steps=4,
        gradient_accumulation_steps=2,
        batch_size=1,
        sampling_strategy="batch_stride",
        source_buckets={},
        tempered_correction_fraction=0.2,
        batch_sampler=sampler,
    )

    assert calls == [5, 6, 7, 8]


def test_cache_lookup_preserves_occurrences_and_projection_gradients():
    projection = nn.Linear(3, 2, bias=False)
    rows = [
        {STAGE0_QUERY_INDEX_KEY: 7},
        {STAGE0_QUERY_INDEX_KEY: 3},
        {STAGE0_QUERY_INDEX_KEY: 7},
    ]
    cache = Stage0FrozenBackboneCache(
        pooled_cpu=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        ),
        query_index_to_cache_row={7: 0, 3: 1},
        identity={"mode": "test"},
        build_seconds=0.1,
    )

    projected = cache.project_batch(
        rows,
        projection_fn=projection,
        device=torch.device("cpu"),
    )
    projected.sum().backward()

    torch.testing.assert_close(projected[0], projected[2])
    assert not torch.equal(projected[0], projected[1])
    assert projection.weight.grad is not None
    assert cache.lookup_count == 3
    assert cache.report()["bytes"] == 2 * 3 * 4


def test_cache_lookup_fails_closed_on_schedule_miss():
    cache = Stage0FrozenBackboneCache(
        pooled_cpu=torch.ones(1, 2),
        query_index_to_cache_row={1: 0},
        identity={},
        build_seconds=0.0,
    )

    with pytest.raises(RuntimeError, match="schedule miss"):
        cache.project_batch(
            [{STAGE0_QUERY_INDEX_KEY: 2}],
            projection_fn=lambda tensor: tensor,
            device=torch.device("cpu"),
        )


class _FakeEncoder(nn.Module):
    def __init__(self, *, trainable_backbone: bool = False):
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)
        self.backbone.weight.requires_grad_(trainable_backbone)
        self.encoded_batches: list[list[str]] = []

    def encode_backbone_pooled(self, texts: list[str]) -> torch.Tensor:
        self.encoded_batches.append(list(texts))
        values = [float(text.rsplit("q", 1)[-1]) for text in texts]
        return torch.tensor([[value, value + 1.0] for value in values])


class _FakeModel(nn.Module):
    def __init__(self, *, trainable_backbone: bool = False):
        super().__init__()
        self.encoder = _FakeEncoder(trainable_backbone=trainable_backbone)

    @staticmethod
    def _serialize_state_for_encoder(text: str) -> str:
        return f"prompt::q{text.removeprefix('q')}"


def test_cache_builder_formats_rows_batches_encoding_and_keeps_native_dtype():
    rows = [{"query": f"q{index}"} for index in range(3)]
    assign_stage0_query_indices(rows)
    model = _FakeModel()

    cache = build_stage0_frozen_backbone_cache(
        model,
        rows,
        batch_size=2,
        identity={"prompt_version": "test-v1"},
    )

    assert model.encoder.encoded_batches == [
        ["prompt::q0", "prompt::q1"],
        ["prompt::q2"],
    ]
    assert cache.pooled_cpu.dtype == torch.float32
    assert cache.pooled_cpu.device.type == "cpu"
    assert cache.query_index_to_cache_row == {0: 0, 1: 1, 2: 2}
    assert cache.report()["identity"]["prompt_version"] == "test-v1"
    assert cache.report()["identity"]["scheduled_query_count"] == 3
    assert len(cache.report()["identity"]["scheduled_query_index_digest"]) == 64
    assert cache.report()["mode"] == "schedule"


def test_cache_builder_rejects_any_trainable_backbone_parameter():
    rows = [{"query": "q0"}]
    assign_stage0_query_indices(rows)

    with pytest.raises(ValueError, match="fully frozen backbone"):
        build_stage0_frozen_backbone_cache(
            _FakeModel(trainable_backbone=True),
            rows,
            batch_size=1,
            identity={},
        )
