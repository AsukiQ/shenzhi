from __future__ import annotations

import inspect

import pytest
import torch

from clstr.memory_utility_gate import (
    HEURISTIC_ALPHA_VERSION,
    MEMORY_UTILITY_FEATURE_NAMES,
    MemoryUtilityGate,
    RELIABILITY_MODES,
    RELIABILITY_FEATURE_SCHEMA_VERSION,
    effective_memory_alpha,
    fuse_route_scores,
    heuristic_memory_alpha,
    memory_utility_features,
    positive_rank_and_utility,
)


def test_reliability_modes_include_trainable_causal_gate() -> None:
    assert "causal_gate" in RELIABILITY_MODES


def test_memory_utility_gate_uses_train_normalization_and_bounded_alpha() -> None:
    gate = MemoryUtilityGate(
        feature_mean=torch.arange(11, dtype=torch.float32),
        feature_scale=torch.full((11,), 2.0),
    )
    with torch.no_grad():
        gate.net[0].weight.zero_()
        gate.net[0].bias.fill_(0.0)

    alpha = gate(torch.arange(11, dtype=torch.float32).view(1, -1))

    assert alpha.shape == (1,)
    assert alpha.item() == pytest.approx(0.5)
    assert torch.equal(gate.feature_scale, torch.full((11,), 2.0))


def test_score_fusion_has_exact_static_and_dynamic_endpoints():
    static = torch.tensor([[3.0, 2.0, float("-inf")]])
    dynamic = torch.tensor([[1.0, 4.0, float("-inf")]])
    valid = torch.tensor([[True, True, False]])
    floor = torch.finfo(static.dtype).min

    static_expected = static.masked_fill(~valid, floor)
    dynamic_expected = dynamic.masked_fill(~valid, floor)

    assert torch.equal(
        fuse_route_scores(static, dynamic, torch.tensor([0.0]), valid),
        static_expected,
    )
    assert torch.equal(
        fuse_route_scores(static, dynamic, torch.tensor([1.0]), valid),
        dynamic_expected,
    )


def test_score_fusion_interpolates_only_valid_entries_without_mutating_inputs():
    static = torch.tensor([[4.0, 2.0, float("nan")]])
    dynamic = torch.tensor([[2.0, 6.0, float("inf")]])
    valid = torch.tensor([[True, True, False]])
    static_before = static.clone()
    dynamic_before = dynamic.clone()

    fused = fuse_route_scores(static, dynamic, torch.tensor([0.25]), valid)

    assert torch.equal(fused[:, :2], torch.tensor([[3.5, 3.0]]))
    assert fused[0, 2] == torch.finfo(fused.dtype).min
    assert torch.allclose(static[:, :2], static_before[:, :2])
    assert torch.allclose(dynamic[:, :2], dynamic_before[:, :2])
    assert torch.isnan(static[0, 2]) and torch.isnan(static_before[0, 2])
    assert torch.isinf(dynamic[0, 2]) and torch.isinf(dynamic_before[0, 2])


@pytest.mark.parametrize(
    "static,dynamic,alpha,valid,error",
    [
        (
            torch.ones(1, 2),
            torch.ones(2, 2),
            torch.tensor([0.5]),
            torch.ones(1, 2, dtype=torch.bool),
            "matching rank-2",
        ),
        (
            torch.ones(1, 2),
            torch.ones(1, 2, dtype=torch.float64),
            torch.tensor([0.5]),
            torch.ones(1, 2, dtype=torch.bool),
            "dtype",
        ),
        (
            torch.ones(1, 2),
            torch.ones(1, 2),
            torch.tensor([1.1]),
            torch.ones(1, 2, dtype=torch.bool),
            r"\[0, 1\]",
        ),
        (
            torch.tensor([[1.0, float("nan")]]),
            torch.ones(1, 2),
            torch.tensor([0.5]),
            torch.ones(1, 2, dtype=torch.bool),
            "valid route scores",
        ),
    ],
)
def test_score_fusion_rejects_invalid_contracts(static, dynamic, alpha, valid, error):
    with pytest.raises(ValueError, match=error):
        fuse_route_scores(static, dynamic, alpha, valid)


def test_zero_history_forces_exact_static_alpha():
    raw = torch.tensor([0.9, 0.2])
    updates = torch.tensor([0.0, 3.0])

    assert torch.equal(
        effective_memory_alpha(raw, updates),
        torch.tensor([0.0, 0.2]),
    )


def test_effective_alpha_rejects_invalid_counts_and_probabilities():
    with pytest.raises(ValueError, match="nonnegative"):
        effective_memory_alpha(torch.tensor([0.5]), torch.tensor([-1.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        effective_memory_alpha(torch.tensor([1.5]), torch.tensor([1.0]))


def test_memory_utility_features_are_finite_detached_and_identity_free():
    floor = torch.finfo(torch.float32).min
    static = torch.tensor(
        [
            [3.0, 1.0, floor],
            [2.0, 2.0, 2.0],
            [5.0, floor, floor],
            [floor, floor, floor],
        ],
        requires_grad=True,
    )
    dynamic = torch.tensor(
        [
            [1.0, 4.0, floor],
            [2.0, 2.0, 2.0],
            [7.0, floor, floor],
            [floor, floor, floor],
        ],
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [False, False, False],
        ]
    )
    static_memory = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        requires_grad=True,
    )
    dynamic_memory = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        requires_grad=True,
    )

    features = memory_utility_features(
        static,
        dynamic,
        valid,
        static_memory,
        dynamic_memory,
        torch.tensor([2.0, 0.0, 3.0, 0.0]),
        update_count_cap=4.0,
        candidate_count_cap=8.0,
    )

    assert features.shape == (4, len(MEMORY_UTILITY_FEATURE_NAMES))
    assert features.shape[1] == 11
    assert torch.isfinite(features).all()
    assert features.requires_grad is False
    entropy_idx = MEMORY_UTILITY_FEATURE_NAMES.index("static_normalized_entropy")
    gap_idx = MEMORY_UTILITY_FEATURE_NAMES.index("static_standardized_gap")
    assert features[2, entropy_idx] == 0.0
    assert features[2, gap_idx] == 0.0
    assert torch.equal(features[3], torch.zeros_like(features[3]))
    assert RELIABILITY_FEATURE_SCHEMA_VERSION == "memory_utility_features_v1"

    parameters = set(inspect.signature(memory_utility_features).parameters)
    assert "benchmark" not in parameters
    assert "benchmark_id" not in parameters
    assert "skill_id" not in parameters
    assert "skill_ids" not in parameters


def test_memory_utility_features_validate_caps_and_shapes():
    logits = torch.ones(1, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    memory = torch.ones(1, 3)
    counts = torch.ones(1)

    with pytest.raises(ValueError, match="caps"):
        memory_utility_features(
            logits,
            logits,
            valid,
            memory,
            memory,
            counts,
            update_count_cap=0.0,
            candidate_count_cap=2.0,
        )
    with pytest.raises(ValueError, match="memory"):
        memory_utility_features(
            logits,
            logits,
            valid,
            memory,
            torch.ones(2, 3),
            counts,
            update_count_cap=2.0,
            candidate_count_cap=2.0,
        )


def test_heuristic_alpha_uses_the_versioned_deterministic_formula():
    features = torch.zeros(2, len(MEMORY_UTILITY_FEATURE_NAMES), requires_grad=True)
    features.data[0, MEMORY_UTILITY_FEATURE_NAMES.index("top1_agreement")] = 1.0
    features.data[0, MEMORY_UTILITY_FEATURE_NAMES.index("dynamic_standardized_gap")] = 2.0
    features.data[0, MEMORY_UTILITY_FEATURE_NAMES.index("static_standardized_gap")] = 1.0
    features.data[0, MEMORY_UTILITY_FEATURE_NAMES.index("js_divergence")] = 0.25

    alpha = heuristic_memory_alpha(features)

    assert alpha.requires_grad is False
    assert alpha[0] == pytest.approx(torch.sigmoid(torch.tensor(1.25)).item())
    assert alpha[1] == pytest.approx(torch.sigmoid(torch.tensor(-0.5)).item())
    assert HEURISTIC_ALPHA_VERSION == "memory_utility_heuristic_v1"


def test_positive_rank_and_utility_handles_multi_positive_and_strict_zero_rows():
    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0],
            [3.0, 4.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )
    positive = torch.tensor(
        [
            [False, True, True],
            [True, False, False],
            [False, False, True],
        ]
    )

    ranks, utilities, eligible = positive_rank_and_utility(logits, positive, valid)

    assert ranks.tolist() == [2, 2, 0]
    assert eligible.tolist() == [True, True, False]
    expected = torch.logsumexp(torch.tensor([2.0, 1.0]), dim=0) - torch.logsumexp(
        torch.tensor([4.0, 2.0, 1.0]),
        dim=0,
    )
    assert utilities[0] == pytest.approx(expected.item())
    assert utilities[2] == 0.0
