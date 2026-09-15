from __future__ import annotations

import inspect

import torch

from clstr.vnext_candidates import (
    candidate_membership_mask,
    label_natural_support,
    masked_topk_tensor,
    natural_compressed_support,
    static_plus_dynamic_extra_support,
    training_only_teacher_retained_support,
)


def test_candidate_membership_is_set_based_and_respects_validity() -> None:
    candidates = torch.tensor([[9, 4, 7, 2], [8, 6, 5, 3]])
    candidate_valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    members = torch.tensor([[7, 9, 1], [3, 8, 0]])
    member_valid = torch.tensor(
        [[True, True, False], [True, False, False]]
    )
    observed = candidate_membership_mask(
        candidates,
        candidate_valid,
        members,
        member_valid,
    )
    assert observed.tolist() == [
        [True, False, True, False],
        [False, False, False, True],
    ]


def test_masked_topk_tensor_is_vectorized_and_never_marks_illegal_slots_valid() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.7, 0.4], [4.0, 3.0, 2.0, 1.0]])
    legal = torch.tensor([[True, True, False, True], [False, False, True, False]])
    ids, valid = masked_topk_tensor(logits, legal, k=3)
    assert ids[0].tolist() == [1, 3, 0]
    assert valid[0].tolist() == [True, True, True]
    assert ids[1, 0].item() == 2
    assert valid[1].tolist() == [True, False, False]


def test_natural_compressor_selects_only_from_coarse_support_without_target_input() -> None:
    output = natural_compressed_support(
        torch.tensor([[9, 4, 7, 2], [8, 6, 5, 3]]),
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
        torch.tensor([[0.1, 0.9, 0.4, 0.2], [0.3, 0.8, -9.0, -10.0]]),
        compressed_m=2,
    )
    assert output.candidate_ids.tolist() == [[4, 7], [6, 8]]
    assert output.valid_mask.tolist() == [[True, True], [True, True]]
    assert set(output.candidate_ids[0].tolist()).issubset(
        set(output.coarse_candidate_ids[0].tolist())
    )


def test_full_width_reranking_preserves_every_natural_candidate() -> None:
    coarse_ids = torch.tensor([[9, 4, 7, 2], [8, 6, 5, 3]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    output = natural_compressed_support(
        coarse_ids,
        valid,
        torch.tensor([[0.1, 0.9, 0.4, 0.2], [0.3, 0.8, -9.0, -10.0]]),
        compressed_m=4,
    )
    for row in range(2):
        before = set(coarse_ids[row][valid[row]].tolist())
        after = set(output.candidate_ids[row][output.valid_mask[row]].tolist())
        assert after == before
        assert int(output.valid_mask[row].sum()) == int(valid[row].sum())


def test_natural_compressor_rejects_empty_coarse_support() -> None:
    with torch.no_grad():
        try:
            natural_compressed_support(
                torch.tensor([[0, 1]]),
                torch.tensor([[False, False]]),
                torch.tensor([[1.0, 0.0]]),
                compressed_m=1,
            )
        except ValueError as exc:
            assert "natural coarse candidate" in str(exc)
        else:
            raise AssertionError("empty natural support must fail closed")


def test_natural_support_labels_misses_without_mutating_candidates() -> None:
    candidates = torch.tensor([[3, 1], [0, 2]])
    positives = torch.tensor(
        [[False, False, False, True], [False, True, False, False]]
    )
    labels = label_natural_support(
        candidates,
        torch.ones_like(candidates, dtype=torch.bool),
        positives,
    )
    assert candidates.tolist() == [[3, 1], [0, 2]]
    assert labels.positive_mask.tolist() == [[True, False], [False, False]]
    assert labels.eligible_mask.tolist() == [True, False]


def test_training_teacher_retains_best_positive_without_mutating_natural_support() -> None:
    natural = natural_compressed_support(
        torch.tensor([[0, 1, 2, 3]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        compressed_m=2,
    )
    original_ids = natural.candidate_ids.clone()
    positive = torch.tensor([[False, False, True, True]])
    full_scores = torch.tensor([[0.0, 0.0, 8.0, 9.0]])
    teacher = training_only_teacher_retained_support(
        natural,
        positive,
        full_scores,
        torch.tensor([True]),
    )
    assert torch.equal(natural.candidate_ids, original_ids)
    assert teacher.retained_mask.tolist() == [True]
    assert teacher.natural_positive_hit.tolist() == [False]
    assert teacher.candidate_ids.tolist() == [[0, 3]]
    labels = label_natural_support(
        teacher.candidate_ids,
        teacher.valid_mask,
        positive,
    )
    assert labels.eligible_mask.tolist() == [True]


def test_training_teacher_is_exact_noop_when_retention_is_disabled() -> None:
    natural = natural_compressed_support(
        torch.tensor([[0, 1, 2]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[3.0, 2.0, 1.0]]),
        compressed_m=2,
    )
    teacher = training_only_teacher_retained_support(
        natural,
        torch.tensor([[False, False, True]]),
        torch.tensor([[0.0, 0.0, 9.0]]),
        torch.tensor([False]),
    )
    assert torch.equal(teacher.candidate_ids, natural.candidate_ids)
    assert torch.equal(teacher.valid_mask, natural.valid_mask)
    assert teacher.retained_mask.tolist() == [False]


def test_natural_candidate_api_has_no_label_or_positive_input() -> None:
    parameters = inspect.signature(natural_compressed_support).parameters
    assert "positive" not in parameters
    assert "positive_mask" not in parameters
    assert "target" not in parameters
    membership_parameters = inspect.signature(candidate_membership_mask).parameters
    assert "positive" not in membership_parameters
    assert "target" not in membership_parameters


def test_static_topk_union_adds_only_memory_extras_and_masks_zero_history() -> None:
    static_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    static_valid = torch.ones_like(static_ids, dtype=torch.bool)
    dynamic_logits = torch.tensor(
        [[9.0, 2.0, 1.0, 8.0, 7.0], [9.0, 2.0, 1.0, 8.0, 7.0]]
    )
    legal = torch.ones_like(dynamic_logits, dtype=torch.bool)
    support, extras, extra_valid = static_plus_dynamic_extra_support(
        static_ids,
        static_valid,
        dynamic_logits,
        legal,
        torch.tensor([True, False]),
        dynamic_extra_k=2,
    )
    assert extras[0].tolist() == [3, 4]
    assert extra_valid[0].tolist() == [True, True]
    assert support.candidate_ids[0][support.valid_mask[0]].tolist() == [0, 1, 2, 3, 4]
    assert not bool(extra_valid[1].any())
    assert support.candidate_ids[1][support.valid_mask[1]].tolist() == [0, 1, 2]


def test_static_topk_union_does_not_append_static_tail_without_topk_change() -> None:
    static_ids = torch.tensor([[0, 1, 2]])
    static_valid = torch.ones_like(static_ids, dtype=torch.bool)
    dynamic_logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    legal = torch.ones_like(dynamic_logits, dtype=torch.bool)
    support, _extras, extra_valid = static_plus_dynamic_extra_support(
        static_ids,
        static_valid,
        dynamic_logits,
        legal,
        torch.tensor([True]),
        dynamic_extra_k=2,
    )
    assert not bool(extra_valid.any())
    assert support.candidate_ids[0][support.valid_mask[0]].tolist() == [0, 1, 2]
