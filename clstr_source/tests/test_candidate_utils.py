from clstr.candidate_utils import inject_positive_candidate


def test_inject_positive_candidate_replaces_last_negative_without_duplicating_positive():
    assert inject_positive_candidate([2, 4, 6], positive=8, k=3) == [2, 4, 8]
    assert inject_positive_candidate([2, 8, 6], positive=8, k=3) == [2, 8, 6]


def test_inject_positive_candidate_preserves_short_empty_and_zero_k_behavior():
    assert inject_positive_candidate([2], positive=8, k=3) == [2, 8]
    assert inject_positive_candidate([], positive=8, k=3) == [8]
    assert inject_positive_candidate([2, 4], positive=8, k=0) == [8]
    assert inject_positive_candidate([8, 2], positive=8, k=0) == []
