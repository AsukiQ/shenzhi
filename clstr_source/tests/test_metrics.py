import pytest

from clstr.metrics import belief_mse, hit_at_k, stop_f1


def test_hit_at_k_reports_whether_relevant_skill_is_in_prefix():
    assert hit_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0


def test_stop_f1_handles_normal_and_empty_counts():
    assert stop_f1(tp=1, fp=1, fn=0) == pytest.approx(2 / 3)
    assert stop_f1(tp=0, fp=0, fn=0) == 0.0


def test_belief_mse_returns_mean_squared_error():
    assert belief_mse([1, 2], [1, 4]) == 2.0
