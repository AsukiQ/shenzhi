from clstr.traject_split import assign_traject_split


def test_assign_traject_split_is_stable_at_trajectory_level():
    first = assign_traject_split("traject::sequential::Travel::traj_query::17")
    second = assign_traject_split("traject::sequential::Travel::traj_query::17")

    assert first == second
    assert first in {"train", "dev", "test"}


def test_assign_traject_split_supports_disjoint_train_and_test_partitions():
    train_ids = {
        trajectory_id
        for idx in range(500)
        if assign_traject_split(f"traject::sequential::Travel::traj_query::{idx}") == "train"
        for trajectory_id in [f"traject::sequential::Travel::traj_query::{idx}"]
    }
    test_ids = {
        trajectory_id
        for idx in range(500)
        if assign_traject_split(f"traject::sequential::Travel::traj_query::{idx}") == "test"
        for trajectory_id in [f"traject::sequential::Travel::traj_query::{idx}"]
    }

    assert train_ids
    assert test_ids
    assert train_ids.isdisjoint(test_ids)
