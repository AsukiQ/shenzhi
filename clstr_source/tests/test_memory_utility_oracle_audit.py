from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path

from scripts.audit_clstr_memory_utility_oracle import (
    finalize_memory_utility_oracle_audit_report,
    load_route_records_from_manifests,
    macro_mrr,
    run_memory_utility_oracle_audit,
    split_records_by_trajectory,
)


def test_finalized_oracle_audit_is_self_hashed_and_pins_source_rows(
    tmp_path: Path,
) -> None:
    records = _qualifying_records()
    route_manifest = tmp_path / "route-manifest.json"
    route_manifest.write_text("{}", encoding="utf-8")
    base = run_memory_utility_oracle_audit(
        records,
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=10,
        seed=17,
    )
    finalized = finalize_memory_utility_oracle_audit_report(
        base,
        records=records,
        route_manifests=[{"manifest_path": str(route_manifest)}],
        route_manifest_identity={"model_checkpoint_chain_digest": "chain-a"},
    )
    recorded = finalized.pop("manifest_sha256")
    assert recorded == canonical_digest(finalized)
    assert finalized["route_record_source"] == {
        "record_count": len(records),
        "row_digest_sha256": canonical_digest(
            [row["row_digest"] for row in records]
        ),
    }
    assert finalized["route_manifests"][0][
        "manifest_file_identity"
    ] == sha256_path(route_manifest)


def _route_record(
    *,
    benchmark: str,
    trajectory_id: str,
    row_index: int,
    dynamic_better: bool,
    causal_update_count: int = 1,
    has_positive: bool = True,
    sequential_benchmark: bool = True,
) -> dict:
    if dynamic_better:
        static_logits = [0.0, 10.0]
        dynamic_logits = [4.0, 0.0]
        feature = 1.0
    else:
        static_logits = [4.0, 0.0]
        dynamic_logits = [0.0, 10.0]
        feature = -1.0
    return {
        "schema_version": "memory_utility_route_record_v1",
        "row_digest": hashlib.sha256(
            f"{benchmark}|{trajectory_id}|{row_index}".encode("utf-8")
        ).hexdigest(),
        "trajectory_id": trajectory_id,
        "task_id": f"{trajectory_id}::{row_index}",
        "benchmark": benchmark,
        "sequential_benchmark": sequential_benchmark,
        "causal_update_count": causal_update_count,
        "features": [feature] + [0.0] * 10,
        "static_logits": static_logits,
        "dynamic_logits": dynamic_logits,
        "valid_mask": [True, True],
        "positive_mask": [has_positive, False],
    }


def _qualifying_records() -> list[dict]:
    records = []
    for benchmark_index in range(4):
        benchmark = f"bench_{benchmark_index}"
        for trajectory_index in range(12):
            trajectory_id = f"{benchmark}/traj_{trajectory_index}"
            for row_index in range(5):
                records.append(
                    _route_record(
                        benchmark=benchmark,
                        trajectory_id=trajectory_id,
                        row_index=row_index,
                        dynamic_better=row_index < 3,
                    )
                )
        records.append(
            _route_record(
                benchmark=benchmark,
                trajectory_id=f"{benchmark}/zero_history",
                row_index=0,
                dynamic_better=True,
                causal_update_count=0,
            )
        )
        records.append(
            _route_record(
                benchmark=benchmark,
                trajectory_id=f"{benchmark}/missing_positive",
                row_index=0,
                dynamic_better=False,
                has_positive=False,
            )
        )
    records.extend(
        _route_record(
            benchmark="too_small",
            trajectory_id=f"too_small/traj_{idx}",
            row_index=0,
            dynamic_better=bool(idx % 2),
        )
        for idx in range(3)
    )
    return records


def test_split_records_by_trajectory_is_disjoint_and_deterministic():
    records = _qualifying_records()

    first = split_records_by_trajectory(records, dev_fraction=0.25, seed=17)
    second = split_records_by_trajectory(records, dev_fraction=0.25, seed=17)

    train_ids = {row["trajectory_id"] for row in first["train"]}
    dev_ids = {row["trajectory_id"] for row in first["dev"]}
    assert train_ids.isdisjoint(dev_ids)
    assert first["train_trajectory_ids"] == second["train_trajectory_ids"]
    assert first["dev_trajectory_ids"] == second["dev_trajectory_ids"]


def test_macro_mrr_is_unweighted_across_benchmarks():
    rows = [
        *[{"benchmark": "large", "rank": 1} for _ in range(20)],
        {"benchmark": "small", "rank": 2},
    ]

    assert macro_mrr(rows, rank_key="rank") == pytest.approx(0.75)


def test_oracle_audit_freezes_qualification_selects_global_alpha_and_recommends_gate():
    original_torch_threads = torch.get_num_threads()
    report = run_memory_utility_oracle_audit(
        _qualifying_records(),
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=50,
        seed=17,
    )

    assert report["status"] == "ok"
    assert report["qualifying_benchmarks"] == ["bench_0", "bench_1", "bench_2", "bench_3"]
    assert report["qualification_frozen_before_split"] is True
    assert report["best_fixed_alpha"] in [round(index * 0.05, 2) for index in range(21)]
    assert report["metrics"]["rank_oracle_macro_mrr"] == pytest.approx(1.0)
    assert report["metrics"]["rank_oracle_macro_mrr"] > report["metrics"]["best_fixed_macro_mrr"]
    assert report["held_out_linear_auroc"] >= 0.99
    assert report["linear_fit"]["solver"] == "torch_regularized_newton_logistic_v1"
    assert report["linear_fit"]["torch_num_threads"] == 1
    assert report["linear_fit"]["iterations"] <= 50
    assert report["linear_fit"]["fit_status"] == "converged"
    assert torch.get_num_threads() == original_torch_threads
    assert report["learned_gate_recommended"] is True
    direct = report["direct_utility_eligibility"]
    assert direct["both_utility_signs_present"] is True
    assert direct["utility_oracle_gain_over_static"] >= 0.005
    assert direct["linear_selector_gain_capture_fraction"] >= 0.5
    assert direct["worst_source_regret_vs_static"] <= 0.005
    assert direct["eligible"] is True
    anchored = report["anchored_harm_eligibility"]
    assert anchored["alpha_base"] == report["best_fixed_alpha"]
    assert anchored["both_harm_signs_present"] is True
    assert anchored["linear_harm_selector_auroc"] >= 0.99
    assert anchored["linear_fit"]["class_balanced"] is True
    assert anchored["linear_fit"]["decision_threshold_source"] == "train_macro_mrr_no_regret"
    assert anchored["fixed_or_static_utility_oracle_gain_over_fixed"] >= 0.005
    assert anchored["linear_harm_selector_gain_over_fixed"] >= 0.002
    assert anchored["worst_source_regret_vs_fixed"] <= 0.005
    assert anchored["eligible"] is True
    assert report["recommendation_thresholds"]["minimum_utility_oracle_gain"] == 0.005
    assert report["recommendation_thresholds"]["minimum_gain_capture_fraction"] == 0.5
    assert report["recommendation_thresholds"]["maximum_source_regret"] == 0.005
    assert report["recommendation_thresholds"]["minimum_fixed_or_static_oracle_gain"] == 0.005
    assert report["recommendation_thresholds"]["minimum_harm_selector_gain_over_fixed"] == 0.002
    assert report["recommendation_thresholds"]["maximum_source_regret_vs_fixed"] == 0.005
    assert report["bootstrap"]["resampling_unit"] == "trajectory_id"
    assert report["bootstrap"]["stratified_by_benchmark"] is True
    assert report["exclusions"]["zero_history_rows"] == 4
    assert report["exclusions"]["missing_positive_rows"] == 4
    train_ids = set(report["split"]["train_trajectory_ids"])
    dev_ids = set(report["split"]["dev_trajectory_ids"])
    assert train_ids.isdisjoint(dev_ids)
    for held_out_benchmark, result in report["leave_one_benchmark_out"].items():
        assert result["dev_benchmarks"] == [held_out_benchmark]
        assert held_out_benchmark not in result["train_benchmarks"]
        assert len(result["train_benchmarks"]) == 3
        assert result["static_macro_mrr"] >= 0.0
        assert result["linear_selector_macro_mrr"] >= 0.0
        assert result["regret_vs_static"] <= 0.005


def test_direct_utility_audit_rejects_four_row_gate_snapshot():
    report = run_memory_utility_oracle_audit(
        _qualifying_records()[:4],
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=0,
        seed=17,
    )

    assert report["learned_gate_recommended"] is False
    assert report["direct_utility_eligibility"]["eligible"] is False
    assert report["anchored_harm_eligibility"]["eligible"] is False
    assert "insufficient_qualifying_benchmarks" in report["recommendation_blockers"]


def test_oracle_audit_recommendation_fails_when_utility_is_not_predictable():
    records = _qualifying_records()
    for row in records:
        row["features"] = [0.0] * 11

    report = run_memory_utility_oracle_audit(
        records,
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=20,
        seed=17,
    )

    assert report["held_out_linear_auroc"] == pytest.approx(0.5)
    assert report["learned_gate_recommended"] is False
    assert "insufficient_harm_selector_gain_over_fixed" in report["recommendation_blockers"]


def test_oracle_audit_counts_cross_benchmark_winners_only_on_sequential_benchmarks():
    records = _qualifying_records()
    for row in records:
        row["sequential_benchmark"] = row["benchmark"] == "bench_0"

    report = run_memory_utility_oracle_audit(
        records,
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=10,
        seed=17,
    )

    assert report["learned_gate_recommended"] is True
    assert "insufficient_sequential_cross_benchmark_winner_support" not in report[
        "recommendation_blockers"
    ]
    assert report["sequential_qualifying_benchmarks"] == ["bench_0"]


def test_oracle_audit_derives_count_caps_only_from_train_trajectories():
    records = _qualifying_records()
    eligible = [
        row
        for row in records
        if row["benchmark"].startswith("bench_")
        and row["causal_update_count"] > 0
        and any(row["positive_mask"])
    ]
    preliminary = split_records_by_trajectory(eligible, dev_fraction=0.25, seed=17)
    dev_ids = {row["trajectory_id"] for row in preliminary["dev"]}
    for row in records:
        if row["causal_update_count"] > 0:
            row["causal_update_count"] = 100 if row["trajectory_id"] in dev_ids else 2

    report = run_memory_utility_oracle_audit(
        records,
        minimum_benchmark_rows=50,
        minimum_benchmark_trajectories=10,
        minimum_qualifying_benchmarks=4,
        bootstrap_samples=10,
        seed=17,
    )

    assert report["count_feature_caps"] == {
        "causal_update_count": 2.0,
        "valid_candidate_count": 2.0,
        "derived_from": "train_trajectories_only",
    }


def _manifest_base(records_path: Path) -> dict:
    records_digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    row_digests = [
        json.loads(line)["row_digest"]
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "schema_version": "memory_utility_route_manifest_v1",
        "records_path": str(records_path),
        "records_sha256": records_digest,
        "record_count": 1,
        "source_rows_digest": canonical_digest(row_digests),
        "candidate_union_version": "memory_union_v1",
        "candidate_selection_version": "stable_declared_pool_v1",
        "pool_protocol": "known_global",
        "static_k": 500,
        "dynamic_extra_k": 64,
        "final_k": 64,
        "feature_schema": "memory_utility_features_v1",
        "heuristic_alpha_version": "memory_utility_heuristic_v1",
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "model_checkpoint_chain_digest": "model-a",
        "skill_mapping_digest": "mapping-a",
        "sequential_benchmarks": ["bench_0", "bench_1", "bench_2", "bench_3"],
    }


def test_manifest_loader_rejects_mixed_route_record_identity(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_qualifying_records()[0]) + "\n", encoding="utf-8")
    first = tmp_path / "first.manifest.json"
    second = tmp_path / "second.manifest.json"
    base = _manifest_base(records_path)
    first.write_text(json.dumps(base), encoding="utf-8")
    second.write_text(
        json.dumps({**base, "model_checkpoint_chain_digest": "model-b"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest identity"):
        load_route_records_from_manifests([first, second])


def test_manifest_loader_fails_closed_for_missing_identity_and_tampered_records(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_qualifying_records()[0]) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "records.manifest.json"
    base = _manifest_base(records_path)

    missing = dict(base)
    missing.pop("source_rows_digest")
    manifest_path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="source_rows_digest"):
        load_route_records_from_manifests([manifest_path])

    missing_mapping = dict(base)
    missing_mapping.pop("skill_mapping_digest")
    manifest_path.write_text(json.dumps(missing_mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="skill_mapping_digest"):
        load_route_records_from_manifests([manifest_path])

    wrong_source_digest = dict(base)
    wrong_source_digest["source_rows_digest"] = "tampered"
    manifest_path.write_text(json.dumps(wrong_source_digest), encoding="utf-8")
    with pytest.raises(ValueError, match="source_rows_digest mismatch"):
        load_route_records_from_manifests([manifest_path])

    manifest_path.write_text(json.dumps(base), encoding="utf-8")
    records_path.write_text(
        records_path.read_text(encoding="utf-8") + json.dumps(_qualifying_records()[1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="records_sha256"):
        load_route_records_from_manifests([manifest_path])


def test_manifest_loader_rejects_duplicate_records_file(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(_qualifying_records()[0]) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "records.manifest.json"
    manifest_path.write_text(json.dumps(_manifest_base(records_path)), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate route records file"):
        load_route_records_from_manifests([manifest_path, manifest_path])


def test_oracle_audit_rejects_wrong_schema_and_restores_torch_threads():
    row = _qualifying_records()[0]
    row["schema_version"] = "wrong"
    original_torch_threads = torch.get_num_threads()

    with pytest.raises(ValueError, match="route record schema"):
        run_memory_utility_oracle_audit([row], bootstrap_samples=0)

    assert torch.get_num_threads() == original_torch_threads


def test_trajectory_split_namespaces_ids_shared_across_benchmarks():
    first = _route_record(
        benchmark="bench_a",
        trajectory_id="shared",
        row_index=0,
        dynamic_better=True,
    )
    second = _route_record(
        benchmark="bench_b",
        trajectory_id="shared",
        row_index=0,
        dynamic_better=False,
    )

    split = split_records_by_trajectory([first, second], dev_fraction=0.25, seed=17)

    assert split["train_trajectory_ids"] == ["bench_a::shared", "bench_b::shared"]
    assert split["dev_trajectory_ids"] == []


def test_manifest_loader_rejects_duplicate_row_digests_across_distinct_files(tmp_path):
    record = _qualifying_records()[0]
    manifests = []
    for name in ("first", "second"):
        records_path = tmp_path / f"{name}.jsonl"
        records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        manifest_path = tmp_path / f"{name}.manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest_base(records_path)),
            encoding="utf-8",
        )
        manifests.append(manifest_path)

    with pytest.raises(ValueError, match="duplicate route record row_digest"):
        load_route_records_from_manifests(manifests)


def test_oracle_audit_requires_nonempty_row_digest():
    row = _qualifying_records()[0]
    row.pop("row_digest")

    with pytest.raises(ValueError, match="row_digest"):
        run_memory_utility_oracle_audit([row], bootstrap_samples=0)


def test_manifest_loader_allows_per_manifest_caps_and_sequential_declarations(tmp_path):
    manifests = []
    for benchmark_index in range(2):
        record = _route_record(
            benchmark=f"bench_{benchmark_index}",
            trajectory_id=f"bench_{benchmark_index}/traj",
            row_index=0,
            dynamic_better=bool(benchmark_index),
        )
        records_path = tmp_path / f"bench_{benchmark_index}.jsonl"
        records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        manifest = _manifest_base(records_path)
        manifest["feature_update_count_cap"] = float(4 + benchmark_index)
        manifest["feature_candidate_count_cap"] = float(64 + benchmark_index)
        manifest["sequential_benchmarks"] = [f"bench_{benchmark_index}"]
        manifest_path = tmp_path / f"bench_{benchmark_index}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append(manifest_path)

    loaded = load_route_records_from_manifests(manifests)

    assert len(loaded["records"]) == 2
    assert "sequential_benchmarks" not in loaded["identity"]
    assert "feature_update_count_cap" not in loaded["identity"]
    assert "feature_candidate_count_cap" not in loaded["identity"]


def test_manifest_loader_rejects_sequential_flag_inconsistent_with_manifest(tmp_path):
    record = _route_record(
        benchmark="bench_0",
        trajectory_id="bench_0/traj",
        row_index=0,
        dynamic_better=True,
        sequential_benchmark=True,
    )
    records_path = tmp_path / "bench_0.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = _manifest_base(records_path)
    manifest["sequential_benchmarks"] = []
    manifest_path = tmp_path / "bench_0.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sequential_benchmark disagrees"):
        load_route_records_from_manifests([manifest_path])
