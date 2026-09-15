from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_clstr_matched_history_e1_test import audit_test_reports


METHODS = ("serialized", "gru", "transformer", "lstr")
SEEDS = (23, 31, 47)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_three_seed_test_audit_binds_dev_selection_and_fixed_support(tmp_path: Path):
    train_report_paths = []
    report_paths = []
    contract = {
        "optimizer_steps": 0,
        "checkpoint_selection": "preselected_on_disjoint_e1_dev",
        "test_access": "single_frozen_evaluation",
    }
    common_inputs = {
        "foundation_checkpoint_sha256": "a" * 64,
        "skills_sha256": "b" * 64,
        "test_rows_sha256": "c" * 64,
        "test_data_report_sha256": "d" * 64,
        "inventory_catalogs_sha256": "e" * 64,
    }
    for seed in SEEDS:
        for method in METHODS:
            train_report = tmp_path / f"train_{method}_{seed}.json"
            train_report_paths.append(str(train_report.resolve()))
            rank = 1 if method == "lstr" else 2 if method == "transformer" else 3
            predictions = tmp_path / f"predictions_{method}_{seed}.jsonl"
            with predictions.open("w", encoding="utf-8") as handle:
                for benchmark in ("toolbench_g3", "tau2"):
                    for trajectory_index in range(2):
                        handle.write(
                            json.dumps(
                                {
                                    "benchmark": benchmark,
                                    "trajectory_id": f"{benchmark}-{trajectory_index}",
                                    "decision_index": 1,
                                    "candidate_support_sha256": "f" * 64,
                                    "candidate_support_size": 10,
                                    "static_rank": 4,
                                    "rank": rank,
                                    "reciprocal_rank": 1.0 / rank,
                                }
                            )
                            + "\n"
                        )
            report = {
                "schema_version": "clstr_matched_history_e1_test_run_v1",
                "status": "ok",
                "source_worktree_clean": True,
                "source_commit": "1" * 40,
                "seed": seed,
                "encoder_kind": method,
                "contract": contract,
                "test_prediction_count": 4,
                "artifacts": {
                    "test_predictions_path": str(predictions),
                    "test_predictions_sha256": _sha256(predictions),
                },
                "inputs": {
                    **common_inputs,
                    "train_report_path": str(train_report),
                },
                "metrics": {
                    "toolbench_g3": {
                        "mrr": 1.0 / rank,
                        "recall_at_1": float(rank == 1),
                        "recall_at_5": 1.0,
                        "recall_at_20": 1.0,
                    },
                    "tau2": {
                        "mrr": 1.0 / rank,
                        "recall_at_1": float(rank == 1),
                        "recall_at_5": 1.0,
                        "recall_at_20": 1.0,
                    },
                    "macro_mrr": 1.0 / rank,
                },
            }
            report_path = tmp_path / f"report_{method}_{seed}.json"
            _write_json(report_path, report)
            report_paths.append(report_path)
    dev_audit = tmp_path / "dev_audit.json"
    _write_json(
        dev_audit,
        {
            "schema_version": "clstr_matched_history_e1_audit_v1",
            "status": "ok",
            "seeds": list(SEEDS),
            "strongest_generic_encoder": "transformer",
            "report_paths": train_report_paths,
        },
    )
    audit = audit_test_reports(
        report_paths=report_paths,
        dev_audit_path=dev_audit,
        expected_seeds=set(SEEDS),
        bootstrap_draws=100,
    )
    assert audit["status"] == "ok"
    assert audit["dev_selected_strongest_generic_encoder"] == "transformer"
    assert audit["test_checkpoint_selection"] is False
    assert (
        audit["lstr_minus_dev_selected_generic_clustered_mrr"]["tau2"][
            "mean_difference"
        ]
        == 0.5
    )
