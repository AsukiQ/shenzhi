from __future__ import annotations

import json
from pathlib import Path

import pytest

from clstr.vnext_eval import _stage2_release_selection_contract
from clstr.vnext_matched_release import (
    MATCHED_RELEASE_BOOTSTRAP_PROTOCOL,
    MATCHED_RELEASE_SELECTION_SCHEMA,
    build_matched_release_selection,
    file_sha256,
    json_digest,
    paired_cluster_bootstrap,
)


def test_matched_release_bootstrap_uses_canonical_fsum_protocol() -> None:
    assert MATCHED_RELEASE_BOOTSTRAP_PROTOCOL.endswith("_fsum12_v2")

    rows = [
        _rank_record("trajectory", index, factual=3, static=7, mismatch=9)
        for index in range(13)
    ]
    report = paired_cluster_bootstrap(rows, comparison="static", samples=20)
    for metric in report["metrics"].values():
        for field in ("mean", "ci_low", "ci_high"):
            assert metric[field] == float(f"{metric[field]:.12g}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metric(mean: float, low: float, *, count: int = 100, clusters: int = 50) -> dict:
    return {
        "count": count,
        "cluster_count": clusters,
        "mean": mean,
        "ci_low": low,
        "ci_high": mean + 0.05,
    }


def _validation_record() -> dict:
    toolbench = {
        "row_count": 100,
        "cluster_count": 50,
        "route_mrr_delta": _metric(0.08, 0.02),
        "route_recall_at_1_delta": _metric(0.04, -0.005),
        "route_recall_at_5_delta": _metric(0.10, 0.03),
        "raw_route_mrr_delta": _metric(0.075, 0.015),
        "candidate_union_minus_static_recall": _metric(0.0, 0.0),
    }
    return {
        "step": 500,
        "gates": {
            "causal_route_gate": True,
            "causal_safety_gate": True,
            "cluster_sufficiency_gate": True,
            "coarse_recall_gate": True,
            "exact_fallback_gate": True,
            "gradient_health_gate": True,
            "robust_prefix_dev_gate": True,
        },
        "ordinary_dev": {
            "overall": {"route_mrr_delta": _metric(0.09, 0.04)},
            "per_family": {"toolbench": toolbench},
            "per_source": {"toolbench_g3": dict(toolbench)},
        },
    }


def _rank_record(
    trajectory: str,
    index: int,
    *,
    factual: int,
    static: int,
    mismatch: int | None,
) -> dict:
    def score(rank: int | None) -> dict:
        return {"end_to_end_route_rank": rank}

    return {
        "benchmark": "tau2",
        "trajectory_id": trajectory,
        "source_index": index,
        "factual": score(factual),
        "static": score(static),
        "masked": score(static),
        "mismatch": score(mismatch),
        "mismatch_donor_source_index": index + 10 if mismatch is not None else None,
        "order_shuffle": score(static),
        "order_shuffle_eligible": True,
    }


def _dev_report(
    *,
    benchmark: str,
    checkpoint: Path,
    skills: Path,
    manifest: Path,
    records: Path,
    protocol: Path,
    row_count: int,
    recurrent_count: int,
    factual_metrics: dict,
    static_metrics: dict,
) -> dict:
    return {
        "schema_version": "clstr_vnext_checkpoint_native_eval_v20",
        "status": "ok",
        "blockers": [],
        "benchmark": benchmark,
        "method": "clstr_vnext_stage2_checkpoint_native",
        "evaluation_scope": "stage2_complete_method",
        "stage2_release_selection_contract": None,
        "requested_stage2_checkpoint_path": str(checkpoint.resolve()),
        "checkpoint": {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_step": 500,
            "training_skills_path": str(skills.resolve()),
            "training_skills_sha256": file_sha256(skills),
        },
        "corpus": {
            "task_split": "dev",
            "matched_prebuilt_manifest_path": str(manifest.resolve()),
        },
        "corpus_source_contract": {
            "task_split": "dev",
            "matched_union_manifest_sha256": json.loads(
                manifest.read_text(encoding="utf-8")
            )["manifest_sha256"],
        },
        "non_static_memory_metrics_release_eligible": True,
        "release_eligible_metrics": [
            "factual.end_to_end_route",
            "factual.full_pool_recall",
        ],
        "rows": {
            "routing_row_count": row_count,
            "history_channel": {"status": "ok", "leaked_row_count": 0},
        },
        "route_records_path": str(records.resolve()),
        "protocol_manifest_path": str(protocol.resolve()),
        "metrics": {
            "factual": {
                "candidate_union_recall": 1.0,
                "end_to_end_route": factual_metrics,
            },
            "static": {"end_to_end_route": static_metrics},
        },
        "memory": {"uses_recurrent_m_t_count": recurrent_count},
    }


def _fixture(tmp_path: Path, *, mismatch_eligible: bool = True) -> dict[str, Path]:
    output = tmp_path / "stage2"
    checkpoint = output / "checkpoints" / "clstr_vnext_stage2-step500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    skills = tmp_path / "selected_skills.jsonl"
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")
    validation = _validation_record()
    original = {
        "status": "action_required",
        "selected_step": 0,
        "validation_records": [validation],
    }
    train_report = {
        "status": "action_required",
        "finite_loss": True,
        "selection": original,
        "positive_injection_count": 0,
        "training_teacher_retained_rows": 0,
        "objective_warm_start": {
            "enabled": True,
            "model_state_exact": True,
            "optimizer_restored": False,
            "trainer_progress_restored": False,
        },
    }
    _write_json(output / "stage2_selection.json", original)
    _write_json(output / "stage2_quality_gate.json", original)
    _write_json(output / "train_report.json", train_report)
    _write_json(
        output / "source_manifest.json",
        {"source_contract": {"git_commit": "checkpoint-source", "dirty": False}},
    )

    tau_rows_path = tmp_path / "tau2_dev_rows.jsonl"
    toolsandbox_rows_path = tmp_path / "toolsandbox_dev_rows.jsonl"
    _write_jsonl(tau_rows_path, [{"matched_data_split": "dev"}])
    _write_jsonl(toolsandbox_rows_path, [{"matched_data_split": "dev"}])
    manifest = {
        "schema_version": "clstr_matched_multibench_union_v1",
        "status": "ok",
        "split_seed": "unit",
        "route_files": {
            "tau2_dev": {
                "path": str(tau_rows_path.resolve()),
                "sha256": file_sha256(tau_rows_path),
            },
            "toolsandbox_dev": {
                "path": str(toolsandbox_rows_path.resolve()),
                "sha256": file_sha256(toolsandbox_rows_path),
            },
        },
        "tau2_split_manifest": {
            "official_test_preserved": True,
            "manifest_sha256": "tau-split",
        },
        "toolsandbox_split_manifest": {
            "family_to_split": {"family": "dev"},
            "manifest_sha256": "sandbox-split",
        },
    }
    manifest["manifest_sha256"] = json_digest(manifest)
    manifest_path = tmp_path / "matched_union_manifest.json"
    _write_json(manifest_path, manifest)

    tau_records = [
        _rank_record(
            "tau-a",
            0,
            factual=1,
            static=6,
            mismatch=8 if mismatch_eligible else None,
        ),
        _rank_record(
            "tau-b",
            1,
            factual=1,
            static=7,
            mismatch=9 if mismatch_eligible else None,
        ),
    ]
    tau_records_path = tmp_path / "tau2_records.jsonl"
    _write_jsonl(tau_records_path, tau_records)
    tau_protocol = tmp_path / "tau2_protocol.json"
    _write_json(tau_protocol, {"status": "ok"})
    tau_report = tmp_path / "tau2_report.json"
    _write_json(
        tau_report,
        _dev_report(
            benchmark="tau2",
            checkpoint=checkpoint,
            skills=skills,
            manifest=manifest_path,
            records=tau_records_path,
            protocol=tau_protocol,
            row_count=2,
            recurrent_count=2,
            factual_metrics={"mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0},
            static_metrics={"mrr": 13 / 84, "recall@1": 0.0, "recall@5": 0.0},
        ),
    )

    sandbox_records = [
        {
            **_rank_record(
                "sandbox-a",
                0,
                factual=1,
                static=1,
                mismatch=None,
            ),
            "benchmark": "toolsandbox",
            "order_shuffle_eligible": False,
        }
    ]
    sandbox_records_path = tmp_path / "toolsandbox_records.jsonl"
    _write_jsonl(sandbox_records_path, sandbox_records)
    sandbox_protocol = tmp_path / "toolsandbox_protocol.json"
    _write_json(sandbox_protocol, {"status": "ok"})
    sandbox_report = tmp_path / "toolsandbox_report.json"
    _write_json(
        sandbox_report,
        _dev_report(
            benchmark="toolsandbox",
            checkpoint=checkpoint,
            skills=skills,
            manifest=manifest_path,
            records=sandbox_records_path,
            protocol=sandbox_protocol,
            row_count=1,
            recurrent_count=1,
            factual_metrics={"mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0},
            static_metrics={"mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0},
        ),
    )
    return {
        "output": output,
        "checkpoint": checkpoint,
        "skills": skills,
        "manifest": manifest_path,
        "tau_report": tau_report,
        "sandbox_report": sandbox_report,
        "tau_records": tau_records_path,
    }


def test_paired_cluster_bootstrap_requires_trajectory_specific_history() -> None:
    rows = [
        _rank_record("a", 0, factual=1, static=2, mismatch=3),
        _rank_record("b", 1, factual=1, static=4, mismatch=5),
    ]
    result = paired_cluster_bootstrap(rows, comparison="mismatch", samples=100)
    assert result["eligible"] is True
    assert result["cluster_count"] == 2
    assert result["metrics"]["mrr"]["ci_low"] > 0.0
    assert result["metrics"]["recall_at_1"]["ci_low"] > 0.0
    assert result["metrics"]["recall_at_5"]["ci_low"] == 0.0


def test_matched_release_accepts_positive_dev_and_binds_every_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    payload = build_matched_release_selection(
        stage2_output_dir=paths["output"],
        selected_step=500,
        checkpoint_path=paths["checkpoint"],
        training_skills_path=paths["skills"],
        matched_union_manifest_path=paths["manifest"],
        tau2_dev_report_path=paths["tau_report"],
        toolsandbox_dev_report_path=paths["sandbox_report"],
        source_commit="release-source",
    )
    assert payload["schema_version"] == MATCHED_RELEASE_SELECTION_SCHEMA
    assert payload["status"] == "ok"
    assert payload["blockers"] == []
    assert payload["closed_set_dispatch_required"] is False
    selection_path = tmp_path / "matched_release.json"
    _write_json(selection_path, payload)
    contract = _stage2_release_selection_contract(
        selection_path,
        stage2_checkpoint_path=paths["checkpoint"],
    )
    assert contract["selection_mode"] == "matched_multibench_unified_recurrent"
    assert contract["selected_step"] == 500
    assert contract["uses_test_metrics"] is False
    assert contract["closed_set_dispatch_required"] is False

    paths["tau_records"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="route-record|route_records_path"):
        _stage2_release_selection_contract(
            selection_path,
            stage2_checkpoint_path=paths["checkpoint"],
        )


def test_matched_release_fails_closed_without_mismatched_history_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, mismatch_eligible=False)
    payload = build_matched_release_selection(
        stage2_output_dir=paths["output"],
        selected_step=500,
        checkpoint_path=paths["checkpoint"],
        training_skills_path=paths["skills"],
        matched_union_manifest_path=paths["manifest"],
        tau2_dev_report_path=paths["tau_report"],
        toolsandbox_dev_report_path=paths["sandbox_report"],
        source_commit="release-source",
    )
    assert payload["status"] == "action_required"
    assert "tau2_mismatch_comparison_missing" in payload["blockers"]


def test_matched_release_cli_and_launcher_use_unified_recurrent_contract() -> None:
    root = Path(__file__).parents[1]
    selector = root.joinpath(
        "scripts/select_clstr_vnext_matched_release.py"
    ).read_text(encoding="utf-8")
    launcher = root.joinpath(
        "scripts/sbatch/run_clstr_vnext_benchmark_eval.sh"
    ).read_text(encoding="utf-8")
    evaluator = root.joinpath("clstr/vnext_eval.py").read_text(encoding="utf-8")
    assert "build_matched_release_selection" in selector
    assert "clstr_vnext_stage2_matched_multibench_selection_v1" in launcher
    assert "matched_multibench_unified_recurrent" in launcher
    assert "matched unified recurrent release may not use the closed-set foundation" in launcher
    assert "MATCHED_RELEASE_SELECTION_SCHEMA" in evaluator
    assert "matched unified recurrent release may not use deployment dispatch" in evaluator
