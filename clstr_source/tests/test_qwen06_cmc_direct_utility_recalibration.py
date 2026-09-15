from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from scripts.run_qwen06_cmc_direct_utility_recalibration import (
    run_qwen06_cmc_direct_utility_recalibration,
)


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")


def _record(
    *,
    benchmark: str,
    trajectory_index: int,
    row_index: int,
    dynamic_better: bool,
) -> dict:
    if dynamic_better:
        static_logits, dynamic_logits, feature = [0.0, 10.0], [4.0, 0.0], 1.0
    else:
        static_logits, dynamic_logits, feature = [4.0, 0.0], [0.0, 10.0], -1.0
    trajectory_id = f"{benchmark}/trajectory-{trajectory_index}"
    return {
        "schema_version": "memory_utility_route_record_v1",
        "row_digest": hashlib.sha256(
            f"{benchmark}|{trajectory_id}|{row_index}".encode("utf-8")
        ).hexdigest(),
        "trajectory_id": trajectory_id,
        "task_id": f"{trajectory_id}::{row_index}",
        "benchmark": benchmark,
        "sequential_benchmark": True,
        "causal_update_count": 1,
        "features": [feature] + [0.0] * 10,
        "static_logits": static_logits,
        "dynamic_logits": dynamic_logits,
        "valid_mask": [True, True],
        "positive_mask": [True, False],
    }


def _qualifying_records() -> list[dict]:
    return [
        _record(
            benchmark=benchmark,
            trajectory_index=trajectory_index,
            row_index=row_index,
            dynamic_better=row_index < 3,
        )
        for benchmark in BENCHMARKS
        for trajectory_index in range(12)
        for row_index in range(5)
    ]


def _write_inputs(
    tmp_path: Path,
    records: list[dict],
    *,
    name: str = "selected",
) -> tuple[Path, Path]:
    records_path = tmp_path / f"{name}.route_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    route_manifest = {
        "schema_version": "memory_utility_route_manifest_v1",
        "records_path": str(records_path.resolve()),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "record_count": len(records),
        "source_rows_digest": canonical_digest(
            [row["row_digest"] for row in records]
        ),
        "candidate_union_version": "memory_union_v1",
        "candidate_selection_version": "stable_declared_pool_v1",
        "pool_protocol": "static_plus_dynamic_extra",
        "static_k": 500,
        "dynamic_extra_k": 64,
        "final_k": 64,
        "feature_schema": "memory_utility_features_v1",
        "heuristic_alpha_version": "memory_utility_heuristic_v1",
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "model_checkpoint_chain_digest": "selected-stage4-chain",
        "skill_mapping_digest": "selected-skill-mapping",
        "sequential_benchmarks": list(BENCHMARKS),
    }
    route_manifest_path = tmp_path / f"{name}.route_manifest.json"
    route_manifest_path.write_text(
        json.dumps(route_manifest, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint = tmp_path / f"{name}.stage4.pt"
    checkpoint.write_bytes(b"selected-stage4-delta")
    validation_report = tmp_path / f"{name}.validation.json"
    validation_report.write_text("{}\n", encoding="utf-8")
    selection = {
        "schema_version": "stage4_dynamic_selection_v1",
        "status": "ok",
        "release_status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "selected_step": 3000,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(validation_report.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_report)[
            "sha256"
        ],
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "static_macro_mrr": 0.75,
            "raw_dynamic_macro_mrr": 0.75,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.75, "raw_dynamic_mrr": 0.75}
                for benchmark in BENCHMARKS
            },
        },
        "candidate_union": {"static_k": 500, "dynamic_extra_k": 64, "final_k": 64},
        "cmc_fused_selection": {
            "balanced_macro_mrr": 0.76,
            "regret": 0.0,
            "alpha_zero_exact": True,
            "alpha_one_exact": True,
            "zero_history_exact": True,
            "by_benchmark": {
                benchmark: {"fused_mrr": 0.76} for benchmark in BENCHMARKS
            },
        },
        "validation_route_records": {
            **sha256_path(records_path),
            "record_count": len(records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in records]
            ),
        },
    }
    selection["manifest_sha256"] = canonical_digest(selection)
    selection_path = tmp_path / f"{name}.stage4_dynamic_selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True),
        encoding="utf-8",
    )
    return selection_path, route_manifest_path


def test_recalibration_stops_after_negative_audit(tmp_path: Path) -> None:
    records = _qualifying_records()[:4]
    selection_path, route_manifest_path = _write_inputs(tmp_path, records)
    output_dir = tmp_path / "negative"

    report = run_qwen06_cmc_direct_utility_recalibration(
        dynamic_selection_path=selection_path,
        route_manifest_path=route_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=0,
    )

    assert report["status"] == "not_recommended"
    assert report["gate_report"] is None
    assert report["checkpoint"] is None
    assert (output_dir / "oracle_audit.json").is_file()
    assert not (output_dir / "gate" / "memory_utility_gate.pt").exists()
    payload = dict(report)
    recorded = payload.pop("manifest_sha256")
    assert recorded == canonical_digest(payload)


def test_recalibration_trains_and_binds_promoted_gate(tmp_path: Path) -> None:
    records = _qualifying_records()
    selection_path, route_manifest_path = _write_inputs(tmp_path, records)
    output_dir = tmp_path / "positive"

    report = run_qwen06_cmc_direct_utility_recalibration(
        dynamic_selection_path=selection_path,
        route_manifest_path=route_manifest_path,
        output_dir=output_dir,
        bootstrap_samples=10,
    )

    assert report["status"] == "ok"
    assert report["promoted"] is True
    assert report["gate_report"] == sha256_path(output_dir / "gate" / "gate_report.json")
    assert report["checkpoint"] == sha256_path(
        output_dir / "gate" / "memory_utility_gate.pt"
    )
    assert report["gate_output_semantics"] == "anchored_harm_suppression_alpha"
    assert report["alpha_base"] == json.loads(
        (output_dir / "oracle_audit.json").read_text(encoding="utf-8")
    )["best_fixed_alpha"]
    assert report["selected_stage4_checkpoint_sha256"] == json.loads(
        selection_path.read_text(encoding="utf-8")
    )["selected_checkpoint_sha256"]
    payload = dict(report)
    recorded = payload.pop("manifest_sha256")
    assert recorded == canonical_digest(payload)


def test_recalibration_rejects_manifest_other_than_selected_validation(
    tmp_path: Path,
) -> None:
    records = _qualifying_records()
    selection_path, _selected_manifest = _write_inputs(tmp_path, records)
    changed = list(records)
    changed[0] = _record(
        benchmark="toolbench_g3",
        trajectory_index=99,
        row_index=0,
        dynamic_better=True,
    )
    _unused_selection, wrong_manifest = _write_inputs(
        tmp_path,
        changed,
        name="wrong",
    )

    with pytest.raises(ValueError, match="selected validation route records"):
        run_qwen06_cmc_direct_utility_recalibration(
            dynamic_selection_path=selection_path,
            route_manifest_path=wrong_manifest,
            output_dir=tmp_path / "mismatch",
            bootstrap_samples=0,
        )


def test_direct_utility_recalibration_launcher_is_model_free_and_predeclared() -> None:
    text = Path(
        "scripts/sbatch/run_qwen06_cmc_direct_utility_recalibration.sh"
    ).read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=" in text
    assert "--temperature 0.5" in text
    assert "--fused_rank_weight 1.0" in text
    assert "--static_no_regret_weight 1.0" in text
    assert "--direct_gate_bce_weight 1.0" in text
    assert "--max_steps 300" in text
    assert "--learning_rate 0.01" in text
    assert "--seed 17" in text
    assert "Qwen" not in text
    assert "load_model" not in text
