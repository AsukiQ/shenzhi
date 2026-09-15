from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest

from clstr.memory_utility_gate import (
    AnchoredMemoryUtilityGate,
    MemoryUtilityGate,
)
from clstr.memory_utility_gate_train import (
    direct_harm_targets,
    direct_utility_targets,
    load_memory_utility_gate_checkpoint,
    resolve_reliability_gate,
    train_memory_utility_gate,
)
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path


BENCHMARKS = ("toolbench_g3", "traject_bench", "alfworld", "webshop")


def test_reliability_resolver_loads_learned_and_optional_cmc_overlay(
    monkeypatch,
    tmp_path,
):
    import clstr.memory_utility_gate_train as gate_train

    sentinel = torch.nn.Identity()
    checkpoint = tmp_path / "gate.pt"
    checkpoint.write_bytes(b"gate")
    monkeypatch.setattr(
        gate_train,
        "load_memory_utility_gate_checkpoint",
        lambda *args, **kwargs: (
            sentinel,
            {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": "gate-sha",
                "audit_manifest_sha256": "audit-sha",
                "feature_update_count_cap": 4.0,
                "feature_candidate_count_cap": 64.0,
            },
        ),
    )

    gate, report = resolve_reliability_gate(
        reliability_mode="learned",
        gate_checkpoint_path=checkpoint,
        expected_gate_sha256="gate-sha",
        expected_audit_sha256="audit-sha",
        device="cpu",
    )
    assert gate is sentinel
    assert report["gate_loaded"] is True
    assert report["feature_update_count_cap"] == 4.0
    gate, report = resolve_reliability_gate(
        reliability_mode="cmc",
        gate_checkpoint_path=checkpoint,
        expected_gate_sha256="gate-sha",
        expected_audit_sha256="audit-sha",
        device="cpu",
    )
    assert gate is sentinel
    assert report["gate_loaded"] is True
    assert report["gate_source"] == "direct_utility_overlay"
    gate, report = resolve_reliability_gate(
        reliability_mode="cmc",
        gate_checkpoint_path=None,
    )
    assert gate is None
    assert report == {
        "mode": "cmc",
        "gate_loaded": False,
        "gate_source": "stage4_checkpoint",
    }
    with pytest.raises(ValueError, match="requires a gate checkpoint"):
        resolve_reliability_gate(
            reliability_mode="learned",
            gate_checkpoint_path=None,
        )
    with pytest.raises(ValueError, match="must not load"):
        resolve_reliability_gate(
            reliability_mode="fixed_alpha",
            gate_checkpoint_path=checkpoint,
        )


def _record(
    benchmark: str,
    trajectory_id: str,
    row_index: int,
    *,
    dynamic_better: bool,
    identical_endpoints: bool = False,
) -> dict:
    if identical_endpoints:
        static_logits = dynamic_logits = [4.0, 0.0]
        feature = 0.0
    elif dynamic_better:
        static_logits, dynamic_logits, feature = [0.0, 4.0], [5.0, 0.0], 1.0
    else:
        static_logits, dynamic_logits, feature = [5.0, 0.0], [0.0, 5.0], -1.0
    return {
        "schema_version": "memory_utility_route_record_v1",
        "row_digest": hashlib.sha256(
            f"{benchmark}|{trajectory_id}|{row_index}".encode("utf-8")
        ).hexdigest(),
        "benchmark": benchmark,
        "trajectory_id": trajectory_id,
        "causal_update_count": 1,
        "features": [feature] + [0.0] * 10,
        "static_logits": static_logits,
        "dynamic_logits": dynamic_logits,
        "valid_mask": [True, True],
        "positive_mask": [True, False],
    }


def _records(prefix: str, *, identical_endpoints: bool = False) -> list[dict]:
    return [
        _record(
            benchmark,
            f"{prefix}-{trajectory_index}",
            trajectory_index,
            dynamic_better=(trajectory_index % 2 == 0),
            identical_endpoints=identical_endpoints,
        )
        for benchmark in BENCHMARKS
        for trajectory_index in range(6)
    ]


def _audit(
    *,
    recommended: bool,
    route_records: list[dict],
    route_manifest_identity: dict,
) -> dict:
    payload = {
        "status": "ok",
        "learned_gate_recommended": recommended,
        "recommendation_blockers": [] if recommended else ["utility_not_predictable"],
        "split": {
            "train_trajectory_ids": [
                f"{benchmark}::train-{trajectory_index}"
                for benchmark in BENCHMARKS
                for trajectory_index in range(4)
            ],
            "dev_trajectory_ids": [
                f"{benchmark}::train-{trajectory_index}"
                for benchmark in BENCHMARKS
                for trajectory_index in (4, 5)
            ],
        },
        "count_feature_caps": {
            "causal_update_count": 4.0,
            "valid_candidate_count": 64.0,
            "derived_from": "train_trajectories_only",
        },
        "best_fixed_alpha": 0.85,
        "anchored_harm_eligibility": {
            "alpha_base": 0.85,
            "both_harm_signs_present": True,
            "fixed_or_static_utility_oracle_gain_over_fixed": 0.10,
            "linear_harm_selector_gain_over_fixed": 0.08,
            "worst_source_regret_vs_fixed": 0.0,
            "eligible": recommended,
        },
        "route_record_source": {
            "record_count": len(route_records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in route_records]
            ),
        },
        "route_manifests": [
            {
                "manifest_path": route_manifest_identity["path"],
                "manifest_file_identity": dict(route_manifest_identity),
            }
        ],
        "route_manifest_identity": {
            "model_checkpoint_chain_digest": "selected-stage4-chain"
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload


def _dynamic_selection(
    tmp_path: Path,
    calibration_records: list[dict],
) -> tuple[dict, dict]:
    checkpoint = tmp_path / "selected-stage4.pt"
    checkpoint.write_bytes(b"stage4-delta")
    validation_report = tmp_path / "selected-validation.json"
    validation_report.write_text("{}", encoding="utf-8")
    validation_records_path = tmp_path / "selected-validation.route_records.jsonl"
    validation_records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_records),
        encoding="utf-8",
    )
    route_manifest_path = tmp_path / "selected-validation.route_manifest.json"
    route_manifest = {
        "schema_version": "memory_utility_route_manifest_v1",
        "records_path": str(validation_records_path.resolve()),
        "records_sha256": hashlib.sha256(
            validation_records_path.read_bytes()
        ).hexdigest(),
        "record_count": len(calibration_records),
        "source_rows_digest": canonical_digest(
            [row["row_digest"] for row in calibration_records]
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
    route_manifest_path.write_text(
        json.dumps(route_manifest, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "stage4_dynamic_selection_v1",
        "status": "ok",
        "release_status": "ok",
        "stage4_method": "counterfactual_memory_calibration_v1",
        "selected_step": 800,
        "selected_checkpoint_path": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint)["sha256"],
        "selected_validation_report_path": str(validation_report.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_report)["sha256"],
        "router_integrity": {
            "full_router_digest": "router-full",
            "fast_router_digest": "router-fast",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "report_sha256": "stage2-baseline-sha256",
            "full_router_digest": "router-full",
            "static_macro_mrr": 0.60,
            "dynamic_macro_mrr": 0.60,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60, "dynamic_mrr": 0.60}
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
                benchmark: {"mrr": 0.76} for benchmark in BENCHMARKS
            },
        },
        "validation_route_records": {
            **sha256_path(validation_records_path),
            "record_count": len(calibration_records),
            "row_digest_sha256": canonical_digest(
                [row["row_digest"] for row in calibration_records]
            ),
        },
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload, sha256_path(route_manifest_path)


def test_direct_utility_targets_are_detached_signed_and_confidence_weighted() -> None:
    static = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    dynamic = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    valid = torch.ones_like(static, dtype=torch.bool)
    positive = torch.tensor([[True, False], [True, False]])

    target, weight, delta = direct_utility_targets(
        static,
        dynamic,
        valid,
        positive,
        temperature=0.5,
    )

    assert target[0] > 0.5 and target[1] < 0.5
    assert torch.all((weight >= 0) & (weight <= 1))
    assert not target.requires_grad
    assert not weight.requires_grad
    assert not delta.requires_grad


def test_direct_harm_targets_and_anchored_gate_preserve_fixed_alpha_by_default() -> None:
    static = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
    fixed = torch.tensor([[0.0, 4.0], [4.0, 0.0]], requires_grad=True)
    valid = torch.ones_like(static, dtype=torch.bool)
    positive = torch.tensor([[True, False], [True, False]])

    target, weight, delta = direct_harm_targets(
        static,
        fixed,
        valid,
        positive,
        temperature=0.5,
    )

    assert target[0] > 0.5 and target[1] < 0.5
    assert torch.all((weight >= 0) & (weight <= 1))
    assert not target.requires_grad
    assert not weight.requires_grad
    assert not delta.requires_grad

    harm_gate = MemoryUtilityGate()
    features = torch.zeros((2, 11), dtype=torch.float32)
    wrapper = AnchoredMemoryUtilityGate(harm_gate, alpha_base=0.85)
    assert torch.allclose(
        wrapper(features),
        0.85 * (1.0 - harm_gate(features)),
    )


def test_gate_training_refuses_a_negative_audit(tmp_path: Path) -> None:
    route_records = _records("train")
    selection, route_manifest_identity = _dynamic_selection(tmp_path, route_records)
    report = train_memory_utility_gate(
        route_records=route_records,
        audit_report=_audit(
            recommended=False,
            route_records=route_records,
            route_manifest_identity=route_manifest_identity,
        ),
        dynamic_selection=selection,
        output_dir=tmp_path,
        temperature=0.5,
        fused_rank_weight=1.0,
        static_no_regret_weight=1.0,
        direct_gate_bce_weight=1.0,
        max_steps=20,
        learning_rate=0.01,
    )
    assert report["status"] == "not_recommended"
    assert report["checkpoint_path"] is None
    assert (tmp_path / "gate_report.json").is_file()
    assert not (tmp_path / "memory_utility_gate.pt").exists()


def test_gate_training_writes_audit_bound_checkpoint_when_validation_improves(
    tmp_path: Path,
) -> None:
    route_records = _records("train")
    selection, route_manifest_identity = _dynamic_selection(tmp_path, route_records)
    audit = _audit(
        recommended=True,
        route_records=route_records,
        route_manifest_identity=route_manifest_identity,
    )
    report = train_memory_utility_gate(
        route_records=route_records,
        audit_report=audit,
        dynamic_selection=selection,
        output_dir=tmp_path,
        temperature=0.5,
        fused_rank_weight=1.0,
        static_no_regret_weight=1.0,
        direct_gate_bce_weight=1.0,
        max_steps=300,
        learning_rate=0.01,
    )
    assert report["promoted"] is True, (
        report["validation"]["promotion_checks"],
        report["dev_summary"]["alpha"],
    )
    payload = torch.load(report["checkpoint_path"], map_location="cpu")
    gate, load_report = load_memory_utility_gate_checkpoint(
        report["checkpoint_path"],
        expected_sha256=report["checkpoint_sha256"],
        expected_audit_sha256=audit["manifest_sha256"],
    )
    assert report["status"] == "ok"
    assert report["promoted"] is True
    assert payload["stage"] == "clstr_memory_utility_gate"
    assert payload["schema_version"] == "memory_utility_gate_checkpoint_v2"
    assert payload["feature_schema"] == "memory_utility_features_v1"
    assert payload["gate_output_semantics"] == "anchored_harm_suppression_alpha"
    assert payload["alpha_base"] == 0.85
    assert payload["zero_history_fallback"] == "exact_static"
    assert payload["audit_manifest_sha256"] == audit["manifest_sha256"]
    assert payload["harm_probability_gate_state_dict"]
    assert set(payload["trainable_parameter_names"]) == {
        "net.0.weight",
        "net.0.bias",
    }
    assert payload["objective"] == {
        "temperature": 0.5,
        "fused_rank_weight": 1.0,
        "static_no_regret_weight": 1.0,
        "direct_gate_bce_weight": 1.0,
        "source_balanced": True,
        "harm_sign_balanced": True,
    }
    assert isinstance(gate, AnchoredMemoryUtilityGate)
    assert gate.training is False
    assert load_report["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert load_report["feature_update_count_cap"] == 4.0
    assert load_report["feature_candidate_count_cap"] == 64.0
    assert load_report["alpha_base"] == 0.85
    assert load_report["gate_output_semantics"] == "anchored_harm_suppression_alpha"
    checks = report["validation"]["promotion_checks"]
    assert checks["positive_source_balanced_fused_mrr_delta_vs_fixed"] is True
    assert checks["harmful_rows_improve"] is True
    assert checks["helpful_rows_preserved"] is True
    assert checks["harmful_helpful_alpha_separated"] is True


def test_gate_training_reuses_audit_row_eligibility_within_selected_trajectories(
    tmp_path: Path,
) -> None:
    route_records = _records("train")
    for row_index, trajectory_index in enumerate((0, 4), start=len(route_records)):
        row = _record(
            "toolbench_g3",
            f"train-{trajectory_index}",
            row_index,
            dynamic_better=True,
        )
        row["positive_mask"] = [False, False]
        route_records.append(row)
    selection, route_manifest_identity = _dynamic_selection(tmp_path, route_records)
    audit = _audit(
        recommended=True,
        route_records=route_records,
        route_manifest_identity=route_manifest_identity,
    )

    report = train_memory_utility_gate(
        route_records=route_records,
        audit_report=audit,
        dynamic_selection=selection,
        output_dir=tmp_path / "gate",
        temperature=0.5,
        fused_rank_weight=1.0,
        static_no_regret_weight=1.0,
        direct_gate_bce_weight=1.0,
        max_steps=300,
        learning_rate=0.01,
    )

    assert report["status"] == "ok"
    assert report["calibration_rows"] == {
        "train": 16,
        "dev": 8,
        "excluded_from_selected_trajectories": 2,
    }


def test_gate_training_keeps_fixed_alpha_when_validation_has_no_advantage(
    tmp_path: Path,
) -> None:
    route_records = _records("train", identical_endpoints=True)
    selection, route_manifest_identity = _dynamic_selection(tmp_path, route_records)
    report = train_memory_utility_gate(
        route_records=route_records,
        audit_report=_audit(
            recommended=True,
            route_records=route_records,
            route_manifest_identity=route_manifest_identity,
        ),
        dynamic_selection=selection,
        output_dir=tmp_path,
        temperature=0.5,
        fused_rank_weight=1.0,
        static_no_regret_weight=1.0,
        direct_gate_bce_weight=1.0,
        max_steps=300,
        learning_rate=0.01,
    )
    assert report["status"] == "ok"
    assert report["promoted"] is False
    assert report["checkpoint_path"] is None
    assert (tmp_path / "gate_report.json").is_file()
    assert not (tmp_path / "memory_utility_gate.pt").exists()
