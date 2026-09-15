import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.memory_utility_gate import MemoryUtilityGate
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_final_chain import (
    _resolve_training_layout,
    _validate_cmc_checkpoint_parent_digests,
    _validate_reliability,
)
from clstr.qwen_clstr_lineage import SCHEMA_VERSION as LINEAGE_SCHEMA_VERSION
from clstr.qwen_clstr_lineage import sha256_path


STATE_QUERY = {
    "prompt_version": "clstr_causal_state_v1",
    "instruction": (
        "Given an agent task or current execution state and interaction history, "
        "retrieve the skill or tool document most useful for the next action."
    ),
    "max_chars": 2000,
    "truncation": "head_tail_v1",
}


def test_final_chain_accepts_joint_causal_gate_reliability() -> None:
    reliability = {
        "mode": "causal_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "safe_memory_residual_bound": 2.0,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)

    assert _validate_reliability(reliability) == reliability


def test_final_chain_accepts_checkpoint_native_cmc_reliability() -> None:
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)

    assert _validate_reliability(reliability) == reliability


def _candidate_provenance_reliability(**overrides) -> dict:
    reliability = {
        "mode": "cmc_candidate_provenance",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
        "safe_memory_residual_bound": 2.0,
        "selected_stage4_checkpoint_sha256": "b" * 64,
        "base_reliability_source": "stage4_checkpoint",
        "deployed_reliability_source": "candidate_provenance_positive_residual",
    }
    reliability.update(overrides)
    reliability["reliability_sha256"] = canonical_digest(reliability)
    return reliability


def test_final_chain_accepts_candidate_provenance_reliability() -> None:
    reliability = _candidate_provenance_reliability()

    assert _validate_reliability(reliability) == reliability


def test_final_chain_accepts_candidate_admission_reliability() -> None:
    reliability = {
        "mode": "candidate_admission_residual",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "safe_memory_residual_bound": 2.0,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
        "base_cmc_checkpoint_sha256": "a" * 64,
        "deployed_reliability_source": (
            "candidate_admission_constrained_residual"
        ),
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)

    assert _validate_reliability(reliability) == reliability


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"safe_memory_residual_bound": 1.0}, "residual bound"),
        ({"feature_update_count_cap": 1.0}, "feature count caps"),
        ({"gate_checkpoint": {"path": "unused", "sha256": "a" * 64}}, "gate checkpoint"),
        ({"base_reliability_source": "post_hoc_gate"}, "base reliability source"),
        ({"deployed_reliability_source": "benchmark_switch"}, "deployed reliability source"),
    ],
)
def test_final_chain_rejects_invalid_candidate_provenance_reliability(
    overrides,
    message,
) -> None:
    reliability = _candidate_provenance_reliability(**overrides)

    with pytest.raises(ValueError, match=message):
        _validate_reliability(reliability)


def test_final_chain_accepts_cmc_direct_utility_overlay_reliability(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "memory_utility_gate.pt"
    gate_path.write_bytes(b"gate")
    gate_report_path = tmp_path / "gate_report.json"
    gate_report_path.write_text("{}\n", encoding="utf-8")
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": sha256_path(gate_path),
        "gate_report": sha256_path(gate_report_path),
        "audit_manifest_sha256": "a" * 64,
        "selected_stage4_checkpoint_sha256": "b" * 64,
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "base_gate_source": "stage4_checkpoint",
        "deployed_gate_source": "direct_utility_overlay",
        "gate_output_semantics": "anchored_harm_suppression_alpha",
        "alpha_base": 0.85,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)

    assert _validate_reliability(reliability) == reliability


def test_final_chain_rejects_cmc_reliability_without_training_feature_caps() -> None:
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)

    with pytest.raises(ValueError, match="feature count caps"):
        _validate_reliability(reliability)


def test_final_chain_rejects_cmc_checkpoint_with_wrong_stage2_parent_digest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage4-cmc.pt"
    payload = {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "parent_stage2_full_router_digest": "wrong-full",
        "parent_stage2_fast_router_digest": "router-fast",
        "model_state_dict": {},
    }
    torch.save(payload, checkpoint)
    router_integrity = {
        "full_router_digest": "router-full",
        "fast_router_digest": "router-fast",
    }

    with pytest.raises(ValueError, match="parent Stage2 router digest mismatch"):
        _validate_cmc_checkpoint_parent_digests(
            checkpoint,
            router_integrity=router_integrity,
        )

    payload["parent_stage2_full_router_digest"] = "router-full"
    torch.save(payload, checkpoint)
    report = _validate_cmc_checkpoint_parent_digests(
        checkpoint,
        router_integrity=router_integrity,
    )
    assert report == {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "parent_stage2_full_router_digest": "router-full",
        "parent_stage2_fast_router_digest": "router-fast",
    }


def test_final_chain_prefers_isolated_safe_memory_training_layout(
    tmp_path: Path,
) -> None:
    legacy_stage2 = tmp_path / "stage2_full"
    legacy_stage4 = tmp_path / "stage4_safe_full"
    safe_stage2 = tmp_path / "safe_memory_fusion" / "stage2_full"
    safe_stage4 = tmp_path / "safe_memory_fusion" / "stage4_full"
    for path in (legacy_stage2, legacy_stage4, safe_stage2, safe_stage4):
        path.mkdir(parents=True)
    (safe_stage2 / "lineage.json").write_text("{}\n", encoding="utf-8")
    (safe_stage4 / "stage4_selection.json").write_text("{}\n", encoding="utf-8")

    layout = _resolve_training_layout(tmp_path)

    assert layout["stage2_dir"] == safe_stage2.resolve()
    assert layout["stage4_dir"] == safe_stage4.resolve()


def test_final_chain_prefers_current_cmc_training_layout(tmp_path: Path) -> None:
    cmc_stage2 = tmp_path / "stage2_anchored_full10000_v1"
    cmc_stage4 = tmp_path / "stage4_cmc_full"
    legacy_stage2 = tmp_path / "stage2_full"
    legacy_stage4 = tmp_path / "stage4_safe_full"
    for path in (cmc_stage2, cmc_stage4, legacy_stage2, legacy_stage4):
        path.mkdir(parents=True)
    (cmc_stage2 / "lineage.json").write_text("{}\n", encoding="utf-8")
    (cmc_stage4 / "stage4_selection.json").write_text("{}\n", encoding="utf-8")
    (legacy_stage2 / "lineage.json").write_text("{}\n", encoding="utf-8")
    (legacy_stage4 / "stage4_selection.json").write_text("{}\n", encoding="utf-8")

    layout = _resolve_training_layout(tmp_path)

    assert layout["stage2_dir"] == cmc_stage2.resolve()
    assert layout["stage4_dir"] == cmc_stage4.resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parent_record(path: Path, checkpoint_sha256: str) -> dict:
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_path(path)["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
    }


def _lineage(
    *,
    stage: str,
    checkpoint_path: Path,
    backbone: dict,
    skill_pool: dict,
    data_manifest: dict,
    parents: dict,
    identity_parent_role: str | None = None,
    stage_metadata: dict | None = None,
) -> dict:
    payload = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "stage": stage,
        "route_scorer": "unified_memory",
        "state_query": dict(STATE_QUERY),
        "backbone": dict(backbone),
        "skill_pool": dict(skill_pool),
        "data_manifest": dict(data_manifest),
        "checkpoint": sha256_path(checkpoint_path),
        "parents": parents,
    }
    if identity_parent_role is not None:
        payload["identity_parent_role"] = identity_parent_role
    if stage_metadata is not None:
        payload["stage_metadata"] = stage_metadata
    return payload


def _build_valid_run(tmp_path: Path) -> dict[str, Path]:
    run_root = tmp_path / "outputs" / "qwen06_clstr_postfix"
    model_path = tmp_path / "models" / "Qwen3-Embedding-0.6B"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    skills_path = tmp_path / "data" / "skill_pool.jsonl"
    manifest_path = tmp_path / "data" / "manifest.json"
    skills_path.parent.mkdir(parents=True)
    skills_path.write_text('{"skill_id":"skill/a"}\n', encoding="utf-8")
    manifest_path.write_text('{"status":"ok"}\n', encoding="utf-8")
    backbone = {**sha256_path(model_path), "frozen": True}
    skill_pool = sha256_path(skills_path)
    data_manifest = sha256_path(manifest_path)

    checkpoint_paths = {
        "stage0": run_root / "stage0_full" / "checkpoints" / "clstr_unified_retrieval_v2-step5000.pt",
        "stage1": run_root / "stage1_full" / "checkpoints" / "clstr_stage1_heads-step3000.pt",
        "stage2": run_root / "stage2_full" / "checkpoints" / "clstr_full_base-step10000.pt",
        "stage4": run_root / "stage4_safe_full" / "checkpoints" / "clstr_stage4_safe-step800.pt",
    }
    for role, path in checkpoint_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-checkpoint".encode("utf-8"))

    lineage_paths = {
        "stage0": run_root / "stage0_full" / "lineage-step5000.json",
        "stage1": run_root / "stage1_full" / "lineage.json",
        "stage2": run_root / "stage2_full" / "lineage.json",
        "stage4": run_root / "stage4_safe_full" / "lineage.json",
    }
    stage0 = _lineage(
        stage="stage0",
        checkpoint_path=checkpoint_paths["stage0"],
        backbone=backbone,
        skill_pool=skill_pool,
        data_manifest=data_manifest,
        parents={},
    )
    _write_json(lineage_paths["stage0"], stage0)
    stage1 = _lineage(
        stage="stage1",
        checkpoint_path=checkpoint_paths["stage1"],
        backbone=backbone,
        skill_pool=skill_pool,
        data_manifest=data_manifest,
        parents={
            "stage0": _parent_record(lineage_paths["stage0"], stage0["checkpoint"]["sha256"]),
        },
    )
    _write_json(lineage_paths["stage1"], stage1)
    stage2 = _lineage(
        stage="stage2",
        checkpoint_path=checkpoint_paths["stage2"],
        backbone=backbone,
        skill_pool=skill_pool,
        data_manifest=data_manifest,
        parents={
            "stage0": _parent_record(lineage_paths["stage0"], stage0["checkpoint"]["sha256"]),
            "stage1": _parent_record(lineage_paths["stage1"], stage1["checkpoint"]["sha256"]),
        },
    )
    _write_json(lineage_paths["stage2"], stage2)
    validation_report_path = (
        run_root / "stage4_safe_full" / "validation" / "step800.json"
    )
    _write_json(validation_report_path, {"status": "ok", "step": 800})
    reliability = {
        "mode": "fixed_alpha",
        "fixed_alpha": 0.5,
        "gate_checkpoint": None,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    stage4_selection_path = run_root / "stage4_safe_full" / "stage4_selection.json"
    stage4_selection = {
        "schema_version": "stage4_selection_v1",
        "status": "ok",
        "release_status": "ok",
        "dynamic_release_status": "ok",
        "release_checks": {
            "dynamic_release_ok": True,
            "fused_macro_improvement": True,
            "fused_per_benchmark_nonregression": True,
            "zero_history_exact": True,
        },
        "selected_checkpoint_path": str(checkpoint_paths["stage4"].resolve()),
        "selected_checkpoint_sha256": sha256_path(checkpoint_paths["stage4"])[
            "sha256"
        ],
        "selected_validation_report_path": str(validation_report_path.resolve()),
        "selected_validation_report_sha256": sha256_path(validation_report_path)[
            "sha256"
        ],
        "router_integrity": {
            "full_router_digest": "stage2-router-digest",
            "fast_router_digest": "stage2-fast-router-digest",
            "static_logits_exact": True,
            "max_abs_static_logit_difference": 0.0,
        },
        "stage2_baseline": {
            "report_sha256": "stage2-baseline-sha256",
            "full_router_digest": "stage2-router-digest",
            "static_macro_mrr": 0.60,
            "dynamic_macro_mrr": 0.62,
            "by_benchmark": {
                benchmark: {"static_mrr": 0.60, "dynamic_mrr": 0.62}
                for benchmark in (
                    "toolbench_g3",
                    "traject_bench",
                    "alfworld",
                    "webshop",
                )
            },
        },
        "candidate_union": {
            "static_k": 500,
            "dynamic_extra_k": 64,
            "final_k": 64,
        },
        "reliability": reliability,
        "reliability_validation": {
            "balanced_macro_mrr": 0.64,
            "zero_history_exact": True,
            "by_benchmark": {
                benchmark: {"mrr": 0.64}
                for benchmark in (
                    "toolbench_g3",
                    "traject_bench",
                    "alfworld",
                    "webshop",
                )
            },
        },
    }
    stage4_selection["manifest_sha256"] = canonical_digest(stage4_selection)
    _write_json(stage4_selection_path, stage4_selection)
    stage4 = _lineage(
        stage="stage4",
        checkpoint_path=checkpoint_paths["stage4"],
        backbone=backbone,
        skill_pool=skill_pool,
        data_manifest=data_manifest,
        parents={
            "stage0": _parent_record(lineage_paths["stage0"], stage0["checkpoint"]["sha256"]),
            "stage2": _parent_record(lineage_paths["stage2"], stage2["checkpoint"]["sha256"]),
        },
        identity_parent_role="stage2",
        stage_metadata={
            "stage4_selection": sha256_path(stage4_selection_path),
            "router_integrity": stage4_selection["router_integrity"],
            "reliability": reliability,
        },
    )
    _write_json(lineage_paths["stage4"], stage4)

    stage0_report = run_root / "stage0_audits" / "step5000" / "report.json"
    _write_json(stage0_report, {"status": "ok"})
    _write_json(
        run_root / "stage0_full" / "stage0_selection.json",
        {
            "status": "ok",
            "release_status": "ok",
            "selected_checkpoint_path": str(checkpoint_paths["stage0"]),
            "selected_lineage_path": str(lineage_paths["stage0"]),
            "selected_report_path": str(stage0_report),
        },
    )
    for role in ("stage1", "stage2"):
        _write_json(
            run_root / f"{role}_full" / f"{role}_quality_gate.json",
            {
                "status": "ok",
                "checkpoint_path": str(checkpoint_paths[role]),
            },
        )
    _write_json(
        run_root / "stage4_safe_full" / "stage4_quality_gate.json",
        {
            "status": "ok",
            "checkpoint_path": str(checkpoint_paths["stage4"]),
            "selection_path": str(stage4_selection_path),
            "selected_validation_report_path": str(validation_report_path),
            "router_integrity": stage4_selection["router_integrity"],
        },
    )
    return {
        "run_root": run_root,
        "model_path": model_path,
        "skills_path": skills_path,
        "manifest_path": manifest_path,
        "stage4_selection": stage4_selection_path,
        "stage4_validation_report": validation_report_path,
        **{f"{role}_checkpoint": path for role, path in checkpoint_paths.items()},
        **{f"{role}_lineage": path for role, path in lineage_paths.items()},
    }


def _convert_valid_run_to_cmc(paths: dict[str, Path]) -> None:
    stage4_checkpoint = paths["stage4_checkpoint"]
    torch.save(
        {
            "stage4_method": "counterfactual_memory_calibration_v1",
            "parent_stage2_full_router_digest": "stage2-router-digest",
            "parent_stage2_fast_router_digest": "stage2-fast-router-digest",
            "model_state_dict": {},
        },
        stage4_checkpoint,
    )
    selection = json.loads(paths["stage4_selection"].read_text(encoding="utf-8"))
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    selection.update(
        {
            "stage4_method": "counterfactual_memory_calibration_v1",
            "promotion_status": "promoted",
            "selected_checkpoint_sha256": sha256_path(stage4_checkpoint)["sha256"],
            "reliability": reliability,
            "cmc_fused_selection": {
                "balanced_macro_mrr": 0.64,
                "regret": 0.0,
                "alpha_zero_exact": True,
                "alpha_one_exact": True,
                "zero_history_exact": True,
                "by_benchmark": {
                    benchmark: {"fused_mrr": 0.64}
                    for benchmark in (
                        "toolbench_g3",
                        "traject_bench",
                        "alfworld",
                        "webshop",
                    )
                },
            },
            "reliability_validation": {
                "balanced_macro_mrr": 0.64,
                "zero_history_exact": True,
            },
        }
    )
    selection.pop("manifest_sha256", None)
    selection["manifest_sha256"] = canonical_digest(selection)
    _write_json(paths["stage4_selection"], selection)
    lineage = json.loads(paths["stage4_lineage"].read_text(encoding="utf-8"))
    lineage["checkpoint"] = sha256_path(stage4_checkpoint)
    lineage["stage_metadata"]["stage4_selection"] = sha256_path(
        paths["stage4_selection"]
    )
    lineage["stage_metadata"]["reliability"] = reliability
    _write_json(paths["stage4_lineage"], lineage)


def _write_promoted_gate_report(
    tmp_path: Path,
    *,
    selected_stage4_checkpoint_sha256: str,
    promoted: bool = True,
) -> Path:
    gate = MemoryUtilityGate()
    gate_path = tmp_path / "direct_utility" / "memory_utility_gate.pt"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "clstr_memory_utility_gate",
            "schema_version": "memory_utility_gate_checkpoint_v2",
            "feature_schema": "memory_utility_features_v1",
            "feature_names": [],
            "gate_output_semantics": "anchored_harm_suppression_alpha",
            "alpha_base": 0.85,
            "audit_manifest_sha256": "a" * 64,
            "zero_history_fallback": "exact_static",
            "reliability_changes_memory_state": False,
            "selected_stage4_checkpoint_sha256": selected_stage4_checkpoint_sha256,
            "training_seed": 17,
            "trainable_parameter_names": ["net.0.weight", "net.0.bias"],
            "objective": {
                "temperature": 0.5,
                "fused_rank_weight": 1.0,
                "static_no_regret_weight": 1.0,
                "direct_gate_bce_weight": 1.0,
                "source_balanced": True,
                "harm_sign_balanced": True,
            },
            "feature_update_count_cap": 4.0,
            "feature_candidate_count_cap": 64.0,
            "harm_probability_gate_state_dict": gate.state_dict(),
            "validation": {},
        },
        gate_path,
    )
    payload = {
        "schema_version": "memory_utility_gate_report_v2",
        "status": "ok",
        "promoted": promoted,
        "checkpoint_path": str(gate_path.resolve()) if promoted else None,
        "checkpoint_sha256": sha256_path(gate_path)["sha256"] if promoted else None,
        "audit_manifest_sha256": "a" * 64,
        "selected_stage4_checkpoint_sha256": selected_stage4_checkpoint_sha256,
        "gate_output_semantics": "anchored_harm_suppression_alpha",
        "alpha_base": 0.85,
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "trainable_parameter_names": ["net.0.weight", "net.0.bias"],
        "objective": {
            "temperature": 0.5,
            "fused_rank_weight": 1.0,
            "static_no_regret_weight": 1.0,
            "direct_gate_bce_weight": 1.0,
            "source_balanced": True,
            "harm_sign_balanced": True,
        },
        "validation": {"promotion_checks": {"all": promoted}},
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    report_path = tmp_path / "direct_utility" / "gate_report.json"
    _write_json(report_path, payload)
    return report_path


def test_resolve_qwen_clstr_final_chain_applies_promoted_cmc_gate_overlay(
    tmp_path: Path,
) -> None:
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    _convert_valid_run_to_cmc(paths)
    base = resolve_qwen_clstr_final_chain(paths["run_root"])
    gate_report_path = _write_promoted_gate_report(
        tmp_path,
        selected_stage4_checkpoint_sha256=sha256_path(paths["stage4_checkpoint"])[
            "sha256"
        ],
    )

    manifest = resolve_qwen_clstr_final_chain(
        paths["run_root"],
        reliability_gate_report_path=gate_report_path,
    )

    assert manifest["checkpoint_chain_digest"] == base["checkpoint_chain_digest"]
    assert manifest["manifest_sha256"] != base["manifest_sha256"]
    assert manifest["reliability"]["mode"] == "cmc_candidate_gate"
    assert manifest["reliability"]["audit_manifest_sha256"] == "a" * 64
    assert manifest["reliability"]["base_gate_source"] == "stage4_checkpoint"
    assert manifest["reliability"]["deployed_gate_source"] == "direct_utility_overlay"
    assert manifest["reliability"]["gate_output_semantics"] == "anchored_harm_suppression_alpha"
    assert manifest["reliability"]["alpha_base"] == 0.85
    assert manifest["reliability"]["gate_report"] == sha256_path(gate_report_path)


def test_resolve_qwen_clstr_final_chain_applies_candidate_provenance_overlay(
    tmp_path: Path,
) -> None:
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    _convert_valid_run_to_cmc(paths)
    base = resolve_qwen_clstr_final_chain(paths["run_root"])

    manifest = resolve_qwen_clstr_final_chain(
        paths["run_root"],
        candidate_provenance_overlay=True,
    )

    reliability = manifest["reliability"]
    assert manifest["checkpoint_chain_digest"] == base["checkpoint_chain_digest"]
    assert manifest["manifest_sha256"] != base["manifest_sha256"]
    assert reliability["mode"] == "cmc_candidate_provenance"
    assert reliability["fixed_alpha"] is None
    assert reliability["gate_checkpoint"] is None
    assert reliability["safe_memory_residual_bound"] == 2.0
    assert reliability["feature_update_count_cap"] == 16.0
    assert reliability["feature_candidate_count_cap"] == 256.0
    assert reliability["selected_stage4_checkpoint_sha256"] == sha256_path(
        paths["stage4_checkpoint"]
    )["sha256"]
    assert reliability["base_reliability_source"] == "stage4_checkpoint"
    assert reliability["deployed_reliability_source"] == (
        "candidate_provenance_positive_residual"
    )
    digest_payload = dict(reliability)
    assert digest_payload.pop("reliability_sha256") == canonical_digest(digest_payload)


def test_resolve_qwen_clstr_final_chain_rejects_unpromoted_cmc_gate_overlay(
    tmp_path: Path,
) -> None:
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    _convert_valid_run_to_cmc(paths)
    gate_report_path = _write_promoted_gate_report(
        tmp_path,
        selected_stage4_checkpoint_sha256=sha256_path(paths["stage4_checkpoint"])[
            "sha256"
        ],
        promoted=False,
    )

    with pytest.raises(ValueError, match="direct utility gate report is not promoted"):
        resolve_qwen_clstr_final_chain(
            paths["run_root"],
            reliability_gate_report_path=gate_report_path,
        )


def test_resolve_qwen_clstr_final_chain_accepts_one_linked_frozen_chain(tmp_path):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    manifest = resolve_qwen_clstr_final_chain(paths["run_root"])

    assert manifest["schema_version"] == "qwen06_clstr_final_chain_v2"
    assert manifest["status"] == "ok"
    assert manifest["checkpoint_chain_digest"]
    assert manifest["final_checkpoint_role"] == "stage4"
    assert manifest["checkpoints"]["stage4"]["path"] == str(paths["stage4_checkpoint"].resolve())
    assert manifest["stage4_selection"] == sha256_path(paths["stage4_selection"])
    assert manifest["reliability"]["mode"] == "fixed_alpha"
    assert manifest["reliability"]["fixed_alpha"] == pytest.approx(0.5)
    assert manifest["backbone"]["frozen"] is True


def test_resolve_qwen_clstr_final_chain_accepts_real_safe_stage4_gate_shape(
    tmp_path,
):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    gate_path = paths["run_root"] / "stage4_safe_full" / "stage4_quality_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    validation_path = paths["stage4_validation_report"]
    selection = json.loads(paths["stage4_selection"].read_text(encoding="utf-8"))
    gate.pop("selected_validation_report_path")
    gate.pop("router_integrity")
    gate["selected_validation_report"] = sha256_path(validation_path)
    gate["stage4_selection"] = selection
    _write_json(gate_path, gate)

    manifest = resolve_qwen_clstr_final_chain(paths["run_root"])

    assert manifest["status"] == "ok"
    assert manifest["stage4_selected_validation_report"] == sha256_path(
        validation_path
    )


def test_resolve_qwen_clstr_final_chain_rejects_stage4_selection_self_hash_drift(
    tmp_path,
):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    selection = json.loads(paths["stage4_selection"].read_text(encoding="utf-8"))
    selection["reliability"]["fixed_alpha"] = 0.75
    _write_json(paths["stage4_selection"], selection)

    with pytest.raises(ValueError, match="Stage4 selection self-hash mismatch"):
        resolve_qwen_clstr_final_chain(paths["run_root"])


def test_resolve_qwen_clstr_final_chain_rejects_bad_quality_gate(tmp_path):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    gate_path = paths["run_root"] / "stage2_full" / "stage2_quality_gate.json"
    _write_json(gate_path, {"status": "action_required"})

    with pytest.raises(ValueError, match="Stage2 quality gate is not ok"):
        resolve_qwen_clstr_final_chain(paths["run_root"])


def test_resolve_qwen_clstr_final_chain_rejects_checkpoint_drift(tmp_path):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    paths["stage2_checkpoint"].write_bytes(b"changed-after-lineage")

    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        resolve_qwen_clstr_final_chain(paths["run_root"])


def test_resolve_qwen_clstr_final_chain_rejects_nonfrozen_backbone(tmp_path):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    lineage = json.loads(paths["stage0_lineage"].read_text(encoding="utf-8"))
    lineage["backbone"]["frozen"] = False
    _write_json(paths["stage0_lineage"], lineage)

    with pytest.raises(ValueError, match="Qwen backbone must remain frozen"):
        resolve_qwen_clstr_final_chain(paths["run_root"])


def test_resolve_qwen_clstr_final_chain_rejects_wrong_stage4_identity_parent(tmp_path):
    from clstr.qwen_clstr_final_chain import resolve_qwen_clstr_final_chain

    paths = _build_valid_run(tmp_path)
    lineage = json.loads(paths["stage4_lineage"].read_text(encoding="utf-8"))
    lineage["identity_parent_role"] = "stage0"
    _write_json(paths["stage4_lineage"], lineage)

    with pytest.raises(ValueError, match="Stage4 identity parent must be stage2"):
        resolve_qwen_clstr_final_chain(paths["run_root"])


def test_resolve_qwen_clstr_final_chain_cli_writes_manifest(tmp_path):
    paths = _build_valid_run(tmp_path)
    output_path = tmp_path / "final_chain.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/resolve_qwen06_clstr_final_chain.py",
            "--run_root",
            str(paths["run_root"]),
            "--output_path",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["manifest_sha256"]


def test_resolve_qwen_clstr_final_chain_cli_accepts_promoted_gate_report(tmp_path):
    paths = _build_valid_run(tmp_path)
    _convert_valid_run_to_cmc(paths)
    gate_report_path = _write_promoted_gate_report(
        tmp_path,
        selected_stage4_checkpoint_sha256=sha256_path(paths["stage4_checkpoint"])[
            "sha256"
        ],
    )
    output_path = tmp_path / "overlay_final_chain.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/resolve_qwen06_clstr_final_chain.py",
            "--run_root",
            str(paths["run_root"]),
            "--reliability_gate_report_path",
            str(gate_report_path),
            "--output_path",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reliability"]["deployed_gate_source"] == (
        "direct_utility_overlay"
    )


def test_resolve_qwen_clstr_final_chain_cli_accepts_candidate_provenance_overlay(
    tmp_path,
):
    paths = _build_valid_run(tmp_path)
    _convert_valid_run_to_cmc(paths)
    output_path = tmp_path / "candidate_provenance_final_chain.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/resolve_qwen06_clstr_final_chain.py",
            "--run_root",
            str(paths["run_root"]),
            "--candidate_provenance_overlay",
            "--output_path",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reliability"]["mode"] == "cmc_candidate_provenance"
    assert payload["reliability"]["gate_checkpoint"] is None
