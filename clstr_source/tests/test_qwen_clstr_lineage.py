from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clstr.qwen_clstr_lineage import (
    create_derived_lineage_manifest,
    create_lineage_manifest,
    validate_lineage_manifest,
    validate_qwen_clstr_checkpoint,
    validate_qwen_clstr_derived_checkpoint,
)


def _write_identity_inputs(tmp_path: Path, *, frozen: bool = True):
    model = tmp_path / "Qwen3-Embedding-0.6B"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
    pool = tmp_path / "skill_pool.jsonl"
    pool.write_text('{"skill_id":"skill/a"}\n', encoding="utf-8")
    data_manifest = tmp_path / "manifest.json"
    data_manifest.write_text('{"status":"ok","skill_count":1}\n', encoding="utf-8")
    checkpoint = tmp_path / "stage0.pt"
    torch.save(
        {
            "stage": "clstr_unified_retrieval_v2",
            "config": {
                "base_model_name": str(model),
                "freeze_backbone": frozen,
                "route_scorer": "unified_memory",
                "state_query_prompt_version": "clstr_causal_state_v1",
                "state_query_instruction": (
                    "Given an agent task or current execution state and interaction history, "
                    "retrieve the skill or tool document most useful for the next action."
                ),
                "state_query_max_chars": 2000,
                "state_query_truncation": "head_tail_v1",
            },
            "trainable_parameter_policy": {
                "trains_encoder_backbone": not frozen,
            },
            "model_state_dict": {
                "skill_table.E": torch.zeros(1, 4),
            },
        },
        checkpoint,
    )
    return model, pool, data_manifest, checkpoint


def test_lineage_manifest_records_qwen_model_pool_data_and_checkpoint_hashes(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)

    manifest = create_lineage_manifest(
        stage="stage0",
        checkpoint_path=checkpoint,
        expected_checkpoint_stage="clstr_unified_retrieval_v2",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={},
    )

    assert manifest["schema_version"] == "qwen06_clstr_lineage_v1"
    assert manifest["backbone"]["frozen"] is True
    assert manifest["route_scorer"] == "unified_memory"
    assert manifest["state_query"]["prompt_version"] == "clstr_causal_state_v1"
    assert manifest["checkpoint"]["sha256"]


def test_lineage_rejects_trainable_backbone_checkpoint(tmp_path):
    model, _pool, _data_manifest, checkpoint = _write_identity_inputs(
        tmp_path,
        frozen=False,
    )

    with pytest.raises(ValueError, match="backbone must remain frozen"):
        validate_qwen_clstr_checkpoint(
            checkpoint,
            expected_stage="clstr_unified_retrieval_v2",
            expected_model_path=model,
        )


def test_lineage_rejects_skill_pool_mismatch(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    pool.write_text('{"skill_id":"skill/changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="skill pool digest mismatch"):
        validate_lineage_manifest(
            manifest_path,
            expected_stage="stage0",
            expected_parent_manifests={},
        )


def test_lineage_rejects_expected_input_path_mismatch_before_downstream_training(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path / "recorded")
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    _other_model, other_pool, _other_data, _other_checkpoint = _write_identity_inputs(
        tmp_path / "other"
    )

    with pytest.raises(ValueError, match="expected skill pool path mismatch"):
        validate_lineage_manifest(
            manifest_path,
            expected_stage="stage0",
            expected_parent_manifests={},
            expected_model_path=model,
            expected_skill_pool_path=other_pool,
            expected_data_manifest_path=data_manifest,
        )


def test_lineage_rejects_prompt_contract_mismatch(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu")
    payload["config"]["state_query_max_chars"] = 1500
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="state query prompt contract mismatch"):
        create_lineage_manifest(
            stage="stage0",
            checkpoint_path=checkpoint,
            expected_checkpoint_stage="clstr_unified_retrieval_v2",
            model_path=model,
            skill_pool_path=pool,
            data_manifest_path=data_manifest,
            parent_manifests={},
        )


def test_lineage_rejects_changed_checkpoint_digest(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    manifest_path = tmp_path / "lineage.json"
    manifest_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    payload = torch.load(checkpoint, map_location="cpu")
    payload["model_state_dict"]["skill_table.E"] = torch.ones(1, 4)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        validate_lineage_manifest(
            manifest_path,
            expected_stage="stage0",
            expected_parent_manifests={},
        )


def test_lineage_rejects_wrong_parent_role_and_unsupported_schema(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    child_path = tmp_path / "child.json"
    child = create_lineage_manifest(
        stage="stage1",
        checkpoint_path=checkpoint,
        expected_checkpoint_stage="clstr_unified_retrieval_v2",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={"stage0": parent_path},
    )
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ValueError, match="parent lineage roles mismatch"):
        validate_lineage_manifest(
            child_path,
            expected_stage="stage1",
            expected_parent_manifests={"wrong_role": parent_path},
        )

    child["schema_version"] = "unsupported"
    child_path.write_text(json.dumps(child), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported lineage schema version"):
        validate_lineage_manifest(
            child_path,
            expected_stage="stage1",
            expected_parent_manifests={"stage0": parent_path},
        )


def test_lineage_revalidates_parent_checkpoint_file(tmp_path):
    parent_model, parent_pool, parent_data, parent_checkpoint = _write_identity_inputs(
        tmp_path / "parent"
    )
    child_model, child_pool, child_data, child_checkpoint = _write_identity_inputs(
        tmp_path / "child"
    )
    parent_path = tmp_path / "parent_lineage.json"
    parent_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=parent_checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=parent_model,
                skill_pool_path=parent_pool,
                data_manifest_path=parent_data,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    child_path = tmp_path / "child_lineage.json"
    child_path.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage1",
                checkpoint_path=child_checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=child_model,
                skill_pool_path=child_pool,
                data_manifest_path=child_data,
                parent_manifests={"stage0": parent_path},
            )
        ),
        encoding="utf-8",
    )
    payload = torch.load(parent_checkpoint, map_location="cpu")
    payload["model_state_dict"]["skill_table.E"] = torch.ones(1, 4)
    torch.save(payload, parent_checkpoint)

    with pytest.raises(ValueError, match="parent checkpoint file digest mismatch"):
        validate_lineage_manifest(
            child_path,
            expected_stage="stage1",
            expected_parent_manifests={"stage0": parent_path},
        )


def test_lineage_cli_creates_and_validates_manifest(tmp_path):
    model, pool, data_manifest, checkpoint = _write_identity_inputs(tmp_path)
    output_path = tmp_path / "lineage.json"
    script = Path("scripts/audit_qwen06_clstr_lineage.py")

    created = subprocess.run(
        [
            sys.executable,
            str(script),
            "create",
            "--stage",
            "stage0",
            "--checkpoint_path",
            str(checkpoint),
            "--expected_checkpoint_stage",
            "clstr_unified_retrieval_v2",
            "--model_path",
            str(model),
            "--skill_pool_path",
            str(pool),
            "--data_manifest_path",
            str(data_manifest),
            "--output_path",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["stage"] == "stage0"

    validated = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate",
            "--manifest_path",
            str(output_path),
            "--expected_stage",
            "stage0",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["status"] == "ok"


def _write_stage0_and_stage2_parent_lineages(tmp_path: Path):
    model, pool, data_manifest, stage0_checkpoint = _write_identity_inputs(tmp_path)
    stage0_lineage = tmp_path / "stage0_lineage.json"
    stage0_lineage.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage0",
                checkpoint_path=stage0_checkpoint,
                expected_checkpoint_stage="clstr_unified_retrieval_v2",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={},
            )
        ),
        encoding="utf-8",
    )
    stage2_checkpoint = tmp_path / "stage2.pt"
    stage2_payload = torch.load(stage0_checkpoint, map_location="cpu")
    stage2_payload["stage"] = "clstr_full_base_component_complete"
    torch.save(stage2_payload, stage2_checkpoint)
    stage2_lineage = tmp_path / "stage2_lineage.json"
    stage2_lineage.write_text(
        json.dumps(
            create_lineage_manifest(
                stage="stage2",
                checkpoint_path=stage2_checkpoint,
                expected_checkpoint_stage="clstr_full_base_component_complete",
                model_path=model,
                skill_pool_path=pool,
                data_manifest_path=data_manifest,
                parent_manifests={"stage0": stage0_lineage},
            )
        ),
        encoding="utf-8",
    )
    return model, pool, data_manifest, stage0_lineage, stage2_lineage


def test_derived_lineage_inherits_validated_parent_identity_for_configless_stage4(tmp_path):
    model, pool, data_manifest, stage0_lineage, stage2_lineage = (
        _write_stage0_and_stage2_parent_lineages(tmp_path)
    )
    stage4_checkpoint = tmp_path / "stage4.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
            },
            "train_report": {
                "status": "ok",
                "safe_memory_protocol_version": "stage4_safe_memory_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        stage4_checkpoint,
    )

    manifest = create_derived_lineage_manifest(
        stage="stage4",
        checkpoint_path=stage4_checkpoint,
        expected_checkpoint_stage="clstr_stage4_transition_conditioned_next_skill",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={"stage0": stage0_lineage, "stage2": stage2_lineage},
        identity_parent_role="stage2",
        stage_metadata={
            "stage4_selection": {"sha256": "selection-sha256"},
            "router_integrity": {"full_router_digest": "router-digest"},
            "reliability": {"mode": "fixed_alpha", "fixed_alpha": 0.5},
        },
    )

    stage2_parent = json.loads(stage2_lineage.read_text(encoding="utf-8"))
    assert manifest["stage"] == "stage4"
    assert manifest["route_scorer"] == "unified_memory"
    assert manifest["backbone"] == stage2_parent["backbone"]
    assert manifest["state_query"] == stage2_parent["state_query"]
    assert set(manifest["parents"]) == {"stage0", "stage2"}
    assert manifest["stage_metadata"]["stage4_selection"]["sha256"] == (
        "selection-sha256"
    )


def test_derived_lineage_rejects_frozen_routing_state_in_stage4_checkpoint(tmp_path):
    checkpoint = tmp_path / "stage4.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
                "encoder.proj.weight": torch.zeros(1, 1),
            },
            "train_report": {
                "status": "ok",
                "safe_memory_protocol_version": "stage4_safe_memory_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="forbidden Stage4 delta key"):
        validate_qwen_clstr_derived_checkpoint(
            checkpoint,
            expected_stage="clstr_stage4_transition_conditioned_next_skill",
        )


def test_selected_safe_stage4_lineage_accepts_running_intermediate_checkpoint(
    tmp_path,
):
    model, pool, data_manifest, stage0_lineage, stage2_lineage = (
        _write_stage0_and_stage2_parent_lineages(tmp_path)
    )
    checkpoint = tmp_path / "stage4-selected.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
            },
            "train_report": {
                "status": "running",
                "safe_memory_protocol_version": "stage4_safe_memory_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        checkpoint,
    )

    manifest = create_derived_lineage_manifest(
        stage="stage4",
        checkpoint_path=checkpoint,
        expected_checkpoint_stage="clstr_stage4_transition_conditioned_next_skill",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={"stage0": stage0_lineage, "stage2": stage2_lineage},
        identity_parent_role="stage2",
        stage_metadata={
            "stage4_selection": {"sha256": "selection-sha256"},
            "router_integrity": {"full_router_digest": "router-digest"},
            "reliability": {"mode": "fixed_alpha", "fixed_alpha": 1.0},
        },
    )

    assert manifest["checkpoint"]["path"] == str(checkpoint.resolve())


@pytest.mark.parametrize(
    "router_key",
    ["initial_belief_head.weight", "unified_retriever.weight"],
)
def test_safe_stage4_lineage_rejects_router_heads(
    tmp_path: Path,
    router_key: str,
) -> None:
    checkpoint = tmp_path / "stage4-safe.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
                router_key: torch.zeros(1, 1),
            },
            "train_report": {
                "status": "ok",
                "safe_memory_protocol_version": "stage4_safe_memory_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="forbidden Stage4 delta key"):
        validate_qwen_clstr_derived_checkpoint(
            checkpoint,
            expected_stage="clstr_stage4_transition_conditioned_next_skill",
        )


def test_safe_stage4_lineage_requires_validation_selected_protocol(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage4-legacy.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
            },
            "train_report": {
                "status": "ok",
                "valid_or_test_used_for_training": False,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="safe-memory protocol"):
        validate_qwen_clstr_derived_checkpoint(
            checkpoint,
            expected_stage="clstr_stage4_transition_conditioned_next_skill",
        )


def test_cmc_stage4_lineage_accepts_exact_delta_and_records_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage4-cmc.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": "counterfactual_memory_calibration_v1",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "route_memory_residual_adapter.net.3.weight": torch.zeros(1, 1),
                "route_memory_candidate_utility_gate.net.0.weight": torch.zeros(1, 1),
            },
            "parent_stage2_full_router_digest": "full-router-digest",
            "parent_stage2_fast_router_digest": "fast-router-digest",
            "train_report": {
                "status": "ok",
                "stage4_method": "counterfactual_memory_calibration_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "feature_schema": "memory_utility_features_v1",
                "feature_update_count_cap": 16.0,
                "feature_candidate_count_cap": 256.0,
                "freeze_report": {
                    "frozen_routing_foundation": True,
                    "parent_stage2_full_router_digest": "full-router-digest",
                    "parent_stage2_fast_router_digest": "fast-router-digest",
                },
            },
        },
        checkpoint,
    )

    report = validate_qwen_clstr_derived_checkpoint(
        checkpoint,
        expected_stage="clstr_stage4_transition_conditioned_next_skill",
    )

    assert report["stage4_method"] == "counterfactual_memory_calibration_v1"
    assert report["feature_schema"] == "memory_utility_features_v1"
    assert report["feature_update_count_cap"] == 16.0
    assert report["feature_candidate_count_cap"] == 256.0
    assert report["parent_stage2_full_router_digest"] == "full-router-digest"
    assert report["parent_stage2_fast_router_digest"] == "fast-router-digest"


def test_cmc_derived_lineage_pins_training_and_parent_identity(tmp_path: Path) -> None:
    model, pool, data_manifest, stage0_lineage, stage2_lineage = (
        _write_stage0_and_stage2_parent_lineages(tmp_path)
    )
    checkpoint = tmp_path / "stage4-cmc.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "stage4_method": "counterfactual_memory_calibration_v1",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "route_memory_residual_adapter.net.3.weight": torch.zeros(1, 1),
                "route_memory_candidate_utility_gate.net.0.weight": torch.zeros(1, 1),
            },
            "parent_stage2_full_router_digest": "full-router-digest",
            "parent_stage2_fast_router_digest": "fast-router-digest",
            "train_report": {
                "status": "ok",
                "stage4_method": "counterfactual_memory_calibration_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "feature_schema": "memory_utility_features_v1",
                "feature_update_count_cap": 16.0,
                "feature_candidate_count_cap": 256.0,
                "freeze_report": {
                    "frozen_routing_foundation": True,
                    "parent_stage2_full_router_digest": "full-router-digest",
                    "parent_stage2_fast_router_digest": "fast-router-digest",
                },
            },
        },
        checkpoint,
    )

    manifest = create_derived_lineage_manifest(
        stage="stage4",
        checkpoint_path=checkpoint,
        expected_checkpoint_stage="clstr_stage4_transition_conditioned_next_skill",
        model_path=model,
        skill_pool_path=pool,
        data_manifest_path=data_manifest,
        parent_manifests={"stage0": stage0_lineage, "stage2": stage2_lineage},
        identity_parent_role="stage2",
        stage_metadata={"stage4_selection": {"sha256": "selection-sha256"}},
    )

    assert manifest["stage_metadata"]["cmc"] == {
        "stage4_method": "counterfactual_memory_calibration_v1",
        "feature_schema": "memory_utility_features_v1",
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
        "parent_stage2_full_router_digest": "full-router-digest",
        "parent_stage2_fast_router_digest": "fast-router-digest",
    }


def test_lineage_cli_creates_derived_stage4_manifest(tmp_path):
    model, pool, data_manifest, stage0_lineage, stage2_lineage = (
        _write_stage0_and_stage2_parent_lineages(tmp_path)
    )
    stage4_checkpoint = tmp_path / "stage4.pt"
    torch.save(
        {
            "stage": "clstr_stage4_transition_conditioned_next_skill",
            "checkpoint_excludes_frozen_qwen_backbone": True,
            "checkpoint_excludes_frozen_routing_foundation": True,
            "model_state_dict": {
                "transition.weight": torch.zeros(1, 1),
                "gate.weight": torch.zeros(1, 1),
                "action_proj.weight": torch.zeros(1, 1),
            },
            "train_report": {
                "status": "ok",
                "safe_memory_protocol_version": "stage4_safe_memory_v1",
                "valid_or_test_used_for_training": False,
                "validation_used_for_checkpoint_selection": True,
                "route_scorer": "unified_memory",
                "freeze_report": {"frozen_routing_foundation": True},
            },
        },
        stage4_checkpoint,
    )
    output_path = tmp_path / "stage4_lineage.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_qwen06_clstr_lineage.py",
            "create-derived",
            "--stage",
            "stage4",
            "--checkpoint_path",
            str(stage4_checkpoint),
            "--expected_checkpoint_stage",
            "clstr_stage4_transition_conditioned_next_skill",
            "--model_path",
            str(model),
            "--skill_pool_path",
            str(pool),
            "--data_manifest_path",
            str(data_manifest),
            "--parent",
            f"stage0={stage0_lineage}",
            "--parent",
            f"stage2={stage2_lineage}",
            "--identity_parent_role",
            "stage2",
            "--output_path",
            str(output_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "stage4"
    assert manifest["identity_parent_role"] == "stage2"

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/audit_qwen06_clstr_lineage.py",
            "validate",
            "--manifest_path",
            str(output_path),
            "--expected_stage",
            "stage4",
            "--expected_parent",
            f"stage0={stage0_lineage}",
            "--expected_parent",
            f"stage2={stage2_lineage}",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["status"] == "ok"
