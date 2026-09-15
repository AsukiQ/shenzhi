from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_clstr_vnext_full_chain import finalize_full_chain


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _chain_fixture(tmp_path: Path) -> dict[str, Path]:
    run_root = tmp_path / "run"
    precompute_root = run_root / "precompute"
    stage0_root = run_root / "stage0"
    candidate_root = run_root / "candidate_compressor"
    stage2_root = run_root / "stage2"
    for path in (precompute_root, stage0_root, candidate_root, stage2_root):
        path.mkdir(parents=True)

    data_manifest = tmp_path / "data_manifest.json"
    _write_json(data_manifest, {"status": "ok", "files": {}})
    data_sha = _sha256(data_manifest)
    stage0_dev_checkpoint = (
        stage0_root / "checkpoints" / "clstr_vnext_stage0-step4500.pt"
    )
    stage0_checkpoint = (
        stage0_root / "checkpoints" / "clstr_vnext_stage0-step5000.pt"
    )
    candidate_checkpoint = candidate_root / "checkpoints" / "candidate.pt"
    stage2_checkpoint = stage2_root / "checkpoints" / "stage2.pt"
    skills = stage0_root / "selected_skills.jsonl"
    stage0_checkpoint.parent.mkdir()
    candidate_checkpoint.parent.mkdir()
    stage2_checkpoint.parent.mkdir()
    stage0_dev_checkpoint.write_bytes(b"stage0-dev-checkpoint")
    stage0_checkpoint.write_bytes(b"stage0-checkpoint")
    candidate_checkpoint.write_bytes(b"candidate-checkpoint")
    stage2_checkpoint.write_bytes(b"stage2-checkpoint")
    skills.write_text('{"skill_id":"a"}\n', encoding="utf-8")

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_paths = {}
    for role in ("skill", "state", "action", "result"):
        path = cache_root / f"{role}.json"
        _write_json(path, {"role": role})
        cache_paths[role] = path.resolve()

    skill_cache = {
        "cache_identity": "skill-cache-v1",
        "manifest_path": str(cache_paths["skill"]),
        "manifest_sha256": _sha256(cache_paths["skill"]),
    }

    def text_cache(role: str) -> dict:
        return {
            "cache_identity": "encoder-cache-v1",
            "cache_tensor_path": str(cache_paths[role]),
        }

    backbone_contract = {
        "contract_digest": "backbone-v1",
        "model_root": "/model",
        "files": [{"path": "model.safetensors"}],
    }
    backbone = {
        "contract_digest": "backbone-v1",
        "contract": backbone_contract,
    }
    data_contract = {"contract_sha256": data_sha}
    stage0_selection = {
        "status": "ok",
        "selected_step": 4500,
        "selected_checkpoint_path": str(stage0_dev_checkpoint.resolve()),
    }
    candidate_digest = "candidate-foundation-v1"
    stage2_initial_candidate_digest = "stage2-initial-route-state-v1"
    static_route_digest = "static-route-foundation-v1"
    candidate_selection = {
        "status": "ok",
        "selected_step": 120,
        "selected_checkpoint_path": str(candidate_checkpoint.resolve()),
        "selected_checkpoint_sha256": _sha256(candidate_checkpoint),
        "parent_stage0_checkpoint_sha256": _sha256(stage0_checkpoint),
        "selected_candidate_foundation_digest": candidate_digest,
        "objective_mode": "static_route_query_residual",
        "candidate_protocol": "natural_top500_static_route_query_residual_v1",
    }
    stage2_selection = {
        "status": "ok",
        "selected_step": 200,
        "selected_checkpoint_path": str(stage2_checkpoint.resolve()),
        "selection_reason": "first_gate_passing_minimal_intervention",
    }
    stage0_run_contract = {
        "inputs": {
            "selected_skills_sha256": _sha256(skills),
            "data_contract_sha256": data_sha,
            "frozen_backbone_snapshot_contract": backbone_contract,
        },
        "frozen_cache_identity": "encoder-cache-v1",
        "skill_cache_identity": "skill-cache-v1",
    }
    stage2_run_contract = {
        "inputs": {
            "stage0_checkpoint_sha256": _sha256(stage0_checkpoint),
            "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
            "candidate_foundation_digest": stage2_initial_candidate_digest,
            "static_route_foundation_digest": static_route_digest,
            "skills_sha256": _sha256(skills),
            "data_contract_sha256": data_sha,
            "frozen_backbone_snapshot_contract": backbone_contract,
        },
        "frozen_cache_identities": {
            "state": "encoder-cache-v1",
            "action": "encoder-cache-v1",
            "result": "encoder-cache-v1",
        },
        "optimization": {
            "horizon_protocol": "family_first_supported_full_bptt_le16_v1",
        },
    }
    precompute_report = {
        "status": "ok",
        "stage": "clstr_vnext_precompute",
        "data_contract": data_contract,
        "frozen_backbone_snapshot": backbone,
        "skill_cache": skill_cache,
        "state_cache": text_cache("state"),
        "action_cache": text_cache("action"),
        "result_cache": text_cache("result"),
    }
    stage0_report = {
        "status": "ok",
        "stage": "clstr_vnext_stage0",
        "step": 5000,
        "checkpoint_path": str(stage0_checkpoint.resolve()),
        "selection": stage0_selection,
        "data_contract": data_contract,
        "frozen_backbone_snapshot": backbone,
        "skill_cache": skill_cache,
        "cache": text_cache("state"),
        "run_contract": stage0_run_contract,
    }
    candidate_run_contract = {
        "inputs": {
            "stage0_checkpoint_sha256": _sha256(stage0_checkpoint),
            "skills_sha256": _sha256(skills),
            "data_contract_sha256": data_sha,
            "frozen_backbone_snapshot_contract": backbone_contract,
        },
        "optimization": {
            "candidate_protocol": "natural_top500_static_route_query_residual_v1"
        },
    }
    candidate_report = {
        "status": "ok",
        "stage": "clstr_vnext_candidate_compressor",
        "selection": candidate_selection,
        "data_contract": data_contract,
        "frozen_backbone_snapshot": backbone,
        "cache": text_cache("state"),
        "run_contract": candidate_run_contract,
        "parent_stage0_checkpoint_sha256": _sha256(stage0_checkpoint),
        "candidate_foundation_digest": candidate_digest,
        "objective_mode": "static_route_query_residual",
        "training_objective": "natural_top500_static_route_query_listwise_v1",
    }
    stage2_report = {
        "status": "ok",
        "stage": "clstr_vnext_stage2",
        "selection": stage2_selection,
        "data": {"data_contract": data_contract},
        "frozen_backbone_snapshot": backbone,
        "state_cache": text_cache("state"),
        "action_cache": text_cache("action"),
        "result_cache": text_cache("result"),
        "run_contract": stage2_run_contract,
        "candidate_foundation_digest": stage2_initial_candidate_digest,
        "static_route_foundation_digest": static_route_digest,
        "static_route_foundation_digest_final": static_route_digest,
        "sampling": {
            "horizon_protocol": "family_first_supported_full_bptt_le16_v1",
        },
    }
    _write_json(precompute_root / "precompute_report.json", precompute_report)
    _write_json(stage0_root / "stage0_selection.json", stage0_selection)
    _write_json(stage0_root / "train_report.json", stage0_report)
    _write_json(candidate_root / "compressor_selection.json", candidate_selection)
    _write_json(candidate_root / "train_report.json", candidate_report)
    _write_json(stage2_root / "stage2_selection.json", stage2_selection)
    _write_json(stage2_root / "train_report.json", stage2_report)
    _write_json(stage0_root / "source_manifest.json", {"status": "ok"})
    _write_json(candidate_root / "source_manifest.json", {"status": "ok"})
    _write_json(stage2_root / "source_manifest.json", {"status": "ok"})
    return {
        "run_root": run_root,
        "data_manifest": data_manifest,
        "stage2_report": stage2_root / "train_report.json",
        "cache_root": cache_root,
    }


def test_vnext_finalizer_accepts_consistent_static_rerank_chain(
    tmp_path: Path,
) -> None:
    paths = _chain_fixture(tmp_path)
    payload = finalize_full_chain(paths["run_root"], paths["data_manifest"])
    assert payload["status"] == "ok"
    assert payload["lineage"]["status"] == "ok"
    assert payload["stage0"]["selected_step"] == 5000
    assert payload["stage0"]["dev_selected_step"] == 4500
    assert payload["stage0"]["checkpoint_policy"] == "final_training_checkpoint_v1"
    assert payload["stage0"]["checkpoint_path"].endswith(
        "clstr_vnext_stage0-step5000.pt"
    )
    assert payload["lineage"]["frozen_backbone_contract_digest"] == "backbone-v1"
    assert payload["lineage"]["static_route_foundation_digest"] == (
        "static-route-foundation-v1"
    )
    assert payload["lineage"]["candidate_foundation_digest"] == (
        "candidate-foundation-v1"
    )
    assert payload["lineage"]["stage2_initial_candidate_foundation_digest"] == (
        "stage2-initial-route-state-v1"
    )
    assert payload["horizon_protocol"] == (
        "family_first_supported_full_bptt_le16_v1"
    )
    assert payload["canonical_stage_chain"] == [
        "precompute",
        "stage0_static",
        "static_route_query_adapter",
        "stage2_causal_memory",
    ]
    assert payload["static_reranker"]["component"] == "static_route_query_residual"
    assert paths["run_root"].joinpath("final_chain_manifest.json").is_file()


def test_vnext_finalizer_accepts_identity_only_resume_cache_metadata(
    tmp_path: Path,
) -> None:
    paths = _chain_fixture(tmp_path)
    report_path = paths["run_root"] / "stage0" / "train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["skill_cache"] = {
        "status": "restored_from_resume_checkpoint",
        "cache_identity": report["skill_cache"]["cache_identity"],
        "skill_count": 1,
    }
    _write_json(report_path, report)

    payload = finalize_full_chain(
        paths["run_root"],
        paths["data_manifest"],
        write_manifest=False,
    )
    assert payload["lineage"]["skill_cache"]["cache_identity"] == (
        "skill-cache-v1"
    )
    assert payload["lineage"]["skill_cache"]["manifest_path"].endswith(
        "skill.json"
    )


def test_vnext_finalizer_rejects_conflicting_optional_cache_path(
    tmp_path: Path,
) -> None:
    paths = _chain_fixture(tmp_path)
    stale = paths["cache_root"] / "skill-stale.json"
    _write_json(stale, {"role": "skill", "version": "stale"})
    report_path = paths["run_root"] / "stage0" / "train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["skill_cache"]["manifest_path"] = str(stale.resolve())
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="skill cache path"):
        finalize_full_chain(
            paths["run_root"],
            paths["data_manifest"],
            write_manifest=False,
        )


def test_vnext_finalizer_requires_canonical_nested_stage2_data_contract(
    tmp_path: Path,
) -> None:
    paths = _chain_fixture(tmp_path)
    report = json.loads(paths["stage2_report"].read_text(encoding="utf-8"))
    report["data_contract"] = report.pop("data")["data_contract"]
    _write_json(paths["stage2_report"], report)

    with pytest.raises(ValueError, match="nested data contract"):
        finalize_full_chain(
            paths["run_root"],
            paths["data_manifest"],
            write_manifest=False,
        )


@pytest.mark.parametrize(
    "drift",
    ["stage0_parent", "backbone", "state_cache", "static_route_foundation"],
)
def test_vnext_finalizer_rejects_mixed_lineage(
    tmp_path: Path,
    drift: str,
) -> None:
    paths = _chain_fixture(tmp_path)
    report = json.loads(paths["stage2_report"].read_text(encoding="utf-8"))
    if drift == "stage0_parent":
        report["run_contract"]["inputs"][
            "stage0_checkpoint_sha256"
        ] = "0" * 64
    elif drift == "backbone":
        changed = {
            "contract_digest": "backbone-v0",
            "model_root": "/old-model",
            "files": [{"path": "model.safetensors"}],
        }
        report["frozen_backbone_snapshot"] = {
            "contract_digest": "backbone-v0",
            "contract": changed,
        }
        report["run_contract"]["inputs"]["frozen_backbone_snapshot_contract"] = changed
    elif drift == "state_cache":
        stale_path = paths["cache_root"] / "state-old.json"
        _write_json(stale_path, {"role": "state", "version": "old"})
        report["state_cache"] = {
            "cache_identity": "encoder-cache-v0",
            "cache_tensor_path": str(stale_path.resolve()),
        }
        report["run_contract"]["frozen_cache_identities"]["state"] = (
            "encoder-cache-v0"
        )
    else:
        report["static_route_foundation_digest_final"] = "stale-route-foundation"
    _write_json(paths["stage2_report"], report)
    with pytest.raises(ValueError, match="lineage mismatch"):
        finalize_full_chain(
            paths["run_root"],
            paths["data_manifest"],
            write_manifest=False,
        )
