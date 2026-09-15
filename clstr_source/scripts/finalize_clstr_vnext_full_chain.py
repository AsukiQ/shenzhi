#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing final-chain artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"final-chain artifact is not a JSON object: {path}")
    return payload


def _require_equal(label: str, *values: Any) -> Any:
    if not values or any(value is None or value == "" for value in values):
        raise ValueError(f"final-chain lineage lacks {label}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"final-chain lineage mismatch: {label}")
    return first


def _resolved_report_path(value: Any, *, label: str) -> str:
    if not str(value or ""):
        raise ValueError(f"final-chain lineage lacks {label}")
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise ValueError(f"final-chain cache manifest is missing: {path}")
    return str(path)


def _cache_lineage(
    label: str,
    reports: list[dict[str, Any] | None],
    *,
    path_key: str,
) -> dict[str, Any]:
    if all(report is None for report in reports):
        return {"present": False}
    if any(not isinstance(report, dict) for report in reports):
        raise ValueError(f"final-chain lineage mismatch: {label} cache presence")
    typed_reports = [dict(report or {}) for report in reports]
    identity = _require_equal(
        f"{label} cache identity",
        *(str(report.get("cache_identity") or "") for report in typed_reports),
    )
    reported_paths = [
        _resolved_report_path(report.get(path_key), label=f"{label} cache path")
        for report in typed_reports
        if str(report.get(path_key) or "")
    ]
    if not reported_paths:
        raise ValueError(f"final-chain lineage lacks {label} cache path")
    manifest_path = _require_equal(
        f"{label} cache path",
        *reported_paths,
    )
    return {
        "present": True,
        "cache_identity": identity,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(Path(manifest_path)),
    }


def _snapshot_digest(report: dict[str, Any], *, label: str) -> str:
    snapshot = report.get("frozen_backbone_snapshot") or {}
    contract = snapshot.get("contract") or {}
    return str(
        _require_equal(
            f"{label} frozen-backbone contract digest",
            str(snapshot.get("contract_digest") or ""),
            str(contract.get("contract_digest") or ""),
        )
    )


def finalize_full_chain(
    run_root: str | Path,
    data_manifest: str | Path,
    *,
    write_manifest: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    data_manifest = Path(data_manifest).resolve()
    paths = {
        "precompute_report": run_root / "precompute" / "precompute_report.json",
        "stage0_report": run_root / "stage0" / "train_report.json",
        "stage0_selection": run_root / "stage0" / "stage0_selection.json",
        "candidate_report": run_root / "candidate_compressor" / "train_report.json",
        "candidate_selection": run_root / "candidate_compressor" / "compressor_selection.json",
        "stage2_report": run_root / "stage2" / "train_report.json",
        "stage2_selection": run_root / "stage2" / "stage2_selection.json",
        "stage0_source": run_root / "stage0" / "source_manifest.json",
        "candidate_source": run_root / "candidate_compressor" / "source_manifest.json",
        "stage2_source": run_root / "stage2" / "source_manifest.json",
    }
    precompute = load(paths["precompute_report"])
    stage0_report = load(paths["stage0_report"])
    stage0 = load(paths["stage0_selection"])
    candidate_report = load(paths["candidate_report"])
    candidate = load(paths["candidate_selection"])
    stage2_report = load(paths["stage2_report"])
    stage2 = load(paths["stage2_selection"])
    for path in (
        data_manifest,
        paths["stage0_source"],
        paths["candidate_source"],
        paths["stage2_source"],
    ):
        if not path.is_file():
            raise ValueError(f"final-chain input is missing: {path}")
    if precompute.get("status") != "ok" or precompute.get("stage") != "clstr_vnext_precompute":
        raise ValueError("precompute report has not completed the canonical stage")
    for label, report, selection, expected_stage in (
        ("Stage0", stage0_report, stage0, "clstr_vnext_stage0"),
        (
            "Static route adapter",
            candidate_report,
            candidate,
            "clstr_vnext_candidate_compressor",
        ),
        ("Stage2", stage2_report, stage2, "clstr_vnext_stage2"),
    ):
        if report.get("stage") != expected_stage:
            raise ValueError(f"{label} report is not from the canonical stage")
        if report.get("status") != selection.get("status"):
            raise ValueError(f"{label} report and selection statuses differ")
        if report.get("selection") != selection:
            raise ValueError(f"{label} train report does not embed its final selection")

    candidate_optimization = (
        (candidate_report.get("run_contract") or {}).get("optimization") or {}
    )
    _require_equal(
        "static route-adapter objective mode",
        candidate_report.get("objective_mode"),
        candidate.get("objective_mode"),
        "static_route_query_residual",
    )
    _require_equal(
        "static route-adapter candidate protocol",
        candidate_optimization.get("candidate_protocol"),
        candidate.get("candidate_protocol"),
        "natural_top500_static_route_query_residual_v1",
    )
    _require_equal(
        "static route-adapter training objective",
        candidate_report.get("training_objective"),
        "natural_top500_static_route_query_listwise_v1",
    )

    stage0_final_step = int(stage0_report.get("step") or 0)
    stage0_checkpoint = Path(str(stage0_report.get("checkpoint_path") or "")).resolve()
    if (
        stage0_final_step <= 0
        or stage0_checkpoint.name
        != f"clstr_vnext_stage0-step{stage0_final_step}.pt"
    ):
        raise ValueError("Stage0 train report does not name its final checkpoint")
    candidate_checkpoint = Path(str(candidate["selected_checkpoint_path"])).resolve()
    stage2_checkpoint = Path(str(stage2["selected_checkpoint_path"])).resolve()
    skills_path = (run_root / "stage0" / "selected_skills.jsonl").resolve()
    for path in (
        stage0_checkpoint,
        candidate_checkpoint,
        stage2_checkpoint,
        skills_path,
    ):
        if not path.is_file():
            raise ValueError(f"final-chain input is missing: {path}")

    precompute_backbone = _snapshot_digest(precompute, label="precompute")
    stage0_backbone = _snapshot_digest(stage0_report, label="Stage0")
    candidate_backbone = _snapshot_digest(candidate_report, label="static route adapter")
    stage2_backbone = _snapshot_digest(stage2_report, label="Stage2")
    backbone_digest = str(
        _require_equal(
            "precompute/Stage0/static-route-adapter/Stage2 frozen-backbone contract digest",
            precompute_backbone,
            stage0_backbone,
            candidate_backbone,
            stage2_backbone,
        )
    )
    stage0_inputs = ((stage0_report.get("run_contract") or {}).get("inputs") or {})
    candidate_inputs = ((candidate_report.get("run_contract") or {}).get("inputs") or {})
    stage2_inputs = ((stage2_report.get("run_contract") or {}).get("inputs") or {})
    stage2_data_contract = (
        (stage2_report.get("data") or {}).get("data_contract") or {}
    )
    if not isinstance(stage2_data_contract, dict) or not stage2_data_contract:
        raise ValueError("Stage2 train report lacks its nested data contract")
    for label, inputs, report in (
        ("Stage0", stage0_inputs, stage0_report),
        ("Static route adapter", candidate_inputs, candidate_report),
        ("Stage2", stage2_inputs, stage2_report),
    ):
        _require_equal(
            f"{label} report/run-contract frozen-backbone contract",
            inputs.get("frozen_backbone_snapshot_contract"),
            (report.get("frozen_backbone_snapshot") or {}).get("contract"),
        )

    skill_cache = _cache_lineage(
        "skill",
        [precompute.get("skill_cache"), stage0_report.get("skill_cache")],
        path_key="manifest_path",
    )
    reported_skill_manifest_digests = [
        str(report.get("manifest_sha256"))
        for report in (
            precompute.get("skill_cache") or {},
            stage0_report.get("skill_cache") or {},
        )
        if str(report.get("manifest_sha256") or "")
    ]
    _require_equal(
        "skill cache manifest digest",
        *reported_skill_manifest_digests,
        skill_cache.get("manifest_sha256"),
    )
    state_cache = _cache_lineage(
        "state",
        [
            precompute.get("state_cache"),
            stage0_report.get("cache"),
            candidate_report.get("cache"),
            stage2_report.get("state_cache"),
        ],
        path_key="cache_tensor_path",
    )
    action_cache = _cache_lineage(
        "action",
        [precompute.get("action_cache"), stage2_report.get("action_cache")],
        path_key="cache_tensor_path",
    )
    result_cache = _cache_lineage(
        "result",
        [precompute.get("result_cache"), stage2_report.get("result_cache")],
        path_key="cache_tensor_path",
    )
    _require_equal(
        "Stage0 run-contract skill cache identity",
        (stage0_report.get("run_contract") or {}).get("skill_cache_identity"),
        skill_cache.get("cache_identity"),
    )
    _require_equal(
        "Stage0 run-contract state cache identity",
        (stage0_report.get("run_contract") or {}).get("frozen_cache_identity"),
        state_cache.get("cache_identity"),
    )
    stage2_cache_contract = (
        (stage2_report.get("run_contract") or {}).get("frozen_cache_identities") or {}
    )
    for role, lineage in (
        ("state", state_cache),
        ("action", action_cache),
        ("result", result_cache),
    ):
        expected = None if not lineage.get("present") else lineage.get("cache_identity")
        if stage2_cache_contract.get(role) != expected:
            raise ValueError(f"final-chain lineage mismatch: Stage2 {role} cache contract")
    horizon_protocol_values = [
        str(
            ((stage2_report.get("run_contract") or {}).get("optimization") or {}).get(
                "horizon_protocol"
            )
            or ""
        )
    ]
    reported_sampling_horizon = str(
        (stage2_report.get("sampling") or {}).get("horizon_protocol") or ""
    )
    if reported_sampling_horizon:
        horizon_protocol_values.append(reported_sampling_horizon)
    horizon_protocol = _require_equal(
        "Stage2 horizon protocol",
        *horizon_protocol_values,
    )

    stage0_checkpoint_sha = sha256(stage0_checkpoint)
    candidate_checkpoint_sha = sha256(candidate_checkpoint)
    stage2_checkpoint_sha = sha256(stage2_checkpoint)
    skills_sha = sha256(skills_path)
    _require_equal(
        "static route-adapter parent Stage0 checkpoint digest",
        candidate_inputs.get("stage0_checkpoint_sha256"),
        candidate_report.get("parent_stage0_checkpoint_sha256"),
        candidate.get("parent_stage0_checkpoint_sha256"),
        stage0_checkpoint_sha,
    )
    _require_equal(
        "Stage2 parent static route-adapter checkpoint digest",
        stage2_inputs.get("candidate_checkpoint_sha256"),
        candidate.get("selected_checkpoint_sha256"),
        candidate_checkpoint_sha,
    )
    _require_equal(
        "Stage2 transitive parent Stage0 checkpoint digest",
        stage2_inputs.get("stage0_checkpoint_sha256"),
        stage0_checkpoint_sha,
    )
    candidate_digest = _require_equal(
        "static route-adapter candidate digest",
        candidate.get("selected_candidate_foundation_digest"),
        candidate_report.get("candidate_foundation_digest"),
    )
    stage2_initial_candidate_digest = _require_equal(
        "Stage2 initial route-state digest",
        stage2_inputs.get("candidate_foundation_digest"),
        stage2_report.get("candidate_foundation_digest"),
    )
    static_route_digest = _require_equal(
        "frozen Stage2 static route foundation digest",
        stage2_inputs.get("static_route_foundation_digest"),
        stage2_report.get("static_route_foundation_digest"),
        stage2_report.get("static_route_foundation_digest_final"),
    )
    _require_equal(
        "Stage0 selected skills digest",
        stage0_inputs.get("selected_skills_sha256"),
        candidate_inputs.get("skills_sha256"),
        stage2_inputs.get("skills_sha256"),
        skills_sha,
    )

    data_manifest_sha = sha256(data_manifest)
    _require_equal(
        "precompute/Stage0/static-route-adapter/Stage2 data-contract digest",
        (precompute.get("data_contract") or {}).get("contract_sha256"),
        (stage0_report.get("data_contract") or {}).get("contract_sha256"),
        stage0_inputs.get("data_contract_sha256"),
        candidate_inputs.get("data_contract_sha256"),
        stage2_data_contract.get("contract_sha256"),
        stage2_inputs.get("data_contract_sha256"),
        data_manifest_sha,
    )

    payload = {
        "schema_version": "clstr_vnext_full_chain_route_query_residual_v1",
        "status": (
            "ok"
            if stage0.get("status") == "ok"
            and candidate.get("status") == "ok"
            and stage2.get("status") == "ok"
            else "action_required"
        ),
        "data_manifest": {"path": str(data_manifest), "sha256": data_manifest_sha},
        "precompute_report": {
            "path": str(paths["precompute_report"].resolve()),
            "sha256": sha256(paths["precompute_report"]),
            "cache_identities": {
                "skill": skill_cache.get("cache_identity"),
                "state": state_cache.get("cache_identity"),
                "action": action_cache.get("cache_identity"),
                "result": result_cache.get("cache_identity"),
            },
            "frozen_backbone_snapshot": precompute["frozen_backbone_snapshot"],
        },
        "stage0": {
            "selected_step": int(stage0_final_step),
            "checkpoint_policy": "final_training_checkpoint_v1",
            "dev_selected_step": int(stage0["selected_step"]),
            "dev_selected_checkpoint_path": str(
                Path(str(stage0["selected_checkpoint_path"])).resolve()
            ),
            "checkpoint_path": str(stage0_checkpoint),
            "checkpoint_sha256": stage0_checkpoint_sha,
            "skills_path": str(skills_path),
            "skills_sha256": skills_sha,
            "selection_sha256": sha256(paths["stage0_selection"]),
            "train_report_sha256": sha256(paths["stage0_report"]),
            "source_manifest_sha256": sha256(paths["stage0_source"]),
        },
        "static_reranker": {
            "component": "static_route_query_residual",
            "selected_step": int(candidate["selected_step"]),
            "checkpoint_path": str(candidate_checkpoint),
            "checkpoint_sha256": candidate_checkpoint_sha,
            "selection_sha256": sha256(paths["candidate_selection"]),
            "train_report_sha256": sha256(paths["candidate_report"]),
            "source_manifest_sha256": sha256(paths["candidate_source"]),
            "objective_mode": candidate_report.get("objective_mode"),
        },
        "stage2": {
            "selected_step": int(stage2["selected_step"]),
            "checkpoint_path": str(stage2_checkpoint),
            "checkpoint_sha256": stage2_checkpoint_sha,
            "selection_reason": stage2.get("selection_reason"),
            "selection_sha256": sha256(paths["stage2_selection"]),
            "train_report_sha256": sha256(paths["stage2_report"]),
            "source_manifest_sha256": sha256(paths["stage2_source"]),
        },
        "lineage": {
            "status": "ok",
            "frozen_backbone_contract_digest": backbone_digest,
            "skill_cache": skill_cache,
            "state_cache": state_cache,
            "action_cache": action_cache,
            "result_cache": result_cache,
            "static_reranker_parent_stage0_checkpoint_sha256": stage0_checkpoint_sha,
            "stage2_parent_static_reranker_checkpoint_sha256": candidate_checkpoint_sha,
            "candidate_foundation_digest": candidate_digest,
            "stage2_initial_candidate_foundation_digest": (
                stage2_initial_candidate_digest
            ),
            "static_route_foundation_digest": static_route_digest,
            "selected_skills_sha256": skills_sha,
            "data_contract_sha256": data_manifest_sha,
        },
        "horizon_protocol": horizon_protocol,
        "canonical_stage_chain": [
            "precompute",
            "stage0_static",
            "static_route_query_adapter",
            "stage2_causal_memory",
        ],
        "legacy_stage1_or_stage4_reachable": False,
    }
    if write_manifest:
        target = run_root / "final_chain_manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return payload


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: finalize_clstr_vnext_full_chain.py RUN_ROOT DATA_MANIFEST")
    try:
        payload = finalize_full_chain(sys.argv[1], sys.argv[2])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
