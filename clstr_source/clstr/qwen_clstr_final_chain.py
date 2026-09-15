from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from clstr.counterfactual_memory_calibration import (
    CMC_FEATURE_CANDIDATE_COUNT_CAP,
    CMC_FEATURE_UPDATE_COUNT_CAP,
)
from clstr.memory_utility_records import canonical_digest, checkpoint_chain_digest
from clstr.qwen_clstr_lineage import QWEN_MODEL_BASENAME, SCHEMA_VERSION, sha256_path
from clstr.safe_memory_ranking import CANDIDATE_PROVENANCE_RESIDUAL_BOUND
from clstr.state_query_prompt import resolve_state_query_prompt_contract


FINAL_CHAIN_SCHEMA_VERSION = "qwen06_clstr_final_chain_v2"


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _resolved_recorded_path(value: Any, *, base: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} path is missing")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_ok(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("status") != "ok":
        raise ValueError(f"{label} is not ok")


def _validate_reliability(reliability: Any) -> dict[str, Any]:
    if not isinstance(reliability, dict):
        raise ValueError("Stage4 reliability identity is missing")
    payload = dict(reliability)
    recorded = str(payload.pop("reliability_sha256", ""))
    if not recorded or recorded != canonical_digest(payload):
        raise ValueError("Stage4 reliability self-hash mismatch")
    mode = str(reliability.get("mode") or "")
    if mode == "fixed_alpha":
        alpha = float(reliability.get("fixed_alpha", float("nan")))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Stage4 fixed alpha must be in [0, 1]")
        if reliability.get("gate_checkpoint") is not None:
            raise ValueError("fixed-alpha reliability cannot carry a gate checkpoint")
    elif mode == "learned":
        gate = reliability.get("gate_checkpoint")
        if not isinstance(gate, dict) or not gate.get("path") or not gate.get("sha256"):
            raise ValueError("learned reliability requires a gate identity")
        _validate_recorded_identity(gate, label="Stage4 reliability gate")
        if not reliability.get("audit_manifest_sha256"):
            raise ValueError("learned reliability requires an audit identity")
    elif mode == "causal_gate":
        if reliability.get("fixed_alpha") is not None:
            raise ValueError("causal-gate reliability cannot carry fixed alpha")
        if reliability.get("gate_checkpoint") is not None:
            raise ValueError("causal-gate reliability is stored in the Stage4 checkpoint")
        residual_bound = float(
            reliability.get("safe_memory_residual_bound", float("nan"))
        )
        if not math.isfinite(residual_bound) or residual_bound <= 0.0:
            raise ValueError(
                "causal-gate safe-memory residual bound must be finite and positive"
            )
    elif mode == "cmc_candidate_gate":
        if reliability.get("fixed_alpha") is not None:
            raise ValueError("CMC reliability cannot carry fixed alpha")
        update_cap = float(
            reliability.get("feature_update_count_cap", float("nan"))
        )
        candidate_cap = float(
            reliability.get("feature_candidate_count_cap", float("nan"))
        )
        gate = reliability.get("gate_checkpoint")
        if gate is None:
            if (
                update_cap != CMC_FEATURE_UPDATE_COUNT_CAP
                or candidate_cap != CMC_FEATURE_CANDIDATE_COUNT_CAP
            ):
                raise ValueError(
                    "CMC reliability feature count caps do not match training"
                )
        else:
            _validate_recorded_identity(gate, label="CMC direct utility gate")
            _validate_recorded_identity(
                reliability.get("gate_report"),
                label="CMC direct utility gate report",
            )
            if not math.isfinite(update_cap) or not math.isfinite(candidate_cap):
                raise ValueError("CMC direct utility feature caps must be finite")
            if update_cap <= 0.0 or candidate_cap <= 0.0:
                raise ValueError("CMC direct utility feature caps must be positive")
            if not str(reliability.get("audit_manifest_sha256") or ""):
                raise ValueError("CMC direct utility overlay requires an audit identity")
            if not str(
                reliability.get("selected_stage4_checkpoint_sha256") or ""
            ):
                raise ValueError("CMC direct utility overlay requires a Stage4 identity")
            if reliability.get("base_gate_source") != "stage4_checkpoint":
                raise ValueError("CMC direct utility base gate source mismatch")
            if reliability.get("deployed_gate_source") != "direct_utility_overlay":
                raise ValueError("CMC direct utility deployed gate source mismatch")
            if (
                reliability.get("gate_output_semantics")
                != "anchored_harm_suppression_alpha"
            ):
                raise ValueError("CMC anchored gate output semantics mismatch")
            alpha_base = float(reliability.get("alpha_base", float("nan")))
            if not math.isfinite(alpha_base) or not 0.0 < alpha_base <= 1.0:
                raise ValueError("CMC anchored gate alpha base is invalid")
    elif mode == "candidate_admission_residual":
        if reliability.get("fixed_alpha") is not None:
            raise ValueError(
                "candidate admission reliability cannot carry fixed alpha"
            )
        if reliability.get("gate_checkpoint") is not None:
            raise ValueError(
                "candidate admission reliability cannot carry a gate checkpoint"
            )
        if float(
            reliability.get("safe_memory_residual_bound", float("nan"))
        ) != 2.0:
            raise ValueError("candidate admission residual bound must equal 2.0")
        if (
            float(reliability.get("feature_update_count_cap", float("nan")))
            != CMC_FEATURE_UPDATE_COUNT_CAP
            or float(
                reliability.get("feature_candidate_count_cap", float("nan"))
            )
            != CMC_FEATURE_CANDIDATE_COUNT_CAP
        ):
            raise ValueError(
                "candidate admission feature caps do not match training"
            )
        base_sha = str(reliability.get("base_cmc_checkpoint_sha256") or "")
        if len(base_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in base_sha.lower()
        ):
            raise ValueError("candidate admission base CMC identity is invalid")
        if reliability.get("deployed_reliability_source") != (
            "candidate_admission_constrained_residual"
        ):
            raise ValueError(
                "candidate admission deployed reliability source mismatch"
            )
    elif mode == "cmc_candidate_provenance":
        if reliability.get("fixed_alpha") is not None:
            raise ValueError("candidate provenance reliability cannot carry fixed alpha")
        if reliability.get("gate_checkpoint") is not None:
            raise ValueError("candidate provenance reliability cannot carry a gate checkpoint")
        update_cap = float(
            reliability.get("feature_update_count_cap", float("nan"))
        )
        candidate_cap = float(
            reliability.get("feature_candidate_count_cap", float("nan"))
        )
        if (
            update_cap != CMC_FEATURE_UPDATE_COUNT_CAP
            or candidate_cap != CMC_FEATURE_CANDIDATE_COUNT_CAP
        ):
            raise ValueError(
                "candidate provenance feature count caps do not match CMC training"
            )
        residual_bound = float(
            reliability.get("safe_memory_residual_bound", float("nan"))
        )
        if residual_bound != CANDIDATE_PROVENANCE_RESIDUAL_BOUND:
            raise ValueError("candidate provenance residual bound must equal 2.0")
        if reliability.get("base_reliability_source") != "stage4_checkpoint":
            raise ValueError("candidate provenance base reliability source mismatch")
        if reliability.get("deployed_reliability_source") != (
            "candidate_provenance_positive_residual"
        ):
            raise ValueError("candidate provenance deployed reliability source mismatch")
        stage4_sha = str(
            reliability.get("selected_stage4_checkpoint_sha256") or ""
        )
        if len(stage4_sha) != 64 or any(
            character not in "0123456789abcdef" for character in stage4_sha.lower()
        ):
            raise ValueError("candidate provenance Stage4 checkpoint identity is invalid")
    else:
        raise ValueError("unsupported Stage4 reliability mode")
    return dict(reliability)


def _validate_cmc_checkpoint_parent_digests(
    checkpoint_path: str | Path,
    *,
    router_integrity: Any,
) -> dict[str, str]:
    if not isinstance(router_integrity, dict):
        raise ValueError("CMC router integrity is missing")
    expected_full = str(router_integrity.get("full_router_digest") or "")
    expected_fast = str(router_integrity.get("fast_router_digest") or "")
    if not expected_full or not expected_fast:
        raise ValueError("CMC router integrity digests are missing")
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("CMC Stage4 checkpoint payload must be a dict")
    method = str(payload.get("stage4_method") or "")
    if method not in {
        "counterfactual_memory_calibration_v1",
        "candidate_admission_residual_v1",
    }:
        raise ValueError("selected Stage4 checkpoint is not a CMC-derived delta")
    recorded_full = str(payload.get("parent_stage2_full_router_digest") or "")
    recorded_fast = str(payload.get("parent_stage2_fast_router_digest") or "")
    if recorded_full != expected_full or recorded_fast != expected_fast:
        raise ValueError("CMC parent Stage2 router digest mismatch")
    report = {
        "stage4_method": method,
        "parent_stage2_full_router_digest": recorded_full,
        "parent_stage2_fast_router_digest": recorded_fast,
    }
    if method == "candidate_admission_residual_v1":
        report["base_cmc_checkpoint_sha256"] = str(
            payload.get("base_cmc_checkpoint_sha256") or ""
        )
    return report


def _read_stage4_selection(path: Path) -> dict[str, Any]:
    selection = _read_json_object(path, label="Stage4 selection")
    digest_payload = dict(selection)
    recorded = str(digest_payload.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(digest_payload):
        raise ValueError("Stage4 selection self-hash mismatch")
    if selection.get("schema_version") != "stage4_selection_v1":
        raise ValueError("unsupported Stage4 selection schema")
    _require_ok(selection, label="Stage4 selection")
    if selection.get("release_status") != "ok":
        raise ValueError("Stage4 selection is not release-safe")
    checks = selection.get("release_checks") or {}
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Stage4 selection release checks are incomplete")
    router = selection.get("router_integrity") or {}
    if (
        not isinstance(router, dict)
        or router.get("static_logits_exact") is not True
        or float(router.get("max_abs_static_logit_difference", float("inf")))
        != 0.0
    ):
        raise ValueError("Stage4 selection lacks exact router integrity")
    _validate_reliability(selection.get("reliability"))
    return selection


def _direct_utility_reliability_overlay(
    report_path: str | Path,
    *,
    selected_stage4_checkpoint_sha256: str,
) -> dict[str, Any]:
    resolved_report_path = Path(report_path).resolve()
    report = _read_json_object(
        resolved_report_path,
        label="direct utility gate report",
    )
    payload = dict(report)
    recorded = str(payload.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(payload):
        raise ValueError("direct utility gate report self-hash mismatch")
    if report.get("schema_version") != "memory_utility_gate_report_v2":
        raise ValueError("unsupported direct utility gate report schema")
    if report.get("status") != "ok" or report.get("promoted") is not True:
        raise ValueError("direct utility gate report is not promoted")
    if str(report.get("selected_stage4_checkpoint_sha256") or "") != str(
        selected_stage4_checkpoint_sha256
    ):
        raise ValueError("direct utility gate Stage4 identity mismatch")
    trainable_names = list(report.get("trainable_parameter_names") or [])
    if set(trainable_names) != {"net.0.weight", "net.0.bias"}:
        raise ValueError("direct utility gate trainable scope mismatch")
    expected_objective = {
        "temperature": 0.5,
        "fused_rank_weight": 1.0,
        "static_no_regret_weight": 1.0,
        "direct_gate_bce_weight": 1.0,
        "source_balanced": True,
        "harm_sign_balanced": True,
    }
    if report.get("objective") != expected_objective:
        raise ValueError("direct utility gate objective identity mismatch")
    promotion_checks = dict(
        (report.get("validation") or {}).get("promotion_checks") or {}
    )
    if not promotion_checks or not all(value is True for value in promotion_checks.values()):
        raise ValueError("direct utility gate promotion checks are incomplete")
    checkpoint_path = Path(str(report.get("checkpoint_path") or "")).resolve()
    checkpoint_identity = sha256_path(checkpoint_path)
    if checkpoint_identity["sha256"] != str(report.get("checkpoint_sha256") or ""):
        raise ValueError("direct utility gate checkpoint identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("direct utility gate checkpoint must be a dict")
    audit_sha = str(report.get("audit_manifest_sha256") or "")
    output_semantics = str(report.get("gate_output_semantics") or "")
    alpha_base = float(report.get("alpha_base", 0.0))
    if output_semantics != "anchored_harm_suppression_alpha":
        raise ValueError("direct utility gate output semantics mismatch")
    if not math.isfinite(alpha_base) or not 0.0 < alpha_base <= 1.0:
        raise ValueError("direct utility gate alpha base mismatch")
    update_cap = float(report.get("feature_update_count_cap", 0.0))
    candidate_cap = float(report.get("feature_candidate_count_cap", 0.0))
    checkpoint_contract = {
        "stage": "clstr_memory_utility_gate",
        "schema_version": "memory_utility_gate_checkpoint_v2",
        "feature_schema": "memory_utility_features_v1",
        "gate_output_semantics": output_semantics,
        "alpha_base": alpha_base,
        "audit_manifest_sha256": audit_sha,
        "selected_stage4_checkpoint_sha256": selected_stage4_checkpoint_sha256,
        "zero_history_fallback": "exact_static",
        "reliability_changes_memory_state": False,
        "trainable_parameter_names": trainable_names,
        "objective": expected_objective,
        "feature_update_count_cap": update_cap,
        "feature_candidate_count_cap": candidate_cap,
    }
    for key, expected in checkpoint_contract.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"direct utility gate checkpoint {key} mismatch")
    state = checkpoint.get("harm_probability_gate_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("direct utility gate checkpoint state is missing")
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": checkpoint_identity,
        "gate_report": sha256_path(resolved_report_path),
        "audit_manifest_sha256": audit_sha,
        "selected_stage4_checkpoint_sha256": selected_stage4_checkpoint_sha256,
        "feature_update_count_cap": update_cap,
        "feature_candidate_count_cap": candidate_cap,
        "base_gate_source": "stage4_checkpoint",
        "deployed_gate_source": "direct_utility_overlay",
        "gate_output_semantics": output_semantics,
        "alpha_base": alpha_base,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    return _validate_reliability(reliability)


def _expected_state_query() -> dict[str, Any]:
    contract = resolve_state_query_prompt_contract(
        prompt_version="clstr_causal_state_v1",
        max_chars=2000,
        truncation="head_tail_v1",
    )
    return {
        "prompt_version": contract["state_query_prompt_version"],
        "instruction": contract["state_query_instruction"],
        "max_chars": contract["state_query_max_chars"],
        "truncation": contract["state_query_truncation"],
    }


def _validate_recorded_identity(
    recorded: Any,
    *,
    label: str,
    actual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(recorded, dict):
        raise ValueError(f"{label} identity is missing")
    if not recorded.get("path") or not recorded.get("sha256"):
        raise ValueError(f"{label} identity is missing")
    current = actual or sha256_path(recorded["path"])
    if str(current["sha256"]) != str(recorded.get("sha256") or ""):
        raise ValueError(f"{label} digest mismatch")
    if int(current["size"]) != int(recorded.get("size", -1)):
        raise ValueError(f"{label} size mismatch")
    if int(current["file_count"]) != int(recorded.get("file_count", -1)):
        raise ValueError(f"{label} file count mismatch")
    return current


def _parent_record(
    *,
    role: str,
    child: dict[str, Any],
    child_path: Path,
    expected_parent_path: Path,
    parent: dict[str, Any],
) -> None:
    parents = child.get("parents") or {}
    if not isinstance(parents, dict):
        raise ValueError(f"{child_path.name} lineage parents must be an object")
    record = parents.get(role)
    if not isinstance(record, dict):
        raise ValueError(f"parent lineage record missing: {role}")
    recorded_path = _resolved_recorded_path(
        record.get("manifest_path"),
        base=child_path.parent,
        label=f"parent {role}",
    )
    if recorded_path != expected_parent_path:
        raise ValueError(f"parent lineage path mismatch: {role}")
    if sha256_path(expected_parent_path)["sha256"] != str(record.get("manifest_sha256") or ""):
        raise ValueError(f"parent manifest digest mismatch: {role}")
    parent_checkpoint = parent.get("checkpoint") or {}
    if str(parent_checkpoint.get("sha256") or "") != str(record.get("checkpoint_sha256") or ""):
        raise ValueError(f"parent checkpoint digest mismatch: {role}")


def _gate_checkpoint_path(
    gate: dict[str, Any],
    *,
    fallback: Path,
    run_root: Path,
    label: str,
) -> Path:
    raw = gate.get("checkpoint_path")
    path = fallback if not raw else _resolved_recorded_path(raw, base=run_root, label=label)
    if path != fallback.resolve():
        raise ValueError(f"{label} checkpoint path mismatch")
    return path


def _resolve_training_layout(run_root: str | Path) -> dict[str, Path]:
    run_root = Path(run_root).resolve()
    candidate_stage2 = run_root / "stage2_anchored_full10000_v1"
    candidate_stage4 = run_root / "stage4_candidate_admission_full"
    if (
        (candidate_stage2 / "lineage.json").is_file()
        and (candidate_stage4 / "stage4_selection.json").is_file()
    ):
        return {
            "stage2_dir": candidate_stage2.resolve(),
            "stage4_dir": candidate_stage4.resolve(),
        }
    cmc_stage2 = run_root / "stage2_anchored_full10000_v1"
    cmc_stage4 = run_root / "stage4_cmc_full"
    if (
        (cmc_stage2 / "lineage.json").is_file()
        and (cmc_stage4 / "stage4_selection.json").is_file()
    ):
        return {
            "stage2_dir": cmc_stage2.resolve(),
            "stage4_dir": cmc_stage4.resolve(),
        }
    safe_root = run_root / "safe_memory_fusion"
    safe_stage2 = safe_root / "stage2_full"
    safe_stage4 = safe_root / "stage4_full"
    if (
        (safe_stage2 / "lineage.json").is_file()
        and (safe_stage4 / "stage4_selection.json").is_file()
    ):
        return {
            "stage2_dir": safe_stage2.resolve(),
            "stage4_dir": safe_stage4.resolve(),
        }
    return {
        "stage2_dir": (run_root / "stage2_full").resolve(),
        "stage4_dir": (run_root / "stage4_safe_full").resolve(),
    }


def resolve_qwen_clstr_final_chain(
    run_root: str | Path,
    *,
    reliability_gate_report_path: str | Path | None = None,
    candidate_provenance_overlay: bool = False,
) -> dict[str, Any]:
    if candidate_provenance_overlay and reliability_gate_report_path is not None:
        raise ValueError(
            "candidate provenance and direct utility reliability overlays are mutually exclusive"
        )
    run_root = Path(run_root).resolve()
    training_layout = _resolve_training_layout(run_root)
    stage2_dir = training_layout["stage2_dir"]
    stage4_dir = training_layout["stage4_dir"]
    selection_path = run_root / "stage0_full" / "stage0_selection.json"
    selection = _read_json_object(selection_path, label="Stage0 selection")
    _require_ok(selection, label="Stage0 selection")
    if selection.get("release_status") != "ok":
        raise ValueError("Stage0 selection is not release-safe")
    stage4_selection_path = (stage4_dir / "stage4_selection.json").resolve()
    stage4_selection = _read_stage4_selection(stage4_selection_path)
    selected_stage4_checkpoint = _resolved_recorded_path(
        stage4_selection.get("selected_checkpoint_path"),
        base=run_root,
        label="selected Stage4 checkpoint",
    )
    selected_validation_report = _resolved_recorded_path(
        stage4_selection.get("selected_validation_report_path"),
        base=run_root,
        label="selected Stage4 validation report",
    )
    if sha256_path(selected_stage4_checkpoint)["sha256"] != str(
        stage4_selection.get("selected_checkpoint_sha256") or ""
    ):
        raise ValueError("selected Stage4 checkpoint digest mismatch")
    if sha256_path(selected_validation_report)["sha256"] != str(
        stage4_selection.get("selected_validation_report_sha256") or ""
    ):
        raise ValueError("selected Stage4 validation report digest mismatch")
    cmc_parent_router_digests = None
    if stage4_selection.get("stage4_method") in {
        "counterfactual_memory_calibration_v1",
        "candidate_admission_residual_v1",
    }:
        cmc_parent_router_digests = _validate_cmc_checkpoint_parent_digests(
            selected_stage4_checkpoint,
            router_integrity=stage4_selection.get("router_integrity"),
        )
        if (
            stage4_selection.get("stage4_method")
            == "candidate_admission_residual_v1"
            and cmc_parent_router_digests["base_cmc_checkpoint_sha256"]
            != str(
                dict(stage4_selection.get("reliability") or {}).get(
                    "base_cmc_checkpoint_sha256"
                )
                or ""
            )
        ):
            raise ValueError("candidate admission base CMC identity mismatch")

    checkpoint_paths = {
        "stage0": _resolved_recorded_path(
            selection.get("selected_checkpoint_path"),
            base=run_root,
            label="Stage0 checkpoint",
        ),
        "stage1": (run_root / "stage1_full" / "checkpoints" / "clstr_stage1_heads-step3000.pt").resolve(),
        "stage2": (stage2_dir / "checkpoints" / "clstr_full_base-step10000.pt").resolve(),
        "stage4": selected_stage4_checkpoint,
    }
    lineage_paths = {
        "stage0": _resolved_recorded_path(
            selection.get("selected_lineage_path"),
            base=run_root,
            label="Stage0 lineage",
        ),
        "stage1": (run_root / "stage1_full" / "lineage.json").resolve(),
        "stage2": (stage2_dir / "lineage.json").resolve(),
        "stage4": (stage4_dir / "lineage.json").resolve(),
    }
    selected_report_path = _resolved_recorded_path(
        selection.get("selected_report_path"),
        base=run_root,
        label="Stage0 selected report",
    )
    selected_report = _read_json_object(selected_report_path, label="Stage0 selected report")
    _require_ok(selected_report, label="Stage0 selected report")

    gate_paths = {
        "stage1": run_root / "stage1_full" / "stage1_quality_gate.json",
        "stage2": stage2_dir / "stage2_quality_gate.json",
        "stage4": stage4_dir / "stage4_quality_gate.json",
    }
    gates: dict[str, dict[str, Any]] = {}
    for role in ("stage1", "stage2", "stage4"):
        label = f"{role.title()} quality gate"
        gate = _read_json_object(gate_paths[role], label=label)
        _require_ok(gate, label=label)
        _gate_checkpoint_path(
            gate,
            fallback=checkpoint_paths[role],
            run_root=run_root,
            label=label,
        )
        gates[role] = gate
    stage4_gate = gates["stage4"]
    if _resolved_recorded_path(
        stage4_gate.get("selection_path"),
        base=run_root,
        label="Stage4 quality-gate selection",
    ) != stage4_selection_path:
        raise ValueError("Stage4 quality-gate selection path mismatch")
    gate_validation_identity = stage4_gate.get("selected_validation_report")
    if isinstance(gate_validation_identity, dict):
        gate_validation_report = _resolved_recorded_path(
            gate_validation_identity.get("path"),
            base=run_root,
            label="Stage4 quality-gate validation report",
        )
        if gate_validation_identity != sha256_path(gate_validation_report):
            raise ValueError("Stage4 quality-gate validation report digest mismatch")
    else:
        gate_validation_report = _resolved_recorded_path(
            stage4_gate.get("selected_validation_report_path"),
            base=run_root,
            label="Stage4 quality-gate validation report",
        )
    if gate_validation_report != selected_validation_report:
        raise ValueError("Stage4 quality-gate validation report mismatch")
    gate_selection = stage4_gate.get("stage4_selection")
    if isinstance(gate_selection, dict):
        if gate_selection != stage4_selection:
            raise ValueError("Stage4 quality-gate selection payload mismatch")
        gate_router_integrity = gate_selection.get("router_integrity")
    else:
        gate_router_integrity = stage4_gate.get("router_integrity")
    if gate_router_integrity != stage4_selection.get("router_integrity"):
        raise ValueError("Stage4 quality-gate router integrity mismatch")

    lineages = {
        role: _read_json_object(path, label=f"{role.title()} lineage")
        for role, path in lineage_paths.items()
    }
    expected_state_query = _expected_state_query()
    for role, lineage in lineages.items():
        if lineage.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported {role} lineage schema version")
        if str(lineage.get("stage") or "") != role:
            raise ValueError(f"{role.title()} lineage stage mismatch")
        if lineage.get("route_scorer") != "unified_memory":
            raise ValueError("Qwen CLSTR requires route_scorer=unified_memory")
        if lineage.get("state_query") != expected_state_query:
            raise ValueError("state query prompt contract mismatch")
        backbone = lineage.get("backbone") or {}
        if not isinstance(backbone, dict) or backbone.get("frozen") is not True:
            raise ValueError("Qwen backbone must remain frozen")
        if Path(str(backbone.get("path") or "")).resolve().name != QWEN_MODEL_BASENAME:
            raise ValueError("Qwen backbone identity mismatch")

    identity_roles = ("backbone", "skill_pool", "data_manifest")
    stage0_lineage = lineages["stage0"]
    for role in ("stage1", "stage2", "stage4"):
        for identity_role in identity_roles:
            if lineages[role].get(identity_role) != stage0_lineage.get(identity_role):
                raise ValueError(f"{role.title()} {identity_role} identity mismatch")

    shared_identities = {
        role: _validate_recorded_identity(stage0_lineage.get(role), label=role.replace("_", " "))
        for role in identity_roles
    }
    checkpoint_identities: dict[str, dict[str, Any]] = {}
    for role, checkpoint_path in checkpoint_paths.items():
        recorded = lineages[role].get("checkpoint") or {}
        if Path(str(recorded.get("path") or "")).resolve() != checkpoint_path:
            raise ValueError(f"{role.title()} checkpoint path mismatch")
        checkpoint_identities[role] = _validate_recorded_identity(
            recorded,
            label=f"{role.title()} checkpoint",
        )

    expected_parent_roles = {
        "stage0": set(),
        "stage1": {"stage0"},
        "stage2": {"stage0", "stage1"},
        "stage4": {"stage0", "stage2"},
    }
    for role, expected_roles in expected_parent_roles.items():
        parents = lineages[role].get("parents") or {}
        if not isinstance(parents, dict) or set(parents) != expected_roles:
            raise ValueError(f"{role.title()} parent lineage roles mismatch")
        for parent_role in sorted(expected_roles):
            _parent_record(
                role=parent_role,
                child=lineages[role],
                child_path=lineage_paths[role],
                expected_parent_path=lineage_paths[parent_role],
                parent=lineages[parent_role],
            )
    if lineages["stage4"].get("identity_parent_role") != "stage2":
        raise ValueError("Stage4 identity parent must be stage2")
    stage4_metadata = lineages["stage4"].get("stage_metadata") or {}
    if not isinstance(stage4_metadata, dict):
        raise ValueError("Stage4 lineage metadata is missing")
    if stage4_metadata.get("stage4_selection") != sha256_path(
        stage4_selection_path
    ):
        raise ValueError("Stage4 lineage selection identity mismatch")
    if stage4_metadata.get("router_integrity") != stage4_selection.get(
        "router_integrity"
    ):
        raise ValueError("Stage4 lineage router integrity mismatch")
    if stage4_metadata.get("reliability") != stage4_selection.get("reliability"):
        raise ValueError("Stage4 lineage reliability mismatch")

    reliability = dict(stage4_selection["reliability"])
    if reliability_gate_report_path is not None:
        if stage4_selection.get("stage4_method") != (
            "counterfactual_memory_calibration_v1"
        ):
            raise ValueError("direct utility gate overlay requires CMC Stage4")
        reliability = _direct_utility_reliability_overlay(
            reliability_gate_report_path,
            selected_stage4_checkpoint_sha256=str(
                stage4_selection["selected_checkpoint_sha256"]
            ),
        )
    elif candidate_provenance_overlay:
        if stage4_selection.get("stage4_method") != (
            "counterfactual_memory_calibration_v1"
        ):
            raise ValueError("candidate provenance overlay requires CMC Stage4")
        reliability = {
            "mode": "cmc_candidate_provenance",
            "fixed_alpha": None,
            "gate_checkpoint": None,
            "feature_update_count_cap": CMC_FEATURE_UPDATE_COUNT_CAP,
            "feature_candidate_count_cap": CMC_FEATURE_CANDIDATE_COUNT_CAP,
            "safe_memory_residual_bound": CANDIDATE_PROVENANCE_RESIDUAL_BOUND,
            "selected_stage4_checkpoint_sha256": str(
                stage4_selection["selected_checkpoint_sha256"]
            ),
            "base_reliability_source": "stage4_checkpoint",
            "deployed_reliability_source": (
                "candidate_provenance_positive_residual"
            ),
        }
        reliability["reliability_sha256"] = canonical_digest(reliability)
        reliability = _validate_reliability(reliability)

    payload: dict[str, Any] = {
        "schema_version": FINAL_CHAIN_SCHEMA_VERSION,
        "status": "ok",
        "run_root": str(run_root),
        "final_checkpoint_role": "stage4",
        "checkpoint_chain_digest": checkpoint_chain_digest(checkpoint_paths),
        "checkpoints": checkpoint_identities,
        "lineages": {
            role: {**sha256_path(path), "stage": role}
            for role, path in lineage_paths.items()
        },
        "quality_gates": {
            role: {**sha256_path(path), "status": gates[role]["status"]}
            for role, path in gate_paths.items()
        },
        "stage0_selection": sha256_path(selection_path),
        "stage0_selected_report": sha256_path(selected_report_path),
        "stage4_selection": sha256_path(stage4_selection_path),
        "stage4_selected_validation_report": sha256_path(
            selected_validation_report
        ),
        "reliability": reliability,
        "cmc_parent_router_digests": cmc_parent_router_digests,
        "backbone": {**shared_identities["backbone"], "frozen": True},
        "skill_pool": shared_identities["skill_pool"],
        "data_manifest": shared_identities["data_manifest"],
        "route_scorer": "unified_memory",
        "state_query": expected_state_query,
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload
