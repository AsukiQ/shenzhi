#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.memory_utility_gate_train import train_memory_utility_gate
from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_lineage import sha256_path
from scripts.audit_clstr_memory_utility_oracle import (
    finalize_memory_utility_oracle_audit_report,
    load_route_records_from_manifests,
    run_memory_utility_oracle_audit,
)


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _require_self_hash(payload: dict[str, Any], *, label: str) -> None:
    copy = dict(payload)
    recorded = str(copy.pop("manifest_sha256", ""))
    if not recorded or recorded != canonical_digest(copy):
        raise ValueError(f"{label} self-hash mismatch")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_route_identity(
    dynamic_selection: dict[str, Any],
    loaded: dict[str, Any],
) -> dict[str, Any]:
    expected = dict(dynamic_selection.get("validation_route_records") or {})
    manifests = list(loaded.get("manifests") or [])
    if len(manifests) != 1:
        raise ValueError("direct utility recalibration requires one selected route manifest")
    manifest = manifests[0]
    records = list(loaded.get("records") or [])
    row_digest = canonical_digest([str(row.get("row_digest") or "") for row in records])
    actual_path = Path(str(manifest.get("records_path") or "")).resolve()
    expected_path = Path(str(expected.get("path") or "")).resolve()
    actual_file_identity = sha256_path(actual_path)
    if (
        actual_path != expected_path
        or actual_file_identity["sha256"] != str(expected.get("sha256") or "")
        or int(actual_file_identity["size"]) != int(expected.get("size", -1))
        or int(actual_file_identity["file_count"])
        != int(expected.get("file_count", -1))
        or int(manifest.get("record_count", -1))
        != int(expected.get("record_count", -1))
        or str(manifest.get("source_rows_digest") or "")
        != str(expected.get("row_digest_sha256") or "")
        or len(records) != int(expected.get("record_count", -1))
        or row_digest != str(expected.get("row_digest_sha256") or "")
    ):
        raise ValueError("route manifest does not match selected validation route records")
    return {
        **actual_file_identity,
        "record_count": len(records),
        "row_digest_sha256": row_digest,
    }


def run_qwen06_cmc_direct_utility_recalibration(
    *,
    dynamic_selection_path: str | Path,
    route_manifest_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    temperature: float = 0.5,
    fused_rank_weight: float = 1.0,
    static_no_regret_weight: float = 1.0,
    direct_gate_bce_weight: float = 1.0,
    max_steps: int = 300,
    learning_rate: float = 0.01,
    seed: int = 17,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dynamic_path = Path(dynamic_selection_path).resolve()
    dynamic = _read_json(dynamic_path, label="CMC dynamic selection")
    _require_self_hash(dynamic, label="CMC dynamic selection")
    if (
        dynamic.get("status") != "ok"
        or dynamic.get("release_status") != "ok"
        or dynamic.get("stage4_method")
        != "counterfactual_memory_calibration_v1"
    ):
        raise ValueError("selected Stage4 endpoint is not release-safe CMC")
    selected_checkpoint = sha256_path(dynamic["selected_checkpoint_path"])
    if selected_checkpoint["sha256"] != str(
        dynamic.get("selected_checkpoint_sha256") or ""
    ):
        raise ValueError("selected Stage4 checkpoint SHA-256 mismatch")
    selected_validation = sha256_path(dynamic["selected_validation_report_path"])
    if selected_validation["sha256"] != str(
        dynamic.get("selected_validation_report_sha256") or ""
    ):
        raise ValueError("selected Stage4 validation report SHA-256 mismatch")
    route_path = Path(route_manifest_path).resolve()
    loaded = load_route_records_from_manifests([route_path])
    route_identity = _selected_route_identity(dynamic, loaded)
    audit = run_memory_utility_oracle_audit(
        loaded["records"],
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )
    audit = finalize_memory_utility_oracle_audit_report(
        audit,
        records=loaded["records"],
        route_manifests=loaded["manifests"],
        route_manifest_identity=loaded["identity"],
    )
    audit_path = output / "oracle_audit.json"
    _atomic_json(audit_path, audit)
    objective = {
        "temperature": float(temperature),
        "fused_rank_weight": float(fused_rank_weight),
        "static_no_regret_weight": float(static_no_regret_weight),
        "direct_gate_bce_weight": float(direct_gate_bce_weight),
        "source_balanced": True,
        "harm_sign_balanced": True,
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
    }
    gate_report: dict[str, Any] | None = None
    gate_report_path: Path | None = None
    checkpoint_identity: dict[str, Any] | None = None
    if audit.get("learned_gate_recommended") is True:
        gate_report = train_memory_utility_gate(
            route_records=loaded["records"],
            audit_report=audit,
            dynamic_selection=dynamic,
            output_dir=output / "gate",
            temperature=temperature,
            fused_rank_weight=fused_rank_weight,
            static_no_regret_weight=static_no_regret_weight,
            direct_gate_bce_weight=direct_gate_bce_weight,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
        )
        gate_report_path = output / "gate" / "gate_report.json"
        if gate_report.get("promoted") is True:
            checkpoint_identity = sha256_path(gate_report["checkpoint_path"])
    promoted = bool(gate_report and gate_report.get("promoted") is True)
    status = (
        "ok"
        if promoted
        else "not_promoted"
        if gate_report is not None
        else "not_recommended"
    )
    report = {
        "schema_version": "qwen06_cmc_direct_utility_recalibration_v1",
        "status": status,
        "promoted": promoted,
        "dynamic_selection": sha256_path(dynamic_path),
        "selected_stage4_checkpoint_sha256": dynamic[
            "selected_checkpoint_sha256"
        ],
        "selected_validation_route_records": route_identity,
        "route_manifest": sha256_path(route_path),
        "oracle_audit": sha256_path(audit_path),
        "gate_report": (
            None if gate_report_path is None else sha256_path(gate_report_path)
        ),
        "checkpoint": checkpoint_identity,
        "gate_output_semantics": (
            None
            if gate_report is None
            else gate_report.get("gate_output_semantics")
        ),
        "alpha_base": (
            None if gate_report is None else gate_report.get("alpha_base")
        ),
        "objective": objective,
    }
    report["manifest_sha256"] = canonical_digest(report)
    _atomic_json(output / "direct_utility_recalibration_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and optionally refit the Qwen06 CMC direct-utility gate."
    )
    parser.add_argument("--dynamic_selection_path", required=True)
    parser.add_argument("--route_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fused_rank_weight", type=float, default=1.0)
    parser.add_argument("--static_no_regret_weight", type=float, default=1.0)
    parser.add_argument("--direct_gate_bce_weight", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run_qwen06_cmc_direct_utility_recalibration(
        dynamic_selection_path=args.dynamic_selection_path,
        route_manifest_path=args.route_manifest_path,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        temperature=args.temperature,
        fused_rank_weight=args.fused_rank_weight,
        static_no_regret_weight=args.static_no_regret_weight,
        direct_gate_bce_weight=args.direct_gate_bce_weight,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
