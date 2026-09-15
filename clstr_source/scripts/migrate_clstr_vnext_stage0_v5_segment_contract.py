#!/usr/bin/env python3
"""Create an auditable v5 checkpoint for a longer deterministic Stage0 segment.

Stage0 v5 accidentally included the segment-local scheduled occurrence count in
its immutable run contract.  Extending a completed segment therefore failed
closed even when every semantic training setting was unchanged.  This tool
copies a checkpoint and updates only that count and the derived contract digest.
It never overwrites the source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


V5_SCHEMA = "clstr_vnext_stage0_unified_static_run_v5"
MIGRATION_SCHEMA = "clstr_vnext_stage0_v5_segment_contract_migration_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(payload: dict[str, Any]) -> dict[str, Any]:
    values = copy.deepcopy(payload)
    values.pop("contract_digest", None)
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    values["contract_digest"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return values


def migrate_contract(
    observed: dict[str, Any],
    *,
    checkpoint_step: int,
    segment_end_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(observed.get("schema_version") or "") != V5_SCHEMA:
        raise ValueError("segment migration requires a Stage0 v5 run contract")
    if int(segment_end_step) <= int(checkpoint_step):
        raise ValueError("segment end must be greater than checkpoint step")
    optimization = observed.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("Stage0 v5 run contract lacks optimization settings")
    batch_size = int(optimization.get("batch_size") or 0)
    accumulation = int(optimization.get("gradient_accumulation_steps") or 0)
    if batch_size <= 0 or accumulation <= 0:
        raise ValueError("Stage0 v5 run contract has invalid batch settings")
    prior_count = int(optimization.get("scheduled_training_occurrence_count") or -1)
    expected_prior_count = int(checkpoint_step) * batch_size * accumulation
    if prior_count != expected_prior_count:
        raise ValueError(
            "source checkpoint occurrence count does not match its completed step"
        )
    target_count = (
        int(segment_end_step) - int(checkpoint_step)
    ) * batch_size * accumulation
    migrated = copy.deepcopy(observed)
    migrated["optimization"]["scheduled_training_occurrence_count"] = target_count
    migrated = _contract(migrated)
    record = {
        "schema_version": MIGRATION_SCHEMA,
        "checkpoint_step": int(checkpoint_step),
        "segment_start_step": int(checkpoint_step) + 1,
        "segment_end_step": int(segment_end_step),
        "old_scheduled_training_occurrence_count": prior_count,
        "new_scheduled_training_occurrence_count": target_count,
        "old_contract_digest": str(observed.get("contract_digest") or ""),
        "new_contract_digest": str(migrated["contract_digest"]),
        "changed_contract_fields": [
            "optimization.scheduled_training_occurrence_count",
            "contract_digest",
        ],
    }
    return migrated, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--segment_end_step", required=True, type=int)
    parser.add_argument("--manifest_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input_checkpoint).resolve()
    destination = Path(args.output_checkpoint).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    for path in (destination, manifest_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Stage0 checkpoint payload must be a mapping")
    checkpoint_step = int(payload.get("step") or -1)
    observed = payload.get("run_contract")
    if not isinstance(observed, dict):
        raise ValueError("Stage0 checkpoint lacks a run contract")
    migrated, record = migrate_contract(
        observed,
        checkpoint_step=checkpoint_step,
        segment_end_step=int(args.segment_end_step),
    )
    migrated_payload = dict(payload)
    migrated_payload["run_contract"] = migrated
    migrated_payload["segment_resume_contract_migration"] = {
        **record,
        "source_checkpoint_path": str(source),
        "source_checkpoint_sha256": source_sha256,
        "non_contract_payload_objects_reused": True,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(migrated_payload, temporary)
    temporary.replace(destination)
    manifest = {
        **record,
        "status": "ok",
        "source_checkpoint_path": str(source),
        "source_checkpoint_sha256": source_sha256,
        "output_checkpoint_path": str(destination),
        "output_checkpoint_sha256": _sha256(destination),
        "source_checkpoint_preserved": _sha256(source) == source_sha256,
        "model_optimizer_trainer_payload_mutation": "none",
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
