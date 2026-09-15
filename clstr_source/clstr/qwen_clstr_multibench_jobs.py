from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_multibench_submit import QWEN_CLSTR_MULTIBENCH_STAGE_ORDER


SacctRunner = Callable[[list[str]], str]


def _read_registry(path: str | Path) -> tuple[Path, dict[str, Any]]:
    registry_path = Path(path).resolve()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid multibench submission registry: {registry_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("multibench submission registry must contain an object")
    return registry_path, payload


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _normalize_state(value: str) -> str:
    token = str(value or "").strip().split(maxsplit=1)[0] if str(value or "").strip() else ""
    return token.rstrip("+")


def collect_qwen_clstr_multibench_job_evidence(
    *,
    registry_path: str | Path,
    output_path: str | Path,
    sacct_runner: SacctRunner,
) -> dict[str, Any]:
    resolved_registry, registry = _read_registry(registry_path)
    stage_order = registry.get("stage_order")
    if tuple(stage_order or ()) != QWEN_CLSTR_MULTIBENCH_STAGE_ORDER:
        raise ValueError("submission registry stage order mismatch")
    if registry.get("status") != "submitted":
        raise ValueError("submission registry is not fully submitted")
    stored_stages = registry.get("stages")
    if not isinstance(stored_stages, Mapping):
        raise ValueError("submission registry stages must be an object")

    job_id_by_stage: dict[str, str] = {}
    for stage in stage_order:
        entry = stored_stages.get(stage)
        if not isinstance(entry, Mapping) or entry.get("status") != "submitted":
            raise ValueError(f"submission registry stage is not submitted: {stage}")
        job_id = str(entry.get("job_id") or "").strip()
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError(f"invalid Slurm job ID for stage {stage}")
        job_id_by_stage[stage] = job_id

    job_ids = [job_id_by_stage[stage] for stage in stage_order]
    command = [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode",
    ]
    raw_output = str(sacct_runner(command) or "")
    rows_by_job_id: dict[str, list[dict[str, str]]] = {}
    expected_job_ids = set(job_ids)
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id = parts[0].strip()
        if job_id not in expected_job_ids:
            continue
        rows_by_job_id.setdefault(job_id, []).append(
            {
                "job_id": job_id,
                "state": _normalize_state(parts[1]),
                "exit_code": parts[2].strip(),
            }
        )

    blockers: list[str] = []
    jobs: dict[str, dict[str, str]] = {}
    for stage in stage_order:
        job_id = job_id_by_stage[stage]
        rows = rows_by_job_id.get(job_id) or []
        if not rows:
            blockers.append(f"{stage}:missing_job_state")
            continue
        if len(rows) != 1:
            blockers.append(f"{stage}:duplicate_job_state")
            continue
        row = rows[0]
        jobs[stage] = row
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
            blockers.append(
                f"{stage}:job_not_completed:{row['state']}:{row['exit_code']}"
            )

    evidence: dict[str, Any] = {
        "schema_version": "qwen06_clstr_multibench_job_evidence_v1",
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "stage_order": list(stage_order),
        "input_fingerprint": registry.get("input_fingerprint"),
        "plan_fingerprint": registry.get("plan_fingerprint"),
        "submission_identity": registry.get("submission_identity"),
        "registry_path": str(resolved_registry),
        "registry_sha256": hashlib.sha256(resolved_registry.read_bytes()).hexdigest(),
        "jobs": jobs,
    }
    evidence["evidence_sha256"] = canonical_digest(evidence)
    _atomic_write_json(output_path, evidence)
    return evidence
