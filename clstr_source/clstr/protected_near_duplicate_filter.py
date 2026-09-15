from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from clstr.clean_training_preflight import _row_sha256


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _match_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    leakage = report.get("leakage") or {}
    matches: list[dict[str, Any]] = []
    for protected in leakage.values():
        if not isinstance(protected, dict):
            continue
        for key, value in protected.items():
            if not key.endswith("_content_overlap") or not isinstance(value, dict):
                continue
            for row in value.get("matched_train_rows") or []:
                if isinstance(row, dict):
                    matches.append(dict(row))
    return matches


def _stream_filter(
    source: Path,
    target: Path,
    *,
    row_hashes: set[str],
    trajectory_ids: set[str],
    query_ids: set[str],
) -> dict[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    output_rows = 0
    removed_rows = 0
    with source.open("r", encoding="utf-8") as source_handle, target.open(
        "w",
        encoding="utf-8",
    ) as target_handle:
        for line_number, line in enumerate(source_handle, start=1):
            if not line.strip():
                continue
            input_rows += 1
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {source}:{line_number}")
            trajectory_id = str(value.get("trajectory_id") or "")
            provenance = value.get("provenance") if isinstance(value.get("provenance"), dict) else {}
            query_id = str(value.get("query_id") or provenance.get("query_id") or "")
            remove = bool(
                _row_sha256(value) in row_hashes
                or (trajectory_id and trajectory_id in trajectory_ids)
                or (query_id and query_id in query_ids)
            )
            if remove:
                removed_rows += 1
                continue
            target_handle.write(line if line.endswith("\n") else line + "\n")
            output_rows += 1
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_rows": removed_rows,
    }


def filter_protected_near_duplicates(
    *,
    input_root: str | Path,
    preflight_report_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    preflight_report_path = Path(preflight_report_path)
    report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    policy = report.get("leakage_policy") or {}
    contract = report.get("preflight_contract") or {}
    if not bool(contract.get("protected_near_duplicate_blocking")):
        raise ValueError("filter requires a blocking protected-near-duplicate audit")
    if bool(policy.get("has_exact_leakage")):
        raise ValueError("exact protected leakage must be resolved before near-duplicate filtering")
    if bool(policy.get("has_history_channel_blocker")):
        raise ValueError("history-channel blockers cannot be repaired by near-duplicate filtering")
    if not bool(policy.get("has_near_duplicate")):
        raise ValueError("preflight report contains no protected near duplicates")
    matches = _match_rows(report)
    if not matches:
        raise ValueError("preflight report lacks complete matched-train row identities")
    row_hashes = {
        str(row.get("train_row_sha256") or "")
        for row in matches
        if str(row.get("train_row_sha256") or "")
    }
    trajectory_ids = {
        str(row.get("trajectory_id") or "")
        for row in matches
        if str(row.get("trajectory_id") or "")
    }
    query_ids = {
        str(row.get("query_id") or "")
        for row in matches
        if str(row.get("query_id") or "")
    }
    output_root.mkdir(parents=True, exist_ok=False)
    trajectories = _stream_filter(
        input_root / "trajectories.jsonl",
        output_root / "trajectories.jsonl",
        row_hashes=row_hashes,
        trajectory_ids=trajectory_ids,
        query_ids=set(),
    )
    retrieval = _stream_filter(
        input_root / "retrieval.jsonl",
        output_root / "retrieval.jsonl",
        row_hashes=row_hashes,
        trajectory_ids=trajectory_ids,
        query_ids=query_ids,
    )
    shutil.copy2(input_root / "skill_pool.jsonl", output_root / "skill_pool.jsonl")
    manifest = {
        "status": "ok",
        "schema_version": "clstr_protected_near_duplicate_filter_v1",
        "input_root": str(input_root.resolve()),
        "preflight_report_path": str(preflight_report_path.resolve()),
        "preflight_report_sha256": _file_sha256(preflight_report_path),
        "matched_train_row_count": len(matches),
        "matched_row_sha256_count": len(row_hashes),
        "matched_trajectory_id_count": len(trajectory_ids),
        "matched_query_id_count": len(query_ids),
        "trajectories": trajectories,
        "retrieval": retrieval,
        "files_sha256": {
            name: _file_sha256(output_root / name)
            for name in (
                "skill_pool.jsonl",
                "retrieval.jsonl",
                "trajectories.jsonl",
            )
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest
