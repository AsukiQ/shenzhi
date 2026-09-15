from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from clstr.memory_utility_gate import positive_rank_and_utility


ROUTE_RECORD_SCHEMA_VERSION = "memory_utility_route_record_v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "memory_utility_route_manifest_v1"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()
        byte_view = tensor.reshape(-1).view(torch.uint8)
        return {
            "__tensor__": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported route-record digest value: {type(value).__name__}")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_chain_digest(
    checkpoints: dict[str, str | Path | None],
) -> str:
    entries: list[dict[str, Any]] = []
    for role, raw_path in sorted(checkpoints.items()):
        if raw_path is None:
            entries.append({"role": str(role), "present": False, "sha256": None})
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"route-record checkpoint is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            {
                "role": str(role),
                "present": True,
                "size": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    return canonical_digest(entries)


def default_route_manifest_path(records_path: str | Path) -> Path:
    return Path(records_path).with_suffix(".manifest.json")


def _rank_and_utility(
    logits: list[float],
    positive_mask: list[bool],
    valid_mask: list[bool],
) -> tuple[int, float]:
    tensor = torch.tensor(logits, dtype=torch.float32).view(1, -1)
    positive = torch.tensor(positive_mask, dtype=torch.bool).view(1, -1)
    valid = torch.tensor(valid_mask, dtype=torch.bool).view(1, -1)
    rank, utility, _eligible = positive_rank_and_utility(tensor, positive, valid)
    return int(rank.item()), float(utility.item())


def build_memory_utility_route_records(
    rows: list[dict[str, Any]],
    batch_output: Any,
    *,
    skill_id_to_idx: dict[str, int],
    sequential_benchmarks: set[str],
    source_start: int,
) -> list[dict[str, Any]]:
    candidate_rows = batch_output.candidate_union.candidate_rows
    if len(rows) != len(candidate_rows):
        raise ValueError("route record batch output must contain one candidate row per source row")
    ordered_skill_ids = [
        skill_id
        for skill_id, _idx in sorted(
            ((str(skill_id), int(idx)) for skill_id, idx in skill_id_to_idx.items()),
            key=lambda item: item[1],
        )
    ]
    records: list[dict[str, Any]] = []
    for row_idx, (row, candidate_indices) in enumerate(zip(rows, candidate_rows)):
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError("route recording requires a nonempty trajectory_id on every row")
        benchmark = str(row.get("source_benchmark") or row.get("benchmark") or "unknown")
        width = len(candidate_indices)
        if width:
            static_logits = [
                float(value)
                for value in batch_output.static_candidate_logits[row_idx, :width]
                .detach()
                .cpu()
                .tolist()
            ]
            dynamic_logits = [
                float(value)
                for value in batch_output.dynamic_candidate_logits[row_idx, :width]
                .detach()
                .cpu()
                .tolist()
            ]
            valid_mask = [
                bool(value)
                for value in batch_output.candidate_valid_mask[row_idx, :width]
                .detach()
                .cpu()
                .tolist()
            ]
            positive_mask = [
                bool(value)
                for value in batch_output.candidate_positive_mask[row_idx, :width]
                .detach()
                .cpu()
                .tolist()
            ]
            candidate_skill_ids = [ordered_skill_ids[int(index)] for index in candidate_indices]
        else:
            candidate_indices = [-1]
            candidate_skill_ids = ["__no_legal_candidate__"]
            static_logits = [0.0]
            dynamic_logits = [0.0]
            valid_mask = [False]
            positive_mask = [False]
        static_rank, static_utility = _rank_and_utility(
            static_logits,
            positive_mask,
            valid_mask,
        )
        dynamic_rank, dynamic_utility = _rank_and_utility(
            dynamic_logits,
            positive_mask,
            valid_mask,
        )
        row_digest = canonical_digest(
            {
                key: value
                for key, value in row.items()
                if not str(key).startswith("_")
            }
        )
        task_id = str(row.get("task_id") or row.get("row_id") or trajectory_id)
        records.append(
            {
                "schema_version": ROUTE_RECORD_SCHEMA_VERSION,
                "row_digest": row_digest,
                "source_row_index": int(source_start + row_idx),
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "benchmark": benchmark,
                "sequential_benchmark": benchmark in sequential_benchmarks,
                "causal_update_count": float(
                    batch_output.causal_update_count[row_idx].detach().cpu().item()
                ),
                "features": [
                    float(value)
                    for value in batch_output.features[row_idx].detach().cpu().tolist()
                ],
                "candidate_indices": [int(index) for index in candidate_indices],
                "candidate_skill_ids": candidate_skill_ids,
                "static_logits": static_logits,
                "dynamic_logits": dynamic_logits,
                "valid_mask": valid_mask,
                "positive_mask": positive_mask,
                "static_rank": static_rank,
                "dynamic_rank": dynamic_rank,
                "static_utility": static_utility,
                "dynamic_utility": dynamic_utility,
                "raw_alpha": float(batch_output.raw_alpha[row_idx].detach().cpu().item()),
                "effective_alpha": float(
                    batch_output.effective_alpha[row_idx].detach().cpu().item()
                ),
            }
        )
    return records


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} line {line_no}: expected JSON object")
        rows.append(loaded)
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_memory_utility_route_record_target(
    records_path: str | Path,
    *,
    manifest_identity: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> None:
    records_file = Path(records_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else default_route_manifest_path(records_file)
    )
    if not records_file.exists() and not manifest_file.exists():
        return
    if not records_file.is_file() or not manifest_file.is_file():
        raise ValueError("route records and companion manifest must either both exist or both be absent")
    manifest = _read_json(manifest_file)
    if manifest.get("schema_version") != ROUTE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported route record manifest schema")
    for key, expected in manifest_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"route record manifest identity mismatch: {key}")
    actual_digest = hashlib.sha256(records_file.read_bytes()).hexdigest()
    if manifest.get("records_sha256") != actual_digest:
        raise ValueError("route record manifest records_sha256 mismatch")
    if int(manifest.get("record_count", -1)) != len(_read_jsonl(records_file)):
        raise ValueError("route record manifest record_count mismatch")


def persist_memory_utility_route_records(
    records_path: str | Path,
    new_records: list[dict[str, Any]],
    *,
    manifest_identity: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    records_file = Path(records_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else default_route_manifest_path(records_file)
    )
    existing_records: list[dict[str, Any]] = []
    validate_memory_utility_route_record_target(
        records_file,
        manifest_identity=manifest_identity,
        manifest_path=manifest_file,
    )
    if records_file.exists() or manifest_file.exists():
        existing_records = _read_jsonl(records_file)

    existing_digests = {str(row.get("row_digest") or "") for row in existing_records}
    new_digests = [str(row.get("row_digest") or "") for row in new_records]
    if any(not digest for digest in new_digests):
        raise ValueError("route records require nonempty row_digest")
    if len(set(new_digests)) != len(new_digests) or any(
        digest in existing_digests for digest in new_digests
    ):
        raise ValueError("duplicate route record row_digest")
    combined = [*existing_records, *new_records]
    records_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in combined
    )
    records_digest = hashlib.sha256(records_text.encode("utf-8")).hexdigest()
    source_rows_digest = canonical_digest([row["row_digest"] for row in combined])
    manifest = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "records_path": (
            records_file.name
            if records_file.parent.resolve() == manifest_file.parent.resolve()
            else str(records_file.resolve())
        ),
        "records_sha256": records_digest,
        "record_count": len(combined),
        "source_rows_digest": source_rows_digest,
        **manifest_identity,
    }
    manifest_text = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_write_text(records_file, records_text)
    _atomic_write_text(manifest_file, manifest_text)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_file),
        "written_count": len(new_records),
        "total_count": len(combined),
    }
