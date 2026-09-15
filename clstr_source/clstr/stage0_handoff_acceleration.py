from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


ROW_SHARDED_CACHE_FORMAT = "row_sharded_v1"
ROW_SHARDED_CACHE_VERSION = 1
LEGACY_EXECUTION_SCHEDULE_VERSION = "legacy_split_batches_v1"


@dataclass(frozen=True)
class LegacyScheduleRowKeyPlan:
    row_keys: tuple[str, ...]
    batch_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RawStage0Candidates:
    current_indices: tuple[int, ...]
    current_scores: tuple[float, ...]
    next_indices: tuple[int, ...]
    next_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_indices", tuple(int(idx) for idx in self.current_indices))
        object.__setattr__(self, "current_scores", tuple(float(score) for score in self.current_scores))
        object.__setattr__(self, "next_indices", tuple(int(idx) for idx in self.next_indices))
        object.__setattr__(self, "next_scores", tuple(float(score) for score in self.next_scores))
        if len(self.current_indices) != len(self.current_scores):
            raise ValueError("current candidate indices and scores must match")
        if len(self.next_indices) != len(self.next_scores):
            raise ValueError("next candidate indices and scores must match")
        for label, indices, scores in (
            ("current", self.current_indices, self.current_scores),
            ("next", self.next_indices, self.next_scores),
        ):
            if any(idx < 0 for idx in indices):
                raise ValueError(f"{label} candidate indices must be nonnegative")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{label} candidate indices must be unique")
            if any(not math.isfinite(score) for score in scores):
                raise ValueError(f"{label} candidate scores must be finite")


def stable_json_digest(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_legacy_schedule_row_key_plan(
    current_queries: Sequence[str],
    next_queries: Sequence[str],
    ordered_inventory_skill_ids: Sequence[Sequence[str]],
    *,
    batch_size: int,
) -> LegacyScheduleRowKeyPlan:
    if not (
        len(current_queries)
        == len(next_queries)
        == len(ordered_inventory_skill_ids)
    ):
        raise ValueError("legacy schedule row-key inputs must have matching lengths")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("legacy schedule batch_size must be positive")

    row_keys: list[str] = []
    batch_ranges: list[tuple[int, int]] = []
    for start in range(0, len(current_queries), batch_size):
        end = min(len(current_queries), start + batch_size)
        batch_ranges.append((start, end))
        batch_signature = stable_json_digest(
            {
                "execution_schedule_version": LEGACY_EXECUTION_SCHEDULE_VERSION,
                "encode_batch_size": batch_size,
                "current_queries": [str(value) for value in current_queries[start:end]],
                "next_queries": [str(value) for value in next_queries[start:end]],
                "ordered_inventory_skill_ids": [
                    [str(item) for item in inventory]
                    for inventory in ordered_inventory_skill_ids[start:end]
                ],
            }
        )
        row_keys.extend(
            stable_json_digest(
                {
                    "batch_signature": batch_signature,
                    "row_offset": row_offset,
                }
            )
            for row_offset in range(end - start)
        )
    return LegacyScheduleRowKeyPlan(
        row_keys=tuple(row_keys),
        batch_ranges=tuple(batch_ranges),
    )


def stage0_handoff_global_identity(
    *,
    checkpoint_digest: str | dict[str, object],
    skills_digest: str,
    top_m: int,
    candidate_count: int,
    query_mode: str,
    inventory_min_candidates: int,
    initial_belief_top_k: int | None,
    state_text_format: str,
    skill_text_format: str,
    state_query_prompt_version: str = "unspecified",
    skill_embedding_digest: str,
    declared_pool_order_digest: str,
    candidate_selection_version: str,
    tie_break_policy: str,
    unified_static_scorer_digest: str,
    encode_batch_size: int = 8,
) -> dict[str, object]:
    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("row_sharded_v1 candidate_count must be positive")
    encode_batch_size = int(encode_batch_size)
    if encode_batch_size <= 0:
        raise ValueError("row_sharded_v1 encode_batch_size must be positive")
    identity: dict[str, object] = {
        "format": ROW_SHARDED_CACHE_FORMAT,
        "version": ROW_SHARDED_CACHE_VERSION,
        "static_candidate_scorer": "unified_static",
        "checkpoint_digest": checkpoint_digest,
        "skills_digest": str(skills_digest),
        "top_m": int(top_m),
        "candidate_count": candidate_count,
        "query_mode": str(query_mode),
        "inventory_min_candidates": int(inventory_min_candidates),
        "initial_belief_top_k": None if initial_belief_top_k is None else int(initial_belief_top_k),
        "state_text_format": str(state_text_format),
        "skill_text_format": str(skill_text_format),
        "state_query_prompt_version": str(state_query_prompt_version),
        "skill_embedding_digest": str(skill_embedding_digest),
        "declared_pool_order_digest": str(declared_pool_order_digest),
        "candidate_selection_version": str(candidate_selection_version),
        "tie_break_policy": str(tie_break_policy),
        "unified_static_scorer_digest": str(unified_static_scorer_digest),
        "encode_batch_size": encode_batch_size,
        "execution_schedule_version": LEGACY_EXECUTION_SCHEDULE_VERSION,
    }
    identity["global_key"] = stable_json_digest(identity)
    return identity


def row_sharded_cache_entry_dir(
    cache_root: str | Path,
    global_identity: dict[str, object],
) -> Path:
    global_key = str(global_identity.get("global_key") or "")
    if not global_key:
        raise ValueError("row_sharded_v1 global identity is missing global_key")
    return Path(cache_root) / ROW_SHARDED_CACHE_FORMAT / global_key[:2] / global_key


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _empty_manifest(global_identity: dict[str, object]) -> dict[str, object]:
    return {
        "format": ROW_SHARDED_CACHE_FORMAT,
        "version": ROW_SHARDED_CACHE_VERSION,
        "global_key": global_identity["global_key"],
        "global_identity": global_identity,
        "shards": [],
        "row_count": 0,
    }


def _validate_global_identity(global_identity: dict[str, object]) -> None:
    if global_identity.get("format") != ROW_SHARDED_CACHE_FORMAT:
        raise ValueError("row_sharded_v1 global identity has unsupported format")
    if global_identity.get("version") != ROW_SHARDED_CACHE_VERSION:
        raise ValueError("row_sharded_v1 global identity has unsupported version")
    global_key = str(global_identity.get("global_key") or "")
    identity_payload = {
        key: value
        for key, value in global_identity.items()
        if key != "global_key"
    }
    if global_key != stable_json_digest(identity_payload):
        raise ValueError("row_sharded_v1 global identity digest mismatch")
    try:
        candidate_count = int(global_identity["candidate_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row_sharded_v1 global identity has invalid candidate_count") from exc
    if candidate_count <= 0:
        raise ValueError("row_sharded_v1 global identity candidate_count must be positive")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(
    cache_root: str | Path,
    global_identity: dict[str, object],
) -> tuple[Path, dict[str, object] | None]:
    _validate_global_identity(global_identity)
    entry_dir = row_sharded_cache_entry_dir(cache_root, global_identity)
    manifest_path = entry_dir / "manifest.json"
    if not manifest_path.is_file():
        return entry_dir, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("row_sharded_v1 manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("row_sharded_v1 manifest must be a JSON object")
    expected_manifest_keys = {
        "format",
        "version",
        "global_key",
        "global_identity",
        "shards",
        "row_count",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("row_sharded_v1 manifest has unexpected fields")
    if manifest.get("format") != ROW_SHARDED_CACHE_FORMAT:
        raise ValueError("row_sharded_v1 manifest format mismatch")
    if manifest.get("version") != ROW_SHARDED_CACHE_VERSION:
        raise ValueError("row_sharded_v1 manifest version mismatch")
    if manifest.get("global_key") != global_identity.get("global_key"):
        raise ValueError("row_sharded_v1 manifest global key mismatch")
    if manifest.get("global_identity") != global_identity:
        raise ValueError("row_sharded_v1 manifest global identity mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("row_sharded_v1 manifest shards must be a list")
    total_rows = 0
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "row_count"}:
            raise ValueError("row_sharded_v1 manifest shard entry is invalid")
        shard_relative = Path(str(shard.get("path") or ""))
        if not str(shard_relative) or shard_relative.is_absolute() or ".." in shard_relative.parts:
            raise ValueError("row_sharded_v1 manifest shard path is unsafe")
        shard_digest = str(shard.get("sha256") or "")
        if len(shard_digest) != 64 or any(char not in "0123456789abcdef" for char in shard_digest):
            raise ValueError("row_sharded_v1 manifest shard digest is invalid")
        try:
            shard_rows = int(shard["row_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("row_sharded_v1 manifest shard row_count is invalid") from exc
        if shard_rows <= 0:
            raise ValueError("row_sharded_v1 manifest shard row_count must be positive")
        total_rows += shard_rows
    try:
        manifest_rows = int(manifest.get("row_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("row_sharded_v1 manifest row_count is invalid") from exc
    if manifest_rows != total_rows:
        raise ValueError("row_sharded_v1 manifest row_count does not match shards")
    return entry_dir, manifest


def _pack_raw_records(
    records: Sequence[tuple[str, RawStage0Candidates]],
    *,
    candidate_count: int,
) -> dict[str, object]:
    row_count = len(records)
    current_indices = torch.full((row_count, candidate_count), -1, dtype=torch.int32)
    current_scores = torch.full((row_count, candidate_count), float("nan"), dtype=torch.float32)
    next_indices = torch.full((row_count, candidate_count), -1, dtype=torch.int32)
    next_scores = torch.full((row_count, candidate_count), float("nan"), dtype=torch.float32)
    row_keys: list[str] = []
    for row_idx, (row_key, raw) in enumerate(records):
        row_keys.append(str(row_key))
        current_width = len(raw.current_indices)
        next_width = len(raw.next_indices)
        if current_width > candidate_count or next_width > candidate_count:
            raise ValueError("row_sharded_v1 raw candidate width exceeds candidate_count")
        if current_width:
            current_indices[row_idx, :current_width] = torch.tensor(raw.current_indices, dtype=torch.int32)
            current_scores[row_idx, :current_width] = torch.tensor(raw.current_scores, dtype=torch.float32)
        if next_width:
            next_indices[row_idx, :next_width] = torch.tensor(raw.next_indices, dtype=torch.int32)
            next_scores[row_idx, :next_width] = torch.tensor(raw.next_scores, dtype=torch.float32)
    return {
        "row_keys": tuple(row_keys),
        "current_indices": current_indices,
        "current_scores": current_scores,
        "next_indices": next_indices,
        "next_scores": next_scores,
    }


def _validate_shard_payload(
    payload: object,
    *,
    expected_rows: int,
    candidate_count: int,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("row_sharded_v1 shard payload must be a dictionary")
    expected_keys = {
        "row_keys",
        "current_indices",
        "current_scores",
        "next_indices",
        "next_scores",
    }
    if set(payload) != expected_keys:
        raise ValueError("row_sharded_v1 shard payload has unexpected fields")
    row_keys = payload["row_keys"]
    if not isinstance(row_keys, (list, tuple)) or len(row_keys) != expected_rows:
        raise ValueError("row_sharded_v1 shard row_keys do not match row_count")
    if any(not isinstance(row_key, str) or not row_key for row_key in row_keys):
        raise ValueError("row_sharded_v1 shard row_keys must be non-empty strings")
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("row_sharded_v1 shard contains duplicate row keys")

    tensor_specs = {
        "current_indices": torch.int32,
        "current_scores": torch.float32,
        "next_indices": torch.int32,
        "next_scores": torch.float32,
    }
    expected_shape = (expected_rows, candidate_count)
    for key, dtype in tensor_specs.items():
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"row_sharded_v1 shard {key} must be a tensor")
        if tensor.dtype != dtype:
            raise ValueError(f"row_sharded_v1 shard {key} must use {dtype}")
        if tensor.ndim != 2 or tuple(tensor.shape) != expected_shape:
            raise ValueError(f"row_sharded_v1 shard {key} has invalid shape")

    for prefix in ("current", "next"):
        indices = payload[f"{prefix}_indices"]
        scores = payload[f"{prefix}_scores"]
        for row_idx in range(expected_rows):
            index_row = indices[row_idx]
            score_row = scores[row_idx]
            valid_mask = index_row >= 0
            invalid_mask = ~valid_mask
            if bool((index_row < -1).any()):
                raise ValueError(f"row_sharded_v1 shard {prefix} indices contain invalid padding")
            valid_count = int(valid_mask.sum().item())
            if valid_count and not bool(valid_mask[:valid_count].all()):
                raise ValueError(f"row_sharded_v1 shard {prefix} indices padding is not trailing")
            if bool(valid_mask[valid_count:].any()):
                raise ValueError(f"row_sharded_v1 shard {prefix} indices padding is not trailing")
            if valid_count and not bool(torch.isfinite(score_row[:valid_count]).all()):
                raise ValueError(f"row_sharded_v1 shard {prefix} valid scores must be finite")
            if bool(invalid_mask.any()) and not bool(torch.isnan(score_row[invalid_mask]).all()):
                raise ValueError(f"row_sharded_v1 shard {prefix} padding scores must be NaN")
            valid_indices = [int(value) for value in index_row[:valid_count].tolist()]
            if len(valid_indices) != len(set(valid_indices)):
                raise ValueError(f"row_sharded_v1 shard {prefix} candidate indices must be unique")
    return payload


def _raw_record_from_shard(payload: dict[str, object], row_idx: int) -> RawStage0Candidates:
    current_indices_tensor = payload["current_indices"]
    current_scores_tensor = payload["current_scores"]
    next_indices_tensor = payload["next_indices"]
    next_scores_tensor = payload["next_scores"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            current_indices_tensor,
            current_scores_tensor,
            next_indices_tensor,
            next_scores_tensor,
        )
    ):
        raise ValueError("row_sharded_v1 shard candidate payload must contain tensors")
    current_mask = current_indices_tensor[row_idx] >= 0
    next_mask = next_indices_tensor[row_idx] >= 0
    return RawStage0Candidates(
        current_indices=tuple(int(value) for value in current_indices_tensor[row_idx][current_mask].tolist()),
        current_scores=tuple(float(value) for value in current_scores_tensor[row_idx][current_mask].tolist()),
        next_indices=tuple(int(value) for value in next_indices_tensor[row_idx][next_mask].tolist()),
        next_scores=tuple(float(value) for value in next_scores_tensor[row_idx][next_mask].tolist()),
    )


def _load_all_cached_records(
    entry_dir: Path,
    manifest: dict[str, object],
) -> dict[str, RawStage0Candidates]:
    records: dict[str, RawStage0Candidates] = {}
    candidate_count = int(manifest["global_identity"]["candidate_count"])
    for shard in manifest.get("shards") or []:
        shard_path = entry_dir / str(shard["path"])
        if not shard_path.is_file():
            raise ValueError(f"row_sharded_v1 shard is missing: {shard_path}")
        if file_sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"row_sharded_v1 shard digest mismatch: {shard_path}")
        try:
            loaded_payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"row_sharded_v1 shard is unreadable: {shard_path}") from exc
        payload = _validate_shard_payload(
            loaded_payload,
            expected_rows=int(shard["row_count"]),
            candidate_count=candidate_count,
        )
        for row_idx, row_key in enumerate(payload["row_keys"]):
            row_key = str(row_key)
            record = _raw_record_from_shard(payload, row_idx)
            existing = records.get(row_key)
            if existing is not None and existing != record:
                raise ValueError(f"row_sharded_v1 conflicting duplicate row key: {row_key}")
            records[row_key] = record
    return records


def append_row_sharded_cache(
    cache_root: str | Path,
    global_identity: dict[str, object],
    records: dict[str, RawStage0Candidates],
    *,
    shard_size: int = 2048,
) -> dict[str, object]:
    shard_size = int(shard_size)
    if shard_size <= 0:
        raise ValueError("row_sharded_v1 shard_size must be positive")
    entry_dir, manifest = _load_manifest(cache_root, global_identity)
    manifest = _empty_manifest(global_identity) if manifest is None else manifest
    existing_records = _load_all_cached_records(entry_dir, manifest) if manifest["shards"] else {}
    existing_rows = 0
    new_records: dict[str, RawStage0Candidates] = {}
    for row_key, record in records.items():
        row_key = str(row_key)
        if not row_key:
            raise ValueError("row_sharded_v1 row key must be non-empty")
        existing = existing_records.get(row_key)
        if existing is None:
            new_records[row_key] = record
            continue
        if existing != record:
            raise ValueError(f"row_sharded_v1 conflicting duplicate row key: {row_key}")
        existing_rows += 1
    entry_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = entry_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    items = list(new_records.items())
    new_shards: list[dict[str, object]] = []
    candidate_count = int(global_identity["candidate_count"])
    for start in range(0, len(items), shard_size):
        chunk = items[start : start + shard_size]
        shard_name = f"shard-{uuid.uuid4().hex}.pt"
        shard_path = shards_dir / shard_name
        temporary = shards_dir / f".{shard_name}.{uuid.uuid4().hex}.tmp"
        torch.save(_pack_raw_records(chunk, candidate_count=candidate_count), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, shard_path)
        new_shards.append(
            {
                "path": str(shard_path.relative_to(entry_dir)),
                "sha256": file_sha256(shard_path),
                "row_count": len(chunk),
            }
        )
    manifest_shards = list(manifest.get("shards") or []) + new_shards
    manifest["shards"] = manifest_shards
    manifest["row_count"] = int(manifest.get("row_count") or 0) + len(items)
    _atomic_write_json(entry_dir / "manifest.json", manifest)
    return {
        "format": ROW_SHARDED_CACHE_FORMAT,
        "cache_path": str(entry_dir),
        "written_rows": len(items),
        "written_shards": len(new_shards),
        "cached_rows": int(manifest["row_count"]),
        "existing_rows": existing_rows,
    }


def reset_row_sharded_cache(
    cache_root: str | Path,
    global_identity: dict[str, object],
) -> dict[str, object]:
    entry_dir, manifest = _load_manifest(cache_root, global_identity)
    previous_rows = 0 if manifest is None else int(manifest["row_count"])
    entry_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(entry_dir / "manifest.json", _empty_manifest(global_identity))
    return {
        "format": ROW_SHARDED_CACHE_FORMAT,
        "cache_path": str(entry_dir),
        "previous_rows": previous_rows,
        "cached_rows": 0,
    }


def load_row_sharded_cache(
    cache_root: str | Path,
    global_identity: dict[str, object],
    requested_row_keys: Sequence[str],
) -> tuple[dict[str, RawStage0Candidates], dict[str, object]]:
    entry_dir, manifest = _load_manifest(cache_root, global_identity)
    if manifest is None:
        return {}, {
            "format": ROW_SHARDED_CACHE_FORMAT,
            "cache_path": str(entry_dir),
            "cached_rows": 0,
            "hit_rows": 0,
            "miss_rows": len(requested_row_keys),
        }
    cached_records = _load_all_cached_records(entry_dir, manifest)
    requested = [str(row_key) for row_key in requested_row_keys]
    hits = {row_key: cached_records[row_key] for row_key in requested if row_key in cached_records}
    return hits, {
        "format": ROW_SHARDED_CACHE_FORMAT,
        "cache_path": str(entry_dir),
        "cached_rows": len(cached_records),
        "hit_rows": sum(1 for row_key in requested if row_key in cached_records),
        "miss_rows": sum(1 for row_key in requested if row_key not in cached_records),
    }
