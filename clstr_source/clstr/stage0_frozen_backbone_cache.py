from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import time
from typing import Any

import torch

from clstr.history_channel import strip_history_sections


STAGE0_QUERY_INDEX_KEY = "_stage0_query_index"


@dataclass
class Stage0FrozenBackboneCache:
    pooled_cpu: torch.Tensor
    query_index_to_cache_row: dict[int, int]
    identity: dict[str, Any]
    build_seconds: float
    lookup_count: int = 0

    def project_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        projection_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        query_indices = [int(row[STAGE0_QUERY_INDEX_KEY]) for row in rows]
        missing = [
            query_index
            for query_index in query_indices
            if query_index not in self.query_index_to_cache_row
        ]
        if missing:
            raise RuntimeError(
                f"Stage0 frozen-backbone cache schedule miss: {missing[:8]}"
            )
        cache_rows = torch.tensor(
            [self.query_index_to_cache_row[query_index] for query_index in query_indices],
            dtype=torch.long,
        )
        pooled = self.pooled_cpu.index_select(0, cache_rows).to(
            device=device,
            non_blocking=True,
        )
        self.lookup_count += len(query_indices)
        return projection_fn(pooled)

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "schedule",
            "unique_row_count": int(self.pooled_cpu.size(0)),
            "hidden_size": int(self.pooled_cpu.size(1)),
            "dtype": str(self.pooled_cpu.dtype),
            "bytes": int(self.pooled_cpu.numel() * self.pooled_cpu.element_size()),
            "pinned_memory": bool(self.pooled_cpu.is_pinned()),
            "build_seconds": float(self.build_seconds),
            "lookup_count": int(self.lookup_count),
            "identity": dict(self.identity),
        }


def assign_stage0_query_indices(rows: Sequence[dict[str, Any]]) -> None:
    for query_index, row in enumerate(rows):
        row[STAGE0_QUERY_INDEX_KEY] = int(query_index)


def plan_stage0_scheduled_rows(
    *,
    queries: list[dict[str, Any]],
    start_step: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    sampling_strategy: str,
    source_buckets: dict[str, list[dict[str, Any]]],
    tempered_correction_fraction: float,
    batch_sampler: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    seen: set[int] = set()
    for step in range(int(start_step), int(max_steps) + 1):
        for micro_step in range(max(1, int(gradient_accumulation_steps))):
            micro_index = (
                (step - 1) * max(1, int(gradient_accumulation_steps))
                + micro_step
                + 1
            )
            batch = batch_sampler(
                queries,
                micro_index,
                batch_size,
                sampling_strategy=sampling_strategy,
                source_buckets=source_buckets,
                tempered_correction_fraction=tempered_correction_fraction,
            )
            for row in batch:
                if STAGE0_QUERY_INDEX_KEY not in row:
                    raise ValueError(
                        f"scheduled Stage0 row lacks {STAGE0_QUERY_INDEX_KEY}"
                    )
                query_index = int(row[STAGE0_QUERY_INDEX_KEY])
                if query_index in seen:
                    continue
                seen.add(query_index)
                planned.append(row)
    return planned


def _scheduled_query_index_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        query_index = int(row[STAGE0_QUERY_INDEX_KEY])
        digest.update(query_index.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def build_stage0_frozen_backbone_cache(
    model: Any,
    scheduled_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    identity: Mapping[str, Any],
) -> Stage0FrozenBackboneCache:
    if any(param.requires_grad for param in model.encoder.backbone.parameters()):
        raise ValueError(
            "Stage0 frozen-backbone cache requires a fully frozen backbone"
        )
    if not scheduled_rows:
        raise ValueError("Stage0 frozen-backbone cache requires scheduled rows")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Stage0 frozen-backbone cache batch_size must be positive")

    started = time.perf_counter()
    pooled_batches: list[torch.Tensor] = []
    for start in range(0, len(scheduled_rows), batch_size):
        batch = scheduled_rows[start : start + batch_size]
        texts = [
            model._serialize_state_for_encoder(
                strip_history_sections(str(row["query"]))
            )
            for row in batch
        ]
        with torch.no_grad():
            pooled = model.encoder.encode_backbone_pooled(texts)
        if pooled.ndim != 2 or pooled.size(0) != len(batch):
            raise ValueError(
                "Stage0 frozen-backbone cache encoder output must be "
                f"[batch, hidden], got {list(pooled.shape)}"
            )
        pooled_batches.append(
            pooled.detach().to(device="cpu").contiguous()
        )

    pooled_cpu = torch.cat(pooled_batches, dim=0).contiguous()
    if torch.cuda.is_available():
        pooled_cpu = pooled_cpu.pin_memory()
    query_index_to_cache_row = {
        int(row[STAGE0_QUERY_INDEX_KEY]): cache_row
        for cache_row, row in enumerate(scheduled_rows)
    }
    cache_identity = {
        **dict(identity),
        "scheduled_query_count": len(scheduled_rows),
        "scheduled_query_index_digest": _scheduled_query_index_digest(
            scheduled_rows
        ),
    }
    return Stage0FrozenBackboneCache(
        pooled_cpu=pooled_cpu,
        query_index_to_cache_row=query_index_to_cache_row,
        identity=cache_identity,
        build_seconds=time.perf_counter() - started,
    )
