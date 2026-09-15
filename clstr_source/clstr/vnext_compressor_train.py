from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from clstr.history_channel import audit_history_channel_rows
from clstr.model import CLSTRModel
from clstr.vnext_candidates import (
    label_natural_support,
    masked_topk_tensor,
    natural_compressed_support,
)
from clstr.vnext_data import runtime_visible_mask
from clstr.vnext_losses import natural_candidate_topk_coverage_loss
from clstr.vnext_stage0_train import (
    _filter_rows,
    _partition_stage0_eligible_rows,
    _positive_mask,
    _positive_ranks,
    _require_legal_stage0_positives,
    _rows_by_source,
    _stage0_source_family,
    _stage0_step_bucket,
)
from clstr.vnext_stage2_train import (
    _causal_route_state_text,
    _load_stage0_model,
    _prepare_trajectories,
)
from clstr.vnext_training import (
    candidate_foundation_digest,
    capture_vnext_source_manifest,
    configure_vnext_candidate_compressor,
    configure_vnext_static_route_adapter,
    derive_inventory_catalog_subset,
    file_sha256,
    immutable_run_contract,
    load_inventory_catalogs,
    load_or_build_frozen_text_cache,
    persist_vnext_source_manifest,
    read_jsonl,
    require_canonical_trainability,
    require_verified_data_contract,
    rewrite_inventory_catalog_references,
    seed_vnext_run,
    semantic_source_id,
    stable_stratified_cap_rows,
    static_foundation_digest,
    vnext_checkpoint_state,
    write_json,
)


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _initialize_static_reranker_scale(
    model: CLSTRModel,
    initial: float,
    *,
    objective_mode: str = "static_top500_rerank",
) -> dict[str, float | bool]:
    """Increase identity-reranker learning scale without changing step-zero scores."""

    if objective_mode == "static_route_query_residual":
        residual = model.vnext.static_route_query_delta
        output = residual.adapter.output
        component = "route_query_residual"
    elif objective_mode == "static_top500_rerank":
        residual = model.vnext.candidate_compressor
        output = residual.output
        component = "candidate_mlp"
    else:
        raise ValueError(f"unsupported identity reranker objective: {objective_mode}")
    if bool(torch.count_nonzero(output.weight.detach()).item()) or bool(
        torch.count_nonzero(output.bias.detach()).item()
    ):
        raise ValueError("static reranker scale reset requires an exact identity output")
    initial = float(initial)
    maximum = float(residual.scale.maximum)
    if not math.isfinite(initial) or not 0.0 < initial < maximum:
        raise ValueError("static reranker initial scale must be in (0, maximum)")
    ratio = initial / maximum
    raw = math.log(ratio / (1.0 - ratio))
    with torch.no_grad():
        residual.scale.raw.fill_(raw)
    realized = float(residual.scale().detach().float().cpu().item())
    if not math.isclose(realized, initial, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise RuntimeError("static reranker scale initialization is not exact")
    return {
        "identity_output_verified": True,
        "component": component,
        "initial": initial,
        "maximum": maximum,
        "realized": realized,
    }


def _static_reranker_scale_value(model: CLSTRModel, objective_mode: str) -> float:
    residual = (
        model.vnext.static_route_query_delta
        if objective_mode == "static_route_query_residual"
        else model.vnext.candidate_compressor
    )
    return float(residual.scale().detach().float().cpu().item())


def _static_reranker_protocol(objective_mode: str) -> str:
    if objective_mode == "static_route_query_residual":
        return "natural_top500_static_route_query_residual_v1"
    if objective_mode == "static_top500_rerank":
        return "natural_top500_memory_free_successor_route_reranker_v3"
    return "natural_top500_memory_free_compressor_top64_v1"


def _static_reranker_training_objective(objective_mode: str) -> str:
    if objective_mode == "static_route_query_residual":
        return "natural_top500_static_route_query_listwise_v1"
    if objective_mode == "static_top500_rerank":
        return "natural_top500_static_listwise_v1"
    return "top64_boundary_plus_weak_listwise_v1"


def _prepare_successor_route_rows(
    rows: list[dict[str, Any]],
    selected_skill_ids: set[str],
    catalogs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flatten verified ordered trajectories into per-decision route rows."""

    trajectories, report = _prepare_trajectories(
        rows,
        selected_skill_ids,
        catalogs,
        sequence_role="ordinary",
    )
    flattened: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for row in trajectory:
            copied = dict(row)
            copied["_vnext_kind"] = "trajectory_successor_route"
            copied["_vnext_current_query"] = str(row["state_text_current"])
            copied["_vnext_causal_query"] = _causal_route_state_text(row)
            copied["_vnext_query"] = copied["_vnext_causal_query"]
            copied["_vnext_query_channel"] = "compact_causal_skill_action_v1"
            flattened.append(copied)
    if not flattened:
        raise ValueError("static reranker has no verified ordered route rows")
    return flattened, report


def _schedule_compressor_batches(
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if not rows:
        raise ValueError("candidate compressor requires retrieval rows")
    if min(int(max_steps), int(gradient_accumulation_steps), int(batch_size)) <= 0:
        raise ValueError("compressor schedule dimensions must be positive")
    grouped = _rows_by_source(rows)
    sources = sorted(grouped)
    cursor = 0
    exposures: Counter[str] = Counter()
    schedule: list[list[dict[str, Any]]] = []
    for step in range(1, int(max_steps) + 1):
        for micro_step in range(int(gradient_accumulation_steps)):
            rng = random.Random(int(seed) + step * 1009 + micro_step)
            batch: list[dict[str, Any]] = []
            for offset in range(int(batch_size)):
                source = sources[(cursor + offset) % len(sources)]
                batch.append(rng.choice(grouped[source]))
                exposures[source] += 1
            cursor = (cursor + int(batch_size)) % len(sources)
            rng.shuffle(batch)
            schedule.append(batch)
    return schedule, {
        "protocol": "capability_source_row_round_robin_v1",
        "input_rows_by_source": {
            source: len(source_rows) for source, source_rows in sorted(grouped.items())
        },
        "sampled_exposures": dict(sorted(exposures.items())),
        "scheduled_batch_count": len(schedule),
    }


def _candidate_ranks(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[int | None]:
    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("compressor rank tensors must match [batch, candidates]")
    positive = positive_mask.to(device=logits.device, dtype=torch.bool)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    columns = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)
    ranks: list[int | None] = []
    for row_index in range(int(logits.size(0))):
        positions = (positive[row_index] & valid[row_index]).nonzero(
            as_tuple=False
        ).flatten()
        if not int(positions.numel()):
            ranks.append(None)
            continue
        positive_logits = logits[row_index].index_select(0, positions)
        best_position = positions[
            torch.argmax(positive_logits).to(dtype=torch.long)
        ]
        target_score = logits[row_index, best_position]
        rank = 1 + int(
            (
                valid[row_index]
                & (
                    (logits[row_index] > target_score)
                    | (
                        logits[row_index].eq(target_score)
                        & columns[0].lt(best_position)
                    )
                )
            )
            .sum()
            .detach()
            .cpu()
            .item()
        )
        ranks.append(rank)
    return ranks


def _natural_support_listwise_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Rank positives already present in natural support without injection."""

    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("static reranker tensors must match [batch, candidates]")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    eligible = positive.any(dim=-1) & (valid & ~positive).any(dim=-1)
    eligible_count = int(eligible.sum().detach().cpu().item())
    natural_miss_count = int((~positive.any(dim=-1)).sum().detach().cpu().item())
    if eligible_count <= 0:
        return logits.sum() * 0.0, {
            "eligible_rows": 0,
            "natural_miss_rows": natural_miss_count,
            "direct_topk_hit_rows": int(positive.any(dim=-1).sum().detach().cpu().item()),
            "recoverable_rows": 0,
            "mean_boundary_loss": 0.0,
            "mean_listwise_nll": 0.0,
        }
    selected_logits = logits[eligible].float()
    selected_valid = valid[eligible]
    selected_positive = positive[eligible]
    floor = torch.finfo(selected_logits.dtype).min
    all_logsumexp = torch.logsumexp(
        selected_logits.masked_fill(~selected_valid, floor),
        dim=-1,
    )
    positive_logsumexp = torch.logsumexp(
        selected_logits.masked_fill(~selected_positive, floor),
        dim=-1,
    )
    loss = (all_logsumexp - positive_logsumexp).mean().to(logits.dtype)
    return loss, {
        "eligible_rows": eligible_count,
        "natural_miss_rows": natural_miss_count,
        "direct_topk_hit_rows": int(positive.any(dim=-1).sum().detach().cpu().item()),
        "recoverable_rows": 0,
        "mean_boundary_loss": 0.0,
        "mean_listwise_nll": float(loss.detach().float().cpu().item()),
    }


def _strict_summary(
    records: list[dict[str, Any]],
    *,
    support_k: int = 64,
) -> dict[str, float | int]:
    count = len(records)
    if count <= 0:
        empty = {
            "row_count": 0,
            "full_pool_recall_at_100": 0.0,
            "full_pool_recall_at_500": 0.0,
            "direct_recall_at_64": 0.0,
            "compressed_recall_at_64": 0.0,
            "direct_mrr": 0.0,
            "compressed_mrr": 0.0,
            "compressed_minus_direct_recall_at_64": 0.0,
            "compressed_minus_direct_mrr": 0.0,
            "recovered_at_64": 0,
            "lost_at_64": 0,
            "net_recovered_at_64": 0,
        }
        return {
            **empty,
            "support_k": int(support_k),
            "base_recall_at_k": 0.0,
            "reranked_recall_at_k": 0.0,
            "base_mrr": 0.0,
            "reranked_mrr": 0.0,
            "reranked_minus_base_recall_at_k": 0.0,
            "reranked_minus_base_mrr": 0.0,
            "rank_improved_rows": 0,
            "rank_worsened_rows": 0,
            "net_rank_improved_rows": 0,
        }
    direct_ranks = [record["direct_rank"] for record in records]
    compressed_ranks = [record["compressed_rank"] for record in records]
    recovered = sum(
        direct_rank is None and compressed_rank is not None
        for direct_rank, compressed_rank in zip(direct_ranks, compressed_ranks)
    )
    lost = sum(
        direct_rank is not None and compressed_rank is None
        for direct_rank, compressed_rank in zip(direct_ranks, compressed_ranks)
    )
    direct_recall = sum(rank is not None for rank in direct_ranks) / count
    compressed_recall = sum(rank is not None for rank in compressed_ranks) / count
    direct_mrr = sum(0.0 if rank is None else 1.0 / rank for rank in direct_ranks) / count
    compressed_mrr = sum(
        0.0 if rank is None else 1.0 / rank for rank in compressed_ranks
    ) / count
    improved = sum(
        direct is not None and compressed is not None and compressed < direct
        for direct, compressed in zip(direct_ranks, compressed_ranks)
    )
    worsened = sum(
        direct is not None and compressed is not None and compressed > direct
        for direct, compressed in zip(direct_ranks, compressed_ranks)
    )
    legacy = {
        "row_count": count,
        "full_pool_recall_at_100": sum(
            int(int(record["full_pool_rank"]) <= 100) for record in records
        )
        / count,
        "full_pool_recall_at_500": sum(
            int(int(record["full_pool_rank"]) <= 500) for record in records
        )
        / count,
        "direct_recall_at_64": direct_recall,
        "compressed_recall_at_64": compressed_recall,
        "direct_mrr": direct_mrr,
        "compressed_mrr": compressed_mrr,
        "compressed_minus_direct_recall_at_64": compressed_recall - direct_recall,
        "compressed_minus_direct_mrr": compressed_mrr - direct_mrr,
        "recovered_at_64": int(recovered),
        "lost_at_64": int(lost),
        "net_recovered_at_64": int(recovered - lost),
    }
    return {
        **legacy,
        "support_k": int(support_k),
        "base_recall_at_k": direct_recall,
        "reranked_recall_at_k": compressed_recall,
        "base_mrr": direct_mrr,
        "reranked_mrr": compressed_mrr,
        "reranked_minus_base_recall_at_k": compressed_recall - direct_recall,
        "reranked_minus_base_mrr": compressed_mrr - direct_mrr,
        "rank_improved_rows": int(improved),
        "rank_worsened_rows": int(worsened),
        "net_rank_improved_rows": int(improved - worsened),
    }


def _compressor_validation_gates(report: dict[str, Any]) -> dict[str, Any]:
    def noninferior(summary: dict[str, Any]) -> bool:
        base_recall = float(
            summary.get("base_recall_at_k", summary.get("direct_recall_at_64", 0.0))
            or 0.0
        )
        reranked_recall = float(
            summary.get(
                "reranked_recall_at_k",
                summary.get("compressed_recall_at_64", 0.0),
            )
            or 0.0
        )
        base_mrr = float(summary.get("base_mrr", summary.get("direct_mrr", 0.0)) or 0.0)
        reranked_mrr = float(
            summary.get("reranked_mrr", summary.get("compressed_mrr", 0.0)) or 0.0
        )
        return bool(
            int(summary.get("row_count") or 0) > 0
            and reranked_recall >= base_recall
            and reranked_mrr >= base_mrr
        )

    gates = {
        name: noninferior(dict(report.get(name) or {}))
        for name in ("overall", "trajectory", "toolbench")
    }
    overall_trajectory = bool(gates["overall"] and gates["trajectory"])
    toolbench = bool(gates["toolbench"])
    return {
        "noninferior_to_base": overall_trajectory,
        "toolbench_noninferior_to_base": toolbench,
        # Backward-compatible aliases for old Top64 compressor diagnostics.
        "noninferior_to_direct_top64": overall_trajectory,
        "toolbench_noninferior_to_direct_top64": toolbench,
        "by_scope": gates,
        "pass": all(gates.values()),
    }


@torch.no_grad()
def evaluate_vnext_candidate_compressor(
    model: CLSTRModel,
    rows: list[dict[str, Any]],
    *,
    state_cache: Any,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    coarse_k: int,
    compressed_m: int,
    batch_size: int,
    autocast_dtype: torch.dtype,
    objective_mode: str = "topk_boundary",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    delta_abs_sum = torch.zeros((), device=device, dtype=torch.float32)
    delta_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    delta_count = torch.zeros((), device=device, dtype=torch.long)
    delta_abs_max = torch.zeros((), device=device, dtype=torch.float32)
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        causal_states = state_cache.batch(
            [str(row["_vnext_causal_query"]) for row in batch],
            device=device,
        )
        current_states = state_cache.batch(
            [str(row["_vnext_current_query"]) for row in batch],
            device=device,
        )
        legal = runtime_visible_mask(
            batch,
            skill_id_to_idx,
            device=device,
            inventory_catalogs=catalogs,
        )
        positive = _positive_mask(batch, skill_id_to_idx, device=device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            belief = model.vnext_initial_belief(
                causal_states,
                legal,
                top_k=belief_top_k,
            )
            causal_query, route_delta_query, _route_query = (
                model.vnext.static_query_components(
                causal_states,
                belief,
                )
            )
            full_logits = model.vnext_full_pool_logits(causal_query, head="recall")
            coarse_ids, coarse_valid = masked_topk_tensor(
                full_logits,
                legal,
                k=coarse_k,
            )
            # Proposal membership comes from the full-pool scorer, while the
            # fixed-support base uses the exact candidate kernel deployed by
            # Stage2.  This avoids a BF16 full-matmul/einsum identity drift.
            base_logits = model.vnext_candidate_logits(
                causal_query,
                coarse_ids,
                coarse_valid,
                head="route",
            )
            if objective_mode == "static_route_query_residual":
                route_delta_logits = model.vnext_candidate_logits(
                    route_delta_query,
                    coarse_ids,
                    coarse_valid,
                    head="route",
                ).masked_fill(
                    ~coarse_valid,
                    0.0,
                )
                reranked_logits = (base_logits + route_delta_logits).masked_fill(
                    ~coarse_valid,
                    torch.finfo(base_logits.dtype).min,
                )
                bounded_delta = route_delta_logits
            else:
                compression = model.vnext_candidate_compression_scores(
                    causal_query,
                    (
                        causal_states
                        if objective_mode == "static_top500_rerank"
                        else current_states
                    ),
                    coarse_ids,
                    coarse_valid,
                    base_logits,
                )
                reranked_logits = compression.logits
                bounded_delta = compression.bounded_delta
            valid_delta = bounded_delta[coarse_valid].float()
            if int(valid_delta.numel()):
                delta_abs_sum += valid_delta.abs().sum()
                delta_square_sum += valid_delta.square().sum()
                delta_count += int(valid_delta.numel())
                delta_abs_max = torch.maximum(delta_abs_max, valid_delta.abs().max())
            compressed = natural_compressed_support(
                coarse_ids,
                coarse_valid,
                reranked_logits,
                compressed_m=compressed_m,
            )
        full_ranks = _positive_ranks(full_logits.float(), positive, legal).cpu().tolist()
        direct_ids = coarse_ids[:, : int(compressed_m)]
        direct_valid = coarse_valid[:, : int(compressed_m)]
        direct_logits = base_logits[:, : int(compressed_m)]
        direct_labels = label_natural_support(direct_ids, direct_valid, positive)
        compressed_labels = label_natural_support(
            compressed.candidate_ids,
            compressed.valid_mask,
            positive,
        )
        direct_ranks = _candidate_ranks(
            direct_logits.float(),
            direct_labels.positive_mask,
            direct_valid,
        )
        compressed_ranks = _candidate_ranks(
            compressed.compression_logits.float(),
            compressed_labels.positive_mask,
            compressed.valid_mask,
        )
        for row, full_rank, direct_rank, compressed_rank in zip(
            batch,
            full_ranks,
            direct_ranks,
            compressed_ranks,
        ):
            records.append(
                {
                    "source": semantic_source_id(row),
                    "family": _stage0_source_family(row),
                    "step": _stage0_step_bucket(row),
                    "full_pool_rank": int(full_rank),
                    "direct_rank": direct_rank,
                    "compressed_rank": compressed_rank,
                }
            )
    per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        per_source[str(record["source"])].append(record)
        per_family[str(record["family"])].append(record)
        per_step[str(record["step"])].append(record)
    overall = _strict_summary(records, support_k=compressed_m)
    trajectory_records = [
        record
        for record in records
        if str(record["family"])
        in {"toolbench", "traject", "alfworld", "webshop", "agentgym"}
    ]
    trajectory = _strict_summary(trajectory_records, support_k=compressed_m)
    toolbench = _strict_summary(
        per_family.get("toolbench", []),
        support_k=compressed_m,
    )
    selection_score = 0.25 * (
        float(overall["reranked_recall_at_k"])
        + float(overall["reranked_mrr"])
        + float(trajectory["reranked_recall_at_k"])
        + float(trajectory["reranked_mrr"])
    )
    return {
        "protocol": _static_reranker_protocol(objective_mode),
        "coarse_k": int(coarse_k),
        "compressed_m": int(compressed_m),
        "positive_injection_count": 0,
        "overall": overall,
        "trajectory": trajectory,
        "toolbench": toolbench,
        "per_source": {
            key: _strict_summary(value, support_k=compressed_m)
            for key, value in sorted(per_source.items())
        },
        "per_family": {
            key: _strict_summary(value, support_k=compressed_m)
            for key, value in sorted(per_family.items())
        },
        "per_step": {
            key: _strict_summary(value, support_k=compressed_m)
            for key, value in sorted(per_step.items())
        },
        "selection_score": float(selection_score),
        "reranker_delta": {
            "valid_candidate_count": int(delta_count.detach().cpu().item()),
            "mean_absolute": float(
                (delta_abs_sum / delta_count.clamp_min(1)).detach().cpu().item()
            ),
            "rms": float(
                (delta_square_sum / delta_count.clamp_min(1))
                .sqrt()
                .detach()
                .cpu()
                .item()
            ),
            "absolute_max": float(delta_abs_max.detach().cpu().item()),
        },
    }


def _save_compressor_checkpoint(
    path: Path,
    *,
    model: CLSTRModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: dict[str, Any],
    trainability: dict[str, Any],
    skills_path: Path,
    static_digest: str,
    parent_stage0_checkpoint_sha256: str,
    run_contract: dict[str, Any],
    objective_mode: str,
    coarse_k: int,
    compressed_m: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "clstr_vnext_candidate_compressor",
            "step": int(step),
            "model_state_dict": vnext_checkpoint_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "trainability": trainability,
            "skills_path": str(skills_path.resolve()),
            "static_foundation_digest": static_digest,
            "candidate_foundation_digest": candidate_foundation_digest(model),
            "parent_stage0_checkpoint_sha256": parent_stage0_checkpoint_sha256,
            "objective_mode": str(objective_mode),
            "coarse_k": int(coarse_k),
            "compressed_m": int(compressed_m),
            "run_contract": run_contract,
        },
        temporary,
    )
    temporary.replace(path)


def train_vnext_candidate_compressor(
    *,
    stage0_checkpoint_path: str | Path,
    skills_path: str | Path,
    retrieval_rows_path: str | Path | None = None,
    retrieval_dev_rows_path: str | Path | None = None,
    trajectory_rows_path: str | Path | None = None,
    trajectory_dev_rows_path: str | Path | None = None,
    inventory_catalogs_path: str | Path,
    data_contract_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 1200,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1.0e-4,
    belief_top_k: int = 64,
    coarse_k: int = 500,
    compressed_m: int = 64,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    frozen_cache_dir: str | Path | None = None,
    seed: int = 37,
    checkpoint_interval: int = 300,
    validation_interval: int = 300,
    validation_batch_size: int = 16,
    max_dev_rows: int | None = 4096,
    max_train_rows: int | None = None,
    minimum_dev_score_gain: float = 0.0,
    coverage_margin: float = 0.05,
    recoverable_row_weight: float = 4.0,
    listwise_loss_weight: float = 0.05,
    objective_mode: str = "topk_boundary",
    static_reranker_scale_initial: float = 1.0,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(checkpoint_interval) != int(validation_interval):
        raise ValueError("compressor checkpoint and validation intervals must match")
    objective_mode = str(objective_mode or "topk_boundary").strip().lower()
    if objective_mode not in {
        "topk_boundary",
        "static_top500_rerank",
        "static_route_query_residual",
    }:
        raise ValueError(f"unsupported candidate objective mode: {objective_mode}")
    if objective_mode == "topk_boundary" and not 0 < int(compressed_m) < int(coarse_k):
        raise ValueError("top-k compressor requires 0 < compressed_m < coarse_k")
    if objective_mode in {"static_top500_rerank", "static_route_query_residual"} and not (
        int(coarse_k) == 500 and int(compressed_m) == int(coarse_k)
    ):
        raise ValueError("static reranker requires coarse_k=compressed_m=500")
    if objective_mode in {"static_top500_rerank", "static_route_query_residual"}:
        if trajectory_rows_path is None or trajectory_dev_rows_path is None:
            raise ValueError(
                "static reranker requires verified ordered trajectory train/dev rows"
            )
        training_rows_path = Path(trajectory_rows_path)
        dev_rows_path = Path(trajectory_dev_rows_path)
        training_row_kind = "trajectory_successor_route"
        verified_row_inputs = {
            "trajectory_rows": training_rows_path,
            "trajectory_dev_rows": dev_rows_path,
        }
    else:
        if retrieval_rows_path is None or retrieval_dev_rows_path is None:
            raise ValueError(
                "top-k compressor requires verified retrieval train/dev rows"
            )
        training_rows_path = Path(retrieval_rows_path)
        dev_rows_path = Path(retrieval_dev_rows_path)
        training_row_kind = "retrieval"
        verified_row_inputs = {
            "retrieval_rows": training_rows_path,
            "retrieval_dev_rows": dev_rows_path,
        }
    reproducibility = seed_vnext_run(seed)
    source_manifest = capture_vnext_source_manifest(
        Path(__file__).resolve().parents[1],
        (
            "clstr/model.py",
            "clstr/vnext_core.py",
            "clstr/vnext_candidates.py",
            "clstr/vnext_data.py",
            "clstr/vnext_losses.py",
            "clstr/vnext_training.py",
            "clstr/vnext_stage0_train.py",
            "clstr/vnext_stage2_train.py",
            "clstr/vnext_compressor_train.py",
            "scripts/run_clstr_vnext_candidate_compressor_train.py",
            "scripts/resolve_clstr_vnext_full_inputs.py",
            "scripts/sbatch/run_clstr_vnext_candidate_compressor.sh",
        ),
        require_clean=bool(require_clean_source),
    )
    source_manifest_record = persist_vnext_source_manifest(output_dir, source_manifest)
    data_contract = require_verified_data_contract(
        data_contract_path,
        {
            "training_skills": skills_path,
            **verified_row_inputs,
            "inventory_catalogs": inventory_catalogs_path,
        },
    )
    skills = read_jsonl(skills_path)
    selected_skill_ids = {_skill_id(row) for row in skills}
    catalogs, catalog_mapping, catalog_report = derive_inventory_catalog_subset(
        load_inventory_catalogs(inventory_catalogs_path),
        selected_skill_ids,
    )
    raw_train = read_jsonl(
        training_rows_path,
        max_rows=(
            None
            if objective_mode in {"static_top500_rerank", "static_route_query_residual"}
            else max_train_rows
        ),
    )
    raw_dev = read_jsonl(dev_rows_path)
    rewritten_train = rewrite_inventory_catalog_references(
        raw_train,
        catalog_mapping,
        catalogs,
    )
    rewritten_dev = rewrite_inventory_catalog_references(
        raw_dev,
        catalog_mapping,
        catalogs,
    )
    if objective_mode in {"static_top500_rerank", "static_route_query_residual"}:
        train_rows, train_sequence_report = _prepare_successor_route_rows(
            rewritten_train,
            selected_skill_ids,
            catalogs,
        )
        dev_rows, dev_sequence_report = _prepare_successor_route_rows(
            rewritten_dev,
            selected_skill_ids,
            catalogs,
        )
        train_rows, train_sampling = stable_stratified_cap_rows(
            train_rows,
            max_train_rows,
        )
    else:
        train_rows = _filter_rows(
            rewritten_train,
            selected_skill_ids,
            kind=training_row_kind,
        )
        dev_rows = _filter_rows(
            rewritten_dev,
            selected_skill_ids,
            kind=training_row_kind,
        )
        train_sequence_report = None
        dev_sequence_report = None
        train_sampling = {
            "protocol": "read_prefix_v1",
            "input_row_count": len(train_rows),
            "selected_row_count": len(train_rows),
        }
    _require_legal_stage0_positives(train_rows, catalogs)
    _require_legal_stage0_positives(dev_rows, catalogs)
    train_rows, train_eligibility = _partition_stage0_eligible_rows(
        train_rows,
        catalogs,
        kind="compressor_train",
    )
    dev_rows, dev_eligibility = _partition_stage0_eligible_rows(
        dev_rows,
        catalogs,
        kind="compressor_dev",
    )
    dev_rows, dev_sampling = stable_stratified_cap_rows(dev_rows, max_dev_rows)
    history_channel = audit_history_channel_rows(
        [*train_rows, *dev_rows],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    if history_channel.get("status") != "ok":
        raise ValueError("compressor data failed current/causal channel audit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("candidate compressor training requires a CUDA Slurm node")
    model, model_config, stage0_payload, backbone_snapshot = _load_stage0_model(
        stage0_checkpoint_path,
        skills_path,
        device=device,
    )
    static_digest = static_foundation_digest(model)
    if static_digest != str(stage0_payload.get("static_foundation_digest") or ""):
        raise ValueError("compressor parent Stage0 static foundation changed")
    static_reranker_initialization = (
        _initialize_static_reranker_scale(
            model,
            static_reranker_scale_initial,
            objective_mode=objective_mode,
        )
        if objective_mode in {"static_top500_rerank", "static_route_query_residual"}
        else None
    )
    trainability = (
        configure_vnext_static_route_adapter(model)
        if objective_mode == "static_route_query_residual"
        else configure_vnext_candidate_compressor(model)
    )
    require_canonical_trainability(trainability)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("candidate compressor has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate))
    schedule, sampling = _schedule_compressor_batches(
        train_rows,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        seed=seed,
    )
    cache_root = (
        Path(frozen_cache_dir)
        if frozen_cache_dir is not None
        else output_dir / "frozen_qwen_cache"
    )
    state_cache = load_or_build_frozen_text_cache(
        model,
        (
            text
            for row in [*train_rows, *dev_rows]
            for text in (
                str(row["_vnext_current_query"]),
                str(row["_vnext_causal_query"]),
            )
        ),
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    skill_id_to_idx = {_skill_id(row): index for index, row in enumerate(skills)}
    parent_sha = file_sha256(stage0_checkpoint_path)
    catalog_identity = immutable_run_contract(
        {
            "catalogs": [
                {
                    "catalog_id": catalog_id,
                    "digest": str(row.get("inventory_catalog_digest") or ""),
                    "pool_size": int(row.get("inventory_pool_size") or 0),
                }
                for catalog_id, row in sorted(catalogs.items())
            ]
        }
    )["contract_digest"]
    run_contract = immutable_run_contract(
        {
            "schema_version": (
                "clstr_vnext_static_route_query_residual_run_v1"
                if objective_mode == "static_route_query_residual"
                else "clstr_vnext_successor_route_reranker_run_v3"
                if objective_mode == "static_top500_rerank"
                else "clstr_vnext_candidate_compressor_run_v1"
            ),
            "inputs": {
                "stage0_checkpoint_sha256": parent_sha,
                "stage0_static_foundation_digest": static_digest,
                "skills_sha256": file_sha256(skills_path),
                "training_rows_kind": training_row_kind,
                "training_rows_sha256": file_sha256(training_rows_path),
                "dev_rows_sha256": file_sha256(dev_rows_path),
                "inventory_catalogs_sha256": file_sha256(inventory_catalogs_path),
                "data_contract_sha256": file_sha256(data_contract_path),
                "derived_catalog_identity": catalog_identity,
                "frozen_backbone_snapshot_contract": backbone_snapshot["contract"],
            },
            "model_config": model_config,
            "optimization": {
                "batch_size": int(batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "learning_rate": float(learning_rate),
                "belief_top_k": int(belief_top_k),
                "coarse_k": int(coarse_k),
                "compressed_m": int(compressed_m),
                "seed": int(seed),
                "checkpoint_interval": int(checkpoint_interval),
                "validation_interval": int(validation_interval),
                "validation_batch_size": int(validation_batch_size),
                "max_dev_rows": None if max_dev_rows is None else int(max_dev_rows),
                "max_train_rows": None if max_train_rows is None else int(max_train_rows),
                "minimum_dev_score_gain": float(minimum_dev_score_gain),
                "coverage_margin": float(coverage_margin),
                "recoverable_row_weight": float(recoverable_row_weight),
                "listwise_loss_weight": float(listwise_loss_weight),
                "training_objective": _static_reranker_training_objective(
                    objective_mode
                ),
                "objective_mode": objective_mode,
                "static_reranker_scale_initial": (
                    float(static_reranker_scale_initial)
                    if objective_mode
                    in {"static_top500_rerank", "static_route_query_residual"}
                    else None
                ),
                "sampling_protocol": "capability_source_row_round_robin_v1",
                "candidate_protocol": _static_reranker_protocol(objective_mode),
                "positive_injection_count": 0,
                "require_clean_source": bool(require_clean_source),
            },
            "frozen_cache_identity": state_cache.cache_identity,
            "source_contract_digest": source_manifest_record["source_contract_digest"],
        }
    )

    autocast_dtype = torch.bfloat16
    initial_validation = evaluate_vnext_candidate_compressor(
        model,
        dev_rows,
        state_cache=state_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
        batch_size=validation_batch_size,
        autocast_dtype=autocast_dtype,
        objective_mode=objective_mode,
    )
    initial_validation["quality_gates"] = _compressor_validation_gates(
        initial_validation
    )
    if objective_mode == "static_route_query_residual" and float(
        (initial_validation.get("reranker_delta") or {}).get("absolute_max") or 0.0
    ) != 0.0:
        raise RuntimeError(
            "zero-initialized static route adapter changed step-zero logits"
        )
    validation_records = [
        {
            "step": 0,
            "candidate_foundation_digest": candidate_foundation_digest(model),
            "reranker_scale": _static_reranker_scale_value(model, objective_mode),
            **initial_validation,
        }
    ]
    initial_score = float(initial_validation["selection_score"])
    initial_checkpoint = (
        output_dir / "checkpoints" / "clstr_vnext_compressor-step0.pt"
    )
    _save_compressor_checkpoint(
        initial_checkpoint,
        model=model,
        optimizer=optimizer,
        step=0,
        model_config=model_config,
        trainability=trainability,
        skills_path=Path(skills_path),
        static_digest=static_digest,
        parent_stage0_checkpoint_sha256=parent_sha,
        run_contract=run_contract,
        objective_mode=objective_mode,
        coarse_k=coarse_k,
        compressed_m=compressed_m,
    )

    optimizer.zero_grad(set_to_none=True)
    schedule_index = 0
    loss_curve: list[dict[str, Any]] = []
    gradient_norms: list[float] = []
    exposure: Counter[str] = Counter()
    started = time.perf_counter()
    for step in range(1, int(max_steps) + 1):
        step_loss = 0.0
        step_eligible = 0
        step_misses = 0
        step_direct_hits = 0
        step_recoverable = 0
        step_boundary_loss = 0.0
        step_listwise_nll = 0.0
        for _micro in range(int(gradient_accumulation_steps)):
            batch = schedule[schedule_index]
            schedule_index += 1
            for row in batch:
                exposure[semantic_source_id(row)] += 1
            causal_states = state_cache.batch(
                [str(row["_vnext_causal_query"]) for row in batch],
                device=device,
            )
            current_states = state_cache.batch(
                [str(row["_vnext_current_query"]) for row in batch],
                device=device,
            )
            legal = runtime_visible_mask(
                batch,
                skill_id_to_idx,
                device=device,
                inventory_catalogs=catalogs,
            )
            positive = _positive_mask(batch, skill_id_to_idx, device=device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            ):
                belief = model.vnext_initial_belief(
                    causal_states,
                    legal,
                    top_k=belief_top_k,
                )
                causal_query, _unused_route = model.vnext.static_queries(
                    causal_states,
                    belief,
                )
                full_logits = model.vnext_full_pool_logits(causal_query, head="recall")
                coarse_ids, coarse_valid = masked_topk_tensor(
                    full_logits,
                    legal,
                    k=coarse_k,
                )
                base_logits = model.vnext_candidate_logits(
                    causal_query,
                    coarse_ids,
                    coarse_valid,
                    head="route",
                )
                natural_labels = label_natural_support(
                    coarse_ids,
                    coarse_valid,
                    positive,
                )
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                if objective_mode == "static_route_query_residual":
                    _unused_recall, route_delta_query, _unused_route = (
                        model.vnext.static_query_components(
                            causal_states.detach(),
                            belief.detach(),
                        )
                    )
                    route_delta_logits = model.vnext_candidate_logits(
                        route_delta_query,
                        coarse_ids,
                        coarse_valid,
                        head="route",
                    ).masked_fill(~coarse_valid, 0.0)
                    reranked_logits = (base_logits.detach() + route_delta_logits).masked_fill(
                        ~coarse_valid,
                        torch.finfo(base_logits.dtype).min,
                    )
                    objective_loss, objective_report = _natural_support_listwise_nll(
                        reranked_logits,
                        natural_labels.positive_mask,
                        coarse_valid,
                    )
                else:
                    compression = model.vnext_candidate_compression_scores(
                        causal_query.detach(),
                        (
                            causal_states
                            if objective_mode == "static_top500_rerank"
                            else current_states
                        ),
                        coarse_ids,
                        coarse_valid,
                        base_logits.detach(),
                    )
                    if objective_mode == "static_top500_rerank":
                        objective_loss, objective_report = (
                            _natural_support_listwise_nll(
                                compression.logits,
                                natural_labels.positive_mask,
                                coarse_valid,
                            )
                        )
                    else:
                        loss_output = natural_candidate_topk_coverage_loss(
                            compression.logits,
                            base_logits.detach(),
                            natural_labels.positive_mask,
                            coarse_valid,
                            k=compressed_m,
                            margin=coverage_margin,
                            recoverable_weight=recoverable_row_weight,
                            listwise_weight=listwise_loss_weight,
                        )
                        objective_loss = loss_output.loss
                        objective_report = loss_output.report
                loss = objective_loss / int(gradient_accumulation_steps)
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"nonfinite compressor loss at step {step}")
            loss.backward()
            step_loss += float(objective_loss.detach().float().cpu().item())
            step_eligible += int(objective_report["eligible_rows"])
            step_misses += int(objective_report["natural_miss_rows"])
            step_direct_hits += int(objective_report["direct_topk_hit_rows"])
            step_recoverable += int(objective_report["recoverable_rows"])
            step_boundary_loss += float(objective_report["mean_boundary_loss"])
            step_listwise_nll += float(objective_report["mean_listwise_nll"])
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            .detach()
            .float()
            .cpu()
            .item()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite compressor gradient at step {step}")
        gradient_norms.append(gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_curve.append(
            {
                "step": int(step),
                "loss": step_loss / int(gradient_accumulation_steps),
                "eligible_rows": int(step_eligible),
                "natural_miss_rows": int(step_misses),
                "direct_topk_hit_rows": int(step_direct_hits),
                "recoverable_rows": int(step_recoverable),
                "mean_boundary_loss": step_boundary_loss
                / int(gradient_accumulation_steps),
                "mean_listwise_nll": step_listwise_nll
                / int(gradient_accumulation_steps),
                "positive_injection_count": 0,
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        should_validate = bool(
            step == int(max_steps)
            or (int(validation_interval) > 0 and step % int(validation_interval) == 0)
        )
        if should_validate:
            validation = evaluate_vnext_candidate_compressor(
                model,
                dev_rows,
                state_cache=state_cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
                batch_size=validation_batch_size,
                autocast_dtype=autocast_dtype,
                objective_mode=objective_mode,
            )
            validation["quality_gates"] = _compressor_validation_gates(validation)
            validation_records.append(
                {
                    "step": int(step),
                    "candidate_foundation_digest": candidate_foundation_digest(model),
                    "reranker_scale": _static_reranker_scale_value(
                        model,
                        objective_mode,
                    ),
                    **validation,
                }
            )
            checkpoint_path = (
                output_dir / "checkpoints" / f"clstr_vnext_compressor-step{step}.pt"
            )
            _save_compressor_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                model_config=model_config,
                trainability=trainability,
                skills_path=Path(skills_path),
                static_digest=static_digest,
                parent_stage0_checkpoint_sha256=parent_sha,
                run_contract=run_contract,
                objective_mode=objective_mode,
                coarse_k=coarse_k,
                compressed_m=compressed_m,
            )
            if static_foundation_digest(model) != static_digest:
                raise ValueError("compressor training changed the Stage0 foundation")

    passing_records = [
        record
        for record in validation_records
        if int(record["step"]) > 0
        and float(record["selection_score"]) - initial_score
        > float(minimum_dev_score_gain)
        and bool((record.get("quality_gates") or {}).get("pass"))
    ]
    selected_record = max(
        passing_records or validation_records,
        key=lambda record: (
            float(record["selection_score"]),
            -int(record["step"]),
        ),
    )
    best_step = int(selected_record["step"])
    best_score = float(selected_record["selection_score"])
    best_checkpoint = (
        output_dir
        / "checkpoints"
        / f"clstr_vnext_compressor-step{best_step}.pt"
    )
    score_gain = float(best_score - initial_score)
    selected_gates = dict(selected_record["quality_gates"])
    noninferior = bool(selected_gates["noninferior_to_base"])
    toolbench_noninferior = bool(selected_gates["toolbench_noninferior_to_base"])
    selected_checkpoint_sha256 = file_sha256(best_checkpoint)
    selected_candidate_digest = str(
        selected_record["candidate_foundation_digest"]
    )
    gradient_health_pass = bool(
        gradient_norms
        and all(math.isfinite(value) for value in gradient_norms)
        and any(value > 0.0 for value in gradient_norms)
    )
    status = (
        "ok"
        if passing_records and gradient_health_pass
        else "action_required"
    )
    selection = {
        "status": status,
        "initial_score": initial_score,
        "selected_score": float(best_score),
        "score_gain": score_gain,
        "selected_step": int(best_step),
        "selected_checkpoint_path": str(best_checkpoint.resolve()),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "parent_stage0_checkpoint_sha256": parent_sha,
        "selected_candidate_foundation_digest": selected_candidate_digest,
        "objective_mode": objective_mode,
        "candidate_protocol": _static_reranker_protocol(objective_mode),
        "selection_source": (
            "successor_route_dev_overall_trajectory_toolbench_noninferiority"
        ),
        "positive_injection_count": 0,
        "gradient_health_pass": gradient_health_pass,
        "noninferior_to_base": noninferior,
        "toolbench_noninferior_to_base": toolbench_noninferior,
        "noninferior_to_direct_top64": noninferior,
        "toolbench_noninferior_to_direct_top64": toolbench_noninferior,
        "selected_quality_gates": selected_gates,
        "validation_records": validation_records,
    }
    write_json(output_dir / "compressor_selection.json", selection)
    write_json(output_dir / "compressor_quality_gate.json", selection)
    report = {
        "status": status,
        "stage": "clstr_vnext_candidate_compressor",
        "training_objective": _static_reranker_training_objective(objective_mode),
        "objective_mode": objective_mode,
        "coverage_objective": {
            "top_k": int(compressed_m),
            "margin": float(coverage_margin),
            "recoverable_row_weight": float(recoverable_row_weight),
            "listwise_loss_weight": float(listwise_loss_weight),
        },
        "step": int(max_steps),
        "checkpoint_path": str(
            (
                output_dir
                / "checkpoints"
                / f"clstr_vnext_compressor-step{int(max_steps)}.pt"
            ).resolve()
        ),
        "parent_stage0_checkpoint_path": str(Path(stage0_checkpoint_path).resolve()),
        "parent_stage0_checkpoint_sha256": parent_sha,
        "static_foundation_digest": static_digest,
        "candidate_foundation_digest": selected_candidate_digest,
        "candidate_foundation_digest_final": candidate_foundation_digest(model),
        "data_contract": data_contract,
        "inventory_catalogs": catalog_report,
        "history_channel": history_channel,
        "loss_eligibility": {
            "train": train_eligibility,
            "dev": dev_eligibility,
        },
        "successor_route_data": {
            "training_row_kind": training_row_kind,
            "train_sequence_report": train_sequence_report,
            "dev_sequence_report": dev_sequence_report,
            "train_sampling": train_sampling,
        },
        "static_reranker_initialization": static_reranker_initialization,
        "static_reranker_scale_final": _static_reranker_scale_value(
            model,
            objective_mode,
        ),
        "dev_sampling": dev_sampling,
        "sampling": {
            **sampling,
            "cumulative_exposures": dict(sorted(exposure.items())),
        },
        "trainability": trainability,
        "cache": state_cache.report(),
        "frozen_backbone_snapshot": backbone_snapshot,
        "source_manifest": source_manifest_record,
        "reproducibility": reproducibility,
        "run_contract": run_contract,
        "loss_curve": loss_curve,
        "finite_loss": all(math.isfinite(float(row["loss"])) for row in loss_curve),
        "gradient_health": {
            "finite": all(math.isfinite(value) for value in gradient_norms),
            "step_count": len(gradient_norms),
            "nonzero_step_count": sum(value > 0.0 for value in gradient_norms),
            "minimum_nonzero": min(
                (value for value in gradient_norms if value > 0.0),
                default=0.0,
            ),
            "maximum": max(gradient_norms, default=0.0),
        },
        "selection": selection,
        "positive_injection_count": 0,
    }
    write_json(output_dir / "train_report.json", report)
    return report
