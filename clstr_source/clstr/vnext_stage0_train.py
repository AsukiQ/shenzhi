from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from clstr.history_channel import (
    audit_history_channel_rows,
    compact_causal_state_from_row,
)
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.stage_checkpoint_init import load_compatible_state_dict
from clstr.vnext_data import (
    runtime_visible_mask,
    runtime_visible_skill_id_set,
)
from clstr.vnext_losses import (
    dense_full_pool_nll_and_hard_negative,
)
from clstr.vnext_training import (
    capture_vnext_source_manifest,
    configure_vnext_stage0,
    derive_inventory_catalog_subset,
    file_sha256,
    immutable_run_contract,
    load_inventory_catalogs,
    load_or_capture_frozen_backbone_snapshot,
    load_or_build_frozen_text_cache,
    load_or_build_skill_embedding_cache,
    mapped_legacy_stage0_state,
    move_optimizer_state_to_device,
    persist_vnext_source_manifest,
    read_jsonl,
    require_canonical_trainability,
    require_canonical_vnext_checkpoint_state,
    require_matching_run_contract,
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


def _checkpoint_state(model: CLSTRModel) -> dict[str, torch.Tensor]:
    return vnext_checkpoint_state(model)


def _save_checkpoint(
    path: Path,
    *,
    model: CLSTRModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: dict[str, Any],
    trainability: dict[str, Any],
    skills_path: Path,
    run_contract: dict[str, Any],
    trainer_progress: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": "clstr_vnext_stage0",
            "step": int(step),
            "model_state_dict": _checkpoint_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "trainability": trainability,
            "skills_path": str(skills_path.resolve()),
            "static_foundation_digest": static_foundation_digest(model),
            "run_contract": run_contract,
            "trainer_progress": trainer_progress or {},
        },
        temporary,
    )
    temporary.replace(path)


def _select_skills(
    skills: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    *,
    max_skills: int | None,
) -> list[dict[str, Any]]:
    if max_skills is None or int(max_skills) <= 0 or int(max_skills) >= len(skills):
        return skills
    required: list[str] = []
    for row in retrieval_rows:
        required.extend(str(item) for item in row.get("required_tool_set_skill_ids") or [])
    for row in static_rows:
        required.extend(
            str(item)
            for item in row.get("current_state_route_set_skill_ids") or []
        )
    required_set = {item for item in required if item}
    by_id = {_skill_id(row): row for row in skills}
    missing = sorted(required_set - set(by_id))
    if missing:
        raise ValueError(f"Stage0 rows reference unknown skills: {missing[:4]}")
    selected_ids = list(sorted(required_set))
    if len(selected_ids) > int(max_skills):
        raise ValueError(
            "max_skills is smaller than the complete positive set required by selected Stage0 rows"
        )
    selected_set = set(selected_ids)
    for row in skills:
        skill_id = _skill_id(row)
        if len(selected_ids) >= int(max_skills):
            break
        if skill_id and skill_id not in selected_set:
            selected_ids.append(skill_id)
            selected_set.add(skill_id)
    return [by_id[skill_id] for skill_id in selected_ids]


def _filter_rows(
    rows: list[dict[str, Any]],
    selected_skill_ids: set[str],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["_vnext_kind"] = kind
        current_query = str(row.get("state_text_current") or "").strip()
        if not current_query:
            raise ValueError(
                "canonical Stage0 row lacks an explicit history-free state_text_current"
            )
        causal_query = compact_causal_state_from_row(row)
        copied["_vnext_current_query"] = current_query
        copied["_vnext_causal_query"] = causal_query
        copied["_vnext_query"] = causal_query
        copied["_vnext_query_channel"] = "compact_causal_skill_action_v1"
        if kind == "retrieval":
            positives = [
                str(item)
                for item in row.get("required_tool_set_skill_ids") or []
                if str(item) in selected_skill_ids
            ]
        elif kind == "static_route":
            positives = [
                str(item)
                for item in row.get("current_state_route_set_skill_ids") or []
                if str(item) in selected_skill_ids
            ]
        else:
            raise ValueError(f"unsupported Stage0 row kind: {kind}")
        if not positives:
            continue
        copied["_vnext_positive_skill_ids"] = sorted(set(positives))
        output.append(copied)
    return output


def _schedule_batches(
    retrieval_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    *,
    start_step: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if not retrieval_rows or not static_rows:
        raise ValueError("vNext Stage0 requires both retrieval and static-route rows")
    if int(batch_size) < 2:
        raise ValueError("source-balanced Stage0 requires batch_size >= 2")
    grouped = {
        "retrieval": _rows_by_source(retrieval_rows),
        "static_route": _rows_by_source(static_rows),
    }
    schedule: list[list[dict[str, Any]]] = []
    exposure: Counter[str] = Counter()
    half = max(1, int(batch_size) // 2)
    counts_per_batch = {
        "retrieval": half,
        "static_route": int(batch_size) - half,
    }
    completed_batches = max(0, int(start_step) - 1) * int(gradient_accumulation_steps)
    cursors = {
        kind: (completed_batches * count) % len(grouped[kind])
        for kind, count in counts_per_batch.items()
    }
    for step in range(int(start_step), int(max_steps) + 1):
        for micro_step in range(int(gradient_accumulation_steps)):
            rng = random.Random(int(seed) + step * 1009 + micro_step)
            batch: list[dict[str, Any]] = []
            for kind, count in (
                ("retrieval", counts_per_batch["retrieval"]),
                ("static_route", counts_per_batch["static_route"]),
            ):
                sources = sorted(grouped[kind])
                for offset in range(int(count)):
                    source = sources[(cursors[kind] + offset) % len(sources)]
                    batch.append(rng.choice(grouped[kind][source]))
                    exposure[f"{kind}:{source}"] += 1
                cursors[kind] = (cursors[kind] + int(count)) % len(sources)
            rng.shuffle(batch)
            schedule.append(batch)
    return schedule, {
        "protocol": "capability_source_row_round_robin_v1",
        "input_rows_by_kind_source": {
            kind: {source: len(rows) for source, rows in sorted(groups.items())}
            for kind, groups in sorted(grouped.items())
        },
        "sampled_exposures": dict(sorted(exposure.items())),
        "scheduled_batch_count": len(schedule),
        "start_step": int(start_step),
    }


def _rows_by_source(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[semantic_source_id(row)].append(row)
    if not grouped:
        raise ValueError("source-balanced sampling received an empty row group")
    return {source: grouped[source] for source in sorted(grouped)}


def _positive_mask(
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(len(rows), len(skill_id_to_idx), dtype=torch.bool, device=device)
    for row_idx, row in enumerate(rows):
        indices = [
            skill_id_to_idx[skill_id]
            for skill_id in row["_vnext_positive_skill_ids"]
            if skill_id in skill_id_to_idx
        ]
        if indices:
            mask[row_idx, torch.tensor(indices, dtype=torch.long, device=device)] = True
    return mask


def _positive_set_cardinality_report(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    distribution: Counter[int] = Counter()
    multi_positive_by_source: Counter[str] = Counter()
    for row in rows:
        count = len(
            {
                str(skill_id)
                for skill_id in row.get("_vnext_positive_skill_ids") or []
                if str(skill_id)
            }
        )
        distribution[count] += 1
        if count > 1:
            multi_positive_by_source[semantic_source_id(row)] += 1
    return {
        "row_count": len(rows),
        "positive_count_distribution": {
            str(count): int(rows_with_count)
            for count, rows_with_count in sorted(distribution.items())
        },
        "multi_positive_row_count": int(
            sum(
                rows_with_count
                for count, rows_with_count in distribution.items()
                if count > 1
            )
        ),
        "multi_positive_by_source": dict(sorted(multi_positive_by_source.items())),
    }


def _require_legal_stage0_positives(
    rows: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        positives = {str(item) for item in row.get("_vnext_positive_skill_ids") or []}
        legal = runtime_visible_skill_id_set(row, catalogs)
        if not positives or not positives.issubset(legal):
            raise ValueError("Stage0 row contains a positive outside its runtime inventory")


def _partition_stage0_eligible_rows(
    rows: list[dict[str, Any]],
    catalogs: dict[str, dict[str, Any]],
    *,
    kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exclude rows whose legal pool provides no ranking supervision."""

    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    excluded_by_source: Counter[str] = Counter()
    for row in rows:
        positives = {str(item) for item in row.get("_vnext_positive_skill_ids") or []}
        legal = runtime_visible_skill_id_set(row, catalogs)
        if not positives:
            reason = "no_positive"
        elif not positives.issubset(legal):
            raise ValueError("Stage0 row contains a positive outside its runtime inventory")
        elif not (legal - positives):
            reason = "no_legal_negative"
        else:
            eligible.append(row)
            continue
        excluded[reason] += 1
        excluded_by_source[f"{semantic_source_id(row)}:{reason}"] += 1
    if not eligible:
        raise ValueError(f"Stage0 {kind} has no discriminative rows with a legal negative")
    return eligible, {
        "kind": kind,
        "input_row_count": len(rows),
        "eligible_row_count": len(eligible),
        "excluded_row_count": len(rows) - len(eligible),
        "excluded_reasons": dict(sorted(excluded.items())),
        "excluded_by_source_reason": dict(sorted(excluded_by_source.items())),
    }


def _build_model(
    skills: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    model_dim: int,
    max_length: int,
    torch_dtype: str,
    skill_table_batch_size: int,
    belief_top_k: int,
    local_files_only: bool,
    frozen_backbone_snapshot_digest: str | None = None,
    frozen_backbone_snapshot_manifest_path: str | None = None,
) -> tuple[CLSTRModel, dict[str, Any]]:
    config = CLSTRConfig(
        base_model_name=model_name_or_path,
        d=int(model_dim),
        d_a=int(model_dim),
        top_k=100,
        encoder_pooling="last_token",
        cross_encoder_pooling="last_token",
        tokenizer_padding_side="left",
        torch_dtype=torch_dtype,
        freeze_backbone=True,
        trust_remote_code=True,
        max_length=int(max_length),
        projection_init="identity",
        normalize_embeddings=True,
        local_files_only=bool(local_files_only),
        defer_skill_table_init=True,
        skill_text_format="skillret_official",
        state_query_prompt_version="clstr_causal_state_v1",
        state_query_max_chars=2000,
        state_query_truncation="head_tail_v1",
        skill_table_batch_size=int(skill_table_batch_size),
        skill_table_adapter_init="identity",
        use_cross_encoder=False,
        initial_belief_top_k=int(belief_top_k),
        frozen_backbone_snapshot_digest=frozen_backbone_snapshot_digest,
        frozen_backbone_snapshot_manifest_path=(
            frozen_backbone_snapshot_manifest_path
        ),
    )
    return CLSTRModel(config, skills), vars(config)


def _stable_cap_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    selected, _report = stable_stratified_cap_rows(rows, limit)
    return selected


def _positive_ranks(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    masked_positive = logits.masked_fill(~(positive & legal), floor)
    best_values, best_indices = masked_positive.max(dim=-1)
    skill_ids = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)
    return 1 + (
        legal
        & (
            (logits > best_values.unsqueeze(-1))
            | (
                logits.eq(best_values.unsqueeze(-1))
                & skill_ids.lt(best_indices.unsqueeze(-1))
            )
        )
    ).sum(dim=-1)


def _ranking_metrics(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
) -> dict[str, float]:
    ranks = _positive_ranks(logits, positive, legal)
    reciprocal = ranks.float().reciprocal()
    return {
        "mrr_sum": float(reciprocal.sum().cpu().item()),
        "recall_at_1_sum": float((ranks <= 1).float().sum().cpu().item()),
        "recall_at_10_sum": float((ranks <= 10).float().sum().cpu().item()),
        "recall_at_100_sum": float((ranks <= 100).float().sum().cpu().item()),
        "recall_at_500_sum": float((ranks <= 500).float().sum().cpu().item()),
        "row_count": float(ranks.numel()),
    }


def _all_positive_topk_coverage(
    logits: torch.Tensor,
    positive: torch.Tensor,
    legal: torch.Tensor,
    *,
    k: int = 500,
) -> tuple[dict[str, float], torch.Tensor]:
    """Report deployment support for every legal positive in each row.

    Ties at the TopK boundary follow the same lower-skill-index convention as
    `_positive_ranks`.  The returned row mask is true only when every positive
    for that row is naturally present in TopK.
    """

    if (
        logits.ndim != 2
        or positive.shape != logits.shape
        or legal.shape != logits.shape
    ):
        raise ValueError("all-positive coverage inputs must match [batch, skills]")
    k = min(int(k), int(logits.size(1)))
    if k <= 0:
        raise ValueError("all-positive coverage k must be positive")
    legal_positive = positive.to(dtype=torch.bool) & legal.to(dtype=torch.bool)
    positive_count = legal_positive.sum(dim=-1)
    eligible = positive_count > 0
    if not bool(eligible.all().item()):
        raise ValueError("all-positive coverage requires a legal positive per row")

    floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~legal, floor)
    boundary = torch.topk(masked, k=k, dim=-1).values[:, -1]
    above = legal & logits.gt(boundary.unsqueeze(-1))
    tied = legal & logits.eq(boundary.unsqueeze(-1))
    remaining = (k - above.sum(dim=-1)).clamp_min(0)
    tied_count = tied.sum(dim=-1)
    if bool(tied_count.le(remaining).all().item()):
        selected = above | tied
    else:
        tied_order = tied.to(dtype=torch.int32).cumsum(
            dim=-1,
            dtype=torch.int32,
        )
        selected = above | (tied & tied_order.le(remaining.unsqueeze(-1)))
    positive_hit_count = (legal_positive & selected).sum(dim=-1)
    complete = positive_hit_count.eq(positive_count)
    return (
        {
            "row_count": float(logits.size(0)),
            "positive_target_count": float(positive_count.sum().cpu().item()),
            "positive_target_hit_count": float(
                positive_hit_count.sum().cpu().item()
            ),
            "complete_row_count": float(complete.sum().cpu().item()),
            "top_k": float(k),
        },
        complete,
    )


def _stage0_step_bucket(row: dict[str, Any]) -> str:
    raw = row.get("decision_step_index")
    if raw in (None, ""):
        return "unknown"
    step = int(raw)
    return str(step) if step <= 3 else "4+"


def _stage0_source_family(row: dict[str, Any]) -> str:
    source = semantic_source_id(row).lower()
    for family, markers in (
        ("toolbench", ("toolbench",)),
        ("alfworld", ("alfworld",)),
        ("webshop", ("webshop",)),
        ("agentgym", ("agentgym",)),
        ("traject", ("traject",)),
        ("skillret", ("skillret",)),
    ):
        if any(marker in source for marker in markers):
            return family
    return source or "unknown"


def _rank_summary(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {
            "row_count": 0,
            "mrr": 0.0,
            "recall_at_1": 0.0,
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "recall_at_500": 0.0,
        }
    count = len(ranks)
    return {
        "row_count": count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "recall_at_1": sum(int(rank <= 1) for rank in ranks) / count,
        "recall_at_10": sum(int(rank <= 10) for rank in ranks) / count,
        "recall_at_100": sum(int(rank <= 100) for rank in ranks) / count,
        "recall_at_500": sum(int(rank <= 500) for rank in ranks) / count,
    }


def _exact_factual_route_metrics(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    legal: torch.Tensor,
    skill_id_to_idx: dict[str, int],
) -> dict[str, float]:
    nll_sum = 0.0
    reciprocal_sum = 0.0
    recall_at_1_sum = 0.0
    recall_at_10_sum = 0.0
    recall_at_100_sum = 0.0
    recall_at_500_sum = 0.0
    weight_sum = 0.0
    target_count = 0
    floor = torch.finfo(logits.dtype).min
    legal_log_partition = torch.logsumexp(logits.masked_fill(~legal, floor), dim=-1)
    skill_indices = torch.arange(logits.size(1), device=logits.device)
    for row_index, row in enumerate(rows):
        target_counts = row.get("current_state_route_factual_target_counts") or {}
        if not isinstance(target_counts, dict) or not target_counts:
            raise ValueError("static-route dev row lacks factual target counts")
        for skill_id, raw_count in sorted(target_counts.items()):
            if str(skill_id) not in skill_id_to_idx:
                raise ValueError("factual static-route target is absent from the skill table")
            target_index = int(skill_id_to_idx[str(skill_id)])
            if not bool(legal[row_index, target_index]):
                raise ValueError("factual static-route target is outside the legal inventory")
            count = float(raw_count)
            if not math.isfinite(count) or count <= 0.0:
                raise ValueError("factual static-route target count must be positive")
            target_score = logits[row_index, target_index]
            rank = 1 + int(
                (
                    legal[row_index]
                    & (
                        (logits[row_index] > target_score)
                        | (
                            logits[row_index].eq(target_score)
                            & skill_indices.lt(target_index)
                        )
                    )
                )
                .sum()
                .cpu()
                .item()
            )
            nll_sum += count * float(
                (legal_log_partition[row_index] - target_score).float().cpu().item()
            )
            reciprocal_sum += count / rank
            recall_at_1_sum += count * int(rank <= 1)
            recall_at_10_sum += count * int(rank <= 10)
            recall_at_100_sum += count * int(rank <= 100)
            recall_at_500_sum += count * int(rank <= 500)
            weight_sum += count
            target_count += 1
    return {
        "nll_sum": nll_sum,
        "mrr_sum": reciprocal_sum,
        "recall_at_1_sum": recall_at_1_sum,
        "recall_at_10_sum": recall_at_10_sum,
        "recall_at_100_sum": recall_at_100_sum,
        "recall_at_500_sum": recall_at_500_sum,
        "weight_sum": weight_sum,
        "target_count": float(target_count),
    }


@torch.no_grad()
def _evaluate_stage0(
    model: CLSTRModel,
    *,
    retrieval_rows: list[dict[str, Any]],
    static_rows: list[dict[str, Any]],
    cache: Any,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    belief_top_k: int,
    batch_size: int,
    max_rows_per_kind: int | None,
    autocast_dtype: torch.dtype,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind, rows, head in (
        ("retrieval", _stable_cap_rows(retrieval_rows, max_rows_per_kind), "recall"),
        ("static_route", _stable_cap_rows(static_rows, max_rows_per_kind), "route"),
    ):
        totals: Counter[str] = Counter()
        source_ranks: dict[str, list[int]] = defaultdict(list)
        family_ranks: dict[str, list[int]] = defaultdict(list)
        step_ranks: dict[str, list[int]] = defaultdict(list)
        family_step_ranks: dict[str, list[int]] = defaultdict(list)
        all_positive_source_rows: Counter[str] = Counter()
        all_positive_source_complete: Counter[str] = Counter()
        all_positive_family_rows: Counter[str] = Counter()
        all_positive_family_complete: Counter[str] = Counter()
        raw_causal_totals: Counter[str] = Counter()
        raw_causal_source_ranks: dict[str, list[int]] = defaultdict(list)
        raw_causal_family_ranks: dict[str, list[int]] = defaultdict(list)
        unique_queries: set[str] = set()
        for start in range(0, len(rows), int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            states = cache.batch(
                [str(row["_vnext_query"]) for row in batch],
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
                belief = model.vnext_initial_belief(states, legal, top_k=belief_top_k)
                recall_query, route_query = model.vnext.static_queries(states, belief)
                query = recall_query if head == "recall" else route_query
                logits = model.vnext_full_pool_logits(query, head=head)
                raw_causal_logits = (
                    model.vnext_full_pool_logits(
                        F.normalize(states.float(), p=2, dim=-1).to(states.dtype),
                        head="recall",
                    )
                    if kind == "static_route"
                    else None
                )
                row_nll = (
                    torch.logsumexp(logits.masked_fill(~legal, torch.finfo(logits.dtype).min), dim=-1)
                    - torch.logsumexp(
                        logits.masked_fill(~(positive & legal), torch.finfo(logits.dtype).min),
                        dim=-1,
                    )
                )
            metrics = _ranking_metrics(logits.float(), positive, legal)
            coverage, complete_rows = _all_positive_topk_coverage(
                logits.float(),
                positive,
                legal,
                k=500,
            )
            ranks = _positive_ranks(logits.float(), positive, legal).cpu().tolist()
            complete_values = complete_rows.cpu().tolist()
            for row, raw_rank, complete in zip(batch, ranks, complete_values):
                rank = int(raw_rank)
                source = semantic_source_id(row)
                family = _stage0_source_family(row)
                step_bucket = _stage0_step_bucket(row)
                source_ranks[source].append(rank)
                family_ranks[family].append(rank)
                step_ranks[step_bucket].append(rank)
                family_step_ranks[f"{family}:{step_bucket}"].append(rank)
                all_positive_source_rows[source] += 1
                all_positive_source_complete[source] += int(bool(complete))
                all_positive_family_rows[family] += 1
                all_positive_family_complete[family] += int(bool(complete))
                unique_queries.add(str(row["_vnext_query"]))
            totals.update(metrics)
            for name, value in coverage.items():
                if name != "top_k":
                    totals[f"all_positive_{name}"] += float(value)
            totals["nll_sum"] += float(row_nll.float().sum().cpu().item())
            if kind == "static_route":
                if raw_causal_logits is None:
                    raise RuntimeError("static-route raw causal baseline is missing")
                raw_metrics = _ranking_metrics(
                    raw_causal_logits.float(),
                    positive,
                    legal,
                )
                raw_ranks = _positive_ranks(
                    raw_causal_logits.float(),
                    positive,
                    legal,
                ).cpu().tolist()
                raw_causal_totals.update(raw_metrics)
                for row, raw_rank in zip(batch, raw_ranks):
                    rank = int(raw_rank)
                    raw_causal_source_ranks[semantic_source_id(row)].append(rank)
                    raw_causal_family_ranks[_stage0_source_family(row)].append(rank)
                exact = _exact_factual_route_metrics(
                    logits.float(),
                    batch,
                    legal,
                    skill_id_to_idx,
                )
                for name, value in exact.items():
                    totals[f"exact_factual_{name}"] += float(value)
        row_count = int(totals["row_count"])
        if row_count <= 0:
            raise ValueError(f"Stage0 validation has no rows for {kind}")
        output[kind] = {
            "row_count": row_count,
            "unique_query_count": len(unique_queries),
            "unique_query_rate": float(len(unique_queries)) / row_count,
            "nll": float(totals["nll_sum"]) / row_count,
            "mrr": float(totals["mrr_sum"]) / row_count,
            "recall_at_1": float(totals["recall_at_1_sum"]) / row_count,
            "recall_at_10": float(totals["recall_at_10_sum"]) / row_count,
            "recall_at_100": float(totals["recall_at_100_sum"]) / row_count,
            "recall_at_500": float(totals["recall_at_500_sum"]) / row_count,
            "per_source": {
                name: _rank_summary(values)
                for name, values in sorted(source_ranks.items())
            },
            "per_family": {
                name: _rank_summary(values)
                for name, values in sorted(family_ranks.items())
            },
            "per_step": {
                name: _rank_summary(values)
                for name, values in sorted(step_ranks.items())
            },
            "per_family_step": {
                name: _rank_summary(values)
                for name, values in sorted(family_step_ranks.items())
            },
            "all_positive_top500": {
                "protocol": "natural_complete_positive_top500_v1",
                "requested_top_k": 500,
                "effective_top_k": min(500, len(skill_id_to_idx)),
                "row_count": int(totals["all_positive_row_count"]),
                "positive_target_count": int(
                    totals["all_positive_positive_target_count"]
                ),
                "positive_target_coverage": float(
                    totals["all_positive_positive_target_hit_count"]
                )
                / max(float(totals["all_positive_positive_target_count"]), 1.0),
                "complete_row_recall": float(
                    totals["all_positive_complete_row_count"]
                )
                / max(float(totals["all_positive_row_count"]), 1.0),
                "per_source_complete_row_recall": {
                    name: {
                        "row_count": int(count),
                        "complete_row_recall": float(
                            all_positive_source_complete[name]
                        )
                        / int(count),
                    }
                    for name, count in sorted(all_positive_source_rows.items())
                },
                "per_family_complete_row_recall": {
                    name: {
                        "row_count": int(count),
                        "complete_row_recall": float(
                            all_positive_family_complete[name]
                        )
                        / int(count),
                    }
                    for name, count in sorted(all_positive_family_rows.items())
                },
            },
        }
        if kind == "static_route":
            raw_row_count = int(raw_causal_totals["row_count"])
            if raw_row_count != row_count:
                raise RuntimeError("raw causal baseline row count mismatch")
            raw_causal_baseline = {
                "protocol": "frozen_raw_causal_full_pool_v1",
                "row_count": raw_row_count,
                "mrr": float(raw_causal_totals["mrr_sum"]) / raw_row_count,
                "recall_at_1": float(raw_causal_totals["recall_at_1_sum"])
                / raw_row_count,
                "recall_at_10": float(raw_causal_totals["recall_at_10_sum"])
                / raw_row_count,
                "recall_at_100": float(raw_causal_totals["recall_at_100_sum"])
                / raw_row_count,
                "recall_at_500": float(raw_causal_totals["recall_at_500_sum"])
                / raw_row_count,
                "per_source": {
                    name: _rank_summary(values)
                    for name, values in sorted(raw_causal_source_ranks.items())
                },
                "per_family": {
                    name: _rank_summary(values)
                    for name, values in sorted(raw_causal_family_ranks.items())
                },
            }
            output[kind]["raw_causal_baseline"] = raw_causal_baseline
            output[kind]["learned_route_minus_raw_causal"] = {
                metric: float(output[kind][metric])
                - float(raw_causal_baseline[metric])
                for metric in (
                    "mrr",
                    "recall_at_1",
                    "recall_at_10",
                    "recall_at_100",
                    "recall_at_500",
                )
            }
            exact_weight = float(totals["exact_factual_weight_sum"])
            if exact_weight <= 0.0:
                raise ValueError("Stage0 static validation has no factual target weight")
            output[kind]["exact_factual"] = {
                "target_count": int(totals["exact_factual_target_count"]),
                "target_weight": exact_weight,
                "nll": float(totals["exact_factual_nll_sum"]) / exact_weight,
                "mrr": float(totals["exact_factual_mrr_sum"]) / exact_weight,
                "recall_at_1": float(totals["exact_factual_recall_at_1_sum"])
                / exact_weight,
                "recall_at_10": float(totals["exact_factual_recall_at_10_sum"])
                / exact_weight,
                "recall_at_100": float(totals["exact_factual_recall_at_100_sum"])
                / exact_weight,
                "recall_at_500": float(totals["exact_factual_recall_at_500_sum"])
                / exact_weight,
            }
    trajectory_retrieval_family_rows = [
        output["retrieval"]["per_family"][name]
        for name in ("toolbench", "traject", "alfworld", "webshop", "agentgym")
        if name in output["retrieval"]["per_family"]
    ]
    trajectory_retrieval_row_count = sum(
        int(row["row_count"]) for row in trajectory_retrieval_family_rows
    )
    trajectory_retrieval_mrr = (
        sum(
            float(row["mrr"]) * int(row["row_count"])
            for row in trajectory_retrieval_family_rows
        )
        / trajectory_retrieval_row_count
        if trajectory_retrieval_row_count
        else float(output["retrieval"]["mrr"])
    )
    trajectory_retrieval_recall_at_100 = (
        sum(
            float(row["recall_at_100"]) * int(row["row_count"])
            for row in trajectory_retrieval_family_rows
        )
        / trajectory_retrieval_row_count
        if trajectory_retrieval_row_count
        else float(output["retrieval"]["recall_at_100"])
    )
    trajectory_route_family_rows = [
        output["static_route"]["per_family"][name]
        for name in ("toolbench", "traject", "alfworld", "webshop", "agentgym")
        if name in output["static_route"]["per_family"]
    ]
    trajectory_route_row_count = sum(
        int(row["row_count"]) for row in trajectory_route_family_rows
    )
    trajectory_route_mrr = (
        sum(
            float(row["mrr"]) * int(row["row_count"])
            for row in trajectory_route_family_rows
        )
        / trajectory_route_row_count
        if trajectory_route_row_count
        else float(output["static_route"]["mrr"])
    )
    trajectory_route_recall_at_100 = (
        sum(
            float(row["recall_at_100"]) * int(row["row_count"])
            for row in trajectory_route_family_rows
        )
        / trajectory_route_row_count
        if trajectory_route_row_count
        else float(output["static_route"]["recall_at_100"])
    )
    exact_factual_route_mrr = float(output["static_route"]["exact_factual"]["mrr"])
    output["selection_components"] = {
        "retrieval_mrr": float(output["retrieval"]["mrr"]),
        "static_route_mrr": float(output["static_route"]["mrr"]),
        "exact_factual_static_route_mrr": exact_factual_route_mrr,
        "trajectory_static_route_mrr": float(trajectory_route_mrr),
        "trajectory_static_route_recall_at_100": float(
            trajectory_route_recall_at_100
        ),
        "trajectory_static_route_row_count": int(trajectory_route_row_count),
        "trajectory_retrieval_mrr": float(trajectory_retrieval_mrr),
        "trajectory_retrieval_recall_at_100": float(
            trajectory_retrieval_recall_at_100
        ),
        "trajectory_retrieval_row_count": int(trajectory_retrieval_row_count),
        "weights": {
            "static_route_mrr": 0.4,
            "trajectory_static_route_mrr": 0.25,
            "trajectory_static_route_recall_at_100": 0.15,
            "exact_factual_static_route_mrr": 0.15,
            "retrieval_mrr": 0.05,
        },
    }
    output["selection_score"] = (
        0.4 * float(output["static_route"]["mrr"])
        + 0.25 * float(trajectory_route_mrr)
        + 0.15 * float(trajectory_route_recall_at_100)
        + 0.15 * exact_factual_route_mrr
        + 0.05 * float(output["retrieval"]["mrr"])
    )
    return output


def train_vnext_stage0(
    *,
    skills_path: str | Path,
    retrieval_rows_path: str | Path,
    retrieval_dev_rows_path: str | Path,
    static_route_rows_path: str | Path,
    static_route_dev_rows_path: str | Path,
    inventory_catalogs_path: str | Path,
    data_contract_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    model_dim: int = 1024,
    max_length: int = 2048,
    torch_dtype: str = "bfloat16",
    skill_table_batch_size: int = 32,
    belief_top_k: int = 64,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    frozen_cache_dir: str | Path | None = None,
    backbone_snapshot_path: str | Path | None = None,
    skill_cache_shard_size: int = 2048,
    hard_negative_loss_weight: float = 0.2,
    hard_negative_margin: float = 0.1,
    hard_negative_top_k: int = 32,
    max_skills: int | None = None,
    max_retrieval_rows: int | None = None,
    max_static_rows: int | None = None,
    seed: int = 17,
    checkpoint_interval: int = 400,
    validation_interval: int = 400,
    validation_batch_size: int = 128,
    max_dev_rows_per_kind: int | None = 4096,
    minimum_dev_score_gain: float = 0.0,
    resume_checkpoint_path: str | Path | None = None,
    legacy_init_checkpoint_path: str | Path | None = None,
    legacy_init_skills_path: str | Path | None = None,
    local_files_only: bool = True,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(checkpoint_interval) != int(validation_interval):
        raise ValueError(
            "canonical segmented Stage0 requires checkpoint_interval == validation_interval"
        )
    if resume_checkpoint_path and legacy_init_checkpoint_path:
        raise ValueError("Stage0 resume and legacy initialization are mutually exclusive")
    if bool(legacy_init_checkpoint_path) != bool(legacy_init_skills_path):
        raise ValueError("legacy Stage0 initialization requires checkpoint and skills paths")
    if not math.isfinite(float(hard_negative_loss_weight)) or float(hard_negative_loss_weight) < 0.0:
        raise ValueError("hard_negative_loss_weight must be finite and nonnegative")
    if not math.isfinite(float(hard_negative_margin)) or float(hard_negative_margin) < 0.0:
        raise ValueError("hard_negative_margin must be finite and nonnegative")
    if int(hard_negative_top_k) <= 0:
        raise ValueError("hard_negative_top_k must be positive")
    reproducibility = seed_vnext_run(seed)
    source_manifest = capture_vnext_source_manifest(
        Path(__file__).resolve().parents[1],
        (
            "clstr/model.py",
            "clstr/belief.py",
            "clstr/history_channel.py",
            "clstr/vnext_core.py",
            "clstr/vnext_data.py",
            "clstr/vnext_losses.py",
            "clstr/vnext_training.py",
            "clstr/vnext_stage0_train.py",
            "scripts/run_clstr_vnext_stage0_train.py",
            "scripts/resolve_clstr_vnext_full_inputs.py",
            "scripts/sbatch/run_clstr_vnext_full_stage0_segment.sh",
        ),
        require_clean=bool(require_clean_source),
    )
    source_manifest_record = persist_vnext_source_manifest(output_dir, source_manifest)
    data_contract = require_verified_data_contract(
        data_contract_path,
        {
            "training_skills": skills_path,
            "retrieval_rows": retrieval_rows_path,
            "retrieval_dev_rows": retrieval_dev_rows_path,
            "static_route_rows": static_route_rows_path,
            "static_route_dev_rows": static_route_dev_rows_path,
            "inventory_catalogs": inventory_catalogs_path,
        },
    )
    raw_skills = read_jsonl(skills_path)
    retrieval_raw = read_jsonl(retrieval_rows_path, max_rows=max_retrieval_rows)
    static_raw = read_jsonl(static_route_rows_path, max_rows=max_static_rows)
    retrieval_dev_raw = read_jsonl(retrieval_dev_rows_path)
    static_dev_raw = read_jsonl(static_route_dev_rows_path)
    retrieval_dev_for_skill_selection, _retrieval_skill_cap = stable_stratified_cap_rows(
        retrieval_dev_raw,
        max_dev_rows_per_kind,
    )
    static_dev_for_skill_selection, _static_skill_cap = stable_stratified_cap_rows(
        static_dev_raw,
        max_dev_rows_per_kind,
    )
    skills = _select_skills(
        raw_skills,
        [*retrieval_raw, *retrieval_dev_for_skill_selection],
        [*static_raw, *static_dev_for_skill_selection],
        max_skills=max_skills,
    )
    selected_skill_ids = {_skill_id(row) for row in skills}
    catalogs, catalog_mapping, catalog_report = derive_inventory_catalog_subset(
        load_inventory_catalogs(inventory_catalogs_path),
        selected_skill_ids,
    )
    retrieval_rows = _filter_rows(
        rewrite_inventory_catalog_references(retrieval_raw, catalog_mapping, catalogs),
        selected_skill_ids,
        kind="retrieval",
    )
    static_rows = _filter_rows(
        rewrite_inventory_catalog_references(static_raw, catalog_mapping, catalogs),
        selected_skill_ids,
        kind="static_route",
    )
    retrieval_dev_rows = _filter_rows(
        rewrite_inventory_catalog_references(
            retrieval_dev_raw,
            catalog_mapping,
            catalogs,
        ),
        selected_skill_ids,
        kind="retrieval",
    )
    static_dev_rows = _filter_rows(
        rewrite_inventory_catalog_references(
            static_dev_raw,
            catalog_mapping,
            catalogs,
        ),
        selected_skill_ids,
        kind="static_route",
    )
    _require_legal_stage0_positives(retrieval_rows, catalogs)
    _require_legal_stage0_positives(static_rows, catalogs)
    _require_legal_stage0_positives(retrieval_dev_rows, catalogs)
    _require_legal_stage0_positives(static_dev_rows, catalogs)
    retrieval_rows, retrieval_eligibility = _partition_stage0_eligible_rows(
        retrieval_rows,
        catalogs,
        kind="retrieval_train",
    )
    static_rows, static_eligibility = _partition_stage0_eligible_rows(
        static_rows,
        catalogs,
        kind="static_route_train",
    )
    retrieval_dev_rows, retrieval_dev_eligibility = _partition_stage0_eligible_rows(
        retrieval_dev_rows,
        catalogs,
        kind="retrieval_dev",
    )
    static_dev_rows, static_dev_eligibility = _partition_stage0_eligible_rows(
        static_dev_rows,
        catalogs,
        kind="static_route_dev",
    )
    retrieval_dev_rows, retrieval_dev_sampling = stable_stratified_cap_rows(
        retrieval_dev_rows,
        max_dev_rows_per_kind,
    )
    static_dev_rows, static_dev_sampling = stable_stratified_cap_rows(
        static_dev_rows,
        max_dev_rows_per_kind,
    )
    history_channel = audit_history_channel_rows(
        [*retrieval_rows, *static_rows, *retrieval_dev_rows, *static_dev_rows],
        require_explicit_current=True,
        require_structured_current=True,
        require_explicit_causal=True,
    )
    if history_channel.get("status") != "ok":
        raise ValueError("Stage0 data contract failed structured current/causal audit")
    backbone_snapshot = load_or_capture_frozen_backbone_snapshot(
        model_name_or_path,
        manifest_path=(
            Path(backbone_snapshot_path)
            if backbone_snapshot_path is not None
            else output_dir / "backbone_snapshot.json"
        ),
    )
    selected_skills_path = output_dir / "selected_skills.jsonl"
    if len(skills) == len(raw_skills) and all(
        _skill_id(selected) == _skill_id(raw)
        for selected, raw in zip(skills, raw_skills)
    ):
        selected_skills_path.write_bytes(Path(skills_path).read_bytes())
    else:
        selected_skills_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skills),
            encoding="utf-8",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("vNext Stage0 training requires a CUDA Slurm node")
    model, model_config = _build_model(
        skills,
        model_name_or_path=model_name_or_path,
        model_dim=model_dim,
        max_length=max_length,
        torch_dtype=torch_dtype,
        skill_table_batch_size=skill_table_batch_size,
        belief_top_k=belief_top_k,
        local_files_only=local_files_only,
        frozen_backbone_snapshot_digest=backbone_snapshot["contract_digest"],
        frozen_backbone_snapshot_manifest_path=backbone_snapshot["path"],
    )
    model.to(device)
    model.eval()
    cache_root = (
        Path(frozen_cache_dir)
        if frozen_cache_dir is not None
        else output_dir / "frozen_qwen_cache"
    )
    resume_payload: dict[str, Any] | None = None
    start_step = 1
    if resume_checkpoint_path:
        resume_payload = torch.load(resume_checkpoint_path, map_location="cpu")
        require_canonical_vnext_checkpoint_state(
            resume_payload.get("model_state_dict")
        )
    legacy_init_report: dict[str, Any] = {"enabled": False}
    if legacy_init_checkpoint_path is not None:
        legacy_skill_rows = read_jsonl(legacy_init_skills_path)
        legacy_skill_ids = [_skill_id(row) for row in legacy_skill_rows]
        selected_skill_ids_ordered = [_skill_id(row) for row in skills]
        legacy_payload = torch.load(legacy_init_checkpoint_path, map_location="cpu")
        legacy_state = (
            legacy_payload.get("model_state_dict")
            if isinstance(legacy_payload, dict)
            else None
        )
        mapped_state, mapping_report = mapped_legacy_stage0_state(
            model,
            legacy_state,
            legacy_skill_ids=legacy_skill_ids,
            current_skill_ids=selected_skill_ids_ordered,
        )
        load_report = load_compatible_state_dict(
            model,
            mapped_state,
            partial_load_mode="vnext_stage0_legacy_foundation_init",
        )
        if set(mapping_report["required_targets"]) - set(load_report["loaded_keys"]):
            raise ValueError("legacy Stage0 foundation did not load every required tensor")
        legacy_init_report = {
            "enabled": True,
            "checkpoint_path": str(Path(legacy_init_checkpoint_path).resolve()),
            "checkpoint_sha256": file_sha256(legacy_init_checkpoint_path),
            "skills_path": str(Path(legacy_init_skills_path).resolve()),
            "skills_sha256": file_sha256(legacy_init_skills_path),
            "checkpoint_step": int(legacy_payload.get("step") or 0),
            "mapping": mapping_report,
            "load_report": load_report,
        }
        skill_cache = {
            "status": "restored_from_legacy_checkpoint",
            "skill_count": len(skills),
            "checkpoint_sha256": legacy_init_report["checkpoint_sha256"],
            "id_aligned": True,
            "cache_identity": (
                "legacy-id-aligned:"
                f"{legacy_init_report['checkpoint_sha256']}:"
                f"{file_sha256(selected_skills_path)}"
            ),
        }
    elif resume_payload is not None:
        load_compatible_state_dict(
            model,
            resume_payload["model_state_dict"],
            partial_load_mode="vnext_stage0_resume",
        )
        start_step = int(resume_payload["step"]) + 1
        prior_inputs = dict(
            (resume_payload.get("run_contract") or {}).get("inputs") or {}
        )
        prior_legacy_checkpoint = prior_inputs.get("legacy_init_checkpoint_sha256")
        prior_legacy_skills = prior_inputs.get("legacy_init_skills_sha256")
        legacy_init_report = {
            "enabled": bool(prior_legacy_checkpoint),
            "resumed_from_run_contract": True,
            "checkpoint_sha256": prior_legacy_checkpoint,
            "skills_sha256": prior_legacy_skills,
        }
        skill_cache = {
            "status": "restored_from_resume_checkpoint",
            "skill_count": len(skills),
            "checkpoint_path": str(Path(resume_checkpoint_path).resolve()),
            "cache_identity": str(
                (resume_payload.get("run_contract") or {}).get(
                    "skill_cache_identity"
                )
                or ""
            ),
        }
        if not skill_cache["cache_identity"]:
            raise ValueError("Stage0 resume lacks its skill-cache identity")
    else:
        skill_cache = load_or_build_skill_embedding_cache(
            model,
            skills,
            cache_root=cache_root,
            batch_size=skill_table_batch_size,
            cache_shard_size=skill_cache_shard_size,
        )
    trainability = configure_vnext_stage0(model)
    require_canonical_trainability(trainability)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate))
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
    schedule, sampling_report = _schedule_batches(
        retrieval_rows,
        static_rows,
        start_step=start_step,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        seed=seed,
    )
    scheduled_training_rows = [row for batch in schedule for row in batch]
    cache = load_or_build_frozen_text_cache(
        model,
        (
            row["_vnext_query"]
            for row in [
                *scheduled_training_rows,
                *retrieval_dev_rows,
                *static_dev_rows,
            ]
        ),
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
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
            "schema_version": "clstr_vnext_stage0_unified_static_run_v7",
            "inputs": {
                "skills_sha256": file_sha256(skills_path),
                "retrieval_rows_sha256": file_sha256(retrieval_rows_path),
                "retrieval_dev_rows_sha256": file_sha256(retrieval_dev_rows_path),
                "static_route_rows_sha256": file_sha256(static_route_rows_path),
                "static_route_dev_rows_sha256": file_sha256(static_route_dev_rows_path),
                "inventory_catalogs_sha256": file_sha256(inventory_catalogs_path),
                "data_contract_sha256": file_sha256(data_contract_path),
                "selected_skills_sha256": file_sha256(selected_skills_path),
                "legacy_init_checkpoint_sha256": legacy_init_report.get(
                    "checkpoint_sha256"
                ),
                "legacy_init_skills_sha256": legacy_init_report.get("skills_sha256"),
                "frozen_backbone_snapshot_contract": backbone_snapshot["contract"],
                "derived_catalog_identity": catalog_identity,
            },
            "model_config": model_config,
            "optimization": {
                "batch_size": int(batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "learning_rate": float(learning_rate),
                "belief_top_k": int(belief_top_k),
                "full_pool_scoring": "single_dense_deployment_identical_matmul_v1",
                "hard_negative_loss_weight": float(hard_negative_loss_weight),
                "hard_negative_margin": float(hard_negative_margin),
                "hard_negative_top_k": int(hard_negative_top_k),
                "hard_negative_positive_anchor_by_kind": {
                    "retrieval": "weakest_legal_positive_v1",
                    "static_route": "set_logsumexp_v1",
                },
                "seed": int(seed),
                "max_skills": None if max_skills is None else int(max_skills),
                "max_retrieval_rows": (
                    None if max_retrieval_rows is None else int(max_retrieval_rows)
                ),
                "max_static_rows": (
                    None if max_static_rows is None else int(max_static_rows)
                ),
                "validation_batch_size": int(validation_batch_size),
                "max_dev_rows_per_kind": (
                    None
                    if max_dev_rows_per_kind is None
                    else int(max_dev_rows_per_kind)
                ),
                "validation_interval": int(validation_interval),
                "checkpoint_interval": int(checkpoint_interval),
                "minimum_dev_score_gain": float(minimum_dev_score_gain),
                "sampling_protocol": "capability_source_row_round_robin_v1",
                "segment_extension_protocol": (
                    "deterministic_absolute_optimizer_step_v1"
                ),
                "state_cache_protocol": "schedule_aware_preprojection_v1",
                "dev_selection_protocol": "full_pool_unified_static_primary_v4",
                "candidate_foundation_diagnostic": (
                    "raw_causal_vs_trained_unified_static_full_pool_v2"
                ),
                "cache_shard_size": int(cache_shard_size),
                "skill_cache_shard_size": int(skill_cache_shard_size),
                "require_clean_source": bool(require_clean_source),
            },
            "frozen_cache_identity": cache.cache_identity,
            "skill_cache_identity": skill_cache["cache_identity"],
            "source_contract_digest": source_manifest_record[
                "source_contract_digest"
            ],
        }
    )
    if resume_payload is not None:
        require_matching_run_contract(resume_payload.get("run_contract"), run_contract)
    prior_progress = (
        dict(resume_payload.get("trainer_progress") or {})
        if resume_payload is not None
        else {}
    )
    if resume_payload is not None and int(start_step) > 1:
        if not prior_progress:
            raise ValueError("Stage0 resume checkpoint lacks cumulative trainer_progress")
        if int(prior_progress.get("completed_step", -1)) != int(start_step - 1):
            raise ValueError("Stage0 resume trainer_progress does not end at checkpoint step")
    skill_id_to_idx = {_skill_id(row): idx for idx, row in enumerate(skills)}
    resume_validation = _evaluate_stage0(
        model,
        retrieval_rows=retrieval_dev_rows,
        static_rows=static_dev_rows,
        cache=cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        belief_top_k=belief_top_k,
        batch_size=validation_batch_size,
        max_rows_per_kind=max_dev_rows_per_kind,
        autocast_dtype=(torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16),
    )
    if resume_payload is None:
        validation_records: list[dict[str, Any]] = [
            {"step": int(start_step - 1), **resume_validation}
        ]
        initial_score = float(resume_validation["selection_score"])
        best_step = int(start_step - 1)
        best_score = initial_score
        best_checkpoint = output_dir / "checkpoints" / f"clstr_vnext_stage0-step{best_step}.pt"
    else:
        prior_selection_path = output_dir / "stage0_selection.json"
        if not prior_selection_path.is_file():
            raise ValueError("Stage0 resume requires prior stage0_selection.json in output_dir")
        prior_selection = json.loads(prior_selection_path.read_text(encoding="utf-8"))
        validation_records = list(prior_selection.get("validation_records") or [])
        if not validation_records or int(validation_records[-1].get("step", -1)) != int(
            start_step - 1
        ):
            raise ValueError("Stage0 resume selection does not end at checkpoint step")
        if abs(
            float(validation_records[-1].get("selection_score") or 0.0)
            - float(resume_validation["selection_score"])
        ) > 1.0e-8:
            raise ValueError("Stage0 resume validation does not reproduce prior checkpoint")
        initial_score = float(prior_selection["initial_score"])
        best_step = int(prior_selection["selected_step"])
        best_score = float(prior_selection["selected_score"])
        best_checkpoint = Path(str(prior_selection["selected_checkpoint_path"]))
    loss_curve: list[dict[str, Any]] = list(prior_progress.get("loss_curve") or [])
    kind_source_exposures: Counter[str] = Counter(
        prior_progress.get("kind_source_exposures") or {}
    )
    kind_exposures: Counter[str] = Counter(prior_progress.get("kind_exposures") or {})
    optimizer.zero_grad(set_to_none=True)
    schedule_index = 0
    started = time.perf_counter()
    prior_elapsed_seconds = float(prior_progress.get("elapsed_seconds") or 0.0)
    autocast_dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    def current_selection(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "selected_step": int(best_step),
            "selected_checkpoint_path": str(best_checkpoint.resolve()),
            "selected_score": float(best_score),
            "initial_score": float(initial_score),
            "score_gain": float(best_score - initial_score),
            "minimum_score_gain": float(minimum_dev_score_gain),
            "selection_source": "heldout_full_pool_unified_static_primary_dev_v4",
            "validation_records": validation_records,
        }

    def trainer_progress() -> dict[str, Any]:
        return {
            "completed_step": int(loss_curve[-1]["step"]) if loss_curve else int(start_step - 1),
            "loss_curve": loss_curve,
            "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
            "kind_source_exposures": dict(sorted(kind_source_exposures.items())),
            "kind_exposures": dict(sorted(kind_exposures.items())),
            "validation_records": validation_records,
            "selected_step": int(best_step),
            "selected_score": float(best_score),
            "selected_checkpoint_path": str(best_checkpoint.resolve()),
        }

    if start_step == 1:
        _save_checkpoint(
            output_dir / "checkpoints" / "clstr_vnext_stage0-step0.pt",
            model=model,
            optimizer=optimizer,
            step=0,
            model_config=model_config,
            trainability=trainability,
            skills_path=selected_skills_path,
            run_contract=run_contract,
            trainer_progress=trainer_progress(),
        )
    write_json(output_dir / "trainer_progress.json", trainer_progress())
    write_json(output_dir / "stage0_selection.json", current_selection("training"))
    for step in range(start_step, int(max_steps) + 1):
        step_losses: list[float] = []
        kind_losses: Counter[str] = Counter()
        kind_hard_losses: Counter[str] = Counter()
        for _micro in range(int(gradient_accumulation_steps)):
            batch = schedule[schedule_index]
            schedule_index += 1
            for row in batch:
                kind = str(row["_vnext_kind"])
                source = semantic_source_id(row)
                kind_exposures[kind] += 1
                kind_source_exposures[f"{kind}:{source}"] += 1
            texts = [row["_vnext_query"] for row in batch]
            h_t = cache.batch(texts, device=device)
            legal = runtime_visible_mask(
                batch,
                skill_id_to_idx,
                device=device,
                inventory_catalogs=catalogs,
            )
            positive = _positive_mask(batch, skill_id_to_idx, device=device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                b_t = model.vnext_initial_belief(h_t, legal, top_k=belief_top_k)
                # Stage0 owns the unified static foundation only. The separate
                # static-route residual is trained by the downstream adapter
                # stage and must not enter this objective, even as a zero-valued
                # tensor with distinct storage.
                unified_query = model.vnext.static_query(h_t, b_t)
                full_logits = model.vnext_full_pool_logits(
                    unified_query,
                    head="recall",
                )
                losses: list[torch.Tensor] = []
                for kind in ("retrieval", "static_route"):
                    indices = [idx for idx, row in enumerate(batch) if row["_vnext_kind"] == kind]
                    if not indices:
                        continue
                    index = torch.tensor(indices, dtype=torch.long, device=device)
                    positives = positive.index_select(0, index)
                    legal_rows = legal.index_select(0, index)
                    output = dense_full_pool_nll_and_hard_negative(
                        full_logits.index_select(0, index),
                        positives,
                        legal_rows,
                        hard_negative_top_k=hard_negative_top_k,
                        hard_negative_margin=hard_negative_margin,
                        hard_negative_positive_anchor=(
                            "weakest_legal_positive_v1"
                            if kind == "retrieval"
                            else "set_logsumexp_v1"
                        ),
                    )
                    objective = output.nll_loss + float(hard_negative_loss_weight) * output.hard_negative_loss
                    losses.append(objective)
                    kind_losses[kind] += float(output.nll_loss.detach().float().cpu().item())
                    kind_hard_losses[kind] += float(
                        output.hard_negative_loss.detach().float().cpu().item()
                    )
                if not losses:
                    raise RuntimeError("Stage0 batch contains no eligible objective")
                loss = torch.stack(losses).mean() / int(gradient_accumulation_steps)
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"nonfinite vNext Stage0 loss at step {step}")
            loss.backward()
            step_losses.append(float(loss.detach().float().cpu().item()) * int(gradient_accumulation_steps))
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": sum(step_losses) / max(len(step_losses), 1),
            "retrieval_loss_sum": float(kind_losses["retrieval"]),
            "static_route_loss_sum": float(kind_losses["static_route"]),
            "retrieval_hard_negative_loss_sum": float(kind_hard_losses["retrieval"]),
            "static_route_hard_negative_loss_sum": float(kind_hard_losses["static_route"]),
            "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
        }
        loss_curve.append(record)
        should_validate = bool(
            step == int(max_steps)
            or (int(validation_interval) > 0 and step % int(validation_interval) == 0)
        )
        checkpoint_path = output_dir / "checkpoints" / f"clstr_vnext_stage0-step{step}.pt"
        if should_validate:
            validation = _evaluate_stage0(
                model,
                retrieval_rows=retrieval_dev_rows,
                static_rows=static_dev_rows,
                cache=cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                belief_top_k=belief_top_k,
                batch_size=validation_batch_size,
                max_rows_per_kind=max_dev_rows_per_kind,
                autocast_dtype=autocast_dtype,
            )
            validation_records.append({"step": int(step), **validation})
            score = float(validation["selection_score"])
            if score > best_score:
                best_score = score
                best_step = int(step)
                best_checkpoint = checkpoint_path
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                model_config=model_config,
                trainability=trainability,
                skills_path=selected_skills_path,
                run_contract=run_contract,
                trainer_progress=trainer_progress(),
            )
            write_json(output_dir / "trainer_progress.json", trainer_progress())
            write_json(
                output_dir / "stage0_selection.json",
                current_selection("training"),
            )
    final_checkpoint = output_dir / "checkpoints" / f"clstr_vnext_stage0-step{max_steps}.pt"
    elapsed = prior_elapsed_seconds + time.perf_counter() - started
    dev_score_gain = float(best_score - initial_score)
    quality_status = (
        "ok"
        if best_step > 0
        and dev_score_gain >= float(minimum_dev_score_gain)
        else "action_required"
    )
    selection = current_selection(quality_status)
    write_json(output_dir / "stage0_selection.json", selection)
    write_json(output_dir / "stage0_quality_gate.json", selection)
    write_json(output_dir / "trainer_progress.json", trainer_progress())
    _save_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        step=max_steps,
        model_config=model_config,
        trainability=trainability,
        skills_path=selected_skills_path,
        run_contract=run_contract,
        trainer_progress=trainer_progress(),
    )
    report = {
        "status": quality_status,
        "stage": "clstr_vnext_stage0",
        "step": int(max_steps),
        "start_step": int(start_step),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "selected_skill_count": len(skills),
        "retrieval_row_count": len(retrieval_rows),
        "static_route_row_count": len(static_rows),
        "retrieval_dev_row_count": len(retrieval_dev_rows),
        "static_route_dev_row_count": len(static_dev_rows),
        "inventory_catalogs": catalog_report,
        "data_contract": data_contract,
        "history_channel": history_channel,
        "loss_eligibility": {
            "retrieval_train": retrieval_eligibility,
            "static_route_train": static_eligibility,
            "retrieval_dev": retrieval_dev_eligibility,
            "static_route_dev": static_dev_eligibility,
        },
        "positive_set_cardinality": {
            "retrieval_train": _positive_set_cardinality_report(retrieval_rows),
            "static_route_train": _positive_set_cardinality_report(static_rows),
            "retrieval_dev": _positive_set_cardinality_report(retrieval_dev_rows),
            "static_route_dev": _positive_set_cardinality_report(static_dev_rows),
        },
        "sampling": {
            **sampling_report,
            "cumulative_kind_exposures": dict(sorted(kind_exposures.items())),
            "cumulative_kind_source_exposures": dict(
                sorted(kind_source_exposures.items())
            ),
        },
        "dev_selection": {
            "retrieval": retrieval_dev_sampling,
            "static_route": static_dev_sampling,
        },
        "trainability": trainability,
        "cache": cache.report(),
        "skill_cache": skill_cache,
        "frozen_backbone_snapshot": backbone_snapshot,
        "legacy_initialization": legacy_init_report,
        "source_manifest": source_manifest_record,
        "reproducibility": reproducibility,
        "loss_curve": loss_curve,
        "elapsed_seconds": elapsed,
        "steps_per_second": len(loss_curve) / max(elapsed, 1.0e-9),
        "static_foundation_digest": static_foundation_digest(model),
        "run_contract": run_contract,
        "finite_loss": all(math.isfinite(float(row["loss"])) for row in loss_curve),
        "objective": {
            "scorer": "unified_static_query_v1",
            "full_pool_multi_positive_nll": True,
            "query_normalization": False,
            "hard_negative_loss_weight": float(hard_negative_loss_weight),
            "hard_negative_margin": float(hard_negative_margin),
            "hard_negative_top_k": int(hard_negative_top_k),
            "hard_negative_positive_anchor_by_kind": {
                "retrieval": "weakest_legal_positive_v1",
                "static_route": "set_logsumexp_v1",
            },
            "positive_injection": False,
        },
        "selection": selection,
    }
    write_json(output_dir / "train_report.json", report)
    return report
