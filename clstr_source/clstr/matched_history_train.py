from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from clstr.matched_history_data import (
    history_events_for_decision,
    load_matched_history_trajectories,
    serialize_factual_history,
)
from clstr.matched_history_encoders import (
    HistoryEncoderKind,
    build_matched_history_encoder,
    trainable_parameter_count,
)
from clstr.history_channel import serialize_compact_causal_state
from clstr.vnext_candidates import masked_topk_tensor
from clstr.vnext_core import BoundedLayerScale, NonAffineRMSNorm, TwoLayerAdapter
from clstr.vnext_data import runtime_visible_mask
from clstr.vnext_eval import load_vnext_stage2_for_evaluation
from clstr.vnext_training import (
    file_sha256,
    load_inventory_catalogs,
    load_or_build_frozen_text_cache,
)


MATCHED_HISTORY_TRAIN_SCHEMA = "clstr_matched_history_e1_train_v1"


class MatchedHistoryRouteScorer(nn.Module):
    """One common within-support scorer shared by every history encoder."""

    def __init__(self, d: int, *, hidden_dim: int = 256) -> None:
        super().__init__()
        self.d = int(d)
        self.adapter = TwoLayerAdapter(4 * self.d, self.d, hidden_dim=int(hidden_dim))
        self.scale = BoundedLayerScale(initial=0.05, maximum=2.0)
        self.query_norm = NonAffineRMSNorm(self.d)

    def forward(
        self,
        base_query: torch.Tensor,
        current_state: torch.Tensor,
        history_memory: torch.Tensor,
        reference_memory: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        shapes = {
            tuple(base_query.shape),
            tuple(current_state.shape),
            tuple(history_memory.shape),
            tuple(reference_memory.shape),
        }
        if len(shapes) != 1 or base_query.ndim != 2 or int(base_query.size(-1)) != self.d:
            raise ValueError("matched-history scorer inputs must share shape [batch, d]")
        if history_mask.ndim != 1 or int(history_mask.numel()) != int(base_query.size(0)):
            raise ValueError("matched-history scorer mask must have one value per row")
        delta = self.adapter(
            torch.cat(
                (
                    current_state,
                    history_memory,
                    reference_memory,
                    history_memory - reference_memory,
                ),
                dim=-1,
            )
        )
        candidate = self.query_norm(
            base_query + self.scale().to(delta.dtype) * torch.tanh(delta)
        )
        return torch.where(
            history_mask.to(device=base_query.device, dtype=torch.bool).unsqueeze(-1),
            candidate,
            base_query,
        )


def multi_positive_support_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or positive_mask.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("matched-history loss tensors must share [batch, candidates]")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    eligible = positive.any(dim=-1)
    if not bool(eligible.any().item()):
        return logits.sum() * 0.0, eligible
    floor = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~valid, floor), dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    return (denominator[eligible] - numerator[eligible]).mean(), eligible


def best_positive_support_rank(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if any(
        tensor.shape != logits.shape
        for tensor in (candidate_ids, positive_mask, valid_mask)
    ):
        raise ValueError("matched-history rank tensors must share [batch, candidates]")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive = positive_mask.to(device=logits.device, dtype=torch.bool) & valid
    floor = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive, floor)
    best_values = positive_logits.max(dim=-1).values
    sentinel = int(candidate_ids.max().item()) + 1 if candidate_ids.numel() else 1
    tied_positive_ids = torch.where(
        positive & logits.eq(best_values.unsqueeze(-1)),
        candidate_ids,
        torch.full_like(candidate_ids, sentinel),
    )
    best_ids = tied_positive_ids.min(dim=-1).values
    ranks = 1 + (
        valid
        & (
            logits.gt(best_values.unsqueeze(-1))
            | (logits.eq(best_values.unsqueeze(-1)) & candidate_ids.lt(best_ids.unsqueeze(-1)))
        )
    ).sum(dim=-1)
    return torch.where(positive.any(dim=-1), ranks, torch.zeros_like(ranks))


def _anchors(
    trajectories: list[list[dict[str, Any]]],
) -> dict[str, list[tuple[list[dict[str, Any]], int]]]:
    output: dict[str, list[tuple[list[dict[str, Any]], int]]] = defaultdict(list)
    for rows in trajectories:
        benchmark = str(rows[0].get("benchmark") or "")
        for decision_index in range(1, len(rows)):
            output[benchmark].append((rows, decision_index))
    if set(output) != {"toolbench_g3", "tau2"} or any(not values for values in output.values()):
        raise ValueError("matched-history data must contain history-bearing ToolBench and tau2 rows")
    return dict(output)


def _stable_subset(
    values: list[tuple[list[dict[str, Any]], int]],
    limit: int | None,
) -> list[tuple[list[dict[str, Any]], int]]:
    ordered = sorted(
        values,
        key=lambda item: (
            str(item[0][0].get("trajectory_id") or ""),
            int(item[1]),
        ),
    )
    return ordered if limit is None else ordered[: max(0, int(limit))]


def _balanced_sample(
    anchors: dict[str, list[tuple[list[dict[str, Any]], int]]],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[list[dict[str, Any]], int]]:
    benchmarks = ("toolbench_g3", "tau2")
    return [
        rng.choice(anchors[benchmarks[index % len(benchmarks)]])
        for index in range(int(batch_size))
    ]


@dataclass
class MatchedHistoryBatch:
    current_state: torch.Tensor
    initial_memory: torch.Tensor
    static_memory: torch.Tensor
    base_query: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_embeddings: torch.Tensor
    positive_mask: torch.Tensor
    event_states: torch.Tensor
    event_skills: torch.Tensor
    event_actions: torch.Tensor
    event_results: torch.Tensor
    event_mask: torch.Tensor
    result_mask: torch.Tensor
    serialized_history_embedding: torch.Tensor | None
    benchmarks: list[str]


def _batch_from_anchors(
    anchors: Sequence[tuple[list[dict[str, Any]], int]],
    *,
    foundation: Any,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    history_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    max_horizon: int,
    support_k: int,
    belief_top_k: int,
) -> MatchedHistoryBatch:
    rows = [trajectory[index] for trajectory, index in anchors]
    events = [
        history_events_for_decision(trajectory, index, max_horizon=max_horizon)
        for trajectory, index in anchors
    ]
    current = state_cache.batch(
        [str(row["state_text_current"]) for row in rows],
        device=device,
    )
    legal = runtime_visible_mask(
        rows,
        skill_id_to_idx,
        device=device,
        inventory_catalogs=catalogs,
    )
    initial_rows = [trajectory[0] for trajectory, _index in anchors]
    initial_states = state_cache.batch(
        [str(row["state_text_current"]) for row in initial_rows],
        device=device,
    )
    initial_legal = runtime_visible_mask(
        initial_rows,
        skill_id_to_idx,
        device=device,
        inventory_catalogs=catalogs,
    )
    static_texts = [
        serialize_compact_causal_state(
            str(row["state_text_current"]),
            event_rows,
            max_events=8,
            max_action_chars=256,
        )[0]
        for row, event_rows in zip(rows, events)
    ]
    static_states = state_cache.batch(static_texts, device=device)
    with torch.no_grad():
        initial = foundation.vnext_initial_belief(
            initial_states,
            initial_legal,
            top_k=belief_top_k,
        )
        static_memory = foundation.vnext_initial_belief(
            static_states,
            legal,
            top_k=belief_top_k,
        )
        static_recall, _static_delta, base_query = foundation.vnext.static_query_components(
            static_states,
            static_memory,
        )
        full_static = foundation.vnext_full_pool_logits(static_recall, head="recall")
        candidate_ids, candidate_valid = masked_topk_tensor(
            full_static,
            legal,
            k=support_k,
        )
        skill_embeddings = foundation.vnext_normalized_skill_embeddings(dtype=current.dtype)
        candidate_embeddings = skill_embeddings.index_select(
            0,
            candidate_ids.reshape(-1),
        ).view(candidate_ids.size(0), candidate_ids.size(1), -1)

    batch_size = len(rows)
    horizon = max((len(value) for value in events), default=0)
    d = int(current.size(-1))
    shape = (batch_size, horizon, d)
    event_states = torch.zeros(shape, device=device, dtype=current.dtype)
    event_skills = torch.zeros(shape, device=device, dtype=current.dtype)
    event_actions = torch.zeros(shape, device=device, dtype=current.dtype)
    event_results = torch.zeros(shape, device=device, dtype=current.dtype)
    event_mask = torch.zeros(batch_size, horizon, device=device, dtype=torch.bool)
    result_mask = torch.zeros_like(event_mask)
    positions: list[tuple[int, int]] = []
    flat_states: list[str] = []
    flat_actions: list[str] = []
    flat_results: list[str] = []
    flat_result_positions: list[tuple[int, int]] = []
    flat_skill_indices: list[int] = []
    for batch_index, event_rows in enumerate(events):
        for event_index, event in enumerate(event_rows):
            positions.append((batch_index, event_index))
            flat_states.append(str(event["state_text"]))
            flat_actions.append(str(event["action_text"]))
            flat_skill_indices.append(skill_id_to_idx[str(event["skill_id"])])
            event_mask[batch_index, event_index] = True
            if bool(event.get("result_executed")):
                flat_result_positions.append((batch_index, event_index))
                flat_results.append(str(event["result_text"]))
                result_mask[batch_index, event_index] = True
    if positions:
        state_values = state_cache.batch(flat_states, device=device)
        action_values = action_cache.batch(flat_actions, device=device)
        skill_values = foundation.vnext_normalized_skill_embeddings(
            dtype=current.dtype
        ).index_select(
            0,
            torch.tensor(flat_skill_indices, device=device, dtype=torch.long),
        )
        for flat_index, (batch_index, event_index) in enumerate(positions):
            event_states[batch_index, event_index] = state_values[flat_index]
            event_actions[batch_index, event_index] = action_values[flat_index]
            event_skills[batch_index, event_index] = skill_values[flat_index]
    if flat_results:
        if result_cache is None:
            raise RuntimeError("matched-history visible results require a result cache")
        result_values = result_cache.batch(flat_results, device=device)
        for flat_index, (batch_index, event_index) in enumerate(flat_result_positions):
            event_results[batch_index, event_index] = result_values[flat_index]

    serialized = None
    if history_cache is not None:
        serialized = history_cache.batch(
            [serialize_factual_history(value) for value in events],
            device=device,
        )
    positive_mask = torch.zeros_like(candidate_valid)
    for row_index, row in enumerate(rows):
        positive_indices = torch.tensor(
            [skill_id_to_idx[str(value)] for value in row["positive_skill_ids"]],
            device=device,
            dtype=torch.long,
        )
        positive_mask[row_index] = (
            candidate_ids[row_index].unsqueeze(-1) == positive_indices.unsqueeze(0)
        ).any(dim=-1) & candidate_valid[row_index]
    return MatchedHistoryBatch(
        current_state=current,
        initial_memory=initial,
        static_memory=static_memory,
        base_query=base_query,
        candidate_ids=candidate_ids,
        candidate_valid=candidate_valid,
        candidate_embeddings=candidate_embeddings,
        positive_mask=positive_mask,
        event_states=event_states,
        event_skills=event_skills,
        event_actions=event_actions,
        event_results=event_results,
        event_mask=event_mask,
        result_mask=result_mask,
        serialized_history_embedding=serialized,
        benchmarks=[str(row["benchmark"]) for row in rows],
    )


def _route_logits(
    encoder: nn.Module,
    scorer: MatchedHistoryRouteScorer,
    batch: MatchedHistoryBatch,
    *,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory = encoder(
        batch.initial_memory,
        batch.event_states,
        batch.event_skills,
        batch.event_actions,
        batch.event_results,
        batch.event_mask,
        batch.result_mask,
        serialized_history_embedding=batch.serialized_history_embedding,
    )
    query = scorer(
        batch.base_query,
        batch.current_state,
        memory,
        batch.static_memory,
        batch.event_mask.any(dim=-1),
    )
    logits = temperature.to(query.dtype) * torch.einsum(
        "bd,bcd->bc",
        query,
        batch.candidate_embeddings,
    )
    static_logits = temperature.to(query.dtype) * torch.einsum(
        "bd,bcd->bc",
        batch.base_query,
        batch.candidate_embeddings,
    )
    floor = torch.finfo(logits.dtype).min
    return (
        logits.masked_fill(~batch.candidate_valid, floor),
        static_logits.masked_fill(~batch.candidate_valid, floor),
    )


def _metric_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "support_hit": 0.0,
        "mrr": 0.0,
        "r1": 0.0,
        "r5": 0.0,
        "r20": 0.0,
        "static_mrr": 0.0,
        "static_r1": 0.0,
        "static_r5": 0.0,
        "static_r20": 0.0,
    }


def _finalize_metrics(accumulator: dict[str, float]) -> dict[str, float | int]:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("matched-history metric group is empty")
    return {
        "count": count,
        "candidate_recall_at_500": accumulator["support_hit"] / count,
        "mrr": accumulator["mrr"] / count,
        "recall_at_1": accumulator["r1"] / count,
        "recall_at_5": accumulator["r5"] / count,
        "recall_at_20": accumulator["r20"] / count,
        "static_mrr": accumulator["static_mrr"] / count,
        "static_recall_at_1": accumulator["static_r1"] / count,
        "static_recall_at_5": accumulator["static_r5"] / count,
        "static_recall_at_20": accumulator["static_r20"] / count,
    }


def _evaluate(
    encoder: nn.Module,
    scorer: MatchedHistoryRouteScorer,
    anchors: dict[str, list[tuple[list[dict[str, Any]], int]]],
    *,
    foundation: Any,
    state_cache: Any,
    action_cache: Any,
    result_cache: Any | None,
    history_cache: Any | None,
    catalogs: dict[str, dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    device: torch.device,
    max_horizon: int,
    support_k: int,
    belief_top_k: int,
    batch_size: int,
    collect_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoder.eval()
    scorer.eval()
    temperature = foundation.vnext.static_query.temperature().detach()
    accumulators = {benchmark: _metric_accumulator() for benchmark in anchors}
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for benchmark, values in sorted(anchors.items()):
            for start in range(0, len(values), int(batch_size)):
                anchor_batch = values[start : start + int(batch_size)]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    batch = _batch_from_anchors(
                        anchor_batch,
                        foundation=foundation,
                        state_cache=state_cache,
                        action_cache=action_cache,
                        result_cache=result_cache,
                        history_cache=history_cache,
                        catalogs=catalogs,
                        skill_id_to_idx=skill_id_to_idx,
                        device=device,
                        max_horizon=max_horizon,
                        support_k=support_k,
                        belief_top_k=belief_top_k,
                    )
                    logits, static_logits = _route_logits(
                        encoder,
                        scorer,
                        batch,
                        temperature=temperature,
                    )
                ranks = best_positive_support_rank(
                    logits.float(),
                    batch.candidate_ids,
                    batch.positive_mask,
                    batch.candidate_valid,
                )
                static_ranks = best_positive_support_rank(
                    static_logits.float(),
                    batch.candidate_ids,
                    batch.positive_mask,
                    batch.candidate_valid,
                )
                for row_index, (rank, static_rank) in enumerate(
                    zip(ranks.tolist(), static_ranks.tolist())
                ):
                    target = accumulators[benchmark]
                    target["count"] += 1
                    target["support_hit"] += float(rank > 0)
                    target["mrr"] += 0.0 if rank <= 0 else 1.0 / float(rank)
                    target["r1"] += float(0 < rank <= 1)
                    target["r5"] += float(0 < rank <= 5)
                    target["r20"] += float(0 < rank <= 20)
                    target["static_mrr"] += (
                        0.0 if static_rank <= 0 else 1.0 / float(static_rank)
                    )
                    target["static_r1"] += float(0 < static_rank <= 1)
                    target["static_r5"] += float(0 < static_rank <= 5)
                    target["static_r20"] += float(0 < static_rank <= 20)
                    if collect_predictions:
                        trajectory, decision_index = anchor_batch[row_index]
                        row = trajectory[decision_index]
                        valid_ids = batch.candidate_ids[row_index][
                            batch.candidate_valid[row_index]
                        ].detach().cpu().tolist()
                        support_payload = json.dumps(
                            valid_ids,
                            separators=(",", ":"),
                        )
                        predictions.append(
                            {
                                "benchmark": benchmark,
                                "trajectory_id": str(row["trajectory_id"]),
                                "decision_index": int(decision_index),
                                "candidate_support_sha256": hashlib.sha256(
                                    support_payload.encode("utf-8")
                                ).hexdigest(),
                                "candidate_support_size": len(valid_ids),
                                "positive_skill_ids": list(row["positive_skill_ids"]),
                                "support_hit": bool(rank > 0),
                                "rank": int(rank),
                                "reciprocal_rank": (
                                    0.0 if rank <= 0 else 1.0 / float(rank)
                                ),
                                "static_rank": int(static_rank),
                                "static_reciprocal_rank": (
                                    0.0
                                    if static_rank <= 0
                                    else 1.0 / float(static_rank)
                                ),
                            }
                        )
    metrics = {
        benchmark: _finalize_metrics(values)
        for benchmark, values in sorted(accumulators.items())
    }
    metrics["macro_mrr"] = sum(float(value["mrr"]) for value in metrics.values()) / len(
        metrics
    )
    encoder.train()
    scorer.train()
    return metrics, predictions


def _collect_cache_texts(
    anchors: Iterable[tuple[list[dict[str, Any]], int]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    states: list[str] = []
    actions: list[str] = []
    results: list[str] = []
    histories: list[str] = []
    for trajectory, index in anchors:
        states.append(str(trajectory[index]["state_text_current"]))
        events = history_events_for_decision(trajectory, index)
        states.append(str(trajectory[0]["state_text_current"]))
        states.append(
            serialize_compact_causal_state(
                str(trajectory[index]["state_text_current"]),
                events,
                max_events=8,
                max_action_chars=256,
            )[0]
        )
        histories.append(serialize_factual_history(events))
        for event in events:
            states.append(str(event["state_text"]))
            actions.append(str(event["action_text"]))
            if bool(event.get("result_executed")):
                results.append(str(event["result_text"]))
    return states, actions, results, histories


def _source_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def train_matched_history_e1(
    *,
    encoder_kind: HistoryEncoderKind,
    foundation_checkpoint_path: str | Path,
    skills_path: str | Path,
    train_rows_path: str | Path,
    dev_rows_path: str | Path,
    inventory_catalogs_path: str | Path,
    output_dir: str | Path,
    frozen_cache_dir: str | Path,
    max_steps: int = 500,
    batch_size: int = 8,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 0.01,
    seed: int = 23,
    max_horizon: int = 16,
    support_k: int = 500,
    belief_top_k: int = 64,
    cache_batch_size: int = 128,
    cache_shard_size: int = 4096,
    validation_interval: int = 100,
    validation_batch_size: int = 32,
    max_train_rows_per_benchmark: int | None = None,
    max_dev_rows_per_benchmark: int | None = None,
) -> dict[str, Any]:
    if int(max_steps) <= 0 or int(batch_size) <= 0 or int(validation_interval) <= 0:
        raise ValueError("matched-history steps, batch size, and validation interval must be positive")
    if int(max_horizon) != 16 or int(support_k) != 500:
        raise ValueError("canonical E1 requires horizon 16 and fixed Static Top-500 support")
    if not torch.cuda.is_available():
        raise RuntimeError("matched-history E1 training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    random.seed(int(seed))
    torch.set_float32_matmul_precision("high")

    paths = [
        Path(foundation_checkpoint_path).resolve(),
        Path(skills_path).resolve(),
        Path(train_rows_path).resolve(),
        Path(dev_rows_path).resolve(),
        Path(inventory_catalogs_path).resolve(),
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("matched-history E1 is missing an immutable input")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("matched-history output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    foundation, _skills, skill_id_to_idx, foundation_report = (
        load_vnext_stage2_for_evaluation(
            checkpoint_path=paths[0],
            training_skills_path=paths[1],
            benchmark_skills=[],
            device=device,
        )
    )
    foundation.eval()
    for parameter in foundation.parameters():
        parameter.requires_grad_(False)
    catalogs = load_inventory_catalogs(paths[4])
    train_anchors = _anchors(load_matched_history_trajectories(paths[2]))
    dev_anchors = _anchors(load_matched_history_trajectories(paths[3]))
    train_anchors = {
        benchmark: _stable_subset(values, max_train_rows_per_benchmark)
        for benchmark, values in train_anchors.items()
    }
    dev_anchors = {
        benchmark: _stable_subset(values, max_dev_rows_per_benchmark)
        for benchmark, values in dev_anchors.items()
    }
    all_selected = [
        item
        for collection in (*train_anchors.values(), *dev_anchors.values())
        for item in collection
    ]
    state_texts, action_texts, result_texts, history_texts = _collect_cache_texts(
        all_selected
    )
    cache_root = Path(frozen_cache_dir).resolve()
    state_cache = load_or_build_frozen_text_cache(
        foundation,
        state_texts,
        role="state",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    action_cache = load_or_build_frozen_text_cache(
        foundation,
        action_texts,
        role="action",
        batch_size=cache_batch_size,
        cache_root=cache_root,
        cache_shard_size=cache_shard_size,
    )
    result_cache = (
        load_or_build_frozen_text_cache(
            foundation,
            result_texts,
            role="result",
            batch_size=cache_batch_size,
            cache_root=cache_root,
            cache_shard_size=cache_shard_size,
        )
        if result_texts
        else None
    )
    history_cache = (
        load_or_build_frozen_text_cache(
            foundation,
            history_texts,
            role="matched_history",
            batch_size=cache_batch_size,
            cache_root=cache_root,
            cache_shard_size=cache_shard_size,
        )
        if str(encoder_kind) == "serialized"
        else None
    )

    d = int(foundation.vnext.d)
    encoder = build_matched_history_encoder(
        encoder_kind,
        d,
        max_horizon=max_horizon,
    ).to(device)
    scorer = MatchedHistoryRouteScorer(d).to(device)
    parameters = [*encoder.parameters(), *scorer.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = random.Random(int(seed))
    validation_records: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_step = 0
    best_path = output / "best.pt"
    started = time.perf_counter()
    temperature = foundation.vnext.static_query.temperature().detach()
    for step in range(1, int(max_steps) + 1):
        selected = _balanced_sample(train_anchors, batch_size=batch_size, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch = _batch_from_anchors(
                selected,
                foundation=foundation,
                state_cache=state_cache,
                action_cache=action_cache,
                result_cache=result_cache,
                history_cache=history_cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                max_horizon=max_horizon,
                support_k=support_k,
                belief_top_k=belief_top_k,
            )
            logits, _static_logits = _route_logits(
                encoder,
                scorer,
                batch,
                temperature=temperature,
            )
            loss, eligible = multi_positive_support_loss(
                logits.float(),
                batch.positive_mask,
                batch.candidate_valid,
            )
        if not bool(eligible.any().item()) or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("matched-history batch lacks a finite eligible objective")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("matched-history gradient norm is not finite")
        optimizer.step()

        if step % int(validation_interval) == 0 or step == int(max_steps):
            metrics, _predictions = _evaluate(
                encoder,
                scorer,
                dev_anchors,
                foundation=foundation,
                state_cache=state_cache,
                action_cache=action_cache,
                result_cache=result_cache,
                history_cache=history_cache,
                catalogs=catalogs,
                skill_id_to_idx=skill_id_to_idx,
                device=device,
                max_horizon=max_horizon,
                support_k=support_k,
                belief_top_k=belief_top_k,
                batch_size=validation_batch_size,
                collect_predictions=False,
            )
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "gradient_norm": float(gradient_norm.detach().cpu().item()),
                "metrics": metrics,
            }
            validation_records.append(record)
            score = float(metrics["macro_mrr"])
            if score > best_score:
                best_score = score
                best_step = step
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "schema_version": MATCHED_HISTORY_TRAIN_SCHEMA,
                        "encoder_kind": str(encoder_kind),
                        "seed": int(seed),
                        "step": step,
                        "encoder_state_dict": encoder.state_dict(),
                        "scorer_state_dict": scorer.state_dict(),
                        "metrics": metrics,
                    },
                    temporary,
                )
                temporary.replace(best_path)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    selected_payload = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(selected_payload["encoder_state_dict"])
    scorer.load_state_dict(selected_payload["scorer_state_dict"])
    final_metrics, prediction_records = _evaluate(
        encoder,
        scorer,
        dev_anchors,
        foundation=foundation,
        state_cache=state_cache,
        action_cache=action_cache,
        result_cache=result_cache,
        history_cache=history_cache,
        catalogs=catalogs,
        skill_id_to_idx=skill_id_to_idx,
        device=device,
        max_horizon=max_horizon,
        support_k=support_k,
        belief_top_k=belief_top_k,
        batch_size=validation_batch_size,
        collect_predictions=True,
    )
    if abs(float(final_metrics["macro_mrr"]) - float(best_score)) > 1.0e-12:
        raise RuntimeError("matched-history selected checkpoint did not reproduce its dev score")
    predictions_path = output / "dev_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": MATCHED_HISTORY_TRAIN_SCHEMA,
        "status": "ok",
        "encoder_kind": str(encoder_kind),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "best_step": int(best_step),
        "best_macro_mrr": float(best_score),
        "best_checkpoint_path": str(best_path),
        "best_checkpoint_sha256": file_sha256(best_path),
        "best_metrics": final_metrics,
        "dev_predictions_path": str(predictions_path),
        "dev_predictions_sha256": file_sha256(predictions_path),
        "dev_prediction_count": len(prediction_records),
        "trainable_parameters": {
            "history_encoder": trainable_parameter_count(encoder),
            "shared_route_scorer": trainable_parameter_count(scorer),
            "total": trainable_parameter_count(encoder) + trainable_parameter_count(scorer),
        },
        "contract": {
            "max_horizon": int(max_horizon),
            "candidate_support": "bounded_last8_skill_action_frozen_static_top500",
            "support_k": int(support_k),
            "belief_top_k": int(belief_top_k),
            "same_shared_scorer": True,
            "same_multi_positive_objective": True,
            "frozen_foundation": True,
            "dynamic_candidate_extras": False,
            "selector": False,
            "history_decisions_only": True,
            "checkpoint_selection": "macro_dev_mrr_across_toolbench_g3_and_tau2",
        },
        "train_rows_by_benchmark": {
            key: len(value) for key, value in sorted(train_anchors.items())
        },
        "dev_rows_by_benchmark": {
            key: len(value) for key, value in sorted(dev_anchors.items())
        },
        "inputs": {
            "foundation": foundation_report,
            "train_rows_path": str(paths[2]),
            "train_rows_sha256": file_sha256(paths[2]),
            "dev_rows_path": str(paths[3]),
            "dev_rows_sha256": file_sha256(paths[3]),
            "inventory_catalogs_path": str(paths[4]),
            "inventory_catalogs_sha256": file_sha256(paths[4]),
        },
        "cache": {
            "state": state_cache.report(),
            "action": action_cache.report(),
            "result": None if result_cache is None else result_cache.report(),
            "matched_history": None if history_cache is None else history_cache.report(),
        },
        "validation_records": validation_records,
        "elapsed_seconds": time.perf_counter() - started,
        "source_commit": _source_commit(root),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    (output / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
