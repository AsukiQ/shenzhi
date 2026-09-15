#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.history_channel import strip_history_sections

POOLED_ATOL = 0.0
PROJECTED_ATOL = 1.0e-6
LOGITS_ATOL = 1.0e-5
LOSS_ATOL = 1.0e-6
GRADIENT_ATOL = 2.0e-5
PARAMETER_UPDATE_ATOL = 2.0e-6
MIN_SPEEDUP = 1.5

PARITY_THRESHOLDS = {
    "pooled_max_abs_diff": POOLED_ATOL,
    "projected_max_abs_diff": PROJECTED_ATOL,
    "logits_max_abs_diff": LOGITS_ATOL,
    "loss_abs_diff": LOSS_ATOL,
    "gradient_max_abs_diff": GRADIENT_ATOL,
    "parameter_update_max_abs_diff": PARAMETER_UPDATE_ATOL,
}


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"tensor shape mismatch: {list(left.shape)} != {list(right.shape)}"
        )
    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    if difference.numel() == 0:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    return {
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
    }


def _mapping_difference(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    per_parameter: dict[str, dict[str, float]] = {}
    max_abs_diff = 0.0
    means: list[float] = []
    for name in sorted(set(left) | set(right)):
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None and right_value is None:
            current = {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
        elif left_value is None or right_value is None:
            current = {"max_abs_diff": math.inf, "mean_abs_diff": math.inf}
        else:
            current = tensor_difference(left_value, right_value)
        per_parameter[name] = current
        max_abs_diff = max(max_abs_diff, float(current["max_abs_diff"]))
        means.append(float(current["mean_abs_diff"]))
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum(means) / max(len(means), 1),
        "per_parameter": per_parameter,
    }


def decide_cache_gate(
    *,
    parity: dict[str, float],
    steady_state_speedup: float,
    projected_end_to_end_speedup: float,
    backbone_trainable_parameter_count: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    for key, threshold in PARITY_THRESHOLDS.items():
        value = float(parity.get(key, math.inf))
        if not math.isfinite(value) or value > threshold:
            blockers.append(f"{key}_above_{threshold:g}")
    if float(steady_state_speedup) < MIN_SPEEDUP:
        blockers.append("steady_state_speedup_below_1.5")
    if float(projected_end_to_end_speedup) < MIN_SPEEDUP:
        blockers.append("projected_end_to_end_speedup_below_1.5")
    if int(backbone_trainable_parameter_count) != 0:
        blockers.append("backbone_not_fully_frozen")
    return {
        "status": "ok" if not blockers else "blocked",
        "blockers": blockers,
        "minimum_speedup": MIN_SPEEDUP,
        "parity_thresholds": dict(PARITY_THRESHOLDS),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prepare_queries(
    *,
    data_root: Path,
    seed: int,
    checkpoint_step: int,
    target_step: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    sampling_strategy: str,
    tempered_correction_fraction: float,
    expand_alias_positives: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[int]],
]:
    from clstr.retrieval_warmup import (
        _batch_queries_for_step,
        _build_training_buckets,
        _filter_queries_to_skill_pool,
        _skill_pool_alias_positive_ids_by_skill_id,
        assign_stage0_query_indices,
        load_retrieval_warmup_rows,
        plan_stage0_scheduled_rows,
    )

    skill_rows, query_rows, positives_by_query = load_retrieval_warmup_rows(
        data_root,
        data_format="unified_v2",
    )
    skill_id_to_idx = {
        str(row["skill_id"]): index
        for index, row in enumerate(skill_rows)
    }
    alias_positive_ids = (
        _skill_pool_alias_positive_ids_by_skill_id(skill_rows)
        if expand_alias_positives
        else {}
    )
    queries = _filter_queries_to_skill_pool(
        query_rows,
        positives_by_query,
        skill_id_to_idx,
        max_queries=None,
        alias_positive_ids_by_skill_id=alias_positive_ids,
    )
    random.Random(seed).shuffle(queries)
    assign_stage0_query_indices(queries)
    source_buckets = _build_training_buckets(
        queries,
        sampling_strategy=sampling_strategy,
    )
    start_step = int(checkpoint_step) + 1
    scheduled_rows = plan_stage0_scheduled_rows(
        queries=queries,
        start_step=start_step,
        max_steps=target_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        sampling_strategy=sampling_strategy,
        source_buckets=source_buckets,
        tempered_correction_fraction=tempered_correction_fraction,
        batch_sampler=_batch_queries_for_step,
    )
    first_micro_index = (
        checkpoint_step * gradient_accumulation_steps + 1
    )
    fixed_batch = _batch_queries_for_step(
        queries,
        first_micro_index,
        batch_size,
        sampling_strategy=sampling_strategy,
        source_buckets=source_buckets,
        tempered_correction_fraction=tempered_correction_fraction,
    )
    positive_indices = [
        [int(index) for index in row["positive_indices"]]
        for row in fixed_batch
    ]
    return skill_rows, scheduled_rows, fixed_batch, positive_indices


def _configure_stage0_trainable_parameters(
    model: Any,
    config: dict[str, Any],
) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def enable_parameters(parameters) -> None:
        for parameter in parameters:
            parameter.requires_grad_(True)

    if bool(config.get("train_encoder_projection", True)):
        enable_parameters(model.encoder.proj.parameters())
    if bool(config.get("train_skill_adapter", True)):
        enable_parameters(model.skill_table.W.parameters())
    model.skill_table.E.requires_grad_(
        bool(config.get("train_skill_embeddings", False))
    )
    model.skill_table.logit_scale_retr.requires_grad_(
        bool(config.get("train_retrieval_scale", True))
    )
    model.skill_table.skill_bias_retr.requires_grad_(
        bool(config.get("train_skill_bias", False))
    )
    if str(config.get("route_scorer")) == "unified_memory":
        enable_parameters(model.initial_belief_head.parameters())
        enable_parameters(model.unified_retriever.parameters())
        model.skill_table.logit_scale_belief.requires_grad_(True)
        model.skill_table.skill_bias_belief.requires_grad_(True)
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _forward_from_pooled(
    model: Any,
    pooled: torch.Tensor,
    positive_indices: list[list[int]],
    *,
    belief_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from clstr.retrieval_warmup import _multi_positive_nll

    h = model.encoder.project_pooled(pooled)
    m_0 = model.initial_belief(h, top_k=belief_top_k)
    logits = model.unified_route_logits(h, m_0)
    loss = _multi_positive_nll(logits, positive_indices)
    return h, logits, loss


def _snapshot_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


def _restore_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    snapshot: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(
                snapshot[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
            parameter.grad = None


def _run_single_update(
    model: Any,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    pooled: torch.Tensor,
    positive_indices: list[list[int]],
    *,
    belief_top_k: int,
    learning_rate: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_parameters],
        lr=learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    h, logits, loss = _forward_from_pooled(
        model,
        pooled,
        positive_indices,
        belief_top_k=belief_top_k,
    )
    loss.backward()
    gradients = {
        name: (
            None
            if parameter.grad is None
            else parameter.grad.detach().cpu().clone()
        )
        for name, parameter in named_parameters
    }
    optimizer.step()
    updated_parameters = _snapshot_parameters(named_parameters)
    return {
        "projected": h.detach().cpu(),
        "logits": logits.detach().cpu(),
        "loss": float(loss.detach().cpu().item()),
        "gradients": gradients,
        "updated_parameters": updated_parameters,
    }


def _timed_training_iterations(
    *,
    model: Any,
    forward_loss,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float, int]:
    for _ in range(max(0, int(warmup_iterations))):
        model.zero_grad(set_to_none=True)
        forward_loss().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(max(1, int(timed_iterations))):
        model.zero_grad(set_to_none=True)
        forward_loss().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    return elapsed / max(1, int(timed_iterations)), peak_memory


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from clstr.retrieval_warmup import (
        _multi_positive_nll,
        _stage0_resume_expected_config,
        build_stage0_frozen_backbone_cache,
    )
    from clstr.stage_checkpoint_init import (
        build_clstr_model_from_stage0_checkpoint,
        checkpoint_payload,
    )
    from clstr.stage0_skill_pool_identity import (
        validate_verified_resume_skill_table,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Stage0 cache benchmark requires CUDA")
    checkpoint_path = Path(args.checkpoint_path)
    skills_path = Path(args.skills_path)
    data_root = Path(args.data_root)
    payload = checkpoint_payload(checkpoint_path, "stage0")
    raw_config = dict(payload.get("config") or {})
    checkpoint_step = int(payload.get("step") or 0)
    if str(raw_config.get("route_scorer")) != "unified_memory":
        raise ValueError("cache benchmark requires route_scorer=unified_memory")
    if bool(raw_config.get("train_encoder_backbone", False)):
        raise ValueError("cache benchmark refuses a trainable encoder backbone")
    if int(args.target_step) <= checkpoint_step:
        raise ValueError("target_step must exceed checkpoint step")

    model, model_config, load_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
    )
    device = torch.device("cuda")
    model.to(device)
    model.train()
    named_parameters = _configure_stage0_trainable_parameters(model, raw_config)
    if not named_parameters:
        raise ValueError("cache benchmark found no Stage0 trainable parameters")
    backbone_trainable_parameter_count = sum(
        int(parameter.numel())
        for parameter in model.encoder.backbone.parameters()
        if parameter.requires_grad
    )

    gradient_accumulation_steps = int(
        raw_config.get("gradient_accumulation_steps")
        or args.gradient_accumulation_steps
    )
    sampling_strategy = str(
        raw_config.get("sampling_strategy") or "handoff_balanced"
    )
    tempered_correction_fraction = float(
        raw_config.get("tempered_correction_fraction", 0.2)
    )
    skill_rows, scheduled_rows, fixed_batch, positive_indices = _prepare_queries(
        data_root=data_root,
        seed=int(args.seed),
        checkpoint_step=checkpoint_step,
        target_step=int(args.target_step),
        batch_size=int(args.batch_size),
        gradient_accumulation_steps=gradient_accumulation_steps,
        sampling_strategy=sampling_strategy,
        tempered_correction_fraction=tempered_correction_fraction,
        expand_alias_positives=bool(
            raw_config.get("expand_alias_positives", True)
        ),
    )
    selected_skills = _read_jsonl(skills_path)
    selected_ids = [str(row.get("skill_id") or "") for row in selected_skills]
    data_ids = [str(row.get("skill_id") or "") for row in skill_rows]
    if selected_ids != data_ids:
        raise ValueError("selected skills do not match ordered training skill pool")
    verified_resume_skill_table = validate_verified_resume_skill_table(
        checkpoint_state=dict(payload.get("model_state_dict") or {}),
        checkpoint_config=raw_config,
        checkpoint_identity=raw_config.get("skill_pool_identity"),
        resume_skill_rows=selected_skills,
        current_skill_rows=skill_rows,
        expected_shape=tuple(model.skill_table.E.shape),
        expected_config=_stage0_resume_expected_config(model),
        skill_text_fn=model.skill_table.skill_text_fn,
    )

    cache_identity = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "target_step": int(args.target_step),
        "sampling_strategy": sampling_strategy,
        "seed": int(args.seed),
        "state_query_prompt_version": model_config[
            "state_query_prompt_version"
        ],
    }
    cache = build_stage0_frozen_backbone_cache(
        model,
        scheduled_rows,
        batch_size=int(args.cache_batch_size),
        identity=cache_identity,
    )
    cache_report = cache.report()

    saved_skill_embeddings = model.skill_table.E.detach().clone()
    torch.cuda.synchronize()
    skill_rebuild_started = time.perf_counter()
    model.rebuild_skill_table()
    torch.cuda.synchronize()
    skill_rebuild_seconds = time.perf_counter() - skill_rebuild_started
    with torch.no_grad():
        model.skill_table.E.copy_(saved_skill_embeddings)
    del saved_skill_embeddings

    query_texts = [strip_history_sections(str(row["query"])) for row in fixed_batch]
    formatted_texts = [
        model._serialize_state_for_encoder(text)
        for text in query_texts
    ]
    with torch.no_grad():
        pooled_uncached = model.encoder.encode_backbone_pooled(formatted_texts)
    cache_positions = torch.tensor(
        [
            cache.query_index_to_cache_row[int(row["_stage0_query_index"])]
            for row in fixed_batch
        ],
        dtype=torch.long,
    )
    pooled_cached = cache.pooled_cpu.index_select(0, cache_positions).to(device)
    pooled_difference = tensor_difference(pooled_uncached, pooled_cached)

    base_parameters = _snapshot_parameters(named_parameters)
    uncached_update = _run_single_update(
        model,
        named_parameters,
        pooled_uncached,
        positive_indices,
        belief_top_k=int(raw_config.get("belief_top_k") or 64),
        learning_rate=float(args.learning_rate),
    )
    _restore_parameters(named_parameters, base_parameters)
    cached_update = _run_single_update(
        model,
        named_parameters,
        pooled_cached,
        positive_indices,
        belief_top_k=int(raw_config.get("belief_top_k") or 64),
        learning_rate=float(args.learning_rate),
    )
    _restore_parameters(named_parameters, base_parameters)

    projected_difference = tensor_difference(
        uncached_update["projected"],
        cached_update["projected"],
    )
    logits_difference = tensor_difference(
        uncached_update["logits"],
        cached_update["logits"],
    )
    gradient_difference = _mapping_difference(
        uncached_update["gradients"],
        cached_update["gradients"],
    )
    update_difference = _mapping_difference(
        uncached_update["updated_parameters"],
        cached_update["updated_parameters"],
    )
    parity = {
        "pooled_max_abs_diff": pooled_difference["max_abs_diff"],
        "pooled_mean_abs_diff": pooled_difference["mean_abs_diff"],
        "projected_max_abs_diff": projected_difference["max_abs_diff"],
        "projected_mean_abs_diff": projected_difference["mean_abs_diff"],
        "logits_max_abs_diff": logits_difference["max_abs_diff"],
        "logits_mean_abs_diff": logits_difference["mean_abs_diff"],
        "loss_abs_diff": abs(
            uncached_update["loss"] - cached_update["loss"]
        ),
        "gradient_max_abs_diff": gradient_difference["max_abs_diff"],
        "gradient_mean_abs_diff": gradient_difference["mean_abs_diff"],
        "parameter_update_max_abs_diff": update_difference["max_abs_diff"],
        "parameter_update_mean_abs_diff": update_difference["mean_abs_diff"],
        "gradient_per_parameter": gradient_difference["per_parameter"],
        "parameter_update_per_parameter": update_difference["per_parameter"],
    }

    belief_top_k = int(raw_config.get("belief_top_k") or 64)

    def uncached_loss():
        h = model.encode_states(query_texts)
        m_0 = model.initial_belief(h, top_k=belief_top_k)
        logits = model.unified_route_logits(h, m_0)
        return _multi_positive_nll(logits, positive_indices)

    def cached_loss():
        h = cache.project_batch(
            fixed_batch,
            projection_fn=model.encoder.project_pooled,
            device=device,
        )
        m_0 = model.initial_belief(h, top_k=belief_top_k)
        logits = model.unified_route_logits(h, m_0)
        return _multi_positive_nll(logits, positive_indices)

    uncached_seconds, uncached_peak = _timed_training_iterations(
        model=model,
        forward_loss=uncached_loss,
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    cached_seconds, cached_peak = _timed_training_iterations(
        model=model,
        forward_loss=cached_loss,
        warmup_iterations=int(args.warmup_iterations),
        timed_iterations=int(args.timed_iterations),
    )
    steady_state_speedup = uncached_seconds / max(cached_seconds, 1.0e-12)
    continuation_microbatches = (
        (int(args.target_step) - checkpoint_step)
        * gradient_accumulation_steps
    )
    projected_uncached_seconds = (
        skill_rebuild_seconds
        + continuation_microbatches * uncached_seconds
    )
    projected_cached_seconds = (
        float(cache_report["build_seconds"])
        + continuation_microbatches * cached_seconds
    )
    projected_end_to_end_speedup = (
        projected_uncached_seconds / max(projected_cached_seconds, 1.0e-12)
    )
    decision = decide_cache_gate(
        parity=parity,
        steady_state_speedup=steady_state_speedup,
        projected_end_to_end_speedup=projected_end_to_end_speedup,
        backbone_trainable_parameter_count=backbone_trainable_parameter_count,
    )
    return {
        "status": decision["status"],
        "blockers": decision["blockers"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "target_step": int(args.target_step),
        "model_load_report": load_report,
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fixed_batch_query_ids": [str(row["query_id"]) for row in fixed_batch],
        "scheduled_unique_query_count": len(scheduled_rows),
        "backbone_trainable_parameter_count": backbone_trainable_parameter_count,
        "trainable_parameter_names": [name for name, _parameter in named_parameters],
        "verified_resume_skill_table": verified_resume_skill_table,
        "parity": parity,
        "timing": {
            "warmup_iterations": int(args.warmup_iterations),
            "timed_iterations": int(args.timed_iterations),
            "uncached_seconds_per_microbatch": uncached_seconds,
            "cached_seconds_per_microbatch": cached_seconds,
            "steady_state_speedup": steady_state_speedup,
            "skill_table_rebuild_seconds": skill_rebuild_seconds,
            "cache_build_seconds": float(cache_report["build_seconds"]),
            "continuation_microbatches": continuation_microbatches,
            "projected_uncached_seconds": projected_uncached_seconds,
            "projected_cached_seconds": projected_cached_seconds,
            "projected_end_to_end_speedup": projected_end_to_end_speedup,
            "uncached_peak_gpu_memory_bytes": uncached_peak,
            "cached_peak_gpu_memory_bytes": cached_peak,
            "peak_gpu_memory_bytes": max(uncached_peak, cached_peak),
        },
        "cache": cache.report(),
        "gate": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen-Qwen Stage0 pooled-cache parity and speed."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_step", type=int, default=2400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--timed_iterations", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    report = run_audit(args)
    _write_json(Path(args.output_path), report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
