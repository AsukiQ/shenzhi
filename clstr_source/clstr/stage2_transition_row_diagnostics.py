from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.action_adapter import UniversalActionAdapter
from clstr.full_base_train import (
    DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    STATE_QUERY_ROLE,
    TRANSITION_SCORING_MODE,
    TRANSITION_TEXT_ROLE,
    _apply_replay_prefix_beliefs,
    _apply_skill_text_format,
    _attach_full_base_embedding_cache,
    _attach_policy_embedding_cache,
    _attach_stage0_topm_candidates,
    _batch_action_text_embedding_or_none,
    _batch_cached_or_encode,
    _cap_rows_by_benchmark,
    _embedding_cache_policy,
    _equivalent_skill_ids_by_skill_id,
    _filter_rows_by_allowed_benchmarks,
    _filter_rows_by_train_split,
    _limit_rows_for_smoke,
    _normalize_loss_weights,
    _read_jsonl,
    _row_candidate_indices,
    _stage0_candidate_prior_scores_tensor,
    _skill_id,
    _skipped_embedding_cache_report,
    _transition_candidate_logits_for_mode,
)
from clstr.stage2_real_topm_eval import DEFAULT_STAGE2_V2_LOSS_WEIGHTS
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_existing_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rank_label(logits: torch.Tensor, label: int) -> int:
    if logits.ndim != 1:
        raise ValueError("rank_label expects a rank-1 logits tensor")
    label = int(label)
    order = torch.argsort(logits.detach(), descending=True)
    positions = (order.detach().cpu() == label).nonzero(as_tuple=False)
    return int(positions[0].item()) + 1 if positions.numel() else int(logits.numel()) + 1


def _history_line_count(text: Any) -> int:
    value = str(text or "").strip()
    if not value:
        return 0
    return len([line for line in value.splitlines() if line.strip()])


def _namespace(skill_id: str) -> str:
    return str(skill_id or "").split("/", 1)[0]


def _failure_type(rank: int, stage0_rank: int | None) -> str:
    if rank <= 1:
        return "hit_top1"
    if rank <= 5:
        return "hit_top5_not_top1"
    if stage0_rank is not None and stage0_rank <= 5:
        return "miss_top5_stage0_top5"
    if stage0_rank is not None and stage0_rank <= 20:
        return "miss_top5_stage0_top20"
    return "miss_top5_stage0_low_rank"


def build_transition_row_records(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    candidate_indices: list[list[int]] | None,
    skill_ids: dict[int, str],
    top_k: int = 10,
    row_offset: int = 0,
    transition_skill_head_type: str = "",
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if logits.ndim != 2:
        raise ValueError("transition diagnostics logits must be rank-2 [rows, candidates]")
    top_k = max(1, int(top_k))
    labels_cpu = [int(item) for item in labels.detach().cpu().tolist()]
    records: list[dict[str, Any]] = []
    log_probs = F.log_softmax(logits.float(), dim=-1)
    for local_row_idx, (row, label) in enumerate(zip(rows, labels_cpu)):
        row_logits = logits[local_row_idx].detach()
        width = int(row_logits.numel())
        rank = rank_label(row_logits, label)
        k = min(top_k, width)
        top = torch.topk(row_logits, k=k)
        top_local_indices = [int(item) for item in top.indices.detach().cpu().tolist()]
        top_scores = [float(item) for item in top.values.detach().cpu().tolist()]
        if candidate_indices is None:
            global_candidates = list(range(width))
            stage0_rank = None
        else:
            global_candidates = [int(item) for item in candidate_indices[local_row_idx]]
            stage0_rank = int(label) + 1
        gold_global_idx = int(global_candidates[label]) if 0 <= label < len(global_candidates) else -1
        top_global_indices = [
            int(global_candidates[idx]) if 0 <= idx < len(global_candidates) else int(idx)
            for idx in top_local_indices
        ]
        gold_skill_id = str(row.get("next_skill_id") or skill_ids.get(gold_global_idx, str(gold_global_idx)))
        top_skill_ids = [str(skill_ids.get(idx, str(idx))) for idx in top_global_indices]
        top1_skill_id = top_skill_ids[0] if top_skill_ids else ""
        record = {
            "row_index": int(row.get("_diagnostic_filtered_index", row_offset + local_row_idx)),
            "benchmark": str(row.get("benchmark") or "unknown"),
            "source_quality": str(row.get("source_quality") or "unknown"),
            "candidate_source": str(row.get("candidate_source") or "unknown"),
            "task_id": row.get("task_id"),
            "trajectory_id": row.get("trajectory_id"),
            "step_index": row.get("step_index"),
            "skill_id": row.get("skill_id"),
            "next_skill_id": row.get("next_skill_id"),
            "current_skill_index": row.get("current_skill_index"),
            "gold_next_skill_index": gold_global_idx,
            "candidate_count": width,
            "stage0_gold_rank": stage0_rank,
            "stage2_gold_rank": rank,
            "stage2_hit@1": rank <= 1,
            "stage2_hit@5": rank <= 5,
            "stage2_hit@10": rank <= 10,
            "stage2_hit@20": rank <= 20,
            "stage2_mrr": float(1.0 / rank),
            "stage2_cross_entropy": float(-log_probs[local_row_idx, label].detach().cpu().item()),
            "gold_score": float(row_logits[label].detach().cpu().item()) if 0 <= label < width else None,
            "top_skill_indices": top_global_indices,
            "top_skill_ids": top_skill_ids,
            "top_scores": top_scores,
            "top1_skill_id": top1_skill_id,
            "top1_same_namespace_as_gold": bool(_namespace(top1_skill_id) == _namespace(gold_skill_id)),
            "failure_type": _failure_type(rank, stage0_rank),
            "history_line_count": _history_line_count(row.get("history_text")),
            "state_text_chars": len(str(row.get("state_text") or "")),
            "action_text": row.get("action_text"),
            "next_action_text": row.get("next_action_text"),
            "done": bool(row.get("done")),
            "transition_skill_head_type": transition_skill_head_type,
        }
        if equivalent_skill_ids_by_skill_id is not None:
            record["equivalent_next_skill_ids"] = equivalent_skill_ids_by_skill_id.get(gold_skill_id, [])
        provenance = row.get("provenance")
        if isinstance(provenance, dict):
            record["provenance_source_id"] = provenance.get("source_id")
            record["provenance_split"] = provenance.get("split")
            record["provenance_bucket"] = provenance.get("bucket")
        records.append(record)
    return records


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count <= 0:
        return {
            "row_count": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "mrr": 0.0,
        }

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "row_count": count,
        "recall@1": sum(1 for row in rows if bool(row.get("stage2_hit@1"))) / count,
        "recall@5": sum(1 for row in rows if bool(row.get("stage2_hit@5"))) / count,
        "recall@10": sum(1 for row in rows if bool(row.get("stage2_hit@10"))) / count,
        "recall@20": sum(1 for row in rows if bool(row.get("stage2_hit@20"))) / count,
        "stage0_recall@5": sum(
            1 for row in rows if isinstance(row.get("stage0_gold_rank"), int) and int(row["stage0_gold_rank"]) <= 5
        )
        / count,
        "stage0_recall@10": sum(
            1 for row in rows if isinstance(row.get("stage0_gold_rank"), int) and int(row["stage0_gold_rank"]) <= 10
        )
        / count,
        "stage0_recall@20": sum(
            1 for row in rows if isinstance(row.get("stage0_gold_rank"), int) and int(row["stage0_gold_rank"]) <= 20
        )
        / count,
        "mrr": mean("stage2_mrr"),
        "mean_stage0_gold_rank": mean("stage0_gold_rank"),
        "mean_stage2_gold_rank": mean("stage2_gold_rank"),
        "mean_stage2_cross_entropy": mean("stage2_cross_entropy"),
        "stage2_improved_vs_stage0_fraction": (
            sum(
                1
                for row in rows
                if isinstance(row.get("stage0_gold_rank"), int)
                and isinstance(row.get("stage2_gold_rank"), int)
                and int(row["stage2_gold_rank"]) < int(row["stage0_gold_rank"])
            )
            / count
        ),
        "stage2_worse_than_stage0_fraction": (
            sum(
                1
                for row in rows
                if isinstance(row.get("stage0_gold_rank"), int)
                and isinstance(row.get("stage2_gold_rank"), int)
                and int(row["stage2_gold_rank"]) > int(row["stage0_gold_rank"])
            )
            / count
        ),
    }


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: _stats(group_rows) for name, group_rows in sorted(grouped.items())}


def summarize_transition_row_records(path_or_rows: str | Path | list[dict[str, Any]]) -> dict[str, Any]:
    rows = _read_existing_jsonl(path_or_rows) if not isinstance(path_or_rows, list) else list(path_or_rows)
    return {
        "row_count": len(rows),
        "overall": _stats(rows),
        "failure_type_counts": dict(sorted(Counter(str(row.get("failure_type") or "unknown") for row in rows).items())),
        "by_benchmark": _breakdown(rows, "benchmark"),
        "by_source_quality": _breakdown(rows, "source_quality"),
        "by_candidate_source": _breakdown(rows, "candidate_source"),
        "top1_same_namespace_as_gold_fraction": (
            sum(1 for row in rows if bool(row.get("top1_same_namespace_as_gold"))) / len(rows)
            if rows
            else 0.0
        ),
    }


def compute_transition_row_diagnostics_batch(
    *,
    model: Any,
    batch: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    skill_ids_by_idx: dict[int, str],
    equivalent_skill_ids_by_skill_id: dict[str, list[str]] | None,
    device: torch.device,
    top_k: int = 10,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
) -> list[dict[str, Any]]:
    if not batch:
        return []
    h = _batch_cached_or_encode(
        model,
        batch,
        "_state_embedding",
        "state_text",
        device,
        text_role=STATE_QUERY_ROLE,
    )
    skill_count = max(1, len(skill_id_to_idx))
    initial_belief = getattr(model, "initial_belief", None)
    if not callable(initial_belief):
        raise ValueError("transition-row diagnostics require model.initial_belief")
    m_obs = initial_belief(h)
    m_obs, _replay_prefix_used_count = _apply_replay_prefix_beliefs(
        model,
        batch,
        m_obs,
        skill_id_to_idx,
        skill_count,
        device,
    )
    item_indices = [
        idx
        for idx, row in enumerate(batch)
        if (row.get("loss_mask") or {}).get("L_trans_skill_ce") and row.get("next_skill_id") in skill_id_to_idx
    ]
    if not item_indices:
        return []
    rows = [batch[idx] for idx in item_indices]
    indices = torch.tensor(item_indices, dtype=torch.long, device=device)
    obs_emb = _batch_cached_or_encode(
        model,
        rows,
        "_next_observation_embedding",
        "next_observation_text",
        device,
        text_role=TRANSITION_TEXT_ROLE,
    )
    action_emb = _batch_action_text_embedding_or_none(model, rows, device)
    current_labels = torch.tensor(
        [skill_id_to_idx.get(str(row.get("skill_id")), 0) for row in rows],
        dtype=torch.long,
        device=device,
    )
    next_labels_global = torch.tensor(
        [skill_id_to_idx[str(row.get("next_skill_id"))] for row in rows],
        dtype=torch.long,
        device=device,
    )
    candidate_rows = [
        _row_candidate_indices(row, "stage0_next_candidate_skill_indices", len(skill_id_to_idx))
        or _row_candidate_indices(row, "stage0_candidate_skill_indices", len(skill_id_to_idx))
        for row in rows
    ]
    use_candidates = bool(
        candidate_rows
        and all(candidate_rows)
        and all(int(label.detach().cpu().item()) in row for label, row in zip(next_labels_global, candidate_rows))
    )
    if use_candidates:
        max_width = max(len(row) for row in candidate_rows)
        candidate_ids = torch.zeros(len(candidate_rows), max_width, dtype=torch.long, device=device)
        candidate_valid_mask = torch.zeros(len(candidate_rows), max_width, dtype=torch.bool, device=device)
        local_labels: list[int] = []
        for row_idx, (row, label) in enumerate(zip(candidate_rows, next_labels_global.detach().cpu().tolist())):
            width = len(row)
            candidate_ids[row_idx, :width] = torch.tensor(row, dtype=torch.long, device=device)
            candidate_valid_mask[row_idx, :width] = True
            local_labels.append(row.index(int(label)))
        labels = torch.tensor(local_labels, dtype=torch.long, device=device)
        logits, transition_skill_head_type, _prior_logits, _residual_logits = _transition_candidate_logits_for_mode(
            model,
            h.index_select(0, indices),
            m_obs.index_select(0, indices),
            current_labels,
            obs_emb,
            action_emb,
            skill_count,
            candidate_ids=candidate_ids,
            candidate_valid_mask=candidate_valid_mask,
            candidate_stage0_prior_scores=_stage0_candidate_prior_scores_tensor(
                rows,
                candidate_rows,
                width=max_width,
                device=device,
                dtype=h.dtype,
            ),
            residual_lambda=float(transition_residual_lambda),
            scoring_mode=str(transition_scoring_mode or TRANSITION_SCORING_MODE),
        )
        return build_transition_row_records(
            logits=logits,
            labels=labels,
            rows=rows,
            candidate_indices=candidate_rows,
            skill_ids=skill_ids_by_idx,
            top_k=top_k,
            transition_skill_head_type=transition_skill_head_type,
            equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
        )
    raise RuntimeError(
        "row-level transition diagnostics currently require Stage0 top-M candidates containing the next skill"
    )


def _markdown_summary(report: dict[str, Any]) -> str:
    overall = (report.get("summary") or {}).get("overall") or {}
    failures = (report.get("summary") or {}).get("failure_type_counts") or {}
    by_benchmark = (report.get("summary") or {}).get("by_benchmark") or {}
    lines = [
        "# Stage2 Row-Level Transition Diagnostics",
        "",
        f"- status: `{report.get('status')}`",
        f"- row count: `{overall.get('row_count')}`",
        f"- recall@1: `{overall.get('recall@1')}`",
        f"- recall@5: `{overall.get('recall@5')}`",
        f"- recall@10: `{overall.get('recall@10')}`",
        f"- recall@20: `{overall.get('recall@20')}`",
        f"- Stage0 recall@5: `{overall.get('stage0_recall@5')}`",
        f"- Stage0 recall@10: `{overall.get('stage0_recall@10')}`",
        f"- Stage0 recall@20: `{overall.get('stage0_recall@20')}`",
        f"- MRR: `{overall.get('mrr')}`",
        f"- mean Stage0 gold rank: `{overall.get('mean_stage0_gold_rank')}`",
        f"- mean Stage2 gold rank: `{overall.get('mean_stage2_gold_rank')}`",
        f"- improved vs Stage0 fraction: `{overall.get('stage2_improved_vs_stage0_fraction')}`",
        f"- worse than Stage0 fraction: `{overall.get('stage2_worse_than_stage0_fraction')}`",
        "",
        "## Failure Types",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in failures.items())
    if by_benchmark:
        lines.extend(
            [
                "",
                "## By Benchmark",
                "",
                "| benchmark | rows | stage0 r@5 | stage0 r@20 | stage2 r@5 | stage2 r@20 | mean stage0 rank | mean stage2 rank | worse frac |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in sorted(by_benchmark.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(name),
                        str(metrics.get("row_count")),
                        str(metrics.get("stage0_recall@5")),
                        str(metrics.get("stage0_recall@20")),
                        str(metrics.get("recall@5")),
                        str(metrics.get("recall@20")),
                        str(metrics.get("mean_stage0_gold_rank")),
                        str(metrics.get("mean_stage2_gold_rank")),
                        str(metrics.get("stage2_worse_than_stage0_fraction")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def run_stage2_transition_row_diagnostics(
    *,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    top_m: int = 350,
    top_k: int = 10,
    batch_size: int = 8,
    max_rows: int | None = None,
    max_diagnostic_rows: int | None = None,
    allowed_benchmarks: set[str] | list[str] | tuple[str, ...] | None = None,
    benchmark_caps: dict[str, int] | None = None,
    embedding_cache_mode: str = "auto",
    embedding_cache_max_rows: int = 20000,
    skill_text_format: str | None = None,
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 100,
    transition_residual_lambda: float = DEFAULT_TRANSITION_RESIDUAL_LAMBDA,
    transition_scoring_mode: str = TRANSITION_SCORING_MODE,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage0_checkpoint_path = Path(stage0_checkpoint_path)
    stage2_checkpoint_path = Path(stage2_checkpoint_path)
    train_path = Path(train_path)
    skills_path = Path(skills_path)
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=stage0_checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        stage2_checkpoint_path,
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    rows = _read_jsonl(train_path)
    raw_row_count = len(rows)
    rows, train_split_filter_report = _filter_rows_by_train_split(rows)
    rows, benchmark_filter_report = _filter_rows_by_allowed_benchmarks(rows, allowed_benchmarks)
    rows, benchmark_caps_report = _cap_rows_by_benchmark(rows, benchmark_caps)
    rows, max_rows_filter_report = _limit_rows_for_smoke(rows, max_rows)
    for idx, row in enumerate(rows):
        row["_diagnostic_filtered_index"] = idx
    skills = _read_jsonl(skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    skill_ids_by_idx = {idx: str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)}
    equivalent_skill_ids_by_skill_id = _equivalent_skill_ids_by_skill_id(skills)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    skill_text_format_report = _apply_skill_text_format(model, model_config, skill_text_format)
    loss_weights = _normalize_loss_weights(DEFAULT_STAGE2_V2_LOSS_WEIGHTS)
    setup_status_path = output_dir / "setup_status.jsonl"
    if setup_status_path.exists():
        setup_status_path.unlink()
    rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        rows,
        skills,
        skill_id_to_idx,
        top_m=top_m,
        positive_missing_policy="skip",
        query_mode=stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=stage0_checkpoint_path,
        manifest_path=output_dir / "stage0_candidate_handoff_no_inject.json",
        encode_batch_size=stage0_candidate_encode_batch_size,
        device=device,
        setup_status_path=setup_status_path,
        progress_interval_batches=stage0_candidate_progress_interval_batches,
    )
    if max_diagnostic_rows is not None:
        rows = rows[: max(0, int(max_diagnostic_rows))]
    qwen_external_encoder = bool(
        getattr(model, "qwen_external_metadata", {})
        and getattr(model, "qwen_external_metadata", {}).get("qwen_external_encoder")
    )
    cache_policy = _embedding_cache_policy(
        mode=embedding_cache_mode,
        row_count=len(rows),
        qwen_external_encoder=qwen_external_encoder,
        max_rows=embedding_cache_max_rows,
    )
    embedding_cache_batch_size = 8 if qwen_external_encoder else 256
    if rows and cache_policy["cache_enabled"]:
        full_base_embedding_cache_report = _attach_full_base_embedding_cache(
            model,
            rows,
            encode_batch_size=embedding_cache_batch_size,
        )
        policy_embedding_cache_report = _attach_policy_embedding_cache(
            model,
            rows,
            encode_batch_size=embedding_cache_batch_size,
        )
    else:
        full_base_embedding_cache_report = _skipped_embedding_cache_report("full_base_replay", rows, cache_policy)
        policy_embedding_cache_report = _skipped_embedding_cache_report("policy_candidates", rows, cache_policy)
    action_adapter = UniversalActionAdapter(int(model_config.get("d", 128)), hidden_dim=int(model_config.get("d", 128))).to(device)
    action_adapter.eval()
    if hasattr(model, "eval"):
        model.eval()

    del action_adapter
    batch_size = max(1, int(batch_size))
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            records.extend(
                compute_transition_row_diagnostics_batch(
                    model=model,
                    batch=batch,
                    skill_id_to_idx=skill_id_to_idx,
                    skill_ids_by_idx=skill_ids_by_idx,
                    equivalent_skill_ids_by_skill_id=equivalent_skill_ids_by_skill_id,
                    device=device,
                    top_k=top_k,
                    transition_residual_lambda=transition_residual_lambda,
                    transition_scoring_mode=transition_scoring_mode,
                )
            )
    rows_path = output_dir / "transition_row_diagnostics.jsonl"
    _write_jsonl(rows_path, records)
    summary = summarize_transition_row_records(records)
    report = {
        "status": "ok" if records else "action_required",
        "blockers": [] if records else ["no_transition_rows_diagnosed"],
        "output_dir": str(output_dir),
        "row_diagnostics_path": str(rows_path),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "config": {
            "top_m": int(top_m),
            "top_k": int(top_k),
            "batch_size": int(batch_size),
            "max_rows": max_rows,
            "max_diagnostic_rows": max_diagnostic_rows,
            "embedding_cache_mode": str(embedding_cache_mode),
            "embedding_cache_max_rows": int(embedding_cache_max_rows),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
        },
        "data_filters": {
            "raw_row_count": raw_row_count,
            "train_split_filter": train_split_filter_report,
            "benchmark_filter": benchmark_filter_report,
            "benchmark_caps": benchmark_caps_report,
            "max_rows_filter": max_rows_filter_report,
        },
        "stage0_candidate_handoff": handoff_report,
        "model_load": {
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "skill_text_format": skill_text_format_report,
        },
        "embedding_cache": {
            "policy": cache_policy,
            "full_base": full_base_embedding_cache_report,
            "policy_candidates": policy_embedding_cache_report,
        },
        "summary": summary,
    }
    report_path = output_dir / "transition_row_diagnostics_report.json"
    markdown_path = output_dir / "transition_row_diagnostics_report.md"
    _write_json(report_path, report)
    markdown_path.write_text(_markdown_summary(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report
