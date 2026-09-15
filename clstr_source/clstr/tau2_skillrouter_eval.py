from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
from clstr.tau2_route_eval import (
    _clean_benchmark_name,
    _mean,
    _mrr,
    _rank_position,
    _recall_at,
    _write_json,
    _write_jsonl,
    load_tau2_route_corpus,
)


def _load_skillrouter_adapter_checkpoint(adapter_checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(adapter_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"SkillRouter adapter checkpoint must be a dict: {checkpoint_path}")
    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"SkillRouter adapter checkpoint missing adapter_state_dict: {checkpoint_path}")
    q_weight = state.get("q_proj.weight")
    if not isinstance(q_weight, torch.Tensor):
        raise ValueError(f"SkillRouter adapter checkpoint missing q_proj.weight: {checkpoint_path}")
    d_weight = state.get("d_proj.weight")
    if d_weight is not None and not isinstance(d_weight, torch.Tensor):
        raise ValueError(f"SkillRouter adapter checkpoint has invalid d_proj.weight: {checkpoint_path}")
    adapter_state = {"q_proj.weight": q_weight.float()}
    if isinstance(d_weight, torch.Tensor):
        adapter_state["d_proj.weight"] = d_weight.float()
    return adapter_state, {
        "checkpoint_path": str(checkpoint_path),
        "method": str(payload.get("method") or ""),
        "step": payload.get("step"),
        "uses_clstr_heads": bool(payload.get("uses_clstr_heads", False)),
        "uses_clstr_transition": bool(payload.get("uses_clstr_transition", False)),
        "config": payload.get("config", {}),
    }


def _apply_skillrouter_adapter(
    *,
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    adapter_checkpoint_path: str | Path | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if adapter_checkpoint_path is None:
        return (
            F.normalize(query_embs.float(), p=2, dim=-1),
            F.normalize(skill_embs.float(), p=2, dim=-1),
            None,
        )
    adapter_state, adapter_report = _load_skillrouter_adapter_checkpoint(adapter_checkpoint_path)
    q_weight = adapter_state["q_proj.weight"]
    d_weight = adapter_state.get("d_proj.weight")
    if query_embs.size(-1) != q_weight.size(-1):
        raise ValueError(
            "query embedding dimension does not match adapter q_proj input: "
            f"{query_embs.size(-1)} != {q_weight.size(-1)}"
        )
    projected_queries = query_embs.float() @ q_weight.t()
    if d_weight is None:
        projected_skills = skill_embs.float()
    else:
        if skill_embs.size(-1) != d_weight.size(-1):
            raise ValueError(
                "skill embedding dimension does not match adapter d_proj input: "
                f"{skill_embs.size(-1)} != {d_weight.size(-1)}"
            )
        projected_skills = skill_embs.float() @ d_weight.t()
    adapter_report["q_proj_shape"] = list(q_weight.shape)
    adapter_report["d_proj_shape"] = None if d_weight is None else list(d_weight.shape)
    return (
        F.normalize(projected_queries.float(), p=2, dim=-1),
        F.normalize(projected_skills.float(), p=2, dim=-1),
        adapter_report,
    )


def _tau2_skillrouter_query_text(row: dict[str, Any]) -> str:
    state = str(
        row.get("state_text_full")
        or row.get("state_text")
        or row.get("query_text")
        or ""
    )
    return (
        "Instruct: Given a task description and interaction history, retrieve the most relevant "
        "skill document that would help an agent choose the next tool action\nQuery:"
        f"{state[:1500]}"
    )


def _metrics_from_ranks(ranks: list[int | None], candidate_counts: list[int]) -> dict[str, float]:
    valid = [rank for rank in ranks if rank is not None]
    denom = len(ranks)
    if denom <= 0:
        return {
            "next_tool_recall@1": 0.0,
            "next_tool_recall@5": 0.0,
            "next_tool_mrr": 0.0,
            "candidate_count": _mean(candidate_counts),
        }
    return {
        "next_tool_recall@1": sum(1 for rank in valid if rank <= 1) / denom,
        "next_tool_recall@5": sum(1 for rank in valid if rank <= 5) / denom,
        "next_tool_mrr": sum(1.0 / float(rank) for rank in valid) / denom,
        "candidate_count": _mean(candidate_counts),
    }


def _rank_candidates(
    *,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    candidate_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skill_ids = [str(skill.get("skill_id") or "") for skill in skills]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids) if skill_id}
    scores = query_embs.float() @ skill_embs.float().t()
    ranked_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    missing_positive = 0
    missing_candidate_skill = 0
    for row_idx, row in enumerate(rows):
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or []]
        scored: list[tuple[str, float]] = []
        for candidate_id in candidates:
            skill_idx = skill_id_to_idx.get(candidate_id)
            if skill_idx is None:
                missing_candidate_skill += 1
                continue
            scored.append((candidate_id, float(scores[row_idx, skill_idx].item())))
        scored.sort(key=lambda item: (-item[1], item[0]))
        if candidate_count is not None:
            scored = scored[: max(1, int(candidate_count))]
        ranked_ids = [item[0] for item in scored]
        positive_id = str(row.get("next_skill_id") or "")
        rank = _rank_position(ranked_ids, positive_id)
        if rank is None:
            missing_positive += 1
        ranks.append(rank)
        candidate_counts.append(len(ranked_ids))
        copied = dict(row)
        copied["candidate_next_skill_ids"] = ranked_ids
        copied["candidate_next_skillrouter_scores"] = [float(item[1]) for item in scored]
        copied["positive_next_skill_rank"] = rank
        ranked_rows.append(copied)
    return ranked_rows, {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": len(ranked_rows) - missing_positive,
        "positive_missing_rows": int(missing_positive),
        "candidate_skill_missing_count": int(missing_candidate_skill),
        "candidate_count_mean": _mean(candidate_counts),
        "metrics": _metrics_from_ranks(ranks, candidate_counts),
        "positive_skillrouter_domain_recall@1": _recall_at([r for r in ranks if r is not None], 1),
        "positive_skillrouter_domain_recall@5": _recall_at([r for r in ranks if r is not None], 5),
        "positive_skillrouter_domain_mrr": _mrr([r for r in ranks if r is not None]),
    }


def _metrics_by_domain(ranked_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in ranked_rows:
        grouped.setdefault(str(row.get("domain") or "unknown"), []).append(row)
    result: dict[str, dict[str, float]] = {}
    for domain, rows in sorted(grouped.items()):
        ranks = [row.get("positive_next_skill_rank") for row in rows]
        candidate_counts = [len(row.get("candidate_next_skill_ids") or []) for row in rows]
        result[domain] = {
            **_metrics_from_ranks([rank if isinstance(rank, int) else None for rank in ranks], candidate_counts),
            "row_count": float(len(rows)),
        }
    return result


def run_tau2_skillrouter_frozen_eval(
    *,
    benchmark_name: str = "tau2",
    data_root: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    domains: Iterable[str] | None = None,
    max_tasks_per_domain: int | None = None,
    max_eval_rows: int | None = None,
    candidate_count: int | None = None,
    batch_size: int = 16,
    max_length: int = 2048,
    adapter_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_tau2_route_corpus(
        data_root,
        domains=domains,
        max_tasks_per_domain=max_tasks_per_domain,
        benchmark_name=benchmark_name,
    )
    source_rows = corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    skills_path = output_dir / f"{benchmark_name}_skill_pool.jsonl"
    source_rows_path = output_dir / f"{benchmark_name}_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)

    query_texts = [_tau2_skillrouter_query_text(row) for row in source_rows]
    skill_texts = [_skillrouter_skill_text(skill) for skill in corpus.skills]
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=query_texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=skill_texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    query_embs, skill_embs, adapter_report = _apply_skillrouter_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    ranked_rows, ranking_report = _rank_candidates(
        rows=source_rows,
        skills=corpus.skills,
        query_embs=query_embs,
        skill_embs=skill_embs,
        candidate_count=candidate_count,
    )
    ranked_path = output_dir / "tau2_skillrouter_ranked_rows.jsonl"
    _write_jsonl(ranked_path, ranked_rows)
    blockers: list[str] = []
    if not source_rows:
        blockers.append("no_source_eval_rows")
    if int(ranking_report.get("candidate_skill_missing_count") or 0) > 0:
        blockers.append("candidate_skill_missing")
    method = "skillrouter_finetuned_biencoder_adapter" if adapter_checkpoint_path is not None else "skillrouter_frozen_biencoder"
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": benchmark_name,
        "method": method,
        "baseline_family": (
            "SkillRouter-compatible finetuned bi-encoder adapter over tau2 domain-local action skills"
            if adapter_checkpoint_path is not None
            else "SkillRouter-compatible frozen bi-encoder over tau2 domain-local action skills"
        ),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "ranked_rows_path": str(ranked_path),
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(ranked_rows),
        "corpus_report": corpus.report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "metrics_by_domain": _metrics_by_domain(ranked_rows),
        "config": {
            "benchmark_name": benchmark_name,
            "domains": None if domains is None else list(domains),
            "max_tasks_per_domain": max_tasks_per_domain,
            "max_eval_rows": max_eval_rows,
            "candidate_count": candidate_count,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
            "query_format": "skillrouter_tau2_state_history",
            "skill_format": "skillrouter_same_pool_text",
        },
        "paper_scope_note": (
            f"This is a SkillRouter-style bi-encoder baseline on the same {benchmark_name} domain-local action skill pool. "
            "It does not use CLSTR Stage2 transition heads or Stage4 online memory."
        ),
    }
    report_name = (
        f"{benchmark_name}_skillrouter_finetuned_eval_report.json"
        if adapter_checkpoint_path is not None
        else f"{benchmark_name}_skillrouter_frozen_eval_report.json"
    )
    _write_json(output_dir / report_name, report)
    if benchmark_name == "tau2":
        legacy_report_name = "tau2_skillrouter_finetuned_eval_report.json" if adapter_checkpoint_path is not None else "tau2_skillrouter_frozen_eval_report.json"
        _write_json(output_dir / legacy_report_name, report)
    return report
