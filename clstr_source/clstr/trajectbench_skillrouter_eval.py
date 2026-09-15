from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.logged_online_stage4_train import (
    build_logged_online_stage4_rows_with_stage0_handoff,
    split_logged_online_stage4_rows,
)
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint
from clstr.tau2_skillrouter_eval import _apply_skillrouter_adapter
from clstr.skillret_official import _skillrouter_query_text, _skillrouter_skill_text


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "").strip()


def _row_query_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("state_text") or row.get("query") or "").strip(),
        str(row.get("history_text") or "").strip(),
        str(row.get("action_text") or "").strip(),
    ]
    return _skillrouter_query_text({"query": "\n".join(part for part in parts if part)})


def _rank_position(ranked_ids: list[str], positive_id: str) -> int | None:
    try:
        return ranked_ids.index(positive_id) + 1
    except ValueError:
        return None


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _metrics_from_ranks(ranks: list[int | None], candidate_counts: list[int]) -> dict[str, float]:
    denom = len(ranks)
    if denom <= 0:
        return {
            "next_skill_recall@1": 0.0,
            "next_skill_recall@5": 0.0,
            "next_skill_mrr": 0.0,
            "candidate_count": _mean(candidate_counts),
        }
    valid = [rank for rank in ranks if rank is not None]
    return {
        "next_skill_recall@1": sum(1 for rank in valid if rank <= 1) / denom,
        "next_skill_recall@5": sum(1 for rank in valid if rank <= 5) / denom,
        "next_skill_mrr": sum(1.0 / float(rank) for rank in valid) / denom,
        "candidate_count": _mean(candidate_counts),
    }


def _selected_candidate_skills(rows: list[dict[str, Any]], skills: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_ids = {
        str(candidate_id)
        for row in rows
        for candidate_id in (row.get("candidate_next_skill_ids") or [])
        if str(candidate_id).strip()
    }
    selected = [skill for skill in skills if _skill_id(skill) in candidate_ids]
    selected_ids = {_skill_id(skill) for skill in selected}
    return selected, {
        "candidate_skill_id_count": len(candidate_ids),
        "selected_skill_count": len(selected),
        "missing_candidate_skill_count": len(candidate_ids - selected_ids),
    }


def _rank_candidate_rows(
    *,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    candidate_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skill_ids = [_skill_id(skill) for skill in skills]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids) if skill_id}
    scores = query_embs.float() @ skill_embs.float().t()
    ranked_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    missing_positive = 0
    missing_candidate_skill = 0
    for row_idx, row in enumerate(rows):
        scored: list[tuple[str, float]] = []
        for candidate_id in [str(item) for item in row.get("candidate_next_skill_ids") or []]:
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
        "metrics": _metrics_from_ranks(ranks, candidate_counts),
    }


def run_trajectbench_skillrouter_eval_on_rows(
    *,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    adapter_checkpoint_path: str | Path | None = None,
    batch_size: int = 16,
    max_length: int = 2048,
    candidate_count: int | None = None,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_skills, skill_selection_report = _selected_candidate_skills(rows, skills)
    _write_jsonl(output_dir / "trajectbench_eval_rows.jsonl", rows)
    _write_jsonl(output_dir / "trajectbench_candidate_skill_pool.jsonl", selected_skills)
    query_texts = [_row_query_text(row) for row in rows]
    skill_texts = [_skillrouter_skill_text(skill) for skill in selected_skills]
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
        query_embs=F.normalize(query_embs.float(), p=2, dim=-1),
        skill_embs=F.normalize(skill_embs.float(), p=2, dim=-1),
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    ranked_rows, ranking_report = _rank_candidate_rows(
        rows=rows,
        skills=selected_skills,
        query_embs=query_embs,
        skill_embs=skill_embs,
        candidate_count=candidate_count,
    )
    ranked_path = output_dir / "trajectbench_skillrouter_ranked_rows.jsonl"
    _write_jsonl(ranked_path, ranked_rows)
    blockers: list[str] = []
    if not rows:
        blockers.append("no_eval_rows")
    if not selected_skills:
        blockers.append("no_candidate_skills")
    if ranking_report["candidate_skill_missing_count"]:
        blockers.append("candidate_skill_missing")
    method = "skillrouter_finetuned_biencoder_adapter" if adapter_checkpoint_path is not None else "skillrouter_frozen_biencoder"
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "trajectbench",
        "method": method,
        "output_dir": str(output_dir),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "source_eval_rows": len(rows),
        "retained_eval_rows": len(ranked_rows),
        "skill_selection_report": skill_selection_report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "ranked_rows_path": str(ranked_path),
        "config": {
            "candidate_count": candidate_count,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "query_format": "skillrouter_state_history",
            "skill_format": "skillret_official_full_text",
        },
        "paper_scope_note": (
            "SkillRouter-style bi-encoder adapter reranks the same TrajectBench Stage0 top-M candidate rows "
            "used by the complete CLSTR route. It does not use CLSTR Stage2/Stage4 heads."
        ),
    }
    if extra_report:
        report.update(extra_report)
    _write_json(output_dir / "trajectbench_skillrouter_finetuned_eval_report.json", report)
    return report


def run_trajectbench_skillrouter_finetuned_eval(
    *,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    routing_checkpoint_path: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    adapter_checkpoint_path: str | Path | None = None,
    prebuilt_stage4_rows_path: str | Path | None = None,
    max_rows: int | None = None,
    eval_rows: int = 256,
    eval_split_mode: str = "trajectory_prefix",
    trajectory_eval_steps: int = 1,
    candidate_count: int | None = None,
    stage0_top_m: int = 350,
    stage0_positive_missing_policy: str = "skip",
    stage0_handoff_query_mode: str = "skillrouter_state",
    stage0_candidate_encode_batch_size: int = 16,
    stage0_candidate_progress_interval_batches: int = 100,
    batch_size: int = 16,
    max_length: int = 2048,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skills = _read_jsonl(skills_path)
    if prebuilt_stage4_rows_path is not None:
        rows = _read_jsonl(prebuilt_stage4_rows_path)
        data_report = {
            "candidate_source": "prebuilt_stage4_rows",
            "prebuilt_stage4_rows_path": str(prebuilt_stage4_rows_path),
            "stage4_rows": len(rows),
            "max_rows": max_rows,
            "stage0_top_m": stage0_top_m,
        }
        routing_report: dict[str, Any] = {
            "candidate_source": "prebuilt_stage4_rows",
            "routing_checkpoint_path": str(routing_checkpoint_path),
            "skipped_stage0_handoff": True,
        }
        model_config: dict[str, Any] = {}
    else:
        model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
            checkpoint_path=routing_checkpoint_path,
            skills_path=skills_path,
            model_cache_dir=output_dir / "model_cache",
        )
        rows, data_report = build_logged_online_stage4_rows_with_stage0_handoff(
            model=model,
            trajectories_path=trajectories_path,
            skills_path=skills_path,
            output_dir=output_dir,
            stage0_top_m=stage0_top_m,
            stage0_positive_missing_policy=stage0_positive_missing_policy,
            stage0_handoff_query_mode=stage0_handoff_query_mode,
            stage0_candidate_encode_batch_size=stage0_candidate_encode_batch_size,
            stage0_candidate_progress_interval_batches=stage0_candidate_progress_interval_batches,
            candidate_count=candidate_count,
            max_rows=max_rows,
            allowed_benchmarks={"traject_bench"},
            benchmark_caps=None,
            routing_checkpoint_path=routing_checkpoint_path,
        )
    stream_rows, eval_set, split_report = split_logged_online_stage4_rows(
        rows,
        eval_rows=eval_rows,
        eval_split_mode=eval_split_mode,
        trajectory_eval_steps=trajectory_eval_steps,
        eval_benchmark_caps=None,
    )
    _write_jsonl(output_dir / "trajectbench_stage4_rows.jsonl", rows)
    report = run_trajectbench_skillrouter_eval_on_rows(
        rows=eval_set,
        skills=skills,
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        adapter_checkpoint_path=adapter_checkpoint_path,
        batch_size=batch_size,
        max_length=max_length,
        candidate_count=candidate_count,
        extra_report={
            "trajectories_path": str(trajectories_path),
            "skills_path": str(skills_path),
            "routing_checkpoint_path": str(routing_checkpoint_path),
            "prebuilt_stage4_rows_path": None if prebuilt_stage4_rows_path is None else str(prebuilt_stage4_rows_path),
            "data_report": data_report,
            "split_report": split_report,
            "stream_row_count": len(stream_rows),
            "eval_row_count": len(eval_set),
            "routing_init": routing_report,
            "model_config": model_config,
        },
    )
    return report
