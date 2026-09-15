from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.global_pool_route_eval import GlobalPoolCorpus, merge_global_skill_pool


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_toolbench_g3_global_pool_corpus(
    *,
    eval_trajectories_path: str | Path,
    skills_path: str | Path = "data/toolbench_g3/skills.jsonl",
    max_eval_rows: int | None = None,
) -> GlobalPoolCorpus:
    source_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in _read_jsonl(eval_trajectories_path):
        next_skill_id = str(row.get("next_skill_id") or "").strip()
        state_text = str(row.get("state_text") or "").strip()
        if not next_skill_id or not state_text:
            skipped += 1
            continue
        copied = dict(row)
        copied["benchmark"] = "toolbench_g3"
        copied["source_benchmark"] = "toolbench_g3"
        copied["next_skill_id"] = next_skill_id
        source_rows.append(copied)
        if max_eval_rows is not None and len(source_rows) >= max(0, int(max_eval_rows)):
            break
    skills = _read_jsonl(skills_path)
    return GlobalPoolCorpus(
        benchmark="toolbench_g3",
        skills=skills,
        source_rows=source_rows,
        report={
            "benchmark": "toolbench_g3",
            "eval_trajectories_path": str(eval_trajectories_path),
            "skills_path": str(skills_path),
            "source_row_count": len(source_rows),
            "skill_count": len(skills),
            "skipped_rows": int(skipped),
            "max_eval_rows": max_eval_rows,
        },
    )


def _query_text(row: dict[str, Any]) -> str:
    state = str(row.get("state_text") or row.get("query_text") or row.get("query") or "")
    return (
        "Instruct: Given a task description and interaction history, retrieve the most relevant "
        "skill document that would help an agent choose the next tool action\nQuery:"
        f"{state[:1500]}"
    )


def _mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _metrics_from_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, float]:
    denom = len(ranks)
    if denom <= 0:
        return {"mrr": 0.0, **{f"recall@{k}": 0.0 for k in top_ks}}
    valid = [rank for rank in ranks if rank is not None]
    metrics = {
        f"recall@{k}": sum(1 for rank in valid if rank <= int(k)) / denom
        for k in top_ks
    }
    metrics["mrr"] = sum(1.0 / float(rank) for rank in valid) / denom
    return metrics


def _rank_full_pool(
    *,
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    score_batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    ranks: list[int | None] = []
    missing_positive = 0
    ranked_rows: list[dict[str, Any]] = []
    skill_embs_device = skill_embs.to(device=device, dtype=torch.float32)
    skill_embs_t = skill_embs_device.t().contiguous()
    score_batch_size = max(1, int(score_batch_size))
    with torch.no_grad():
        for start in range(0, len(rows), score_batch_size):
            batch = rows[start : start + score_batch_size]
            q = query_embs[start : start + len(batch)].to(device=device, dtype=torch.float32)
            scores = q @ skill_embs_t
            positive_indices: list[int] = []
            positive_present: list[bool] = []
            for row in batch:
                idx = skill_id_to_idx.get(str(row.get("next_skill_id") or ""))
                positive_present.append(idx is not None)
                positive_indices.append(0 if idx is None else int(idx))
            idx_tensor = torch.tensor(positive_indices, dtype=torch.long, device=device)
            pos_scores = scores.gather(1, idx_tensor.view(-1, 1)).squeeze(1)
            rank_tensor = (scores > pos_scores.view(-1, 1)).sum(dim=1) + 1
            for offset, row in enumerate(batch):
                copied = dict(row)
                if not positive_present[offset]:
                    rank = None
                    missing_positive += 1
                else:
                    rank = int(rank_tensor[offset].detach().cpu().item())
                copied["positive_full_pool_rank"] = rank
                ranked_rows.append(copied)
                ranks.append(rank)
    metrics = _metrics_from_ranks(ranks, top_ks=(1, 5, 20, 50, 100, 500))
    return ranked_rows, {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": len(ranked_rows) - missing_positive,
        "positive_missing_rows": int(missing_positive),
        "skill_count": int(skill_embs.size(0)),
        "score_batch_size": int(score_batch_size),
        "rank_mean": _mean([rank for rank in ranks if rank is not None]),
        "metrics": metrics,
    }


def run_global_pool_skillrouter_eval(
    *,
    corpus: GlobalPoolCorpus,
    base_skills_path: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    adapter_checkpoint_path: str | Path | None = None,
    max_eval_rows: int | None = None,
    encode_batch_size: int = 16,
    score_batch_size: int = 64,
    max_length: int = 2048,
) -> dict[str, Any]:
    import torch

    from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
    from clstr.tau2_skillrouter_eval import _apply_skillrouter_adapter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_skills = _read_jsonl(base_skills_path)
    merged_skills, pool_report = merge_global_skill_pool(
        base_skills,
        corpus.skills,
        dynamic_source_label=corpus.benchmark,
    )
    source_rows = (
        corpus.source_rows[: max(0, int(max_eval_rows))]
        if max_eval_rows is not None
        else list(corpus.source_rows)
    )
    skills_path = output_dir / "global_skill_pool.jsonl"
    source_rows_path = output_dir / f"{corpus.benchmark}_source_rows.jsonl"
    _write_jsonl(skills_path, merged_skills)
    _write_jsonl(source_rows_path, source_rows)
    skill_id_to_idx = {
        str(skill.get("skill_id") or ""): idx
        for idx, skill in enumerate(merged_skills)
        if str(skill.get("skill_id") or "")
    }
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_query_text(row) for row in source_rows],
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_skill_text(skill) for skill in merged_skills],
        batch_size=encode_batch_size,
        max_length=max_length,
    )
    query_embs, skill_embs, adapter_report = _apply_skillrouter_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranked_rows, ranking_report = _rank_full_pool(
        query_embs=query_embs,
        skill_embs=skill_embs,
        rows=source_rows,
        skill_id_to_idx=skill_id_to_idx,
        score_batch_size=score_batch_size,
        device=device,
    )
    _write_jsonl(output_dir / f"{corpus.benchmark}_global_pool_skillrouter_ranked_rows.jsonl", ranked_rows)
    method = (
        "skillrouter_finetuned_global_pool_biencoder_adapter"
        if adapter_checkpoint_path
        else "skillrouter_frozen_global_pool_biencoder"
    )
    blockers: list[str] = []
    if not source_rows:
        blockers.append("no_source_eval_rows")
    if int(ranking_report.get("positive_missing_rows") or 0) > 0:
        blockers.append("positive_missing_from_global_pool")
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": corpus.benchmark,
        "method": method,
        "output_dir": str(output_dir),
        "base_skills_path": str(base_skills_path),
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(ranked_rows),
        "corpus_report": corpus.report,
        "global_pool_report": pool_report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "config": {
            "max_eval_rows": max_eval_rows,
            "encode_batch_size": int(encode_batch_size),
            "score_batch_size": int(score_batch_size),
            "max_length": int(max_length),
        },
        "paper_scope_note": (
            "SkillRouter-style full-pool bi-encoder ranking over the same merged global skill pool used by CLSTR. "
            "For dynamically appended skills, this measures unseen-skill transfer."
        ),
    }
    report_name = f"{corpus.benchmark}_global_pool_skillrouter_eval_report.json"
    _write_json(output_dir / report_name, report)
    return report
