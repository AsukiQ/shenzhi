from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.retrieval_metrics import evaluate_retrieval_run


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


def _write_progress(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    _write_json(path, payload)


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _skillrouter_query_text(row: dict[str, Any]) -> str:
    query = str(row.get("query_text") or row.get("query") or row.get("instruction") or "")
    return (
        "Instruct: Given a task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery:"
        f"{query[:1500]}"
    )


def _skillrouter_skill_text(skill: dict[str, Any]) -> str:
    name = str(skill.get("name") or "").strip()
    desc = str(skill.get("description") or "").strip()
    executor = str(skill.get("executor_desc") or "").strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    return " | ".join(part for part in (name, desc, executor, body) if part)


def _encode_skillrouter_texts(
    *,
    model_name_or_path: str | Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    from clstr.skillret_official import _encode_hf_texts

    return _encode_hf_texts(str(model_name_or_path), texts, batch_size, max_length)


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_tool_ids")
    if values is None:
        single = row.get("positive_skill_id") or row.get("skill_id") or row.get("tool_id") or row.get("doc_id")
        values = [single] if single else []
    if isinstance(values, str):
        values = [values]
    return [str(item) for item in values if item]


def _canonical_alias_map(skills: list[dict[str, Any]]) -> dict[str, str]:
    alias_to_canonical: dict[str, str] = {}
    for skill in skills:
        sid = str(skill.get("skill_id") or "")
        if not sid:
            continue
        canonical = str(skill.get("canonical_skill_id") or sid)
        alias_to_canonical[sid] = canonical
        alias_to_canonical[canonical] = canonical
        aliases = skill.get("alias_skill_ids") or []
        if isinstance(aliases, list):
            for alias in aliases:
                if alias:
                    alias_to_canonical[str(alias)] = canonical
    return alias_to_canonical


def _canonical_positive_ids(row: dict[str, Any], alias_to_canonical: dict[str, str], skill_ids: set[str]) -> list[str]:
    positives: list[str] = []
    seen: set[str] = set()
    for skill_id in _positive_skill_ids(row):
        canonical = alias_to_canonical.get(skill_id, skill_id)
        if canonical in skill_ids and canonical not in seen:
            positives.append(canonical)
            seen.add(canonical)
    return positives


def _write_canonical_qrels(
    *,
    path: str | Path,
    retrieval_rows: list[dict[str, Any]],
    alias_to_canonical: dict[str, str],
    skill_ids: set[str],
) -> tuple[int, int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in retrieval_rows:
        query_id = str(row.get("query_id") or row.get("qid") or row.get("task_id") or "")
        if not query_id:
            skipped += 1
            continue
        positives = _canonical_positive_ids(row, alias_to_canonical, skill_ids)
        if not positives:
            skipped += 1
            continue
        for skill_id in positives:
            rows.append({"query_id": query_id, "skill_id": skill_id, "relevance": 1})
    return _write_jsonl(path, rows), skipped


def _write_trec_run(
    *,
    path: str | Path,
    query_ids: list[str],
    skill_ids: list[str],
    scores: torch.Tensor,
    top_k: int,
    run_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k = min(int(top_k), len(skill_ids))
    values, indices = torch.topk(scores, k=k, dim=1)
    with path.open("w", encoding="utf-8") as handle:
        for row_idx, query_id in enumerate(query_ids):
            for rank, (skill_idx, score) in enumerate(
                zip(indices[row_idx].tolist(), values[row_idx].tolist()),
                start=1,
            ):
                handle.write(
                    f"{query_id}\tQ0\t{skill_ids[int(skill_idx)]}\t{rank}\t{float(score):.8f}\t{run_name}\n"
                )


def _write_trec_run_batched(
    *,
    path: str | Path,
    query_ids: list[str],
    skill_ids: list[str],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    top_k: int,
    run_name: str,
    query_batch_size: int,
    progress_path: str | Path | None = None,
    progress_base: dict[str, Any] | None = None,
) -> dict[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k = min(int(top_k), len(skill_ids))
    batch_size = max(1, int(query_batch_size))
    progress_base = dict(progress_base or {})
    processed = 0
    batches = 0
    skill_embs_t = skill_embs.t().contiguous()

    def write_progress(phase: str) -> None:
        _write_progress(
            progress_path,
            {
                **progress_base,
                "phase": phase,
                "processed_queries": processed,
                "processed_batches": batches,
                "total_queries": len(query_ids),
                "skill_count": len(skill_ids),
                "top_k": k,
                "score_batch_size": batch_size,
            },
        )

    write_progress("writing_run")
    with path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for start in range(0, len(query_ids), batch_size):
            end = min(start + batch_size, len(query_ids))
            scores = query_embs[start:end] @ skill_embs_t
            values, indices = torch.topk(scores, k=k, dim=1)
            for row_idx, query_id in enumerate(query_ids[start:end]):
                for rank, (skill_idx, score) in enumerate(
                    zip(indices[row_idx].tolist(), values[row_idx].tolist()),
                    start=1,
                ):
                    handle.write(
                        f"{query_id}\tQ0\t{skill_ids[int(skill_idx)]}\t{rank}\t{float(score):.8f}\t{run_name}\n"
                    )
                processed += 1
            batches += 1
            handle.flush()
            write_progress("writing_run")
    write_progress("done")
    return {"processed_queries": processed, "processed_batches": batches}


def run_stage0_skillrouter_frozen_baseline(
    *,
    data_root: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final",
    output_dir: str | Path = "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak",
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    top_k: int = 100,
    batch_size: int = 16,
    score_batch_size: int = 1024,
    max_length: int = 2048,
    max_queries: int | None = None,
    max_skills: int | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    progress_base = {
        "status": "running",
        "method": "stage0_skillrouter_frozen_baseline",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
    }
    _write_progress(progress_path, {**progress_base, "phase": "loading_data"})
    skills = _read_jsonl(data_root / "skill_pool.jsonl")
    retrieval_rows = _read_jsonl(data_root / "retrieval.jsonl")
    if max_skills is not None:
        skills = skills[: int(max_skills)]
    if max_queries is not None:
        retrieval_rows = retrieval_rows[: int(max_queries)]

    skill_ids = [str(row["skill_id"]) for row in skills]
    skill_id_set = set(skill_ids)
    alias_to_canonical = _canonical_alias_map(skills)
    query_rows = [
        row
        for row in retrieval_rows
        if str(row.get("query_id") or row.get("qid") or row.get("task_id") or "")
        and _canonical_positive_ids(row, alias_to_canonical, skill_id_set)
    ]
    query_ids = [str(row.get("query_id") or row.get("qid") or row.get("task_id")) for row in query_rows]

    qrels_path = output_dir / "qrels.jsonl"
    _write_progress(
        progress_path,
        {
            **progress_base,
            "phase": "writing_qrels",
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
        },
    )
    qrel_count, skipped_qrel_rows = _write_canonical_qrels(
        path=qrels_path,
        retrieval_rows=query_rows,
        alias_to_canonical=alias_to_canonical,
        skill_ids=skill_id_set,
    )

    if not skill_ids or not query_rows or qrel_count <= 0:
        report = {
            "status": "action_required",
            "method": "stage0_skillrouter_frozen_baseline",
            "blockers": ["missing_skill_pool_or_qrels"],
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "skill_pool_path": str(data_root / "skill_pool.jsonl"),
            "retrieval_path": str(data_root / "retrieval.jsonl"),
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
            "qrel_count": qrel_count,
            "skipped_qrel_rows": skipped_qrel_rows,
        }
        _write_json(output_dir / "manifest.json", report)
        _write_progress(progress_path, {**progress_base, "status": "action_required", "phase": "done", **report})
        return report

    _write_progress(
        progress_path,
        {
            **progress_base,
            "phase": "encoding_queries",
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
            "qrel_count": qrel_count,
            "batch_size": int(batch_size),
            "score_batch_size": int(score_batch_size),
        },
    )
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=model_name_or_path,
        texts=[_skillrouter_query_text(row) for row in query_rows],
        batch_size=batch_size,
        max_length=max_length,
    )
    _write_progress(
        progress_path,
        {
            **progress_base,
            "phase": "encoding_skills",
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
            "qrel_count": qrel_count,
            "batch_size": int(batch_size),
            "score_batch_size": int(score_batch_size),
        },
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=model_name_or_path,
        texts=[_skillrouter_skill_text(row) for row in skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    query_embs = F.normalize(query_embs.float(), p=2, dim=-1)
    skill_embs = F.normalize(skill_embs.float(), p=2, dim=-1)

    run_path = output_dir / "run.tsv"
    run_write_report = _write_trec_run_batched(
        path=run_path,
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        top_k=top_k,
        run_name="stage0_skillrouter_frozen_baseline",
        query_batch_size=score_batch_size,
        progress_path=progress_path,
        progress_base={
            **progress_base,
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
            "qrel_count": qrel_count,
            "batch_size": int(batch_size),
        },
    )
    _write_progress(
        progress_path,
        {
            **progress_base,
            "phase": "evaluating_metrics",
            "skill_count": len(skill_ids),
            "query_count": len(query_rows),
            "qrel_count": qrel_count,
            "batch_size": int(batch_size),
            "score_batch_size": int(score_batch_size),
            **run_write_report,
        },
    )
    metrics = evaluate_retrieval_run(
        qrels_path=qrels_path,
        run_path=run_path,
        output_dir=output_dir,
        run_format="trec",
        k_values=(20, 50, 100),
        benchmark="clstr_unified_stage0",
        method="stage0_skillrouter_frozen_baseline",
    )
    manifest = {
        "status": "ok",
        "method": "stage0_skillrouter_frozen_baseline",
        "baseline_family": "SkillRouter-compatible frozen bi-encoder over CLSTR canonical skill pool",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "model_name_or_path": str(model_name_or_path),
        "skill_pool_path": str(data_root / "skill_pool.jsonl"),
        "retrieval_path": str(data_root / "retrieval.jsonl"),
        "qrels_path": str(qrels_path),
        "run_path": str(run_path),
        "metrics_path": str(output_dir / "metrics.json"),
        "run_format": "trec",
        "qrels_id_space": "canonical_skill_id",
        "uses_canonical_skill_pool": True,
        "skill_count": len(skill_ids),
        "query_count": len(query_rows),
        "qrel_count": qrel_count,
        "skipped_qrel_rows": skipped_qrel_rows,
        "top_k": int(top_k),
        "batch_size": int(batch_size),
        "score_batch_size": int(score_batch_size),
        "max_length": int(max_length),
        "scoring_mode": "batched_query_topk",
        "run_write_report": run_write_report,
        "progress_path": str(progress_path),
        "skill_ids": skill_ids,
        "canonical_skill_ids": skill_ids,
        "metrics": metrics.get("metrics", {}),
        "note": "This is a same-pool frozen bi-encoder baseline; it does not use SkillRouter's original skill pool or cross-encoder reranker.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_progress(progress_path, {**progress_base, "status": "ok", "phase": "done", **run_write_report})
    return manifest
