from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.baseline_corpus_types import (
    UnifiedSkillRouterCorpus,
    UnifiedSkillRouterQuery,
)
from clstr.skillret_official import _skillrouter_query_text, _skillrouter_skill_text
from clstr.skillrouter_style import (
    SkillRouterStyleAdapter,
    _encode_raw_texts,
    _load_backbone,
    _query_text_for_mode,
    _resolve_hf_pooling,
    _resolve_query_text_mode,
    _resolve_tokenizer_padding_side,
)


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


def _iter_jsonl(path: str | Path, max_rows: int | None = None):
    yielded = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_rows is not None and yielded >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yielded += 1
                yield row


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_training_progress(
    output_dir: str | Path,
    *,
    stage: str,
    step: int | None = None,
    max_steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "running",
        "stage": str(stage),
        "updated_at_unix": time.time(),
    }
    if step is not None:
        payload["step"] = int(step)
    if max_steps is not None:
        payload["max_steps"] = int(max_steps)
        if step is not None and int(max_steps) > 0:
            payload["progress_fraction"] = min(1.0, max(0.0, int(step) / int(max_steps)))
    if extra:
        payload.update(extra)
    _write_json(Path(output_dir) / "progress.json", payload)
    return payload


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("id") or "").strip()


def _query_id(row: dict[str, Any], fallback: int) -> str:
    task_id = str(row.get("task_id") or row.get("trajectory_id") or fallback).strip()
    step = row.get("step_index")
    return f"{task_id}::{step}" if step is not None else task_id


def _routing_enabled(row: dict[str, Any]) -> bool:
    mask = row.get("loss_mask")
    if not isinstance(mask, dict):
        return True
    return bool(mask.get("routing") or mask.get("L_trans_skill_ce") or mask.get("L_policy"))


def _alias_sets(skills: list[dict[str, Any]]) -> dict[str, set[str]]:
    known = {_skill_id(skill) for skill in skills if _skill_id(skill)}
    groups: dict[str, set[str]] = {}
    for skill in skills:
        sid = _skill_id(skill)
        if not sid:
            continue
        values = {sid}
        canonical = str(skill.get("canonical_skill_id") or "").strip()
        if canonical in known:
            values.add(canonical)
        aliases = skill.get("alias_skill_ids")
        if isinstance(aliases, list):
            values.update(str(item).strip() for item in aliases if str(item).strip() in known)
        for value in values:
            groups.setdefault(value, set()).update(values)
    return groups


def _select_skills(
    skills: list[dict[str, Any]],
    required_skill_ids: set[str],
    max_skills: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_skills is None:
        return list(skills), {
            "max_skills": None,
            "required_skill_count": len(required_skill_ids),
            "required_skill_truncated": False,
        }
    max_skills = max(1, int(max_skills))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for skill in skills:
        sid = _skill_id(skill)
        if sid in required_skill_ids and sid not in selected_ids:
            selected.append(skill)
            selected_ids.add(sid)
            if len(selected) >= max_skills:
                break
    required_truncated = len(required_skill_ids - selected_ids) > 0
    for skill in skills:
        if len(selected) >= max_skills:
            break
        sid = _skill_id(skill)
        if sid and sid not in selected_ids:
            selected.append(skill)
            selected_ids.add(sid)
    return selected, {
        "max_skills": max_skills,
        "required_skill_count": len(required_skill_ids),
        "required_skill_truncated": required_truncated,
        "selected_required_skill_count": len(required_skill_ids & selected_ids),
    }


def _query_to_dict(query: UnifiedSkillRouterQuery) -> dict[str, Any]:
    row = {
        "query_id": query.query_id,
        "query": query.query,
        "benchmark": query.benchmark,
        "positive_skill_ids": list(query.positive_skill_ids),
        "positive_indices": list(query.positive_indices),
    }
    if query.candidate_skill_ids is not None:
        row["candidate_skill_ids"] = list(query.candidate_skill_ids)
    for key in (
        "source_id",
        "kind",
        "split_group_identity",
        "runtime_visible_catalog_id",
        "inventory_catalog_digest",
        "history_mode",
    ):
        value = getattr(query, key)
        if value:
            row[key] = value
    return row


def load_unified_skillrouter_training_corpus(
    data_root: str | Path,
    *,
    max_rows: int | None = None,
    max_skills: int | None = None,
    eval_rows: int = 2048,
    seed: int = 13,
) -> UnifiedSkillRouterCorpus:
    data_root = Path(data_root)
    skills_all = _read_jsonl(data_root / "skill_pool.jsonl")
    full_skill_ids = {_skill_id(skill) for skill in skills_all if _skill_id(skill)}
    aliases = _alias_sets(skills_all)
    raw_queries: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_rows_seen = 0
    for idx, row in enumerate(_iter_jsonl(data_root / "trajectories.jsonl", max_rows=max_rows)):
        source_rows_seen += 1
        if not _routing_enabled(row):
            skipped["routing_mask_disabled"] += 1
            continue
        next_skill_id = str(row.get("next_skill_id") or "").strip()
        if not next_skill_id:
            skipped["missing_next_skill_id"] += 1
            continue
        if next_skill_id not in full_skill_ids:
            skipped["positive_missing_from_skill_pool"] += 1
            continue
        state_text = str(row.get("state_text") or row.get("query") or "").strip()
        if not state_text:
            skipped["missing_state_text"] += 1
            continue
        positives = sorted(aliases.get(next_skill_id, {next_skill_id}) & full_skill_ids)
        raw_queries.append(
            {
                "query_id": _query_id(row, idx),
                "query": state_text,
                "benchmark": str(row.get("benchmark") or row.get("source_benchmark") or "unknown"),
                "positive_skill_ids": positives,
            }
        )
    required = {skill_id for query in raw_queries for skill_id in query["positive_skill_ids"]}
    skills, skill_selection = _select_skills(skills_all, required, max_skills)
    skill_id_to_idx = {_skill_id(skill): idx for idx, skill in enumerate(skills) if _skill_id(skill)}
    usable: list[UnifiedSkillRouterQuery] = []
    for query in raw_queries:
        positive_ids = [skill_id for skill_id in query["positive_skill_ids"] if skill_id in skill_id_to_idx]
        if not positive_ids:
            skipped["positive_removed_by_skill_cap"] += 1
            continue
        usable.append(
            UnifiedSkillRouterQuery(
                query_id=str(query["query_id"]),
                query=str(query["query"]),
                benchmark=str(query["benchmark"]),
                positive_skill_ids=positive_ids,
                positive_indices=[int(skill_id_to_idx[skill_id]) for skill_id in positive_ids],
            )
        )
    if not skills:
        raise ValueError(f"no skills found under {data_root}")
    if not usable:
        raise ValueError("no unified routing rows have positives inside the selected skill pool")
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    eval_count = min(max(0, int(eval_rows)), max(0, len(shuffled) - 1))
    eval_queries = shuffled[:eval_count]
    train_queries = shuffled[eval_count:] if eval_count else shuffled
    if not train_queries:
        train_queries = shuffled
        eval_queries = []
    benchmark_counts = Counter(query.benchmark for query in usable)
    report = {
        "data_root": str(data_root),
        "source_rows_seen": source_rows_seen,
        "skill_count_before_cap": len(skills_all),
        "skill_count": len(skills),
        "raw_query_count": len(raw_queries),
        "usable_query_count": len(usable),
        "train_query_count": len(train_queries),
        "eval_query_count": len(eval_queries),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "skipped_reasons": dict(sorted(skipped.items())),
        "skill_selection": skill_selection,
    }
    return UnifiedSkillRouterCorpus(
        skills=skills,
        train_queries=train_queries,
        eval_queries=eval_queries,
        report=report,
    )


def _positive_mask(queries: list[UnifiedSkillRouterQuery], skill_count: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(queries), skill_count), dtype=torch.bool, device=device)
    for row_idx, query in enumerate(queries):
        for skill_idx in query.positive_indices:
            if 0 <= int(skill_idx) < skill_count:
                mask[row_idx, int(skill_idx)] = True
    return mask


def _multi_positive_nll(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, min_value)
    return (torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(positive_logits.float(), dim=-1)).mean()


def _ranking_metrics_from_logits(logits: torch.Tensor, positive_mask: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0:
        return {
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@20": 0.0,
            "recall@50": 0.0,
            "mrr": 0.0,
        }
    order = torch.argsort(logits, dim=-1, descending=True)
    positives_ranked = positive_mask.gather(1, order)
    first_positive = positives_ranked.float().argmax(dim=-1) + 1
    has_positive = positives_ranked.any(dim=-1)
    first_positive = torch.where(has_positive, first_positive, torch.full_like(first_positive, logits.size(-1) + 1))
    return {
        "recall@1": float((first_positive <= 1).float().mean().cpu().item()),
        "recall@5": float((first_positive <= 5).float().mean().cpu().item()),
        "recall@20": float((first_positive <= 20).float().mean().cpu().item()),
        "recall@50": float((first_positive <= 50).float().mean().cpu().item()),
        "mrr": float((1.0 / first_positive.float()).mean().cpu().item()),
    }


def _evaluate_queries(
    adapter: SkillRouterStyleAdapter,
    raw_query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    queries: list[UnifiedSkillRouterQuery],
    *,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    if not queries:
        return {}
    positive_mask_all = _positive_mask(queries, doc_embs.size(0), device=device)
    metric_rows: list[dict[str, float]] = []
    adapter.eval()
    with torch.no_grad():
        projected_docs = adapter.encode_docs(doc_embs.to(device))
        for start in range(0, len(queries), max(1, int(batch_size))):
            q_raw = raw_query_embs[start : start + max(1, int(batch_size))].to(device)
            q_embs = adapter.encode_queries(q_raw)
            logits = (q_embs @ projected_docs.T) / float(temperature)
            metric_rows.append(
                _ranking_metrics_from_logits(logits, positive_mask_all[start : start + q_raw.size(0)])
            )
    keys = metric_rows[0].keys()
    return {key: float(sum(row[key] for row in metric_rows) / len(metric_rows)) for key in keys}


def run_unified_skillrouter_finetune(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    encoder_model_path: str = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    max_rows: int | None = None,
    max_skills: int | None = None,
    eval_rows: int = 2048,
    max_steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 5.0e-5,
    temperature: float = 0.05,
    seed: int = 13,
    encoder_max_length: int = 2048,
    encoder_batch_size: int = 16,
    torch_dtype: str | None = "bfloat16",
    tokenizer_padding_side: str = "left",
    projection_init: str = "identity",
    train_doc_projection: bool = False,
    eval_every: int = 100,
    log_every: int = 10,
    checkpoint_every: int = 400,
    pooling: str = "auto",
    query_text_mode: str = "auto",
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_progress(output_dir, stage="loading_data", extra={"data_root": str(data_root)})
    corpus = load_unified_skillrouter_training_corpus(
        data_root,
        max_rows=max_rows,
        max_skills=max_skills,
        eval_rows=eval_rows,
        seed=seed,
    )
    _write_jsonl(output_dir / "selected_skills.jsonl", corpus.skills)
    _write_jsonl(output_dir / "train_queries.jsonl", [_query_to_dict(query) for query in corpus.train_queries])
    _write_jsonl(output_dir / "eval_queries.jsonl", [_query_to_dict(query) for query in corpus.eval_queries])
    print(
        "[unified-skillrouter-ft] data ready "
        f"train={len(corpus.train_queries)} eval={len(corpus.eval_queries)} skills={len(corpus.skills)}",
        flush=True,
    )
    resolved_pooling = _resolve_hf_pooling(encoder_model_path, pooling)
    resolved_query_text_mode = _resolve_query_text_mode(encoder_model_path, query_text_mode, resolved_pooling)
    resolved_padding_side = _resolve_tokenizer_padding_side(
        encoder_model_path,
        tokenizer_padding_side,
        resolved_pooling,
    )
    _write_training_progress(
        output_dir,
        stage="loading_backbone",
        extra={
            "encoder_model_path": str(encoder_model_path),
            "pooling": resolved_pooling,
            "query_text_mode": resolved_query_text_mode,
            "tokenizer_padding_side": resolved_padding_side,
        },
    )
    model, tokenizer, device = _load_backbone(
        encoder_model_path,
        torch_dtype,
        resolved_padding_side,
        pooling=resolved_pooling,
    )
    skill_texts = [_skillrouter_skill_text(skill) for skill in corpus.skills]
    train_query_texts = [
        _query_text_for_mode({"query": query.query}, resolved_query_text_mode) for query in corpus.train_queries
    ]
    eval_query_texts = [
        _query_text_for_mode({"query": query.query}, resolved_query_text_mode) for query in corpus.eval_queries
    ]
    print("[unified-skillrouter-ft] encoding skills", flush=True)
    _write_training_progress(
        output_dir,
        stage="encoding_skills",
        extra={"skill_count": len(skill_texts), "encoder_batch_size": int(encoder_batch_size)},
    )
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        skill_texts,
        encoder_batch_size,
        encoder_max_length,
        pooling=resolved_pooling,
    ).to(device)
    print("[unified-skillrouter-ft] encoding train queries", flush=True)
    _write_training_progress(
        output_dir,
        stage="encoding_train_queries",
        extra={"train_query_count": len(train_query_texts), "encoder_batch_size": int(encoder_batch_size)},
    )
    raw_train_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        train_query_texts,
        encoder_batch_size,
        encoder_max_length,
        pooling=resolved_pooling,
    ).to(device)
    print("[unified-skillrouter-ft] encoding eval queries", flush=True)
    _write_training_progress(
        output_dir,
        stage="encoding_eval_queries",
        extra={"eval_query_count": len(eval_query_texts), "encoder_batch_size": int(encoder_batch_size)},
    )
    raw_eval_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        eval_query_texts,
        encoder_batch_size,
        encoder_max_length,
        pooling=resolved_pooling,
    )
    adapter = SkillRouterStyleAdapter(
        hidden_size=int(model.config.hidden_size),
        d=int(model.config.hidden_size),
        projection_init=projection_init,
    ).to(device)
    if not train_doc_projection:
        for param in adapter.d_proj.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW((param for param in adapter.parameters() if param.requires_grad), lr=float(learning_rate))
    positive_mask_all = _positive_mask(corpus.train_queries, len(corpus.skills), device=device)
    fixed_doc_embs = None
    if not train_doc_projection:
        with torch.no_grad():
            fixed_doc_embs = adapter.encode_docs(raw_skill_embs).detach()
    metrics_path = output_dir / "training_metrics.jsonl"
    metric_rows: list[dict[str, float]] = []
    max_steps = max(1, int(max_steps))
    batch_size = max(1, int(batch_size))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(checkpoint_every))
    started_at = time.time()

    def save_checkpoint(step: int, metrics: dict[str, float]) -> Path:
        checkpoint_path = checkpoint_dir / f"unified_skillrouter_finetune-step{step}.pt"
        config = {
            "encoder_model_path": str(encoder_model_path),
            "temperature": float(temperature),
            "projection_init": projection_init,
            "encoder_max_length": int(encoder_max_length),
            "torch_dtype": torch_dtype,
            "tokenizer_padding_side": resolved_padding_side,
            "pooling": resolved_pooling,
            "train_doc_projection": bool(train_doc_projection),
            "query_template": resolved_query_text_mode,
            "query_text_mode": resolved_query_text_mode,
            "doc_template": "skillret_official_full_text",
            "training_objective": "multi_positive_full_pool_nll",
            "backbone": "frozen",
        }
        payload = {
            "method": "unified_skillrouter_style_finetune",
            "step": int(step),
            "adapter_state_dict": adapter.state_dict(),
            "config": config,
            "corpus_report": corpus.report,
            "last_metrics": metrics,
            "uses_clstr_heads": False,
            "uses_clstr_transition": False,
            "source_note": "CLSTR-side SkillRouter-style adapter finetune on unified CLSTR train trajectories.",
        }
        torch.save(payload, checkpoint_path)
        torch.save(payload, checkpoint_dir / "latest.pt")
        return checkpoint_path

    _write_training_progress(
        output_dir,
        stage="training",
        step=0,
        max_steps=max_steps,
        extra={
            "skill_count": len(corpus.skills),
            "train_query_count": len(corpus.train_queries),
            "eval_query_count": len(corpus.eval_queries),
        },
    )
    for step in range(1, max_steps + 1):
        start = ((step - 1) * batch_size) % len(corpus.train_queries)
        batch_indices = [(start + offset) % len(corpus.train_queries) for offset in range(batch_size)]
        q_raw = raw_train_query_embs[batch_indices]
        pos_mask = positive_mask_all[batch_indices]
        adapter.train()
        query_embs = adapter.encode_queries(q_raw)
        doc_embs = fixed_doc_embs if fixed_doc_embs is not None else adapter.encode_docs(raw_skill_embs)
        logits = (query_embs @ doc_embs.T) / float(temperature)
        loss = _multi_positive_nll(logits, pos_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            train_metrics = _ranking_metrics_from_logits(logits, pos_mask)
        row: dict[str, float] = {
            "step": float(step),
            "loss": float(loss.detach().cpu().item()),
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }
        if corpus.eval_queries and (step == 1 or step == max_steps or step % max(1, int(eval_every)) == 0):
            eval_metrics = _evaluate_queries(
                adapter,
                raw_eval_query_embs,
                raw_skill_embs.detach().cpu(),
                corpus.eval_queries,
                batch_size=max(1, int(encoder_batch_size)),
                temperature=temperature,
                device=device,
            )
            row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
        metric_rows.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if checkpoint_every and (step == 1 or step == max_steps or step % checkpoint_every == 0):
            checkpoint_path = save_checkpoint(step, row)
            row["checkpoint_step"] = float(step)
            row["checkpoint_path"] = str(checkpoint_path)  # type: ignore[assignment]
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            elapsed = max(0.0, time.time() - started_at)
            steps_per_second = step / elapsed if elapsed > 0 else 0.0
            remaining = (max_steps - step) / steps_per_second if steps_per_second > 0 else None
            _write_training_progress(
                output_dir,
                stage="training",
                step=step,
                max_steps=max_steps,
                extra={
                    "loss": row["loss"],
                    "train_recall@1": row["train_recall@1"],
                    "train_recall@5": row["train_recall@5"],
                    "elapsed_seconds": elapsed,
                    "eta_seconds": remaining,
                },
            )
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            print(
                "[unified-skillrouter-ft] "
                f"step={step}/{max_steps} loss={row['loss']:.6f} "
                f"train_recall@1={row['train_recall@1']:.4f} "
                f"train_recall@5={row['train_recall@5']:.4f}",
                flush=True,
            )
    config = {
        "encoder_model_path": str(encoder_model_path),
        "temperature": float(temperature),
        "projection_init": projection_init,
        "encoder_max_length": int(encoder_max_length),
        "torch_dtype": torch_dtype,
        "tokenizer_padding_side": resolved_padding_side,
        "pooling": resolved_pooling,
        "train_doc_projection": bool(train_doc_projection),
        "query_template": resolved_query_text_mode,
        "query_text_mode": resolved_query_text_mode,
        "doc_template": "skillret_official_full_text",
        "training_objective": "multi_positive_full_pool_nll",
        "backbone": "frozen",
    }
    checkpoint_path = save_checkpoint(max_steps, metric_rows[-1] if metric_rows else {})
    report = {
        "status": "ok",
        "method": "unified_skillrouter_style_finetune",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "train_doc_projection": bool(train_doc_projection),
        "corpus_report": corpus.report,
        "last_metrics": metric_rows[-1] if metric_rows else {},
        "model_config": config,
        "uses_clstr_heads": False,
        "note": "Official-compatible SkillRouter-style baseline; train/eval data are unified CLSTR train rows, not external benchmark eval rows.",
    }
    _write_json(output_dir / "train_report.json", report)
    _write_json(output_dir / "metrics.json", report)
    _write_training_progress(
        output_dir,
        stage="complete",
        step=max_steps,
        max_steps=max_steps,
        extra={"status": "ok", "checkpoint_path": str(checkpoint_path), "last_metrics": metric_rows[-1] if metric_rows else {}},
    )
    return report
