from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.bge_reranker import _resolve_dtype, bge_reranker_score_tensor
from clstr.exact_candidate_cache import load_or_build_exact_candidate_rankings
from clstr.bge_sr_reranker_train import (
    _load_reranker_resume_state,
    _file_sha256,
    _capture_rng_state,
    _load_sr_embedding_checkpoint_config,
    _move_optimizer_state_to_device,
    _prune_old_checkpoints,
    _rank_candidates,
    _ranking_device_contract,
    _read_jsonl,
    _skill_id,
    _write_json,
    _write_jsonl,
    _write_reranker_progress,
    _validate_resume_training_contract,
    _restore_rng_state,
)
from clstr.skillret_official import _skillrouter_skill_text
from clstr.skillrouter_style import (
    _encode_raw_texts,
    _load_backbone,
    _query_text_for_mode,
    _resolve_hf_pooling,
    _resolve_query_text_mode,
    _resolve_tokenizer_padding_side,
)
from clstr.training_monitor import TrainingMonitor


TOOLRANK_OBJECTIVE = "toolrex_true_false_pair_bce_over_tool_embed_candidates"


def build_toolrank_pairwise_pairs(
    *,
    queries: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    ranked_skill_ids_by_query: dict[str, list[str]],
    top_k: int = 20,
    negatives_per_query: int = 5,
    max_pairs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(top_k) <= 0:
        raise ValueError("ToolREx top_k must be positive")
    if int(negatives_per_query) < 0:
        raise ValueError("ToolREx negatives_per_query must be nonnegative")
    skill_by_id = {_skill_id(skill): skill for skill in skills if _skill_id(skill)}
    pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    positive_pair_count = 0
    negative_pair_count = 0
    for query in queries:
        if max_pairs is not None and len(pairs) >= int(max_pairs):
            break
        query_id = str(query.get("query_id") or "").strip()
        query_text = str(query.get("query") or "").strip()
        positives = [
            str(item).strip()
            for item in query.get("positive_skill_ids", [])
            if str(item).strip()
        ]
        if not query_id or not query_text or not positives:
            skipped["missing_query_or_positive"] += 1
            continue
        valid_positives = [skill_id for skill_id in dict.fromkeys(positives) if skill_id in skill_by_id]
        if not valid_positives:
            skipped["positive_missing_from_skills"] += 1
            continue
        ranked = [
            str(skill_id).strip()
            for skill_id in ranked_skill_ids_by_query.get(query_id, [])
            if str(skill_id).strip() in skill_by_id
        ][: int(top_k)]
        negatives = [
            skill_id
            for skill_id in dict.fromkeys(ranked)
            if skill_id not in set(valid_positives)
        ][: max(0, int(negatives_per_query))]
        if not negatives:
            skipped["missing_negative_candidates"] += 1
            continue
        for skill_id in valid_positives:
            if max_pairs is not None and len(pairs) >= int(max_pairs):
                break
            pairs.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "skill_id": skill_id,
                    "doc_text": _skillrouter_skill_text(skill_by_id[skill_id]),
                    "label": 1,
                }
            )
            positive_pair_count += 1
        for skill_id in negatives:
            if max_pairs is not None and len(pairs) >= int(max_pairs):
                break
            pairs.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "skill_id": skill_id,
                    "doc_text": _skillrouter_skill_text(skill_by_id[skill_id]),
                    "label": 0,
                }
            )
            negative_pair_count += 1
    return pairs, {
        "pair_count": len(pairs),
        "positive_pair_count": positive_pair_count,
        "negative_pair_count": negative_pair_count,
        "query_count": len(queries),
        "top_k": int(top_k),
        "negatives_per_query": int(negatives_per_query),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _pair_batch_to_texts(pairs: list[dict[str, Any]]) -> tuple[list[str], list[str], torch.Tensor]:
    queries = [str(row.get("query") or "") for row in pairs]
    docs = [str(row.get("doc_text") or "") for row in pairs]
    labels = torch.tensor([float(row.get("label") or 0.0) for row in pairs], dtype=torch.float32)
    return queries, docs, labels


def _pairwise_batch_loss(model, tokenizer, device, pairs: list[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, dict[str, float]]:
    queries, docs, labels = _pair_batch_to_texts(pairs)
    tokens = tokenizer(
        queries,
        docs,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    labels = labels.to(device)
    scores = bge_reranker_score_tensor(model(**tokens).logits).float()
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    with torch.no_grad():
        predictions = (torch.sigmoid(scores) >= 0.5).float()
        positive_mask = labels >= 0.5
        negative_mask = ~positive_mask
        metrics = {
            "accuracy": float((predictions == labels).float().mean().cpu().item()),
            "positive_accuracy": float((predictions[positive_mask] == 1.0).float().mean().cpu().item())
            if positive_mask.any()
            else 0.0,
            "negative_accuracy": float((predictions[negative_mask] == 0.0).float().mean().cpu().item())
            if negative_mask.any()
            else 0.0,
            "positive_score_mean": float(scores[positive_mask].mean().cpu().item()) if positive_mask.any() else 0.0,
            "negative_score_mean": float(scores[negative_mask].mean().cpu().item()) if negative_mask.any() else 0.0,
            "pair_count": float(labels.numel()),
        }
    return loss, metrics


def _build_pairs_from_tool_embed(
    *,
    tool_embed_output_dir: Path,
    tool_embed_checkpoint_path: Path,
    encoder_max_length: int,
    encoder_batch_size: int,
    torch_dtype: str,
    top_k: int,
    rank_batch_size: int,
    negatives_per_query: int,
    max_train_queries: int | None,
    max_eval_queries: int | None,
    max_train_pairs: int | None,
    max_eval_pairs: int | None,
    output_dir: Path,
    candidate_cache_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_skills_path = tool_embed_output_dir / "selected_skills.jsonl"
    train_queries_path = tool_embed_output_dir / "train_queries.jsonl"
    eval_queries_path = tool_embed_output_dir / "eval_queries.jsonl"
    skills = _read_jsonl(selected_skills_path)
    train_queries = _read_jsonl(train_queries_path, max_rows=max_train_queries)
    eval_queries = _read_jsonl(eval_queries_path, max_rows=max_eval_queries)
    config = _load_sr_embedding_checkpoint_config(tool_embed_checkpoint_path)
    pooling = _resolve_hf_pooling(tool_embed_checkpoint_path, str(config.get("pooling") or "cls"))
    query_text_mode = _resolve_query_text_mode(tool_embed_checkpoint_path, str(config.get("query_text_mode") or "raw"), pooling)
    padding_side = _resolve_tokenizer_padding_side(
        tool_embed_checkpoint_path,
        str(config.get("tokenizer_padding_side") or "right"),
        pooling,
    )
    skill_texts = [_skillrouter_skill_text(skill) for skill in skills]
    all_queries = train_queries + eval_queries
    query_texts = [_query_text_for_mode({"query": row.get("query", "")}, query_text_mode) for row in all_queries]
    skill_ids = [_skill_id(skill) for skill in skills]
    query_ids = [str(row.get("query_id")) for row in all_queries]
    candidate_skill_ids_by_query = [
        (
            [str(item) for item in row.get("candidate_skill_ids") or []]
            if "candidate_skill_ids" in row
            else None
        )
        for row in all_queries
    ]
    score_device, ranking_device = _ranking_device_contract()

    def build_ranked() -> dict[str, list[str]]:
        model, tokenizer, _device = _load_backbone(
            str(tool_embed_checkpoint_path),
            torch_dtype,
            padding_side,
            pooling=pooling,
        )
        skill_embs = _encode_raw_texts(
            model,
            tokenizer,
            skill_texts,
            encoder_batch_size,
            encoder_max_length,
            pooling=pooling,
        )
        query_embs = _encode_raw_texts(
            model,
            tokenizer,
            query_texts,
            encoder_batch_size,
            encoder_max_length,
            pooling=pooling,
        )
        return _rank_candidates(
            query_ids=query_ids,
            query_embs=query_embs,
            skill_embs=skill_embs,
            skill_ids=skill_ids,
            top_k=top_k,
            batch_size=rank_batch_size,
            candidate_skill_ids_by_query=candidate_skill_ids_by_query,
            score_device=score_device,
        )

    ranked, candidate_cache_report = load_or_build_exact_candidate_rankings(
        candidate_cache_dir=candidate_cache_dir,
        checkpoint_path=tool_embed_checkpoint_path,
        selected_skills_path=selected_skills_path,
        train_queries_path=train_queries_path,
        eval_queries_path=eval_queries_path,
        pooling=pooling,
        query_text_mode=query_text_mode,
        tokenizer_padding_side=padding_side,
        torch_dtype=torch_dtype,
        max_length=encoder_max_length,
        top_k=top_k,
        max_train_queries=max_train_queries,
        max_eval_queries=max_eval_queries,
        skill_ids=skill_ids,
        query_ids=query_ids,
        builder=build_ranked,
        candidate_skill_ids_by_query=candidate_skill_ids_by_query,
        ranking_device=ranking_device,
    )
    train_pairs, train_report = build_toolrank_pairwise_pairs(
        queries=train_queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=top_k,
        negatives_per_query=negatives_per_query,
        max_pairs=max_train_pairs,
    )
    eval_pairs, eval_report = build_toolrank_pairwise_pairs(
        queries=eval_queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=top_k,
        negatives_per_query=negatives_per_query,
        max_pairs=max_eval_pairs,
    )
    _write_jsonl(output_dir / "train_toolrank_pairs.jsonl", train_pairs)
    _write_jsonl(output_dir / "eval_toolrank_pairs.jsonl", eval_pairs)
    return train_pairs, eval_pairs, {
        "skill_count": len(skills),
        "train_query_count": len(train_queries),
        "eval_query_count": len(eval_queries),
        "train_pair_report": train_report,
        "eval_pair_report": eval_report,
        "tool_embed_checkpoint_path": str(tool_embed_checkpoint_path),
        "pooling": pooling,
        "query_text_mode": query_text_mode,
        "tokenizer_padding_side": padding_side,
        "ranking_device": ranking_device,
        "candidate_cache": candidate_cache_report,
    }


def run_bge_toolrex_reranker_train(
    *,
    tool_embed_output_dir: str | Path,
    tool_embed_checkpoint_path: str | Path,
    output_dir: str | Path,
    reranker_model_path: str = "models/BAAI/bge-reranker-v2-m3",
    max_steps: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1.0e-5,
    top_k: int = 20,
    negatives_per_query: int = 5,
    encoder_max_length: int = 512,
    encoder_batch_size: int = 64,
    rank_batch_size: int = 256,
    reranker_max_length: int = 512,
    torch_dtype: str = "bfloat16",
    seed: int = 13,
    eval_every: int = 50,
    log_every: int = 10,
    checkpoint_every: int = 400,
    max_train_queries: int | None = None,
    max_eval_queries: int | None = None,
    max_train_pairs: int | None = None,
    max_eval_pairs: int | None = None,
    freeze_backbone: bool = False,
    resume_checkpoint_path: str | Path | None = None,
    max_checkpoints_to_keep: int = 2,
    candidate_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if float(learning_rate) <= 0:
        raise ValueError("ToolREx reranker learning_rate must be positive")
    if int(top_k) <= 0:
        raise ValueError("ToolREx reranker top_k must be positive")
    if int(negatives_per_query) < 0:
        raise ValueError("ToolREx negatives_per_query must be nonnegative")
    if int(encoder_max_length) <= 0 or int(reranker_max_length) <= 0:
        raise ValueError("ToolREx reranker max lengths must be positive")
    batch_size = max(1, int(batch_size))
    random.seed(seed)
    torch.manual_seed(seed)
    tool_embed_output_dir = Path(tool_embed_output_dir)
    tool_embed_checkpoint_path = Path(tool_embed_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_state = _load_reranker_resume_state(resume_checkpoint_path) if resume_checkpoint_path is not None else None
    _write_reranker_progress(output_dir, stage="building_toolrank_pairs")
    train_pairs, eval_pairs, pair_report = _build_pairs_from_tool_embed(
        tool_embed_output_dir=tool_embed_output_dir,
        tool_embed_checkpoint_path=tool_embed_checkpoint_path,
        encoder_max_length=encoder_max_length,
        encoder_batch_size=encoder_batch_size,
        torch_dtype=torch_dtype,
        top_k=top_k,
        rank_batch_size=rank_batch_size,
        negatives_per_query=negatives_per_query,
        max_train_queries=max_train_queries,
        max_eval_queries=max_eval_queries,
        max_train_pairs=max_train_pairs,
        max_eval_pairs=max_eval_pairs,
        output_dir=output_dir,
        candidate_cache_dir=candidate_cache_dir,
    )
    if not train_pairs:
        raise ValueError("no ToolRex Tool-Rank true/false training pairs")
    training_contract = {
        "schema_version": "bge_toolrex_reranker_training_contract_v1",
        "reranker_model_path": str(Path(reranker_model_path).resolve()),
        "training_data": {
            "train_pairs_sha256": _file_sha256(output_dir / "train_toolrank_pairs.jsonl"),
            "eval_pairs_sha256": _file_sha256(output_dir / "eval_toolrank_pairs.jsonl"),
        },
        "objective": {
            "training_objective": TOOLRANK_OBJECTIVE,
            "top_k": int(top_k),
            "negatives_per_query": int(negatives_per_query),
            "reranker_max_length": int(reranker_max_length),
            "freeze_backbone": bool(freeze_backbone),
        },
        "schedule": {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
        },
        "numerics": {"torch_dtype": str(torch_dtype)},
    }
    resume_contract_status = _validate_resume_training_contract(
        resume_state,
        training_contract,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_reranker_progress(output_dir, stage="loading_reranker", extra=pair_report)
    load_reranker_path = resume_state["checkpoint_path"] if resume_state is not None else reranker_model_path
    tokenizer = AutoTokenizer.from_pretrained(load_reranker_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_reranker_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=_resolve_dtype(torch_dtype),
    ).to(device)
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("classifier") or name.startswith("score"))
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("ToolREx reranker configuration leaves no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(
        output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_steps=max(1, int(checkpoint_every)),
        curve_interval_steps=max(1, int(checkpoint_every)),
        reset=resume_state is None,
    )
    if resume_state is not None:
        if monitor.metrics_path.exists():
            monitor.history = _read_jsonl(monitor.metrics_path)
        if resume_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
    rng_resume_status = _restore_rng_state(resume_state)
    max_steps = max(1, int(max_steps))
    checkpoint_every = max(0, int(checkpoint_every))
    start_step = int(resume_state["step"]) + 1 if resume_state is not None else 1
    started_at = time.time()
    metric_rows: list[dict[str, Any]] = list(monitor.history)

    def save_checkpoint(step: int, metrics: dict[str, Any]) -> Path:
        checkpoint_path = checkpoint_dir / f"bge_tool_rank-step{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        payload = {
            "method": "bge_tool_rank_pairwise_ce_finetune",
            "step": int(step),
            "model_name_or_path": str(reranker_model_path),
            "training_contract": training_contract,
            "config": {
                "tool_embed_output_dir": str(tool_embed_output_dir),
                "tool_embed_checkpoint_path": str(tool_embed_checkpoint_path),
                "reranker_model_path": str(reranker_model_path),
                "top_k": int(top_k),
                "negatives_per_query": int(negatives_per_query),
                "reranker_max_length": int(reranker_max_length),
                "training_objective": TOOLRANK_OBJECTIVE,
                "freeze_backbone": bool(freeze_backbone),
                "max_checkpoints_to_keep": int(max_checkpoints_to_keep),
            },
            "last_metrics": metrics,
            "pair_report": pair_report,
        }
        torch.save(
            {
                **payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": _capture_rng_state(),
            },
            checkpoint_path / "training_state.pt",
        )
        _write_json(checkpoint_path / "train_checkpoint_report.json", payload)
        _write_json(
            checkpoint_dir / "latest.json",
            {"path": str(checkpoint_path.resolve()), "step": int(step)},
        )
        _write_json(
            checkpoint_dir / "retention_report.json",
            _prune_old_checkpoints(
                checkpoint_dir,
                latest_path=checkpoint_path,
                max_checkpoints_to_keep=max_checkpoints_to_keep,
            ),
        )
        return checkpoint_path

    _write_reranker_progress(
        output_dir,
        stage="training",
        step=max(0, start_step - 1),
        max_steps=max_steps,
        extra={
            "train_pair_count": len(train_pairs),
            "eval_pair_count": len(eval_pairs),
            "training_objective": TOOLRANK_OBJECTIVE,
            "resume_enabled": bool(resume_state is not None),
            "resume_checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
        },
    )
    if start_step > max_steps:
        report = {
            "status": "ok",
            "method": "bge_tool_rank_pairwise_ce_finetune",
            "output_dir": str(output_dir),
            "checkpoint_path": resume_state["checkpoint_path"] if resume_state is not None else None,
            "message": "resume checkpoint is already at or beyond max_steps",
            "resume_contract_status": resume_contract_status,
            "rng_resume_status": rng_resume_status,
            "last_metrics": metric_rows[-1] if metric_rows else (resume_state.get("last_metrics", {}) if resume_state else {}),
            **monitor.paths_report(),
        }
        _write_json(output_dir / "train_report.json", report)
        _write_json(output_dir / "metrics.json", report)
        return report
    last_checkpoint_path: Path | None = None
    for step in range(start_step, max_steps + 1):
        start = ((step - 1) * batch_size) % len(train_pairs)
        batch = [train_pairs[(start + offset) % len(train_pairs)] for offset in range(batch_size)]
        model.train()
        loss, train_metrics = _pairwise_batch_loss(model, tokenizer, device, batch, reranker_max_length)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "training_objective": TOOLRANK_OBJECTIVE,
        }
        if eval_pairs and (step == 1 or step == max_steps or step % max(1, int(eval_every)) == 0):
            model.eval()
            eval_batch = eval_pairs[: min(len(eval_pairs), batch_size)]
            with torch.no_grad():
                _loss, eval_metrics = _pairwise_batch_loss(model, tokenizer, device, eval_batch, reranker_max_length)
            row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
        if checkpoint_every and (step == 1 or step == max_steps or step % checkpoint_every == 0):
            last_checkpoint_path = save_checkpoint(step, row)
            row["checkpoint_path"] = str(last_checkpoint_path)
        metric_rows.append(row)
        monitor.record(row)
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            elapsed = time.time() - started_at
            _write_reranker_progress(
                output_dir,
                stage="training",
                step=step,
                max_steps=max_steps,
                extra={
                    "loss": row["loss"],
                    "train_accuracy": row["train_accuracy"],
                    "train_positive_accuracy": row["train_positive_accuracy"],
                    "train_negative_accuracy": row["train_negative_accuracy"],
                    "elapsed_seconds": elapsed,
                    "training_objective": TOOLRANK_OBJECTIVE,
                },
            )
            print(
                "[bge-toolrex-tool-rank] "
                f"step={step}/{max_steps} loss={row['loss']:.6f} "
                f"acc={row['train_accuracy']:.4f} pos_acc={row['train_positive_accuracy']:.4f} "
                f"neg_acc={row['train_negative_accuracy']:.4f}",
                flush=True,
            )
    checkpoint_path = last_checkpoint_path or save_checkpoint(
        max_steps,
        metric_rows[-1] if metric_rows else {},
    )
    report = {
        "status": "ok",
        "method": "bge_tool_rank_pairwise_ce_finetune",
        "training_objective": TOOLRANK_OBJECTIVE,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "tool_embed_output_dir": str(tool_embed_output_dir),
        "tool_embed_checkpoint_path": str(tool_embed_checkpoint_path),
        "reranker_model_path": str(reranker_model_path),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "top_k": int(top_k),
        "negatives_per_query": int(negatives_per_query),
        "pair_report": pair_report,
        "last_metrics": metric_rows[-1] if metric_rows else {},
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
            "contract_status": resume_contract_status,
            "rng_status": rng_resume_status,
        },
        **monitor.paths_report(),
    }
    _write_json(output_dir / "train_report.json", report)
    _write_json(output_dir / "metrics.json", report)
    _write_reranker_progress(
        output_dir,
        stage="complete",
        step=max_steps,
        max_steps=max_steps,
        extra={"status": "ok", "checkpoint_path": str(checkpoint_path), "last_metrics": metric_rows[-1] if metric_rows else {}},
    )
    return report
