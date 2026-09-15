from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.encoders import _resolve_torch_dtype
from clstr.skillret import load_skillret_training_rows
from clstr.skillret_official import (
    _default_hf_pooling,
    _last_token_pool,
    _pool_hf_hidden,
    _rank_embedding_rows,
    _select_eval_subset,
    _skillrouter_query_text,
    _skillrouter_skill_text,
    load_official_qrels,
    load_official_queries,
    load_official_skills,
    write_json,
    write_jsonl,
    write_trec_run,
)


SUPPORTED_POOLING_MODES = {"auto", "last_token", "cls", "masked_mean"}
SUPPORTED_QUERY_TEXT_MODES = {"auto", "skillrouter", "raw", "raw_state"}


class SkillRouterStyleAdapter(nn.Module):
    def __init__(self, hidden_size: int, d: int, projection_init: str = "identity"):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, d, bias=False)
        self.d_proj = nn.Linear(hidden_size, d, bias=False)
        if projection_init == "identity":
            if hidden_size != d:
                raise ValueError(
                    "identity projection requires model_dim to match hidden size "
                    f"({d} != {hidden_size})"
                )
            with torch.no_grad():
                eye = torch.eye(d)
                self.q_proj.weight.copy_(eye)
                self.d_proj.weight.copy_(eye)
        elif projection_init != "default":
            raise ValueError(f"unsupported projection_init: {projection_init}")

    def encode_queries(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_proj(raw), p=2, dim=-1)

    def encode_docs(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.d_proj(raw), p=2, dim=-1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _load_backbone(
    base_model_name: str,
    torch_dtype: str | None,
    tokenizer_padding_side: str,
    pooling: str = "auto",
):
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_pooling = _resolve_hf_pooling(base_model_name, pooling)
    resolved_padding_side = _resolve_tokenizer_padding_side(
        base_model_name,
        tokenizer_padding_side,
        resolved_pooling,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side=resolved_padding_side,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    model = AutoModel.from_pretrained(base_model_name, **model_kwargs)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, device


def _resolve_hf_pooling(model_name_or_path: str | Path, pooling: str | None = "auto") -> str:
    mode = str(pooling or "auto").strip().lower()
    if mode not in SUPPORTED_POOLING_MODES:
        raise ValueError(f"unsupported pooling mode: {pooling}")
    if mode == "auto":
        return _default_hf_pooling(str(model_name_or_path))
    return mode


def _resolve_tokenizer_padding_side(
    model_name_or_path: str | Path,
    tokenizer_padding_side: str | None,
    pooling: str,
) -> str:
    side = str(tokenizer_padding_side or "auto").strip().lower()
    if side == "auto":
        return "right" if pooling in {"cls", "masked_mean"} else "left"
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported tokenizer_padding_side: {tokenizer_padding_side}")
    return side


def _resolve_query_text_mode(
    model_name_or_path: str | Path,
    query_text_mode: str | None,
    pooling: str,
) -> str:
    mode = str(query_text_mode or "auto").strip().lower()
    if mode not in SUPPORTED_QUERY_TEXT_MODES:
        raise ValueError(f"unsupported query_text_mode: {query_text_mode}")
    if mode == "raw_state":
        return "raw"
    if mode == "auto":
        label = Path(str(model_name_or_path)).name.lower()
        return "raw" if pooling in {"cls", "masked_mean"} or label in {"bge-m3", "bge_m3"} else "skillrouter"
    return mode


def _query_text_for_mode(query: dict[str, Any], query_text_mode: str) -> str:
    mode = str(query_text_mode or "skillrouter").strip().lower()
    if mode in {"raw", "raw_state"}:
        return str(query.get("query", "")).strip()
    if mode == "skillrouter":
        return _skillrouter_query_text(query)
    raise ValueError(f"unsupported query_text_mode: {query_text_mode}")


def _encode_raw_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    pooling: str = "last_token",
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        tok = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {key: value.to(device) for key, value in tok.items()}
        with torch.no_grad():
            out = model(**tok)
            if pooling == "last_token":
                pooled = _last_token_pool(out.last_hidden_state, tok["attention_mask"])
            else:
                pooled = _pool_hf_hidden(out.last_hidden_state, tok["attention_mask"], pooling)
        encoded.append(pooled.float().cpu())
    if not encoded:
        hidden_size = int(model.config.hidden_size)
        return torch.empty(0, hidden_size)
    return torch.cat(encoded, dim=0)


def _load_training_rows(
    data_root: str | Path,
    max_skills: int | None,
    max_queries: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], int]:
    skill_rows, query_rows, positives_by_query = load_skillret_training_rows(data_root, split="train")
    if max_skills is not None:
        skill_rows = skill_rows[:max_skills]
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skill_rows)}
    usable_queries: list[dict[str, Any]] = []
    for query in query_rows:
        qid = str(query["query_id"])
        positive_indices = [
            skill_id_to_idx[skill_id]
            for skill_id in positives_by_query.get(qid, query.get("positive_skill_ids", []))
            if skill_id in skill_id_to_idx
        ]
        if positive_indices:
            copied = dict(query)
            copied["positive_indices"] = positive_indices
            usable_queries.append(copied)
        if max_queries is not None and len(usable_queries) >= max_queries:
            break
    if not skill_rows:
        raise ValueError(f"no SKILLRET train skills found under {data_root}")
    if not usable_queries:
        raise ValueError("no SKILLRET train queries have positives inside the selected skill pool")
    return skill_rows, usable_queries, positives_by_query, len(query_rows)


def run_skillrouter_style_finetune(
    data_root: str | Path,
    base_model_name: str,
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 2,
    model_dim: int = 1024,
    max_skills: int | None = None,
    max_queries: int | None = None,
    learning_rate: float = 1.0e-6,
    temperature: float = 0.05,
    seed: int = 13,
    max_length: int = 32768,
    skill_batch_size: int = 1,
    query_batch_size: int | None = None,
    torch_dtype: str | None = "bfloat16",
    tokenizer_padding_side: str = "left",
    projection_init: str = "identity",
    pooling: str = "auto",
    query_text_mode: str = "auto",
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    skill_rows, queries, _positives_by_query, query_count_before_filter = _load_training_rows(
        data_root,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    resolved_pooling = _resolve_hf_pooling(base_model_name, pooling)
    resolved_query_text_mode = _resolve_query_text_mode(base_model_name, query_text_mode, resolved_pooling)
    resolved_padding_side = _resolve_tokenizer_padding_side(base_model_name, tokenizer_padding_side, resolved_pooling)
    model, tokenizer, device = _load_backbone(
        base_model_name,
        torch_dtype,
        resolved_padding_side,
        pooling=resolved_pooling,
    )
    skill_texts = [_skillrouter_skill_text(skill) for skill in skill_rows]
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        skill_texts,
        skill_batch_size,
        max_length,
        pooling=resolved_pooling,
    ).to(device)
    adapter = SkillRouterStyleAdapter(
        hidden_size=int(model.config.hidden_size),
        d=model_dim,
        projection_init=projection_init,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)

    metrics: dict[str, float] = {"loss": 0.0, "recall_at_1": 0.0, "recall_at_50": 0.0}
    encode_query_batch_size = query_batch_size or batch_size
    for step in range(1, max_steps + 1):
        batch = [queries[(step - 1 + offset) % len(queries)] for offset in range(batch_size)]
        labels = torch.tensor([row["positive_indices"][0] for row in batch], device=device, dtype=torch.long)
        query_texts = [_query_text_for_mode(row, resolved_query_text_mode) for row in batch]
        raw_query_embs = _encode_raw_texts(
            model,
            tokenizer,
            query_texts,
            encode_query_batch_size,
            max_length,
            pooling=resolved_pooling,
        ).to(device)
        query_embs = adapter.encode_queries(raw_query_embs)
        doc_embs = adapter.encode_docs(raw_skill_embs)
        logits = (query_embs @ doc_embs.T) / temperature
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        top50 = torch.topk(logits.detach(), k=min(50, logits.size(-1)), dim=-1).indices
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "recall_at_1": float((logits.detach().argmax(dim=-1) == labels).float().mean().cpu().item()),
            "recall_at_50": float((top50 == labels.unsqueeze(-1)).any(dim=-1).float().mean().cpu().item()),
        }

    checkpoint_path = checkpoint_dir / f"skillrouter_style-step{max_steps}.pt"
    config = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "max_length": max_length,
        "torch_dtype": torch_dtype,
        "tokenizer_padding_side": resolved_padding_side,
        "projection_init": projection_init,
        "pooling": resolved_pooling,
        "temperature": temperature,
        "training_objective": "full_pool_infonce",
        "backbone": "frozen",
        "query_template": resolved_query_text_mode,
        "doc_template": "skillret_official_full_text",
    }
    torch.save(
        {
            "method": "skillrouter_style_finetune",
            "source_note": "CLSTR-side SkillRouter-style bi-encoder adapter finetune; not an official SkillRouter repo training reproduction.",
            "step": max_steps,
            "config": config,
            "adapter_state_dict": adapter.state_dict(),
            "metrics": metrics,
            "skill_count": len(skill_rows),
            "query_count": len(queries),
        },
        checkpoint_path,
    )
    report = {
        "status": "ok",
        "method": "skillrouter_style_finetune",
        "training_objective": "full_pool_infonce",
        "paper_reference": "skillrouter.pdf Section 4 / Appendix F: bi-encoder retrieval with InfoNCE over normalized embeddings, tau=0.05",
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "steps": max_steps,
        "batch_size": batch_size,
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "query_count_before_filter": query_count_before_filter,
        "temperature": temperature,
        "metrics": metrics,
        "model_config": config,
        "uses_clstr_heads": False,
        "uses_skillrouter_eval_labels": False,
        "uses_skillsbench_trajectories": False,
        "note": "SkillRouter public repo has eval code only; this is a CLSTR-side SkillRouter-style adapter finetune baseline on SKILLRET train.",
    }
    _write_json(output_dir / "train_report.json", report)
    return report


def _project_in_batches(adapter: SkillRouterStyleAdapter, raw: torch.Tensor, kind: str, batch_size: int, device: torch.device) -> torch.Tensor:
    projected: list[torch.Tensor] = []
    adapter.eval()
    for start in range(0, raw.size(0), batch_size):
        batch = raw[start : start + batch_size].to(device)
        with torch.no_grad():
            if kind == "query":
                out = adapter.encode_queries(batch)
            elif kind == "doc":
                out = adapter.encode_docs(batch)
            else:
                raise ValueError(kind)
        projected.append(out.cpu())
    return torch.cat(projected, dim=0)


def export_skillrouter_style_official_run(
    data_root: str | Path,
    base_model_name: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    batch_size: int = 1,
    max_length: int = 32768,
    max_queries: int | None = None,
    max_skills: int | None = None,
    run_name: str = "skillrouter_style_finetune_full",
    pooling: str = "auto",
    query_text_mode: str = "auto",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config", {})
    model_name = str(config.get("base_model_name", base_model_name))
    model_dim = int(config.get("d", 1024))
    torch_dtype = config.get("torch_dtype", "bfloat16")
    resolved_pooling = _resolve_hf_pooling(model_name, config.get("pooling", pooling))
    tokenizer_padding_side = str(config.get("tokenizer_padding_side", "auto"))
    resolved_padding_side = _resolve_tokenizer_padding_side(model_name, tokenizer_padding_side, resolved_pooling)
    resolved_query_text_mode = _resolve_query_text_mode(
        model_name,
        config.get("query_template", query_text_mode),
        resolved_pooling,
    )
    model, tokenizer, device = _load_backbone(
        model_name,
        torch_dtype,
        resolved_padding_side,
        pooling=resolved_pooling,
    )
    adapter = SkillRouterStyleAdapter(
        hidden_size=int(model.config.hidden_size),
        d=model_dim,
        projection_init="default",
    ).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()

    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    query_texts = [_query_text_for_mode(query, resolved_query_text_mode) for query in queries]
    skill_texts = [_skillrouter_skill_text(skill) for skill in skills]
    raw_query_embs = _encode_raw_texts(model, tokenizer, query_texts, batch_size, max_length, pooling=resolved_pooling)
    raw_skill_embs = _encode_raw_texts(model, tokenizer, skill_texts, batch_size, max_length, pooling=resolved_pooling)
    query_embs = _project_in_batches(adapter, raw_query_embs, "query", max(1, batch_size), device)
    skill_embs = _project_in_batches(adapter, raw_skill_embs, "doc", max(1, batch_size), device)
    skill_ids = [str(skill["skill_id"]) for skill in skills]
    query_ids = [str(query["query_id"]) for query in queries]
    ranked_by_query = _rank_embedding_rows(query_ids, query_embs, skill_embs, skill_ids, top_k)
    run_path = output_dir / "run.tsv"
    predictions_path = output_dir / "predictions.jsonl"
    write_trec_run(run_path, ranked_by_query, run_name=run_name)
    write_jsonl(
        predictions_path,
        (
            {
                "query_id": query_id,
                "ranked_skill_ids": [skill_id for skill_id, _ in ranked],
                "scores": [score for _, score in ranked],
                "split": split,
            }
            for query_id, ranked in ranked_by_query.items()
        ),
    )
    report = {
        "status": "ok",
        "method": "skillrouter_style_finetune",
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "base_model_name": model_name,
        "pooling": resolved_pooling,
        "query_text_mode": resolved_query_text_mode,
        "checkpoint_path": str(checkpoint_path),
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "SKILLRET official test static retrieval; not a SkillsBench closed-loop result.",
    }
    _write_json(output_dir / "run_report.json", report)
    return report
