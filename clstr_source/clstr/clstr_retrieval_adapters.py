from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.skillret import load_skillret_training_rows
from clstr.skillret_official import (
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
from clstr.skillrouter_style import _encode_raw_texts, _load_backbone


class CLSTRQDocAdapter(nn.Module):
    def __init__(self, hidden_size: int, d: int, projection_init: str = "identity"):
        super().__init__()
        self.q_adapter = nn.Linear(hidden_size, d, bias=False)
        self.d_adapter = nn.Linear(hidden_size, d, bias=False)
        if projection_init == "identity":
            if hidden_size != d:
                raise ValueError(
                    "identity projection_init requires model_dim to match hidden size "
                    f"({d} != {hidden_size})"
                )
            with torch.no_grad():
                eye = torch.eye(d)
                self.q_adapter.weight.copy_(eye)
                self.d_adapter.weight.copy_(eye)
        elif projection_init != "default":
            raise ValueError(f"unsupported projection_init: {projection_init}")

    def encode_queries(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_adapter(raw), p=2, dim=-1)

    def encode_docs(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.d_adapter(raw), p=2, dim=-1)


class CLSTRListwiseRerankHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        q = query_embs.unsqueeze(1).expand(-1, doc_embs.size(1), -1)
        features = torch.cat([q, doc_embs, q * doc_embs], dim=-1)
        return self.net(features).squeeze(-1)


class CLSTRResidualListwiseRerankHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(d * 3 + 1, d),
            nn.GELU(),
        )
        self.out = nn.Linear(d, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> torch.Tensor:
        q = query_embs.unsqueeze(1).expand(-1, doc_embs.size(1), -1)
        base_feature = base_scores.unsqueeze(-1)
        features = torch.cat([q, doc_embs, q * doc_embs, base_feature], dim=-1)
        return self.out(self.hidden(features)).squeeze(-1)


def write_clstr_qdoc_rerank_design_report(output_path: str | Path) -> dict[str, Any]:
    report = {
        "status": "ok",
        "role": "SKILLRET static routing foundation alignment",
        "not_core_claim": "q/doc adapters and listwise rerank are routing foundation components, not CLSTR closed-loop novelty.",
        "current_clstr_static_retrieval": {
            "has_full_pool_retrieval_loss": True,
            "missing_symmetric_qdoc_adapter": True,
            "official_eval_uses_rerank": False,
            "official_eval_path": "CLSTR official export ranks with skill_table.logits only.",
        },
        "planned_alignment": {
            "qdoc": "frozen SkillRouter backbone, last-token pooling, left padding, q_adapter/doc_adapter, full-pool InfoNCE.",
            "rerank": "residual listwise head over q/doc retriever topK candidates; final score preserves retriever score and adds a learned reranker delta.",
        },
        "rerank_root_cause_guard": {
            "risk": "a standalone embedding-pair MLP can replace and degrade a strong first-stage retriever ranking",
            "mitigation": "zero-initialized residual delta and base retriever score fusion",
        },
        "data_boundary": {
            "train": "SKILLRET train split only",
            "eval": "official SKILLRET test split with qrels used only for metrics",
            "not_used": [
                "SkillRouter benchmark eval labels",
                "SkillsBench held-out trajectories",
                "SkillsBench clean_router_data",
            ],
        },
    }
    write_json(output_path, report)
    return report


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
            copied["positive_skill_ids_in_pool"] = [
                str(skill_id)
                for skill_id in positives_by_query.get(qid, query.get("positive_skill_ids", []))
                if skill_id in skill_id_to_idx
            ]
            usable_queries.append(copied)
        if max_queries is not None and len(usable_queries) >= max_queries:
            break
    if not skill_rows:
        raise ValueError(f"no SKILLRET train skills found under {data_root}")
    if not usable_queries:
        raise ValueError("no SKILLRET train queries have positives inside the selected skill pool")
    return skill_rows, usable_queries, positives_by_query, len(query_rows)


def _load_qdoc_checkpoint(
    checkpoint_path: str | Path,
    fallback_base_model_name: str,
    fallback_model_dim: int = 1024,
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config", {})
    base_model_name = str(config.get("base_model_name", fallback_base_model_name))
    model_dim = int(config.get("d", fallback_model_dim))
    torch_dtype = config.get("torch_dtype", "bfloat16")
    padding_side = str(config.get("tokenizer_padding_side", "left"))
    max_length = int(config.get("max_length", 32768))
    return payload, config, base_model_name, model_dim, torch_dtype, padding_side, max_length


def _load_qdoc_adapter_for_checkpoint(
    checkpoint_path: str | Path,
    base_model_name: str,
    model_dim: int | None = None,
):
    payload, config, model_name, d, torch_dtype, padding_side, max_length = _load_qdoc_checkpoint(
        checkpoint_path,
        fallback_base_model_name=base_model_name,
        fallback_model_dim=model_dim or 1024,
    )
    model, tokenizer, device = _load_backbone(model_name, torch_dtype, padding_side)
    adapter = CLSTRQDocAdapter(
        hidden_size=int(model.config.hidden_size),
        d=int(config.get("d", d)),
        projection_init="default",
    ).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()
    return model, tokenizer, adapter, device, model_name, int(config.get("d", d)), max_length


def _project_qdoc_in_batches(
    adapter: CLSTRQDocAdapter,
    raw: torch.Tensor,
    kind: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, raw.size(0), batch_size):
        batch = raw[start : start + batch_size].to(device)
        with torch.no_grad():
            if kind == "query":
                projected = adapter.encode_queries(batch)
            elif kind == "doc":
                projected = adapter.encode_docs(batch)
            else:
                raise ValueError(kind)
        out.append(projected.cpu())
    return torch.cat(out, dim=0)


def run_clstr_qdoc_warmup(
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
    model, tokenizer, device = _load_backbone(base_model_name, torch_dtype, tokenizer_padding_side)
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_skill_text(skill) for skill in skill_rows],
        skill_batch_size,
        max_length,
    ).to(device)
    adapter = CLSTRQDocAdapter(
        hidden_size=int(model.config.hidden_size),
        d=model_dim,
        projection_init=projection_init,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    encode_query_batch_size = query_batch_size or batch_size
    metrics: dict[str, float] = {"loss": 0.0, "recall_at_1": 0.0, "recall_at_50": 0.0}
    for step in range(1, max_steps + 1):
        batch = [queries[(step - 1 + offset) % len(queries)] for offset in range(batch_size)]
        labels = torch.tensor([row["positive_indices"][0] for row in batch], device=device, dtype=torch.long)
        raw_query_embs = _encode_raw_texts(
            model,
            tokenizer,
            [_skillrouter_query_text(row) for row in batch],
            encode_query_batch_size,
            max_length,
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

    checkpoint_path = checkpoint_dir / f"clstr_qdoc-step{max_steps}.pt"
    config = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "max_length": max_length,
        "torch_dtype": torch_dtype,
        "tokenizer_padding_side": tokenizer_padding_side,
        "projection_init": projection_init,
        "temperature": temperature,
        "pooling": "last_token",
        "training_objective": "full_pool_infonce",
        "backbone": "frozen",
        "query_template": "skillrouter",
        "doc_template": "skillret_official_full_text",
    }
    torch.save(
        {
            "method": "clstr_qdoc",
            "source_note": "CLSTR q/doc adapter retrieval foundation; not CLSTR closed-loop evidence.",
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
        "method": "clstr_qdoc",
        "training_objective": "full_pool_infonce",
        "routing_foundation_role": "static_retrieval_not_closed_loop_core",
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
        "uses_skillret_test_qrels": False,
        "uses_skillrouter_eval_labels": False,
        "uses_skillsbench_trajectories": False,
    }
    write_json(output_dir / "train_report.json", report)
    return report


def export_clstr_qdoc_official_run(
    data_root: str | Path,
    base_model_name: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    batch_size: int = 1,
    max_length: int | None = None,
    max_queries: int | None = None,
    max_skills: int | None = None,
    run_name: str = "clstr_qdoc_full",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    model, tokenizer, adapter, device, model_name, _model_dim, ckpt_max_length = _load_qdoc_adapter_for_checkpoint(
        checkpoint_path,
        base_model_name,
    )
    effective_max_length = max_length or ckpt_max_length
    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    raw_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_query_text(query) for query in queries],
        batch_size,
        effective_max_length,
    )
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_skill_text(skill) for skill in skills],
        batch_size,
        effective_max_length,
    )
    query_embs = _project_qdoc_in_batches(adapter, raw_query_embs, "query", max(1, batch_size), device)
    skill_embs = _project_qdoc_in_batches(adapter, raw_skill_embs, "doc", max(1, batch_size), device)
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
        "method": "clstr_qdoc",
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "base_model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "SKILLRET official test static retrieval; not a SkillsBench closed-loop result.",
    }
    write_json(output_dir / "run_report.json", report)
    return report


def build_listwise_rerank_candidates(
    query_ids: list[str],
    ranked_by_query: dict[str, list[tuple[str, float]]],
    positives_by_query: dict[str, list[str]],
    candidate_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        positives = [str(item) for item in positives_by_query.get(str(query_id), [])]
        if not positives:
            continue
        positive = positives[0]
        positive_set = set(positives)
        negatives = [
            str(skill_id)
            for skill_id, _score in ranked_by_query.get(str(query_id), [])
            if str(skill_id) not in positive_set
        ]
        candidate_ids = [positive] + negatives[: max(candidate_k - 1, 0)]
        if len(candidate_ids) < 2:
            continue
        rows.append(
            {
                "query_id": str(query_id),
                "positive_skill_id": positive,
                "all_positive_skill_ids": positives,
                "candidate_skill_ids": candidate_ids[:candidate_k],
            }
        )
    return rows


def _score_rerank_candidates(
    reranker: CLSTRListwiseRerankHead,
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    candidate_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    q = query_embs.to(device)
    candidate_doc_embs = skill_embs[candidate_indices.detach().cpu()].to(device)
    return reranker(q, candidate_doc_embs)


def _score_residual_rerank_candidates(
    reranker: CLSTRResidualListwiseRerankHead,
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    candidate_indices: torch.Tensor,
    base_scores: torch.Tensor,
    device: torch.device,
    base_score_temperature: float = 0.05,
    residual_weight: float = 1.0,
) -> torch.Tensor:
    if base_score_temperature <= 0:
        raise ValueError("base_score_temperature must be positive")
    q = query_embs.to(device)
    candidate_doc_embs = skill_embs[candidate_indices.detach().cpu()].to(device)
    base = base_scores.to(device)
    delta = reranker(q, candidate_doc_embs, base)
    return (base / base_score_temperature) + (residual_weight * delta)


def run_clstr_qdoc_rerank_warmup(
    data_root: str | Path,
    base_model_name: str,
    qdoc_checkpoint_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 2,
    model_dim: int = 1024,
    max_skills: int | None = None,
    max_queries: int | None = None,
    candidate_k: int = 50,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    max_length: int | None = None,
    encode_batch_size: int = 1,
    reranker_mode: str = "residual",
    base_score_temperature: float = 0.05,
    residual_weight: float = 1.0,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    skill_rows, queries, positives_by_query, query_count_before_filter = _load_training_rows(
        data_root,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    model, tokenizer, adapter, device, model_name, d, ckpt_max_length = _load_qdoc_adapter_for_checkpoint(
        qdoc_checkpoint_path,
        base_model_name,
        model_dim,
    )
    effective_max_length = max_length or ckpt_max_length
    skill_ids = [str(row["skill_id"]) for row in skill_rows]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_skill_text(skill) for skill in skill_rows],
        encode_batch_size,
        effective_max_length,
    )
    skill_embs = _project_qdoc_in_batches(adapter, raw_skill_embs, "doc", max(1, encode_batch_size), device)
    if reranker_mode == "legacy_mlp":
        reranker: nn.Module = CLSTRListwiseRerankHead(d).to(device)
        reranker_type = "embedding_pair_mlp_head"
    elif reranker_mode == "residual":
        reranker = CLSTRResidualListwiseRerankHead(d).to(device)
        reranker_type = "residual_embedding_pair_mlp_head"
    else:
        raise ValueError(f"unsupported reranker_mode: {reranker_mode}")
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=learning_rate)
    metrics = {"loss": 0.0, "accuracy": 0.0, "candidate_count": 0.0}
    for step in range(1, max_steps + 1):
        batch_queries = [queries[(step - 1 + offset) % len(queries)] for offset in range(batch_size)]
        raw_query_embs = _encode_raw_texts(
            model,
            tokenizer,
            [_skillrouter_query_text(row) for row in batch_queries],
            max(1, encode_batch_size),
            effective_max_length,
        )
        query_embs = _project_qdoc_in_batches(adapter, raw_query_embs, "query", max(1, encode_batch_size), device)
        sims = query_embs @ skill_embs.T
        candidate_rows: list[list[int]] = []
        used_query_offsets: list[int] = []
        for query_offset, (query, sim_row) in enumerate(zip(batch_queries, sims)):
            qid = str(query["query_id"])
            positive_ids = [str(item) for item in positives_by_query.get(qid, query.get("positive_skill_ids_in_pool", []))]
            positive_indices = [skill_id_to_idx[item] for item in positive_ids if item in skill_id_to_idx]
            if not positive_indices:
                continue
            positive_idx = positive_indices[0]
            positive_set = set(positive_indices)
            top_count = min(len(skill_ids), candidate_k + len(positive_set) + 32)
            ranked = torch.topk(sim_row, k=top_count).indices.tolist()
            negatives = [idx for idx in ranked if idx not in positive_set]
            row = [positive_idx] + negatives[: max(candidate_k - 1, 0)]
            if len(row) >= 2:
                candidate_rows.append(row[:candidate_k])
                used_query_offsets.append(query_offset)
        if not candidate_rows:
            continue
        row_width = min(len(row) for row in candidate_rows)
        candidate_rows = [row[:row_width] for row in candidate_rows]
        usable_query_embs = query_embs[used_query_offsets]
        candidate_tensor = torch.tensor(candidate_rows, device=device, dtype=torch.long)
        base_scores = sims[used_query_offsets].gather(1, candidate_tensor.detach().cpu())
        if reranker_mode == "legacy_mlp":
            logits = _score_rerank_candidates(
                reranker,  # type: ignore[arg-type]
                usable_query_embs,
                skill_embs,
                candidate_tensor,
                device,
            )
        else:
            logits = _score_residual_rerank_candidates(
                reranker,  # type: ignore[arg-type]
                usable_query_embs,
                skill_embs,
                candidate_tensor,
                base_scores,
                device,
                base_score_temperature=base_score_temperature,
                residual_weight=residual_weight,
            )
        labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "accuracy": float((logits.detach().argmax(dim=-1) == labels).float().mean().cpu().item()),
            "candidate_count": float(row_width),
        }

    checkpoint_path = checkpoint_dir / f"clstr_qdoc_rerank-step{max_steps}.pt"
    config = {
        "base_model_name": model_name,
        "qdoc_checkpoint_path": str(qdoc_checkpoint_path),
        "d": d,
        "max_length": effective_max_length,
        "candidate_k": candidate_k,
        "training_objective": (
            "residual_listwise_cross_entropy_over_retriever_topk"
            if reranker_mode == "residual"
            else "listwise_cross_entropy_over_retriever_topk"
        ),
        "reranker_mode": reranker_mode,
        "reranker_type": reranker_type,
        "base_score_temperature": base_score_temperature,
        "residual_weight": residual_weight,
        "backbone": "frozen",
    }
    torch.save(
        {
            "method": "clstr_qdoc_rerank",
            "source_note": "CLSTR q/doc listwise rerank routing foundation; not CLSTR closed-loop evidence.",
            "step": max_steps,
            "config": config,
            "reranker_state_dict": reranker.state_dict(),
            "metrics": metrics,
            "skill_count": len(skill_rows),
            "query_count": len(queries),
        },
        checkpoint_path,
    )
    report = {
        "status": "ok",
        "method": "clstr_qdoc_rerank",
        "training_objective": config["training_objective"],
        "false_negative_filtering": "exclude_all_query_positive_skill_ids_from_negatives",
        "routing_foundation_role": "static_rerank_not_closed_loop_core",
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "qdoc_checkpoint": str(qdoc_checkpoint_path),
        "steps": max_steps,
        "batch_size": batch_size,
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "query_count_before_filter": query_count_before_filter,
        "candidate_k": candidate_k,
        "metrics": metrics,
        "reranker_mode": reranker_mode,
        "base_score_temperature": base_score_temperature,
        "residual_weight": residual_weight,
        "uses_skillret_test_qrels": False,
        "uses_skillrouter_eval_labels": False,
        "uses_skillsbench_trajectories": False,
        "model_config": config,
    }
    write_json(output_dir / "train_report.json", report)
    return report


def export_clstr_qdoc_rerank_official_run(
    data_root: str | Path,
    base_model_name: str,
    qdoc_checkpoint_path: str | Path,
    rerank_checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    batch_size: int = 1,
    max_length: int | None = None,
    max_queries: int | None = None,
    max_skills: int | None = None,
    run_name: str = "clstr_qdoc_rerank_full",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    model, tokenizer, adapter, device, model_name, d, ckpt_max_length = _load_qdoc_adapter_for_checkpoint(
        qdoc_checkpoint_path,
        base_model_name,
    )
    effective_max_length = max_length or ckpt_max_length
    payload = torch.load(rerank_checkpoint_path, map_location="cpu")
    rerank_config = payload.get("config", {})
    reranker_mode = str(rerank_config.get("reranker_mode", "legacy_mlp"))
    reranker_type = str(rerank_config.get("reranker_type", "embedding_pair_mlp_head"))
    if reranker_mode == "residual" or reranker_type == "residual_embedding_pair_mlp_head":
        reranker: nn.Module = CLSTRResidualListwiseRerankHead(d).to(device)
    else:
        reranker = CLSTRListwiseRerankHead(d).to(device)
    reranker.load_state_dict(payload["reranker_state_dict"])
    reranker.eval()
    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    raw_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_query_text(query) for query in queries],
        batch_size,
        effective_max_length,
    )
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        [_skillrouter_skill_text(skill) for skill in skills],
        batch_size,
        effective_max_length,
    )
    query_embs = _project_qdoc_in_batches(adapter, raw_query_embs, "query", max(1, batch_size), device)
    skill_embs = _project_qdoc_in_batches(adapter, raw_skill_embs, "doc", max(1, batch_size), device)
    skill_ids = [str(skill["skill_id"]) for skill in skills]
    query_ids = [str(query["query_id"]) for query in queries]
    retrieved = _rank_embedding_rows(query_ids, query_embs, skill_embs, skill_ids, top_k)
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    ranked_by_query: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(query_ids), max(1, batch_size)):
        batch_query_ids = query_ids[start : start + max(1, batch_size)]
        batch_query_embs = query_embs[start : start + len(batch_query_ids)]
        candidate_rows = [
            [skill_id_to_idx[skill_id] for skill_id, _score in retrieved[qid]]
            for qid in batch_query_ids
        ]
        base_score_rows = [
            [float(score) for _skill_id, score in retrieved[qid]]
            for qid in batch_query_ids
        ]
        candidate_tensor = torch.tensor(candidate_rows, device=device, dtype=torch.long)
        base_scores = torch.tensor(base_score_rows, dtype=torch.float32)
        with torch.no_grad():
            if reranker_mode == "residual" or reranker_type == "residual_embedding_pair_mlp_head":
                scores = _score_residual_rerank_candidates(
                    reranker,  # type: ignore[arg-type]
                    batch_query_embs,
                    skill_embs,
                    candidate_tensor,
                    base_scores,
                    device,
                    base_score_temperature=float(rerank_config.get("base_score_temperature", 0.05)),
                    residual_weight=float(rerank_config.get("residual_weight", 1.0)),
                ).cpu()
            else:
                scores = _score_rerank_candidates(
                    reranker,  # type: ignore[arg-type]
                    batch_query_embs,
                    skill_embs,
                    candidate_tensor,
                    device,
                ).cpu()
        for qid, row_indices, row_scores in zip(batch_query_ids, candidate_rows, scores):
            pairs = [
                (skill_ids[idx], float(score))
                for idx, score in zip(row_indices, row_scores.tolist())
            ]
            ranked_by_query[qid] = sorted(pairs, key=lambda item: item[1], reverse=True)
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
        "method": "clstr_qdoc_rerank",
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "base_model_name": model_name,
        "qdoc_checkpoint_path": str(qdoc_checkpoint_path),
        "rerank_checkpoint_path": str(rerank_checkpoint_path),
        "reranker_mode": reranker_mode,
        "reranker_type": reranker_type,
        "score_fusion": (
            "base_retriever_score_over_temperature_plus_reranker_delta"
            if reranker_mode == "residual" or reranker_type == "residual_embedding_pair_mlp_head"
            else "reranker_score_replaces_retriever_score"
        ),
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "SKILLRET official test static rerank; not a SkillsBench closed-loop result.",
    }
    write_json(output_dir / "run_report.json", report)
    return report
