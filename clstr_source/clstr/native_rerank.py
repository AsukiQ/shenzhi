from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.skillret import load_skillret_training_rows
from clstr.skillret_official import (
    _select_eval_subset,
    _skillrouter_query_text,
    load_official_qrels,
    load_official_queries,
    load_official_skills,
    write_json,
    write_jsonl,
    write_trec_run,
)


class CLSTRNativeResidualRerankHead(nn.Module):
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
        features = torch.cat([q, doc_embs, q * doc_embs, base_scores.unsqueeze(-1)], dim=-1)
        return self.out(self.hidden(features)).squeeze(-1)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_state_dict(model: CLSTRModel, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }


def _config_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(payload.get("config") or {})
    required = {
        "base_model_name": cfg.get("base_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "d": int(cfg.get("d", 256)),
        "d_a": int(cfg.get("d_a", max(4, int(cfg.get("d", 256)) // 4))),
        "top_k": int(cfg.get("top_k", 50)),
        "encoder_pooling": cfg.get("encoder_pooling", "masked_mean"),
        "cross_encoder_pooling": cfg.get("cross_encoder_pooling", "masked_mean"),
        "tokenizer_padding_side": cfg.get("tokenizer_padding_side"),
        "torch_dtype": cfg.get("torch_dtype"),
        "freeze_backbone": bool(cfg.get("freeze_backbone", False)),
        "max_length": cfg.get("max_length"),
        "projection_init": cfg.get("projection_init", "default"),
        "normalize_embeddings": bool(cfg.get("normalize_embeddings", False)),
        "hf_cache_dir": cfg.get("hf_cache_dir"),
        "local_files_only": bool(cfg.get("local_files_only", False)),
        "defer_skill_table_init": True,
        "skill_text_format": cfg.get("skill_text_format", "skillret_official"),
        "skill_table_batch_size": int(cfg.get("skill_table_batch_size", 1)),
        "skill_table_adapter_init": cfg.get("skill_table_adapter_init", "default"),
        "use_cross_encoder": bool(cfg.get("use_cross_encoder", False)),
        "query_text_format": cfg.get("query_text_format", "skillrouter"),
    }
    return required


def _write_skill_rows(path: Path, skill_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skill_rows),
        encoding="utf-8",
    )


def _build_model_from_checkpoint(
    checkpoint_path: str | Path,
    skill_rows: list[dict[str, Any]],
    skills_path: Path,
) -> tuple[CLSTRModel, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = _config_from_checkpoint(payload)
    _write_skill_rows(skills_path, skill_rows)
    skills = load_eval_pool(skills_path)
    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(config["base_model_name"]),
            d=int(config["d"]),
            d_a=int(config["d_a"]),
            top_k=int(config["top_k"]),
            encoder_pooling=str(config["encoder_pooling"]),
            cross_encoder_pooling=str(config["cross_encoder_pooling"]),
            tokenizer_padding_side=config["tokenizer_padding_side"],
            torch_dtype=config["torch_dtype"],
            freeze_backbone=bool(config["freeze_backbone"]),
            max_length=config["max_length"],
            projection_init=str(config["projection_init"]),
            normalize_embeddings=bool(config["normalize_embeddings"]),
            hf_cache_dir=config.get("hf_cache_dir"),
            local_files_only=bool(config.get("local_files_only", False)),
            defer_skill_table_init=True,
            skill_text_format=str(config["skill_text_format"]),
            skill_table_batch_size=int(config["skill_table_batch_size"]),
            skill_table_adapter_init=str(config["skill_table_adapter_init"]),
            use_cross_encoder=bool(config["use_cross_encoder"]),
        ),
        skills,
    )
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(_filter_state_dict(model, state_dict), strict=False)
    return model, config, payload


def write_clstr_native_rerank_design_report(output_path: str | Path) -> dict[str, Any]:
    report = {
        "status": "ok",
        "role": "CLSTR native routing stack with residual listwise rerank",
        "native_routing_components": {
            "state_query_adapter": "StateEncoder.proj",
            "skill_query_adapter": "SkillTable.W",
            "doc_side_table": "SkillTable.E",
            "native_score": "SkillTable.W(h) @ SkillTable.E.T",
        },
        "qdoc_adapter_policy": {
            "enters_mainline": False,
            "reason": "q/doc adapter overlaps with StateEncoder.proj, SkillTable.W, and SkillTable.E responsibilities.",
            "role": "SKILLRET static ablation only",
        },
        "native_rerank": {
            "candidate_source": "CLSTRModel native skill_table.logits topK",
            "score_fusion": "native_retriever_score_over_temperature_plus_zero_initialized_delta",
            "qdoc_adapter_used": False,
        },
        "loss_taxonomy": {
            "native_residual_listwise_rerank": "L_retr",
            "retrieval_infonce": "L_retr",
            "stop_bce_proxy": "auxiliary proxy for STOP policy action, not a new core loss",
            "auxiliary_imitation_proxy": "L_act pretraining proxy",
            "transition_proxy": "L_trans",
        },
        "paper_positioning": "Listwise rerank is a routing-foundation objective under L_retr, not CLSTR closed-loop novelty.",
    }
    write_json(output_path, report)
    return report


def score_native_residual_rerank_candidates(
    reranker: CLSTRNativeResidualRerankHead,
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
    docs = skill_embs[candidate_indices.detach().cpu()].to(device)
    base = base_scores.to(device)
    delta = reranker(q, docs, base)
    return (base / base_score_temperature) + residual_weight * delta


def _query_text(query: dict[str, Any], query_text_format: str) -> str:
    if query_text_format == "skillrouter":
        return _skillrouter_query_text(query)
    if query_text_format == "raw":
        return str(query.get("query", ""))
    raise ValueError(f"unsupported query_text_format: {query_text_format}")


def _usable_train_queries(
    query_rows: list[dict[str, Any]],
    positives_by_query: dict[str, list[str]],
    skill_id_to_idx: dict[str, int],
    max_queries: int | None,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for query in query_rows:
        qid = str(query["query_id"])
        positive_indices = [
            skill_id_to_idx[skill_id]
            for skill_id in positives_by_query.get(qid, query.get("positive_skill_ids", []))
            if skill_id in skill_id_to_idx
        ]
        if positive_indices:
            row = dict(query)
            row["positive_indices"] = positive_indices
            queries.append(row)
        if max_queries is not None and len(queries) >= max_queries:
            break
    return queries


def run_clstr_native_rerank_warmup(
    data_root: str | Path,
    base_checkpoint_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 1,
    batch_size: int = 2,
    candidate_k: int = 50,
    max_skills: int | None = None,
    max_queries: int | None = None,
    learning_rate: float = 1.0e-4,
    seed: int = 23,
    base_score_temperature: float = 0.05,
    residual_weight: float = 1.0,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    skill_rows, query_rows, positives_by_query = load_skillret_training_rows(data_root, split="train")
    if max_skills is not None:
        skill_rows = skill_rows[:max_skills]
    if not skill_rows:
        raise ValueError(f"no SKILLRET train skills found under {data_root}")
    skill_ids = [str(row["skill_id"]) for row in skill_rows]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    queries = _usable_train_queries(query_rows, positives_by_query, skill_id_to_idx, max_queries)
    if not queries:
        raise ValueError("no SKILLRET train queries have positives inside the selected skill pool")

    model, config, _payload = _build_model_from_checkpoint(
        base_checkpoint_path,
        skill_rows,
        output_dir / "selected_train_skills.jsonl",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.rebuild_skill_table()
    for param in model.parameters():
        param.requires_grad_(False)
    reranker = CLSTRNativeResidualRerankHead(int(config["d"])).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=learning_rate)
    query_text_format = str(config.get("query_text_format", "skillrouter"))
    metrics: dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "candidate_count": 0.0}

    for step in range(1, max_steps + 1):
        batch = [queries[(step - 1 + offset) % len(queries)] for offset in range(batch_size)]
        with torch.no_grad():
            h = model.encode_states([_query_text(row, query_text_format) for row in batch])
            native_query = model.skill_table.W(h)
            logits = native_query @ model.skill_table.E.t()
        candidate_rows: list[list[int]] = []
        used_offsets: list[int] = []
        for offset, (query, logit_row) in enumerate(zip(batch, logits)):
            positive_indices = [int(idx) for idx in query["positive_indices"]]
            positive = positive_indices[0]
            positive_set = set(positive_indices)
            top_count = min(len(skill_ids), candidate_k + len(positive_set) + 32)
            ranked = torch.topk(logit_row, k=top_count).indices.tolist()
            negatives = [idx for idx in ranked if idx not in positive_set]
            row = [positive] + negatives[: max(candidate_k - 1, 0)]
            if len(row) >= 2:
                candidate_rows.append(row[:candidate_k])
                used_offsets.append(offset)
        if not candidate_rows:
            continue
        row_width = min(len(row) for row in candidate_rows)
        candidate_rows = [row[:row_width] for row in candidate_rows]
        candidate_tensor = torch.tensor(candidate_rows, device=device, dtype=torch.long)
        base_scores = logits[used_offsets].gather(1, candidate_tensor)
        scores = score_native_residual_rerank_candidates(
            reranker,
            native_query[used_offsets],
            model.skill_table.E,
            candidate_tensor,
            base_scores,
            device,
            base_score_temperature=base_score_temperature,
            residual_weight=residual_weight,
        )
        labels = torch.zeros(scores.size(0), device=device, dtype=torch.long)
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "accuracy": float((scores.detach().argmax(dim=-1) == labels).float().mean().cpu().item()),
            "candidate_count": float(row_width),
        }

    checkpoint_path = checkpoint_dir / f"clstr_native_rerank-step{max_steps}.pt"
    rerank_config = {
        "base_checkpoint_path": str(base_checkpoint_path),
        "base_checkpoint_sha256": _sha256(base_checkpoint_path),
        "d": int(config["d"]),
        "candidate_k": candidate_k,
        "training_objective": "L_retr_native_residual_listwise",
        "loss_taxonomy": "L_retr",
        "reranker_type": "native_residual_listwise_head",
        "base_score_temperature": base_score_temperature,
        "residual_weight": residual_weight,
        "qdoc_adapter_used": False,
        "base_clstr_config": config,
    }
    torch.save(
        {
            "method": "clstr_native_rerank",
            "source_note": "CLSTR-native residual listwise rerank; belongs to L_retr routing foundation.",
            "step": max_steps,
            "config": rerank_config,
            "reranker_state_dict": reranker.state_dict(),
            "metrics": metrics,
            "skill_count": len(skill_rows),
            "query_count": len(queries),
        },
        checkpoint_path,
    )
    report = {
        "status": "ok",
        "method": "clstr_native_rerank",
        "training_objective": "L_retr_native_residual_listwise",
        "loss_taxonomy": {"belongs_to": "L_retr", "is_new_core_loss": False},
        "base_checkpoint": str(base_checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "steps": max_steps,
        "batch_size": batch_size,
        "candidate_k": candidate_k,
        "skill_count": len(skill_rows),
        "query_count": len(queries),
        "metrics": metrics,
        "uses_skillret_test_qrels": False,
        "uses_skillrouter_eval_labels": False,
        "qdoc_adapter_used": False,
        "model_config": rerank_config,
    }
    write_json(output_dir / "train_report.json", report)
    return report


def export_clstr_native_rerank_official_run(
    data_root: str | Path,
    base_checkpoint_path: str | Path,
    rerank_checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    batch_size: int = 1,
    max_queries: int | None = None,
    max_skills: int | None = None,
    run_name: str = "clstr_native_rerank_full",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    model, config, _payload = _build_model_from_checkpoint(
        base_checkpoint_path,
        skills,
        output_dir / "selected_eval_skills.jsonl",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.rebuild_skill_table()
    payload = torch.load(rerank_checkpoint_path, map_location="cpu")
    rerank_config = dict(payload.get("config") or {})
    reranker = CLSTRNativeResidualRerankHead(int(rerank_config.get("d", config["d"]))).to(device)
    reranker.load_state_dict(payload["reranker_state_dict"])
    reranker.eval()
    query_text_format = str(config.get("query_text_format", "skillrouter"))
    skill_ids = [str(skill["skill_id"]) for skill in skills]
    ranked_by_query: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(queries), max(1, batch_size)):
        batch = queries[start : start + max(1, batch_size)]
        query_ids = [str(row["query_id"]) for row in batch]
        with torch.no_grad():
            h = model.encode_states([_query_text(row, query_text_format) for row in batch])
            native_query = model.skill_table.W(h)
            logits = native_query @ model.skill_table.E.t()
            values, indices = torch.topk(logits, k=min(top_k, len(skill_ids)), dim=1)
            scores = score_native_residual_rerank_candidates(
                reranker,
                native_query,
                model.skill_table.E,
                indices,
                values.cpu(),
                device,
                base_score_temperature=float(rerank_config.get("base_score_temperature", 0.05)),
                residual_weight=float(rerank_config.get("residual_weight", 1.0)),
            ).cpu()
        for qid, row_indices, row_scores in zip(query_ids, indices.cpu(), scores):
            pairs = [
                (skill_ids[idx], float(score))
                for idx, score in zip(row_indices.tolist(), row_scores.tolist())
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
        "method": "clstr_native_rerank",
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "base_checkpoint_path": str(base_checkpoint_path),
        "rerank_checkpoint_path": str(rerank_checkpoint_path),
        "qdoc_adapter_used": False,
        "loss_taxonomy": "L_retr",
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "SKILLRET official test static routing/rerank; not a SkillsBench closed-loop result.",
    }
    write_json(output_dir / "run_report.json", report)
    return report


def build_clstr_native_rerank_decision_report(
    base_metrics: dict[str, Any],
    native_metrics: dict[str, Any],
    qdoc_metrics: dict[str, Any],
    output_path: str | Path,
    tolerance: float = 0.001,
) -> dict[str, Any]:
    base_ndcg = float(base_metrics.get("NDCG@10", 0.0))
    native_ndcg = float(native_metrics.get("NDCG@10", 0.0))
    adopted = native_ndcg >= base_ndcg - tolerance
    report = {
        "status": "ok",
        "native_rerank_adopted": adopted,
        "decision_rule": "adopt if native NDCG@10 is not clearly below base CLSTR routing",
        "tolerance": tolerance,
        "base_clstr_metrics": {key: base_metrics.get(key) for key in ["NDCG@10", "Recall@10", "MAP@10"]},
        "native_rerank_metrics": {key: native_metrics.get(key) for key in ["NDCG@10", "Recall@10", "MAP@10"]},
        "qdoc_rerank_metrics": {key: qdoc_metrics.get(key) for key in ["NDCG@10", "Recall@10", "MAP@10"]},
        "delta_native_vs_base": {
            key: round(float(native_metrics.get(key, 0.0)) - float(base_metrics.get(key, 0.0)), 5)
            for key in ["NDCG@10", "Recall@10", "MAP@10"]
        },
        "qdoc_rerank_role": "static_ablation_not_mainline_init",
        "qdoc_adapter_used_for_mainline": False,
        "loss_taxonomy": {"listwise_rerank": "L_retr", "is_new_core_loss": False},
        "downstream_recommendation": (
            "use_native_rerank_sidecar_for_auxiliary_trajectory_pretrain"
            if adopted
            else "use_base_clstr_retrieval_checkpoint_only"
        ),
    }
    write_json(output_path, report)
    return report


def build_native_routing_init_manifest(
    base_clstr_checkpoint: str | Path,
    native_rerank_checkpoint: str | Path,
    native_rerank_metrics: dict[str, Any],
    decision_report: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    adopted = bool(decision_report.get("native_rerank_adopted"))
    manifest = {
        "status": "ok",
        "artifact_role": "CLSTR native routing init for auxiliary trajectory pretrain",
        "base_clstr_checkpoint": str(base_clstr_checkpoint),
        "base_clstr_checkpoint_sha256": _sha256(base_clstr_checkpoint),
        "native_rerank_adopted": adopted,
        "native_rerank_checkpoint": str(native_rerank_checkpoint) if adopted else None,
        "native_rerank_checkpoint_sha256": _sha256(native_rerank_checkpoint) if adopted else None,
        "downstream_init_checkpoint": str(native_rerank_checkpoint if adopted else base_clstr_checkpoint),
        "downstream_init_for": "auxiliary_trajectory_pretrain",
        "qdoc_adapter_used": False,
        "qdoc_rerank_role": "static_ablation_not_mainline_init",
        "loss_taxonomy": {"listwise_rerank": "L_retr", "is_new_core_loss": False},
        "native_rerank_metrics": native_rerank_metrics,
        "decision_report": decision_report,
    }
    write_json(output_path, manifest)
    return manifest
