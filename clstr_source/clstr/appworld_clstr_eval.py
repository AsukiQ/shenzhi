from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from clstr.appworld_eval import compute_retrieval_metrics
from clstr.appworld_routing import read_jsonl, write_json, write_jsonl
from clstr.belief import subspace_obs
from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.device_utils import resolve_device
from clstr.model import CLSTRConfig, CLSTRModel


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        if int(row.get("relevance", 1)) <= 0:
            continue
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["skill_id"]))
    return qrels


def _load_checkpoint_model(
    *,
    model_config_path: str | Path,
    skill_pool_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[CLSTRModel, dict[str, Any]]:
    model_cfg = _load_yaml(model_config_path)
    skills = load_eval_pool(Path(skill_pool_path))
    model = CLSTRModel(CLSTRConfig(**model_cfg), skills)
    device = resolve_device()
    model.to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    report = {
        "checkpoint_path": str(checkpoint_path),
        "model_config_path": str(model_config_path),
        "skill_pool_path": str(skill_pool_path),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "device": str(device),
    }
    if isinstance(payload, dict):
        for key in (
            "stage",
            "step",
            "metrics",
            "train_config",
            "epoch_stats",
            "model_config_path",
            "base_checkpoint_path",
        ):
            if key in payload:
                report[key] = payload[key]
    return model, report


def _standardize_scores(scores: torch.Tensor) -> torch.Tensor:
    values = scores.float()
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1.0e-8)


def rank_clstr_skills_for_query(
    *,
    model: CLSTRModel,
    query: str,
    skill_ids: list[str],
    top_k: int,
    ranking_mode: str = "skill_table",
    candidate_top_k: int | None = None,
    policy_blend_alpha: float = 0.25,
    allow_legacy_policy_skill_router: bool = False,
) -> tuple[list[str], list[float], dict[str, Any]]:
    routing_logits = model.skill_table.retrieval_logits(model.encode_states([query])).squeeze(0)
    if routing_logits.ndim != 1:
        raise ValueError("CLSTR skill routing logits must be rank-1 after squeeze")
    if len(skill_ids) != int(routing_logits.numel()):
        raise ValueError("skill_ids length must match CLSTR skill table size")

    output_k = min(max(1, int(top_k)), int(routing_logits.numel()))
    mode = str(ranking_mode).lower()
    if mode == "skill_table":
        values, indices = torch.topk(routing_logits, k=output_k)
        ranked = [skill_ids[int(idx)] for idx in indices.detach().cpu().tolist()]
        scores = [float(value) for value in values.detach().cpu().tolist()]
        return ranked, scores, {"ranking_mode": mode}

    if mode not in {"policy_head", "policy_blend"}:
        raise ValueError(f"unsupported AppWorld CLSTR ranking_mode: {ranking_mode}")
    if not bool(allow_legacy_policy_skill_router):
        raise ValueError(
            "ranking_mode policy_head/policy_blend is a legacy_policy_skill_router path. "
            "It uses the policy head as a skill-candidate text reranker, which is not the "
            "current CLSTR training/eval contract. Pass allow_legacy_policy_skill_router=True "
            "only to reproduce legacy AppWorld experiments."
        )

    raw_candidate_k = int(candidate_top_k) if candidate_top_k is not None else max(output_k, int(getattr(model, "K", output_k)))
    candidate_k = min(max(1, raw_candidate_k), int(routing_logits.numel()))
    base_values, base_indices = torch.topk(routing_logits, k=candidate_k)
    candidates = [int(idx) for idx in base_indices.detach().cpu().tolist()]

    h_t = model.encode_states([query])
    m_t = subspace_obs(model.skill_table, h_t)
    candidate_embs = model.batch_cross_encode([query], [candidates])
    policy_logits = model.policy_forward(h_t, m_t, candidate_embs, routing_logits=base_values).squeeze(0)[: len(candidates)]
    ranking_scores = policy_logits
    diagnostics = {
        "ranking_mode": mode,
        "legacy_policy_skill_router": True,
        "candidate_top_k": candidate_k,
        "base_candidate_skill_ids": [skill_ids[idx] for idx in candidates],
        "base_candidate_scores": [float(value) for value in base_values.detach().cpu().tolist()],
    }
    if mode == "policy_blend":
        alpha = min(max(float(policy_blend_alpha), 0.0), 1.0)
        ranking_scores = (1.0 - alpha) * _standardize_scores(base_values) + alpha * _standardize_scores(policy_logits)
        diagnostics["policy_blend_alpha"] = alpha

    values, local_indices = torch.topk(ranking_scores, k=min(output_k, len(candidates)))
    ranked_indices = [candidates[int(local_idx)] for local_idx in local_indices.detach().cpu().tolist()]
    ranked = [skill_ids[idx] for idx in ranked_indices]
    scores = [float(value) for value in values.detach().cpu().tolist()]
    return ranked, scores, diagnostics


def run_appworld_clstr_routing_eval(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    tasks_path: str | Path = "data/appworld_routing/dev_tasks.jsonl",
    qrels_path: str | Path = "data/appworld_routing/dev_qrels.jsonl",
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    output_dir: str | Path = "outputs/appworld_clstr_eval",
    top_k: int = 20,
    ranking_mode: str = "skill_table",
    candidate_top_k: int | None = None,
    policy_blend_alpha: float = 0.25,
    allow_legacy_policy_skill_router: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    qrels = _load_qrels(qrels_path)
    model, load_report = _load_checkpoint_model(
        model_config_path=model_config_path,
        skill_pool_path=skill_pool_path,
        checkpoint_path=checkpoint_path,
    )
    skill_ids = [str(getattr(skill, "skill_id", None) or idx) for idx, skill in enumerate(model.skills)]

    predictions: dict[str, list[str]] = {}
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for task in tasks:
            query_id = str(task.get("query_id") or task.get("task_id"))
            query = str(task.get("query") or task.get("instruction_text") or "")
            ranked, scores, diagnostics = rank_clstr_skills_for_query(
                model=model,
                query=query,
                skill_ids=skill_ids,
                top_k=top_k,
                ranking_mode=ranking_mode,
                candidate_top_k=candidate_top_k,
                policy_blend_alpha=policy_blend_alpha,
                allow_legacy_policy_skill_router=allow_legacy_policy_skill_router,
            )
            predictions[query_id] = ranked
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "task_id": task.get("task_id"),
                    "split": task.get("split"),
                    "ranked_skill_ids": ranked,
                    "scores": scores,
                    "ranking": diagnostics,
                }
            )

    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    metrics = compute_retrieval_metrics(predictions, qrels, ks=(1, 5, 10, top_k))
    report = {
        "status": "ok",
        "method": f"clstr_checkpoint_{str(ranking_mode).lower()}",
        "tasks_path": str(tasks_path),
        "qrels_path": str(qrels_path),
        "skill_pool_path": str(skill_pool_path),
        "predictions_path": str(predictions_path),
        "top_k": top_k,
        "ranking_mode": str(ranking_mode).lower(),
        "candidate_top_k": candidate_top_k,
        "policy_blend_alpha": float(policy_blend_alpha),
        "allow_legacy_policy_skill_router": bool(allow_legacy_policy_skill_router),
        "query_count": len(tasks),
        "qrel_query_count": len(qrels),
        "skill_count": len(model.skills),
        "metrics": metrics,
        "checkpoint": load_report,
        "caveat": "Routing-layer AppWorld evaluation against task-to-SkillX positives; not full DB-state AppWorld completion.",
    }
    write_json(output_dir / "report.json", report)
    return report
