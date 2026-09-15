from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from clstr.appworld_eval import compute_retrieval_metrics
from clstr.appworld_routing import read_jsonl, write_json, write_jsonl


def skillrouter_base_query_text(task: dict[str, Any]) -> str:
    query = str(task.get("query") or task.get("instruction_text") or "")
    return (
        "Instruct: Given a task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery:"
        f"{query[:1500]}"
    )


def skillrouter_base_skill_text(skill: dict[str, Any]) -> str:
    name = str(skill.get("name", "")).strip()
    desc = str(skill.get("description", "")).strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    executor = str(skill.get("executor_desc", "")).strip()
    return f"{name} | {desc} | {executor} | {body}"


class SkillRouterBaseScorer(torch.nn.Module):
    """Small trainable scorer over frozen SkillRouter-style embeddings."""

    def __init__(self, dim: int, skill_count: int) -> None:
        super().__init__()
        self.query_proj = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.query_proj.weight)
        self.skill_bias = torch.nn.Parameter(torch.zeros(skill_count))
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def score(self, query_embs: torch.Tensor, skill_embs: torch.Tensor) -> torch.Tensor:
        projected = F.normalize(self.query_proj(query_embs.float()), p=2, dim=-1)
        skills = F.normalize(skill_embs.float(), p=2, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (projected @ skills.t()) + self.skill_bias.unsqueeze(0)


def multi_positive_nll(logits: torch.Tensor, positive_indices: Sequence[Sequence[int]]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row_idx, positives in enumerate(positive_indices):
        valid = sorted({int(idx) for idx in positives if 0 <= int(idx) < logits.size(1)})
        if not valid:
            continue
        row = logits[row_idx]
        pos = torch.tensor(valid, dtype=torch.long, device=logits.device)
        losses.append(torch.logsumexp(row, dim=0) - torch.logsumexp(row.index_select(0, pos), dim=0))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        if int(row.get("relevance", 1)) <= 0:
            continue
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["skill_id"]))
    return qrels


def _positive_indices(
    query_ids: Sequence[str],
    skill_ids: Sequence[str],
    qrels: dict[str, set[str]],
) -> tuple[list[int], list[list[int]]]:
    skill_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    selected_query_indices: list[int] = []
    positives: list[list[int]] = []
    for query_idx, query_id in enumerate(query_ids):
        row = [skill_to_idx[skill_id] for skill_id in sorted(qrels.get(query_id, set())) if skill_id in skill_to_idx]
        if row:
            selected_query_indices.append(query_idx)
            positives.append(row)
    return selected_query_indices, positives


def _rank_with_scores(
    *,
    query_ids: Sequence[str],
    skill_ids: Sequence[str],
    scores: torch.Tensor,
    top_k: int,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    k = min(int(top_k), len(skill_ids))
    values, indices = torch.topk(scores, k=k, dim=1)
    predictions: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for row_idx, query_id in enumerate(query_ids):
        ranked = [str(skill_ids[int(idx)]) for idx in indices[row_idx].detach().cpu().tolist()]
        row_scores = [float(value) for value in values[row_idx].detach().cpu().tolist()]
        predictions[str(query_id)] = ranked
        rows.append({"query_id": str(query_id), "ranked_skill_ids": ranked, "scores": row_scores})
    return predictions, rows


def save_skillrouter_base_checkpoint(
    path: str | Path,
    *,
    model: SkillRouterBaseScorer,
    skill_ids: Sequence[str],
    report: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dim": int(model.query_proj.in_features),
            "skill_count": int(len(skill_ids)),
            "skill_ids": list(skill_ids),
            "report": dict(report),
        },
        path,
    )


def load_skillrouter_base_checkpoint(path: str | Path) -> tuple[SkillRouterBaseScorer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model = SkillRouterBaseScorer(dim=int(payload["dim"]), skill_count=int(payload["skill_count"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    return model, metadata


def train_skillrouter_base_from_embeddings(
    *,
    query_ids: Sequence[str],
    skill_ids: Sequence[str],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    qrels: dict[str, set[str]],
    output_dir: str | Path,
    epochs: int = 100,
    learning_rate: float = 0.05,
    weight_decay: float = 0.0,
    seed: int = 13,
    top_k: int = 20,
    device: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    torch.manual_seed(int(seed))
    selected_query_indices, positives = _positive_indices(query_ids, skill_ids, qrels)
    if not selected_query_indices:
        report = {
            "status": "blocked",
            "method": "skillrouter_base_trainable_scorer",
            "blocker": "no_qrels_match_query_and_skill_ids",
            "train_query_count": 0,
            "skill_count": len(skill_ids),
        }
        write_json(output_dir / "report.json", report)
        return report

    train_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_query_embs = query_embs[selected_query_indices].float().to(train_device)
    train_skill_embs = skill_embs.float().to(train_device)
    model = SkillRouterBaseScorer(dim=int(query_embs.size(1)), skill_count=len(skill_ids)).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    with torch.no_grad():
        initial_loss = float(multi_positive_nll(model.score(train_query_embs, train_skill_embs), positives).detach().cpu())
    loss_value = initial_loss
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = multi_positive_nll(model.score(train_query_embs, train_skill_embs), positives)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        final_scores = model.score(train_query_embs, train_skill_embs).detach().cpu()
    train_predictions, _rows = _rank_with_scores(
        query_ids=[query_ids[idx] for idx in selected_query_indices],
        skill_ids=skill_ids,
        scores=final_scores,
        top_k=top_k,
    )
    train_qrels = {query_ids[idx]: qrels[query_ids[idx]] for idx in selected_query_indices}
    metrics = compute_retrieval_metrics(train_predictions, train_qrels, ks=(1, 5, 10, top_k))
    checkpoint_path = output_dir / "model.pt"
    report = {
        "status": "ok",
        "method": "skillrouter_base_trainable_scorer",
        "baseline_family": "SkillRouter frozen embedding plus trainable AppWorld routing scorer",
        "train_query_count": len(selected_query_indices),
        "skill_count": len(skill_ids),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "initial_loss": initial_loss,
        "final_loss": loss_value,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "caveat": "Local AppWorld trainable scorer over frozen SkillRouter embeddings; not full-parameter SkillRouter fine-tuning.",
    }
    save_skillrouter_base_checkpoint(checkpoint_path, model=model.cpu(), skill_ids=skill_ids, report=report)
    write_json(output_dir / "report.json", report)
    return report


def evaluate_skillrouter_base_from_embeddings(
    *,
    query_ids: Sequence[str],
    skill_ids: Sequence[str],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    checkpoint_path: str | Path,
    qrels_path: str | Path,
    output_dir: str | Path,
    top_k: int = 20,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    model, metadata = load_skillrouter_base_checkpoint(checkpoint_path)
    checkpoint_skill_ids = [str(item) for item in metadata.get("skill_ids", [])]
    if checkpoint_skill_ids and list(skill_ids) != checkpoint_skill_ids:
        report = {
            "status": "blocked",
            "method": "skillrouter_base_trainable_scorer",
            "blocker": "skill_id_order_mismatch",
            "checkpoint_path": str(checkpoint_path),
            "input_skill_count": len(skill_ids),
            "checkpoint_skill_count": len(checkpoint_skill_ids),
        }
        write_json(output_dir / "report.json", report)
        return report

    with torch.no_grad():
        scores = model.score(query_embs.float(), skill_embs.float())
    predictions, rows = _rank_with_scores(query_ids=query_ids, skill_ids=skill_ids, scores=scores, top_k=top_k)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    qrels = _load_qrels(qrels_path)
    metrics = compute_retrieval_metrics(predictions, qrels, ks=(1, 5, 10, top_k))
    report = {
        "status": "ok",
        "method": "skillrouter_base_trainable_scorer",
        "baseline_family": "SkillRouter frozen embedding plus trainable AppWorld routing scorer",
        "checkpoint_path": str(checkpoint_path),
        "qrels_path": str(qrels_path),
        "predictions_path": str(predictions_path),
        "top_k": int(top_k),
        "query_count": len(query_ids),
        "skill_count": len(skill_ids),
        "qrel_query_count": len(qrels),
        "metrics": metrics,
        "checkpoint": {
            "dim": metadata.get("dim"),
            "skill_count": metadata.get("skill_count"),
            "train_report": metadata.get("report", {}),
        },
        "caveat": "Routing-layer AppWorld eval; not DB-state task-completion.",
    }
    write_json(output_dir / "report.json", report)
    return report


def _encode_hf_texts_local(
    *,
    model_name_or_path: str | Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    from clstr.skillret_official import _encode_hf_texts

    return _encode_hf_texts(str(model_name_or_path), texts, batch_size, max_length)


def run_appworld_skillrouter_base_train(
    *,
    tasks_path: str | Path = "data/appworld_routing/train_tasks.jsonl",
    qrels_path: str | Path = "data/appworld_routing/train_qrels.jsonl",
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    output_dir: str | Path = "outputs/appworld_skillrouter_base_train_v1",
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    batch_size: int = 8,
    max_length: int = 1024,
    epochs: int = 120,
    learning_rate: float = 0.05,
    weight_decay: float = 0.0,
    top_k: int = 20,
    seed: int = 13,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    skills = read_jsonl(skill_pool_path)
    qrels = _load_qrels(qrels_path)
    query_ids = [str(row.get("query_id") or row.get("task_id")) for row in tasks]
    skill_ids = [str(row["skill_id"]) for row in skills]
    query_embs = _encode_hf_texts_local(
        model_name_or_path=model_name_or_path,
        texts=[skillrouter_base_query_text(row) for row in tasks],
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_hf_texts_local(
        model_name_or_path=model_name_or_path,
        texts=[skillrouter_base_skill_text(row) for row in skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    report = train_skillrouter_base_from_embeddings(
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        qrels=qrels,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        top_k=top_k,
    )
    report.update(
        {
            "tasks_path": str(tasks_path),
            "qrels_path": str(qrels_path),
            "skill_pool_path": str(skill_pool_path),
            "model_name_or_path": str(model_name_or_path),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
        }
    )
    write_json(Path(output_dir) / "report.json", report)
    return report


def run_appworld_skillrouter_base_eval(
    *,
    tasks_path: str | Path = "data/appworld_routing/dev_tasks.jsonl",
    qrels_path: str | Path = "data/appworld_routing/dev_qrels.jsonl",
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    checkpoint_path: str | Path = "outputs/appworld_skillrouter_base_train_v1/model.pt",
    output_dir: str | Path = "outputs/appworld_skillrouter_base_eval/dev_v1",
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    batch_size: int = 8,
    max_length: int = 1024,
    top_k: int = 20,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    skills = read_jsonl(skill_pool_path)
    query_ids = [str(row.get("query_id") or row.get("task_id")) for row in tasks]
    skill_ids = [str(row["skill_id"]) for row in skills]
    query_embs = _encode_hf_texts_local(
        model_name_or_path=model_name_or_path,
        texts=[skillrouter_base_query_text(row) for row in tasks],
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_hf_texts_local(
        model_name_or_path=model_name_or_path,
        texts=[skillrouter_base_skill_text(row) for row in skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    report = evaluate_skillrouter_base_from_embeddings(
        query_ids=query_ids,
        skill_ids=skill_ids,
        query_embs=query_embs,
        skill_embs=skill_embs,
        checkpoint_path=checkpoint_path,
        qrels_path=qrels_path,
        output_dir=output_dir,
        top_k=top_k,
    )
    report.update(
        {
            "tasks_path": str(tasks_path),
            "skill_pool_path": str(skill_pool_path),
            "model_name_or_path": str(model_name_or_path),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
        }
    )
    write_json(Path(output_dir) / "report.json", report)
    return report
