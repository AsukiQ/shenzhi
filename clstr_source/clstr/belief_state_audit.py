from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.belief import resolve_belief_top_k


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _pairwise_cosine_mean(vectors: torch.Tensor, *, max_rows: int = 256) -> float:
    vectors = vectors.detach().float()
    if vectors.size(0) <= 1:
        return 1.0
    if vectors.size(0) > int(max_rows):
        vectors = vectors[: int(max_rows)]
    normalized = F.normalize(vectors, p=2, dim=-1)
    sims = normalized @ normalized.t()
    mask = ~torch.eye(sims.size(0), dtype=torch.bool, device=sims.device)
    if not bool(mask.any()):
        return 1.0
    return float(sims[mask].mean().detach().cpu().item())


def _topk_probability_memory(
    *,
    logits: torch.Tensor,
    skill_embeddings: torch.Tensor,
    support_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    skill_count = int(logits.size(-1))
    if int(support_size) >= skill_count:
        probs = torch.softmax(logits, dim=-1)
        return probs, probs @ skill_embeddings, skill_count, "full"
    top_values, top_indices = torch.topk(logits, k=int(support_size), dim=-1)
    probs = torch.softmax(top_values, dim=-1)
    selected = skill_embeddings.index_select(0, top_indices.reshape(-1)).view(logits.size(0), int(support_size), -1)
    memory = torch.bmm(probs.unsqueeze(1), selected).squeeze(1)
    return probs, memory, int(support_size), "topk"


def _logit_source_report(skill_table: Any, h: torch.Tensor, source: str) -> dict[str, Any]:
    if source == "belief":
        logits = skill_table.belief_logits(h)
        scale = getattr(skill_table, "logit_scale_belief", None)
        bias = getattr(skill_table, "skill_bias_belief", None)
    elif source == "retrieval":
        logits = skill_table.retrieval_logits(h)
        scale = getattr(skill_table, "logit_scale_retr", None)
        bias = getattr(skill_table, "skill_bias_retr", None)
    else:
        raise ValueError(f"unsupported logit source: {source}")
    logits = logits.float()
    skill_count = max(1, int(logits.size(-1)))
    if source == "belief":
        support_size = resolve_belief_top_k(skill_table, skill_count)
    else:
        support_size = skill_count
    skill_embeddings = skill_table.E.to(device=logits.device, dtype=logits.dtype)
    probs, m, probability_support_size, probability_support = _topk_probability_memory(
        logits=logits,
        skill_embeddings=skill_embeddings,
        support_size=support_size,
    )
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
    effective_support = entropy.exp()
    top_k = min(5, int(probs.size(-1)))
    top_values = torch.topk(probs, k=top_k, dim=-1).values
    top1 = top_values[:, 0]
    top5 = top_values.sum(dim=-1)
    report = {
        "logit_scale": None if scale is None else _as_float(torch.as_tensor(scale).detach().exp().clamp(max=100.0)),
        "raw_logit_scale": None if scale is None else _as_float(torch.as_tensor(scale).detach()),
        "bias": None if bias is None else _tensor_summary(torch.as_tensor(bias)),
        "logits": _tensor_summary(logits),
        "probability_support": probability_support,
        "probability_support_size": int(probability_support_size),
        "entropy_mean": float(entropy.detach().cpu().mean().item()),
        "effective_support_mean": float(effective_support.detach().cpu().mean().item()),
        "effective_support_fraction_mean": float((effective_support / float(skill_count)).detach().cpu().mean().item()),
        "top1_mass_mean": float(top1.detach().cpu().mean().item()),
        "top5_mass_mean": float(top5.detach().cpu().mean().item()),
        "m_norm_mean": float(m.norm(dim=-1).detach().cpu().mean().item()),
        "m_norm_std": float(m.norm(dim=-1).detach().cpu().std(unbiased=False).item()) if m.size(0) > 1 else 0.0,
        "pairwise_m_cosine_mean": _pairwise_cosine_mean(m),
    }
    return report


def _benchmark_counts(labels: list[str] | None, row_count: int) -> dict[str, dict[str, int]]:
    if not labels:
        return {}
    counts = Counter(str(label or "<missing>") for label in labels[:row_count])
    return {key: {"row_count": int(value)} for key, value in sorted(counts.items())}


def audit_belief_tensors(
    *,
    skill_table: Any,
    state_embeddings: torch.Tensor,
    benchmark_labels: list[str] | None = None,
    max_effective_support_fraction: float = 0.50,
    min_top5_mass: float = 0.05,
    max_pairwise_cosine: float = 0.995,
) -> dict[str, Any]:
    if state_embeddings.ndim != 2:
        raise ValueError(f"state_embeddings must be rank-2, got shape={tuple(state_embeddings.shape)}")
    if state_embeddings.size(0) <= 0:
        raise ValueError("state_embeddings must contain at least one row")
    if not hasattr(skill_table, "belief_logits"):
        raise TypeError("skill_table must expose belief_logits(h)")
    if not hasattr(skill_table, "retrieval_logits"):
        raise TypeError("skill_table must expose retrieval_logits(h)")
    if not hasattr(skill_table, "E"):
        raise TypeError("skill_table must expose E")

    h = state_embeddings.detach()
    skill_count = int(skill_table.E.size(0))
    belief = _logit_source_report(skill_table, h, "belief")
    retrieval = _logit_source_report(skill_table, h, "retrieval")
    blockers: list[str] = []
    if float(belief["effective_support_fraction_mean"]) > float(max_effective_support_fraction):
        blockers.append("belief_effective_support_too_large")
    if float(belief["top5_mass_mean"]) < float(min_top5_mass):
        blockers.append("belief_top5_mass_too_low")
    if float(belief["pairwise_m_cosine_mean"]) > float(max_pairwise_cosine):
        blockers.append("belief_pairwise_cosine_too_high")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "metadata": {
            "row_count": int(h.size(0)),
            "embedding_dim": int(h.size(1)),
            "skill_count": int(skill_count),
            "thresholds": {
                "max_effective_support_fraction": float(max_effective_support_fraction),
                "min_top5_mass": float(min_top5_mass),
                "max_pairwise_cosine": float(max_pairwise_cosine),
            },
        },
        "belief": belief,
        "retrieval": retrieval,
        "by_benchmark": _benchmark_counts(benchmark_labels, int(h.size(0))),
        "paper_scope_note": (
            "This audit checks whether the inference-time belief memory is sharp and nonconstant. "
            "It is not a routing metric and does not replace strict benchmark evaluation."
        ),
    }


def write_belief_audit_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def sample_state_rows(path: str | Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        state_text = str(row.get("state_text") or row.get("task_text") or row.get("goal_text") or "").strip()
        if not state_text:
            continue
        rows.append(row)
        if len(rows) >= int(max_rows):
            break
    return rows
