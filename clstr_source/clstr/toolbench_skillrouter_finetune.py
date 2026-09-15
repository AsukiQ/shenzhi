from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EvalCoreTask:
    task_id: str
    instruction_text: str
    positive_skill_ids: list[str]
    relevance: dict[str, float]


@dataclass(frozen=True)
class EvalCoreSplit:
    tasks: list[EvalCoreTask]
    skills: list[dict[str, Any]]


def _build_task_query_text(task: EvalCoreTask, *, query_variant: str = "history") -> str:
    text = str(task.instruction_text or "").strip()
    if query_variant == "history":
        return text
    if query_variant == "goal_only":
        kept_lines: list[str] = []
        for line in text.splitlines():
            normalized = line.strip().lower()
            if normalized.startswith(("benchmark:", "trajectory_type:", "previous_tools:")):
                continue
            if line.strip():
                kept_lines.append(line.strip())
        return "\n".join(kept_lines).strip()
    raise ValueError(f"unsupported query_variant: {query_variant}")


class SkillRouterBiEncoderAdapter(nn.Module):
    def __init__(self, hidden_size: int, projection_init: str = "identity") -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.d_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        if projection_init == "identity":
            with torch.no_grad():
                eye = torch.eye(hidden_size)
                self.q_proj.weight.copy_(eye)
                self.d_proj.weight.copy_(eye)
        elif projection_init != "default":
            raise ValueError(f"unsupported projection_init: {projection_init}")

    def encode_queries(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_proj(raw), p=2, dim=-1)

    def encode_docs(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.d_proj(raw), p=2, dim=-1)


def _iter_jsonl_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file() and path.name.endswith(".jsonl"):
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.name.endswith(".jsonl"))
    raise FileNotFoundError(path)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in _iter_jsonl_paths(path):
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_relevance(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_positives(task_id: str, relevance: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    entry = relevance.get(task_id) or {}
    positives = [str(item) for item in entry.get("core_gt_ids") or entry.get("gt_skill_ids") or []]
    rel_map = {str(key): float(value) for key, value in (entry.get("relevance") or {}).items()}
    if not positives and rel_map:
        positives = [skill_id for skill_id, score in rel_map.items() if float(score) > 0]
    return sorted(set(positives)), rel_map


def load_eval_core_split(data_root: str | Path, split: str, max_tasks: int | None = None) -> EvalCoreSplit:
    split_dir = Path(data_root) / split
    skills = _read_jsonl(split_dir / "easy")
    skill_ids = {str(row.get("skill_id") or "") for row in skills}
    relevance = _load_relevance(split_dir / "relevance.json")
    tasks: list[EvalCoreTask] = []
    for row in _read_jsonl(split_dir / "tasks.jsonl"):
        task_id = str(row.get("task_id") or "")
        positives, rel_map = _task_positives(task_id, relevance)
        positives = [skill_id for skill_id in positives if skill_id in skill_ids]
        if not task_id or not positives:
            continue
        tasks.append(
            EvalCoreTask(
                task_id=task_id,
                instruction_text=str(row.get("instruction_text") or ""),
                positive_skill_ids=positives,
                relevance={skill_id: rel_map.get(skill_id, 1.0) for skill_id in positives},
            )
        )
        if max_tasks is not None and len(tasks) >= int(max_tasks):
            break
    return EvalCoreSplit(tasks=tasks, skills=skills)


def select_skill_pool_with_required_positives(
    skills: list[dict[str, Any]],
    required_skill_ids: set[str],
    max_skills: int | None = None,
) -> list[dict[str, Any]]:
    if max_skills is None:
        return skills
    max_skills = max(1, int(max_skills))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for row in skills:
        skill_id = str(row.get("skill_id") or "")
        if skill_id in required_skill_ids and skill_id not in selected_ids:
            selected.append(row)
            selected_ids.add(skill_id)
    for row in skills:
        if len(selected) >= max_skills:
            break
        skill_id = str(row.get("skill_id") or "")
        if skill_id and skill_id not in selected_ids:
            selected.append(row)
            selected_ids.add(skill_id)
    return selected


def _import_official_skillrouter(official_repo_path: str | Path):
    official_repo_path = str(Path(official_repo_path))
    if official_repo_path not in sys.path:
        sys.path.insert(0, official_repo_path)
    from src.common import (  # type: ignore
        format_query,
        format_rerank_prompt,
        format_skill,
        get_reranker_template_tokens,
        last_token_pool,
        load_reranker_model,
        tokenize_reranker_text,
    )
    from src.metrics import compute_all_metrics  # type: ignore
    from src.run_open_model_eval import score_candidates_with_reranker  # type: ignore

    return {
        "format_query": format_query,
        "format_skill": format_skill,
        "format_rerank_prompt": format_rerank_prompt,
        "last_token_pool": last_token_pool,
        "compute_all_metrics": compute_all_metrics,
        "load_reranker_model": load_reranker_model,
        "score_candidates_with_reranker": score_candidates_with_reranker,
        "get_reranker_template_tokens": get_reranker_template_tokens,
        "tokenize_reranker_text": tokenize_reranker_text,
    }


def _load_embedding_backbone(model_path: str, torch_dtype: str | None):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    dtype = _resolve_torch_dtype_local(torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def _resolve_torch_dtype_local(value: str | None) -> torch.dtype | None:
    if value is None or str(value).lower() in {"", "none", "auto"}:
        return None
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {value}")


def _encode_raw_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    last_token_pool,
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch = texts[start : start + max(1, int(batch_size))]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
            pooled = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
        encoded.append(pooled.float().cpu())
    if not encoded:
        return torch.empty(0, int(model.config.hidden_size), dtype=torch.float32)
    return torch.cat(encoded, dim=0)


def _project_in_batches(
    adapter: SkillRouterBiEncoderAdapter,
    raw: torch.Tensor,
    *,
    kind: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    projected: list[torch.Tensor] = []
    for start in range(0, raw.size(0), max(1, int(batch_size))):
        batch = raw[start : start + max(1, int(batch_size))].to(device)
        with torch.no_grad():
            if kind == "query":
                out = adapter.encode_queries(batch)
            elif kind == "doc":
                out = adapter.encode_docs(batch)
            else:
                raise ValueError(kind)
        projected.append(out.cpu())
    return torch.cat(projected, dim=0)


def _positive_mask(tasks: list[EvalCoreTask], skill_id_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(tasks), len(skill_id_to_idx)), dtype=torch.bool, device=device)
    for row_idx, task in enumerate(tasks):
        for skill_id in task.positive_skill_ids:
            skill_idx = skill_id_to_idx.get(skill_id)
            if skill_idx is not None:
                mask[row_idx, skill_idx] = True
    return mask


def _multi_positive_nll(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, min_value)
    return (torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(positive_logits.float(), dim=-1)).mean()


def _aggregate(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted(metric_rows[0])
    return {key: float(sum(float(row[key]) for row in metric_rows) / len(metric_rows)) for key in keys}


def _evaluate_rankings(
    *,
    tasks: list[EvalCoreTask],
    skills: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    adapter: SkillRouterBiEncoderAdapter,
    official: dict[str, Any],
    output_dir: Path,
    retrieval_top_k: int,
    eval_batch_size: int,
    device: torch.device,
    reranker_model_path: str | None,
    reranker_max_length: int,
    reranker_batch_size: int,
    prompt_format: str,
    query_variant: str = "history",
) -> dict[str, Any]:
    del eval_batch_size
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir = output_dir / "retrieval"
    reranked_dir = output_dir / "reranked"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    reranked_dir.mkdir(parents=True, exist_ok=True)

    skill_ids = [str(row["skill_id"]) for row in skills]
    skill_id_set = set(skill_ids)
    adapter.eval()
    query_projected = _project_in_batches(adapter, query_embs, kind="query", batch_size=256, device=device)
    skill_projected = _project_in_batches(adapter, skill_embs, kind="doc", batch_size=512, device=device)
    scores = query_projected @ skill_projected.T
    top_k = min(max(1, int(retrieval_top_k)), len(skill_ids))
    values, indices = torch.topk(scores, k=top_k, dim=-1)

    retrieval_results: dict[str, list[str]] = {}
    reranked_results: dict[str, list[str]] = {}
    retrieval_metrics: list[dict[str, float]] = []
    pipeline_metrics: list[dict[str, float]] = []

    rr_model = None
    rr_tokenizer = None
    if reranker_model_path:
        rr_model, rr_tokenizer = official["load_reranker_model"](reranker_model_path)
        rr_model.to(device).eval()

    for task_idx, task in enumerate(tasks):
        ranked_indices = indices[task_idx].tolist()
        ranked_ids = [skill_ids[int(idx)] for idx in ranked_indices]
        retrieval_results[task.task_id] = ranked_ids
        gt_ids = set(task.positive_skill_ids) & skill_id_set
        if not gt_ids:
            continue
        relevance = {key: float(value) for key, value in task.relevance.items() if key in skill_id_set}
        retrieval_metrics.append(official["compute_all_metrics"](ranked_ids, gt_ids, relevance or None))

        if rr_model is None or rr_tokenizer is None:
            reranked_results[task.task_id] = ranked_ids
            pipeline_metrics.append(official["compute_all_metrics"](ranked_ids, gt_ids, relevance or None))
            continue

        candidates = [skills[int(idx)] for idx in ranked_indices]
        rr_scores = official["score_candidates_with_reranker"](
            rr_model,
            rr_tokenizer,
            _build_task_query_text(task, query_variant=query_variant),
            candidates,
            prompt_format,
            reranker_max_length,
            reranker_batch_size,
            device,
        )
        ranked_pairs = sorted(zip(ranked_ids, rr_scores), key=lambda item: item[1], reverse=True)
        reranked_ids = [skill_id for skill_id, _score in ranked_pairs]
        reranked_results[task.task_id] = reranked_ids
        pipeline_metrics.append(official["compute_all_metrics"](reranked_ids, gt_ids, relevance or None))

    _write_json(retrieval_dir / "easy.json", retrieval_results)
    _write_json(reranked_dir / "easy.json", reranked_results)
    summary = {
        "easy": {
            "retrieval": {
                "all": {**_aggregate(retrieval_metrics), "count": len(retrieval_metrics)}
            },
            "pipeline": {
                "all": {**_aggregate(pipeline_metrics), "count": len(pipeline_metrics)}
            },
        }
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_toolbench_skillrouter_finetune(
    *,
    data_root: str | Path,
    encoder_model_path: str,
    output_dir: str | Path,
    official_repo_path: str | Path = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter",
    reranker_model_path: str | None = None,
    max_train_tasks: int | None = None,
    max_eval_tasks: int | None = None,
    max_skills: int | None = None,
    max_steps: int = 512,
    batch_size: int = 8,
    learning_rate: float = 5.0e-5,
    temperature: float = 0.05,
    seed: int = 13,
    encoder_max_length: int = 4096,
    reranker_max_length: int = 4096,
    encoder_batch_size: int = 16,
    reranker_batch_size: int = 8,
    torch_dtype: str | None = "bfloat16",
    retrieval_top_k: int = 20,
    prompt_format: str = "flat-full",
    projection_init: str = "identity",
    train_doc_projection: bool = False,
    query_variant: str = "history",
    log_every: int = 10,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[skillrouter-ft] loading official SkillRouter helpers", flush=True)
    official = _import_official_skillrouter(official_repo_path)
    print("[skillrouter-ft] loading train/eval split", flush=True)
    train_split = load_eval_core_split(data_root, "train", max_tasks=max_train_tasks)
    eval_split = load_eval_core_split(data_root, "eval", max_tasks=max_eval_tasks)
    required = {skill_id for task in train_split.tasks + eval_split.tasks for skill_id in task.positive_skill_ids}
    skills = select_skill_pool_with_required_positives(train_split.skills, required, max_skills=max_skills)
    skill_id_to_idx = {str(row["skill_id"]): idx for idx, row in enumerate(skills)}
    train_tasks = [
        task
        for task in train_split.tasks
        if any(skill_id in skill_id_to_idx for skill_id in task.positive_skill_ids)
    ]
    eval_tasks = [
        task
        for task in eval_split.tasks
        if any(skill_id in skill_id_to_idx for skill_id in task.positive_skill_ids)
    ]
    if not train_tasks:
        raise ValueError("no train tasks have positives in selected skill pool")
    if not eval_tasks:
        raise ValueError("no eval tasks have positives in selected skill pool")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "[skillrouter-ft] data ready "
        f"train={len(train_tasks)} eval={len(eval_tasks)} skills={len(skills)} "
        f"device={device} train_doc_projection={train_doc_projection}",
        flush=True,
    )
    model, tokenizer = _load_embedding_backbone(encoder_model_path, torch_dtype=torch_dtype)
    model.to(device).eval()
    skill_texts = [official["format_skill"](skill) for skill in skills]
    train_query_texts = [official["format_query"](_build_task_query_text(task, query_variant=query_variant)) for task in train_tasks]
    eval_query_texts = [official["format_query"](_build_task_query_text(task, query_variant=query_variant)) for task in eval_tasks]
    print("[skillrouter-ft] encoding skills", flush=True)
    raw_skill_embs = _encode_raw_texts(
        model,
        tokenizer,
        skill_texts,
        max_length=encoder_max_length,
        batch_size=encoder_batch_size,
        device=device,
        last_token_pool=official["last_token_pool"],
    ).to(device)
    print("[skillrouter-ft] encoding train queries", flush=True)
    raw_train_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        train_query_texts,
        max_length=encoder_max_length,
        batch_size=encoder_batch_size,
        device=device,
        last_token_pool=official["last_token_pool"],
    ).to(device)
    print("[skillrouter-ft] encoding eval queries", flush=True)
    raw_eval_query_embs = _encode_raw_texts(
        model,
        tokenizer,
        eval_query_texts,
        max_length=encoder_max_length,
        batch_size=encoder_batch_size,
        device=device,
        last_token_pool=official["last_token_pool"],
    )

    adapter = SkillRouterBiEncoderAdapter(
        hidden_size=int(model.config.hidden_size),
        projection_init=projection_init,
    ).to(device)
    if not train_doc_projection:
        for param in adapter.d_proj.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate))
    positive_mask_all = _positive_mask(train_tasks, skill_id_to_idx, device=device)
    metric_rows: list[dict[str, float]] = []
    batch_size = max(1, int(batch_size))
    fixed_doc_embs = None
    if not train_doc_projection:
        with torch.no_grad():
            fixed_doc_embs = adapter.encode_docs(raw_skill_embs).detach()
    print("[skillrouter-ft] training", flush=True)
    for step in range(1, max(1, int(max_steps)) + 1):
        start = ((step - 1) * batch_size) % len(train_tasks)
        batch_indices = [(start + offset) % len(train_tasks) for offset in range(batch_size)]
        q_raw = raw_train_query_embs[batch_indices]
        pos_mask = positive_mask_all[batch_indices]
        query_embs = adapter.encode_queries(q_raw)
        doc_embs = fixed_doc_embs if fixed_doc_embs is not None else adapter.encode_docs(raw_skill_embs)
        logits = (query_embs @ doc_embs.T) / float(temperature)
        loss = _multi_positive_nll(logits, pos_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            top1 = logits.argmax(dim=-1)
            recall1 = pos_mask.gather(1, top1.unsqueeze(-1)).float().mean()
            top5 = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
            recall5 = pos_mask.gather(1, top5).any(dim=-1).float().mean()
        metric_rows.append(
            {
                "step": float(step),
                "loss": float(loss.detach().cpu().item()),
                "train_recall@1": float(recall1.detach().cpu().item()),
                "train_recall@5": float(recall5.detach().cpu().item()),
            }
        )
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            print(
                "[skillrouter-ft] "
                f"step={step}/{max_steps} loss={metric_rows[-1]['loss']:.6f} "
                f"train_recall@1={metric_rows[-1]['train_recall@1']:.4f} "
                f"train_recall@5={metric_rows[-1]['train_recall@5']:.4f}",
                flush=True,
            )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"toolbench_skillrouter_finetune-step{max_steps}.pt"
    torch.save(
        {
            "method": "official_compatible_skillrouter_finetune",
            "step": int(max_steps),
            "adapter_state_dict": adapter.state_dict(),
            "config": {
                "encoder_model_path": str(encoder_model_path),
                "official_repo_path": str(official_repo_path),
                "temperature": float(temperature),
                "projection_init": projection_init,
                "encoder_max_length": int(encoder_max_length),
                "torch_dtype": torch_dtype,
                "train_doc_projection": bool(train_doc_projection),
                "query_variant": query_variant,
            },
            "train_task_count": len(train_tasks),
            "eval_task_count": len(eval_tasks),
            "skill_count": len(skills),
            "last_metrics": metric_rows[-1] if metric_rows else {},
        },
        checkpoint_path,
    )
    _write_jsonl(output_dir / "training_metrics.jsonl", metric_rows)
    print("[skillrouter-ft] evaluating retrieval + official reranker", flush=True)
    summary = _evaluate_rankings(
        tasks=eval_tasks,
        skills=skills,
        query_embs=raw_eval_query_embs,
        skill_embs=raw_skill_embs.detach().cpu(),
        adapter=adapter,
        official=official,
        output_dir=output_dir,
        retrieval_top_k=retrieval_top_k,
        eval_batch_size=encoder_batch_size,
        device=device,
        reranker_model_path=reranker_model_path,
        reranker_max_length=reranker_max_length,
        reranker_batch_size=reranker_batch_size,
        prompt_format=prompt_format,
        query_variant=query_variant,
    )
    report = {
        "status": "ok",
        "method": "official_compatible_skillrouter_finetune",
        "official_repo_path": str(official_repo_path),
        "data_root": str(data_root),
        "encoder_model_path": str(encoder_model_path),
        "reranker_model_path": reranker_model_path,
        "checkpoint_path": str(checkpoint_path),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "train_doc_projection": bool(train_doc_projection),
        "query_variant": query_variant,
        "train_task_count": len(train_tasks),
        "eval_task_count": len(eval_tasks),
        "skill_count": len(skills),
        "max_train_tasks": max_train_tasks,
        "max_eval_tasks": max_eval_tasks,
        "max_skills": max_skills,
        "last_train_metrics": metric_rows[-1] if metric_rows else {},
        "summary": summary,
        "note": "Official SkillRouter public repo provides eval/model code, not a train entrypoint. This baseline uses the official embedding/reranker models, official text formatting, official metrics, and official-compatible eval_core data.",
    }
    _write_json(output_dir / "train_eval_report.json", report)
    return report
