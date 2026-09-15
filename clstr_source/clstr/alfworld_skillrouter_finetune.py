from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from clstr.alfworld_policy_train import (
    PolicyExample,
    build_goal_conditioned_state_text,
    compute_policy_diagnostic_metrics,
    load_policy_examples,
)
from clstr.external_data import write_json


def skillrouter_alfworld_query_text(state_text: str) -> str:
    return (
        "Instruct: Given an ALFWorld goal, observation, and action history, "
        "retrieve the next admissible text action that best advances the task\n"
        f"Query:{str(state_text)[:1800]}"
    )


def skillrouter_alfworld_action_text(action: str) -> str:
    return f"ALFWorld admissible action: {str(action)}"


class SkillRouterActionProjectionAdapter(nn.Module):
    def __init__(self, hidden_size: int, projection_init: str = "identity") -> None:
        super().__init__()
        self.q_proj = nn.Linear(int(hidden_size), int(hidden_size), bias=False)
        self.d_proj = nn.Linear(int(hidden_size), int(hidden_size), bias=False)
        if projection_init == "identity":
            with torch.no_grad():
                eye = torch.eye(int(hidden_size))
                self.q_proj.weight.copy_(eye)
                self.d_proj.weight.copy_(eye)
        elif projection_init != "default":
            raise ValueError(f"unsupported projection_init: {projection_init}")

    def encode_queries(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_proj(raw.float()), p=2, dim=-1)

    def encode_docs(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.d_proj(raw.float()), p=2, dim=-1)


@dataclass
class _ActionAdapterCache:
    query_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    labels: torch.Tensor
    done_labels: torch.Tensor

    def select(self, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(indices, dtype=torch.long)
        return (
            self.query_embeddings.index_select(0, idx).to(device),
            self.candidate_embeddings.index_select(0, idx).to(device),
            self.candidate_mask.index_select(0, idx).to(device),
            self.labels.index_select(0, idx).to(device),
            self.done_labels.index_select(0, idx).to(device),
        )


def _example_from_record(row: dict[str, Any]) -> PolicyExample:
    candidates = [str(item) for item in row.get("candidate_actions", row.get("admissible_commands_t", []))]
    if "label_index" in row:
        label_index = int(row["label_index"])
        expert_action = candidates[label_index]
    else:
        expert_action = str(row.get("expert_action", row.get("expert_action_t", "")))
        label_index = candidates.index(expert_action)
    return PolicyExample(
        split=str(row.get("split", "train")),
        gamefile=str(row.get("gamefile", "")),
        state_text=str(row.get("state_text") or build_goal_conditioned_state_text(row)),
        candidate_actions=candidates,
        expert_action=expert_action,
        label_index=label_index,
        done=bool(row.get("done", row.get("done_t", False))),
    )


def _split_examples(
    examples: list[PolicyExample],
    eval_fraction: float,
    seed: int,
) -> tuple[list[PolicyExample], list[PolicyExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or eval_fraction <= 0:
        return shuffled, shuffled
    eval_count = max(1, int(math.ceil(len(shuffled) * float(eval_fraction))))
    eval_examples = shuffled[:eval_count]
    train_examples = shuffled[eval_count:] or shuffled
    return train_examples, eval_examples


def _candidate_length_stats(examples: list[PolicyExample]) -> dict[str, float]:
    lengths = [len(example.candidate_actions) for example in examples]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": round(float(sum(lengths) / len(lengths)), 4),
    }


def _pad_examples(examples: list[PolicyExample]) -> tuple[list[list[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(example.candidate_actions) for example in examples)
    rows: list[list[str]] = []
    masks: list[list[bool]] = []
    labels: list[int] = []
    done: list[float] = []
    for example in examples:
        actions = list(example.candidate_actions)
        labels.append(int(example.label_index))
        done.append(1.0 if example.done else 0.0)
        if len(actions) < width:
            rows.append(actions + [actions[-1]] * (width - len(actions)))
            masks.append([True] * len(actions) + [False] * (width - len(actions)))
        else:
            rows.append(actions)
            masks.append([True] * len(actions))
    return rows, torch.tensor(masks, dtype=torch.bool), torch.tensor(labels, dtype=torch.long), torch.tensor(done)


def _normalize_embedding(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().float()
    if tensor.ndim != 1:
        raise ValueError("text_embedding_fn must return a 1-D tensor per text")
    return F.normalize(tensor, p=2, dim=0)


def _build_cache(
    examples: list[PolicyExample],
    text_embedding_fn: Callable[[str], torch.Tensor],
) -> _ActionAdapterCache:
    candidate_rows, mask, labels, done = _pad_examples(examples)
    query_embeddings = torch.stack(
        [_normalize_embedding(text_embedding_fn(skillrouter_alfworld_query_text(example.state_text))) for example in examples],
        dim=0,
    )
    flat_actions = [skillrouter_alfworld_action_text(action) for row in candidate_rows for action in row]
    candidate_embeddings = torch.stack([_normalize_embedding(text_embedding_fn(text)) for text in flat_actions], dim=0)
    candidate_embeddings = candidate_embeddings.view(len(examples), len(candidate_rows[0]), -1)
    return _ActionAdapterCache(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_mask=mask,
        labels=labels,
        done_labels=done.float(),
    )


def _scores(
    adapter: SkillRouterActionProjectionAdapter | None,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if adapter is None:
        q = F.normalize(query_embeddings.float(), p=2, dim=-1)
        d = F.normalize(candidate_embeddings.float(), p=2, dim=-1)
    else:
        q = adapter.encode_queries(query_embeddings)
        d = adapter.encode_docs(candidate_embeddings)
    logits = torch.einsum("bd,bcd->bc", q, d) / max(float(temperature), 1.0e-6)
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


def _evaluate_cache(
    adapter: SkillRouterActionProjectionAdapter | None,
    cache: _ActionAdapterCache,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> dict[str, Any]:
    total = 0
    metric_sums: Counter[str] = Counter()
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        for start in range(0, int(cache.labels.numel()), max(1, int(batch_size))):
            indices = list(range(start, min(start + max(1, int(batch_size)), int(cache.labels.numel()))))
            q, d, mask, labels, done = cache.select(indices, device)
            logits = _scores(adapter, q, d, mask, temperature)
            metrics = compute_policy_diagnostic_metrics(logits, labels, done, None)
            for key, value in metrics.items():
                metric_sums[key] += float(value) * len(indices)
            total += len(indices)
    return {
        "status": "ok",
        "example_count": int(total),
        "metrics": {key: round(float(value) / max(1, total), 6) for key, value in sorted(metric_sums.items())},
        "not_closed_loop_success": True,
    }


def train_alfworld_skillrouter_action_adapter_with_embeddings(
    *,
    replay_path: str | Path,
    output_dir: str | Path,
    text_embedding_fn: Callable[[str], torch.Tensor],
    examples: list[dict[str, Any]] | None = None,
    output_checkpoint_path: str | Path | None = None,
    max_steps: int = 512,
    batch_size: int = 8,
    learning_rate: float = 5.0e-5,
    temperature: float = 0.05,
    seed: int = 13,
    eval_fraction: float = 0.1,
    projection_init: str = "identity",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    skipped: dict[str, int] = {}
    if examples is None:
        loaded_examples, skipped = load_policy_examples(replay_path)
    else:
        loaded_examples = [_example_from_record(row) for row in examples]
    if not loaded_examples:
        raise ValueError("no usable ALFWorld SkillRouter action-adapter examples")
    if any(str(example.split).lower() != "train" for example in loaded_examples):
        raise ValueError("ALFWorld SkillRouter action-adapter training received non-train examples")

    train_examples, eval_examples = _split_examples(loaded_examples, eval_fraction=eval_fraction, seed=seed)
    train_cache = _build_cache(train_examples, text_embedding_fn)
    eval_cache = _build_cache(eval_examples, text_embedding_fn)
    hidden_size = int(train_cache.query_embeddings.size(-1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = SkillRouterActionProjectionAdapter(hidden_size=hidden_size, projection_init=projection_init).to(device)

    baseline_eval = _evaluate_cache(None, eval_cache, device, batch_size=batch_size, temperature=temperature)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate))
    last_metrics: dict[str, float] = {}
    adapter.train()
    for step_idx in range(1, max(1, int(max_steps)) + 1):
        start = ((step_idx - 1) * max(1, int(batch_size))) % len(train_examples)
        indices = [(start + offset) % len(train_examples) for offset in range(max(1, int(batch_size)))]
        q, d, mask, labels, done = train_cache.select(indices, device)
        logits = _scores(adapter, q, d, mask, temperature)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step_idx == 1 or step_idx == int(max_steps):
            last_metrics = {
                "loss": float(loss.detach().cpu().item()),
                **compute_policy_diagnostic_metrics(logits.detach(), labels.detach(), done.detach(), None),
            }

    eval_report = _evaluate_cache(adapter, eval_cache, device, batch_size=batch_size, temperature=temperature)
    baseline_metrics = baseline_eval.get("metrics", {})
    eval_metrics = eval_report.get("metrics", {})
    delta = {
        key: round(float(eval_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0)), 6)
        for key in ("expert_action_recall@1", "expert_action_recall@5", "expert_action_mrr")
    }
    checkpoint_path = (
        Path(output_checkpoint_path)
        if output_checkpoint_path is not None
        else output_dir / "checkpoints" / f"skillrouter_alfworld_action_adapter-step{int(max_steps)}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "alfworld_skillrouter_action_adapter",
        "step": int(max_steps),
        "hidden_size": hidden_size,
        "adapter_state_dict": adapter.cpu().state_dict(),
        "training_objective": "multi_candidate_admissible_action_ce",
        "training_data": "ALFWorld train replay only",
        "base_embedding_normalized": True,
        "query_text_format": "skillrouter_alfworld_query_text",
        "action_text_format": "skillrouter_alfworld_action_text",
        "temperature": float(temperature),
        "projection_init": projection_init,
        "metrics": last_metrics,
        "eval_metrics": eval_metrics,
        "baseline_eval_metrics": baseline_metrics,
        "delta_vs_frozen_skillrouter": delta,
    }
    torch.save(payload, checkpoint_path)
    report = {
        "status": "ok",
        "training_objective": "multi_candidate_admissible_action_ce",
        "training_data": "ALFWorld train replay only",
        "checkpoint": str(checkpoint_path),
        "policy_sample_count": len(loaded_examples),
        "train_sample_count": len(train_examples),
        "eval_sample_count": len(eval_examples),
        "candidate_length_stats": _candidate_length_stats(loaded_examples),
        "skipped_reasons": skipped,
        "uses_alfworld_valid_or_test_for_training": False,
        "frozen_skillrouter_backbone": True,
        "trainable_modules": ["skillrouter_action_projection_adapter.q_proj", "skillrouter_action_projection_adapter.d_proj"],
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "projection_init": projection_init,
        "metrics": last_metrics,
        "eval_metrics": eval_metrics,
        "baseline_eval_metrics": baseline_metrics,
        "delta_vs_frozen_skillrouter": delta,
        "gate_pass_offline": delta.get("expert_action_recall@1", 0.0) >= 0.0
        and eval_metrics.get("expert_action_recall@1", 0.0) >= baseline_metrics.get("expert_action_recall@1", 0.0),
        "not_closed_loop_success": True,
    }
    write_json(output_dir / "train_report.json", report)
    write_json(output_dir / "eval_report.json", {**eval_report, "baseline_eval": baseline_eval, "delta_vs_frozen_skillrouter": delta})
    return report


def _resolve_torch_dtype(value: str | None) -> torch.dtype | None:
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


def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, lengths]


def _load_skillrouter_encoder(model_name_or_path: str | Path, torch_dtype: str | None):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    dtype = _resolve_torch_dtype(torch_dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(str(model_name_or_path), **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, device


def run_alfworld_skillrouter_action_adapter_finetune(
    *,
    replay_path: str | Path = "data/alfworld_policy_replay/train_replay.jsonl",
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    max_steps: int = 512,
    batch_size: int = 8,
    learning_rate: float = 5.0e-5,
    temperature: float = 0.05,
    seed: int = 13,
    eval_fraction: float = 0.1,
    projection_init: str = "identity",
    encode_batch_size: int = 16,
    max_length: int = 2048,
    torch_dtype: str | None = "bfloat16",
) -> dict[str, Any]:
    examples, skipped = load_policy_examples(replay_path)
    if not examples:
        raise ValueError(f"no usable ALFWorld train replay policy examples in {replay_path}")
    unique_texts: list[str] = []
    seen: set[str] = set()
    for example in examples:
        query = skillrouter_alfworld_query_text(example.state_text)
        if query not in seen:
            seen.add(query)
            unique_texts.append(query)
        for action in example.candidate_actions:
            text = skillrouter_alfworld_action_text(action)
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

    tokenizer, model, device = _load_skillrouter_encoder(model_name_or_path, torch_dtype=torch_dtype)
    embedding_map: dict[str, torch.Tensor] = {}
    for start in range(0, len(unique_texts), max(1, int(encode_batch_size))):
        batch = unique_texts[start : start + max(1, int(encode_batch_size))]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
            pooled = _last_token_pool(out.last_hidden_state, tokens["attention_mask"])
            pooled = F.normalize(pooled.float(), p=2, dim=-1).cpu()
        for text, emb in zip(batch, pooled):
            embedding_map[text] = emb

    rows = [
        {
            "split": example.split,
            "state_text": example.state_text,
            "candidate_actions": example.candidate_actions,
            "label_index": example.label_index,
            "done": example.done,
        }
        for example in examples
    ]
    report = train_alfworld_skillrouter_action_adapter_with_embeddings(
        replay_path=replay_path,
        output_dir=output_dir,
        examples=rows,
        text_embedding_fn=lambda text: embedding_map[str(text)],
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        seed=seed,
        eval_fraction=eval_fraction,
        projection_init=projection_init,
    )
    report["model_name_or_path"] = str(model_name_or_path)
    report["encode_batch_size"] = int(encode_batch_size)
    report["encoder_max_length"] = int(max_length)
    report["skipped_reasons"] = skipped
    report["unique_encoded_text_count"] = len(unique_texts)
    write_json(Path(output_dir) / "train_report.json", report)
    return report
