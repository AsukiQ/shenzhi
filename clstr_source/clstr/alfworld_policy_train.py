from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from clstr.action_adapter import UniversalActionAdapter
from clstr.aux_pretrain import _build_model_from_routing_init, _encode_text_batches, _gpu_report
from clstr.aux_trajectories import load_aux_training_rows
from clstr.belief import subspace_obs
from clstr.external_data import write_json


@dataclass(frozen=True)
class PolicyExample:
    split: str
    gamefile: str
    state_text: str
    candidate_actions: list[str]
    expert_action: str
    label_index: int
    done: bool


@dataclass
class PolicyFeatureCache:
    state_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    labels: torch.Tensor
    done_labels: torch.Tensor
    encode_batch_size: int

    def select(self, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(indices, dtype=torch.long)
        return (
            self.state_embeddings.index_select(0, idx).to(device),
            self.candidate_embeddings.index_select(0, idx).to(device),
            self.candidate_mask.index_select(0, idx).to(device),
            self.labels.index_select(0, idx).to(device),
            self.done_labels.index_select(0, idx).to(device),
        )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_goal_conditioned_state_text(row: dict[str, Any], history_window: int = 6) -> str:
    history = [str(item) for item in row.get("history_t", []) if str(item).strip()]
    compact_history = " | ".join(history[-history_window:]) if history else "<empty>"
    return "\n".join(
        [
            f"goal: {str(row.get('goal_text') or '').strip()}",
            f"task_type: {str(row.get('task_type') or '').strip()}",
            f"observation: {str(row.get('observation_t') or '').strip()}",
            f"history: {compact_history}",
        ]
    )


def load_policy_examples(
    replay_path: str | Path,
    history_window: int = 6,
) -> tuple[list[PolicyExample], dict[str, int]]:
    examples: list[PolicyExample] = []
    skipped: Counter[str] = Counter()
    for row in _read_jsonl(replay_path):
        split = str(row.get("split", ""))
        if split != "train":
            skipped["non_train_split"] += 1
            continue
        if not bool(row.get("usable_for_policy", row.get("expert_action_in_admissible", False))):
            skipped[str(row.get("skip_reason") or "unusable_for_policy")] += 1
            continue
        candidate_actions = [str(item) for item in row.get("admissible_commands_t", [])]
        expert_action = str(row.get("expert_action_t") or "")
        if not expert_action or expert_action not in candidate_actions:
            skipped["expert_action_not_in_admissible"] += 1
            continue
        examples.append(
            PolicyExample(
                split=split,
                gamefile=str(row.get("gamefile") or ""),
                state_text=build_goal_conditioned_state_text(row, history_window=history_window),
                candidate_actions=candidate_actions,
                expert_action=expert_action,
                label_index=candidate_actions.index(expert_action),
                done=bool(row.get("done_t", False)),
            )
        )
    return examples, dict(sorted(skipped.items()))


def _candidate_length_stats(examples: list[PolicyExample]) -> dict[str, float]:
    lengths = [len(example.candidate_actions) for example in examples]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": round(float(sum(lengths) / len(lengths)), 4),
    }


def _pad_candidates(examples: list[PolicyExample]) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
    max_width = max(len(example.candidate_actions) for example in examples)
    padded: list[list[str]] = []
    masks: list[list[bool]] = []
    labels: list[int] = []
    for example in examples:
        row = list(example.candidate_actions)
        labels.append(example.label_index)
        if len(row) < max_width:
            pad = row[-1]
            padded.append(row + [pad] * (max_width - len(row)))
            masks.append([True] * len(row) + [False] * (max_width - len(row)))
        else:
            padded.append(row)
            masks.append([True] * len(row))
    return padded, torch.tensor(masks, dtype=torch.bool), torch.tensor(labels, dtype=torch.long)


def _encode_policy_batch(
    model: Any,
    examples: list[PolicyExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_rows, candidate_mask, labels = _pad_candidates(examples)
    state_texts = [example.state_text for example in examples]
    flat_candidates = [text for row in candidate_rows for text in row]
    with torch.no_grad():
        state_embs = model.encode_observations(state_texts).detach().to(device)
        candidate_embs = model.encode_observations(flat_candidates).detach().to(device)
    candidate_embs = candidate_embs.view(len(examples), len(candidate_rows[0]), -1)
    return state_embs, candidate_embs, candidate_mask.to(device), labels.to(device)


def _build_policy_feature_cache(
    model: Any,
    examples: list[PolicyExample],
    encode_batch_size: int,
) -> PolicyFeatureCache:
    candidate_rows, candidate_mask, labels = _pad_candidates(examples)
    state_texts = [example.state_text for example in examples]
    flat_candidates = [text for row in candidate_rows for text in row]
    state_embeddings = _encode_text_batches(
        model,
        state_texts,
        encode_batch_size,
        label="ALFWorld policy state embeddings",
    )
    flat_candidate_embeddings = _encode_text_batches(
        model,
        flat_candidates,
        encode_batch_size,
        label="ALFWorld admissible action embeddings",
    )
    candidate_embeddings = flat_candidate_embeddings.view(len(examples), len(candidate_rows[0]), -1)
    done_labels = torch.tensor([1.0 if example.done else 0.0 for example in examples], dtype=torch.float32)
    return PolicyFeatureCache(
        state_embeddings=state_embeddings,
        candidate_embeddings=candidate_embeddings,
        candidate_mask=candidate_mask,
        labels=labels,
        done_labels=done_labels,
        encode_batch_size=encode_batch_size,
    )


def _stop_logits(model: Any, state_embs: torch.Tensor) -> torch.Tensor:
    stop_head = getattr(model, "stop_head", None)
    if stop_head is None:
        return torch.zeros(state_embs.size(0), device=state_embs.device)
    skill_table = getattr(model, "skill_table", None)
    if skill_table is not None:
        with torch.no_grad():
            if hasattr(skill_table, "belief_logits") and hasattr(skill_table, "E"):
                m_obs = subspace_obs(skill_table, state_embs)
            elif hasattr(skill_table, "retrieval_logits") and hasattr(skill_table, "E"):
                logits = skill_table.retrieval_logits(state_embs)
                m_obs = torch.softmax(logits, dim=-1) @ skill_table.E.to(state_embs.device)
            else:
                m_obs = torch.zeros_like(state_embs)
    else:
        m_obs = torch.zeros_like(state_embs)
    return stop_head(state_embs, m_obs).squeeze(-1)


def compute_policy_diagnostic_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    done_labels: torch.Tensor | None = None,
    stop_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    if scores.numel() == 0:
        return {
            "expert_action_recall@1": 0.0,
            "expert_action_recall@5": 0.0,
            "expert_action_mrr": 0.0,
            "stop_accuracy": 0.0,
        }
    top1 = scores.argmax(dim=-1)
    k = min(5, scores.size(-1))
    topk = torch.topk(scores, k=k, dim=-1).indices
    ranks = torch.argsort(scores, dim=-1, descending=True)
    reciprocal_ranks = []
    for row, label in zip(ranks, labels):
        positions = (row == label).nonzero(as_tuple=False)
        reciprocal_ranks.append(1.0 / float(int(positions[0].item()) + 1) if positions.numel() else 0.0)
    if done_labels is not None and stop_logits is not None:
        stop_pred = (torch.sigmoid(stop_logits) >= 0.5).float()
        stop_hits = int((stop_pred == done_labels.float()).sum().item())
        stop_accuracy = float(stop_hits / max(1, int(done_labels.numel())))
    else:
        stop_accuracy = 0.0
    recall1_hits = int((top1 == labels).sum().item())
    recall5_hits = int((topk == labels.unsqueeze(-1)).any(dim=-1).sum().item())
    denom = max(1, int(labels.numel()))
    return {
        "expert_action_recall@1": float(recall1_hits / denom),
        "expert_action_recall@5": float(recall5_hits / denom),
        "expert_action_mrr": float(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))),
        "stop_accuracy": stop_accuracy,
    }


def _evaluate_policy(
    model: Any,
    action_adapter: UniversalActionAdapter,
    examples: list[PolicyExample],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if not examples:
        return {"status": "skipped", "reason": "no eval examples", "metrics": {}}
    model.eval()
    action_adapter.eval()
    metric_sums: Counter[str] = Counter()
    total = 0
    with torch.no_grad():
        for start in range(0, len(examples), max(1, batch_size)):
            batch = examples[start : start + max(1, batch_size)]
            state_embs, candidate_embs, candidate_mask, labels = _encode_policy_batch(model, batch, device)
            scores = action_adapter(state_embs, candidate_embs, candidate_mask)
            done_labels = torch.tensor([1.0 if example.done else 0.0 for example in batch], device=device)
            stop = _stop_logits(model, state_embs)
            metrics = compute_policy_diagnostic_metrics(scores, labels, done_labels, stop)
            for key, value in metrics.items():
                metric_sums[key] += float(value) * len(batch)
            total += len(batch)
    return {
        "status": "ok",
        "example_count": total,
        "metrics": {key: round(value / max(1, total), 6) for key, value in sorted(metric_sums.items())},
        "not_closed_loop_success": True,
    }


def _evaluate_policy_cache(
    model: Any,
    action_adapter: UniversalActionAdapter | None,
    cache: PolicyFeatureCache,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if cache.labels.numel() == 0:
        return {"status": "skipped", "reason": "no eval examples", "metrics": {}}
    model.eval()
    if action_adapter is not None:
        action_adapter.eval()
    metric_sums: Counter[str] = Counter()
    total = 0
    with torch.no_grad():
        for start in range(0, int(cache.labels.numel()), max(1, batch_size)):
            indices = list(range(start, min(start + max(1, batch_size), int(cache.labels.numel()))))
            state_embs, candidate_embs, candidate_mask, labels, done_labels = cache.select(indices, device)
            if action_adapter is None:
                state_norm = F.normalize(state_embs.float(), p=2, dim=-1)
                candidate_norm = F.normalize(candidate_embs.float(), p=2, dim=-1)
                scores = torch.einsum("bd,bcd->bc", state_norm, candidate_norm)
                scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
            else:
                scores = action_adapter(state_embs, candidate_embs, candidate_mask)
            stop = _stop_logits(model, state_embs)
            metrics = compute_policy_diagnostic_metrics(scores, labels, done_labels, stop)
            for key, value in metrics.items():
                metric_sums[key] += float(value) * len(indices)
            total += len(indices)
    return {
        "status": "ok",
        "example_count": total,
        "metrics": {key: round(value / max(1, total), 6) for key, value in sorted(metric_sums.items())},
        "score_source": "routing_init_cosine_scorer" if action_adapter is None else "alfworld_l_policy_adapter",
        "not_closed_loop_success": True,
    }


def _freeze_routing_foundation_for_policy(model: Any, stop_loss_weight: float) -> list[str]:
    if hasattr(model, "parameters"):
        for param in model.parameters():
            param.requires_grad_(False)
    trainable = ["universal_action_adapter"]
    if stop_loss_weight > 0 and getattr(model, "stop_head", None) is not None:
        for param in model.stop_head.parameters():
            param.requires_grad_(True)
        trainable.append("stop_head")
    return trainable


def _split_examples(
    examples: list[PolicyExample],
    eval_fraction: float,
    seed: int,
) -> tuple[list[PolicyExample], list[PolicyExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or eval_fraction <= 0:
        return shuffled, shuffled
    eval_count = max(1, int(math.ceil(len(shuffled) * eval_fraction)))
    eval_examples = shuffled[:eval_count]
    train_examples = shuffled[eval_count:] or shuffled
    return train_examples, eval_examples


def train_alfworld_policy_head_with_model(
    model: Any,
    model_config: dict[str, Any],
    routing_report: dict[str, Any],
    replay_path: str | Path,
    output_dir: str | Path,
    max_steps: int = 20000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    eval_fraction: float = 0.05,
    stop_loss_weight: float = 0.05,
    history_window: int = 6,
    feature_cache_encode_batch_size: int = 128,
) -> dict[str, Any]:
    replay_path = Path(replay_path)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    examples, skipped = load_policy_examples(replay_path, history_window=history_window)
    if not examples:
        raise ValueError(f"no usable ALFWorld train replay policy examples in {replay_path}")
    train_examples, eval_examples = _split_examples(examples, eval_fraction=eval_fraction, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(model, "to"):
        model.to(device)
    model.eval()
    train_cache = _build_policy_feature_cache(
        model,
        train_examples,
        encode_batch_size=feature_cache_encode_batch_size,
    )
    eval_cache = _build_policy_feature_cache(
        model,
        eval_examples,
        encode_batch_size=feature_cache_encode_batch_size,
    )
    d = int(model_config.get("d") or train_cache.state_embeddings.size(-1))
    action_adapter = UniversalActionAdapter(d=d, hidden_dim=d).to(device)
    trainable_modules = _freeze_routing_foundation_for_policy(model, stop_loss_weight=stop_loss_weight)
    routing_init_baseline_eval_report = _evaluate_policy_cache(
        model,
        None,
        eval_cache,
        device,
        batch_size=batch_size,
    )
    params = list(action_adapter.parameters())
    if stop_loss_weight > 0 and getattr(model, "stop_head", None) is not None:
        params.extend(param for param in model.stop_head.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    metrics: dict[str, float] = {}
    model.eval()
    action_adapter.train()
    for step_idx in range(1, max(1, int(max_steps)) + 1):
        start = ((step_idx - 1) * max(1, batch_size)) % len(train_examples)
        indices = [(start + offset) % len(train_examples) for offset in range(max(1, batch_size))]
        state_embs, candidate_embs, candidate_mask, labels, done_labels = train_cache.select(indices, device)
        scores = action_adapter(state_embs, candidate_embs, candidate_mask)
        policy_loss = F.cross_entropy(scores, labels)
        stop = _stop_logits(model, state_embs)
        stop_loss = F.binary_cross_entropy_with_logits(stop, done_labels)
        loss = policy_loss + float(stop_loss_weight) * stop_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(action_adapter.parameters(), 1.0)
        if stop_loss_weight > 0 and getattr(model, "stop_head", None) is not None:
            torch.nn.utils.clip_grad_norm_(model.stop_head.parameters(), 1.0)
        optimizer.step()
        if step_idx == max_steps or step_idx == 1:
            metrics = {
                "loss": float(loss.detach().cpu().item()),
                "policy_ce_loss": float(policy_loss.detach().cpu().item()),
                "stop_bce_loss": float(stop_loss.detach().cpu().item()),
                **compute_policy_diagnostic_metrics(scores.detach(), labels.detach(), done_labels, stop.detach()),
            }
        if max_steps >= 1000 and (step_idx == 1 or step_idx % max(1000, max_steps // 20) == 0):
            print(
                f"[alfworld_policy_train] step {step_idx}/{max_steps} loss={metrics.get('loss', 0.0):.4f}",
                file=sys.stderr,
                flush=True,
            )

    eval_report = _evaluate_policy_cache(model, action_adapter, eval_cache, device, batch_size=batch_size)
    baseline_metrics = routing_init_baseline_eval_report.get("metrics", {})
    eval_metrics = eval_report.get("metrics", {})
    delta_vs_baseline = {
        key: round(float(eval_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0)), 6)
        for key in ("expert_action_recall@1", "expert_action_recall@5", "expert_action_mrr", "stop_accuracy")
    }
    eval_report.update(
        {
            "training_objective": "L_policy_admissible_action_ce",
            "training_data": "ALFWorld train replay only",
            "routing_init_baseline": routing_init_baseline_eval_report,
            "delta_vs_routing_init_baseline": delta_vs_baseline,
            "gate_pass_offline": delta_vs_baseline.get("expert_action_recall@1", 0.0) > 0.0,
            "not_closed_loop_success": True,
        }
    )
    write_json(output_dir / "eval_report.json", eval_report)
    checkpoint_path = checkpoint_dir / f"alfworld_policy-step{max_steps}.pt"
    torch.save(
        {
            "stage": "alfworld_policy_l_policy",
            "step": max_steps,
            "config": model_config,
            "training_objective": "L_policy_admissible_action_ce",
            "training_data": "ALFWorld train replay only",
            "frozen_routing_foundation": True,
            "qdoc_adapter_used": False,
            "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
            "universal_action_adapter_state_dict": action_adapter.state_dict(),
            "trainable_modules": trainable_modules,
            "routing_init": routing_report,
            "metrics": metrics,
            "eval_metrics": eval_report.get("metrics", {}),
            "routing_init_baseline_eval_metrics": baseline_metrics,
            "delta_vs_routing_init_baseline": delta_vs_baseline,
        },
        checkpoint_path,
    )
    report = {
        "status": "ok",
        "training_objective": "L_policy_admissible_action_ce",
        "training_data": "ALFWorld train replay only",
        "checkpoint": str(checkpoint_path),
        "eval_report": str(output_dir / "eval_report.json"),
        "replay_path": str(replay_path),
        "frozen_routing_foundation": True,
        "frozen_modules": [
            "encoder.backbone",
            "encoder.proj",
            "skill_table.W",
            "skill_table.E",
        ],
        "qdoc_adapter_used": False,
        "trainable_modules": trainable_modules,
        "policy_sample_count": len(examples),
        "train_sample_count": len(train_examples),
        "eval_sample_count": len(eval_examples),
        "candidate_length_stats": _candidate_length_stats(examples),
        "expert_action_coverage": round(
            float(len(examples) / max(1, len(examples) + sum(value for key, value in skipped.items() if key != "non_train_split"))),
            6,
        ),
        "skipped_reasons": skipped,
        "uses_alfworld_valid_or_test_for_training": False,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history_window": history_window,
        "stop_loss_weight": stop_loss_weight,
        "feature_cache": {
            "used": True,
            "train_state_rows": int(train_cache.state_embeddings.size(0)),
            "eval_state_rows": int(eval_cache.state_embeddings.size(0)),
            "candidate_width": int(train_cache.candidate_embeddings.size(1)),
            "encode_batch_size": feature_cache_encode_batch_size,
        },
        "metrics": metrics,
        "eval_metrics": eval_report.get("metrics", {}),
        "routing_init_baseline_eval_metrics": baseline_metrics,
        "delta_vs_routing_init_baseline": delta_vs_baseline,
        "gate_pass_offline": delta_vs_baseline.get("expert_action_recall@1", 0.0) > 0.0,
        "routing_init": routing_report,
        "gpu": _gpu_report(device),
        "not_rl_fine_tuning": True,
    }
    write_json(output_dir / "train_report.json", report)
    return report


def run_alfworld_policy_train(
    replay_path: str | Path,
    output_dir: str | Path,
    routing_init_manifest: str | Path = "outputs/clstr_native_routing_init/manifest.json",
    aux_data_root: str | Path = "data/aux_trajectories",
    max_steps: int = 20000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    eval_fraction: float = 0.05,
    stop_loss_weight: float = 0.05,
    history_window: int = 6,
    feature_cache_encode_batch_size: int = 128,
) -> dict[str, Any]:
    skill_rows = load_aux_training_rows(Path(aux_data_root))[0]
    model, model_config, routing_report = _build_model_from_routing_init(
        routing_init_manifest,
        skill_rows,
        Path(output_dir) / "model_cache",
    )
    return train_alfworld_policy_head_with_model(
        model=model,
        model_config=model_config,
        routing_report=routing_report,
        replay_path=replay_path,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        eval_fraction=eval_fraction,
        stop_loss_weight=stop_loss_weight,
        history_window=history_window,
        feature_cache_encode_batch_size=feature_cache_encode_batch_size,
    )
