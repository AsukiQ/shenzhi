from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.encoders import _resolve_torch_dtype
from clstr.matched_baseline_corpus import (
    _stable_stratified_query_cap,
    load_prepared_matched_baseline_corpus,
)
from clstr.skillret_official import _pool_hf_hidden, _skillrouter_skill_text
from clstr.skillrouter_style import _query_text_for_mode
from clstr.training_monitor import TrainingMonitor
from clstr.unified_skillrouter_finetune import (
    UnifiedSkillRouterQuery,
    load_unified_skillrouter_training_corpus,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resume_training_contract(
    resume_state: dict[str, Any] | None,
    current_contract: dict[str, Any],
) -> str:
    if resume_state is None:
        return "not_applicable"
    saved_contract = resume_state.get("training_contract")
    if saved_contract is None:
        return "legacy_unverified"
    if saved_contract != current_contract:
        raise ValueError(
            "resume checkpoint training contract differs from the current corpus or objective"
        )
    return "verified"


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(resume_state: dict[str, Any] | None) -> str:
    if resume_state is None:
        return "not_applicable"
    rng_state = resume_state.get("rng_state")
    if not isinstance(rng_state, dict):
        return "legacy_unverified"
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch_cpu"])
    cuda_states = rng_state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("resume checkpoint CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_states)
    return "restored"


def _prune_old_checkpoints(
    checkpoint_dir: str | Path,
    *,
    latest_path: str | Path,
    max_checkpoints_to_keep: int,
) -> dict[str, Any]:
    keep = int(max_checkpoints_to_keep)
    if keep <= 0:
        return {"enabled": False, "max_checkpoints_to_keep": keep, "removed": []}
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = Path(latest_path).resolve()
    candidates = [path for path in checkpoint_dir.iterdir() if path.is_dir()]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    kept: list[Path] = []
    removed: list[str] = []
    for path in candidates:
        if path.resolve() == latest_path:
            kept.append(path)
            continue
        if len(kept) < keep:
            kept.append(path)
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return {
        "enabled": True,
        "max_checkpoints_to_keep": keep,
        "kept": [str(path) for path in kept],
        "removed": removed,
    }


def _build_embedding_training_monitor(
    output_dir: str | Path,
    *,
    checkpoint_dir: str | Path,
    checkpoint_every: int,
    reset: bool = True,
) -> TrainingMonitor:
    monitor = TrainingMonitor(
        output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_steps=max(1, int(checkpoint_every)),
        curve_interval_steps=max(1, int(checkpoint_every)),
        reset=reset,
    )
    if not reset and monitor.metrics_path.exists():
        history: list[dict[str, Any]] = []
        with monitor.metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                history.append(json.loads(line))
        monitor.history = history
        monitor._write_loss_curve()
        monitor._write_diagnostic_curves()
    return monitor


def _load_embedding_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if path.is_dir() and (path / "latest_checkpoint.json").exists():
        latest = json.loads((path / "latest_checkpoint.json").read_text(encoding="utf-8"))
        path = Path(latest["path"])
    elif path.is_dir() and (path / "latest.json").exists():
        latest = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        path = Path(latest["path"])
    state_path = path / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"resume training_state.pt not found under {path}")
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"resume state must be a dict: {state_path}")
    return {
        **state,
        "checkpoint_path": str(path),
        "training_state_path": str(state_path),
        "step": int(state.get("step") or 0),
    }


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _query_to_dict(query: UnifiedSkillRouterQuery) -> dict[str, Any]:
    row = {
        "query_id": query.query_id,
        "query": query.query,
        "benchmark": query.benchmark,
        "positive_skill_ids": list(query.positive_skill_ids),
        "positive_indices": list(query.positive_indices),
    }
    if query.candidate_skill_ids is not None:
        row["candidate_skill_ids"] = list(query.candidate_skill_ids)
    for key in (
        "source_id",
        "kind",
        "split_group_identity",
        "runtime_visible_catalog_id",
        "inventory_catalog_digest",
        "history_mode",
    ):
        value = getattr(query, key)
        if value:
            row[key] = value
    return row


def _write_embedding_progress(
    output_dir: str | Path,
    *,
    stage: str,
    step: int | None = None,
    max_steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "running",
        "stage": str(stage),
        "updated_at_unix": time.time(),
    }
    if step is not None:
        payload["step"] = int(step)
    if max_steps is not None:
        payload["max_steps"] = int(max_steps)
        if step is not None and int(max_steps) > 0:
            payload["progress_fraction"] = min(1.0, max(0.0, int(step) / int(max_steps)))
    if extra:
        payload.update(extra)
    _write_json(Path(output_dir) / "progress.json", payload)
    return payload


def _ensure_finite_loss(loss: torch.Tensor, *, output_dir: str | Path, step: int) -> None:
    _ensure_finite_tensor(loss, output_dir=output_dir, step=step, name="loss")


def _ensure_finite_tensor(tensor: torch.Tensor, *, output_dir: str | Path, step: int, name: str) -> None:
    if torch.isfinite(tensor.detach()).all():
        return
    reason = f"non-finite {name}"
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    blocker = {
        "status": "blocked",
        "reason": reason,
        "step": int(step),
        "tensor_name": str(name),
        "finite_count": int(finite.numel()),
        "total_count": int(detached.numel()),
        "finite_min": float(finite.min().cpu().item()) if finite.numel() else None,
        "finite_max": float(finite.max().cpu().item()) if finite.numel() else None,
        "action": "stop training and fix numerical stability before resubmitting",
    }
    _write_json(Path(output_dir) / "blocker_report.json", blocker)
    _write_embedding_progress(
        output_dir,
        stage="blocked",
        step=step,
        extra={"status": "blocked", "reason": reason},
    )
    raise FloatingPointError(f"{reason} at step {step}")


def _ensure_finite_gradients(
    model: torch.nn.Module,
    *,
    output_dir: str | Path,
    step: int,
    action: str = "error",
    max_nonfinite_fraction: float = 0.0,
) -> dict[str, Any]:
    action = str(action or "error")
    if action not in {"error", "zero"}:
        raise ValueError(f"unsupported nonfinite gradient action: {action}")
    total_nonfinite = 0
    total_sanitized = 0
    affected: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite_mask = torch.isfinite(grad)
        if finite_mask.all():
            continue
        nonfinite_count = int((~finite_mask).sum().cpu().item())
        total_count = int(grad.numel())
        fraction = nonfinite_count / max(total_count, 1)
        total_nonfinite += nonfinite_count
        affected.append(
            {
                "parameter_name": str(name),
                "nonfinite_count": nonfinite_count,
                "total_count": total_count,
                "nonfinite_fraction": float(fraction),
            }
        )
        if action == "zero" and fraction <= float(max_nonfinite_fraction):
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            total_sanitized += nonfinite_count
            continue
        reason = "non-finite gradient"
        blocker = {
            "status": "blocked",
            "reason": reason,
            "step": int(step),
            "parameter_name": str(name),
            "nonfinite_count": nonfinite_count,
            "total_count": total_count,
            "nonfinite_fraction": float(fraction),
            "max_nonfinite_fraction": float(max_nonfinite_fraction),
            "action": "stop training and fix numerical stability before resubmitting",
            "affected_parameters": affected,
        }
        _write_json(Path(output_dir) / "blocker_report.json", blocker)
        _write_embedding_progress(
            output_dir,
            stage="blocked",
            step=step,
            extra={"status": "blocked", "reason": reason, "parameter_name": str(name)},
        )
        raise FloatingPointError(f"{reason} in {name} at step {step}")
    return {
        "nonfinite_gradient_action": action,
        "nonfinite_gradient_count": int(total_nonfinite),
        "sanitized_gradient_count": int(total_sanitized),
        "affected_gradient_parameters": affected,
    }


def _copy_bge_pooling_config(source_model_dir: str | Path, target_dir: str | Path) -> bool:
    source_path = Path(source_model_dir) / "1_Pooling" / "config.json"
    if not source_path.exists():
        return False
    target_path = Path(target_dir) / "1_Pooling" / "config.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def _skill_id(skill: dict[str, Any]) -> str:
    return str(skill.get("skill_id") or skill.get("id") or "").strip()


def _load_bge_encoder(model_name_or_path: str | Path, torch_dtype: str) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    dtype = _resolve_torch_dtype(torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(str(model_name_or_path), **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def _encode_texts_trainable(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    pooling: str = "cls",
    use_bf16_autocast: bool = False,
) -> torch.Tensor:
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    tokens = {key: value.to(device) for key, value in tokens.items()}
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_autocast and device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        out = model(**tokens)
    embs = _pool_hf_hidden(out.last_hidden_state, tokens["attention_mask"], pooling)
    return F.normalize(embs.float(), p=2, dim=-1)


@torch.no_grad()
def _encode_texts_in_batches(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    pooling: str = "cls",
    use_bf16_autocast: bool = False,
) -> torch.Tensor:
    model.eval()
    encoded: list[torch.Tensor] = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        embs = _encode_texts_trainable(
            model,
            tokenizer,
            texts[start : start + max(1, int(batch_size))],
            max_length=max_length,
            pooling=pooling,
            use_bf16_autocast=use_bf16_autocast,
        )
        encoded.append(embs.detach().cpu())
    if not encoded:
        hidden = int(model.config.hidden_size)
        return torch.empty(0, hidden)
    return torch.cat(encoded, dim=0)


def _mine_hard_negative_indices(
    scores: torch.Tensor,
    positive_indices_by_query: list[set[int]],
    *,
    top_k: int,
    legal_indices_by_query: list[set[int] | None] | None = None,
) -> list[list[int]]:
    if legal_indices_by_query is not None:
        if len(legal_indices_by_query) != scores.size(0):
            raise ValueError("legal candidate rows do not match score rows")
        mined: list[list[int]] = [[] for _ in range(scores.size(0))]
        rows_by_catalog: dict[tuple[int, ...] | None, list[int]] = {}
        for row_idx, legal in enumerate(legal_indices_by_query):
            legal_tuple = (
                None
                if legal is None
                else tuple(
                    sorted(
                        int(index)
                        for index in legal
                        if 0 <= int(index) < scores.size(-1)
                    )
                )
            )
            rows_by_catalog.setdefault(legal_tuple, []).append(row_idx)
        for catalog_indices, row_indices in rows_by_catalog.items():
            row_tensor = torch.tensor(
                row_indices,
                dtype=torch.long,
                device=scores.device,
            )
            if catalog_indices is None:
                legal_tuple = range(scores.size(-1))
                catalog_scores = scores.index_select(0, row_tensor)
                local_index = None
            else:
                legal_tuple = catalog_indices
                if not legal_tuple:
                    continue
                legal_tensor = torch.tensor(
                    legal_tuple,
                    dtype=torch.long,
                    device=scores.device,
                )
                catalog_scores = scores.index_select(0, row_tensor).index_select(
                    1,
                    legal_tensor,
                )
                local_index = {
                    skill_index: index for index, skill_index in enumerate(legal_tuple)
                }
            positive_mask = torch.zeros_like(catalog_scores, dtype=torch.bool)
            for local_row, row_idx in enumerate(row_indices):
                positions = [
                    index if local_index is None else local_index[index]
                    for index in positive_indices_by_query[row_idx]
                    if local_index is None or index in local_index
                ]
                if positions:
                    positive_mask[
                        local_row,
                        torch.tensor(positions, dtype=torch.long, device=scores.device),
                    ] = True
            count = min(max(0, int(top_k)), len(legal_tuple))
            masked = catalog_scores.masked_fill(
                positive_mask,
                torch.finfo(catalog_scores.dtype).min,
            )
            selected_local = torch.topk(masked, k=count, dim=-1).indices.cpu().tolist()
            for local_row, selected in enumerate(selected_local):
                positives = positive_indices_by_query[row_indices[local_row]]
                mined[row_indices[local_row]] = [
                    legal_tuple[index]
                    for index in selected
                    if legal_tuple[index] not in positives
                ][: max(0, int(top_k))]
        return mined
    mined: list[list[int]] = []
    width = scores.size(-1)
    extra = max((len(items) for items in positive_indices_by_query), default=0) + 8
    shortlist_k = min(width, max(1, int(top_k) + extra))
    order = torch.topk(scores, k=shortlist_k, dim=-1).indices.cpu().tolist()
    for row_idx, ranked in enumerate(order):
        positives = positive_indices_by_query[row_idx]
        selected = [int(idx) for idx in ranked if int(idx) not in positives]
        if len(selected) < int(top_k) and shortlist_k < width:
            full_order = torch.argsort(scores[row_idx], descending=True).cpu().tolist()
            selected = [int(idx) for idx in full_order if int(idx) not in positives]
        mined.append(selected[: max(0, int(top_k))])
    return mined


def _build_semantic_hard_negatives(
    *,
    model,
    tokenizer,
    skills: list[dict[str, Any]],
    queries: list[UnifiedSkillRouterQuery],
    max_queries: int | None,
    top_k: int,
    encode_batch_size: int,
    score_batch_size: int,
    max_length: int,
    output_dir: Path,
    use_bf16_autocast: bool,
    source_balanced_selection: bool = False,
    selection_seed: int = 13,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    selected_queries, selection_report = _select_semantic_hard_negative_queries(
        queries,
        max_queries=max_queries,
        source_balanced=source_balanced_selection,
        seed=selection_seed,
    )
    skill_ids = [_skill_id(skill) for skill in skills]
    skill_id_to_idx = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    skill_texts = [_skillrouter_skill_text(skill) for skill in skills]
    query_texts = [_query_text_for_mode({"query": query.query}, "raw") for query in selected_queries]
    _write_embedding_progress(
        output_dir,
        stage="mining_encode_skills",
        extra={"skill_count": len(skill_texts), "query_count": len(selected_queries)},
    )
    skill_embs = _encode_texts_in_batches(
        model,
        tokenizer,
        skill_texts,
        batch_size=encode_batch_size,
        max_length=max_length,
        pooling="cls",
        use_bf16_autocast=use_bf16_autocast,
    )
    _write_embedding_progress(
        output_dir,
        stage="mining_encode_queries",
        extra={"skill_count": len(skill_texts), "query_count": len(selected_queries)},
    )
    query_embs = _encode_texts_in_batches(
        model,
        tokenizer,
        query_texts,
        batch_size=encode_batch_size,
        max_length=max_length,
        pooling="cls",
        use_bf16_autocast=use_bf16_autocast,
    )
    hard: dict[str, list[str]] = {}
    score_device = next(model.parameters()).device
    skill_embs_for_score = skill_embs.to(score_device)
    for start in range(0, len(selected_queries), max(1, int(score_batch_size))):
        end = min(len(selected_queries), start + max(1, int(score_batch_size)))
        scores = query_embs[start:end].to(score_device) @ skill_embs_for_score.T
        positives = [set(query.positive_indices) for query in selected_queries[start:end]]
        selected_batch = selected_queries[start:end]
        use_legal_catalogs = any(query.candidate_skill_ids is not None for query in selected_batch)
        legal_indices = (
            [
                (
                    None
                    if query.candidate_skill_ids is None
                    else {
                        skill_id_to_idx[skill_id]
                        for skill_id in query.candidate_skill_ids
                        if skill_id in skill_id_to_idx
                    }
                )
                for query in selected_batch
            ]
            if use_legal_catalogs
            else None
        )
        mined_indices = _mine_hard_negative_indices(
            scores,
            positives,
            top_k=top_k,
            legal_indices_by_query=legal_indices,
        )
        for query, indices in zip(selected_queries[start:end], mined_indices):
            hard[query.query_id] = [skill_ids[idx] for idx in indices if 0 <= idx < len(skill_ids)]
        if start == 0 or end == len(selected_queries) or end % (max(1, int(score_batch_size)) * 10) == 0:
            _write_embedding_progress(
                output_dir,
                stage="mining_rank_queries",
                extra={"ranked_queries": end, "query_count": len(selected_queries)},
            )
    _write_json(output_dir / "hard_negatives_by_query.json", hard)
    return hard, selection_report


def _select_semantic_hard_negative_queries(
    queries: list[UnifiedSkillRouterQuery],
    *,
    max_queries: int | None,
    source_balanced: bool,
    seed: int,
) -> tuple[list[UnifiedSkillRouterQuery], dict[str, Any]]:
    if max_queries is not None and int(max_queries) <= 0:
        selected: list[UnifiedSkillRouterQuery] = []
    elif source_balanced:
        selected = _stable_stratified_query_cap(
            queries,
            max_queries,
            seed=int(seed),
        )
    else:
        selected = (
            list(queries)
            if max_queries is None
            else list(queries[: int(max_queries)])
        )
    protocol = (
        "capability_first_source_round_robin_cap_v2"
        if source_balanced
        else "input_prefix_v1"
    )
    return selected, {
        "protocol": protocol,
        "seed": int(seed) if source_balanced else None,
        "source_query_count": len(queries),
        "selected_query_count": len(selected),
        "by_kind_source": dict(
            sorted(
                Counter(
                    f"{str(query.kind)}:{str(query.source_id)}"
                    for query in selected
                ).items()
            )
        ),
    }


def _should_build_semantic_hard_negatives(
    *,
    hard_negative_top_k: int,
    max_hard_negative_queries: int | None,
) -> bool:
    if int(hard_negative_top_k) <= 0:
        return False
    if max_hard_negative_queries is not None and int(max_hard_negative_queries) <= 0:
        return False
    return True


def _validate_hard_negative_cache(
    value: Any,
    *,
    queries: list[UnifiedSkillRouterQuery],
    all_skill_ids: set[str],
    top_k: int,
) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError("hard-negative cache must be a query-to-skill mapping")
    query_by_id = {query.query_id: query for query in queries}
    validated: dict[str, list[str]] = {}
    for raw_query_id, raw_skill_ids in value.items():
        query_id = str(raw_query_id)
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"hard-negative cache references an unknown query: {query_id}")
        if not isinstance(raw_skill_ids, list):
            raise ValueError(f"hard-negative cache row is not a list: {query_id}")
        skill_ids = [str(skill_id).strip() for skill_id in raw_skill_ids]
        if (
            any(not skill_id for skill_id in skill_ids)
            or len(skill_ids) != len(set(skill_ids))
            or len(skill_ids) > max(0, int(top_k))
        ):
            raise ValueError(f"hard-negative cache row is invalid: {query_id}")
        legal = (
            all_skill_ids
            if query.candidate_skill_ids is None
            else set(query.candidate_skill_ids)
        )
        forbidden = set(query.positive_skill_ids)
        if any(
            skill_id not in all_skill_ids
            or skill_id not in legal
            or skill_id in forbidden
            for skill_id in skill_ids
        ):
            raise ValueError(
                f"hard-negative cache row violates the query catalog: {query_id}"
            )
        validated[query_id] = skill_ids
    return validated


def sample_negative_skill_ids(
    *,
    query_id: str,
    positive_skill_ids: set[str],
    all_skill_ids: list[str],
    candidate_skill_ids: list[str] | None = None,
    all_skill_id_set: set[str] | None = None,
    hard_negatives_by_query: dict[str, list[str]],
    count: int,
    rng_seed: int,
) -> list[str]:
    rng = random.Random(rng_seed)
    target_count = max(0, int(count))
    if target_count == 0:
        return []
    known_skill_ids = all_skill_id_set or set(all_skill_ids)
    allowed = None if candidate_skill_ids is None else set(candidate_skill_ids)
    selected: list[str] = []
    selected_set: set[str] = set()
    for skill_id in hard_negatives_by_query.get(query_id, []):
        if (
            (allowed is not None and skill_id not in allowed)
            or skill_id not in known_skill_ids
            or skill_id in positive_skill_ids
            or skill_id in selected_set
        ):
            continue
        selected.append(skill_id)
        selected_set.add(skill_id)
        if len(selected) >= target_count:
            return selected
    if allowed is not None:
        candidates = list(
            dict.fromkeys(
                skill_id
                for skill_id in candidate_skill_ids or []
                if skill_id in known_skill_ids
                and skill_id not in positive_skill_ids
                and skill_id not in selected_set
            )
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: max(0, target_count - len(selected))])
        return selected
    attempts = 0
    max_attempts = max(100, max(0, target_count - len(selected)) * 20)
    while len(selected) < target_count and attempts < max_attempts and all_skill_ids:
        attempts += 1
        skill_id = all_skill_ids[rng.randrange(len(all_skill_ids))]
        if skill_id in positive_skill_ids or skill_id in selected_set:
            continue
        selected.append(skill_id)
        selected_set.add(skill_id)
    if len(selected) < target_count:
        candidates = [
            skill_id
            for skill_id in all_skill_ids
            if skill_id not in positive_skill_ids and skill_id not in selected_set
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[: max(0, target_count - len(selected))])
    return selected


def _multi_positive_legal_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != positive_mask.shape or logits.shape != legal_mask.shape:
        raise ValueError("in-batch logits and masks must have identical shapes")
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("every in-batch query requires at least one positive document")
    if torch.any(positive_mask & ~legal_mask):
        raise ValueError("in-batch positive document is outside the legal catalog")
    min_value = torch.finfo(logits.dtype).min
    legal_logits = logits.masked_fill(~legal_mask, min_value)
    positive_logits = legal_logits.masked_fill(~positive_mask, min_value)
    return (
        torch.logsumexp(legal_logits.float(), dim=-1)
        - torch.logsumexp(positive_logits.float(), dim=-1)
    ).mean()


def _ranking_metrics(logits: torch.Tensor, positive_mask: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(logits, dim=-1, descending=True)
    positives_ranked = positive_mask.gather(1, order)
    ranks = positives_ranked.float().argmax(dim=-1).float() + 1
    has_positive = positives_ranked.any(dim=-1)
    ranks = torch.where(
        has_positive,
        ranks,
        torch.full_like(ranks, logits.size(-1) + 1),
    )
    return {
        "recall@1": float((ranks <= 1).float().mean().detach().cpu().item()),
        "recall@5": float((ranks <= min(5, logits.size(-1))).float().mean().detach().cpu().item()),
        "mrr": float((1.0 / ranks).mean().detach().cpu().item()),
    }


def _batch_from_queries(
    *,
    queries: list[UnifiedSkillRouterQuery],
    skills: list[dict[str, Any]],
    skill_id_to_skill: dict[str, dict[str, Any]],
    all_skill_ids: list[str],
    hard_negatives_by_query: dict[str, list[str]],
    negatives_per_query: int,
    rng_seed: int,
    all_skill_id_set: set[str] | None = None,
) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor, dict[str, Any]]:
    query_texts: list[str] = []
    doc_texts: list[str] = []
    doc_skill_ids: list[str] = []
    usable_queries: list[UnifiedSkillRouterQuery] = []
    skipped: Counter[str] = Counter()
    rng = random.Random(rng_seed)
    for query in queries:
        positive_ids = [skill_id for skill_id in query.positive_skill_ids if skill_id in skill_id_to_skill]
        if not positive_ids:
            skipped["missing_positive_skill"] += 1
            continue
        positive_id = rng.choice(positive_ids)
        usable_queries.append(query)
        query_texts.append(_query_text_for_mode({"query": query.query}, "raw"))
        doc_texts.append(_skillrouter_skill_text(skill_id_to_skill[positive_id]))
        doc_skill_ids.append(positive_id)
    for query in usable_queries:
        positive_ids = {skill_id for skill_id in query.positive_skill_ids if skill_id in skill_id_to_skill}
        for skill_id in sample_negative_skill_ids(
            query_id=query.query_id,
            positive_skill_ids=positive_ids,
            all_skill_ids=all_skill_ids,
            candidate_skill_ids=query.candidate_skill_ids,
            all_skill_id_set=all_skill_id_set,
            hard_negatives_by_query=hard_negatives_by_query,
            count=negatives_per_query,
            rng_seed=rng_seed + len(doc_texts),
        ):
            if skill_id in skill_id_to_skill:
                doc_texts.append(_skillrouter_skill_text(skill_id_to_skill[skill_id]))
                doc_skill_ids.append(skill_id)
    positive_mask = torch.zeros(
        len(usable_queries),
        len(doc_skill_ids),
        dtype=torch.bool,
    )
    legal_mask = torch.zeros_like(positive_mask)
    if all_skill_id_set is None:
        all_skill_id_set = set(all_skill_ids)
    for row_index, query in enumerate(usable_queries):
        positives = set(query.positive_skill_ids)
        legal = (
            all_skill_id_set
            if query.candidate_skill_ids is None
            else set(query.candidate_skill_ids)
        )
        for column_index, skill_id in enumerate(doc_skill_ids):
            positive_mask[row_index, column_index] = skill_id in positives
            legal_mask[row_index, column_index] = skill_id in legal
    return query_texts, doc_texts, positive_mask, legal_mask, {
        "skipped_reasons": dict(skipped),
        "query_count": len(usable_queries),
        "document_count": len(doc_skill_ids),
    }


def _prepared_sampling_groups(
    queries: list[UnifiedSkillRouterQuery],
) -> dict[str, dict[str, list[UnifiedSkillRouterQuery]]]:
    grouped: dict[str, dict[str, list[UnifiedSkillRouterQuery]]] = {}
    for query in queries:
        kind = str(query.kind)
        source = str(query.source_id)
        if kind not in {"retrieval", "static_route"} or not source:
            raise ValueError("prepared matched query lacks a sampling kind/source")
        grouped.setdefault(kind, {}).setdefault(source, []).append(query)
    if not grouped:
        raise ValueError("prepared matched corpus has no source-balanced groups")
    return grouped


def _prepared_source_balanced_batch(
    groups: dict[str, dict[str, list[UnifiedSkillRouterQuery]]],
    *,
    global_batch_index: int,
    batch_size: int,
    seed: int,
) -> list[UnifiedSkillRouterQuery]:
    kinds = [kind for kind in ("retrieval", "static_route") if kind in groups]
    if not kinds:
        raise ValueError("prepared sampling groups have no supported capability")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("prepared batch size must be positive")
    if len(kinds) == 2:
        base_count = batch_size // 2
        extra_kind_index = int(global_batch_index) % 2
        counts = {
            kind: base_count + int(batch_size % 2 == 1 and index == extra_kind_index)
            for index, kind in enumerate(kinds)
        }
        prior_allocations = {
            kinds[0]: int(global_batch_index) * base_count
            + (
                (int(global_batch_index) + 1) // 2
                if batch_size % 2 == 1
                else 0
            ),
            kinds[1]: int(global_batch_index) * base_count
            + (
                int(global_batch_index) // 2
                if batch_size % 2 == 1
                else 0
            ),
        }
    else:
        counts = {kinds[0]: batch_size}
        prior_allocations = {kinds[0]: int(global_batch_index) * batch_size}
    batch: list[UnifiedSkillRouterQuery] = []
    for kind_index, kind in enumerate(kinds):
        sources = sorted(groups[kind])
        count = counts[kind]
        for offset in range(count):
            source_position = (prior_allocations[kind] + offset) % len(sources)
            source = sources[source_position]
            rows = groups[kind][source]
            rng = random.Random(
                int(seed)
                + int(global_batch_index) * 1009
                + kind_index * 104729
                + offset
            )
            batch.append(rng.choice(rows))
    random.Random(int(seed) + int(global_batch_index) * 65537).shuffle(batch)
    return batch


def _save_full_finetune_checkpoint(
    *,
    model,
    tokenizer,
    output_dir: Path,
    base_model_dir: str | Path,
    checkpoint_dir: Path,
    step: int,
    payload: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    checkpoint_prefix: str = "bge-m3-sr-emb",
    max_checkpoints_to_keep: int = 2,
) -> Path:
    path = checkpoint_dir / f"{checkpoint_prefix}-step{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    _copy_bge_pooling_config(base_model_dir, path)
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
            **payload,
        },
        path / "training_state.pt",
    )
    _write_json(path / "train_checkpoint_report.json", payload)
    _write_json(
        checkpoint_dir / "latest.json",
        {"path": str(path.resolve()), "step": int(step)},
    )
    _write_json(
        output_dir / "latest_checkpoint.json",
        {"path": str(path.resolve()), "step": int(step)},
    )
    _write_json(
        checkpoint_dir / "retention_report.json",
        _prune_old_checkpoints(
            checkpoint_dir,
            latest_path=path,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
        ),
    )
    return path


def run_bge_sr_embedding_full_train(
    *,
    data_root: str | Path | None,
    output_dir: str | Path,
    prepared_corpus_dir: str | Path | None = None,
    encoder_model_path: str = "models/BAAI/bge-m3",
    max_rows: int | None = None,
    max_skills: int | None = None,
    eval_rows: int = 2048,
    max_steps: int = 100,
    batch_size: int = 8,
    negatives_per_query: int = 7,
    hard_negative_top_k: int = 32,
    max_hard_negative_queries: int | None = 20000,
    learning_rate: float = 2.0e-5,
    temperature: float = 0.05,
    seed: int = 13,
    max_length: int = 512,
    encode_batch_size: int = 64,
    mine_score_batch_size: int = 128,
    torch_dtype: str = "bfloat16",
    eval_every: int = 100,
    log_every: int = 10,
    checkpoint_every: int = 400,
    max_checkpoints_to_keep: int = 2,
    use_bf16_autocast: bool = True,
    mining_use_bf16_autocast: bool | None = None,
    gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,
    resume_checkpoint_path: str | Path | None = None,
    nonfinite_gradient_action: str = "error",
    max_nonfinite_gradient_fraction: float = 0.0,
    method_name: str = "bge_sr_emb_full_encoder_finetune",
    checkpoint_prefix: str = "bge-m3-sr-emb",
    training_objective: str = "multi_positive_inbatch_infonce_with_semantic_hard_negatives",
    doc_template: str = "skillret_official_full_text",
    log_prefix: str = "bge-sr-emb-full",
) -> dict[str, Any]:
    if float(learning_rate) <= 0:
        raise ValueError("embedding learning_rate must be positive")
    if float(temperature) <= 0:
        raise ValueError("embedding temperature must be positive")
    if int(max_length) <= 0:
        raise ValueError("embedding max_length must be positive")
    if int(negatives_per_query) < 0:
        raise ValueError("embedding negatives_per_query must be nonnegative")
    batch_size = max(1, int(batch_size))
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_state = _load_embedding_resume_state(resume_checkpoint_path) if resume_checkpoint_path is not None else None
    monitor = _build_embedding_training_monitor(
        output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=max(1, int(checkpoint_every)),
        reset=resume_state is None,
    )
    if mining_use_bf16_autocast is None:
        mining_use_bf16_autocast = use_bf16_autocast
    _write_embedding_progress(
        output_dir,
        stage="loading_data",
        extra={
            "data_root": None if data_root is None else str(data_root),
            "prepared_corpus_dir": (
                None if prepared_corpus_dir is None else str(prepared_corpus_dir)
            ),
        },
    )
    if prepared_corpus_dir is not None:
        corpus = load_prepared_matched_baseline_corpus(
            prepared_corpus_dir,
            max_rows=max_rows,
            max_eval_rows=eval_rows,
            max_skills=max_skills,
            seed=seed,
        )
    else:
        if data_root is None:
            raise ValueError("data_root is required when prepared_corpus_dir is not set")
        corpus = load_unified_skillrouter_training_corpus(
            data_root,
            max_rows=max_rows,
            max_skills=max_skills,
            eval_rows=eval_rows,
            seed=seed,
        )
    _write_jsonl(output_dir / "selected_skills.jsonl", corpus.skills)
    _write_jsonl(
        output_dir / "train_queries.jsonl",
        (_query_to_dict(query) for query in corpus.train_queries),
    )
    _write_jsonl(
        output_dir / "eval_queries.jsonl",
        (_query_to_dict(query) for query in corpus.eval_queries),
    )
    _write_json(output_dir / "corpus_report.json", corpus.report)
    skill_id_to_skill = {
        _skill_id(skill): skill for skill in corpus.skills if _skill_id(skill)
    }
    all_skill_ids = list(skill_id_to_skill)
    all_skill_id_set = set(all_skill_ids)
    hard_negative_mining_enabled = _should_build_semantic_hard_negatives(
        hard_negative_top_k=hard_negative_top_k,
        max_hard_negative_queries=max_hard_negative_queries,
    )
    hard_negative_query_selection_protocol = (
        "disabled"
        if not hard_negative_mining_enabled
        else (
            "capability_first_source_round_robin_cap_v2"
            if prepared_corpus_dir is not None
            else "input_prefix_v1"
        )
    )
    negative_sampling_protocol = (
        "hard_first_uniform_without_replacement_v3_direct_local_catalog"
        if prepared_corpus_dir is not None
        else "hard_first_uniform_without_replacement_v2"
    )
    training_contract = {
        "schema_version": "bge_embedding_training_contract_v1",
        "corpus": {
            "selected_skills_sha256": _file_sha256(output_dir / "selected_skills.jsonl"),
            "train_queries_sha256": _file_sha256(output_dir / "train_queries.jsonl"),
            "eval_queries_sha256": _file_sha256(output_dir / "eval_queries.jsonl"),
            "prepared_mode": prepared_corpus_dir is not None,
        },
        "encoder_model_path": str(Path(encoder_model_path).resolve()),
        "objective": {
            "training_objective": str(training_objective),
            "doc_template": str(doc_template),
            "temperature": float(temperature),
            "negatives_per_query": int(negatives_per_query),
            "negative_sampling_protocol": negative_sampling_protocol,
            "hard_negative_top_k": int(hard_negative_top_k),
            "max_hard_negative_queries": (
                None
                if max_hard_negative_queries is None
                else int(max_hard_negative_queries)
            ),
            "max_length": int(max_length),
        },
        "schedule": {
            "seed": int(seed),
            "micro_batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "learning_rate": float(learning_rate),
            "sampling_protocol": (
                "capability_source_round_robin_v1"
                if prepared_corpus_dir is not None
                else "legacy_seeded_row_order_v1"
            ),
        },
        "numerics": {
            "torch_dtype": str(torch_dtype),
            "use_bf16_autocast": bool(use_bf16_autocast),
            "mining_use_bf16_autocast": bool(mining_use_bf16_autocast),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "nonfinite_gradient_action": str(nonfinite_gradient_action),
            "max_nonfinite_gradient_fraction": float(max_nonfinite_gradient_fraction),
        },
    }
    if prepared_corpus_dir is not None:
        training_contract["objective"][
            "hard_negative_query_selection_protocol"
        ] = hard_negative_query_selection_protocol
    resume_contract_status = _validate_resume_training_contract(
        resume_state,
        training_contract,
    )
    prepared_groups = (
        _prepared_sampling_groups(corpus.train_queries)
        if prepared_corpus_dir is not None
        else None
    )
    _write_embedding_progress(
        output_dir,
        stage="loading_encoder",
        extra={"encoder_model_path": str(encoder_model_path), "corpus_report": corpus.report},
    )
    load_model_path = resume_state["checkpoint_path"] if resume_state is not None else encoder_model_path
    model, tokenizer, device = _load_bge_encoder(load_model_path, torch_dtype)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    hard_negative_cache_path = output_dir / "hard_negatives_by_query.json"
    hard_negative_report_path = output_dir / "hard_negative_report.json"
    hard_negative_contract = {
        "schema_version": "bge_embedding_hard_negative_contract_v1",
        "selected_skills_sha256": _file_sha256(output_dir / "selected_skills.jsonl"),
        "train_queries_sha256": _file_sha256(output_dir / "train_queries.jsonl"),
        "encoder_model_path": str(encoder_model_path),
        "hard_negative_top_k": int(hard_negative_top_k),
        "max_hard_negative_queries": (
            None
            if max_hard_negative_queries is None
            else int(max_hard_negative_queries)
        ),
        "max_length": int(max_length),
        "mining_use_bf16_autocast": bool(mining_use_bf16_autocast),
        "query_selection_protocol": hard_negative_query_selection_protocol,
        "query_selection_seed": (
            int(seed)
            if hard_negative_query_selection_protocol
            == "capability_first_source_round_robin_cap_v2"
            else None
        ),
    }
    if (
        resume_state is not None
        and prepared_corpus_dir is not None
        and (not hard_negative_cache_path.is_file() or not hard_negative_report_path.is_file())
    ):
        raise ValueError(
            "prepared matched resume requires the original hard-negative cache and report"
        )
    if resume_state is not None and hard_negative_cache_path.exists():
        cached_report = (
            json.loads(hard_negative_report_path.read_text(encoding="utf-8"))
            if hard_negative_report_path.is_file()
            else {}
        )
        if prepared_corpus_dir is not None and (
            cached_report.get("contract") != hard_negative_contract
            or cached_report.get("hard_negatives_sha256")
            != _file_sha256(hard_negative_cache_path)
        ):
            raise ValueError(
                "prepared matched resume hard-negative cache does not match the corpus contract"
            )
        hard_negatives = _validate_hard_negative_cache(
            json.loads(hard_negative_cache_path.read_text(encoding="utf-8")),
            queries=corpus.train_queries,
            all_skill_ids=all_skill_id_set,
            top_k=hard_negative_top_k,
        )
        _write_embedding_progress(
            output_dir,
            stage="hard_negative_cache_loaded",
            extra={"query_count": len(hard_negatives), "path": str(hard_negative_cache_path)},
        )
    elif not hard_negative_mining_enabled:
        hard_negatives = {}
        _write_json(hard_negative_cache_path, hard_negatives)
        _write_json(
            hard_negative_report_path,
            {
                "query_count": 0,
                "top_k": int(hard_negative_top_k),
                "enabled": False,
                "reason": "disabled_by_hard_negative_top_k_or_max_hard_negative_queries",
                "contract": hard_negative_contract,
                "hard_negatives_sha256": _file_sha256(hard_negative_cache_path),
            },
        )
        _write_embedding_progress(
            output_dir,
            stage="hard_negative_mining_skipped",
            extra={"hard_negative_top_k": int(hard_negative_top_k), "max_hard_negative_queries": max_hard_negative_queries},
        )
    else:
        built_hard_negatives, hard_negative_selection_report = (
            _build_semantic_hard_negatives(
                model=model,
                tokenizer=tokenizer,
                skills=corpus.skills,
                queries=corpus.train_queries,
                max_queries=max_hard_negative_queries,
                top_k=hard_negative_top_k,
                encode_batch_size=encode_batch_size,
                score_batch_size=mine_score_batch_size,
                max_length=max_length,
                output_dir=output_dir,
                use_bf16_autocast=bool(mining_use_bf16_autocast),
                source_balanced_selection=prepared_corpus_dir is not None,
                selection_seed=seed,
            )
        )
        hard_negatives = _validate_hard_negative_cache(
            built_hard_negatives,
            queries=corpus.train_queries,
            all_skill_ids=all_skill_id_set,
            top_k=hard_negative_top_k,
        )
        _write_json(
            hard_negative_report_path,
            {
                "query_count": len(hard_negatives),
                "top_k": int(hard_negative_top_k),
                "enabled": True,
                "contract": hard_negative_contract,
                "query_selection": hard_negative_selection_report,
                "hard_negatives_sha256": _file_sha256(hard_negative_cache_path),
            },
        )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    start_step = 1
    if resume_state is not None:
        if resume_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        start_step = int(resume_state["step"]) + 1
    rng_resume_status = _restore_rng_state(resume_state)
    metric_rows: list[dict[str, Any]] = list(monitor.history)
    max_steps = max(1, int(max_steps))
    if start_step > max_steps:
        report = {
            "status": "ok",
            "method": str(method_name),
            "resume": {
                "enabled": True,
                "checkpoint_path": resume_state["checkpoint_path"],
                "training_state_path": resume_state["training_state_path"],
                "checkpoint_step": int(resume_state["step"]),
            },
            "message": "resume checkpoint is already at or beyond max_steps",
            "resume_contract_status": resume_contract_status,
            "rng_resume_status": rng_resume_status,
            "output_dir": str(output_dir),
            "max_steps": int(max_steps),
            "last_metrics": metric_rows[-1] if metric_rows else resume_state.get("last_metrics", {}),
            **monitor.paths_report(),
        }
        _write_json(output_dir / "train_report.json", report)
        _write_json(output_dir / "metrics.json", report)
        return report
    checkpoint_every = max(0, int(checkpoint_every))
    started_at = time.time()
    _write_embedding_progress(
        output_dir,
        stage="training",
        step=0,
        max_steps=max_steps,
        extra={
            "skill_count": len(corpus.skills),
            "train_query_count": len(corpus.train_queries),
            "eval_query_count": len(corpus.eval_queries),
            "negatives_per_query": int(negatives_per_query),
            "micro_batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
            "use_bf16_autocast": bool(use_bf16_autocast),
            "mining_use_bf16_autocast": bool(mining_use_bf16_autocast),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "resume_enabled": bool(resume_state is not None),
            "resume_checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
            "nonfinite_gradient_action": str(nonfinite_gradient_action),
            "max_nonfinite_gradient_fraction": float(max_nonfinite_gradient_fraction),
            "sampling_protocol": (
                "capability_source_round_robin_v1"
                if prepared_groups is not None
                else "legacy_seeded_row_order_v1"
            ),
        },
    )

    def save_checkpoint(step: int, metrics: dict[str, Any]) -> Path:
        payload = {
            "method": str(method_name),
            "step": int(step),
            "model_name_or_path": str(encoder_model_path),
            "training_contract": training_contract,
            "config": {
                "data_root": None if data_root is None else str(data_root),
                "prepared_corpus_dir": (
                    None if prepared_corpus_dir is None else str(prepared_corpus_dir)
                ),
                "encoder_model_path": str(encoder_model_path),
                "temperature": float(temperature),
                "max_length": int(max_length),
                "negatives_per_query": int(negatives_per_query),
                "negative_sampling_protocol": negative_sampling_protocol,
                "hard_negative_query_selection_protocol": (
                    hard_negative_query_selection_protocol
                ),
                "hard_negative_top_k": int(hard_negative_top_k),
                "training_objective": str(training_objective),
                "pooling": "cls",
                "query_text_mode": "raw",
                "doc_template": str(doc_template),
                "backbone": "full_finetune",
                "torch_dtype": str(torch_dtype),
                "use_bf16_autocast": bool(use_bf16_autocast),
                "mining_use_bf16_autocast": bool(mining_use_bf16_autocast),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "micro_batch_size": int(batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
                "max_checkpoints_to_keep": int(max_checkpoints_to_keep),
                "sampling_protocol": (
                    "capability_source_round_robin_v1"
                    if prepared_groups is not None
                    else "legacy_seeded_row_order_v1"
                ),
                "in_batch_candidate_contract": "per_query_legal_catalog_multi_positive_logsumexp_v1",
            },
            "corpus_report": corpus.report,
            "last_metrics": metrics,
        }
        return _save_full_finetune_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            base_model_dir=load_model_path,
            checkpoint_dir=checkpoint_dir,
            step=step,
            payload=payload,
            optimizer=optimizer,
            checkpoint_prefix=checkpoint_prefix,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
        )

    last_checkpoint_path: Path | None = None
    for step in range(start_step, max_steps + 1):
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_metrics: list[dict[str, float]] = []
        for accum_idx in range(gradient_accumulation_steps):
            global_batch_index = (step - 1) * gradient_accumulation_steps + accum_idx
            if prepared_groups is not None:
                batch_queries = _prepared_source_balanced_batch(
                    prepared_groups,
                    global_batch_index=global_batch_index,
                    batch_size=batch_size,
                    seed=seed,
                )
            else:
                start = (global_batch_index * batch_size) % len(corpus.train_queries)
                batch_queries = [
                    corpus.train_queries[(start + offset) % len(corpus.train_queries)]
                    for offset in range(batch_size)
                ]
            query_texts, doc_texts, positive_mask_cpu, legal_mask_cpu, batch_report = _batch_from_queries(
                queries=batch_queries,
                skills=corpus.skills,
                skill_id_to_skill=skill_id_to_skill,
                all_skill_ids=all_skill_ids,
                hard_negatives_by_query=hard_negatives,
                negatives_per_query=negatives_per_query,
                rng_seed=seed + step * 1000 + accum_idx,
                all_skill_id_set=all_skill_id_set,
            )
            if not query_texts or not doc_texts:
                raise ValueError(f"empty training batch at step {step} accum {accum_idx}: {batch_report}")
            query_embs = _encode_texts_trainable(
                model,
                tokenizer,
                query_texts,
                max_length=max_length,
                pooling="cls",
                use_bf16_autocast=use_bf16_autocast,
            )
            _ensure_finite_tensor(query_embs, output_dir=output_dir, step=step, name="query_embeddings")
            doc_embs = _encode_texts_trainable(
                model,
                tokenizer,
                doc_texts,
                max_length=max_length,
                pooling="cls",
                use_bf16_autocast=use_bf16_autocast,
            )
            _ensure_finite_tensor(doc_embs, output_dir=output_dir, step=step, name="doc_embeddings")
            logits = (query_embs @ doc_embs.T) / float(temperature)
            _ensure_finite_tensor(logits, output_dir=output_dir, step=step, name="logits")
            positive_mask = positive_mask_cpu.to(device)
            legal_mask = legal_mask_cpu.to(device)
            micro_loss = _multi_positive_legal_nll(
                logits,
                positive_mask,
                legal_mask,
            )
            _ensure_finite_loss(micro_loss, output_dir=output_dir, step=step)
            (micro_loss / gradient_accumulation_steps).backward()
            accumulated_loss += float(micro_loss.detach().cpu().item())
            with torch.no_grad():
                min_value = torch.finfo(logits.dtype).min
                accumulated_metrics.append(
                    _ranking_metrics(
                        logits.detach().masked_fill(~legal_mask, min_value),
                        positive_mask,
                    )
                )
        gradient_guard_report = _ensure_finite_gradients(
            model,
            output_dir=output_dir,
            step=step,
            action=nonfinite_gradient_action,
            max_nonfinite_fraction=max_nonfinite_gradient_fraction,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = {
            key: float(sum(row[key] for row in accumulated_metrics) / len(accumulated_metrics))
            for key in accumulated_metrics[0]
        }
        row: dict[str, Any] = {
            "step": int(step),
            "loss": accumulated_loss / gradient_accumulation_steps,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "nonfinite_gradient_count": int(gradient_guard_report["nonfinite_gradient_count"]),
            "sanitized_gradient_count": int(gradient_guard_report["sanitized_gradient_count"]),
            "nonfinite_gradient_action": str(gradient_guard_report["nonfinite_gradient_action"]),
        }
        if corpus.eval_queries and (step == 1 or step == max_steps or step % max(1, int(eval_every)) == 0):
            model.eval()
            eval_batch = corpus.eval_queries[: min(len(corpus.eval_queries), batch_size)]
            with torch.no_grad():
                eq, ed, epositive_cpu, elegal_cpu, _ = _batch_from_queries(
                    queries=eval_batch,
                    skills=corpus.skills,
                    skill_id_to_skill=skill_id_to_skill,
                    all_skill_ids=all_skill_ids,
                    hard_negatives_by_query=hard_negatives,
                    negatives_per_query=negatives_per_query,
                    rng_seed=seed + 100000 + step,
                    all_skill_id_set=all_skill_id_set,
                )
                eq_embs = _encode_texts_trainable(
                    model,
                    tokenizer,
                    eq,
                    max_length=max_length,
                    pooling="cls",
                    use_bf16_autocast=use_bf16_autocast,
                )
                ed_embs = _encode_texts_trainable(
                    model,
                    tokenizer,
                    ed,
                    max_length=max_length,
                    pooling="cls",
                    use_bf16_autocast=use_bf16_autocast,
                )
                eval_logits = (eq_embs @ ed_embs.T) / float(temperature)
                eval_positive = epositive_cpu.to(device)
                eval_legal = elegal_cpu.to(device)
                eval_metrics = _ranking_metrics(
                    eval_logits.masked_fill(
                        ~eval_legal,
                        torch.finfo(eval_logits.dtype).min,
                    ),
                    eval_positive,
                )
            model.train()
            row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
        if checkpoint_every and (step == max_steps or step % checkpoint_every == 0):
            last_checkpoint_path = save_checkpoint(step, row)
            row["checkpoint_path"] = str(last_checkpoint_path)
        metric_rows.append(row)
        monitor.record(row)
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            elapsed = time.time() - started_at
            _write_embedding_progress(
                output_dir,
                stage="training",
                step=step,
                max_steps=max_steps,
                extra={
                    "loss": row["loss"],
                    "train_recall@1": row["train_recall@1"],
                    "train_recall@5": row["train_recall@5"],
                    "elapsed_seconds": elapsed,
                },
            )
            print(
                f"[{log_prefix}] "
                f"step={step}/{max_steps} loss={row['loss']:.6f} "
                f"train_recall@1={row['train_recall@1']:.4f} train_recall@5={row['train_recall@5']:.4f}",
                flush=True,
            )
    checkpoint_path = last_checkpoint_path or save_checkpoint(
        max_steps,
        metric_rows[-1] if metric_rows else {},
    )
    report = {
        "status": "ok",
        "method": str(method_name),
        "data_root": None if data_root is None else str(data_root),
        "prepared_corpus_dir": (
            None if prepared_corpus_dir is None else str(prepared_corpus_dir)
        ),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "encoder_model_path": str(encoder_model_path),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "negatives_per_query": int(negatives_per_query),
        "hard_negative_top_k": int(hard_negative_top_k),
        "hard_negative_query_count": len(hard_negatives),
        "negative_sampling_protocol": negative_sampling_protocol,
        "hard_negative_query_selection_protocol": (
            hard_negative_query_selection_protocol
        ),
        "in_batch_candidate_contract": "per_query_legal_catalog_multi_positive_logsumexp_v1",
        "sampling_protocol": (
            "capability_source_round_robin_v1"
            if prepared_groups is not None
            else "legacy_seeded_row_order_v1"
        ),
        "corpus_report": corpus.report,
        "last_metrics": metric_rows[-1] if metric_rows else {},
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
            "contract_status": resume_contract_status,
            "rng_status": rng_resume_status,
        },
        "gradient_guard": {
            "nonfinite_gradient_action": str(nonfinite_gradient_action),
            "max_nonfinite_gradient_fraction": float(max_nonfinite_gradient_fraction),
        },
        **monitor.paths_report(),
        "model_config": {
            "pooling": "cls",
            "query_text_mode": "raw",
            "backbone": "full_finetune",
            "training_objective": str(training_objective),
            "doc_template": str(doc_template),
            "torch_dtype": str(torch_dtype),
            "use_bf16_autocast": bool(use_bf16_autocast),
            "mining_use_bf16_autocast": bool(mining_use_bf16_autocast),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "micro_batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(batch_size) * int(gradient_accumulation_steps),
            "nonfinite_gradient_action": str(nonfinite_gradient_action),
            "max_nonfinite_gradient_fraction": float(max_nonfinite_gradient_fraction),
        },
    }
    _write_json(output_dir / "train_report.json", report)
    _write_json(output_dir / "metrics.json", report)
    _write_embedding_progress(
        output_dir,
        stage="complete",
        step=max_steps,
        max_steps=max_steps,
        extra={"status": "ok", "checkpoint_path": str(checkpoint_path), "last_metrics": metric_rows[-1] if metric_rows else {}},
    )
    return report
