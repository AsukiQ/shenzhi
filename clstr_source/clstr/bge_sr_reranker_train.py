from __future__ import annotations

import json
import hashlib
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from clstr.bge_reranker import _resolve_dtype, bge_reranker_score_tensor
from clstr.exact_candidate_cache import load_or_build_exact_candidate_rankings
from clstr.skillret_official import _skillrouter_skill_text
from clstr.skillrouter_style import (
    _encode_raw_texts,
    _load_backbone,
    _query_text_for_mode,
    _resolve_hf_pooling,
    _resolve_query_text_mode,
    _resolve_tokenizer_padding_side,
)
from clstr.training_monitor import TrainingMonitor


def _read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(rows) >= int(max_rows):
                break
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
            "resume checkpoint training contract differs from the current training groups or objective"
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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_reranker_progress(
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


def _load_reranker_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if path.is_dir() and (path / "latest.json").exists():
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


def _skill_id(skill: dict[str, Any]) -> str:
    return str(skill.get("skill_id") or skill.get("id") or "").strip()


def _ranking_device_contract() -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    contract = (
        f"cuda::{properties.name}::cc{properties.major}.{properties.minor}"
    )
    return "cuda", contract


def build_listwise_reranker_groups(
    *,
    queries: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    ranked_skill_ids_by_query: dict[str, list[str]],
    top_k: int = 20,
    max_groups: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(top_k) <= 0:
        raise ValueError("reranker top_k must be positive")
    skill_by_id = {_skill_id(skill): skill for skill in skills if _skill_id(skill)}
    groups: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for query in queries:
        if max_groups is not None and len(groups) >= int(max_groups):
            break
        query_id = str(query.get("query_id") or "").strip()
        query_text = str(query.get("query") or "").strip()
        positives = {str(item).strip() for item in query.get("positive_skill_ids", []) if str(item).strip()}
        if not query_id or not query_text or not positives:
            skipped["missing_query_or_positive"] += 1
            continue
        candidate_ids: list[str] = []
        seen: set[str] = set()
        for skill_id in ranked_skill_ids_by_query.get(query_id, []):
            skill_id = str(skill_id).strip()
            if skill_id and skill_id in skill_by_id and skill_id not in seen:
                candidate_ids.append(skill_id)
                seen.add(skill_id)
            if len(candidate_ids) >= int(top_k):
                break
        if not candidate_ids:
            skipped["missing_candidates"] += 1
            continue
        positive_candidate_indices = [
            idx for idx, skill_id in enumerate(candidate_ids) if skill_id in positives
        ]
        if not positive_candidate_indices:
            skipped["positive_not_in_topk"] += 1
            continue
        groups.append(
            {
                "query_id": query_id,
                "query": query_text,
                "positive_skill_ids": sorted(positives),
                "candidate_skill_ids": candidate_ids,
                "candidate_texts": [_skillrouter_skill_text(skill_by_id[skill_id]) for skill_id in candidate_ids],
                "label_index": int(positive_candidate_indices[0]),
                "positive_candidate_indices": positive_candidate_indices,
            }
        )
    return groups, {
        "group_count": len(groups),
        "query_count": len(queries),
        "top_k": int(top_k),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _rank_candidates(
    *,
    query_ids: list[str],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    skill_ids: list[str],
    top_k: int,
    batch_size: int,
    candidate_skill_ids_by_query: list[list[str] | None] | None = None,
    score_device: str | torch.device | None = None,
) -> dict[str, list[str]]:
    if query_embs.ndim != 2 or skill_embs.ndim != 2:
        raise ValueError("candidate ranking embeddings must be rank-2 tensors")
    if query_embs.size(0) != len(query_ids) or skill_embs.size(0) != len(skill_ids):
        raise ValueError("candidate ranking IDs do not match their embedding rows")
    if query_embs.size(-1) != skill_embs.size(-1):
        raise ValueError("query and skill embedding dimensions differ")
    if query_embs.device != skill_embs.device:
        raise ValueError("query and skill embeddings must be on the same device")
    if int(top_k) <= 0:
        raise ValueError("candidate ranking top_k must be positive")
    if score_device is not None:
        target_device = torch.device(score_device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA candidate ranking was requested but CUDA is unavailable")
        query_embs = query_embs.to(target_device)
        skill_embs = skill_embs.to(target_device)

    def exact_scores(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if queries.device.type != "cuda":
            return queries @ candidates.T
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return queries @ candidates.T
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    if candidate_skill_ids_by_query is not None:
        if len(candidate_skill_ids_by_query) != len(query_ids):
            raise ValueError("candidate catalog rows do not match query rows")
        skill_id_to_idx = {skill_id: index for index, skill_id in enumerate(skill_ids)}
        known_skill_ids = set(skill_id_to_idx)
        grouped_positions: dict[tuple[int, ...] | None, list[int]] = {}
        catalog_indices_cache: dict[
            tuple[str, ...] | None,
            tuple[int, ...] | None,
        ] = {None: None}
        all_indices = tuple(range(len(skill_ids)))
        for position, candidate_ids in enumerate(candidate_skill_ids_by_query):
            catalog_key = None if candidate_ids is None else tuple(candidate_ids)
            if catalog_key in catalog_indices_cache:
                indices = catalog_indices_cache[catalog_key]
            else:
                candidate_set = set(candidate_ids or [])
                if len(candidate_ids or []) != len(candidate_set):
                    raise ValueError(
                        f"query catalog contains duplicate skills: {query_ids[position]}"
                    )
                unknown = sorted(candidate_set - known_skill_ids)
                if unknown:
                    raise ValueError(
                        f"query catalog references unknown skills: "
                        f"{query_ids[position]}: {unknown[:4]}"
                    )
                indices = tuple(
                    skill_id_to_idx[skill_id] for skill_id in candidate_ids or []
                )
                catalog_indices_cache[catalog_key] = indices
            if indices == ():
                raise ValueError(f"query has no legal candidate skills: {query_ids[position]}")
            grouped_positions.setdefault(indices, []).append(position)
        ranked: dict[str, list[str]] = {}
        for catalog_indices, positions in grouped_positions.items():
            legal_indices = all_indices if catalog_indices is None else catalog_indices
            if catalog_indices is None:
                legal_embs = skill_embs
            else:
                legal_tensor = torch.tensor(
                    legal_indices,
                    dtype=torch.long,
                    device=skill_embs.device,
                )
                legal_embs = skill_embs.index_select(0, legal_tensor)
            for start in range(0, len(positions), max(1, int(batch_size))):
                batch_positions = positions[start : start + max(1, int(batch_size))]
                position_tensor = torch.tensor(
                    batch_positions,
                    dtype=torch.long,
                    device=query_embs.device,
                )
                q = query_embs.index_select(0, position_tensor)
                scores = exact_scores(q, legal_embs)
                top = torch.topk(
                    scores,
                    k=min(int(top_k), scores.size(-1)),
                    dim=-1,
                ).indices.cpu().tolist()
                for position, local_indices in zip(batch_positions, top):
                    ranked[query_ids[position]] = [
                        skill_ids[legal_indices[int(index)]] for index in local_indices
                    ]
        return ranked
    ranked: dict[str, list[str]] = {}
    for start in range(0, len(query_ids), max(1, int(batch_size))):
        q = query_embs[start : start + max(1, int(batch_size))]
        scores = exact_scores(q, skill_embs)
        top = torch.topk(scores, k=min(int(top_k), scores.size(-1)), dim=-1).indices.cpu().tolist()
        for offset, indices in enumerate(top):
            ranked[query_ids[start + offset]] = [skill_ids[int(idx)] for idx in indices]
    return ranked


def _load_sr_embedding_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    report_path = Path(checkpoint_path) / "train_checkpoint_report.json"
    if not report_path.exists():
        return {"pooling": "cls", "query_text_mode": "raw", "tokenizer_padding_side": "right"}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        return {"pooling": "cls", "query_text_mode": "raw", "tokenizer_padding_side": "right"}
    return config


def _build_groups_from_sr_embedding(
    *,
    sr_embedding_output_dir: Path,
    sr_embedding_checkpoint_path: Path,
    encoder_max_length: int,
    encoder_batch_size: int,
    torch_dtype: str,
    top_k: int,
    rank_batch_size: int,
    max_train_queries: int | None,
    max_eval_queries: int | None,
    max_train_groups: int | None,
    max_eval_groups: int | None,
    output_dir: Path,
    candidate_cache_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_skills_path = sr_embedding_output_dir / "selected_skills.jsonl"
    train_queries_path = sr_embedding_output_dir / "train_queries.jsonl"
    eval_queries_path = sr_embedding_output_dir / "eval_queries.jsonl"
    skills = _read_jsonl(selected_skills_path)
    train_queries = _read_jsonl(train_queries_path, max_rows=max_train_queries)
    eval_queries = _read_jsonl(eval_queries_path, max_rows=max_eval_queries)
    config = _load_sr_embedding_checkpoint_config(sr_embedding_checkpoint_path)
    pooling = _resolve_hf_pooling(sr_embedding_checkpoint_path, str(config.get("pooling") or "cls"))
    query_text_mode = _resolve_query_text_mode(sr_embedding_checkpoint_path, str(config.get("query_text_mode") or "raw"), pooling)
    padding_side = _resolve_tokenizer_padding_side(
        sr_embedding_checkpoint_path,
        str(config.get("tokenizer_padding_side") or "right"),
        pooling,
    )
    skill_texts = [_skillrouter_skill_text(skill) for skill in skills]
    all_queries = train_queries + eval_queries
    query_texts = [_query_text_for_mode({"query": row.get("query", "")}, query_text_mode) for row in all_queries]
    skill_ids = [_skill_id(skill) for skill in skills]
    query_ids = [str(row.get("query_id")) for row in all_queries]
    candidate_skill_ids_by_query = [
        (
            [str(item) for item in row.get("candidate_skill_ids") or []]
            if "candidate_skill_ids" in row
            else None
        )
        for row in all_queries
    ]
    score_device, ranking_device = _ranking_device_contract()

    def build_ranked() -> dict[str, list[str]]:
        model, tokenizer, _device = _load_backbone(
            str(sr_embedding_checkpoint_path),
            torch_dtype,
            padding_side,
            pooling=pooling,
        )
        skill_embs = _encode_raw_texts(
            model,
            tokenizer,
            skill_texts,
            encoder_batch_size,
            encoder_max_length,
            pooling=pooling,
        )
        query_embs = _encode_raw_texts(
            model,
            tokenizer,
            query_texts,
            encoder_batch_size,
            encoder_max_length,
            pooling=pooling,
        )
        return _rank_candidates(
            query_ids=query_ids,
            query_embs=query_embs,
            skill_embs=skill_embs,
            skill_ids=skill_ids,
            top_k=top_k,
            batch_size=rank_batch_size,
            candidate_skill_ids_by_query=candidate_skill_ids_by_query,
            score_device=score_device,
        )

    ranked, candidate_cache_report = load_or_build_exact_candidate_rankings(
        candidate_cache_dir=candidate_cache_dir,
        checkpoint_path=sr_embedding_checkpoint_path,
        selected_skills_path=selected_skills_path,
        train_queries_path=train_queries_path,
        eval_queries_path=eval_queries_path,
        pooling=pooling,
        query_text_mode=query_text_mode,
        tokenizer_padding_side=padding_side,
        torch_dtype=torch_dtype,
        max_length=encoder_max_length,
        top_k=top_k,
        max_train_queries=max_train_queries,
        max_eval_queries=max_eval_queries,
        skill_ids=skill_ids,
        query_ids=query_ids,
        builder=build_ranked,
        candidate_skill_ids_by_query=candidate_skill_ids_by_query,
        ranking_device=ranking_device,
    )
    train_groups, train_report = build_listwise_reranker_groups(
        queries=train_queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=top_k,
        max_groups=max_train_groups,
    )
    eval_groups, eval_report = build_listwise_reranker_groups(
        queries=eval_queries,
        skills=skills,
        ranked_skill_ids_by_query=ranked,
        top_k=top_k,
        max_groups=max_eval_groups,
    )
    _write_jsonl(output_dir / "train_reranker_groups.jsonl", train_groups)
    _write_jsonl(output_dir / "eval_reranker_groups.jsonl", eval_groups)
    return train_groups, eval_groups, {
        "skill_count": len(skills),
        "train_query_count": len(train_queries),
        "eval_query_count": len(eval_queries),
        "train_group_report": train_report,
        "eval_group_report": eval_report,
        "sr_embedding_checkpoint_path": str(sr_embedding_checkpoint_path),
        "pooling": pooling,
        "query_text_mode": query_text_mode,
        "tokenizer_padding_side": padding_side,
        "ranking_device": ranking_device,
        "candidate_cache": candidate_cache_report,
    }


def _group_batch_to_pairs(groups: list[dict[str, Any]]) -> tuple[list[str], list[str], torch.Tensor, list[int]]:
    if not groups:
        raise ValueError("reranker batch is empty")
    queries: list[str] = []
    docs: list[str] = []
    positive_rows: list[list[int]] = []
    sizes: list[int] = []
    for group in groups:
        candidate_texts = [str(item) for item in group.get("candidate_texts", [])]
        if not candidate_texts:
            raise ValueError("reranker group has no candidate texts")
        sizes.append(len(candidate_texts))
        positive_rows.append(
            [
                int(item)
                for item in group.get("positive_candidate_indices")
                or [group["label_index"]]
            ]
        )
        queries.extend([str(group.get("query", ""))] * len(candidate_texts))
        docs.extend(candidate_texts)
    positive_mask = torch.zeros(len(groups), max(sizes), dtype=torch.bool)
    for row_index, indices in enumerate(positive_rows):
        for index in indices:
            if 0 <= index < sizes[row_index]:
                positive_mask[row_index, index] = True
    if not positive_mask.any(dim=-1).all():
        raise ValueError("every reranker group requires a positive candidate")
    return queries, docs, positive_mask, sizes


def _listwise_batch_loss(model, tokenizer, device, groups: list[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, dict[str, float]]:
    queries, docs, positive_mask, sizes = _group_batch_to_pairs(groups)
    tokens = tokenizer(
        queries,
        docs,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    flat_scores = bge_reranker_score_tensor(model(**tokens).logits)
    if flat_scores.numel() != sum(sizes):
        raise ValueError("reranker score count does not match the ragged candidate batch")
    min_value = torch.finfo(flat_scores.dtype).min
    split_scores = flat_scores.split(sizes)
    scores = torch.stack(
        [
            torch.nn.functional.pad(
                row,
                (0, max(sizes) - row.numel()),
                value=min_value,
            )
            for row in split_scores
        ],
        dim=0,
    )
    positive_mask = positive_mask.to(device)
    positive_scores = scores.masked_fill(
        ~positive_mask,
        min_value,
    )
    loss = (
        torch.logsumexp(scores.float(), dim=-1)
        - torch.logsumexp(positive_scores.float(), dim=-1)
    ).mean()
    with torch.no_grad():
        order = torch.argsort(scores, dim=-1, descending=True)
        positives_ranked = positive_mask.gather(1, order)
        ranks = positives_ranked.float().argmax(dim=-1).float() + 1
        metrics = {
            "recall@1": float((ranks <= 1).float().mean().cpu().item()),
            "recall@5": float(
                (
                    ranks
                    <= torch.tensor(
                        [min(5, size) for size in sizes],
                        dtype=ranks.dtype,
                        device=ranks.device,
                    )
                )
                .float()
                .mean()
                .cpu()
                .item()
            ),
            "mrr": float((1.0 / ranks).mean().cpu().item()),
        }
    return loss, metrics


def run_bge_sr_reranker_train(
    *,
    sr_embedding_output_dir: str | Path,
    sr_embedding_checkpoint_path: str | Path,
    output_dir: str | Path,
    reranker_model_path: str = "models/BAAI/bge-reranker-v2-m3",
    max_steps: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1.0e-5,
    top_k: int = 20,
    encoder_max_length: int = 512,
    encoder_batch_size: int = 64,
    rank_batch_size: int = 256,
    reranker_max_length: int = 512,
    torch_dtype: str = "bfloat16",
    seed: int = 13,
    eval_every: int = 50,
    log_every: int = 10,
    checkpoint_every: int = 400,
    max_train_queries: int | None = None,
    max_eval_queries: int | None = None,
    max_train_groups: int | None = None,
    max_eval_groups: int | None = None,
    freeze_backbone: bool = False,
    resume_checkpoint_path: str | Path | None = None,
    max_checkpoints_to_keep: int = 2,
    candidate_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if float(learning_rate) <= 0:
        raise ValueError("SR reranker learning_rate must be positive")
    if int(top_k) <= 0:
        raise ValueError("SR reranker top_k must be positive")
    if int(encoder_max_length) <= 0 or int(reranker_max_length) <= 0:
        raise ValueError("SR reranker max lengths must be positive")
    batch_size = max(1, int(batch_size))
    random.seed(seed)
    torch.manual_seed(seed)
    sr_embedding_output_dir = Path(sr_embedding_output_dir)
    sr_embedding_checkpoint_path = Path(sr_embedding_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_state = _load_reranker_resume_state(resume_checkpoint_path) if resume_checkpoint_path is not None else None
    _write_reranker_progress(output_dir, stage="building_groups")
    train_groups, eval_groups, group_report = _build_groups_from_sr_embedding(
        sr_embedding_output_dir=sr_embedding_output_dir,
        sr_embedding_checkpoint_path=sr_embedding_checkpoint_path,
        encoder_max_length=encoder_max_length,
        encoder_batch_size=encoder_batch_size,
        torch_dtype=torch_dtype,
        top_k=top_k,
        rank_batch_size=rank_batch_size,
        max_train_queries=max_train_queries,
        max_eval_queries=max_eval_queries,
        max_train_groups=max_train_groups,
        max_eval_groups=max_eval_groups,
        output_dir=output_dir,
        candidate_cache_dir=candidate_cache_dir,
    )
    if not train_groups:
        raise ValueError("no train reranker groups with positives in top-k")
    training_contract = {
        "schema_version": "bge_sr_reranker_training_contract_v1",
        "reranker_model_path": str(Path(reranker_model_path).resolve()),
        "training_data": {
            "train_groups_sha256": _file_sha256(output_dir / "train_reranker_groups.jsonl"),
            "eval_groups_sha256": _file_sha256(output_dir / "eval_reranker_groups.jsonl"),
        },
        "objective": {
            "training_objective": "multi_positive_listwise_nll_over_sr_emb_topk",
            "top_k": int(top_k),
            "reranker_max_length": int(reranker_max_length),
            "freeze_backbone": bool(freeze_backbone),
        },
        "schedule": {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
        },
        "numerics": {"torch_dtype": str(torch_dtype)},
    }
    resume_contract_status = _validate_resume_training_contract(
        resume_state,
        training_contract,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_reranker_progress(output_dir, stage="loading_reranker", extra=group_report)
    load_reranker_path = resume_state["checkpoint_path"] if resume_state is not None else reranker_model_path
    tokenizer = AutoTokenizer.from_pretrained(load_reranker_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_reranker_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=_resolve_dtype(torch_dtype),
    ).to(device)
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("classifier") or name.startswith("score"))
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("SR reranker configuration leaves no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(
        output_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_steps=max(1, int(checkpoint_every)),
        curve_interval_steps=max(1, int(checkpoint_every)),
        reset=resume_state is None,
    )
    if resume_state is not None:
        if monitor.metrics_path.exists():
            history: list[dict[str, Any]] = []
            with monitor.metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        history.append(json.loads(line))
            monitor.history = history
        if resume_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
    rng_resume_status = _restore_rng_state(resume_state)
    max_steps = max(1, int(max_steps))
    checkpoint_every = max(0, int(checkpoint_every))
    start_step = int(resume_state["step"]) + 1 if resume_state is not None else 1
    started_at = time.time()
    metric_rows: list[dict[str, Any]] = list(monitor.history)

    def save_checkpoint(step: int, metrics: dict[str, Any]) -> Path:
        checkpoint_path = checkpoint_dir / f"bge_sr_rank-step{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        payload = {
            "method": "bge_sr_rank_listwise_finetune",
            "step": int(step),
            "model_name_or_path": str(reranker_model_path),
            "training_contract": training_contract,
            "config": {
                "sr_embedding_output_dir": str(sr_embedding_output_dir),
                "sr_embedding_checkpoint_path": str(sr_embedding_checkpoint_path),
                "reranker_model_path": str(reranker_model_path),
                "top_k": int(top_k),
                "reranker_max_length": int(reranker_max_length),
                "training_objective": "multi_positive_listwise_nll_over_sr_emb_topk",
                "freeze_backbone": bool(freeze_backbone),
                "max_checkpoints_to_keep": int(max_checkpoints_to_keep),
            },
            "last_metrics": metrics,
            "group_report": group_report,
        }
        torch.save(
            {
                **payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": _capture_rng_state(),
            },
            checkpoint_path / "training_state.pt",
        )
        _write_json(checkpoint_path / "train_checkpoint_report.json", payload)
        _write_json(
            checkpoint_dir / "latest.json",
            {"path": str(checkpoint_path.resolve()), "step": int(step)},
        )
        _write_json(
            checkpoint_dir / "retention_report.json",
            _prune_old_checkpoints(
                checkpoint_dir,
                latest_path=checkpoint_path,
                max_checkpoints_to_keep=max_checkpoints_to_keep,
            ),
        )
        return checkpoint_path

    _write_reranker_progress(
        output_dir,
        stage="training",
        step=max(0, start_step - 1),
        max_steps=max_steps,
        extra={
            "train_group_count": len(train_groups),
            "eval_group_count": len(eval_groups),
            "resume_enabled": bool(resume_state is not None),
            "resume_checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
        },
    )
    if start_step > max_steps:
        report = {
            "status": "ok",
            "method": "bge_sr_rank_listwise_finetune",
            "output_dir": str(output_dir),
            "checkpoint_path": resume_state["checkpoint_path"] if resume_state is not None else None,
            "message": "resume checkpoint is already at or beyond max_steps",
            "resume_contract_status": resume_contract_status,
            "rng_resume_status": rng_resume_status,
            "last_metrics": metric_rows[-1] if metric_rows else (resume_state.get("last_metrics", {}) if resume_state else {}),
            **monitor.paths_report(),
        }
        _write_json(output_dir / "train_report.json", report)
        _write_json(output_dir / "metrics.json", report)
        return report
    last_checkpoint_path: Path | None = None
    for step in range(start_step, max_steps + 1):
        start = ((step - 1) * batch_size) % len(train_groups)
        batch = [train_groups[(start + offset) % len(train_groups)] for offset in range(batch_size)]
        model.train()
        loss, train_metrics = _listwise_batch_loss(model, tokenizer, device, batch, reranker_max_length)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }
        if eval_groups and (step == 1 or step == max_steps or step % max(1, int(eval_every)) == 0):
            model.eval()
            eval_batch = eval_groups[: min(len(eval_groups), batch_size)]
            with torch.no_grad():
                _loss, eval_metrics = _listwise_batch_loss(model, tokenizer, device, eval_batch, reranker_max_length)
            row.update({f"eval_{key}": value for key, value in eval_metrics.items()})
        if checkpoint_every and (step == 1 or step == max_steps or step % checkpoint_every == 0):
            last_checkpoint_path = save_checkpoint(step, row)
            row["checkpoint_path"] = str(last_checkpoint_path)
        metric_rows.append(row)
        monitor.record(row)
        if step == 1 or step == max_steps or step % max(1, int(log_every)) == 0:
            elapsed = time.time() - started_at
            _write_reranker_progress(
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
                "[bge-sr-rank] "
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
        "method": "bge_sr_rank_listwise_finetune",
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "sr_embedding_output_dir": str(sr_embedding_output_dir),
        "sr_embedding_checkpoint_path": str(sr_embedding_checkpoint_path),
        "reranker_model_path": str(reranker_model_path),
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "top_k": int(top_k),
        "group_report": group_report,
        "last_metrics": metric_rows[-1] if metric_rows else {},
        "resume": {
            "enabled": bool(resume_state is not None),
            "checkpoint_path": None if resume_state is None else resume_state["checkpoint_path"],
            "start_step": int(start_step),
            "contract_status": resume_contract_status,
            "rng_status": rng_resume_status,
        },
        **monitor.paths_report(),
    }
    _write_json(output_dir / "train_report.json", report)
    _write_json(output_dir / "metrics.json", report)
    _write_reranker_progress(
        output_dir,
        stage="complete",
        step=max_steps,
        max_steps=max_steps,
        extra={"status": "ok", "checkpoint_path": str(checkpoint_path), "last_metrics": metric_rows[-1] if metric_rows else {}},
    )
    return report
