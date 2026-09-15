from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _skill_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or fallback)


def _is_appworld_compatible(row: dict[str, Any], skill_id: str) -> bool:
    return bool(row.get("appworld_executor_compatible") is True or str(row.get("executor_domain") or "").lower() == "appworld" or skill_id.startswith("skillx/appworld/"))


def _extract_appended_appworld_rows(
    *,
    base_rows: Sequence[dict[str, Any]],
    dynamic_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_ids = [_skill_id(row, str(idx)) for idx, row in enumerate(base_rows)]
    dynamic_ids = [_skill_id(row, str(idx)) for idx, row in enumerate(dynamic_rows)]
    prefix_matches = dynamic_ids[: len(base_ids)] == base_ids
    base_id_set = set(base_ids)
    if prefix_matches:
        candidate_rows = list(dynamic_rows[len(base_ids) :])
        extraction_mode = "checkpoint_prefix_tail"
    else:
        candidate_rows = [
            row
            for row in dynamic_rows
            if _skill_id(row, "") not in base_id_set
            and (
                row.get("is_appended_after_checkpoint") is True
                or str(row.get("skill_pool_role") or "") == "appworld_appended_after_checkpoint"
                or _is_appworld_compatible(row, _skill_id(row, ""))
            )
        ]
        extraction_mode = "metadata_filter_fallback"
    appended_rows = [
        dict(row)
        for row in candidate_rows
        if _is_appworld_compatible(row, _skill_id(row, ""))
    ]
    return appended_rows, {
        "base_skill_count": len(base_rows),
        "dynamic_skill_count": len(dynamic_rows),
        "prefix_matches": bool(prefix_matches),
        "extraction_mode": extraction_mode,
        "appended_appworld_candidate_count": len(appended_rows),
    }


def _positive_skill_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("positive_skill_ids")
    if values is None:
        values = row.get("positive_skill_id")
    if values is None:
        values = row.get("skill_id")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if item]


def _query_text(row: dict[str, Any]) -> str:
    return str(row.get("query_text") or row.get("state_text") or row.get("query") or row.get("instruction_text") or "")


def _query_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("query_id") or row.get("task_id") or row.get("id") or f"query-{idx}")


def _format_query_text(text: str, query_text_format: str) -> str:
    mode = str(query_text_format or "raw").strip().lower()
    if mode in {"", "raw", "none"}:
        return text
    if mode in {"skillrouter", "skillrouter_state", "appworld_skillrouter_query"}:
        return (
            "Instruct: Given a task description, retrieve the most relevant "
            "skill document that would help an agent complete the task\nQuery:"
            f"{text[:1500]}"
        )
    raise ValueError(f"unsupported query_text_format: {query_text_format}")


def _positive_pairs(retrieval_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    pairs: list[dict[str, str]] = []
    skipped_missing_text = 0
    skipped_missing_positive = 0
    for idx, row in enumerate(retrieval_rows):
        text = _query_text(row)
        positives = _positive_skill_ids(row)
        if not text:
            skipped_missing_text += 1
            continue
        if not positives:
            skipped_missing_positive += 1
            continue
        for skill_id in positives:
            pairs.append(
                {
                    "query_id": _query_id(row, idx),
                    "query_text": text,
                    "positive_skill_id": skill_id,
                }
            )
    return pairs, {
        "source_rows": len(retrieval_rows),
        "positive_pair_count": len(pairs),
        "unique_query_count": len({row["query_id"] for row in pairs}),
        "skipped_missing_text": skipped_missing_text,
        "skipped_missing_positive": skipped_missing_positive,
    }


def sample_positive_pairs(
    pairs: Sequence[dict[str, str]],
    *,
    max_pairs: int | None = None,
    sampling_strategy: str = "head",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    total = len(pairs)
    strategy = str(sampling_strategy or "head").strip().lower()
    if max_pairs is None or int(max_pairs) <= 0 or int(max_pairs) >= total:
        return list(pairs), {
            "sampling_strategy": strategy,
            "positive_pair_count_before_cap": total,
            "positive_pair_count_after_cap": total,
            "selected_pair_indices": list(range(total)),
        }
    cap = int(max_pairs)
    if strategy == "head":
        indices = list(range(cap))
    elif strategy == "stride":
        indices = []
        seen: set[int] = set()
        for offset in range(cap):
            idx = min(total - 1, int(offset * total / cap))
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        cursor = 0
        while len(indices) < cap and cursor < total:
            if cursor not in seen:
                seen.add(cursor)
                indices.append(cursor)
            cursor += 1
        indices = sorted(indices)
    else:
        raise ValueError(f"unsupported sampling_strategy: {sampling_strategy}")
    return [dict(pairs[idx]) for idx in indices], {
        "sampling_strategy": strategy,
        "positive_pair_count_before_cap": total,
        "positive_pair_count_after_cap": len(indices),
        "selected_pair_indices": indices,
    }


def _rank_one(scores: torch.Tensor, positive_index: int, mask: torch.Tensor | None = None) -> int | None:
    if positive_index < 0 or positive_index >= int(scores.numel()):
        return None
    if mask is not None:
        if not bool(mask[positive_index].item()):
            return None
        scores = scores[mask]
        positive_score = scores[(mask.nonzero(as_tuple=False).view(-1) == positive_index).nonzero(as_tuple=False).view(-1)[0]]
    else:
        positive_score = scores[positive_index]
    return int((scores > positive_score).sum().item()) + 1


def _metric_summary(ranks: Sequence[int | None], *, k_values: Sequence[int]) -> dict[str, Any]:
    valid_ranks = [int(rank) for rank in ranks if rank is not None]
    output: dict[str, Any] = {
        "positive_ranks": valid_ranks,
        "ranked_positive_count": len(valid_ranks),
        "missing_positive_count": len(ranks) - len(valid_ranks),
        "mrr": 0.0,
        "mean_rank": None,
    }
    if valid_ranks:
        output["mrr"] = round(sum(1.0 / float(rank) for rank in valid_ranks) / float(len(valid_ranks)), 6)
        output["mean_rank"] = round(sum(valid_ranks) / float(len(valid_ranks)), 6)
    for k in k_values:
        key = f"recall@{int(k)}"
        output[key] = round(sum(1 for rank in valid_ranks if rank <= int(k)) / float(len(valid_ranks)), 6) if valid_ranks else 0.0
    return output


def rank_positive_pairs(
    *,
    logits: torch.Tensor,
    skill_ids: Sequence[str],
    positive_skill_ids: Sequence[str],
    appworld_compatible_mask: Sequence[bool],
    k_values: Sequence[int] = (1, 5, 10, 20, 50, 100, 350),
) -> dict[str, Any]:
    if logits.ndim != 2:
        raise ValueError("logits must be rank-2 [pairs, skills]")
    if int(logits.shape[1]) != len(skill_ids):
        raise ValueError("logits skill dimension must match skill_ids")
    if int(logits.shape[0]) != len(positive_skill_ids):
        raise ValueError("logits batch dimension must match positive_skill_ids")
    if len(appworld_compatible_mask) != len(skill_ids):
        raise ValueError("appworld_compatible_mask must match skill_ids")
    id_to_index = {str(skill_id): idx for idx, skill_id in enumerate(skill_ids)}
    mask = torch.tensor(list(appworld_compatible_mask), dtype=torch.bool, device=logits.device)
    global_ranks: list[int | None] = []
    appworld_ranks: list[int | None] = []
    missing_ids: list[str] = []
    for row_scores, positive_skill_id in zip(logits, positive_skill_ids):
        positive_index = id_to_index.get(str(positive_skill_id))
        if positive_index is None:
            missing_ids.append(str(positive_skill_id))
            global_ranks.append(None)
            appworld_ranks.append(None)
            continue
        global_ranks.append(_rank_one(row_scores, positive_index))
        appworld_ranks.append(_rank_one(row_scores, positive_index, mask=mask))
    k_values = [int(k) for k in k_values]
    return {
        "positive_pair_count": len(positive_skill_ids),
        "skill_count": len(skill_ids),
        "appworld_compatible_skill_count": int(mask.sum().item()),
        "missing_positive_skill_ids": sorted(set(missing_ids)),
        "k_values": k_values,
        "global": _metric_summary(global_ranks, k_values=k_values),
        "appworld_compatible": _metric_summary(appworld_ranks, k_values=k_values),
    }


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _progress_path(output_path: str | Path | None) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path)
    return path.with_suffix(path.suffix + ".progress.json")


def audit_appworld_dynamic_routing(
    *,
    checkpoint_path: str | Path,
    base_skill_pool_path: str | Path,
    dynamic_skill_pool_path: str | Path,
    retrieval_path: str | Path,
    output_path: str | Path | None = None,
    model_cache_dir: str | Path | None = None,
    batch_size: int = 8,
    k_values: Sequence[int] = (1, 5, 10, 20, 50, 100, 350),
    max_pairs: int | None = None,
    sampling_strategy: str = "head",
    device: str | torch.device | None = "auto",
    query_text_format: str | None = "auto",
) -> dict[str, Any]:
    progress = _progress_path(output_path)
    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "loading_inputs", "processed_pairs": 0, "total_pairs": 0})
    base_rows = _read_jsonl(base_skill_pool_path)
    dynamic_rows = _read_jsonl(dynamic_skill_pool_path)
    appended_rows, extraction_report = _extract_appended_appworld_rows(
        base_rows=base_rows,
        dynamic_rows=dynamic_rows,
    )
    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "loading_checkpoint", "processed_pairs": 0, "total_pairs": 0})
    model, model_config, checkpoint_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(checkpoint_path),
        skills_path=Path(base_skill_pool_path),
        model_cache_dir=model_cache_dir,
    )
    applied_query_text_format = (
        str(model_config.get("query_text_format") or "raw")
        if query_text_format is None or str(query_text_format).strip().lower() == "auto"
        else str(query_text_format)
    )
    run_device = _resolve_device(device)
    model.to(run_device).eval()
    if progress is not None:
        _write_json(
            progress,
            {
                "status": "running",
                "phase": "appending_dynamic_skills",
                "processed_pairs": 0,
                "total_pairs": 0,
                "appended_appworld_candidate_count": len(appended_rows),
            },
        )
    append_report = model.append_skills(appended_rows)
    if progress is not None:
        _write_json(
            progress,
            {
                "status": "running",
                "phase": "preparing_pairs",
                "processed_pairs": 0,
                "total_pairs": 0,
                "appended_count": int(append_report.get("appended_count") or 0),
            },
        )
    skill_rows = list(getattr(model, "skills", []))
    skill_ids = [_skill_id(dict(row), str(idx)) for idx, row in enumerate(skill_rows)]
    appworld_mask = [_is_appworld_compatible(dict(row), skill_id) for row, skill_id in zip(skill_rows, skill_ids)]
    retrieval_rows = _read_jsonl(retrieval_path)
    pairs, pair_report = _positive_pairs(retrieval_rows)
    pairs, sampling_report = sample_positive_pairs(
        pairs,
        max_pairs=max_pairs,
        sampling_strategy=sampling_strategy,
    )
    pair_report["sampling"] = sampling_report
    if not pairs:
        report = {
            "status": "blocked",
            "blockers": ["no_positive_pairs"],
            "checkpoint_path": str(checkpoint_path),
            "base_skill_pool_path": str(base_skill_pool_path),
            "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
            "retrieval_path": str(retrieval_path),
            "extraction_report": extraction_report,
            "append_report": append_report,
            "pair_report": pair_report,
        }
        if output_path is not None:
            _write_json(output_path, report)
        if progress is not None:
            _write_json(progress, {"status": "blocked", "phase": "no_positive_pairs", "processed_pairs": 0, "total_pairs": 0})
        return report

    if progress is not None:
        _write_json(progress, {"status": "running", "phase": "ranking", "processed_pairs": 0, "total_pairs": len(pairs)})
    all_logits: list[torch.Tensor] = []
    processed = 0
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            h = model.encode_states([_format_query_text(row["query_text"], applied_query_text_format) for row in batch])
            logits = model.skill_table.retrieval_logits(h).detach().cpu()
            all_logits.append(logits)
            processed += len(batch)
            if progress is not None:
                _write_json(
                    progress,
                    {
                        "status": "running",
                        "phase": "ranking",
                        "processed_pairs": processed,
                        "total_pairs": len(pairs),
                    },
                )
    logits = torch.cat(all_logits, dim=0)
    rank_metrics = rank_positive_pairs(
        logits=logits,
        skill_ids=skill_ids,
        positive_skill_ids=[row["positive_skill_id"] for row in pairs],
        appworld_compatible_mask=appworld_mask,
        k_values=k_values,
    )
    blockers: list[str] = []
    if int(append_report.get("appended_count", 0)) <= 0:
        blockers.append("no_appworld_skills_appended")
    if rank_metrics["global"]["ranked_positive_count"] <= 0:
        blockers.append("no_ranked_global_positives")
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "checkpoint_path": str(checkpoint_path),
        "base_skill_pool_path": str(base_skill_pool_path),
        "dynamic_skill_pool_path": str(dynamic_skill_pool_path),
        "retrieval_path": str(retrieval_path),
        "model_config": model_config,
        "checkpoint_report": checkpoint_report,
        "extraction_report": extraction_report,
        "append_report": append_report,
        "pair_report": pair_report,
        "device": str(run_device),
        "batch_size": batch_size,
        "query_text_format": applied_query_text_format,
        "rank_metrics": rank_metrics,
        "progress_path": None if progress is None else str(progress),
        "interpretation": {
            "global": "Rank before executor masking; this tests large-skill-pool retrieval.",
            "appworld_compatible": "Rank after keeping only AppWorld-compatible skills; this tests executor handoff quality.",
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
        if progress is not None:
            _write_json(progress, {"status": "ok", "phase": "done", "processed_pairs": processed, "total_pairs": len(pairs)})
    return report
