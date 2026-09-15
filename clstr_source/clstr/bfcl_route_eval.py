from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.memory_candidate_recall import (
    candidate_recall_protocol_metadata,
    declared_candidate_pool_size,
    source_rows_have_causal_sequence,
)
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
from clstr.tau2_skillrouter_eval import _apply_skillrouter_adapter

STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE = "stage0_rank_prior_plus_transition_residual"


DEFAULT_BFCL_ROUTE_CATEGORIES = [
    "BFCL_v3_multiple",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel_multiple",
    "BFCL_v3_multi_turn_base",
    "BFCL_v3_multi_turn_long_context",
    "BFCL_v3_multi_turn_miss_func",
    "BFCL_v3_multi_turn_miss_param",
]

MULTI_TURN_CATEGORIES = {
    "BFCL_v3_multi_turn_base",
    "BFCL_v3_multi_turn_composite",
    "BFCL_v3_multi_turn_long_context",
    "BFCL_v3_multi_turn_miss_func",
    "BFCL_v3_multi_turn_miss_param",
}


@dataclass(frozen=True)
class BFCLRouteCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _read_json_or_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _clean_name(name: Any) -> str:
    return str(name or "").strip()


def _bfcl_skill_id(category: str, raw_id: str, candidate_idx: int, name: str) -> str:
    safe_name = _clean_name(name).replace("/", "_")
    return f"bfcl/{category}/{raw_id}/{candidate_idx}/{safe_name}"


def _bfcl_multiturn_skill_id(name: str) -> str:
    safe_name = _clean_name(name).replace("/", "_")
    return f"bfcl/multi_turn/{safe_name}"


def _start_skill() -> dict[str, Any]:
    return {
        "skill_id": "bfcl/__start__",
        "name": "BFCL.START",
        "description": "Initial BFCL state before any function call.",
        "executor_desc": "Initial BFCL state before any function call.",
        "body": "No function has been selected yet.",
        "skill_md": "No function has been selected yet.",
        "source_benchmark": "bfcl",
    }


def _skill_from_function(skill_id: str, fn: dict[str, Any], *, category: str) -> dict[str, Any]:
    name = _clean_name(fn.get("name"))
    description = str(fn.get("description") or "")
    parameters = fn.get("parameters") or fn.get("inputSchema") or {}
    body = f"Function name: {name}\nCategory: {category}\nDescription: {description}\nParameters: {json.dumps(parameters, ensure_ascii=False, sort_keys=True)}"
    return {
        "skill_id": skill_id,
        "name": name,
        "description": description or f"BFCL function {name}",
        "executor_desc": description or f"BFCL function {name}",
        "body": body,
        "skill_md": body,
        "source_benchmark": "bfcl",
        "bfcl_category": category,
        "function_name": name,
    }


def _question_text(question: Any) -> str:
    turns = []
    for turn_group in question if isinstance(question, list) else []:
        if isinstance(turn_group, list):
            for message in turn_group:
                if isinstance(message, dict):
                    turns.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        elif isinstance(turn_group, dict):
            turns.append(f"{turn_group.get('role', 'user')}: {turn_group.get('content', '')}")
    return "\n".join(turns)


def _single_turn_state_text(*, category: str, raw_id: str, question: Any, previous_calls: list[str]) -> str:
    parts = [
        "benchmark: BFCL",
        f"category: {category}",
        f"bfcl_id: {raw_id}",
        "conversation:",
        _question_text(question),
    ]
    if previous_calls:
        parts.extend(["previous_calls:", "\n".join(previous_calls)])
    return "\n".join(part for part in parts if part)


def _multiturn_state_text(
    *,
    category: str,
    raw_id: str,
    question: list[Any],
    turn_index: int,
    previous_calls: list[str],
) -> str:
    visible_turns = question[: turn_index + 1]
    parts = [
        "benchmark: BFCL",
        f"category: {category}",
        f"bfcl_id: {raw_id}",
        "conversation_so_far:",
        _question_text(visible_turns),
    ]
    if previous_calls:
        parts.extend(["previous_calls:", "\n".join(previous_calls)])
    return "\n".join(part for part in parts if part)


def _answer_names_from_ground_truth(ground_truth: Any) -> list[str]:
    names: list[str] = []
    if isinstance(ground_truth, dict):
        names.extend(_clean_name(key) for key in ground_truth.keys() if _clean_name(key))
    elif isinstance(ground_truth, str):
        match = re.match(r"\s*([A-Za-z_][\w.]*)\s*\(", ground_truth)
        if match:
            names.append(match.group(1))
    elif isinstance(ground_truth, list):
        for item in ground_truth:
            names.extend(_answer_names_from_ground_truth(item))
    return names


def _answers_by_id(data_root: Path, category: str) -> dict[str, Any]:
    path = data_root / "possible_answer" / f"{category}.json"
    if not path.exists():
        return {}
    rows = _read_json_or_jsonl(path)
    return {str(row.get("id")): row.get("ground_truth") for row in rows if isinstance(row, dict)}


def _load_multiturn_functions(data_root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted((data_root / "multi_turn_func_doc").glob("*.json")):
        for row in _read_json_or_jsonl(path):
            if isinstance(row, dict) and _clean_name(row.get("name")):
                docs.append(row)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for fn in docs:
        name = _clean_name(fn.get("name"))
        if name in seen:
            continue
        seen.add(name)
        deduped.append(fn)
    return deduped


def _metrics_from_ranks(ranks: list[int | None], candidate_counts: list[int]) -> dict[str, float]:
    denom = len(ranks)
    if denom <= 0:
        return {
            "next_tool_recall@1": 0.0,
            "next_tool_recall@5": 0.0,
            "next_tool_mrr": 0.0,
            "candidate_count": _mean(candidate_counts),
        }
    valid = [rank for rank in ranks if rank is not None]
    return {
        "next_tool_recall@1": sum(1 for rank in valid if rank <= 1) / denom,
        "next_tool_recall@5": sum(1 for rank in valid if rank <= 5) / denom,
        "next_tool_mrr": sum(1.0 / float(rank) for rank in valid) / denom,
        "candidate_count": _mean(candidate_counts),
    }


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _rank_candidates_with_stage0_prior(
    model: Any,
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    batch_size: int,
    device: torch.device | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"ranked_rows": 0}
    device = torch.device(device or _model_device(model))
    ranked_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    missing_candidates = 0
    missing_skills = 0
    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "eval", None)):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            encoded = model.encode_observations([str(row.get("state_text") or "") for row in batch]).to(device)
            logits = model.skill_table.retrieval_logits(encoded).detach().float().cpu()
            for row_offset, row in enumerate(batch):
                scored: list[tuple[str, float]] = []
                for candidate_id in [str(item) for item in row.get("candidate_next_skill_ids") or []]:
                    skill_idx = skill_id_to_idx.get(candidate_id)
                    if skill_idx is None:
                        missing_skills += 1
                        continue
                    scored.append((candidate_id, float(logits[row_offset, int(skill_idx)].item())))
                scored.sort(key=lambda item: (-item[1], item[0]))
                ranked_ids = [item[0] for item in scored]
                positive_id = str(row.get("next_skill_id") or "")
                rank = ranked_ids.index(positive_id) + 1 if positive_id in ranked_ids else None
                if rank is None:
                    missing_candidates += 1
                copied = dict(row)
                copied["candidate_next_skill_ids"] = ranked_ids
                copied["candidate_next_skill_indices"] = [int(skill_id_to_idx[item]) for item in ranked_ids]
                copied["candidate_next_prior_scores"] = [float(item[1]) for item in scored]
                copied["positive_next_skill_rank"] = rank
                ranked_rows.append(copied)
                ranks.append(rank)
                candidate_counts.append(len(ranked_ids))
    if was_training and callable(getattr(model, "train", None)):
        model.train()
    metrics = _metrics_from_ranks(ranks, candidate_counts)
    return ranked_rows, {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": len(ranked_rows) - missing_candidates,
        "positive_missing_rows": int(missing_candidates),
        "candidate_skill_missing_count": int(missing_skills),
        "positive_stage0_domain_recall@1": metrics["next_tool_recall@1"],
        "positive_stage0_domain_recall@5": metrics["next_tool_recall@5"],
        "positive_stage0_domain_mrr": metrics["next_tool_mrr"],
        "candidate_count_mean": metrics["candidate_count"],
    }


def _strict_stage4_metrics(eval_metrics: dict[str, Any], *, retained_rows: int, source_rows: int) -> dict[str, float]:
    denom = float(source_rows) if source_rows else 0.0

    def strict_value(key: str) -> float:
        value = float(eval_metrics.get(key, 0.0) or 0.0)
        return value * float(retained_rows) / denom if denom > 0 else 0.0

    return {
        "strict_stage4_next_skill_recall@1": strict_value("stage4_next_skill_recall@1"),
        "strict_stage4_next_skill_recall@5": strict_value("stage4_next_skill_recall@5"),
        "strict_stage4_next_skill_mrr": strict_value("stage4_next_skill_mrr"),
        "retained_row_fraction": float(retained_rows) / denom if denom > 0 else 0.0,
        "retained_rows": float(retained_rows),
        "source_rows": float(source_rows),
        "retained_stage4_candidate_count": float(eval_metrics.get("stage4_candidate_count", 0.0) or 0.0),
    }


def load_bfcl_route_corpus(
    data_root: str | Path,
    *,
    categories: Iterable[str] | None = None,
    max_rows_per_category: int | None = None,
    include_trivial: bool = False,
) -> BFCLRouteCorpus:
    data_root = Path(data_root)
    categories = list(categories or DEFAULT_BFCL_ROUTE_CATEGORIES)
    skills: list[dict[str, Any]] = [_start_skill()]
    source_rows: list[dict[str, Any]] = []
    category_counts: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    skill_ids_seen = {"bfcl/__start__"}
    multiturn_functions = _load_multiturn_functions(data_root)
    multiturn_name_to_skill_id = {
        _clean_name(fn.get("name")): _bfcl_multiturn_skill_id(_clean_name(fn.get("name"))) for fn in multiturn_functions
    }
    for fn in multiturn_functions:
        sid = multiturn_name_to_skill_id[_clean_name(fn.get("name"))]
        if sid not in skill_ids_seen:
            skills.append(_skill_from_function(sid, fn, category="BFCL_v3_multi_turn"))
            skill_ids_seen.add(sid)

    for category in categories:
        category_path = data_root / f"{category}.json"
        counts = {
            "raw_rows": 0,
            "source_rows": 0,
            "skipped_trivial_candidate_rows": 0,
            "missing_answer_rows": 0,
            "missing_gt_candidate_rows": 0,
        }
        if not category_path.exists():
            counts["missing_file"] = 1
            category_counts[category] = counts
            skipped["missing_category_file"] += 1
            continue
        rows = _read_json_or_jsonl(category_path)
        if max_rows_per_category is not None:
            rows = rows[: max(0, int(max_rows_per_category))]
        answers = _answers_by_id(data_root, category)
        counts["raw_rows"] = len(rows)
        for row in rows:
            if not isinstance(row, dict):
                skipped["row_not_dict"] += 1
                continue
            raw_id = str(row.get("id") or "")
            ground_truth = answers.get(raw_id)
            if ground_truth is None:
                counts["missing_answer_rows"] += 1
                continue
            if category in MULTI_TURN_CATEGORIES:
                candidate_ids = list(multiturn_name_to_skill_id.values())
                candidate_names = list(multiturn_name_to_skill_id.keys())
                if len(candidate_ids) <= 1 and not include_trivial:
                    counts["skipped_trivial_candidate_rows"] += 1
                    continue
                question = row.get("question") if isinstance(row.get("question"), list) else []
                previous_calls: list[str] = []
                previous_skill_id = "bfcl/__start__"
                turns = ground_truth if isinstance(ground_truth, list) else []
                for turn_index, turn_calls in enumerate(turns):
                    call_names = _answer_names_from_ground_truth(turn_calls)
                    for call_index, name in enumerate(call_names):
                        if name not in multiturn_name_to_skill_id:
                            counts["missing_gt_candidate_rows"] += 1
                            previous_calls.append(name)
                            previous_skill_id = "bfcl/__start__"
                            continue
                        source_rows.append(
                            {
                                "benchmark": "bfcl",
                                "source_benchmark": "bfcl",
                                "split": "eval",
                                "task_id": f"{category}/{raw_id}/{turn_index}/{call_index}",
                                "trajectory_id": f"bfcl/{category}/{raw_id}",
                                "bfcl_category": category,
                                "bfcl_id": raw_id,
                                "turn_index": int(turn_index),
                                "call_index": int(call_index),
                                "state_text": _multiturn_state_text(
                                    category=category,
                                    raw_id=raw_id,
                                    question=question,
                                    turn_index=turn_index,
                                    previous_calls=previous_calls,
                                ),
                                "history_text": "\n".join(previous_calls),
                                "action_text": previous_calls[-1] if previous_calls else "previous_function: START",
                                "next_observation_text": "",
                                "skill_id": previous_skill_id,
                                "next_skill_id": multiturn_name_to_skill_id[name],
                                "next_skill_name": name,
                                "candidate_next_skill_ids": list(candidate_ids),
                                "candidate_next_skill_names": list(candidate_names),
                                "provenance": {
                                    "source": "official_bfcl",
                                    "raw_source_path": str(category_path),
                                    "bfcl_category": category,
                                    "bfcl_id": raw_id,
                                },
                            }
                        )
                        counts["source_rows"] += 1
                        previous_calls.append(name)
                        previous_skill_id = multiturn_name_to_skill_id[name]
            else:
                functions = [fn for fn in row.get("function") or [] if isinstance(fn, dict) and _clean_name(fn.get("name"))]
                if len(functions) <= 1 and not include_trivial:
                    counts["skipped_trivial_candidate_rows"] += 1
                    continue
                candidate_ids: list[str] = []
                name_to_skill_id: dict[str, str] = {}
                for idx, fn in enumerate(functions):
                    name = _clean_name(fn.get("name"))
                    sid = _bfcl_skill_id(category, raw_id, idx, name)
                    candidate_ids.append(sid)
                    name_to_skill_id[name] = sid
                    if sid not in skill_ids_seen:
                        skills.append(_skill_from_function(sid, fn, category=category))
                        skill_ids_seen.add(sid)
                previous_calls: list[str] = []
                previous_skill_id = "bfcl/__start__"
                for call_index, name in enumerate(_answer_names_from_ground_truth(ground_truth)):
                    if name not in name_to_skill_id:
                        counts["missing_gt_candidate_rows"] += 1
                        continue
                    source_rows.append(
                        {
                            "benchmark": "bfcl",
                            "source_benchmark": "bfcl",
                            "split": "eval",
                            "task_id": f"{category}/{raw_id}/0/{call_index}",
                            "trajectory_id": f"bfcl/{category}/{raw_id}",
                            "bfcl_category": category,
                            "bfcl_id": raw_id,
                            "turn_index": 0,
                            "call_index": int(call_index),
                            "state_text": _single_turn_state_text(
                                category=category,
                                raw_id=raw_id,
                                question=row.get("question"),
                                previous_calls=previous_calls,
                            ),
                            "history_text": "\n".join(previous_calls),
                            "action_text": previous_calls[-1] if previous_calls else "previous_function: START",
                            "next_observation_text": "",
                            "skill_id": previous_skill_id,
                            "next_skill_id": name_to_skill_id[name],
                            "next_skill_name": name,
                            "candidate_next_skill_ids": list(candidate_ids),
                            "candidate_next_skill_names": [_clean_name(fn.get("name")) for fn in functions],
                            "provenance": {
                                "source": "official_bfcl",
                                "raw_source_path": str(category_path),
                                "bfcl_category": category,
                                "bfcl_id": raw_id,
                            },
                        }
                    )
                    counts["source_rows"] += 1
                    previous_calls.append(name)
                    previous_skill_id = name_to_skill_id[name]
        category_counts[category] = counts

    report = {
        "status": "ok",
        "benchmark": "bfcl",
        "data_root": str(data_root),
        "categories": categories,
        "skill_count": len(skills),
        "source_row_count": len(source_rows),
        "include_trivial": bool(include_trivial),
        "category_counts": category_counts,
        "skipped_reasons": dict(sorted(skipped.items())),
        "paper_scope_note": "Official BFCL-derived next-function routing rows, not official BFCL leaderboard generation score.",
    }
    return BFCLRouteCorpus(skills=skills, source_rows=source_rows, report=report)


def _rank_rows(
    *,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, float]]]:
    skill_ids = [str(skill.get("skill_id") or "") for skill in skills]
    skill_id_to_idx = {sid: idx for idx, sid in enumerate(skill_ids)}
    scores = query_embs.float() @ skill_embs.float().t()
    ranked_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    by_category: dict[str, list[tuple[int | None, int]]] = defaultdict(list)
    missing_candidate_skill = 0
    for row_idx, row in enumerate(rows):
        scored: list[tuple[str, float]] = []
        for candidate_id in [str(item) for item in row.get("candidate_next_skill_ids") or []]:
            skill_idx = skill_id_to_idx.get(candidate_id)
            if skill_idx is None:
                missing_candidate_skill += 1
                continue
            scored.append((candidate_id, float(scores[row_idx, skill_idx].item())))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked_ids = [item[0] for item in scored]
        positive_id = str(row.get("next_skill_id") or "")
        rank = ranked_ids.index(positive_id) + 1 if positive_id in ranked_ids else None
        copied = dict(row)
        copied["candidate_next_skill_ids"] = ranked_ids
        copied["candidate_next_skillrouter_scores"] = [float(item[1]) for item in scored]
        copied["positive_next_skill_rank"] = rank
        ranked_rows.append(copied)
        ranks.append(rank)
        candidate_counts.append(len(ranked_ids))
        by_category[str(row.get("bfcl_category") or "unknown")].append((rank, len(ranked_ids)))
    metrics = _metrics_from_ranks(ranks, candidate_counts)
    report = {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": sum(1 for rank in ranks if rank is not None),
        "positive_missing_rows": sum(1 for rank in ranks if rank is None),
        "candidate_skill_missing_count": int(missing_candidate_skill),
        "candidate_count_mean": _mean(candidate_counts),
        "metrics": metrics,
    }
    by_category_metrics = {
        category: {
            **_metrics_from_ranks([rank for rank, _count in values], [count for _rank, count in values]),
            "row_count": float(len(values)),
        }
        for category, values in sorted(by_category.items())
    }
    return ranked_rows, report, by_category_metrics


def _skillrouter_query_text(row: dict[str, Any]) -> str:
    return (
        "Instruct: Given a task description and interaction history, retrieve the most relevant "
        "function tool that should be called next\nQuery:"
        f"{str(row.get('state_text') or '')[:1500]}"
    )


def run_bfcl_skillrouter_frozen_eval(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    categories: Iterable[str] | None = None,
    max_rows_per_category: int | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 16,
    max_length: int = 2048,
    adapter_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_bfcl_route_corpus(
        data_root,
        categories=categories,
        max_rows_per_category=max_rows_per_category,
    )
    source_rows = (
        corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    )
    skills_path = output_dir / "bfcl_skill_pool.jsonl"
    source_rows_path = output_dir / "bfcl_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_query_text(row) for row in source_rows],
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_skill_text(skill) for skill in corpus.skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    query_embs, skill_embs, adapter_report = _apply_skillrouter_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    ranked_rows, ranking_report, metrics_by_category = _rank_rows(
        rows=source_rows,
        skills=corpus.skills,
        query_embs=query_embs,
        skill_embs=skill_embs,
    )
    ranked_path = output_dir / "bfcl_skillrouter_ranked_rows.jsonl"
    _write_jsonl(ranked_path, ranked_rows)
    blockers: list[str] = []
    if not source_rows:
        blockers.append("no_source_eval_rows")
    if int(ranking_report.get("candidate_skill_missing_count") or 0) > 0:
        blockers.append("candidate_skill_missing")
    method = "skillrouter_finetuned_biencoder_adapter" if adapter_checkpoint_path is not None else "skillrouter_frozen_biencoder"
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "bfcl",
        "method": method,
        "baseline_family": (
            "SkillRouter-compatible finetuned bi-encoder adapter on official BFCL-derived next-function routing"
            if adapter_checkpoint_path is not None
            else "SkillRouter-compatible frozen bi-encoder on official BFCL-derived next-function routing"
        ),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "ranked_rows_path": str(ranked_path),
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(ranked_rows),
        "corpus_report": corpus.report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "metrics_by_category": metrics_by_category,
        "config": {
            "categories": None if categories is None else list(categories),
            "max_rows_per_category": max_rows_per_category,
            "max_eval_rows": max_eval_rows,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        },
        "paper_scope_note": "Official BFCL-derived next-function routing evaluation; not an official BFCL leaderboard score.",
    }
    report_name = "bfcl_skillrouter_finetuned_eval_report.json" if adapter_checkpoint_path is not None else "bfcl_skillrouter_frozen_eval_report.json"
    _write_json(output_dir / report_name, report)
    return report


def _stage0_prior_eval_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    for row in rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or []]
        positive = str(row.get("next_skill_id") or "")
        candidate_counts.append(len(candidates))
        ranks.append(candidates.index(positive) + 1 if positive in candidates else None)
    return {
        "stage4_act_count": float(len(rows)),
        "stage4_next_skill_recall@1": _metrics_from_ranks(ranks, candidate_counts)["next_tool_recall@1"],
        "stage4_next_skill_recall@5": _metrics_from_ranks(ranks, candidate_counts)["next_tool_recall@5"],
        "stage4_next_skill_mrr": _metrics_from_ranks(ranks, candidate_counts)["next_tool_mrr"],
        "stage4_candidate_count": _mean(candidate_counts),
    }


def _stage4_rows_from_ranked_bfcl(
    source_rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    candidate_counts: list[int] = []
    for row in source_rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or [] if str(item) in skill_id_to_idx]
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if next_skill_id not in skill_id_to_idx:
            skipped["next_skill_not_in_pool"] += 1
            continue
        if next_skill_id not in candidates:
            skipped["next_positive_missing_from_candidates"] += 1
            continue
        prior_scores = list(row.get("candidate_next_prior_scores") or [])
        if len(prior_scores) != len(row.get("candidate_next_skill_ids") or []):
            prior_scores = [-float(idx) for idx in range(len(row.get("candidate_next_skill_ids") or []))]
        prior_scores = prior_scores[: len(candidates)]
        positive_pos = candidates.index(next_skill_id)
        rows.append(
            {
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("call_index", row.get("turn_index", 0)),
                "benchmark": "bfcl",
                "source_benchmark": "bfcl",
                "bfcl_category": row.get("bfcl_category"),
                "state_text": str(row.get("state_text") or ""),
                "action_text": str(row.get("action_text") or ""),
                "next_observation_text": str(row.get("next_observation_text") or ""),
                "skill_id": current_skill_id,
                "next_skill_id": next_skill_id,
                "skill_idx": int(skill_id_to_idx[current_skill_id]),
                "positive_next_skill_idx": int(skill_id_to_idx[next_skill_id]),
                "positive_next_skill_position": int(positive_pos),
                "candidate_next_skill_ids": candidates,
                "candidate_next_skill_indices": [int(skill_id_to_idx[item]) for item in candidates],
                "candidate_next_prior_scores": [float(value) for value in prior_scores],
                "positive_injected": False,
                "provenance": {
                    "source": "official_bfcl_derived_stage0_ranked",
                    "original_provenance": row.get("provenance") or {},
                },
            }
        )
        candidate_counts.append(len(candidates))
    return rows, {
        "source_rows": len(source_rows),
        "stage4_rows": len(rows),
        "positive_injected_rows": 0,
        "candidate_source": "official_bfcl_stage0_ranked",
        "candidate_count_mean": _mean(candidate_counts),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _numeric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def _build_bfcl_clstr_report(
    *,
    output_dir: str | Path,
    data_root: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    skills_path: str | Path,
    source_eval_rows: int,
    retained_eval_rows: int,
    corpus_report: dict[str, Any],
    route_data_report: dict[str, Any],
    memory_report: dict[str, Any],
    stage0_prior_eval: dict[str, Any],
    base_eval: dict[str, Any],
    stage4_eval: dict[str, Any],
    config: dict[str, Any],
    stage0_prior_report: dict[str, Any],
    base_eval_by_benchmark: dict[str, Any],
    stage4_eval_by_benchmark: dict[str, Any],
    model_load: dict[str, Any],
    candidate_recall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_prior = _strict_stage4_metrics(stage0_prior_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_base = _strict_stage4_metrics(base_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_stage4 = _strict_stage4_metrics(stage4_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    if int(route_data_report.get("positive_injected_rows") or 0) > 0:
        blockers.append("gold_positive_injected")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "bfcl",
        "stage4_route": "official_bfcl_stage0_ranked_logged_online_memory",
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "skills_path": str(skills_path),
        "source_eval_rows": int(source_eval_rows),
        "retained_eval_rows": int(retained_eval_rows),
        "corpus_report": corpus_report,
        "stage0_prior_report": stage0_prior_report,
        "route_data_report": route_data_report,
        "memory_report": memory_report,
        "stage0_prior_eval": stage0_prior_eval,
        "base_eval": base_eval,
        "stage4_eval": stage4_eval,
        "base_eval_by_benchmark": base_eval_by_benchmark,
        "stage4_eval_by_benchmark": stage4_eval_by_benchmark,
        "strict": {"stage0_prior": strict_prior, "base": strict_base, "stage4": strict_stage4},
        "strict_delta_base_vs_stage0_prior": _numeric_delta(strict_base, strict_prior),
        "strict_delta_stage4_vs_base": _numeric_delta(strict_stage4, strict_base),
        "strict_delta_stage4_vs_stage0_prior": _numeric_delta(strict_stage4, strict_prior),
        "candidate_recall": candidate_recall or {},
        "config": config,
        "model_load": model_load,
        "paper_scope_note": "Official BFCL-derived next-function routing evaluation; not an official BFCL leaderboard score.",
    }


def run_bfcl_full_clstr_route_eval(
    *,
    data_root: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    output_dir: str | Path,
    categories: Iterable[str] | None = None,
    max_rows_per_category: int | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 8,
    stage0_candidate_batch_size: int = 16,
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.25,
    transition_scoring_mode: str = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
) -> dict[str, Any]:
    from clstr.logged_online_stage4_train import (
        attach_trajectory_prefix_online_memory_scores,
        evaluate_logged_online_stage4_rows,
        evaluate_logged_online_stage4_rows_by_benchmark,
    )
    from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    corpus = load_bfcl_route_corpus(
        data_root,
        categories=categories,
        max_rows_per_category=max_rows_per_category,
    )
    source_rows = (
        corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    )
    skills_path = output_dir / "bfcl_skill_pool.jsonl"
    source_rows_path = output_dir / "bfcl_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    skill_id_to_idx = {str(skill.get("skill_id") or ""): idx for idx, skill in enumerate(corpus.skills)}
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    stage4_load_report = None
    if stage4_checkpoint_path:
        stage4_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    ranked_rows, stage0_prior_report = _rank_candidates_with_stage0_prior(
        model,
        source_rows,
        skill_id_to_idx,
        batch_size=stage0_candidate_batch_size,
        device=device,
    )
    stage4_rows, route_data_report = _stage4_rows_from_ranked_bfcl(ranked_rows, skill_id_to_idx)
    scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        stage4_rows,
        feedback_rows=stage4_rows,
        next_skill_bonus=online_memory_next_skill_bonus,
        exact_transition_bonus=online_memory_exact_transition_bonus,
        memory_mode=online_memory_mode,
    )
    stage0_prior_eval = _stage0_prior_eval_from_rows(stage4_rows)
    base_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
    )
    base_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
    )
    stage4_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
    )
    stage4_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
    )
    local_pool_size = declared_candidate_pool_size(ranked_rows)
    candidate_recall = candidate_recall_protocol_metadata(
        pool_protocol="benchmark_local",
        candidate_source=str(
            route_data_report.get("candidate_source") or "official_bfcl_stage0_ranked"
        ),
        legal_pool_size=local_pool_size,
        static_k=local_pool_size,
        dynamic_extra_k=64,
        final_k=max(1, local_pool_size),
        causal_sequential=source_rows_have_causal_sequence(source_rows),
    )
    candidate_recall.update(
        {
            "predeclared_sequential_subset_only": True,
            "requires_positive_causal_update": True,
        }
    )
    report = _build_bfcl_clstr_report(
        output_dir=output_dir,
        data_root=data_root,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        skills_path=skills_path,
        source_eval_rows=len(source_rows),
        retained_eval_rows=len(stage4_rows),
        corpus_report=corpus.report,
        route_data_report=route_data_report,
        memory_report=memory_report,
        stage0_prior_eval=stage0_prior_eval,
        base_eval=base_eval,
        stage4_eval=stage4_eval,
        stage0_prior_report=stage0_prior_report,
        base_eval_by_benchmark=base_eval_by_benchmark,
        stage4_eval_by_benchmark=stage4_eval_by_benchmark,
        config={
            "categories": None if categories is None else list(categories),
            "max_rows_per_category": max_rows_per_category,
            "max_eval_rows": max_eval_rows,
            "batch_size": int(batch_size),
            "stage0_candidate_batch_size": int(stage0_candidate_batch_size),
            "online_memory_mode": online_memory_mode,
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
        },
        model_load={
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
        },
        candidate_recall=candidate_recall,
    )
    _write_json(output_dir / "bfcl_full_clstr_route_eval_report.json", report)
    return report
