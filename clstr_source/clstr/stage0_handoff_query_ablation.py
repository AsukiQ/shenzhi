from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from clstr.full_base_train import (
    _encode,
    _read_jsonl,
    _skill_id,
    _skill_logits_and_memory,
    _skillrouter_query_text_from_raw,
    _split_name,
    _stage0_handoff_query_text,
)
from clstr.stage0_audit_sampling import select_audit_rows
from clstr.stage0_handoff_audit import TRAIN_SPLIT_NAMES
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint


QUERY_VARIANTS = (
    "baseline_skillrouter_state",
    "goal_only",
    "state_no_observation",
    "state_sanitized_observation",
    "goal_history_action",
)

_TAG_RE = re.compile(r"<[^>]+>")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compact_text(text: str, *, max_chars: int) -> str:
    text = html.unescape(str(text or ""))
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def sanitize_observation_for_handoff(value: Any, *, max_chars: int = 512) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return _compact_text(text, max_chars=max_chars)
    if not isinstance(parsed, dict):
        return _compact_text(str(parsed), max_chars=max_chars)
    error = str(parsed.get("error") or "").strip()
    response = parsed.get("response")
    if error:
        return _compact_text(f"tool_error: {error}", max_chars=max_chars)
    if response is None:
        return _compact_text(json.dumps(parsed, ensure_ascii=False), max_chars=max_chars)
    return _compact_text(str(response), max_chars=max_chars)


def _raw_handoff_query(row: dict[str, Any], *, target: str, variant: str) -> str:
    state = str(row.get("state_text") or "").strip()
    goal = str(row.get("goal_text") or row.get("task_text") or "").strip()
    history = str(row.get("history_text") or "").strip()
    action = str(row.get("action_text") or row.get("expert_action") or "").strip()
    if variant == "baseline_skillrouter_state":
        return _stage0_handoff_query_text(row, target=target, mode="raw_state")
    if variant == "goal_only":
        return f"goal: {goal}" if goal else state
    if variant == "state_no_observation":
        parts = [state]
        if target == "next" and action:
            parts.append(f"previous_action: {action}")
        return "\n".join(part for part in parts if part)
    if variant == "state_sanitized_observation":
        parts = [state]
        if target == "next" and action:
            parts.append(f"previous_action: {action}")
        if target == "next":
            observation = sanitize_observation_for_handoff(row.get("next_observation_text"), max_chars=512)
            if observation:
                parts.append(f"observation_summary: {observation}")
        return "\n".join(part for part in parts if part)
    if variant == "goal_history_action":
        parts = [f"goal: {goal}" if goal else state]
        if history:
            parts.append(f"previous_tools: {history}")
        if target == "next" and action:
            parts.append(f"previous_action: {action}")
        return "\n".join(part for part in parts if part)
    raise ValueError(f"unsupported query ablation variant: {variant}")


def build_handoff_ablation_query(row: dict[str, Any], *, target: str = "next", variant: str) -> str:
    if variant not in QUERY_VARIANTS:
        raise ValueError(f"unsupported query ablation variant: {variant}")
    raw = _raw_handoff_query(row, target=target, variant=variant)
    return _skillrouter_query_text_from_raw(raw)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str | None,
    skill_id_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if _split_name(row) not in TRAIN_SPLIT_NAMES:
            continue
        if benchmark and str(row.get("benchmark") or "") != benchmark:
            continue
        if not (row.get("loss_mask") or {}).get("L_trans_skill_ce"):
            continue
        if str(row.get("next_skill_id") or "") not in skill_id_to_idx:
            continue
        filtered.append(row)
    return filtered


def _empty_counts(k_values: tuple[int, ...]) -> dict[str, Any]:
    return {
        "total": 0,
        "hits": {int(k): 0 for k in k_values},
        "rank_sum": 0,
        "query_chars_sum": 0,
    }


def _add_rank(counts: dict[str, Any], rank: int, query_chars: int, k_values: tuple[int, ...]) -> None:
    counts["total"] += 1
    counts["rank_sum"] += int(rank)
    counts["query_chars_sum"] += int(query_chars)
    for k in k_values:
        if int(rank) <= int(k):
            counts["hits"][int(k)] += 1


def _finalize_counts(counts: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    total = max(1, int(counts["total"]))
    report: dict[str, Any] = {
        "total": int(counts["total"]),
        "mean_rank": float(counts["rank_sum"] / total),
        "mean_query_chars": float(counts["query_chars_sum"] / total),
    }
    for k in k_values:
        report[f"recall@{int(k)}"] = float(counts["hits"][int(k)] / total)
    return report


def _rank_from_logits(row_logits: torch.Tensor, label: int) -> int:
    gold_score = row_logits[int(label)]
    return int((row_logits > gold_score).sum().detach().cpu().item()) + 1


def audit_stage0_handoff_query_ablation_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_path: str | Path | None = None,
    benchmark: str | None = "toolbench_g3",
    variants: tuple[str, ...] | list[str] = QUERY_VARIANTS,
    top_k_values: tuple[int, ...] | list[int] = (20, 50, 100, 200, 500),
    max_rows: int | None = 1024,
    batch_size: int = 16,
    model_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    selected_variants = tuple(str(item) for item in variants)
    unsupported = [item for item in selected_variants if item not in QUERY_VARIANTS]
    if unsupported:
        raise ValueError(f"unsupported query ablation variants: {unsupported}")
    k_values = tuple(sorted({max(1, int(k)) for k in top_k_values}))

    skills = _read_jsonl(skills_path)
    skill_ids = [str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    all_rows = _read_jsonl(train_path)
    filtered = _filter_rows(all_rows, benchmark=benchmark, skill_id_to_idx=skill_id_to_idx)
    rows = select_audit_rows(filtered, max_rows)
    if not rows:
        raise ValueError(f"no rows available for query ablation: benchmark={benchmark}")

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=model_cache_dir,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    variant_reports: dict[str, Any] = {}
    with torch.no_grad():
        for variant in selected_variants:
            global_counts = _empty_counts(k_values)
            by_benchmark: dict[str, dict[str, Any]] = defaultdict(lambda: _empty_counts(k_values))
            for start in range(0, len(rows), max(1, int(batch_size))):
                batch = rows[start : start + max(1, int(batch_size))]
                texts = [build_handoff_ablation_query(row, target="next", variant=variant) for row in batch]
                h = _encode(model, texts).to(device)
                logits, _m = _skill_logits_and_memory(model, h, len(skill_ids))
                logits = logits[:, : len(skill_ids)]
                for local_idx, row in enumerate(batch):
                    positive = skill_id_to_idx[str(row.get("next_skill_id") or "")]
                    rank = _rank_from_logits(logits[local_idx], positive)
                    query_chars = len(texts[local_idx])
                    benchmark_name = str(row.get("benchmark") or "<missing>")
                    _add_rank(global_counts, rank, query_chars, k_values)
                    _add_rank(by_benchmark[benchmark_name], rank, query_chars, k_values)
            variant_reports[variant] = {
                "global": _finalize_counts(global_counts, k_values),
                "benchmarks": {
                    name: _finalize_counts(counts, k_values)
                    for name, counts in sorted(by_benchmark.items())
                },
            }

    report = {
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "benchmark": benchmark,
        "available_rows": len(filtered),
        "row_count": len(rows),
        "max_rows": max_rows,
        "skill_count": len(skill_ids),
        "variants": variant_reports,
        "model_config": model_config,
        "routing_init": routing_report,
        "note": "Query variants use only goal/state/history/action/current observation fields; next_skill_id and next_action_text are labels only.",
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
