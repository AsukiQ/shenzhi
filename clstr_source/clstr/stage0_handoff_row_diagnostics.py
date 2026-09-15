from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import torch

from clstr.full_base_train import (
    STAGE0_HANDOFF_QUERY_MODES,
    _encode_stage0_handoff_queries,
    _equivalent_skill_ids_by_skill_id,
    _read_jsonl,
    _explicit_inventory_skill_ids_ordered,
    _skill_id,
    _stage0_unified_static_logits,
    _split_name,
    _stage0_handoff_query_text,
)
from clstr.stage0_audit_sampling import select_rows_round_robin
from clstr.stage0_handoff_audit import TRAIN_SPLIT_NAMES
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint
from clstr.trajectory_inventory import backfill_tool_inventory_from_trajectory_rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _short(value: Any, limit: int = 512) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _skill_summary(skill: dict[str, Any] | None) -> dict[str, Any]:
    if not skill:
        return {}
    input_schema = skill.get("input_schema") or {}
    required = input_schema.get("required_parameters") if isinstance(input_schema, dict) else None
    optional = input_schema.get("optional_parameters") if isinstance(input_schema, dict) else None
    return {
        "skill_id": skill.get("skill_id") or skill.get("id"),
        "canonical_skill_id": skill.get("canonical_skill_id"),
        "name": skill.get("name"),
        "source": skill.get("source"),
        "description": _short(skill.get("description"), 320),
        "body": _short(skill.get("body"), 320),
        "required_parameters": required or [],
        "optional_parameters": optional or [],
        "alias_skill_ids": skill.get("alias_skill_ids") or [],
        "source_files": skill.get("source_files") or [],
    }


def _rank_from_logits(row_logits: torch.Tensor, label: int) -> int:
    if row_logits.ndim != 1:
        raise ValueError("rank expects a rank-1 logits tensor")
    label = int(label)
    if label < 0 or label >= int(row_logits.numel()):
        raise ValueError(f"label out of range: {label}")
    gold_score = row_logits[label]
    return int((row_logits > gold_score).sum().detach().cpu().item()) + 1


def _rank_bucket(rank: int) -> str:
    if rank <= 20:
        return "<=20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    if rank <= 200:
        return "101-200"
    if rank <= 500:
        return "201-500"
    if rank <= 1000:
        return "501-1000"
    return ">1000"


def _traject_bucket_key(row: dict[str, Any]) -> tuple[str, str]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return (
        str(provenance.get("domain") or "unknown_domain"),
        str(provenance.get("trajectory_type") or "unknown_type"),
    )


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    skill_id_to_idx: dict[str, int],
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if _split_name(row) not in TRAIN_SPLIT_NAMES:
            continue
        if str(row.get("benchmark") or "") != benchmark:
            continue
        if not (row.get("loss_mask") or {}).get("L_trans_skill_ce"):
            continue
        if str(row.get("next_skill_id") or "") not in skill_id_to_idx:
            continue
        filtered.append(row)
    return filtered


def _group_stats(records: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {"row_count": 0}
    ranks = [int(row["gold_rank"]) for row in records]
    stats: dict[str, Any] = {
        "row_count": count,
        "mean_rank": float(sum(ranks) / count),
        "median_rank": float(median(ranks)),
        "rank_bucket_counts": dict(sorted(Counter(str(row.get("rank_bucket")) for row in records).items())),
    }
    for k in k_values:
        stats[f"recall@{k}"] = sum(1 for rank in ranks if rank <= int(k)) / count
    return stats


def _breakdown(records: list[dict[str, Any]], key: str, k_values: tuple[int, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: _group_stats(items, k_values) for name, items in sorted(groups.items())}


def summarize_stage0_handoff_row_records(
    records: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (20, 50, 100, 200, 500),
) -> dict[str, Any]:
    missed_500 = [row for row in records if int(row.get("gold_rank") or 0) > 500]
    top1_when_missed = Counter(
        str((row.get("top_predictions") or [{}])[0].get("skill_id") or "<missing>")
        for row in missed_500
        if row.get("top_predictions")
    )
    gold_missed = Counter(str(row.get("next_skill_id") or "<missing>") for row in missed_500)
    return {
        "overall": _group_stats(records, k_values),
        "by_domain": _breakdown(records, "domain", k_values),
        "by_trajectory_type": _breakdown(records, "trajectory_type", k_values),
        "missed_500_count": len(missed_500),
        "top_gold_skill_ids_missed_500": gold_missed.most_common(20),
        "top1_skill_ids_when_gold_missed_500": top1_when_missed.most_common(20),
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    overall = summary.get("overall") or {}
    lines = [
        "# Stage0 TrajectBench Handoff Row Diagnostics",
        "",
        f"- status: `{report.get('status')}`",
        f"- row count: `{overall.get('row_count')}`",
        f"- mean rank: `{overall.get('mean_rank')}`",
        f"- median rank: `{overall.get('median_rank')}`",
        f"- recall@100: `{overall.get('recall@100')}`",
        f"- recall@200: `{overall.get('recall@200')}`",
        f"- recall@500: `{overall.get('recall@500')}`",
        f"- missed@500 count: `{summary.get('missed_500_count')}`",
        "",
        "## Rank Buckets",
        "",
    ]
    for name, count in (overall.get("rank_bucket_counts") or {}).items():
        lines.append(f"- `{name}`: `{count}`")
    for title, key in (("By Domain", "by_domain"), ("By Trajectory Type", "by_trajectory_type")):
        groups = summary.get(key) or {}
        if not groups:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| group | rows | r@100 | r@200 | r@500 | mean rank |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in groups.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(name),
                        str(metrics.get("row_count")),
                        str(metrics.get("recall@100")),
                        str(metrics.get("recall@200")),
                        str(metrics.get("recall@500")),
                        str(metrics.get("mean_rank")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def audit_stage0_handoff_row_diagnostics_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    train_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    benchmark: str = "traject_bench",
    query_mode: str = "skillrouter_state",
    max_rows: int | None = 960,
    batch_size: int = 8,
    top_k: int = 10,
    k_values: tuple[int, ...] | list[int] = (20, 50, 100, 200, 500),
    model_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if query_mode not in STAGE0_HANDOFF_QUERY_MODES:
        raise ValueError(f"unsupported Stage0 handoff query mode: {query_mode}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skills = _read_jsonl(skills_path)
    skill_ids = [str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)]
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    skill_by_id = {skill_id: skill for skill_id, skill in zip(skill_ids, skills)}
    equivalent_by_id = _equivalent_skill_ids_by_skill_id(skills)
    raw_rows = _read_jsonl(train_path)
    raw_rows, tool_inventory_backfill_report = backfill_tool_inventory_from_trajectory_rows(raw_rows)
    filtered = _filter_rows(raw_rows, benchmark=benchmark, skill_id_to_idx=skill_id_to_idx)
    rows = select_rows_round_robin(filtered, max_rows, key_fn=_traject_bucket_key)
    if not rows:
        raise ValueError(f"no rows available for Stage0 handoff row diagnostics: benchmark={benchmark}")

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=checkpoint_path,
        skills_path=skills_path,
        model_cache_dir=model_cache_dir or output_dir / "model_cache",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    k_values = tuple(sorted({max(1, int(k)) for k in k_values}))
    top_k = max(1, int(top_k))
    skill_count = len(skill_ids)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            texts = [_stage0_handoff_query_text(row, target="next", mode=query_mode) for row in batch]
            h = _encode_stage0_handoff_queries(
                model,
                texts,
                query_mode=query_mode,
            ).to(device)
            logits = _stage0_unified_static_logits(model, h, skill_count)
            logits = logits[:, :skill_count]
            for local_idx, row in enumerate(batch):
                next_skill_id = str(row.get("next_skill_id") or "")
                gold_idx = int(skill_id_to_idx[next_skill_id])
                row_logits = logits[local_idx]
                full_pool_gold_rank = _rank_from_logits(row_logits, gold_idx)
                inventory_indices = [
                    int(skill_id_to_idx[skill_id])
                    for skill_id in _explicit_inventory_skill_ids_ordered(row)
                    if skill_id in skill_id_to_idx
                ]
                if inventory_indices and gold_idx in inventory_indices:
                    inventory_tensor = torch.tensor(inventory_indices, dtype=torch.long, device=row_logits.device)
                    ranking_logits = row_logits.index_select(0, inventory_tensor)
                    local_gold_idx = inventory_indices.index(gold_idx)
                    gold_rank = _rank_from_logits(ranking_logits, local_gold_idx)
                    top = torch.topk(ranking_logits, k=min(top_k, len(inventory_indices)))
                    top_indices = [inventory_indices[int(idx)] for idx in top.indices.detach().cpu().tolist()]
                else:
                    gold_rank = full_pool_gold_rank
                    top = torch.topk(row_logits, k=min(top_k, skill_count))
                    top_indices = [int(idx) for idx in top.indices.detach().cpu().tolist()]
                top_scores = [float(score) for score in top.values.detach().cpu().tolist()]
                provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
                current_skill_id = str(row.get("skill_id") or "")
                record = {
                    "row_index": start + local_idx,
                    "benchmark": benchmark,
                    "domain": str(provenance.get("domain") or "unknown_domain"),
                    "trajectory_type": str(provenance.get("trajectory_type") or "unknown_type"),
                    "task_id": row.get("task_id"),
                    "trajectory_id": row.get("trajectory_id"),
                    "step_index": row.get("step_index"),
                    "current_skill_id": current_skill_id,
                    "current_skill": _skill_summary(skill_by_id.get(current_skill_id)),
                    "next_skill_id": next_skill_id,
                    "next_skill": _skill_summary(skill_by_id.get(next_skill_id)),
                    "equivalent_next_skill_ids": equivalent_by_id.get(next_skill_id, []),
                    "gold_rank": int(gold_rank),
                    "full_pool_gold_rank": int(full_pool_gold_rank),
                    "inventory_candidate_count": len(inventory_indices),
                    "rank_bucket": _rank_bucket(gold_rank),
                    "gold_score": float(row_logits[gold_idx].detach().cpu().item()),
                    "top_predictions": [
                        {
                            "rank": rank + 1,
                            "skill_id": skill_ids[idx],
                            "score": score,
                            "skill": _skill_summary(skill_by_id.get(skill_ids[idx])),
                        }
                        for rank, (idx, score) in enumerate(zip(top_indices, top_scores))
                    ],
                    "query_text": _short(texts[local_idx], 1400),
                    "goal_text": _short(row.get("goal_text"), 700),
                    "state_text": _short(row.get("state_text"), 900),
                    "history_text": _short(row.get("history_text"), 700),
                    "action_text": _short(row.get("action_text"), 500),
                    "next_observation_text": _short(row.get("next_observation_text"), 700),
                    "provenance": provenance,
                }
                for k in k_values:
                    record[f"hit@{k}"] = bool(gold_rank <= int(k))
                records.append(record)

    rows_path = output_dir / "row_diagnostics.jsonl"
    _write_jsonl(rows_path, records)
    miss_path = output_dir / "missed_gt500.jsonl"
    _write_jsonl(miss_path, [row for row in records if int(row["gold_rank"]) > 500])
    summary = summarize_stage0_handoff_row_records(records, k_values=k_values)
    report = {
        "status": "ok",
        "checkpoint_path": str(checkpoint_path),
        "train_path": str(train_path),
        "skills_path": str(skills_path),
        "output_dir": str(output_dir),
        "row_diagnostics_path": str(rows_path),
        "missed_gt500_path": str(miss_path),
        "config": {
            "benchmark": benchmark,
            "query_mode": query_mode,
            "max_rows": max_rows,
            "batch_size": int(batch_size),
            "top_k": int(top_k),
            "k_values": list(k_values),
            "row_selection": "domain_trajectory_type_round_robin",
        },
        "data": {
            "raw_row_count": len(raw_rows),
            "filtered_row_count": len(filtered),
            "sampled_row_count": len(records),
            "skill_count": skill_count,
        },
        "tool_inventory_backfill": tool_inventory_backfill_report,
        "summary": summary,
        "model_config": model_config,
        "routing_init": routing_report,
    }
    _write_json(output_dir / "report.json", report)
    (output_dir / "summary.md").write_text(_markdown_summary(report), encoding="utf-8")
    return report
