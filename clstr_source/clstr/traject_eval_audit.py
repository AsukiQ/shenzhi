from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clstr.toolret_eval_audit import audit_toolret_eval_data


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trajectory_reconstruction_report(
    *,
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    qrel_query_ids = {
        str(row.get("query_id"))
        for row in qrels
        if row.get("query_id") is not None and int(row.get("relevance", 1)) > 0
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_trajectory_id = 0
    missing_step_index = 0
    positive_queries_with_trajectory_metadata = 0
    positive_queries_with_order_metadata = 0

    for row in queries:
        query_id = row.get("query_id")
        trajectory_id = row.get("trajectory_id")
        step_index = row.get("step_index")
        if trajectory_id in (None, ""):
            missing_trajectory_id += 1
        else:
            grouped[str(trajectory_id)].append(row)
            if query_id is not None and str(query_id) in qrel_query_ids:
                positive_queries_with_trajectory_metadata += 1
        if step_index is None:
            missing_step_index += 1
        elif query_id is not None and str(query_id) in qrel_query_ids:
            positive_queries_with_order_metadata += 1

    trajectory_type_counts = Counter()
    steps_per_trajectory: list[int] = []
    for rows in grouped.values():
        steps_per_trajectory.append(len(rows))
        trajectory_types = [str(row.get("trajectory_type") or "unknown") for row in rows]
        trajectory_type_counts[trajectory_types[0] if trajectory_types else "unknown"] += 1

    ordered_sequence_ready = (
        bool(qrel_query_ids)
        and positive_queries_with_trajectory_metadata == len(qrel_query_ids)
        and positive_queries_with_order_metadata == len(qrel_query_ids)
    )
    return {
        "trajectory_count": len(grouped),
        "trajectory_type_counts": dict(sorted(trajectory_type_counts.items())),
        "queries_with_trajectory_id": len(queries) - missing_trajectory_id,
        "queries_missing_trajectory_id": missing_trajectory_id,
        "queries_with_step_index": len(queries) - missing_step_index,
        "queries_missing_step_index": missing_step_index,
        "positive_qrel_query_count": len(qrel_query_ids),
        "positive_qrel_queries_with_trajectory_metadata": positive_queries_with_trajectory_metadata,
        "positive_qrel_queries_with_order_metadata": positive_queries_with_order_metadata,
        "max_steps_per_trajectory": max(steps_per_trajectory, default=0),
        "ordered_sequence_proxy_ready": ordered_sequence_ready,
    }


def _metric_readiness(
    *,
    base_status: str,
    positive_qrel_count: int,
    trajectory_report: dict[str, Any],
) -> dict[str, Any]:
    routing_ready = base_status == "ok" and positive_qrel_count > 0
    sequence_ready = routing_ready and bool(trajectory_report.get("ordered_sequence_proxy_ready"))
    return {
        "routing_eval": {
            "status": "ready" if routing_ready else "blocked",
            "metric_scope": "stepwise retrieval/routing metrics such as Recall@K, NDCG@K, MAP@K",
            "requires": ["queries.jsonl", "skills.jsonl", "qrels.jsonl", "CLSTR retrieval run"],
        },
        "trajectory_sequence_proxy": {
            "status": "ready" if sequence_ready else "blocked",
            "metric_scope": "trajectory-level top-1 skill sequence proxy grouped by trajectory_id and step_index",
            "requires": ["stepwise qrels", "trajectory_id", "step_index", "retrieval run"],
        },
        "official_em_inclusion": {
            "status": "partial_proxy_ready" if sequence_ready else "blocked",
            "metric_scope": "selection-only EM/Inclusion can be approximated from predicted skill/tool IDs; official tool-call outputs still need executor integration",
            "requires": ["predicted tool sequence mapped from run.tsv", "ground-truth trajectory sequence"],
        },
        "official_usage": {
            "status": "blocked",
            "metric_scope": "official Usage compares parameterized tool calls, not only selected skill IDs",
            "requires": ["downstream LM executor that emits tool names and parameter values"],
        },
        "official_traj_satisfy": {
            "status": "blocked",
            "metric_scope": "official trajectory satisfaction uses LLM judging over predicted trajectories",
            "requires": ["official TRAJECT evaluator", "judge model/API or local judge adapter"],
        },
        "official_acc": {
            "status": "blocked",
            "metric_scope": "official answer accuracy requires executed/generated final answers and LLM/answer matching",
            "requires": ["downstream executor outputs", "official answer evaluator"],
        },
    }


def audit_traject_eval_data(
    *,
    data_dir: str | Path = "data/traject_eval_traject_split_test",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    report = audit_toolret_eval_data(data_dir=data_dir)
    data_dir = Path(data_dir)
    if report.get("status") == "ok":
        queries = _read_jsonl(data_dir / "queries.jsonl")
        qrels = _read_jsonl(data_dir / "qrels.jsonl")
        trajectory_report = _trajectory_reconstruction_report(queries=queries, qrels=qrels)
        report = {
            **report,
            "trajectory_reconstruction": trajectory_report,
            "paper_metric_readiness": _metric_readiness(
                base_status=str(report.get("status")),
                positive_qrel_count=int(report.get("positive_qrel_count", 0)),
                trajectory_report=trajectory_report,
            ),
        }
    report = {
        **report,
        "benchmark": "TRAJECT-Bench",
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
