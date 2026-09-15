from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from clstr.retrieval_metrics import load_jsonl_run, load_qrels_jsonl, load_trec_run


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _step_index(row: dict[str, Any]) -> int:
    value = row.get("step_index")
    if value is None:
        return 0
    return int(value)


def _load_query_metadata(queries_path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["query_id"]): row
        for row in _read_jsonl(queries_path)
        if row.get("query_id") is not None
    }


def _top1_by_query(run_path: str | Path, run_format: str) -> dict[str, str]:
    if run_format == "trec":
        run = load_trec_run(run_path)
    elif run_format == "jsonl":
        run = load_jsonl_run(run_path)
    else:
        raise ValueError(f"unsupported run format: {run_format}")
    return {
        query_id: ranked[0][0]
        for query_id, ranked in run.items()
        if ranked
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _score_group(trajectories: list[dict[str, Any]]) -> dict[str, float]:
    if not trajectories:
        return {
            "trajectory_count": 0.0,
            "step_count": 0.0,
            "missing_run_steps": 0.0,
            "step_top1_accuracy": 0.0,
            "ordered_exact_match": 0.0,
            "unordered_exact_match": 0.0,
            "inclusion": 0.0,
        }

    step_correct = 0
    step_total = 0
    missing_run_steps = 0
    ordered_exact: list[float] = []
    unordered_exact: list[float] = []
    inclusion: list[float] = []

    for trajectory in trajectories:
        steps = trajectory["steps"]
        ordered_ok = True
        gt_skill_ids: list[str] = []
        pred_skill_ids: list[str] = []
        for step in steps:
            positives = step["positive_skill_ids"]
            predicted = step.get("predicted_skill_id")
            gt_skill_ids.extend(positives)
            if predicted is None:
                missing_run_steps += 1
                ordered_ok = False
            else:
                pred_skill_ids.append(predicted)
                if predicted not in positives:
                    ordered_ok = False
            if predicted is not None and predicted in positives:
                step_correct += 1
            step_total += 1

        gt_set = set(gt_skill_ids)
        pred_set = set(pred_skill_ids)
        ordered_exact.append(1.0 if ordered_ok and len(pred_skill_ids) == len(steps) else 0.0)
        unordered_exact.append(1.0 if gt_set == pred_set else 0.0)
        inclusion.append(len(gt_set & pred_set) / max(1, len(gt_set)))

    return {
        "trajectory_count": float(len(trajectories)),
        "step_count": float(step_total),
        "missing_run_steps": float(missing_run_steps),
        "step_top1_accuracy": round(step_correct / step_total, 6) if step_total else 0.0,
        "ordered_exact_match": _mean(ordered_exact),
        "unordered_exact_match": _mean(unordered_exact),
        "inclusion": _mean(inclusion),
    }


def _build_trajectories(
    *,
    query_meta: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    top1: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_query_metadata = 0
    missing_trajectory_metadata = 0
    for query_id, positives in qrels.items():
        meta = query_meta.get(query_id)
        if meta is None:
            missing_query_metadata += 1
            continue
        trajectory_id = meta.get("trajectory_id")
        if trajectory_id in (None, "") or meta.get("step_index") is None:
            missing_trajectory_metadata += 1
            continue
        grouped[str(trajectory_id)].append(
            {
                "query_id": query_id,
                "step_index": _step_index(meta),
                "trajectory_type": str(meta.get("trajectory_type") or "unknown"),
                "positive_skill_ids": sorted(positives),
                "predicted_skill_id": top1.get(query_id),
            }
        )

    trajectories: list[dict[str, Any]] = []
    for trajectory_id, steps in grouped.items():
        ordered_steps = sorted(steps, key=lambda item: item["step_index"])
        trajectory_type = ordered_steps[0]["trajectory_type"] if ordered_steps else "unknown"
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_type": trajectory_type,
                "steps": ordered_steps,
            }
        )
    return sorted(trajectories, key=lambda item: item["trajectory_id"]), {
        "missing_query_metadata": missing_query_metadata,
        "missing_trajectory_metadata": missing_trajectory_metadata,
    }


def evaluate_traject_sequence_proxy(
    *,
    queries_path: str | Path,
    qrels_path: str | Path,
    run_path: str | Path,
    output_dir: str | Path,
    run_format: str = "trec",
    method: str = "unknown",
) -> dict[str, Any]:
    query_meta = _load_query_metadata(queries_path)
    qrels = load_qrels_jsonl(qrels_path)
    top1 = _top1_by_query(run_path, run_format)
    trajectories, dropped = _build_trajectories(query_meta=query_meta, qrels=qrels, top1=top1)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_type[trajectory["trajectory_type"]].append(trajectory)

    blockers: list[str] = []
    if not qrels:
        blockers.append("no_positive_qrels")
    if not trajectories:
        blockers.append("no_reconstructable_trajectories")

    report = {
        "status": "ok" if not blockers else "action_required",
        "benchmark": "TRAJECT-Bench",
        "method": method,
        "metric_scope": "TRAJECT selection-only sequence proxy",
        "official_metric_caveat": (
            "This report approximates EM/Inclusion over selected skill IDs only. "
            "It does not compute official TRAJECT Usage, Traj-Satisfy, or Acc."
        ),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "run_path": str(run_path),
        "run_format": run_format,
        "blockers": blockers,
        "qrel_query_count": len(qrels),
        "run_query_count": len(top1),
        "trajectory_count": len(trajectories),
        "dropped": dropped,
        "metrics": _score_group(trajectories),
        "by_trajectory_type": {
            trajectory_type: _score_group(rows)
            for trajectory_type, rows in sorted(by_type.items())
        },
    }
    _write_json(Path(output_dir) / "traject_sequence_proxy_metrics.json", report)
    return report
