from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


Qrels = dict[str, dict[str, int]]
Run = dict[str, list[tuple[str, float]]]


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


def _first_present(row: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def load_qrels_jsonl(path: str | Path) -> Qrels:
    """Load generic query-to-skill/tool relevance labels.

    Accepted document id fields intentionally cover CLSTR unified rows and
    common ToolRet/TRAJECT naming. Non-positive relevance rows are ignored.
    """
    qrels: Qrels = {}
    for row in _read_jsonl(path):
        query_id = _first_present(row, ("query_id", "qid", "task_id"))
        doc_id = _first_present(row, ("skill_id", "tool_id", "doc_id", "document_id", "api_id"))
        relevance = int(row.get("relevance", row.get("score", 1)))
        if query_id is None or doc_id is None or relevance <= 0:
            continue
        qrels.setdefault(str(query_id), {})[str(doc_id)] = relevance
    return qrels


def load_trec_run(path: str | Path) -> Run:
    ranked: dict[str, list[tuple[int, str, float]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"invalid TREC run line: {line}")
            query_id, _, doc_id, rank, score, _run_name = parts[:6]
            ranked.setdefault(str(query_id), []).append((int(rank), str(doc_id), float(score)))
    return {
        query_id: [(doc_id, score) for _rank, doc_id, score in sorted(rows, key=lambda item: item[0])]
        for query_id, rows in ranked.items()
    }


def load_jsonl_run(path: str | Path) -> Run:
    run: Run = {}
    for row in _read_jsonl(path):
        query_id = _first_present(row, ("query_id", "qid", "task_id"))
        ranked_ids = _first_present(
            row,
            (
                "ranked_skill_ids",
                "ranked_tool_ids",
                "ranked_doc_ids",
                "ranked_ids",
                "predicted_skill_ids",
                "predicted_tool_ids",
            ),
        )
        if query_id is None or ranked_ids is None:
            continue
        scores = row.get("scores")
        if scores is None:
            scores = [float(len(ranked_ids) - idx) for idx in range(len(ranked_ids))]
        if len(scores) != len(ranked_ids):
            raise ValueError(f"score count does not match ranked id count for query {query_id}")
        run[str(query_id)] = [(str(doc_id), float(score)) for doc_id, score in zip(ranked_ids, scores)]
    return run


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def _average_precision_at_k(ranked_doc_ids: Sequence[str], positives: dict[str, int], k: int) -> float:
    if not positives:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(positives), k)


def compute_ranking_metrics(qrels: Qrels, run: Run, k_values: Sequence[int] = (5, 10)) -> dict[str, float]:
    """Compute ToolRet/SKILLRET-style retrieval metrics without pytrec_eval."""
    query_ids = sorted(qrels)
    metrics: dict[str, float] = {
        "evaluated_queries": float(len(query_ids)),
        "missing_run_queries": float(sum(1 for query_id in query_ids if query_id not in run)),
    }
    if not query_ids:
        for k in k_values:
            metrics[f"NDCG@{k}"] = 0.0
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"MAP@{k}"] = 0.0
            metrics[f"Precision@{k}"] = 0.0
        return metrics

    totals = {
        metric_name: {int(k): 0.0 for k in k_values}
        for metric_name in ("NDCG", "Recall", "MAP", "Precision")
    }
    for query_id in query_ids:
        positives = qrels[query_id]
        ranked_doc_ids = [doc_id for doc_id, _score in run.get(query_id, [])]
        for k in k_values:
            k = int(k)
            top_ids = ranked_doc_ids[:k]
            hit_count = sum(1 for doc_id in top_ids if doc_id in positives)
            totals["Recall"][k] += hit_count / len(positives) if positives else 0.0
            totals["Precision"][k] += hit_count / k if k else 0.0
            totals["MAP"][k] += _average_precision_at_k(ranked_doc_ids, positives, k)

            rels = [positives.get(doc_id, 0) for doc_id in top_ids]
            ideal_rels = sorted(positives.values(), reverse=True)[:k]
            ideal = _dcg(ideal_rels)
            totals["NDCG"][k] += _dcg(rels) / ideal if ideal > 0.0 else 0.0

    denom = float(len(query_ids))
    for metric_name in ("NDCG", "Recall", "MAP", "Precision"):
        for k in k_values:
            metrics[f"{metric_name}@{int(k)}"] = round(totals[metric_name][int(k)] / denom, 6)
    return metrics


def evaluate_retrieval_run(
    *,
    qrels_path: str | Path,
    run_path: str | Path,
    output_dir: str | Path,
    run_format: str = "jsonl",
    k_values: Sequence[int] = (5, 10),
    benchmark: str = "generic_retrieval",
    method: str = "unknown",
) -> dict[str, Any]:
    qrels = load_qrels_jsonl(qrels_path)
    if run_format == "jsonl":
        run = load_jsonl_run(run_path)
    elif run_format == "trec":
        run = load_trec_run(run_path)
    else:
        raise ValueError(f"unsupported run format: {run_format}")
    metrics = compute_ranking_metrics(qrels, run, k_values=k_values)
    report = {
        "status": "ok",
        "benchmark": benchmark,
        "method": method,
        "qrels_path": str(qrels_path),
        "run_path": str(run_path),
        "run_format": run_format,
        "k_values": [int(k) for k in k_values],
        "qrel_query_count": len(qrels),
        "run_query_count": len(run),
        "metrics": metrics,
    }
    _write_json(Path(output_dir) / "metrics.json", report)
    return report
