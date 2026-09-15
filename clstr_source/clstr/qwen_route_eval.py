from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.bge_reranker import BGERerankerConfig, BGERerankerScorer
from clstr.full_base_train import _read_jsonl, _skill_id
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
from clstr.stage4_act_train import _as_stage4_handoff_row
from clstr.tau2_route_eval import _clean_benchmark_name, load_tau2_route_corpus
from clstr.toolbench_qwen_rerank import (
    aggregate_metric_rows,
    ranking_metrics_from_ranked_skill_ids,
    strict_metrics,
)
from clstr.toolbench_qwen_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    Qwen3RerankerConfig,
    Qwen3RerankerScorer,
    build_qwen3_reranker_document,
    rank_candidate_skill_ids_by_scores,
)


@dataclass(frozen=True)
class QwenRouteCorpus:
    benchmark: str
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    candidate_source: str
    report: dict[str, Any]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _route_eval_row_key(row: dict[str, Any], *, row_index: int) -> str:
    return (
        f"idx={int(row_index)}"
        f"|task={str(row.get('task_id') or '')}"
        f"|traj={str(row.get('trajectory_id') or '')}"
        f"|step={str(row.get('step_index') or '')}"
        f"|domain={str(row.get('domain') or '')}"
        f"|pos={str(row.get('next_skill_id') or row.get('positive_skill_id') or '')}"
    )


def _prediction_paths(output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    return (
        output_dir / "route_eval_predictions.jsonl",
        output_dir / "qwen3_embedding_reranker_predictions.jsonl",
    )


def _load_existing_route_predictions(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    generic_predictions_path, legacy_predictions_path = _prediction_paths(output_dir)
    source_path = generic_predictions_path if generic_predictions_path.is_file() else legacy_predictions_path
    if not source_path.is_file():
        return {}
    predictions: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_key = str(row.get("row_key") or "")
            metrics = row.get("metrics")
            if row_key and isinstance(metrics, dict):
                predictions[row_key] = row
    return predictions


def _append_route_eval_prediction(output_dir: str | Path, prediction: dict[str, Any], *, save_predictions: bool) -> None:
    if not save_predictions:
        return
    generic_predictions_path, legacy_predictions_path = _prediction_paths(output_dir)
    _append_jsonl(generic_predictions_path, prediction)
    _append_jsonl(legacy_predictions_path, prediction)


def select_embedding_topk(
    *,
    candidate_skill_ids: list[str],
    candidate_scores: list[float],
    top_k: int,
) -> list[str]:
    pairs = [
        (idx, str(skill_id), float(score))
        for idx, (skill_id, score) in enumerate(zip(candidate_skill_ids, candidate_scores))
    ]
    pairs.sort(key=lambda item: (-item[2], item[0]))
    return [skill_id for _, skill_id, _ in pairs[: max(1, int(top_k))]]


def metric_row_from_reranked_candidates(
    *,
    positive_skill_id: str,
    embedding_candidate_skill_ids: list[str],
    reranked_skill_ids: list[str],
) -> dict[str, float]:
    metrics = ranking_metrics_from_ranked_skill_ids(
        ranked_skill_ids=reranked_skill_ids,
        candidate_skill_ids=embedding_candidate_skill_ids,
        positive_skill_id=str(positive_skill_id),
        source_row_count=1,
    )
    metrics["positive_in_embedding_topk"] = 1.0 if str(positive_skill_id) in set(embedding_candidate_skill_ids) else 0.0
    return metrics


def is_reranker_disabled(value: str | Path | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "off", "false", "embedding_only", "retrieval_only"}


def reranker_family(model_name_or_path: str | Path) -> str:
    label = str(model_name_or_path).lower()
    if "bge-reranker" in label or "bge_reranker" in label:
        return "bge_sequence_classifier"
    config_path = Path(model_name_or_path) / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
        architectures = {str(item).lower() for item in config.get("architectures", [])}
        model_type = str(config.get("model_type") or "").lower()
        if (
            "xlmrobertaforsequenceclassification" in architectures
            or model_type == "xlm-roberta"
        ) and int(config.get("num_labels") or len(config.get("id2label", {})) or 1) == 1:
            return "bge_sequence_classifier"
    return "qwen3_causal_yes_no"


def embedding_model_family(model_name_or_path: str | Path) -> str:
    label = str(model_name_or_path).lower()
    if "bge-m3" in label or "bge_m3" in label:
        return "bge_m3"
    if "qwen3-embedding" in label or "qwen3_embedding" in label:
        return "qwen3_embedding"
    if "tool-embed" in label or "toolembed" in label:
        return "tool_embed"
    if "skillrouter-embedding" in label or "skillrouter_embedding" in label:
        return "skillrouter_embedding"
    return "generic_hf_embedding"


def _compact_model_label(model_name_or_path: str | Path) -> str:
    label = Path(str(model_name_or_path)).name.lower()
    replacements = {
        "qwen3-embedding-": "qwen3_embedding_",
        "qwen3-reranker-": "qwen3_reranker_",
        "tool-embed-": "toolembed_",
        ".": "_",
        "-": "_",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return "_".join(part for part in label.split("_") if part)


def build_qwen_route_method_name(
    *,
    embedding_model_name_or_path: str | Path,
    reranker_model_name_or_path: str | Path | None,
) -> str:
    embedding_label = _compact_model_label(embedding_model_name_or_path)
    if is_reranker_disabled(reranker_model_name_or_path):
        return f"{embedding_label}_embedding_only_route_eval"
    reranker_label = _compact_model_label(reranker_model_name_or_path)
    return f"{embedding_label}_plus_{reranker_label}"


def load_embedding_adapter_checkpoint(adapter_checkpoint_path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(adapter_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"embedding adapter checkpoint must be a dict: {checkpoint_path}")
    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"embedding adapter checkpoint missing adapter_state_dict: {checkpoint_path}")
    q_weight = state.get("q_proj.weight")
    if not isinstance(q_weight, torch.Tensor):
        raise ValueError(f"embedding adapter checkpoint missing q_proj.weight: {checkpoint_path}")
    d_weight = state.get("d_proj.weight")
    if d_weight is not None and not isinstance(d_weight, torch.Tensor):
        raise ValueError(f"embedding adapter checkpoint has invalid d_proj.weight: {checkpoint_path}")
    adapter_state = {"q_proj.weight": q_weight.float()}
    if isinstance(d_weight, torch.Tensor):
        adapter_state["d_proj.weight"] = d_weight.float()
    return adapter_state, {
        "checkpoint_path": str(checkpoint_path),
        "method": str(payload.get("method") or ""),
        "step": payload.get("step"),
        "uses_clstr_heads": bool(payload.get("uses_clstr_heads", False)),
        "uses_clstr_transition": bool(payload.get("uses_clstr_transition", False)),
        "config": payload.get("config", {}),
    }


def apply_embedding_adapter(
    *,
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    adapter_checkpoint_path: str | Path | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if adapter_checkpoint_path is None:
        return (
            F.normalize(query_embs.float(), p=2, dim=-1),
            F.normalize(skill_embs.float(), p=2, dim=-1),
            None,
        )
    adapter_state, adapter_report = load_embedding_adapter_checkpoint(adapter_checkpoint_path)
    q_weight = adapter_state["q_proj.weight"]
    d_weight = adapter_state.get("d_proj.weight")
    if query_embs.size(-1) != q_weight.size(-1):
        raise ValueError(
            "query embedding dimension does not match adapter q_proj input: "
            f"{query_embs.size(-1)} != {q_weight.size(-1)}"
        )
    projected_queries = query_embs.float() @ q_weight.t()
    if d_weight is None:
        projected_skills = skill_embs.float()
    else:
        if skill_embs.size(-1) != d_weight.size(-1):
            raise ValueError(
                "skill embedding dimension does not match adapter d_proj input: "
                f"{skill_embs.size(-1)} != {d_weight.size(-1)}"
            )
        projected_skills = skill_embs.float() @ d_weight.t()
    adapter_report["q_proj_shape"] = list(q_weight.shape)
    adapter_report["d_proj_shape"] = None if d_weight is None else list(d_weight.shape)
    return (
        F.normalize(projected_queries.float(), p=2, dim=-1),
        F.normalize(projected_skills.float(), p=2, dim=-1),
        adapter_report,
    )


def summarize_route_metric_rows(metric_rows: list[dict[str, float]], *, source_rows: int) -> dict[str, Any]:
    retained_metrics = aggregate_metric_rows(metric_rows)
    strict = strict_metrics(retained_metrics, retained_rows=len(metric_rows), source_rows=source_rows)
    embedding_recall = (
        sum(float(row.get("positive_in_embedding_topk") or 0.0) for row in metric_rows) / len(metric_rows)
        if metric_rows
        else 0.0
    )
    scale = len(metric_rows) / max(1, int(source_rows)) if int(source_rows) > 0 else 0.0
    return {
        "retained_metrics": retained_metrics,
        "strict": strict,
        "embedding_recall@topk": float(embedding_recall),
        "strict_embedding_recall@topk": float(embedding_recall) * scale,
    }


def _write_route_eval_outputs(
    *,
    output_dir: str | Path,
    report: dict[str, Any],
    predictions: Iterable[dict[str, Any]],
    save_predictions: bool,
) -> dict[str, Path | None]:
    output_dir = Path(output_dir)
    legacy_report_path = output_dir / "qwen3_embedding_reranker_route_eval_report.json"
    generic_report_path = output_dir / "route_eval_report.json"
    legacy_predictions_path = output_dir / "qwen3_embedding_reranker_predictions.jsonl"
    generic_predictions_path = output_dir / "route_eval_predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    blocker_report_path = output_dir / "blocker_report.json"

    _write_json(legacy_report_path, report)
    _write_json(generic_report_path, report)
    metrics_out: Path | None = None
    blocker_out: Path | None = None
    if str(report.get("status") or "") == "ok":
        _write_json(
            metrics_path,
            {
                "status": report.get("status"),
                "benchmark": report.get("benchmark"),
                "method": report.get("method"),
                "source_eval_rows": report.get("source_eval_rows"),
                "retained_eval_rows": report.get("retained_eval_rows"),
                "retained_metrics": report.get("retained_metrics"),
                "strict": report.get("strict"),
                "embedding_recall@topk": report.get("embedding_recall@topk"),
                "strict_embedding_recall@topk": report.get("strict_embedding_recall@topk"),
                "report_path": str(generic_report_path),
                "legacy_report_path": str(legacy_report_path),
            },
        )
        metrics_out = metrics_path
    else:
        _write_json(
            blocker_report_path,
            {
                "status": report.get("status"),
                "benchmark": report.get("benchmark"),
                "method": report.get("method"),
                "blockers": report.get("blockers", []),
                "source_eval_rows": report.get("source_eval_rows"),
                "retained_eval_rows": report.get("retained_eval_rows"),
                "report_path": str(generic_report_path),
                "legacy_report_path": str(legacy_report_path),
            },
        )
        blocker_out = blocker_report_path
    if save_predictions:
        prediction_rows = list(predictions)
        _write_jsonl(legacy_predictions_path, prediction_rows)
        _write_jsonl(generic_predictions_path, prediction_rows)
    return {
        "legacy_report_path": legacy_report_path,
        "generic_report_path": generic_report_path,
        "metrics_path": metrics_out,
        "blocker_report_path": blocker_out,
        "legacy_predictions_path": legacy_predictions_path if save_predictions else None,
        "generic_predictions_path": generic_predictions_path if save_predictions else None,
    }


def _raw_route_state_text(row: dict[str, Any], *, max_chars: int = 2000) -> str:
    state = str(
        row.get("state_text")
        or row.get("query_text")
        or row.get("query")
        or row.get("instruction")
        or row.get("task")
        or ""
    ).strip()
    return state[: max(0, int(max_chars))]


def _qwen_route_query_text(row: dict[str, Any], *, benchmark: str, max_chars: int = 2000) -> str:
    state = _raw_route_state_text(row, max_chars=max_chars)
    return (
        f"Instruct: Given a {benchmark} task state and interaction history, retrieve the next skill/tool "
        "document an agent should use.\nQuery:"
        f"{state}"
    )


def route_query_text(
    row: dict[str, Any],
    *,
    benchmark: str,
    embedding_family: str,
    query_text_mode: str = "auto",
    max_chars: int = 2000,
) -> str:
    mode = str(query_text_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "raw_state" if str(embedding_family).lower() == "bge_m3" else "qwen_instruct"
    if mode in {"raw", "raw_state", "state"}:
        return _raw_route_state_text(row, max_chars=max_chars)
    if mode in {"qwen", "qwen_instruct", "skillrouter_instruct", "instruct"}:
        return _qwen_route_query_text(row, benchmark=benchmark, max_chars=max_chars)
    raise ValueError(f"unsupported query_text_mode: {query_text_mode}")


def _skill_ids(skills: list[dict[str, Any]]) -> list[str]:
    return [str(_skill_id(skill, idx)) for idx, skill in enumerate(skills)]


def _select_candidate_skills_for_rows(
    *,
    source_rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_ids = {
        str(candidate_id)
        for row in source_rows
        for candidate_id in (row.get("candidate_next_skill_ids") or row.get("candidate_skill_ids") or [])
        if str(candidate_id).strip()
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for idx, skill in enumerate(skills):
        skill_id = str(_skill_id(skill, idx))
        if skill_id in candidate_ids:
            selected.append(skill)
            selected_ids.add(skill_id)
    return selected, {
        "candidate_skill_id_count": len(candidate_ids),
        "selected_skill_count": len(selected),
        "missing_candidate_skill_count": len(candidate_ids - selected_ids),
    }


def _candidate_ids_for_row(row: dict[str, Any], *, candidate_source: str, all_skill_ids: list[str]) -> list[str]:
    if candidate_source == "full_pool":
        return list(all_skill_ids)
    values = row.get("candidate_next_skill_ids") or row.get("candidate_skill_ids") or []
    return [str(item) for item in values if str(item)]


def _hash_texts(*, model_name_or_path: str | Path, max_length: int, texts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(str(model_name_or_path).encode("utf-8", errors="ignore"))
    digest.update(str(int(max_length)).encode("utf-8"))
    digest.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        encoded = str(text).encode("utf-8", errors="ignore")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:24]


def _encode_texts_cached(
    *,
    model_name_or_path: str | Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    cache_dir: str | Path | None,
    cache_prefix: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32), {"cache": "empty", "count": 0}
    cache_path: Path | None = None
    cache_key = _hash_texts(model_name_or_path=model_name_or_path, max_length=max_length, texts=texts)
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_prefix}_{cache_key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            embeddings = payload["embeddings"].float()
            return embeddings, {
                "cache": "hit",
                "cache_path": str(cache_path),
                "count": len(texts),
                "embedding_dim": int(embeddings.size(-1)) if embeddings.ndim == 2 else 0,
            }
    embeddings = _encode_skillrouter_texts(
        model_name_or_path=model_name_or_path,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
    ).float()
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    report = {
        "cache": "miss",
        "cache_path": None if cache_path is None else str(cache_path),
        "count": len(texts),
        "embedding_dim": int(embeddings.size(-1)) if embeddings.ndim == 2 else 0,
    }
    if cache_path is not None:
        torch.save(
            {
                "embeddings": embeddings.cpu(),
                "model_name_or_path": str(model_name_or_path),
                "max_length": int(max_length),
                "text_count": len(texts),
                "cache_key": cache_key,
            },
            cache_path,
        )
    return embeddings.cpu(), report


def load_qwen_route_corpus(
    *,
    benchmark: str,
    toolbench_eval_trajectories_path: str | Path = "outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    toolbench_skills_path: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
    tau_data_root: str | Path = ".tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data",
    tau_domains: Iterable[str] | None = ("airline", "retail", "telecom"),
    tau_max_tasks_per_domain: int | None = None,
    toolsandbox_scenarios_root: str | Path = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios",
    toolsandbox_tools_root: str | Path | None = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools",
    toolsandbox_max_scenarios: int | None = None,
    trajectbench_eval_rows_path: str | Path | None = None,
    trajectbench_skills_path: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
) -> QwenRouteCorpus:
    benchmark_name = _clean_benchmark_name(benchmark)
    if benchmark_name in {"toolbench", "toolbench_g3", "toolbench-g3"}:
        skills = _read_jsonl(toolbench_skills_path)
        source_rows = [_as_stage4_handoff_row(row) for row in _read_jsonl(toolbench_eval_trajectories_path)]
        return QwenRouteCorpus(
            benchmark="toolbench_g3",
            skills=skills,
            source_rows=source_rows,
            candidate_source="full_pool",
            report={
                "benchmark": "toolbench_g3",
                "toolbench_eval_trajectories_path": str(toolbench_eval_trajectories_path),
                "toolbench_skills_path": str(toolbench_skills_path),
                "source_row_count": len(source_rows),
                "skill_count": len(skills),
                "candidate_source": "full_pool",
            },
        )
    if benchmark_name in {"tau2", "tau3"}:
        corpus = load_tau2_route_corpus(
            tau_data_root,
            domains=tau_domains,
            max_tasks_per_domain=tau_max_tasks_per_domain,
            benchmark_name=benchmark_name,
        )
        return QwenRouteCorpus(
            benchmark=benchmark_name,
            skills=corpus.skills,
            source_rows=corpus.source_rows,
            candidate_source="row_candidates",
            report={**corpus.report, "candidate_source": "row_candidates"},
        )
    if benchmark_name == "toolsandbox":
        from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus

        corpus = load_toolsandbox_route_corpus(
            scenarios_root=toolsandbox_scenarios_root,
            tools_root=toolsandbox_tools_root,
            max_scenarios=toolsandbox_max_scenarios,
        )
        return QwenRouteCorpus(
            benchmark="toolsandbox",
            skills=corpus.skills,
            source_rows=corpus.source_rows,
            candidate_source="row_candidates",
            report={**corpus.report, "candidate_source": "row_candidates"},
        )
    if benchmark_name in {"trajectbench", "traject_bench"}:
        if trajectbench_eval_rows_path is None:
            raise ValueError("trajectbench qwen route eval requires --trajectbench_eval_rows_path")
        source_rows = _read_jsonl(trajectbench_eval_rows_path)
        all_skills = _read_jsonl(trajectbench_skills_path)
        selected_skills, selection_report = _select_candidate_skills_for_rows(
            source_rows=source_rows,
            skills=all_skills,
        )
        return QwenRouteCorpus(
            benchmark="trajectbench",
            skills=selected_skills,
            source_rows=source_rows,
            candidate_source="row_candidates",
            report={
                "benchmark": "trajectbench",
                "trajectbench_eval_rows_path": str(trajectbench_eval_rows_path),
                "trajectbench_skills_path": str(trajectbench_skills_path),
                "source_row_count": len(source_rows),
                "source_skill_count": len(all_skills),
                **selection_report,
                "candidate_source": "row_candidates",
            },
        )
    raise ValueError(f"unsupported qwen route benchmark: {benchmark}")


def _embedding_topk_for_rows(
    *,
    source_rows: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    skill_ids: list[str],
    skill_id_to_idx: dict[str, int],
    candidate_source: str,
    top_k: int,
    score_batch_size: int,
) -> tuple[list[list[str]], dict[str, Any]]:
    selected_by_row: list[list[str]] = [[] for _ in source_rows]
    skipped: Counter[str] = Counter()
    candidate_counts: list[int] = []
    top_k = max(1, int(top_k))
    if candidate_source == "full_pool":
        k = min(top_k, len(skill_ids))
        skill_embs_t = skill_embs.float().t().contiguous()
        for start in range(0, len(source_rows), max(1, int(score_batch_size))):
            end = min(start + max(1, int(score_batch_size)), len(source_rows))
            scores = query_embs[start:end].float() @ skill_embs_t
            _, indices = torch.topk(scores, k=k, dim=1)
            for row_offset, row_indices in enumerate(indices.tolist()):
                selected = [skill_ids[int(idx)] for idx in row_indices]
                selected_by_row[start + row_offset] = selected
                candidate_counts.append(len(selected))
        return selected_by_row, {
            "candidate_source": candidate_source,
            "rows": len(source_rows),
            "top_k": k,
            "candidate_count_mean": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
            "skipped_reasons": dict(skipped),
        }

    for row_idx, row in enumerate(source_rows):
        candidates = [
            skill_id
            for skill_id in _candidate_ids_for_row(row, candidate_source=candidate_source, all_skill_ids=skill_ids)
            if skill_id in skill_id_to_idx
        ]
        if not candidates:
            skipped["no_valid_candidates"] += 1
            continue
        candidate_indices = [skill_id_to_idx[skill_id] for skill_id in candidates]
        scores = (query_embs[row_idx].float().unsqueeze(0) @ skill_embs[candidate_indices].float().t()).squeeze(0)
        selected = select_embedding_topk(
            candidate_skill_ids=candidates,
            candidate_scores=[float(item) for item in scores.detach().cpu().tolist()],
            top_k=min(top_k, len(candidates)),
        )
        selected_by_row[row_idx] = selected
        candidate_counts.append(len(selected))
    return selected_by_row, {
        "candidate_source": candidate_source,
        "rows": len(source_rows),
        "top_k": top_k,
        "candidate_count_mean": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "skipped_reasons": dict(skipped),
    }


def run_qwen_embedding_reranker_route_eval(
    *,
    benchmark: str,
    output_dir: str | Path,
    embedding_model_name_or_path: str | Path = "models/Qwen3-Embedding-0.6B",
    reranker_model_name_or_path: str | Path | None = "models/Qwen3-Reranker-0.6B",
    top_k: int = 100,
    max_eval_rows: int | None = None,
    embedding_batch_size: int = 16,
    embedding_max_length: int = 2048,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 2048,
    max_skill_chars: int = 900,
    torch_dtype: str = "bfloat16",
    local_files_only: bool = True,
    score_mode: str = "logit_diff",
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
    query_text_mode: str = "auto",
    score_batch_size: int = 64,
    row_progress_interval: int = 25,
    save_predictions: bool = True,
    embedding_cache_dir: str | Path | None = None,
    adapter_checkpoint_path: str | Path | None = None,
    toolbench_eval_trajectories_path: str | Path = "outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    toolbench_skills_path: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
    tau_data_root: str | Path = ".tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data",
    tau_domains: Iterable[str] | None = ("airline", "retail", "telecom"),
    tau_max_tasks_per_domain: int | None = None,
    toolsandbox_scenarios_root: str | Path = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios",
    toolsandbox_tools_root: str | Path | None = "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools",
    toolsandbox_max_scenarios: int | None = None,
    trajectbench_eval_rows_path: str | Path | None = None,
    trajectbench_skills_path: str | Path = "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    embedding_family = embedding_model_family(embedding_model_name_or_path)

    def write_progress(phase: str, **extra: Any) -> None:
        _write_json(
            progress_path,
            {
                "status": "running",
                "phase": phase,
                "benchmark": benchmark,
                "output_dir": str(output_dir),
                "embedding_model_name_or_path": str(embedding_model_name_or_path),
                "embedding_model_family": embedding_family,
                "reranker_model_name_or_path": None
                if is_reranker_disabled(reranker_model_name_or_path)
                else str(reranker_model_name_or_path),
                "elapsed_seconds": time.perf_counter() - started,
                **extra,
            },
        )

    write_progress("loading_corpus")
    corpus = load_qwen_route_corpus(
        benchmark=benchmark,
        toolbench_eval_trajectories_path=toolbench_eval_trajectories_path,
        toolbench_skills_path=toolbench_skills_path,
        tau_data_root=tau_data_root,
        tau_domains=tau_domains,
        tau_max_tasks_per_domain=tau_max_tasks_per_domain,
        toolsandbox_scenarios_root=toolsandbox_scenarios_root,
        toolsandbox_tools_root=toolsandbox_tools_root,
        toolsandbox_max_scenarios=toolsandbox_max_scenarios,
        trajectbench_eval_rows_path=trajectbench_eval_rows_path,
        trajectbench_skills_path=trajectbench_skills_path,
    )
    source_rows = corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    skill_ids = _skill_ids(corpus.skills)
    skill_id_to_idx = {skill_id: idx for idx, skill_id in enumerate(skill_ids)}
    skills_by_id = {skill_id: corpus.skills[idx] for idx, skill_id in enumerate(skill_ids)}
    _write_jsonl(output_dir / f"{corpus.benchmark}_source_rows.jsonl", source_rows)
    _write_jsonl(output_dir / f"{corpus.benchmark}_skill_pool.jsonl", corpus.skills)

    skill_texts = [_skillrouter_skill_text(skill) for skill in corpus.skills]
    query_texts = [
        route_query_text(
            row,
            benchmark=corpus.benchmark,
            embedding_family=embedding_family,
            query_text_mode=query_text_mode,
        )
        for row in source_rows
    ]
    write_progress(
        "encoding_queries",
        corpus_benchmark=corpus.benchmark,
        source_eval_rows=len(source_rows),
        skill_count=len(corpus.skills),
        candidate_source=corpus.candidate_source,
    )
    query_embs, query_encode_report = _encode_texts_cached(
        model_name_or_path=embedding_model_name_or_path,
        texts=query_texts,
        batch_size=embedding_batch_size,
        max_length=embedding_max_length,
        cache_dir=None,
        cache_prefix=f"{corpus.benchmark}_queries",
    )
    write_progress("encoding_skills", query_encode_report=query_encode_report)
    skill_embs, skill_encode_report = _encode_texts_cached(
        model_name_or_path=embedding_model_name_or_path,
        texts=skill_texts,
        batch_size=embedding_batch_size,
        max_length=embedding_max_length,
        cache_dir=embedding_cache_dir,
        cache_prefix=f"{corpus.benchmark}_skills",
    )
    query_embs, skill_embs, adapter_report = apply_embedding_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    write_progress("embedding_topk", skill_encode_report=skill_encode_report, adapter_report=adapter_report)
    selected_by_row, embedding_report = _embedding_topk_for_rows(
        source_rows=source_rows,
        query_embs=query_embs,
        skill_embs=skill_embs,
        skill_ids=skill_ids,
        skill_id_to_idx=skill_id_to_idx,
        candidate_source=corpus.candidate_source,
        top_k=top_k,
        score_batch_size=score_batch_size,
    )
    reranker_enabled = not is_reranker_disabled(reranker_model_name_or_path)
    reranker_kind = None if not reranker_enabled else reranker_family(str(reranker_model_name_or_path))
    scorer: Any | None = None
    if reranker_enabled:
        write_progress("loading_reranker", reranker_family=reranker_kind, embedding_report=embedding_report)
        if reranker_kind == "bge_sequence_classifier":
            scorer = BGERerankerScorer(
                BGERerankerConfig(
                    model_name_or_path=str(reranker_model_name_or_path),
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    batch_size=reranker_batch_size,
                    max_length=reranker_max_length,
                )
            )
        else:
            scorer = Qwen3RerankerScorer(
                Qwen3RerankerConfig(
                    model_name_or_path=str(reranker_model_name_or_path),
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    batch_size=reranker_batch_size,
                    max_length=reranker_max_length,
                    max_skill_chars=max_skill_chars,
                    instruction=instruction,
                    score_mode=score_mode,
                )
            )
    else:
        write_progress("scoring_rows", reranker_family=reranker_kind, embedding_report=embedding_report)
    doc_by_skill_id = {
        skill_id: build_qwen3_reranker_document(
            {**skills_by_id[skill_id], "skill_id": skill_id},
            max_skill_chars=max_skill_chars,
        )
        for skill_id in skill_ids
    }

    metric_rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []
    existing_predictions = _load_existing_route_predictions(output_dir) if save_predictions else {}
    resumed_prediction_count = 0
    new_prediction_count = 0
    skipped: Counter[str] = Counter()
    scoring_started = time.perf_counter()
    for row_idx, (row, selected_skill_ids) in enumerate(zip(source_rows, selected_by_row), start=1):
        row_key = _route_eval_row_key(row, row_index=row_idx)
        existing_prediction = existing_predictions.get(row_key)
        if existing_prediction is not None:
            metrics = existing_prediction.get("metrics") or {}
            metric_rows.append({str(key): float(value) for key, value in metrics.items()})
            predictions.append(existing_prediction)
            resumed_prediction_count += 1
            if row_progress_interval > 0 and (
                row_idx == 1 or row_idx % int(row_progress_interval) == 0 or row_idx == len(source_rows)
            ):
                write_progress(
                    "scoring_rows",
                    corpus_benchmark=corpus.benchmark,
                    scored_rows=row_idx,
                    total_rows=len(source_rows),
                    retained_rows=len(metric_rows),
                    resumed_rows=resumed_prediction_count,
                    newly_scored_rows=new_prediction_count,
                    skipped_reasons=dict(sorted(skipped.items())),
                    reranker_family=reranker_kind,
                    embedding_report=embedding_report,
                )
            continue
        positive_id = str(row.get("next_skill_id") or row.get("positive_skill_id") or "")
        if not positive_id:
            skipped["missing_positive_skill_id"] += 1
            continue
        if positive_id not in skill_id_to_idx:
            skipped["positive_not_in_skill_pool"] += 1
            continue
        selected_skill_ids = [skill_id for skill_id in selected_skill_ids if skill_id in doc_by_skill_id]
        if not selected_skill_ids:
            skipped["empty_embedding_topk"] += 1
            continue
        if scorer is None:
            scores = []
            ranked_skill_ids = list(selected_skill_ids)
        else:
            documents = [doc_by_skill_id[skill_id] for skill_id in selected_skill_ids]
            scores = scorer.score_pairs(queries=[query_texts[row_idx - 1]] * len(documents), documents=documents)
            ranked_skill_ids = rank_candidate_skill_ids_by_scores(
                candidate_skill_ids=selected_skill_ids,
                scores=scores,
            )
        metrics = metric_row_from_reranked_candidates(
            positive_skill_id=positive_id,
            embedding_candidate_skill_ids=selected_skill_ids,
            reranked_skill_ids=ranked_skill_ids,
        )
        metric_rows.append(metrics)
        predictions.append(
            {
                "row_key": row_key,
                "source_row_index": row_idx,
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index"),
                "domain": row.get("domain"),
                "next_skill_id": positive_id,
                "embedding_candidate_skill_ids": selected_skill_ids,
                "reranker_scores": scores,
                "ranked_skill_ids": ranked_skill_ids,
                "metrics": metrics,
            }
        )
        _append_route_eval_prediction(output_dir, predictions[-1], save_predictions=save_predictions)
        new_prediction_count += 1
        if row_progress_interval > 0 and (
            row_idx == 1 or row_idx % int(row_progress_interval) == 0 or row_idx == len(source_rows)
        ):
            elapsed = time.perf_counter() - scoring_started
            write_progress(
                "scoring_rows",
                corpus_benchmark=corpus.benchmark,
                scored_rows=row_idx,
                total_rows=len(source_rows),
                retained_rows=len(metric_rows),
                resumed_rows=resumed_prediction_count,
                newly_scored_rows=new_prediction_count,
                skipped_reasons=dict(sorted(skipped.items())),
                reranker_family=reranker_kind,
                embedding_report=embedding_report,
            )
            print(
                f"[qwen-route] benchmark={corpus.benchmark} scored_rows={row_idx}/{len(source_rows)} "
                f"retained={len(metric_rows)} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    summary = summarize_route_metric_rows(metric_rows, source_rows=len(source_rows))
    report = {
        "status": "ok" if source_rows and metric_rows else "action_required",
        "blockers": [] if source_rows and metric_rows else ["no_source_or_retained_rows"],
        "benchmark": corpus.benchmark,
        "method": build_qwen_route_method_name(
            embedding_model_name_or_path=embedding_model_name_or_path,
            reranker_model_name_or_path=reranker_model_name_or_path,
        ),
        "embedding_model_name_or_path": str(embedding_model_name_or_path),
        "embedding_model_family": embedding_family,
        "reranker_model_name_or_path": None if not reranker_enabled else str(reranker_model_name_or_path),
        "reranker_family": reranker_kind,
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "output_dir": str(output_dir),
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(metric_rows),
        "top_k": int(top_k),
        "candidate_source": corpus.candidate_source,
        "corpus_report": corpus.report,
        "query_encode_report": query_encode_report,
        "skill_encode_report": skill_encode_report,
        "embedding_report": embedding_report,
        "skipped_reasons": dict(sorted(skipped.items())),
        "resumed_prediction_rows": int(resumed_prediction_count),
        "newly_scored_rows": int(new_prediction_count),
        "retained_metrics": summary["retained_metrics"],
        "strict": summary["strict"],
        "embedding_recall@topk": summary["embedding_recall@topk"],
        "strict_embedding_recall@topk": summary["strict_embedding_recall@topk"],
        "timing": {
            "total_seconds": time.perf_counter() - started,
            "scoring_seconds": time.perf_counter() - scoring_started,
        },
        "config": {
            "benchmark": benchmark,
            "top_k": int(top_k),
            "max_eval_rows": max_eval_rows,
            "embedding_batch_size": int(embedding_batch_size),
            "embedding_max_length": int(embedding_max_length),
            "reranker_batch_size": int(reranker_batch_size),
            "reranker_max_length": int(reranker_max_length),
            "reranker_enabled": bool(reranker_enabled),
            "max_skill_chars": int(max_skill_chars),
            "torch_dtype": str(torch_dtype),
            "local_files_only": bool(local_files_only),
            "score_mode": str(score_mode),
            "query_text_mode": str(query_text_mode),
            "score_batch_size": int(score_batch_size),
            "embedding_cache_dir": None if embedding_cache_dir is None else str(embedding_cache_dir),
            "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
            "tau_domains": None if tau_domains is None else list(tau_domains),
            "tau_max_tasks_per_domain": tau_max_tasks_per_domain,
            "toolsandbox_max_scenarios": toolsandbox_max_scenarios,
            "trajectbench_eval_rows_path": None if trajectbench_eval_rows_path is None else str(trajectbench_eval_rows_path),
            "trajectbench_skills_path": str(trajectbench_skills_path),
        },
        "paper_scope_note": (
            "The embedding model retrieves candidate skills with the same text-only route rows, then "
            "the configured reranker reranks the retrieved top-k when enabled. Strict metrics keep "
            "source rows in the denominator."
        ),
    }
    output_paths = _write_route_eval_outputs(
        output_dir=output_dir,
        report=report,
        predictions=predictions,
        save_predictions=save_predictions,
    )
    _write_json(
        progress_path,
        {
            "status": report["status"],
            "phase": "done",
            "benchmark": corpus.benchmark,
            "output_dir": str(output_dir),
            "embedding_model_name_or_path": str(embedding_model_name_or_path),
            "embedding_model_family": embedding_family,
            "reranker_model_name_or_path": None if not reranker_enabled else str(reranker_model_name_or_path),
            "reranker_family": reranker_kind,
            "source_eval_rows": len(source_rows),
            "retained_eval_rows": len(metric_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "report_path": str(output_paths["generic_report_path"]),
            "legacy_report_path": str(output_paths["legacy_report_path"]),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report
