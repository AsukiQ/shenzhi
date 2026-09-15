from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.model import CLSTRConfig, CLSTRModel
from clstr.skillret_official import _filter_state_dict_for_model, resolve_clstr_export_config


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _write_trec_run(path: str | Path, ranked_by_query: dict[str, list[tuple[str, float]]], run_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, ranked in ranked_by_query.items():
            for rank, (skill_id, score) in enumerate(ranked, start=1):
                handle.write(f"{query_id}\tQ0\t{skill_id}\t{rank}\t{float(score):.8f}\t{run_name}\n")


def _write_progress(path: str | Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _query_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("query_id") or row.get("qid") or row.get("task_id") or row.get("id") or f"query-{index}")


def _query_text(row: dict[str, Any]) -> str:
    text = row.get("query_text")
    if text is not None:
        return str(text)
    query = str(row.get("query") or row.get("text") or row.get("instruction_text") or "")
    instruction = str(row.get("instruction") or "")
    if instruction and query:
        return f"{instruction}\nTask: {query}"
    return instruction or query


def _format_query_text(row: dict[str, Any], query_text_format: str) -> str:
    text = _query_text(row)
    if query_text_format == "raw":
        return text
    if query_text_format == "skillrouter":
        return (
            "Instruct: Given a task description, retrieve the most relevant "
            "skill document that would help an agent complete the task\nQuery:"
            f"{text[:1500]}"
        )
    raise ValueError(f"unsupported query_text_format: {query_text_format}")


def _build_model(
    *,
    skills_path: str | Path,
    base_model_name: str,
    checkpoint_path: str | Path | None,
    top_k: int,
    model_dim: int,
    batch_size: int,
    encoder_pooling: str,
    cross_encoder_pooling: str,
    tokenizer_padding_side: str | None,
    torch_dtype: str | None,
    freeze_backbone: bool,
    trust_remote_code: bool,
    max_length: int | None,
    projection_init: str,
    normalize_embeddings: bool,
    skill_text_format: str,
    skill_table_batch_size: int,
    skill_table_adapter_init: str,
    use_cross_encoder: bool,
    hf_cache_dir: str | None,
    local_files_only: bool,
) -> tuple[CLSTRModel, dict[str, Any]]:
    del batch_size
    resolved_config = resolve_clstr_export_config(
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        model_dim=model_dim,
        top_k=top_k,
        encoder_pooling=encoder_pooling,
        cross_encoder_pooling=cross_encoder_pooling,
        tokenizer_padding_side=tokenizer_padding_side,
        torch_dtype=torch_dtype,
        freeze_backbone=freeze_backbone,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        projection_init=projection_init,
        normalize_embeddings=normalize_embeddings,
        skill_text_format=skill_text_format,
        skill_table_batch_size=skill_table_batch_size,
        skill_table_adapter_init=skill_table_adapter_init,
        use_cross_encoder=use_cross_encoder,
        hf_cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    top_k = int(resolved_config["top_k"])
    skills = load_eval_pool(Path(skills_path))
    model = CLSTRModel(
        CLSTRConfig(
            base_model_name=str(resolved_config["base_model_name"]),
            d=int(resolved_config["d"]),
            d_a=int(resolved_config["d_a"]),
            top_k=top_k,
            encoder_pooling=str(resolved_config["encoder_pooling"]),
            cross_encoder_pooling=str(resolved_config["cross_encoder_pooling"]),
            tokenizer_padding_side=resolved_config["tokenizer_padding_side"],
            torch_dtype=resolved_config["torch_dtype"],
            freeze_backbone=bool(resolved_config["freeze_backbone"]),
            trust_remote_code=bool(resolved_config["trust_remote_code"]),
            max_length=resolved_config["max_length"],
            projection_init=str(resolved_config["projection_init"]),
            normalize_embeddings=bool(resolved_config["normalize_embeddings"]),
            hf_cache_dir=resolved_config.get("hf_cache_dir"),
            local_files_only=bool(resolved_config.get("local_files_only", False)),
            defer_skill_table_init=bool(resolved_config["defer_skill_table_init"]),
            skill_text_format=str(resolved_config["skill_text_format"]),
            skill_table_batch_size=int(resolved_config["skill_table_batch_size"]),
            skill_table_adapter_init=str(resolved_config["skill_table_adapter_init"]),
            use_cross_encoder=bool(resolved_config["use_cross_encoder"]),
        ),
        skills,
    )
    load_report: dict[str, Any] = {
        "base_model_name": str(resolved_config["base_model_name"]),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "resolved_config": resolved_config,
        "loaded_state_keys": 0,
    }
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        filtered = _filter_state_dict_for_model(model, state_dict)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        load_report.update(
            {
                "loaded_state_keys": len(filtered),
                "loaded_skill_table_embeddings": "skill_table.E" in filtered,
                "missing_keys": sorted(missing),
                "unexpected_keys": sorted(unexpected),
            }
        )
        if isinstance(payload, dict):
            for key in ("stage", "step", "metrics", "train_config", "train_safety"):
                if key in payload:
                    load_report[key] = payload[key]
    return model, load_report


def export_clstr_retrieval_run(
    *,
    queries_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    base_model_name: str,
    checkpoint_path: str | Path | None = None,
    top_k: int = 50,
    batch_size: int = 16,
    model_dim: int = 1024,
    encoder_pooling: str = "masked_mean",
    cross_encoder_pooling: str = "masked_mean",
    tokenizer_padding_side: str | None = None,
    torch_dtype: str | None = None,
    freeze_backbone: bool = False,
    trust_remote_code: bool = True,
    max_length: int | None = None,
    projection_init: str = "default",
    normalize_embeddings: bool = False,
    skill_text_format: str = "clstr",
    skill_table_batch_size: int = 32,
    skill_table_adapter_init: str = "default",
    use_cross_encoder: bool = False,
    hf_cache_dir: str | None = None,
    local_files_only: bool = False,
    query_text_format: str = "raw",
    run_name: str = "clstr_retrieval",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    run_path = output / "run.tsv"
    progress_path = output / "progress.json"
    queries = _read_jsonl(queries_path)
    skill_rows = _read_jsonl(skills_path)
    skill_ids = [str(row.get("skill_id") or row.get("id") or idx) for idx, row in enumerate(skill_rows)]
    output_k = min(max(1, int(top_k)), len(skill_ids))
    progress_base = {
        "status": "running",
        "phase": "loading_model",
        "processed_queries": 0,
        "total_queries": len(queries),
        "skill_count": len(skill_ids),
        "top_k": output_k,
        "batch_size": int(batch_size),
    }
    _write_progress(progress_path, progress_base)
    model, load_report = _build_model(
        skills_path=skills_path,
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        top_k=top_k,
        model_dim=model_dim,
        batch_size=batch_size,
        encoder_pooling=encoder_pooling,
        cross_encoder_pooling=cross_encoder_pooling,
        tokenizer_padding_side=tokenizer_padding_side,
        torch_dtype=torch_dtype,
        freeze_backbone=freeze_backbone,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        projection_init=projection_init,
        normalize_embeddings=normalize_embeddings,
        skill_text_format=skill_text_format,
        skill_table_batch_size=skill_table_batch_size,
        skill_table_adapter_init=skill_table_adapter_init,
        use_cross_encoder=use_cross_encoder,
        hf_cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if bool(load_report.get("loaded_skill_table_embeddings", False)):
        load_report["skill_table_rebuild"] = "skipped_checkpoint_embeddings"
        _write_progress(progress_path, {**progress_base, "phase": "using_checkpoint_skill_table"})
    else:
        load_report["skill_table_rebuild"] = "rebuilt_from_encoder"
        _write_progress(progress_path, {**progress_base, "phase": "rebuild_skill_table"})
        model.rebuild_skill_table()

    processed = 0
    batches = 0
    with predictions_path.open("w", encoding="utf-8") as pred_handle, run_path.open("w", encoding="utf-8") as run_handle:
        _write_progress(progress_path, {**progress_base, "phase": "retrieving"})
        with torch.no_grad():
            for start in range(0, len(queries), batch_size):
                batch = queries[start : start + batch_size]
                h = model.encode_states([_format_query_text(row, query_text_format) for row in batch])
                logits = model.skill_table.retrieval_logits(h)
                values, indices = torch.topk(logits, k=output_k, dim=1)
                for offset, (query, row_values, row_indices) in enumerate(zip(batch, values.cpu(), indices.cpu())):
                    qid = _query_id(query, start + offset)
                    ranked = [
                        (skill_ids[int(idx)], float(score))
                        for idx, score in zip(row_indices.tolist(), row_values.tolist())
                    ]
                    pred_handle.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "ranked_skill_ids": [skill_id for skill_id, _score in ranked],
                                "scores": [score for _skill_id, score in ranked],
                                "split": query.get("split"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    for rank, (skill_id, score) in enumerate(ranked, start=1):
                        run_handle.write(f"{qid}\tQ0\t{skill_id}\t{rank}\t{float(score):.8f}\t{run_name}\n")
                    processed += 1
                batches += 1
                pred_handle.flush()
                run_handle.flush()
                _write_progress(
                    progress_path,
                    {
                        **progress_base,
                        "phase": "retrieving",
                        "processed_queries": processed,
                        "processed_batches": batches,
                    },
                )
    report = {
        "status": "ok",
        "method": run_name,
        "queries_path": str(queries_path),
        "skills_path": str(skills_path),
        "predictions_path": str(predictions_path),
        "run_path": str(run_path),
        "progress_path": str(progress_path),
        "run_format": "trec",
        "query_count": len(queries),
        "skill_count": len(skill_ids),
        "top_k": output_k,
        "batch_size": int(batch_size),
        "query_text_format": query_text_format,
        "model": load_report,
        "caveat": "Generic CLSTR static retrieval export; benchmark success requires evaluating the produced run against benchmark qrels.",
    }
    _write_json(output / "run_report.json", report)
    _write_progress(
        progress_path,
        {
            **progress_base,
            "status": "ok",
            "phase": "done",
            "processed_queries": processed,
            "processed_batches": batches,
        },
    )
    return report
