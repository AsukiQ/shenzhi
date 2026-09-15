from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.bridges.skillrouter.datasets import load_eval_pool
from clstr.model import CLSTRConfig, CLSTRModel


OFFICIAL_METRIC_NAMES = [
    "NDCG@5",
    "NDCG@10",
    "NDCG@15",
    "Recall@5",
    "Recall@10",
    "Recall@15",
    "Completeness@5",
    "Completeness@10",
    "Completeness@15",
    "MAP@5",
    "MAP@10",
    "MAP@15",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_official_protocol_report(official_repo: str | Path, output_path: str | Path) -> dict[str, Any]:
    official_repo = Path(official_repo)
    required_files = [
        "README.md",
        "skillret/eval.py",
        "scripts/run_eval_embedding.sh",
        "scripts/run_eval_rerank.sh",
    ]
    missing = [name for name in required_files if not (official_repo / name).exists()]
    report = {
        "status": "missing" if missing else "ok",
        "official_repo": str(official_repo),
        "missing_files": missing,
        "data_path_convention": "HuggingFace dataset ThakiCloud/SKILLRET subsets skills/queries/qrels with official train/test splits",
        "normalized_data_supported": "CLSTR data/skillret mirrors official skills.jsonl, queries.jsonl, qrels.jsonl with split fields",
        "qrels_format": "query_id -> skill_id -> integer relevance",
        "run_format": "TREC run TSV: query_id Q0 skill_id rank score run_name",
        "metric_backend": "pytrec_eval RelevanceEvaluator over map_cut, ndcg_cut, recall, P",
        "metric_names": OFFICIAL_METRIC_NAMES,
        "eval_split": "test",
        "dependencies": ["pytrec_eval", "datasets", "sentence_transformers", "transformers", "faiss"],
        "official_baseline_models": [
            "ThakiCloud/SKILLRET-Embedding-0.6B",
            "ThakiCloud/SKILLRET-Embedding-8B",
            "ThakiCloud/SKILLRET-Reranker-0.6B",
        ],
        "supports_custom_run_file": True,
        "notes": [
            "Final SKILLRET table must use the official test split.",
            "Custom train/dev splits are only allowed for tuning or smoke checks.",
            "SKILLRET is static retrieval/rerank evidence, not closed-loop harness success.",
        ],
    }
    write_json(output_path, report)
    return report


def _rows_for_split(path: Path, split: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if str(row.get("split", "")) == split]


def load_official_qrels(data_root: str | Path, split: str = "test") -> dict[str, dict[str, int]]:
    data_root = Path(data_root)
    qrels: dict[str, dict[str, int]] = {}
    for row in _rows_for_split(data_root / "qrels.jsonl", split):
        qrels.setdefault(str(row["query_id"]), {})[str(row["skill_id"])] = int(row.get("relevance", 1))
    return qrels


def load_official_queries(data_root: str | Path, split: str = "test") -> list[dict[str, Any]]:
    return _rows_for_split(Path(data_root) / "queries.jsonl", split)


def load_official_skills(data_root: str | Path, split: str = "test") -> list[dict[str, Any]]:
    return _rows_for_split(Path(data_root) / "skills.jsonl", split)


def _select_eval_subset(
    skills: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    max_skills: int | None,
    max_queries: int | None,
    include_eval_positives: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if max_queries is not None:
        queries = queries[:max_queries]
    if max_skills is None or max_skills >= len(skills):
        return skills, queries, "all"
    selected = list(skills[:max_skills])
    if not include_eval_positives:
        return selected, queries, "first_n_no_qrel_peek"
    selected_ids = {str(row["skill_id"]) for row in selected}
    by_id = {str(row["skill_id"]): row for row in skills}
    required_ids: list[str] = []
    for query in queries:
        required_ids.extend(qrels.get(str(query["query_id"]), {}).keys())
    for skill_id in required_ids:
        if skill_id not in selected_ids and skill_id in by_id:
            if len(selected) >= max_skills:
                removed = selected.pop()
                selected_ids.discard(str(removed["skill_id"]))
            selected.append(by_id[skill_id])
            selected_ids.add(skill_id)
    return selected, queries, "first_n_plus_eval_positives"


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(skill.get("name", "")),
            str(skill.get("description", "")),
            str(skill.get("executor_desc", "")),
            str(skill.get("body", skill.get("skill_md", ""))),
        ]
    )


def lexical_rank(query: dict[str, Any], skills: list[dict[str, Any]]) -> list[tuple[str, float]]:
    query_tokens = _tokenize(str(query.get("query", "")))
    precomputed = [
        (str(skill["skill_id"]), _tokenize(_skill_text(skill)), idx)
        for idx, skill in enumerate(skills)
    ]
    return _lexical_rank_with_precomputed(query_tokens, precomputed)


def _lexical_rank_with_precomputed(
    query_tokens: set[str],
    precomputed_skills: list[tuple[str, set[str], int]],
) -> list[tuple[str, float]]:
    scored = []
    for skill_id, skill_tokens, idx in precomputed_skills:
        overlap = len(query_tokens & skill_tokens)
        denom = math.sqrt(max(len(query_tokens), 1) * max(len(skill_tokens), 1))
        score = overlap / denom if denom else 0.0
        score += 1.0e-9 * (len(precomputed_skills) - idx)
        scored.append((skill_id, float(score)))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def write_trec_run(path: str | Path, ranked_by_query: dict[str, list[tuple[str, float]]], run_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, ranked in ranked_by_query.items():
            for rank, (skill_id, score) in enumerate(ranked, start=1):
                handle.write(f"{query_id}\tQ0\t{skill_id}\t{rank}\t{float(score):.8f}\t{run_name}\n")


def load_trec_run(path: str | Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"invalid TREC run line: {line}")
            query_id, _, skill_id, _, score, _ = parts[:6]
            results.setdefault(query_id, {})[skill_id] = float(score)
    return results


def export_lexical_official_run(
    data_root: str | Path,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    max_queries: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=None,
        max_queries=max_queries,
    )
    precomputed_skills = [
        (str(skill["skill_id"]), _tokenize(_skill_text(skill)), idx)
        for idx, skill in enumerate(skills)
    ]
    ranked_by_query = {}
    for query in queries:
        ranked_by_query[str(query["query_id"])] = _lexical_rank_with_precomputed(
            _tokenize(str(query.get("query", ""))),
            precomputed_skills,
        )[:top_k]
    run_path = output_dir / "run.tsv"
    predictions_path = output_dir / "predictions.jsonl"
    write_trec_run(run_path, ranked_by_query, run_name="lexical_official")
    write_jsonl(
        predictions_path,
        (
            {
                "query_id": query_id,
                "ranked_skill_ids": [skill_id for skill_id, _ in ranked],
                "scores": [score for _, score in ranked],
                "split": split,
            }
            for query_id, ranked in ranked_by_query.items()
        ),
    )
    report = {
        "status": "ok",
        "method": "lexical_official",
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "Sanity baseline using official SKILLRET test split and TREC run format.",
    }
    write_json(output_dir / "run_report.json", report)
    return report


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _masked_mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    return (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _default_hf_pooling(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path)
    pooling_config_path = model_path / "1_Pooling" / "config.json"
    if pooling_config_path.exists():
        try:
            pooling_config = json.loads(pooling_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pooling_config = {}
        if bool(pooling_config.get("pooling_mode_cls_token")):
            return "cls"
        if bool(pooling_config.get("pooling_mode_mean_tokens")):
            return "masked_mean"
    label = str(model_name_or_path).lower()
    if "bge-m3" in label or "bge_m3" in label:
        return "cls"
    return "last_token"


def _pool_hf_hidden(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "last_token":
        return _last_token_pool(last_hidden_states, attention_mask)
    if pooling == "masked_mean":
        return _masked_mean_pool(last_hidden_states, attention_mask)
    if pooling == "cls":
        return last_hidden_states[:, 0]
    raise ValueError(f"unsupported hf pooling mode: {pooling}")


def _encode_hf_texts(
    model_name_or_path: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    pooling: str | None = None,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    resolved_pooling = pooling or _default_hf_pooling(model_name_or_path)
    padding_side = "right" if resolved_pooling in {"cls", "masked_mean"} else "left"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side=padding_side)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=dtype)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    encoded_batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        tok = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {key: value.to(device) for key, value in tok.items()}
        with torch.no_grad():
            out = model(**tok)
            embs = _pool_hf_hidden(out.last_hidden_state, tok["attention_mask"], resolved_pooling)
            embs = F.normalize(embs, p=2, dim=1)
        encoded_batches.append(embs.cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(encoded_batches, dim=0)


def _skillrouter_query_text(query: dict[str, Any]) -> str:
    return (
        "Instruct: Given a task description, retrieve the most relevant "
        "skill document that would help an agent complete the task\nQuery:"
        f"{str(query.get('query', ''))[:1500]}"
    )


def _skillrouter_skill_text(skill: dict[str, Any]) -> str:
    name = str(skill.get("name", "")).strip()
    desc = str(skill.get("description", "")).strip()
    body = str(skill.get("skill_md", skill.get("body", ""))).strip()
    return f"{name} | {desc} | {body}"


def write_skillret_serialization_report(
    output_path: str | Path,
    official_repo: str | Path = "/root/autodl-tmp/skillret_repo",
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    report = {
        "status": "ok",
        "official_repo": str(official_repo),
        "official_reference_files": [
            str(official_repo / "skillret" / "config.py"),
            str(official_repo / "skillret" / "eval.py"),
        ],
        "query_format": {
            "skillrouter": {
                "template": (
                    "Instruct: Given a task description, retrieve the most relevant "
                    "skill document that would help an agent complete the task\nQuery:{query}"
                ),
                "official_behavior_key": "pipizhao/SkillRouter-Embedding-0.6B",
                "pooling": "last_token",
                "padding_side": "left",
                "normalization": "l2",
            },
            "skillret_embedding": {
                "template": "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: {query}",
                "official_behavior_key": "ThakiCloud/SKILLRET-Embedding-0.6B",
            },
        },
        "doc_format": {
            "source": "official_skillret_embedding_text_for_skill",
            "template": "{name} | {description} | {skill_md}",
            "truncation": "tokenizer truncation only; no manual skill text truncation before tokenization",
        },
        "clstr_usage": {
            "state_encoder": "same SkillRouter query template for SKILLRET query text",
            "skill_table": "rebuild E from current SKILLRET skill pool using official document text",
            "benchmark_role": "static retrieval/rerank evidence only",
        },
    }
    write_json(output_path, report)
    return report


def _rank_embedding_rows(
    query_ids: list[str],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
    skill_ids: list[str],
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    ranked: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, query_embs.size(0), 64):
        sims = query_embs[start : start + 64] @ skill_embs.T
        values, indices = torch.topk(sims, k=min(top_k, len(skill_ids)), dim=1)
        for offset, (row_values, row_indices) in enumerate(zip(values, indices)):
            qid = query_ids[start + offset]
            ranked[qid] = [
                (skill_ids[idx], float(score))
                for idx, score in zip(row_indices.tolist(), row_values.tolist())
            ]
    return ranked


def export_hf_encoder_official_run(
    data_root: str | Path,
    model_name_or_path: str,
    output_dir: str | Path,
    split: str = "test",
    top_k: int = 50,
    batch_size: int = 16,
    max_length: int = 1024,
    max_queries: int | None = None,
    max_skills: int | None = None,
    run_name: str = "hf_encoder_official",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    qrels = load_official_qrels(data_root, split)
    skills, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    query_texts = [_skillrouter_query_text(query) for query in queries]
    skill_texts = [_skillrouter_skill_text(skill) for skill in skills]
    query_embs = _encode_hf_texts(model_name_or_path, query_texts, batch_size, max_length)
    skill_embs = _encode_hf_texts(model_name_or_path, skill_texts, batch_size, max_length)
    skill_ids = [str(skill["skill_id"]) for skill in skills]
    query_ids = [str(query["query_id"]) for query in queries]
    ranked_by_query = _rank_embedding_rows(query_ids, query_embs, skill_embs, skill_ids, top_k)
    run_path = output_dir / "run.tsv"
    predictions_path = output_dir / "predictions.jsonl"
    write_trec_run(run_path, ranked_by_query, run_name=run_name)
    write_jsonl(
        predictions_path,
        (
            {
                "query_id": query_id,
                "ranked_skill_ids": [skill_id for skill_id, _ in ranked],
                "scores": [score for _, score in ranked],
                "split": split,
            }
            for query_id, ranked in ranked_by_query.items()
        ),
    )
    report = {
        "status": "ok",
        "method": run_name,
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skills),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "model_or_checkpoint": model_name_or_path,
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
    }
    write_json(output_dir / "run_report.json", report)
    return report


def _write_temp_skill_pool(skill_rows: list[dict[str, Any]]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
    with handle:
        for row in skill_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Path(handle.name)


def _filter_state_dict_for_model(model: CLSTRModel, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }


def resolve_clstr_export_config(
    base_model_name: str,
    checkpoint_path: str | Path | None = None,
    model_dim: int = 16,
    top_k: int = 50,
    d_a: int | None = None,
    encoder_pooling: str = "masked_mean",
    cross_encoder_pooling: str = "masked_mean",
    tokenizer_padding_side: str | None = None,
    torch_dtype: str | None = None,
    freeze_backbone: bool = False,
    trust_remote_code: bool = True,
    max_length: int | None = None,
    projection_init: str = "default",
    normalize_embeddings: bool = False,
    defer_skill_table_init: bool = True,
    skill_text_format: str = "skillret_official",
    skill_table_batch_size: int = 1,
    skill_table_adapter_init: str = "default",
    use_cross_encoder: bool = False,
    hf_cache_dir: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "base_model_name": base_model_name,
        "d": model_dim,
        "d_a": d_a if d_a is not None else max(4, model_dim // 4),
        "top_k": top_k,
        "encoder_pooling": encoder_pooling,
        "cross_encoder_pooling": cross_encoder_pooling,
        "tokenizer_padding_side": tokenizer_padding_side,
        "torch_dtype": torch_dtype,
        "freeze_backbone": freeze_backbone,
        "trust_remote_code": trust_remote_code,
        "max_length": max_length,
        "projection_init": projection_init,
        "normalize_embeddings": normalize_embeddings,
        "defer_skill_table_init": defer_skill_table_init,
        "skill_text_format": skill_text_format,
        "skill_table_batch_size": skill_table_batch_size,
        "skill_table_adapter_init": skill_table_adapter_init,
        "use_cross_encoder": use_cross_encoder,
        "hf_cache_dir": hf_cache_dir,
        "local_files_only": local_files_only,
    }
    if checkpoint_path is None:
        return resolved
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = payload.get("config", {})
    if not isinstance(checkpoint_cfg, dict):
        return resolved
    for key in (
        "base_model_name",
        "d",
        "d_a",
        "top_k",
        "encoder_pooling",
        "cross_encoder_pooling",
        "tokenizer_padding_side",
        "torch_dtype",
        "freeze_backbone",
        "trust_remote_code",
        "max_length",
        "projection_init",
        "normalize_embeddings",
        "defer_skill_table_init",
        "skill_text_format",
        "skill_table_batch_size",
        "skill_table_adapter_init",
        "use_cross_encoder",
        "hf_cache_dir",
        "local_files_only",
    ):
        if key in checkpoint_cfg and checkpoint_cfg[key] is not None:
            resolved[key] = checkpoint_cfg[key]
    return resolved


def export_clstr_official_run(
    data_root: str | Path,
    base_model_name: str,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    split: str = "test",
    top_k: int = 50,
    model_dim: int = 16,
    batch_size: int = 16,
    max_queries: int | None = None,
    max_skills: int | None = None,
    encoder_pooling: str = "masked_mean",
    cross_encoder_pooling: str = "masked_mean",
    tokenizer_padding_side: str | None = None,
    torch_dtype: str | None = None,
    freeze_backbone: bool = False,
    trust_remote_code: bool = True,
    max_length: int | None = None,
    projection_init: str = "default",
    normalize_embeddings: bool = False,
    skill_text_format: str = "skillret_official",
    skill_table_batch_size: int = 1,
    skill_table_adapter_init: str = "default",
    use_cross_encoder: bool = False,
    run_name: str = "clstr_skillret_warmup",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
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
    )
    top_k = int(resolved_config["top_k"])
    qrels = load_official_qrels(data_root, split)
    skill_rows, queries, skill_selection = _select_eval_subset(
        load_official_skills(data_root, split),
        load_official_queries(data_root, split),
        qrels,
        max_skills=max_skills,
        max_queries=max_queries,
    )
    skill_path = _write_temp_skill_pool(skill_rows)
    try:
        skills = load_eval_pool(skill_path)
    finally:
        skill_path.unlink(missing_ok=True)
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
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(_filter_state_dict_for_model(model, state_dict), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.rebuild_skill_table()
    skill_ids = [str(row["skill_id"]) for row in skill_rows]
    ranked_by_query: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        with torch.no_grad():
            h = model.encode_states([_skillrouter_query_text(row) for row in batch])
            logits = model.skill_table.retrieval_logits(h)
            values, indices = torch.topk(logits, k=min(top_k, len(skill_ids)), dim=1)
        for query, row_values, row_indices in zip(batch, values.cpu(), indices.cpu()):
            ranked_by_query[str(query["query_id"])] = [
                (skill_ids[idx], float(score))
                for idx, score in zip(row_indices.tolist(), row_values.tolist())
            ]
    run_path = output_dir / "run.tsv"
    predictions_path = output_dir / "predictions.jsonl"
    write_trec_run(run_path, ranked_by_query, run_name=run_name)
    write_jsonl(
        predictions_path,
        (
            {
                "query_id": query_id,
                "ranked_skill_ids": [skill_id for skill_id, _ in ranked],
                "scores": [score for _, score in ranked],
                "split": split,
            }
            for query_id, ranked in ranked_by_query.items()
        ),
    )
    report = {
        "status": "ok",
        "method": run_name,
        "split": split,
        "query_count": len(queries),
        "skill_count": len(skill_rows),
        "skill_pool_selection": skill_selection,
        "top_k": top_k,
        "base_model_name": str(resolved_config["base_model_name"]),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "resolved_config": resolved_config,
        "run_path": str(run_path),
        "predictions_path": str(predictions_path),
        "caveat": "CLSTR run is evaluated on official SKILLRET test split as static retrieval only.",
    }
    write_json(output_dir / "run_report.json", report)
    return report


def _manual_trec_eval(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    metric_sums = {name: 0.0 for name in OFFICIAL_METRIC_NAMES}
    query_count = 0
    for query_id, rels in qrels.items():
        ranked_ids = [
            skill_id
            for skill_id, _ in sorted(results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True)
        ]
        relevant = {skill_id for skill_id, relevance in rels.items() if relevance > 0}
        if not relevant:
            continue
        query_count += 1
        ideal_rels = sorted([float(value) for value in rels.values()], reverse=True)
        ranked_rels = [float(rels.get(skill_id, 0.0)) for skill_id in ranked_ids]
        for k in k_values:
            hits = [1.0 if skill_id in relevant else 0.0 for skill_id in ranked_ids[:k]]
            hit_count = sum(hits)
            recall = hit_count / len(relevant)
            precision_sum = 0.0
            found = 0.0
            for idx, is_hit in enumerate(hits, start=1):
                if is_hit:
                    found += 1.0
                    precision_sum += found / idx
            ap = precision_sum / len(relevant)
            dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ranked_rels[:k]))
            ideal_dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels[:k]))
            metric_sums[f"NDCG@{k}"] += dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            metric_sums[f"Recall@{k}"] += recall
            metric_sums[f"Completeness@{k}"] += 1.0 if recall == 1.0 else 0.0
            metric_sums[f"MAP@{k}"] += ap
    denom = max(query_count, 1)
    return {name: round(value / denom, 5) for name, value in metric_sums.items()}


def compute_official_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: tuple[int, ...] = (5, 10, 15),
) -> tuple[dict[str, float], str]:
    try:
        import pytrec_eval

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {
                "map_cut." + ",".join(str(k) for k in k_values),
                "ndcg_cut." + ",".join(str(k) for k in k_values),
                "recall." + ",".join(str(k) for k in k_values),
                "P." + ",".join(str(k) for k in k_values),
            },
        )
        scores = evaluator.evaluate(results)
        metric_sums = {name: 0.0 for name in OFFICIAL_METRIC_NAMES}
        for query_scores in scores.values():
            for k in k_values:
                metric_sums[f"NDCG@{k}"] += query_scores[f"ndcg_cut_{k}"]
                metric_sums[f"Recall@{k}"] += query_scores[f"recall_{k}"]
                metric_sums[f"Completeness@{k}"] += float(query_scores[f"recall_{k}"] == 1.0)
                metric_sums[f"MAP@{k}"] += query_scores[f"map_cut_{k}"]
        denom = max(len(scores), 1)
        return {name: round(value / denom, 5) for name, value in metric_sums.items()}, "pytrec_eval"
    except ModuleNotFoundError:
        return _manual_trec_eval(qrels, results, k_values), "manual_pytrec_compatible_fallback"


def evaluate_official_run(
    data_root: str | Path,
    run_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    results = load_trec_run(run_path)
    all_qrels = load_official_qrels(data_root, split)
    qrels = {query_id: all_qrels[query_id] for query_id in results if query_id in all_qrels}
    metric_values, backend = compute_official_metrics(qrels, results)
    metrics = {
        "status": "ok",
        "protocol": "official_skillret_pytrec_eval",
        "metric_backend": backend,
        "split": split,
        "data_root": str(data_root),
        "run_path": str(run_path),
        "query_count": len(qrels),
        "result_query_count": len(results),
        "caveat": "SKILLRET is static retrieval/rerank evidence, not a SkillsBench held-out closed-loop harness result.",
        **metric_values,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def write_official_comparison_table(
    eval_root: str | Path,
    run_names: list[str],
    output_table_path: str | Path,
    output_summary_path: str | Path,
) -> dict[str, Any]:
    eval_root = Path(eval_root)
    rows = []
    skipped_non_paper_smoke: list[str] = []
    skipped_non_paper_runs: list[str] = []
    for run_name in run_names:
        metrics_path = eval_root / run_name / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            marker = str(payload.get("paper_role", "")).lower()
            model_ref = str(payload.get("model_or_checkpoint", payload.get("checkpoint", "")))
            if marker == "non_paper_smoke" or "tiny-hf-model" in model_ref:
                skipped_non_paper_smoke.append(run_name)
                continue
            if marker != "paper_static_retrieval" or str(payload.get("split", "")).lower() != "test":
                skipped_non_paper_runs.append(run_name)
                continue
            payload["run_name"] = run_name
            payload["metrics_path"] = str(metrics_path)
            rows.append(payload)
    headers = [
        "method",
        "init",
        "train data",
        "trainable params / frozen backbone",
        "model/checkpoint",
        "NDCG@5",
        "NDCG@10",
        "Recall@10",
        "Completeness@10",
        "MAP@10",
    ]
    lines = [
        "# SKILLRET Official-Protocol Static Retrieval Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("method") or row.get("run_name", "")),
            str(row.get("init", "")),
            str(row.get("train_data", "")),
            str(row.get("trainable_params", row.get("frozen_backbone", ""))),
            str(row.get("model_or_checkpoint", row.get("checkpoint", ""))),
            _fmt_metric(row.get("NDCG@5", 0.0)),
            _fmt_metric(row.get("NDCG@10", 0.0)),
            _fmt_metric(row.get("Recall@10", 0.0)),
            _fmt_metric(row.get("Completeness@10", 0.0)),
            _fmt_metric(row.get("MAP@10", 0.0)),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Caveat: SKILLRET is static retrieval/rerank evidence, not a SkillsBench held-out closed-loop harness result.",
            "Q/doc rows are static routing ablations and are not used as the CLSTR mainline trajectory-training initialization.",
            "CLSTR-native rerank is the mainline routing-stack candidate when qdoc_adapter_used=false.",
            "Q/doc adapters and listwise rerank are treated as a routing foundation rather than the CLSTR closed-loop contribution.",
            "Retrieval InfoNCE and listwise rerank are both categorized as L_retr, not new CLSTR core losses.",
            "These metrics do not prove CLSTR belief-filter closed-loop success; SkillsBench harness evaluation remains required.",
        ]
    )
    output_table_path = Path(output_table_path)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "ok" if rows else "missing",
        "eval_root": str(eval_root),
        "run_names": run_names,
        "rows": rows,
        "skipped_non_paper_smoke": skipped_non_paper_smoke,
        "skipped_non_paper_runs": skipped_non_paper_runs,
        "comparison_table": str(output_table_path),
        "caveat": "SKILLRET is static retrieval/rerank evidence, not a SkillsBench held-out closed-loop harness result.",
    }
    write_json(output_summary_path, summary)
    return summary
