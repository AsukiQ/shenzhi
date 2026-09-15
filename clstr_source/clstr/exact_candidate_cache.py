from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

EXACT_CANDIDATE_CACHE_SCHEMA = "clstr_exact_candidate_rankings_v2"


def _json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_inference_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"embedding checkpoint does not exist: {path}")
    selected: list[Path] = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        name = candidate.name
        if name == "training_state.pt":
            continue
        if (
            candidate.suffix
            in {".safetensors", ".model", ".tiktoken", ".vocab", ".py"}
            or (name.startswith("pytorch_model") and name.endswith(".bin"))
            or name.startswith("tokenizer")
            or name.startswith("vocab")
            or name in {
                "config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "merges.txt",
                "modules.json",
                "sentence_bert_config.json",
                "train_checkpoint_report.json",
            }
            or name.endswith(".index.json")
            or relative == "1_Pooling/config.json"
        ):
            selected.append(candidate)
    if not selected:
        selected = [
            candidate
            for candidate in sorted(item for item in path.rglob("*") if item.is_file())
            if candidate.name != "training_state.pt"
        ]
    if not selected:
        raise ValueError(f"embedding checkpoint has no inference files: {path}")
    return selected


def checkpoint_inference_digest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    files = _checkpoint_inference_files(path)
    root = path if path.is_dir() else path.parent
    entries = [
        {
            "path": file.relative_to(root).as_posix(),
            "size": file.stat().st_size,
            "sha256": _file_sha256(file),
        }
        for file in files
    ]
    return {
        "file_count": len(entries),
        "files": entries,
        "digest": _json_digest(entries),
    }


def candidate_cache_spec(
    *,
    checkpoint_path: str | Path,
    selected_skills_path: str | Path,
    train_queries_path: str | Path,
    eval_queries_path: str | Path,
    pooling: str,
    query_text_mode: str,
    tokenizer_padding_side: str,
    torch_dtype: str,
    max_length: int,
    top_k: int,
    max_train_queries: int | None,
    max_eval_queries: int | None,
    skill_ids: list[str],
    query_ids: list[str],
    candidate_skill_ids_by_query: list[list[str] | None] | None,
    ranking_device: str,
) -> dict[str, Any]:
    source_files = {}
    for key, raw_path in (
        ("selected_skills", selected_skills_path),
        ("train_queries", train_queries_path),
        ("eval_queries", eval_queries_path),
    ):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"candidate cache source file is missing: {path}")
        source_files[key] = {
            "sha256": _file_sha256(path),
        }
    return {
        "schema_version": EXACT_CANDIDATE_CACHE_SCHEMA,
        "checkpoint": checkpoint_inference_digest(checkpoint_path),
        "source_files": source_files,
        "encoder_contract": {
            "pooling": str(pooling),
            "query_text_mode": str(query_text_mode),
            "tokenizer_padding_side": str(tokenizer_padding_side),
            "torch_dtype": str(torch_dtype),
            "max_length": int(max_length),
            "normalization": "existing_encoder_exact_v1",
        },
        "ranking_contract": {
            "top_k": int(top_k),
            "max_train_queries": (
                None if max_train_queries is None else int(max_train_queries)
            ),
            "max_eval_queries": (
                None if max_eval_queries is None else int(max_eval_queries)
            ),
            "skill_ids_sha256": _json_digest(skill_ids),
            "query_ids_sha256": _json_digest(query_ids),
            "candidate_catalogs_sha256": _json_digest(
                candidate_skill_ids_by_query
            ),
            "candidate_scope": "per_query_runtime_visible_catalog_when_declared_v1",
            "ranking": "exact_dense_inner_product_topk_v1",
            "ranking_device": str(ranking_device),
            "ranking_dtype": "float32",
            "cuda_tf32": False,
        },
    }


def _validate_ranked(
    ranked: Any,
    *,
    query_ids: list[str],
    skill_ids: list[str],
    top_k: int,
    candidate_skill_ids_by_query: list[list[str] | None] | None = None,
) -> dict[str, list[str]]:
    if not isinstance(ranked, dict):
        raise ValueError("candidate cache rankings must be a mapping")
    expected_queries = set(query_ids)
    if set(ranked) != expected_queries:
        raise ValueError("candidate cache query IDs differ from the requested corpus")
    known_skills = set(skill_ids)
    if candidate_skill_ids_by_query is not None:
        if len(candidate_skill_ids_by_query) != len(query_ids):
            raise ValueError("candidate cache catalogs differ from the requested queries")
        legal_by_query: dict[str, set[str]] = {}
        catalog_cache: dict[tuple[str, ...] | None, set[str]] = {}
        for query_id, raw_candidates in zip(query_ids, candidate_skill_ids_by_query):
            catalog_key = (
                None if raw_candidates is None else tuple(raw_candidates)
            )
            legal = catalog_cache.get(catalog_key)
            if legal is None:
                candidates = skill_ids if raw_candidates is None else raw_candidates
                legal = set(candidates)
                if not candidates or len(candidates) != len(legal):
                    raise ValueError(
                        f"candidate cache catalog is empty or duplicated: {query_id}"
                    )
                unknown = sorted(legal - known_skills)
                if unknown:
                    raise ValueError(
                        f"candidate cache catalog references unknown skills: {unknown[:4]}"
                    )
                catalog_cache[catalog_key] = legal
            legal_by_query[query_id] = legal
    else:
        legal_by_query = {query_id: known_skills for query_id in query_ids}
    validated: dict[str, list[str]] = {}
    for query_id in query_ids:
        values = ranked.get(query_id)
        if not isinstance(values, list):
            raise ValueError(f"candidate cache row is not a list: {query_id}")
        resolved = [str(item) for item in values]
        if len(resolved) > int(top_k) or len(resolved) != len(set(resolved)):
            raise ValueError(f"candidate cache row violates top-k uniqueness: {query_id}")
        unknown = sorted(set(resolved) - known_skills)
        if unknown:
            raise ValueError(f"candidate cache row references unknown skills: {unknown[:4]}")
        legal = legal_by_query[query_id]
        illegal = sorted(set(resolved) - legal)
        if illegal:
            raise ValueError(
                f"candidate cache row leaves its legal catalog: {query_id}: {illegal[:4]}"
            )
        expected_count = min(int(top_k), len(legal))
        if len(resolved) != expected_count:
            raise ValueError(
                f"candidate cache row has {len(resolved)} results, expected "
                f"{expected_count}: {query_id}"
            )
        validated[query_id] = resolved
    return validated


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_or_build_exact_candidate_rankings(
    *,
    candidate_cache_dir: str | Path | None,
    checkpoint_path: str | Path,
    selected_skills_path: str | Path,
    train_queries_path: str | Path,
    eval_queries_path: str | Path,
    pooling: str,
    query_text_mode: str,
    tokenizer_padding_side: str,
    torch_dtype: str,
    max_length: int,
    top_k: int,
    max_train_queries: int | None,
    max_eval_queries: int | None,
    skill_ids: list[str],
    query_ids: list[str],
    builder: Callable[[], dict[str, list[str]]],
    candidate_skill_ids_by_query: list[list[str] | None] | None = None,
    ranking_device: str = "cpu",
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if (
        len(skill_ids) != len(set(skill_ids))
        or not skill_ids
        or any(not str(skill_id).strip() for skill_id in skill_ids)
    ):
        raise ValueError("candidate cache requires unique nonempty skill IDs")
    if (
        len(query_ids) != len(set(query_ids))
        or not query_ids
        or any(not str(query_id).strip() for query_id in query_ids)
    ):
        raise ValueError("candidate cache requires unique nonempty query IDs")
    if int(top_k) <= 0:
        raise ValueError("candidate cache top_k must be positive")
    if candidate_cache_dir is None:
        ranked = _validate_ranked(
            builder(),
            query_ids=query_ids,
            skill_ids=skill_ids,
            top_k=top_k,
            candidate_skill_ids_by_query=candidate_skill_ids_by_query,
        )
        return ranked, {
            "enabled": False,
            "hit": False,
            "protocol": EXACT_CANDIDATE_CACHE_SCHEMA,
        }
    spec = candidate_cache_spec(
        checkpoint_path=checkpoint_path,
        selected_skills_path=selected_skills_path,
        train_queries_path=train_queries_path,
        eval_queries_path=eval_queries_path,
        pooling=pooling,
        query_text_mode=query_text_mode,
        tokenizer_padding_side=tokenizer_padding_side,
        torch_dtype=torch_dtype,
        max_length=max_length,
        top_k=top_k,
        max_train_queries=max_train_queries,
        max_eval_queries=max_eval_queries,
        skill_ids=skill_ids,
        query_ids=query_ids,
        candidate_skill_ids_by_query=candidate_skill_ids_by_query,
        ranking_device=ranking_device,
    )
    identity = _json_digest(spec)
    root = Path(candidate_cache_dir) / identity
    manifest_path = root / "manifest.json"
    rankings_path = root / "ranked_skill_ids_by_query.json"
    lock_path = Path(candidate_cache_dir) / f"{identity}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_reason: str | None = None
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file() and rankings_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("status") != "ok"
                    or manifest.get("schema_version") != EXACT_CANDIDATE_CACHE_SCHEMA
                    or manifest.get("identity") != identity
                    or manifest.get("spec") != spec
                    or manifest.get("rankings_sha256") != _file_sha256(rankings_path)
                ):
                    raise ValueError("cache manifest contract mismatch")
                ranked = _validate_ranked(
                    json.loads(rankings_path.read_text(encoding="utf-8")),
                    query_ids=query_ids,
                    skill_ids=skill_ids,
                    top_k=top_k,
                    candidate_skill_ids_by_query=candidate_skill_ids_by_query,
                )
                return ranked, {
                    "enabled": True,
                    "hit": True,
                    "identity": identity,
                    "cache_dir": str(root.resolve()),
                    "protocol": EXACT_CANDIDATE_CACHE_SCHEMA,
                }
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                invalid_reason = str(exc)
        ranked = _validate_ranked(
            builder(),
            query_ids=query_ids,
            skill_ids=skill_ids,
            top_k=top_k,
            candidate_skill_ids_by_query=candidate_skill_ids_by_query,
        )
        root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(rankings_path, ranked)
        manifest = {
            "status": "ok",
            "schema_version": EXACT_CANDIDATE_CACHE_SCHEMA,
            "identity": identity,
            "spec": spec,
            "rankings_path": str(rankings_path.resolve()),
            "rankings_sha256": _file_sha256(rankings_path),
            "query_count": len(ranked),
        }
        _atomic_write_json(manifest_path, manifest)
        return ranked, {
            "enabled": True,
            "hit": False,
            "identity": identity,
            "cache_dir": str(root.resolve()),
            "invalid_reason": invalid_reason,
            "protocol": EXACT_CANDIDATE_CACHE_SCHEMA,
        }
