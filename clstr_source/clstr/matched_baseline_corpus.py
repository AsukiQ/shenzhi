from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from clstr.baseline_corpus_types import (
    UnifiedSkillRouterCorpus,
    UnifiedSkillRouterQuery,
)
from clstr.history_channel import compact_causal_state_from_row


MATCHED_BASELINE_CORPUS_SCHEMA = "clstr_matched_baseline_corpus_v1"
MATCHED_BASELINE_QUERY_SCHEMA = "clstr_matched_baseline_query_v1"
REQUIRED_VNEXT_SCHEMA_VERSION = "clstr_vnext_semantic_v1"
MATCHED_BASELINE_QUERY_CONTRACT = {
    "training_query_field": "query",
    "training_query_channel": "compact_causal_skill_action_v1",
    "current_state_field": "query_current",
    "causal_state_field": "query_causal",
    "target_semantics": {
        "retrieval": "required_tool_set_skill_ids",
        "static_route": "current_state_route_set_skill_ids",
    },
    "candidate_contract": "runtime_visible_inventory_exact_v1",
    "split_contract": "vnext_manifest_locked_no_resplit_v1",
}
REQUIRED_VNEXT_MODEL_INPUT_CONTRACT = {
    "history_free_current_state_channel": True,
    "route_query_channel": "compact_causal_skill_action_v1",
    "route_query_history_max_events": 8,
    "raw_tool_results_in_route_query": False,
    "explicit_benchmark_or_source_label_added_to_query": False,
    "executed_skill_ids_in_route_query": True,
    "executed_skill_ids_may_be_namespaced": True,
    "static_dynamic_route_query_identical": True,
    "training_target_semantics": "current_skill_before_memory_update_v1",
}
REQUIRED_VNEXT_FILES = (
    "training_skills",
    "inventory_catalogs",
    "retrieval_rows",
    "retrieval_dev_rows",
    "static_route_rows",
    "static_route_dev_rows",
)


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


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
            yield row


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _load_inventory_catalogs(path: str | Path) -> dict[str, dict[str, Any]]:
    catalogs: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        catalog_id = str(row.get("inventory_catalog_id") or "").strip()
        if not catalog_id or catalog_id in catalogs:
            raise ValueError(f"invalid or duplicate inventory catalog: {catalog_id}")
        catalogs[catalog_id] = row
    return catalogs


def _semantic_source_id(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    value = str(
        provenance.get("source_id")
        or row.get("source")
        or row.get("benchmark")
        or ""
    ).strip()
    if not value:
        raise ValueError("matched baseline query lacks a source identity")
    return value


def _require_verified_data_contract(
    manifest_path: Path,
    expected_paths: dict[str, Path],
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("matched baseline export requires an approved vNext manifest")
    if payload.get("schema_version") != REQUIRED_VNEXT_SCHEMA_VERSION:
        raise ValueError("matched baseline export requires the finalized vNext schema")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("vNext manifest lacks its file table")
    model_input_contract = payload.get("model_input_contract")
    if not isinstance(model_input_contract, dict) or any(
        model_input_contract.get(key) != value
        for key, value in REQUIRED_VNEXT_MODEL_INPUT_CONTRACT.items()
    ):
        raise ValueError("vNext manifest uses an unsupported model-input contract")
    verified: dict[str, Any] = {}
    for key, expected_path in sorted(expected_paths.items()):
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"vNext manifest lacks file entry: {key}")
        recorded_path = Path(str(entry.get("path") or ""))
        if not recorded_path.is_absolute():
            recorded_path = manifest_path.parent / recorded_path
        recorded_digest = str(entry.get("sha256") or "")
        if not expected_path.is_file() or not recorded_path.is_file() or not recorded_digest:
            raise ValueError(f"vNext manifest has an invalid file entry: {key}")
        observed_digest = _file_sha256(expected_path)
        if observed_digest != recorded_digest:
            raise ValueError(f"vNext artifact digest differs from manifest: {key}")
        verified[key] = {
            "path": str(expected_path.resolve()),
            "manifest_path": str(recorded_path.resolve()),
            "content_alias": expected_path.resolve() != recorded_path.resolve(),
            "sha256": observed_digest,
        }
    return {
        "status": "ok",
        "contract_path": str(manifest_path.resolve()),
        "contract_sha256": _file_sha256(manifest_path),
        "schema_version": str(payload.get("schema_version") or ""),
        "model_input_contract": model_input_contract,
        "verified_files": verified,
    }


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or row.get("id") or "").strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _manifest_paths(manifest_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"vNext manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("vNext manifest lacks its file table")
    paths: dict[str, Path] = {}
    for key in REQUIRED_VNEXT_FILES:
        entry = files.get(key)
        if not isinstance(entry, dict) or not str(entry.get("path") or "").strip():
            raise ValueError(f"vNext manifest lacks required file entry: {key}")
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths[key] = path
    verified = _require_verified_data_contract(manifest_path, paths)
    return paths, verified


def _validate_skills(skills: list[dict[str, Any]]) -> tuple[list[str], set[str]]:
    ordered: list[str] = []
    for row in skills:
        skill_id = _skill_id(row)
        if not skill_id:
            raise ValueError("matched baseline skill pool contains an empty skill ID")
        ordered.append(skill_id)
    if len(ordered) != len(set(ordered)):
        raise ValueError("matched baseline skill pool contains duplicate skill IDs")
    if not ordered:
        raise ValueError("matched baseline skill pool is empty")
    return ordered, set(ordered)


def _validate_catalogs(
    catalogs: dict[str, dict[str, Any]],
    known_skill_ids: set[str],
) -> dict[str, list[str]]:
    candidates_by_catalog: dict[str, list[str]] = {}
    for catalog_id, catalog in sorted(catalogs.items()):
        candidates = sorted(
            {
                str(item).strip()
                for item in catalog.get("runtime_visible_skill_ids") or []
                if str(item).strip()
            }
        )
        if not candidates:
            raise ValueError(f"inventory catalog is empty: {catalog_id}")
        unknown = sorted(set(candidates) - known_skill_ids)
        if unknown:
            raise ValueError(
                f"inventory catalog references skills outside training_skills: "
                f"{catalog_id}: {unknown[:4]}"
            )
        observed_digest = str(catalog.get("inventory_catalog_digest") or "")
        expected_digest = _json_digest(candidates)
        if observed_digest != expected_digest:
            raise ValueError(f"inventory catalog digest mismatch: {catalog_id}")
        candidates_by_catalog[catalog_id] = candidates
    if not candidates_by_catalog:
        raise ValueError("matched baseline corpus has no inventory catalogs")
    return candidates_by_catalog


def _source_row_id(row: dict[str, Any], *, kind: str) -> str:
    values = (
        (
            row.get("current_state_route_group_identity"),
            row.get("query_id"),
            row.get("trajectory_id"),
            row.get("task_id"),
        )
        if kind == "static_route"
        else (
            row.get("query_id"),
            row.get("trajectory_id"),
            row.get("task_id"),
            row.get("current_state_route_group_identity"),
        )
    )
    for value in values:
        resolved = str(value or "").strip()
        if resolved:
            return resolved
    return _json_digest(
        {
            "kind": kind,
            "state_text_current": row.get("state_text_current"),
            "split_group_identity": row.get("split_group_identity"),
            "inventory_catalog_digest": row.get("inventory_catalog_digest"),
        }
    )


def _query_from_vnext_row(
    row: dict[str, Any],
    *,
    kind: str,
    split: str,
    candidates_by_catalog: dict[str, list[str]],
    known_skill_ids: set[str],
    source_manifest_sha256: str,
    candidate_sets_by_catalog: dict[str, set[str]] | None = None,
    catalog_digests: dict[str, str] | None = None,
    full_pool_catalogs: set[str] | None = None,
) -> dict[str, Any] | None:
    if str(row.get("data_split") or "").strip().lower() != split:
        raise ValueError(f"vNext {kind} row split does not match its artifact: {split}")
    current_query = str(row.get("state_text_current") or "").strip()
    if not current_query:
        raise ValueError(f"vNext {kind} row lacks state_text_current")
    causal_query = compact_causal_state_from_row(row)
    target_field = (
        "required_tool_set_skill_ids"
        if kind == "retrieval"
        else "current_state_route_set_skill_ids"
    )
    positives = sorted(
        {
            str(item).strip()
            for item in row.get(target_field) or []
            if str(item).strip()
        }
    )
    if not positives:
        return None
    unknown_positives = sorted(set(positives) - known_skill_ids)
    if unknown_positives:
        raise ValueError(
            f"vNext {kind} row references unknown positives: {unknown_positives[:4]}"
        )
    catalog_id = str(row.get("runtime_visible_catalog_id") or "").strip()
    if catalog_id not in candidates_by_catalog:
        raise ValueError(f"vNext {kind} row references an unknown catalog: {catalog_id}")
    candidates = candidates_by_catalog[catalog_id]
    candidate_set = (
        candidate_sets_by_catalog[catalog_id]
        if candidate_sets_by_catalog is not None
        else set(candidates)
    )
    positive_set = set(positives)
    if not positive_set.issubset(candidate_set):
        raise ValueError(f"vNext {kind} row has positives outside its legal catalog")
    if candidate_set.issubset(positive_set):
        return None
    catalog_digest = str(row.get("inventory_catalog_digest") or "").strip()
    expected_catalog_digest = (
        catalog_digests[catalog_id]
        if catalog_digests is not None
        else _json_digest(candidates)
    )
    if catalog_digest != expected_catalog_digest:
        raise ValueError(f"vNext {kind} row catalog digest does not match catalog contents")
    split_group_identity = str(row.get("split_group_identity") or "").strip()
    if not split_group_identity:
        raise ValueError(f"vNext {kind} row lacks split_group_identity")
    if kind == "static_route" and not str(
        row.get("current_state_route_group_identity") or ""
    ).strip():
        raise ValueError("vNext static-route row lacks its grouped-state identity")
    split_schema_version = str(row.get("split_schema_version") or "").strip()
    split_assignment_source = str(row.get("split_assignment_source") or "").strip()
    if not split_schema_version or not split_assignment_source:
        raise ValueError(f"vNext {kind} row lacks locked split provenance")
    source_id = _semantic_source_id(row)
    source_row_id = _source_row_id(row, kind=kind)
    query_identity = {
        "kind": kind,
        "source_id": source_id,
        "source_row_id": source_row_id,
        "split": split,
        "split_group_identity": split_group_identity,
        "inventory_catalog_digest": catalog_digest,
        "query_sha256": hashlib.sha256(causal_query.encode("utf-8")).hexdigest(),
        "positive_skill_ids": positives,
    }
    query = {
        "schema_version": MATCHED_BASELINE_QUERY_SCHEMA,
        "query_id": f"matched::{kind}::{_json_digest(query_identity)}",
        "query": causal_query,
        "query_current": current_query,
        "query_causal": causal_query,
        "query_channel": "compact_causal_skill_action_v1",
        "history_mode": "bounded_skill_action_prefix_max8_v1",
        "benchmark": str(
            row.get("benchmark")
            or row.get("source_benchmark")
            or source_id
        ),
        "source_id": source_id,
        "source_row_id": source_row_id,
        "kind": kind,
        "positive_skill_ids": positives,
        "split": split,
        "split_group_identity": split_group_identity,
        "split_schema_version": split_schema_version,
        "split_assignment_source": split_assignment_source,
        "runtime_visible_catalog_id": catalog_id,
        "inventory_catalog_digest": catalog_digest,
        "inventory_pool_size": len(candidates),
        "source_manifest_sha256": source_manifest_sha256,
    }
    is_full_pool = (
        catalog_id in full_pool_catalogs
        if full_pool_catalogs is not None
        else candidate_set == known_skill_ids
    )
    if not is_full_pool:
        query["candidate_skill_ids"] = candidates
    else:
        query["candidate_scope"] = "all_selected_skills"
    return query


def _materialize_split(
    rows_by_kind: dict[str, list[dict[str, Any]]],
    *,
    split: str,
    candidates_by_catalog: dict[str, list[str]],
    known_skill_ids: set[str],
    source_manifest_sha256: str,
    candidate_sets_by_catalog: dict[str, set[str]],
    catalog_digests: dict[str, str],
    full_pool_catalogs: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for kind in ("retrieval", "static_route"):
        for row in rows_by_kind[kind]:
            query = _query_from_vnext_row(
                row,
                kind=kind,
                split=split,
                candidates_by_catalog=candidates_by_catalog,
                known_skill_ids=known_skill_ids,
                source_manifest_sha256=source_manifest_sha256,
                candidate_sets_by_catalog=candidate_sets_by_catalog,
                catalog_digests=catalog_digests,
                full_pool_catalogs=full_pool_catalogs,
            )
            if query is None:
                excluded[f"{kind}:no_positive_or_legal_negative"] += 1
                continue
            queries.append(query)
    queries.sort(key=lambda row: str(row["query_id"]))
    query_ids = [str(row["query_id"]) for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"matched baseline {split} query IDs are not unique")
    return queries, {
        "query_count": len(queries),
        "by_kind": dict(sorted(Counter(str(row["kind"]) for row in queries).items())),
        "by_source": dict(
            sorted(Counter(str(row["source_id"]) for row in queries).items())
        ),
        "multi_positive_query_count": sum(
            int(len(row["positive_skill_ids"]) > 1) for row in queries
        ),
        "excluded": dict(sorted(excluded.items())),
    }


def build_matched_baseline_corpus(
    *,
    vnext_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(vnext_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, verified_contract = _manifest_paths(manifest_path)
    source_manifest_sha256 = _file_sha256(manifest_path)
    skills = _read_jsonl(paths["training_skills"])
    _ordered_skill_ids, known_skill_ids = _validate_skills(skills)
    catalogs = _load_inventory_catalogs(paths["inventory_catalogs"])
    candidates_by_catalog = _validate_catalogs(catalogs, known_skill_ids)
    candidate_sets_by_catalog = {
        catalog_id: set(candidates)
        for catalog_id, candidates in candidates_by_catalog.items()
    }
    catalog_digests = {
        catalog_id: _json_digest(candidates)
        for catalog_id, candidates in candidates_by_catalog.items()
    }
    full_pool_catalogs = {
        catalog_id
        for catalog_id, candidates in candidate_sets_by_catalog.items()
        if candidates == known_skill_ids
    }
    train_queries, train_report = _materialize_split(
        {
            "retrieval": _read_jsonl(paths["retrieval_rows"]),
            "static_route": _read_jsonl(paths["static_route_rows"]),
        },
        split="train",
        candidates_by_catalog=candidates_by_catalog,
        known_skill_ids=known_skill_ids,
        source_manifest_sha256=source_manifest_sha256,
        candidate_sets_by_catalog=candidate_sets_by_catalog,
        catalog_digests=catalog_digests,
        full_pool_catalogs=full_pool_catalogs,
    )
    eval_queries, eval_report = _materialize_split(
        {
            "retrieval": _read_jsonl(paths["retrieval_dev_rows"]),
            "static_route": _read_jsonl(paths["static_route_dev_rows"]),
        },
        split="dev",
        candidates_by_catalog=candidates_by_catalog,
        known_skill_ids=known_skill_ids,
        source_manifest_sha256=source_manifest_sha256,
        candidate_sets_by_catalog=candidate_sets_by_catalog,
        catalog_digests=catalog_digests,
        full_pool_catalogs=full_pool_catalogs,
    )
    if not train_queries or not eval_queries:
        raise ValueError("matched baseline corpus requires nonempty train and dev queries")
    train_groups = {str(row["split_group_identity"]) for row in train_queries}
    eval_groups = {str(row["split_group_identity"]) for row in eval_queries}
    overlap = sorted(train_groups & eval_groups)
    if overlap:
        raise ValueError(f"matched baseline split groups cross train/dev: {overlap[:4]}")
    output_files = {
        "selected_skills": output_dir / "selected_skills.jsonl",
        "train_queries": output_dir / "train_queries.jsonl",
        "eval_queries": output_dir / "eval_queries.jsonl",
    }
    counts = {
        "selected_skills": _write_jsonl(output_files["selected_skills"], skills),
        "train_queries": _write_jsonl(output_files["train_queries"], train_queries),
        "eval_queries": _write_jsonl(output_files["eval_queries"], eval_queries),
    }
    manifest = {
        "status": "ok",
        "schema_version": MATCHED_BASELINE_CORPUS_SCHEMA,
        "query_schema_version": MATCHED_BASELINE_QUERY_SCHEMA,
        "query_contract": MATCHED_BASELINE_QUERY_CONTRACT,
        "source_vnext_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": source_manifest_sha256,
            "schema_version": verified_contract["schema_version"],
            "model_input_contract": verified_contract["model_input_contract"],
            "verified_files": verified_contract["verified_files"],
        },
        "counts": counts,
        "train_report": train_report,
        "eval_report": eval_report,
        "split_group_overlap_count": 0,
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for key, path in output_files.items()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _verified_prepared_paths(corpus_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared matched baseline manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "ok"
        or manifest.get("schema_version") != MATCHED_BASELINE_CORPUS_SCHEMA
    ):
        raise ValueError("prepared matched baseline corpus manifest is not approved")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("prepared matched baseline manifest lacks files")
    if manifest.get("query_schema_version") != MATCHED_BASELINE_QUERY_SCHEMA:
        raise ValueError("prepared matched baseline query schema is unsupported")
    if manifest.get("query_contract") != MATCHED_BASELINE_QUERY_CONTRACT:
        raise ValueError("prepared matched baseline query contract is unsupported")
    paths: dict[str, Path] = {}
    for key in ("selected_skills", "train_queries", "eval_queries"):
        entry = files.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"prepared matched baseline manifest lacks {key}")
        path = corpus_dir / f"{key}.jsonl"
        digest = str(entry.get("sha256") or "")
        if not path.is_file() or not digest or _file_sha256(path) != digest:
            raise ValueError(f"prepared matched baseline file digest mismatch: {key}")
        paths[key] = path
    return paths, manifest


def _query_dataclass(
    row: dict[str, Any],
    *,
    split: str,
    skill_id_to_idx: dict[str, int],
    known_skill_ids: set[str],
    source_manifest_sha256: str,
    catalog_validation_cache: dict[
        tuple[str, ...] | None,
        tuple[set[str], str],
    ],
) -> UnifiedSkillRouterQuery:
    if str(row.get("schema_version") or "") != MATCHED_BASELINE_QUERY_SCHEMA:
        raise ValueError("prepared query uses an unsupported schema")
    if str(row.get("split") or "") != split:
        raise ValueError(f"prepared query split mismatch: expected {split}")
    raw_positives = [
        str(item).strip()
        for item in row.get("positive_skill_ids") or []
        if str(item).strip()
    ]
    unknown_positives = sorted(set(raw_positives) - known_skill_ids)
    if unknown_positives:
        raise ValueError(f"prepared query has unknown positives: {unknown_positives[:4]}")
    positives = raw_positives
    candidates = (
        [
            str(item).strip()
            for item in row.get("candidate_skill_ids") or []
            if str(item).strip()
        ]
        if "candidate_skill_ids" in row
        else None
    )
    if not positives or len(positives) != len(set(positives)):
        raise ValueError("prepared query lacks a unique nonempty positive set")
    query_id = str(row.get("query_id") or "").strip()
    query_text = str(row.get("query") or "").strip()
    query_current = str(row.get("query_current") or "").strip()
    query_causal = str(row.get("query_causal") or "").strip()
    kind = str(row.get("kind") or "").strip()
    source_id = str(row.get("source_id") or "").strip()
    split_group_identity = str(row.get("split_group_identity") or "").strip()
    split_schema_version = str(row.get("split_schema_version") or "").strip()
    split_assignment_source = str(row.get("split_assignment_source") or "").strip()
    runtime_visible_catalog_id = str(
        row.get("runtime_visible_catalog_id") or ""
    ).strip()
    if (
        not query_id
        or not query_text
        or not query_current
        or query_text != query_causal
        or kind not in {"retrieval", "static_route"}
        or not source_id
        or not split_group_identity
        or not split_schema_version
        or not split_assignment_source
        or not runtime_visible_catalog_id
    ):
        raise ValueError("prepared query lacks a valid identity, text, or capability kind")
    if row.get("query_channel") != "compact_causal_skill_action_v1":
        raise ValueError("prepared query uses an unsupported query channel")
    if row.get("history_mode") != "bounded_skill_action_prefix_max8_v1":
        raise ValueError("prepared query uses an unsupported history mode")
    if str(row.get("source_manifest_sha256") or "") != source_manifest_sha256:
        raise ValueError("prepared query source manifest identity mismatch")
    catalog_key = None if candidates is None else tuple(candidates)
    cached_catalog = catalog_validation_cache.get(catalog_key)
    if cached_catalog is None:
        legal_candidates = known_skill_ids if candidates is None else set(candidates)
        if candidates is not None and len(candidates) != len(legal_candidates):
            raise ValueError("prepared query candidate IDs are not unique")
        unknown_candidates = sorted(legal_candidates - known_skill_ids)
        if unknown_candidates:
            raise ValueError(
                f"prepared query has unknown candidates: {unknown_candidates[:4]}"
            )
        cached_catalog = (
            legal_candidates,
            _json_digest(sorted(legal_candidates)),
        )
        catalog_validation_cache[catalog_key] = cached_catalog
    legal_candidates, expected_catalog_digest = cached_catalog
    if candidates is None:
        if row.get("candidate_scope") != "all_selected_skills":
            raise ValueError("prepared query omits candidates without an all-skill scope")
    elif row.get("candidate_scope") is not None:
        raise ValueError("prepared query declares both local and all-skill candidate scopes")
    if not set(positives).issubset(legal_candidates):
        raise ValueError("prepared query positive is outside its candidate catalog")
    if str(row.get("inventory_catalog_digest") or "") != expected_catalog_digest:
        raise ValueError("prepared query inventory digest does not match its candidates")
    if int(row.get("inventory_pool_size") or 0) != len(legal_candidates):
        raise ValueError("prepared query inventory size does not match its candidates")
    return UnifiedSkillRouterQuery(
        query_id=query_id,
        query=query_text,
        benchmark=str(row.get("benchmark") or "unknown"),
        positive_skill_ids=positives,
        positive_indices=[skill_id_to_idx[item] for item in positives],
        candidate_skill_ids=candidates,
        source_id=source_id,
        kind=kind,
        split_group_identity=split_group_identity,
        runtime_visible_catalog_id=runtime_visible_catalog_id,
        inventory_catalog_digest=str(row.get("inventory_catalog_digest") or ""),
        history_mode=str(row.get("history_mode") or ""),
    )


def _stable_stratified_query_cap(
    queries: list[UnifiedSkillRouterQuery],
    limit: int | None,
    *,
    seed: int,
) -> list[UnifiedSkillRouterQuery]:
    """Select a deterministic capability/source-balanced subset.

    The full corpus is returned unchanged. Capped pilot and validation subsets
    alternates capabilities first, then round-robins sources inside each
    capability. This mirrors prepared training exposure even when the two
    capabilities have different source counts.
    """

    if limit is None or int(limit) >= len(queries):
        return list(queries)
    if int(limit) <= 0:
        raise ValueError("prepared matched query caps must be positive")
    grouped: dict[str, dict[str, list[UnifiedSkillRouterQuery]]] = {}
    for query in queries:
        kind = str(query.kind)
        source = str(query.source_id)
        grouped.setdefault(kind, {}).setdefault(source, []).append(query)
    if not grouped:
        return []
    ordered_kinds = [
        kind
        for kind in ("retrieval", "static_route")
        if kind in grouped
    ]
    ordered_kinds.extend(
        kind for kind in sorted(grouped) if kind not in ordered_kinds
    )
    ordered_sources: dict[str, list[str]] = {}
    for kind in ordered_kinds:
        sources = sorted(grouped[kind])
        random.Random(
            int.from_bytes(
                hashlib.sha256(
                    f"{int(seed)}\0{kind}\0sources".encode("utf-8")
                ).digest()[:8],
                "big",
            )
        ).shuffle(sources)
        ordered_sources[kind] = sources
        for source in sources:
            rows = sorted(
                grouped[kind][source],
                key=lambda item: item.query_id,
            )
            row_seed = int.from_bytes(
                hashlib.sha256(
                    f"{int(seed)}\0{kind}\0{source}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            random.Random(row_seed).shuffle(rows)
            grouped[kind][source] = rows
    selected: list[UnifiedSkillRouterQuery] = []
    row_cursors = {
        (kind, source): 0
        for kind in ordered_kinds
        for source in ordered_sources[kind]
    }
    source_cursors = {kind: 0 for kind in ordered_kinds}
    while len(selected) < int(limit):
        added = False
        for kind in ordered_kinds:
            sources = ordered_sources[kind]
            for _ in range(len(sources)):
                source_position = source_cursors[kind] % len(sources)
                source_cursors[kind] += 1
                source = sources[source_position]
                key = (kind, source)
                cursor = row_cursors[key]
                rows = grouped[kind][source]
                if cursor >= len(rows):
                    continue
                selected.append(rows[cursor])
                row_cursors[key] = cursor + 1
                added = True
                break
            if len(selected) >= int(limit):
                break
        if not added:
            break
    return selected


def load_prepared_matched_baseline_corpus(
    corpus_dir: str | Path,
    *,
    max_rows: int | None = None,
    max_eval_rows: int | None = None,
    max_skills: int | None = None,
    seed: int = 13,
) -> UnifiedSkillRouterCorpus:
    corpus_dir = Path(corpus_dir)
    paths, manifest = _verified_prepared_paths(corpus_dir)
    source_manifest = manifest.get("source_vnext_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("prepared matched baseline lacks its source manifest contract")
    if source_manifest.get("schema_version") != REQUIRED_VNEXT_SCHEMA_VERSION:
        raise ValueError("prepared matched baseline source schema is unsupported")
    source_model_input_contract = source_manifest.get("model_input_contract")
    if not isinstance(source_model_input_contract, dict) or any(
        source_model_input_contract.get(key) != value
        for key, value in REQUIRED_VNEXT_MODEL_INPUT_CONTRACT.items()
    ):
        raise ValueError("prepared matched baseline source contract is unsupported")
    source_manifest_sha256 = str(source_manifest.get("sha256") or "").strip()
    try:
        digest_value = int(source_manifest_sha256, 16)
    except ValueError:
        digest_value = -1
    if len(source_manifest_sha256) != 64 or digest_value < 0:
        raise ValueError("prepared matched baseline source manifest digest is invalid")
    skills = _read_jsonl(paths["selected_skills"])
    skill_ids, known_skill_ids = _validate_skills(skills)
    if max_skills is not None and int(max_skills) < len(skills):
        raise ValueError(
            "prepared matched corpus cannot be skill-capped without changing its legal catalogs"
        )
    skill_id_to_idx = {skill_id: index for index, skill_id in enumerate(skill_ids)}
    catalog_validation_cache: dict[
        tuple[str, ...] | None,
        tuple[set[str], str],
    ] = {}
    train_queries = [
        _query_dataclass(
            row,
            split="train",
            skill_id_to_idx=skill_id_to_idx,
            known_skill_ids=known_skill_ids,
            source_manifest_sha256=source_manifest_sha256,
            catalog_validation_cache=catalog_validation_cache,
        )
        for row in _iter_jsonl(paths["train_queries"])
    ]
    eval_queries = [
        _query_dataclass(
            row,
            split="dev",
            skill_id_to_idx=skill_id_to_idx,
            known_skill_ids=known_skill_ids,
            source_manifest_sha256=source_manifest_sha256,
            catalog_validation_cache=catalog_validation_cache,
        )
        for row in _iter_jsonl(paths["eval_queries"])
    ]
    counts = manifest.get("counts")
    expected_counts = {
        "selected_skills": len(skills),
        "train_queries": len(train_queries),
        "eval_queries": len(eval_queries),
    }
    if not isinstance(counts, dict) or any(
        int(counts.get(key) or -1) != value for key, value in expected_counts.items()
    ):
        raise ValueError("prepared matched baseline manifest counts do not match its files")
    train_query_ids = [query.query_id for query in train_queries]
    eval_query_ids = [query.query_id for query in eval_queries]
    if len(train_query_ids) != len(set(train_query_ids)):
        raise ValueError("prepared train query IDs are not unique")
    if len(eval_query_ids) != len(set(eval_query_ids)):
        raise ValueError("prepared dev query IDs are not unique")
    if set(train_query_ids) & set(eval_query_ids):
        raise ValueError("prepared query IDs cross train/dev")
    source_train_query_count = len(train_queries)
    source_eval_query_count = len(eval_queries)
    train_queries = _stable_stratified_query_cap(
        train_queries,
        max_rows,
        seed=int(seed),
    )
    eval_queries = _stable_stratified_query_cap(
        eval_queries,
        max_eval_rows,
        seed=int(seed) + 1,
    )
    random.Random(int(seed)).shuffle(train_queries)
    if not train_queries or not eval_queries:
        raise ValueError("prepared matched corpus has an empty selected train/dev split")
    train_groups = {query.split_group_identity for query in train_queries}
    eval_groups = {query.split_group_identity for query in eval_queries}
    if train_groups & eval_groups:
        raise ValueError("prepared matched corpus split groups cross train/dev")
    report = {
        "prepared_corpus_dir": str(corpus_dir.resolve()),
        "prepared_manifest_path": str((corpus_dir / "manifest.json").resolve()),
        "prepared_manifest_sha256": _file_sha256(corpus_dir / "manifest.json"),
        "source_vnext_manifest": manifest.get("source_vnext_manifest"),
        "skill_count": len(skills),
        "train_query_count": len(train_queries),
        "eval_query_count": len(eval_queries),
        "source_train_query_count": source_train_query_count,
        "source_eval_query_count": source_eval_query_count,
        "max_rows": max_rows,
        "max_eval_rows": max_eval_rows,
        "selection_protocol": "capability_first_source_round_robin_cap_v2",
        "seeded_train_order": True,
        "query_contract": manifest.get("query_contract"),
        "benchmark_counts": dict(
            sorted(Counter(query.benchmark for query in [*train_queries, *eval_queries]).items())
        ),
        "kind_counts": dict(
            sorted(Counter(query.kind for query in [*train_queries, *eval_queries]).items())
        ),
    }
    return UnifiedSkillRouterCorpus(
        skills=skills,
        train_queries=train_queries,
        eval_queries=eval_queries,
        report=report,
    )
