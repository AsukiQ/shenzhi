from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from clstr.history_channel import (
    actual_causal_observation,
    audit_history_channel_rows,
    materialize_structured_retrieval_state,
    materialize_structured_current_state,
)
from clstr.matched_multibench_data import validate_tau2_split_manifest
from clstr.toolbench_clean_training_export import (
    _is_toolbench_retrieval_row,
    _is_toolbench_trajectory_row,
    _provenance,
    _row_answer_path,
    _row_query_ids,
    _row_signature,
    build_toolbench_eval_exclusion,
)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_semantic_jsonl_prefix(
    base_path: Path,
    union_path: Path,
) -> tuple[int, list[dict[str, Any]]]:
    base_count = 0
    with base_path.open(encoding="utf-8") as base, union_path.open(
        encoding="utf-8"
    ) as union:
        for line_no, base_line in enumerate(base, start=1):
            union_line = union.readline()
            if not union_line:
                raise ValueError("matched union ended before its clean-base prefix")
            if _canonical_digest(json.loads(base_line)) != _canonical_digest(
                json.loads(union_line)
            ):
                raise ValueError(
                    f"matched union semantic prefix differs from clean base at line {line_no}"
                )
            base_count += 1
        tail = [json.loads(line) for line in union if line.strip()]
    return base_count, tail


def _verify_exact_file_prefix(base_path: Path, union_path: Path) -> list[dict[str, Any]]:
    with base_path.open("rb") as base, union_path.open("rb") as union:
        while chunk := base.read(16 * 1024 * 1024):
            if union.read(len(chunk)) != chunk:
                raise ValueError("matched union byte prefix differs from clean base")
        tail_bytes = union.read()
    return [
        json.loads(line)
        for line in tail_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]


def _structured_history_audit_rows(
    rows: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        try:
            yield materialize_structured_current_state(row, replace_state_text=True)
        except ValueError as exc:
            yield {
                **row,
                "state_text_current": "",
                "current_state_materialization_error": str(exc),
            }


def _structured_retrieval_audit_rows(
    rows: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        try:
            yield materialize_structured_retrieval_state(
                row,
                replace_state_text=True,
            )
        except ValueError as exc:
            yield {
                **row,
                "state_text_current": "",
                "current_state_materialization_error": str(exc),
            }


def _skill_source(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").strip()
    if source:
        return source
    skill_id = str(row.get("skill_id") or "").lower()
    if skill_id.startswith("bfcl/"):
        return "bfcl"
    if skill_id.startswith("apibank/"):
        return "apibank"
    if skill_id.startswith("toolbench-g3/"):
        return "ToolBench-G3"
    if skill_id.startswith("traject/"):
        return "TRAJECT-Bench"
    if skill_id.startswith("alfworld/"):
        return "alfworld"
    if skill_id.startswith("webshop/"):
        return "webshop"
    return "<missing>"


_CONTENT_TEXT_KEYS = (
    "state_text_full",
    "state_text",
    "query_text",
    "query",
    "goal_text",
    "task_text",
    "history_text",
    "action_text",
    "expert_action",
    "next_observation_text",
)


def _row_content_text(row: dict[str, Any]) -> str:
    semantic = str(row.get("split_semantic_text") or "").strip()
    if semantic:
        return semantic
    values = [str(row.get(key) or "") for key in _CONTENT_TEXT_KEYS]
    return " ".join(value for value in values if value.strip())


def _normalize_content_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _content_shingles(text: str, n: int = 5, max_chars: int | None = None) -> set[str]:
    normalized = _normalize_content_text(text)
    if not normalized:
        return set()
    if max_chars is not None and max_chars > 0:
        normalized = normalized[: int(max_chars)].rstrip()
    tokens = normalized.split()
    if not tokens:
        return set()

    # Use token n-gram features instead of character shingles. This keeps the
    # audit deterministic while avoiding huge posting lists from common
    # character fragments on large retrieval corpora.
    features = {f"1:{token}" for token in tokens}
    max_ngram = max(1, min(int(n), 3))
    for width in range(2, max_ngram + 1):
        if len(tokens) < width:
            continue
        features.update(
            f"{width}:{' '.join(tokens[idx : idx + width])}"
            for idx in range(0, len(tokens) - width + 1)
        )
    return features


def _row_stable_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "query_id", "trajectory_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "<missing>"


def _row_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_match_reference(
    row: dict[str, Any],
    *,
    eval_id: str,
    score: float,
    match_kind: str,
) -> dict[str, Any]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return {
        "train_id": _row_stable_id(row),
        "train_row_sha256": _row_sha256(row),
        "trajectory_id": str(row.get("trajectory_id") or provenance.get("trajectory_id") or ""),
        "task_id": str(row.get("task_id") or provenance.get("task_id") or ""),
        "query_id": str(row.get("query_id") or provenance.get("query_id") or ""),
        "source": str(row.get("source") or provenance.get("source_id") or ""),
        "benchmark": str(row.get("benchmark") or ""),
        "eval_id": str(eval_id),
        "score": round(float(score), 6),
        "match_kind": str(match_kind),
    }


def _content_overlap_report(
    *,
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    threshold: float = 0.55,
    max_chars: int | None = 1000,
    max_shingle_postings: int | None = 256,
    max_query_features: int | None = 64,
    max_candidates_per_row: int | None = 128,
    max_examples: int = 10,
) -> dict[str, Any]:
    eval_entries: list[dict[str, Any]] = []
    eval_exact: dict[str, list[int]] = {}
    eval_shingle_index: dict[str, set[int]] = {}
    for eval_idx, row in enumerate(eval_rows):
        text = _normalize_content_text(_row_content_text(row))
        if not text:
            continue
        shingles = _content_shingles(text, max_chars=max_chars)
        entry = {
            "row": row,
            "id": _row_stable_id(row),
            "signature": text,
            "shingles": shingles,
        }
        eval_entries.append(entry)
        eval_exact.setdefault(text, []).append(len(eval_entries) - 1)
        for shingle in shingles:
            eval_shingle_index.setdefault(shingle, set()).add(len(eval_entries) - 1)

    skipped_high_posting_shingles = 0
    if max_shingle_postings is not None and max_shingle_postings > 0:
        max_postings = int(max_shingle_postings)
        for shingle in list(eval_shingle_index):
            if len(eval_shingle_index[shingle]) > max_postings:
                skipped_high_posting_shingles += 1
                del eval_shingle_index[shingle]

    exact_hits = 0
    near_hits = 0
    candidate_pairs_scored = 0
    skipped_query_features = 0
    truncated_candidate_rows = 0
    examples: list[dict[str, Any]] = []
    matched_train_rows: list[dict[str, Any]] = []
    for row in train_rows:
        train_text = _normalize_content_text(_row_content_text(row))
        if not train_text:
            continue
        train_id = _row_stable_id(row)
        train_shingles = _content_shingles(train_text, max_chars=max_chars)
        matched_idx: int | None = None
        score = 0.0
        exact_indices = eval_exact.get(train_text) or []
        if exact_indices:
            exact_hits += 1
            matched_idx = exact_indices[0]
            score = 1.0
        elif train_shingles:
            candidate_counts: Counter[int] = Counter()
            query_shingles = [
                shingle for shingle in train_shingles if shingle in eval_shingle_index
            ]
            query_shingles.sort(
                key=lambda shingle: (
                    len(eval_shingle_index.get(shingle, ())),
                    -int(str(shingle).split(":", 1)[0]) if ":" in str(shingle) else 0,
                    str(shingle),
                )
            )
            if max_query_features is not None and max_query_features > 0:
                original_count = len(query_shingles)
                query_shingles = query_shingles[: int(max_query_features)]
                skipped_query_features += max(0, original_count - len(query_shingles))
            for shingle in query_shingles:
                candidate_counts.update(eval_shingle_index.get(shingle, set()))
            candidate_pairs_scored += len(candidate_counts)
            candidate_items = candidate_counts.most_common()
            if max_candidates_per_row is not None and max_candidates_per_row > 0:
                if len(candidate_items) > int(max_candidates_per_row):
                    truncated_candidate_rows += 1
                candidate_items = candidate_items[: int(max_candidates_per_row)]
            for eval_idx, overlap in candidate_items:
                eval_shingles = eval_entries[eval_idx]["shingles"]
                union = len(train_shingles | eval_shingles)
                if union <= 0:
                    continue
                jaccard = float(overlap) / float(union)
                min_size = min(len(train_shingles), len(eval_shingles))
                containment = float(overlap) / float(min_size) if min_size > 0 else 0.0
                candidate_score = max(jaccard, containment)
                if candidate_score >= float(threshold):
                    matched_idx = eval_idx
                    score = candidate_score
                    break
        if matched_idx is None:
            continue
        near_hits += 1
        eval_entry = eval_entries[matched_idx]
        matched_train_rows.append(
            _row_match_reference(
                row,
                eval_id=str(eval_entry["id"]),
                score=score,
                match_kind="exact" if score == 1.0 else "near_duplicate",
            )
        )
        if len(examples) < int(max_examples):
            examples.append(
                {
                    "train_id": train_id,
                    "eval_id": eval_entry["id"],
                    "score": round(float(score), 6),
                    "train_text": train_text[:240],
                    "eval_text": str(eval_entry["signature"])[:240],
                }
            )

    return {
        "eval_rows_indexed": len(eval_entries),
        "train_rows_checked": len(train_rows),
        "exact_signature_hits": int(exact_hits),
        "near_duplicate_hits": int(near_hits),
        "threshold": float(threshold),
        "near_duplicate_feature_family": "token_unigram_bigram_trigram_containment",
        "near_duplicate_max_chars": None if max_chars is None else int(max_chars),
        "near_duplicate_max_shingle_postings": (
            None if max_shingle_postings is None else int(max_shingle_postings)
        ),
        "near_duplicate_max_query_features": (
            None if max_query_features is None else int(max_query_features)
        ),
        "near_duplicate_max_candidates_per_row": (
            None if max_candidates_per_row is None else int(max_candidates_per_row)
        ),
        "indexed_eval_shingles": int(len(eval_shingle_index)),
        "skipped_high_posting_shingles": int(skipped_high_posting_shingles),
        "skipped_query_features": int(skipped_query_features),
        "truncated_candidate_rows": int(truncated_candidate_rows),
        "candidate_pairs_scored": int(candidate_pairs_scored),
        "matched_train_row_count": len(matched_train_rows),
        "matched_train_rows_sha256": hashlib.sha256(
            json.dumps(
                matched_train_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "matched_train_rows": matched_train_rows,
        "examples": examples,
    }


def _data_composition(data_root: Path) -> dict[str, Any]:
    trajectories_by_benchmark: Counter[str] = Counter()
    retrieval_by_source: Counter[str] = Counter()
    skill_pool_by_source: Counter[str] = Counter()
    rows = {
        "trajectories_rows": 0,
        "retrieval_rows": 0,
        "skill_pool_rows": 0,
    }

    for row in _iter_jsonl(data_root / "trajectories.jsonl"):
        rows["trajectories_rows"] += 1
        trajectories_by_benchmark[str(row.get("benchmark") or "<missing>")] += 1
    for row in _iter_jsonl(data_root / "retrieval.jsonl"):
        rows["retrieval_rows"] += 1
        retrieval_by_source[str(row.get("source") or "<missing>")] += 1
    for row in _iter_jsonl(data_root / "skill_pool.jsonl"):
        rows["skill_pool_rows"] += 1
        skill_pool_by_source[_skill_source(row)] += 1

    return {
        "data_root": str(data_root),
        "files_sha256": {
            "trajectories.jsonl": _file_sha256(data_root / "trajectories.jsonl"),
            "retrieval.jsonl": _file_sha256(data_root / "retrieval.jsonl"),
            "skill_pool.jsonl": _file_sha256(data_root / "skill_pool.jsonl"),
        },
        "files": rows,
        "trajectories_by_benchmark": dict(trajectories_by_benchmark.most_common()),
        "retrieval_by_source": dict(retrieval_by_source.most_common()),
        "skill_pool_by_source": dict(skill_pool_by_source.most_common()),
    }


def _toolbench_eval_hits(
    data_root: Path,
    toolbench_eval_trajectories_path: str | Path,
    *,
    near_duplicate_threshold: float = 0.55,
    near_duplicate_max_chars: int | None = 1000,
    near_duplicate_max_shingle_postings: int | None = 256,
    near_duplicate_max_query_features: int | None = 64,
    near_duplicate_max_candidates_per_row: int | None = 128,
) -> dict[str, Any]:
    exclusion = build_toolbench_eval_exclusion(toolbench_eval_trajectories_path)
    eval_rows = [row for row in _iter_jsonl(toolbench_eval_trajectories_path) if isinstance(row, dict)]
    query_ids = set(exclusion.query_ids)
    trajectory_hits: Counter[str] = Counter()
    retrieval_hits: Counter[str] = Counter()
    trajectory_rows_for_content: list[dict[str, Any]] = []
    retrieval_rows_for_content: list[dict[str, Any]] = []
    toolbench_trajectory_rows = 0
    toolbench_retrieval_rows = 0

    for row in _iter_jsonl(data_root / "trajectories.jsonl"):
        if not _is_toolbench_trajectory_row(row):
            continue
        toolbench_trajectory_rows += 1
        trajectory_rows_for_content.append(row)
        if str(row.get("task_id") or "").strip() in exclusion.task_ids:
            trajectory_hits["task_id_hits"] += 1
        if str(row.get("trajectory_id") or "").strip() in exclusion.trajectory_ids:
            trajectory_hits["trajectory_id_hits"] += 1
        if _row_answer_path(row) in exclusion.answer_paths:
            trajectory_hits["answer_path_hits"] += 1
        if _row_query_ids(row) & query_ids:
            trajectory_hits["query_id_hits"] += 1
        if _row_signature(row) in exclusion.signatures:
            trajectory_hits["signature_hits"] += 1

    for row in _iter_jsonl(data_root / "retrieval.jsonl"):
        if not _is_toolbench_retrieval_row(row):
            continue
        toolbench_retrieval_rows += 1
        retrieval_rows_for_content.append(row)
        provenance = _provenance(row)
        if str(provenance.get("task_id") or "").strip() in exclusion.task_ids:
            retrieval_hits["task_id_hits"] += 1
        if str(provenance.get("trajectory_id") or "").strip() in exclusion.trajectory_ids:
            retrieval_hits["trajectory_id_hits"] += 1
        if _row_answer_path(row) in exclusion.answer_paths:
            retrieval_hits["answer_path_hits"] += 1
        if _row_query_ids(row) & query_ids:
            retrieval_hits["query_id_hits"] += 1

    return {
        "eval_trajectories_path": str(toolbench_eval_trajectories_path),
        "exclusion": exclusion.report(),
        "toolbench_trajectory_rows": toolbench_trajectory_rows,
        "trajectory_hits": dict(trajectory_hits),
        "trajectory_content_overlap": _content_overlap_report(
            eval_rows=eval_rows,
            train_rows=trajectory_rows_for_content,
            threshold=near_duplicate_threshold,
            max_chars=near_duplicate_max_chars,
            max_shingle_postings=near_duplicate_max_shingle_postings,
            max_query_features=near_duplicate_max_query_features,
            max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
        "toolbench_retrieval_rows": toolbench_retrieval_rows,
        "retrieval_hits": dict(retrieval_hits),
        "retrieval_content_overlap": _content_overlap_report(
            eval_rows=eval_rows,
            train_rows=retrieval_rows_for_content,
            threshold=near_duplicate_threshold,
            max_chars=near_duplicate_max_chars,
            max_shingle_postings=near_duplicate_max_shingle_postings,
            max_query_features=near_duplicate_max_query_features,
            max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
    }


def _trajectbench_eval_hits(
    data_root: Path,
    traject_eval_queries_path: str | Path,
    *,
    near_duplicate_threshold: float = 0.55,
    near_duplicate_max_chars: int | None = 1000,
    near_duplicate_max_shingle_postings: int | None = 256,
    near_duplicate_max_query_features: int | None = 64,
    near_duplicate_max_candidates_per_row: int | None = 128,
) -> dict[str, Any]:
    queries_path = Path(traject_eval_queries_path)
    qrels_path = queries_path.with_name("qrels.jsonl")
    eval_query_ids: set[str] = set()
    eval_trajectory_ids: set[str] = set()
    query_text_by_id: dict[str, str] = {}
    positive_skill_ids_by_query: dict[str, set[str]] = {}
    signature_set: set[tuple[str, str]] = set()
    eval_rows_for_content: list[dict[str, Any]] = []

    for row in _iter_jsonl(queries_path):
        query_id = str(row.get("query_id") or "").strip()
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if query_id:
            eval_query_ids.add(query_id)
            query_text_by_id[query_id] = str(row.get("query_text") or "")
            eval_rows_for_content.append(row)
        if trajectory_id:
            eval_trajectory_ids.add(trajectory_id)

    if qrels_path.is_file():
        for row in _iter_jsonl(qrels_path):
            if int(row.get("relevance", 1) or 0) <= 0:
                continue
            query_id = str(row.get("query_id") or "").strip()
            skill_id = str(row.get("skill_id") or row.get("positive_skill_id") or "").strip()
            if query_id and skill_id:
                positive_skill_ids_by_query.setdefault(query_id, set()).add(skill_id)

    for query_id, skill_ids in positive_skill_ids_by_query.items():
        query_text = query_text_by_id.get(query_id)
        if query_text is None:
            continue
        for skill_id in skill_ids:
            signature_set.add((query_text, skill_id))

    hits: Counter[str] = Counter()
    traject_trajectory_rows = 0
    traject_rows_for_content: list[dict[str, Any]] = []
    for row in _iter_jsonl(data_root / "trajectories.jsonl"):
        if str(row.get("benchmark") or "") != "traject_bench":
            continue
        traject_trajectory_rows += 1
        traject_rows_for_content.append(row)
        if str(row.get("task_id") or "").strip() in eval_query_ids:
            hits["task_id_hits"] += 1
        if str(row.get("trajectory_id") or "").strip() in eval_trajectory_ids:
            hits["trajectory_id_hits"] += 1
        state_text = str(row.get("state_text_full") or row.get("state_text") or "")
        skill_id = str(row.get("skill_id") or "")
        if (state_text, skill_id) in signature_set:
            hits["signature_hits"] += 1

    return {
        "eval_queries_path": str(queries_path),
        "eval_qrels_path": str(qrels_path) if qrels_path.is_file() else None,
        "eval_query_ids": len(eval_query_ids),
        "eval_trajectory_ids": len(eval_trajectory_ids),
        "eval_positive_signatures": len(signature_set),
        "traject_trajectory_rows": traject_trajectory_rows,
        "trajectory_hits": dict(hits),
        "trajectory_content_overlap": _content_overlap_report(
            eval_rows=eval_rows_for_content,
            train_rows=traject_rows_for_content,
            threshold=near_duplicate_threshold,
            max_chars=near_duplicate_max_chars,
            max_shingle_postings=near_duplicate_max_shingle_postings,
            max_query_features=near_duplicate_max_query_features,
            max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
    }


def _matched_benchmark_eval_hits(
    data_root: Path,
    eval_rows_path: str | Path,
    *,
    benchmark: str,
    near_duplicate_threshold: float,
    near_duplicate_max_chars: int | None,
    near_duplicate_max_shingle_postings: int | None,
    near_duplicate_max_query_features: int | None,
    near_duplicate_max_candidates_per_row: int | None,
) -> dict[str, Any]:
    eval_rows = list(_iter_jsonl(eval_rows_path))
    train_rows = [
        row
        for row in _iter_jsonl(data_root / "trajectories.jsonl")
        if str(row.get("benchmark") or "").strip().lower() == benchmark
    ]
    eval_groups = {
        str(row.get("matched_split_group_identity") or "").strip()
        for row in eval_rows
        if str(row.get("matched_split_group_identity") or "").strip()
    }
    train_groups = {
        str(
            row.get("locked_split_group_identity")
            or row.get("matched_split_group_identity")
            or ""
        ).strip()
        for row in train_rows
        if str(
            row.get("locked_split_group_identity")
            or row.get("matched_split_group_identity")
            or ""
        ).strip()
    }
    eval_trajectories = {
        str(row.get("trajectory_id") or "").strip()
        for row in eval_rows
        if str(row.get("trajectory_id") or "").strip()
    }
    train_trajectories = {
        str(row.get("trajectory_id") or "").strip()
        for row in train_rows
        if str(row.get("trajectory_id") or "").strip()
    }
    group_overlap = sorted(eval_groups & train_groups)
    trajectory_overlap = sorted(eval_trajectories & train_trajectories)
    return {
        "benchmark": benchmark,
        "eval_rows_path": str(Path(eval_rows_path).resolve()),
        "eval_row_count": len(eval_rows),
        "train_row_count": len(train_rows),
        "eval_group_count": len(eval_groups),
        "train_group_count": len(train_groups),
        "group_overlap_count": len(group_overlap),
        "group_overlap_examples": group_overlap[:10],
        "trajectory_overlap_count": len(trajectory_overlap),
        "trajectory_overlap_examples": trajectory_overlap[:10],
        "content_overlap": _content_overlap_report(
            eval_rows=eval_rows,
            train_rows=train_rows,
            threshold=near_duplicate_threshold,
            max_chars=near_duplicate_max_chars,
            max_shingle_postings=near_duplicate_max_shingle_postings,
            max_query_features=near_duplicate_max_query_features,
            max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
    }


def audit_incremental_matched_union_preflight(
    *,
    data_root: str | Path,
    base_data_root: str | Path,
    base_clean_preflight_report: str | Path,
    matched_union_manifest_path: str | Path,
    toolbench_eval_trajectories_path: str | Path,
    traject_eval_queries_path: str | Path | None,
    tau2_test_rows_path: str | Path,
    toolsandbox_test_rows_path: str | Path,
    output_path: str | Path | None = None,
    near_duplicate_threshold: float = 0.55,
    near_duplicate_max_chars: int | None = 1000,
    near_duplicate_max_shingle_postings: int | None = 256,
    near_duplicate_max_query_features: int | None = 64,
    near_duplicate_max_candidates_per_row: int | None = 128,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    base_root = Path(base_data_root).resolve()
    base_report_path = Path(base_clean_preflight_report).resolve()
    union_manifest_path = Path(matched_union_manifest_path).resolve()
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_contract = base_report.get("preflight_contract") or {}
    if (
        base_report.get("status") != "ok"
        or not bool(base_contract.get("structured_current_state_required"))
        or not bool(base_contract.get("protected_near_duplicate_blocking"))
    ):
        raise ValueError("incremental preflight requires a strict clean-base report")
    base_composition = base_report.get("composition") or {}
    if Path(str(base_composition.get("data_root") or "")).resolve() != base_root:
        raise ValueError("clean-base preflight root mismatch")
    for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl"):
        expected = str((base_composition.get("files_sha256") or {}).get(name) or "")
        if not expected or _file_sha256(base_root / name) != expected:
            raise ValueError(f"clean-base file changed after preflight: {name}")

    union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    unsigned_manifest = dict(union_manifest)
    observed_manifest_digest = str(unsigned_manifest.pop("manifest_sha256", ""))
    if (
        union_manifest.get("status") != "ok"
        or union_manifest.get("schema_version") != "clstr_matched_multibench_union_v1"
        or observed_manifest_digest != _canonical_digest(unsigned_manifest)
    ):
        raise ValueError("matched union manifest is invalid")
    if Path(str(union_manifest.get("base_data_root") or "")).resolve() != base_root:
        raise ValueError("matched union manifest names the wrong clean base")
    if union_manifest.get("base_files_sha256") != base_composition.get("files_sha256"):
        raise ValueError("matched union clean-base digest contract mismatch")
    tau2_split_manifest = union_manifest.get("tau2_split_manifest") or {}
    validate_tau2_split_manifest(tau2_split_manifest)
    if not bool(tau2_split_manifest.get("official_test_preserved")):
        raise ValueError("matched union does not preserve the official Tau2 test split")
    for name, entry in (union_manifest.get("output_files") or {}).items():
        path = Path(str(entry.get("path") or "")).resolve()
        if path != data_root / name or _file_sha256(path) != str(entry.get("sha256") or ""):
            raise ValueError(f"matched union output digest mismatch: {name}")
    if _file_sha256(data_root / "retrieval.jsonl") != _file_sha256(
        base_root / "retrieval.jsonl"
    ):
        raise ValueError("matched union retrieval stream is not the clean-base copy")

    base_skill_count, appended_skills = _verify_semantic_jsonl_prefix(
        base_root / "skill_pool.jsonl", data_root / "skill_pool.jsonl"
    )
    appended_trajectories = _verify_exact_file_prefix(
        base_root / "trajectories.jsonl", data_root / "trajectories.jsonl"
    )
    counts = union_manifest.get("counts") or {}
    if base_skill_count != int(counts.get("base_skill_count", -1)):
        raise ValueError("matched union base skill count does not reproduce")
    if len(appended_skills) != int(counts.get("appended_skill_count", -1)):
        raise ValueError("matched union appended skill count does not reproduce")
    if len(appended_trajectories) != int(
        counts.get("appended_training_trajectory_rows", -1)
    ):
        raise ValueError("matched union appended trajectory count does not reproduce")
    if any(
        not str(row.get("skill_id") or "").startswith(("tau2/", "toolsandbox/"))
        for row in appended_skills
    ):
        raise ValueError("matched union appends a skill outside Tau2/ToolSandbox")
    appended_blockers: Counter[str] = Counter()
    for row in appended_trajectories:
        benchmark = str(row.get("benchmark") or "")
        state = str(row.get("state_text_current") or "").lower()
        appended_blockers["unexpected_benchmark"] += int(
            benchmark not in {"tau2", "toolsandbox"}
        )
        appended_blockers["invalid_split"] += int(
            row.get("locked_data_split") not in {"train", "dev"}
        )
        appended_blockers["missing_group"] += int(
            not str(row.get("locked_split_group_identity") or "")
        )
        appended_blockers["history_in_h_t"] += int("history:" in state)
        appended_blockers["source_label_in_h_t"] += int(
            "benchmark:" in state or "scenario:" in state or "scenario_group:" in state
        )
        provenance = row.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        result_text = str(row.get("next_observation_text") or "").strip()
        observation_source = str(row.get("observation_source") or "")
        source_id = str(provenance.get("source_id") or "")
        if benchmark == "tau2":
            appended_blockers["invalid_tau2_successful_result_contract"] += int(
                not result_text
                or not actual_causal_observation(row)
                or observation_source
                != "tau2_official_successful_rollout_tool_result"
                or source_id != "tau2_official_successful_rollout_v1"
            )
        elif benchmark == "toolsandbox":
            appended_blockers["invalid_toolsandbox_action_only_contract"] += int(
                bool(result_text)
                or observation_source != "action_only_no_tool_result"
                or source_id != "matched_toolsandbox_action_only_v1"
            )
    appended_history = audit_history_channel_rows(
        _structured_history_audit_rows(appended_trajectories),
        require_explicit_current=True,
        require_actual_replay_observation=False,
        require_structured_current=True,
    )

    matched_reports = {
        "tau2_test": _matched_benchmark_eval_hits(
            data_root,
            tau2_test_rows_path,
            benchmark="tau2",
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_max_chars=near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
        "toolsandbox_test": _matched_benchmark_eval_hits(
            data_root,
            toolsandbox_test_rows_path,
            benchmark="toolsandbox",
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_max_chars=near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
        ),
    }
    exact_leakage = any(
        int(value.get("group_overlap_count") or 0)
        or int(value.get("trajectory_overlap_count") or 0)
        for value in matched_reports.values()
    )
    observed_near_duplicate = any(
        int((value.get("content_overlap") or {}).get("near_duplicate_hits") or 0)
        for value in matched_reports.values()
    )
    # Tau2 publishes explicit train/test task identities.  Telecom deliberately
    # reuses the same user-facing issue prompt across different hidden fault
    # configurations, so prompt equality alone is not task leakage.  Preserve
    # and validate the official split, disclose prompt overlap, and continue to
    # block exact group/trajectory overlap.  ToolSandbox has no such official
    # split, so its family-held-out content overlap remains blocking.
    matched_reports["tau2_test"]["content_overlap_policy"] = {
        "blocking": False,
        "basis": "official_tau2_train_test_task_split",
        "prompt_overlap_disclosed": True,
    }
    matched_reports["toolsandbox_test"]["content_overlap_policy"] = {
        "blocking": True,
        "basis": "held_out_template_family_split",
        "prompt_overlap_disclosed": True,
    }
    blocking_near_duplicate = bool(
        int(
            (
                matched_reports["toolsandbox_test"].get("content_overlap") or {}
            ).get("near_duplicate_hits")
            or 0
        )
    )
    base_protected = base_report.get("protected_eval_inputs") or {}
    expected_toolbench = str(base_protected.get("toolbench_eval_trajectories_sha256") or "")
    if _file_sha256(toolbench_eval_trajectories_path) != expected_toolbench:
        raise ValueError("ToolBench protected input differs from clean-base preflight")
    if traject_eval_queries_path is not None:
        expected_traject = str(base_protected.get("traject_eval_queries_sha256") or "")
        if not expected_traject or _file_sha256(traject_eval_queries_path) != expected_traject:
            raise ValueError("TrajectBench protected input differs from clean-base preflight")
    blockers = {
        key: value for key, value in appended_blockers.items() if int(value) > 0
    }
    history_blocker = appended_history.get("status") != "ok"
    status = (
        "ok"
        if not blockers
        and not history_blocker
        and not exact_leakage
        and not blocking_near_duplicate
        else "error"
    )
    composition = _data_composition(data_root)
    protected_inputs = {
        **base_protected,
        "tau2_test_rows_path": str(Path(tau2_test_rows_path).resolve()),
        "tau2_test_rows_sha256": _file_sha256(tau2_test_rows_path),
        "toolsandbox_test_rows_path": str(Path(toolsandbox_test_rows_path).resolve()),
        "toolsandbox_test_rows_sha256": _file_sha256(toolsandbox_test_rows_path),
        "matched_union_manifest_path": str(union_manifest_path),
        "matched_union_manifest_sha256": _file_sha256(union_manifest_path),
    }
    report = {
        "status": status,
        "preflight_contract": {
            "schema_version": "clstr_clean_training_preflight_v3",
            "structured_current_state_required": True,
            "protected_near_duplicate_blocking": True,
            "incremental_matched_union_verified": True,
        },
        "leakage_policy": {
            "exact_leakage_blocks_training": True,
            "near_duplicate_blocks_training": True,
            "has_exact_leakage": bool(exact_leakage),
            "has_near_duplicate": bool(observed_near_duplicate),
            "has_blocking_near_duplicate": bool(blocking_near_duplicate),
            "tau2_official_split_prompt_overlap_is_disclosed_not_blocking": True,
            "history_channel_blocks_training": True,
            "has_history_channel_blocker": bool(history_blocker),
        },
        "history_channel": {
            "clean_base": base_report.get("history_channel") or {},
            "matched_appended_trajectory_stream": appended_history,
        },
        "composition": composition,
        "leakage": {
            "toolbench_eval": (base_report.get("leakage") or {}).get("toolbench_eval"),
            "trajectbench_eval": (base_report.get("leakage") or {}).get("trajectbench_eval"),
            **matched_reports,
        },
        "incremental_union_audit": {
            "base_clean_preflight_report": str(base_report_path),
            "base_clean_preflight_sha256": _file_sha256(base_report_path),
            "matched_union_manifest": str(union_manifest_path),
            "matched_union_logical_digest": observed_manifest_digest,
            "base_skill_count": base_skill_count,
            "appended_skill_count": len(appended_skills),
            "appended_trajectory_count": len(appended_trajectories),
            "appended_blockers": blockers,
        },
        "protected_eval_inputs": protected_inputs,
        "paper_scope_note": {
            "toolbench": "reuses exact digest-bound clean-base audit",
            "trajectbench": "reuses exact digest-bound clean-base audit",
            "tau2": "official test protected from official-train-derived union",
            "toolsandbox": "template-family test protected from grouped train/dev union",
        },
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def audit_clean_training_preflight(
    *,
    data_root: str | Path,
    toolbench_eval_trajectories_path: str | Path,
    traject_eval_queries_path: str | Path | None = None,
    output_path: str | Path | None = None,
    near_duplicate_threshold: float = 0.55,
    near_duplicate_max_chars: int | None = 1000,
    near_duplicate_max_shingle_postings: int | None = 256,
    near_duplicate_max_query_features: int | None = 64,
    near_duplicate_max_candidates_per_row: int | None = 128,
    fail_on_near_duplicate: bool = True,
    require_structured_current_state: bool = False,
    tau2_test_rows_path: str | Path | None = None,
    toolsandbox_test_rows_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    required = ["trajectories.jsonl", "retrieval.jsonl", "skill_pool.jsonl"]
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing unified training files under {data_root}: {missing}")

    composition = _data_composition(data_root)
    toolbench_eval = _toolbench_eval_hits(
        data_root,
        toolbench_eval_trajectories_path,
        near_duplicate_threshold=near_duplicate_threshold,
        near_duplicate_max_chars=near_duplicate_max_chars,
        near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
        near_duplicate_max_query_features=near_duplicate_max_query_features,
        near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
    )
    trajectbench_eval = (
        _trajectbench_eval_hits(
            data_root,
            traject_eval_queries_path,
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_max_chars=near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
        )
        if traject_eval_queries_path is not None and Path(traject_eval_queries_path).is_file()
        else None
    )
    tau2_test = (
        _matched_benchmark_eval_hits(
            data_root,
            tau2_test_rows_path,
            benchmark="tau2",
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_max_chars=near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
        )
        if tau2_test_rows_path is not None and Path(tau2_test_rows_path).is_file()
        else None
    )
    toolsandbox_test = (
        _matched_benchmark_eval_hits(
            data_root,
            toolsandbox_test_rows_path,
            benchmark="toolsandbox",
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_max_chars=near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=near_duplicate_max_candidates_per_row,
        )
        if toolsandbox_test_rows_path is not None
        and Path(toolsandbox_test_rows_path).is_file()
        else None
    )
    has_toolbench_exact_leakage = bool(
        toolbench_eval["trajectory_hits"]
        or toolbench_eval["retrieval_hits"]
    )
    has_toolbench_near_duplicate = bool(
        int((toolbench_eval.get("trajectory_content_overlap") or {}).get("near_duplicate_hits") or 0) > 0
        or int((toolbench_eval.get("retrieval_content_overlap") or {}).get("near_duplicate_hits") or 0) > 0
    )
    has_trajectbench_exact_leakage = bool(
        trajectbench_eval is not None
        and trajectbench_eval.get("trajectory_hits")
    )
    has_trajectbench_near_duplicate = bool(
        trajectbench_eval is not None
        and int((trajectbench_eval.get("trajectory_content_overlap") or {}).get("near_duplicate_hits") or 0) > 0
    )
    matched_reports = [
        report for report in (tau2_test, toolsandbox_test) if report is not None
    ]
    has_matched_exact_leakage = any(
        int(report.get("group_overlap_count") or 0) > 0
        or int(report.get("trajectory_overlap_count") or 0) > 0
        for report in matched_reports
    )
    has_matched_near_duplicate = any(
        int((report.get("content_overlap") or {}).get("near_duplicate_hits") or 0)
        > 0
        for report in matched_reports
    )
    has_exact_leakage = (
        has_toolbench_exact_leakage
        or has_trajectbench_exact_leakage
        or has_matched_exact_leakage
    )
    has_near_duplicate = (
        has_toolbench_near_duplicate
        or has_trajectbench_near_duplicate
        or has_matched_near_duplicate
    )
    if require_structured_current_state:
        trajectory_audit_rows = _structured_history_audit_rows(
            _iter_jsonl(data_root / "trajectories.jsonl")
        )
        retrieval_audit_rows = _structured_retrieval_audit_rows(
            _iter_jsonl(data_root / "retrieval.jsonl")
        )
    else:
        trajectory_audit_rows = _iter_jsonl(data_root / "trajectories.jsonl")
        retrieval_audit_rows = (
            {
                "state_text_current": str(row.get("query_text") or row.get("query") or ""),
                "step_index": 1,
            }
            for row in _iter_jsonl(data_root / "retrieval.jsonl")
        )
    history_channel = {
        "trajectory_stream": audit_history_channel_rows(
            trajectory_audit_rows,
            require_explicit_current=True,
            require_actual_replay_observation=False,
            require_structured_current=require_structured_current_state,
        ),
        "retrieval_stream": audit_history_channel_rows(
            retrieval_audit_rows,
            require_explicit_current=True,
            require_structured_current=require_structured_current_state,
        ),
    }
    history_channel_blocks = any(
        value["status"] != "ok" for value in history_channel.values()
    )
    blocks_training = bool(
        has_exact_leakage
        or (fail_on_near_duplicate and has_near_duplicate)
        or history_channel_blocks
    )
    if blocks_training:
        status = "error"
    elif has_near_duplicate:
        status = "warning"
    else:
        status = "ok"
    leakage_policy = {
        "exact_leakage_blocks_training": True,
        "near_duplicate_blocks_training": bool(fail_on_near_duplicate),
        "has_exact_leakage": bool(has_exact_leakage),
        "has_near_duplicate": bool(has_near_duplicate),
        "history_channel_blocks_training": True,
        "has_history_channel_blocker": bool(history_channel_blocks),
    }
    report = {
        "status": status,
        "preflight_contract": {
            "schema_version": "clstr_clean_training_preflight_v3",
            "structured_current_state_required": bool(
                require_structured_current_state
            ),
            "protected_near_duplicate_blocking": bool(fail_on_near_duplicate),
        },
        "leakage_policy": leakage_policy,
        "history_channel": history_channel,
        "composition": composition,
        "leakage": {
            "toolbench_eval": toolbench_eval,
        },
        "protected_eval_inputs": {
            "toolbench_eval_trajectories_path": str(
                Path(toolbench_eval_trajectories_path).resolve()
            ),
            "toolbench_eval_trajectories_sha256": _file_sha256(
                toolbench_eval_trajectories_path
            ),
            "traject_eval_queries_path": (
                str(Path(traject_eval_queries_path).resolve())
                if traject_eval_queries_path is not None
                and Path(traject_eval_queries_path).is_file()
                else None
            ),
            "traject_eval_queries_sha256": (
                _file_sha256(traject_eval_queries_path)
                if traject_eval_queries_path is not None
                and Path(traject_eval_queries_path).is_file()
                else None
            ),
            "tau2_test_rows_path": (
                str(Path(tau2_test_rows_path).resolve())
                if tau2_test_rows_path is not None
                and Path(tau2_test_rows_path).is_file()
                else None
            ),
            "tau2_test_rows_sha256": (
                _file_sha256(tau2_test_rows_path)
                if tau2_test_rows_path is not None
                and Path(tau2_test_rows_path).is_file()
                else None
            ),
            "toolsandbox_test_rows_path": (
                str(Path(toolsandbox_test_rows_path).resolve())
                if toolsandbox_test_rows_path is not None
                and Path(toolsandbox_test_rows_path).is_file()
                else None
            ),
            "toolsandbox_test_rows_sha256": (
                _file_sha256(toolsandbox_test_rows_path)
                if toolsandbox_test_rows_path is not None
                and Path(toolsandbox_test_rows_path).is_file()
                else None
            ),
        },
        "paper_scope_note": {
            "toolbench": "clean only when trajectory_hits and retrieval_hits are empty",
            "trajectbench": "requires separate split audit before held-out claims",
            "bfcl_apibank": "diagnostic if same-benchmark schema rows are used for training",
        },
    }
    if trajectbench_eval is not None:
        report["leakage"]["trajectbench_eval"] = trajectbench_eval
    if tau2_test is not None:
        report["leakage"]["tau2_test"] = tau2_test
    if toolsandbox_test is not None:
        report["leakage"]["toolsandbox_test"] = toolsandbox_test
    if output_path is not None:
        _write_json(output_path, report)
    return report
