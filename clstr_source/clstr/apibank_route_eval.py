from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from clstr.full_base_train import UNIFIED_MEMORY_ROUTE_SCORER

from clstr.bfcl_route_eval import (
    _mean,
    _metrics_from_ranks,
    _numeric_delta,
    _rank_candidates_with_stage0_prior,
    _stage0_prior_eval_from_rows,
    _strict_stage4_metrics,
)
from clstr.logged_online_stage4_train import (
    attach_trajectory_prefix_online_memory_scores,
    evaluate_logged_online_stage4_rows,
    evaluate_logged_online_stage4_rows_by_benchmark,
)
from clstr.memory_candidate_recall import (
    candidate_recall_protocol_metadata,
    declared_candidate_pool_size,
    source_rows_have_causal_sequence,
)
from clstr.stage4_act_train import STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
from clstr.tau2_skillrouter_eval import _apply_skillrouter_adapter


DEFAULT_APIBANK_ROUTE_FILES = [
    "test-data/level-1-api.json",
    "test-data/level-2-api.json",
    "test-data/level-3-batch-inf.json",
]


@dataclass(frozen=True)
class APIBankRouteCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


def apibank_start_skill_id() -> str:
    return "apibank/__start__"


def apibank_api_skill_id(name: str) -> str:
    safe = str(name or "").strip().replace("/", "_")
    return f"apibank/{safe}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _name_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def parse_apibank_call_names(text: Any) -> list[str]:
    inner = _first_bracket_content(str(text or ""))
    names: list[str] = []
    index = 0
    while index < len(inner):
        while index < len(inner) and inner[index] in " \t\r\n,":
            index += 1
        start = index
        if index < len(inner) and (inner[index].isalpha() or inner[index] == "_"):
            index += 1
            while index < len(inner) and (inner[index].isalnum() or inner[index] == "_"):
                index += 1
            name = inner[start:index]
            while index < len(inner) and inner[index].isspace():
                index += 1
            if index < len(inner) and inner[index] == "(":
                names.append(name)
                index = _skip_balanced_parens(inner, index)
                continue
        index = max(index + 1, start + 1)
    return names


def _first_bracket_content(text: str) -> str:
    start = text.find("[")
    if start < 0:
        return ""
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return text[start + 1 :]


def _skip_balanced_parens(text: str, start: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        else:
            if char in {"'", '"'}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index + 1
        index += 1
    return index


def _balanced_dict_strings(text: str) -> list[str]:
    strings: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        start = index
        depth = 0
        quote: str | None = None
        escaped = False
        while index < len(text):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            else:
                if char in {"'", '"'}:
                    quote = char
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        strings.append(text[start : index + 1])
                        index += 1
                        break
            index += 1
        else:
            break
    return strings


def _parse_dict_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for candidate in _balanced_dict_strings(text):
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
            except Exception:
                continue
            if isinstance(value, dict):
                objects.append(value)
            break
    return objects


def _api_doc_name(doc: dict[str, Any]) -> str:
    for key in ("apiCode", "name", "api_name", "function"):
        value = doc.get(key)
        if value:
            return str(value).strip()
    return ""


def _api_doc_description(doc: dict[str, Any]) -> str:
    return str(doc.get("description") or doc.get("desc") or doc.get("documentation") or "").strip()


def _api_doc_parameters(doc: dict[str, Any]) -> Any:
    return doc.get("parameters") or doc.get("input_parameters") or doc.get("inputSchema") or {}


def _api_doc_response(doc: dict[str, Any]) -> Any:
    return doc.get("response") or doc.get("output_parameters") or doc.get("output_schema") or {}


def _iter_api_docs(obj: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if _api_doc_name(obj):
        yield obj
    for key in ("output", "response", "best_matchs", "best_matches"):
        value = obj.get(key)
        if isinstance(value, dict):
            yield from _iter_api_docs(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _iter_api_docs(item)


def _visible_api_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = "\n".join(str(row.get(key) or "") for key in ("instruction", "input"))
    docs_by_key: dict[str, dict[str, Any]] = {}
    for obj in _parse_dict_objects(text):
        for doc in _iter_api_docs(obj):
            name = _api_doc_name(doc)
            key = _name_key(name)
            if not key or key in docs_by_key:
                continue
            docs_by_key[key] = {
                "name": name,
                "description": _api_doc_description(doc),
                "parameters": _api_doc_parameters(doc),
                "response": _api_doc_response(doc),
            }
    return list(docs_by_key.values())


def _skill_from_api_doc(doc: dict[str, Any]) -> dict[str, Any]:
    name = str(doc.get("name") or "").strip()
    description = str(doc.get("description") or f"APIBank API {name}").strip()
    parameters = doc.get("parameters") if isinstance(doc.get("parameters"), dict) else {}
    response = doc.get("response") if isinstance(doc.get("response"), dict) else {}
    body = (
        f"API name: {name}\n"
        f"Benchmark: APIBank\n"
        f"Description: {description}\n"
        f"Input parameters: {json.dumps(parameters, ensure_ascii=False, sort_keys=True)}\n"
        f"Output parameters: {json.dumps(response, ensure_ascii=False, sort_keys=True)}"
    )
    return {
        "skill_id": apibank_api_skill_id(name),
        "name": name,
        "description": description,
        "executor_desc": description,
        "body": body,
        "skill_md": body,
        "source_benchmark": "apibank",
        "api_name": name,
    }


def _start_skill() -> dict[str, Any]:
    return {
        "skill_id": apibank_start_skill_id(),
        "name": "APIBank.START",
        "description": "Initial APIBank state before any API request.",
        "executor_desc": "Initial APIBank state before any API request.",
        "body": "No API request has been selected yet.",
        "skill_md": "No API request has been selected yet.",
        "source_benchmark": "apibank",
    }


def _previous_call_names(row: dict[str, Any]) -> list[str]:
    input_text = str(row.get("input") or "")
    names: list[str] = []
    for match in re.finditer(r"API-Request:\s*\[", input_text):
        names.extend(parse_apibank_call_names(input_text[match.start() :]))
    return names


def _state_text(*, file_name: str, row_id: str, row: dict[str, Any], docs: list[dict[str, Any]]) -> str:
    input_text = str(row.get("input") or "").strip()
    visible_docs = []
    for doc in docs:
        visible_docs.append(f"{doc.get('name')}: {doc.get('description')}")
    parts = [
        "benchmark: APIBank",
        f"file: {file_name}",
        f"row_id: {row_id}",
        "dialogue_and_api_context:",
        input_text,
    ]
    if visible_docs:
        parts.extend(["visible_api_evidence:", "\n".join(visible_docs)])
    return "\n".join(part for part in parts if part)


def _trajectory_and_step(file_name: str, row: dict[str, Any], row_id: str, fallback_idx: int) -> tuple[str, int]:
    if "level-3" in file_name and row.get("sample_id") is not None:
        step = int(row.get("api_id") if row.get("api_id") is not None else fallback_idx)
        return f"apibank/{file_name}/sample-{row.get('sample_id')}", step
    if "level-2" in file_name and row.get("file"):
        step = int(row.get("id") if row.get("id") is not None else fallback_idx)
        return f"apibank/{file_name}/{row.get('file')}", step
    return f"apibank/{file_name}/{row_id}", 0


def load_apibank_route_corpus(
    data_root: str | Path,
    *,
    files: Iterable[str] | None = None,
    prebuilt_source_rows_path: str | Path | None = None,
    prebuilt_skills_path: str | Path | None = None,
    include_trivial: bool = False,
    max_rows_per_file: int | None = None,
) -> APIBankRouteCorpus:
    data_root = Path(data_root)
    files = list(files or DEFAULT_APIBANK_ROUTE_FILES)
    skills: list[dict[str, Any]] = [_start_skill()]
    skill_ids_seen = {apibank_start_skill_id()}
    source_rows: list[dict[str, Any]] = []
    file_counts: dict[str, dict[str, Any]] = {}
    skipped = Counter()

    for file_name in files:
        path = data_root / file_name
        counts = {
            "raw_rows": 0,
            "source_rows": 0,
            "missing_file": 0,
            "row_not_dict": 0,
            "missing_gold_call_rows": 0,
            "missing_visible_api_docs_rows": 0,
            "missing_gt_candidate_rows": 0,
            "skipped_trivial_candidate_rows": 0,
        }
        if not path.exists():
            counts["missing_file"] = 1
            skipped["missing_file"] += 1
            file_counts[file_name] = counts
            continue
        raw_rows = _read_json(path)
        if not isinstance(raw_rows, list):
            skipped["file_not_list"] += 1
            file_counts[file_name] = counts
            continue
        selected_rows = raw_rows[: max(0, int(max_rows_per_file))] if max_rows_per_file is not None else raw_rows
        counts["raw_rows"] = len(selected_rows)
        for fallback_idx, row in enumerate(selected_rows):
            if not isinstance(row, dict):
                counts["row_not_dict"] += 1
                continue
            gold_names = parse_apibank_call_names(row.get("expected_output") or row.get("output"))
            if not gold_names:
                counts["missing_gold_call_rows"] += 1
                continue
            docs = _visible_api_docs(row)
            if not docs:
                counts["missing_visible_api_docs_rows"] += 1
                continue
            doc_by_key = {_name_key(doc.get("name")): doc for doc in docs}
            candidate_docs = list(docs)
            candidate_ids = [apibank_api_skill_id(str(doc.get("name") or "")) for doc in candidate_docs]
            if any(_name_key(gold_name) not in doc_by_key for gold_name in gold_names):
                counts["missing_gt_candidate_rows"] += 1
                continue
            if len(candidate_ids) <= 1 and not include_trivial:
                counts["skipped_trivial_candidate_rows"] += 1
                continue
            row_id = str(row.get("id") if row.get("id") is not None else row.get("sample_id") if row.get("sample_id") is not None else fallback_idx)
            previous_names = _previous_call_names(row)
            current_name = previous_names[-1] if previous_names else ""
            current_skill_id = apibank_api_skill_id(current_name) if current_name else apibank_start_skill_id()
            for doc in candidate_docs:
                skill = _skill_from_api_doc(doc)
                if skill["skill_id"] not in skill_ids_seen:
                    skills.append(skill)
                    skill_ids_seen.add(skill["skill_id"])
            if current_skill_id not in skill_ids_seen and current_name:
                skill = _skill_from_api_doc({"name": current_name, "description": f"Previously called API `{current_name}`."})
                skills.append(skill)
                skill_ids_seen.add(skill["skill_id"])
            history_text = "\n".join(previous_names)
            trajectory_id, step_index = _trajectory_and_step(file_name, row, row_id, fallback_idx)
            for call_index, gold_name in enumerate(gold_names):
                gold_doc = doc_by_key[_name_key(gold_name)]
                next_skill_id = apibank_api_skill_id(str(gold_doc.get("name") or gold_name))
                source_rows.append(
                    {
                        "benchmark": "apibank",
                        "source_benchmark": "apibank",
                        "split": "eval",
                        "task_id": f"{file_name}/{row_id}/{call_index}",
                        "trajectory_id": trajectory_id,
                        "apibank_file": file_name,
                        "row_id": row_id,
                        "step_index": int(step_index + call_index),
                        "call_index": int(call_index),
                        "state_text": _state_text(file_name=file_name, row_id=row_id, row=row, docs=candidate_docs),
                        "history_text": history_text,
                        "action_text": previous_names[-1] if previous_names else "previous_api: START",
                        "next_observation_text": "",
                        "skill_id": current_skill_id,
                        "next_skill_id": next_skill_id,
                        "next_skill_name": str(gold_doc.get("name") or gold_name),
                        "candidate_next_skill_ids": list(candidate_ids),
                        "candidate_next_skill_names": [str(doc.get("name") or "") for doc in candidate_docs],
                        "provenance": {
                            "source": "official_apibank",
                            "raw_source_path": str(path),
                            "file": file_name,
                            "raw_file": row.get("file"),
                            "raw_id": row.get("id"),
                            "sample_id": row.get("sample_id"),
                        },
                    }
                )
                counts["source_rows"] += 1
        file_counts[file_name] = counts

    report = {
        "status": "ok",
        "benchmark": "apibank",
        "data_root": str(data_root),
        "files": files,
        "include_trivial": bool(include_trivial),
        "skill_count": len(skills),
        "source_row_count": len(source_rows),
        "file_counts": file_counts,
        "skipped_reasons": dict(sorted(skipped.items())),
        "paper_scope_note": "Official APIBank-derived next-API routing rows; not APIBank generation/response exact-match scoring.",
    }
    return APIBankRouteCorpus(skills=skills, source_rows=source_rows, report=report)


def _skillrouter_query_text(row: dict[str, Any]) -> str:
    return (
        "Instruct: Given a dialogue state and visible API evidence, retrieve the API that should be called next\n"
        f"Query:{str(row.get('state_text') or '')[:1800]}"
    )


def _rank_rows(
    *,
    rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    query_embs: torch.Tensor,
    skill_embs: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, float]]]:
    skill_ids = [str(skill.get("skill_id") or "") for skill in skills]
    skill_id_to_idx = {sid: idx for idx, sid in enumerate(skill_ids)}
    scores = query_embs.float() @ skill_embs.float().t()
    ranked_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    by_file: dict[str, list[tuple[int | None, int]]] = defaultdict(list)
    missing_candidate_skill = 0
    for row_idx, row in enumerate(rows):
        scored: list[tuple[str, float]] = []
        for candidate_id in [str(item) for item in row.get("candidate_next_skill_ids") or []]:
            skill_idx = skill_id_to_idx.get(candidate_id)
            if skill_idx is None:
                missing_candidate_skill += 1
                continue
            scored.append((candidate_id, float(scores[row_idx, skill_idx].item())))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked_ids = [item[0] for item in scored]
        positive_id = str(row.get("next_skill_id") or "")
        rank = ranked_ids.index(positive_id) + 1 if positive_id in ranked_ids else None
        copied = dict(row)
        copied["candidate_next_skill_ids"] = ranked_ids
        copied["candidate_next_skillrouter_scores"] = [float(item[1]) for item in scored]
        copied["positive_next_skill_rank"] = rank
        ranked_rows.append(copied)
        ranks.append(rank)
        candidate_counts.append(len(ranked_ids))
        by_file[str(row.get("apibank_file") or "unknown")].append((rank, len(ranked_ids)))
    metrics = _metrics_from_ranks(ranks, candidate_counts)
    report = {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": sum(1 for rank in ranks if rank is not None),
        "positive_missing_rows": sum(1 for rank in ranks if rank is None),
        "candidate_skill_missing_count": int(missing_candidate_skill),
        "candidate_count_mean": _mean(candidate_counts),
        "metrics": metrics,
    }
    by_file_metrics = {
        file_name: {
            **_metrics_from_ranks([rank for rank, _count in values], [count for _rank, count in values]),
            "row_count": float(len(values)),
        }
        for file_name, values in sorted(by_file.items())
    }
    return ranked_rows, report, by_file_metrics


def run_apibank_skillrouter_frozen_eval(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    files: Iterable[str] | None = None,
    prebuilt_source_rows_path: str | Path | None = None,
    prebuilt_skills_path: str | Path | None = None,
    include_trivial: bool = False,
    max_rows_per_file: int | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 16,
    max_length: int = 2048,
    adapter_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prebuilt_source_rows_path is not None and prebuilt_skills_path is not None:
        prebuilt_source_rows_path = Path(prebuilt_source_rows_path)
        prebuilt_skills_path = Path(prebuilt_skills_path)
        corpus = APIBankRouteCorpus(
            skills=_read_jsonl(prebuilt_skills_path),
            source_rows=_read_jsonl(prebuilt_source_rows_path),
            report={
                "status": "ok",
                "benchmark": "apibank",
                "source": "prebuilt",
                "prebuilt_source_rows_path": str(prebuilt_source_rows_path),
                "prebuilt_skills_path": str(prebuilt_skills_path),
            },
        )
    else:
        corpus = load_apibank_route_corpus(
            data_root,
            files=files,
            include_trivial=include_trivial,
            max_rows_per_file=max_rows_per_file,
        )
    source_rows = (
        corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    )
    skills_path = output_dir / "apibank_skill_pool.jsonl"
    source_rows_path = output_dir / "apibank_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_query_text(row) for row in source_rows],
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_skill_text(skill) for skill in corpus.skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    query_embs, skill_embs, adapter_report = _apply_skillrouter_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    ranked_rows, ranking_report, metrics_by_file = _rank_rows(
        rows=source_rows,
        skills=corpus.skills,
        query_embs=query_embs,
        skill_embs=skill_embs,
    )
    ranked_path = output_dir / "apibank_skillrouter_ranked_rows.jsonl"
    _write_jsonl(ranked_path, ranked_rows)
    blockers: list[str] = []
    if not source_rows:
        blockers.append("no_source_eval_rows")
    if int(ranking_report.get("candidate_skill_missing_count") or 0) > 0:
        blockers.append("candidate_skill_missing")
    method = "skillrouter_finetuned_biencoder_adapter" if adapter_checkpoint_path is not None else "skillrouter_frozen_biencoder"
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "apibank",
        "method": method,
        "baseline_family": (
            "SkillRouter-compatible finetuned bi-encoder adapter on official APIBank-derived next-API routing"
            if adapter_checkpoint_path is not None
            else "SkillRouter-compatible frozen bi-encoder on official APIBank-derived next-API routing"
        ),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "ranked_rows_path": str(ranked_path),
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(ranked_rows),
        "corpus_report": corpus.report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "metrics_by_file": metrics_by_file,
        "config": {
            "files": None if files is None else list(files),
            "prebuilt_source_rows_path": None if prebuilt_source_rows_path is None else str(prebuilt_source_rows_path),
            "prebuilt_skills_path": None if prebuilt_skills_path is None else str(prebuilt_skills_path),
            "include_trivial": bool(include_trivial),
            "max_rows_per_file": max_rows_per_file,
            "max_eval_rows": max_eval_rows,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
        },
        "paper_scope_note": "Official APIBank-derived next-API routing evaluation; not APIBank generation exact-match scoring.",
    }
    _write_json(output_dir / "apibank_skillrouter_frozen_eval_report.json", report)
    return report


def _stage4_rows_from_ranked_apibank(
    source_rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    candidate_counts: list[int] = []
    for row in source_rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or [] if str(item) in skill_id_to_idx]
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if next_skill_id not in skill_id_to_idx:
            skipped["next_skill_not_in_pool"] += 1
            continue
        if next_skill_id not in candidates:
            skipped["next_positive_missing_from_candidates"] += 1
            continue
        prior_scores = list(row.get("candidate_next_prior_scores") or [])
        if len(prior_scores) != len(row.get("candidate_next_skill_ids") or []):
            prior_scores = [-float(idx) for idx in range(len(row.get("candidate_next_skill_ids") or []))]
        prior_scores = prior_scores[: len(candidates)]
        positive_pos = candidates.index(next_skill_id)
        rows.append(
            {
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index", row.get("call_index", 0)),
                "benchmark": "apibank",
                "source_benchmark": "apibank",
                "apibank_file": row.get("apibank_file"),
                "state_text": str(row.get("state_text") or ""),
                "action_text": str(row.get("action_text") or ""),
                "next_observation_text": str(row.get("next_observation_text") or ""),
                "skill_id": current_skill_id,
                "next_skill_id": next_skill_id,
                "skill_idx": int(skill_id_to_idx[current_skill_id]),
                "positive_next_skill_idx": int(skill_id_to_idx[next_skill_id]),
                "positive_next_skill_position": int(positive_pos),
                "candidate_next_skill_ids": candidates,
                "candidate_next_skill_indices": [int(skill_id_to_idx[item]) for item in candidates],
                "candidate_next_prior_scores": [float(value) for value in prior_scores],
                "positive_injected": False,
                "provenance": {
                    "source": "official_apibank_derived_stage0_ranked",
                    "original_provenance": row.get("provenance") or {},
                },
            }
        )
        candidate_counts.append(len(candidates))
    return rows, {
        "source_rows": len(source_rows),
        "stage4_rows": len(rows),
        "positive_injected_rows": 0,
        "candidate_source": "official_apibank_stage0_ranked",
        "candidate_count_mean": _mean(candidate_counts),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _build_apibank_clstr_report(
    *,
    output_dir: str | Path,
    data_root: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    skills_path: str | Path,
    source_eval_rows: int,
    retained_eval_rows: int,
    corpus_report: dict[str, Any],
    route_data_report: dict[str, Any],
    memory_report: dict[str, Any],
    stage0_prior_eval: dict[str, Any],
    base_eval: dict[str, Any],
    stage4_eval: dict[str, Any],
    config: dict[str, Any],
    stage0_prior_report: dict[str, Any],
    base_eval_by_benchmark: dict[str, Any],
    stage4_eval_by_benchmark: dict[str, Any],
    model_load: dict[str, Any],
    candidate_recall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_prior = _strict_stage4_metrics(stage0_prior_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_base = _strict_stage4_metrics(base_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_stage4 = _strict_stage4_metrics(stage4_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    if int(route_data_report.get("positive_injected_rows") or 0) > 0:
        blockers.append("gold_positive_injected")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "apibank",
        "stage4_route": "official_apibank_stage0_ranked_logged_online_memory",
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "skills_path": str(skills_path),
        "source_eval_rows": int(source_eval_rows),
        "retained_eval_rows": int(retained_eval_rows),
        "corpus_report": corpus_report,
        "stage0_prior_report": stage0_prior_report,
        "route_data_report": route_data_report,
        "memory_report": memory_report,
        "stage0_prior_eval": stage0_prior_eval,
        "base_eval": base_eval,
        "stage4_eval": stage4_eval,
        "base_eval_by_benchmark": base_eval_by_benchmark,
        "stage4_eval_by_benchmark": stage4_eval_by_benchmark,
        "strict": {"stage0_prior": strict_prior, "base": strict_base, "stage4": strict_stage4},
        "strict_delta_base_vs_stage0_prior": _numeric_delta(strict_base, strict_prior),
        "strict_delta_stage4_vs_base": _numeric_delta(strict_stage4, strict_base),
        "strict_delta_stage4_vs_stage0_prior": _numeric_delta(strict_stage4, strict_prior),
        "candidate_recall": candidate_recall or {},
        "config": config,
        "model_load": model_load,
        "paper_scope_note": "Official APIBank-derived next-API routing evaluation; not APIBank generation exact-match scoring.",
    }


def run_apibank_full_clstr_route_eval(
    *,
    data_root: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    output_dir: str | Path,
    files: Iterable[str] | None = None,
    prebuilt_source_rows_path: str | Path | None = None,
    prebuilt_skills_path: str | Path | None = None,
    include_trivial: bool = False,
    max_rows_per_file: int | None = None,
    max_eval_rows: int | None = None,
    batch_size: int = 8,
    stage0_candidate_batch_size: int = 16,
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.25,
    transition_scoring_mode: str = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    if prebuilt_source_rows_path is not None and prebuilt_skills_path is not None:
        prebuilt_source_rows_path = Path(prebuilt_source_rows_path)
        prebuilt_skills_path = Path(prebuilt_skills_path)
        corpus = APIBankRouteCorpus(
            skills=_read_jsonl(prebuilt_skills_path),
            source_rows=_read_jsonl(prebuilt_source_rows_path),
            report={
                "status": "ok",
                "benchmark": "apibank",
                "source": "prebuilt",
                "prebuilt_source_rows_path": str(prebuilt_source_rows_path),
                "prebuilt_skills_path": str(prebuilt_skills_path),
            },
        )
    else:
        corpus = load_apibank_route_corpus(
            data_root,
            files=files,
            include_trivial=include_trivial,
            max_rows_per_file=max_rows_per_file,
        )
    source_rows = (
        corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    )
    skills_path = output_dir / "apibank_skill_pool.jsonl"
    source_rows_path = output_dir / "apibank_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    skill_id_to_idx = {str(skill.get("skill_id") or ""): idx for idx, skill in enumerate(corpus.skills)}
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=skills_path,
        model_cache_dir=output_dir / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    stage4_load_report = None
    if stage4_checkpoint_path:
        stage4_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    ranked_rows, stage0_prior_report = _rank_candidates_with_stage0_prior(
        model,
        source_rows,
        skill_id_to_idx,
        batch_size=stage0_candidate_batch_size,
        device=device,
    )
    stage4_rows, route_data_report = _stage4_rows_from_ranked_apibank(ranked_rows, skill_id_to_idx)
    scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        stage4_rows,
        feedback_rows=stage4_rows,
        next_skill_bonus=online_memory_next_skill_bonus,
        exact_transition_bonus=online_memory_exact_transition_bonus,
        memory_mode=online_memory_mode,
    )
    stage0_prior_eval = _stage0_prior_eval_from_rows(stage4_rows)
    base_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
        progress_path=output_dir / "base_eval_progress.json",
        progress_label="base_eval",
    )
    base_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
    )
    stage4_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
        progress_path=output_dir / "stage4_eval_progress.json",
        progress_label="stage4_eval",
    )
    stage4_eval_by_benchmark = evaluate_logged_online_stage4_rows_by_benchmark(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        route_scorer=route_scorer,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
    )
    local_pool_size = declared_candidate_pool_size(ranked_rows)
    candidate_recall = candidate_recall_protocol_metadata(
        pool_protocol="benchmark_local",
        candidate_source=str(
            route_data_report.get("candidate_source") or "official_apibank_stage0_ranked"
        ),
        legal_pool_size=local_pool_size,
        static_k=local_pool_size,
        dynamic_extra_k=64,
        final_k=max(1, local_pool_size),
        causal_sequential=source_rows_have_causal_sequence(source_rows),
    )
    report = _build_apibank_clstr_report(
        output_dir=output_dir,
        data_root=data_root,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        skills_path=skills_path,
        source_eval_rows=len(source_rows),
        retained_eval_rows=len(stage4_rows),
        corpus_report=corpus.report,
        route_data_report=route_data_report,
        memory_report=memory_report,
        stage0_prior_eval=stage0_prior_eval,
        base_eval=base_eval,
        stage4_eval=stage4_eval,
        config={
            "files": None if files is None else list(files),
            "prebuilt_source_rows_path": None if prebuilt_source_rows_path is None else str(prebuilt_source_rows_path),
            "prebuilt_skills_path": None if prebuilt_skills_path is None else str(prebuilt_skills_path),
            "include_trivial": bool(include_trivial),
            "max_rows_per_file": max_rows_per_file,
            "max_eval_rows": max_eval_rows,
            "batch_size": int(batch_size),
            "stage0_candidate_batch_size": int(stage0_candidate_batch_size),
            "online_memory_mode": online_memory_mode,
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
            "route_scorer": str(route_scorer),
        },
        stage0_prior_report=stage0_prior_report,
        base_eval_by_benchmark=base_eval_by_benchmark,
        stage4_eval_by_benchmark=stage4_eval_by_benchmark,
        model_load={
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
        },
        candidate_recall=candidate_recall,
    )
    _write_json(output_dir / "apibank_full_clstr_route_eval_report.json", report)
    return report
