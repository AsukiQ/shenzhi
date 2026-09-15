from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BFCL_CATEGORIES = [
    "BFCL_v3_multiple",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel_multiple",
    "BFCL_v3_multi_turn_base",
    "BFCL_v3_multi_turn_long_context",
    "BFCL_v3_multi_turn_miss_func",
    "BFCL_v3_multi_turn_miss_param",
]

DEFAULT_APIBANK_TRAIN_FILES = [
    "training-data/lv1-api-train.json",
    "training-data/lv2-api-train.json",
    "training-data/lv3-api-train.json",
]


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def _short_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
    except OSError:
        if src.is_file():
            shutil.copy2(src, dst)


def _parse_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [tool for tool in value if isinstance(tool, dict) and str(tool.get("name") or "").strip()]


def _parse_function_call(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or "").strip()
    if not name:
        return None
    return value


def _unitool_skill_id(source: str, tool: dict[str, Any]) -> str:
    name = str(tool.get("name") or "").strip()
    return f"unitoolcall/{source}/{_slug(name)}/{_short_hash(tool)}"


def _unitool_skill_row(source: str, tool: dict[str, Any]) -> dict[str, Any]:
    skill_id = _unitool_skill_id(source, tool)
    description = str(tool.get("description") or tool.get("desc") or f"Function {tool.get('name')}")
    return {
        "skill_id": skill_id,
        "canonical_skill_id": skill_id,
        "name": str(tool.get("name") or skill_id),
        "description": description,
        "source": f"UniToolCall/{source}",
        "environment": tool.get("domain") or tool.get("category"),
        "executor_desc": description,
        "input_schema": tool.get("inputSchema") or tool.get("parameters") or {},
        "output_schema": tool.get("outputSchema") or {},
        "body": "",
        "provenance": {
            "source_dataset": "EIT-NLP/UniToolCall",
            "source_split": source,
            "tool_hash": _short_hash(tool),
        },
    }


def _conversation_query_text(
    *,
    user_text: str,
    previous_calls: list[str],
    previous_observations: list[str],
) -> str:
    prev_calls = " | ".join(previous_calls) if previous_calls else "<empty>"
    prev_obs = " | ".join(previous_observations[-3:]) if previous_observations else "<empty>"
    return "\n".join(
        [
            f"goal: {user_text}",
            "benchmark: UniToolCall",
            "task: function-schema next-tool routing",
            f"previous_function_calls: {prev_calls}",
            f"recent_observations: {prev_obs}",
        ]
    )


def _iter_unitool_retrieval_rows(
    *,
    path: Path,
    source: str,
    skill_rows_by_id: dict[str, dict[str, Any]],
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"UniToolCall file must contain a list: {path}")

    rows: list[dict[str, Any]] = []
    report = {
        "source": f"unitoolcall_{source}",
        "path": str(path),
        "input_rows": len(payload),
        "used_dialog_rows": 0,
        "retrieval_rows": 0,
        "skipped_no_tools": 0,
        "skipped_missing_positive_tool": 0,
    }
    for dialog_idx, dialog in enumerate(payload):
        if max_rows is not None and report["used_dialog_rows"] >= max_rows:
            break
        if not isinstance(dialog, dict):
            continue
        tools = _parse_tools(dialog.get("tools"))
        if not tools:
            report["skipped_no_tools"] += 1
            continue
        tool_by_name = {str(tool.get("name") or "").strip(): tool for tool in tools}
        tool_skill_ids = []
        for tool in tools:
            skill_id = _unitool_skill_id(source, tool)
            skill_rows_by_id.setdefault(skill_id, _unitool_skill_row(source, tool))
            tool_skill_ids.append(skill_id)

        conversations = dialog.get("conversations") or []
        if not isinstance(conversations, list):
            continue
        user_text = ""
        previous_calls: list[str] = []
        previous_observations: list[str] = []
        emitted = 0
        for turn_idx, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or "").strip()
            value = turn.get("value")
            if role == "human" and not user_text:
                user_text = str(value or "")
                continue
            if role == "observation":
                previous_observations.append(str(value or ""))
                continue
            if role != "function_call":
                continue
            call = _parse_function_call(value)
            if call is None:
                continue
            tool = tool_by_name.get(str(call.get("name") or "").strip())
            if tool is None:
                report["skipped_missing_positive_tool"] += 1
                previous_calls.append(json.dumps(call, ensure_ascii=False, sort_keys=True))
                continue
            positive_skill_id = _unitool_skill_id(source, tool)
            rows.append(
                {
                    "source": f"unitoolcall_{source}",
                    "query_id": f"unitoolcall::{source}::{dialog_idx}::{turn_idx}",
                    "query_text": _conversation_query_text(
                        user_text=user_text,
                        previous_calls=previous_calls,
                        previous_observations=previous_observations,
                    ),
                    "positive_skill_id": positive_skill_id,
                    "negative_skill_ids": [sid for sid in tool_skill_ids if sid != positive_skill_id],
                    "provenance": json.dumps(
                        {
                            "source_dataset": "EIT-NLP/UniToolCall",
                            "source_file": str(path),
                            "dialog_index": dialog_idx,
                            "turn_index": turn_idx,
                            "tool_name": call.get("name"),
                            "split": "train",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            report["retrieval_rows"] += 1
            emitted += 1
            previous_calls.append(json.dumps(call, ensure_ascii=False, sort_keys=True))
        if emitted:
            report["used_dialog_rows"] += 1
    return rows, report


def _official_bfcl_rows(
    *,
    bfcl_root: Path,
    skill_rows_by_id: dict[str, dict[str, Any]],
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from clstr.bfcl_route_eval import load_bfcl_route_corpus

    corpus = load_bfcl_route_corpus(
        bfcl_root,
        categories=DEFAULT_BFCL_CATEGORIES,
        max_rows_per_category=None,
    )
    for skill in corpus.skills:
        skill_rows_by_id.setdefault(str(skill["skill_id"]), skill)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(corpus.source_rows):
        if max_rows is not None and idx >= max_rows:
            break
        positive = str(row.get("next_skill_id") or "").strip()
        if not positive:
            continue
        candidate_ids = [str(x) for x in row.get("candidate_next_skill_ids") or [] if str(x)]
        rows.append(
            {
                "source": "official_bfcl_function_schema",
                "query_id": f"official_bfcl::{row.get('bfcl_category')}::{row.get('bfcl_id')}::{row.get('turn_index')}::{row.get('call_index')}",
                "query_text": str(row.get("state_text") or ""),
                "positive_skill_id": positive,
                "negative_skill_ids": [sid for sid in candidate_ids if sid != positive],
                "provenance": json.dumps(
                    {
                        "source_dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                        "bfcl_category": row.get("bfcl_category"),
                        "bfcl_id": row.get("bfcl_id"),
                        "split": "train_augmentation",
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows, {
        "source": "official_bfcl_function_schema",
        "input_rows": len(corpus.source_rows),
        "retrieval_rows": len(rows),
        "corpus_report": corpus.report,
    }


def _validate_apibank_train_files(files: Iterable[str]) -> list[str]:
    validated: list[str] = []
    for file_name in files:
        path = Path(str(file_name))
        parts = set(path.parts)
        if "test-data" in parts or "training-data" not in parts:
            raise ValueError(
                "APIBank train augmentation only accepts training-data files; "
                f"got {file_name!r}"
            )
        validated.append(str(path.as_posix()))
    return validated


def _official_apibank_train_rows(
    *,
    apibank_root: Path,
    skill_rows_by_id: dict[str, dict[str, Any]],
    files: Iterable[str],
    max_rows_per_file: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from clstr.apibank_route_eval import load_apibank_route_corpus

    train_files = _validate_apibank_train_files(files)
    corpus = load_apibank_route_corpus(
        apibank_root,
        files=train_files,
        include_trivial=True,
        max_rows_per_file=max_rows_per_file,
    )
    for skill in corpus.skills:
        skill_rows_by_id.setdefault(str(skill["skill_id"]), skill)

    rows: list[dict[str, Any]] = []
    for row in corpus.source_rows:
        positive = str(row.get("next_skill_id") or "").strip()
        if not positive:
            continue
        candidate_ids = [str(x) for x in row.get("candidate_next_skill_ids") or [] if str(x)]
        rows.append(
            {
                "source": "official_apibank_function_schema",
                "query_id": (
                    "official_apibank_train::"
                    f"{row.get('apibank_file')}::{row.get('row_id')}::{row.get('call_index')}"
                ),
                "query_text": str(row.get("state_text") or ""),
                "positive_skill_id": positive,
                "negative_skill_ids": [sid for sid in candidate_ids if sid != positive],
                "provenance": json.dumps(
                    {
                        "source_dataset": "liminghao1630/API-Bank",
                        "apibank_file": row.get("apibank_file"),
                        "row_id": row.get("row_id"),
                        "call_index": row.get("call_index"),
                        "split": "train_augmentation",
                    },
                    ensure_ascii=False,
                ),
            }
        )

    return rows, {
        "source": "official_apibank_function_schema",
        "split_policy": "train_only",
        "input_rows": sum(
            int((counts or {}).get("raw_rows") or 0)
            for counts in (corpus.report.get("file_counts") or {}).values()
        ),
        "retrieval_rows": len(rows),
        "files": train_files,
        "test_data_files_used": 0,
        "corpus_report": corpus.report,
    }


def _source_inventory_row_from_aug_report(report: dict[str, Any], aug_root: Path) -> dict[str, Any]:
    source = str(report.get("source") or "function_schema_augmentation")
    benchmark_by_source = {
        "official_apibank_function_schema": "APIBank",
        "official_bfcl_function_schema": "BFCL",
        "unitoolcall_apigen": "UniToolCall/APIGen",
        "unitoolcall_bfcl": "UniToolCall/BFCL",
    }
    remote_by_source = {
        "official_apibank_function_schema": "https://huggingface.co/datasets/liminghao1630/API-Bank",
        "official_bfcl_function_schema": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "unitoolcall_apigen": "https://huggingface.co/datasets/EIT-NLP/UniToolCall",
        "unitoolcall_bfcl": "https://huggingface.co/datasets/EIT-NLP/UniToolCall",
    }
    split_policy = str(report.get("split_policy") or "train_conversion_only; do not evaluate on same rows")
    return {
        "source_id": source,
        "benchmark": benchmark_by_source.get(source, source),
        "role": "function_schema_retrieval_augmentation",
        "split_policy": split_policy,
        "remote_url": remote_by_source.get(source),
        "available": True,
        "resolved_path": str(aug_root),
        "first_existing_path": str(aug_root),
        "train_allowed": True,
        "reason_if_missing": None,
    }


def merge_function_schema_augmentation(
    *,
    base_data_root: str | Path,
    function_schema_augmentation_root: str | Path,
    output_dir: str | Path,
    training_note: str | None = None,
) -> dict[str, Any]:
    base_data_root = Path(base_data_root)
    aug_root = Path(function_schema_augmentation_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_skill_rows_skipped = 0
    for path in (base_data_root / "skill_pool.jsonl", aug_root / "skill_pool.jsonl"):
        for row in _iter_jsonl(path):
            skill_id = str(row.get("skill_id") or "").strip()
            if not skill_id:
                continue
            if skill_id in skill_rows_by_id:
                duplicate_skill_rows_skipped += 1
                continue
            skill_rows_by_id[skill_id] = row

    skill_count = _write_jsonl(
        output_dir / "skill_pool.jsonl",
        (skill_rows_by_id[skill_id] for skill_id in sorted(skill_rows_by_id)),
    )

    retrieval_by_source: dict[str, int] = {}
    retrieval_rows_skipped_missing_positive = 0
    retrieval_count = 0
    with (output_dir / "retrieval.jsonl").open("w", encoding="utf-8") as handle:
        for path in (base_data_root / "retrieval.jsonl", aug_root / "retrieval.jsonl"):
            for row in _iter_jsonl(path):
                positive = str(row.get("positive_skill_id") or "").strip()
                if positive and positive not in skill_rows_by_id:
                    retrieval_rows_skipped_missing_positive += 1
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                retrieval_count += 1
                source = str(row.get("source") or "unknown")
                retrieval_by_source[source] = retrieval_by_source.get(source, 0) + 1

    source_rows = list(_iter_jsonl(base_data_root / "source_inventory.jsonl"))
    existing_sources = {str(row.get("source_id") or "") for row in source_rows}
    aug_manifest = _read_json(aug_root / "manifest.json") if (aug_root / "manifest.json").exists() else {}
    added_sources: list[str] = []
    for source_report in aug_manifest.get("sources") or []:
        source_id = str(source_report.get("source") or "")
        if not source_id or source_id in existing_sources:
            continue
        source_rows.append(_source_inventory_row_from_aug_report(source_report, aug_root))
        existing_sources.add(source_id)
        added_sources.append(source_id)
    _write_jsonl(output_dir / "source_inventory.jsonl", source_rows)

    for name in ("trajectories.jsonl", "skill_aliases.jsonl", "skill_dedup_borderline_candidates.jsonl", "leakage_audit.json"):
        _link_or_copy(base_data_root / name, output_dir / name)

    manifest = {
        "status": "ok",
        "base_data_root": str(base_data_root),
        "function_schema_augmentation_root": str(aug_root),
        "output_dir": str(output_dir),
        "skill_count": skill_count,
        "duplicate_skill_rows_skipped": duplicate_skill_rows_skipped,
        "retrieval_row_count": retrieval_count,
        "retrieval_rows_skipped_missing_positive": retrieval_rows_skipped_missing_positive,
        "retrieval_by_source": dict(sorted(retrieval_by_source.items())),
        "files": {
            "skill_pool": "skill_pool.jsonl",
            "retrieval": "retrieval.jsonl",
            "trajectories": "trajectories.jsonl",
            "source_inventory": "source_inventory.jsonl",
        },
        "training_note": training_note
        or "Stage0/Stage1 candidate: base CLSTR data plus function-schema retrieval augmentation. Do not evaluate on augmentation rows as held-out claims.",
        "source_inventory": {
            "source_count": len(source_rows),
            "added_sources": added_sources,
            "available_source_count": sum(1 for row in source_rows if row.get("available")),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_function_schema_augmentation(
    *,
    unitool_root: str | Path | None,
    bfcl_root: str | Path | None,
    apibank_root: str | Path | None = None,
    output_dir: str | Path,
    include_unitool_apigen: bool = True,
    include_unitool_bfcl: bool = False,
    include_official_bfcl: bool = True,
    include_official_apibank_train: bool = False,
    apibank_train_files: Iterable[str] | None = None,
    max_unitool_rows: int | None = None,
    max_bfcl_rows: int | None = None,
    max_apibank_rows_per_file: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    skill_rows_by_id: dict[str, dict[str, Any]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    if unitool_root is not None:
        unitool_root = Path(unitool_root)
        if include_unitool_apigen:
            rows, report = _iter_unitool_retrieval_rows(
                path=unitool_root / "train" / "train_converted_APIGen.json",
                source="apigen",
                skill_rows_by_id=skill_rows_by_id,
                max_rows=max_unitool_rows,
            )
            retrieval_rows.extend(rows)
            sources.append(report)
        if include_unitool_bfcl:
            rows, report = _iter_unitool_retrieval_rows(
                path=unitool_root / "test" / "test_converted_bfcl.json",
                source="bfcl",
                skill_rows_by_id=skill_rows_by_id,
                max_rows=max_unitool_rows,
            )
            retrieval_rows.extend(rows)
            sources.append(report)

    if bfcl_root is not None and include_official_bfcl:
        rows, report = _official_bfcl_rows(
            bfcl_root=Path(bfcl_root),
            skill_rows_by_id=skill_rows_by_id,
            max_rows=max_bfcl_rows,
        )
        retrieval_rows.extend(rows)
        sources.append(report)

    if apibank_root is not None and include_official_apibank_train:
        rows, report = _official_apibank_train_rows(
            apibank_root=Path(apibank_root),
            skill_rows_by_id=skill_rows_by_id,
            files=apibank_train_files or DEFAULT_APIBANK_TRAIN_FILES,
            max_rows_per_file=max_apibank_rows_per_file,
        )
        retrieval_rows.extend(rows)
        sources.append(report)

    skill_rows = [skill_rows_by_id[skill_id] for skill_id in sorted(skill_rows_by_id)]
    retrieval_rows = sorted(retrieval_rows, key=lambda row: str(row.get("query_id") or ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    skill_count = _write_jsonl(output_dir / "skill_pool.jsonl", skill_rows)
    retrieval_count = _write_jsonl(output_dir / "retrieval.jsonl", retrieval_rows)
    manifest = {
        "status": "ok",
        "output_dir": str(output_dir),
        "skill_count": skill_count,
        "retrieval_row_count": retrieval_count,
        "sources": sources,
        "files": {
            "skill_pool": "skill_pool.jsonl",
            "retrieval": "retrieval.jsonl",
        },
        "paper_scope_note": "Function-schema data augmentation for Stage0/Stage1 training; do not evaluate on the same held-out source split.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BFCL/UniToolCall function-schema retrieval augmentation data.")
    parser.add_argument("--unitool_root", default=".tmp/benchmark_probe_direct/EIT-NLP__UniToolCall")
    parser.add_argument("--bfcl_root", default=".tmp/benchmark_probe_direct/gorilla-llm__Berkeley-Function-Calling-Leaderboard")
    parser.add_argument("--apibank_root", default=".tmp/benchmark_probe_direct/liminghao1630__API-Bank")
    parser.add_argument("--output_dir", default="data/function_schema_augmentation_v1")
    parser.add_argument("--include_unitool_apigen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_unitool_bfcl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_official_bfcl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_official_apibank_train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apibank_train_files", nargs="*", default=None)
    parser.add_argument("--max_unitool_rows", type=int)
    parser.add_argument("--max_bfcl_rows", type=int)
    parser.add_argument("--max_apibank_rows_per_file", type=int)
    parser.add_argument("--merge_base_data_root")
    parser.add_argument("--merge_output_dir")
    args = parser.parse_args(argv)
    report = build_function_schema_augmentation(
        unitool_root=args.unitool_root,
        bfcl_root=args.bfcl_root,
        apibank_root=args.apibank_root,
        output_dir=args.output_dir,
        include_unitool_apigen=args.include_unitool_apigen,
        include_unitool_bfcl=args.include_unitool_bfcl,
        include_official_bfcl=args.include_official_bfcl,
        include_official_apibank_train=args.include_official_apibank_train,
        apibank_train_files=args.apibank_train_files,
        max_unitool_rows=args.max_unitool_rows,
        max_bfcl_rows=args.max_bfcl_rows,
        max_apibank_rows_per_file=args.max_apibank_rows_per_file,
    )
    if args.merge_base_data_root and args.merge_output_dir:
        report = {
            "augmentation": report,
            "merge": merge_function_schema_augmentation(
                base_data_root=args.merge_base_data_root,
                function_schema_augmentation_root=args.output_dir,
                output_dir=args.merge_output_dir,
            ),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    status = report.get("status") or (report.get("merge") or {}).get("status")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
