#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _skill_id(row: dict[str, Any]) -> str:
    return str(
        row.get("skill_id")
        or row.get("canonical_skill_id")
        or row.get("tool_id")
        or row.get("id")
        or ""
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 67,557-skill ToolBench corpus for matched Qwen SR/ToolREx "
            "evaluation while preserving all existing native ToolREx profiles."
        )
    )
    parser.add_argument("--baseline_project_root", required=True)
    parser.add_argument("--native_profiled_corpus_dir", required=True)
    parser.add_argument("--global_skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_native_skill_count", type=int, default=27_486)
    parser.add_argument("--expected_global_skill_count", type=int, default=67_557)
    parser.add_argument("--expected_source_row_count", type=int, default=1_362)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    baseline_root = Path(args.baseline_project_root).expanduser().resolve()
    if not (baseline_root / "clstr/qwen_multibench_corpora.py").is_file():
        raise ValueError("baseline project lacks the Qwen multibench implementation")
    sys.path.insert(0, str(baseline_root))

    from clstr.qwen_method_benchmark_format import canonicalize_benchmark_skill
    from clstr.qwen_method_documents import tool_document_hash
    from clstr.qwen_multibench_corpora import (
        file_sha256,
        write_frozen_qwen_route_corpus,
    )
    from clstr.qwen_multibench_protocol import (
        build_benchmark_manifest,
        load_benchmark_manifest,
    )
    from clstr.toolrex_document_resolution import TOOLREX_FALLBACK_RESOLUTION

    native_root = Path(args.native_profiled_corpus_dir).expanduser().resolve()
    native_manifest_path = native_root / "manifest.json"
    native_skills_path = native_root / "skills.jsonl"
    native_rows_path = native_root / "source_rows.jsonl"
    global_skills_path = Path(args.global_skills_path).expanduser().resolve()

    native_manifest = load_benchmark_manifest(native_manifest_path)
    if native_manifest.get("benchmark") != "toolbench_g3":
        raise ValueError("native profiled corpus is not ToolBench-G3")
    native_skills = _read_jsonl(native_skills_path)
    native_rows = _read_jsonl(native_rows_path)
    global_skills = _read_jsonl(global_skills_path)
    if len(native_skills) != int(args.expected_native_skill_count):
        raise ValueError("native ToolBench skill count changed")
    if len(global_skills) != int(args.expected_global_skill_count):
        raise ValueError("global ToolBench skill count changed")
    if len(native_rows) != int(args.expected_source_row_count):
        raise ValueError("ToolBench source-row count changed")
    if native_rows != list(native_manifest.get("canonical_source_rows") or []):
        raise ValueError("native ToolBench rows differ from their frozen manifest")

    native_by_id = {_skill_id(row): dict(row) for row in native_skills}
    global_by_id: dict[str, dict[str, Any]] = {}
    for raw in global_skills:
        canonical = canonicalize_benchmark_skill(raw, benchmark="toolbench_g3")
        skill_id = _skill_id(canonical)
        if not skill_id or skill_id in global_by_id:
            raise ValueError(f"invalid or duplicate global skill ID: {skill_id}")
        global_by_id[skill_id] = canonical
    if not set(native_by_id).issubset(global_by_id):
        raise ValueError("global pool is missing native ToolBench skill IDs")

    merged_skills: list[dict[str, Any]] = []
    accepted_profile_count = 0
    inherited_fallback_count = 0
    global_extension_fallback_count = 0
    for skill_id in sorted(global_by_id):
        if skill_id in native_by_id:
            skill = native_by_id[skill_id]
            accepted_profile_count += int(
                isinstance(skill.get("accepted_toolrex_profile"), dict)
            )
            inherited_fallback_count += int(
                not isinstance(skill.get("accepted_toolrex_profile"), dict)
            )
            merged_skills.append(skill)
            continue
        skill = global_by_id[skill_id]
        merged_skills.append(
            {
                **skill,
                "toolrex_document_resolution": TOOLREX_FALLBACK_RESOLUTION,
                "toolrex_profile_fallback": {
                    "source": "global_pool_unprofiled_extension",
                    "reason": "global_pool_unprofiled_extension",
                    "original_document_hash": tool_document_hash(skill),
                },
            }
        )
        global_extension_fallback_count += 1

    source_sha256 = {
        str(native_manifest_path): file_sha256(native_manifest_path),
        str(native_skills_path): file_sha256(native_skills_path),
        str(native_rows_path): file_sha256(native_rows_path),
        str(global_skills_path): file_sha256(global_skills_path),
    }
    manifest = build_benchmark_manifest(
        benchmark="toolbench_g3",
        split=str(native_manifest.get("split") or "eval"),
        source_rows=native_rows,
        skills=merged_skills,
        source_sha256=source_sha256,
        formatting_version=str(native_manifest.get("formatting_version") or ""),
        candidate_source="full_pool",
    )
    target_ids = {str(row["positive_skill_id"]) for row in native_rows}
    merged_ids = {str(row["skill_id"]) for row in merged_skills}
    if target_ids - merged_ids:
        raise ValueError("global ToolBench pool is missing evaluation targets")

    preparation_report = {
        "schema_version": 1,
        "status": "ok",
        "benchmark": "toolbench_g3",
        "pool_scope": "global_67557",
        "source_row_count": len(native_rows),
        "skill_count": len(merged_skills),
        "native_skill_count": len(native_skills),
        "accepted_toolrex_profile_count": accepted_profile_count,
        "inherited_toolrex_fallback_count": inherited_fallback_count,
        "global_extension_fallback_count": global_extension_fallback_count,
        "toolrex_profile_coverage": accepted_profile_count / len(merged_skills),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": source_sha256,
    }
    report = write_frozen_qwen_route_corpus(
        frozen={
            "manifest": manifest,
            "source_rows": list(manifest["canonical_source_rows"]),
            "skills": list(manifest["canonical_skills"]),
            "preparation_report": preparation_report,
        },
        output_dir=args.output_dir,
        resume=bool(args.resume),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
