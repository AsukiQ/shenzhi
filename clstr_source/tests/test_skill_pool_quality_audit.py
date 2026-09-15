from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clstr.skill_pool_quality_audit import (
    audit_clstr_skill_pool_quality,
    review_skill_dedup_borderline_candidates,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_skill_pool_quality_audit_accepts_canonical_merge_and_multi_positive_qrels(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    write_json(
        data_root / "manifest.json",
        {
            "skill_pool": {
                "raw_skill_count": 3,
                "canonical_skill_count": 2,
                "merged_group_count": 1,
                "borderline_review": {
                    "status": "complete",
                    "candidate_count": 0,
                    "method": "manual_review_v1",
                },
            }
        },
    )
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolbench-g3/weather/get-weather",
                "canonical_skill_id": "toolbench-g3/weather/get-weather",
                "alias_skill_ids": ["toolbench-g3/weather/get-weather", "toolret/get-weather"],
                "name": "Get Weather",
                "description": "Fetches weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/news-search",
                "canonical_skill_id": "toolret/news-search",
                "alias_skill_ids": ["toolret/news-search"],
                "name": "Search News",
                "description": "Searches news.",
                "input_schema": {"query": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q-weather",
                "query_text": "Need weather.",
                "positive_skill_ids": [
                    "toolbench-g3/weather/get-weather",
                    "toolret/get-weather",
                ],
                "negative_skill_ids": ["toolret/news-search"],
            },
            {
                "query_id": "q-news",
                "query_text": "Need news.",
                "positive_skill_id": "toolret/news-search",
                "negative_skill_ids": ["toolbench-g3/weather/get-weather"],
            },
        ],
    )

    report = audit_clstr_skill_pool_quality(data_root=data_root, output_path=data_root / "skill_pool_quality.json")

    assert report["status"] == "ok"
    assert report["raw_skill_count"] == 3
    assert report["canonical_skill_count"] == 2
    assert report["merged_group_count"] == 1
    assert report["borderline_review_status"] == "complete"
    assert report["missing_positive_ref_count"] == 0
    assert report["placeholder_skill_count"] == 0
    assert report["duplicate_canonical_key_count"] == 0
    assert report["multi_positive_query_count"] == 1
    assert report["positive_alias_ref_count"] == 1
    assert json.loads((data_root / "skill_pool_quality.json").read_text(encoding="utf-8")) == report


def test_skill_pool_quality_audit_blocks_unmerged_duplicate_key_and_missing_positive_ref(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    write_json(
        data_root / "manifest.json",
        {
            "skill_pool": {
                "raw_skill_count": 2,
                "canonical_skill_count": 2,
                "merged_group_count": 0,
                "borderline_review": {
                    "status": "pending_manual_or_llm_review",
                    "candidate_count": 2,
                },
            }
        },
    )
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolret/weather-city",
                "canonical_skill_id": "toolret/weather-city",
                "name": "Get Weather",
                "description": "Fetches weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolbench-g3/weather/get-weather",
                "canonical_skill_id": "toolbench-g3/weather/get-weather",
                "name": "Get Weather",
                "description": "Returns current weather.",
                "input_schema": {"city": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q-weather",
                "query_text": "Need weather.",
                "positive_skill_id": "missing/weather",
                "negative_skill_ids": ["toolret/weather-city"],
            }
        ],
    )

    report = audit_clstr_skill_pool_quality(data_root=data_root)

    assert report["status"] == "action_required"
    assert report["duplicate_canonical_key_count"] == 1
    assert report["missing_positive_ref_count"] == 1
    assert report["missing_positive_refs"] == ["missing/weather"]
    assert "duplicate_canonical_keys" in report["blockers"]
    assert "missing_positive_refs" in report["blockers"]
    assert "skill_pool_borderline_review_pending" in report["blockers"]


def test_skill_pool_quality_audit_cli_fails_on_action_required(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    write_json(data_root / "manifest.json", {"skill_pool": {"total_skills": 0}})
    write_jsonl(data_root / "skill_pool.jsonl", [])
    write_jsonl(data_root / "retrieval.jsonl", [])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_clstr_skill_pool_quality.py",
            "--data_root",
            str(data_root),
            "--fail_on_action_required",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    assert "missing_skill_pool" in proc.stdout


def test_conservative_borderline_review_unblocks_low_confidence_candidates(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    write_json(
        data_root / "manifest.json",
        {
            "skill_pool": {
                "raw_skill_count": 2,
                "canonical_skill_count": 2,
                "merged_group_count": 0,
                "borderline_review": {
                    "status": "pending_manual_or_llm_review",
                    "candidate_count": 1,
                    "method": "fuzzy_name_or_partial_param_overlap_v1",
                },
            }
        },
    )
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolret/weather-city",
                "canonical_skill_id": "toolret/weather-city",
                "name": "Get Weather By City",
                "description": "Fetches weather by city.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolret/weather-zip",
                "canonical_skill_id": "toolret/weather-zip",
                "name": "Get Weather By ZIP",
                "description": "Fetches weather by ZIP code.",
                "input_schema": {"zip": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q-city",
                "query_text": "Need city weather.",
                "positive_skill_id": "toolret/weather-city",
                "negative_skill_ids": ["toolret/weather-zip"],
            }
        ],
    )
    candidates_path = tmp_path / "borderline_candidates.jsonl"
    reviewed_path = tmp_path / "borderline_candidates.reviewed.jsonl"
    review_report_path = tmp_path / "borderline_review_report.json"
    write_jsonl(
        candidates_path,
        [
            {
                "left_skill_id": "toolret/weather-city",
                "right_skill_id": "toolret/weather-zip",
                "left_name": "Get Weather By City",
                "right_name": "Get Weather By ZIP",
                "name_similarity": 0.75,
                "param_overlap": 0.0,
                "reasons": ["fuzzy_name_similarity"],
                "review_status": "pending_manual_or_llm",
            }
        ],
    )

    before = audit_clstr_skill_pool_quality(data_root=data_root)
    review = review_skill_dedup_borderline_candidates(
        candidates_path=candidates_path,
        output_path=reviewed_path,
        report_path=review_report_path,
    )
    after = audit_clstr_skill_pool_quality(
        data_root=data_root,
        borderline_review_report_path=review_report_path,
    )

    assert "skill_pool_borderline_review_pending" in before["blockers"]
    assert review["status"] == "complete"
    assert review["reviewed_candidate_count"] == 1
    assert review["keep_separate_count"] == 1
    assert review["unresolved_candidate_count"] == 0
    assert after["status"] == "ok"
    assert after["borderline_review_status"] == "complete_by_conservative_review"
    assert "skill_pool_borderline_review_pending" not in after["blockers"]


def test_conservative_borderline_review_keeps_high_confidence_duplicates_blocking(tmp_path):
    data_root = tmp_path / "data/clstr_unified_pretrain_v2"
    write_json(
        data_root / "manifest.json",
        {
            "skill_pool": {
                "raw_skill_count": 2,
                "canonical_skill_count": 2,
                "merged_group_count": 0,
                "borderline_review": {
                    "status": "pending_manual_or_llm_review",
                    "candidate_count": 1,
                    "method": "fuzzy_name_or_partial_param_overlap_v1",
                },
            }
        },
    )
    write_jsonl(
        data_root / "skill_pool.jsonl",
        [
            {
                "skill_id": "toolret/weather-a",
                "canonical_skill_id": "toolret/weather-a",
                "name": "Get Weather",
                "description": "Fetches weather.",
                "input_schema": {"city": {"type": "str"}},
            },
            {
                "skill_id": "toolbench-g3/weather/get-weather",
                "canonical_skill_id": "toolbench-g3/weather/get-weather",
                "name": "Get Weather",
                "description": "Returns weather.",
                "input_schema": {"city": {"type": "str"}},
            },
        ],
    )
    write_jsonl(
        data_root / "retrieval.jsonl",
        [
            {
                "query_id": "q-city",
                "query_text": "Need city weather.",
                "positive_skill_id": "toolret/weather-a",
                "negative_skill_ids": ["toolbench-g3/weather/get-weather"],
            }
        ],
    )
    candidates_path = tmp_path / "borderline_candidates.jsonl"
    review_report_path = tmp_path / "borderline_review_report.json"
    write_jsonl(
        candidates_path,
        [
            {
                "left_skill_id": "toolret/weather-a",
                "right_skill_id": "toolbench-g3/weather/get-weather",
                "left_name": "Get Weather",
                "right_name": "Get Weather",
                "name_similarity": 1.0,
                "param_overlap": 1.0,
                "reasons": ["fuzzy_name_similarity", "param_signature_partial_overlap"],
                "review_status": "pending_manual_or_llm",
            }
        ],
    )

    review = review_skill_dedup_borderline_candidates(
        candidates_path=candidates_path,
        report_path=review_report_path,
    )
    report = audit_clstr_skill_pool_quality(
        data_root=data_root,
        borderline_review_report_path=review_report_path,
    )

    assert review["status"] == "action_required"
    assert review["unresolved_candidate_count"] == 1
    assert report["status"] == "action_required"
    assert "skill_pool_borderline_review_pending" in report["blockers"]
