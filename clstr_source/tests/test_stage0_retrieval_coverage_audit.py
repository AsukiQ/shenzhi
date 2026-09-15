import json
from pathlib import Path

from clstr.stage0_retrieval_coverage_audit import (
    build_stage0_missing_positive_subset,
    build_stage0_retrieval_coverage_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage0_retrieval_coverage_audit_reports_source_slices_and_missing_rows(tmp_path):
    retrieval_path = tmp_path / "retrieval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        retrieval_path,
        [
            {
                "source": "toolbench_g3",
                "query_id": "q1",
                "query_text": "first",
                "positive_skill_id": "skill/a",
                "negative_skill_ids": [],
                "provenance": {"source_dataset": "unit"},
            },
            {
                "source": "trajectory_derived_alfworld",
                "query_id": "q2",
                "query_text": "second",
                "positive_skill_id": "skill/z",
                "negative_skill_ids": [],
                "provenance": {"source_dataset": "unit"},
            },
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {"query_id": "q1", "ranked_skill_ids": ["skill/x", "skill/a", "skill/b"]},
            {"query_id": "q2", "ranked_skill_ids": ["skill/x", "skill/y", "skill/w"]},
        ],
    )

    report = build_stage0_retrieval_coverage_audit(
        retrieval_rows_path=retrieval_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        k_values=[1, 2, 3],
        negative_k=2,
    )

    assert report["row_count"] == 2
    assert report["mismatched_query_id_count"] == 0
    assert report["overall"]["recall@1"] == 0.0
    assert report["overall"]["recall@2"] == 0.5
    assert report["overall"]["recall@3"] == 0.5
    assert report["by_source"]["toolbench_g3"]["recall@2"] == 1.0
    assert report["by_source"]["trajectory_derived_alfworld"]["missing@3"] == 1

    missing = [json.loads(line) for line in (output_dir / "stage0_missing_positive_rows.jsonl").read_text().splitlines()]
    assert len(missing) == 1
    assert missing[0]["query_id"] == "q2"
    assert missing[0]["positive_skill_id"] == "skill/z"
    assert missing[0]["negative_skill_ids"] == ["skill/x", "skill/y"]
    assert missing[0]["provenance"]["candidate_positive_missing"] is True
    assert missing[0]["provenance"]["audit_max_k"] == 3


def test_stage0_retrieval_coverage_audit_separates_absent_positive_from_topk_miss(tmp_path):
    retrieval_path = tmp_path / "retrieval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    skill_pool_path = tmp_path / "skill_pool.jsonl"
    output_dir = tmp_path / "audit"
    _write_jsonl(
        retrieval_path,
        [
            {
                "source": "appworld_stage0_api_correction",
                "query_id": "absent-positive",
                "query_text": "Required API refs: apis.spotify.create_playlist",
                "positive_skill_id": "skillx/appworld/spotify-create-new-playlist-21",
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "query_id": "absent-positive",
                "ranked_skill_ids": ["toolbench/search", "toolret/weather"],
            }
        ],
    )
    _write_jsonl(
        skill_pool_path,
        [
            {"skill_id": "toolbench/search"},
            {"skill_id": "toolret/weather"},
        ],
    )

    report = build_stage0_retrieval_coverage_audit(
        retrieval_rows_path=retrieval_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        k_values=[1, 2],
        skill_pool_path=skill_pool_path,
    )

    assert report["overall"]["rows"] == 1
    assert report["overall"]["positive_absent_from_pool"] == 1
    assert report["overall"]["missing_at_max_k"] == 0
    assert report["by_source"]["appworld_stage0_api_correction"]["positive_absent_from_pool"] == 1
    assert report["written_missing_rows"] == 0
    assert (output_dir / "stage0_missing_positive_rows.jsonl").read_text(encoding="utf-8") == ""


def test_stage0_missing_positive_subset_filters_sources_and_caps_per_source(tmp_path):
    input_path = tmp_path / "missing.jsonl"
    output_dir = tmp_path / "subset"
    _write_jsonl(
        input_path,
        [
            {"source": "toolbench_g3_stage0_missing_positive", "query_id": "tb1", "positive_skill_id": "a"},
            {"source": "toolbench_g3_stage0_missing_positive", "query_id": "tb2", "positive_skill_id": "b"},
            {"source": "trajectory_derived_traject_bench_stage0_missing_positive", "query_id": "tr1", "positive_skill_id": "c"},
            {"source": "skillret_stage0_missing_positive", "query_id": "sr1", "positive_skill_id": "d"},
        ],
    )

    report = build_stage0_missing_positive_subset(
        input_path=input_path,
        output_dir=output_dir,
        source_allowlist=["toolbench_g3", "trajectory_derived_traject_bench"],
        per_source_cap=1,
    )

    assert report["input_count"] == 4
    assert report["kept_count"] == 2
    assert report["skipped_by_source_filter"] == 1
    assert report["skipped_by_cap"] == 1
    assert report["kept_by_source"] == {"toolbench_g3": 1, "trajectory_derived_traject_bench": 1}
    rows = [json.loads(line) for line in (output_dir / "stage0_missing_positive_subset.jsonl").read_text().splitlines()]
    assert [row["query_id"] for row in rows] == ["tb1", "tr1"]
    assert rows[0]["provenance"]["subset_source"] == "toolbench_g3"
