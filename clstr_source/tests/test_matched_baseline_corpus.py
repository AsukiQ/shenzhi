from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clstr.matched_baseline_corpus import (
    _stable_stratified_query_cap,
    _query_from_vnext_row,
    build_matched_baseline_corpus,
    load_prepared_matched_baseline_corpus,
)
from clstr.baseline_corpus_types import UnifiedSkillRouterQuery


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _vnext_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    skills = [
        {"skill_id": "skill/a", "name": "A"},
        {"skill_id": "skill/b", "name": "B"},
        {"skill_id": "skill/c", "name": "C"},
        {"skill_id": "skill/d", "name": "D"},
    ]
    candidates = ["skill/a", "skill/b", "skill/c"]
    catalog_digest = _json_digest(candidates)
    catalogs = [
        {
            "inventory_catalog_id": "local-tools",
            "inventory_catalog_digest": catalog_digest,
            "runtime_visible_skill_ids": candidates,
            "inventory_pool_size": len(candidates),
        }
    ]

    def row(
        *,
        split: str,
        group: str,
        query_id: str,
        target_field: str,
        targets: list[str],
        source: str,
    ) -> dict:
        output = {
            "data_split": split,
            "split_group_identity": group,
            "split_schema_version": "fixture_split_v1",
            "split_assignment_source": "fixture",
            "query_id": query_id,
            "source": source,
            "benchmark": "toolsandbox",
            "state_text_current": f"current state {query_id}",
            "state_text_causal": f"unused full causal {query_id}",
            "causal_prefix_events": [
                {
                    "step_index": 0,
                    "skill_id": "skill/d",
                    "action_text": "inspect previous state",
                    "actual_result_text": "privileged result must not enter static query",
                    "actual_result_executed": True,
                }
            ],
            "runtime_visible_catalog_id": "local-tools",
            "inventory_catalog_digest": catalog_digest,
            target_field: targets,
            "next_skill_id": "skill/d",
        }
        if target_field == "current_state_route_set_skill_ids":
            output["current_state_route_group_identity"] = f"static-group-{query_id}"
        return output

    files = {
        "training_skills": tmp_path / "training_skills.jsonl",
        "inventory_catalogs": tmp_path / "inventory_catalogs.jsonl",
        "retrieval_rows": tmp_path / "retrieval_rows.jsonl",
        "retrieval_dev_rows": tmp_path / "retrieval_dev_rows.jsonl",
        "static_route_rows": tmp_path / "static_route_rows.jsonl",
        "static_route_dev_rows": tmp_path / "static_route_dev_rows.jsonl",
    }
    _write_jsonl(files["training_skills"], skills)
    _write_jsonl(files["inventory_catalogs"], catalogs)
    _write_jsonl(
        files["retrieval_rows"],
        [
            row(
                split="train",
                group="train-retrieval",
                query_id="retrieval-train",
                target_field="required_tool_set_skill_ids",
                targets=["skill/a", "skill/b"],
                source="matched_toolsandbox",
            )
        ],
    )
    _write_jsonl(
        files["retrieval_dev_rows"],
        [
            row(
                split="dev",
                group="dev-retrieval",
                query_id="retrieval-dev",
                target_field="required_tool_set_skill_ids",
                targets=["skill/b"],
                source="matched_toolsandbox",
            )
        ],
    )
    _write_jsonl(
        files["static_route_rows"],
        [
            row(
                split="train",
                group="train-static",
                query_id="static-train",
                target_field="current_state_route_set_skill_ids",
                targets=["skill/c"],
                source="matched_tau2",
            )
        ],
    )
    _write_jsonl(
        files["static_route_dev_rows"],
        [
            row(
                split="dev",
                group="dev-static",
                query_id="static-dev",
                target_field="current_state_route_set_skill_ids",
                targets=["skill/a"],
                source="matched_tau2",
            )
        ],
    )
    manifest = {
        "status": "ok",
        "schema_version": "clstr_vnext_semantic_v1",
        "model_input_contract": {
            "history_free_current_state_channel": True,
            "route_query_channel": "compact_causal_skill_action_v1",
            "route_query_history_max_events": 8,
            "raw_tool_results_in_route_query": False,
            "explicit_benchmark_or_source_label_added_to_query": False,
            "executed_skill_ids_in_route_query": True,
            "executed_skill_ids_may_be_namespaced": True,
            "static_dynamic_route_query_identical": True,
            "training_target_semantics": "current_skill_before_memory_update_v1",
        },
        "files": {
            key: {"path": str(path.resolve()), "sha256": _digest(path)}
            for key, path in files.items()
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_matched_corpus_preserves_vnext_targets_splits_and_catalogs(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"

    report = build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train_queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (output_dir / "eval_queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["status"] == "ok"
    assert report["split_group_overlap_count"] == 0
    retrieval = next(row for row in train_rows if row["kind"] == "retrieval")
    assert retrieval["positive_skill_ids"] == ["skill/a", "skill/b"]
    assert "skill/d" not in retrieval["positive_skill_ids"]
    assert retrieval["candidate_skill_ids"] == ["skill/a", "skill/b", "skill/c"]
    assert "causal_skill_action_prefix:" in retrieval["query"]
    assert "actual_result_text" not in retrieval["query"]
    assert {row["split_group_identity"] for row in train_rows}.isdisjoint(
        {row["split_group_identity"] for row in eval_rows}
    )

    corpus = load_prepared_matched_baseline_corpus(output_dir, seed=3)
    loaded_retrieval = next(query for query in corpus.train_queries if query.kind == "retrieval")
    assert loaded_retrieval.positive_skill_ids == ["skill/a", "skill/b"]
    assert loaded_retrieval.positive_indices == [0, 1]
    assert loaded_retrieval.candidate_skill_ids == ["skill/a", "skill/b", "skill/c"]


def test_prepared_corpus_fails_closed_after_file_tampering(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"
    build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )
    with (output_dir / "train_queries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    try:
        load_prepared_matched_baseline_corpus(output_dir)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a digest mismatch")


def test_full_pool_candidates_are_implicit_in_prepared_rows() -> None:
    candidates = ["skill/a", "skill/b"]
    digest = _json_digest(candidates)

    query = _query_from_vnext_row(
        {
            "data_split": "train",
            "split_group_identity": "train-group",
            "split_schema_version": "fixture_split_v1",
            "split_assignment_source": "fixture",
            "query_id": "q0",
            "source": "source-a",
            "benchmark": "toolbench",
            "state_text_current": "need tool a",
            "causal_prefix_events": [],
            "runtime_visible_catalog_id": "public-global",
            "inventory_catalog_digest": digest,
            "required_tool_set_skill_ids": ["skill/a"],
        },
        kind="retrieval",
        split="train",
        candidates_by_catalog={"public-global": candidates},
        known_skill_ids=set(candidates),
        source_manifest_sha256="manifest",
    )

    assert query is not None
    assert "candidate_skill_ids" not in query
    assert query["candidate_scope"] == "all_selected_skills"


def test_prepared_corpus_uses_local_files_after_directory_move(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"
    build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )
    moved_dir = tmp_path / "moved"
    output_dir.rename(moved_dir)

    corpus = load_prepared_matched_baseline_corpus(moved_dir)

    assert len(corpus.train_queries) == 2
    assert len(corpus.eval_queries) == 2


def test_prepared_corpus_rejects_unsupported_semantic_contract(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"
    build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["query_contract"]["training_query_channel"] = "current_only"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        load_prepared_matched_baseline_corpus(output_dir)
    except ValueError as exc:
        assert "query contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsupported query semantics to be rejected")


def test_prepared_corpus_rejects_stale_source_schema(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"
    build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_vnext_manifest"]["schema_version"] = "legacy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        load_prepared_matched_baseline_corpus(output_dir)
    except ValueError as exc:
        assert "source schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a stale prepared source schema to be rejected")


def test_vnext_manifest_relative_paths_resolve_from_manifest_directory(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    for entry in manifest["files"].values():
        entry["path"] = Path(entry["path"]).name
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=tmp_path / "prepared",
    )

    assert report["status"] == "ok"


def test_vnext_manifest_rejects_incompatible_route_query_contract(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["model_input_contract"]["raw_tool_results_in_route_query"] = True
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        build_matched_baseline_corpus(
            vnext_manifest_path=source_manifest,
            output_dir=tmp_path / "prepared",
        )
    except ValueError as exc:
        assert "model-input contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected incompatible query semantics to be rejected")


def test_vnext_manifest_rejects_pre_finalized_schema(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["schema_version"] = "clstr_vnext_fixture_v1"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        build_matched_baseline_corpus(
            vnext_manifest_path=source_manifest,
            output_dir=tmp_path / "prepared",
        )
    except ValueError as exc:
        assert "finalized vNext schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a stale vNext schema to be rejected")


def test_prepared_query_caps_balance_capability_and_source() -> None:
    queries = [
        UnifiedSkillRouterQuery(
            query_id=f"{kind}-{source}-{index}",
            query="query",
            benchmark=source,
            positive_skill_ids=["skill/a"],
            positive_indices=[0],
            source_id=source,
            kind=kind,
        )
        for kind in ("retrieval", "static_route")
        for source in ("tau2", "toolsandbox")
        for index in range(3)
    ]

    selected = _stable_stratified_query_cap(queries, 4, seed=17)

    assert len(selected) == 4
    assert {(query.kind, query.source_id) for query in selected} == {
        ("retrieval", "tau2"),
        ("retrieval", "toolsandbox"),
        ("static_route", "tau2"),
        ("static_route", "toolsandbox"),
    }


def test_prepared_query_caps_balance_capabilities_before_unequal_source_counts() -> None:
    queries = [
        UnifiedSkillRouterQuery(
            query_id=f"retrieval-{source}-{index}",
            query="query",
            benchmark=source,
            positive_skill_ids=["skill/a"],
            positive_indices=[0],
            source_id=source,
            kind="retrieval",
        )
        for source in ("api", "tau2", "toolsandbox")
        for index in range(4)
    ] + [
        UnifiedSkillRouterQuery(
            query_id=f"static-route-{index}",
            query="query",
            benchmark="trajectory",
            positive_skill_ids=["skill/a"],
            positive_indices=[0],
            source_id="trajectory",
            kind="static_route",
        )
        for index in range(8)
    ]

    selected = _stable_stratified_query_cap(queries, 8, seed=17)

    assert sum(query.kind == "retrieval" for query in selected) == 4
    assert sum(query.kind == "static_route" for query in selected) == 4
    assert {query.source_id for query in selected if query.kind == "retrieval"} == {
        "api",
        "tau2",
        "toolsandbox",
    }


def test_prepared_loader_honors_eval_row_cap(tmp_path: Path) -> None:
    source_manifest = _vnext_fixture(tmp_path / "source")
    output_dir = tmp_path / "prepared"
    build_matched_baseline_corpus(
        vnext_manifest_path=source_manifest,
        output_dir=output_dir,
    )

    corpus = load_prepared_matched_baseline_corpus(
        output_dir,
        max_eval_rows=1,
        seed=3,
    )

    assert len(corpus.eval_queries) == 1
    assert corpus.report["source_eval_query_count"] == 2
    assert corpus.report["eval_query_count"] == 1
    assert corpus.report["max_eval_rows"] == 1
