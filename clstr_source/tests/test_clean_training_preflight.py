from __future__ import annotations

import json
from pathlib import Path

from clstr.clean_training_preflight import (
    audit_clean_training_preflight,
    audit_incremental_matched_union_preflight,
)
from clstr.history_channel import materialize_history_free_state
from clstr.matched_multibench_data import canonical_digest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "trajectories.jsonl":
        rows = [
            materialize_history_free_state(row, replace_state_text=True)
            for row in rows
        ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_unified_root(root: Path, *, leaked: bool) -> Path:
    eval_path = root.parent / "toolbench_eval.jsonl"
    eval_row = {
        "benchmark": "toolbench_g3",
        "task_id": "toolbench-g3-10018::0",
        "trajectory_id": "toolbench-g3-10018",
        "state_text": "eval state",
        "skill_id": "toolbench-g3/weather/current",
        "next_skill_id": "toolbench-g3/weather/alerts",
        "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/10018_ChatGPT_DFS_woFilter_w2.json"},
    }
    _write_jsonl(eval_path, [eval_row])
    _write_jsonl(
        root / "skill_pool.jsonl",
        [
            {"skill_id": "toolbench-g3/weather/current", "source": "ToolBench-G3"},
            {"skill_id": "toolbench-g3/weather/alerts", "source": "ToolBench-G3"},
            {"skill_id": "alfworld/go", "source": "alfworld"},
        ],
    )
    if leaked:
        toolbench_row = dict(eval_row)
        retrieval_query_id = "toolbench-g3-10018"
    else:
        toolbench_row = {
            "benchmark": "toolbench_g3",
            "task_id": "toolbench-g3-7::0",
            "trajectory_id": "toolbench-g3-7",
            "state_text": "safe state",
            "skill_id": "toolbench-g3/weather/current",
            "next_skill_id": "toolbench-g3/weather/current",
            "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/7_ChatGPT_DFS_woFilter_w2.json"},
        }
        retrieval_query_id = "toolbench-g3-7"
    _write_jsonl(
        root / "trajectories.jsonl",
        [
            toolbench_row,
            {
                "benchmark": "alfworld",
                "task_id": "alf-1::0",
                "trajectory_id": "alf-1",
                "state_text": "alf state",
                "skill_id": "alfworld/go",
                "next_skill_id": "alfworld/go",
            },
        ],
    )
    _write_jsonl(
        root / "retrieval.jsonl",
        [
            {
                "source": "toolbench_g3",
                "query_id": retrieval_query_id,
                "query_text": "weather query",
                "positive_skill_id": "toolbench-g3/weather/current",
            },
            {
                "source": "trajectory_derived_alfworld",
                "query_id": "alf-1::0",
                "query_text": "alf query",
                "positive_skill_id": "alfworld/go",
            },
        ],
    )
    return eval_path


def test_clean_training_preflight_fails_on_toolbench_eval_leakage(tmp_path):
    root = tmp_path / "contaminated"
    eval_path = _write_unified_root(root, leaked=True)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
    )

    assert report["status"] == "error"
    assert report["leakage"]["toolbench_eval"]["trajectory_hits"]["task_id_hits"] == 1
    assert report["leakage"]["toolbench_eval"]["retrieval_hits"]["query_id_hits"] == 1


def test_clean_training_preflight_accepts_clean_toolbench_root_and_reports_composition(tmp_path):
    root = tmp_path / "clean"
    eval_path = _write_unified_root(root, leaked=False)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
    )

    assert report["status"] == "ok", report
    assert report["composition"]["files"] == {
        "trajectories_rows": 2,
        "retrieval_rows": 2,
        "skill_pool_rows": 3,
    }
    assert report["leakage"]["toolbench_eval"]["trajectory_hits"] == {}
    assert report["leakage"]["toolbench_eval"]["retrieval_hits"] == {}
    assert report["leakage"]["toolbench_eval"]["trajectory_content_overlap"]["near_duplicate_hits"] == 0
    assert report["leakage"]["toolbench_eval"]["retrieval_content_overlap"]["near_duplicate_hits"] == 0


def test_vnext_clean_preflight_requires_structured_current_and_binds_hashes(tmp_path):
    root = tmp_path / "structured_clean"
    eval_path = _write_unified_root(root, leaked=False)
    rows = [
        json.loads(line)
        for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for index, row in enumerate(rows):
        row["goal_text"] = f"structured goal {index}"
    _write_jsonl(root / "trajectories.jsonl", rows)
    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
        require_structured_current_state=True,
    )
    assert report["status"] == "ok"
    assert report["preflight_contract"]["structured_current_state_required"] is True
    assert report["history_channel"]["trajectory_stream"][
        "structured_current_state_rows"
    ] == 2
    assert set(report["composition"]["files_sha256"]) == {
        "skill_pool.jsonl",
        "retrieval.jsonl",
        "trajectories.jsonl",
    }


def test_clean_training_preflight_fails_on_toolbench_content_near_duplicate_without_id_overlap(tmp_path):
    root = tmp_path / "near_duplicate"
    eval_path = _write_unified_root(root, leaked=False)
    _write_jsonl(
        eval_path,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-eval::0",
                "trajectory_id": "toolbench-g3-eval",
                "state_text": "Find weather alerts for Paris today and summarize the warnings.",
                "action_text": "call weather.alerts",
                "next_observation_text": "Warnings returned.",
                "skill_id": "toolbench-g3/weather/current",
                "next_skill_id": "toolbench-g3/weather/alerts",
                "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/eval.json"},
            }
        ],
    )
    rows = [json.loads(line) for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0] = {
        **rows[0],
        "task_id": "toolbench-g3-train-safe::0",
        "trajectory_id": "toolbench-g3-train-safe",
        "state_text": "find weather alerts for paris today summarize warnings",
        "state_text_current": "find weather alerts for paris today summarize warnings",
        "state_text_full": "find weather alerts for paris today summarize warnings",
        "action_text": "call weather alerts",
        "next_observation_text": "warning results returned",
        "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/train-safe.json"},
    }
    _write_jsonl(root / "trajectories.jsonl", rows)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
    )

    assert report["status"] == "error"
    overlap = report["leakage"]["toolbench_eval"]["trajectory_content_overlap"]
    assert overlap["near_duplicate_hits"] == 1
    assert overlap["examples"][0]["train_id"] == "toolbench-g3-train-safe::0"
    assert overlap["matched_train_row_count"] == 1
    assert overlap["matched_train_rows"][0]["trajectory_id"] == "toolbench-g3-train-safe"
    assert len(overlap["matched_train_rows"][0]["train_row_sha256"]) == 64


def test_clean_training_preflight_can_treat_near_duplicates_as_diagnostic(tmp_path):
    root = tmp_path / "near_duplicate_diagnostic"
    eval_path = _write_unified_root(root, leaked=False)
    _write_jsonl(
        eval_path,
        [
            {
                "benchmark": "toolbench_g3",
                "task_id": "toolbench-g3-eval::0",
                "trajectory_id": "toolbench-g3-eval",
                "state_text": "Find weather alerts for Paris today and summarize the warnings.",
                "action_text": "call weather.alerts",
                "next_observation_text": "Warnings returned.",
                "skill_id": "toolbench-g3/weather/current",
                "next_skill_id": "toolbench-g3/weather/alerts",
                "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/eval.json"},
            }
        ],
    )
    rows = [json.loads(line) for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0] = {
        **rows[0],
        "task_id": "toolbench-g3-train-safe::0",
        "trajectory_id": "toolbench-g3-train-safe",
        "state_text": "find weather alerts for paris today summarize warnings",
        "state_text_current": "find weather alerts for paris today summarize warnings",
        "state_text_full": "find weather alerts for paris today summarize warnings",
        "action_text": "call weather alerts",
        "next_observation_text": "warning results returned",
        "provenance": {"answer_path": "/tmp/ToolBench/data/answer/G3_answer/train-safe.json"},
    }
    _write_jsonl(root / "trajectories.jsonl", rows)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
        fail_on_near_duplicate=False,
    )

    assert report["status"] == "warning"
    assert report["leakage_policy"]["exact_leakage_blocks_training"] is True
    assert report["leakage_policy"]["near_duplicate_blocks_training"] is False
    assert report["leakage_policy"]["has_exact_leakage"] is False
    assert report["leakage_policy"]["has_near_duplicate"] is True


def test_clean_training_preflight_content_overlap_reports_runtime_controls(tmp_path):
    root = tmp_path / "controlled_overlap"
    eval_path = _write_unified_root(root, leaked=False)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=eval_path,
        near_duplicate_max_chars=24,
        near_duplicate_max_shingle_postings=1,
        near_duplicate_max_query_features=3,
        near_duplicate_max_candidates_per_row=2,
    )

    overlap = report["leakage"]["toolbench_eval"]["trajectory_content_overlap"]
    assert overlap["near_duplicate_feature_family"] == "token_unigram_bigram_trigram_containment"
    assert overlap["near_duplicate_max_chars"] == 24
    assert overlap["near_duplicate_max_shingle_postings"] == 1
    assert overlap["near_duplicate_max_query_features"] == 3
    assert overlap["near_duplicate_max_candidates_per_row"] == 2
    assert "skipped_high_posting_shingles" in overlap
    assert "candidate_pairs_scored" in overlap


def test_clean_training_preflight_blocks_matched_toolsandbox_family_overlap(tmp_path):
    root = tmp_path / "matched_family_leak"
    toolbench_eval = _write_unified_root(root, leaked=False)
    shared_group = "a" * 64
    rows = [
        json.loads(line)
        for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows.append(
        {
            "benchmark": "toolsandbox",
            "task_id": "toolsandbox/train::0",
            "trajectory_id": "toolsandbox/train",
            "goal_text": "send a message",
            "state_text": "goal: send a message",
            "skill_id": "toolsandbox/send_message",
            "next_skill_id": "",
            "split_semantic_text": "send a message",
            "locked_split_group_identity": shared_group,
        }
    )
    _write_jsonl(root / "trajectories.jsonl", rows)
    test_rows = tmp_path / "toolsandbox_test.jsonl"
    _write_jsonl(
        test_rows,
        [
            {
                "benchmark": "toolsandbox",
                "task_id": "toolsandbox/test::0",
                "trajectory_id": "toolsandbox/test",
                "state_text": "different held-out wording",
                "state_text_current": "different held-out wording",
                "split_semantic_text": "different held-out wording",
                "matched_split_group_identity": shared_group,
            }
        ],
    )
    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=toolbench_eval,
        toolsandbox_test_rows_path=test_rows,
    )
    assert report["status"] == "error"
    assert report["leakage"]["toolsandbox_test"]["group_overlap_count"] == 1


def test_clean_training_preflight_accepts_disjoint_matched_test_groups(tmp_path):
    root = tmp_path / "matched_clean"
    toolbench_eval = _write_unified_root(root, leaked=False)
    rows = [
        json.loads(line)
        for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows.append(
        {
            "benchmark": "tau2",
            "task_id": "tau2/train::0",
            "trajectory_id": "tau2/train",
            "goal_text": "book a flight",
            "state_text": "goal: book a flight",
            "skill_id": "tau2/airline/book",
            "next_skill_id": "",
            "split_semantic_text": "book a flight",
            "locked_split_group_identity": "a" * 64,
        }
    )
    _write_jsonl(root / "trajectories.jsonl", rows)
    test_rows = tmp_path / "tau2_test.jsonl"
    _write_jsonl(
        test_rows,
        [
            {
                "benchmark": "tau2",
                "task_id": "tau2/test::0",
                "trajectory_id": "tau2/test",
                "state_text": "refund an order",
                "state_text_current": "refund an order",
                "split_semantic_text": "refund an order",
                "matched_split_group_identity": "b" * 64,
            }
        ],
    )
    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=toolbench_eval,
        tau2_test_rows_path=test_rows,
    )
    assert report["status"] == "ok"
    assert report["leakage"]["tau2_test"]["group_overlap_count"] == 0


def test_incremental_matched_preflight_reuses_digest_bound_clean_base(tmp_path):
    base = tmp_path / "base"
    toolbench_eval = _write_unified_root(base, leaked=False)
    base_rows = [
        json.loads(line)
        for line in (base / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for index, row in enumerate(base_rows):
        row["goal_text"] = f"base goal {index}"
    _write_jsonl(base / "trajectories.jsonl", base_rows)
    base_report_path = base / "clean_preflight.json"
    base_report = audit_clean_training_preflight(
        data_root=base,
        toolbench_eval_trajectories_path=toolbench_eval,
        output_path=base_report_path,
        require_structured_current_state=True,
    )
    assert base_report["status"] == "ok"

    union = tmp_path / "union"
    union.mkdir()
    base_skills = (base / "skill_pool.jsonl").read_text(encoding="utf-8")
    (union / "skill_pool.jsonl").write_text(
        base_skills + json.dumps({"skill_id": "tau2/airline/book"}) + "\n"
        + json.dumps({"skill_id": "toolsandbox/send_message"}) + "\n",
        encoding="utf-8",
    )
    (union / "retrieval.jsonl").write_bytes((base / "retrieval.jsonl").read_bytes())
    appended = [
        {
            "benchmark": benchmark,
            "trajectory_id": f"{benchmark}/train",
            "task_id": f"{benchmark}/train::0",
            "step_index": 0,
            "goal_text": f"train {benchmark}",
            "state_text": f"goal: train {benchmark}",
            "state_text_current": f"goal: train {benchmark}",
            "skill_id": skill,
            "next_skill_id": "",
            "action_text": "execute",
            "next_observation_text": (
                "actual successful tool result" if benchmark == "tau2" else ""
            ),
            "observation_source": (
                "tau2_official_successful_rollout_tool_result"
                if benchmark == "tau2"
                else "action_only_no_tool_result"
            ),
            "provenance": {
                "source_id": (
                    "tau2_official_successful_rollout_v1"
                    if benchmark == "tau2"
                    else "matched_toolsandbox_action_only_v1"
                )
            },
            "split_semantic_text": f"train {benchmark}",
            "locked_data_split": "train",
            "locked_split_group_identity": group,
        }
        for benchmark, skill, group in (
            ("tau2", "tau2/airline/book", "a" * 64),
            ("toolsandbox", "toolsandbox/send_message", "c" * 64),
        )
    ]
    with (union / "trajectories.jsonl").open("wb") as handle:
        handle.write((base / "trajectories.jsonl").read_bytes())
        for row in appended:
            handle.write((json.dumps(row) + "\n").encode())

    def sha(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "clstr_matched_multibench_union_v1",
        "status": "ok",
        "base_data_root": str(base.resolve()),
        "base_files_sha256": base_report["composition"]["files_sha256"],
        "counts": {
            "base_skill_count": 3,
            "appended_skill_count": 2,
            "appended_training_trajectory_rows": 2,
        },
        "output_files": {
            name: {"path": str((union / name).resolve()), "sha256": sha(union / name)}
            for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl")
        },
        "tau2_split_manifest": {
            "schema_version": "clstr_tau2_official_test_train_dev_split_v1",
            "seed": "test",
            "dev_fraction_within_official_train": 0.2,
            "official_test_preserved": True,
            "domains": ["airline"],
            "task_to_split": {
                "airline/train": "train",
                "airline/test": "test",
            },
            "splits": {
                "dev": [],
                "test": ["airline/test"],
                "train": ["airline/train"],
            },
            "task_count_by_split": {"dev": 0, "test": 1, "train": 1},
        },
    }
    manifest["tau2_split_manifest"]["manifest_sha256"] = canonical_digest(
        manifest["tau2_split_manifest"]
    )
    manifest["manifest_sha256"] = canonical_digest(manifest)
    manifest_path = union / "matched_union_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tau_test = tmp_path / "tau_test.jsonl"
    tools_test = tmp_path / "tools_test.jsonl"
    _write_jsonl(
        tau_test,
        [
            {
                "benchmark": "tau2",
                "trajectory_id": "tau2/test",
                "state_text_current": "refund order",
                "split_semantic_text": "refund order",
                "matched_split_group_identity": "b" * 64,
            }
        ],
    )
    _write_jsonl(
        tools_test,
        [
            {
                "benchmark": "toolsandbox",
                "trajectory_id": "toolsandbox/test",
                "state_text_current": "search weather",
                "split_semantic_text": "search weather",
                "matched_split_group_identity": "d" * 64,
            }
        ],
    )
    report = audit_incremental_matched_union_preflight(
        data_root=union,
        base_data_root=base,
        base_clean_preflight_report=base_report_path,
        matched_union_manifest_path=manifest_path,
        toolbench_eval_trajectories_path=toolbench_eval,
        traject_eval_queries_path=None,
        tau2_test_rows_path=tau_test,
        toolsandbox_test_rows_path=tools_test,
    )
    assert report["status"] == "ok", report
    assert report["preflight_contract"]["incremental_matched_union_verified"] is True
    assert report["incremental_union_audit"]["appended_trajectory_count"] == 2


def test_incremental_preflight_discloses_official_tau2_prompt_reuse(tmp_path):
    base = tmp_path / "base"
    toolbench_eval = _write_unified_root(base, leaked=False)
    base_rows = [
        json.loads(line)
        for line in (base / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for index, row in enumerate(base_rows):
        row["goal_text"] = f"base goal {index}"
    _write_jsonl(base / "trajectories.jsonl", base_rows)
    base_report_path = base / "clean_preflight.json"
    base_report = audit_clean_training_preflight(
        data_root=base,
        toolbench_eval_trajectories_path=toolbench_eval,
        output_path=base_report_path,
        require_structured_current_state=True,
    )

    union = tmp_path / "union"
    union.mkdir()
    (union / "skill_pool.jsonl").write_bytes((base / "skill_pool.jsonl").read_bytes())
    (union / "retrieval.jsonl").write_bytes((base / "retrieval.jsonl").read_bytes())
    appended = {
        "benchmark": "tau2",
        "trajectory_id": "tau2/airline/train",
        "task_id": "tau2/airline/train::0",
        "step_index": 0,
        "goal_text": "shared official prompt",
        "state_text": "goal: shared official prompt",
        "state_text_current": "goal: shared official prompt",
        "skill_id": "skill_a",
        "next_skill_id": "",
        "action_text": "execute",
        "next_observation_text": "actual successful tool result",
        "observation_source": "tau2_official_successful_rollout_tool_result",
        "provenance": {"source_id": "tau2_official_successful_rollout_v1"},
        "split_semantic_text": "shared official prompt",
        "locked_data_split": "train",
        "locked_split_group_identity": "a" * 64,
    }
    with (union / "trajectories.jsonl").open("wb") as handle:
        handle.write((base / "trajectories.jsonl").read_bytes())
        handle.write((json.dumps(appended) + "\n").encode())

    def sha(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    tau_manifest = {
        "schema_version": "clstr_tau2_official_test_train_dev_split_v1",
        "seed": "test",
        "dev_fraction_within_official_train": 0.2,
        "official_test_preserved": True,
        "domains": ["airline"],
        "task_to_split": {
            "airline/train": "train",
            "airline/test": "test",
        },
        "splits": {
            "dev": [],
            "test": ["airline/test"],
            "train": ["airline/train"],
        },
        "task_count_by_split": {"dev": 0, "test": 1, "train": 1},
    }
    tau_manifest["manifest_sha256"] = canonical_digest(tau_manifest)
    manifest = {
        "schema_version": "clstr_matched_multibench_union_v1",
        "status": "ok",
        "base_data_root": str(base.resolve()),
        "base_files_sha256": base_report["composition"]["files_sha256"],
        "counts": {
            "base_skill_count": 3,
            "appended_skill_count": 0,
            "appended_training_trajectory_rows": 1,
        },
        "output_files": {
            name: {"path": str((union / name).resolve()), "sha256": sha(union / name)}
            for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl")
        },
        "tau2_split_manifest": tau_manifest,
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    manifest_path = union / "matched_union_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tau_test = tmp_path / "tau_test.jsonl"
    tools_test = tmp_path / "tools_test.jsonl"
    _write_jsonl(
        tau_test,
        [
            {
                "benchmark": "tau2",
                "trajectory_id": "tau2/airline/test",
                "state_text_current": "goal: shared official prompt",
                "split_semantic_text": "shared official prompt",
                "matched_split_group_identity": "b" * 64,
            }
        ],
    )
    _write_jsonl(tools_test, [])
    report = audit_incremental_matched_union_preflight(
        data_root=union,
        base_data_root=base,
        base_clean_preflight_report=base_report_path,
        matched_union_manifest_path=manifest_path,
        toolbench_eval_trajectories_path=toolbench_eval,
        traject_eval_queries_path=None,
        tau2_test_rows_path=tau_test,
        toolsandbox_test_rows_path=tools_test,
    )
    assert report["status"] == "ok"
    assert report["leakage_policy"]["has_near_duplicate"] is True
    assert report["leakage_policy"]["has_blocking_near_duplicate"] is False
    assert report["leakage"]["tau2_test"]["content_overlap_policy"]["blocking"] is False


def test_clean_training_preflight_fails_on_traject_eval_overlap(tmp_path):
    root = tmp_path / "traject_contaminated"
    toolbench_eval_path = _write_unified_root(root, leaked=False)
    traject_eval_dir = tmp_path / "traject_eval"
    _write_jsonl(
        traject_eval_dir / "queries.jsonl",
        [
            {
                "query_id": "traject::parallel::Music::hard::7::0",
                "query_text": "goal: make a playlist\nprevious_tools: <empty>",
                "trajectory_id": "traject::parallel::Music::hard::7",
            }
        ],
    )
    _write_jsonl(
        traject_eval_dir / "qrels.jsonl",
        [
            {
                "query_id": "traject::parallel::Music::hard::7::0",
                "skill_id": "traject/music/search",
                "relevance": 1,
            }
        ],
    )
    rows = [json.loads(line) for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "benchmark": "traject_bench",
            "task_id": "traject::parallel::Music::hard::7::0",
            "trajectory_id": "traject::parallel::Music::hard::7",
            "state_text": "goal: make a playlist\nprevious_tools: <empty>",
            "skill_id": "traject/music/search",
            "next_skill_id": "traject/music/detail",
        }
    )
    _write_jsonl(root / "trajectories.jsonl", rows)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=toolbench_eval_path,
        traject_eval_queries_path=traject_eval_dir / "queries.jsonl",
    )

    assert report["status"] == "error"
    hits = report["leakage"]["trajectbench_eval"]["trajectory_hits"]
    assert hits["trajectory_id_hits"] == 1
    assert hits["task_id_hits"] == 1
    assert hits["signature_hits"] == 1


def test_clean_training_preflight_fails_on_traject_content_near_duplicate_without_id_overlap(tmp_path):
    root = tmp_path / "traject_near_duplicate"
    toolbench_eval_path = _write_unified_root(root, leaked=False)
    traject_eval_dir = tmp_path / "traject_eval_near"
    _write_jsonl(
        traject_eval_dir / "queries.jsonl",
        [
            {
                "query_id": "traject::parallel::Music::hard::eval::0",
                "query_text": "goal: create a playlist from relaxing jazz songs previous tools empty",
                "trajectory_id": "traject::parallel::Music::hard::eval",
            }
        ],
    )
    rows = [json.loads(line) for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "benchmark": "traject_bench",
            "task_id": "traject::parallel::Music::hard::train::0",
            "trajectory_id": "traject::parallel::Music::hard::train",
            "state_text": "Goal create playlist relaxing jazz songs previous tools are empty",
            "skill_id": "traject/music/search",
            "next_skill_id": "traject/music/detail",
        }
    )
    _write_jsonl(root / "trajectories.jsonl", rows)

    report = audit_clean_training_preflight(
        data_root=root,
        toolbench_eval_trajectories_path=toolbench_eval_path,
        traject_eval_queries_path=traject_eval_dir / "queries.jsonl",
    )

    assert report["status"] == "error"
    overlap = report["leakage"]["trajectbench_eval"]["trajectory_content_overlap"]
    assert overlap["near_duplicate_hits"] == 1
    assert overlap["examples"][0]["train_id"] == "traject::parallel::Music::hard::train::0"
