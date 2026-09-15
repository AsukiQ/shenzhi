#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_data import (
    MATCHED_HISTORY_DATA_SCHEMA,
    MatchedHistoryTrajectorySkip,
    prepare_matched_history_trajectory,
)
from clstr.matched_multibench_data import route_rows_to_training_trajectories
from clstr.vnext_training import file_sha256, load_inventory_catalogs
from scripts.build_clstr_vnext_views import _canonical_trajectory_rows


TEST_DATA_SCHEMA = "clstr_matched_history_e1_test_data_v1"


def _git_identity(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if status.strip():
        raise RuntimeError("formal E1 test-data build requires a clean source worktree")
    return {"source_commit": commit, "source_worktree_clean": True}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"matched-history test source is empty: {path}")
    return rows


def _selected_skill_ids(path: Path) -> set[str]:
    output: set[str] = set()
    for row in _read_jsonl(path):
        skill_id = str(
            row.get("skill_id")
            or row.get("canonical_skill_id")
            or row.get("id")
            or ""
        ).strip()
        if not skill_id or skill_id in output:
            raise ValueError("selected skill table has an empty or duplicate identity")
        output.add(skill_id)
    return output


def _trajectory_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    output = {
        str(row.get("trajectory_id") or "").strip()
        for row in rows
        if str(row.get("trajectory_id") or "").strip()
    }
    if not output:
        raise ValueError("matched-history rows lack trajectory identities")
    return output


def _compact_trajectory_ids(path: Path) -> set[str]:
    return _trajectory_ids(_read_jsonl(path))


def _catalog_digest(catalogs: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(
        catalogs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _install_training_global_parent_alias(
    catalogs: dict[str, dict[str, Any]],
) -> str:
    if "public_global_67k" in catalogs:
        raise ValueError("training catalogs unexpectedly retain the parent global pool")
    children = [
        catalog_id
        for catalog_id, catalog in catalogs.items()
        if str(catalog.get("inventory_parent_catalog_id") or "")
        == "public_global_67k"
    ]
    if len(children) != 1:
        raise ValueError("training catalogs do not identify one global parent derivation")
    child_id = children[0]
    alias = dict(catalogs[child_id])
    alias["inventory_catalog_id"] = "public_global_67k"
    catalogs["public_global_67k"] = alias
    return child_id


def _rewrite_global_parent_to_training_child(
    rows: list[dict[str, Any]],
    *,
    catalogs: dict[str, dict[str, Any]],
    child_id: str,
) -> None:
    child = catalogs[child_id]
    for row in rows:
        if str(row.get("runtime_visible_catalog_id") or "") != "public_global_67k":
            continue
        row["runtime_visible_catalog_id"] = child_id
        row["inventory_catalog_digest"] = str(
            child.get("inventory_catalog_digest") or ""
        )
        row["inventory_source"] = str(child.get("inventory_source") or "")
        row["inventory_pool_size"] = int(child.get("inventory_pool_size") or 0)
        row["inventory_parent_catalog_id"] = "public_global_67k"
        row["inventory_parent_catalog_digest"] = str(
            child.get("inventory_parent_catalog_digest") or ""
        )
        row["inventory_derivation"] = "dynamic_skill_training_antijoin_v1"


def _test_manifest_sha256(rows: list[dict[str, Any]]) -> str:
    values = {
        str(row.get("matched_split_manifest_sha256") or "").strip()
        for row in rows
    }
    if len(values) != 1 or not next(iter(values)):
        raise ValueError("tau2 test rows do not share one split manifest")
    if {str(row.get("matched_data_split") or "") for row in rows} != {"test"}:
        raise ValueError("tau2 source is not the protected test split")
    return next(iter(values))


def build_test_data(args: argparse.Namespace) -> dict[str, Any]:
    git_identity = _git_identity(ROOT)
    paths = {
        "tau2": Path(args.tau2_test_rows_path).resolve(),
        "toolbench_g3": Path(args.toolbench_test_rows_path).resolve(),
        "skills": Path(args.skills_path).resolve(),
        "catalogs": Path(args.inventory_catalogs_path).resolve(),
        "train": Path(args.train_rows_path).resolve(),
        "dev": Path(args.dev_rows_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise ValueError(f"matched-history test data inputs are missing: {missing}")
    output_path = Path(args.output_path).resolve()
    output_catalogs_path = Path(args.output_catalogs_path).resolve()
    report_path = Path(args.report_path).resolve()
    if (
        output_path in set(paths.values())
        or output_catalogs_path in set(paths.values())
        or report_path in set(paths.values())
    ):
        raise ValueError("matched-history test outputs may not overwrite inputs")
    if output_path.exists() or output_catalogs_path.exists() or report_path.exists():
        raise ValueError("matched-history test outputs must not already exist")

    tau2_source = _read_jsonl(paths["tau2"])
    toolbench_source = _read_jsonl(paths["toolbench_g3"])
    if {str(row.get("benchmark") or "") for row in tau2_source} != {"tau2"}:
        raise ValueError("tau2 test source contains another benchmark")
    if {str(row.get("benchmark") or "") for row in toolbench_source} != {
        "toolbench_g3"
    }:
        raise ValueError("ToolBench test source contains another benchmark")
    if {
        str(row.get("official_eval_split") or "") for row in toolbench_source
    } != {"eval"}:
        raise ValueError("ToolBench source is not the protected evaluation split")
    tau2_manifest_sha256 = _test_manifest_sha256(tau2_source)

    tau2_current_events = route_rows_to_training_trajectories(
        tau2_source,
        benchmark="tau2",
        split_by_trajectory={},
        group_identity_by_trajectory={},
        split_manifest_sha256=tau2_manifest_sha256,
        evaluation_split="test",
    )
    if any("locked_data_split" in row for row in tau2_current_events):
        raise RuntimeError("test materialization unexpectedly created training locks")

    selected_skill_ids = _selected_skill_ids(paths["skills"])
    catalogs = load_inventory_catalogs(paths["catalogs"])
    catalog_digest_before = _catalog_digest(catalogs)
    global_training_catalog_id = _install_training_global_parent_alias(catalogs)
    canonical_input_catalog_ids = set(catalogs)
    canonical, canonical_counts = _canonical_trajectory_rows(
        [*toolbench_source, *tau2_current_events],
        catalogs,
        selected_skill_ids,
    )
    created_catalog_ids = set(catalogs) - canonical_input_catalog_ids
    invalid_created_catalog_ids = {
        catalog_id
        for catalog_id in created_catalog_ids
        if not catalog_id.startswith("runtime_declared_")
    }
    if invalid_created_catalog_ids:
        raise ValueError(
            "test materialization created a non-runtime catalog: "
            f"{sorted(invalid_created_catalog_ids)[:4]}"
        )
    _rewrite_global_parent_to_training_child(
        canonical,
        catalogs=catalogs,
        child_id=global_training_catalog_id,
    )
    del catalogs["public_global_67k"]
    unchanged_training_catalogs = {
        catalog_id: catalog
        for catalog_id, catalog in catalogs.items()
        if catalog_id not in created_catalog_ids
    }
    if _catalog_digest(unchanged_training_catalogs) != catalog_digest_before:
        raise ValueError("test materialization changed a training catalog")
    declared_test_catalog_skill_sets = {
        tuple(
            sorted(
                {
                    str(item)
                    for item in row.get("candidate_next_skill_ids") or []
                    if str(item)
                }
            )
        )
        for row in tau2_source
    }
    for catalog_id in created_catalog_ids:
        catalog = catalogs[catalog_id]
        catalog_skills = tuple(
            sorted(
                {
                    str(item)
                    for item in catalog.get("runtime_visible_skill_ids") or []
                    if str(item)
                }
            )
        )
        if (
            not catalog_skills
            or catalog_skills not in declared_test_catalog_skill_sets
            or not set(catalog_skills).issubset(selected_skill_ids)
        ):
            raise ValueError(
                "test-only catalog is not an evaluator-visible selected-skill set"
            )
    for row in canonical:
        row["data_split"] = "test"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        grouped[str(row["trajectory_id"])].append(row)
    train_ids = _compact_trajectory_ids(paths["train"])
    dev_ids = _compact_trajectory_ids(paths["dev"])
    test_ids = set(grouped)
    if train_ids & dev_ids or test_ids & train_ids or test_ids & dev_ids:
        raise ValueError("matched-history train/dev/test trajectories overlap")

    benchmark_rows: Counter[str] = Counter()
    benchmark_trajectories: Counter[str] = Counter()
    benchmark_decisions: Counter[str] = Counter()
    quarantined: Counter[str] = Counter()
    skipped_trajectories: Counter[str] = Counter()
    skipped_rows: Counter[str] = Counter()
    compact_rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(grouped):
        raw_trajectory = grouped[trajectory_id]
        try:
            compact, trajectory_quarantine = prepare_matched_history_trajectory(
                raw_trajectory,
                selected_skill_ids=selected_skill_ids,
                catalogs=catalogs,
            )
        except MatchedHistoryTrajectorySkip as error:
            skipped_trajectories[error.reason] += 1
            skipped_rows[error.reason] += len(raw_trajectory)
            continue
        benchmark = str(compact[0]["benchmark"])
        benchmark_rows[benchmark] += len(compact)
        benchmark_trajectories[benchmark] += 1
        benchmark_decisions[benchmark] += len(compact) - 1
        quarantined.update(trajectory_quarantine)
        compact_rows.extend(compact)
    if set(benchmark_decisions) != {"toolbench_g3", "tau2"} or any(
        value <= 0 for value in benchmark_decisions.values()
    ):
        raise ValueError("matched-history test data lacks a history-bearing benchmark")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in compact_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    output_catalogs_path.parent.mkdir(parents=True, exist_ok=True)
    with output_catalogs_path.open("w", encoding="utf-8") as handle:
        for catalog_id in sorted(catalogs):
            handle.write(
                json.dumps(
                    catalogs[catalog_id],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    report = {
        "schema_version": TEST_DATA_SCHEMA,
        "compact_schema_version": MATCHED_HISTORY_DATA_SCHEMA,
        "status": "ok",
        **git_identity,
        "source_row_count_by_benchmark": {
            "tau2": len(tau2_source),
            "toolbench_g3": len(toolbench_source),
        },
        "row_count": len(compact_rows),
        "trajectory_count": sum(benchmark_trajectories.values()),
        "history_bearing_decision_count": sum(benchmark_decisions.values()),
        "row_count_by_benchmark": dict(sorted(benchmark_rows.items())),
        "trajectory_count_by_benchmark": dict(
            sorted(benchmark_trajectories.items())
        ),
        "history_bearing_decision_count_by_benchmark": dict(
            sorted(benchmark_decisions.items())
        ),
        "result_quarantine_by_reason": dict(sorted(quarantined.items())),
        "skipped_trajectory_count_by_reason": dict(
            sorted(skipped_trajectories.items())
        ),
        "skipped_row_count_by_reason": dict(sorted(skipped_rows.items())),
        "canonicalization_counts": canonical_counts,
        "test_only_catalogs": {
            catalog_id: {
                "inventory_catalog_digest": str(
                    catalogs[catalog_id].get("inventory_catalog_digest") or ""
                ),
                "inventory_pool_size": int(
                    catalogs[catalog_id].get("inventory_pool_size") or 0
                ),
                "derivation": "evaluator_visible_candidate_next_skill_ids",
            }
            for catalog_id in sorted(created_catalog_ids)
        },
        "split_audit": {
            "train_trajectory_count": len(train_ids),
            "dev_trajectory_count": len(dev_ids),
            "test_source_trajectory_count": len(test_ids),
            "train_dev_overlap": 0,
            "train_test_overlap": 0,
            "dev_test_overlap": 0,
            "checkpoint_selection_uses_test": False,
        },
        "contract": {
            "tau2_temporal_shift": (
                "next_skill_id_target_action_next_observation_to_current_event"
            ),
            "toolbench_temporal_semantics": "current_executed_event",
            "representation_prefix_uses_rows_strictly_before_decision": True,
            "history_bearing_decisions_only": True,
            "maximum_horizon_applied_by_evaluator": 16,
            "candidate_support": "bounded_last8_skill_action_frozen_static_top500",
            "test_checkpoint_selection_forbidden": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sorted(paths.items())
        },
        "tau2_split_manifest_sha256": tau2_manifest_sha256,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "output_catalogs_path": str(output_catalogs_path),
        "output_catalogs_sha256": file_sha256(output_catalogs_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2_test_rows_path", required=True)
    parser.add_argument("--toolbench_test_rows_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--inventory_catalogs_path", required=True)
    parser.add_argument("--train_rows_path", required=True)
    parser.add_argument("--dev_rows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_catalogs_path", required=True)
    parser.add_argument("--report_path", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_test_data(args),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
