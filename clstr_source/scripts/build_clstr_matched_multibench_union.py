#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_multibench_data import (
    TAU2_SPLIT_SCHEMA,
    TOOLSANDBOX_SPLIT_SCHEMA,
    build_tau2_split_manifest,
    build_toolsandbox_split_manifest,
    canonical_digest,
    canonicalize_matched_route_rows,
    route_rows_to_training_trajectories,
    tau2_row_split,
)
from clstr.tau2_successful_rollouts import (
    TAU2_DEFAULT_AGENT_MODELS,
    load_tau2_successful_rollout_corpus,
)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(base_path: Path, output_path: Path, rows: Iterable[dict[str, Any]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    base_has_content = base_path.stat().st_size > 0
    base_ends_with_newline = not base_has_content or _ends_with_newline(base_path)
    with output_path.open("wb") as target:
        with base_path.open("rb") as source:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        if base_has_content and not base_ends_with_newline:
            target.write(b"\n")
        for row in rows:
            target.write(
                (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            count += 1
    return count


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"\n"


def _merge_skills(
    base_path: Path,
    additions: Iterable[dict[str, Any]],
    output_path: Path,
) -> tuple[int, int]:
    additions_by_id: dict[str, dict[str, Any]] = {}
    for row in additions:
        skill_id = str(row.get("skill_id") or "").strip()
        if not skill_id:
            raise ValueError("matched benchmark skill lacks skill_id")
        prior = additions_by_id.get(skill_id)
        if prior is not None:
            if canonical_digest(prior) != canonical_digest(row):
                raise ValueError(f"conflicting matched skill definition: {skill_id}")
            continue
        additions_by_id[skill_id] = row
    seen_ids: set[str] = set()
    base_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl(base_path):
            skill_id = str(
                row.get("skill_id") or row.get("canonical_skill_id") or ""
            ).strip()
            if not skill_id or skill_id in seen_ids:
                raise ValueError("base skill pool contains an invalid/duplicate skill ID")
            seen_ids.add(skill_id)
            base_count += 1
            matched = additions_by_id.get(skill_id)
            if matched is not None and canonical_digest(matched) != canonical_digest(row):
                raise ValueError(f"conflicting matched skill definition: {skill_id}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        appended = [
            row
            for skill_id, row in additions_by_id.items()
            if skill_id not in seen_ids
        ]
        for row in appended:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return base_count, len(appended)


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = str(row.get("matched_data_split") or "")
        if split not in output:
            raise ValueError("matched row has no valid split")
        output[split].append(row)
    return output


def build_union(args: argparse.Namespace) -> dict[str, Any]:
    # These evaluator modules import Torch, so loading is intentionally delayed
    # until the compute-node data build rather than storage-node CLI parsing.
    from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus

    base_root = Path(args.base_data_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"matched union output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl"):
        if not (base_root / name).is_file():
            raise FileNotFoundError(f"base data is missing {name}")

    tau_agent_models = tuple(
        item.strip()
        for item in str(args.tau2_agent_models).split(",")
        if item.strip()
    )
    if not tau_agent_models:
        raise ValueError("tau2_agent_models must contain at least one model")
    tau_train = load_tau2_successful_rollout_corpus(
        args.tau2_data_root,
        results_root=args.tau2_results_root,
        domains=("airline", "retail", "telecom"),
        task_split="train",
        agent_models=tau_agent_models,
    )
    tau_test = load_tau2_successful_rollout_corpus(
        args.tau2_data_root,
        results_root=args.tau2_results_root,
        domains=("airline", "retail", "telecom"),
        task_split="test",
        agent_models=tau_agent_models,
    )
    tau_manifest = build_tau2_split_manifest(
        train_rows=tau_train.source_rows,
        test_rows=tau_test.source_rows,
        train_task_keys=[
            f"{domain}/{task_id}"
            for domain, task_ids in (
                tau_train.report.get("official_split_task_ids_by_domain") or {}
            ).items()
            for task_id in task_ids
        ],
        test_task_keys=[
            f"{domain}/{task_id}"
            for domain, task_ids in (
                tau_test.report.get("official_split_task_ids_by_domain") or {}
            ).items()
            for task_id in task_ids
        ],
        seed=args.split_seed,
        dev_fraction=args.tau2_dev_fraction,
    )
    tau_all_rows = [*tau_train.source_rows, *tau_test.source_rows]
    tau_split_by_trajectory: dict[str, str] = {}
    tau_group_by_trajectory: dict[str, str] = {}
    for row in tau_all_rows:
        trajectory = str(row["trajectory_id"])
        split = tau2_row_split(tau_manifest, row)
        provenance = row.get("provenance") or {}
        task_key = f"{row.get('domain')}/{provenance.get('raw_task_id')}"
        tau_split_by_trajectory[trajectory] = split
        tau_group_by_trajectory[trajectory] = canonical_digest(
            {"schema": TAU2_SPLIT_SCHEMA, "task": task_key}
        )
    tau_routes = canonicalize_matched_route_rows(
        tau_all_rows,
        split_by_trajectory=tau_split_by_trajectory,
        group_identity_by_trajectory=tau_group_by_trajectory,
        split_manifest_sha256=str(tau_manifest["manifest_sha256"]),
    )

    tools = load_toolsandbox_route_corpus(
        scenarios_root=args.toolsandbox_scenarios_root,
        tools_root=args.toolsandbox_tools_root,
    )
    tools_manifest = build_toolsandbox_split_manifest(
        tools.source_rows,
        seed=args.split_seed,
        train_fraction=args.toolsandbox_train_fraction,
        dev_fraction=args.toolsandbox_dev_fraction,
    )
    tools_split_by_trajectory: dict[str, str] = {}
    tools_group_by_trajectory: dict[str, str] = {}
    records = tools_manifest["scenario_records"]
    for row in tools.source_rows:
        provenance = row.get("provenance") or {}
        scenario = str(provenance.get("scenario_name") or "")
        trajectory = str(row["trajectory_id"])
        tools_split_by_trajectory[trajectory] = str(
            tools_manifest["scenario_to_split"][scenario]
        )
        tools_group_by_trajectory[trajectory] = str(records[scenario]["family_id"])
    tools_routes = canonicalize_matched_route_rows(
        tools.source_rows,
        split_by_trajectory=tools_split_by_trajectory,
        group_identity_by_trajectory=tools_group_by_trajectory,
        split_manifest_sha256=str(tools_manifest["manifest_sha256"]),
    )

    route_splits = {
        "tau2": _split_rows(tau_routes),
        "toolsandbox": _split_rows(tools_routes),
    }
    trainable_routes = [
        *route_splits["tau2"]["train"],
        *route_splits["tau2"]["dev"],
        *route_splits["toolsandbox"]["train"],
        *route_splits["toolsandbox"]["dev"],
    ]
    split_by_trajectory = {
        **tau_split_by_trajectory,
        **tools_split_by_trajectory,
    }
    group_by_trajectory = {
        **tau_group_by_trajectory,
        **tools_group_by_trajectory,
    }
    manifest_by_benchmark = {
        "tau2": str(tau_manifest["manifest_sha256"]),
        "toolsandbox": str(tools_manifest["manifest_sha256"]),
    }
    training_rows: list[dict[str, Any]] = []
    for benchmark in ("tau2", "toolsandbox"):
        benchmark_trainable = [
            row for row in trainable_routes if row.get("benchmark") == benchmark
        ]
        training_rows.extend(
            route_rows_to_training_trajectories(
                benchmark_trainable,
                benchmark=benchmark,
                split_by_trajectory=split_by_trajectory,
                group_identity_by_trajectory=group_by_trajectory,
                split_manifest_sha256=manifest_by_benchmark[benchmark],
            )
        )

    split_root = output_root / "matched_splits"
    _write_json(split_root / "tau2_split_manifest.json", tau_manifest)
    _write_json(split_root / "toolsandbox_split_manifest.json", tools_manifest)
    route_files: dict[str, dict[str, Any]] = {}
    for benchmark, splits in route_splits.items():
        benchmark_skills = (
            [*tau_train.skills, *tau_test.skills]
            if benchmark == "tau2"
            else tools.skills
        )
        unique_skills: dict[str, dict[str, Any]] = {}
        for skill in benchmark_skills:
            skill_id = str(skill.get("skill_id") or "")
            prior = unique_skills.get(skill_id)
            if prior is not None and canonical_digest(prior) != canonical_digest(skill):
                raise ValueError(f"benchmark skill definition conflict: {skill_id}")
            unique_skills[skill_id] = skill
        skills_path = split_root / f"{benchmark}_skills.jsonl"
        _write_jsonl(skills_path, unique_skills.values())
        route_files[f"{benchmark}_skills"] = {
            "path": str(skills_path),
            "sha256": _file_sha256(skills_path),
            "rows": len(unique_skills),
        }
        for split, rows in splits.items():
            path = split_root / f"{benchmark}_{split}_rows.jsonl"
            count = _write_jsonl(path, rows)
            route_files[f"{benchmark}_{split}"] = {
                "path": str(path),
                "sha256": _file_sha256(path),
                "rows": count,
            }

    base_skill_count, appended_skill_count = _merge_skills(
        base_root / "skill_pool.jsonl",
        [*tau_train.skills, *tau_test.skills, *tools.skills],
        output_root / "skill_pool.jsonl",
    )
    shutil.copyfile(base_root / "retrieval.jsonl", output_root / "retrieval.jsonl")
    appended_trajectory_count = _append_jsonl(
        base_root / "trajectories.jsonl",
        output_root / "trajectories.jsonl",
        training_rows,
    )
    output_files = {
        name: {
            "path": str(output_root / name),
            "sha256": _file_sha256(output_root / name),
        }
        for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl")
    }
    report = {
        "schema_version": "clstr_matched_multibench_union_v1",
        "status": "ok",
        "base_data_root": str(base_root),
        "base_files_sha256": {
            name: _file_sha256(base_root / name)
            for name in ("skill_pool.jsonl", "retrieval.jsonl", "trajectories.jsonl")
        },
        "split_seed": args.split_seed,
        "tau2_split_manifest": tau_manifest,
        "toolsandbox_split_manifest": tools_manifest,
        "counts": {
            "base_skill_count": base_skill_count,
            "appended_skill_count": appended_skill_count,
            "appended_training_trajectory_rows": appended_trajectory_count,
            "matched_route_rows_by_benchmark_split": {
                benchmark: {
                    split: len(rows)
                    for split, rows in splits.items()
                    if split in {"train", "dev"}
                }
                for benchmark, splits in route_splits.items()
            },
            "appended_training_trajectory_rows_by_benchmark_split": {
                benchmark: {
                    split: sum(
                        int(
                            row.get("benchmark") == benchmark
                            and row.get("locked_data_split") == split
                        )
                        for row in training_rows
                    )
                    for split in ("train", "dev")
                }
                for benchmark in ("tau2", "toolsandbox")
            },
            "excluded_no_call_rows_by_benchmark_split": {
                benchmark: {
                    split: len(route_splits[benchmark][split])
                    - sum(
                        int(
                            row.get("benchmark") == benchmark
                            and row.get("locked_data_split") == split
                        )
                        for row in training_rows
                    )
                    for split in ("train", "dev")
                }
                for benchmark in ("tau2", "toolsandbox")
            },
        },
        "route_files": route_files,
        "output_files": output_files,
        "model_input_contract": {
            "history_free_h_t": False,
            "history_free_current_state_channel": True,
            "route_query_channel": (
                "current_state_plus_bounded_skill_action_prefix_no_results_v1"
            ),
            "route_query_history_max_events": 8,
            "raw_tool_results_in_route_query": False,
            "static_dynamic_route_query_identical": True,
            "recurrent_memory_carries_result_correction": True,
            "explicit_benchmark_or_source_label_added_to_query": False,
            "executed_skill_ids_in_route_query": True,
            "executed_skill_ids_may_be_namespaced": True,
            "tau2_memory": "agent_visible_successful_rollout_actual_tool_results",
            "toolsandbox_memory": "dag_aware_action_only_no_tool_result",
            "tau2_reference_actions_used_as_route_labels": False,
            "tau2_user_simulator_private_fields_in_route_query": False,
            "toolsandbox_route_protocol": "milestone_dag_topological_multi_positive_v1",
            "no_call_stop_rows_appended_to_training": False,
            "test_rows_appended_to_training": False,
            "training_target_semantics": (
                "current_skill_before_memory_update_v1"
            ),
            "equivalent_positive_semantics": (
                "current_target_equivalents_v1"
            ),
        },
        "tau2_successful_rollout_reports": {
            "train": tau_train.report,
            "test": tau_test.report,
        },
    }
    report["manifest_sha256"] = canonical_digest(report)
    _write_json(output_root / "matched_union_manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one leakage-aware Tau2/ToolSandbox matched CLSTR union"
    )
    parser.add_argument("--base_data_root", required=True)
    parser.add_argument("--tau2_data_root", required=True)
    parser.add_argument(
        "--tau2_results_root",
        help="Defaults to <tau2_data_root>/results/final.",
    )
    parser.add_argument(
        "--tau2_agent_models",
        default=",".join(TAU2_DEFAULT_AGENT_MODELS),
        help="Comma-separated official result-file agent model prefixes.",
    )
    parser.add_argument("--toolsandbox_scenarios_root", required=True)
    parser.add_argument("--toolsandbox_tools_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_seed", default="clstr-matched-v1")
    parser.add_argument("--tau2_dev_fraction", type=float, default=0.2)
    parser.add_argument("--toolsandbox_train_fraction", type=float, default=0.6)
    parser.add_argument("--toolsandbox_dev_fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    report = build_union(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
