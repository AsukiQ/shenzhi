#!/usr/bin/env python3
"""Build artifact-bound, no-inference evidence used by the CLSTR paper.

The script intentionally operates only on completed route/evaluator artifacts.
It reports factual-vs-perturbed metrics on identical eligible rows, selects the
pre-specified ToolBench examples before reading their text, and verifies the
paired ALFWorld seen/unseen execution protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


SCHEMA_VERSION = "clstr_paper_existing_artifact_evidence_v1"
RAW_DYNAMIC_RANK = "raw_dynamic_end_to_end_route_rank"
STATIC_RANK = "end_to_end_route_rank"
BOOTSTRAP_PROTOCOL = "trajectory_cluster_percentile_bootstrap_seed29_samples2000_v1"
EXPECTED_QWEN_CHECKPOINT_SHA256 = (
    "100479d031f9d69e1d575efae0a2cd1eea1a69d364a75bd14e262ed4ffda8fa5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired-history, ToolBench-case, and ALFWorld paper evidence",
    )
    parser.add_argument("--toolbench_report_path", required=True)
    parser.add_argument("--toolsandbox_report_path", required=True)
    parser.add_argument("--tau2_report_path", required=True)
    parser.add_argument("--toolbench_source_rows_path", required=True)
    parser.add_argument("--alfworld_seen_metrics_path", required=True)
    parser.add_argument("--alfworld_unseen_metrics_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=29)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: float) -> float:
    return float(f"{float(value):.12g}")


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"required JSON artifact is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {resolved}")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"required JSONL artifact is missing: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {resolved}:{line_number}")
            rows.append(row)
    return rows


def rank_value(rank: Any, *, k: int | None) -> float:
    if not isinstance(rank, (int, float)) or float(rank) <= 0.0:
        return 0.0
    if k is None:
        return 1.0 / float(rank)
    return float(float(rank) <= float(k))


def route_rank(row: dict[str, Any], mode: str) -> Any:
    route = row.get(mode)
    if not isinstance(route, dict):
        return None
    return route.get(RAW_DYNAMIC_RANK)


def static_rank(row: dict[str, Any]) -> Any:
    route = row.get("static")
    if not isinstance(route, dict):
        return None
    return route.get(STATIC_RANK)


def eligible_rows(
    rows: Iterable[dict[str, Any]],
    comparison: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row.get("factual"), dict) or not isinstance(
            row.get(comparison),
            dict,
        ):
            continue
        if comparison == "mismatch" and row.get("mismatch_donor_source_index") is None:
            continue
        if comparison == "order_shuffle" and not bool(row.get("order_shuffle_eligible")):
            continue
        selected.append(row)
    return selected


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * float(probability)))
    return canonical(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def paired_metrics(
    rows: list[dict[str, Any]],
    *,
    comparison: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    selected = eligible_rows(rows, comparison)
    if not selected:
        raise ValueError(f"no rows are eligible for {comparison}")
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            raise ValueError("route row lacks trajectory_id")
        clusters[trajectory_id].append(row)
    specs = (("mrr", None), ("recall_at_1", 1), ("recall_at_5", 5))
    cluster_stats: dict[str, dict[str, Any]] = {}
    for trajectory_id, cluster_rows in clusters.items():
        metric_sums: dict[str, dict[str, float]] = {}
        for metric, k in specs:
            factual = math.fsum(
                rank_value(route_rank(row, "factual"), k=k) for row in cluster_rows
            )
            perturbed = math.fsum(
                rank_value(route_rank(row, comparison), k=k) for row in cluster_rows
            )
            metric_sums[metric] = {
                "factual": factual,
                "perturbed": perturbed,
                "delta": factual - perturbed,
            }
        cluster_stats[trajectory_id] = {
            "count": len(cluster_rows),
            "metrics": metric_sums,
        }
    total_count = len(selected)
    result_metrics: dict[str, dict[str, Any]] = {}
    for metric, k in specs:
        factual = math.fsum(
            rank_value(route_rank(row, "factual"), k=k) for row in selected
        ) / total_count
        perturbed = math.fsum(
            rank_value(route_rank(row, comparison), k=k) for row in selected
        ) / total_count
        result_metrics[metric] = {
            "count": total_count,
            "factual": canonical(factual),
            "perturbed": canonical(perturbed),
            "delta": canonical(factual - perturbed),
        }
    cluster_ids = sorted(cluster_stats)
    random_state = random.Random(int(seed))
    bootstrap: dict[str, list[float]] = {metric: [] for metric, _ in specs}
    for _ in range(int(samples)):
        sampled_ids = [random_state.choice(cluster_ids) for _ in cluster_ids]
        sampled_count = sum(cluster_stats[item]["count"] for item in sampled_ids)
        for metric, _ in specs:
            delta_sum = math.fsum(
                cluster_stats[item]["metrics"][metric]["delta"]
                for item in sampled_ids
            )
            bootstrap[metric].append(canonical(delta_sum / sampled_count))
    for metric, _ in specs:
        values = sorted(bootstrap[metric])
        result_metrics[metric]["delta_ci_low"] = percentile(values, 0.025)
        result_metrics[metric]["delta_ci_high"] = percentile(values, 0.975)
    return {
        "comparison": comparison,
        "row_count": total_count,
        "trajectory_count": len(cluster_ids),
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(samples),
        "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
        "rank_field": RAW_DYNAMIC_RANK,
        "metrics": result_metrics,
    }


def load_route_report(
    path: str | Path,
    *,
    benchmark: str,
    require_corrected_v2_release: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    resolved = Path(path).resolve()
    report = load_json(resolved)
    if (
        report.get("status") != "ok"
        or report.get("blockers")
        or report.get("benchmark") != benchmark
        or report.get("evaluation_scope") != "stage2_complete_method"
    ):
        raise ValueError(f"route report is not approved: {resolved}")
    checkpoint = report.get("checkpoint") or {}
    if checkpoint.get("checkpoint_sha256") != EXPECTED_QWEN_CHECKPOINT_SHA256:
        raise ValueError(f"route report uses another Qwen checkpoint: {resolved}")
    release = report.get("stage2_release_selection_contract") or {}
    if release.get("uses_test_metrics") or release.get("uses_benchmark_eval_rows"):
        raise ValueError(f"route report release used evaluation rows: {resolved}")
    if release.get("selected_checkpoint_sha256") != EXPECTED_QWEN_CHECKPOINT_SHA256:
        raise ValueError(f"route report release names another checkpoint: {resolved}")
    if require_corrected_v2_release:
        if release.get("schema_version") != "clstr_vnext_stage2_matched_multibench_selection_v3":
            raise ValueError("ToolSandbox paper evidence is not release-bound to v3")
        corpus = report.get("corpus") or {}
        contract = report.get("corpus_source_contract") or {}
        if corpus.get("task_split") != "test" or contract.get("task_split") != "test":
            raise ValueError("ToolSandbox paper evidence is not corrected-v2 test")
    records_path = Path(str(report.get("route_records_path") or "")).resolve()
    records = load_jsonl(records_path)
    expected_rows = int((report.get("rows") or {}).get("routing_row_count") or 0)
    if len(records) != expected_rows or any(
        row.get("benchmark") != benchmark for row in records
    ):
        raise ValueError(f"route record contract differs: {records_path}")
    artifact = {
        "report_path": str(resolved),
        "report_sha256": file_sha256(resolved),
        "route_records_path": str(records_path),
        "route_records_sha256": file_sha256(records_path),
        "evaluator_source_commit": str(report.get("evaluator_source_commit") or ""),
        "checkpoint_sha256": str(checkpoint.get("checkpoint_sha256") or ""),
        "release_schema": str(release.get("schema_version") or ""),
        "release_selection_path": str(release.get("selection_path") or ""),
        "release_selection_sha256": str(release.get("selection_sha256") or ""),
    }
    return report, records, artifact


def choose_closest_to_median(
    candidates: list[tuple[float, dict[str, Any]]],
) -> tuple[float, dict[str, Any], float]:
    if not candidates:
        raise ValueError("pre-specified case selection has no candidates")
    median = float(statistics.median(value for value, _ in candidates))
    value, row = min(
        candidates,
        key=lambda item: (
            abs(float(item[0]) - median),
            int(item[1].get("source_index") or 0),
        ),
    )
    return float(value), row, median


def build_case(
    route_row: dict[str, Any],
    source_row: dict[str, Any],
    *,
    case_type: str,
    selected_value: float,
    population_median: float,
    population_count: int,
) -> dict[str, Any]:
    if (
        str(route_row.get("trajectory_id") or "")
        != str(source_row.get("trajectory_id") or "")
        or int(route_row.get("step_index") or 0)
        != int(source_row.get("step_index") or 0)
    ):
        raise ValueError("ToolBench source and route identities differ")
    actual_result_executed = bool(source_row.get("actual_result_executed"))
    result_skill = str(source_row.get("actual_result_skill_id") or "")
    executed_skill = str(route_row.get("executed_skill_id") or "")
    result_text = str(source_row.get("actual_result_text") or "")
    if not (
        actual_result_executed
        and result_skill == executed_skill
        and str(source_row.get("actual_result_event_id") or "")
        and result_text
    ):
        raise ValueError("selected ToolBench case lacks an aligned executed result")
    static = static_rank(route_row)
    dynamic = route_rank(route_row, "factual")
    static_route = route_row.get("static") or {}
    factual_route = route_row.get("factual") or {}
    static_support_recalled = bool(static_route.get("candidate_recalled_at_m"))
    dynamic_support_recalled = bool(factual_route.get("candidate_recalled_at_m"))
    effect = (
        "dynamic-only candidate recovery"
        if not static_support_recalled and dynamic_support_recalled
        else "state-conditioned rank change"
    )
    return {
        "case_type": case_type,
        "selection_rule": (
            "closest observed value to the population median; ties by source_index"
        ),
        "population_count": population_count,
        "population_median": canonical(population_median),
        "selected_value": canonical(selected_value),
        "source_index": int(route_row.get("source_index") or 0),
        "trajectory_id": str(route_row.get("trajectory_id") or ""),
        "step_index": int(route_row.get("step_index") or 0),
        "goal_text": str(source_row.get("goal_text") or ""),
        "authorized_prior_history": str(source_row.get("history_text") or ""),
        "executed_skill_id": executed_skill,
        "executed_action": str(source_row.get("action_text") or ""),
        "aligned_result": result_text,
        "gold_next_skill_id": str(route_row.get("target_skill_id") or ""),
        "static_rank": static,
        "dynamic_rank": dynamic,
        "static_candidate_support_recalled": static_support_recalled,
        "dynamic_candidate_support_recalled": dynamic_support_recalled,
        "route_effect": effect,
    }


def select_toolbench_cases(
    route_rows: list[dict[str, Any]],
    source_rows_path: str | Path,
) -> dict[str, Any]:
    # Selection deliberately precedes loading any natural-language source row.
    recovery: list[tuple[float, dict[str, Any]]] = []
    harmful: list[tuple[float, dict[str, Any]]] = []
    for row in route_rows:
        static_route = row.get("static") or {}
        factual_route = row.get("factual") or {}
        static_rr = rank_value(static_rank(row), k=None)
        dynamic_rr = rank_value(route_rank(row, "factual"), k=None)
        if (
            static_rr == 0.0
            and dynamic_rr > 0.0
            and not bool(static_route.get("candidate_recalled_at_m"))
            and bool(factual_route.get("candidate_recalled_at_m"))
        ):
            recovery.append((dynamic_rr, row))
        if dynamic_rr < static_rr:
            harmful.append((static_rr - dynamic_rr, row))
    recovery_value, recovery_row, recovery_median = choose_closest_to_median(recovery)
    harmful_value, harmful_row, harmful_median = choose_closest_to_median(harmful)
    source_path = Path(source_rows_path).resolve()
    source_rows = load_jsonl(source_path)
    source_by_identity = {
        (str(row.get("trajectory_id") or ""), int(row.get("step_index") or 0)): row
        for row in source_rows
    }
    recovery_key = (
        str(recovery_row.get("trajectory_id") or ""),
        int(recovery_row.get("step_index") or 0),
    )
    harmful_key = (
        str(harmful_row.get("trajectory_id") or ""),
        int(harmful_row.get("step_index") or 0),
    )
    if recovery_key not in source_by_identity or harmful_key not in source_by_identity:
        raise ValueError("selected ToolBench case is absent from the bound source rows")
    return {
        "route_rank_field": RAW_DYNAMIC_RANK,
        "source_rows_path": str(source_path),
        "source_rows_sha256": file_sha256(source_path),
        "selection_reads_trace_after_ranking": True,
        "positive": build_case(
            recovery_row,
            source_by_identity[recovery_key],
            case_type="median_static_candidate_miss_recovered",
            selected_value=recovery_value,
            population_median=recovery_median,
            population_count=len(recovery),
        ),
        "failure": build_case(
            harmful_row,
            source_by_identity[harmful_key],
            case_type="median_reciprocal_rank_loss",
            selected_value=harmful_value,
            population_median=harmful_median,
            population_count=len(harmful),
        ),
    }


ALFWORLD_MATCHED_PROTOCOL_FIELDS = (
    "policy_family",
    "method_release",
    "matched_release_selection_path",
    "stage2_checkpoint_path",
    "stage2_checkpoint_sha256",
    "stage2_checkpoint_step",
    "training_skills_path",
    "training_skills_sha256",
    "executor_model",
    "executor_is_qwen3_14b",
    "executor_interface",
    "scoring_method",
    "qwen_weight",
    "clstr_weight",
    "skill_guidance_top_k",
    "clstr_route_mode",
    "qwen_score_mode",
    "score_normalization",
    "uses_clstr_score_fusion",
    "uses_clstr_skill_guidance",
    "guidance_conditions_qwen",
    "loop_guard",
    "candidate_skill_schema",
    "guidance_candidate_skill_schema",
    "runtime_pseudo_skills_allowed",
    "history_removed_from_clstr_current_state",
    "memory_protocol",
    "memory_update_skill_mode",
    "post_action_result_correction",
)


def alfworld_split_evidence(
    metrics_path: str | Path,
    *,
    expected_split: str,
    expected_episodes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(metrics_path).resolve()
    metrics = load_json(resolved)
    if (
        metrics.get("status") != "ok"
        or metrics.get("split") != expected_split
        or int(metrics.get("episode_count") or 0) != expected_episodes
        or metrics.get("stage2_checkpoint_sha256") != EXPECTED_QWEN_CHECKPOINT_SHA256
    ):
        raise ValueError(f"ALFWorld metrics contract differs: {resolved}")
    run_path = resolved.with_name("run.jsonl")
    run_rows = load_jsonl(run_path)
    if len(run_rows) != expected_episodes:
        raise ValueError(f"ALFWorld run row count differs: {run_path}")
    indices = [int(row.get("episode_index")) for row in run_rows]
    if len(set(indices)) != expected_episodes or sorted(indices) != list(
        range(expected_episodes)
    ):
        raise ValueError(f"ALFWorld episode identities differ: {run_path}")
    if any(row.get("split") != expected_split for row in run_rows):
        raise ValueError(f"ALFWorld run mixes splits: {run_path}")
    successes = sum(int(bool(row.get("success"))) for row in run_rows)
    exact_rate = successes / expected_episodes
    if not math.isclose(
        float(metrics.get("success_rate") or 0.0),
        exact_rate,
        abs_tol=5e-7,
    ):
        raise ValueError(f"ALFWorld metrics and run rows disagree: {resolved}")
    evidence = {
        "split": expected_split,
        "successes": successes,
        "episodes": expected_episodes,
        "success_rate": canonical(exact_rate),
        "runtime_appended_skill_count": int(
            metrics.get("runtime_appended_skill_count") or 0
        ),
        "metrics_path": str(resolved),
        "metrics_sha256": file_sha256(resolved),
        "run_path": str(run_path),
        "run_sha256": file_sha256(run_path),
    }
    return metrics, evidence


def paired_alfworld_evidence(
    seen_path: str | Path,
    unseen_path: str | Path,
) -> dict[str, Any]:
    seen_metrics, seen = alfworld_split_evidence(
        seen_path,
        expected_split="valid_seen",
        expected_episodes=140,
    )
    unseen_metrics, unseen = alfworld_split_evidence(
        unseen_path,
        expected_split="valid_unseen",
        expected_episodes=134,
    )
    differences = {
        field: {
            "seen": seen_metrics.get(field),
            "unseen": unseen_metrics.get(field),
        }
        for field in ALFWORLD_MATCHED_PROTOCOL_FIELDS
        if seen_metrics.get(field) != unseen_metrics.get(field)
    }
    if differences:
        raise ValueError(f"ALFWorld seen/unseen protocols differ: {differences}")
    return {
        "protocol_match": True,
        "matched_protocol_fields": list(ALFWORLD_MATCHED_PROTOCOL_FIELDS),
        "runtime_inventory_is_split_observed": True,
        "seen": seen,
        "unseen": unseen,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CLSTR existing-artifact paper evidence",
        "",
        "## Paired history interventions (core always-dynamic route)",
        "",
        "| Benchmark | Eligible subset | Rows | Traj. | Factual MRR | Perturbed MRR | Delta | 95% cluster CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, blocks in payload["paired_history"].items():
        for comparison, block in blocks.items():
            metric = block["metrics"]["mrr"]
            lines.append(
                "| {benchmark} | {comparison} | {rows} | {trajectories} | "
                "{factual:.4f} | {perturbed:.4f} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] |".format(
                    benchmark=benchmark,
                    comparison=comparison,
                    rows=block["row_count"],
                    trajectories=block["trajectory_count"],
                    factual=metric["factual"],
                    perturbed=metric["perturbed"],
                    delta=metric["delta"],
                    low=metric["delta_ci_low"],
                    high=metric["delta_ci_high"],
                )
            )
    lines.extend(["", "## Deterministic ToolBench cases", ""])
    for name in ("positive", "failure"):
        case = payload["toolbench_cases"][name]
        lines.extend(
            [
                f"### {name.title()}",
                "",
                f"- Identity: `{case['trajectory_id']}`, step {case['step_index']}, source row {case['source_index']}",
                f"- Executed: `{case['executed_skill_id']}` with `{case['executed_action'].strip()}`",
                f"- Aligned result: `{case['aligned_result']}`",
                f"- Gold next skill: `{case['gold_next_skill_id']}`",
                f"- Static rank -> dynamic rank: `{case['static_rank']}` -> `{case['dynamic_rank']}`",
                f"- Effect: {case['route_effect']}",
                "",
            ]
        )
    alfworld = payload["alfworld"]
    lines.extend(
        [
            "## ALFWorld paired protocol",
            "",
            "| Split | Successes | Episodes | Success rate | Runtime-appended skills |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("seen", "unseen"):
        row = alfworld[name]
        lines.append(
            f"| {row['split']} | {row['successes']} | {row['episodes']} | "
            f"{row['success_rate']:.6f} | {row['runtime_appended_skill_count']} |"
        )
    lines.extend(
        [
            "",
            "Seen and unseen share every declared executor/retriever protocol field; "
            "their runtime-appended exact-action inventories are observed separately per split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reports = {
        "toolbench_g3": args.toolbench_report_path,
        "toolsandbox": args.toolsandbox_report_path,
        "tau2": args.tau2_report_path,
    }
    route_rows: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for benchmark, path in reports.items():
        _, rows, artifact = load_route_report(
            path,
            benchmark=benchmark,
            require_corrected_v2_release=benchmark == "toolsandbox",
        )
        route_rows[benchmark] = rows
        artifacts[benchmark] = artifact
    paired_history = {
        benchmark: {
            comparison: paired_metrics(
                rows,
                comparison=comparison,
                seed=args.bootstrap_seed,
                samples=args.bootstrap_samples,
            )
            for comparison in ("mismatch", "order_shuffle")
        }
        for benchmark, rows in route_rows.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "checkpoint_sha256": EXPECTED_QWEN_CHECKPOINT_SHA256,
        "route_semantics": "core_always_dynamic_pre_selector",
        "rank_field": RAW_DYNAMIC_RANK,
        "artifacts": artifacts,
        "paired_history": paired_history,
        "toolbench_cases": select_toolbench_cases(
            route_rows["toolbench_g3"],
            args.toolbench_source_rows_path,
        ),
        "alfworld": paired_alfworld_evidence(
            args.alfworld_seen_metrics_path,
            args.alfworld_unseen_metrics_path,
        ),
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    if output_json.exists() or output_md.exists():
        raise ValueError("paper-evidence outputs must be fresh")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
