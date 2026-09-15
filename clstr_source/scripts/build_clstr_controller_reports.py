#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _metric(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "N/A"
    return f"{number:.4f}"


def _display_path(output_root: Path, path: Path) -> str:
    try:
        return str(Path(output_root.name) / path.relative_to(output_root))
    except ValueError:
        return str(path)


def _row_from_metrics(
    *,
    output_root: Path,
    benchmark: str,
    method: str,
    path: Path,
    training_data: str,
    trajectory_policy_trained: bool,
    controller_complete_decision: bool,
    components_in_action_selection: bool,
    l_policy_source: str,
    note: str,
    closed_loop_evaluated: bool | None = None,
) -> dict[str, Any]:
    metrics = _read_json(path)
    episodes = _int_or_zero(_metric(metrics, "episodes", "episode_count"))
    status = str(metrics.get("status") or ("ok" if metrics else "missing"))
    if closed_loop_evaluated is None:
        closed_loop_evaluated = status == "ok" and episodes > 0
    return {
        "benchmark": benchmark,
        "method": method,
        "training_data": training_data,
        "trajectory_policy_trained": trajectory_policy_trained,
        "controller_complete_decision": controller_complete_decision,
        "transition_belief_stop_in_action_selection": components_in_action_selection,
        "L_policy_source": l_policy_source,
        "status": status,
        "closed_loop_evaluated": bool(closed_loop_evaluated),
        "success_rate": _float_or_none(_metric(metrics, "success_rate")),
        "average_reward": _float_or_none(_metric(metrics, "average_reward", "avg_reward")),
        "average_goal_condition_points": _float_or_none(
            _metric(metrics, "average_goal_condition_points", "avg_goal_condition_points")
        ),
        "average_episode_steps": _float_or_none(_metric(metrics, "average_episode_steps", "avg_steps")),
        "episodes": episodes,
        "metrics_or_audit_path": _display_path(output_root, path),
        "note": note,
    }


def _dbbench_row(output_root: Path) -> dict[str, Any]:
    path = output_root / "dbbench_eval" / "harness_audit_report.json"
    audit = _read_json(path)
    status = str(audit.get("status") or "missing")
    missing = audit.get("missing_items") or []
    return {
        "benchmark": "dbbench",
        "method": "dbbench_controller_gate",
        "training_data": "u-10bei/dbbench SFT is registered only; no eval harness training/eval claim",
        "trajectory_policy_trained": False,
        "controller_complete_decision": False,
        "transition_belief_stop_in_action_selection": False,
        "L_policy_source": "none for closed-loop eval",
        "status": status,
        "closed_loop_evaluated": False,
        "success_rate": None,
        "average_reward": None,
        "average_goal_condition_points": None,
        "average_episode_steps": None,
        "episodes": 0,
        "metrics_or_audit_path": _display_path(output_root, path),
        "note": "blocked by missing items: " + ", ".join(missing) if missing else "blocked: no closed-loop harness audited",
        "missing_items": missing,
        "reproducible_commands": audit.get("reproducible_commands") or {},
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "benchmark",
        "method",
        "training data",
        "trajectory/policy trained",
        "controller-complete decision",
        "transition/belief/STOP in action selection",
        "L_policy source",
        "status",
        "closed-loop evaluated",
        "success_rate",
        "average_reward",
        "average_goal_condition_points",
        "average_episode_steps",
        "episodes",
        "metrics/audit path",
        "note",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        values = [
            row["benchmark"],
            row["method"],
            row["training_data"],
            str(row["trajectory_policy_trained"]),
            str(row["controller_complete_decision"]),
            str(row["transition_belief_stop_in_action_selection"]),
            row["L_policy_source"],
            row["status"],
            str(row["closed_loop_evaluated"]),
            _fmt(row["success_rate"]),
            _fmt(row["average_reward"]),
            _fmt(row["average_goal_condition_points"]),
            _fmt(row["average_episode_steps"]),
            str(row["episodes"]),
            row["metrics_or_audit_path"],
            row["note"],
        ]
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in values) + " |")
    lines.extend(
        [
            "",
            "Caveats:",
            "- Closed-loop metrics come only from actual environment rollouts; offline CE/recall/MRR are not success.",
            "- Full-base training means losses were optimized; controller-complete decision means transition/belief/STOP/loop scores were used during action ranking.",
            "- ALFWorld valid_seen/valid_unseen are eval-only; WebShop test split data is not used for training and cannot be reused as same-split test evidence.",
            "- DBBench remains a blocker/audit row, not a benchmark result.",
        ]
    )
    return "\n".join(lines) + "\n"


def _paper_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    by_method = {row["method"]: row for row in rows}
    alf = by_method["clstr_controller_gate"]
    baseline = by_method["routing_init_baseline"]
    improved = (
        _float_or_none(alf["success_rate"]) is not None
        and _float_or_none(baseline["success_rate"]) is not None
        and float(alf["success_rate"]) > float(baseline["success_rate"])
    )
    sci = by_method["scienceworld_controller_gate"]
    web = by_method["webshop_controller_gate"]
    dbb = by_method["dbbench_controller_gate"]
    lines = [
        "# CLSTR Controller-Complete Closed-Loop Report",
        "",
        "## What Changed",
        "",
        "This stage separates full-base training from controller-complete decision making. Full-base training optimized `L_policy`, `L_trans`, belief, STOP, and routing losses. Controller-complete evaluation ranks admissible actions with policy, transition, belief, STOP, and loop-penalty terms instead of only `argmax(policy_score)`.",
        "",
        "## ALFWorld",
        "",
        f"- Metrics: {alf['metrics_or_audit_path']}",
        "- Run trace: outputs/alfworld_eval/clstr_controller_gate_valid_seen/run.jsonl",
        "- Controller diagnostic: outputs/alfworld_eval/clstr_controller_gate_valid_seen/controller_diagnostic.json",
        "- Blocker: outputs/alfworld_eval/clstr_controller_gate_valid_seen/blocker_report.json",
        "",
        (
            "Result: ALFWorld controller gate improved over routing init."
            if improved
            else "Result: ALFWorld valid_seen controller gate did not improve over routing init."
        ),
        f"success_rate={_fmt(alf['success_rate'])}, average_reward={_fmt(alf['average_reward'])}, average_goal_condition_points={_fmt(alf['average_goal_condition_points'])}, average_episode_steps={_fmt(alf['average_episode_steps'])}, episodes={alf['episodes']}.",
        "The ALFWorld trace records per-candidate policy_score, transition_score, belief_score, STOP probability, loop penalty, and final_score.",
        "",
        "## ScienceWorld",
        "",
        "- Smoke report: outputs/scienceworld_eval/harness_smoke_report.json",
        f"- Eval metrics: {sci['metrics_or_audit_path']}",
        "- Run trace: outputs/scienceworld_eval/clstr_controller_gate/run.jsonl",
        "",
        (
            f"ScienceWorld: official smoke OK; gate closed-loop eval ran for {sci['episodes']} episode(s), success_rate={_fmt(sci['success_rate'])}, average_reward={_fmt(sci['average_reward'])}, average_episode_steps={_fmt(sci['average_episode_steps'])}."
            if sci["status"] == "ok" and sci["closed_loop_evaluated"]
            else f"ScienceWorld: {sci['status']}; no benchmark success is reported."
        ),
        "",
        "## WebShop",
        "",
        "- Smoke report: outputs/webshop_eval/harness_smoke_report.json",
        f"- Eval metrics: {web['metrics_or_audit_path']}",
        "- Run trace: outputs/webshop_eval/clstr_controller_gate/run.jsonl",
        "",
        (
            f"WebShop: official smoke OK; gate closed-loop eval ran for {web['episodes']} episode(s), success_rate={_fmt(web['success_rate'])}, average_reward={_fmt(web['average_reward'])}, average_episode_steps={_fmt(web['average_episode_steps'])}. WebShop test split train-then-same-split eval remains forbidden."
            if web["status"] == "ok" and web["closed_loop_evaluated"]
            else f"WebShop: {web['status']}; no benchmark success is reported."
        ),
        "",
        "## DBBench",
        "",
        f"- Harness audit: {dbb['metrics_or_audit_path']}",
        f"DBBench: {dbb['status']}. The registered SFT/ReAct dataset is not an official closed-loop harness, so no DBBench success metric is reported.",
        "",
        "## Paper-Safe Interpretation",
        "",
        "- Offline recall/CE/MRR are diagnostics only.",
        "- ALFWorld, ScienceWorld, and WebShop rows above are closed-loop environment rollouts when their status is `ok` and `closed_loop_evaluated=true`.",
        "- DBBench remains a hard blocker until an official or reproducible closed-loop harness is installed and smoke-tested.",
        "- See `outputs/clstr_controller_comparison_table.md` and `outputs/clstr_controller_comparison_summary.json`.",
    ]
    if summary.get("component_trace_example"):
        lines.extend(["", "## Component Trace Example", "", json.dumps(summary["component_trace_example"], ensure_ascii=False)[:2000]])
    return "\n".join(lines) + "\n"


def _load_component_trace_example(output_root: Path) -> dict[str, Any] | None:
    run_path = output_root / "alfworld_eval" / "clstr_controller_gate_valid_seen" / "run.jsonl"
    if not run_path.exists():
        return None
    try:
        with run_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                steps = row.get("steps") or []
                if isinstance(steps, list) and steps:
                    first = steps[0]
                    scores = first.get("component_scores") or []
                    return {
                        "benchmark": "alfworld",
                        "chosen_action": first.get("action"),
                        "chosen_reason": first.get("chosen_reason"),
                        "top_component_scores": scores[:3],
                    }
                score_trace = row.get("component_score_trace") or []
                if isinstance(score_trace, list) and score_trace:
                    first_scores = score_trace[0] or []
                    return {
                        "benchmark": "alfworld",
                        "chosen_action": (row.get("chosen_action_trace") or row.get("action_trace") or [None])[0],
                        "chosen_reason": (row.get("chosen_reason_trace") or [None])[0],
                        "top_component_scores": sorted(
                            first_scores,
                            key=lambda item: item.get("final_score", item.get("policy_score", float("-inf"))),
                            reverse=True,
                        )[:3],
                    }
    except Exception:  # noqa: BLE001
        return None
    return None


def build_controller_reports(
    *,
    output_root: str | Path = "outputs",
    output_table_path: str | Path = "outputs/clstr_controller_comparison_table.md",
    output_summary_path: str | Path = "outputs/clstr_controller_comparison_summary.json",
    output_paper_md_path: str | Path = "outputs/clstr_controller_paper_report.md",
    output_paper_json_path: str | Path = "outputs/clstr_controller_paper_report.json",
) -> dict[str, Any]:
    root = Path(output_root)
    rows = [
        _row_from_metrics(
            output_root=root,
            benchmark="alfworld",
            method="routing_init_baseline",
            path=root / "alfworld_eval" / "clstr_routing_init_baseline" / "metrics.json",
            training_data="SKILLRET/CLSTR native routing init only",
            trajectory_policy_trained=False,
            controller_complete_decision=False,
            components_in_action_selection=False,
            l_policy_source="none",
            note="routing/static retrieval baseline; closed-loop ALFWorld rollout",
            closed_loop_evaluated=True,
        ),
        _row_from_metrics(
            output_root=root,
            benchmark="alfworld",
            method="alfworld_policy_only_gate",
            path=root / "alfworld_eval" / "clstr_policy_gate_valid_seen" / "metrics.json",
            training_data="ALFWorld train replay only",
            trajectory_policy_trained=True,
            controller_complete_decision=False,
            components_in_action_selection=False,
            l_policy_source="L_policy CE over admissible_commands_t",
            note="policy-only closed-loop gate; offline recall is not success",
            closed_loop_evaluated=True,
        ),
        _row_from_metrics(
            output_root=root,
            benchmark="alfworld",
            method="clstr_full_base_action_scorer_gate",
            path=root / "alfworld_eval" / "clstr_full_base_gate" / "metrics.json",
            training_data="CLSTR full-base train_allowed rows",
            trajectory_policy_trained=True,
            controller_complete_decision=False,
            components_in_action_selection=False,
            l_policy_source="ALFWorld train replay L_policy plus component losses",
            note="decision used action scorer only",
            closed_loop_evaluated=True,
        ),
        _row_from_metrics(
            output_root=root,
            benchmark="alfworld",
            method="clstr_controller_gate",
            path=root / "alfworld_eval" / "clstr_controller_gate_valid_seen" / "metrics.json",
            training_data="CLSTR full-base checkpoint with controller-complete decision",
            trajectory_policy_trained=True,
            controller_complete_decision=True,
            components_in_action_selection=True,
            l_policy_source="ALFWorld train replay L_policy plus transition/belief/STOP/loop scoring",
            note="transition/belief/STOP/loop participated in action selection",
            closed_loop_evaluated=True,
        ),
        _row_from_metrics(
            output_root=root,
            benchmark="scienceworld",
            method="scienceworld_controller_gate",
            path=root / "scienceworld_eval" / "clstr_controller_gate" / "metrics.json",
            training_data="CLSTR full-base checkpoint; no ScienceWorld eval split used for training",
            trajectory_policy_trained=True,
            controller_complete_decision=True,
            components_in_action_selection=True,
            l_policy_source="full-base heterogeneous trajectory losses where legal",
            note="official ScienceWorld harness smoke OK; gate-sized closed-loop eval",
        ),
        _row_from_metrics(
            output_root=root,
            benchmark="webshop",
            method="webshop_controller_gate",
            path=root / "webshop_eval" / "clstr_controller_gate" / "metrics.json",
            training_data="CLSTR full-base checkpoint; WebShop test train-then-eval forbidden",
            trajectory_policy_trained=True,
            controller_complete_decision=True,
            components_in_action_selection=True,
            l_policy_source="full-base heterogeneous trajectory losses where legal",
            note="official WebShop text env smoke OK; gate-sized closed-loop eval",
        ),
        _dbbench_row(root),
    ]
    component_trace_example = _load_component_trace_example(root)
    summary = {
        "status": "ok",
        "rows": rows,
        "artifact_paths": {
            "comparison_table": str(output_table_path),
            "comparison_summary": str(output_summary_path),
            "paper_report_md": str(output_paper_md_path),
            "paper_report_json": str(output_paper_json_path),
        },
        "component_trace_example": component_trace_example,
        "caveats": [
            "closed-loop metrics only come from environment rollouts",
            "offline diagnostics are not closed-loop success",
            "valid/test splits are eval-only",
            "DBBench remains blocked without an official closed-loop harness",
        ],
    }
    table = _markdown_table(rows)
    Path(output_table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_table_path).write_text(table, encoding="utf-8")
    _write_json(Path(output_summary_path), summary)
    Path(output_paper_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_paper_md_path).write_text(_paper_report(rows, summary), encoding="utf-8")
    _write_json(
        Path(output_paper_json_path),
        {
            "status": "ok",
            "summary": summary,
            "paper_safe_claims": {
                "offline_diagnostic_is_not_success": True,
                "full_base_training_vs_controller_decision_distinguished": True,
                "dbbench_not_reported_as_success": True,
            },
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR controller-complete comparison and paper reports.")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--output_table_path", default="outputs/clstr_controller_comparison_table.md")
    parser.add_argument("--output_summary_path", default="outputs/clstr_controller_comparison_summary.json")
    parser.add_argument("--output_paper_md_path", default="outputs/clstr_controller_paper_report.md")
    parser.add_argument("--output_paper_json_path", default="outputs/clstr_controller_paper_report.json")
    args = parser.parse_args()
    summary = build_controller_reports(
        output_root=args.output_root,
        output_table_path=args.output_table_path,
        output_summary_path=args.output_summary_path,
        output_paper_md_path=args.output_paper_md_path,
        output_paper_json_path=args.output_paper_json_path,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
