#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from clstr.vnext_online_selector import VNextOnlineSelector
from clstr.vnext_toolsandbox_agent import (
    CLSTRVNextToolSandboxUser,
    CLSTRVNextToolSandboxAgent,
    _selection_order_diagnostics,
)
from tool_sandbox.cli import write_result_summary
from tool_sandbox.cli.utils import (
    USER_TYPE_TO_FACTORY,
    RoleImplType,
    get_category_summary,
    resolve_scenarios,
)
from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.execution_environment import ExecutionEnvironment


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_diagnostics(path: Path) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    provider_decisions: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("record_type") == "selection":
                        selections.append(row)
                    elif row.get("record_type") == "provider_decision":
                        provider_decisions.append(row)
                    elif row.get("record_type") in {
                        "memory_update",
                        "untracked_tool_result",
                    }:
                        result_records.append(row)
    no_op_count = sum(bool(row.get("filter_was_noop")) for row in selections)
    filtered_count = sum(
        int(row.get("filtered_tracked_count") or 0) for row in selections
    )
    normalized_order_diagnostics: list[dict[str, Any]] = []
    for row in selections:
        derived = _selection_order_diagnostics(
            candidate_skill_ids=list(row.get("candidate_skill_ids") or []),
            selected=list(row.get("selected") or []),
            history_depth=int(row.get("history_depth") or 0),
        )
        normalized_order_diagnostics.append(
            {
                key: row.get(key, value)
                for key, value in derived.items()
            }
        )
    order_changed_count = sum(
        bool(row.get("rank_order_changed"))
        for row in normalized_order_diagnostics
    )
    memory_applied_count = sum(
        bool(row.get("memory_ranking_applied"))
        for row in normalized_order_diagnostics
    )
    memory_changed_count = sum(
        bool(row.get("memory_ranking_changed"))
        for row in normalized_order_diagnostics
    )
    memory_top1_changed_count = sum(
        bool(row.get("memory_top1_changed"))
        for row in normalized_order_diagnostics
    )
    schema_order_preserved_count = sum(
        bool(row.get("schema_order_preserved")) for row in selections
    )
    clstr_rank_differs_count = sum(
        bool(row.get("clstr_rank_differs_from_input")) for row in selections
    )
    guidance_policies = sorted(
        {
            str(row.get("guidance_policy") or "")
            for row in selections
            if str(row.get("guidance_policy") or "")
        }
    )
    dual_expert_guidance_count = sum(
        bool(row.get("dual_expert_guidance_applied")) for row in selections
    )
    intervention_modes = []
    if filtered_count > 0:
        intervention_modes.append("candidate_set_filter")
    if order_changed_count > 0:
        intervention_modes.append("tool_schema_rank_order")
    guidance_count = sum(
        bool(row.get("provider_guidance_injected")) for row in provider_decisions
    )
    guidance_adopted_count = sum(
        bool(row.get("provider_guidance_adopted")) for row in provider_decisions
    )
    top1_adopted_count = sum(
        bool(row.get("provider_top1_guidance_adopted"))
        for row in provider_decisions
    )
    posthoc_substitution_count = sum(
        bool(row.get("posthoc_tool_substitution")) for row in provider_decisions
    )
    if guidance_count > 0:
        intervention_modes.append("provider_ranked_tool_guidance")
    proposed_call_ids = {
        str(call.get("tool_call_id") or "")
        for row in provider_decisions
        for call in list(row.get("provider_tool_calls") or [])
        if str(call.get("tool_call_id") or "")
    }
    observed_result_ids = {
        str(row.get("tool_call_id") or "")
        for row in result_records
        if str(row.get("tool_call_id") or "")
    }
    observed_proposed_ids = proposed_call_ids.intersection(observed_result_ids)
    return {
        "selection_step_count": len(selections),
        "filter_noop_step_count": no_op_count,
        "filter_noop_step_rate": (
            no_op_count / len(selections) if selections else 0.0
        ),
        "filtered_tracked_tool_total": filtered_count,
        "rank_order_changed_step_count": order_changed_count,
        "rank_order_changed_step_rate": (
            order_changed_count / len(selections) if selections else 0.0
        ),
        "memory_ranking_applied_step_count": memory_applied_count,
        "memory_ranking_changed_step_count": memory_changed_count,
        "memory_ranking_changed_step_rate": (
            memory_changed_count / memory_applied_count
            if memory_applied_count
            else 0.0
        ),
        "memory_top1_changed_step_count": memory_top1_changed_count,
        "memory_top1_changed_step_rate": (
            memory_top1_changed_count / memory_applied_count
            if memory_applied_count
            else 0.0
        ),
        "schema_order_preserved_step_count": schema_order_preserved_count,
        "schema_order_preserved_step_rate": (
            schema_order_preserved_count / len(selections) if selections else 0.0
        ),
        "clstr_rank_differs_from_input_step_count": clstr_rank_differs_count,
        "clstr_rank_differs_from_input_step_rate": (
            clstr_rank_differs_count / len(selections) if selections else 0.0
        ),
        "guidance_policies": guidance_policies,
        "dual_expert_guidance_step_count": dual_expert_guidance_count,
        "dual_expert_guidance_step_rate": (
            dual_expert_guidance_count / len(selections) if selections else 0.0
        ),
        "provider_decision_step_count": len(provider_decisions),
        "provider_guidance_step_count": guidance_count,
        "provider_guidance_step_rate": (
            guidance_count / len(provider_decisions) if provider_decisions else 0.0
        ),
        "provider_guidance_adopted_step_count": guidance_adopted_count,
        "provider_guidance_adopted_step_rate": (
            guidance_adopted_count / guidance_count if guidance_count else 0.0
        ),
        "provider_top1_guidance_adopted_step_count": top1_adopted_count,
        "provider_posthoc_tool_substitution_step_count": posthoc_substitution_count,
        "provider_proposed_tool_call_count": len(proposed_call_ids),
        "provider_observed_result_count": len(observed_proposed_ids),
        "provider_result_coverage_rate": (
            len(observed_proposed_ids) / len(proposed_call_ids)
            if proposed_call_ids
            else 1.0
        ),
        "selector_intervention_modes": intervention_modes,
        "selector_intervention_effective": bool(intervention_modes),
    }


def held_out_toolsandbox_scenarios(manifest_path: str | Path) -> list[str]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split = payload.get("toolsandbox_split_manifest") or {}
    records = split.get("scenario_records") or {}
    family_to_split = split.get("family_to_split") or {}
    selected = sorted(
        name
        for name, record in records.items()
        if family_to_split.get(str(record.get("family_id") or "")) == "test"
    )
    expected = int((split.get("scenario_count_by_split") or {}).get("test") or 0)
    if not selected or len(selected) != expected:
        raise ValueError(
            "ToolSandbox grouped held-out scenario contract differs: "
            f"expected {expected}, observed {len(selected)}"
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLSTR vNext on the official ToolSandbox scenario executor and "
            "report milestone/minefield similarity plus exact success."
        )
    )
    parser.add_argument("--matched_release_selection_path", required=True)
    parser.add_argument("--stage2_checkpoint_path", default=None)
    parser.add_argument("--training_skills_path", default=None)
    parser.add_argument("--benchmark_skills_path", required=True)
    parser.add_argument("--matched_union_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--executor_model", required=True)
    parser.add_argument("--executor_base_url", required=True)
    parser.add_argument("--executor_api_key", default="EMPTY")
    parser.add_argument("--executor_temperature", type=float, default=0.0)
    parser.add_argument("--executor_max_tokens", type=int, default=1024)
    parser.add_argument(
        "--executor_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--intervention_mode",
        choices=["schema_order", "ranked_guidance"],
        default="ranked_guidance",
    )
    parser.add_argument("--guidance_top_k", type=int, default=2)
    parser.add_argument(
        "--user",
        choices=[str(item) for item in USER_TYPE_TO_FACTORY],
        default=str(RoleImplType.GPT_4_o_2024_05_13),
    )
    parser.add_argument("--user_base_url", default=None)
    parser.add_argument("--user_api_key", default=None)
    parser.add_argument("--user_temperature", type=float, default=0.0)
    parser.add_argument("--user_max_tokens", type=int, default=1024)
    parser.add_argument("--executor_random_seed", type=int, default=0)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument(
        "--route_mode",
        choices=["adaptive", "static", "dynamic"],
        default="adaptive",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.matched_union_manifest_path).resolve()
    held_out = held_out_toolsandbox_scenarios(manifest_path)
    desired = held_out if not args.scenario else list(dict.fromkeys(args.scenario))
    invalid = sorted(set(desired) - set(held_out))
    if invalid:
        raise ValueError(
            "requested ToolSandbox scenarios are outside the grouped held-out test: "
            + ", ".join(invalid)
        )
    selector = VNextOnlineSelector(
        matched_release_selection_path=args.matched_release_selection_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        training_skills_path=args.training_skills_path,
        benchmark_skills_path=args.benchmark_skills_path,
        device=args.device,
        coarse_k=args.coarse_k,
        dynamic_extra_k=args.dynamic_extra_k,
    )
    release_manifest = selector.checkpoint_binding["release"][
        "matched_union_manifest"
    ]
    if Path(release_manifest["path"]).resolve() != manifest_path or str(
        release_manifest["sha256"]
    ) != _file_sha256(manifest_path):
        raise ValueError("ToolSandbox split manifest differs from the matched release")
    selection_log_path = output_dir / "retrieval_selections.jsonl"
    if selection_log_path.exists():
        selection_log_path.unlink()
    api_key = (
        os.environ.get("EXECUTOR_API_KEY", "EMPTY")
        if args.executor_api_key == "EMPTY"
        else args.executor_api_key
    )
    agent = CLSTRVNextToolSandboxAgent(
        selector=selector,
        model_name=args.executor_model,
        base_url=args.executor_base_url,
        api_key=api_key,
        top_k=args.top_k,
        selection_log_path=selection_log_path,
        route_mode=args.route_mode,
        executor_temperature=args.executor_temperature,
        executor_max_tokens=args.executor_max_tokens,
        executor_enable_thinking=args.executor_enable_thinking,
        intervention_mode=args.intervention_mode,
        guidance_top_k=args.guidance_top_k,
    )
    user_type = RoleImplType(args.user)
    scenarios = resolve_scenarios(
        desired_scenario_names=desired,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    missing = sorted(set(desired) - set(scenarios))
    if missing:
        raise ValueError(
            "official ToolSandbox repository lacks held-out scenarios: "
            + ", ".join(missing)
        )
    result_summary: list[dict[str, Any]] = []
    for name in desired:
        scenario = scenarios[name]
        agent.begin_scenario(name)
        user_factory = USER_TYPE_TO_FACTORY[user_type]
        user_model_name = getattr(user_factory, "model_name", None)
        if not user_model_name:
            raise ValueError(
                "deterministic ToolSandbox evaluation requires an OpenAI user model"
            )
        user = CLSTRVNextToolSandboxUser(
            model_name=str(user_model_name),
            temperature=args.user_temperature,
            max_tokens=args.user_max_tokens,
        )
        if args.user_base_url or args.user_api_key:
            user.openai_client = OpenAI(
                base_url=args.user_base_url or "https://api.openai.com/v1",
                api_key=args.user_api_key or os.environ.get("OPENAI_API_KEY"),
            )
        environment = ExecutionEnvironment()
        roles = {
            RoleType.USER: user,
            RoleType.EXECUTION_ENVIRONMENT: environment,
            RoleType.AGENT: agent,
        }
        try:
            result = scenario.play_and_evaluate(
                roles=roles,
                output_directory=output_dir,
                scenario_name=name,
            )
            agent.finalize_scenario()
            evaluation = result.evaluation_result
            result_summary.append(
                {
                    "name": name,
                    "categories": scenario.categories,
                    "traceback": None,
                    "exception_type": None,
                    "milestone_similarity": evaluation.milestone_similarity,
                    "minefield_similarity": evaluation.minefield_similarity,
                    "similarity": evaluation.similarity,
                    "turn_count": evaluation.turn_count,
                    "milestone_mapping": evaluation.milestone_mapping,
                    "minefield_mapping": evaluation.minefield_mapping,
                }
            )
        except Exception as exc:
            result_summary.append(
                {
                    "name": name,
                    "categories": scenario.categories,
                    "traceback": str(exc),
                    "exception_type": type(exc).__name__,
                    "milestone_similarity": 0.0,
                    "minefield_similarity": 0.0,
                    "similarity": 0.0,
                    "turn_count": scenario.max_messages,
                    "milestone_mapping": {},
                    "minefield_mapping": {},
                }
            )
        finally:
            user.teardown()
            environment.teardown()
            agent.reset()
    agent.teardown()
    category_summary = get_category_summary(result_summary)
    write_result_summary(
        result_summary=result_summary,
        category_summary=category_summary,
        output_directory=output_dir,
    )
    similarities = [float(row["similarity"]) for row in result_summary]
    milestones = [float(row["milestone_similarity"]) for row in result_summary]
    minefields = [float(row["minefield_similarity"]) for row in result_summary]
    exact = [value >= 1.0 - 1e-12 for value in similarities]
    exception_count = sum(
        row["exception_type"] is not None for row in result_summary
    )
    selection_diagnostics = _selection_diagnostics(selection_log_path)
    report = {
        "status": (
            "ok" if result_summary and exception_count == 0 else "action_required"
        ),
        "metric_scope": "official ToolSandbox held-out scenario execution",
        "method": selector.method,
        "executor_model": args.executor_model,
        "executor_temperature": float(args.executor_temperature),
        "executor_max_tokens": int(args.executor_max_tokens),
        "executor_enable_thinking": bool(args.executor_enable_thinking),
        "executor_random_seed": int(args.executor_random_seed),
        "intervention_mode": args.intervention_mode,
        "guidance_top_k": int(args.guidance_top_k),
        "user": str(user_type),
        "user_base_url": args.user_base_url,
        "user_temperature": float(args.user_temperature),
        "user_max_tokens": int(args.user_max_tokens),
        "split_protocol": "matched_union_grouped_family_test_v1",
        "matched_union_manifest_path": str(manifest_path),
        "matched_union_manifest_sha256": _file_sha256(manifest_path),
        "scenario_count": len(result_summary),
        "mean_milestone_similarity": (
            sum(milestones) / len(milestones) if milestones else 0.0
        ),
        "mean_minefield_similarity": (
            sum(minefields) / len(minefields) if minefields else 0.0
        ),
        "mean_similarity": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "exact_success_rate": sum(exact) / len(exact) if exact else 0.0,
        "exception_count": exception_count,
        "official_result_summary": str(output_dir / "result_summary.json"),
        "selection_log_path": str(selection_log_path),
        "route_mode": args.route_mode,
        **selection_diagnostics,
        "checkpoint_binding": selector.checkpoint_binding,
    }
    _write_json(output_dir / "clstr_task_accuracy.json", report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
