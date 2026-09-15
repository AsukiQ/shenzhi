#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_executor import (
    _default_world_factory,
    _execution_succeeded,
    enrich_task_with_appworld_specs,
    load_appworld_api_refs,
)
from clstr.appworld_official_executor import (
    EvidenceResult,
    OfficialReActExecutorConfig,
    build_official_clstr_state_text,
    build_official_api_docs_context,
    run_official_react_task,
)
from clstr.appworld_current_route_rollout import CurrentRouteRolloutLogger
from clstr.appworld_routing import read_jsonl, write_json, write_jsonl
from clstr.appworld_skill_handoff import build_verified_skill_handoff
from clstr.envs.appworld_env import configure_appworld_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official-style AppWorld ReAct executor with optional CLSTR evidence.")
    parser.add_argument(
        "--method",
        choices=["qwen_only", "static_predictions", "clstr_multistep"],
        default="qwen_only",
    )
    parser.add_argument("--tasks_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--appworld_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root",
    )
    parser.add_argument(
        "--appworld_cache",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld",
    )
    parser.add_argument("--model_name_or_path", default="models/Qwen3-14B")
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--task_ids", default="")
    parser.add_argument("--max_interactions", type=int, default=40)
    parser.add_argument("--max_wrong_completion_retries", type=int, default=0)
    parser.add_argument("--preflight_max_repairs", type=int, default=0)
    parser.add_argument(
        "--completion_precheck_mode",
        choices=["off", "constraint_tokens"],
        default="off",
    )
    parser.add_argument("--timeout_seconds", type=int, default=120)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--experiment_name", default="clstr_appworld_official_executor")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--base_skill_pool_path", default=None)
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--clstr_model_config", default=None)
    parser.add_argument("--clstr_checkpoint_path", default=None)
    parser.add_argument(
        "--ranking_mode",
        choices=[
            "auto",
            "skill_table",
            "policy_head",
            "policy_blend",
            "transition_head",
            "transition_blend",
            "policy_transition_blend",
        ],
        default="auto",
    )
    parser.add_argument("--candidate_top_k", type=int, default=None)
    parser.add_argument(
        "--candidate_source",
        choices=["routing", "routing_belief_union", "routing_belief_union_after_update"],
        default="routing",
    )
    parser.add_argument("--policy_blend_alpha", type=float, default=0.5)
    parser.add_argument("--transition_scoring_mode", default="v4_1b_action_observation")
    parser.add_argument("--transition_residual_lambda", type=float, default=0.0)
    parser.add_argument("--clstr_alpha", type=float, default=0.25)
    parser.add_argument("--recurrent_belief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe_canonical_skills", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_auth_like_skills", type=int, default=None)
    parser.add_argument("--appworld_executor_compatible_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_evidence_chars", type=int, default=1600)
    parser.add_argument(
        "--handoff_prompt_style",
        choices=["structured_evidence", "legacy_hints"],
        default="structured_evidence",
    )
    parser.add_argument(
        "--handoff_visible_skill_limit",
        type=int,
        default=None,
        help=(
            "Optional cap on how many selected skills contribute read/schema evidence to the "
            "executor prompt. State-changing action APIs can still be rescued from deeper skills."
        ),
    )
    parser.add_argument(
        "--clstr_state_context",
        choices=["legacy", "api_schema"],
        default="api_schema",
        help=(
            "State text used by CLSTR routing. 'api_schema' appends public available API "
            "inventory to the routing state; it does not use solution api_refs for ranking."
        ),
    )
    parser.add_argument(
        "--current_route_rollouts_path",
        default="",
        help=(
            "Optional JSONL path for current-route rollout traces used by Stage4/RL diagnostics. "
            "Use 'auto' to write output_dir/current_route_rollouts.jsonl."
        ),
    )
    return parser


def _task_instruction(task: dict[str, Any]) -> str:
    return str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")


def _build_generator(args: argparse.Namespace) -> Any:
    from clstr.qwen_backend import QwenBackendConfig, QwenGenerationBackend

    return QwenGenerationBackend(
        QwenBackendConfig(
            model_name_or_path=str(args.model_name_or_path),
            local_files_only=bool(args.local_files_only),
            torch_dtype=str(args.torch_dtype),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            device=args.device,
            enable_thinking=bool(args.enable_thinking),
        )
    )


def _metadata(generator: Any) -> dict[str, Any]:
    if hasattr(generator, "metadata") and callable(getattr(generator, "metadata")):
        return dict(generator.metadata())
    return {"backend": type(generator).__name__}


def _local_state_text(task: dict[str, Any], history: list[dict[str, Any]]) -> str:
    lines = ["[User Goal]", _task_instruction(task), "", "[Execution History]"]
    if not history:
        lines.append("No previous execution steps.")
    for step in history:
        lines.extend(
            [
                f"Step {int(step.get('step_idx', len(lines)))}",
                "Code:",
                str(step.get("code", "")),
                "Output:",
                str(step.get("execute_output", "")),
            ]
        )
    return "\n".join(lines)


def _legacy_state_text(task: dict[str, Any], history: list[dict[str, Any]]) -> str:
    from clstr.appworld_multistep import build_multistep_state_text

    return build_multistep_state_text(task=task, steps=history, include_selected_skill_ids=False)


class CLSTREvidenceBuilder:
    def __init__(
        self,
        *,
        controller: Any,
        appworld_root: str | Path,
        top_k: int = 5,
        max_chars: int = 1600,
        valid_api_refs_loader: Callable[..., set[tuple[str, str]]] = load_appworld_api_refs,
        state_text_builder: Callable[[dict[str, Any], list[dict[str, Any]]], str] | None = None,
        api_docs_context_builder: Callable[..., str] = build_official_api_docs_context,
        handoff_prompt_style: str = "structured_evidence",
        handoff_visible_skill_limit: int | None = None,
        clstr_state_context: str = "legacy",
    ) -> None:
        self.controller = controller
        self.appworld_root = appworld_root
        self.top_k = int(top_k)
        self.max_chars = int(max_chars)
        self.valid_api_refs_loader = valid_api_refs_loader
        self.state_text_builder = state_text_builder or _local_state_text
        self.api_docs_context_builder = api_docs_context_builder
        self.handoff_prompt_style = str(handoff_prompt_style)
        self.handoff_visible_skill_limit = handoff_visible_skill_limit
        self.clstr_state_context = str(clstr_state_context or "legacy")
        self.last_selection: Any | None = None

    def reset(self, task: dict[str, Any]) -> None:
        self.last_selection = None
        if callable(getattr(self.controller, "reset", None)):
            self.controller.reset(task)

    def __call__(self, *, task: dict[str, Any], history: list[dict[str, Any]]) -> EvidenceResult:
        state_api_docs_context = ""
        if self.clstr_state_context == "api_schema":
            state_api_docs_context = str(
                self.api_docs_context_builder(
                    appworld_root=self.appworld_root,
                    task=task,
                )
                or ""
            )
            state_text = build_official_clstr_state_text(
                task=task,
                history=history,
                api_docs_context=state_api_docs_context,
            )
        else:
            state_text = self.state_text_builder(task, history)
        selection = self.controller.select(task=task, state_text=state_text, steps=history, top_k=self.top_k)
        self.last_selection = selection
        skills = [dict(skill) for skill in getattr(selection, "skills", [])]
        controller_diagnostics = getattr(selection, "diagnostics", {})
        valid_api_refs = self.valid_api_refs_loader(
            appworld_root=self.appworld_root,
            required_apps=[str(item) for item in task.get("required_apps", [])],
            api_refs=[str(item) for item in task.get("api_refs", [])],
        )
        handoff = build_verified_skill_handoff(
            instruction=_task_instruction(task),
            skills=skills,
            valid_api_refs=valid_api_refs,
            required_apps=[str(item) for item in task.get("required_apps", [])],
            controller_diagnostics=controller_diagnostics if isinstance(controller_diagnostics, dict) else {},
            max_chars=self.max_chars,
            prompt_style=self.handoff_prompt_style,
            visible_skill_limit=self.handoff_visible_skill_limit,
        )
        diagnostics = {
            key: value
            for key, value in handoff.items()
            if key not in {"prompt_block"}
        }
        diagnostics["state_text"] = state_text
        diagnostics["state_context"] = self.clstr_state_context
        diagnostics["state_api_docs_chars"] = len(state_api_docs_context)
        diagnostics["controller"] = controller_diagnostics
        diagnostics["selected_skill_ids"] = list(getattr(selection, "selected_skill_ids", []))
        return EvidenceResult(prompt_text=str(handoff.get("prompt_block", "")), diagnostics=diagnostics)

    def observe(self, *, code: str, execute_output: str, step: dict[str, Any]) -> None:
        if self.last_selection is None:
            return
        if callable(getattr(self.controller, "observe", None)):
            self.controller.observe(
                selection=self.last_selection,
                code=code,
                execute_output=execute_output,
                step=step,
            )


def _build_clstr_evidence_builder(args: argparse.Namespace) -> CLSTREvidenceBuilder:
    from scripts.run_appworld_multistep_executor_eval import _build_controller

    controller = _build_controller(
        method=str(args.method),
        skill_pool_path=str(args.skill_pool_path),
        base_skill_pool_path=args.base_skill_pool_path,
        predictions_path=args.predictions_path,
        clstr_model_config_path=args.clstr_model_config,
        clstr_checkpoint_path=args.clstr_checkpoint_path,
        ranking_mode=str(args.ranking_mode),
        candidate_top_k=args.candidate_top_k,
        candidate_source=str(args.candidate_source),
        policy_blend_alpha=float(args.policy_blend_alpha),
        transition_scoring_mode=str(args.transition_scoring_mode),
        transition_residual_lambda=float(args.transition_residual_lambda),
        clstr_alpha=float(args.clstr_alpha),
        recurrent_belief=bool(args.recurrent_belief),
        dedupe_canonical_skills=bool(args.dedupe_canonical_skills),
        max_auth_like_skills=args.max_auth_like_skills,
        appworld_executor_compatible_only=bool(args.appworld_executor_compatible_only),
    )
    return CLSTREvidenceBuilder(
        controller=controller,
        appworld_root=args.appworld_root,
        top_k=int(args.top_k),
        max_chars=int(args.max_evidence_chars),
        state_text_builder=_legacy_state_text,
        handoff_prompt_style=str(args.handoff_prompt_style),
        handoff_visible_skill_limit=args.handoff_visible_skill_limit,
        clstr_state_context=str(args.clstr_state_context),
    )


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _step_execution_failures(row: dict[str, Any]) -> int:
    failures = 0
    for step in row.get("steps", []):
        output = step.get("execute_output", "")
        ok = bool(step.get("preflight_ok", True)) and bool(step.get("execution_attempted", True)) and _execution_succeeded(output)
        step["execution_ok"] = bool(ok)
        failures += int(not ok)
    return failures


def run_official_executor_eval(
    args: argparse.Namespace,
    *,
    generator_factory: Callable[[argparse.Namespace], Any] | None = None,
    world_factory: Callable[..., Any] | None = None,
    evidence_builder_factory: Callable[[argparse.Namespace], Any] | None = None,
) -> dict[str, Any]:
    if args.method not in {"qwen_only", "clstr_multistep"}:
        raise NotImplementedError(f"official executor runtime for method={args.method!r} is not implemented")

    root, cache = configure_appworld_paths(args.appworld_root, args.appworld_cache)
    tasks = read_jsonl(args.tasks_path)
    task_ids = {item.strip() for item in str(getattr(args, "task_ids", "") or "").split(",") if item.strip()}
    if task_ids:
        tasks = [
            task
            for task in tasks
            if str(task.get("task_id") or task.get("query_id") or "") in task_ids
        ]
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]

    output_dir = Path(args.output_dir)
    runs_path = output_dir / "runs.jsonl"
    current_route_rollouts_path = str(getattr(args, "current_route_rollouts_path", "") or "")
    if current_route_rollouts_path == "auto":
        current_route_rollouts_path = str(output_dir / "current_route_rollouts.jsonl")
    current_route_logger = (
        CurrentRouteRolloutLogger(current_route_rollouts_path)
        if current_route_rollouts_path
        else None
    )
    factory = world_factory or _default_world_factory()
    generator = generator_factory(args) if generator_factory else _build_generator(args)
    evidence_builder = None
    if args.method == "clstr_multistep":
        evidence_builder = evidence_builder_factory(args) if evidence_builder_factory else _build_clstr_evidence_builder(args)
    generator_metadata = _metadata(generator)
    run_rows: list[dict[str, Any]] = []
    _log_event(
        "official_executor_start",
        method=str(args.method),
        task_count=len(tasks),
        max_interactions=int(args.max_interactions),
        output_dir=str(output_dir),
    )

    for task_index, raw_task in enumerate(tasks, start=1):
        task = enrich_task_with_appworld_specs(raw_task, root)
        task_id = str(task.get("task_id") or task.get("query_id"))
        query_id = str(task.get("query_id") or task_id)
        _log_event(
            "official_executor_task_start",
            task_index=task_index,
            task_count=len(tasks),
            task_id=task_id,
            query_id=query_id,
        )
        row: dict[str, Any] = {
            "method": str(args.method),
            "task_id": task_id,
            "query_id": query_id,
            "split": task.get("split"),
            "user_goal": _task_instruction(task),
            "reset_ok": False,
            "success": False,
            "task_completed": False,
            "evaluation_success": False,
            "steps": [],
        }
        world = None
        try:
            _log_event(
                "official_executor_world_start",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
            )
            world = factory(
                task_id,
                experiment_name=f"{args.experiment_name}_{args.method}",
                max_interactions=int(args.max_interactions),
                timeout_seconds=int(args.timeout_seconds),
                load_ground_truth=True,
                ground_truth_mode="minimal",
                show_api_response_schemas=False,
            )
            row["reset_ok"] = True
            _log_event(
                "official_executor_world_done",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
            )
            if evidence_builder is not None and callable(getattr(evidence_builder, "reset", None)):
                _log_event(
                    "official_executor_evidence_reset_start",
                    task_index=task_index,
                    task_count=len(tasks),
                    task_id=task_id,
                )
                evidence_builder.reset(task)
                _log_event(
                    "official_executor_evidence_reset_done",
                    task_index=task_index,
                    task_count=len(tasks),
                    task_id=task_id,
                )
            _log_event(
                "official_executor_api_refs_start",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
            )
            valid_api_refs = load_appworld_api_refs(
                appworld_root=root,
                required_apps=[str(item) for item in task.get("required_apps", [])],
                api_refs=[str(item) for item in task.get("api_refs", [])],
            )
            _log_event(
                "official_executor_api_refs_done",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
                valid_api_ref_count=len(valid_api_refs),
            )
            _log_event(
                "official_executor_api_docs_start",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
            )
            api_docs_context = build_official_api_docs_context(appworld_root=root, task=task)
            _log_event(
                "official_executor_api_docs_done",
                task_index=task_index,
                task_count=len(tasks),
                task_id=task_id,
                api_docs_chars=len(str(api_docs_context)),
            )
            result = run_official_react_task(
                task=task,
                world=world,
                generator=generator,
                config=OfficialReActExecutorConfig(
                    max_interactions=int(args.max_interactions),
                    max_wrong_completion_retries=int(args.max_wrong_completion_retries),
                    preflight_max_repairs=int(args.preflight_max_repairs),
                    completion_precheck_mode=str(args.completion_precheck_mode),
                    timeout_seconds=int(args.timeout_seconds),
                    valid_api_refs=valid_api_refs,
                    api_docs_context=api_docs_context,
                ),
                evidence_builder=evidence_builder,
                progress_callback=lambda event, payload, task_index=task_index, task_count=len(tasks): _log_event(
                    event,
                    task_index=task_index,
                    task_count=task_count,
                    **payload,
                ),
            )
            row.update(result)
            _step_execution_failures(row)
            row["final_success"] = bool(row.get("success", False))
        except Exception as exc:
            row["error"] = repr(exc)
        finally:
            if world is not None and callable(getattr(world, "close", None)):
                world.close()
        run_rows.append(row)
        write_jsonl(runs_path, run_rows)
        if current_route_logger is not None:
            current_route_logger.append(row)
        _log_event(
            "official_executor_task_done",
            task_index=task_index,
            task_count=len(tasks),
            task_id=task_id,
            success=bool(row.get("success", False)),
            task_completed=bool(row.get("task_completed", False)),
            evaluation_success=bool(row.get("evaluation_success", False)),
            steps=len(row.get("steps", [])),
            error=str(row.get("error", "")),
        )

    write_jsonl(runs_path, run_rows)
    total_steps = sum(len(row.get("steps", [])) for row in run_rows)
    success_count = sum(1 for row in run_rows if row.get("success"))
    task_completed_count = sum(1 for row in run_rows if row.get("task_completed"))
    evaluate_success_count = sum(1 for row in run_rows if row.get("evaluation_success"))
    generation_failures = sum(1 for row in run_rows if row.get("error") and not row.get("steps"))
    execution_failures = sum(_step_execution_failures(row) for row in run_rows)
    report = {
        "status": "ok",
        "method": str(args.method),
        "tasks_path": str(args.tasks_path),
        "appworld_root": str(root),
        "appworld_cache": str(cache),
        "runs_path": str(runs_path),
        "model": generator_metadata,
        "task_count": len(run_rows),
        "success_count": success_count,
        "success_rate": round(success_count / max(len(run_rows), 1), 6),
        "task_completed_count": task_completed_count,
        "evaluate_success_count": evaluate_success_count,
        "generation_failures": generation_failures,
        "execution_failures": execution_failures,
        "average_steps": round(total_steps / max(len(run_rows), 1), 6),
        "max_interactions": int(args.max_interactions),
        "timeout_seconds": int(args.timeout_seconds),
        "caveat": "Official-style AppWorld ReAct runtime; success requires AppWorld task_completed/evaluate.",
    }
    if current_route_logger is not None:
        report["current_route_rollouts_path"] = str(current_route_logger.path)
    write_json(output_dir / "report.json", report)
    _log_event("official_executor_done", report_path=str(output_dir / "report.json"), success_count=success_count)
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_official_executor_eval(args)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
