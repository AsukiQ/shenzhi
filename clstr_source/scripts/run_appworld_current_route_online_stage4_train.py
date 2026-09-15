#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run current-route AppWorld official executor rollouts and update Stage4 online "
            "from clean current-route preference samples after each task."
        )
    )
    parser.add_argument("--method", choices=["clstr_multistep"], default="clstr_multistep")
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
    parser.add_argument("--completion_precheck_mode", choices=["off", "constraint_tokens"], default="constraint_tokens")
    parser.add_argument("--timeout_seconds", type=int, default=120)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--experiment_name", default="clstr_appworld_current_route_online_stage4")
    parser.add_argument("--skill_pool_path", required=True)
    parser.add_argument("--base_skill_pool_path", required=True)
    parser.add_argument("--clstr_checkpoint_path", required=True)
    parser.add_argument("--ranking_mode", default="policy_transition_blend")
    parser.add_argument("--candidate_top_k", type=int, default=350)
    parser.add_argument(
        "--candidate_source",
        choices=["routing", "routing_belief_union", "routing_belief_union_after_update"],
        default="routing",
    )
    parser.add_argument("--policy_blend_alpha", type=float, default=0.5)
    parser.add_argument("--transition_scoring_mode", default="v4_1b_action_observation")
    parser.add_argument("--transition_residual_lambda", type=float, default=0.0)
    parser.add_argument("--learned_component_min_range", type=float, default=0.1)
    parser.add_argument("--learned_component_trust_top_k", type=int, default=80)
    parser.add_argument("--recurrent_belief", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dedupe_canonical_skills", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_auth_like_skills", type=int, default=1)
    parser.add_argument("--appworld_executor_compatible_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_evidence_chars", type=int, default=1600)
    parser.add_argument("--handoff_visible_skill_limit", type=int, default=5)
    parser.add_argument(
        "--handoff_prompt_style",
        choices=["structured_evidence", "legacy_hints"],
        default="legacy_hints",
    )
    parser.add_argument("--max_online_updates", type=int, default=1)
    parser.add_argument("--online_update_epochs", type=int, default=1)
    parser.add_argument("--replay_success_rollouts_path", default="")
    parser.add_argument("--max_replay_updates", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--train_transition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stability_kl_weight", type=float, default=0.1)
    parser.add_argument(
        "--loss_score_mode",
        choices=["policy_head", "transition_blend", "policy_transition_blend"],
        default="policy_transition_blend",
    )
    parser.add_argument(
        "--enable_failure_stage0_correction",
        action="store_true",
        help=(
            "When a failed rollout has GT API skills outside current candidates, write Stage0 "
            "retrieval correction rows instead of updating Stage4."
        ),
    )
    parser.add_argument(
        "--enable_failure_stage4_correction",
        action="store_true",
        help=(
            "When a failed rollout has GT API-matched skills inside current candidates, update "
            "Stage4 with target-vs-rejected correction samples."
        ),
    )
    parser.add_argument(
        "--enable_online_stage0_correction_update",
        action="store_true",
        help=(
            "Immediately update Stage0 retrieval parameters from missing-candidate correction rows "
            "during the online loop. Stage1/2 remain frozen."
        ),
    )
    parser.add_argument("--online_stage0_learning_rate", type=float, default=None)
    parser.add_argument("--train_online_stage0_skill_adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_online_stage0_retrieval_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_online_stage0_skill_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_online_stage0_encoder_projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--appworld_tasks_root",
        default="",
        help="AppWorld tasks root containing <task_id>/ground_truth/solution.py; defaults to appworld_root/data/tasks.",
    )
    parser.add_argument("--failure_correction_max_targets", type=int, default=3)
    return parser


def _filter_tasks(tasks: list[dict[str, Any]], *, task_ids: str, max_tasks: int | None) -> list[dict[str, Any]]:
    requested = {item.strip() for item in str(task_ids or "").replace(":", ",").split(",") if item.strip()}
    if requested:
        tasks = [
            task
            for task in tasks
            if str(task.get("task_id") or task.get("query_id") or "") in requested
        ]
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    return tasks


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def run_online_stage4_train(
    args: argparse.Namespace,
    *,
    generator_factory: Callable[[argparse.Namespace], Any] | None = None,
    world_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from clstr.appworld_current_route_online_train import run_current_route_online_stage4_train_with_model
    from clstr.appworld_corrective_preference import build_api_correction_dataset_from_rollouts
    from clstr.appworld_executor import _default_world_factory, enrich_task_with_appworld_specs, load_appworld_api_refs
    from clstr.appworld_multistep import CLSTRMultiStepController
    from clstr.appworld_official_executor import OfficialReActExecutorConfig, build_official_api_docs_context, run_official_react_task
    from clstr.appworld_routing import read_jsonl
    from clstr.envs.appworld_env import configure_appworld_paths
    from scripts.run_appworld_multistep_executor_eval import _load_dynamic_checkpoint_model
    from scripts.run_appworld_official_executor_eval import (
        CLSTREvidenceBuilder,
        _build_generator,
        _legacy_state_text,
        _metadata,
        _step_execution_failures,
        _task_instruction,
    )

    root, cache = configure_appworld_paths(args.appworld_root, args.appworld_cache)
    tasks = _filter_tasks(read_jsonl(args.tasks_path), task_ids=args.task_ids, max_tasks=args.max_tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, load_report = _load_dynamic_checkpoint_model(
        checkpoint_path=str(args.clstr_checkpoint_path),
        base_skill_pool_path=str(args.base_skill_pool_path),
        dynamic_skill_pool_path=str(args.skill_pool_path),
    )
    evidence_builder = None
    generator = None
    factory = None
    if tasks:
        skills = list(getattr(model, "skills", None) or read_jsonl(args.skill_pool_path))
        controller = CLSTRMultiStepController(
            model=model,
            skills=skills,
            ranking_mode=str(args.ranking_mode),
            candidate_top_k=args.candidate_top_k,
            candidate_source=str(args.candidate_source),
            policy_blend_alpha=float(args.policy_blend_alpha),
            transition_scoring_mode=str(args.transition_scoring_mode),
            transition_residual_lambda=float(args.transition_residual_lambda),
            learned_component_min_range=float(args.learned_component_min_range),
            learned_component_trust_top_k=args.learned_component_trust_top_k,
            recurrent_belief=bool(args.recurrent_belief),
            dedupe_canonical_skills=bool(args.dedupe_canonical_skills),
            max_auth_like_skills=args.max_auth_like_skills,
            appworld_executor_compatible_only=bool(args.appworld_executor_compatible_only),
        )
        evidence_builder = CLSTREvidenceBuilder(
            controller=controller,
            appworld_root=root,
            top_k=int(args.top_k),
            max_chars=int(args.max_evidence_chars),
            state_text_builder=_legacy_state_text,
            handoff_prompt_style=str(args.handoff_prompt_style),
            handoff_visible_skill_limit=args.handoff_visible_skill_limit,
        )
        generator = generator_factory(args) if generator_factory else _build_generator(args)
        generator_metadata = _metadata(generator)
        factory = world_factory or _default_world_factory()
    else:
        generator_metadata = {"backend": "none", "reason": "replay_only_no_live_tasks"}
    appworld_tasks_root = Path(args.appworld_tasks_root) if str(args.appworld_tasks_root or "") else Path(root) / "data" / "tasks"

    _log_event(
        "current_route_online_stage4_start",
        task_count=len(tasks),
        max_online_updates=int(args.max_online_updates),
        output_dir=str(output_dir),
    )

    def rollout_runner(*, task: dict[str, Any], task_index: int, model: Any) -> dict[str, Any]:
        del model
        enriched = enrich_task_with_appworld_specs(task, root)
        task_id = str(enriched.get("task_id") or enriched.get("query_id"))
        query_id = str(enriched.get("query_id") or task_id)
        _log_event(
            "current_route_online_stage4_task_start",
            task_index=task_index,
            task_count=len(tasks),
            task_id=task_id,
            query_id=query_id,
        )
        row: dict[str, Any] = {
            "method": str(args.method),
            "task_id": task_id,
            "query_id": query_id,
            "split": enriched.get("split"),
            "user_goal": _task_instruction(enriched),
            "reset_ok": False,
            "success": False,
            "task_completed": False,
            "evaluation_success": False,
            "steps": [],
        }
        world = None
        try:
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
            evidence_builder.reset(enriched)
            valid_api_refs = load_appworld_api_refs(
                appworld_root=root,
                required_apps=[str(item) for item in enriched.get("required_apps", [])],
                api_refs=[str(item) for item in enriched.get("api_refs", [])],
            )
            api_docs_context = build_official_api_docs_context(appworld_root=root, task=enriched)
            result = run_official_react_task(
                task=enriched,
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
                progress_callback=lambda event, payload: _log_event(
                    event,
                    task_index=task_index,
                    task_count=len(tasks),
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
        _log_event(
            "current_route_online_stage4_task_done",
            task_index=task_index,
            task_count=len(tasks),
            task_id=task_id,
            success=bool(row.get("success", False)),
            task_completed=bool(row.get("task_completed", False)),
            evaluation_success=bool(row.get("evaluation_success", False)),
            steps=len(row.get("steps", [])),
            error=str(row.get("error", "")),
        )
        return row

    failure_correction_builder = None
    if bool(args.enable_failure_stage0_correction):
        def failure_correction_builder(trace: dict[str, Any]) -> dict[str, Any]:
            dataset = build_api_correction_dataset_from_rollouts(
                rollouts=[trace],
                skill_pool_path=str(args.skill_pool_path),
                appworld_tasks_root=appworld_tasks_root,
                max_targets=int(args.failure_correction_max_targets),
            )
            return {
                "report": dataset.report,
                "samples": dataset.samples,
                "stage0_retrieval_rows": dataset.stage0_retrieval_rows,
            }

    report = run_current_route_online_stage4_train_with_model(
        model=model,
        rollout_runner=rollout_runner,
        tasks=tasks,
        output_dir=output_dir,
        max_online_updates=int(args.max_online_updates),
        online_update_epochs=int(args.online_update_epochs),
        replay_success_rollouts_path=str(args.replay_success_rollouts_path or ""),
        max_replay_updates=int(args.max_replay_updates),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        train_transition=bool(args.train_transition),
        loss_score_mode=str(args.loss_score_mode),
        policy_blend_alpha=float(args.policy_blend_alpha),
        transition_residual_lambda=float(args.transition_residual_lambda),
        transition_scoring_mode=str(args.transition_scoring_mode),
        learned_component_min_range=float(args.learned_component_min_range),
        learned_component_trust_top_k=args.learned_component_trust_top_k,
        stability_kl_weight=float(args.stability_kl_weight),
        failure_correction_builder=failure_correction_builder,
        enable_failure_stage4_correction=bool(args.enable_failure_stage4_correction),
        enable_online_stage0_correction_update=bool(args.enable_online_stage0_correction_update),
        online_stage0_learning_rate=args.online_stage0_learning_rate,
        train_online_stage0_skill_adapter=bool(args.train_online_stage0_skill_adapter),
        train_online_stage0_retrieval_scale=bool(args.train_online_stage0_retrieval_scale),
        train_online_stage0_skill_bias=bool(args.train_online_stage0_skill_bias),
        train_online_stage0_encoder_projection=bool(args.train_online_stage0_encoder_projection),
        checkpoint_metadata={
            "source_clstr_checkpoint_path": str(args.clstr_checkpoint_path),
            "base_skill_pool_path": str(args.base_skill_pool_path),
            "dynamic_skill_pool_path": str(args.skill_pool_path),
            "model_load_report": load_report,
            "generator": generator_metadata,
            "tasks_path": str(args.tasks_path),
            "appworld_root": str(root),
            "appworld_cache": str(cache),
            "appworld_tasks_root": str(appworld_tasks_root),
            "enable_failure_stage0_correction": bool(args.enable_failure_stage0_correction),
            "enable_failure_stage4_correction": bool(args.enable_failure_stage4_correction),
            "enable_online_stage0_correction_update": bool(args.enable_online_stage0_correction_update),
            "online_stage0_learning_rate": args.online_stage0_learning_rate,
            "online_update_epochs": int(args.online_update_epochs),
            "replay_success_rollouts_path": str(args.replay_success_rollouts_path or ""),
            "max_replay_updates": int(args.max_replay_updates),
            "ranking_mode": str(args.ranking_mode),
            "loss_score_mode": str(args.loss_score_mode),
            "policy_blend_alpha": float(args.policy_blend_alpha),
            "transition_scoring_mode": str(args.transition_scoring_mode),
            "transition_residual_lambda": float(args.transition_residual_lambda),
            "learned_component_min_range": float(args.learned_component_min_range),
            "learned_component_trust_top_k": args.learned_component_trust_top_k,
            "stability_kl_weight": float(args.stability_kl_weight),
            "handoff_visible_skill_limit": args.handoff_visible_skill_limit,
            "train_online_stage0_skill_adapter": bool(args.train_online_stage0_skill_adapter),
            "train_online_stage0_retrieval_scale": bool(args.train_online_stage0_retrieval_scale),
            "train_online_stage0_skill_bias": bool(args.train_online_stage0_skill_bias),
            "train_online_stage0_encoder_projection": bool(args.train_online_stage0_encoder_projection),
            "failure_correction_max_targets": int(args.failure_correction_max_targets),
        },
    )
    report["model_load_report"] = load_report
    report["generator"] = generator_metadata
    report["tasks_path"] = str(args.tasks_path)
    report["task_count"] = len(tasks)
    (output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _log_event(
        "current_route_online_stage4_done",
        status=report.get("status"),
        online_update_count=report.get("online_update_count"),
        rollout_count=report.get("rollout_count"),
        report_path=str(output_dir / "train_report.json"),
    )
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = run_online_stage4_train(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
