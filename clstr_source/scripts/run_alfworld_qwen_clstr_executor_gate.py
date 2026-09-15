#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import (  # noqa: E402
    ClstrUnifiedMemoryAdmissibleActionScorer,
    ClstrUnifiedMemoryConcreteActionScorer,
    SkillRouterAdmissibleActionScorer,
    _load_clstr_alfworld_model,
    _make_env_config,
    make_candidate_scorer,
    run_alfworld_closed_loop_eval,
)
from clstr.alfworld_qwen_clstr_gate import QwenClstrHybridActionScorer  # noqa: E402
from clstr.closed_loop_controller import ClosedLoopControllerConfig  # noqa: E402
from clstr.external_data import write_json  # noqa: E402
from clstr.qwen_direct_policy import (  # noqa: E402
    QwenDirectAdmissibleActionScorer,
    QwenDirectLikelihoodActionScorer,
    QwenDirectPolicyConfig,
)


class _LoopGuardComponentScorer:
    """Zero-valued components used only to activate generic closed-loop penalties."""

    def __init__(self) -> None:
        self.last_metadata: list[dict[str, Any]] = []

    def __call__(
        self,
        state_texts: list[str],
        candidate_rows: list[list[str]],
        policy_scores,
        action_histories: list[list[str]],
    ) -> dict[str, Any]:
        del state_texts, action_histories
        zeros = policy_scores.new_zeros(policy_scores.shape)
        self.last_metadata = [
            {
                "component_score_source": "loop_guard_only_zero_components",
                "transition_score_source": "disabled_executor_gate_loop_guard",
                "belief_score_source": "disabled_executor_gate_loop_guard",
                "stop_score_source": "disabled_executor_gate_loop_guard",
                "candidate_count": len(candidates),
            }
            for candidates in candidate_rows
        ]
        return {
            "transition_scores": zeros,
            "belief_scores": zeros,
            "stop_logits": zeros,
        }


def _env_factory_for(official_repo: Path, train_eval_name: str):
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(cfg, train_eval=train_eval_name)

    return env_factory


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _weighted(split_reports: list[dict[str, Any]], metric: str) -> float:
    total = sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports)
    if total <= 0:
        return 0.0
    return round(
        sum(float(row["metrics"].get(metric, 0.0)) * int(row["metrics"].get("episode_count", 0)) for row in split_reports)
        / total,
        6,
    )


def _summarize_policy_trace(run_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_path)
    total_steps = 0
    qwen_hybrid_disagree = 0
    qwen_clstr_disagree = 0
    clstr_hybrid_agree = 0
    qwen_fallbacks = 0
    clstr_recurrent_mt_steps = 0
    clstr_replay_prefix_steps = 0
    clstr_override_allowed = 0
    clstr_override_blocked = 0
    for row in rows:
        for item in row.get("policy_metadata_trace", []) or []:
            if not isinstance(item, dict):
                continue
            total_steps += 1
            qwen_action = item.get("qwen_chosen_action")
            clstr_action = item.get("clstr_chosen_action")
            hybrid_action = item.get("hybrid_chosen_action")
            if qwen_action and hybrid_action:
                qwen_hybrid_disagree += int(str(qwen_action) != str(hybrid_action))
            if qwen_action and clstr_action:
                qwen_clstr_disagree += int(str(qwen_action) != str(clstr_action))
            if clstr_action and hybrid_action:
                clstr_hybrid_agree += int(str(clstr_action) == str(hybrid_action))
            qwen_fallbacks += int(bool(item.get("qwen_fallback_used", item.get("fallback_used", False))))
            clstr_recurrent_mt_steps += int(bool(item.get("uses_recurrent_m_t", False)))
            clstr_replay_prefix_steps += int(float(item.get("clstr_replay_prefix_len", 0) or 0) > 0)
            if "clstr_override_allowed" in item:
                if bool(item.get("clstr_override_allowed", False)):
                    clstr_override_allowed += 1
                else:
                    clstr_override_blocked += 1
    return {
        "episode_count": len(rows),
        "total_steps": total_steps,
        "qwen_clstr_disagreement_rate": round(qwen_clstr_disagree / max(1, total_steps), 6),
        "qwen_hybrid_disagreement_rate": round(qwen_hybrid_disagree / max(1, total_steps), 6),
        "clstr_hybrid_agreement_rate": round(clstr_hybrid_agree / max(1, total_steps), 6),
        "qwen_fallback_rate": round(qwen_fallbacks / max(1, total_steps), 6),
        "clstr_recurrent_mt_step_rate": round(clstr_recurrent_mt_steps / max(1, total_steps), 6),
        "clstr_replay_prefix_step_rate": round(clstr_replay_prefix_steps / max(1, total_steps), 6),
        "clstr_recurrent_mt_steps": clstr_recurrent_mt_steps,
        "clstr_override_allowed_steps": clstr_override_allowed,
        "clstr_override_blocked_steps": clstr_override_blocked,
        "clstr_override_blocked_rate": round(clstr_override_blocked / max(1, total_steps), 6),
    }


def _run_method(
    *,
    official_repo: Path,
    data_dir: Path,
    output_dir: Path,
    splits: list[str],
    run_name: str,
    scorer,
    max_episodes: int | None,
    max_steps: int,
    batch_size: int,
    extra_metrics: dict[str, Any],
    component_scorer=None,
    controller_config: ClosedLoopControllerConfig | None = None,
) -> dict[str, Any]:
    split_reports: list[dict[str, Any]] = []
    combined_run_lines: list[str] = []
    for split in splits:
        split_output = output_dir if len(splits) == 1 else output_dir / split
        config, train_eval_name = _make_env_config(official_repo, data_dir, split)
        result = run_alfworld_closed_loop_eval(
            env_factory=_env_factory_for(official_repo, train_eval_name),
            config=config,
            split=split,
            candidate_scorer=scorer,
            output_dir=split_output,
            max_episodes=max_episodes,
            max_steps=max_steps,
            batch_size=batch_size,
            run_name=run_name,
            component_scorer=component_scorer,
            controller_config=controller_config,
        )
        metrics_path = split_output / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(extra_metrics)
        metrics["trace_summary"] = _summarize_policy_trace(split_output / "run.jsonl")
        write_json(metrics_path, metrics)
        split_reports.append({"status": "ok", "metrics": metrics, "result": result})
        run_path = split_output / "run.jsonl"
        if run_path.exists():
            combined_run_lines.extend(run_path.read_text(encoding="utf-8").splitlines())
    if len(splits) == 1:
        return split_reports[0]
    aggregate = {
        "status": "ok",
        "method": run_name,
        "splits": splits,
        "success_rate": _weighted(split_reports, "success_rate"),
        "average_reward": _weighted(split_reports, "average_reward"),
        "average_goal_condition_points": _weighted(split_reports, "average_goal_condition_points"),
        "average_episode_steps": _weighted(split_reports, "average_episode_steps"),
        "episode_count": sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports),
        "episodes": sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports),
        "split_metrics": [row["metrics"] for row in split_reports],
    }
    aggregate.update(extra_metrics)
    write_json(output_dir / "metrics.json", aggregate)
    (output_dir / "run.jsonl").write_text("\n".join(combined_run_lines) + "\n", encoding="utf-8")
    return {"status": "ok", "metrics": aggregate, "split_reports": split_reports}


def _delta(hybrid: dict[str, Any] | None, qwen_only: dict[str, Any] | None) -> dict[str, Any]:
    if not hybrid or not qwen_only:
        return {}
    hybrid_metrics = hybrid.get("metrics", {})
    qwen_metrics = qwen_only.get("metrics", {})
    return {
        "success_rate_delta": round(float(hybrid_metrics.get("success_rate", 0.0)) - float(qwen_metrics.get("success_rate", 0.0)), 6),
        "average_reward_delta": round(float(hybrid_metrics.get("average_reward", 0.0)) - float(qwen_metrics.get("average_reward", 0.0)), 6),
        "average_goal_condition_points_delta": round(
            float(hybrid_metrics.get("average_goal_condition_points", 0.0))
            - float(qwen_metrics.get("average_goal_condition_points", 0.0)),
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a low-cost ALFWorld Qwen executor vs Qwen+CLSTR prior gate.")
    parser.add_argument("--official_repo", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/alfworld_eval/qwen_clstr_executor_gate_smoke")
    parser.add_argument("--splits", nargs="+", default=["valid_seen"])
    parser.add_argument("--run_name", default="qwen_clstr_executor_gate")
    parser.add_argument("--max_episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/"
            "checkpoints/clstr_full_base-step10000.pt"
        ),
    )
    parser.add_argument("--stage4_checkpoint_path", default=None)
    parser.add_argument("--stage0_checkpoint_path", default=None)
    parser.add_argument("--training_skills_path", default=None)
    parser.add_argument("--benchmark_skill_rows_path", default=None)
    parser.add_argument("--skill_rows_path_override", default=None)
    parser.add_argument("--aux_data_root", default="data/clstr_unified_pretrain_v4_2_progressive_final")
    parser.add_argument("--qwen_model_name_or_path", default="models/Qwen3-14B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--fallback_strategy", default="first_admissible")
    parser.add_argument("--scoring_method", default="generate", choices=["generate", "likelihood"])
    parser.add_argument("--likelihood_batch_size", type=int, default=4)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--qwen_weight", type=float, default=1.0)
    parser.add_argument("--clstr_weight", type=float, default=0.25)
    parser.add_argument("--hybrid_prior", default="clstr", choices=["clstr", "skillrouter"])
    parser.add_argument("--skillrouter_model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--skillrouter_adapter_checkpoint_path", default=None)
    parser.add_argument("--skillrouter_encode_batch_size", type=int, default=16)
    parser.add_argument("--skillrouter_max_length", type=int, default=2048)
    parser.add_argument("--qwen_score_mode", default="auto", choices=["auto", "raw", "proposal_bonus"])
    parser.add_argument(
        "--clstr_prior_mode",
        default="unified_memory_concrete_action",
        choices=["unified_memory", "unified_memory_concrete_action", "static_action"],
    )
    parser.add_argument("--clstr_replay_prefix_max_steps", type=int, default=6)
    parser.add_argument("--reliability_mode", default="cmc")
    parser.add_argument("--fixed_alpha", type=float, default=1.0)
    parser.add_argument("--safe_memory_residual_bound", type=float, default=2.0)
    parser.add_argument("--feature_update_count_cap", type=float, default=16.0)
    parser.add_argument("--feature_candidate_count_cap", type=float, default=256.0)
    parser.add_argument("--no_normalize_scores", action="store_true")
    parser.add_argument("--include_available_actions_in_state", action="store_true")
    parser.add_argument("--loop_guard", action="store_true")
    parser.add_argument("--run_qwen_only", action="store_true")
    parser.add_argument("--run_hybrid", action="store_true")
    args = parser.parse_args()

    data_dir_value = args.data_dir or os.environ.get("ALFWORLD_DATA")
    if not data_dir_value:
        raise ValueError("--data_dir or ALFWORLD_DATA is required")
    final_chain_inputs = {
        "stage0_checkpoint_path": args.stage0_checkpoint_path,
        "checkpoint_path": args.checkpoint_path,
        "stage4_checkpoint_path": args.stage4_checkpoint_path,
        "training_skills_path": args.training_skills_path,
        "benchmark_skill_rows_path": args.benchmark_skill_rows_path,
    }
    final_chain_requested = any(
        (
            args.stage0_checkpoint_path,
            args.training_skills_path,
            args.benchmark_skill_rows_path,
        )
    )
    supplied_final_chain_inputs = {
        key for key, value in final_chain_inputs.items() if value
    }
    if final_chain_requested and supplied_final_chain_inputs != set(final_chain_inputs):
        missing = sorted(set(final_chain_inputs) - supplied_final_chain_inputs)
        raise ValueError(
            "final-chain ALFWorld loading requires all checkpoint and skill inputs: "
            + ", ".join(missing)
        )
    run_qwen_only = bool(args.run_qwen_only)
    run_hybrid = bool(args.run_hybrid)
    if not run_qwen_only and not run_hybrid:
        run_qwen_only = True
        run_hybrid = True
    controller_config = (
        ClosedLoopControllerConfig(
            mode="policy_plus_transition_belief_stop_loop_penalty",
            transition_weight=0.0,
            belief_weight=0.0,
            stop_weight=0.0,
            planner_weight=0.0,
            q_success_weight=0.0,
        )
        if args.loop_guard
        else None
    )
    max_episodes = None if args.max_episodes is not None and args.max_episodes <= 0 else args.max_episodes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer_cls = QwenDirectLikelihoodActionScorer if args.scoring_method == "likelihood" else QwenDirectAdmissibleActionScorer
    qwen_scorer = scorer_cls(
        QwenDirectPolicyConfig(
            model_name_or_path=str(args.qwen_model_name_or_path),
            cache_dir=str(args.cache_dir),
            local_files_only=bool(args.local_files_only),
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            fallback_strategy=args.fallback_strategy,
            likelihood_batch_size=args.likelihood_batch_size,
            use_chat_template=True,
            enable_thinking=False,
        )
    )

    qwen_report = None
    if run_qwen_only:
        qwen_report = _run_method(
            official_repo=Path(args.official_repo),
            data_dir=Path(data_dir_value),
            output_dir=output_dir / "qwen_only",
            splits=list(args.splits),
            run_name=f"{args.run_name}_qwen_only",
            scorer=qwen_scorer,
            max_episodes=max_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            component_scorer=_LoopGuardComponentScorer() if args.loop_guard else None,
            controller_config=controller_config,
            extra_metrics={
                "policy_family": "qwen_direct_admissible",
                "uses_clstr": False,
                "qwen_direct_baseline": True,
                "loop_guard": bool(args.loop_guard),
                "scoring_method": args.scoring_method,
                "model_name_or_path": str(args.qwen_model_name_or_path),
                "checkpoint_path": None,
                "executor_gate": True,
            },
        )

    hybrid_report = None
    if run_hybrid:
        qwen_score_mode = args.qwen_score_mode
        if qwen_score_mode == "auto":
            qwen_score_mode = "proposal_bonus" if args.scoring_method == "generate" else "raw"
        routing_report = None
        if args.hybrid_prior == "skillrouter":
            prior_scorer = SkillRouterAdmissibleActionScorer(
                model_name_or_path=args.skillrouter_model_name_or_path,
                batch_size=args.skillrouter_encode_batch_size,
                max_length=args.skillrouter_max_length,
                adapter_checkpoint_path=args.skillrouter_adapter_checkpoint_path,
            )
            prior_policy_family = "qwen_skillrouter_hybrid_executor_gate"
            output_suffix = "qwen_plus_skillrouter"
        else:
            model, action_adapter, routing_report = _load_clstr_alfworld_model(
                routing_init_manifest=Path(args.routing_init_manifest),
                checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
                stage4_checkpoint_path=Path(args.stage4_checkpoint_path) if args.stage4_checkpoint_path else None,
                skill_rows_path_override=(
                    Path(args.training_skills_path)
                    if args.training_skills_path
                    else (
                        Path(args.skill_rows_path_override)
                        if args.skill_rows_path_override
                        else None
                    )
                ),
                stage0_checkpoint_path=(
                    Path(args.stage0_checkpoint_path)
                    if args.stage0_checkpoint_path
                    else None
                ),
                benchmark_skill_rows_path=(
                    Path(args.benchmark_skill_rows_path)
                    if args.benchmark_skill_rows_path
                    else None
                ),
                output_dir=output_dir / "model_cache",
                data_root=Path(args.aux_data_root),
            )
            if args.clstr_prior_mode == "unified_memory":
                prior_scorer = ClstrUnifiedMemoryAdmissibleActionScorer(
                    model,
                    replay_prefix_max_steps=args.clstr_replay_prefix_max_steps,
                    reliability_mode=args.reliability_mode,
                    fixed_alpha=args.fixed_alpha,
                    safe_memory_residual_bound=args.safe_memory_residual_bound,
                    feature_update_count_cap=args.feature_update_count_cap,
                    feature_candidate_count_cap=args.feature_candidate_count_cap,
                )
            elif args.clstr_prior_mode == "unified_memory_concrete_action":
                prior_scorer = ClstrUnifiedMemoryConcreteActionScorer(
                    model,
                    replay_prefix_max_steps=args.clstr_replay_prefix_max_steps,
                    reliability_mode=args.reliability_mode,
                    fixed_alpha=args.fixed_alpha,
                    safe_memory_residual_bound=args.safe_memory_residual_bound,
                    feature_update_count_cap=args.feature_update_count_cap,
                    feature_candidate_count_cap=args.feature_candidate_count_cap,
                )
            else:
                prior_scorer = make_candidate_scorer(
                    model,
                    action_adapter,
                    include_available_actions_in_state=bool(args.include_available_actions_in_state),
                )
            prior_policy_family = "qwen_clstr_hybrid_executor_gate"
            output_suffix = "qwen_plus_clstr"
        hybrid_scorer = QwenClstrHybridActionScorer(
            qwen_scorer=qwen_scorer,
            clstr_scorer=prior_scorer,
            qwen_weight=args.qwen_weight,
            clstr_weight=args.clstr_weight,
            normalize_scores=not bool(args.no_normalize_scores),
            qwen_score_mode=qwen_score_mode,
            prior_name=args.hybrid_prior,
            policy_family=prior_policy_family,
        )
        hybrid_report = _run_method(
            official_repo=Path(args.official_repo),
            data_dir=Path(data_dir_value),
            output_dir=output_dir / output_suffix,
            splits=list(args.splits),
            run_name=f"{args.run_name}_{output_suffix}",
            scorer=hybrid_scorer,
            max_episodes=max_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            component_scorer=_LoopGuardComponentScorer() if args.loop_guard else None,
            controller_config=controller_config,
            extra_metrics={
                "policy_family": prior_policy_family,
                "uses_clstr": args.hybrid_prior == "clstr",
                "uses_skillrouter": args.hybrid_prior == "skillrouter",
                "is_clstr_result": args.hybrid_prior == "clstr",
                "is_skillrouter_result": args.hybrid_prior == "skillrouter",
                "qwen_direct_baseline": False,
                "scoring_method": args.scoring_method,
                "model_name_or_path": str(args.qwen_model_name_or_path),
                "checkpoint_path": str(args.checkpoint_path) if args.hybrid_prior == "clstr" and args.checkpoint_path else None,
                "stage4_checkpoint_path": (
                    str(args.stage4_checkpoint_path)
                    if args.hybrid_prior == "clstr" and args.stage4_checkpoint_path
                    else None
                ),
                "stage4_overlay_loaded": bool((routing_report or {}).get("stage4_checkpoint"))
                if args.hybrid_prior == "clstr"
                else False,
                "skill_rows_path_override": (
                    str(args.skill_rows_path_override)
                    if args.hybrid_prior == "clstr" and args.skill_rows_path_override
                    else None
                ),
                "routing_init_manifest": str(args.routing_init_manifest) if args.hybrid_prior == "clstr" else None,
                "aux_data_root": str(args.aux_data_root) if args.hybrid_prior == "clstr" else None,
                "hybrid_prior": args.hybrid_prior,
                "skillrouter_model_name_or_path": (
                    str(args.skillrouter_model_name_or_path) if args.hybrid_prior == "skillrouter" else None
                ),
                "skillrouter_adapter_checkpoint_path": (
                    str(args.skillrouter_adapter_checkpoint_path)
                    if args.hybrid_prior == "skillrouter" and args.skillrouter_adapter_checkpoint_path
                    else None
                ),
                "qwen_weight": float(args.qwen_weight),
                "clstr_weight": float(args.clstr_weight),
                f"{args.hybrid_prior}_weight": float(args.clstr_weight),
                "qwen_score_mode": qwen_score_mode,
                "score_normalization": (
                    "none"
                    if args.no_normalize_scores
                    else (
                        f"qwen:proposal_bonus;{args.hybrid_prior}:row_zscore"
                        if qwen_score_mode == "proposal_bonus"
                        else "row_zscore"
                    )
                ),
                "loop_guard": bool(args.loop_guard),
                "include_available_actions_in_state": bool(args.include_available_actions_in_state),
                "clstr_prior_mode": args.clstr_prior_mode if args.hybrid_prior == "clstr" else None,
                "clstr_replay_prefix_max_steps": (
                    int(args.clstr_replay_prefix_max_steps) if args.hybrid_prior == "clstr" else None
                ),
                "memory_utility_reliability_mode": (
                    args.reliability_mode if args.hybrid_prior == "clstr" else None
                ),
                "feature_update_count_cap": (
                    float(args.feature_update_count_cap)
                    if args.hybrid_prior == "clstr"
                    else None
                ),
                "feature_candidate_count_cap": (
                    float(args.feature_candidate_count_cap)
                    if args.hybrid_prior == "clstr"
                    else None
                ),
                "routing_init_report": routing_report,
                "executor_gate": True,
                "not_rl_training": True,
            },
        )

    summary = {
        "status": "ok",
        "method": args.run_name,
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "max_episodes": max_episodes,
        "max_steps": int(args.max_steps),
        "qwen_only": qwen_report.get("metrics") if qwen_report else None,
        "qwen_plus_clstr": hybrid_report.get("metrics") if hybrid_report and args.hybrid_prior == "clstr" else None,
        "qwen_plus_skillrouter": (
            hybrid_report.get("metrics") if hybrid_report and args.hybrid_prior == "skillrouter" else None
        ),
        "delta": _delta(hybrid_report, qwen_report),
        "decision_rule": "continue_to_online_rl_only_if_qwen_plus_clstr_beats_qwen_only_on_reward_or_success",
    }
    write_json(output_dir / "gate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
