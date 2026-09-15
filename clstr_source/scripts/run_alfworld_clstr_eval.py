#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import (
    build_alfworld_env_report,
    build_alfworld_policy_diagnostic_report,
    build_alfworld_protocol_report,
    evaluate_alfworld_clstr,
)
from clstr.external_data import write_json
from clstr.memory_utility_gate import RELIABILITY_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR ALFWorld protocol reporting and closed-loop eval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("protocol-report", help="Inspect the official ALFWorld repo protocol.")
    protocol.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    protocol.add_argument("--output_path", default="outputs/alfworld_eval/protocol_report.json")
    protocol.add_argument("--data_dir", default=None)
    protocol.add_argument("--use_network_turbo", action="store_true")

    env_report = subparsers.add_parser("env-report", help="Record ALFWorld data/env readiness.")
    env_report.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    env_report.add_argument("--output_path", default="outputs/alfworld_eval/env_report.json")
    env_report.add_argument("--data_dir", default=None)
    env_report.add_argument("--use_network_turbo", action="store_true")

    diagnostic = subparsers.add_parser("diagnostic-report", help="Analyze existing ALFWorld CLSTR action traces.")
    diagnostic.add_argument("--eval_root", default="outputs/alfworld_eval")
    diagnostic.add_argument("--output_path", default="outputs/alfworld_policy_replay/diagnostic_report.json")
    diagnostic.add_argument(
        "--methods",
        nargs="+",
        default=["clstr_routing_init_baseline", "clstr_aux_full_obs_cosine"],
    )

    eval_cmd = subparsers.add_parser("eval", help="Run CLSTR closed-loop ALFWorld eval.")
    eval_cmd.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    eval_cmd.add_argument("--data_dir", default=None)
    eval_cmd.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    eval_cmd.add_argument("--checkpoint_path", default=None)
    eval_cmd.add_argument("--stage4_checkpoint_path", default=None)
    eval_cmd.add_argument("--skill_rows_path_override", default=None)
    eval_cmd.add_argument("--stage0_checkpoint_path", default=None)
    eval_cmd.add_argument("--benchmark_skill_rows_path", default=None)
    eval_cmd.add_argument(
        "--scorer_mode",
        default="legacy_concrete_action_head",
        choices=[
            "legacy_concrete_action_head",
            "unified_memory_admissible_action",
            "unified_memory_concrete_action",
        ],
    )
    eval_cmd.add_argument("--replay_prefix_max_steps", type=int, default=6)
    eval_cmd.add_argument("--memory_protocol", default="stateful_post_action_v1")
    eval_cmd.add_argument(
        "--reliability_mode",
        default="dynamic",
        choices=sorted(RELIABILITY_MODES),
    )
    eval_cmd.add_argument("--fixed_alpha", type=float, default=1.0)
    eval_cmd.add_argument("--safe_memory_residual_bound", type=float, default=2.0)
    eval_cmd.add_argument("--feature_update_count_cap", type=float, default=1.0)
    eval_cmd.add_argument("--feature_candidate_count_cap", type=float, default=1.0)
    eval_cmd.add_argument("--memory_utility_gate_checkpoint_path")
    eval_cmd.add_argument("--expected_memory_utility_gate_checkpoint_sha256")
    eval_cmd.add_argument("--expected_memory_utility_gate_audit_sha256")
    eval_cmd.add_argument("--output_dir", default="outputs/alfworld_eval/clstr_eval")
    eval_cmd.add_argument("--splits", nargs="+", default=["valid_seen"])
    eval_cmd.add_argument("--run_name", default="clstr_eval")
    eval_cmd.add_argument("--max_episodes", type=int, default=None)
    eval_cmd.add_argument("--max_steps", type=int, default=50)
    eval_cmd.add_argument("--batch_size", type=int, default=1)
    eval_cmd.add_argument("--aux_data_root", default="data/aux_trajectories")
    eval_cmd.add_argument(
        "--include_available_actions_in_state",
        action="store_true",
        help="Encode state plus AVAILABLE ACTIONS as a latent Qwen planner/query context before CLSTR ranks candidates.",
    )
    eval_cmd.add_argument(
        "--controller_mode",
        default="policy_only",
        choices=[
            "policy_only",
            "policy_plus_transition",
            "policy_plus_transition_belief",
            "policy_plus_transition_belief_stop",
            "policy_plus_transition_belief_stop_loop_penalty",
        ],
    )
    eval_cmd.add_argument("--q_success_weight", type=float, default=0.0)
    eval_cmd.add_argument("--use_network_turbo", action="store_true")
    eval_cmd.add_argument("--protocol_report_path", default="outputs/alfworld_eval/protocol_report.json")
    eval_cmd.add_argument("--env_report_path", default="outputs/alfworld_eval/env_report.json")

    args = parser.parse_args()

    if args.command == "protocol-report":
        report = build_alfworld_protocol_report(
            official_repo=Path(args.official_repo),
            output_path=Path(args.output_path),
            data_dir=Path(args.data_dir) if args.data_dir else None,
            use_network_turbo=args.use_network_turbo,
        )
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.command == "env-report":
        report = build_alfworld_env_report(
            official_repo=Path(args.official_repo),
            data_dir=Path(args.data_dir) if args.data_dir else os.environ.get("ALFWORLD_DATA"),
            use_network_turbo=args.use_network_turbo,
        )
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.command == "diagnostic-report":
        report = build_alfworld_policy_diagnostic_report(
            eval_root=Path(args.eval_root),
            output_path=Path(args.output_path),
            methods=args.methods,
        )
        print(json.dumps(report, ensure_ascii=False))
        return

    data_dir_value = args.data_dir or os.environ.get("ALFWORLD_DATA")
    if not data_dir_value:
        raise ValueError("--data_dir or ALFWORLD_DATA is required")
    data_dir = Path(data_dir_value)
    build_alfworld_protocol_report(
        official_repo=Path(args.official_repo),
        output_path=Path(args.protocol_report_path),
        data_dir=data_dir,
        use_network_turbo=args.use_network_turbo,
    )
    env_ready_report = build_alfworld_env_report(
        official_repo=Path(args.official_repo),
        data_dir=data_dir,
        use_network_turbo=args.use_network_turbo,
    )
    Path(args.env_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.env_report_path).write_text(json.dumps(env_ready_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = []
    combined_run_lines = []
    for split in args.splits:
        split_output = output_dir if len(args.splits) == 1 else output_dir / split
        report = evaluate_alfworld_clstr(
            official_repo=Path(args.official_repo),
            data_dir=data_dir,
            output_dir=split_output,
            routing_init_manifest=Path(args.routing_init_manifest),
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            stage4_checkpoint_path=Path(args.stage4_checkpoint_path) if args.stage4_checkpoint_path else None,
            skill_rows_path_override=(
                Path(args.skill_rows_path_override)
                if args.skill_rows_path_override
                else None
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
            scorer_mode=args.scorer_mode,
            replay_prefix_max_steps=args.replay_prefix_max_steps,
            split=split,
            run_name=args.run_name,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            aux_data_root=Path(args.aux_data_root),
            controller_mode=args.controller_mode,
            include_available_actions_in_state=args.include_available_actions_in_state,
            q_success_weight=args.q_success_weight,
            reliability_mode=args.reliability_mode,
            fixed_alpha=args.fixed_alpha,
            safe_memory_residual_bound=args.safe_memory_residual_bound,
            feature_update_count_cap=args.feature_update_count_cap,
            feature_candidate_count_cap=args.feature_candidate_count_cap,
            memory_utility_gate_checkpoint_path=args.memory_utility_gate_checkpoint_path,
            expected_memory_utility_gate_checkpoint_sha256=args.expected_memory_utility_gate_checkpoint_sha256,
            expected_memory_utility_gate_audit_sha256=args.expected_memory_utility_gate_audit_sha256,
            memory_protocol=args.memory_protocol,
        )
        split_reports.append(report)
        run_path = split_output / "run.jsonl"
        if run_path.exists():
            combined_run_lines.extend(run_path.read_text(encoding="utf-8").splitlines())
    if len(args.splits) > 1:
        total_episodes = sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports)
        def weighted(metric: str) -> float:
            if total_episodes <= 0:
                return 0.0
            return round(
                sum(float(row["metrics"].get(metric, 0.0)) * int(row["metrics"].get("episode_count", 0)) for row in split_reports)
                / total_episodes,
                6,
            )
        def first_metric(name: str, default=None):
            for row in split_reports:
                value = row["metrics"].get(name)
                if value is not None:
                    return value
            return default
        aggregate = {
            "status": "ok",
            "method": args.run_name,
            "splits": args.splits,
            "success_rate": weighted("success_rate"),
            "average_reward": weighted("average_reward"),
            "average_goal_condition_points": weighted("average_goal_condition_points"),
            "average_episode_steps": weighted("average_episode_steps"),
            "episode_count": total_episodes,
            "episodes": total_episodes,
            "trajectory_trained": args.checkpoint_path is not None,
            "training_data": first_metric("training_data", "none"),
            "training_objective": first_metric("training_objective"),
            "policy_head_type": first_metric("policy_head_type"),
            "native_clstr_heads_used": bool(first_metric("native_clstr_heads_used", False)),
            "legacy_universal_action_adapter_role": first_metric("legacy_universal_action_adapter_role"),
            "qdoc_adapter_used": bool(first_metric("qdoc_adapter_used", False)),
            "skill_source_path": first_metric("skill_source_path"),
            "skill_count": first_metric("skill_count"),
            "controller_mode": args.controller_mode,
            "q_success_weight": args.q_success_weight,
            "q_success_participates_in_action_selection": bool(
                float(args.q_success_weight) != 0.0 and args.controller_mode != "policy_only"
            ),
            "qwen_planner_available_actions": bool(args.include_available_actions_in_state),
            "planner_intent_mode": "latent_available_actions_query" if args.include_available_actions_in_state else "none",
            "routing_init_manifest": args.routing_init_manifest,
            "checkpoint_path": args.checkpoint_path,
            "stage4_checkpoint_path": args.stage4_checkpoint_path,
            "skill_rows_path_override": args.skill_rows_path_override,
            "stage0_checkpoint_path": args.stage0_checkpoint_path,
            "benchmark_skill_rows_path": args.benchmark_skill_rows_path,
            "scorer_mode": args.scorer_mode,
            "replay_prefix_max_steps": args.replay_prefix_max_steps,
            "safe_memory_residual_bound": args.safe_memory_residual_bound,
            "split_metrics": [row["metrics"] for row in split_reports],
        }
        write_json(output_dir / "metrics.json", aggregate)
        (output_dir / "run.jsonl").write_text("\n".join(combined_run_lines) + "\n", encoding="utf-8")
        report = {"status": "ok", "metrics": aggregate, "split_reports": split_reports}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
