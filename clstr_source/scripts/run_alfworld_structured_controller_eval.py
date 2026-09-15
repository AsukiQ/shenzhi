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
    _load_clstr_alfworld_model,
    _make_env_config,
    make_candidate_scorer,
    make_controller_component_scorer,
    run_alfworld_closed_loop_eval,
)
from clstr.closed_loop_controller import ClosedLoopControllerConfig
from clstr.external_data import write_json
from clstr.qwen_direct_policy import QwenDirectAdmissibleActionScorer, QwenDirectPolicyConfig
from clstr.qwen_planner_intent import QwenPlannerIntentGenerator
from clstr.structured_controller import StructuredCandidateScorer, StructuredComponentScorer


def _env_factory_for(official_repo: Path, train_eval_name: str):
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from alfworld.agents.environment import get_environment

    def env_factory(cfg, train_eval=None, **_):
        del train_eval
        return get_environment(cfg["env"]["type"])(cfg, train_eval=train_eval_name)

    return env_factory


def _weighted(split_reports: list[dict], metric: str) -> float:
    total = sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports)
    if total <= 0:
        return 0.0
    return round(
        sum(float(row["metrics"].get(metric, 0.0)) * int(row["metrics"].get("episode_count", 0)) for row in split_reports)
        / total,
        6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALFWorld CLSTR-Qwen structured controller eval.")
    parser.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--checkpoint_path", default="outputs/clstr_qwen3_structured_train/checkpoints/clstr_full_base-step1000.pt")
    parser.add_argument("--output_dir", default="outputs/alfworld_eval/clstr_qwen3_structured_gate_valid_seen")
    parser.add_argument("--splits", nargs="+", default=["valid_seen"])
    parser.add_argument("--run_name", default="clstr_qwen3_structured_gate_valid_seen")
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--aux_data_root", default="data/clstr_full_base_train")
    parser.add_argument("--controller_mode", default="policy_plus_transition_belief_stop_loop_penalty")
    parser.add_argument("--planner_weight", type=float, default=3.0)
    parser.add_argument("--q_success_weight", type=float, default=0.0)
    parser.add_argument("--qwen_model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    data_dir_value = args.data_dir or os.environ.get("ALFWORLD_DATA")
    if not data_dir_value:
        raise ValueError("--data_dir or ALFWORLD_DATA is required")
    official_repo = Path(args.official_repo)
    data_dir = Path(data_dir_value)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, action_adapter, routing_report = _load_clstr_alfworld_model(
        routing_init_manifest=Path(args.routing_init_manifest),
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        output_dir=output_dir,
        data_root=Path(args.aux_data_root),
    )
    base_candidate_scorer = make_candidate_scorer(model, action_adapter)
    base_component_scorer = make_controller_component_scorer(model)
    qwen_scorer = QwenDirectAdmissibleActionScorer(
        QwenDirectPolicyConfig(
            model_name_or_path=str(args.qwen_model_name_or_path),
            cache_dir=str(args.cache_dir),
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            fallback_strategy="first_admissible",
            use_chat_template=True,
            enable_thinking=False,
        )
    )
    planner = QwenPlannerIntentGenerator(qwen_scorer)
    structured_candidate_scorer = StructuredCandidateScorer(base_candidate_scorer, planner)
    structured_component_scorer = StructuredComponentScorer(base_component_scorer, structured_candidate_scorer)
    controller_config = ClosedLoopControllerConfig(
        mode=args.controller_mode,
        planner_weight=args.planner_weight,
        q_success_weight=args.q_success_weight,
    )
    max_episodes = None if args.max_episodes is not None and args.max_episodes <= 0 else args.max_episodes
    split_reports = []
    combined_run_lines: list[str] = []
    for split in args.splits:
        split_output = output_dir if len(args.splits) == 1 else output_dir / split
        config, train_eval_name = _make_env_config(official_repo, data_dir, split)
        report = run_alfworld_closed_loop_eval(
            env_factory=_env_factory_for(official_repo, train_eval_name),
            config=config,
            split=split,
            candidate_scorer=structured_candidate_scorer,
            component_scorer=structured_component_scorer,
            controller_config=controller_config,
            output_dir=split_output,
            max_episodes=max_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            run_name=args.run_name,
        )
        metrics_path = split_output / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(
            {
                "trajectory_trained": args.checkpoint_path is not None,
                "training_data": "verified Qwen teacher rollout from ALFWorld train only",
                "policy_family": "clstr_qwen_structured_controller",
                "uses_clstr": True,
                "is_clstr_result": True,
                "qwen_direct_baseline": False,
                "qwen_planner_reference": True,
                "qwen_frozen": True,
                "qwen_direct_generator": False,
                "qwen_direct_generator_in_clstr": False,
                "clstr_final_action": True,
                "structured_controller": True,
                "planner_weight": args.planner_weight,
                "q_success_weight": args.q_success_weight,
                "controller_mode": args.controller_mode,
                "checkpoint_path": args.checkpoint_path,
                "routing_init_manifest": args.routing_init_manifest,
                "qdoc_adapter_used": False,
                "transition_belief_stop_participate_in_action_selection": True,
                "routing_init_report": routing_report,
            }
        )
        write_json(metrics_path, metrics)
        split_reports.append({"status": "ok", "metrics": metrics})
        run_path = split_output / "run.jsonl"
        if run_path.exists():
            combined_run_lines.extend(run_path.read_text(encoding="utf-8").splitlines())
    if len(args.splits) > 1:
        total = sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports)
        aggregate = {
            "status": "ok",
            "method": args.run_name,
            "splits": args.splits,
            "success_rate": _weighted(split_reports, "success_rate"),
            "average_reward": _weighted(split_reports, "average_reward"),
            "average_goal_condition_points": _weighted(split_reports, "average_goal_condition_points"),
            "average_episode_steps": _weighted(split_reports, "average_episode_steps"),
            "episode_count": total,
            "episodes": total,
            "trajectory_trained": args.checkpoint_path is not None,
            "training_data": "verified Qwen teacher rollout from ALFWorld train only",
            "policy_family": "clstr_qwen_structured_controller",
            "uses_clstr": True,
            "is_clstr_result": True,
            "qwen_direct_baseline": False,
            "qwen_planner_reference": True,
            "qwen_frozen": True,
            "qwen_direct_generator": False,
            "qwen_direct_generator_in_clstr": False,
            "clstr_final_action": True,
            "structured_controller": True,
            "planner_weight": args.planner_weight,
            "q_success_weight": args.q_success_weight,
            "controller_mode": args.controller_mode,
            "checkpoint_path": args.checkpoint_path,
            "split_metrics": [row["metrics"] for row in split_reports],
        }
        write_json(output_dir / "metrics.json", aggregate)
        (output_dir / "run.jsonl").write_text("\n".join(combined_run_lines) + "\n", encoding="utf-8")
        report = {"status": "ok", "metrics": aggregate, "split_reports": split_reports}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
