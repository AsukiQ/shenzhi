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

from clstr.alfworld_eval import build_alfworld_env_report, build_alfworld_protocol_report, evaluate_alfworld_qwen_direct
from clstr.external_data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-8B direct ALFWorld admissible-action baseline.")
    parser.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--output_dir", default="outputs/alfworld_eval/qwen3_8b_direct_valid_seen")
    parser.add_argument("--splits", nargs="+", default=["valid_seen"])
    parser.add_argument("--run_name", default="qwen3_8b_direct")
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--fallback_strategy", default="first_admissible")
    parser.add_argument("--scoring_method", default="generate", choices=["generate", "likelihood"])
    parser.add_argument("--likelihood_batch_size", type=int, default=4)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_network_turbo", action="store_true")
    args = parser.parse_args()

    data_dir_value = args.data_dir or os.environ.get("ALFWORLD_DATA")
    if not data_dir_value:
        raise ValueError("--data_dir or ALFWORLD_DATA is required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir_value)
    build_alfworld_protocol_report(
        official_repo=Path(args.official_repo),
        output_path=Path("outputs/alfworld_eval/protocol_report.json"),
        data_dir=data_dir,
        use_network_turbo=args.use_network_turbo,
    )
    write_json(
        Path("outputs/alfworld_eval/env_report.json"),
        build_alfworld_env_report(Path(args.official_repo), data_dir, use_network_turbo=args.use_network_turbo),
    )
    split_reports = []
    combined_run_lines: list[str] = []
    max_episodes = None if args.max_episodes is not None and args.max_episodes <= 0 else args.max_episodes
    for split in args.splits:
        split_output = output_dir if len(args.splits) == 1 else output_dir / split
        try:
            report = evaluate_alfworld_qwen_direct(
                official_repo=Path(args.official_repo),
                data_dir=data_dir,
                output_dir=split_output,
                model_name_or_path=Path(args.model_name_or_path),
                split=split,
                run_name=args.run_name,
                max_episodes=max_episodes,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                cache_dir=Path(args.cache_dir),
                local_files_only=args.local_files_only,
                torch_dtype=args.torch_dtype,
                max_new_tokens=args.max_new_tokens,
                fallback_strategy=args.fallback_strategy,
                scoring_method=args.scoring_method,
                likelihood_batch_size=args.likelihood_batch_size,
            )
        except BaseException as exc:
            split_output.mkdir(parents=True, exist_ok=True)
            blocker = {
                "status": "blocked",
                "stage": "qwen3_8b_direct_admissible_action_eval",
                "split": split,
                "model_name_or_path": args.model_name_or_path,
                "cache_dir": args.cache_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repro_command": (
                    "python scripts/run_qwen3_alfworld_direct_eval.py "
                    f"--model_name_or_path {args.model_name_or_path} --cache_dir {args.cache_dir} "
                    f"--data_dir {data_dir} --splits {split} --max_episodes {args.max_episodes}"
                ),
                "policy_family": "qwen_direct_admissible",
                "uses_clstr": False,
                "not_clstr_result": True,
                "network_turbo_reference": "https://www.autodl.com/docs/network_turbo/",
            }
            write_json(split_output / "blocker_report.json", blocker)
            metrics = dict(blocker)
            metrics.update(
                {
                    "success_rate": None,
                    "average_reward": None,
                    "average_goal_condition_points": None,
                    "average_episode_steps": None,
                    "episodes": 0,
                    "not_closed_loop_success": True,
                }
            )
            write_json(split_output / "metrics.json", metrics)
            report = {"status": "blocked", "metrics": blocker, "blocker": blocker}
        split_reports.append(report)
        run_path = split_output / "run.jsonl"
        if run_path.exists():
            combined_run_lines.extend(run_path.read_text(encoding="utf-8").splitlines())
    if len(args.splits) > 1:
        total_episodes = sum(int(row["metrics"].get("episode_count", 0)) for row in split_reports)
        blocked_reports = [row for row in split_reports if row.get("status") == "blocked" or row.get("metrics", {}).get("status") == "blocked"]

        def weighted(metric: str) -> float:
            if total_episodes <= 0:
                return 0.0
            return round(
                sum(float(row["metrics"].get(metric, 0.0)) * int(row["metrics"].get("episode_count", 0)) for row in split_reports)
                / total_episodes,
                6,
            )

        aggregate = {
            "status": "blocked" if len(blocked_reports) == len(split_reports) else "partial" if blocked_reports else "ok",
            "method": args.run_name,
            "splits": args.splits,
            "success_rate": weighted("success_rate"),
            "average_reward": weighted("average_reward"),
            "average_goal_condition_points": weighted("average_goal_condition_points"),
            "average_episode_steps": weighted("average_episode_steps"),
            "episode_count": total_episodes,
            "episodes": total_episodes,
            "policy_family": "qwen_direct_admissible",
            "uses_clstr": False,
            "is_clstr_result": False,
            "qwen_direct_baseline": True,
            "not_clstr_result": True,
            "not_closed_loop_success": bool(blocked_reports),
            "model_name_or_path": args.model_name_or_path,
            "split_metrics": [row["metrics"] for row in split_reports],
        }
        if blocked_reports:
            aggregate["blocked_split_count"] = len(blocked_reports)
            aggregate["error"] = blocked_reports[0].get("metrics", {}).get("error")
        write_json(output_dir / "metrics.json", aggregate)
        (output_dir / "run.jsonl").write_text("\n".join(combined_run_lines) + "\n", encoding="utf-8")
        report = {"status": "ok", "metrics": aggregate, "split_reports": split_reports}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
