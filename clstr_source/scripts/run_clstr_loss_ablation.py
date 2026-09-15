#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.external_data import write_json
from clstr.loss_ablation import ABLATION_CONFIGS, build_ablation_action_trace_diagnostic, build_ablation_summary
from clstr.full_base_train import run_legacy_clstr_full_base_train_from_routing_init


def _release_training_memory() -> None:
    """Release parent-process training allocations before launching eval."""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_eval(
    checkpoint: str,
    eval_output_dir: Path,
    official_repo: str,
    data_dir: str,
    routing_init_manifest: str,
    aux_data_root: str,
    max_episodes: int,
    max_steps: int,
    planner_weight: float,
    q_success_weight: float,
) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_alfworld_structured_controller_eval.py",
        "--official_repo",
        official_repo,
        "--data_dir",
        data_dir,
        "--routing_init_manifest",
        routing_init_manifest,
        "--checkpoint_path",
        checkpoint,
        "--output_dir",
        str(eval_output_dir),
        "--run_name",
        eval_output_dir.name,
        "--splits",
        "valid_seen",
        "--max_episodes",
        str(max_episodes),
        "--max_steps",
        str(max_steps),
        "--batch_size",
        "1",
        "--aux_data_root",
        aux_data_root,
        "--planner_weight",
        str(planner_weight),
        "--q_success_weight",
        str(q_success_weight),
        "--local_files_only",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return _load_json(eval_output_dir / "metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLSTR DAgger/Q-success loss ablations.")
    parser.add_argument("--train_path", default="data/clstr_dagger_expert_corrected_train/train.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_dagger_expert_corrected_train/skills.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_loss_ablation")
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--official_repo", default="/root/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/alfworld_data")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--eval_max_episodes", type=int, default=20)
    parser.add_argument("--eval_max_steps", type=int, default=50)
    parser.add_argument("--planner_weight", type=float, default=3.0)
    parser.add_argument("--q_success_weight", type=float, default=1.0)
    parser.add_argument("--skill_text_format", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports = []
    configs = ABLATION_CONFIGS[: args.limit] if args.limit and args.limit > 0 else ABLATION_CONFIGS
    for cfg in configs:
        name = str(cfg["name"])
        run_output = output_dir / name
        eval_dir = output_dir / f"{name}_valid_seen_gate"
        train_report_path = run_output / "train_report.json"
        offline_report_path = run_output / "offline_report.json"
        eval_metrics_path = eval_dir / "metrics.json"
        if train_report_path.exists() and offline_report_path.exists() and eval_metrics_path.exists():
            train_report = _load_json(train_report_path)
            offline_report = _load_json(offline_report_path)
            checkpoint = str(train_report.get("checkpoint") or "")
            eval_metrics = _load_json(eval_metrics_path)
        else:
            if train_report_path.exists():
                train_report = _load_json(train_report_path)
            else:
                train_report = run_legacy_clstr_full_base_train_from_routing_init(
                    train_path=Path(args.train_path),
                    skills_path=Path(args.skills_path),
                    output_dir=run_output,
                    routing_init_manifest=Path(args.routing_init_manifest),
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    loss_weights=cfg["loss_weights"],
                    skill_text_format=args.skill_text_format,
                )
            checkpoint = str(train_report["checkpoint"])
            if offline_report_path.exists():
                offline_report = _load_json(offline_report_path)
            else:
                offline_report = {
                    "status": "ok",
                    "metrics": train_report.get("metrics") or {},
                    "loss_activation_counts": train_report.get("loss_activation_counts") or {},
                }
                write_json(offline_report_path, offline_report)
            _release_training_memory()
            if eval_metrics_path.exists():
                eval_metrics = _load_json(eval_metrics_path)
            else:
                eval_metrics = _run_eval(
                    checkpoint=checkpoint,
                    eval_output_dir=eval_dir,
                    official_repo=args.official_repo,
                    data_dir=args.data_dir,
                    routing_init_manifest=args.routing_init_manifest,
                    aux_data_root=str(Path(args.train_path).parent),
                    max_episodes=args.eval_max_episodes,
                    max_steps=args.eval_max_steps,
                    planner_weight=args.planner_weight,
                    q_success_weight=args.q_success_weight,
                )
        action_trace_diagnostic = build_ablation_action_trace_diagnostic(
            eval_output_dir=eval_dir,
            train_report=train_report,
            offline_report=offline_report,
            output_path=run_output / "action_trace_diagnostic.json",
        )
        reports.append(
            {
                "name": name,
                "output_dir": str(run_output),
                "checkpoint": checkpoint,
                "train_report": train_report,
                "offline_report": offline_report,
                "action_trace_diagnostic": action_trace_diagnostic,
                "eval_metrics": eval_metrics,
                "eval_output_dir": str(eval_dir),
            }
        )
    summary = build_ablation_summary(reports, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
