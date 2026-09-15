#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.dagger_preprocess import build_dagger_train_data
from clstr.external_data import write_json
from clstr.full_base_train import run_legacy_clstr_full_base_train_from_routing_init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLSTR on DAgger expert-corrected rollout with Q_success.")
    parser.add_argument("--rollout_path", default="data/alfworld_qwen3_expert_corrected_rollout/train_rollout.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_full_base_train/skills.jsonl")
    parser.add_argument("--output_dir", default="outputs/clstr_dagger_success_train")
    parser.add_argument("--data_output_dir", default="data/clstr_dagger_expert_corrected_train")
    parser.add_argument("--routing_init_manifest", default="outputs/clstr_native_routing_init/manifest.json")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--hard_negative_margin_weight", type=float, default=0.2)
    parser.add_argument("--q_success_weight", type=float, default=0.5)
    parser.add_argument("--transition_loss_weight", type=float, default=0.0)
    parser.add_argument("--transition_skill_ce_loss_weight", type=float, default=0.2)
    parser.add_argument("--belief_loss_weight", type=float, default=0.05)
    parser.add_argument("--stop_loss_weight", type=float, default=0.2)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.2)
    parser.add_argument("--skill_text_format", default=None)
    args = parser.parse_args()

    data_report = build_dagger_train_data(
        rollout_path=Path(args.rollout_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.data_output_dir),
        report_output_dir=Path(args.output_dir),
    )
    loss_weights = {
        "L_policy": args.policy_loss_weight,
        "hard_negative_margin": args.hard_negative_margin_weight,
        "Q_success": args.q_success_weight,
        "L_trans": args.transition_loss_weight,
        "L_trans_skill_ce": args.transition_skill_ce_loss_weight,
        "belief": args.belief_loss_weight,
        "STOP": args.stop_loss_weight,
        "routing": args.retrieval_loss_weight,
    }
    train_report = run_legacy_clstr_full_base_train_from_routing_init(
        train_path=Path(data_report["train_path"]),
        skills_path=Path(data_report["skills_path"]),
        output_dir=Path(args.output_dir),
        routing_init_manifest=Path(args.routing_init_manifest),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        loss_weights=loss_weights,
        skill_text_format=args.skill_text_format,
    )
    train_report["dagger_data_report"] = data_report
    train_report["q_success_action_value_head"] = {
        "enabled": True,
        "interpretation": "estimated_action_value_not_oracle_probability",
        "positive_source": "official_expert_action",
        "hard_negative_source": "qwen_wrong_action",
    }
    write_json(Path(args.output_dir) / "eval_report.json", {"status": "ok", "offline_metrics": train_report.get("metrics", {})})
    write_json(Path(args.output_dir) / "train_report.json", train_report)
    print(json.dumps(train_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
