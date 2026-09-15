#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.belief_state_audit import audit_belief_tensors, sample_state_rows, write_belief_audit_report
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR inference-time belief state sharpness and variance.")
    parser.add_argument("--stage0_checkpoint_path", required=True)
    parser.add_argument("--skills_path", required=True)
    parser.add_argument("--trajectories_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--head_checkpoint_path")
    parser.add_argument("--sample_rows", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_effective_support_fraction", type=float, default=0.50)
    parser.add_argument("--min_top5_mass", type=float, default=0.05)
    parser.add_argument("--max_pairwise_cosine", type=float, default=0.995)
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(args.stage0_checkpoint_path),
        skills_path=Path(args.skills_path),
        model_cache_dir=output_path.parent / "model_cache",
    )
    head_load_report = None
    if args.head_checkpoint_path:
        head_load_report = load_head_checkpoint_into_model(
            model,
            head_checkpoint_path=Path(args.head_checkpoint_path),
            partial_load_mode="belief_audit_optional_head_checkpoint",
        )
    device = torch.device(str(args.device))
    model.to(device)
    model.eval()
    rows = sample_state_rows(args.trajectories_path, max_rows=max(1, int(args.sample_rows)))
    if not rows:
        raise RuntimeError(f"no state_text rows found in {args.trajectories_path}")
    texts = [str(row.get("state_text") or row.get("task_text") or row.get("goal_text") or "") for row in rows]
    labels = [str(row.get("benchmark") or "<missing>") for row in rows]
    with torch.no_grad():
        h = model.encode_states(texts).to(device)
        report = audit_belief_tensors(
            skill_table=model.skill_table,
            state_embeddings=h,
            benchmark_labels=labels,
            max_effective_support_fraction=args.max_effective_support_fraction,
            min_top5_mass=args.min_top5_mass,
            max_pairwise_cosine=args.max_pairwise_cosine,
        )
    report["stage0_checkpoint_path"] = str(args.stage0_checkpoint_path)
    report["head_checkpoint_path"] = None if args.head_checkpoint_path is None else str(args.head_checkpoint_path)
    report["skills_path"] = str(args.skills_path)
    report["trajectories_path"] = str(args.trajectories_path)
    report["model_config"] = model_config
    report["routing_init"] = routing_report
    report["head_load"] = head_load_report or {}
    report["sampled_task_ids"] = [str(row.get("task_id") or "") for row in rows[:20]]
    write_belief_audit_report(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
