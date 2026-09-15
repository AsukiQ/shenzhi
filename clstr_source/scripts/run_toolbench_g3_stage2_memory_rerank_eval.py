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

from clstr.full_base_train import _attach_stage0_topm_candidates, _read_jsonl, _skill_id
from clstr.logged_online_stage4_train import attach_trajectory_prefix_online_memory_scores
from clstr.stage2_memory_rerank_eval import (
    aggregate_stage2_memory_metrics,
    evaluate_stage2_memory_rerank_batch,
    strict_metrics_from_retained,
)
from clstr.stage4_act_train import (
    STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    _as_stage4_handoff_row,
    _build_stage4_next_skill_rows_from_source_rows,
)
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _batch(rows: list[dict], size: int):
    for start in range(0, len(rows), max(1, int(size))):
        yield rows[start : start + max(1, int(size))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolBench-G3 Stage2 real-topM scorer plus Stage4 online-memory score on the clean split."
    )
    parser.add_argument("--stage0_checkpoint_path", default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt")
    parser.add_argument("--stage2_checkpoint_path", default="outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt")
    parser.add_argument("--eval_trajectories_path", default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_m", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=50)
    parser.add_argument("--transition_inventory_mask_mode", default="auto")
    parser.add_argument("--transition_inventory_min_candidates", type=int, default=64)
    parser.add_argument("--transition_positive_mode", default="gold_plus_equivalent")
    parser.add_argument("--transition_residual_lambda", type=float, default=0.25)
    parser.add_argument("--transition_scoring_mode", default=STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE)
    parser.add_argument("--online_memory_mode", default="latest_exact")
    parser.add_argument("--online_memory_weight", type=float, default=1.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    skills = _read_jsonl(args.skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    skill_ids_by_idx = {idx: skill_id for skill_id, idx in skill_id_to_idx.items()}
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(args.stage0_checkpoint_path),
        skills_path=Path(args.skills_path),
        model_cache_dir=output / "model_cache",
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(args.stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    eval_source_rows = _read_jsonl(args.eval_trajectories_path)
    if args.max_eval_rows is not None:
        eval_source_rows = eval_source_rows[: max(0, int(args.max_eval_rows))]
    handoff_rows = [_as_stage4_handoff_row(row) for row in eval_source_rows]
    retained_rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        handoff_rows,
        skills,
        skill_id_to_idx,
        top_m=args.top_m,
        positive_missing_policy="skip",
        query_mode=args.stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=args.stage0_checkpoint_path,
        manifest_path=output / "stage0_candidate_handoff.json",
        encode_batch_size=args.stage0_candidate_encode_batch_size,
        device=device,
        progress_interval_batches=args.stage0_candidate_progress_interval_batches,
    )
    stage4_rows, row_report = _build_stage4_next_skill_rows_from_source_rows(
        retained_rows,
        skill_id_to_idx,
        candidate_count=None,
        max_rows=None,
        allowed_benchmarks={"toolbench_g3"},
        stage0_candidate_handoff_report=handoff_report,
    )
    scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        stage4_rows,
        feedback_rows=stage4_rows,
        next_skill_bonus=args.online_memory_next_skill_bonus,
        exact_transition_bonus=args.online_memory_exact_transition_bonus,
        memory_mode=args.online_memory_mode,
    )
    metric_rows: list[dict] = []
    with torch.no_grad():
        for batch_index, rows in enumerate(_batch(scored_rows, args.batch_size), start=1):
            metrics = evaluate_stage2_memory_rerank_batch(
                model,
                rows,
                skill_id_to_idx,
                skill_ids_by_idx,
                skills,
                device,
                transition_inventory_mask_mode=args.transition_inventory_mask_mode,
                transition_inventory_min_candidates=args.transition_inventory_min_candidates,
                transition_positive_mode=args.transition_positive_mode,
                transition_residual_lambda=args.transition_residual_lambda,
                transition_scoring_mode=args.transition_scoring_mode,
                online_memory_weight=args.online_memory_weight,
            )
            metric_rows.append({"batch_index": batch_index, "batch_size": len(rows), "metrics": metrics})
    aggregate = aggregate_stage2_memory_metrics(metric_rows)
    stage2_metrics = {
        key.removeprefix("stage2_"): value
        for key, value in aggregate.items()
        if key.startswith("stage2_transition_skill_")
    }
    memory_metrics = {
        key.removeprefix("stage2_plus_memory_"): value
        for key, value in aggregate.items()
        if key.startswith("stage2_plus_memory_transition_skill_")
    }
    report = {
        "status": "ok",
        "stage4_route": "stage2_real_topm_scorer_plus_online_memory",
        "output_dir": str(output),
        "source_eval_rows": len(eval_source_rows),
        "retained_eval_rows": len(scored_rows),
        "stage0_candidate_handoff": handoff_report,
        "row_report": row_report,
        "online_memory_report": memory_report,
        "aggregate": aggregate,
        "stage2_only_metrics": stage2_metrics,
        "stage2_plus_memory_metrics": memory_metrics,
        "strict": {
            "stage2_only": strict_metrics_from_retained(
                stage2_metrics,
                retained_rows=len(scored_rows),
                source_rows=len(eval_source_rows),
            ),
            "stage2_plus_memory": strict_metrics_from_retained(
                memory_metrics,
                retained_rows=len(scored_rows),
                source_rows=len(eval_source_rows),
            ),
        },
        "config": vars(args),
        "model_load": {"routing_init": routing_report, "stage2_load": stage2_load_report, "model_config": model_config},
    }
    strict_stage2 = report["strict"]["stage2_only"]
    strict_memory = report["strict"]["stage2_plus_memory"]
    report["strict_delta"] = {
        key: float(strict_memory.get(key, 0.0)) - float(strict_stage2.get(key, 0.0))
        for key in strict_memory
        if isinstance(strict_memory.get(key), (int, float)) and isinstance(strict_stage2.get(key), (int, float))
    }
    _write_json(output / "stage2_memory_rerank_eval_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
