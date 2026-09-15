#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.full_base_train import _attach_stage0_topm_candidates, _read_jsonl, _skill_id
from clstr.stage4_act_train import _as_stage4_handoff_row
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint
from clstr.toolbench_qwen_rerank import (
    aggregate_metric_rows,
    ranking_metrics_from_ranked_skill_ids,
    strict_metrics,
)
from clstr.toolbench_qwen_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    Qwen3RerankerConfig,
    Qwen3RerankerScorer,
    build_qwen3_reranker_document,
    rank_candidate_skill_ids_by_scores,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_indices(row: dict[str, Any]) -> list[int]:
    values = row.get("stage0_next_candidate_skill_indices") or row.get("candidate_next_skill_indices") or []
    output: list[int] = []
    for item in values:
        try:
            output.append(int(item))
        except Exception:
            continue
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-Reranker skill reranking on ToolBench-G3 clean split.")
    parser.add_argument("--stage0_checkpoint_path", default="outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt")
    parser.add_argument("--eval_trajectories_path", default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-Reranker-8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_skill_chars", type=int, default=900)
    parser.add_argument("--instruction", default=DEFAULT_RERANK_INSTRUCTION)
    parser.add_argument("--score_mode", default="logit_diff", choices=["logit_diff", "yes_probability"])
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state", choices=["raw_state", "skillrouter_state"])
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=16)
    parser.add_argument("--stage0_candidate_progress_interval_batches", type=int, default=50)
    parser.add_argument("--row_progress_interval", type=int, default=25)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    skills = _read_jsonl(args.skills_path)
    skill_id_to_idx = {str(_skill_id(skill, idx)): idx for idx, skill in enumerate(skills)}
    skill_ids_by_idx = {idx: skill_id for skill_id, idx in skill_id_to_idx.items()}

    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(args.stage0_checkpoint_path),
        skills_path=Path(args.skills_path),
        model_cache_dir=output_dir / "stage0_model_cache",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    source_rows = _read_jsonl(args.eval_trajectories_path)
    if args.max_eval_rows is not None:
        source_rows = source_rows[: max(0, int(args.max_eval_rows))]
    handoff_rows = [_as_stage4_handoff_row(row) for row in source_rows]
    retained_rows, handoff_report = _attach_stage0_topm_candidates(
        model,
        handoff_rows,
        skills,
        skill_id_to_idx,
        top_m=int(args.top_k),
        positive_missing_policy="skip",
        query_mode=args.stage0_handoff_query_mode,
        allow_full_pool_stage2_debug=False,
        routing_checkpoint_path=args.stage0_checkpoint_path,
        manifest_path=output_dir / "stage0_candidate_handoff.json",
        encode_batch_size=args.stage0_candidate_encode_batch_size,
        device=device,
        progress_interval_batches=args.stage0_candidate_progress_interval_batches,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_rows: list[dict[str, Any]] = []
    for row in retained_rows:
        next_skill_id = str(row.get("next_skill_id") or "")
        candidate_idx = _candidate_indices(row)[: int(args.top_k)]
        candidate_skill_ids = [skill_ids_by_idx[idx] for idx in candidate_idx if idx in skill_ids_by_idx]
        if next_skill_id not in set(candidate_skill_ids):
            continue
        eval_rows.append(
            {
                "row": row,
                "next_skill_id": next_skill_id,
                "candidate_skill_ids": candidate_skill_ids,
            }
        )

    scorer = Qwen3RerankerScorer(
        Qwen3RerankerConfig(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            local_files_only=bool(args.local_files_only),
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            max_skill_chars=int(args.max_skill_chars),
            instruction=str(args.instruction),
            score_mode=str(args.score_mode),
        )
    )

    metric_rows: list[dict[str, float]] = []
    predictions: list[dict[str, Any]] = []
    scoring_started = time.perf_counter()
    for row_idx, item in enumerate(eval_rows, start=1):
        row = item["row"]
        candidate_skill_ids = list(item["candidate_skill_ids"])
        query = str(row.get("state_text") or "")
        documents = [
            build_qwen3_reranker_document(
                {**skills[skill_id_to_idx[skill_id]], "skill_id": skill_id},
                max_skill_chars=int(args.max_skill_chars),
            )
            for skill_id in candidate_skill_ids
            if skill_id in skill_id_to_idx
        ]
        candidate_skill_ids = [skill_id for skill_id in candidate_skill_ids if skill_id in skill_id_to_idx]
        scores = scorer.score_pairs(queries=[query] * len(documents), documents=documents)
        ranked_skill_ids = rank_candidate_skill_ids_by_scores(candidate_skill_ids=candidate_skill_ids, scores=scores)
        metrics = ranking_metrics_from_ranked_skill_ids(
            ranked_skill_ids=ranked_skill_ids,
            candidate_skill_ids=candidate_skill_ids,
            positive_skill_id=str(item["next_skill_id"]),
            source_row_count=1,
        )
        metric_rows.append(metrics)
        predictions.append(
            {
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index"),
                "next_skill_id": item["next_skill_id"],
                "candidate_skill_ids": candidate_skill_ids,
                "scores": scores,
                "ranked_skill_ids": ranked_skill_ids,
                "metrics": metrics,
            }
        )
        if args.row_progress_interval > 0 and (row_idx == 1 or row_idx % int(args.row_progress_interval) == 0 or row_idx == len(eval_rows)):
            elapsed = time.perf_counter() - scoring_started
            print(
                f"[qwen3-reranker] scored_rows={row_idx}/{len(eval_rows)} "
                f"pairs={row_idx * int(args.top_k)} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    retained_metrics = aggregate_metric_rows(metric_rows)
    total_seconds = time.perf_counter() - started
    score_seconds = time.perf_counter() - scoring_started
    report = {
        "status": "ok",
        "method": "qwen3_reranker_skill_rerank",
        "model_name_or_path": args.model_name_or_path,
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(metric_rows),
        "top_k": int(args.top_k),
        "stage0_candidate_handoff": handoff_report,
        "retained_metrics": retained_metrics,
        "strict": strict_metrics(retained_metrics, retained_rows=len(metric_rows), source_rows=len(source_rows)),
        "score": {
            "mean_candidate_count": retained_metrics.get("candidate_count", 0.0),
            "score_mode": args.score_mode,
        },
        "timing": {
            "total_seconds": total_seconds,
            "scoring_seconds": score_seconds,
            "seconds_per_retained_row": score_seconds / len(metric_rows) if metric_rows else 0.0,
        },
        "config": vars(args),
        "model_load": {"routing_init": routing_report, "model_config": model_config},
    }
    _write_json(output_dir / "qwen_reranker_eval_report.json", report)
    if args.save_predictions:
        _write_jsonl(output_dir / "qwen_reranker_predictions.jsonl", predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
