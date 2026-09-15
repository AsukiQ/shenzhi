#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.qwen_route_eval import run_qwen_embedding_reranker_route_eval


def _domains(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B on CLSTR route rows."
    )
    parser.add_argument("--benchmark", required=True, choices=["toolbench_g3", "tau2", "tau3", "toolsandbox", "trajectbench"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_model_name_or_path", default="models/Qwen3-Embedding-0.6B")
    parser.add_argument("--reranker_model_name_or_path", default="models/Qwen3-Reranker-0.6B")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--embedding_max_length", type=int, default=2048)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--reranker_max_length", type=int, default=2048)
    parser.add_argument("--max_skill_chars", type=int, default=900)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--score_mode", default="logit_diff", choices=["logit_diff", "yes_probability"])
    parser.add_argument("--query_text_mode", default="auto", choices=["auto", "raw_state", "qwen_instruct"])
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--row_progress_interval", type=int, default=25)
    parser.add_argument("--embedding_cache_dir", default="outputs/qwen3_embedding_reranker_route_eval/cache")
    parser.add_argument("--adapter_checkpoint_path")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--toolbench_eval_trajectories_path",
        default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    )
    parser.add_argument(
        "--toolbench_skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
    )
    parser.add_argument("--tau_data_root", default=".tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data")
    parser.add_argument("--tau_domains", default="airline,retail,telecom")
    parser.add_argument("--tau_max_tasks_per_domain", type=int)
    parser.add_argument(
        "--toolsandbox_scenarios_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios",
    )
    parser.add_argument(
        "--toolsandbox_tools_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools",
    )
    parser.add_argument("--toolsandbox_max_scenarios", type=int)
    parser.add_argument(
        "--trajectbench_eval_rows_path",
        default="outputs/trajectbench_skillrouter_finetuned_eval/function_aug_v2_adapter_full_20260701_130731/trajectbench_eval_rows.jsonl",
    )
    parser.add_argument(
        "--trajectbench_skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl",
    )
    args = parser.parse_args()

    report = run_qwen_embedding_reranker_route_eval(
        benchmark=args.benchmark,
        output_dir=args.output_dir,
        embedding_model_name_or_path=args.embedding_model_name_or_path,
        reranker_model_name_or_path=args.reranker_model_name_or_path,
        top_k=args.top_k,
        max_eval_rows=args.max_eval_rows,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        max_skill_chars=args.max_skill_chars,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        score_mode=args.score_mode,
        query_text_mode=args.query_text_mode,
        score_batch_size=args.score_batch_size,
        row_progress_interval=args.row_progress_interval,
        save_predictions=args.save_predictions,
        embedding_cache_dir=args.embedding_cache_dir,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
        toolbench_eval_trajectories_path=args.toolbench_eval_trajectories_path,
        toolbench_skills_path=args.toolbench_skills_path,
        tau_data_root=args.tau_data_root,
        tau_domains=_domains(args.tau_domains),
        tau_max_tasks_per_domain=args.tau_max_tasks_per_domain,
        toolsandbox_scenarios_root=args.toolsandbox_scenarios_root,
        toolsandbox_tools_root=args.toolsandbox_tools_root,
        toolsandbox_max_scenarios=args.toolsandbox_max_scenarios,
        trajectbench_eval_rows_path=args.trajectbench_eval_rows_path,
        trajectbench_skills_path=args.trajectbench_skills_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
