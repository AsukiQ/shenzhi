#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.apibank_route_eval import load_apibank_route_corpus
from clstr.bfcl_route_eval import load_bfcl_route_corpus
from clstr.global_pool_route_eval import GlobalPoolCorpus, load_prebuilt_global_pool_corpus
from clstr.global_pool_skillrouter_eval import load_toolbench_g3_global_pool_corpus, run_global_pool_skillrouter_eval
from clstr.tau2_route_eval import load_tau2_route_corpus
from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus


def _csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_corpus(args: argparse.Namespace) -> GlobalPoolCorpus:
    benchmark = str(args.benchmark).strip().lower()
    prebuilt_source_rows_path = str(getattr(args, "prebuilt_source_rows_path", "") or "").strip()
    prebuilt_skills_path = str(getattr(args, "prebuilt_skills_path", "") or "").strip()
    if prebuilt_source_rows_path or prebuilt_skills_path:
        if not prebuilt_source_rows_path or not prebuilt_skills_path:
            raise ValueError("--prebuilt_source_rows_path and --prebuilt_skills_path must be provided together")
        return load_prebuilt_global_pool_corpus(
            benchmark=benchmark,
            source_rows_path=prebuilt_source_rows_path,
            skills_path=prebuilt_skills_path,
        )
    if benchmark == "tau2":
        corpus = load_tau2_route_corpus(
            args.tau2_data_root,
            domains=_csv(args.domains),
            max_tasks_per_domain=args.max_tasks_per_domain,
            benchmark_name="tau2",
        )
        return GlobalPoolCorpus("tau2", corpus.skills, corpus.source_rows, corpus.report)
    if benchmark == "toolsandbox":
        corpus = load_toolsandbox_route_corpus(
            scenarios_root=args.toolsandbox_scenarios_root,
            tools_root=args.toolsandbox_tools_root,
            max_scenarios=args.max_scenarios,
        )
        return GlobalPoolCorpus("toolsandbox", corpus.skills, corpus.source_rows, corpus.report)
    if benchmark == "bfcl":
        corpus = load_bfcl_route_corpus(
            args.bfcl_data_root,
            categories=_csv(args.categories),
            max_rows_per_category=args.max_rows_per_category,
            include_trivial=args.include_trivial,
        )
        return GlobalPoolCorpus("bfcl", corpus.skills, corpus.source_rows, corpus.report)
    if benchmark == "apibank":
        corpus = load_apibank_route_corpus(
            args.apibank_data_root,
            files=_csv(args.files),
            include_trivial=args.include_trivial,
            max_rows_per_file=args.max_rows_per_file,
        )
        return GlobalPoolCorpus("apibank", corpus.skills, corpus.source_rows, corpus.report)
    if benchmark == "toolbench_g3":
        return load_toolbench_g3_global_pool_corpus(
            eval_trajectories_path=args.toolbench_eval_trajectories_path,
            skills_path=args.toolbench_skills_path,
            max_eval_rows=args.max_eval_rows,
        )
    raise ValueError(f"unsupported benchmark: {args.benchmark}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SkillRouter-style bi-encoder on merged global skill pool.")
    parser.add_argument("--benchmark", required=True, choices=["tau2", "toolsandbox", "bfcl", "apibank", "toolbench_g3"])
    parser.add_argument(
        "--base_skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--adapter_checkpoint_path")
    parser.add_argument("--prebuilt_source_rows_path")
    parser.add_argument("--prebuilt_skills_path")

    parser.add_argument("--tau2_data_root", default=".tmp/benchmark_probe_direct/HuggingFaceH4__tau2-bench-data")
    parser.add_argument("--domains", default="airline,retail,telecom")
    parser.add_argument("--max_tasks_per_domain", type=int)
    parser.add_argument(
        "--toolsandbox_scenarios_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios",
    )
    parser.add_argument(
        "--toolsandbox_tools_root",
        default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools",
    )
    parser.add_argument("--max_scenarios", type=int)
    parser.add_argument("--bfcl_data_root", default=".tmp/benchmark_probe_direct/gorilla-llm__Berkeley-Function-Calling-Leaderboard")
    parser.add_argument("--categories")
    parser.add_argument("--max_rows_per_category", type=int)
    parser.add_argument("--apibank_data_root", default=".tmp/benchmark_probe_direct/liminghao1630__API-Bank")
    parser.add_argument("--files")
    parser.add_argument("--max_rows_per_file", type=int)
    parser.add_argument("--include_trivial", action="store_true")
    parser.add_argument(
        "--toolbench_eval_trajectories_path",
        default="outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data/eval_trajectories.jsonl",
    )
    parser.add_argument("--toolbench_skills_path", default="data/toolbench_g3/skills.jsonl")

    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    report = run_global_pool_skillrouter_eval(
        corpus=_build_corpus(args),
        base_skills_path=args.base_skills_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
        max_eval_rows=args.max_eval_rows,
        encode_batch_size=args.encode_batch_size,
        score_batch_size=args.score_batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
