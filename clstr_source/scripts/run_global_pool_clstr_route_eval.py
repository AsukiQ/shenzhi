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
from clstr.full_base_train import ROUTE_SCORERS, UNIFIED_MEMORY_ROUTE_SCORER
from clstr.global_pool_route_eval import GlobalPoolCorpus, load_prebuilt_global_pool_corpus, run_global_pool_clstr_route_eval
from clstr.memory_utility_gate import RELIABILITY_MODES
from clstr.tau2_route_eval import load_tau2_route_corpus
from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus


def _csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_corpus(args: argparse.Namespace) -> GlobalPoolCorpus:
    benchmark = str(args.benchmark).strip().lower()
    if args.prebuilt_source_rows_path or args.prebuilt_skills_path:
        if not args.prebuilt_source_rows_path or not args.prebuilt_skills_path:
            raise ValueError("prebuilt global-pool eval requires both --prebuilt_source_rows_path and --prebuilt_skills_path")
        return load_prebuilt_global_pool_corpus(
            benchmark=benchmark,
            source_rows_path=args.prebuilt_source_rows_path,
            skills_path=args.prebuilt_skills_path,
        )
    if benchmark == "tau2":
        corpus = load_tau2_route_corpus(
            args.tau2_data_root,
            domains=_csv(args.domains),
            max_tasks_per_domain=args.max_tasks_per_domain,
            benchmark_name="tau2",
        )
        return GlobalPoolCorpus(benchmark="tau2", skills=corpus.skills, source_rows=corpus.source_rows, report=corpus.report)
    if benchmark == "toolsandbox":
        corpus = load_toolsandbox_route_corpus(
            scenarios_root=args.toolsandbox_scenarios_root,
            tools_root=args.toolsandbox_tools_root,
            max_scenarios=args.max_scenarios,
        )
        return GlobalPoolCorpus(
            benchmark="toolsandbox",
            skills=corpus.skills,
            source_rows=corpus.source_rows,
            report=corpus.report,
        )
    if benchmark == "bfcl":
        corpus = load_bfcl_route_corpus(
            args.bfcl_data_root,
            categories=_csv(args.categories),
            max_rows_per_category=args.max_rows_per_category,
            include_trivial=args.include_trivial,
        )
        return GlobalPoolCorpus(benchmark="bfcl", skills=corpus.skills, source_rows=corpus.source_rows, report=corpus.report)
    if benchmark == "apibank":
        corpus = load_apibank_route_corpus(
            args.apibank_data_root,
            files=_csv(args.files),
            include_trivial=args.include_trivial,
            max_rows_per_file=args.max_rows_per_file,
        )
        return GlobalPoolCorpus(
            benchmark="apibank",
            skills=corpus.skills,
            source_rows=corpus.source_rows,
            report=corpus.report,
        )
    raise ValueError(f"unsupported benchmark: {args.benchmark}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CLSTR on a clean unified/global skill pool.")
    parser.add_argument("--benchmark", required=True, choices=["tau2", "toolsandbox", "bfcl", "apibank"])
    parser.add_argument(
        "--base_skills_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl",
    )
    parser.add_argument(
        "--stage0_checkpoint_path",
        default="outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        default="outputs/clstr_stage12_belief_seed_fullsupport_rankprior_full_a800_20260706_004234/stage2_full_base/checkpoints/clstr_full_base-step10000.pt",
    )
    parser.add_argument(
        "--stage4_checkpoint_path",
        default="outputs/clstr_stage4_belief_seed_fullsupport_rankprior_full_a800_20260706_from_s12/checkpoints/clstr_stage4_act-step2000.pt",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--route_records_path")
    parser.add_argument("--route_record_manifest_path")
    parser.add_argument("--route_record_model_digest")

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
    parser.add_argument("--prebuilt_source_rows_path")
    parser.add_argument("--prebuilt_skills_path")

    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--stage0_top_m", "--static_k", dest="stage0_top_m", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument("--candidate_count", "--final_k", dest="candidate_count", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stage0_candidate_encode_batch_size", type=int, default=8)
    parser.add_argument("--stage0_handoff_query_mode", default="skillrouter_state")
    parser.add_argument(
        "--online_memory_mode",
        default="latest_exact",
        choices=["latest_exact", "exact_count", "state_conditioned", "inventory_remaining"],
    )
    parser.add_argument("--online_memory_weight", type=float, default=1.0)
    parser.add_argument("--online_memory_next_skill_bonus", type=float, default=0.0)
    parser.add_argument("--online_memory_exact_transition_bonus", type=float, default=5.0)
    parser.add_argument("--transition_residual_lambda", type=float, default=0.25)
    parser.add_argument("--route_scorer", default=UNIFIED_MEMORY_ROUTE_SCORER, choices=sorted(ROUTE_SCORERS))
    parser.add_argument(
        "--reliability_mode",
        default="dynamic",
        choices=sorted(RELIABILITY_MODES),
    )
    parser.add_argument("--fixed_alpha", type=float, default=1.0)
    parser.add_argument("--feature_update_count_cap", type=float, default=1.0)
    parser.add_argument("--feature_candidate_count_cap", type=float, default=1.0)
    parser.add_argument("--memory_utility_gate_checkpoint_path")
    parser.add_argument("--expected_memory_utility_gate_checkpoint_sha256")
    parser.add_argument("--expected_memory_utility_gate_audit_sha256")
    args = parser.parse_args()

    corpus = _build_corpus(args)
    report = run_global_pool_clstr_route_eval(
        corpus=corpus,
        base_skills_path=args.base_skills_path,
        stage0_checkpoint_path=args.stage0_checkpoint_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        stage4_checkpoint_path=args.stage4_checkpoint_path,
        output_dir=args.output_dir,
        max_eval_rows=args.max_eval_rows,
        stage0_top_m=args.stage0_top_m,
        dynamic_extra_k=args.dynamic_extra_k,
        candidate_count=args.candidate_count,
        batch_size=args.batch_size,
        stage0_candidate_encode_batch_size=args.stage0_candidate_encode_batch_size,
        stage0_handoff_query_mode=args.stage0_handoff_query_mode,
        online_memory_mode=args.online_memory_mode,
        online_memory_weight=args.online_memory_weight,
        online_memory_next_skill_bonus=args.online_memory_next_skill_bonus,
        online_memory_exact_transition_bonus=args.online_memory_exact_transition_bonus,
        transition_residual_lambda=args.transition_residual_lambda,
        route_scorer=args.route_scorer,
        reliability_mode=args.reliability_mode,
        fixed_alpha=args.fixed_alpha,
        feature_update_count_cap=args.feature_update_count_cap,
        feature_candidate_count_cap=args.feature_candidate_count_cap,
        memory_utility_gate_checkpoint_path=args.memory_utility_gate_checkpoint_path,
        expected_memory_utility_gate_checkpoint_sha256=args.expected_memory_utility_gate_checkpoint_sha256,
        expected_memory_utility_gate_audit_sha256=args.expected_memory_utility_gate_audit_sha256,
        route_records_path=args.route_records_path,
        route_record_manifest_path=args.route_record_manifest_path,
        route_record_model_digest=args.route_record_model_digest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"ok", "ok_with_notes"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
