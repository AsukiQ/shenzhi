#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.global_pool_skillrouter_eval import load_toolbench_g3_global_pool_corpus
from clstr.matched_multibench_data import canonical_digest
from clstr.tau2_route_eval import load_tau2_route_corpus
from clstr.traject_eval_import import load_trajectbench_route_corpus
from clstr.toolsandbox_route_eval import load_toolsandbox_route_corpus
from clstr.vnext_eval import run_vnext_benchmark_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint-native CLSTR vNext Stage2 model",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=(
            "toolbench",
            "toolbench_g3",
            "toolsandbox",
            "tau2",
            "trajectbench",
            "traject_bench",
        ),
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--training_skills_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_eval_rows", type=int)
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--compressed_m", type=int, default=64)
    parser.add_argument("--dynamic_extra_k", type=int)
    parser.add_argument("--final_k", type=int, default=100)
    parser.add_argument(
        "--candidate_query_mode",
        default="learned_causal",
        choices=("learned_causal", "raw_current", "raw_causal"),
    )
    parser.add_argument(
        "--candidate_compression_mode",
        default="learned",
        choices=("learned", "direct"),
    )
    parser.add_argument("--score_batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=128)
    parser.add_argument("--belief_top_k", type=int, default=64)
    parser.add_argument("--frozen_cache_dir")
    parser.add_argument("--require_verified_toolbench_results", action="store_true")
    parser.add_argument(
        "--toolbench_pool_scope",
        default="global",
        choices=("global", "native"),
    )
    parser.add_argument("--toolbench_native_skills_path")
    parser.add_argument("--allow_stage0_static_diagnostic", action="store_true")
    parser.add_argument("--allow_static_reranker_diagnostic", action="store_true")
    parser.add_argument("--closed_set_foundation_checkpoint_path")
    parser.add_argument("--stage2_release_selection_path")
    parser.add_argument("--unified_router_path")
    parser.add_argument("--support_aware_anchor", action="store_true")
    parser.add_argument(
        "--disable_verified_result_correction",
        action="store_true",
    )

    parser.add_argument("--toolbench_eval_trajectories_path")
    parser.add_argument("--toolbench_skills_path")
    parser.add_argument("--toolsandbox_scenarios_root")
    parser.add_argument("--toolsandbox_tools_root")
    parser.add_argument("--toolsandbox_max_scenarios", type=int)
    parser.add_argument("--tau2_data_root")
    parser.add_argument("--tau2_task_split", default="base", choices=("base", "train", "test", "full"))
    parser.add_argument("--tau2_domains")
    parser.add_argument("--tau2_max_tasks_per_domain", type=int)
    parser.add_argument("--trajectbench_public_data")
    parser.add_argument(
        "--trajectbench_split_partition",
        choices=("dev", "test"),
        default="test",
    )
    parser.add_argument("--prebuilt_source_rows_path")
    parser.add_argument("--prebuilt_skills_path")
    parser.add_argument("--matched_union_manifest_path")
    parser.add_argument("--matched_split", choices=("dev", "test"), default="test")
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus(args: argparse.Namespace):
    benchmark = str(args.benchmark).lower()
    prebuilt_values = (
        args.prebuilt_source_rows_path,
        args.prebuilt_skills_path,
        args.matched_union_manifest_path,
    )
    if any(prebuilt_values):
        if not all(prebuilt_values):
            raise ValueError("matched prebuilt eval requires rows, skills, and union manifest")
        manifest = json.loads(Path(args.matched_union_manifest_path).read_text(encoding="utf-8"))
        if manifest.get("status") != "ok" or manifest.get("schema_version") != "clstr_matched_multibench_union_v1":
            raise ValueError("matched union manifest is invalid")
        unsigned = dict(manifest)
        observed_manifest_sha256 = str(unsigned.pop("manifest_sha256", ""))
        if observed_manifest_sha256 != canonical_digest(unsigned):
            raise ValueError("matched union manifest digest mismatch")
        rows_path = Path(args.prebuilt_source_rows_path).resolve()
        skills_path = Path(args.prebuilt_skills_path).resolve()
        route_files = manifest.get("route_files") or {}
        normalized_benchmark = "toolbench_g3" if benchmark == "toolbench" else benchmark
        if normalized_benchmark not in {"tau2", "toolsandbox"}:
            raise ValueError("matched prebuilt eval currently supports Tau2/ToolSandbox")
        expected_entries = {
            "rows": route_files.get(f"{normalized_benchmark}_{args.matched_split}") or {},
            "skills": route_files.get(f"{normalized_benchmark}_skills") or {},
        }
        expected_paths = {
            "rows": rows_path,
            "skills": skills_path,
        }
        for label, value in expected_entries.items():
            path = Path(str(value.get("path") or "")).resolve()
            if path != expected_paths[label]:
                raise ValueError("prebuilt rows/skills are not bound by the requested benchmark split")
            if str(value.get("sha256") or "") != _file_sha256(path):
                raise ValueError("matched prebuilt route artifact digest mismatch")
        rows = _read_jsonl(rows_path)
        if any(str(row.get("matched_data_split") or "") != args.matched_split for row in rows):
            raise ValueError("prebuilt route rows do not match the requested held-out split")
        return normalized_benchmark, SimpleNamespace(
            skills=_read_jsonl(skills_path),
            source_rows=rows,
            report={
                "benchmark": normalized_benchmark,
                "task_split": args.matched_split,
                "matched_prebuilt_manifest_path": str(Path(args.matched_union_manifest_path).resolve()),
                "matched_prebuilt_source_rows_path": str(rows_path),
                "matched_prebuilt_skills_path": str(skills_path),
                "matched_union_manifest_sha256": observed_manifest_sha256,
            },
        )
    if benchmark in {"toolbench", "toolbench_g3"}:
        if not args.toolbench_eval_trajectories_path or not args.toolbench_skills_path:
            raise ValueError("ToolBench vNext evaluation requires trajectories and skills paths")
        return "toolbench_g3", load_toolbench_g3_global_pool_corpus(
            eval_trajectories_path=args.toolbench_eval_trajectories_path,
            skills_path=args.toolbench_skills_path,
            max_eval_rows=None,
        )
    if benchmark == "toolsandbox":
        if not args.toolsandbox_scenarios_root:
            raise ValueError("ToolSandbox vNext evaluation requires scenarios_root")
        return "toolsandbox", load_toolsandbox_route_corpus(
            scenarios_root=args.toolsandbox_scenarios_root,
            tools_root=args.toolsandbox_tools_root,
            max_scenarios=args.toolsandbox_max_scenarios,
        )
    if benchmark in {"trajectbench", "traject_bench"}:
        if not args.trajectbench_public_data:
            raise ValueError(
                "TrajectBench vNext evaluation requires --trajectbench_public_data"
            )
        return "trajectbench", load_trajectbench_route_corpus(
            args.trajectbench_public_data,
            split_partition=args.trajectbench_split_partition,
        )
    if not args.tau2_data_root:
        raise ValueError("Tau2 vNext evaluation requires data_root")
    domains = (
        [value.strip() for value in str(args.tau2_domains).split(",") if value.strip()]
        if args.tau2_domains
        else None
    )
    return "tau2", load_tau2_route_corpus(
        args.tau2_data_root,
        domains=domains,
        task_split=args.tau2_task_split,
        max_tasks_per_domain=args.tau2_max_tasks_per_domain,
        benchmark_name="tau2",
    )


def main() -> None:
    args = parse_args()
    benchmark, corpus = _load_corpus(args)
    report = run_vnext_benchmark_eval(
        benchmark=benchmark,
        corpus=corpus,
        checkpoint_path=args.checkpoint_path,
        training_skills_path=args.training_skills_path,
        output_dir=args.output_dir,
        max_eval_rows=args.max_eval_rows,
        coarse_k=args.coarse_k,
        compressed_m=args.compressed_m,
        dynamic_extra_k=args.dynamic_extra_k,
        final_k=args.final_k,
        candidate_query_mode=args.candidate_query_mode,
        candidate_compression_mode=args.candidate_compression_mode,
        score_batch_size=args.score_batch_size,
        cache_batch_size=args.cache_batch_size,
        belief_top_k=args.belief_top_k,
        frozen_cache_dir=args.frozen_cache_dir,
        require_verified_toolbench_results=args.require_verified_toolbench_results,
        toolbench_pool_scope=args.toolbench_pool_scope,
        toolbench_native_skills_path=args.toolbench_native_skills_path,
        allow_stage0_static_diagnostic=args.allow_stage0_static_diagnostic,
        allow_static_reranker_diagnostic=args.allow_static_reranker_diagnostic,
        closed_set_foundation_checkpoint_path=(
            args.closed_set_foundation_checkpoint_path
        ),
        stage2_release_selection_path=args.stage2_release_selection_path,
        unified_router_path=args.unified_router_path,
        support_aware_anchor=args.support_aware_anchor,
        disable_verified_result_correction=(
            args.disable_verified_result_correction
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
