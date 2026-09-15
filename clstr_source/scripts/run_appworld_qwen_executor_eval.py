#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_executor import (
    NullSkillProvider,
    PredictionFileSkillProvider,
    QwenCodeGenerator,
    QwenCodeGeneratorConfig,
    VALID_SKILL_CONTEXT_MODES,
    run_appworld_executor_eval,
)


DEFAULT_PREDICTIONS = {
    "skillrouter_embedding": "outputs/appworld_skillrouter_embedding_baseline/dev_a800_v2/predictions.jsonl",
    "skillrouter_base": "outputs/appworld_skillrouter_base_eval/dev_v1/predictions.jsonl",
    "clstr_base": "outputs/appworld_clstr_eval/routing_only_fit_v1_dev/predictions.jsonl",
    "clstr_skillrouter_init": "outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/predictions.jsonl",
    "clstr_act": "outputs/appworld_clstr_eval/act_v2_dev/predictions.jsonl",
}


def _build_skill_provider(
    method: str,
    skill_pool_path: str,
    predictions_path: str | None,
    *,
    dedupe_canonical_skills: bool = False,
    max_auth_like_skills: int | None = None,
):
    if method == "qwen_only":
        return NullSkillProvider()
    resolved = predictions_path or DEFAULT_PREDICTIONS.get(method)
    if not resolved:
        raise ValueError(f"--predictions_path is required for method {method!r}")
    if not Path(resolved).exists():
        raise FileNotFoundError(f"predictions file not found for method {method!r}: {resolved}")
    return PredictionFileSkillProvider(
        skill_pool_path=skill_pool_path,
        predictions_path=resolved,
        dedupe_canonical_skills=dedupe_canonical_skills,
        max_auth_like_skills=max_auth_like_skills,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-8B AppWorld executor benchmark with optional skill routing.")
    parser.add_argument(
        "--method",
        choices=[
            "qwen_only",
            "skillrouter_embedding",
            "skillrouter_base",
            "clstr_base",
            "clstr_skillrouter_init",
            "clstr_act",
        ],
        default="qwen_only",
    )
    parser.add_argument("--tasks_path", default="data/appworld_routing/dev_tasks.jsonl")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--appworld_cache", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-8B")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dedupe_canonical_skills", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_auth_like_skills", type=int, default=None)
    parser.add_argument("--skill_context_mode", choices=sorted(VALID_SKILL_CONTEXT_MODES), default="raw")
    parser.add_argument("--max_tasks", type=int, default=5)
    parser.add_argument("--max_apis_per_app", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout_seconds", type=int, default=60)
    parser.add_argument("--max_interactions", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir or f"outputs/appworld_executor_smoke/{args.method}"
    provider = _build_skill_provider(
        args.method,
        args.skill_pool_path,
        args.predictions_path,
        dedupe_canonical_skills=bool(args.dedupe_canonical_skills),
        max_auth_like_skills=args.max_auth_like_skills,
    )
    generator = QwenCodeGenerator(
        QwenCodeGeneratorConfig(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            local_files_only=bool(args.local_files_only),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            enable_thinking=bool(args.enable_thinking),
        )
    )
    report = run_appworld_executor_eval(
        tasks_path=args.tasks_path,
        skill_pool_path=args.skill_pool_path,
        output_dir=output_dir,
        appworld_root=args.appworld_root,
        appworld_cache=args.appworld_cache,
        method=args.method,
        skill_provider=provider,
        generator=generator,
        max_tasks=args.max_tasks,
        top_k=args.top_k,
        max_apis_per_app=args.max_apis_per_app,
        skill_context_mode=args.skill_context_mode,
        timeout_seconds=args.timeout_seconds,
        max_interactions=args.max_interactions,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
