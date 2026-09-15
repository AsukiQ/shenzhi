#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_skillrouter_eval_command(
    skillrouter_repo: Path,
    data_root: Path,
    encoder_model: Path,
    reranker_model: Path,
    output_dir: Path,
    tiers: list[str],
    task_mode: str,
    retrieval_top_k: int,
    prompt_format: str = "flat-full",
    encoder_batch_size: int = 32,
    reranker_batch_size: int = 8,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.run_open_model_eval",
        "--data_root",
        str(data_root),
        "--encoder_model_or_path",
        str(encoder_model),
        "--reranker_model_or_path",
        str(reranker_model),
        "--task_mode",
        task_mode,
        "--tiers",
        *tiers,
        "--retrieval_top_k",
        str(retrieval_top_k),
        "--prompt_format",
        prompt_format,
        "--encoder_batch_size",
        str(encoder_batch_size),
        "--reranker_batch_size",
        str(reranker_batch_size),
        "--output_dir",
        str(output_dir),
    ]


def format_dry_run(skillrouter_repo: Path, command: list[str]) -> str:
    return "PYTHONPATH=" + str(skillrouter_repo) + " " + " ".join(shlex.quote(part) for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkillRouter benchmark baseline as eval-only from CLSTR.")
    parser.add_argument("--skillrouter_repo", default="/root/autodl-tmp/skillrouter")
    parser.add_argument("--data_root", default="/root/autodl-tmp/clstr/data/skillrouter_eval_core")
    parser.add_argument("--encoder_model", default="/root/autodl-tmp/clstr/.cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--reranker_model", default="/root/autodl-tmp/clstr/.cache/hf_models/SkillRouter-Reranker-0.6B")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/clstr/outputs/skillrouter_baseline/open_model_eval")
    parser.add_argument("--tiers", nargs="+", choices=["easy", "hard"], default=["easy", "hard"])
    parser.add_argument("--task_mode", choices=["core", "all", "single"], default="core")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--prompt_format", choices=["flat-full", "flat-nd", "struct"], default="flat-full")
    parser.add_argument("--encoder_batch_size", type=int, default=32)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    skillrouter_repo = Path(args.skillrouter_repo)
    command = build_skillrouter_eval_command(
        skillrouter_repo=skillrouter_repo,
        data_root=Path(args.data_root),
        encoder_model=Path(args.encoder_model),
        reranker_model=Path(args.reranker_model),
        output_dir=Path(args.output_dir),
        tiers=args.tiers,
        task_mode=args.task_mode,
        retrieval_top_k=args.retrieval_top_k,
        prompt_format=args.prompt_format,
        encoder_batch_size=args.encoder_batch_size,
        reranker_batch_size=args.reranker_batch_size,
    )

    if args.dry_run:
        print(format_dry_run(skillrouter_repo, command))
        return

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(skillrouter_repo) if not existing_pythonpath else f"{skillrouter_repo}:{existing_pythonpath}"
    subprocess.run(command, cwd=Path.cwd(), env=env, check=True)


if __name__ == "__main__":
    main()
