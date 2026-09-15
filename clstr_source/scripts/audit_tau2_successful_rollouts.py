#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.tau2_successful_rollouts import (  # noqa: E402
    TAU2_DEFAULT_AGENT_MODELS,
    load_tau2_successful_rollout_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit agent-visible successful Tau2 rollout supervision."
    )
    parser.add_argument("--tau2_data_root", required=True)
    parser.add_argument("--tau2_results_root")
    parser.add_argument("--task_split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--agent_models",
        default=",".join(TAU2_DEFAULT_AGENT_MODELS),
    )
    parser.add_argument("--output_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(item.strip() for item in args.agent_models.split(",") if item.strip())
    corpus = load_tau2_successful_rollout_corpus(
        args.tau2_data_root,
        results_root=args.tau2_results_root,
        task_split=args.task_split,
        agent_models=models,
    )
    payload = json.dumps(corpus.report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
