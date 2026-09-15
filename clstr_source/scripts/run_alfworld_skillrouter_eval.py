#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.alfworld_eval import evaluate_alfworld_skillrouter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen SkillRouter admissible-action ranking on official ALFWorld closed-loop eval."
    )
    parser.add_argument("--official_repo", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo")
    parser.add_argument("--data_dir", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_data")
    parser.add_argument("--output_dir", default="outputs/alfworld_eval/skillrouter_frozen_admissible")
    parser.add_argument("--model_name_or_path", default=".cache/hf_models/SkillRouter-Embedding-0.6B")
    parser.add_argument("--split", default="valid_seen", choices=["valid_seen", "valid_unseen"])
    parser.add_argument("--run_name", default="skillrouter_frozen_admissible")
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--adapter_checkpoint_path")
    args = parser.parse_args()
    report = evaluate_alfworld_skillrouter(
        official_repo=args.official_repo,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        split=args.split,
        run_name=args.run_name,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
        adapter_checkpoint_path=args.adapter_checkpoint_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
