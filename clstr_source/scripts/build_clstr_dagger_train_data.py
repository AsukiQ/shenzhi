#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.dagger_preprocess import build_dagger_train_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR DAgger expert-corrected train data.")
    parser.add_argument("--rollout_path", default="data/alfworld_qwen3_expert_corrected_rollout/train_rollout.jsonl")
    parser.add_argument("--skills_path", default="data/clstr_full_base_train/skills.jsonl")
    parser.add_argument("--output_dir", default="data/clstr_dagger_expert_corrected_train")
    parser.add_argument("--report_output_dir", default="outputs/clstr_dagger_expert_corrected_train")
    args = parser.parse_args()
    report = build_dagger_train_data(
        rollout_path=Path(args.rollout_path),
        skills_path=Path(args.skills_path),
        output_dir=Path(args.output_dir),
        report_output_dir=Path(args.report_output_dir),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
