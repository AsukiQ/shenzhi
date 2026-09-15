#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skill_embedding import build_enriched_skill_embedding_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLSTR enriched SkillNet skill embedding data.")
    parser.add_argument("--skills_path", default="data/clstr_dagger_expert_corrected_train/skills.jsonl")
    parser.add_argument("--train_path", default="data/clstr_dagger_expert_corrected_train/train.jsonl")
    parser.add_argument("--output_dir", default="data/clstr_dagger_expert_corrected_train_enriched")
    parser.add_argument("--report_output_dir", default="outputs/clstr_dagger_expert_corrected_train_enriched")
    parser.add_argument("--skillnet_root", default="/tmp/SkillNet/experiments/src/skills")
    args = parser.parse_args()

    report = build_enriched_skill_embedding_data(
        skills_path=Path(args.skills_path),
        train_path=Path(args.train_path),
        output_dir=Path(args.output_dir),
        report_output_dir=Path(args.report_output_dir),
        skillnet_root=Path(args.skillnet_root) if args.skillnet_root else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
