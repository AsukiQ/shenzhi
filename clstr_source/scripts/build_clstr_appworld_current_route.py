#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_current_route import DEFAULT_OUTPUT_DIR, DEFAULT_REPO_ROOT, build_appworld_current_route_dataset


def _parse_extra_skill_pool_paths(values: list[str] | None, packed: str | None) -> list[str]:
    paths: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            paths.append(text)
    if packed:
        for item in str(packed).replace(":", ",").split(","):
            text = item.strip()
            if text:
                paths.append(text)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AppWorld SkillX data root that follows the current CLSTR "
            "Stage0/Stage1/Stage2/Stage4 unified schema. This script prepares "
            "data only; it does not launch training."
        )
    )
    parser.add_argument("--repo_root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--train_tasks_path", default="data/appworld_routing/train_tasks.jsonl")
    parser.add_argument("--train_qrels_path", default="data/appworld_routing/train_qrels.jsonl")
    parser.add_argument("--train_replay_path", default="data/appworld_routing/train_replay.jsonl")
    parser.add_argument("--verified_pairs_path", default="data/appworld_act/verified_pairs_train.jsonl")
    parser.add_argument(
        "--base_skill_pool_path",
        default=None,
        help=(
            "Optional checkpoint-prefix skill pool. When set, the output skill_pool keeps this base "
            "pool first and appends AppWorld SkillX skills after it for dynamic registry eval."
        ),
    )
    parser.add_argument(
        "--extra_skill_pool_path",
        action="append",
        default=[],
        help="Optional extra JSONL skill pool appended after AppWorld SkillX as large-pool distractors.",
    )
    parser.add_argument(
        "--extra_skill_pool_paths",
        default=None,
        help="Optional comma- or colon-separated extra JSONL skill pools appended after AppWorld SkillX.",
    )
    args = parser.parse_args()

    manifest = build_appworld_current_route_dataset(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        skill_pool_path=args.skill_pool_path,
        train_tasks_path=args.train_tasks_path,
        train_qrels_path=args.train_qrels_path,
        train_replay_path=args.train_replay_path,
        verified_pairs_path=args.verified_pairs_path,
        base_skill_pool_path=args.base_skill_pool_path,
        extra_skill_pool_paths=_parse_extra_skill_pool_paths(args.extra_skill_pool_path, args.extra_skill_pool_paths),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
