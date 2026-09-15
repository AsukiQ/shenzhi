#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.appworld_dynamic_routing_audit import audit_appworld_dynamic_routing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AppWorld positive-skill ranks after loading a CLSTR checkpoint "
            "with its base skill pool and dynamically appending AppWorld SkillX skills."
        )
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--base_skill_pool_path", default="data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl")
    parser.add_argument("--dynamic_skill_pool_path", default="data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl")
    parser.add_argument("--retrieval_path", default="data/clstr_appworld_dynamic_v4_1b_append/retrieval.jsonl")
    parser.add_argument("--output_path", default="outputs/appworld_dynamic_routing_audit/report.json")
    parser.add_argument("--model_cache_dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 350])
    parser.add_argument("--max_pairs", type=int)
    parser.add_argument("--sampling_strategy", choices=["head", "stride"], default="head")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--query_text_format",
        default="auto",
        choices=["auto", "raw", "skillrouter", "skillrouter_state", "appworld_skillrouter_query"],
        help="Query serialization for Stage0 retrieval. auto uses the checkpoint config.",
    )
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_appworld_dynamic_routing(
        checkpoint_path=args.checkpoint_path,
        base_skill_pool_path=args.base_skill_pool_path,
        dynamic_skill_pool_path=args.dynamic_skill_pool_path,
        retrieval_path=args.retrieval_path,
        output_path=args.output_path,
        model_cache_dir=args.model_cache_dir,
        batch_size=args.batch_size,
        k_values=args.k_values,
        max_pairs=args.max_pairs,
        sampling_strategy=args.sampling_strategy,
        device=args.device,
        query_text_format=args.query_text_format,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_action_required and report.get("status") != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
