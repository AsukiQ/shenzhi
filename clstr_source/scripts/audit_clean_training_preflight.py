#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.clean_training_preflight import (
    audit_clean_training_preflight,
    audit_incremental_matched_union_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CLSTR clean training data before retraining.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--toolbench_eval_trajectories_path", required=True)
    parser.add_argument("--traject_eval_queries_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--near_duplicate_threshold", type=float, default=0.55)
    parser.add_argument("--near_duplicate_max_chars", type=int, default=1000)
    parser.add_argument("--near_duplicate_max_shingle_postings", type=int, default=256)
    parser.add_argument("--near_duplicate_max_query_features", type=int, default=64)
    parser.add_argument("--near_duplicate_max_candidates_per_row", type=int, default=128)
    parser.add_argument("--fail_on_near_duplicate", action="store_true")
    parser.add_argument("--fail_on_leakage", action="store_true")
    parser.add_argument("--require_structured_current_state", action="store_true")
    parser.add_argument("--tau2_test_rows_path")
    parser.add_argument("--toolsandbox_test_rows_path")
    parser.add_argument("--base_data_root")
    parser.add_argument("--base_clean_preflight_report")
    parser.add_argument("--matched_union_manifest_path")
    args = parser.parse_args()

    incremental_values = (
        args.base_data_root,
        args.base_clean_preflight_report,
        args.matched_union_manifest_path,
    )
    if any(incremental_values) and not all(incremental_values):
        parser.error(
            "incremental matched preflight requires base_data_root, "
            "base_clean_preflight_report, and matched_union_manifest_path"
        )
    if all(incremental_values):
        if not args.tau2_test_rows_path or not args.toolsandbox_test_rows_path:
            parser.error("incremental matched preflight requires both held-out test rows")
        report = audit_incremental_matched_union_preflight(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            base_clean_preflight_report=args.base_clean_preflight_report,
            matched_union_manifest_path=args.matched_union_manifest_path,
            toolbench_eval_trajectories_path=args.toolbench_eval_trajectories_path,
            traject_eval_queries_path=args.traject_eval_queries_path,
            tau2_test_rows_path=args.tau2_test_rows_path,
            toolsandbox_test_rows_path=args.toolsandbox_test_rows_path,
            output_path=args.output_path,
            near_duplicate_threshold=args.near_duplicate_threshold,
            near_duplicate_max_chars=args.near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=args.near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=args.near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=args.near_duplicate_max_candidates_per_row,
        )
    else:
        report = audit_clean_training_preflight(
            data_root=args.data_root,
            toolbench_eval_trajectories_path=args.toolbench_eval_trajectories_path,
            traject_eval_queries_path=args.traject_eval_queries_path,
            output_path=args.output_path,
            near_duplicate_threshold=args.near_duplicate_threshold,
            near_duplicate_max_chars=args.near_duplicate_max_chars,
            near_duplicate_max_shingle_postings=args.near_duplicate_max_shingle_postings,
            near_duplicate_max_query_features=args.near_duplicate_max_query_features,
            near_duplicate_max_candidates_per_row=args.near_duplicate_max_candidates_per_row,
            fail_on_near_duplicate=args.fail_on_near_duplicate,
            require_structured_current_state=args.require_structured_current_state,
            tau2_test_rows_path=args.tau2_test_rows_path,
            toolsandbox_test_rows_path=args.toolsandbox_test_rows_path,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_leakage and report.get("status") == "error":
        return 3
    return 0 if report.get("status") in {"ok", "warning", "error"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
