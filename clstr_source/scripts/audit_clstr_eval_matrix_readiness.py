#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.eval_matrix_readiness import audit_eval_matrix_readiness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CLSTR Phase H evaluation matrix readiness without running benchmark evaluation."
    )
    parser.add_argument(
        "--stage4_checkpoint",
        default=(
            "outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act/"
            "checkpoints/clstr_stage4_act-step2000.pt"
        ),
    )
    parser.add_argument("--stage4_output_dir", default="outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act")
    parser.add_argument("--min_stage4_steps", type=int, default=2000)
    parser.add_argument("--toolret_eval_dir", default="data/toolret_eval")
    parser.add_argument("--toolret_run_path", default="outputs/toolret_eval/clstr_retrieval/run.tsv")
    parser.add_argument("--traject_eval_dir", default="data/traject_eval_traject_split_test")
    parser.add_argument(
        "--traject_sequence_proxy_path",
        default="outputs/traject_eval_traject_split_test/traject_sequence_proxy_metrics.json",
    )
    parser.add_argument(
        "--traject_official_metrics_path",
        default="outputs/traject_eval_traject_split_test/official_metrics.json",
    )
    parser.add_argument("--toolbench_g3_data_dir", default="data/toolbench_g3")
    parser.add_argument("--toolbench_g3_source_root", default="../ToolBench/data")
    parser.add_argument("--toolbench_g3_routing_report", default="outputs/toolbench_g3/clstr_routing_eval/metrics.json")
    parser.add_argument("--stabletoolbench_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench")
    parser.add_argument("--converted_answer_path", default="outputs/toolbench_g3/stabletoolbench_converted")
    parser.add_argument("--candidate_model", default="clstr_toolbench_g3")
    parser.add_argument("--test_set", default="G3_instruction")
    parser.add_argument("--api_pool_file")
    parser.add_argument("--appworld_combination_report", default="outputs/appworld_combination_plot/report.json")
    parser.add_argument("--expected_toolbench_g3_answer_files", type=int, default=None)
    parser.add_argument("--output_path", default="outputs/clstr_eval_matrix_readiness/eval_matrix_readiness.json")
    parser.add_argument("--fail_on_action_required", action="store_true")
    args = parser.parse_args()

    report = audit_eval_matrix_readiness(
        stage4_checkpoint=args.stage4_checkpoint,
        stage4_output_dir=args.stage4_output_dir,
        min_stage4_steps=args.min_stage4_steps,
        toolret_eval_dir=args.toolret_eval_dir,
        toolret_run_path=args.toolret_run_path,
        traject_eval_dir=args.traject_eval_dir,
        traject_sequence_proxy_path=args.traject_sequence_proxy_path,
        traject_official_metrics_path=args.traject_official_metrics_path,
        toolbench_g3_data_dir=args.toolbench_g3_data_dir,
        toolbench_g3_source_root=args.toolbench_g3_source_root,
        toolbench_g3_routing_report=args.toolbench_g3_routing_report,
        stabletoolbench_root=args.stabletoolbench_root,
        converted_answer_path=args.converted_answer_path,
        candidate_model=args.candidate_model,
        test_set=args.test_set,
        api_pool_file=args.api_pool_file,
        appworld_combination_report=args.appworld_combination_report,
        expected_toolbench_g3_answer_files=args.expected_toolbench_g3_answer_files,
        output_path=args.output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_action_required and report.get("status") != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
