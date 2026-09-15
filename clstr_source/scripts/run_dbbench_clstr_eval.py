#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_dbbench_harness_audit(
    output_path: str | Path = "outputs/dbbench_eval/harness_audit_report.json",
) -> dict[str, Any]:
    dbbench_package_available = importlib.util.find_spec("dbbench") is not None
    report = {
        "status": "blocked_no_closed_loop_eval_harness",
        "benchmark": "dbbench",
        "closed_loop_evaluated": False,
        "dbbench_package_available": dbbench_package_available,
        "known_sft_dataset": "u-10bei/dbbench_sft_dataset_react_v4",
        "sft_dataset_is_eval_harness": False,
        "reason": (
            "The registered DBBench source is an SFT/ReAct trajectory dataset, not a local official closed-loop "
            "evaluation harness. No DBBench success metric is reported."
        ),
        "missing_items": [
            "official_closed_loop_harness",
            "official_eval_split_definition",
            "closed_loop_eval_command",
            "official_or_reproducible_data_download_command",
            "clstr_env_adapter_for_official_harness",
        ],
        "reproducible_commands": {
            "audit": "python -B scripts/run_dbbench_clstr_eval.py --output_path outputs/dbbench_eval/harness_audit_report.json",
            "check_python_package": "python -B -c 'import importlib.util; print(importlib.util.find_spec(\"dbbench\") is not None)'",
            "install_harness": "python -m pip install <official-dbbench-package-or-repo>",
            "download_data": "<official-dbbench-data-download-command>",
            "smoke_test": "<official-dbbench-smoke-test-command>",
        },
        "next_reproducible_steps": [
            "Locate the official DBBench closed-loop evaluation repository or package.",
            "Install it in a CLSTR-local environment or provide a repo path.",
            "Define train/eval split rules before using u-10bei/dbbench_sft_dataset_react_v4.",
            "Run a smoke episode before reporting CLSTR metrics.",
        ],
        "not_closed_loop_success": True,
    }
    _write_json(Path(output_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DBBench closed-loop harness availability.")
    parser.add_argument("--output_path", default="outputs/dbbench_eval/harness_audit_report.json")
    args = parser.parse_args()
    report = build_dbbench_harness_audit(output_path=Path(args.output_path))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
