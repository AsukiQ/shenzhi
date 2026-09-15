from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clstr.stage2_v4_pre_audit import run_stage2_v4_pre_audit


def _benchmark_caps(value: str | None) -> dict[str, int]:
    if value is None or not str(value).strip():
        return {}
    caps: dict[str, int] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, raw = item.split("=", 1)
        elif ":" in item:
            key, raw = item.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(f"invalid benchmark cap item: {item!r}; expected key=value")
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"invalid benchmark cap item: {item!r}; empty key")
        try:
            caps[key] = int(raw.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid cap for {key!r}: {raw!r}") from exc
    return caps


def _benchmarks(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage2 v4 failure modes from row-level diagnostics.")
    parser.add_argument(
        "--row_diagnostics_path",
        default="outputs/clstr_stage2_transition_row_diag_full_top350_v1/transition_row_diagnostics.jsonl",
    )
    parser.add_argument(
        "--trajectories_path",
        default="data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl",
    )
    parser.add_argument("--output_dir", default="outputs/clstr_stage2_v4_pre_audit_top350_v1")
    parser.add_argument("--benchmarks", type=_benchmarks, default="toolbench_g3,traject_bench")
    parser.add_argument("--example_limit", type=int, default=25)
    parser.add_argument(
        "--benchmark_caps",
        type=_benchmark_caps,
        default="toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000",
    )
    args = parser.parse_args()
    report = run_stage2_v4_pre_audit(
        row_diagnostics_path=args.row_diagnostics_path,
        trajectories_path=args.trajectories_path,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        example_limit=args.example_limit,
        benchmark_caps=args.benchmark_caps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
