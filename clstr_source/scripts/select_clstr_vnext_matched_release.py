#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.vnext_matched_release import build_matched_release_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize one held-out-dev matched-multibench Stage2 release",
    )
    parser.add_argument("--stage2_output_dir", required=True)
    parser.add_argument("--selected_step", required=True, type=int)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--training_skills_path", required=True)
    parser.add_argument("--matched_union_manifest_path", required=True)
    parser.add_argument("--tau2_dev_report_path", required=True)
    parser.add_argument("--toolsandbox_dev_report_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        raise ValueError(f"matched release output already exists: {output_path}")
    source_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_status:
        raise ValueError("matched release finalization requires a clean immutable source")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    selection = build_matched_release_selection(
        stage2_output_dir=args.stage2_output_dir,
        selected_step=args.selected_step,
        checkpoint_path=args.checkpoint_path,
        training_skills_path=args.training_skills_path,
        matched_union_manifest_path=args.matched_union_manifest_path,
        tau2_dev_report_path=args.tau2_dev_report_path,
        toolsandbox_dev_report_path=args.toolsandbox_dev_report_path,
        source_commit=source_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True))
    if selection.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
