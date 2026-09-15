"""Fail-closed verification for Shenzhi canonical vNext Stage0 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(output_dir: Path) -> dict[str, Any]:
    required = {
        name: output_dir / name
        for name in (
            "train_report.json",
            "stage0_selection.json",
            "stage0_quality_gate.json",
            "trainer_progress.json",
            "selected_skills.jsonl",
            "source_manifest.json",
            "backbone_snapshot.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage0 output lacks required artifacts: {missing}")
    train = json.loads(required["train_report.json"].read_text(encoding="utf-8"))
    selection = json.loads(
        required["stage0_selection.json"].read_text(encoding="utf-8")
    )
    quality = json.loads(
        required["stage0_quality_gate.json"].read_text(encoding="utf-8")
    )
    progress = json.loads(
        required["trainer_progress.json"].read_text(encoding="utf-8")
    )
    if train.get("status") != "ok" or selection.get("status") != "ok" or quality.get("status") != "ok":
        raise ValueError("Stage0 trainer/selection/quality status is not ok")
    selected_checkpoint = Path(str(selection.get("selected_checkpoint_path") or ""))
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage0 selected checkpoint is missing: {selected_checkpoint}"
        )
    if Path(str(quality.get("selected_checkpoint_path") or "")).resolve() != selected_checkpoint.resolve():
        raise ValueError("Stage0 quality gate and selection disagree on checkpoint")
    if int(progress.get("completed_step") or -1) < int(selection.get("selected_step") or 0):
        raise ValueError("Stage0 progress does not cover the selected checkpoint")
    score_gain = float(selection.get("score_gain") or 0.0)
    minimum = float(selection.get("minimum_score_gain") or 0.0)
    if score_gain < minimum:
        raise ValueError("Stage0 selected score gain misses its quality threshold")
    return {
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "selected_checkpoint_path": str(selected_checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "selected_step": int(selection["selected_step"]),
        "initial_score": float(selection["initial_score"]),
        "selected_score": float(selection["selected_score"]),
        "score_gain": score_gain,
        "completed_step": int(progress["completed_step"]),
        "retrieval_row_count": int(train.get("retrieval_row_count") or 0),
        "retrieval_dev_row_count": int(train.get("retrieval_dev_row_count") or 0),
        "static_route_row_count": int(train.get("static_route_row_count") or 0),
        "static_route_dev_row_count": int(train.get("static_route_dev_row_count") or 0),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in required.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    report = verify(Path(args.output_dir))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
