#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _score(value: Any) -> float:
    text = str(value)
    if text.endswith(".Solved") or text == "Solved":
        return 1.0
    if text.endswith(".Unsure") or text == "Unsure":
        return 0.5
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--candidate_model", required=True)
    parser.add_argument("--test_set", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.labels_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("StableToolBench labels must be a non-empty object")
    trial_ids = sorted(
        {
            int(trial_id)
            for row in payload.values()
            if isinstance(row, dict)
            for trial_id in (row.get("is_solved") or {})
        }
    )
    scores = []
    missing = 0
    for trial_id in trial_ids:
        total = 0.0
        for row in payload.values():
            labels = row.get("is_solved") if isinstance(row, dict) else {}
            labels = labels if isinstance(labels, dict) else {}
            label = labels.get(str(trial_id), labels.get(trial_id))
            missing += label is None
            total += _score(label)
        scores.append(total / len(payload))
    if not scores:
        raise ValueError("StableToolBench labels contain no judge trials")
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((value - mean) ** 2 for value in scores) / len(scores))
    report = {
        "status": "ok" if missing == 0 else "action_required",
        "metric_scope": "official StableToolBench Solvable Pass Rate (SoPR)",
        "method": args.method,
        "candidate_model": args.candidate_model,
        "test_set": args.test_set,
        "query_count": len(payload),
        "judge_trial_count": len(scores),
        "trial_scores": scores,
        "sopr": mean,
        "sopr_percent": mean * 100.0,
        "std": std,
        "std_percent": std * 100.0,
        "missing_label_count": missing,
        "official_labels_path": str(Path(args.labels_path).resolve()),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
