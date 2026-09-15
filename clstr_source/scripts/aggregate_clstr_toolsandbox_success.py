#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _expected_scenarios(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = payload["toolsandbox_split_manifest"]
    family_to_split = split["family_to_split"]
    return sorted(
        name
        for name, row in split["scenario_records"].items()
        if family_to_split[row["family_id"]] == "test"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_report", action="append", required=True)
    parser.add_argument("--matched_union_manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.matched_union_manifest_path).resolve()
    expected = _expected_scenarios(manifest_path)
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.input_report
    ]
    if any(report.get("status") != "ok" for report in reports):
        raise ValueError("all ToolSandbox shard reports must have status=ok")
    invariants = (
        "method",
        "executor_model",
        "split_protocol",
        "matched_union_manifest_sha256",
    )
    for key in invariants:
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1:
            raise ValueError(f"ToolSandbox shard invariant differs: {key}")
    if reports[0]["matched_union_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("ToolSandbox aggregate manifest digest differs")

    rows: list[dict[str, Any]] = []
    for report in reports:
        summary_path = Path(report["official_result_summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.extend(summary["per_scenario_results"])
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("ToolSandbox shard scenarios overlap")
    if sorted(names) != expected:
        raise ValueError(
            f"ToolSandbox shard coverage differs: expected {len(expected)}, observed {len(names)}"
        )
    if any(row.get("exception_type") is not None for row in rows):
        raise ValueError("ToolSandbox aggregate contains scenario exceptions")

    similarities = [float(row["similarity"]) for row in rows]
    milestones = [float(row["milestone_similarity"]) for row in rows]
    minefields = [float(row["minefield_similarity"]) for row in rows]
    exact_count = sum(value >= 1.0 - 1e-12 for value in similarities)
    output_path = Path(args.output_path).resolve()
    combined_summary_path = output_path.with_name("result_summary.json")
    _write_json(
        combined_summary_path,
        {"per_scenario_results": sorted(rows, key=lambda row: row["name"])},
    )
    report = {
        "status": "ok",
        "metric_scope": "official ToolSandbox held-out scenario execution",
        "method": reports[0]["method"],
        "executor_model": reports[0]["executor_model"],
        "split_protocol": reports[0]["split_protocol"],
        "evaluation_scope": "full_parallel_shards",
        "shard_count": len(reports),
        "scenario_count": len(rows),
        "mean_milestone_similarity": sum(milestones) / len(milestones),
        "mean_minefield_similarity": sum(minefields) / len(minefields),
        "mean_similarity": sum(similarities) / len(similarities),
        "exact_success_count": exact_count,
        "exact_success_rate": exact_count / len(rows),
        "exception_count": 0,
        "matched_union_manifest_path": str(manifest_path),
        "matched_union_manifest_sha256": _sha256(manifest_path),
        "official_result_summary": str(combined_summary_path),
        "input_reports": [str(Path(path).resolve()) for path in args.input_report],
        "checkpoint_binding": reports[0]["checkpoint_binding"],
    }
    _write_json(output_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
