#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_routing import read_json, write_json


def _metric(report: dict, name: str) -> float | None:
    value = (report.get("metrics") or {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def build_comparison(report_paths: list[Path], output_dir: Path) -> dict:
    rows = []
    for path in report_paths:
        report = read_json(path, default={}) or {}
        rows.append(
            {
                "path": str(path),
                "status": report.get("status", "missing"),
                "method": report.get("method", path.parent.name),
                "query_count": report.get("query_count"),
                "skill_count": report.get("skill_count"),
                "recall@1": _metric(report, "recall@1"),
                "recall@5": _metric(report, "recall@5"),
                "recall@20": _metric(report, "recall@20"),
                "mrr@20": _metric(report, "mrr@20"),
                "caveat": report.get("caveat", ""),
            }
        )
    payload = {
        "status": "ok" if rows else "blocked",
        "comparison_role": "AppWorld routing-layer comparison; not official DB-state task-completion.",
        "rows": rows,
    }
    write_json(output_dir / "comparison.json", payload)
    lines = [
        "# AppWorld Routing Comparison",
        "",
        "| method | status | query_count | skill_count | recall@1 | recall@5 | recall@20 | mrr@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {query_count} | {skill_count} | {r1} | {r5} | {r20} | {mrr20} |".format(
                method=row["method"],
                status=row["status"],
                query_count=row["query_count"],
                skill_count=row["skill_count"],
                r1=row["recall@1"],
                r5=row["recall@5"],
                r20=row["recall@20"],
                mrr20=row["mrr@20"],
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AppWorld routing comparison from method reports.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/appworld_routing_comparison")
    args = parser.parse_args()
    payload = build_comparison([Path(item) for item in args.reports], Path(args.output_dir))
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
