#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.skill_pool_quality_audit import review_skill_dedup_borderline_candidates


def _load_builder_module():
    path = ROOT / "scripts" / "build_clstr_unified_pretrain.py"
    spec = importlib.util.spec_from_file_location("build_clstr_unified_pretrain", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_skill_dedup_borderline(
    *,
    skill_pool_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    review_output_path: str | Path | None = None,
    review_report_path: str | Path | None = None,
    max_candidates: int = 20000,
    max_bucket_size: int = 500,
) -> dict[str, Any]:
    skill_pool_path = Path(skill_pool_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    builder = _load_builder_module()

    records: dict[str, dict[str, Any]] = {}
    duplicate_skill_ids: list[str] = []
    for row in _read_jsonl(skill_pool_path):
        sid = str(row.get("skill_id") or row.get("canonical_skill_id") or "")
        if not sid:
            continue
        if sid in records:
            duplicate_skill_ids.append(sid)
            continue
        row = dict(row)
        row["skill_id"] = sid
        records[sid] = row

    alias_map = {sid: sid for sid in records}
    stats = builder.write_borderline_dedup_candidates(
        output_path,
        records,
        alias_map,
        max_bucket_size=max_bucket_size,
        max_candidates=max_candidates,
    )
    report = {
        **stats,
        "skill_pool_path": str(skill_pool_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "skill_count": len(records),
        "duplicate_skill_ids": duplicate_skill_ids[:100],
        "duplicate_skill_id_count": len(duplicate_skill_ids),
        "metric_scope": "Rule-based borderline skill dedup candidates for manual or LLM review; does not auto-merge.",
    }
    _write_json(report_path, report)
    if review_report_path is not None:
        review_report = review_skill_dedup_borderline_candidates(
            candidates_path=output_path,
            output_path=review_output_path,
            report_path=review_report_path,
        )
        report["conservative_review"] = review_report
        _write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit remaining borderline skill dedup candidates.")
    parser.add_argument("--skill_pool_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--review_output_path")
    parser.add_argument("--review_report_path")
    parser.add_argument("--max_candidates", type=int, default=20000)
    parser.add_argument("--max_bucket_size", type=int, default=500)
    args = parser.parse_args()

    report = audit_skill_dedup_borderline(
        skill_pool_path=args.skill_pool_path,
        output_path=args.output_path,
        report_path=args.report_path,
        review_output_path=args.review_output_path,
        review_report_path=args.review_report_path,
        max_candidates=args.max_candidates,
        max_bucket_size=args.max_bucket_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
