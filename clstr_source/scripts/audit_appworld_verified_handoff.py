#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clstr.appworld_skill_handoff import build_verified_skill_handoff


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _required_apps_from_state(state_text: str, fallback: list[str] | None = None) -> list[str]:
    for line in str(state_text or "").splitlines():
        if line.lower().startswith("required apps:"):
            raw = line.split(":", 1)[1]
            return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item) for item in (fallback or []) if str(item)]


def _load_api_refs(appworld_root: str | Path, required_apps: list[str]) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = {
        ("supervisor", "show_profile"),
        ("supervisor", "show_account_passwords"),
        ("supervisor", "complete_task"),
    }
    docs_dir = Path(appworld_root) / "data" / "api_docs" / "standard"
    for app in sorted({str(app) for app in required_apps if str(app) and str(app) != "supervisor"}):
        path = docs_dir / f"{app}.json"
        if not path.exists():
            continue
        try:
            docs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(docs, dict):
            continue
        for api_name in docs:
            refs.add((str(app), str(api_name)))
    return refs


def _compact_instruction(row: dict[str, Any], step: dict[str, Any]) -> str:
    state_text = str(step.get("state_text") or "")
    if state_text:
        return state_text
    return str(row.get("user_goal") or row.get("instruction_text") or row.get("instruction") or row.get("query") or "")


def _decision_counts_add(total: dict[str, int], counts: dict[str, Any]) -> None:
    for key in ("inject_hint", "schema_only", "suppress"):
        total[key] = int(total.get(key, 0)) + int(counts.get(key, 0))


def audit_verified_handoff(
    *,
    runs_path: str | Path,
    skill_pool_path: str | Path,
    appworld_root: str | Path,
    output_dir: str | Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    skill_by_id = {str(row.get("skill_id")): row for row in _read_jsonl(skill_pool_path)}
    run_rows = _read_jsonl(runs_path)
    if max_rows is not None:
        run_rows = run_rows[: int(max_rows)]

    audit_rows: list[dict[str, Any]] = []
    decision_counts = {"inject_hint": 0, "schema_only": 0, "suppress": 0}
    missing_skill_ids: set[str] = set()
    for row in run_rows:
        steps = row.get("steps") if isinstance(row.get("steps"), list) else []
        for step in steps:
            selected_ids = [str(item) for item in step.get("selected_skill_ids", [])]
            skills: list[dict[str, Any]] = []
            for skill_id in selected_ids:
                skill = skill_by_id.get(skill_id)
                if skill is None:
                    missing_skill_ids.add(skill_id)
                    continue
                skills.append(skill)
            instruction = _compact_instruction(row, step)
            required_apps = _required_apps_from_state(
                instruction,
                fallback=[str(item) for item in row.get("required_apps", [])] if isinstance(row.get("required_apps"), list) else [],
            )
            valid_api_refs = _load_api_refs(appworld_root, required_apps)
            handoff = build_verified_skill_handoff(
                instruction=instruction,
                skills=skills,
                valid_api_refs=valid_api_refs,
                required_apps=set(required_apps),
            )
            counts = handoff.get("handoff_decision_counts", {})
            _decision_counts_add(decision_counts, counts if isinstance(counts, dict) else {})
            audit_rows.append(
                {
                    "query_id": str(row.get("query_id") or row.get("task_id") or ""),
                    "task_id": str(row.get("task_id") or row.get("query_id") or ""),
                    "step_idx": int(step.get("step_idx", len(audit_rows))),
                    "selected_skill_ids": selected_ids,
                    "required_apps": required_apps,
                    "handoff_decision_counts": counts,
                    "handoff_decisions": handoff.get("handoff_decisions", []),
                    "useful_apis": handoff.get("useful_apis", []),
                    "state_changing_action_apis": handoff.get("state_changing_action_apis", []),
                    "workflow_hints": handoff.get("workflow_hints", []),
                    "suppressed_schema_refs": handoff.get("suppressed_schema_refs", []),
                    "prompt_block": handoff.get("prompt_block", ""),
                }
            )

    output_dir = Path(output_dir)
    rows_path = output_dir / "verified_handoff_rows.jsonl"
    report_path = output_dir / "verified_handoff_report.json"
    _write_jsonl(rows_path, audit_rows)
    report = {
        "runs_path": str(runs_path),
        "skill_pool_path": str(skill_pool_path),
        "appworld_root": str(appworld_root),
        "output_dir": str(output_dir),
        "rows_path": str(rows_path),
        "report_path": str(report_path),
        "task_count": len(run_rows),
        "step_count": len(audit_rows),
        "decision_counts": decision_counts,
        "missing_skill_count": len(missing_skill_ids),
        "missing_skill_ids": sorted(missing_skill_ids)[:50],
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only audit for AppWorld verified SkillX handoff decisions.")
    parser.add_argument("--runs_path", required=True)
    parser.add_argument("--skill_pool_path", default="data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl")
    parser.add_argument("--appworld_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    report = audit_verified_handoff(
        runs_path=args.runs_path,
        skill_pool_path=args.skill_pool_path,
        appworld_root=args.appworld_root,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
