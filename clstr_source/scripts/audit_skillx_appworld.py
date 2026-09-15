#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.bridges.skillx.appworld_adapter import (
    audit_skillx_appworld,
    audit_skillx_appworld_leakage,
    build_appworld_skill_pool,
)


def _find_skillx_roots(search_root: Path, max_depth: int = 4) -> list[Path]:
    roots: list[Path] = []
    search_root = search_root.resolve()
    skip_names = {".git", ".cache", "__pycache__", "outputs", "models"}
    for current, dirs, _files in os.walk(search_root):
        current_path = Path(current).resolve()
        rel_parts = current_path.relative_to(search_root).parts
        if len(rel_parts) >= max_depth:
            dirs[:] = []
        dirs[:] = [item for item in dirs if item not in skip_names]
        is_likely_repo = current_path.name == "SkillX" or (
            current_path.name.lower() == "skillx" and (current_path / ".git").exists()
        )
        if is_likely_repo:
            roots.append(current_path)
    return sorted(set(roots))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SkillX AppWorld skills for CLSTR.")
    parser.add_argument("--skillx_root")
    parser.add_argument("--search_root", default="/data/home/scyb713/run/xzf/AAAI/autodl-tmp")
    parser.add_argument("--output_dir", default="outputs/skillx_audit")
    parser.add_argument("--min_skill_count", type=int, default=100)
    parser.add_argument("--build_pool", action="store_true")
    parser.add_argument("--pool_output_dir", default="data/appworld_skill_pool")
    parser.add_argument("--leakage_audit", action="store_true")
    parser.add_argument("--skill_pool_path", default="data/appworld_skill_pool/skill_pool.jsonl")
    parser.add_argument("--routing_dir", default="data/appworld_routing")
    parser.add_argument("--near_duplicate_threshold", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.leakage_audit:
        report = audit_skillx_appworld_leakage(
            args.skill_pool_path,
            args.routing_dir,
            output_dir=output_dir,
            skillx_root=args.skillx_root,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.skillx_root:
        skillx_root = Path(args.skillx_root)
    else:
        candidates = _find_skillx_roots(Path(args.search_root))
        if not candidates:
            report = audit_skillx_appworld(
                Path(args.search_root) / "SkillX",
                output_dir=output_dir,
                min_skill_count=args.min_skill_count,
            )
            report["searched_root"] = str(Path(args.search_root))
            report["candidate_roots"] = []
            report["setup_script"] = "scripts/prepare_skillx_appworld.sh"
            blocker_path = output_dir / "blocker_report.json"
            blocker_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
            return
        skillx_root = candidates[0]

    report = audit_skillx_appworld(
        skillx_root,
        output_dir=output_dir,
        min_skill_count=args.min_skill_count,
    )
    if args.build_pool and report["status"] == "ok":
        report["skill_pool_manifest"] = build_appworld_skill_pool(
            skillx_root,
            args.pool_output_dir,
            min_skill_count=args.min_skill_count,
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
