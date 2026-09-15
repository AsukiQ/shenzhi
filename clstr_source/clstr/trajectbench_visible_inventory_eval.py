from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_traject_skill_ids(skills_path: Path) -> list[str]:
    skill_ids: set[str] = set()
    for row in _iter_jsonl(skills_path):
        skill_id = str(row.get("skill_id") or "").strip()
        if skill_id.startswith("traject/"):
            skill_ids.add(skill_id)
    return sorted(skill_ids)


def _domain_from_source_path(value: str) -> str:
    parts = Path(value).parts
    for marker in ("parallel", "sequential"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return str(parts[idx + 1])
    return ""


def _domain_from_skill_row(row: dict[str, Any]) -> str:
    for key in ("environment", "domain", "domain_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("domain", "domain_name"):
        value = str(provenance.get(key) or "").strip()
        if value:
            return value
    source_files = row.get("source_files")
    if isinstance(source_files, list):
        for source_file in source_files:
            domain = _domain_from_source_path(str(source_file))
            if domain:
                return domain
    source_file = provenance.get("source_file") or provenance.get("path")
    if source_file:
        domain = _domain_from_source_path(str(source_file))
        if domain:
            return domain
    return ""


def _domain_from_trajectory_row(row: dict[str, Any]) -> str:
    for key in ("domain", "task"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("domain", "domain_name"):
        value = str(provenance.get(key) or "").strip()
        if value:
            return value
    trajectory_id = str(row.get("trajectory_id") or "")
    parts = trajectory_id.split("::")
    if len(parts) >= 4 and parts[0] == "traject":
        return parts[2]
    return ""


def _load_traject_skill_ids_by_domain(skills_path: Path) -> dict[str, list[str]]:
    by_domain: dict[str, set[str]] = {}
    for row in _iter_jsonl(skills_path):
        skill_id = str(row.get("skill_id") or "").strip()
        if not skill_id.startswith("traject/"):
            continue
        domain = _domain_from_skill_row(row)
        if not domain:
            continue
        by_domain.setdefault(domain, set()).add(skill_id)
    return {domain: sorted(skill_ids) for domain, skill_ids in sorted(by_domain.items())}


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_trajectbench_visible_inventory_eval(
    *,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    visible_inventory_mode: str = "global",
) -> dict[str, Any]:
    trajectories_path = Path(trajectories_path)
    skills_path = Path(skills_path)
    output_path = Path(output_path)
    report_path = Path(report_path) if report_path is not None else None

    if visible_inventory_mode not in {"global", "domain_local"}:
        raise ValueError(f"Unsupported visible_inventory_mode: {visible_inventory_mode}")

    visible_inventory = _load_traject_skill_ids(skills_path)
    if not visible_inventory:
        raise ValueError(f"No traject/* skills found in {skills_path}")
    visible_set = set(visible_inventory)
    visible_by_domain = _load_traject_skill_ids_by_domain(skills_path) if visible_inventory_mode == "domain_local" else {}

    report: dict[str, Any] = {
        "status": "ok",
        "trajectories_path": str(trajectories_path),
        "skills_path": str(skills_path),
        "output_path": str(output_path),
        "visible_inventory_mode": visible_inventory_mode,
        "visible_inventory_source": (
            "skill_pool_traject_domain_local" if visible_inventory_mode == "domain_local" else "skill_pool_traject_global"
        ),
        "visible_inventory_count": len(visible_inventory),
        "visible_inventory_count_mean": float(len(visible_inventory)),
        "domain_local_inventory_domains": sorted(visible_by_domain),
        "domain_local_missing_domain_rows": 0,
        "domain_local_empty_inventory_rows": 0,
        "domain_local_fallback_global_rows": 0,
        "source_rows": 0,
        "written_rows": 0,
        "skipped_non_traject_rows": 0,
        "removed_tool_inventory_rows": 0,
        "current_skill_visible_rows": 0,
        "current_skill_missing_rows": 0,
        "next_skill_rows": 0,
        "next_skill_visible_rows": 0,
        "next_skill_missing_rows": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_sizes: list[int] = []
    with output_path.open("w", encoding="utf-8") as fout:
        for row in _iter_jsonl(trajectories_path):
            report["source_rows"] += 1
            if row.get("benchmark") != "traject_bench":
                report["skipped_non_traject_rows"] += 1
                continue

            normalized = dict(row)
            if "tool_inventory_skill_ids" in normalized:
                report["removed_tool_inventory_rows"] += 1
                normalized.pop("tool_inventory_skill_ids", None)
            row_domain = ""
            row_visible_inventory = visible_inventory
            row_visible_source = "skill_pool_traject_global"
            if visible_inventory_mode == "domain_local":
                row_domain = _domain_from_trajectory_row(normalized)
                if not row_domain:
                    report["domain_local_missing_domain_rows"] += 1
                    report["domain_local_fallback_global_rows"] += 1
                else:
                    domain_inventory = visible_by_domain.get(row_domain) or []
                    if domain_inventory:
                        row_visible_inventory = domain_inventory
                        row_visible_source = "skill_pool_traject_domain_local"
                    else:
                        report["domain_local_empty_inventory_rows"] += 1
                        report["domain_local_fallback_global_rows"] += 1
            normalized["visible_inventory_skill_ids"] = row_visible_inventory
            row_visible_set = set(row_visible_inventory)
            inventory_sizes.append(len(row_visible_inventory))

            provenance = dict(normalized.get("provenance") or {})
            provenance["visible_inventory_source"] = row_visible_source
            if row_domain:
                provenance["visible_inventory_domain"] = row_domain
            provenance["oracle_tool_inventory_removed"] = True
            normalized["provenance"] = provenance

            skill_id = str(normalized.get("skill_id") or "")
            if skill_id in row_visible_set:
                report["current_skill_visible_rows"] += 1
            else:
                report["current_skill_missing_rows"] += 1

            next_skill_id = str(normalized.get("next_skill_id") or "")
            if next_skill_id:
                report["next_skill_rows"] += 1
                if next_skill_id in row_visible_set:
                    report["next_skill_visible_rows"] += 1
                else:
                    report["next_skill_missing_rows"] += 1

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            report["written_rows"] += 1

    report["visible_inventory_count_mean"] = _mean(inventory_sizes)
    report["visible_inventory_count_min"] = min(inventory_sizes) if inventory_sizes else 0
    report["visible_inventory_count_max"] = max(inventory_sizes) if inventory_sizes else 0

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
