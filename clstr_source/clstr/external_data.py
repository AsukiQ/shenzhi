from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _exists_any(root: Path, candidates: list[str]) -> Path | None:
    for rel_path in candidates:
        path = root / rel_path
        if path.exists():
            return path
    return None


def build_clean_router_manifest(
    skillsbench_root: str | Path,
    audit_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    skillsbench_root = Path(skillsbench_root)
    audit_dir = Path(audit_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "audit_exclusions": _exists_any(audit_dir, ["excluded_positive_skill_ids.json"]),
        "train_split": _exists_any(skillsbench_root, ["splits/train_task_ids.json", "train_task_ids.json"]),
        "skill_pool": _exists_any(skillsbench_root, ["skills.jsonl", "skill_pool.jsonl", "data/skills.jsonl"]),
        "successful_trajectories": _exists_any(
            skillsbench_root,
            [
                "successful_trajectories.jsonl",
                "trajectories/successful.jsonl",
                "trajectories/success.jsonl",
            ],
        ),
        "harness_results": _exists_any(
            skillsbench_root,
            [
                "harness_results.jsonl",
                "results/harness_results.jsonl",
                "benchflow_results.jsonl",
            ],
        ),
    }
    missing = [name for name, path in required.items() if path is None]
    train_records: list[dict[str, Any]] = []
    clean_skill_records: list[dict[str, Any]] = []
    clean_task_records: list[dict[str, Any]] = []
    if not missing:
        train_payload = _read_json(required["train_split"])
        train_task_ids = {str(task_id) for task_id in train_payload.get("task_ids", [])}
        excluded_payload = _read_json(required["audit_exclusions"])
        if isinstance(excluded_payload, dict):
            excluded_skill_ids = {
                str(skill_id)
                for values in excluded_payload.values()
                for skill_id in (values if isinstance(values, list) else [values])
            }
        else:
            excluded_skill_ids = {str(skill_id) for skill_id in excluded_payload}
        skill_pool_records = _read_jsonl(required["skill_pool"])
        skill_records_by_id = {str(record.get("skill_id")): record for record in skill_pool_records}
        skill_ids = set(skill_records_by_id)

        for record in _read_jsonl(required["successful_trajectories"]):
            task_id = str(record.get("task_id", ""))
            if task_id not in train_task_ids or not record.get("success", True):
                continue
            steps = record.get("steps", [])
            if not isinstance(steps, list):
                continue
            clean_skill_ids = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                skill_id = step.get("skill_id") or step.get("skill")
                if skill_id is None:
                    continue
                skill_id = str(skill_id)
                if skill_id in excluded_skill_ids:
                    continue
                if skill_id not in skill_ids:
                    continue
                clean_skill_ids.append(skill_id)
            if clean_skill_ids:
                train_records.append(
                    {
                        "task_id": task_id,
                        "skill_ids": clean_skill_ids,
                        "source": "skillsbench_successful_trajectory",
                    }
                )
        if not train_records:
            missing.append("successful_trajectories")
        else:
            used_skill_ids = {
                skill_id
                for record in train_records
                for skill_id in record["skill_ids"]
            }
            clean_skill_records = [
                record for skill_id, record in skill_records_by_id.items() if skill_id in used_skill_ids
            ]
            for task_id in sorted({record["task_id"] for record in train_records}):
                instruction_path = skillsbench_root / "tasks" / task_id / "instruction.md"
                instruction_text = (
                    instruction_path.read_text(encoding="utf-8")
                    if instruction_path.exists()
                    else task_id
                )
                clean_task_records.append({"task_id": task_id, "instruction_text": instruction_text})

    status = "ready" if not missing else "missing_inputs"
    emitted_paths = [
        output_dir / "train.jsonl",
        output_dir / "skills.jsonl",
        output_dir / "train_tasks.jsonl",
    ]
    if status == "ready":
        write_jsonl(output_dir / "train.jsonl", train_records)
        write_jsonl(output_dir / "skills.jsonl", clean_skill_records)
        write_jsonl(output_dir / "train_tasks.jsonl", clean_task_records)
    else:
        for path in emitted_paths:
            if path.exists():
                path.unlink()
    manifest = {
        "status": status,
        "skillsbench_root": str(skillsbench_root),
        "audit_dir": str(audit_dir),
        "output_dir": str(output_dir),
        "inputs": {name: str(path) if path is not None else None for name, path in required.items()},
        "missing_inputs": missing,
        "record_counts": {
            "train": len(train_records),
            "skills": len(clean_skill_records),
            "train_tasks": len(clean_task_records),
        },
        "note": "formal clean router data is not emitted unless all required non-leaking SkillsBench inputs are present",
    }
    historical_manifest_path = skillsbench_root / "historical_import_manifest.json"
    if historical_manifest_path.exists():
        historical_payload = json.loads(historical_manifest_path.read_text(encoding="utf-8"))
        historical_filter = historical_payload.get("filter", historical_payload)
        manifest["historical_import"] = {
            "dataset": historical_filter.get("dataset"),
            "dataset_root": historical_filter.get("dataset_root"),
            "kept_train_successful_with_skills": historical_filter.get("kept_train_successful_with_skills"),
            "total_runs": historical_filter.get("total_runs"),
            "split_seen": historical_filter.get("split_seen"),
            "split_kept_for_training": historical_filter.get("split_kept_for_training"),
            "split_excluded": historical_filter.get("split_excluded"),
            "split_exclusion_reasons": historical_filter.get("split_exclusion_reasons"),
            "excluded": historical_filter.get("excluded"),
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def check_alfworld_root(alfworld_root: str | Path) -> dict[str, Any]:
    root = Path(alfworld_root)
    package_available = importlib.util.find_spec("alfworld") is not None
    adapter_requirements = [
        "ALFWorld package importable as `alfworld`",
        "text-game data mounted under the configured root",
        "admissible-command or high-level action template mapping",
        "environment success/done and invalid-action reporting",
    ]
    if not root.exists():
        return {
            "status": "missing",
            "path": str(root),
            "package_available": package_available,
            "adapter_requirements": adapter_requirements,
        }

    data_candidates = ["json_2.1.1", "data", "alfworld_data"]
    data_path = _exists_any(root, data_candidates)
    status = "ok" if package_available and data_path is not None else "incomplete"
    return {
        "status": status,
        "path": str(root),
        "package_available": package_available,
        "data_path": str(data_path) if data_path is not None else None,
        "adapter_requirements": adapter_requirements,
    }
