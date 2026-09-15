from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _row_id(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    return None


def _required_file_report(data_dir: Path) -> tuple[list[str], dict[str, str]]:
    files = {
        "skills": data_dir / "skills.jsonl",
        "retrieval": data_dir / "retrieval.jsonl",
        "trajectories": data_dir / "trajectories.jsonl",
    }
    missing = [
        str(path)
        for path in files.values()
        if not path.exists() or path.stat().st_size <= 0
    ]
    return missing, {name: str(path) for name, path in files.items()}


def _source_report(source_root: str | Path | None, *, expected_answer_files: int | None = None) -> dict[str, Any]:
    if source_root is None:
        return {
            "status": "not_checked",
            "source_root": None,
            "has_instruction_g3": None,
            "has_answer_g3": None,
            "g3_answer_files": None,
            "expected_g3_answer_files": expected_answer_files,
            "is_data_example": None,
            "static_smoke_assets": {"available": None},
        }
    root = Path(source_root)
    data_root = root / "data" if (root / "data" / "instruction" / "G3_query.json").is_file() else root
    answer_files = sorted((data_root / "answer" / "G3_answer").glob("*.json")) if (data_root / "answer" / "G3_answer").exists() else []
    static_in_domain = data_root / "toolbench_static" / "in_domain.json"
    static_out_of_domain = data_root / "toolbench_static" / "out_of_domain.json"
    is_data_example = "data_example" in {part.lower() for part in data_root.parts}
    has_instruction = (data_root / "instruction" / "G3_query.json").is_file()
    has_answer = bool(answer_files)
    if is_data_example:
        status = "data_example_not_allowed"
    elif has_instruction and has_answer and expected_answer_files is not None and len(answer_files) < expected_answer_files:
        status = "incomplete_answer_files"
    elif has_instruction and has_answer:
        status = "official_g3_available"
    else:
        status = "missing_official_g3_files"
    return {
        "status": status,
        "source_root": str(data_root),
        "has_instruction_g3": has_instruction,
        "has_answer_g3": has_answer,
        "g3_answer_files": len(answer_files),
        "expected_g3_answer_files": expected_answer_files,
        "is_data_example": is_data_example,
        "static_smoke_assets": {
            "available": static_in_domain.is_file() and static_out_of_domain.is_file(),
            "may_replace_official_toolbench_g3": False,
        },
    }


def _flatten_skill_ids(values: Iterable[Any]) -> set[str]:
    ids: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            ids.update(str(item) for item in value if str(item))
        elif str(value):
            ids.add(str(value))
    return ids


def audit_toolbench_g3_data(
    *,
    data_dir: str | Path = "data/toolbench_g3",
    source_root: str | Path | None = None,
    output_path: str | Path | None = None,
    expected_answer_files: int | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    missing_files, files = _required_file_report(data_dir)
    official_source = _source_report(source_root, expected_answer_files=expected_answer_files)
    blockers: list[str] = []
    if missing_files:
        blockers.append("missing_normalized_toolbench_g3_files")
    if official_source["status"] == "data_example_not_allowed":
        blockers.append("source_root_is_data_example_not_official_g3")
    elif official_source["status"] == "incomplete_answer_files":
        blockers.append("source_root_incomplete_g3_answer_files")
    elif official_source["status"] == "missing_official_g3_files":
        blockers.append("source_root_missing_official_g3_files")

    if missing_files:
        report = {
            "status": "action_required",
            "benchmark": "ToolBench-G3",
            "data_dir": str(data_dir),
            "blockers": blockers,
            "missing_files": missing_files,
            "files": files,
            "official_source": official_source,
            "skill_count": 0,
            "unique_skill_count": 0,
            "retrieval_pair_count": 0,
            "trajectory_row_count": 0,
            "may_use_for_paper_main_table": False,
        }
        if output_path is not None:
            _write_json(output_path, report)
        return report

    skills = _read_jsonl(data_dir / "skills.jsonl")
    retrieval = _read_jsonl(data_dir / "retrieval.jsonl")
    trajectories = _read_jsonl(data_dir / "trajectories.jsonl")

    skill_ids = [
        skill_id
        for row in skills
        if (skill_id := _row_id(row, "skill_id", "id", "tool_id")) is not None
    ]
    skill_id_counts = Counter(skill_ids)
    duplicate_skill_ids = sorted(skill_id for skill_id, count in skill_id_counts.items() if count > 1)
    skill_id_set = set(skill_ids)

    retrieval_positive_ids = _flatten_skill_ids(row.get("positive_skill_id") for row in retrieval)
    retrieval_negative_ids = _flatten_skill_ids(row.get("negative_skill_ids") for row in retrieval)
    trajectory_ids = _flatten_skill_ids(
        value
        for row in trajectories
        for value in (row.get("skill_id"), row.get("next_skill_id"))
    )
    missing_retrieval_positive_skill_ids = sorted(retrieval_positive_ids - skill_id_set)
    missing_retrieval_negative_skill_ids = sorted(retrieval_negative_ids - skill_id_set)
    missing_trajectory_skill_ids = sorted(trajectory_ids - skill_id_set)

    if duplicate_skill_ids:
        blockers.append("duplicate_skill_ids")
    if not retrieval:
        blockers.append("no_retrieval_pairs")
    if not trajectories:
        blockers.append("no_trajectory_rows")
    if missing_retrieval_positive_skill_ids:
        blockers.append("missing_retrieval_positive_skill_ids")
    if missing_retrieval_negative_skill_ids:
        blockers.append("missing_retrieval_negative_skill_ids")
    if missing_trajectory_skill_ids:
        blockers.append("missing_trajectory_skill_ids")

    report = {
        "status": "ok" if not blockers else "action_required",
        "benchmark": "ToolBench-G3",
        "data_dir": str(data_dir),
        "blockers": blockers,
        "missing_files": [],
        "files": files,
        "official_source": official_source,
        "skill_count": len(skills),
        "unique_skill_count": len(skill_id_set),
        "retrieval_pair_count": len(retrieval),
        "trajectory_row_count": len(trajectories),
        "duplicate_skill_ids": duplicate_skill_ids,
        "missing_retrieval_positive_skill_ids": missing_retrieval_positive_skill_ids,
        "missing_retrieval_negative_skill_ids": missing_retrieval_negative_skill_ids,
        "missing_trajectory_skill_ids": missing_trajectory_skill_ids,
        "may_use_for_paper_main_table": not blockers,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
