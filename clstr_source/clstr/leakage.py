from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clstr.data import read_jsonl


SPLITS = ("train", "dev", "test")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_skillrouter_eval(skillrouter_eval_root: Path) -> tuple[list[str], dict[str, list[str]]]:
    tasks_path = skillrouter_eval_root / "tasks.jsonl"
    relevance_path = skillrouter_eval_root / "relevance.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"SkillRouter eval tasks are missing: {tasks_path}")
    if not relevance_path.exists():
        raise FileNotFoundError(f"SkillRouter eval relevance labels are missing: {relevance_path}")

    tasks = read_jsonl(tasks_path)
    relevance = json.loads(relevance_path.read_text(encoding="utf-8"))
    query_ids = [str(task["task_id"]) for task in tasks]
    gt_skill_ids = {
        task_id: [str(skill_id) for skill_id in relevance.get(task_id, {}).get("gt_skill_ids", [])]
        for task_id in query_ids
    }
    return query_ids, gt_skill_ids


def _read_task_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [str(row.get("task_id", row.get("id"))) for row in read_jsonl(path)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        values = payload.get("task_ids", payload.get("ids", []))
        return [str(item) for item in values]
    raise ValueError(f"unsupported split id payload: {path}")


def _find_split_file(skillsbench_root: Path, split: str) -> Path | None:
    candidates = [
        skillsbench_root / f"{split}_task_ids.json",
        skillsbench_root / f"{split}.json",
        skillsbench_root / f"{split}.jsonl",
        skillsbench_root / "splits" / f"{split}_task_ids.json",
        skillsbench_root / "splits" / f"{split}.json",
        skillsbench_root / "splits" / f"{split}.jsonl",
    ]
    return next((path for path in candidates if path.exists()), None)


def _write_skillsbench_split_audits(skillsbench_root: Path, output_dir: Path) -> dict[str, Any]:
    if not skillsbench_root.exists():
        for split in SPLITS:
            _write_json(
                output_dir / f"skillsbench_{split}_task_ids.json",
                {"status": "missing", "split": split, "task_ids": []},
            )
        return {"status": "missing", "path": str(skillsbench_root)}

    split_status: dict[str, Any] = {}
    for split in SPLITS:
        split_file = _find_split_file(skillsbench_root, split)
        if split_file is None:
            payload = {"status": "missing", "split": split, "task_ids": []}
        else:
            payload = {"status": "ok", "split": split, "task_ids": _read_task_ids(split_file)}
        _write_json(output_dir / f"skillsbench_{split}_task_ids.json", payload)
        split_status[split] = payload
    return {"status": "ok", "path": str(skillsbench_root), "splits": split_status}


def run_leakage_audit(
    skillrouter_eval_root: str | Path,
    skillsbench_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    skillrouter_eval_root = Path(skillrouter_eval_root)
    skillsbench_root = Path(skillsbench_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_ids, gt_skill_ids = _load_skillrouter_eval(skillrouter_eval_root)
    excluded_positive_skill_ids = sorted({skill_id for ids in gt_skill_ids.values() for skill_id in ids})

    _write_json(output_dir / "skillrouter_eval_query_ids.json", query_ids)
    _write_json(output_dir / "skillrouter_eval_gt_skill_ids.json", gt_skill_ids)
    _write_json(output_dir / "excluded_positive_skill_ids.json", excluded_positive_skill_ids)
    skillsbench_report = _write_skillsbench_split_audits(skillsbench_root, output_dir)
    split_task_ids = {
        split: json.loads((output_dir / f"skillsbench_{split}_task_ids.json").read_text(encoding="utf-8")).get("task_ids", [])
        for split in SPLITS
    }
    query_id_set = set(query_ids)
    leakage_checks = {
        "skillsbench_train_overlaps_skillrouter_eval_queries": sorted(set(split_task_ids["train"]) & query_id_set),
        "skillsbench_dev_overlaps_skillrouter_eval_queries": sorted(set(split_task_ids["dev"]) & query_id_set),
        "skillsbench_test_overlaps_skillrouter_eval_queries": sorted(set(split_task_ids["test"]) & query_id_set),
    }
    leakage_checks["status"] = "failed" if any(leakage_checks[key] for key in leakage_checks) else "ok"

    report = {
        "skillrouter_eval": {
            "status": "ok",
            "path": str(skillrouter_eval_root),
            "query_count": len(query_ids),
            "gt_skill_id_count": len(excluded_positive_skill_ids),
        },
        "skillsbench": skillsbench_report,
        "leakage_checks": leakage_checks,
        "artifacts": {
            "skillrouter_eval_query_ids": str(output_dir / "skillrouter_eval_query_ids.json"),
            "skillrouter_eval_gt_skill_ids": str(output_dir / "skillrouter_eval_gt_skill_ids.json"),
            "skillsbench_train_task_ids": str(output_dir / "skillsbench_train_task_ids.json"),
            "skillsbench_dev_task_ids": str(output_dir / "skillsbench_dev_task_ids.json"),
            "skillsbench_test_task_ids": str(output_dir / "skillsbench_test_task_ids.json"),
            "excluded_positive_skill_ids": str(output_dir / "excluded_positive_skill_ids.json"),
        },
    }
    _write_json(output_dir / "leakage_audit_summary.json", report)
    return report
