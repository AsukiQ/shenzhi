from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REQUIRED_FIELDS = ("name", "description", "body")
OPTIONAL_FIELDS = ("input_schema", "output_schema", "executor_desc", "failure_modes")
APPWORLD_ROUTING_SPLITS = ("train", "dev", "test_normal", "test_challenge")
EVAL_SPLITS = ("dev", "test", "test_normal", "test_challenge")


@dataclass(frozen=True)
class SkillXAuditResult:
    report: dict[str, Any]
    skills: list[dict[str, Any]]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "unnamed-skill"


def _frontmatter_and_body(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    frontmatter: dict[str, str] = {}
    end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    if end is None:
        return frontmatter, text.strip()
    return frontmatter, "\n".join(lines[end + 1 :]).strip()


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dict_field(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _body_from_json_row(row: dict[str, Any]) -> str:
    body = _first_text(row, ("body", "skill_md", "content", "markdown", "prompt"))
    if body:
        return body
    workflow = row.get("workflow")
    if isinstance(workflow, list):
        return "\n".join(str(item) for item in workflow if str(item).strip())
    return ""


def _normalize_json_skill(row: dict[str, Any], source_path: Path, index: int) -> dict[str, Any] | None:
    payload = row
    if isinstance(row.get("skill"), dict):
        payload = row["skill"]

    name = _first_text(payload, ("name", "skill_name", "title", "id", "skill_id"))
    description = _first_text(payload, ("description", "desc", "summary", "instruction", "document"))
    body = _body_from_json_row(payload)
    if not any((name, description, body)):
        return None
    tools = _list_field(payload.get("tools"))
    executor_desc = _first_text(payload, ("executor_desc", "executor", "script", "code", "tool"))
    if not executor_desc and tools:
        executor_desc = ", ".join(tools)
    skill_id = _first_text(payload, ("skill_id", "id")) or _first_text(row, ("skill_id", "id"))
    if not skill_id:
        skill_id = f"skillx/appworld/{_slug(name or source_path.stem)}-{index}"
    return {
        "skill_id": skill_id,
        "name": name or source_path.stem,
        "description": description,
        "input_schema": _dict_field(payload.get("input_schema") or payload.get("inputs")),
        "output_schema": _dict_field(payload.get("output_schema") or payload.get("outputs")),
        "executor_desc": executor_desc,
        "failure_modes": _list_field(payload.get("failure_modes") or payload.get("failures") or payload.get("failure")),
        "body": body,
        "source_dataset": "SkillX-AppWorld",
        "source_path": str(source_path),
        "source_format": source_path.suffix.lower().lstrip(".") or "json",
    }


def _normalize_skill_md(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _frontmatter_and_body(text)
    title = ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
    name = frontmatter.get("name") or title or path.parent.name
    description = frontmatter.get("description") or ""
    if not description:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped
                break
    return {
        "skill_id": f"skillx/appworld/{_slug(name)}",
        "name": name,
        "description": description,
        "input_schema": {},
        "output_schema": {},
        "executor_desc": "",
        "failure_modes": [],
        "body": body,
        "source_dataset": "SkillX-AppWorld",
        "source_path": str(path),
        "source_format": "skill_md",
    }


def _iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        for key in ("skills", "skill_list", "data", "records"):
            if isinstance(value.get(key), list):
                for item in value[key]:
                    if isinstance(item, dict):
                        yield item
                return
        yield value


def _is_appworld_path(path: Path) -> bool:
    return any("appworld" in part.lower() for part in path.parts)


def _candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or not _is_appworld_path(path):
            continue
        name = path.name.lower()
        if name == "skill.md" or path.suffix.lower() in {".json", ".jsonl"}:
            files.append(path)
    return sorted(files)


def load_skillx_appworld_skills(skillx_root: str | Path) -> list[dict[str, Any]]:
    root = Path(skillx_root)
    skills: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in _candidate_files(root):
        if path.name.lower() == "skill.md":
            candidates = [_normalize_skill_md(path)]
        else:
            candidates = [
                normalized
                for idx, row in enumerate(_iter_json_rows(path))
                if (normalized := _normalize_json_skill(row, path, idx)) is not None
            ]
        for skill in candidates:
            base_id = str(skill["skill_id"])
            skill_id = base_id
            suffix = 2
            while skill_id in seen_ids:
                skill_id = f"{base_id}-{suffix}"
                suffix += 1
            skill["skill_id"] = skill_id
            seen_ids.add(skill_id)
            skills.append(skill)
    return skills


def _coverage(skills: list[dict[str, Any]], field: str) -> float:
    if not skills:
        return 0.0
    hits = 0
    for skill in skills:
        value = skill.get(field)
        if isinstance(value, str):
            hits += bool(value.strip())
        elif isinstance(value, (list, dict)):
            hits += bool(value)
        else:
            hits += value is not None
    return round(hits / len(skills), 6)


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "median": 0, "mean": 0.0, "p90": 0, "max": 0}
    ordered = sorted(lengths)
    p90_idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))
    return {
        "min": ordered[0],
        "median": int(median(ordered)),
        "mean": round(mean(ordered), 3),
        "p90": ordered[p90_idx],
        "max": ordered[-1],
    }


def _normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _match_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", str(value).lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _skill_match_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(skill.get("skill_id", "")),
            str(skill.get("name", "")),
            str(skill.get("description", "")),
            str(skill.get("executor_desc", "")),
            json.dumps(skill.get("failure_modes", []), ensure_ascii=False, sort_keys=True),
            str(skill.get("body", "")),
        ]
    )


def _routing_task_instruction(task: dict[str, Any]) -> str:
    return _first_text(task, ("instruction_text", "instruction", "goal", "query"))


def _load_routing_tasks(routing_dir: Path, splits: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    tasks_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = routing_dir / f"{split}_tasks.jsonl"
        if not path.exists():
            tasks_by_split[split] = []
            continue
        tasks = []
        for row in _read_jsonl(path):
            task_id = _first_text(row, ("task_id", "query_id", "id"))
            instruction = _routing_task_instruction(row)
            if not task_id and not instruction:
                continue
            tasks.append(
                {
                    "task_id": task_id,
                    "instruction_text": instruction,
                    "split": str(row.get("split", split)),
                }
            )
        tasks_by_split[split] = tasks
    return tasks_by_split


def _load_skillx_plan_user_tasks(skillx_root: Path | None) -> dict[str, list[str]]:
    if skillx_root is None or not skillx_root.exists():
        return {}
    plan_tasks: dict[str, list[str]] = {}
    for path in sorted((skillx_root / "skillx_db" / "appworld").glob("*/plan.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        plan = payload.get("plan", {}) if isinstance(payload, dict) else {}
        if not isinstance(plan, dict):
            continue
        for user_task in plan:
            instruction = str(user_task).strip()
            if instruction:
                plan_tasks.setdefault(instruction, []).append(str(path))
    return plan_tasks


def _empty_contamination() -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        "task_id_hits": {split: [] for split in APPWORLD_ROUTING_SPLITS},
        "instruction_hits": {split: [] for split in APPWORLD_ROUTING_SPLITS},
        "near_duplicate_instruction_hits": {split: [] for split in APPWORLD_ROUTING_SPLITS},
        "source_plan_instruction_overlaps": {split: [] for split in APPWORLD_ROUTING_SPLITS},
    }


def _hit_record(skill: dict[str, Any], task: dict[str, Any], reason: str, score: float | None = None) -> dict[str, Any]:
    record = {
        "skill_id": skill.get("skill_id"),
        "skill_name": skill.get("name"),
        "task_id": task.get("task_id"),
        "instruction_text": task.get("instruction_text"),
        "reason": reason,
    }
    if score is not None:
        record["score"] = round(score, 6)
    return record


def _count_hits(contamination: dict[str, dict[str, list[dict[str, Any]]]], splits: Iterable[str]) -> int:
    split_set = set(splits)
    return sum(
        len(hits)
        for split_hits in contamination.values()
        for split, hits in split_hits.items()
        if split in split_set
    )


def _render_leakage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SkillX AppWorld Leakage Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Risk level: `{report['risk_level']}`",
        f"- Skill count: {report['skill_count']}",
        f"- dev/test contamination count: {report['dev_test_contamination_count']}",
        f"- train overlap count: {report['train_overlap_count']}",
        "",
        "## Checked Splits",
        "",
    ]
    for split, count in report["checked_splits"].items():
        lines.append(f"- {split}: {count} tasks")
    lines.extend(["", "## Contamination Summary", ""])
    contamination = report["contamination"]
    for check_name, split_hits in contamination.items():
        lines.append(f"### {check_name}")
        any_hit = False
        for split, hits in split_hits.items():
            if not hits:
                continue
            any_hit = True
            lines.append(f"- {split}: {len(hits)} hits")
            for hit in hits[:5]:
                lines.append(
                    f"  - skill={hit.get('skill_id')} task={hit.get('task_id')} reason={hit.get('reason')}"
                )
        if not any_hit:
            lines.append("- none")
        lines.append("")
    lines.extend(
        [
            "## Recommendation",
            "",
            report["recommended_next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def _make_report(skillx_root: Path, skills: list[dict[str, Any]], min_skill_count: int) -> dict[str, Any]:
    field_coverage = {
        field: _coverage(skills, field)
        for field in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS)
    }
    serialized_lengths = [
        len(
            "\n".join(
                [
                    str(skill.get("name", "")),
                    str(skill.get("description", "")),
                    json.dumps(skill.get("input_schema", {}), ensure_ascii=False, sort_keys=True),
                    json.dumps(skill.get("output_schema", {}), ensure_ascii=False, sort_keys=True),
                    str(skill.get("executor_desc", "")),
                    json.dumps(skill.get("failure_modes", []), ensure_ascii=False, sort_keys=True),
                    str(skill.get("body", "")),
                ]
            ).split()
        )
        for skill in skills
    ]
    enough = (
        len(skills) >= min_skill_count
        and field_coverage["name"] >= 0.95
        and field_coverage["description"] >= 0.8
        and field_coverage["body"] >= 0.8
    )
    status = "ok" if enough else "fallback_needed"
    if not skills:
        status = "blocked"
    report = {
        "status": status,
        "skillx_root": str(skillx_root),
        "source_dataset": "SkillX-AppWorld",
        "skill_count": len(skills),
        "min_skill_count": min_skill_count,
        "source_files_count": len({skill.get("source_path") for skill in skills}),
        "field_coverage": field_coverage,
        "serialized_token_length": _length_stats(serialized_lengths),
        "enough_for_clstr_schema": enough,
        "sample_skills": [
            {
                "skill_id": skill.get("skill_id"),
                "name": skill.get("name"),
                "description": skill.get("description"),
                "source_path": skill.get("source_path"),
                "body_tokens": len(str(skill.get("body", "")).split()),
            }
            for skill in skills[:20]
        ],
    }
    if not skills:
        report["blocker"] = "no_appworld_skill_files_found"
        report["recommended_next_step"] = "Download or locate SkillX, then rerun scripts/audit_skillx_appworld.py."
    elif not enough:
        report["recommended_next_step"] = "Inspect SkillX schema manually or use synthetic AppWorld API skill bodies as fallback."
    else:
        report["recommended_next_step"] = "Build data/appworld_skill_pool/skill_pool.jsonl and proceed to AppWorld harness smoke."
    return report


def audit_skillx_appworld(
    skillx_root: str | Path,
    output_dir: str | Path | None = None,
    min_skill_count: int = 100,
) -> dict[str, Any]:
    root = Path(skillx_root)
    if not root.exists():
        report = {
            "status": "blocked",
            "blocker": "skillx_root_missing",
            "skillx_root": str(root),
            "skill_count": 0,
            "min_skill_count": min_skill_count,
            "enough_for_clstr_schema": False,
            "recommended_next_step": "Prepare SkillX on the login node, then rerun scripts/audit_skillx_appworld.py.",
        }
        if output_dir is not None:
            _write_json(Path(output_dir) / "blocker_report.json", report)
        return report
    skills = load_skillx_appworld_skills(root)
    report = _make_report(root, skills, min_skill_count)
    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / ("report.json" if report["status"] == "ok" else "blocker_report.json"), report)
        if skills:
            _write_json(out / "schema.json", {"status": report["status"], "sample_skills": report["sample_skills"]})
    return report


def audit_skillx_appworld_leakage(
    skill_pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    routing_dir: str | Path = "data/appworld_routing",
    output_dir: str | Path | None = None,
    skillx_root: str | Path | None = None,
    near_duplicate_threshold: float = 0.8,
    min_instruction_chars: int = 24,
) -> dict[str, Any]:
    pool_path = Path(skill_pool_path)
    routing_path = Path(routing_dir)
    if not pool_path.exists():
        report = {
            "status": "blocked",
            "blocker": "skill_pool_missing",
            "skill_pool_path": str(pool_path),
            "routing_dir": str(routing_path),
            "skill_count": 0,
            "risk_level": "unknown",
            "dev_test_contamination_count": 0,
            "train_overlap_count": 0,
            "recommended_next_step": "Build data/appworld_skill_pool/skill_pool.jsonl before running leakage audit.",
        }
        if output_dir is not None:
            out = Path(output_dir)
            _write_json(out / "report.json", report)
            _write_text(out / "report.md", _render_leakage_markdown({**report, "checked_splits": {}, "contamination": {}}))
        return report
    if not routing_path.exists():
        report = {
            "status": "blocked",
            "blocker": "routing_dir_missing",
            "skill_pool_path": str(pool_path),
            "routing_dir": str(routing_path),
            "skill_count": 0,
            "risk_level": "unknown",
            "dev_test_contamination_count": 0,
            "train_overlap_count": 0,
            "recommended_next_step": "Build data/appworld_routing split files before running leakage audit.",
        }
        if output_dir is not None:
            out = Path(output_dir)
            _write_json(out / "report.json", report)
            _write_text(out / "report.md", _render_leakage_markdown({**report, "checked_splits": {}, "contamination": {}}))
        return report

    root = Path(skillx_root) if skillx_root is not None else None
    skills = _read_jsonl(pool_path)
    tasks_by_split = _load_routing_tasks(routing_path, APPWORLD_ROUTING_SPLITS)
    source_plan_tasks = _load_skillx_plan_user_tasks(root)
    contamination = _empty_contamination()

    for skill in skills:
        raw_skill_text = _skill_match_text(skill)
        normalized_skill_text = _normalize_for_match(raw_skill_text)
        skill_tokens = _match_tokens(raw_skill_text)
        for split, tasks in tasks_by_split.items():
            for task in tasks:
                task_id = str(task.get("task_id") or "").strip()
                instruction = str(task.get("instruction_text") or "").strip()
                normalized_instruction = _normalize_for_match(instruction)
                if task_id and task_id.lower() in normalized_skill_text:
                    contamination["task_id_hits"][split].append(_hit_record(skill, task, "task_id_substring"))
                if (
                    normalized_instruction
                    and len(normalized_instruction) >= min_instruction_chars
                    and normalized_instruction in normalized_skill_text
                ):
                    contamination["instruction_hits"][split].append(
                        _hit_record(skill, task, "instruction_substring")
                    )
                    continue
                instruction_tokens = _match_tokens(instruction)
                score = _jaccard(skill_tokens, instruction_tokens)
                if instruction_tokens and score >= near_duplicate_threshold:
                    contamination["near_duplicate_instruction_hits"][split].append(
                        _hit_record(skill, task, "instruction_token_jaccard", score=score)
                    )

    normalized_plan_tasks = {
        _normalize_for_match(instruction): {"instruction_text": instruction, "source_paths": paths}
        for instruction, paths in source_plan_tasks.items()
    }
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            normalized_instruction = _normalize_for_match(str(task.get("instruction_text") or ""))
            plan_match = normalized_plan_tasks.get(normalized_instruction)
            if plan_match is None:
                continue
            contamination["source_plan_instruction_overlaps"][split].append(
                {
                    "skill_id": None,
                    "skill_name": "SkillX plan.json",
                    "task_id": task.get("task_id"),
                    "instruction_text": task.get("instruction_text"),
                    "reason": "skillx_plan_user_task_overlap",
                    "source_paths": plan_match["source_paths"],
                }
            )

    dev_test_contamination_count = _count_hits(contamination, EVAL_SPLITS)
    train_overlap_count = _count_hits(contamination, ("train",))
    status = "failed" if dev_test_contamination_count else "ok"
    risk_level = "high" if dev_test_contamination_count else "low"
    report = {
        "status": status,
        "risk_level": risk_level,
        "skill_pool_path": str(pool_path),
        "skillx_root": str(root) if root is not None else None,
        "routing_dir": str(routing_path),
        "skill_count": len(skills),
        "source_plan_task_count": len(source_plan_tasks),
        "checked_splits": {split: len(tasks) for split, tasks in tasks_by_split.items()},
        "near_duplicate_threshold": near_duplicate_threshold,
        "min_instruction_chars": min_instruction_chars,
        "dev_test_contamination_count": dev_test_contamination_count,
        "train_overlap_count": train_overlap_count,
        "contamination": contamination,
        "recommended_next_step": (
            "Do not use the current SkillX AppWorld skill body for clean dev/test claims; rebuild a train-only "
            "or public-doc-only skill pool and rerun this audit."
            if dev_test_contamination_count
            else "No dev/test contamination was detected by string-level checks; keep this report with experiment artifacts."
        ),
        "caveat": (
            "This is a deterministic string-level audit for task ids, full instruction substrings, and high token "
            "Jaccard near-duplicates. It cannot prove absence of semantic leakage."
        ),
    }
    if output_dir is not None:
        out = Path(output_dir)
        _write_json(out / "report.json", report)
        _write_text(out / "report.md", _render_leakage_markdown(report))
    return report


def build_appworld_skill_pool(
    skillx_root: str | Path,
    output_dir: str | Path,
    min_skill_count: int = 100,
) -> dict[str, Any]:
    root = Path(skillx_root)
    out = Path(output_dir)
    report = audit_skillx_appworld(root, min_skill_count=min_skill_count)
    if report["status"] != "ok":
        manifest = {
            "status": "blocked",
            "reason": "skillx_audit_not_ok",
            "audit_status": report["status"],
            "skillx_root": str(root),
            "skill_count": report.get("skill_count", 0),
        }
        _write_json(out / "manifest.json", manifest)
        return manifest
    skills = load_skillx_appworld_skills(root)
    count = _write_jsonl(out / "skill_pool.jsonl", skills)
    manifest = {
        "status": "ok",
        "source_dataset": "SkillX-AppWorld",
        "skillx_root": str(root),
        "skill_count": count,
        "files": {
            "skill_pool": str(out / "skill_pool.jsonl"),
            "manifest": str(out / "manifest.json"),
        },
        "audit": report,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def load_appworld_skill_pool(pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl") -> list[dict[str, Any]]:
    return _read_jsonl(Path(pool_path))


def summarize_appworld_skill_pool(
    pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(pool_path)
    if not path.exists():
        report = {
            "status": "blocked",
            "blocker": "skill_pool_missing",
            "pool_path": str(path),
            "skill_count": 0,
        }
        if output_path is not None:
            _write_json(Path(output_path), report)
        return report

    skills = load_appworld_skill_pool(path)
    field_coverage = {
        field: _coverage(skills, field)
        for field in ("skill_id", "name", "description", "body", "executor_desc", "failure_modes")
    }
    body_lengths = [len(str(skill.get("body", "")).split()) for skill in skills]
    report = {
        "status": "ok" if skills else "blocked",
        "pool_path": str(path),
        "skill_count": len(skills),
        "field_coverage": field_coverage,
        "body_token_length": _length_stats(body_lengths),
        "executor_desc_coverage": field_coverage["executor_desc"],
        "sample_skills": [
            {
                "skill_id": skill.get("skill_id"),
                "name": skill.get("name"),
                "description": skill.get("description"),
                "executor_desc": skill.get("executor_desc"),
                "body_tokens": len(str(skill.get("body", "")).split()),
            }
            for skill in skills[:20]
        ],
    }
    if not skills:
        report["blocker"] = "skill_pool_empty"
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def serialize_appworld_skill_for_embedding(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Skill Name: {skill.get('name', '')}",
            f"Description: {skill.get('description', '')}",
            f"Executor: {skill.get('executor_desc', '')}",
            f"Input Schema: {json.dumps(skill.get('input_schema', {}), ensure_ascii=False, sort_keys=True)}",
            f"Output Schema: {json.dumps(skill.get('output_schema', {}), ensure_ascii=False, sort_keys=True)}",
            f"Failure Modes: {json.dumps(skill.get('failure_modes', []), ensure_ascii=False, sort_keys=True)}",
            "Body:",
            str(skill.get("body", "")),
        ]
    ).strip()


def build_appworld_embedding_inputs(
    pool_path: str | Path = "data/appworld_skill_pool/skill_pool.jsonl",
    output_path: str | Path = "data/appworld_skill_pool/embedding_inputs.jsonl",
    manifest_path: str | Path = "data/appworld_skill_pool/embedding_manifest.json",
) -> dict[str, Any]:
    skills = load_appworld_skill_pool(pool_path)
    rows = [
        {
            "skill_id": skill.get("skill_id"),
            "name": skill.get("name"),
            "embedding_text": serialize_appworld_skill_for_embedding(skill),
        }
        for skill in skills
    ]
    count = _write_jsonl(Path(output_path), rows)
    manifest = {
        "status": "ok" if count == len(skills) and count > 0 else "blocked",
        "source_pool": str(pool_path),
        "skill_count": count,
        "files": {
            "embedding_inputs": str(output_path),
            "manifest": str(manifest_path),
        },
        "schema": {
            "required_fields": ["skill_id", "name", "embedding_text"],
            "serializer": "Skill Name + Description + Executor + Schemas + Failure Modes + Body",
        },
    }
    if not rows:
        manifest["blocker"] = "skill_pool_empty"
    _write_json(Path(manifest_path), manifest)
    return manifest
