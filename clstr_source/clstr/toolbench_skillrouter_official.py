from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_official_tier_pool(path: Path, rows: list[dict[str, Any]]) -> int:
    if path.exists() and path.is_file():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return _write_jsonl(path / "skills.jsonl", rows)


def _stable_score(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _split_query_ids(query_ids: list[str], eval_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not query_ids:
        return set(), set()
    eval_fraction = min(max(float(eval_fraction), 0.0), 1.0)
    sorted_ids = sorted(query_ids, key=lambda item: (_stable_score(item, seed), item))
    eval_count = max(1, int(round(len(sorted_ids) * eval_fraction))) if len(sorted_ids) > 1 else 1
    eval_count = min(eval_count, len(sorted_ids) - 1) if len(sorted_ids) > 1 else 1
    eval_ids = set(sorted_ids[:eval_count])
    train_ids = set(sorted_ids[eval_count:])
    return train_ids, eval_ids


def _split_group_ids(group_ids: list[str], eval_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    return _split_query_ids(group_ids, eval_fraction=eval_fraction, seed=seed)


def _skillrouter_skill_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": str(row.get("skill_id") or row.get("id") or ""),
        "name": str(row.get("name") or row.get("skill_id") or row.get("id") or ""),
        "description": str(row.get("description") or row.get("executor_desc") or ""),
        "body": str(row.get("body") or row.get("skill_md") or row.get("executor_desc") or ""),
    }


def _write_official_eval_core_split(
    *,
    output_dir: Path,
    split_name: str,
    query_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, list[str]],
    skill_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_dir = output_dir / split_name
    tasks: list[dict[str, Any]] = []
    relevance: dict[str, dict[str, Any]] = {}
    for row in query_rows:
        query_id = str(row["query_id"])
        positives = sorted(set(qrels_by_query.get(query_id, [])))
        if not positives:
            continue
        tasks.append(
            {
                "task_id": query_id,
                "instruction_text": str(row.get("query_text") or row.get("instruction_text") or ""),
                "source": "toolbench_g3",
            }
        )
        relevance[query_id] = {
            "task_type": "toolbench_g3",
            "gt_skill_ids": positives,
            "core_gt_ids": positives,
            "relevance": {skill_id: 1 for skill_id in positives},
        }
    official_skills = [_skillrouter_skill_row(row) for row in skill_rows]
    official_skills = [row for row in official_skills if row["skill_id"]]
    task_count = _write_jsonl(split_dir / "tasks.jsonl", tasks)
    skill_count = _write_official_tier_pool(split_dir / "easy", official_skills)
    _write_official_tier_pool(split_dir / "hard", official_skills)
    _write_json(split_dir / "relevance.json", relevance)
    _write_json(
        split_dir / "manifest.json",
        {
            "status": "ok",
            "format": "skillrouter_eval_core_compatible",
            "split": split_name,
            "task_count": task_count,
            "skill_count": skill_count,
            "positive_qrels": sum(len(row["gt_skill_ids"]) for row in relevance.values()),
            "tier_files": ["easy/skills.jsonl", "hard/skills.jsonl"],
            "note": "The easy and hard directories intentionally contain the same ToolBench-G3 skill pool so the official SkillRouter scripts can run unchanged.",
        },
    )
    return {
        "task_count": task_count,
        "skill_count": skill_count,
        "positive_qrels": sum(len(row["gt_skill_ids"]) for row in relevance.values()),
    }


def _trajectory_task_id(row: dict[str, Any], fallback_index: int) -> str:
    explicit = row.get("task_id")
    if explicit:
        return str(explicit)
    trajectory_id = str(row.get("trajectory_id") or row.get("query_id") or "trajectory")
    step_index = row.get("step_index")
    if step_index is not None:
        return f"{trajectory_id}::{step_index}"
    return f"{trajectory_id}::{fallback_index}"


def _trajectory_group_id(row: dict[str, Any], fallback_index: int) -> str:
    return str(row.get("trajectory_id") or row.get("query_id") or row.get("task_id") or f"group-{fallback_index}")


def _is_toolbench_stage2_supervised_row(row: dict[str, Any], skill_ids: set[str]) -> bool:
    if str(row.get("benchmark") or "") != "toolbench_g3":
        return False
    next_skill_id = str(row.get("next_skill_id") or "")
    if not next_skill_id or next_skill_id not in skill_ids:
        return False
    loss_mask = row.get("loss_mask") or {}
    if isinstance(loss_mask, dict) and loss_mask.get("L_trans_skill_ce") is False:
        return False
    return bool(str(row.get("state_text") or "").strip())


def _toolbench_action_alias_map(
    skill_rows: list[dict[str, Any]],
    verification_skill_rows: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    canonical_by_raw_id: dict[str, str] = {}
    for row in skill_rows:
        skill_id = str(row.get("skill_id") or row.get("id") or "").strip()
        if not skill_id.startswith("toolbench-g3/"):
            continue
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        partial = (
            provenance.get("partial_dedup")
            if isinstance(provenance.get("partial_dedup"), dict)
            else {}
        )
        raw_ids = [skill_id]
        raw_ids.extend(str(value).strip() for value in row.get("alias_skill_ids") or [])
        raw_ids.extend(str(value).strip() for value in partial.get("raw_skill_ids") or [])
        for raw_id in raw_ids:
            if raw_id:
                canonical_by_raw_id[raw_id] = skill_id

    aliases: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}
    alias_source = verification_skill_rows if verification_skill_rows is not None else skill_rows
    for row in alias_source:
        raw_skill_id = str(row.get("skill_id") or row.get("id") or "").strip()
        skill_id = canonical_by_raw_id.get(raw_skill_id)
        if skill_id is None:
            continue
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        values = [str(provenance.get("action_alias") or "").strip()]
        values.extend(
            str(value).strip()
            for value in provenance.get("action_aliases") or []
        )
        for value in values:
            alias = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            if not alias:
                continue
            previous = aliases.get(alias)
            if previous is not None and previous != skill_id:
                collisions.setdefault(alias, {previous}).add(skill_id)
                continue
            aliases[alias] = skill_id
    if collisions:
        examples = {
            alias: sorted(skill_ids)
            for alias, skill_ids in sorted(collisions.items())[:8]
        }
        raise ValueError(f"ToolBench action aliases are ambiguous: {examples}")
    return aliases


def _toolbench_answer_events(answer_path: Path) -> set[tuple[str, str]]:
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    tree_root = (
        ((answer.get("tree") or {}).get("tree") or {})
        if isinstance(answer, dict)
        else {}
    )
    if not isinstance(tree_root, dict):
        raise ValueError(f"ToolBench answer has no executable tree: {answer_path}")
    events: set[tuple[str, str]] = set()
    stack = [tree_root]
    while stack:
        node = stack.pop()
        children = [
            child
            for child in (node.get("children") or [])
            if isinstance(child, dict)
        ]
        if str(node.get("node_type") or "") == "Action":
            action_name = str(node.get("description") or "")
            for child in children:
                if str(child.get("node_type") or "") != "Action Input":
                    continue
                action_text = f"{action_name}: {str(child.get('description') or '')}"
                observation = str(child.get("observation") or "")
                events.add((action_text, observation))
        stack.extend(children)
    if not events:
        raise ValueError(f"ToolBench answer has no executed action/input events: {answer_path}")
    return events


def verify_toolbench_executed_results(
    rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    verification_skill_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind legacy ToolBench result text to an executed source-tree event.

    The legacy official trajectory artifact persisted the exact action, result,
    and answer-tree path, but predates the explicit result provenance fields.
    This verifier never infers provenance from result text alone: the action
    alias must map to the row's executed skill and the exact action/result pair
    must occur in the declared ToolBench answer tree.
    """

    alias_to_skill = _toolbench_action_alias_map(
        skill_rows,
        verification_skill_rows,
    )
    answer_cache: dict[Path, set[tuple[str, str]]] = {}
    prepared: list[dict[str, Any]] = []
    aligned_count = 0
    nonempty_result_count = 0
    empty_result_count = 0
    answer_paths: set[str] = set()
    for row in rows:
        if str(row.get("benchmark") or "") != "toolbench_g3":
            prepared.append(dict(row))
            continue
        action_text = str(row.get("action_text") or "").strip()
        action_name, separator, _arguments = action_text.partition(":")
        skill_id = str(row.get("skill_id") or "").strip()
        result_text = str(row.get("next_observation_text") or "")
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        answer_path_text = str(provenance.get("answer_path") or "").strip()
        if not separator or not action_name.strip() or not skill_id:
            raise ValueError("ToolBench result verification requires action_text and skill_id")
        if str(provenance.get("source_dataset") or "") != "ToolBench-G3":
            raise ValueError("ToolBench result verification requires ToolBench-G3 provenance")
        action_alias = re.sub(
            r"[^a-z0-9]+",
            "_",
            action_name.strip().lower(),
        ).strip("_")
        mapped_skill = alias_to_skill.get(action_alias)
        if mapped_skill != skill_id:
            raise ValueError(
                "ToolBench executed action alias does not match skill_id: "
                f"action={action_name.strip()} mapped={mapped_skill} row={skill_id}"
            )
        if not answer_path_text:
            raise ValueError("ToolBench result verification requires provenance.answer_path")
        answer_path = Path(answer_path_text).resolve()
        if not answer_path.is_file():
            raise FileNotFoundError(f"ToolBench answer tree is missing: {answer_path}")
        events = answer_cache.get(answer_path)
        if events is None:
            events = _toolbench_answer_events(answer_path)
            answer_cache[answer_path] = events
        if (action_text, result_text) not in events:
            raise ValueError(
                "ToolBench action/result pair is absent from its declared answer tree: "
                f"task={row.get('task_id')} answer={answer_path}"
            )
        event_payload = {
            "answer_path": str(answer_path),
            "action_text": action_text,
            "result_text": result_text,
            "skill_id": skill_id,
        }
        event_id = hashlib.sha256(
            json.dumps(
                event_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        copied = dict(row)
        copied["observation_source"] = "executed_trace:toolbench_g3"
        copied["actual_result_text"] = result_text
        copied["actual_result_executed"] = bool(result_text)
        copied["actual_result_skill_id"] = skill_id
        copied["actual_result_event_id"] = event_id
        copied["result_event_skill_id"] = skill_id
        prepared.append(copied)
        aligned_count += 1
        nonempty_result_count += int(bool(result_text))
        empty_result_count += int(not result_text)
        answer_paths.add(str(answer_path))
    return prepared, {
        "protocol": "toolbench_exact_answer_tree_action_result_alignment_v1",
        "status": "ok",
        "aligned_row_count": aligned_count,
        "nonempty_result_row_count": nonempty_result_count,
        "empty_result_row_count": empty_result_count,
        "answer_file_count": len(answer_paths),
        "action_alias_count": len(alias_to_skill),
        "verification_skill_count": len(
            verification_skill_rows
            if verification_skill_rows is not None
            else skill_rows
        ),
    }


def export_verified_toolbench_existing_eval(
    *,
    source_eval_trajectories_path: str | Path,
    skills_path: str | Path,
    verification_skills_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Annotate one existing official split without changing row membership."""

    source_path = Path(source_eval_trajectories_path)
    skills_path = Path(skills_path)
    verification_skills_path = Path(verification_skills_path)
    output = Path(output_dir)
    rows = _read_jsonl(source_path)
    skills = _read_jsonl(skills_path)
    verification_skills = _read_jsonl(verification_skills_path)
    verified, alignment = verify_toolbench_executed_results(
        rows,
        skills,
        verification_skills,
    )
    source_identities = [
        (str(row.get("task_id") or ""), str(row.get("trajectory_id") or ""))
        for row in rows
    ]
    verified_identities = [
        (str(row.get("task_id") or ""), str(row.get("trajectory_id") or ""))
        for row in verified
    ]
    if verified_identities != source_identities:
        raise ValueError("verified ToolBench copy changed official row membership or order")
    output_path = output / "eval_trajectories.jsonl"
    row_count = _write_jsonl(output_path, verified)
    report = {
        "status": "ok",
        "protocol": "toolbench_existing_official_split_verified_copy_v1",
        "source_eval_trajectories_path": str(source_path.resolve()),
        "source_eval_trajectories_sha256": _file_sha256(source_path),
        "skills_path": str(skills_path.resolve()),
        "skills_sha256": _file_sha256(skills_path),
        "verification_skills_path": str(verification_skills_path.resolve()),
        "verification_skills_sha256": _file_sha256(verification_skills_path),
        "eval_trajectories_path": str(output_path.resolve()),
        "eval_trajectories_sha256": _file_sha256(output_path),
        "row_count": int(row_count),
        "trajectory_count": len(
            {str(row.get("trajectory_id") or "") for row in verified}
        ),
        "row_membership_and_order_preserved": True,
        "executed_result_alignment": alignment,
    }
    _write_json(output / "manifest.json", report)
    return report


def _trajectory_task_row(row: dict[str, Any], task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "instruction_text": str(row.get("state_text") or ""),
        "source": "toolbench_g3_trajectory",
    }


def _write_official_trajectory_split(
    *,
    output_dir: Path,
    split_name: str,
    rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_dir = output_dir / split_name
    tasks: list[dict[str, Any]] = []
    relevance: dict[str, dict[str, Any]] = {}
    trajectory_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for idx, row in enumerate(rows):
        task_id = _trajectory_task_id(row, idx)
        if task_id in seen_task_ids:
            task_id = f"{task_id}::dup{idx}"
        seen_task_ids.add(task_id)
        next_skill_id = str(row.get("next_skill_id") or "")
        out_row = dict(row)
        out_row["task_id"] = task_id
        out_row["official_eval_split"] = split_name
        trajectory_rows.append(out_row)
        tasks.append(_trajectory_task_row(out_row, task_id))
        relevance[task_id] = {
            "task_type": "toolbench_g3_trajectory",
            "gt_skill_ids": [next_skill_id],
            "core_gt_ids": [next_skill_id],
            "relevance": {next_skill_id: 1},
        }
    official_skills = [_skillrouter_skill_row(row) for row in skill_rows]
    official_skills = [row for row in official_skills if row["skill_id"]]
    task_count = _write_jsonl(split_dir / "tasks.jsonl", tasks)
    skill_count = _write_official_tier_pool(split_dir / "easy", official_skills)
    _write_official_tier_pool(split_dir / "hard", official_skills)
    _write_json(split_dir / "relevance.json", relevance)
    _write_jsonl(output_dir / f"{split_name}_trajectories.jsonl", trajectory_rows)
    _write_json(
        split_dir / "manifest.json",
        {
            "status": "ok",
            "format": "skillrouter_eval_core_compatible",
            "task_definition": "toolbench_g3_trajectory_state_to_next_skill",
            "split": split_name,
            "task_count": task_count,
            "skill_count": skill_count,
            "positive_qrels": sum(len(row["gt_skill_ids"]) for row in relevance.values()),
            "tier_files": ["easy/skills.jsonl", "hard/skills.jsonl"],
        },
    )
    return {
        "task_count": task_count,
        "skill_count": skill_count,
        "positive_qrels": sum(len(row["gt_skill_ids"]) for row in relevance.values()),
    }


def export_toolbench_g3_skillrouter_eval_core(
    *,
    queries_path: str | Path,
    qrels_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    eval_fraction: float = 0.2,
    seed: int = 13,
    max_train_queries: int | None = None,
    max_eval_queries: int | None = None,
    max_skills: int | None = None,
) -> dict[str, Any]:
    queries = _read_jsonl(queries_path)
    qrels = _read_jsonl(qrels_path)
    skills = _read_jsonl(skills_path)
    if max_skills is not None:
        skills = skills[: max(0, int(max_skills))]
    skill_ids = {str(row.get("skill_id") or row.get("id") or "") for row in skills}
    qrels_by_query: dict[str, list[str]] = {}
    skipped_qrels_missing_skill = 0
    for row in qrels:
        query_id = str(row.get("query_id") or "")
        skill_id = str(row.get("skill_id") or "")
        if not query_id or not skill_id:
            continue
        if skill_id not in skill_ids:
            skipped_qrels_missing_skill += 1
            continue
        qrels_by_query.setdefault(query_id, []).append(skill_id)
    usable_queries = [row for row in queries if str(row.get("query_id") or "") in qrels_by_query]
    query_ids = [str(row["query_id"]) for row in usable_queries]
    train_ids, eval_ids = _split_query_ids(query_ids, eval_fraction=eval_fraction, seed=seed)
    train_rows = [row for row in usable_queries if str(row["query_id"]) in train_ids]
    eval_rows = [row for row in usable_queries if str(row["query_id"]) in eval_ids]
    train_rows = sorted(train_rows, key=lambda row: str(row["query_id"]))
    eval_rows = sorted(eval_rows, key=lambda row: str(row["query_id"]))
    if max_train_queries is not None:
        rng = random.Random(seed)
        rng.shuffle(train_rows)
        train_rows = sorted(train_rows[: max(0, int(max_train_queries))], key=lambda row: str(row["query_id"]))
    if max_eval_queries is not None:
        rng = random.Random(seed + 1)
        rng.shuffle(eval_rows)
        eval_rows = sorted(eval_rows[: max(0, int(max_eval_queries))], key=lambda row: str(row["query_id"]))

    output = Path(output_dir)
    train_report = _write_official_eval_core_split(
        output_dir=output,
        split_name="train",
        query_rows=train_rows,
        qrels_by_query=qrels_by_query,
        skill_rows=skills,
    )
    eval_report = _write_official_eval_core_split(
        output_dir=output,
        split_name="eval",
        query_rows=eval_rows,
        qrels_by_query=qrels_by_query,
        skill_rows=skills,
    )
    report = {
        "status": "ok",
        "format": "skillrouter_eval_core_compatible",
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "skills_path": str(skills_path),
        "output_dir": str(output),
        "seed": int(seed),
        "eval_fraction": float(eval_fraction),
        "source_query_count": len(queries),
        "usable_query_count": len(usable_queries),
        "train_query_count": train_report["task_count"],
        "eval_query_count": eval_report["task_count"],
        "skill_count": train_report["skill_count"],
        "train_positive_qrels": train_report["positive_qrels"],
        "eval_positive_qrels": eval_report["positive_qrels"],
        "skipped_qrels_missing_skill": skipped_qrels_missing_skill,
        "official_repo_compatibility": {
            "entrypoint": "python -m src.run_open_model_eval --data_root <output_dir>/eval --tiers easy",
            "prediction_format": "official SkillRouter JSON map task_id -> ranked skill_id list",
        },
    }
    _write_json(output / "manifest.json", report)
    return report


def export_toolbench_g3_trajectory_skillrouter_eval_core(
    *,
    trajectories_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path,
    eval_fraction: float = 0.2,
    seed: int = 13,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
    max_skills: int | None = None,
    verify_executed_result_provenance: bool = False,
    verification_skills_path: str | Path | None = None,
) -> dict[str, Any]:
    trajectories = _read_jsonl(trajectories_path)
    skills = _read_jsonl(skills_path)
    if max_skills is not None:
        skills = skills[: max(0, int(max_skills))]
    skill_ids = {str(row.get("skill_id") or row.get("id") or "") for row in skills}
    usable_rows = [
        row
        for row in trajectories
        if _is_toolbench_stage2_supervised_row(row, skill_ids)
    ]
    result_alignment_report: dict[str, Any] = {
        "status": "disabled",
        "protocol": None,
    }
    if verify_executed_result_provenance:
        verification_skills = (
            _read_jsonl(verification_skills_path)
            if verification_skills_path is not None
            else skills
        )
        usable_rows, result_alignment_report = verify_toolbench_executed_results(
            usable_rows,
            skills,
            verification_skills,
        )
    groups = sorted({_trajectory_group_id(row, idx) for idx, row in enumerate(usable_rows)})
    train_groups, eval_groups = _split_group_ids(groups, eval_fraction=eval_fraction, seed=seed)
    train_rows = [
        row
        for idx, row in enumerate(usable_rows)
        if _trajectory_group_id(row, idx) in train_groups
    ]
    eval_rows = [
        row
        for idx, row in enumerate(usable_rows)
        if _trajectory_group_id(row, idx) in eval_groups
    ]
    train_rows = sorted(
        train_rows,
        key=lambda row: (_trajectory_group_id(row, 0), int(row.get("step_index") or 0), str(row.get("task_id") or "")),
    )
    eval_rows = sorted(
        eval_rows,
        key=lambda row: (_trajectory_group_id(row, 0), int(row.get("step_index") or 0), str(row.get("task_id") or "")),
    )
    if max_train_rows is not None:
        rng = random.Random(seed)
        rng.shuffle(train_rows)
        train_rows = sorted(
            train_rows[: max(0, int(max_train_rows))],
            key=lambda row: (_trajectory_group_id(row, 0), int(row.get("step_index") or 0), str(row.get("task_id") or "")),
        )
    if max_eval_rows is not None:
        rng = random.Random(seed + 1)
        rng.shuffle(eval_rows)
        eval_rows = sorted(
            eval_rows[: max(0, int(max_eval_rows))],
            key=lambda row: (_trajectory_group_id(row, 0), int(row.get("step_index") or 0), str(row.get("task_id") or "")),
        )

    output = Path(output_dir)
    train_report = _write_official_trajectory_split(
        output_dir=output,
        split_name="train",
        rows=train_rows,
        skill_rows=skills,
    )
    eval_report = _write_official_trajectory_split(
        output_dir=output,
        split_name="eval",
        rows=eval_rows,
        skill_rows=skills,
    )
    report = {
        "status": "ok",
        "format": "skillrouter_eval_core_compatible",
        "task_definition": "toolbench_g3_trajectory_state_to_next_skill",
        "trajectories_path": str(trajectories_path),
        "skills_path": str(skills_path),
        "output_dir": str(output),
        "seed": int(seed),
        "eval_fraction": float(eval_fraction),
        "source_row_count": len(trajectories),
        "usable_row_count": len(usable_rows),
        "trajectory_group_count": len(groups),
        "train_group_count": len(train_groups),
        "eval_group_count": len(eval_groups),
        "train_query_count": train_report["task_count"],
        "eval_query_count": eval_report["task_count"],
        "skill_count": train_report["skill_count"],
        "executed_result_alignment": result_alignment_report,
        "train_positive_qrels": train_report["positive_qrels"],
        "eval_positive_qrels": eval_report["positive_qrels"],
        "clstr_eval_trajectories_path": str(output / "eval_trajectories.jsonl"),
        "clstr_train_trajectories_path": str(output / "train_trajectories.jsonl"),
        "official_repo_compatibility": {
            "entrypoint": "python -m src.run_open_model_eval --data_root <output_dir>/eval --tiers easy",
            "prediction_format": "official SkillRouter JSON map task_id -> ranked skill_id list",
        },
    }
    _write_json(output / "manifest.json", report)
    return report
