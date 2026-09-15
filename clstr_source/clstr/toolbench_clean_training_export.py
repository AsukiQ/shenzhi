from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOOLBENCH_QUERY_ID_RE = re.compile(r"(?:toolbench-g3-)?(\d+)")
_TOOLBENCH_ANSWER_PATH_RE = re.compile(r"(?:^|/)(\d+)_")


@dataclass(frozen=True)
class ToolBenchEvalExclusion:
    trajectory_ids: frozenset[str]
    task_ids: frozenset[str]
    answer_paths: frozenset[str]
    query_ids: frozenset[str]
    signatures: frozenset[str]

    def report(self) -> dict[str, int]:
        return {
            "trajectory_ids": len(self.trajectory_ids),
            "task_ids": len(self.task_ids),
            "answer_paths": len(self.answer_paths),
            "query_ids": len(self.query_ids),
            "signatures": len(self.signatures),
        }


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalized_toolbench_query_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("toolbench-g3-"):
        head = text.split("::", 1)[0]
        return head
    match = _TOOLBENCH_QUERY_ID_RE.search(text)
    if not match:
        return ""
    return f"toolbench-g3-{match.group(1)}"


def _query_id_from_answer_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = _TOOLBENCH_ANSWER_PATH_RE.search(text)
    if not match:
        return ""
    return f"toolbench-g3-{match.group(1)}"


def _row_answer_path(row: dict[str, Any]) -> str:
    provenance = _provenance(row)
    return str(row.get("answer_path") or provenance.get("answer_path") or "").strip()


def _row_signature(row: dict[str, Any]) -> str:
    return "\0".join(
        [
            str(row.get("state_text") or ""),
            str(row.get("skill_id") or ""),
            str(row.get("next_skill_id") or ""),
        ]
    )


def _row_query_ids(row: dict[str, Any]) -> set[str]:
    provenance = _provenance(row)
    values = [
        row.get("query_id"),
        row.get("trajectory_id"),
        row.get("task_id"),
        provenance.get("query_id"),
        provenance.get("trajectory_id"),
        provenance.get("task_id"),
    ]
    answer_query_id = _query_id_from_answer_path(_row_answer_path(row))
    output = {answer_query_id} if answer_query_id else set()
    for value in values:
        query_id = _normalized_toolbench_query_id(value)
        if query_id:
            output.add(query_id)
    return output


def _is_toolbench_trajectory_row(row: dict[str, Any]) -> bool:
    return str(row.get("benchmark") or "") == "toolbench_g3"


def _is_toolbench_retrieval_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    provenance = _provenance(row)
    return source in {"toolbench_g3", "trajectory_derived_toolbench_g3"} or (
        source.startswith("trajectory_derived_") and str(provenance.get("benchmark") or "") == "toolbench_g3"
    )


def build_toolbench_eval_exclusion(eval_trajectories_path: str | Path) -> ToolBenchEvalExclusion:
    trajectory_ids: set[str] = set()
    task_ids: set[str] = set()
    answer_paths: set[str] = set()
    query_ids: set[str] = set()
    signatures: set[str] = set()

    for row in _iter_jsonl(eval_trajectories_path):
        if not _is_toolbench_trajectory_row(row):
            continue
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        answer_path = _row_answer_path(row)
        if trajectory_id:
            trajectory_ids.add(trajectory_id)
        if task_id:
            task_ids.add(task_id)
        if answer_path:
            answer_paths.add(answer_path)
        query_ids.update(_row_query_ids(row))
        signatures.add(_row_signature(row))

    return ToolBenchEvalExclusion(
        trajectory_ids=frozenset(trajectory_ids),
        task_ids=frozenset(task_ids),
        answer_paths=frozenset(answer_paths),
        query_ids=frozenset(query_ids),
        signatures=frozenset(signatures),
    )


def _trajectory_exclusion_reason(row: dict[str, Any], exclusion: ToolBenchEvalExclusion) -> str | None:
    if not _is_toolbench_trajectory_row(row):
        return None
    if str(row.get("trajectory_id") or "").strip() in exclusion.trajectory_ids:
        return "trajectory_id"
    if str(row.get("task_id") or "").strip() in exclusion.task_ids:
        return "task_id"
    if _row_answer_path(row) in exclusion.answer_paths:
        return "answer_path"
    if _row_query_ids(row) & set(exclusion.query_ids):
        return "query_id"
    if _row_signature(row) in exclusion.signatures:
        return "signature"
    return None


def _retrieval_exclusion_reason(row: dict[str, Any], exclusion: ToolBenchEvalExclusion) -> str | None:
    if not _is_toolbench_retrieval_row(row):
        return None
    provenance = _provenance(row)
    if str(provenance.get("trajectory_id") or "").strip() in exclusion.trajectory_ids:
        return "trajectory_id"
    if str(provenance.get("task_id") or "").strip() in exclusion.task_ids:
        return "task_id"
    if _row_answer_path(row) in exclusion.answer_paths:
        return "answer_path"
    if _row_query_ids(row) & set(exclusion.query_ids):
        return "query_id"
    return None


def _filter_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    exclusion_fn,
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    input_rows = 0
    for row in _iter_jsonl(input_path):
        input_rows += 1
        reason = exclusion_fn(row)
        if reason:
            reasons[reason] += 1
            continue
        kept.append(row)
    output_rows = _write_jsonl(output_path, kept)
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "removed_rows": input_rows - output_rows,
        "removed_by_reason": dict(sorted(reasons.items())),
    }


def export_toolbench_clean_unified_data(
    *,
    data_root: str | Path,
    eval_trajectories_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = ["trajectories.jsonl", "retrieval.jsonl", "skill_pool.jsonl"]
    missing = [name for name in required_files if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing unified data files under {data_root}: {missing}")

    exclusion = build_toolbench_eval_exclusion(eval_trajectories_path)
    shutil.copyfile(data_root / "skill_pool.jsonl", output_dir / "skill_pool.jsonl")

    trajectories_report = _filter_jsonl(
        input_path=data_root / "trajectories.jsonl",
        output_path=output_dir / "trajectories.jsonl",
        exclusion_fn=lambda row: _trajectory_exclusion_reason(row, exclusion),
    )
    retrieval_report = _filter_jsonl(
        input_path=data_root / "retrieval.jsonl",
        output_path=output_dir / "retrieval.jsonl",
        exclusion_fn=lambda row: _retrieval_exclusion_reason(row, exclusion),
    )

    report = {
        "status": "ok",
        "data_root": str(data_root),
        "eval_trajectories_path": str(eval_trajectories_path),
        "output_dir": str(output_dir),
        "exclusion": exclusion.report(),
        "trajectories": trajectories_report,
        "retrieval": retrieval_report,
        "skill_pool": {
            "copied": True,
            "path": str(output_dir / "skill_pool.jsonl"),
        },
        "note": (
            "ToolBench-G3 eval trajectory ids, task ids, answer paths, query ids, "
            "and exact state/skill/next signatures are excluded from CLSTR training sources."
        ),
    }
    _write_json(output_dir / "manifest.json", report)
    return report

