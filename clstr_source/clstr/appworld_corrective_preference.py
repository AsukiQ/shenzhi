from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ApiCorrectionDataset:
    samples: list[dict[str, Any]]
    stage0_retrieval_rows: list[dict[str, Any]]
    report: dict[str, Any]


_API_REF_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_IGNORED_API_REFS = {
    "apis.supervisor.complete_task",
}
_AUTH_LIKE_API_TERMS = ("access_token_from", "login", "authenticate")
_WRITE_API_PREFIXES = (
    "add_",
    "approve_",
    "batch_",
    "comment",
    "create_",
    "delete_",
    "like_",
    "move_",
    "remove_",
    "remind_",
    "rename_",
    "send_",
    "set_",
    "update_",
    "write_",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path)} line {line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _api_ref(app: str, api: str) -> str:
    return f"apis.{app}.{api}"


def _attr_api_ref(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    api_name = str(node.attr)
    value = node.value
    if not isinstance(value, ast.Attribute):
        return None
    app_name = str(value.attr)
    root = value.value
    if not isinstance(root, ast.Name) or root.id != "apis":
        return None
    return _api_ref(app_name, api_name)


def extract_appworld_api_refs(code: str, *, include_auth: bool = False) -> list[str]:
    refs: set[str] = set()
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        refs.update(_api_ref(app, api) for app, api in _API_REF_RE.findall(str(code or "")))
    else:
        for node in ast.walk(tree):
            ref = _attr_api_ref(node)
            if ref:
                refs.add(ref)
    filtered: list[str] = []
    for ref in sorted(refs):
        if ref in _IGNORED_API_REFS:
            continue
        api_name = ref.rsplit(".", 1)[-1]
        if not include_auth and any(term in api_name for term in _AUTH_LIKE_API_TERMS):
            continue
        filtered.append(ref)
    return filtered


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        str(skill.get(key) or "")
        for key in ("skill_id", "name", "description", "executor_desc", "body", "failure_modes")
    )


def _skill_api_refs(skill: dict[str, Any]) -> set[str]:
    return set(extract_appworld_api_refs(_skill_text(skill), include_auth=True))


def _api_app(ref: str) -> str:
    parts = str(ref).split(".")
    return parts[1] if len(parts) >= 3 else ""


def _api_name(ref: str) -> str:
    return str(ref).rsplit(".", 1)[-1]


def _is_write_api_ref(ref: str) -> bool:
    api = _api_name(ref)
    return api.startswith(_WRITE_API_PREFIXES)


def _select_api_diverse_matches(
    scored: list[dict[str, Any]],
    *,
    required_write_refs: set[str],
    max_targets: int,
) -> list[dict[str, Any]]:
    limit = max(1, int(max_targets))
    if not required_write_refs or limit <= 1:
        return scored[:limit]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for ref in sorted(required_write_refs):
        for item in scored:
            if str(item["skill_id"]) in selected_ids:
                continue
            if ref in set(item.get("write_api_overlap") or []):
                selected.append(item)
                selected_ids.add(str(item["skill_id"]))
                break
        if len(selected) >= limit:
            return selected
    for item in scored:
        if str(item["skill_id"]) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(str(item["skill_id"]))
        if len(selected) >= limit:
            break
    return selected


def match_solution_api_refs_to_candidate_skills(
    *,
    solution_api_refs: Iterable[str],
    candidate_skill_ids: list[str],
    skill_by_id: dict[str, dict[str, Any]],
    max_targets: int = 3,
) -> list[dict[str, Any]]:
    target_refs = [str(ref) for ref in solution_api_refs if str(ref)]
    target_set = set(target_refs)
    required_write_refs = {ref for ref in target_refs if _is_write_api_ref(ref)}
    target_apps = {_api_app(ref) for ref in target_refs}
    scored: list[dict[str, Any]] = []
    for local_idx, skill_id in enumerate(candidate_skill_ids):
        skill = skill_by_id.get(str(skill_id))
        if not skill:
            continue
        skill_refs = _skill_api_refs(skill)
        exact = sorted(target_set & skill_refs)
        write_exact = sorted(required_write_refs & skill_refs)
        if required_write_refs and not write_exact:
            continue
        app_overlap = sorted(target_apps & {_api_app(ref) for ref in skill_refs})
        if not exact:
            continue
        score = 10 * len(exact) + len(app_overlap)
        scored.append(
            {
                "skill_id": str(skill_id),
                "local_index": int(local_idx),
                "score": int(score),
                "exact_api_overlap": exact,
                "write_api_overlap": write_exact,
                "app_overlap": app_overlap,
            }
        )
    scored.sort(
        key=lambda item: (
            item["score"],
            len(item["write_api_overlap"]),
            len(item["exact_api_overlap"]),
            -item["local_index"],
        ),
        reverse=True,
    )
    return _select_api_diverse_matches(
        scored,
        required_write_refs=required_write_refs,
        max_targets=max_targets,
    )


def _stage0_correction_query_text(
    *,
    state_text: str,
    user_goal: str,
    solution_api_refs: list[str],
) -> str:
    parts = []
    if user_goal:
        parts.append(f"User goal: {user_goal}")
    if state_text:
        parts.append(f"State/context: {state_text}")
    if solution_api_refs:
        parts.append(f"Required API refs: {', '.join(sorted(solution_api_refs))}")
    parts.append("Retrieve the reusable skill that can execute the required next tool/API operation.")
    return "\n".join(parts)


def _stage0_retrieval_correction_rows(
    *,
    rollout: dict[str, Any],
    decision: dict[str, Any],
    task_id: str,
    solution_refs: list[str],
    candidate_skill_ids: list[str],
    skill_by_id: dict[str, dict[str, Any]],
    max_targets: int,
) -> list[dict[str, Any]]:
    pool_matches = match_solution_api_refs_to_candidate_skills(
        solution_api_refs=solution_refs,
        candidate_skill_ids=sorted(skill_by_id),
        skill_by_id=skill_by_id,
        max_targets=max_targets,
    )
    if not pool_matches:
        return []
    candidate_set = {str(skill_id) for skill_id in candidate_skill_ids}
    selected_ids, _selected_indices = _aligned_selected_indices(decision)
    negatives: list[str] = []
    for skill_id in [*selected_ids, *candidate_skill_ids]:
        skill_id = str(skill_id)
        if skill_id and skill_id not in negatives and skill_id not in {str(item["skill_id"]) for item in pool_matches}:
            negatives.append(skill_id)
    step_idx = int(decision.get("step_idx", 0) or 0)
    query_text = _stage0_correction_query_text(
        state_text=str(decision.get("state_text") or ""),
        user_goal=str(rollout.get("user_goal") or ""),
        solution_api_refs=solution_refs,
    )
    rows: list[dict[str, Any]] = []
    for match in pool_matches:
        positive_skill_id = str(match["skill_id"])
        if positive_skill_id in candidate_set:
            continue
        rows.append(
            {
                "source": "appworld_stage0_api_correction",
                "query_id": f"{task_id}::stage0_api_correction::{step_idx}",
                "query_text": query_text,
                "positive_skill_id": positive_skill_id,
                "negative_skill_ids": negatives[:50],
                "split": str(rollout.get("split") or rollout.get("data_split") or "diagnostic"),
                "provenance": {
                    "source_dataset": "appworld_current_route_rollout",
                    "task_id": task_id,
                    "query_id": str(rollout.get("query_id") or task_id),
                    "step_idx": step_idx,
                    "solution_api_refs": solution_refs,
                    "target_match_detail": match,
                    "candidate_count": len(candidate_skill_ids),
                    "candidate_positive_missing": True,
                    "supervision_source": "official_solution_api_refs_diagnostic",
                },
            }
        )
    return rows


def _aligned_selected_indices(decision: dict[str, Any]) -> tuple[list[str], list[int]]:
    candidates = [str(item) for item in _as_list(decision.get("candidate_skill_ids"))]
    selected = [str(item) for item in _as_list(decision.get("selected_skill_ids"))]
    proposed: list[int] = []
    for item in _as_list(decision.get("selected_candidate_local_indices")):
        try:
            proposed.append(int(item))
        except (TypeError, ValueError):
            continue
    selected_ids: list[str] = []
    selected_indices: list[int] = []
    for pos, skill_id in enumerate(selected):
        idx: int | None = None
        if pos < len(proposed) and 0 <= proposed[pos] < len(candidates) and candidates[proposed[pos]] == skill_id:
            idx = proposed[pos]
        if idx is None:
            try:
                idx = candidates.index(skill_id)
            except ValueError:
                idx = None
        if idx is None:
            continue
        selected_ids.append(skill_id)
        selected_indices.append(int(idx))
    return selected_ids, selected_indices


def _solution_path(root: str | Path, task_id: str) -> Path:
    return Path(root) / str(task_id) / "ground_truth" / "solution.py"


def _load_skill_by_id(skill_pool_path: str | Path) -> dict[str, dict[str, Any]]:
    skills: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(_read_jsonl(skill_pool_path)):
        skill_id = str(row.get("skill_id") or row.get("canonical_skill_id") or idx)
        skills[skill_id] = row
    return skills


def _is_success_rollout(rollout: dict[str, Any]) -> bool:
    outcome = rollout.get("outcome")
    if isinstance(outcome, dict) and outcome.get("label") == "success":
        return True
    return bool(rollout.get("success") or rollout.get("evaluation_success"))


def build_api_correction_dataset(
    *,
    rollouts_path: str | Path,
    skill_pool_path: str | Path,
    appworld_tasks_root: str | Path,
    max_targets: int = 3,
    include_success_rollouts: bool = False,
) -> ApiCorrectionDataset:
    return build_api_correction_dataset_from_rollouts(
        rollouts=_read_jsonl(rollouts_path),
        skill_pool_path=skill_pool_path,
        appworld_tasks_root=appworld_tasks_root,
        max_targets=max_targets,
        include_success_rollouts=include_success_rollouts,
    )


def build_api_correction_dataset_from_rollouts(
    *,
    rollouts: Iterable[dict[str, Any]],
    skill_pool_path: str | Path,
    appworld_tasks_root: str | Path,
    max_targets: int = 3,
    include_success_rollouts: bool = False,
) -> ApiCorrectionDataset:
    rollout_rows = list(rollouts)
    skill_by_id = _load_skill_by_id(skill_pool_path)
    samples: list[dict[str, Any]] = []
    stage0_retrieval_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    solution_api_refs_by_task: dict[str, list[str]] = {}

    for rollout in rollout_rows:
        if _is_success_rollout(rollout) and not include_success_rollouts:
            skipped["success_rollout"] = skipped.get("success_rollout", 0) + 1
            continue
        task_id = str(rollout.get("task_id") or rollout.get("query_id") or "")
        path = _solution_path(appworld_tasks_root, task_id)
        if not path.exists():
            skipped["missing_solution"] = skipped.get("missing_solution", 0) + 1
            continue
        solution_refs = extract_appworld_api_refs(path.read_text(encoding="utf-8"))
        solution_api_refs_by_task[task_id] = solution_refs
        if not solution_refs:
            skipped["no_solution_api_refs"] = skipped.get("no_solution_api_refs", 0) + 1
            continue
        for decision in _as_list(rollout.get("decisions")):
            if not isinstance(decision, dict):
                continue
            if not bool(decision.get("training_ready")):
                skipped["decision_not_training_ready"] = skipped.get("decision_not_training_ready", 0) + 1
                continue
            candidates = [str(item) for item in _as_list(decision.get("candidate_skill_ids"))]
            matches = match_solution_api_refs_to_candidate_skills(
                solution_api_refs=solution_refs,
                candidate_skill_ids=candidates,
                skill_by_id=skill_by_id,
                max_targets=max_targets,
            )
            if not matches:
                stage0_rows = _stage0_retrieval_correction_rows(
                    rollout=rollout,
                    decision=decision,
                    task_id=task_id,
                    solution_refs=solution_refs,
                    candidate_skill_ids=candidates,
                    skill_by_id=skill_by_id,
                    max_targets=max_targets,
                )
                if stage0_rows:
                    stage0_retrieval_rows.extend(stage0_rows)
                    skipped["no_candidate_api_match"] = skipped.get("no_candidate_api_match", 0) + 1
                    continue
                skipped["no_skill_pool_api_match"] = skipped.get("no_skill_pool_api_match", 0) + 1
                continue
            selected_ids, selected_indices = _aligned_selected_indices(decision)
            target_indices = [int(item["local_index"]) for item in matches]
            target_ids = [str(item["skill_id"]) for item in matches]
            rejected_pairs = [
                (skill_id, idx)
                for skill_id, idx in zip(selected_ids, selected_indices)
                if idx not in set(target_indices)
            ]
            if not rejected_pairs:
                skipped["selected_already_target"] = skipped.get("selected_already_target", 0) + 1
                continue
            step_idx = int(decision.get("step_idx", 0) or 0)
            samples.append(
                {
                    "schema_version": "current_route_preference.v1",
                    "sample_type": "counterfactual_api_correction",
                    "task_id": task_id,
                    "query_id": str(rollout.get("query_id") or task_id),
                    "decision_id": f"{task_id}::{step_idx}",
                    "step_idx": step_idx,
                    "state_text": str(decision.get("state_text") or ""),
                    "user_goal": str(rollout.get("user_goal") or ""),
                    "candidate_skill_ids": candidates,
                    "target_skill_ids": target_ids,
                    "target_local_indices": target_indices,
                    "rejected_skill_ids": [skill_id for skill_id, _idx in rejected_pairs],
                    "rejected_local_indices": [int(idx) for _skill_id, idx in rejected_pairs],
                    "solution_api_refs": solution_refs,
                    "target_match_details": matches,
                    "previous_selected_skill_ids": [str(item) for item in _as_list(decision.get("previous_selected_skill_ids"))],
                    "previous_execute_output": str(decision.get("previous_execute_output") or ""),
                    "ranking_mode": decision.get("ranking_mode"),
                    "candidate_source": decision.get("candidate_source"),
                    "outcome_label": (rollout.get("outcome") or {}).get("label") if isinstance(rollout.get("outcome"), dict) else "",
                    "policy_signal": "counterfactual_api_correction",
                    "reward": 0.0,
                    "weight": 1.0,
                    "supervision_source": "official_solution_api_refs_diagnostic",
                }
            )

    report = {
        "schema_version": "api_correction_preference_report.v1",
        "rollout_count": len(rollout_rows),
        "sample_count": len(samples),
        "task_count": len({sample["task_id"] for sample in samples}),
        "stage0_retrieval_correction_count": len(stage0_retrieval_rows),
        "stage0_retrieval_correction_task_count": len(
            {str((row.get("provenance") or {}).get("task_id") or row.get("query_id")) for row in stage0_retrieval_rows}
        ),
        "skipped": skipped,
        "solution_api_ref_task_count": len(solution_api_refs_by_task),
        "supervision_source": "official_solution_api_refs_diagnostic",
    }
    return ApiCorrectionDataset(samples=samples, stage0_retrieval_rows=stage0_retrieval_rows, report=report)


def write_api_correction_dataset(
    *,
    rollouts_path: str | Path,
    skill_pool_path: str | Path,
    appworld_tasks_root: str | Path,
    output_jsonl_path: str | Path,
    report_json_path: str | Path,
    max_targets: int = 3,
    stage0_retrieval_jsonl_path: str | Path | None = None,
) -> ApiCorrectionDataset:
    dataset = build_api_correction_dataset(
        rollouts_path=rollouts_path,
        skill_pool_path=skill_pool_path,
        appworld_tasks_root=appworld_tasks_root,
        max_targets=max_targets,
    )
    output_path = Path(output_jsonl_path)
    report_path = Path(report_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for sample in dataset.samples:
            fp.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    if stage0_retrieval_jsonl_path is not None:
        stage0_path = Path(stage0_retrieval_jsonl_path)
        stage0_path.parent.mkdir(parents=True, exist_ok=True)
        with stage0_path.open("w", encoding="utf-8") as fp:
            for row in dataset.stage0_retrieval_rows:
                fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report_path.write_text(
        json.dumps(dataset.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset
