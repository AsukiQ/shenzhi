from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from clstr.appworld_eval import compute_retrieval_metrics
from clstr.appworld_routing import read_jsonl, write_json


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def canonical_skill_key(skill: dict[str, Any]) -> str:
    name = _normalize_text(str(skill.get("name", "")))
    executor = _normalize_text(str(skill.get("executor_desc", "")))
    return f"{name}::{executor}"


def _load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        if int(row.get("relevance", 1)) <= 0:
            continue
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["skill_id"]))
    return qrels


def _load_predictions(path: str | Path) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or row.get("task_id"))
        predictions[query_id] = [str(item) for item in row.get("ranked_skill_ids", [])]
    return predictions


def _load_executor_runs(path: str | Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or row.get("task_id"))
        if query_id:
            runs[query_id] = row
    return runs


def _canonicalize_sets(
    qrels: dict[str, set[str]],
    skill_to_key: dict[str, str],
) -> dict[str, set[str]]:
    return {
        query_id: {skill_to_key.get(skill_id, skill_id) for skill_id in positives}
        for query_id, positives in qrels.items()
    }


def _canonicalize_predictions(
    predictions: dict[str, list[str]],
    skill_to_key: dict[str, str],
) -> dict[str, list[str]]:
    canonical: dict[str, list[str]] = {}
    for query_id, ranked in predictions.items():
        row: list[str] = []
        seen: set[str] = set()
        for skill_id in ranked:
            key = skill_to_key.get(skill_id, skill_id)
            if key in seen:
                continue
            seen.add(key)
            row.append(key)
        canonical[query_id] = row
    return canonical


def _rank_of_hit(ranked: Sequence[str], positives: set[str]) -> int | None:
    for idx, skill_id in enumerate(ranked, start=1):
        if skill_id in positives:
            return idx
    return None


_APPWORLD_APPS = (
    "amazon",
    "file_system",
    "gmail",
    "phone",
    "simple_note",
    "splitwise",
    "spotify",
    "supervisor",
    "todoist",
    "venmo",
)


def _skill_text(skill: dict[str, Any]) -> str:
    return " ".join(
        str(skill.get(key, ""))
        for key in ("skill_id", "name", "description", "executor_desc", "body", "skill_md")
    ).lower()


def _skill_apps(skill: dict[str, Any]) -> set[str]:
    text = _skill_text(skill)
    apps: set[str] = set()
    for app in _APPWORLD_APPS:
        if app in text or app.replace("_", " ") in text or app.replace("_", "-") in text:
            apps.add(app)
    return apps


def _is_auth_like_skill(skill: dict[str, Any]) -> bool:
    text = _skill_text(skill)
    return any(token in text for token in ("auth", "login", "credential", "password", "account_password"))


def _executor_success(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return bool(row.get("evaluation_success", row.get("success", False)))


def _executor_failed(row: dict[str, Any] | None) -> bool:
    return bool(row) and not _executor_success(row)


def _selected_skill_ids(
    *,
    run: dict[str, Any] | None,
    predictions: dict[str, list[str]],
    query_id: str,
    top_k: int,
) -> list[str]:
    if run and isinstance(run.get("selected_skill_ids"), list):
        return [str(skill_id) for skill_id in run.get("selected_skill_ids", [])][: int(top_k)]
    return predictions.get(query_id, [])[: int(top_k)]


def _skill_summary(
    *,
    skill_id: str,
    skills_by_id: dict[str, dict[str, Any]],
    skill_to_key: dict[str, str],
    cluster_sizes: dict[str, int],
    positives: set[str],
    canonical_positives: set[str],
    required_apps: set[str],
) -> dict[str, Any]:
    skill = skills_by_id.get(skill_id, {"skill_id": skill_id})
    key = skill_to_key.get(skill_id, skill_id)
    apps = sorted(_skill_apps(skill))
    allowed_apps = set(required_apps) | {"supervisor"}
    off_app = bool(set(apps) - allowed_apps) if apps else False
    return {
        "skill_id": skill_id,
        "name": str(skill.get("name", "")),
        "apps": apps,
        "auth_like": _is_auth_like_skill(skill),
        "off_app": off_app,
        "id_positive": skill_id in positives,
        "canonical_positive": key in canonical_positives,
        "canonical_cluster_size": cluster_sizes.get(key, 1),
    }


def _topk_stats(
    *,
    skill_ids: Sequence[str],
    skills_by_id: dict[str, dict[str, Any]],
    skill_to_key: dict[str, str],
    required_apps: set[str],
) -> dict[str, Any]:
    auth_like_count = 0
    off_app_count = 0
    canonical_keys: list[str] = []
    allowed_apps = set(required_apps) | {"supervisor"}
    for skill_id in skill_ids:
        skill = skills_by_id.get(skill_id, {"skill_id": skill_id})
        if _is_auth_like_skill(skill):
            auth_like_count += 1
        apps = _skill_apps(skill)
        if apps and bool(apps - allowed_apps):
            off_app_count += 1
        canonical_keys.append(skill_to_key.get(skill_id, skill_id))
    return {
        "top_k": len(skill_ids),
        "auth_like_count": auth_like_count,
        "off_app_skill_count": off_app_count,
        "duplicate_slot_count": len(canonical_keys) - len(set(canonical_keys)),
    }


def build_duplicate_aware_eval_report(
    *,
    skill_pool_path: str | Path,
    qrels_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    method: str,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, Any]:
    skills = read_jsonl(skill_pool_path)
    skill_to_key = {str(skill["skill_id"]): canonical_skill_key(skill) for skill in skills}
    cluster_sizes: dict[str, int] = {}
    for key in skill_to_key.values():
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1

    qrels = _load_qrels(qrels_path)
    predictions = _load_predictions(predictions_path)
    canonical_qrels = _canonicalize_sets(qrels, skill_to_key)
    canonical_predictions = _canonicalize_predictions(predictions, skill_to_key)

    miss_rows: list[dict[str, Any]] = []
    for query_id, positives in qrels.items():
        ranked = predictions.get(query_id, [])
        id_hit = _rank_of_hit(ranked, positives)
        if id_hit == 1:
            continue
        canonical_hit = _rank_of_hit(canonical_predictions.get(query_id, []), canonical_qrels.get(query_id, set()))
        miss_rows.append(
            {
                "query_id": query_id,
                "id_hit_rank": id_hit,
                "canonical_hit_rank": canonical_hit,
                "canonical_hit@1": canonical_hit == 1,
                "top1_skill_id": ranked[0] if ranked else None,
                "positive_skill_ids": sorted(positives),
            }
        )

    report = {
        "status": "ok",
        "method": method,
        "skill_pool_path": str(skill_pool_path),
        "qrels_path": str(qrels_path),
        "predictions_path": str(predictions_path),
        "id_metrics": compute_retrieval_metrics(predictions, qrels, ks=ks),
        "canonical_metrics": compute_retrieval_metrics(canonical_predictions, canonical_qrels, ks=ks),
        "skill_count": len(skills),
        "canonical_skill_count": len(set(skill_to_key.values())),
        "duplicate_cluster_count": sum(1 for size in cluster_sizes.values() if size > 1),
        "miss_rows": miss_rows,
        "caveat": "Canonical metrics merge SkillX duplicate skills by normalized name plus executor_desc.",
    }
    write_json(Path(output_dir) / "report.json", report)
    return report


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def build_train_dev_overlap_report(
    *,
    train_tasks_path: str | Path,
    dev_tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    train = read_jsonl(train_tasks_path)
    dev = read_jsonl(dev_tasks_path)
    train_sets = {tuple(sorted(str(item) for item in row.get("positive_skill_ids", []))) for row in train}
    train_positive_skills = {
        str(skill_id)
        for row in train
        for skill_id in row.get("positive_skill_ids", [])
    }
    dev_positive_skills = {
        str(skill_id)
        for row in dev
        for skill_id in row.get("positive_skill_ids", [])
    }
    train_tokens = [
        (
            str(row.get("task_id") or row.get("query_id")),
            _tokenize(str(row.get("instruction_text") or row.get("query") or "")),
            str(row.get("instruction_text") or row.get("query") or ""),
        )
        for row in train
    ]
    nearest_rows: list[dict[str, Any]] = []
    for row in dev:
        dev_tokens = _tokenize(str(row.get("instruction_text") or row.get("query") or ""))
        best = (0.0, "", "")
        for train_id, tokens, instruction in train_tokens:
            score = _jaccard(dev_tokens, tokens)
            if score > best[0]:
                best = (score, train_id, instruction)
        nearest_rows.append(
            {
                "dev_task_id": str(row.get("task_id") or row.get("query_id")),
                "nearest_train_task_id": best[1],
                "jaccard": round(float(best[0]), 6),
                "dev_instruction": str(row.get("instruction_text") or row.get("query") or ""),
                "train_instruction": best[2],
            }
        )
    exact_reuse = sum(
        1
        for row in dev
        if tuple(sorted(str(item) for item in row.get("positive_skill_ids", []))) in train_sets
    )
    jaccards = [row["jaccard"] for row in nearest_rows]
    report = {
        "status": "ok",
        "train_tasks_path": str(train_tasks_path),
        "dev_tasks_path": str(dev_tasks_path),
        "train_task_count": len(train),
        "dev_task_count": len(dev),
        "exact_positive_set_reuse_count": exact_reuse,
        "exact_positive_set_reuse_rate": round(exact_reuse / max(len(dev), 1), 6),
        "train_positive_skill_count": len(train_positive_skills),
        "dev_positive_skill_count": len(dev_positive_skills),
        "dev_positive_skills_seen_in_train": len(dev_positive_skills & train_positive_skills),
        "dev_positive_skills_seen_in_train_rate": round(
            len(dev_positive_skills & train_positive_skills) / max(len(dev_positive_skills), 1),
            6,
        ),
        "nearest_instruction_jaccard": {
            "avg": round(sum(jaccards) / max(len(jaccards), 1), 6),
            "max": round(max(jaccards) if jaccards else 0.0, 6),
            "ge_0.8_count": sum(1 for value in jaccards if value >= 0.8),
        },
        "nearest_rows": sorted(nearest_rows, key=lambda item: item["jaccard"], reverse=True),
        "caveat": "High positive-set reuse favors static supervised routing baselines.",
    }
    write_json(Path(output_dir) / "report.json", report)
    return report


def build_executor_failure_topk_report(
    *,
    skill_pool_path: str | Path,
    tasks_path: str | Path,
    qrels_path: str | Path,
    run_paths: Mapping[str, str | Path],
    prediction_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    focus_method: str = "clstr_base",
    reference_methods: Sequence[str] = ("skillrouter_base", "qwen_only"),
    top_k: int = 5,
) -> dict[str, Any]:
    skills = read_jsonl(skill_pool_path)
    skills_by_id = {str(skill["skill_id"]): skill for skill in skills}
    skill_to_key = {str(skill["skill_id"]): canonical_skill_key(skill) for skill in skills}
    cluster_sizes: dict[str, int] = {}
    for key in skill_to_key.values():
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1

    tasks = {
        str(row.get("query_id") or row.get("task_id")): row
        for row in read_jsonl(tasks_path)
    }
    qrels = _load_qrels(qrels_path)
    canonical_qrels = _canonicalize_sets(qrels, skill_to_key)
    runs_by_method = {method: _load_executor_runs(path) for method, path in run_paths.items()}
    predictions_by_method = {
        method: _load_predictions(path)
        for method, path in prediction_paths.items()
    }

    query_ids = sorted(
        set(tasks)
        | set(qrels)
        | {query_id for runs in runs_by_method.values() for query_id in runs}
        | {query_id for predictions in predictions_by_method.values() for query_id in predictions}
    )

    method_summaries: dict[str, dict[str, Any]] = {}
    for method, runs in runs_by_method.items():
        rows = [runs[query_id] for query_id in query_ids if query_id in runs]
        method_summaries[method] = {
            "task_count": len(rows),
            "success_count": sum(1 for row in rows if _executor_success(row)),
            "execution_failure_count": sum(1 for row in rows if bool(row.get("generation_ok", True)) and not bool(row.get("execution_ok", False))),
            "task_completed_count": sum(1 for row in rows if bool(row.get("task_completed", False))),
        }
        method_summaries[method]["success_rate"] = round(
            method_summaries[method]["success_count"] / max(method_summaries[method]["task_count"], 1),
            6,
        )

    focus_runs = runs_by_method.get(focus_method, {})
    focus_predictions = predictions_by_method.get(focus_method, {})
    reference_methods = list(reference_methods)
    focus_failure_rows: list[dict[str, Any]] = []

    for query_id in query_ids:
        focus_run = focus_runs.get(query_id)
        if not _executor_failed(focus_run):
            continue
        reference_success = [
            method
            for method in reference_methods
            if _executor_success(runs_by_method.get(method, {}).get(query_id))
        ]
        if not reference_success:
            continue

        task = tasks.get(query_id, {})
        required_apps = {str(app) for app in task.get("required_apps", [])}
        positives = qrels.get(query_id, set())
        canonical_positives = canonical_qrels.get(query_id, set())
        focus_top_ids = _selected_skill_ids(
            run=focus_run,
            predictions=focus_predictions,
            query_id=query_id,
            top_k=top_k,
        )
        focus_canonical_ids = _canonicalize_predictions({query_id: focus_top_ids}, skill_to_key).get(query_id, [])
        row: dict[str, Any] = {
            "query_id": query_id,
            "instruction": str(task.get("instruction_text") or task.get("query") or ""),
            "required_apps": sorted(required_apps),
            "reference_methods_success": reference_success,
            "focus_run": {
                "success": _executor_success(focus_run),
                "execution_ok": bool(focus_run.get("execution_ok", False)) if focus_run else False,
                "task_completed": bool(focus_run.get("task_completed", False)) if focus_run else False,
                "execute_output": str(focus_run.get("execute_output", ""))[:500] if focus_run else "",
            },
            "focus_id_hit_rank": _rank_of_hit(focus_top_ids, positives),
            "focus_canonical_hit_rank": _rank_of_hit(focus_canonical_ids, canonical_positives),
            "positive_skill_ids": sorted(positives),
            "focus_topk_stats": _topk_stats(
                skill_ids=focus_top_ids,
                skills_by_id=skills_by_id,
                skill_to_key=skill_to_key,
                required_apps=required_apps,
            ),
            "focus_topk_skills": [
                _skill_summary(
                    skill_id=skill_id,
                    skills_by_id=skills_by_id,
                    skill_to_key=skill_to_key,
                    cluster_sizes=cluster_sizes,
                    positives=positives,
                    canonical_positives=canonical_positives,
                    required_apps=required_apps,
                )
                for skill_id in focus_top_ids
            ],
            "reference_topk": {},
        }
        for method in reference_success:
            top_ids = _selected_skill_ids(
                run=runs_by_method.get(method, {}).get(query_id),
                predictions=predictions_by_method.get(method, {}),
                query_id=query_id,
                top_k=top_k,
            )
            row["reference_topk"][method] = [
                _skill_summary(
                    skill_id=skill_id,
                    skills_by_id=skills_by_id,
                    skill_to_key=skill_to_key,
                    cluster_sizes=cluster_sizes,
                    positives=positives,
                    canonical_positives=canonical_positives,
                    required_apps=required_apps,
                )
                for skill_id in top_ids
            ]
        focus_failure_rows.append(row)

    aggregate_stats = {
        "rows_with_focus_top1_auth_like": sum(
            1 for row in focus_failure_rows if row["focus_topk_skills"] and row["focus_topk_skills"][0]["auth_like"]
        ),
        "rows_with_focus_top1_off_app": sum(
            1 for row in focus_failure_rows if row["focus_topk_skills"] and row["focus_topk_skills"][0]["off_app"]
        ),
        "rows_with_no_focus_id_hit_in_topk": sum(1 for row in focus_failure_rows if row["focus_id_hit_rank"] is None),
        "rows_with_focus_canonical_hit_in_topk": sum(1 for row in focus_failure_rows if row["focus_canonical_hit_rank"] is not None),
    }

    report = {
        "status": "ok",
        "skill_pool_path": str(skill_pool_path),
        "tasks_path": str(tasks_path),
        "qrels_path": str(qrels_path),
        "focus_method": focus_method,
        "reference_methods": reference_methods,
        "top_k": int(top_k),
        "method_summaries": method_summaries,
        "focus_failure_reference_success_count": len(focus_failure_rows),
        "aggregate_focus_failure_stats": aggregate_stats,
        "focus_failure_reference_success_rows": focus_failure_rows,
        "caveat": "This is a diagnostic over existing executor runs and routing top-k; it does not rerun AppWorld.",
    }
    write_json(Path(output_dir) / "report.json", report)
    return report
