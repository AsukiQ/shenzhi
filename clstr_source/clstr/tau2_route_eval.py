from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from clstr.full_base_train import UNIFIED_MEMORY_ROUTE_SCORER, _skill_id
from clstr.history_channel import (
    audit_history_channel_rows,
    materialize_history_free_state,
    router_state_text,
)
from clstr.logged_online_stage4_train import (
    attach_trajectory_prefix_online_memory_scores,
    evaluate_logged_online_stage4_rows,
    evaluate_logged_online_stage4_rows_by_benchmark,
)
from clstr.memory_candidate_recall import (
    candidate_recall_protocol_metadata,
    declared_candidate_pool_size,
    row_positive_skill_ids,
    source_rows_have_causal_sequence,
)
from clstr.memory_utility_gate_train import resolve_reliability_gate
from clstr.memory_utility_records import canonical_digest
from clstr.mt_ablation_eval import evaluate_mt_ablation_rows
from clstr.native_benchmark_checkpoint_adapter import (
    restore_native_benchmark_checkpoint_chain,
)
from clstr.stage4_act_train import STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
from clstr.toolbench_full_clstr_route_eval import strict_metric_contract, strict_stage4_metrics


@dataclass(frozen=True)
class Tau2RouteCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


TAU2_TASK_SPLITS = frozenset({"base", "train", "test", "full"})
ROUTER_STATE_CONTRACT = "history_free_current_state_v1"


def tau_family_start_skill_id(benchmark_name: str, domain: str) -> str:
    return f"{_clean_benchmark_name(benchmark_name)}/{_clean_domain(domain)}/__start__"


def tau_family_action_skill_id(benchmark_name: str, domain: str, action_name: str) -> str:
    return f"{_clean_benchmark_name(benchmark_name)}/{_clean_domain(domain)}/{_clean_action_name(action_name)}"


def tau2_start_skill_id(domain: str) -> str:
    return tau_family_start_skill_id("tau2", domain)


def tau2_action_skill_id(domain: str, action_name: str) -> str:
    return tau_family_action_skill_id("tau2", domain, action_name)


def _clean_benchmark_name(benchmark_name: str) -> str:
    return str(benchmark_name or "tau2").strip().lower()


def _clean_domain(domain: str) -> str:
    return str(domain or "").strip().lower()


def _clean_action_name(action_name: str) -> str:
    return str(action_name or "").strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _task_id(task: dict[str, Any], fallback: int) -> str:
    return str(task.get("id") or task.get("task_id") or fallback).strip()


def _task_instructions(task: dict[str, Any]) -> dict[str, Any]:
    scenario = task.get("user_scenario") if isinstance(task.get("user_scenario"), dict) else {}
    instructions = scenario.get("instructions") if isinstance(scenario.get("instructions"), dict) else {}
    return instructions if isinstance(instructions, dict) else {}


def _task_description_text(task: dict[str, Any]) -> str:
    description = task.get("description")
    if isinstance(description, dict):
        return "\n".join(f"{key}: {value}" for key, value in sorted(description.items()) if value)
    return str(description or "").strip()


def _action_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = task.get("evaluation_criteria") if isinstance(task.get("evaluation_criteria"), dict) else {}
    actions = criteria.get("actions") if isinstance(criteria.get("actions"), list) else []
    return [action for action in actions if isinstance(action, dict) and _clean_action_name(action.get("name"))]


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _action_text(action: dict[str, Any] | None) -> str:
    if not action:
        return "previous_action: START"
    name = _clean_action_name(action.get("name"))
    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    return f"previous_action: {name}\nprevious_arguments: {_jsonish(args)}"


def _target_action_text(action: dict[str, Any]) -> str:
    name = _clean_action_name(action.get("name"))
    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    return f"action: {name}\narguments: {_jsonish(args)}"


def _task_semantic_text(task: dict[str, Any]) -> str:
    instructions = _task_instructions(task)
    values = [_task_description_text(task)]
    values.extend(
        _jsonish(instructions.get(key))
        for key in ("reason_for_call", "known_info", "unknown_info", "task_instructions")
        if instructions.get(key)
    )
    return "\n".join(value for value in values if value.strip())


def _history_text(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return ""
    lines = []
    for idx, action in enumerate(actions, start=1):
        name = _clean_action_name(action.get("name"))
        args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        lines.append(f"{idx}. {name}({_jsonish(args)})")
    return "\n".join(lines)


def _state_text(
    *,
    benchmark_name: str,
    domain: str,
    policy_text: str,
    task: dict[str, Any],
    history_actions: list[dict[str, Any]],
    include_history: bool = True,
) -> str:
    instructions = _task_instructions(task)
    parts = [f"domain: {domain}", f"task_description: {_task_description_text(task)}"]
    for key in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
        value = instructions.get(key)
        if value:
            parts.append(f"{key}: {_jsonish(value)}")
    if policy_text.strip():
        parts.append(f"domain_policy:\n{policy_text.strip()}")
    history = _history_text(history_actions)
    if include_history and history:
        parts.append(f"history:\n{history}")
    return "\n".join(parts)


def _skill_row(benchmark_name: str, domain: str, action_name: str | None, *, policy_text: str) -> dict[str, Any]:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    domain = _clean_domain(domain)
    if action_name is None:
        skill_id = tau_family_start_skill_id(benchmark_name, domain)
        name = f"{domain}.START"
        description = f"Initial {benchmark_name} state before any {domain} tool action."
        body = f"Domain policy context for {domain}. No executable tool action has been taken yet."
    else:
        action = _clean_action_name(action_name)
        skill_id = tau_family_action_skill_id(benchmark_name, domain, action)
        name = f"{domain}.{action}"
        description = f"Use {domain} API action `{action}` when the dialogue state requires it."
        body = f"Action name: {action}\nDomain: {domain}\nPolicy excerpt:\n{policy_text.strip()[:2000]}"
    return {
        "skill_id": skill_id,
        "name": name,
        "description": description,
        "executor_desc": description,
        "body": body,
        "skill_md": body,
        "source_benchmark": benchmark_name,
        "domain": domain,
    }


def _domain_dirs(data_root: Path, domains: Iterable[str] | None) -> list[tuple[str, Path]]:
    root = data_root / "domains"
    if domains is None:
        domain_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    else:
        domain_names = [_clean_domain(domain) for domain in domains if _clean_domain(domain)]
    return [(domain, root / domain) for domain in domain_names]


def _read_domain_policy(domain_dir: Path) -> str:
    for name in ("policy.md", "main_policy.md", "policy_solo.md", "main_policy_solo.md"):
        path = domain_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def _ordered_domain_tasks(
    raw_tasks: list[Any],
    *,
    domain: str,
    domain_dir: Path,
    task_split: str,
    skipped: Counter[str],
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for fallback_idx, task in enumerate(raw_tasks):
        if not isinstance(task, dict):
            skipped["task_not_dict"] += 1
            continue
        task_id = _task_id(task, fallback_idx)
        if not task_id:
            raise ValueError(f"Tau2 {domain} task has no stable ID")
        if task_id in tasks_by_id:
            raise ValueError(f"Tau2 {domain} contains duplicate task ID: {task_id}")
        tasks_by_id[task_id] = task

    if task_split == "full":
        ordered_ids = sorted(tasks_by_id)
    else:
        split_path = domain_dir / "split_tasks.json"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Tau2 {domain} {task_split} split requires {split_path}"
            )
        split_payload = _read_json(split_path)
        if not isinstance(split_payload, dict):
            raise ValueError(f"Tau2 {domain} split_tasks.json must be an object")
        raw_ids = split_payload.get(task_split)
        if not isinstance(raw_ids, list):
            raise ValueError(f"Tau2 {domain} split is missing: {task_split}")
        ordered_ids = [str(task_id) for task_id in raw_ids]
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError(f"Tau2 {domain} {task_split} split has duplicate task IDs")
        missing = [task_id for task_id in ordered_ids if task_id not in tasks_by_id]
        if missing:
            raise ValueError(
                f"Tau2 {domain} {task_split} split has missing task IDs: {missing[:5]}"
            )
    return [(task_id, tasks_by_id[task_id]) for task_id in ordered_ids], ordered_ids


def load_tau2_route_corpus(
    data_root: str | Path,
    *,
    domains: Iterable[str] | None = None,
    task_split: str = "full",
    max_tasks_per_domain: int | None = None,
    benchmark_name: str = "tau2",
) -> Tau2RouteCorpus:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    task_split = str(task_split or "full").strip().lower()
    if task_split not in TAU2_TASK_SPLITS:
        raise ValueError("task_split must be base, train, test, or full")
    data_root = Path(data_root)
    skills: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    task_count = 0
    action_step_count = 0
    action_names_by_domain: dict[str, set[str]] = {}
    task_counts_by_domain: Counter[str] = Counter()
    step_counts_by_domain: Counter[str] = Counter()
    refusal_examples_by_domain: dict[str, list[str]] = {}
    split_task_ids_by_domain: dict[str, list[str]] = {}
    policies: dict[str, str] = {}

    for domain, domain_dir in _domain_dirs(data_root, domains):
        tasks_path = domain_dir / "tasks.json"
        if not tasks_path.exists():
            skipped["missing_tasks_json"] += 1
            continue
        policy_text = _read_domain_policy(domain_dir)
        policies[domain] = policy_text
        raw_tasks = _read_json(tasks_path)
        if not isinstance(raw_tasks, list):
            skipped["tasks_json_not_list"] += 1
            continue
        selected_tasks, split_task_ids = _ordered_domain_tasks(
            raw_tasks,
            domain=domain,
            domain_dir=domain_dir,
            task_split=task_split,
            skipped=skipped,
        )
        if max_tasks_per_domain is not None:
            selected_tasks = selected_tasks[: max(0, int(max_tasks_per_domain))]
            split_task_ids = split_task_ids[: max(0, int(max_tasks_per_domain))]
        split_task_ids_by_domain[domain] = split_task_ids
        domain_action_names: set[str] = set()
        parsed_tasks: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
        for task_id, task in selected_tasks:
            task_actions = _action_rows(task)
            task_count += 1
            task_counts_by_domain[domain] += 1
            if not task_actions:
                skipped["empty_action_task"] += 1
                refusal_examples_by_domain.setdefault(domain, []).append(task_id)
            for action in task_actions:
                domain_action_names.add(_clean_action_name(action.get("name")))
            parsed_tasks.append((task_id, task, task_actions))
        action_names_by_domain[domain] = domain_action_names
        skills.append(_skill_row(benchmark_name, domain, None, policy_text=policy_text))
        for action_name in sorted(domain_action_names):
            skills.append(_skill_row(benchmark_name, domain, action_name, policy_text=policy_text))
        domain_candidates = [
            tau_family_action_skill_id(benchmark_name, domain, action_name)
            for action_name in sorted(domain_action_names)
        ]
        for task_id, task, task_actions in parsed_tasks:
            trajectory_id = f"{benchmark_name}/{domain}/{task_id}"
            if not task_actions:
                state_text_current = _state_text(
                    domain=domain,
                    benchmark_name=benchmark_name,
                    policy_text=policy_text,
                    task=task,
                    history_actions=[],
                    include_history=False,
                )
                source_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "source_benchmark": benchmark_name,
                        "split": task_split,
                        "task_id": f"{trajectory_id}::stop",
                        "trajectory_id": trajectory_id,
                        "step_index": 0,
                        "domain": domain,
                        "state_text": state_text_current,
                        "state_text_current": state_text_current,
                        "state_text_full": state_text_current,
                        "history_text": "",
                        "action_text": _action_text(None),
                        "target_action_text": "",
                        "split_semantic_text": _task_semantic_text(task),
                        "next_observation_text": "no_tool_action_required",
                        "skill_id": tau_family_start_skill_id(
                            benchmark_name,
                            domain,
                        ),
                        "next_skill_id": "",
                        "candidate_next_skill_ids": [],
                        "route_target": "STOP",
                        "loss_mask": {
                            "routing": False,
                            "L_trans_skill_ce": False,
                        },
                        "provenance": {
                            "source": f"{benchmark_name}_no_tool_task",
                            "domain": domain,
                            "raw_task_id": task_id,
                        },
                    }
                )
                continue
            for step_index, action in enumerate(task_actions):
                action_name = _clean_action_name(action.get("name"))
                history_actions = task_actions[:step_index]
                previous_action = history_actions[-1] if history_actions else None
                state_text_current = _state_text(
                    domain=domain,
                    benchmark_name=benchmark_name,
                    policy_text=policy_text,
                    task=task,
                    history_actions=history_actions,
                    include_history=False,
                )
                state_text_full = _state_text(
                    domain=domain,
                    benchmark_name=benchmark_name,
                    policy_text=policy_text,
                    task=task,
                    history_actions=history_actions,
                    include_history=True,
                )
                source_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "source_benchmark": benchmark_name,
                        "split": task_split,
                        "task_id": f"{trajectory_id}::{step_index}",
                        "trajectory_id": trajectory_id,
                        "step_index": step_index,
                        "domain": domain,
                        "state_text": state_text_current,
                        "state_text_current": state_text_current,
                        "state_text_full": state_text_full,
                        "history_text": _history_text(history_actions),
                        "action_text": _action_text(previous_action),
                        "target_action_text": _target_action_text(action),
                        "split_semantic_text": _task_semantic_text(task),
                        "next_observation_text": f"oracle_next_action_arguments: {_jsonish(action.get('arguments') or {})}",
                        "observation_source": "oracle_next_action_arguments",
                        "skill_id": tau_family_action_skill_id(
                            benchmark_name,
                            domain,
                            _clean_action_name(previous_action.get("name")),
                        )
                        if previous_action
                        else tau_family_start_skill_id(benchmark_name, domain),
                        "next_skill_id": tau_family_action_skill_id(benchmark_name, domain, action_name),
                        "candidate_next_skill_ids": list(domain_candidates),
                        "route_target": "TOOL",
                        "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                        "provenance": {
                            "source": f"{benchmark_name}_oracle_actions",
                            "domain": domain,
                            "raw_task_id": task_id,
                            "raw_action_id": action.get("action_id"),
                        },
                    }
                )
                action_step_count += 1
                step_counts_by_domain[domain] += 1

    skill_ids_seen: set[str] = set()
    deduped_skills: list[dict[str, Any]] = []
    for skill in skills:
        sid = str(skill.get("skill_id") or "")
        if not sid or sid in skill_ids_seen:
            continue
        skill_ids_seen.add(sid)
        deduped_skills.append(skill)

    history_channel = audit_history_channel_rows(
        source_rows,
        require_explicit_current=True,
    )
    report = {
        "benchmark": benchmark_name,
        "data_root": str(data_root),
        "domains": [domain for domain, _ in _domain_dirs(data_root, domains)],
        "task_split": task_split,
        "split_task_ids_by_domain": split_task_ids_by_domain,
        "split_membership_sha256": canonical_digest(split_task_ids_by_domain),
        "task_count": int(task_count),
        "non_routing_refusal_task_count": int(skipped.get("empty_action_task", 0)),
        "non_routing_refusal_examples_by_domain": {
            domain: examples[:5] for domain, examples in sorted(refusal_examples_by_domain.items())
        },
        "action_step_count": int(action_step_count),
        "tool_action_row_count": int(action_step_count),
        "no_tool_row_count": int(skipped.get("empty_action_task", 0)),
        "source_row_count": len(source_rows),
        "skill_count": len(deduped_skills),
        "action_count_by_domain": {domain: len(names) for domain, names in sorted(action_names_by_domain.items())},
        "task_count_by_domain": dict(sorted(task_counts_by_domain.items())),
        "action_step_count_by_domain": dict(sorted(step_counts_by_domain.items())),
        "skipped_reasons": dict(sorted(skipped.items())),
        "history_channel": history_channel,
        "causal_memory_observation_eligible": False,
        "causal_memory_observation_blocker": "oracle action arguments are not environment tool-result observations",
    }
    return Tau2RouteCorpus(skills=deduped_skills, source_rows=source_rows, report=report)


def load_prebuilt_tau2_route_corpus(
    *,
    source_rows_path: str | Path,
    skills_path: str | Path,
    benchmark_name: str = "tau2",
) -> Tau2RouteCorpus:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    source_rows = [
        materialize_history_free_state(row, replace_state_text=True)
        for row in _read_jsonl(source_rows_path)
    ]
    skills = _read_jsonl(skills_path)
    domains = sorted(
        {
            str(row.get("domain") or "").strip()
            for row in source_rows
            if str(row.get("domain") or "").strip()
        }
    )
    report = {
        "benchmark": benchmark_name,
        "source": "prebuilt_jsonl",
        "source_rows_path": str(source_rows_path),
        "skills_path": str(skills_path),
        "domains": domains,
        "task_count": len({str(row.get("trajectory_id") or row.get("task_id") or idx) for idx, row in enumerate(source_rows)}),
        "action_step_count": len(source_rows),
        "source_row_count": len(source_rows),
        "skill_count": len(skills),
        "skipped_reasons": {},
        "history_channel": audit_history_channel_rows(
            source_rows,
            require_explicit_current=True,
        ),
    }
    return Tau2RouteCorpus(skills=skills, source_rows=source_rows, report=report)


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _rank_position(items: list[str], target: str) -> int | None:
    try:
        return items.index(target) + 1
    except ValueError:
        return None


def _ranked_positive_fields(
    row: dict[str, Any],
    candidates: list[str],
    skill_id_to_idx: dict[str, int],
) -> tuple[list[str], list[int], list[int]]:
    ranked = sorted(
        (
            (candidates.index(skill_id), skill_id)
            for skill_id in row_positive_skill_ids(row)
            if skill_id in skill_id_to_idx and skill_id in candidates
        ),
        key=lambda item: (item[0], item[1]),
    )
    return (
        [skill_id for _position, skill_id in ranked],
        [int(skill_id_to_idx[skill_id]) for _position, skill_id in ranked],
        [int(position) for position, _skill_id in ranked],
    )


def rank_tau2_candidates_with_stage0_prior(
    model: Any,
    rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"ranked_rows": 0}
    device = torch.device(device or _model_device(model))
    ranked_rows: list[dict[str, Any]] = []
    positive_ranks: list[int] = []
    missing_candidates = 0
    skipped_missing_skill = 0
    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "eval", None)):
        model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            encoded = model.encode_observations([router_state_text(row) for row in batch])
            encoded = encoded.to(device)
            logits = model.skill_table.retrieval_logits(encoded).detach().float().cpu()
            for row_offset, row in enumerate(batch):
                candidates = [str(item) for item in row.get("candidate_next_skill_ids") or []]
                scored: list[tuple[str, float]] = []
                for candidate_id in candidates:
                    if candidate_id not in skill_id_to_idx:
                        skipped_missing_skill += 1
                        continue
                    scored.append((candidate_id, float(logits[row_offset, int(skill_id_to_idx[candidate_id])].item())))
                scored.sort(key=lambda item: (-item[1], item[0]))
                copied = dict(row)
                copied["candidate_next_skill_ids"] = [item[0] for item in scored]
                copied["candidate_next_skill_indices"] = [int(skill_id_to_idx[item[0]]) for item in scored]
                copied["candidate_next_prior_scores"] = [float(item[1]) for item in scored]
                positive_ids, positive_indices, positive_positions = _ranked_positive_fields(
                    copied,
                    copied["candidate_next_skill_ids"],
                    skill_id_to_idx,
                )
                rank = positive_positions[0] + 1 if positive_positions else None
                if rank is None:
                    missing_candidates += 1
                else:
                    positive_ranks.append(rank)
                    copied["positive_next_skill_ids"] = positive_ids
                    copied["positive_next_skill_indices"] = positive_indices
                    copied["positive_next_skill_positions"] = positive_positions
                    copied["positive_next_skill_position"] = positive_positions[0]
                    copied["positive_next_skill_idx"] = positive_indices[0]
                copied["stage0_candidate_prior_report"] = {
                    "positive_stage0_domain_rank": rank,
                    "candidate_count": len(scored),
                    "stage0_prior_source": "domain_local_stage0_logits",
                }
                ranked_rows.append(copied)
    if was_training and callable(getattr(model, "train", None)):
        model.train()
    ranked_rows_with_positive = len(positive_ranks)
    report = {
        "ranked_rows": len(ranked_rows),
        "ranked_rows_with_positive": ranked_rows_with_positive,
        "positive_missing_rows": int(missing_candidates),
        "candidate_skill_missing_count": int(skipped_missing_skill),
        "positive_stage0_domain_recall@1": _recall_at(positive_ranks, 1),
        "positive_stage0_domain_recall@5": _recall_at(positive_ranks, 5),
        "positive_stage0_domain_mrr": _mrr(positive_ranks),
        "candidate_count_mean": _mean([len(row.get("candidate_next_skill_ids") or []) for row in ranked_rows]),
    }
    return ranked_rows, report


def _recall_at(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank <= int(k)) / len(ranks)


def _mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / float(rank) for rank in ranks) / len(ranks)


def _mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _numeric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) - float(baseline[key])
        for key, value in current.items()
        if isinstance(value, (int, float)) and isinstance(baseline.get(key), (int, float))
    }


def build_tau2_full_clstr_route_report(
    *,
    benchmark_name: str = "tau2",
    output_dir: str | Path,
    data_root: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    skills_path: str | Path,
    source_eval_rows: int,
    retained_eval_rows: int,
    corpus_report: dict[str, Any],
    route_data_report: dict[str, Any],
    memory_report: dict[str, Any],
    stage0_prior_eval: dict[str, Any],
    base_eval: dict[str, Any],
    stage4_eval: dict[str, Any],
    config: dict[str, Any],
    stage0_prior_report: dict[str, Any] | None = None,
    base_eval_by_benchmark: dict[str, Any] | None = None,
    stage4_eval_by_benchmark: dict[str, Any] | None = None,
    mt_ablation_eval: dict[str, Any] | None = None,
    model_load: dict[str, Any] | None = None,
    candidate_recall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    strict_prior = strict_stage4_metrics(
        stage0_prior_eval,
        retained_rows=retained_eval_rows,
        source_rows=source_eval_rows,
    )
    strict_base = strict_stage4_metrics(
        base_eval,
        retained_rows=retained_eval_rows,
        source_rows=source_eval_rows,
    )
    strict_stage4 = strict_stage4_metrics(
        stage4_eval,
        retained_rows=retained_eval_rows,
        source_rows=source_eval_rows,
    )
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    if int(route_data_report.get("positive_injected_rows") or 0) > 0:
        blockers.append("gold_positive_injected")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": benchmark_name,
        "stage4_route": "domain_local_stage0_ranked_logged_online_memory",
        "route_scorer": str(config.get("route_scorer") or UNIFIED_MEMORY_ROUTE_SCORER),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "skills_path": str(skills_path),
        "source_eval_rows": int(source_eval_rows),
        "retained_eval_rows": int(retained_eval_rows),
        "corpus_report": corpus_report,
        "stage0_prior_report": stage0_prior_report or {},
        "route_data_report": route_data_report,
        "memory_report": memory_report,
        "stage0_prior_eval": stage0_prior_eval,
        "base_eval": base_eval,
        "stage4_eval": stage4_eval,
        "base_eval_by_benchmark": base_eval_by_benchmark or {},
        "stage4_eval_by_benchmark": stage4_eval_by_benchmark or {},
        "mt_ablation_eval": mt_ablation_eval or {},
        "delta_base_vs_stage0_prior": _numeric_delta(base_eval, stage0_prior_eval),
        "delta_stage4_vs_base": _numeric_delta(stage4_eval, base_eval),
        "delta_stage4_vs_stage0_prior": _numeric_delta(stage4_eval, stage0_prior_eval),
        "strict": {
            "stage0_prior": strict_prior,
            "base": strict_base,
            "stage4": strict_stage4,
        },
        "metric_contract": strict_metric_contract(),
        "strict_delta_base_vs_stage0_prior": _numeric_delta(strict_base, strict_prior),
        "strict_delta_stage4_vs_base": _numeric_delta(strict_stage4, strict_base),
        "strict_delta_stage4_vs_stage0_prior": _numeric_delta(strict_stage4, strict_prior),
        "candidate_recall": candidate_recall or {},
        "config": config,
        "model_load": model_load or {},
        "paper_scope_note": (
            f"{benchmark_name}/tau-bench is evaluated as a domain-local, state/history-aware next-tool routing benchmark. "
            "It is auxiliary evidence for multi-step routing, not a large global skill-pool headline."
        ),
    }


def _stage4_rows_from_ranked_tau2(
    source_rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    candidate_count: int | None,
    benchmark_name: str = "tau2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    rows = []
    skipped: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for row in source_rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or [] if str(item) in skill_id_to_idx]
        if candidate_count is not None:
            candidates = candidates[: max(1, int(candidate_count))]
        next_skill_id = str(row.get("next_skill_id") or "")
        current_skill_id = str(row.get("skill_id") or "")
        if not current_skill_id or current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if not row_positive_skill_ids(row):
            skipped["next_skill_not_in_pool"] += 1
            continue
        positive_ids, positive_indices, positive_positions = _ranked_positive_fields(
            row,
            candidates,
            skill_id_to_idx,
        )
        if not positive_positions:
            skipped["next_positive_missing_from_candidates"] += 1
            continue
        positive_pos = positive_positions[0]
        prior_scores = list(row.get("candidate_next_prior_scores") or [])
        if len(prior_scores) != len(row.get("candidate_next_skill_ids") or []):
            prior_scores = [-float(idx) for idx in range(len(row.get("candidate_next_skill_ids") or []))]
        prior_scores = prior_scores[: len(candidates)]
        rows.append(
            {
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index"),
                "benchmark": benchmark_name,
                "source_benchmark": benchmark_name,
                "state_text": router_state_text(row),
                "state_text_current": router_state_text(row),
                "state_text_full": str(row.get("state_text_full") or row.get("state_text") or ""),
                "history_text": str(row.get("history_text") or ""),
                "action_text": str(row.get("action_text") or ""),
                "next_observation_text": str(row.get("next_observation_text") or ""),
                "observation_source": str(row.get("observation_source") or ""),
                "skill_id": current_skill_id,
                "next_skill_id": next_skill_id,
                "equivalent_next_skill_ids": [
                    skill_id
                    for skill_id in row_positive_skill_ids(row)
                    if skill_id != next_skill_id
                ],
                "skill_idx": int(skill_id_to_idx[current_skill_id]),
                "positive_next_skill_ids": positive_ids,
                "positive_next_skill_indices": positive_indices,
                "positive_next_skill_positions": positive_positions,
                "positive_next_skill_idx": positive_indices[0],
                "positive_next_skill_position": int(positive_pos),
                "candidate_next_skill_ids": candidates,
                "candidate_pool_protocol": "benchmark_local",
                "candidate_next_skill_indices": [int(skill_id_to_idx[item]) for item in candidates],
                "candidate_next_prior_scores": [float(value) for value in prior_scores],
                "positive_injected": False,
                "provenance": {
                    "source": f"{benchmark_name}_domain_local_stage0_ranked",
                    "original_provenance": row.get("provenance") or {},
                    "candidate_count_requested": candidate_count,
                },
            }
        )
        candidate_counts.append(len(candidates))
    return rows, {
        "source_rows": len(source_rows),
        "stage4_rows": len(rows),
        "positive_injected_rows": 0,
        "candidate_source": f"{benchmark_name}_domain_local_stage0_ranked",
        "candidate_count_requested": candidate_count,
        "candidate_count_mean": _mean(candidate_counts),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _stage0_prior_eval_from_ranked_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks: list[int] = []
    candidate_counts: list[int] = []
    for row in rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or []]
        candidate_counts.append(len(candidates))
        positions = [
            candidates.index(skill_id) + 1
            for skill_id in row_positive_skill_ids(row)
            if skill_id in candidates
        ]
        rank = min(positions) if positions else None
        if rank is not None:
            ranks.append(rank)
    return {
        "stage4_act_count": float(len(rows)),
        "stage4_next_skill_recall@1": _recall_at(ranks, 1),
        "stage4_next_skill_recall@5": _recall_at(ranks, 5),
        "stage4_next_skill_mrr": _mrr(ranks),
        "stage4_candidate_count": _mean(candidate_counts),
    }


def _attach_tau2_action_only_replay_prefixes(
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay executed Tau actions without pretending oracle args are results."""

    max_steps = max(0, int(max_steps))
    prepared = [dict(row) for row in rows]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, row in enumerate(rows):
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if trajectory_id:
            grouped.setdefault(trajectory_id, []).append((row_index, row))

    rows_with_prefix = 0
    total_prefix_steps = 0
    for indexed_rows in grouped.values():
        ordered = sorted(
            indexed_rows,
            key=lambda item: (int(item[1].get("step_index") or 0), item[0]),
        )
        prior_events: list[dict[str, Any]] = []
        previous_step: int | None = None
        for row_index, row in ordered:
            step_index = int(row.get("step_index") or 0)
            if previous_step is None or step_index != previous_step + 1:
                prior_events = []
            if prior_events and max_steps > 0:
                copied = dict(row)
                copied["replay_prefix"] = [
                    dict(step) for step in prior_events[-max_steps:]
                ]
                prepared[row_index] = copied
                rows_with_prefix += 1
                total_prefix_steps += len(copied["replay_prefix"])

            executed_skill_id = str(row.get("next_skill_id") or "").strip()
            if executed_skill_id not in skill_id_to_idx:
                raise ValueError(
                    f"Tau2 action replay skill is absent from model pool: {executed_skill_id}"
                )
            action_name = executed_skill_id.rsplit("/", 1)[-1]
            arguments = str(row.get("next_observation_text") or "").strip()
            prior_events.append(
                {
                    "observation_text": router_state_text(row),
                    "action_text": f"executed_tool: {action_name}\n{arguments}",
                    "next_observation_text": "",
                    "skill_id": executed_skill_id,
                    "skill_idx": int(skill_id_to_idx[executed_skill_id]),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "step_index": step_index,
                    "observation_source": "action_only_no_tool_result",
                    "replay_protocol": "tau2_oracle_action_only_v1",
                }
            )
            previous_step = step_index

    return prepared, {
        "mode": "action_only",
        "protocol": "tau2_oracle_action_only_v1",
        "max_steps": max_steps,
        "row_count": len(rows),
        "trajectory_count": len(grouped),
        "rows_with_replay_prefix": rows_with_prefix,
        "total_prefix_steps": total_prefix_steps,
        "actual_tool_result_observations": False,
        "eligible_claim": "action_history_memory_only",
    }


def run_tau2_full_clstr_route_eval(
    *,
    benchmark_name: str = "tau2",
    data_root: str | Path,
    prebuilt_source_rows_path: str | Path | None = None,
    prebuilt_skills_path: str | Path | None = None,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None = None,
    training_skills_path: str | Path | None = None,
    output_dir: str | Path,
    domains: Iterable[str] | None = None,
    task_split: str = "full",
    max_tasks_per_domain: int | None = None,
    max_eval_rows: int | None = None,
    candidate_count: int | None = None,
    batch_size: int = 8,
    stage0_candidate_batch_size: int = 16,
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.25,
    transition_scoring_mode: str = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    expected_memory_utility_gate_checkpoint_sha256: str | None = None,
    expected_memory_utility_gate_audit_sha256: str | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    write_mt_ablation_report: bool = False,
    mt_ablation_auto_replay_prefix_max_steps: int = 3,
    include_mt_effect_diagnostics: bool = False,
) -> dict[str, Any]:
    benchmark_name = _clean_benchmark_name(benchmark_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    if prebuilt_source_rows_path is not None or prebuilt_skills_path is not None:
        if prebuilt_source_rows_path is None or prebuilt_skills_path is None:
            raise ValueError("prebuilt tau2 eval requires both prebuilt_source_rows_path and prebuilt_skills_path")
        corpus = load_prebuilt_tau2_route_corpus(
            source_rows_path=prebuilt_source_rows_path,
            skills_path=prebuilt_skills_path,
            benchmark_name=benchmark_name,
        )
    else:
        corpus = load_tau2_route_corpus(
            data_root,
            domains=domains,
            max_tasks_per_domain=max_tasks_per_domain,
            benchmark_name=benchmark_name,
            task_split=task_split,
        )
    all_source_rows = list(corpus.source_rows)
    no_tool_rows = [
        row for row in all_source_rows if str(row.get("route_target") or "") == "STOP"
    ]
    source_rows = [
        row for row in all_source_rows if str(row.get("route_target") or "TOOL") != "STOP"
    ]
    if max_eval_rows is not None:
        source_rows = source_rows[: max(0, int(max_eval_rows))]
    skills_path = output_dir / f"{benchmark_name}_skill_pool.jsonl"
    source_rows_path = output_dir / f"{benchmark_name}_source_rows.jsonl"
    no_tool_rows_path = output_dir / f"{benchmark_name}_no_tool_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    _write_jsonl(no_tool_rows_path, no_tool_rows)
    progress_path = output_dir / f"{benchmark_name}_progress.json"
    ranked_rows_path = output_dir / f"{benchmark_name}_ranked_rows.jsonl"
    stage0_prior_report_path = output_dir / f"{benchmark_name}_stage0_prior_report.json"
    stage4_rows_path = output_dir / f"{benchmark_name}_stage4_rows.jsonl"
    route_data_report_path = output_dir / f"{benchmark_name}_route_data_report.json"
    scored_rows_path = output_dir / f"{benchmark_name}_scored_rows.jsonl"
    memory_report_path = output_dir / f"{benchmark_name}_memory_report.json"
    base_eval_path = output_dir / f"{benchmark_name}_base_eval.json"
    stage4_eval_path = output_dir / f"{benchmark_name}_stage4_eval.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skill_pool_adapter_report = None
    if training_skills_path is not None:
        (
            model,
            model_config,
            skill_id_to_idx,
            skill_pool_adapter_report,
        ) = restore_native_benchmark_checkpoint_chain(
            stage0_checkpoint_path=stage0_checkpoint_path,
            stage2_checkpoint_path=stage2_checkpoint_path,
            stage4_checkpoint_path=stage4_checkpoint_path,
            training_skills_path=training_skills_path,
            benchmark_skills=corpus.skills,
            model_cache_dir=output_dir / "model_cache",
            device=device,
            require_safe_memory_delta=True,
        )
        routing_report = dict(skill_pool_adapter_report["stage0"])
        stage2_load_report = dict(skill_pool_adapter_report["stage2"])
        stage4_load_report = skill_pool_adapter_report.get("stage4")
    else:
        skill_id_to_idx = {
            str(_skill_id(skill, idx)): idx for idx, skill in enumerate(corpus.skills)
        }
        model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
            checkpoint_path=Path(stage0_checkpoint_path),
            skills_path=skills_path,
            model_cache_dir=output_dir / "model_cache",
        )
        stage2_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage2_checkpoint_path),
            partial_load_mode="stage2_checkpoint_compatible_state",
        )
        stage4_load_report = None
        if stage4_checkpoint_path:
            stage4_load_report = load_head_checkpoint_into_model(
                model,
                Path(stage4_checkpoint_path),
                partial_load_mode="stage4_checkpoint_compatible_state",
            )
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    memory_utility_gate, reliability_gate_report = resolve_reliability_gate(
        reliability_mode=reliability_mode,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
        expected_gate_sha256=expected_memory_utility_gate_checkpoint_sha256,
        expected_audit_sha256=expected_memory_utility_gate_audit_sha256,
        device=device,
    )
    if reliability_mode == "learned":
        feature_update_count_cap = float(reliability_gate_report["feature_update_count_cap"])
        feature_candidate_count_cap = float(reliability_gate_report["feature_candidate_count_cap"])

    _write_json(
        progress_path,
        {
            "status": "running",
            "benchmark": benchmark_name,
            "stage": "stage0_ranking",
            "output_dir": str(output_dir),
        },
    )
    cached_stage0_report = _read_json_if_exists(stage0_prior_report_path)
    if (
        ranked_rows_path.exists()
        and cached_stage0_report is not None
        and cached_stage0_report.get("router_state_contract")
        == ROUTER_STATE_CONTRACT
    ):
        ranked_rows = _read_jsonl(ranked_rows_path)
        stage0_prior_report = cached_stage0_report
    else:
        ranked_rows, stage0_prior_report = rank_tau2_candidates_with_stage0_prior(
            model,
            source_rows,
            skill_id_to_idx,
            batch_size=stage0_candidate_batch_size,
            device=device,
        )
        stage0_prior_report["router_state_contract"] = ROUTER_STATE_CONTRACT
        _write_jsonl(ranked_rows_path, ranked_rows)
        _write_json(stage0_prior_report_path, stage0_prior_report)

    _write_json(
        progress_path,
        {
            "status": "running",
            "benchmark": benchmark_name,
            "stage": "stage4_rows",
            "ranked_rows": len(ranked_rows),
            "output_dir": str(output_dir),
        },
    )
    cached_route_report = _read_json_if_exists(route_data_report_path)
    if (
        stage4_rows_path.exists()
        and cached_route_report is not None
        and cached_route_report.get("router_state_contract")
        == ROUTER_STATE_CONTRACT
    ):
        stage4_rows = [
            materialize_history_free_state(row, replace_state_text=True)
            for row in _read_jsonl(stage4_rows_path)
        ]
        route_data_report = cached_route_report
    else:
        stage4_rows, route_data_report = _stage4_rows_from_ranked_tau2(
            ranked_rows,
            skill_id_to_idx,
            candidate_count=candidate_count,
            benchmark_name=benchmark_name,
        )
        route_data_report["router_state_contract"] = ROUTER_STATE_CONTRACT
        _write_jsonl(stage4_rows_path, stage4_rows)
        _write_json(route_data_report_path, route_data_report)

    replay_rows, causal_replay_report = _attach_tau2_action_only_replay_prefixes(
        stage4_rows,
        skill_id_to_idx=skill_id_to_idx,
        max_steps=mt_ablation_auto_replay_prefix_max_steps,
    )

    _write_json(
        progress_path,
        {
            "status": "running",
            "benchmark": benchmark_name,
            "stage": "online_memory_scores",
            "stage4_rows": len(stage4_rows),
            "output_dir": str(output_dir),
        },
    )
    cached_memory_report = _read_json_if_exists(memory_report_path)
    if (
        scored_rows_path.exists()
        and cached_memory_report is not None
        and cached_memory_report.get("causal_replay_protocol")
        == causal_replay_report["protocol"]
        and cached_memory_report.get("router_state_contract")
        == ROUTER_STATE_CONTRACT
    ):
        scored_rows = _read_jsonl(scored_rows_path)
        memory_report = cached_memory_report
    else:
        scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
            replay_rows,
            feedback_rows=replay_rows,
            next_skill_bonus=online_memory_next_skill_bonus,
            exact_transition_bonus=online_memory_exact_transition_bonus,
            memory_mode=online_memory_mode,
        )
        memory_report = {
            **memory_report,
            "causal_replay_protocol": causal_replay_report["protocol"],
            "causal_replay": causal_replay_report,
            "router_state_contract": ROUTER_STATE_CONTRACT,
        }
        _write_jsonl(scored_rows_path, scored_rows)
        _write_json(memory_report_path, memory_report)
    local_pool_size = declared_candidate_pool_size(ranked_rows)
    local_static_k = (
        local_pool_size
        if candidate_count is None
        else min(local_pool_size, max(0, int(candidate_count)))
    )
    candidate_eval_kwargs = {
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "skill_id_to_idx": skill_id_to_idx,
        "static_k": local_static_k,
        "dynamic_extra_k": 64,
        "final_k": max(1, local_static_k),
    }
    stage0_prior_eval = _stage0_prior_eval_from_ranked_rows(stage4_rows)
    base_eval = _read_json_if_exists(base_eval_path)
    if (
        base_eval is None
        or base_eval.get("causal_replay_protocol")
        != causal_replay_report["protocol"]
        or base_eval.get("router_state_contract") != ROUTER_STATE_CONTRACT
    ):
        base_eval = evaluate_logged_online_stage4_rows(
            model,
            scored_rows,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=0.0,
            score_calibrator_enabled=False,
            route_scorer=route_scorer,
            reliability_mode="static",
            fixed_alpha=1.0,
            auto_replay_prefix_max_steps=0,
            progress_path=output_dir / f"{benchmark_name}_base_eval_progress.json",
            progress_label=f"{benchmark_name}_base_eval",
            **candidate_eval_kwargs,
        )
        base_eval["causal_replay_protocol"] = causal_replay_report["protocol"]
        base_eval["router_state_contract"] = ROUTER_STATE_CONTRACT
        _write_json(base_eval_path, base_eval)
    base_eval_by_benchmark = {benchmark_name: dict(base_eval)}
    stage4_eval = _read_json_if_exists(stage4_eval_path)
    if (
        stage4_eval is None
        or stage4_eval.get("causal_replay_protocol")
        != causal_replay_report["protocol"]
        or stage4_eval.get("router_state_contract") != ROUTER_STATE_CONTRACT
    ):
        stage4_eval = evaluate_logged_online_stage4_rows(
            model,
            scored_rows,
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=online_memory_weight,
            score_calibrator_enabled=True,
            route_scorer=route_scorer,
            reliability_mode=reliability_mode,
            fixed_alpha=fixed_alpha,
            memory_utility_gate=memory_utility_gate,
            feature_update_count_cap=feature_update_count_cap,
            feature_candidate_count_cap=feature_candidate_count_cap,
            auto_replay_prefix_max_steps=0,
            progress_path=output_dir / f"{benchmark_name}_stage4_eval_progress.json",
            progress_label=f"{benchmark_name}_stage4_eval",
            **candidate_eval_kwargs,
        )
        stage4_eval["causal_replay_protocol"] = causal_replay_report["protocol"]
        stage4_eval["router_state_contract"] = ROUTER_STATE_CONTRACT
        _write_json(stage4_eval_path, stage4_eval)
    stage4_eval_by_benchmark = {benchmark_name: dict(stage4_eval)}
    mt_ablation_eval = None
    if write_mt_ablation_report:
        mt_ablation_eval = evaluate_mt_ablation_rows(
            model,
            scored_rows,
            source_rows=len(source_rows),
            retained_rows=len(stage4_rows),
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=online_memory_weight,
            auto_replay_prefix_max_steps=mt_ablation_auto_replay_prefix_max_steps,
            route_scorer=route_scorer,
            include_pairwise_effect_diagnostics=include_mt_effect_diagnostics,
        )
    candidate_recall = candidate_recall_protocol_metadata(
        pool_protocol="benchmark_local",
        candidate_source=str(
            route_data_report.get("candidate_source")
            or f"{benchmark_name}_domain_local_stage0_ranked"
        ),
        legal_pool_size=local_pool_size,
        static_k=local_static_k,
        dynamic_extra_k=64,
        final_k=max(1, local_static_k),
        causal_sequential=source_rows_have_causal_sequence(source_rows),
    )
    report = build_tau2_full_clstr_route_report(
        benchmark_name=benchmark_name,
        output_dir=output_dir,
        data_root=data_root,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        skills_path=skills_path,
        source_eval_rows=len(source_rows),
        retained_eval_rows=len(stage4_rows),
        corpus_report=corpus.report,
        route_data_report=route_data_report,
        memory_report=memory_report,
        stage0_prior_eval=stage0_prior_eval,
        base_eval=base_eval,
        stage4_eval=stage4_eval,
        base_eval_by_benchmark=base_eval_by_benchmark,
        stage4_eval_by_benchmark=stage4_eval_by_benchmark,
        mt_ablation_eval=mt_ablation_eval,
        stage0_prior_report=stage0_prior_report,
        config={
            "benchmark_name": benchmark_name,
            "router_state_contract": ROUTER_STATE_CONTRACT,
            "candidate_mode": "domain_local_stage0_ranked",
            "domains": None if domains is None else list(domains),
            "task_split": str(task_split),
            "no_tool_row_count": len(no_tool_rows),
            "no_tool_rows_path": str(no_tool_rows_path),
            "max_tasks_per_domain": max_tasks_per_domain,
            "max_eval_rows": max_eval_rows,
            "candidate_count": candidate_count,
            "batch_size": int(batch_size),
            "stage0_candidate_batch_size": int(stage0_candidate_batch_size),
            "online_memory_mode": online_memory_mode,
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
            "route_scorer": str(route_scorer),
            "reliability_mode": reliability_mode,
            "fixed_alpha": float(fixed_alpha),
            "reliability_gate": reliability_gate_report,
            "write_mt_ablation_report": bool(write_mt_ablation_report),
            "mt_ablation_auto_replay_prefix_max_steps": int(mt_ablation_auto_replay_prefix_max_steps),
            "causal_replay": causal_replay_report,
            "include_mt_effect_diagnostics": bool(include_mt_effect_diagnostics),
            "prebuilt_source_rows_path": None if prebuilt_source_rows_path is None else str(prebuilt_source_rows_path),
            "prebuilt_skills_path": None if prebuilt_skills_path is None else str(prebuilt_skills_path),
            "training_skills_path": None if training_skills_path is None else str(training_skills_path),
        },
        model_load={
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
            "skill_pool_adapter": skill_pool_adapter_report or {},
        },
        candidate_recall=candidate_recall,
    )
    _write_json(output_dir / f"{benchmark_name}_full_clstr_route_eval_report.json", report)
    if benchmark_name == "tau2":
        _write_json(output_dir / "tau2_full_clstr_route_eval_report.json", report)
    return report
