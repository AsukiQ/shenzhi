from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clstr.external_data import write_json, write_jsonl
from clstr.full_base_data import allowed_loss_mask
from clstr.full_base_preprocess import _history_text, _infer_skill_id, _safe_int, _state_text
from clstr.qwen_planner_intent import build_structured_planner_state_text


EXTRA_DAGGER_LOSS_KEYS = ("hard_negative_margin", "Q_success")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _skill_ids(skills: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("skill_id")) for row in skills if row.get("skill_id")}


def _candidate_success_labels(candidates: list[str], expert_action: str, hard_negative: str | None, won: bool) -> tuple[list[float], list[float]]:
    labels: list[float] = []
    weights: list[float] = []
    for action in candidates:
        if action == expert_action:
            labels.append(1.0)
            weights.append(1.0)
        elif hard_negative and action == hard_negative:
            labels.append(0.0)
            weights.append(0.5)
        elif won:
            labels.append(0.25)
            weights.append(0.1)
        else:
            labels.append(0.0)
            weights.append(0.1)
    return labels, weights


def convert_expert_corrected_rollout_to_full_base_rows(
    rollout_path: str | Path,
    skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollout_path = Path(rollout_path)
    rows = _read_jsonl(rollout_path)
    known_skill_ids = _skill_ids(skills)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = Counter()
    for row in rows:
        split = str(row.get("split") or "").lower()
        if split != "train":
            excluded[f"excluded_split_{split or 'missing'}"] += 1
            continue
        if str(row.get("bucket") or "") != "dagger_expert_corrected_rollout":
            excluded["excluded_bucket_mismatch"] += 1
            continue
        if not bool(row.get("usable_for_expert_ce")):
            excluded[str(row.get("expert_skip_reason") or "unusable_for_expert_ce")] += 1
            continue
        candidates = [str(item) for item in row.get("admissible_commands_t") or []]
        expert = str(row.get("official_expert_action_t") or row.get("expert_action_t") or "")
        if expert not in candidates:
            excluded["official_expert_action_not_in_admissible"] += 1
            continue
        gamefile = str(row.get("gamefile") or row.get("trajectory_id") or "")
        episode = str(row.get("episode_index") if row.get("episode_index") is not None else "")
        grouped[(gamefile, episode)].append(row)

    converted: list[dict[str, Any]] = []
    for (gamefile, episode), group in sorted(grouped.items(), key=lambda item: item[0]):
        del episode
        usable = sorted(group, key=lambda item: _safe_int(item.get("step_index"), 0))
        prefix: list[dict[str, Any]] = []
        for pos, row in enumerate(usable):
            candidates = [str(item) for item in row.get("admissible_commands_t") or []]
            expert = str(row.get("official_expert_action_t") or row.get("expert_action_t") or "")
            qwen_action = str(row.get("qwen_action_t") or "")
            hard_negative = qwen_action if qwen_action in candidates and qwen_action != expert else None
            goal = str(row.get("goal_text") or "")
            task = str(row.get("task_type") or "")
            observation = str(row.get("observation_t") or "")
            history = _history_text(row.get("history_t"))
            skill_id = _infer_skill_id("alfworld", expert, known_skill_ids)
            next_action = None
            next_skill_id = None
            if pos + 1 < len(usable):
                next_action = str(usable[pos + 1].get("official_expert_action_t") or usable[pos + 1].get("expert_action_t") or "") or None
                next_skill_id = _infer_skill_id("alfworld", next_action, known_skill_ids) if next_action else None
            base_state = _state_text(goal, task, observation, history)
            state_text = build_structured_planner_state_text(base_state, candidates, qwen_action)
            done = bool(row.get("done_t")) if row.get("done_t") is not None else None
            mask = allowed_loss_mask(
                ["L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
                candidates,
                expert,
                row.get("next_observation_t"),
                done,
                skill_id,
                history,
                next_skill_id=next_skill_id,
            )
            mask["hard_negative_margin"] = bool(hard_negative)
            mask["Q_success"] = bool(candidates and expert in candidates)
            q_success_labels, q_success_label_weights = _candidate_success_labels(
                candidates,
                expert,
                hard_negative,
                bool(row.get("won_t")),
            )
            record = {
                "benchmark": "alfworld",
                "task_id": f"{gamefile}:{_safe_int(row.get('step_index'), pos)}",
                "trajectory_id": gamefile,
                "step_index": _safe_int(row.get("step_index"), pos),
                "trajectory_step_count": len(usable),
                "goal_text": goal,
                "task_text": task,
                "state_text": state_text,
                "history_text": history,
                "replay_prefix": list(prefix),
                "action_text": expert,
                "admissible_actions": candidates,
                "expert_action": expert,
                "planner_proposed_action": qwen_action,
                "planner_parse_status": row.get("qwen_action_parse_status") or row.get("parse_status"),
                "planner_fallback_used": bool(row.get("fallback_used")),
                "hard_negative_action": hard_negative,
                "qwen_action_matches_expert": bool(row.get("qwen_action_matches_expert")),
                "q_success_labels": q_success_labels,
                "q_success_label_weights": q_success_label_weights,
                "q_success_label_source": "official_expert_positive_qwen_wrong_hard_negative",
                "next_action_text": next_action,
                "next_skill_id": next_skill_id,
                "next_observation_text": row.get("next_observation_t"),
                "done": done,
                "reward": row.get("reward_t"),
                "skill_id": skill_id,
                "loss_mask": mask,
                "source_quality": "dagger_expert_corrected_rollout",
                "candidate_source": "official_alfworld_admissible_commands",
                "on_policy_rollout": True,
                "m_t_source": "qwen_rollout_prefix_with_official_expert_correction",
                "provenance": {
                    "source_dataset": "qwen3_expert_corrected_rollout",
                    "bucket": "dagger_expert_corrected_rollout",
                    "split": "train",
                    "gamefile": row.get("gamefile"),
                    "expert_action_source": row.get("expert_action_source"),
                    "qwen_action_source": "qwen3_frozen_direct_policy",
                },
            }
            converted.append(record)
            prefix.append(
                {
                    "step_index": record["step_index"],
                    "observation_text": observation,
                    "action_text": expert,
                    "next_observation_text": row.get("next_observation_t"),
                    "skill_id": skill_id,
                }
            )

    loss_counts = Counter(
        key
        for row in converted
        for key, enabled in (row.get("loss_mask") or {}).items()
        if enabled
    )
    report = {
        "status": "ok",
        "source_path": str(rollout_path),
        "source_split": "train",
        "bucket": "dagger_expert_corrected_rollout",
        "converted_count": len(converted),
        "input_row_count": len(rows),
        "excluded_counts": dict(sorted(excluded.items())),
        "valid_or_test_used_for_training": False,
        "loss_activation_counts": dict(sorted(loss_counts.items())),
    }
    return converted, report


def build_dagger_train_data(
    rollout_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path = "data/clstr_dagger_expert_corrected_train",
    report_output_dir: str | Path = "outputs/clstr_dagger_expert_corrected_train",
) -> dict[str, Any]:
    skills = _read_jsonl(skills_path)
    rows, report = convert_expert_corrected_rollout_to_full_base_rows(rollout_path, skills)
    output_dir = Path(output_dir)
    report_output_dir = Path(report_output_dir)
    train_path = output_dir / "train.jsonl"
    out_skills_path = output_dir / "skills.jsonl"
    manifest_path = output_dir / "manifest.json"
    preprocess_report_path = report_output_dir / "preprocess_report.json"
    write_jsonl(train_path, rows)
    write_jsonl(out_skills_path, skills)
    manifest = {
        "status": "ok",
        "train_path": str(train_path),
        "skills_path": str(out_skills_path),
        "row_count": len(rows),
        "source_rollout_path": str(rollout_path),
        "bucket": "dagger_expert_corrected_rollout",
        "valid_or_test_used_for_training": False,
    }
    write_json(manifest_path, manifest)
    report.update({"train_path": str(train_path), "skills_path": str(out_skills_path), "manifest_path": str(manifest_path)})
    write_json(preprocess_report_path, report)
    return report
