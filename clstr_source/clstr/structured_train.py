from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clstr.external_data import write_json, write_jsonl
from clstr.full_base_preprocess import _history_text, _infer_skill_id, _safe_int, _state_text
from clstr.full_base_data import allowed_loss_mask
from clstr.full_base_train import run_legacy_clstr_full_base_train_from_routing_init
from clstr.qwen_planner_intent import build_structured_planner_state_text


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _skill_ids(skills: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("skill_id")) for row in skills if row.get("skill_id")}


def convert_teacher_rollout_to_full_base_rows(
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
        if not bool(row.get("usable_for_policy")):
            excluded["unusable_for_policy"] += 1
            continue
        if bool(row.get("fallback_used")) or str(row.get("parse_status")) == "fallback":
            excluded["fallback_or_invalid_parse"] += 1
            continue
        candidates = [str(item) for item in row.get("admissible_commands_t") or []]
        action = str(row.get("teacher_action_t") or row.get("expert_action_t") or "")
        if action not in candidates:
            excluded["action_not_in_admissible"] += 1
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
            action = str(row.get("teacher_action_t") or row.get("expert_action_t") or "")
            goal = str(row.get("goal_text") or "")
            task = str(row.get("task_type") or "")
            observation = str(row.get("observation_t") or "")
            history = _history_text(row.get("history_t"))
            proposed = str(row.get("qwen_action_t") or row.get("teacher_action_t") or action)
            skill_id = _infer_skill_id("alfworld", action, known_skill_ids)
            next_action = None
            next_skill_id = None
            if pos + 1 < len(usable):
                next_action = str(usable[pos + 1].get("teacher_action_t") or usable[pos + 1].get("expert_action_t") or "") or None
                next_skill_id = _infer_skill_id("alfworld", next_action, known_skill_ids) if next_action else None
            base_state = _state_text(goal, task, observation, history)
            state_text = build_structured_planner_state_text(base_state, candidates, proposed)
            done = bool(row.get("done_t")) if row.get("done_t") is not None else None
            mask = allowed_loss_mask(
                ["L_policy", "L_trans", "L_trans_skill_ce", "belief", "STOP", "routing"],
                candidates,
                action,
                row.get("next_observation_t"),
                done,
                skill_id,
                history,
                next_skill_id=next_skill_id,
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
                "action_text": action,
                "admissible_actions": candidates,
                "expert_action": action,
                "planner_proposed_action": proposed,
                "planner_parse_status": row.get("parse_status"),
                "planner_fallback_used": bool(row.get("fallback_used")),
                "next_action_text": next_action,
                "next_skill_id": next_skill_id,
                "next_observation_text": row.get("next_observation_t"),
                "done": done,
                "reward": row.get("reward_t"),
                "skill_id": skill_id,
                "loss_mask": mask,
                "source_quality": "verified_teacher_rollout",
                "candidate_source": "official_alfworld_admissible_commands",
                "on_policy_rollout": True,
                "m_t_source": "verified_teacher_rollout_prefix_or_observation",
                "provenance": {
                    "source_dataset": "qwen3_verified_teacher_rollout",
                    "bucket": "verified_teacher_rollout",
                    "split": "train",
                    "gamefile": row.get("gamefile"),
                    "expert_action_source": row.get("expert_action_source"),
                },
            }
            converted.append(record)
            prefix.append(
                {
                    "step_index": record["step_index"],
                    "observation_text": observation,
                    "action_text": action,
                    "next_observation_text": row.get("next_observation_t"),
                    "skill_id": skill_id,
                }
            )

    report = {
        "status": "ok",
        "source_path": str(rollout_path),
        "source_split": "train",
        "bucket": "verified_teacher_rollout",
        "converted_count": len(converted),
        "input_row_count": len(rows),
        "excluded_counts": dict(sorted(excluded.items())),
        "valid_or_test_used_for_training": False,
        "loss_activation_counts": dict(
            sorted(
                Counter(
                    key
                    for row in converted
                    for key, enabled in (row.get("loss_mask") or {}).items()
                    if enabled
                ).items()
            )
        ),
    }
    return converted, report


def build_structured_train_data(
    rollout_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path = "data/clstr_qwen3_structured_train",
    report_output_dir: str | Path = "outputs/clstr_qwen3_structured_train",
) -> dict[str, Any]:
    skills = _read_jsonl(skills_path)
    rows, report = convert_teacher_rollout_to_full_base_rows(rollout_path, skills)
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
        "bucket": "verified_teacher_rollout",
        "valid_or_test_used_for_training": False,
    }
    write_json(manifest_path, manifest)
    report.update({"train_path": str(train_path), "skills_path": str(out_skills_path), "manifest_path": str(manifest_path)})
    write_json(preprocess_report_path, report)
    return report


def run_structured_train(
    rollout_path: str | Path,
    skills_path: str | Path,
    output_dir: str | Path = "outputs/clstr_qwen3_structured_train",
    data_output_dir: str | Path = "data/clstr_qwen3_structured_train",
    routing_init_manifest: str | Path = "outputs/clstr_native_routing_init/manifest.json",
    max_steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1.0e-4,
    seed: int = 17,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    data_report = build_structured_train_data(
        rollout_path=rollout_path,
        skills_path=skills_path,
        output_dir=data_output_dir,
        report_output_dir=output_dir,
    )
    train_report = run_legacy_clstr_full_base_train_from_routing_init(
        train_path=Path(data_report["train_path"]),
        skills_path=Path(data_report["skills_path"]),
        output_dir=output_dir,
        routing_init_manifest=routing_init_manifest,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_weights=loss_weights
        or {
            "L_policy": 1.0,
            "L_trans": 0.02,
            "L_trans_skill_ce": 0.2,
            "belief": 0.05,
            "STOP": 0.2,
            "routing": 0.2,
        },
    )
    train_report["structured_data_report"] = data_report
    train_report["planner_imitation_auxiliary"] = "qwen_proposed_action_in_state_and_candidate_planner_scores"
    train_report["qwen_direct_baseline_is_reference_only"] = True
    write_json(Path(output_dir) / "structured_train_report.json", train_report)
    return train_report
