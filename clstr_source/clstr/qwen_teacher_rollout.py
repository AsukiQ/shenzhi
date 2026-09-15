from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from clstr.alfworld_eval import build_alfworld_policy_state_text
from clstr.alfworld_replay import (
    ReplayEnvFactory,
    _default_env_factory,
    _extract_goal_text,
    _list_for_batch,
    _safe_float,
    _task_type_from_gamefile,
    build_alfworld_train_config,
)
from clstr.alfworld_expert_labeler import label_official_expert_action
from clstr.external_data import write_json, write_jsonl
from clstr.qwen_direct_policy import QwenDirectAdmissibleActionScorer, QwenDirectPolicyConfig


def _metadata_item(planner: Any, idx: int) -> dict[str, Any]:
    metadata = getattr(planner, "last_metadata", None)
    if isinstance(metadata, list) and idx < len(metadata) and isinstance(metadata[idx], dict):
        return dict(metadata[idx])
    return {}


def extract_qwen_teacher_rollout(
    official_repo: str | Path,
    data_dir: str | Path,
    data_output_dir: str | Path = "data/alfworld_qwen3_expert_corrected_rollout",
    report_output_dir: str | Path = "outputs/alfworld_qwen3_expert_corrected_rollout",
    max_games: int | None = 50,
    max_steps: int = 50,
    batch_size: int = 1,
    seed: int = 17,
    planner: Any | None = None,
    env_factory: ReplayEnvFactory | None = None,
    qwen_model_name_or_path: str | Path = "models/Qwen3-8B",
    cache_dir: str | Path | None = ".cache/huggingface",
    local_files_only: bool = True,
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir = Path(data_dir)
    data_output_dir = Path(data_output_dir)
    report_output_dir = Path(report_output_dir)
    os.environ["ALFWORLD_DATA"] = str(data_dir)
    config = build_alfworld_train_config(
        official_repo=official_repo,
        data_dir=data_dir,
        max_games=max_games,
        max_steps=max_steps,
        expert_type="handcoded",
    )
    factory = env_factory or _default_env_factory(official_repo)
    wrapper = factory(config, "train")
    if hasattr(wrapper, "game_files"):
        wrapper.game_files = sorted(str(path) for path in wrapper.game_files)
        if max_games is not None:
            wrapper.game_files = wrapper.game_files[: int(max_games)]
        wrapper.num_games = len(wrapper.game_files)
    env = wrapper.init_env(batch_size=max(1, int(batch_size)))
    if hasattr(env, "seed"):
        env.seed(seed)
    planner = planner or QwenDirectAdmissibleActionScorer(
        QwenDirectPolicyConfig(
            model_name_or_path=str(qwen_model_name_or_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            fallback_strategy="first_admissible",
            use_chat_template=True,
            enable_thinking=False,
        )
    )

    target_games = int(max_games) if max_games is not None else int(getattr(wrapper, "num_games", 1))
    rows: list[dict[str, Any]] = []
    parse_counter: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    attempted_games = 0
    successful_games = 0
    episode_index = 0
    try:
        while attempted_games < target_games:
            obs, infos = env.reset()
            observations = [str(item) for item in obs]
            batch_n = len(observations)
            histories: list[list[str]] = [[] for _ in range(batch_n)]
            initial_observations = list(observations)
            gamefiles = [
                str(_list_for_batch(infos, "extra.gamefile", idx, f"train/episode-{episode_index + idx}"))
                for idx in range(batch_n)
            ]
            goals = [_extract_goal_text(initial_observations[idx], gamefiles[idx]) for idx in range(batch_n)]
            task_types = [_task_type_from_gamefile(gamefiles[idx]) for idx in range(batch_n)]
            active = [True] * batch_n
            first_mismatch_steps: list[int | None] = [None] * batch_n
            attempted_games += batch_n
            episode_success = [False] * batch_n
            for step_index in range(max(1, int(max_steps))):
                candidate_rows = [
                    [str(cmd) for cmd in _list_for_batch(infos, "admissible_commands", idx, [])]
                    for idx in range(batch_n)
                ]
                state_texts = [
                    build_alfworld_policy_state_text(goals[idx], task_types[idx], observations[idx], histories[idx])
                    for idx in range(batch_n)
                ]
                scores = planner(state_texts, candidate_rows)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.as_tensor(scores, dtype=torch.float32)
                execute_actions: list[str] = []
                row_indices: list[int | None] = []
                for idx in range(batch_n):
                    candidates = candidate_rows[idx]
                    if not active[idx]:
                        execute_actions.append(candidates[0] if candidates else "look")
                        row_indices.append(None)
                        continue
                    expert_label = label_official_expert_action(infos, idx, candidates)
                    metadata = _metadata_item(planner, idx)
                    if candidates:
                        width = min(len(candidates), int(scores.size(1)))
                        chosen_index = int(torch.argmax(scores[idx, :width]).item())
                        chosen_action = str(candidates[chosen_index])
                    else:
                        chosen_index = 0
                        chosen_action = "look"
                    parsed_action = str(metadata.get("parsed_action") or chosen_action)
                    parse_status = str(metadata.get("parse_status") or "")
                    fallback_used = bool(metadata.get("fallback_used", False))
                    parse_counter[parse_status or "unknown"] += 1
                    in_admissible = chosen_action in candidates
                    qwen_matches_expert = bool(
                        expert_label.official_expert_action is not None
                        and chosen_action == expert_label.official_expert_action
                    )
                    if not qwen_matches_expert and first_mismatch_steps[idx] is None:
                        first_mismatch_steps[idx] = step_index
                    hard_negative = (
                        chosen_action
                        if in_admissible and expert_label.official_expert_action is not None and not qwen_matches_expert
                        else None
                    )
                    usable = bool(expert_label.usable_for_expert_ce)
                    if not in_admissible:
                        skipped_reasons["chosen_action_not_in_admissible"] += 1
                    if fallback_used:
                        skipped_reasons["fallback_action"] += 1
                    if not expert_label.usable_for_expert_ce and expert_label.skip_reason:
                        skipped_reasons[str(expert_label.skip_reason)] += 1
                    execute_actions.append(chosen_action)
                    row_indices.append(len(rows))
                    row = {
                            "split": "train",
                            "bucket": "dagger_expert_corrected_rollout",
                            "official_replay": False,
                            "verified_env_replay": True,
                            "episode_index": episode_index + idx,
                            "step_index": step_index,
                            "gamefile": gamefiles[idx],
                            "task_type": task_types[idx],
                            "goal_text": goals[idx],
                            "initial_observation": initial_observations[idx],
                            "observation_t": observations[idx],
                            "history_t": list(histories[idx]),
                            "admissible_commands_t": candidates,
                            "qwen_action_t": chosen_action,
                            "qwen_action_in_admissible": bool(in_admissible),
                            "qwen_action_matches_expert": qwen_matches_expert,
                            "qwen_action_parse_status": parse_status,
                            "hard_negative_action_t": hard_negative,
                            "first_mismatch_step": first_mismatch_steps[idx],
                            "chosen_action_in_admissible": bool(in_admissible),
                            "usable_for_policy": usable,
                            "parse_status": parse_status,
                            "fallback_used": fallback_used,
                            "raw_model_response": str(metadata.get("raw_model_response", "")),
                            "parsed_action": parsed_action,
                            "chosen_index": chosen_index,
                            "candidate_source": "official_alfworld_admissible_commands",
                            "train_allowed": True,
                            "valid_or_test_used_for_training": False,
                            "provenance": {
                                "source_dataset": "qwen3_expert_corrected_rollout",
                                "split": "train",
                                "bucket": "dagger_expert_corrected_rollout",
                                "qwen_model": str(qwen_model_name_or_path),
                            },
                        }
                    row.update(expert_label.to_record())
                    rows.append(row)
                obs_next, rewards, dones, next_infos = env.step(execute_actions)
                next_observations = [str(item) for item in obs_next]
                for idx, row_index in enumerate(row_indices):
                    if row_index is None:
                        continue
                    done_value = bool(dones[idx]) if idx < len(dones) else False
                    reward_value = _safe_float(rewards[idx] if idx < len(rewards) else 0.0)
                    won = bool(_list_for_batch(next_infos, "won", idx, False))
                    rows[row_index].update(
                        {
                            "reward_t": reward_value,
                            "done_t": done_value,
                            "won_t": won,
                            "goal_condition_success_rate_t": _safe_float(
                                _list_for_batch(next_infos, "goal_condition_success_rate", idx, 0.0)
                            ),
                            "next_observation_t": next_observations[idx],
                        }
                    )
                    histories[idx].append(execute_actions[idx])
                    episode_success[idx] = episode_success[idx] or won
                    active[idx] = not done_value
                infos = next_infos
                observations = next_observations
                if not any(active):
                    break
            successful_games += sum(1 for item in episode_success if item)
            episode_index += batch_n
            if episode_index >= target_games:
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    replay_path = data_output_dir / "train_rollout.jsonl"
    manifest_path = data_output_dir / "manifest.json"
    report_path = report_output_dir / "extraction_report.json"
    write_jsonl(replay_path, rows)
    usable_count = sum(1 for row in rows if bool(row.get("usable_for_policy")))
    exact_count = sum(1 for row in rows if str(row.get("parse_status")) != "fallback" and not bool(row.get("fallback_used")))
    expert_usable_count = sum(1 for row in rows if bool(row.get("usable_for_expert_ce")))
    qwen_match_count = sum(1 for row in rows if bool(row.get("qwen_action_matches_expert")))
    report = {
        "status": "ok",
        "split": "train",
        "train_only": True,
        "bucket": "dagger_expert_corrected_rollout",
        "official_replay": False,
        "verified_env_replay": True,
        "uses_alfworld_valid_or_test_for_training": False,
        "valid_or_test_used_for_training": False,
        "official_repo": str(official_repo),
        "data_dir": str(data_dir),
        "qwen_model_name_or_path": str(qwen_model_name_or_path),
        "train_games_attempted": min(attempted_games, target_games),
        "train_games_successful_replayed": successful_games,
        "extracted_step_count": len(rows),
        "usable_policy_step_count": usable_count,
        "usable_expert_ce_step_count": expert_usable_count,
        "expert_action_in_admissible_rate": round(expert_usable_count / max(1, len(rows)), 6),
        "qwen_action_matches_expert_count": qwen_match_count,
        "qwen_action_matches_expert_rate": round(qwen_match_count / max(1, len(rows)), 6),
        "exact_parse_step_count": exact_count,
        "exact_parse_rate": round(exact_count / max(1, len(rows)), 6),
        "fallback_count": sum(1 for row in rows if bool(row.get("fallback_used"))),
        "parse_status_counts": dict(sorted(parse_counter.items())),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "max_games": max_games,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "replay_path": str(replay_path),
        "manifest_path": str(manifest_path),
    }
    manifest = {
        "status": "ok",
        "split": "train",
        "bucket": "dagger_expert_corrected_rollout",
        "official_replay": False,
        "verified_env_replay": True,
        "replay_path": str(replay_path),
        "extraction_report_path": str(report_path),
        "row_count": len(rows),
        "usable_policy_step_count": usable_count,
        "usable_expert_ce_step_count": expert_usable_count,
        "valid_or_test_used_for_training": False,
        "qwen_model_name_or_path": str(qwen_model_name_or_path),
    }
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    return report
