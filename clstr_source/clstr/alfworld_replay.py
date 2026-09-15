from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

from clstr.external_data import write_json, write_jsonl


ReplayEnvFactory = Callable[[dict[str, Any], str], Any]


def read_replay_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_yaml_config(official_repo: Path) -> dict[str, Any]:
    config_path = official_repo / "configs" / "eval_config.yaml"
    if config_path.exists():
        import yaml

        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "dataset": {
            "data_path": "",
            "eval_id_data_path": None,
            "eval_ood_data_path": None,
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": "",
            "grammar": "",
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "controller": {"type": "tw"},
        "general": {
            "training_method": "dagger",
            "use_cuda": torch.cuda.is_available(),
            "training": {"batch_size": 1},
            "evaluate": {"run_eval": False, "eval_paths": []},
        },
        "dagger": {
            "action_space": "admissible",
            "training": {"max_nb_steps_per_episode": 50},
        },
    }


def build_alfworld_train_config(
    official_repo: str | Path,
    data_dir: str | Path,
    max_games: int | None = None,
    max_steps: int = 50,
    expert_type: str = "handcoded",
) -> dict[str, Any]:
    official_repo = Path(official_repo)
    data_dir = Path(data_dir)
    config = _load_yaml_config(official_repo)
    config["dataset"]["data_path"] = str(data_dir / "json_2.1.1" / "train")
    config["dataset"]["eval_id_data_path"] = None
    config["dataset"]["eval_ood_data_path"] = None
    config["dataset"]["num_train_games"] = int(max_games) if max_games is not None else -1
    config["dataset"]["num_eval_games"] = -1
    config["logic"]["domain"] = str(data_dir / "logic" / "alfred.pddl")
    config["logic"]["grammar"] = str(data_dir / "logic" / "alfred.twl2")
    config["env"]["type"] = "AlfredTWEnv"
    config["env"]["domain_randomization"] = False
    config["env"]["expert_type"] = expert_type
    config["controller"]["type"] = "tw"
    config["general"]["training_method"] = "dagger"
    config["general"]["use_cuda"] = torch.cuda.is_available()
    config["general"].setdefault("training", {})["batch_size"] = 1
    config["general"].setdefault("evaluate", {})["run_eval"] = False
    config["general"]["evaluate"]["eval_paths"] = []
    config["dagger"]["action_space"] = "admissible"
    config["dagger"].setdefault("training", {})["max_nb_steps_per_episode"] = int(max_steps)
    return config


def _default_env_factory(official_repo: Path) -> ReplayEnvFactory:
    def factory(config: dict[str, Any], train_eval: str):
        if str(official_repo) not in sys.path:
            sys.path.insert(0, str(official_repo))
        from alfworld.agents.environment import get_environment

        return get_environment(config["env"]["type"])(config, train_eval=train_eval)

    return factory


_GOAL_RE = re.compile(r"your task is to:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)


def _extract_goal_text(observation: str, gamefile: str | None) -> str:
    match = _GOAL_RE.search(observation or "")
    if match:
        return " ".join(match.group(1).strip().split())
    if gamefile:
        traj_path = Path(gamefile).parent / "traj_data.json"
        if traj_path.exists():
            try:
                traj = json.loads(traj_path.read_text(encoding="utf-8"))
                anns = traj.get("turk_annotations", {}).get("anns", [])
                if anns and anns[0].get("task_desc"):
                    return str(anns[0]["task_desc"])
            except (OSError, json.JSONDecodeError):
                pass
    return ""


def _task_type_from_gamefile(gamefile: str | None) -> str:
    if not gamefile:
        return ""
    try:
        task_dir = Path(gamefile).parents[1].name
    except IndexError:
        return ""
    return task_dir.split("-", 1)[0]


def _expert_action_from_infos(infos: dict[str, Any], batch_index: int) -> str | None:
    plans = infos.get("extra.expert_plan")
    if not plans or batch_index >= len(plans):
        return None
    plan = plans[batch_index]
    if not plan:
        return None
    action = str(plan[0]).strip()
    return action or None


def _list_for_batch(infos: dict[str, Any], key: str, batch_index: int, default: Any = None) -> Any:
    values = infos.get(key)
    if isinstance(values, (list, tuple)) and batch_index < len(values):
        return values[batch_index]
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_alfworld_policy_replay(
    official_repo: str | Path,
    data_dir: str | Path,
    data_output_dir: str | Path = "data/alfworld_policy_replay",
    report_output_dir: str | Path = "outputs/alfworld_policy_replay",
    max_games: int | None = 100,
    max_steps: int = 50,
    batch_size: int = 1,
    expert_type: str = "handcoded",
    seed: int = 17,
    env_factory: ReplayEnvFactory | None = None,
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
        expert_type=expert_type,
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

    target_games = int(max_games) if max_games is not None else int(getattr(wrapper, "num_games", 1))
    rows: list[dict[str, Any]] = []
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
            attempted_games += batch_n
            episode_had_usable = [False] * batch_n
            for step_index in range(max(1, int(max_steps))):
                candidate_rows = [
                    [str(cmd) for cmd in _list_for_batch(infos, "admissible_commands", idx, [])]
                    for idx in range(batch_n)
                ]
                expert_actions = [_expert_action_from_infos(infos, idx) for idx in range(batch_n)]
                expert_in_admissible = [
                    bool(expert_actions[idx] is not None and expert_actions[idx] in candidate_rows[idx])
                    for idx in range(batch_n)
                ]
                skip_now = False
                execute_actions: list[str] = []
                row_indices_for_step: list[int | None] = []
                for idx in range(batch_n):
                    if not active[idx]:
                        execute_actions.append("look")
                        row_indices_for_step.append(None)
                        continue
                    skip_reason = None
                    if expert_actions[idx] is None:
                        skip_reason = "missing_expert_action"
                    elif not expert_in_admissible[idx]:
                        skip_reason = "expert_action_not_in_admissible"
                    if skip_reason:
                        skipped_reasons[skip_reason] += 1
                        rows.append(
                            {
                                "split": "train",
                                "episode_index": episode_index + idx,
                                "step_index": step_index,
                                "gamefile": gamefiles[idx],
                                "task_type": task_types[idx],
                                "goal_text": goals[idx],
                                "initial_observation": initial_observations[idx],
                                "observation_t": observations[idx],
                                "history_t": list(histories[idx]),
                                "admissible_commands_t": candidate_rows[idx],
                                "expert_action_t": expert_actions[idx],
                                "expert_action_in_admissible": False,
                                "usable_for_policy": False,
                                "skip_reason": skip_reason,
                                "done_t": False,
                                "won_t": bool(_list_for_batch(infos, "won", idx, False)),
                                "goal_condition_success_rate_t": _safe_float(
                                    _list_for_batch(infos, "goal_condition_success_rate", idx, 0.0)
                                ),
                                "next_observation_t": None,
                            }
                        )
                        active[idx] = False
                        execute_actions.append(candidate_rows[idx][0] if candidate_rows[idx] else "look")
                        row_indices_for_step.append(None)
                        skip_now = True
                        continue
                    execute_actions.append(str(expert_actions[idx]))
                    row_indices_for_step.append(len(rows))
                    rows.append(
                        {
                            "split": "train",
                            "episode_index": episode_index + idx,
                            "step_index": step_index,
                            "gamefile": gamefiles[idx],
                            "task_type": task_types[idx],
                            "goal_text": goals[idx],
                            "initial_observation": initial_observations[idx],
                            "observation_t": observations[idx],
                            "history_t": list(histories[idx]),
                            "admissible_commands_t": candidate_rows[idx],
                            "expert_action_t": expert_actions[idx],
                            "expert_action_in_admissible": True,
                            "usable_for_policy": True,
                            "skip_reason": None,
                        }
                    )
                    episode_had_usable[idx] = True
                if skip_now and not any(active):
                    break
                obs_next, _rewards, dones, next_infos = env.step(execute_actions)
                next_observations = [str(item) for item in obs_next]
                for idx, row_index in enumerate(row_indices_for_step):
                    if row_index is None:
                        continue
                    done_value = bool(dones[idx]) if idx < len(dones) else False
                    rows[row_index].update(
                        {
                            "done_t": done_value,
                            "won_t": bool(_list_for_batch(next_infos, "won", idx, False)),
                            "goal_condition_success_rate_t": _safe_float(
                                _list_for_batch(next_infos, "goal_condition_success_rate", idx, 0.0)
                            ),
                            "next_observation_t": next_observations[idx],
                        }
                    )
                    histories[idx].append(execute_actions[idx])
                    active[idx] = not done_value
                infos = next_infos
                observations = next_observations
                if not any(active):
                    break
            successful_games += sum(1 for value in episode_had_usable if value)
            episode_index += batch_n
            if episode_index >= target_games:
                break
    finally:
        if hasattr(env, "close"):
            env.close()

    replay_path = data_output_dir / "train_replay.jsonl"
    manifest_path = data_output_dir / "manifest.json"
    extraction_report_path = report_output_dir / "extraction_report.json"
    write_jsonl(replay_path, rows)

    usable_count = sum(1 for row in rows if bool(row.get("usable_for_policy")))
    extracted_count = len(rows)
    report = {
        "status": "ok",
        "split": "train",
        "train_only": True,
        "valid_test_not_used_for_training": True,
        "uses_alfworld_valid_or_test_for_training": False,
        "official_repo": str(official_repo),
        "data_dir": str(data_dir),
        "train_games_attempted": min(attempted_games, target_games),
        "train_games_successful_replayed": successful_games,
        "extracted_step_count": extracted_count,
        "usable_policy_step_count": usable_count,
        "expert_action_in_admissible_rate": round(float(usable_count / max(1, extracted_count)), 6),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "max_games": max_games,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "expert_type": expert_type,
        "replay_path": str(replay_path),
        "manifest_path": str(manifest_path),
        "valid_or_test_used_for_training": False,
    }
    manifest = {
        "status": "ok",
        "split": "train",
        "replay_path": str(replay_path),
        "extraction_report_path": str(extraction_report_path),
        "row_count": extracted_count,
        "usable_policy_step_count": usable_count,
        "valid_or_test_used_for_training": False,
        "official_repo": str(official_repo),
        "data_dir": str(data_dir),
    }
    write_json(manifest_path, manifest)
    write_json(extraction_report_path, report)
    return report
