from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clstr.external_data import write_json


ALFWORLD_GOAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "find",
    "in",
    "into",
    "it",
    "of",
    "on",
    "one",
    "put",
    "the",
    "them",
    "then",
    "to",
    "two",
    "with",
}

ALFWORLD_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}


@dataclass(frozen=True)
class TraceRewardConfig:
    success_reward: float = 1.0
    goal_condition_reward: float = 1.0
    goal_progress_reward: float = 0.05
    goal_interaction_reward: float = 0.10
    irrelevant_action_penalty: float = 0.03
    placed_target_reward: float = 0.25
    reverted_target_penalty: float = 0.20
    wrong_receptacle_penalty: float = 0.12


MANIPULATION_VERBS = {
    "take",
    "pick",
    "move",
    "put",
    "place",
    "open",
    "close",
    "examine",
    "cool",
    "heat",
    "clean",
    "slice",
    "toggle",
    "turnon",
    "turnoff",
}

TASK_VERB_HINTS = {
    "pick_cool_then_place_in_recep": ["cool"],
    "pick_heat_then_place_in_recep": ["heat"],
    "pick_clean_then_place_in_recep": ["clean"],
    "pick_and_place_simple": ["put", "place", "move"],
    "pick_two_obj_and_place": ["put", "place", "move"],
    "pick_and_place_with_movable_recep": ["put", "place", "move"],
    "look_at_obj_in_light": ["examine", "look"],
}


def _alfworld_goal_term_weights(goal_text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z]+", str(goal_text).lower())
    weights: dict[str, float] = {}
    for token in tokens:
        if len(token) < 3 or token in ALFWORLD_GOAL_STOPWORDS:
            continue
        weights[token] = max(weights.get(token, 0.0), 0.5)
    if "find" in tokens:
        start = tokens.index("find") + 1
        stop = len(tokens)
        for marker in ("and", "put", "in", "to"):
            if marker in tokens[start:]:
                stop = min(stop, start + tokens[start:].index(marker))
        for token in tokens[start:stop]:
            if len(token) >= 3 and token not in ALFWORLD_GOAL_STOPWORDS:
                weights[token] = max(weights.get(token, 0.0), 2.0)
    for marker in (" in ", " into ", " to "):
        if marker in f" {str(goal_text).lower()} ":
            tail = f" {str(goal_text).lower()} ".rsplit(marker, 1)[-1]
            for token in re.findall(r"[a-z]+", tail):
                if len(token) >= 3 and token not in ALFWORLD_GOAL_STOPWORDS:
                    weights[token] = max(weights.get(token, 0.0), 1.0)
            break
    return weights


def _alfworld_goal_target_count(goal_text: str) -> int:
    tokens = re.findall(r"[a-z0-9]+", str(goal_text).lower())
    for token in tokens:
        if token.isdigit() and int(token) > 0:
            return int(token)
        if token in ALFWORLD_COUNT_WORDS:
            return ALFWORLD_COUNT_WORDS[token]
    return 1


def _alfworld_goal_object_terms(weights: dict[str, float]) -> set[str]:
    return {term for term, weight in weights.items() if float(weight) >= 2.0}


def _alfworld_goal_receptacle_terms(weights: dict[str, float]) -> set[str]:
    return {term for term, weight in weights.items() if 1.0 <= float(weight) < 2.0}


def _normalize_alfworld_entity(text: str) -> str:
    lowered = re.sub(r"\b(the|a|an)\b", " ", str(text).lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _parse_alfworld_object_transfer(action: str) -> tuple[str, str, str] | None:
    lowered = _normalize_alfworld_entity(action)
    for pattern in (
        r"^take (?P<object>.+?) from (?P<location>.+)$",
        r"^pick up (?P<object>.+?) from (?P<location>.+)$",
        r"^(?:move|put|place) (?P<object>.+?) to (?P<location>.+)$",
    ):
        match = re.match(pattern, lowered)
        if not match:
            continue
        kind = "take" if "from" in pattern else "move"
        return kind, match.group("object").strip(), match.group("location").strip()
    return None


def _contains_goal_term(text: str, terms: set[str]) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in terms)


def estimate_alfworld_progress_report(goal_text: str, step_records: list[dict[str, Any]] | None) -> dict[str, float]:
    weights = _alfworld_goal_term_weights(goal_text)
    if not weights:
        return {
            "progress_points": 0.0,
            "goal_interaction_points": 0.0,
            "irrelevant_action_count": 0.0,
            "target_object_goal_count": 1.0,
            "placed_target_count": 0.0,
            "placed_target_event_count": 0.0,
            "reverted_target_count": 0.0,
            "wrong_receptacle_move_count": 0.0,
        }
    goal_object_terms = _alfworld_goal_object_terms(weights)
    goal_receptacle_terms = _alfworld_goal_receptacle_terms(weights)
    target_object_goal_count = _alfworld_goal_target_count(goal_text)
    seen_terms: set[str] = set()
    target_object_locations: dict[str, str] = {}
    progress_points = 0.0
    goal_interaction_points = 0.0
    irrelevant_action_count = 0.0
    placed_target_event_count = 0.0
    reverted_target_count = 0.0
    wrong_receptacle_move_count = 0.0
    manipulation_verbs = ("take ", "pick up ", "put ", "place ", "open ", "close ", "examine ")
    for step in step_records or []:
        action = str(step.get("chosen_action") or step.get("action_text") or "").lower()
        evidence = f"{action}\n{step.get('next_observation_text') or ''}".lower()
        for term, weight in weights.items():
            if term in seen_terms or term not in evidence:
                continue
            seen_terms.add(term)
            progress_points += float(weight)
        if any(verb in f" {action}" for verb in manipulation_verbs):
            if any(term in action for term in goal_object_terms):
                goal_interaction_points += 1.0
            elif not any(term in action for term in weights):
                irrelevant_action_count += 1.0
        transfer = _parse_alfworld_object_transfer(action)
        if transfer is None or not goal_object_terms or not goal_receptacle_terms:
            continue
        kind, obj, location = transfer
        if not _contains_goal_term(obj, goal_object_terms):
            continue
        obj_key = _normalize_alfworld_entity(obj)
        location_is_target = _contains_goal_term(location, goal_receptacle_terms)
        if kind == "take":
            if location_is_target:
                reverted_target_count += 1.0
            target_object_locations[obj_key] = "inventory"
        elif location_is_target:
            if target_object_locations.get(obj_key) != "target":
                placed_target_event_count += 1.0
            target_object_locations[obj_key] = "target"
        else:
            wrong_receptacle_move_count += 1.0
            target_object_locations[obj_key] = "other"
    placed_target_count = min(
        float(target_object_goal_count),
        float(sum(1 for value in target_object_locations.values() if value == "target")),
    )
    return {
        "progress_points": float(progress_points),
        "goal_interaction_points": float(goal_interaction_points),
        "irrelevant_action_count": float(irrelevant_action_count),
        "target_object_goal_count": float(target_object_goal_count),
        "placed_target_count": float(placed_target_count),
        "placed_target_event_count": float(placed_target_event_count),
        "reverted_target_count": float(reverted_target_count),
        "wrong_receptacle_move_count": float(wrong_receptacle_move_count),
    }


def shape_trace_progress_reward(
    success: bool,
    goal_condition_points: float,
    progress: dict[str, float],
    config: TraceRewardConfig | None = None,
) -> float:
    config = config or TraceRewardConfig()
    reward = 0.0
    reward += float(config.success_reward) if bool(success) else 0.0
    reward += float(config.goal_condition_reward) * float(goal_condition_points)
    reward += float(config.goal_progress_reward) * float(progress.get("progress_points", 0.0))
    reward += float(config.goal_interaction_reward) * float(progress.get("goal_interaction_points", 0.0))
    reward -= float(config.irrelevant_action_penalty) * float(progress.get("irrelevant_action_count", 0.0))
    reward += float(config.placed_target_reward) * float(progress.get("placed_target_count", 0.0))
    reward -= float(config.reverted_target_penalty) * float(progress.get("reverted_target_count", 0.0))
    reward -= float(config.wrong_receptacle_penalty) * float(progress.get("wrong_receptacle_move_count", 0.0))
    return float(reward)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _camel_to_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _trial_dir_from_gamefile(gamefile: str | Path | None) -> Path | None:
    if not gamefile:
        return None
    path = Path(str(gamefile))
    if path.name == "game.tw-pddl":
        return path.parent
    if path.is_dir():
        return path
    return path.parent


def _task_goal_from_pddl(task_type: str, pddl_params: dict[str, Any]) -> str:
    obj = _camel_to_compact(str(pddl_params.get("object_target") or "object"))
    parent = _camel_to_compact(str(pddl_params.get("parent_target") or "receptacle"))
    movable = _camel_to_compact(str(pddl_params.get("mrecep_target") or ""))
    toggle = _camel_to_compact(str(pddl_params.get("toggle_target") or ""))
    target_count = "two " if str(task_type) == "pick_two_obj_and_place" else ""
    if task_type == "pick_cool_then_place_in_recep":
        return f"find {target_count}{obj} and cool {obj} with fridge and put it in {parent}"
    if task_type == "pick_heat_then_place_in_recep":
        return f"find {target_count}{obj} and heat {obj} and put it in {parent}"
    if task_type == "pick_clean_then_place_in_recep":
        return f"find {target_count}{obj} and clean {obj} and put it in {parent}"
    if task_type == "pick_and_place_with_movable_recep" and movable:
        return f"find {target_count}{obj} and put it in {movable} then put it in {parent}"
    if task_type == "look_at_obj_in_light":
        light = toggle or parent
        return f"find {obj} and examine it with {light}"
    return f"find {target_count}{obj} and put it in {parent}"


def _task_goal_from_folder(gamefile: str | Path | None) -> str:
    trial_dir = _trial_dir_from_gamefile(gamefile)
    if trial_dir is None:
        return ""
    task_dir = trial_dir.parent.name
    parts = task_dir.split("-")
    if len(parts) < 4:
        return task_dir.replace("_", " ")
    task_type, obj, movable, parent = parts[:4]
    return _task_goal_from_pddl(
        task_type,
        {
            "object_target": obj,
            "mrecep_target": "" if movable == "None" else movable,
            "parent_target": parent,
        },
    )


def resolve_alfworld_goal_text(row: dict[str, Any]) -> str:
    explicit = str(row.get("goal_text") or "").strip()
    if explicit:
        return explicit
    trial_dir = _trial_dir_from_gamefile(row.get("gamefile"))
    if trial_dir is not None:
        traj_path = trial_dir / "traj_data.json"
        if traj_path.exists():
            try:
                data = json.loads(traj_path.read_text(encoding="utf-8"))
                task_type = str(data.get("task_type") or trial_dir.parent.name.split("-", 1)[0])
                pddl_params = data.get("pddl_params") if isinstance(data.get("pddl_params"), dict) else {}
                if pddl_params:
                    return _task_goal_from_pddl(task_type, pddl_params)
            except (OSError, json.JSONDecodeError):
                pass
    return _task_goal_from_folder(row.get("gamefile"))


def _actions(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in row.get("action_trace", []) or row.get("chosen_action_trace", []) or []]


def _step_records(actions: list[str]) -> list[dict[str, Any]]:
    return [{"chosen_action": action, "next_observation_text": ""} for action in actions]


def _has_tail_two_action_cycle(actions: list[str], tail_len: int = 10) -> bool:
    tail = actions[-tail_len:]
    if len(tail) < 6:
        return False
    a, b = tail[-2], tail[-1]
    if a == b:
        return False
    return all(action == (a if idx % 2 == 0 else b) for idx, action in enumerate(tail[-6:]))


def _max_consecutive_repeat(actions: list[str]) -> int:
    if not actions:
        return 0
    best = 1
    run = 1
    previous = actions[0]
    for action in actions[1:]:
        if action == previous:
            run += 1
            best = max(best, run)
        else:
            previous = action
            run = 1
    return best


def _action_verb(action: str) -> str:
    return str(action).strip().lower().split(maxsplit=1)[0] if str(action).strip() else ""


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in terms)


def _row_trace_progress(row: dict[str, Any], method_name: str) -> dict[str, Any]:
    actions = _actions(row)
    goal_text = resolve_alfworld_goal_text(row)
    records = _step_records(actions)
    progress = estimate_alfworld_progress_report(goal_text, records)
    weights = _alfworld_goal_term_weights(goal_text)
    object_terms = _alfworld_goal_object_terms(weights)
    receptacle_terms = _alfworld_goal_receptacle_terms(weights)
    goal_verbs = {verb for verb in MANIPULATION_VERBS if re.search(rf"\b{re.escape(verb)}\b", goal_text.lower())}
    manipulation_count = 0
    goal_object_action_count = 0
    goal_receptacle_action_count = 0
    goal_verb_action_count = 0
    transfer_count = 0
    for action in actions:
        verb = _action_verb(action)
        if verb in MANIPULATION_VERBS:
            manipulation_count += 1
        if _contains_any(action, object_terms):
            goal_object_action_count += 1
        if _contains_any(action, receptacle_terms):
            goal_receptacle_action_count += 1
        if verb in goal_verbs:
            goal_verb_action_count += 1
        if _parse_alfworld_object_transfer(action) is not None:
            transfer_count += 1
    progress_reward = shape_trace_progress_reward(
        success=bool(row.get("success", False)),
        goal_condition_points=float(row.get("goal_condition_points", 0.0) or 0.0),
        progress=progress,
    )
    return {
        "method": method_name,
        "episode_index": row.get("episode_index"),
        "gamefile": row.get("gamefile"),
        "goal_text": goal_text,
        "success": bool(row.get("success", False)),
        "points": float(row.get("points", 0.0) or 0.0),
        "goal_condition_points": float(row.get("goal_condition_points", 0.0) or 0.0),
        "steps": int(float(row.get("steps", len(actions)) or 0.0)),
        "action_count": len(actions),
        "progress_reward": round(float(progress_reward), 6),
        "manipulation_action_count": manipulation_count,
        "transfer_action_count": transfer_count,
        "goal_object_action_count": goal_object_action_count,
        "goal_receptacle_action_count": goal_receptacle_action_count,
        "goal_verb_action_count": goal_verb_action_count,
        "max_consecutive_repeat": _max_consecutive_repeat(actions),
        "tail_two_action_cycle": _has_tail_two_action_cycle(actions),
        **{key: round(float(value), 6) for key, value in progress.items()},
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows)), 6)


def build_trace_progress_report(run_path: str | Path, method_name: str | None = None) -> dict[str, Any]:
    run_path = Path(run_path)
    rows = _read_jsonl(run_path)
    method_name = method_name or (str(rows[0].get("method")) if rows else run_path.parent.name)
    progress_rows = [_row_trace_progress(row, method_name) for row in rows]
    verb_counter: Counter[str] = Counter()
    for row in rows:
        verb_counter.update(_action_verb(action) for action in _actions(row))
    return {
        "status": "ok",
        "method": method_name,
        "run_path": str(run_path),
        "episode_count": len(progress_rows),
        "success_rate": _mean(progress_rows, "success"),
        "mean_points": _mean(progress_rows, "points"),
        "mean_goal_condition_points": _mean(progress_rows, "goal_condition_points"),
        "mean_progress_reward": _mean(progress_rows, "progress_reward"),
        "mean_progress_points": _mean(progress_rows, "progress_points"),
        "mean_goal_interaction_points": _mean(progress_rows, "goal_interaction_points"),
        "mean_irrelevant_action_count": _mean(progress_rows, "irrelevant_action_count"),
        "mean_placed_target_count": _mean(progress_rows, "placed_target_count"),
        "mean_placed_target_event_count": _mean(progress_rows, "placed_target_event_count"),
        "mean_reverted_target_count": _mean(progress_rows, "reverted_target_count"),
        "mean_wrong_receptacle_move_count": _mean(progress_rows, "wrong_receptacle_move_count"),
        "mean_manipulation_action_count": _mean(progress_rows, "manipulation_action_count"),
        "mean_transfer_action_count": _mean(progress_rows, "transfer_action_count"),
        "mean_goal_object_action_count": _mean(progress_rows, "goal_object_action_count"),
        "mean_goal_receptacle_action_count": _mean(progress_rows, "goal_receptacle_action_count"),
        "mean_goal_verb_action_count": _mean(progress_rows, "goal_verb_action_count"),
        "tail_two_action_cycle_rate": _mean(progress_rows, "tail_two_action_cycle"),
        "mean_max_consecutive_repeat": _mean(progress_rows, "max_consecutive_repeat"),
        "top_verbs": [[verb, count] for verb, count in verb_counter.most_common(10) if verb],
        "rows": progress_rows,
    }


def compare_trace_progress_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = [
        "mean_progress_reward",
        "mean_progress_points",
        "mean_goal_interaction_points",
        "mean_placed_target_count",
        "mean_placed_target_event_count",
        "mean_irrelevant_action_count",
        "mean_reverted_target_count",
        "mean_wrong_receptacle_move_count",
        "tail_two_action_cycle_rate",
    ]
    deltas = {
        f"{metric}_delta": round(float(candidate.get(metric, 0.0) or 0.0) - float(reference.get(metric, 0.0) or 0.0), 6)
        for metric in metrics
    }
    positive = (
        deltas["mean_progress_reward_delta"] > 0.0
        or deltas["mean_goal_interaction_points_delta"] > 0.0
        or deltas["mean_placed_target_count_delta"] > 0.0
    )
    regression = (
        deltas["mean_wrong_receptacle_move_count_delta"] > 0.0
        or deltas["mean_reverted_target_count_delta"] > 0.0
        or deltas["tail_two_action_cycle_rate_delta"] > 0.0
    )
    status = "ok" if positive and not regression else "action_required"
    recommendation = "trace_progress_uplift_observed" if status == "ok" else "do_not_start_full_rl_from_this_gate"
    return {
        "status": status,
        "recommendation": recommendation,
        "reference_method": reference.get("method"),
        "candidate_method": candidate.get("method"),
        "reference_episode_count": reference.get("episode_count"),
        "candidate_episode_count": candidate.get("episode_count"),
        **deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline ALFWorld trace-progress gate reports.")
    parser.add_argument("--reference_run", required=True)
    parser.add_argument("--candidate_run", required=True)
    parser.add_argument("--reference_name", default="reference")
    parser.add_argument("--candidate_name", default="candidate")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    reference = build_trace_progress_report(args.reference_run, method_name=args.reference_name)
    candidate = build_trace_progress_report(args.candidate_run, method_name=args.candidate_name)
    comparison = compare_trace_progress_reports(reference, candidate)
    report = {
        "status": comparison["status"],
        "comparison": comparison,
        "reference": reference,
        "candidate": candidate,
    }
    write_json(Path(args.output_path), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
