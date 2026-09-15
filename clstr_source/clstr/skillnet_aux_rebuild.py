from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_ENVIRONMENTS = ("alfworld", "scienceworld", "webshop")
TRAINING_SPLITS = {"train"}
VALID_OR_TEST_SPLITS = {"dev", "valid", "validation", "valid_seen", "valid_unseen", "test"}


@dataclass(frozen=True)
class SkillNetMapping:
    skill_id: str | None
    skill_name: str | None
    confidence: str
    reason: str


def _json_dumps(row: Any) -> str:
    return json.dumps(row, ensure_ascii=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(row) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, text

    metadata: dict[str, str] = {}
    key: str | None = None
    folded: list[str] = []
    for raw_line in lines[1:end_idx]:
        if not raw_line.strip():
            if key is not None and folded:
                folded.append("")
            continue
        if not raw_line.startswith((" ", "\t")) and ":" in raw_line:
            if key is not None and folded:
                metadata[key] = " ".join(part.strip() for part in folded if part.strip()).strip()
            current_key, value = raw_line.split(":", 1)
            key = current_key.strip()
            value = value.strip()
            if value in {">", ">-", "|", "|-"}:
                folded = []
            else:
                metadata[key] = value.strip("\"'")
                folded = []
                key = None
        elif key is not None:
            folded.append(raw_line.strip())
    if key is not None and folded:
        metadata[key] = " ".join(part.strip() for part in folded if part.strip()).strip()
    body = "\n".join(lines[end_idx + 1 :])
    return metadata, body


def load_skillnet_skills(skillnet_root: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load SkillNet SKILL.md files grouped by benchmark environment."""
    root = Path(skillnet_root)
    skills: dict[str, dict[str, dict[str, Any]]] = {}
    for environment in SUPPORTED_ENVIRONMENTS:
        env_root = root / environment
        env_skills: dict[str, dict[str, Any]] = {}
        if not env_root.exists():
            skills[environment] = env_skills
            continue
        for skill_path in sorted(env_root.glob("*/SKILL.md")):
            text = skill_path.read_text(encoding="utf-8")
            metadata, body = _extract_frontmatter(text)
            name = metadata.get("name") or skill_path.parent.name
            description = metadata.get("description") or ""
            skill_id = f"{environment}/{name}"
            env_skills[skill_id] = {
                "skill_id": skill_id,
                "name": name,
                "description": description,
                "environment": environment,
                "source": "skillnet_skill_pool",
                "source_path": str(skill_path),
                "body": body,
                "not_skillsbench_skill": True,
            }
        skills[environment] = env_skills
    return skills


def _normalize_action(action_text: str) -> str:
    action = action_text.strip().lower()
    action = re.sub(r"\s+", " ", action)
    return action


def _lookup_skill(
    environment: str,
    skills_by_env: dict[str, dict[str, dict[str, Any]]],
    preferred_names: list[str],
    reason: str,
    confidence: str = "high",
) -> SkillNetMapping:
    env_skills = skills_by_env.get(environment, {})
    for name in preferred_names:
        skill_id = f"{environment}/{name}"
        if skill_id in env_skills:
            return SkillNetMapping(skill_id, env_skills[skill_id]["name"], confidence, reason)
    for name in preferred_names:
        for skill_id, skill in env_skills.items():
            if name in skill["name"]:
                return SkillNetMapping(skill_id, skill["name"], "medium", f"{reason}; fuzzy_skill_name_match")
    return SkillNetMapping(None, None, "unmapped", f"{reason}; skill_not_found")


def _map_alfworld(action: str, skills_by_env: dict[str, dict[str, dict[str, Any]]]) -> SkillNetMapping:
    if action.startswith(("go to ", "go ", "walk to ", "move to ")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-location-navigator", "alfworld-receptacle-navigator", "alfworld-navigation-planner"],
            "alfworld_navigation_command",
        )
    if action.startswith(("take ", "pick up ", "grab ")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-object-picker", "alfworld-object-retriever"],
            "alfworld_pickup_command",
        )
    if action.startswith(("put ", "place ", "move ")) and " to " in action:
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-object-placer", "alfworld-object-storer", "alfworld-object-transporter"],
            "alfworld_place_or_move_command",
        )
    if action.startswith("open "):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-receptacle-opener", "alfworld-open-receptacle", "alfworld-receptacle-operator"],
            "alfworld_open_receptacle_command",
        )
    if action.startswith("close "):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-receptacle-closer", "alfworld-receptacle-operator"],
            "alfworld_close_receptacle_command",
        )
    if action.startswith(("clean ", "wash ")):
        return _lookup_skill("alfworld", skills_by_env, ["alfworld-clean-object"], "alfworld_clean_command")
    if action.startswith(("heat ", "cook ", "warm ")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-heat-object-with-appliance", "alfworld-object-heater", "alfworld-temperature-regulator"],
            "alfworld_heat_command",
        )
    if action.startswith(("cool ", "chill ")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-object-cooler", "alfworld-temperature-regulator"],
            "alfworld_cool_command",
        )
    if action.startswith(("use ", "turn on ", "toggle ")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-device-operator", "alfworld-tool-user"],
            "alfworld_device_or_tool_command",
        )
    if action.startswith(("look", "examine", "inspect")):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-object-state-inspector", "alfworld-environment-scanner"],
            "alfworld_inspection_command",
            confidence="medium",
        )
    if action.startswith("inventory"):
        return _lookup_skill(
            "alfworld",
            skills_by_env,
            ["alfworld-inventory-management"],
            "alfworld_inventory_command",
            confidence="medium",
        )
    return _lookup_skill(
        "alfworld",
        skills_by_env,
        ["alfworld-task-verifier", "alfworld-goal-interpreter"],
        "alfworld_unrecognized_action_template",
        confidence="low",
    )


def _map_scienceworld(action: str, skills_by_env: dict[str, dict[str, dict[str, Any]]]) -> SkillNetMapping:
    if action.startswith(("teleport ", "teleport to ", "go to ", "move to ")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-room-navigator", "scienceworld-room-teleporter"],
            "scienceworld_room_navigation_command",
        )
    if action.startswith(("look around", "look", "examine", "inspect")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-room-scanner", "scienceworld-room-explorer", "scienceworld-container-inspector"],
            "scienceworld_inspection_command",
            confidence="medium",
        )
    if action.startswith("focus "):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-object-focuser", "scienceworld-task-focuser"],
            "scienceworld_focus_command",
        )
    if action.startswith(("take ", "pick up ", "get ", "fetch ")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-object-retriever", "scienceworld-item-fetcher", "scienceworld-tool-fetcher"],
            "scienceworld_retrieval_command",
        )
    if action.startswith(("put ", "place ", "move ")) and " to " in action:
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-object-placer", "scienceworld-container-transfer", "scienceworld-container-relocator"],
            "scienceworld_place_or_transfer_command",
        )
    if action.startswith(("mix ", "combine ")):
        return _lookup_skill("scienceworld", skills_by_env, ["scienceworld-mixture-creator"], "scienceworld_mix_command")
    if action.startswith(("measure ", "read temperature", "use thermometer")) or "temperature" in action:
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-temperature-measurer", "scienceworld-measurement-taker"],
            "scienceworld_measurement_command",
        )
    if action.startswith(("activate ", "turn on ", "use stove", "use oven")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-device-activator", "scienceworld-tool-user"],
            "scienceworld_device_activation_command",
        )
    if action.startswith("connect "):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-circuit-connector", "scienceworld-circuit-builder"],
            "scienceworld_circuit_connection_command",
        )
    if action.startswith(("pour ", "fill ")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-liquid-pourer", "scienceworld-liquid-filler"],
            "scienceworld_liquid_transfer_command",
        )
    if action.startswith(("wait", "wait1")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-controlled-waiting", "scienceworld-process-pauser"],
            "scienceworld_wait_command",
        )
    if action.startswith(("open ", "close ")):
        return _lookup_skill(
            "scienceworld",
            skills_by_env,
            ["scienceworld-environment-isolation", "scienceworld-container-inspector"],
            "scienceworld_open_close_command",
            confidence="medium",
        )
    return _lookup_skill(
        "scienceworld",
        skills_by_env,
        ["scienceworld-task-interpreter", "scienceworld-task-parser"],
        "scienceworld_unrecognized_action_template",
        confidence="low",
    )


def _is_webshop_product_click(action: str) -> bool:
    if not action.startswith("click[") or action in {"click[buy now]", "click[next >]", "click[< prev]", "click[back to search]"}:
        return False
    target = action.removeprefix("click[").removesuffix("]")
    return bool(re.search(r"\b[A-Z0-9]{8,}\b", target.upper())) or " - " in target or target[:1].isalnum()


def _map_webshop(action: str, skills_by_env: dict[str, dict[str, dict[str, Any]]]) -> SkillNetMapping:
    if action.startswith("search["):
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-search-executor", "webshop-initial-search", "webshop-product-search"],
            "webshop_search_command",
        )
    if action in {"click[next >]", "click[< prev]", "click[back to search]", "click[prev <]"}:
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-result-page-navigator"],
            "webshop_result_navigation_click",
        )
    if "buy now" in action or "checkout" in action:
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-purchase-executor", "webshop-purchase-initiator"],
            "webshop_purchase_click",
        )
    if any(token in action for token in ("description", "features", "reviews", "details")):
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-product-detail-inspector", "webshop-product-detail-check"],
            "webshop_detail_inspection_click",
        )
    if any(token in action for token in ("color", "size", "flavor", "pack", "quantity", "option")):
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-variant-chooser", "webshop-option-selector", "webshop-attribute-selector"],
            "webshop_variant_or_option_click",
        )
    if _is_webshop_product_click(action):
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-product-selector", "webshop-product-detail-navigator"],
            "webshop_product_selection_click",
            confidence="medium",
        )
    if action.startswith("click["):
        return _lookup_skill(
            "webshop",
            skills_by_env,
            ["webshop-action-executor"],
            "webshop_generic_click",
            confidence="low",
        )
    return _lookup_skill(
        "webshop",
        skills_by_env,
        ["webshop-action-executor"],
        "webshop_unrecognized_action_template",
        confidence="low",
    )


def map_action_to_skillnet_skill(
    environment: str,
    action_text: str,
    skills_by_env: dict[str, dict[str, dict[str, Any]]],
) -> SkillNetMapping:
    env = (environment or "").lower().replace("-", "_")
    action = _normalize_action(action_text)
    if env not in SUPPORTED_ENVIRONMENTS:
        return SkillNetMapping(None, None, "unmapped", f"unsupported_environment:{env or 'missing'}")
    if not skills_by_env.get(env):
        return SkillNetMapping(None, None, "unmapped", f"missing_skillnet_skill_pool:{env}")
    if env == "alfworld":
        return _map_alfworld(action, skills_by_env)
    if env == "scienceworld":
        return _map_scienceworld(action, skills_by_env)
    return _map_webshop(action, skills_by_env)


def _step_is_training_usable(record: dict[str, Any], step: dict[str, Any]) -> tuple[bool, str | None]:
    split = str(record.get("split") or "unknown").lower()
    if split not in TRAINING_SPLITS:
        return False, "non_train_split"
    if step.get("skillnet_skill_id") is None:
        return False, "unmapped_skillnet_skill"
    if step.get("skillnet_mapping_confidence") not in {"high", "medium"}:
        return False, "low_confidence_skillnet_mapping"
    if not str(step.get("action_text") or "").strip():
        return False, "missing_action_text"
    return True, None


def _remap_record(
    record: dict[str, Any],
    skills_by_env: dict[str, dict[str, dict[str, Any]]],
    confidence_counts: Counter[str],
    usable_step_counter: Counter[str],
) -> dict[str, Any]:
    remapped = dict(record)
    remapped_steps: list[dict[str, Any]] = []
    for step in record.get("steps") or []:
        if not isinstance(step, dict):
            continue
        mapping = map_action_to_skillnet_skill(str(record.get("environment") or ""), str(step.get("action_text") or ""), skills_by_env)
        new_step = dict(step)
        new_step["original_pseudo_skill_id"] = step.get("pseudo_skill_id")
        new_step["original_pseudo_skill_low_confidence"] = step.get("pseudo_skill_low_confidence")
        new_step["skillnet_skill_id"] = mapping.skill_id
        new_step["skillnet_skill_name"] = mapping.skill_name
        new_step["skillnet_mapping_confidence"] = mapping.confidence
        new_step["skillnet_mapping_reason"] = mapping.reason
        if mapping.skill_id is not None:
            new_step["pseudo_skill_id"] = mapping.skill_id
            new_step["pseudo_skill_low_confidence"] = mapping.confidence not in {"high", "medium"}
        usable, unusable_reason = _step_is_training_usable(record, new_step)
        new_step["usable_for_skillnet_training"] = usable
        if unusable_reason is not None:
            new_step["unusable_reason"] = unusable_reason
        confidence_counts[mapping.confidence] += 1
        if usable:
            usable_step_counter["usable"] += 1
        else:
            usable_step_counter[unusable_reason or "unusable"] += 1
        remapped_steps.append(new_step)
    remapped["steps"] = remapped_steps
    remapped["skill_pool_source"] = "skillnet"
    remapped["usable_for_skillnet_training"] = any(step["usable_for_skillnet_training"] for step in remapped_steps)
    return remapped


def _skill_rows(skills_by_env: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment in SUPPORTED_ENVIRONMENTS:
        for skill in skills_by_env.get(environment, {}).values():
            rows.append(
                {
                    key: value
                    for key, value in skill.items()
                    if key
                    in {
                        "skill_id",
                        "name",
                        "description",
                        "environment",
                        "source",
                        "source_path",
                        "body",
                        "not_skillsbench_skill",
                    }
                }
            )
    return sorted(rows, key=lambda row: row["skill_id"])


def _pseudo_skill_rows(skills_by_env: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = _skill_rows(skills_by_env)
    for row in rows:
        row["source"] = "skillnet_skill_pool_as_aux_pseudo_skill"
        row["low_confidence_mapping"] = False
    return rows


def _load_source_manifest(source_data_dir: Path) -> dict[str, Any]:
    path = source_data_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unreadable"}


def rebuild_aux_with_skillnet(
    source_data_dir: str | Path,
    skillnet_root: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    training_splits: Iterable[str] = ("train",),
) -> dict[str, Any]:
    source_data_dir = Path(source_data_dir)
    skillnet_root = Path(skillnet_root)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    trajectories_path = source_data_dir / "trajectories.jsonl"
    if not trajectories_path.exists():
        raise FileNotFoundError(f"missing auxiliary trajectories: {trajectories_path}")

    global TRAINING_SPLITS
    previous_training_splits = TRAINING_SPLITS
    TRAINING_SPLITS = {split.lower() for split in training_splits}
    try:
        skills_by_env = load_skillnet_skills(skillnet_root)
        available_envs = {env for env, rows in skills_by_env.items() if rows}
        missing_envs = {env: "missing_skillnet_skill_pool" for env in ["crafter"] if env not in available_envs}

        source_manifest = _load_source_manifest(source_data_dir)
        skill_rows = _skill_rows(skills_by_env)
        _write_jsonl(output_dir / "skills.jsonl", skill_rows)
        _write_jsonl(output_dir / "pseudo_skills.jsonl", _pseudo_skill_rows(skills_by_env))
        for environment in SUPPORTED_ENVIRONMENTS:
            _write_jsonl(output_dir / environment / "skills.jsonl", sorted(skills_by_env.get(environment, {}).values(), key=lambda row: row["skill_id"]))

        env_handles: dict[str, Any] = {}
        combined_path = output_dir / "trajectories.jsonl"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_handle = combined_path.open("w", encoding="utf-8")
        confidence_counts: Counter[str] = Counter()
        usable_step_counter: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        split_counts_by_environment: dict[str, Counter[str]] = defaultdict(Counter)
        environment_counts: Counter[str] = Counter()
        step_counts_by_env: Counter[str] = Counter()
        usable_steps_by_env: Counter[str] = Counter()
        excluded_counts: Counter[str] = Counter()
        valid_or_test_training_steps = 0
        written_trajectory_count = 0
        written_step_count = 0

        try:
            for record in _read_jsonl(trajectories_path):
                environment = str(record.get("environment") or "unknown").lower()
                if environment not in SUPPORTED_ENVIRONMENTS or not skills_by_env.get(environment):
                    excluded_counts[environment] += 1
                    missing_envs.setdefault(environment, "missing_skillnet_skill_pool")
                    continue
                remapped = _remap_record(record, skills_by_env, confidence_counts, usable_step_counter)
                split = str(remapped.get("split") or "unknown").lower()
                split_counts[split] += 1
                split_counts_by_environment[environment][split] += 1
                environment_counts[environment] += 1
                step_count = len(remapped.get("steps") or [])
                usable_steps = sum(1 for step in remapped.get("steps") or [] if step.get("usable_for_skillnet_training"))
                step_counts_by_env[environment] += step_count
                usable_steps_by_env[environment] += usable_steps
                if split in VALID_OR_TEST_SPLITS:
                    valid_or_test_training_steps += usable_steps
                line = _json_dumps(remapped) + "\n"
                combined_handle.write(line)
                if environment not in env_handles:
                    env_path = output_dir / environment / "trajectories.jsonl"
                    env_path.parent.mkdir(parents=True, exist_ok=True)
                    env_handles[environment] = env_path.open("w", encoding="utf-8")
                env_handles[environment].write(line)
                written_trajectory_count += 1
                written_step_count += step_count
        finally:
            combined_handle.close()
            for handle in env_handles.values():
                handle.close()

        skill_counts = {env: len(skills_by_env.get(env, {})) for env in SUPPORTED_ENVIRONMENTS}
        manifest = {
            "status": "ok" if written_trajectory_count else "missing",
            "data_role": "auxiliary_skillnet_rebuilt_not_official_replay",
            "source_aux_data_dir": str(source_data_dir),
            "source_aux_manifest_path": str(source_data_dir / "manifest.json"),
            "source_aux_manifest_record_counts": source_manifest.get("record_counts", {}),
            "skillnet_root": str(skillnet_root),
            "skillnet_environments": sorted(available_envs),
            "excluded_environments": dict(sorted(missing_envs.items())),
            "record_counts": {"trajectories": written_trajectory_count, "steps": written_step_count},
            "split_counts": dict(sorted(split_counts.items())),
            "split_counts_by_environment": {
                env: dict(sorted(counter.items())) for env, counter in sorted(split_counts_by_environment.items())
            },
            "environment_counts": dict(sorted(environment_counts.items())),
            "step_counts_by_environment": dict(sorted(step_counts_by_env.items())),
            "skill_counts": dict(sorted(skill_counts.items())),
            "skill_count_total": sum(skill_counts.values()),
            "mapping_confidence_counts": dict(sorted(confidence_counts.items())),
            "training_splits": sorted(TRAINING_SPLITS),
            "training_usable_step_count": usable_step_counter["usable"],
            "training_usable_steps_by_environment": dict(sorted(usable_steps_by_env.items())),
            "unusable_step_counts": {
                key: value for key, value in sorted(usable_step_counter.items()) if key != "usable"
            },
            "uses_valid_or_test_for_training": valid_or_test_training_steps > 0,
            "valid_or_test_training_step_count": valid_or_test_training_steps,
            "excluded_trajectory_counts": dict(sorted(excluded_counts.items())),
            "quality_caveats": {
                "crafter_excluded_due_to_missing_skillnet_pool": "crafter" in missing_envs,
                "not_equivalent_to_official_replay": True,
                "candidate_admissible_commands_available": False,
                "expert_action_in_admissible_available": False,
                "requires_official_replay_for_full_l_policy": True,
            },
            "forbidden_outputs": {
                "skillsbench_successful_trajectories": "not_written",
                "skillsbench_harness_results": "not_written",
                "clean_router_train_jsonl": "not_written",
            },
        }
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(report_dir / "rebuild_report.json", manifest)
        return manifest
    finally:
        TRAINING_SPLITS = previous_training_splits
