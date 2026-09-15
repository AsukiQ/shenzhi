from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AlfWorldSkillMapping:
    skill_id: str | None
    confidence: str
    reason: str
    normalized_action: str


def normalize_alfworld_action(action: str) -> str:
    return " ".join(str(action or "").strip().lower().split())


def _select(preferred: list[str], known: set[str]) -> str | None:
    if not known:
        return preferred[0] if preferred else None
    for skill_id in preferred:
        if skill_id in known:
            return skill_id
    return None


def _result(
    normalized_action: str,
    known: set[str],
    preferred: list[str],
    reason: str,
    confidence: str = "high",
) -> AlfWorldSkillMapping:
    skill_id = _select(preferred, known)
    if skill_id is None:
        return AlfWorldSkillMapping(
            skill_id=None,
            confidence="unmapped",
            reason=f"{reason}; preferred_skill_not_in_pool",
            normalized_action=normalized_action,
        )
    return AlfWorldSkillMapping(
        skill_id=skill_id,
        confidence=confidence,
        reason=reason,
        normalized_action=normalized_action,
    )


def map_alfworld_action_to_skill_id(
    action: str,
    known_skill_ids: Iterable[str] | None = None,
) -> AlfWorldSkillMapping:
    text = normalize_alfworld_action(action)
    known = {str(item) for item in known_skill_ids or [] if str(item).strip()}
    exact_concrete_skill_id = f"alfworld_action/{text}"
    if exact_concrete_skill_id in known:
        return AlfWorldSkillMapping(
            skill_id=exact_concrete_skill_id,
            confidence="exact",
            reason="exact_concrete_action_skill",
            normalized_action=text,
        )
    if text.startswith(("take ", "pick up ", "grab ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-object-picker", "alfworld/alfworld-object-retriever"],
            "alfworld_pickup_command",
        )
    if text.startswith(("go to ", "go ", "walk to ", "move to ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-location-navigator", "alfworld/alfworld-receptacle-navigator"],
            "alfworld_navigation_command",
        )
    if text.startswith("open "):
        return _result(
            text,
            known,
            ["alfworld/alfworld-receptacle-opener", "alfworld/alfworld-open-receptacle"],
            "alfworld_open_receptacle_command",
        )
    if text.startswith("close "):
        return _result(
            text,
            known,
            ["alfworld/alfworld-receptacle-closer", "alfworld/alfworld-receptacle-operator"],
            "alfworld_close_receptacle_command",
        )
    if text.startswith(("put ", "place ", "move ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-object-placer", "alfworld/alfworld-object-storer"],
            "alfworld_place_or_move_command",
        )
    if text.startswith(("clean ", "wash ")):
        return _result(text, known, ["alfworld/alfworld-clean-object"], "alfworld_clean_command")
    if text.startswith(("heat ", "warm ", "cook ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-heat-object-with-appliance", "alfworld/alfworld-object-heater"],
            "alfworld_heat_command",
        )
    if text.startswith(("cool ", "chill ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-object-cooler", "alfworld/alfworld-temperature-regulator"],
            "alfworld_cool_command",
        )
    if text.startswith(("look", "examine", "inspect")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-object-state-inspector", "alfworld/alfworld-environment-scanner"],
            "alfworld_inspection_command",
            confidence="medium",
        )
    if text.startswith("inventory"):
        return _result(
            text,
            known,
            ["alfworld/alfworld-inventory-management", "alfworld/alfworld-environment-scanner"],
            "alfworld_inventory_command",
            confidence="medium",
        )
    if text.startswith("help"):
        return _result(
            text,
            known,
            ["alfworld/alfworld-environment-scanner"],
            "alfworld_help_command",
            confidence="low",
        )
    if text.startswith(("use ", "turn on ", "toggle ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-device-operator", "alfworld/alfworld-tool-user"],
            "alfworld_device_or_tool_command",
            confidence="medium",
        )
    if text.startswith(("slice ", "cut ")):
        return _result(
            text,
            known,
            ["alfworld/alfworld-device-operator", "alfworld/alfworld-tool-user"],
            "alfworld_cut_with_tool_command",
            confidence="medium",
        )
    return AlfWorldSkillMapping(
        skill_id=None,
        confidence="unmapped",
        reason="unmapped_alfworld_action",
        normalized_action=text,
    )


def alfworld_action_to_skill_id(action: str, known_skill_ids: Iterable[str] | None = None) -> str | None:
    return map_alfworld_action_to_skill_id(action, known_skill_ids).skill_id
