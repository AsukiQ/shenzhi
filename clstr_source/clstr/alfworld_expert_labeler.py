from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OfficialExpertLabel:
    official_expert_action: str | None
    expert_action_in_admissible: bool
    usable_for_expert_ce: bool
    skip_reason: str | None
    expert_action_source: str = "official_alfworld_extra_expert_plan"

    def to_record(self) -> dict[str, Any]:
        return {
            "official_expert_action_t": self.official_expert_action,
            "expert_action_t": self.official_expert_action,
            "teacher_action_t": self.official_expert_action,
            "expert_action_source": self.expert_action_source,
            "expert_action_in_admissible": bool(self.expert_action_in_admissible),
            "usable_for_expert_ce": bool(self.usable_for_expert_ce),
            "usable_for_policy": bool(self.usable_for_expert_ce),
            "expert_skip_reason": self.skip_reason,
        }


def _first_plan_action(infos: dict[str, Any], batch_index: int) -> str | None:
    plans = infos.get("extra.expert_plan")
    if not isinstance(plans, (list, tuple)) or batch_index >= len(plans):
        return None
    plan = plans[batch_index]
    if not plan:
        return None
    action = str(plan[0]).strip()
    return action or None


def label_official_expert_action(
    infos: dict[str, Any],
    batch_index: int,
    admissible_commands: list[str],
) -> OfficialExpertLabel:
    candidates = [str(item) for item in admissible_commands or []]
    action = _first_plan_action(infos, batch_index)
    if action is None:
        return OfficialExpertLabel(
            official_expert_action=None,
            expert_action_in_admissible=False,
            usable_for_expert_ce=False,
            skip_reason="missing_official_expert_action",
        )
    in_admissible = action in candidates
    return OfficialExpertLabel(
        official_expert_action=action,
        expert_action_in_admissible=bool(in_admissible),
        usable_for_expert_ce=bool(in_admissible),
        skip_reason=None if in_admissible else "official_expert_action_not_in_admissible",
    )
