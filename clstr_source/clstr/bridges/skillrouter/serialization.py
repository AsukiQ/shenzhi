from __future__ import annotations

import json
from typing import Any


def serialize_skill_text(skill: dict[str, Any]) -> str:
    def stable_field(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return "\n".join(
        [
            f"name:{skill.get('name', '')}",
            f"desc:{skill.get('description', '')}",
            f"in:{stable_field(skill.get('input_schema', {}))}",
            f"out:{stable_field(skill.get('output_schema', {}))}",
            f"exec:{skill.get('executor_desc', '')}",
            f"fail:{stable_field(skill.get('failure_modes', []))}",
            f"body:{skill.get('body', '')}",
        ]
    )
