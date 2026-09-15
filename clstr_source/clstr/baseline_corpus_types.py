from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedSkillRouterQuery:
    query_id: str
    query: str
    benchmark: str
    positive_skill_ids: list[str]
    positive_indices: list[int]
    candidate_skill_ids: list[str] | None = None
    source_id: str = ""
    kind: str = ""
    split_group_identity: str = ""
    runtime_visible_catalog_id: str = ""
    inventory_catalog_digest: str = ""
    history_mode: str = ""


@dataclass(frozen=True)
class UnifiedSkillRouterCorpus:
    skills: list[dict[str, Any]]
    train_queries: list[UnifiedSkillRouterQuery]
    eval_queries: list[UnifiedSkillRouterQuery]
    report: dict[str, Any]
