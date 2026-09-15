"""Deterministic graph-query intent planning for paper retrieval.

The planner deliberately recognizes only relationship-oriented language.  A
normal topical query stays on the lexical/dense path; an explicit metadata
filter is handled separately by :class:`SearchFilters`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

try:
    from .paper_search import SearchFilters
except ImportError:
    from paper_search import SearchFilters


@dataclass(frozen=True)
class GraphQueryPlan:
    """The bounded graph operation selected for one request."""

    expansion_relation: str | None = None
    expansion_relations: tuple[str, ...] = ()
    filter_requested: bool = False
    trigger: str = "none"
    matched_terms: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.filter_requested or self.expansion_relation is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_RELATION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "authors",
        (
            "同作者",
            "同一作者",
            "作者的其他",
            "作者其他论文",
            "同作者的其他论文",
            "同作者其他论文",
            "其他作者工作",
            "相关作者",
            "same author",
            "same authors",
            "same researchers",
            "other papers by",
            "author expansion",
        ),
    ),
    (
        "keywords",
        (
            "同关键词",
            "相同关键词",
            "共享关键词",
            "same keyword",
            "same keywords",
            "shared keywords",
        ),
    ),
    (
        "subjects",
        (
            "同主题",
            "相同主题",
            "同领域",
            "相同领域",
            "相关领域",
            "same subject",
            "same field",
            "related field",
        ),
    ),
    (
        "venue_year",
        (
            "同会议",
            "同会",
            "同年份",
            "附近年份",
            "近年论文",
            "同 venue",
            "same venue",
            "same conference",
            "same year",
            "nearby year",
        ),
    ),
)


def _first_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    folded = str(text or "").casefold()
    # Prefer the longest phrase so callers that remove matched spans do not
    # leave a trailing fragment such as ``的其他论文`` behind a shorter
    # ``同作者`` match.
    return tuple(
        term
        for term in sorted(terms, key=lambda value: len(str(value)), reverse=True)
        if str(term).casefold() in folded
    )


def infer_graph_query_plan(query: str, filters: SearchFilters | None = None) -> GraphQueryPlan:
    """Infer a safe, bounded graph operation without extracting entities.

    Entity values remain bound by the HTTP filter fields or by the returned
    seed papers.  This avoids inventing author/method identifiers from free
    text; entity extraction can be added later when audited data is available.
    """

    active_filters = filters or SearchFilters()
    filter_requested = bool(
        active_filters.year_gte is not None
        or active_filters.year_lte is not None
        or active_filters.conference
        or active_filters.author
        or active_filters.keyword
        or active_filters.subject
    )
    text = str(query or "")
    matches: list[tuple[int, str, tuple[str, ...]]] = []
    for relation, terms in _RELATION_TERMS:
        matched = _first_terms(text, terms)
        if matched:
            first_position = min(text.casefold().find(term.casefold()) for term in matched)
            matches.append((first_position, relation, matched))
    if matches:
        matches.sort(key=lambda item: (item[0], item[1]))
        relations = tuple(dict.fromkeys(item[1] for item in matches))
        matched_terms = tuple(term for _pos, _relation, terms in matches for term in terms)
        trigger = "filter_and_relation" if filter_requested else "relation_language"
        return GraphQueryPlan(
            expansion_relation=relations[0],
            expansion_relations=relations,
            filter_requested=filter_requested,
            trigger=trigger,
            matched_terms=matched_terms,
        )
    return GraphQueryPlan(
        expansion_relations=(),
        filter_requested=filter_requested,
        trigger="explicit_filter" if filter_requested else "none",
    )
