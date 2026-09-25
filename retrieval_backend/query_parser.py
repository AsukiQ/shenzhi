"""Conservative parsing of structured constraints from paper-search queries.

The parser intentionally extracts only high-confidence patterns.  It never
guesses an author or venue from arbitrary text; uncertain text remains in the
semantic query and is handled by the translation service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import json
import re
from pathlib import Path
from typing import Any
import unicodedata

try:
    from .graph_query import infer_graph_query_plan
    from .paper_search import SearchFilters
except ImportError:
    from graph_query import infer_graph_query_plan
    from paper_search import SearchFilters


_YEAR = r"(?:19|20)\d{2}"
_FILLER_PHRASES = (
    "请帮我找",
    "帮我找",
    "查找",
    "搜索",
    "查询",
    "发表的",
    "论文",
    "找",
    "在",
    "然后",
    "并且",
    "再",
)


@dataclass(frozen=True)
class ParsedQuery:
    original_query: str
    semantic_query: str
    graph_query: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    extracted: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "semantic_query": self.semantic_query,
            "graph_query": self.graph_query,
            "filters": asdict(self.filters),
            "extracted": list(self.extracted),
        }


def _normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).strip()


def _mask(text: list[str], start: int, end: int) -> None:
    for index in range(max(0, start), min(len(text), end)):
        text[index] = " "


def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < right and end > left for left, right in spans)


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、;/；]+", value) if item.strip()]


_CONFERENCES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "IEEE Symposium on Security and Privacy": "S&P",
            "Security and Privacy": "S&P",
            "NeurIPS": "NeurIPS",
            "NIPS": "NeurIPS",
            "AAAI": "AAAI",
            "ICML": "ICML",
            "VLDB": "VLDB",
            "S&P": "S&P",
            "KDD": "KDD",
            "ACL": "ACL",
            "EMNLP": "EMNLP",
            "CVPR": "CVPR",
            "ICCV": "ICCV",
            "ECCV": "ECCV",
            "WWW": "WWW",
            "SIGMOD": "SIGMOD",
            "ICDE": "ICDE",
            "ICLR": "ICLR",
            "IJCAI": "IJCAI",
            "COLT": "COLT",
            "AISTATS": "AISTATS",
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


class QueryParser:
    """Parse only explicit, deterministic query constraints."""

    def __init__(self, glossary_path: str | Path | None = None):
        self.glossary_path = Path(glossary_path) if glossary_path else None
        self._term_map = self._load_term_map()

    def _load_term_map(self) -> dict[str, str]:
        if not self.glossary_path or not self.glossary_path.is_file():
            return {}
        try:
            raw = json.loads(self.glossary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        rows = raw.get("terms", []) if isinstance(raw, dict) else raw
        output: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("status", "verified") != "verified":
                continue
            zh = row.get("zh") or row.get("zh_terms") or row.get("zh_term")
            en = row.get("en") or row.get("en_terms") or row.get("en_term")
            zh_values = [zh] if isinstance(zh, str) else list(zh or [])
            en_values = [en] if isinstance(en, str) else list(en or [])
            if not en_values:
                continue
            for value in zh_values:
                value = _normalize(value)
                if value:
                    output[value.casefold()] = _normalize(en_values[0])
        return output

    def _canonical_term(self, value: str) -> str | None:
        normalized = _normalize(value)
        if not normalized:
            return None
        if re.fullmatch(r"[\x00-\x7f]+", normalized):
            return normalized
        return self._term_map.get(normalized.casefold())

    def parse(self, query: str) -> ParsedQuery:
        original = _normalize(query)
        working = list(original)
        spans: list[tuple[int, int]] = []
        extracted: list[dict[str, Any]] = []
        year_gte: int | None = None
        year_lte: int | None = None
        conferences: list[str] = []
        authors: list[str] = []
        keywords: list[str] = []
        subjects: list[str] = []

        def add_match(kind: str, match: re.Match[str], value: Any, *, remove: bool = True) -> None:
            start, end = match.span()
            if _overlaps(spans, start, end):
                return
            spans.append((start, end))
            if remove:
                _mask(working, start, end)
            extracted.append({"type": kind, "value": value, "text": match.group(0)})

        # Process ranges and relative bounds before exact years.
        for match in re.finditer(
            rf"(?P<start>{_YEAR})\s*(?:年\s*)?(?:-|–|—|到|至)\s*(?P<end>{_YEAR})\s*年?",
            original,
            re.IGNORECASE,
        ):
            start_year, end_year = int(match.group("start")), int(match.group("end"))
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            # If a query contains more than one bounded range, keep the
            # intersection rather than silently letting the last match win.
            year_gte = start_year if year_gte is None else max(year_gte, start_year)
            year_lte = end_year if year_lte is None else min(year_lte, end_year)
            add_match("year_range", match, {"year_gte": start_year, "year_lte": end_year})

        for match in re.finditer(
            rf"(?P<year>{_YEAR})\s*年?\s*(?:以后|之后|起|以来|及以后|以上)", original
        ):
            value = int(match.group("year"))
            year_gte = value if year_gte is None else max(year_gte, value)
            add_match("year_gte", match, value)

        for match in re.finditer(
            rf"(?P<year>{_YEAR})\s*年?\s*(?:以前|之前|以内|及以前|以下)", original
        ):
            value = int(match.group("year"))
            year_lte = value if year_lte is None else min(year_lte, value)
            add_match("year_lte", match, value)

        for match in re.finditer(r"近\s*(?P<count>\d+)\s*年", original):
            count = max(1, int(match.group("count")))
            value = date.today().year - count + 1
            year_gte = value if year_gte is None else max(year_gte, value)
            add_match("recent_years", match, {"count": count, "year_gte": value})

        for match in re.finditer(rf"(?<!\d)(?P<year>{_YEAR})\s*年(?!\s*(?:以后|之后|以前|之前|到|至))", original):
            year = int(match.group("year"))
            if year_gte is None:
                year_gte = year
            if year_lte is None:
                year_lte = year
            add_match("year_exact", match, year)

        # English-style queries commonly write a bare year (``ICML 2024``).
        # Treat a standalone four-digit year as an exact bound, while avoiding
        # numbers embedded in identifiers or ordinary words.
        for match in re.finditer(rf"(?<![A-Za-z0-9])(?P<year>{_YEAR})(?![A-Za-z0-9])", original):
            if _overlaps(spans, *match.span()):
                continue
            prefix = original[max(0, match.start() - 8) : match.start()].casefold()
            if prefix.endswith("arxiv:") or prefix.endswith("paper:"):
                continue
            year = int(match.group("year"))
            if year_gte is None:
                year_gte = year
            if year_lte is None:
                year_lte = year
            add_match("year_exact", match, year)

        # Explicit conference names are a conservative allow-list.  They can
        # be written without a marker, e.g. "AAAI 的图神经网络论文".
        for label, canonical in _CONFERENCES:
            # Allow a venue immediately followed by a year (``AAAI2024``),
            # but avoid matching it inside a longer alphabetic token.
            pattern = rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])"
            flags = re.IGNORECASE if re.fullmatch(r"[A-Za-z& .]+", label) else 0
            for match in re.finditer(pattern, original, flags):
                if _overlaps(spans, *match.span()):
                    continue
                if canonical not in conferences:
                    conferences.append(canonical)
                add_match("conference", match, canonical)

        # Mark relation phrases before parsing explicit entity filters. This
        # prevents phrases such as ``same author`` from being mistaken for an
        # author value by the broad English author pattern below.
        graph_plan = infer_graph_query_plan(original)
        for term in graph_plan.matched_terms:
            for match in re.finditer(re.escape(term), original, re.IGNORECASE):
                if not _overlaps(spans, *match.span()):
                    add_match("relation", match, term)

        # Only explicit author markers are accepted.  Free text is never
        # guessed as an author name.
        author_patterns = (
            # Stop a Chinese name before common clause delimiters.  Without
            # this boundary, ``作者是张三的图神经网络论文`` would incorrectly
            # capture the whole six-character tail as the author.
            r"(?:作者(?:是|为)|作者\s*[:：])\s*(?P<name>(?:(?![的和与及])\s*[\u3400-\u9fff]){2,6})(?=的|，|,|。|；|;|、|和|与|及|论文|$)",
            r"由\s*(?P<name>(?:(?![的和与及])\s*[\u3400-\u9fff]){2,6})(?=的|，|,|。|；|;|、|和|与|及|论文)",
            r"(?:作者(?:是|为)|作者\s*[:：])\s*(?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})(?=的|，|,|。|；|;|、|和|与|及|论文|$)",
            r"(?:author(?:\s+is)?|by)\s*[:：]?\s*(?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})(?=\s+(?:of|on|about|in|for|and|or|same|related|other)|\s*['’]s|\s*[,.;:!?]|$)",
        )
        for pattern in author_patterns:
            for match in re.finditer(pattern, original, re.IGNORECASE):
                name = _normalize(match.group("name")).rstrip("的")
                # Conjunctions and relation words delimit an author value in
                # natural-language requests (e.g. ``author is Alice and
                # related field``). Keep the explicit name, not the clause.
                name = re.split(
                    r"\s+(?:and|or|same|related|other)\b",
                    name,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                if not name or _overlaps(spans, *match.span()):
                    continue
                authors.append(name)
                add_match("author", match, name)

        # Keyword/subject filters require an explicit marker. Chinese values
        # are promoted only when the verified glossary has an English mapping.
        field_patterns = (
            ("keyword", r"(?:关键词|关键字|keywords?)\s*(?:是|为|[:：])?\s*(?P<value>[^，,。；;\n]+)"),
            ("subject", r"(?:主题|领域|方向|subjects?)\s*(?:是|为|[:：])?\s*(?P<value>[^，,。；;\n]+)"),
        )
        for kind, pattern in field_patterns:
            for match in re.finditer(pattern, original, re.IGNORECASE):
                values = _split_values(match.group("value"))
                accepted: list[str] = []
                for value in values:
                    canonical = self._canonical_term(value)
                    if canonical:
                        accepted.append(canonical)
                if kind == "keyword":
                    keywords.extend(accepted)
                else:
                    subjects.extend(accepted)
                # An unverified value is intentionally left in the semantic
                # query for translation/retrieval; never erase it just
                # because it followed an explicit field marker.
                if accepted:
                    add_match(kind, match, accepted)
                else:
                    extracted.append({"type": kind, "value": [], "text": match.group(0), "accepted": False})

        # Remove only generic request scaffolding.  This makes a pure filter
        # query such as ``2020年以后在AAAI发表的论文`` an explicit browse
        # request while retaining domain nouns like ``图神经网络``.
        for term in _FILLER_PHRASES:
            for match in re.finditer(re.escape(term), original):
                if _overlaps(spans, *match.span()):
                    continue
                if term == "在" and match.start() > 0:
                    next_char = original[match.end() : match.end() + 1]
                    # Do not erase the ``在`` inside a domain word such as
                    # ``存在``; venue clauses (``在AAAI``) are followed by a
                    # Latin venue token and remain safely removable.
                    if not next_char.isascii():
                        continue
                spans.append(match.span())
                _mask(working, *match.span())

        # Remove standalone conjunctions left between multiple relation
        # clauses, while preserving words embedded in domain terms.
        for match in re.finditer(r"(?i)(?<![A-Za-z])(and|or|then|also)(?![A-Za-z])", original):
            if not _overlaps(spans, *match.span()):
                spans.append(match.span())
                _mask(working, *match.span())

        semantic = _normalize("".join(working))
        filters = SearchFilters(
            year_gte=year_gte,
            year_lte=year_lte,
            conference=list(dict.fromkeys(conferences)),
            author=list(dict.fromkeys(authors)),
            keyword=list(dict.fromkeys(keywords)),
            subject=list(dict.fromkeys(subjects)),
        )
        return ParsedQuery(original, semantic, original, filters, tuple(extracted))


def merge_filters(parsed: SearchFilters, explicit: SearchFilters) -> SearchFilters:
    """Explicit API fields override parsed values when supplied."""

    return SearchFilters(
        year_gte=explicit.year_gte if explicit.year_gte is not None else parsed.year_gte,
        year_lte=explicit.year_lte if explicit.year_lte is not None else parsed.year_lte,
        conference=list(dict.fromkeys(explicit.conference or parsed.conference)),
        author=list(dict.fromkeys(explicit.author or parsed.author)),
        keyword=list(dict.fromkeys(explicit.keyword or parsed.keyword)),
        subject=list(dict.fromkeys(explicit.subject or parsed.subject)),
        institution=list(dict.fromkeys(explicit.institution or parsed.institution)),
    )
