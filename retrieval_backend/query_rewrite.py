"""Chinese-to-English query rewriting without training or third-party dependencies.

The glossary path is deterministic and safe to run offline.  An optional HTTP
translator can rewrite the remaining sentence, but its output is only an
additional recall query; the original user query is never replaced.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


@dataclass(frozen=True)
class GlossaryTerm:
    term_id: str
    zh: tuple[str, ...]
    en: tuple[str, ...]
    category: str = "general"
    status: str = "verified"
    weight: float = 1.0


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    translated_query: str
    matched_terms: tuple[dict[str, Any], ...] = ()
    language: str = "other"
    source: str = "identity"
    fallback: bool = False
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryRewriteService:
    def __init__(
        self,
        glossary_path: str | Path | None = None,
        *,
        translator_url: str | None = None,
        timeout_seconds: float = 0.8,
        cache_size: int = 2048,
    ) -> None:
        self.glossary_path = Path(glossary_path) if glossary_path else None
        self.translator_url = translator_url.rstrip("/") if translator_url else None
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, str], RewriteResult] = OrderedDict()
        self._lock = threading.RLock()
        self._terms = self._load_terms()

    def _load_terms(self) -> tuple[GlossaryTerm, ...]:
        if not self.glossary_path or not self.glossary_path.is_file():
            return ()
        raw = json.loads(self.glossary_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("terms", [])
        terms: list[GlossaryTerm] = []
        for row in raw:
            if not isinstance(row, dict) or row.get("status", "verified") != "verified":
                continue
            zh = row.get("zh") or row.get("zh_terms") or row.get("zh_term")
            en = row.get("en") or row.get("en_terms") or row.get("en_term")
            zh_values = (zh,) if isinstance(zh, str) else tuple(str(x) for x in (zh or []))
            en_values = (en,) if isinstance(en, str) else tuple(str(x) for x in (en or []))
            zh_values = tuple(x.strip() for x in zh_values if x and x.strip())
            en_values = tuple(x.strip() for x in en_values if x and x.strip())
            if zh_values and en_values:
                terms.append(GlossaryTerm(str(row.get("term_id", zh_values[0])), zh_values, en_values,
                                          str(row.get("category", "general")), "verified",
                                          float(row.get("weight", 1.0))))
        return tuple(sorted(terms, key=lambda x: max(map(len, x.zh)), reverse=True))

    def _glossary_rewrite(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        rewritten = str(query)
        matches: list[dict[str, Any]] = []
        for term in self._terms:
            for zh in sorted(term.zh, key=len, reverse=True):
                if zh not in rewritten:
                    continue
                en = term.en[0]
                rewritten = rewritten.replace(zh, f" {en} ")
                matches.append({"term_id": term.term_id, "zh": zh, "en": en, "category": term.category})
                break
        if matches:
            # For a Chinese natural-language request, keep the canonical
            # English domain terms and any user-supplied Latin acronym.  Drop
            # Chinese request filler (请找/论文/近三年等) that cannot match the
            # English-only corpus and would otherwise dilute the FTS query.
            english_terms = [str(item["en"]) for item in matches]
            latin_terms = _LATIN_TOKEN_RE.findall(query)
            return " ".join(dict.fromkeys([*english_terms, *latin_terms])), matches
        latin_terms = _LATIN_TOKEN_RE.findall(query)
        if latin_terms:
            return " ".join(dict.fromkeys(latin_terms)), matches
        return " ".join(rewritten.split()), matches

    def _remote_rewrite(self, query: str, matched_terms: list[dict[str, Any]]) -> str | None:
        if not self.translator_url:
            return None
        payload = json.dumps({"query": query, "terms": matched_terms}, ensure_ascii=False).encode()
        request = urllib.request.Request(
            self.translator_url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read(256 * 1024).decode("utf-8"))
            value = data.get("translated_query") or data.get("translation")
            return str(value).strip() if isinstance(value, str) and value.strip() else None
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
            return None

    def rewrite(self, query: str) -> RewriteResult:
        original = " ".join(str(query or "").split())
        language = "zh" if contains_cjk(original) else "other"
        if not original or language != "zh":
            return RewriteResult(original, original, language=language)
        key = (original, str(self.glossary_path or ""))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        rewritten, matches = self._glossary_rewrite(original)
        remote = self._remote_rewrite(original, matches)
        translated = remote or rewritten
        source = "translator+glossary" if remote else "glossary"
        fallback = not bool(translated) or translated == original
        result = RewriteResult(original, translated, tuple(matches), language, source, fallback,
                               0.9 if remote else (0.65 if matches else None))
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result
