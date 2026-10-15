import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import httpx
import requests
from bs4 import BeautifulSoup

from config import ARXIV_HTML_CACHE_FIRST, DATA_DIR, KG_MAX_SECTION_CHARS
from kg_cache import content_hash

logger = logging.getLogger(__name__)


HIGH_VALUE_KEYWORDS = (
    "abstract",
    "introduction",
    "method",
    "approach",
    "experiment",
    "evaluation",
    "benchmark",
    "metric",
    "result",
    "analysis",
    "ablation",
    "conclusion",
    "limitation",
    "future",
    "acknowledg",
)

ALWAYS_INCLUDE_KEYWORDS = (
    "acknowledg",
)


@dataclass
class PaperSection:
    section_id: str
    title: str
    level: int
    text: str


class ArxivHTMLProcessor:
    def fetch(self, url: str) -> str:
        cache_dir = Path(DATA_DIR) / "html_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{content_hash(url)[:16]}.html"
        if ARXIV_HTML_CACHE_FIRST and cache_path.exists():
            logger.info("Using cached HTML first: %s", cache_path)
            return cache_path.read_text(encoding="utf-8")

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        last_error = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    cache_path.write_text(response.text, encoding="utf-8")
                    return response.text
            except Exception as exc:
                last_error = exc
                logger.warning("httpx fetch failed (%s/3): %s", attempt + 1, exc)
                time.sleep(1 + attempt)

        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except Exception as exc:
            last_error = exc
            logger.warning("requests fetch failed: %s", exc)

        if cache_path.exists():
            logger.warning("Using cached HTML after fetch failure: %s", cache_path)
            return cache_path.read_text(encoding="utf-8")

        raise RuntimeError(f"Failed to fetch arXiv HTML and no cache exists: {last_error}")

    def parse(self, url: str, html: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")
        self._remove_noise(soup)

        arxiv_id = self._parse_arxiv_id(url, soup)
        title = self._meta_content(soup, "citation_title") or self._first_heading(soup)
        authors = self._extract_authors(soup)
        abstract = self._extract_abstract(soup)
        sections = self._extract_sections(soup)

        if abstract:
            sections.insert(0, PaperSection("abstract", "Abstract", 1, abstract))

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": url,
            "sections": [s.__dict__ for s in sections if s.text.strip()],
        }

    def high_value_sections(self, paper: Dict, max_sections: int = 8) -> List[Dict]:
        selected = []
        for section in paper.get("sections", []):
            title = section.get("title", "").lower()
            text = section.get("text", "")
            if any(k in title for k in HIGH_VALUE_KEYWORDS):
                copied = dict(section)
                copied["text"] = text[:KG_MAX_SECTION_CHARS]
                selected.append(copied)

        if not selected and paper.get("sections"):
            selected = paper["sections"][:max_sections]

        selected = selected[:max_sections]
        selected_ids = {s.get("section_id") for s in selected}

        for section in paper.get("sections", []):
            title = section.get("title", "").lower()
            if not any(k in title for k in ALWAYS_INCLUDE_KEYWORDS):
                continue
            if section.get("section_id") in selected_ids:
                continue
            copied = dict(section)
            copied["text"] = copied.get("text", "")[:KG_MAX_SECTION_CHARS]
            if len(selected) < max_sections:
                selected.append(copied)
            elif selected:
                selected[-1] = copied
            break

        return selected

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

    def _parse_arxiv_id(self, url: str, soup: BeautifulSoup) -> str:
        match = re.search(r"arxiv\.org/(?:html|abs|pdf)/([^/?#]+)", url)
        if match:
            return match.group(1).replace(".pdf", "")
        identifier = self._meta_content(soup, "citation_arxiv_id")
        return identifier or url.rstrip("/").split("/")[-1]

    def _meta_content(self, soup: BeautifulSoup, name: str) -> str:
        tag = soup.find("meta", attrs={"name": name})
        return tag.get("content", "").strip() if tag else ""

    def _first_heading(self, soup: BeautifulSoup) -> str:
        heading = soup.find(["h1", "h2"])
        return heading.get_text(" ", strip=True) if heading else ""

    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        candidates = [
            soup.select_one(".ltx_abstract"),
            soup.find(attrs={"class": re.compile("abstract", re.I)}),
        ]
        for candidate in candidates:
            if candidate:
                text = candidate.get_text(" ", strip=True)
                return re.sub(r"^Abstract\s*", "", text, flags=re.I).strip()
        return ""

    def _extract_authors(self, soup: BeautifulSoup) -> List[str]:
        authors = [m.get("content", "").strip() for m in soup.find_all("meta", attrs={"name": "citation_author"})]
        authors = [a for a in authors if a]
        if authors:
            return authors

        person_names = []
        for item in soup.select(".ltx_personname"):
            raw = item.get_text(" ", strip=True)
            person_names.extend(re.split(r"\s*,\s*|\s+\band\b\s+", raw))
        authors = [self._clean_author_name(x) for x in person_names]
        authors = [a for a in authors if a]
        if authors:
            return list(dict.fromkeys(authors))

        block = soup.select_one(".ltx_authors")
        if not block:
            return []
        text = block.get_text(" ", strip=True)
        text = re.split(r"\{|\}|@|Corresponding author|Abstract", text)[0]
        parts = re.split(r",|\band\b", text)
        authors = [self._clean_author_name(x) for x in parts]
        return [a for a in dict.fromkeys(authors) if a]

    def _clean_author_name(self, raw: str) -> str:
        name = re.sub(r"[\d*†‡§¶]+", "", raw or "")
        name = re.sub(r"\S+@\S+", "", name)
        name = re.sub(r"\b(Didi Chuxing|Yonsei University|Seoul National University|Peng Cheng Laboratory)\b.*", "", name, flags=re.I)
        name = re.sub(r"\s+", " ", name).strip(" ,;")
        if not name or len(name) > 80:
            return ""
        org_tokens = (
            "academy",
            "beijing",
            "china",
            "institute",
            "laboratory",
            "school",
            "telecommunications",
            "university",
        )
        if any(token in name.lower() for token in org_tokens):
            return ""
        if len(name.split()) < 2:
            return ""
        return name

    def _extract_sections(self, soup: BeautifulSoup) -> List[PaperSection]:
        sections = []

        section_tags = soup.select("section")
        if section_tags:
            for idx, tag in enumerate(section_tags, start=1):
                heading = tag.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if not heading:
                    continue
                title = heading.get_text(" ", strip=True)
                level = self._heading_level(heading.name)
                text = tag.get_text(" ", strip=True)
                text = text.replace(title, "", 1).strip()
                if text:
                    sections.append(PaperSection(str(tag.get("id") or idx), title, level, text))
            if sections:
                return sections

        headings = soup.find_all(["h2", "h3", "h4"])
        for idx, heading in enumerate(headings, start=1):
            level = self._heading_level(heading.name)
            title = heading.get_text(" ", strip=True)
            chunks = []
            for sibling in heading.next_siblings:
                name = getattr(sibling, "name", None)
                if name in ("h2", "h3", "h4") and self._heading_level(name) <= level:
                    break
                if hasattr(sibling, "get_text"):
                    chunks.append(sibling.get_text(" ", strip=True))
            text = " ".join(c for c in chunks if c)
            if text:
                sections.append(PaperSection(str(heading.get("id") or idx), title, level, text))
        return sections

    def _heading_level(self, name: str) -> int:
        match = re.match(r"h(\d)", name or "")
        return int(match.group(1)) if match else 2
