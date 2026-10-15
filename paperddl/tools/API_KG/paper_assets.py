import logging
import mimetypes
import re
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import DATA_DIR

logger = logging.getLogger(__name__)


SKIP_IMAGE_PATTERNS = (
    "favicon",
    "logo",
    "icons/",
    "static/",
    "math",
)


class PaperAssetDownloader:
    def extract_images(self, page_url: str, html: str, max_images: int = 1) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        candidates = []

        for idx, img in enumerate(soup.find_all("img"), start=1):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue
            image_url = urljoin(page_url, src)
            if self._should_skip(image_url, img):
                continue

            figure = img.find_parent(["figure", "div"])
            full_figure = img.find_parent(attrs={"class": re.compile(r"\bltx_figure\b")})
            if full_figure:
                figure = full_figure
            caption = ""
            if figure:
                caption_tag = figure.find(["figcaption", "caption"])
                if caption_tag:
                    caption = caption_tag.get_text(" ", strip=True)

            alt = img.get("alt", "").strip()
            score = self._score_image(image_url, caption, alt, idx)
            candidates.append({
                "url": image_url,
                "caption": caption,
                "alt": alt,
                "score": score,
                "order": idx,
            })

        candidates.sort(key=lambda x: (-x["score"], x["order"]))
        return candidates[:max_images]

    def download_images(self, arxiv_id: str, images: List[Dict]) -> List[Dict]:
        image_dir = Path(DATA_DIR) / "images" / self._safe_name(arxiv_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            for idx, image in enumerate(images, start=1):
                try:
                    response = client.get(image["url"], headers={"User-Agent": "Mozilla/5.0"})
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";")[0].strip()
                    if content_type and not content_type.startswith("image/"):
                        logger.warning("Skipping non-image response: %s (%s)", image["url"], content_type)
                        continue

                    ext = self._extension(image["url"], content_type)
                    path = image_dir / f"paper_image_{idx}{ext}"
                    path.write_bytes(response.content)
                    copied = dict(image)
                    copied.update({
                        "local_path": str(path),
                        "content_type": content_type,
                        "bytes": len(response.content),
                    })
                    downloaded.append(copied)
                except Exception as exc:
                    logger.warning("Image download failed %s: %s", image.get("url"), exc)

        return downloaded

    def get_paper_images(self, page_url: str, html: str, arxiv_id: str, max_images: int = 1) -> List[Dict]:
        candidates = self.extract_images(page_url, html, max_images=max_images)
        return self.download_images(arxiv_id, candidates)

    def _score_image(self, url: str, caption: str, alt: str, order: int) -> int:
        text = f"{url} {caption} {alt}".lower()
        score = max(0, 50 - order)
        if "figure" in text or "fig." in text:
            score += 40
        if any(token in text for token in ("overview", "framework", "architecture", "pipeline", "method")):
            score += 30
        if any(token in text for token in ("table", "logo", "icon")):
            score -= 30
        return score

    def _should_skip(self, url: str, img) -> bool:
        lower = url.lower()
        if lower.startswith("data:"):
            return True
        parent = img.find_parent(["div", "span", "figure"])
        parent_class = " ".join(parent.get("class", [])).lower() if parent else ""
        if any(token in parent_class for token in ("page_logo", "navbar", "header", "footer")):
            return True
        return any(pattern in lower for pattern in SKIP_IMAGE_PATTERNS)

    def _extension(self, url: str, content_type: str) -> str:
        path_ext = Path(urlparse(url).path).suffix
        if path_ext and re.match(r"^\.[A-Za-z0-9]{2,5}$", path_ext):
            return path_ext
        guessed = mimetypes.guess_extension(content_type or "")
        return guessed or ".img"

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
