import argparse
import json
import logging
from pathlib import Path

from academic_kg_extractor import AcademicKGExtractor
from arxiv_html_processor import ArxivHTMLProcessor
from config import (
    ARXIV_HTML_TEST_URL,
    DATA_DIR,
    KG_PROMPT_VERSION,
    KG_SCHEMA_VERSION,
    LLM_MODEL,
)
from kg_cache import build_cache, content_hash
from kg_schema import build_quality_report, merge_section_kgs
from paper_assets import PaperAssetDownloader


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_url(url: str, dry_run: bool = False, max_sections: int = 8) -> dict:
    processor = ArxivHTMLProcessor()
    html = processor.fetch(url)
    paper = processor.parse(url, html)
    images = PaperAssetDownloader().get_paper_images(url, html, paper["arxiv_id"], max_images=1)
    paper["images"] = images
    sections = processor.high_value_sections(paper, max_sections=max_sections)

    logger.info("Paper: %s (%s)", paper.get("title", ""), paper.get("arxiv_id", ""))
    logger.info("Selected %s high-value sections from %s parsed sections.",
                len(sections), len(paper.get("sections", [])))

    cache = build_cache()
    extractor = AcademicKGExtractor(dry_run=dry_run)
    section_kgs = []
    stats = {"cache_hits": 0, "api_calls": 0, "sections": len(sections)}

    try:
        for section in sections:
            section_id = section.get("section_id") or section.get("title", "")
            hash_value = content_hash(section.get("text", ""))
            cached = cache.get_extraction(
                paper_id=paper["arxiv_id"],
                section_id=section_id,
                hash_value=hash_value,
                schema_version=KG_SCHEMA_VERSION,
                prompt_version=KG_PROMPT_VERSION,
                model=LLM_MODEL,
            )
            if cached:
                logger.info("Cache hit: %s", section.get("title", section_id))
                stats["cache_hits"] += 1
                section_kgs.append(cached["kg"] if "kg" in cached else cached)
                continue

            logger.info("%s: %s", "Dry-run extract" if dry_run else "API extract", section.get("title", section_id))
            kg = extractor.extract_section(paper, section)
            stats["api_calls"] += 0 if dry_run else 1
            section_kgs.append(kg)
            cache.save_extraction(
                paper_id=paper["arxiv_id"],
                section_id=section_id,
                hash_value=hash_value,
                schema_version=KG_SCHEMA_VERSION,
                prompt_version=KG_PROMPT_VERSION,
                model=LLM_MODEL,
                kg=kg,
            )

        merged = merge_section_kgs(section_kgs)
        quality_report = build_quality_report(merged)
        output = {
            "paper": {
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "url": paper.get("url", ""),
                "images": images,
            },
            "kg": merged,
            "assets": {
                "images": images,
            },
            "quality_report": quality_report,
            "stats": stats,
        }

        output_path = Path(DATA_DIR) / f"{paper['arxiv_id'].replace('/', '_')}_kg.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved merged KG: %s", output_path)

        if hasattr(cache, "save_paper_kg"):
            cache.save_paper_kg(output["paper"], merged)
            logger.info("Saved merged KG nodes and edges to Neo4j.")

        return output
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser(description="Extract compact academic KG from an arXiv HTML paper.")
    parser.add_argument("--url", default=ARXIV_HTML_TEST_URL, help="arXiv HTML URL")
    parser.add_argument("--dry-run", action="store_true", help="Run without LLM API calls")
    parser.add_argument("--max-sections", type=int, default=8, help="Maximum high-value sections to process")
    args = parser.parse_args()

    result = process_url(args.url, dry_run=args.dry_run, max_sections=args.max_sections)
    print(json.dumps({
        "paper": result["paper"],
        "stats": result["stats"],
        "node_count": len(result["kg"]["nodes"]),
        "edge_count": len(result["kg"]["edges"]),
        "claim_count": result["quality_report"]["claim_count"],
        "evidence_count": result["quality_report"]["evidence_count"],
        "isolated_node_count": result["quality_report"]["isolated_node_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
