import argparse
import csv
import json
from pathlib import Path

from config import DATA_DIR
from main_arxiv_html import process_url


DEFAULT_URLS = [
    "https://arxiv.org/html/2602.22727v1",
    "https://arxiv.org/html/2406.10125v1",
    "https://arxiv.org/html/2508.12692v1",
    "https://arxiv.org/html/2409.05211v1",
    "https://arxiv.org/html/2401.04647",
    "https://arxiv.org/html/2604.13940v1",
]


def load_urls(args) -> list:
    urls = []
    if args.urls:
        urls.extend(args.urls)
    if args.url_file:
        path = Path(args.url_file)
        urls.extend([
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ])
    return urls or DEFAULT_URLS


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate arXiv HTML KG extraction quality.")
    parser.add_argument("urls", nargs="*", help="arXiv HTML URLs")
    parser.add_argument("--url-file", help="Text file containing one URL per line")
    parser.add_argument("--max-sections", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = []
    for url in load_urls(args):
        result = process_url(url, dry_run=args.dry_run, max_sections=args.max_sections)
        report = result["quality_report"]
        images = result["paper"].get("images", [])
        row = {
            "arxiv_id": result["paper"]["arxiv_id"],
            "title": result["paper"]["title"],
            "url": result["paper"]["url"],
            "sections": result["stats"]["sections"],
            "cache_hits": result["stats"]["cache_hits"],
            "api_calls": result["stats"]["api_calls"],
            "nodes": report["node_count"],
            "edges": report["edge_count"],
            "claims": report["claim_count"],
            "evidence": report["evidence_count"],
            "results": report["result_count"],
            "isolated": report["isolated_node_count"],
            "low_confidence": len(report["low_confidence_nodes"]),
            "missing_evidence": len(report["missing_evidence_nodes"]),
            "image_count": len(images),
            "first_image_path": images[0].get("local_path", "") if images else "",
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out_path = Path(DATA_DIR) / "batch_eval_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
