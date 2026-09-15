"""从清洗后的论文文档生成论文版 CLSTR/SkillRouter Stage0 数据。

这是弱监督 bootstrap 数据：每篇论文用标题和摘要首句生成查询，正例是
该论文自身，hard negative 从同会议/同年份中确定性抽取。它适合先跑 smoke
和检索器初始化，不应冒充人工标注的最终评测集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm_tokens(value: str) -> set[str]:
    value = str(value or "").lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9_+.-]{1,}", value))
    for chunk in re.findall(r"[\u3400-\u9fff]+", value):
        words.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return words


def split_for(paper_id: str) -> str:
    # 以 canonical ID 哈希切分，保证同一论文的两个 query variant 不跨 split。
    bucket = int(hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def first_sentence(abstract: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(abstract or "")).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)
    return parts[0][:limit]


def make_query_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    pid = str(doc["paper_id"])
    title = str(doc.get("title") or "").strip()
    abstract = first_sentence(str(doc.get("abstract") or ""))
    rows: list[dict[str, Any]] = []
    if title:
        rows.append(
            {
                "query_id": f"{pid}::title",
                "query_text": f"Find papers about {title}",
                "positive_skill_id": pid,
                "variant": "title_template",
            }
        )
    if abstract:
        rows.append(
            {
                "query_id": f"{pid}::abstract",
                "query_text": abstract,
                "positive_skill_id": pid,
                "variant": "abstract_lead",
            }
        )
    return rows


def choose_negatives(
    doc_index: int,
    pool: list[dict[str, Any]],
    groups: dict[tuple[str, int | None], list[int]],
    group_position: dict[int, int],
    token_sets: list[set[str]],
    *,
    count: int,
    candidate_window: int = 96,
) -> list[str]:
    doc = pool[doc_index]
    pid = str(doc["paper_id"])
    conf = str(doc.get("conference") or "").casefold()
    year = doc.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    candidates = groups.get((conf, year), [])
    # 只比较同会议/年份中 canonical ID 相邻的固定窗口，避免大会议组内
    # O(n^2) 扫描。组过小时再用全局确定性步长补足候选。
    position = group_position.get(doc_index, 0)
    half = max(int(candidate_window) // 2, count)
    local = candidates[max(0, position - half) : min(len(candidates), position + half + 1)]
    candidate_indices = [idx for idx in local if idx != doc_index]
    if len(candidate_indices) < count:
        stride = 7919  # 与数据规模互质倾向较强的固定质数步长
        cursor = int(hashlib.sha256(pid.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
        seen = set(candidate_indices)
        seen.add(doc_index)
        while len(candidate_indices) < max(count, candidate_window) and len(seen) < len(pool):
            if cursor not in seen:
                candidate_indices.append(cursor)
                seen.add(cursor)
            cursor = (cursor + stride) % len(pool)
    query_tokens = token_sets[doc_index]
    scored: list[tuple[float, str]] = []
    for index in candidate_indices:
        other = pool[index]
        other_id = str(other["paper_id"])
        if other_id == pid:
            continue
        overlap = len(query_tokens & token_sets[index])
        # 同会议/年份候选优先，词面稍有重叠者作为 hard negative；ID 作为稳定 tie-break。
        scored.append((-float(overlap), other_id))
    scored.sort()
    return [paper_id for _score, paper_id in scored[: max(0, int(count))]]


def build(documents_path: Path, output_dir: Path, *, negative_count: int = 8, max_papers: int | None = None) -> dict[str, Any]:
    pool: list[dict[str, Any]] = []
    for doc in read_jsonl(documents_path):
        if not doc.get("paper_id") or not doc.get("retrieval_eligible", True):
            continue
        pool.append(doc)
        if max_papers is not None and len(pool) >= int(max_papers):
            break
    if not pool:
        raise ValueError("no eligible paper documents")
    pool.sort(key=lambda d: str(d["paper_id"]))
    groups: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    for index, doc in enumerate(pool):
        year = doc.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        groups[(str(doc.get("conference") or "").casefold(), year)].append(index)
    group_position: dict[int, int] = {}
    for indices in groups.values():
        for position, index in enumerate(indices):
            group_position[index] = position
    token_sets = [
        norm_tokens(f"{doc.get('title', '')} {' '.join(doc.get('keywords') or [])}")
        for doc in pool
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "skill_pool.jsonl"
    retrieval_path = output_dir / "retrieval.jsonl"
    split_rows: dict[str, list[str]] = defaultdict(list)
    query_count = 0
    with pool_path.open("w", encoding="utf-8") as pool_handle, retrieval_path.open("w", encoding="utf-8") as retrieval_handle:
        for doc_index, doc in enumerate(pool):
            pid = str(doc["paper_id"])
            skill = {
                "skill_id": pid,
                "canonical_skill_id": pid,
                "name": doc.get("title", ""),
                "description": doc.get("abstract", ""),
                "body": doc.get("search_text", ""),
                "paper_id": pid,
                "source_id": doc.get("source_id", ""),
                "conference": doc.get("conference"),
                "year": doc.get("year"),
                "authors": doc.get("authors", []),
                "keywords": doc.get("keywords", []),
            }
            pool_handle.write(json.dumps(skill, ensure_ascii=False) + "\n")
            split = split_for(pid)
            for query in make_query_rows(doc):
                query["split"] = split
                query["source"] = "paper_weak_bootstrap"
                query["provenance"] = {"paper_id": pid, "variant": query["variant"], "label_quality": "weak"}
                query["negative_skill_ids"] = choose_negatives(
                    doc_index,
                    pool,
                    groups,
                    group_position,
                    token_sets,
                    count=negative_count,
                )
                retrieval_handle.write(json.dumps(query, ensure_ascii=False) + "\n")
                split_rows[split].append(query["query_id"])
                query_count += 1
    (output_dir / "splits.json").write_text(json.dumps({"splits": {k: {"query_ids": v} for k, v in sorted(split_rows.items())}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": "clstr_unified_v2_compatible_paper_bootstrap",
        "source_documents": str(documents_path),
        "paper_count": len(pool),
        "query_count": query_count,
        "negative_count": int(negative_count),
        "label_quality": "weak_bootstrap",
        "splits": {key: len(value) for key, value in sorted(split_rows.items())},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negative-count", type=int, default=8)
    parser.add_argument("--max-papers", type=int)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.documents), Path(args.output_dir), negative_count=args.negative_count, max_papers=args.max_papers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
