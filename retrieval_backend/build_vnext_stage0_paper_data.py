"""把论文弱监督检索数据适配为 canonical CLSTR vNext Stage0 数据。

输入仍然是旧 bootstrap 产生的论文候选表和 query-positive 对。这个脚本只做
数据契约迁移，不生成虚假的多步轨迹：每条论文查询都是一个零历史决策点，
retrieval/static-route 两个 Stage0 监督头共享同一查询和正例论文。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CURRENT_STATE_CONTRACT = "clstr_structured_current_state_v1"
CAUSAL_STATE_CONTRACT = "clstr_structured_causal_state_v1"
SCHEMA_VERSION = "shenzhi_paper_vnext_stage0_data_v1"
CATALOG_ID = "shenzhi_paper_public_global_v1"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def load_skill_ids(skills_path: Path) -> tuple[int, list[str]]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(skills_path):
        candidate_id = skill_id(row)
        if not candidate_id:
            raise ValueError("paper skill row lacks skill_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate paper skill_id: {candidate_id}")
        seen.add(candidate_id)
        ordered.append(candidate_id)
    if len(ordered) < 2:
        raise ValueError("vNext Stage0 requires at least two paper candidates")
    return len(ordered), sorted(ordered)


def write_skills(source: Path, destination: Path) -> int:
    count = 0
    with destination.open("w", encoding="utf-8") as output:
        for row in read_jsonl(source):
            candidate_id = skill_id(row)
            # 只规范 vNext/SkillRouter 实际使用的公共字段，其余论文元数据原样保留。
            copied = dict(row)
            copied["skill_id"] = candidate_id
            copied["canonical_skill_id"] = candidate_id
            copied["candidate_type"] = "Paper"
            copied["provenance"] = {
                "source_id": "shenzhi_paper_data_v1",
                "paper_id": str(row.get("paper_id") or candidate_id),
                "source_record_id": str(row.get("source_id") or ""),
            }
            output.write(json.dumps(copied, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def structured_zero_history_row(
    raw: dict[str, Any],
    *,
    positive_id: str,
    catalog_digest: str,
    catalog_size: int,
) -> dict[str, Any]:
    query_id = str(raw.get("query_id") or "").strip()
    query_text = str(raw.get("query_text") or raw.get("query") or "").strip()
    if not query_id or not query_text:
        raise ValueError("paper retrieval row lacks query_id/query_text")
    paper_id = str((raw.get("provenance") or {}).get("paper_id") or positive_id)
    components = {
        "goal_text": query_text,
        "task_text": "",
        "current_observation_text": "",
    }
    current = f"goal: {query_text}"
    empty_events: list[dict[str, Any]] = []
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "query_text": query_text,
        "task_id": paper_id,
        "task_group_identity": paper_id,
        "split_group_identity": paper_id,
        "source": "paper_weak_bootstrap",
        "benchmark": "shenzhi_paper_retrieval",
        "variant": str(raw.get("variant") or "weak_query"),
        "split": str(raw.get("split") or "").strip(),
        "step_index": 0,
        "decision_step_index": 0,
        "state_text": current,
        "state_text_full": query_text,
        "state_text_current": current,
        "current_state_components": components,
        "current_state_contract": CURRENT_STATE_CONTRACT,
        "current_state_materialization_source": "paper_query_text_v1",
        "current_state_full_text_sha256": text_digest(query_text),
        "current_state_components_sha256": json_digest(components),
        "state_text_causal": current,
        "causal_state_contract": CAUSAL_STATE_CONTRACT,
        "causal_prefix_events": empty_events,
        "causal_prefix_event_count": 0,
        "causal_prefix_sha256": json_digest(empty_events),
        "causal_state_sha256": text_digest(current),
        "zero_history_causal_equals_current": True,
        "required_tool_set_skill_ids": [positive_id],
        "current_state_route_set_skill_ids": [positive_id],
        "current_state_route_factual_target_counts": {positive_id: 1},
        "future_label_skill_ids": [positive_id],
        "runtime_visible_catalog_id": CATALOG_ID,
        "inventory_catalog_digest": catalog_digest,
        "inventory_source": "immutable_shenzhi_paper_candidate_pool",
        "inventory_available_before_decision": True,
        "inventory_pool_size": int(catalog_size),
        "inventory_protocol": "public_global",
        "capabilities": {
            "required_tool_set": True,
            "current_state_route_set": True,
            "ordered_next_tool": False,
            "actual_execution_result": False,
            "causal_branch_pair": False,
        },
        "provenance": {
            "source_id": "paper_weak_bootstrap",
            "paper_id": paper_id,
            "variant": str(raw.get("variant") or "weak_query"),
            "label_quality": str(
                (raw.get("provenance") or {}).get("label_quality") or "weak"
            ),
            "history_semantics": "zero_history_single_step_query",
        },
    }


def contract_entry(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def build(
    skills_path: Path,
    retrieval_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_skills": output_dir / "skills.jsonl",
        "retrieval_rows": output_dir / "retrieval_train.jsonl",
        "retrieval_dev_rows": output_dir / "retrieval_dev.jsonl",
        "static_route_rows": output_dir / "static_route_train.jsonl",
        "static_route_dev_rows": output_dir / "static_route_dev.jsonl",
        "inventory_catalogs": output_dir / "inventory_catalogs.jsonl",
    }

    skill_count, sorted_skill_ids = load_skill_ids(skills_path)
    written_skills = write_skills(skills_path, paths["training_skills"])
    if written_skills != skill_count:
        raise RuntimeError("paper skill count changed while writing vNext data")

    catalog_digest = json_digest(sorted_skill_ids)
    catalog = {
        "inventory_catalog_id": CATALOG_ID,
        "inventory_catalog_digest": catalog_digest,
        "inventory_source": "immutable_shenzhi_paper_candidate_pool",
        "inventory_available_before_decision": True,
        "inventory_pool_size": skill_count,
        "runtime_visible_skill_ids": sorted_skill_ids,
    }
    paths["inventory_catalogs"].write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    handles = {
        key: path.open("w", encoding="utf-8")
        for key, path in paths.items()
        if key not in {"training_skills", "inventory_catalogs"}
    }
    split_counts: Counter[str] = Counter()
    positive_ids: set[str] = set()
    known = set(sorted_skill_ids)
    try:
        for raw in read_jsonl(retrieval_path):
            positive_id = str(
                raw.get("positive_skill_id")
                or (raw.get("positive_skill_ids") or [""])[0]
            ).strip()
            if positive_id not in known:
                raise ValueError(
                    f"paper query references unknown positive skill: {positive_id}"
                )
            split = str(raw.get("split") or "").strip().lower()
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"unsupported paper query split: {split}")
            row = structured_zero_history_row(
                raw,
                positive_id=positive_id,
                catalog_digest=catalog_digest,
                catalog_size=skill_count,
            )
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            if split == "train":
                handles["retrieval_rows"].write(encoded)
                handles["static_route_rows"].write(encoded)
            elif split == "dev":
                handles["retrieval_dev_rows"].write(encoded)
                handles["static_route_dev_rows"].write(encoded)
            # test 是冻结评测集，不放进 canonical Stage0 trainer 输入。
            split_counts[split] += 1
            positive_ids.add(positive_id)
    finally:
        for handle in handles.values():
            handle.close()

    if not split_counts["train"] or not split_counts["dev"]:
        raise ValueError("vNext Stage0 data requires nonempty train and dev splits")

    files = {key: contract_entry(path) for key, path in paths.items()}
    contract = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "task": "single_step_paper_retrieval",
        "label_quality": "weak_bootstrap",
        "stage_contract": "canonical_clstr_vnext_stage0",
        "history_contract": {
            "structured_current_state_required": True,
            "structured_causal_state_required": True,
            "all_rows_are_zero_history": True,
            "multi_step_trajectory_claimed": False,
        },
        "inventory": {
            "catalog_id": CATALOG_ID,
            "catalog_digest": catalog_digest,
            "candidate_count": skill_count,
        },
        "counts": {
            "skill_count": skill_count,
            "unique_positive_skill_count": len(positive_ids),
            "query_splits": dict(sorted(split_counts.items())),
            "retrieval_train_rows": split_counts["train"],
            "retrieval_dev_rows": split_counts["dev"],
            "static_route_train_rows": split_counts["train"],
            "static_route_dev_rows": split_counts["dev"],
        },
        "source_files": {
            "skills": contract_entry(skills_path),
            "retrieval": contract_entry(retrieval_path),
        },
        "files": files,
        "notes": [
            "The test split is excluded from all trainer inputs.",
            "Canonical Stage0 mines hard negatives from the legal full candidate pool.",
            "No multi-step paper trajectory is synthesized by this adapter.",
        ],
    }
    contract_path = output_dir / "data_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract | {
        "data_contract_path": str(contract_path.resolve()),
        "data_contract_sha256": file_sha256(contract_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical CLSTR vNext Stage0 paper retrieval data"
    )
    parser.add_argument("--skills", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build(
        Path(args.skills),
        Path(args.retrieval),
        Path(args.output_dir),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
