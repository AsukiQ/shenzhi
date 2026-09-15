"""Build canonical vNext Stage0 data for executable retrieval-operation skills."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

try:
    from .multistep_search import ACTION_SKILLS, parse_action_skill
except ImportError:
    from multistep_search import ACTION_SKILLS, parse_action_skill


CURRENT_STATE_CONTRACT = "clstr_structured_current_state_v1"
CAUSAL_STATE_CONTRACT = "clstr_structured_causal_state_v1"
SCHEMA_VERSION = "shenzhi_action_vnext_stage0_data_v1"
CATALOG_ID = "shenzhi_retrieval_action_global_v1"


ACTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "SEARCH_PAPERS": (
        "Find papers about {topic}.",
        "Search the paper collection for work on {topic}.",
        "检索与{topic}有关的论文。",
        "先查找关于{topic}的候选论文。",
    ),
    "EXPAND_AUTHORS": (
        "From the current papers, find more work by the same authors on {topic}.",
        "Expand the result set through the researchers who wrote the current papers about {topic}.",
        "沿当前论文作者继续找与{topic}相关的论文。",
        "看看这些作者还发表过哪些{topic}方向的工作。",
    ),
    "EXPAND_KEYWORDS": (
        "Broaden the current {topic} results using their keywords.",
        "Follow the keyword links of the current papers to find related {topic} work.",
        "根据当前论文的关键词扩展{topic}相关结果。",
        "沿关键词节点继续检索{topic}论文。",
    ),
    "EXPAND_SUBJECTS": (
        "Explore the subject areas of the current papers to find broader work on {topic}.",
        "Expand by the subject categories attached to the current {topic} papers.",
        "按照当前论文所属主题方向扩展{topic}检索。",
        "沿学科或主题节点寻找更多{topic}论文。",
    ),
    "EXPAND_VENUE_YEAR": (
        "Find nearby {topic} papers from the same venues and publication years.",
        "Explore recent work around the conferences and years of the current {topic} results.",
        "按照当前论文的会议和相近年份查找{topic}相关工作。",
        "找同会议、近年份的{topic}论文。",
    ),
    "RERANK_RESULTS": (
        "Rerank the accumulated {topic} papers by relevance to the original request.",
        "Choose the most relevant papers from the current {topic} candidate set.",
        "把当前{topic}候选论文按相关性重新排序。",
        "从已有{topic}结果中精排出最相关的论文。",
    ),
    "STOP": (
        "The current {topic} results are sufficient; stop searching and return them.",
        "No further expansion is needed for {topic}; finish the retrieval.",
        "当前{topic}结果已经足够，停止检索并返回。",
        "不需要继续扩展{topic}，结束搜索。",
    ),
}

TOPICS = (
    "knowledge graph completion",
    "large language model reasoning",
    "medical image segmentation",
    "federated learning privacy",
    "multimodal retrieval",
    "graph neural networks",
    "robust machine learning",
    "知识图谱补全",
    "大语言模型推理",
    "医学图像分割",
    "联邦学习隐私保护",
    "多模态检索",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def contract_entry(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _row(
    *,
    action: str,
    base_action: str,
    slot_index: int | None,
    text: str,
    split: str,
    index: int,
    catalog_digest: str,
    catalog_size: int,
) -> dict[str, Any]:
    components = {
        "goal_text": text,
        "task_text": "select the next executable paper-retrieval operation",
        "current_observation_text": "no execution history is available at this static decision",
    }
    current = "\n".join(
        (
            f"goal: {components['goal_text']}",
            f"task: {components['task_text']}",
            f"observation: {components['current_observation_text']}",
        )
    )
    empty_events: list[dict[str, Any]] = []
    identity = f"action-stage0/{split}/{action}/{index:06d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": identity,
        "query_text": text,
        "task_id": identity,
        "task_group_identity": identity,
        "split_group_identity": identity,
        "source": "shenzhi_action_weak_bootstrap",
        "benchmark": "shenzhi_retrieval_action_routing",
        "split": split,
        "step_index": 0,
        "decision_step_index": 0,
        "state_text": current,
        "state_text_full": current,
        "state_text_current": current,
        "current_state_components": components,
        "current_state_contract": CURRENT_STATE_CONTRACT,
        "current_state_materialization_source": "action_template_v1",
        "current_state_full_text_sha256": text_digest(current),
        "current_state_components_sha256": json_digest(components),
        "state_text_causal": current,
        "causal_state_contract": CAUSAL_STATE_CONTRACT,
        "causal_prefix_events": empty_events,
        "causal_prefix_event_count": 0,
        "causal_prefix_sha256": json_digest(empty_events),
        "causal_state_sha256": text_digest(current),
        "zero_history_causal_equals_current": True,
        "required_tool_set_skill_ids": [action],
        "current_state_route_set_skill_ids": [action],
        "current_state_route_factual_target_counts": {action: 1},
        "future_label_skill_ids": [action],
        "runtime_visible_catalog_id": CATALOG_ID,
        "inventory_catalog_digest": catalog_digest,
        "inventory_source": "immutable_shenzhi_retrieval_action_pool",
        "inventory_available_before_decision": True,
        "inventory_pool_size": catalog_size,
        "inventory_protocol": "global_action_pool",
        "capabilities": {
            "required_tool_set": True,
            "current_state_route_set": True,
            "ordered_next_tool": False,
            "actual_execution_result": False,
            "causal_branch_pair": False,
        },
        "provenance": {
            "source_id": "shenzhi_action_weak_bootstrap",
            "label_quality": "weak_bootstrap",
            "generator": "action_intent_templates_v1",
            "target_action": action,
            "base_action": base_action,
            "slot_index": slot_index,
        },
    }


def build(output_dir: Path, *, train_per_action: int, dev_per_action: int, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_skills": output_dir / "skills.jsonl",
        "retrieval_rows": output_dir / "retrieval_train.jsonl",
        "retrieval_dev_rows": output_dir / "retrieval_dev.jsonl",
        "static_route_rows": output_dir / "static_route_train.jsonl",
        "static_route_dev_rows": output_dir / "static_route_dev.jsonl",
        "inventory_catalogs": output_dir / "inventory_catalogs.jsonl",
    }
    skills = [dict(row) for row in ACTION_SKILLS]
    skill_ids = [str(row["skill_id"]) for row in skills]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("action skill inventory contains duplicate IDs")
    if {parse_action_skill(skill_id)[0] for skill_id in skill_ids} != set(ACTION_TEMPLATES):
        raise ValueError("action skill inventory and template base actions differ")
    with paths["training_skills"].open("w", encoding="utf-8") as handle:
        for row in skills:
            copied = dict(row)
            copied["canonical_skill_id"] = copied["skill_id"]
            copied["candidate_type"] = "RetrievalOperation"
            copied["provenance"] = {
                "source_id": "shenzhi_retrieval_backend",
                "executor_binding": copied["skill_id"],
            }
            handle.write(json.dumps(copied, ensure_ascii=False, sort_keys=True) + "\n")
    catalog_digest = json_digest(sorted(skill_ids))
    catalog = {
        "inventory_catalog_id": CATALOG_ID,
        "inventory_catalog_digest": catalog_digest,
        "inventory_source": "immutable_shenzhi_retrieval_action_pool",
        "inventory_available_before_decision": True,
        "inventory_pool_size": len(skill_ids),
        "runtime_visible_skill_ids": sorted(skill_ids),
    }
    paths["inventory_catalogs"].write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rng = random.Random(int(seed))
    counts: Counter[str] = Counter()
    for split, count, retrieval_key, route_key in (
        ("train", int(train_per_action), "retrieval_rows", "static_route_rows"),
        ("dev", int(dev_per_action), "retrieval_dev_rows", "static_route_dev_rows"),
    ):
        with paths[retrieval_key].open("w", encoding="utf-8") as retrieval, paths[
            route_key
        ].open("w", encoding="utf-8") as route:
            rows: list[dict[str, Any]] = []
            for action in sorted(skill_ids):
                base_action, slot_index = parse_action_skill(action)
                templates = ACTION_TEMPLATES[base_action]
                for index in range(count):
                    template = templates[index % len(templates)]
                    topic = TOPICS[(index + rng.randrange(len(TOPICS))) % len(TOPICS)]
                    text = template.format(topic=topic)
                    if slot_index is not None:
                        text += (
                            f" Use deterministic current-result slot {slot_index + 1}; "
                            "the executor will bind its value."
                        )
                    rows.append(
                        _row(
                            action=action,
                            base_action=base_action,
                            slot_index=slot_index,
                            text=text,
                            split=split,
                            index=index,
                            catalog_digest=catalog_digest,
                            catalog_size=len(skill_ids),
                        )
                    )
                    counts[f"{split}:{action}"] += 1
            rows.sort(key=lambda row: str(row["query_id"]))
            for row in rows:
                encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                retrieval.write(encoded)
                route.write(encoded)
    files = {key: contract_entry(path) for key, path in paths.items()}
    contract = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "task": "static_retrieval_operation_routing",
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
            "candidate_count": len(skill_ids),
        },
        "counts": {
            "skill_count": len(skill_ids),
            "retrieval_train_rows": int(train_per_action) * len(skill_ids),
            "retrieval_dev_rows": int(dev_per_action) * len(skill_ids),
            "static_route_train_rows": int(train_per_action) * len(skill_ids),
            "static_route_dev_rows": int(dev_per_action) * len(skill_ids),
            "by_split_action": dict(sorted(counts.items())),
        },
        "files": files,
        "notes": [
            "This is weak-bootstrap operation-intent supervision, not user trajectory data.",
            "Every skill is bound to one executable backend operation.",
            "Stage2 must inherit this exact action skill table rather than the paper-ID table.",
        ],
    }
    contract_path = output_dir / "data_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **contract,
        "data_contract_path": str(contract_path.resolve()),
        "data_contract_sha256": file_sha256(contract_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-per-action", type=int, default=24)
    parser.add_argument("--dev-per-action", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                Path(args.output_dir),
                train_per_action=args.train_per_action,
                dev_per_action=args.dev_per_action,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
