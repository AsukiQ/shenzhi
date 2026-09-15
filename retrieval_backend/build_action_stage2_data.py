"""Generate replayable weak-bootstrap Stage2 trajectories from real execution."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .multistep_search import (
        EXPAND_AUTHORS,
        EXPAND_KEYWORDS,
        EXPAND_SUBJECTS,
        EXPAND_VENUE_YEAR,
        SEARCH_PAPERS,
        STOP,
        MultiStepPaperSearch,
        MultiStepSearchState,
        action_skill_id,
    )
    from .paper_search import PaperSearchIndex, SearchFilters
    from .retrieval_pipeline import HybridPaperSearch
except ImportError:
    from multistep_search import (
        EXPAND_AUTHORS,
        EXPAND_KEYWORDS,
        EXPAND_SUBJECTS,
        EXPAND_VENUE_YEAR,
        SEARCH_PAPERS,
        STOP,
        MultiStepPaperSearch,
        MultiStepSearchState,
        action_skill_id,
    )
    from paper_search import PaperSearchIndex, SearchFilters
    from retrieval_pipeline import HybridPaperSearch


CURRENT_STATE_CONTRACT = "clstr_structured_current_state_v1"
CAUSAL_STATE_CONTRACT = "clstr_structured_causal_state_v1"
SCHEMA_VERSION = "shenzhi_action_vnext_stage2_data_v1"
SOURCE_ID = "shenzhi_action_executable_weak_bootstrap"


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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def contract_entry(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _load_catalog(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    if len(rows) != 1:
        raise ValueError("action Stage2 expects exactly one global inventory catalog")
    return rows[0]


def _query_pool(
    path: Path,
    *,
    limit: int,
    buffer_factor: int = 10,
) -> list[dict[str, str]]:
    """Load a buffered, deterministic query pool.

    Some queries cannot execute a planned relation expansion (for example a
    lexical query with no returned papers or no relation values).  The builder
    skips those rows, so reading exactly ``limit`` inputs can exhaust the pool
    before the requested number of trajectories is accepted.  Keep a bounded
    deterministic buffer and let the caller report exhaustion explicitly.
    """
    values: list[dict[str, str]] = []
    seen: set[str] = set()
    requested = max(0, int(limit))
    pool_limit = max(requested, requested * max(1, int(buffer_factor)))
    for row in read_jsonl(path):
        query = str(row.get("query") or row.get("query_text") or "").strip()
        query_id = str(row.get("query_id") or "").strip()
        if not query or not query_id or query in seen:
            continue
        seen.add(query)
        values.append({"query_id": query_id, "query": query})
        if len(values) >= pool_limit:
            break
    if len(values) < requested:
        raise ValueError(
            f"query source only supplied {len(values)} unique rows; "
            f"at least {requested} are required"
        )
    return values


def _canonical_event(row: dict[str, Any]) -> dict[str, Any]:
    result = str(row.get("actual_result_text") or "")
    return {
        "step_index": int(row["step_index"]),
        "skill_id": str(row["target_skill_id"]),
        "action_text": str(row["action_text"]),
        "result_text": result,
        "result_executed": bool(row.get("actual_result_executed") and result),
    }


def _causal_state(current: str, events: list[dict[str, Any]]) -> str:
    if not events:
        return current
    lines = [current, "causal_prefix:"]
    for event in events:
        step = int(event["step_index"])
        lines.append(f"event[{step}].skill_id: {event['skill_id']}")
        lines.append(f"event[{step}].action: {event['action_text']}")
        if event["result_executed"]:
            lines.append(f"event[{step}].result: {event['result_text']}")
    return "\n".join(lines)


def _state_components(query: str, step: int, max_steps: int) -> dict[str, str]:
    return {
        "goal_text": query,
        "task_text": "retrieve relevant papers through executable search operations",
        "current_observation_text": (
            f"retrieval_phase={'initial' if step == 0 else 'followup'}; "
            f"remaining_step_budget={max(0, int(max_steps) - int(step))}; filters={{}}"
        ),
    }


def _current_state(components: dict[str, str]) -> str:
    return "\n".join(
        (
            f"goal: {components['goal_text']}",
            f"task: {components['task_text']}",
            f"observation: {components['current_observation_text']}",
        )
    )


def _row_from_event(
    *,
    query_id: str,
    query: str,
    trajectory_id: str,
    split: str,
    event: Any,
    prior_rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    max_steps: int,
    pair_support_only: bool,
) -> dict[str, Any]:
    step = int(event.step_index)
    components = _state_components(query, step, max_steps)
    current = _current_state(components)
    prefix_events = [_canonical_event(row) for row in prior_rows]
    causal = _causal_state(current, prefix_events)
    result_text = str(event.result_text)
    skill_id = str(event.skill_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "task_id": trajectory_id,
        "task_group_identity": trajectory_id,
        "split_group_identity": trajectory_id,
        "query_id": query_id,
        "goal_text": query,
        "task_text": query,
        "source": SOURCE_ID,
        "benchmark": "shenzhi_multistep_paper_retrieval",
        "split": split,
        "step_index": step,
        "decision_step_index": step,
        "target_skill_id": skill_id,
        "skill_id": skill_id,
        "equivalent_next_skill_ids": [],
        "action_text": str(event.action_text),
        "actual_result_text": result_text,
        "actual_result_executed": bool(result_text),
        "result_novel_for_memory": bool(result_text),
        "state_text": current,
        "state_text_full": current,
        "state_text_current": current,
        "current_state_components": components,
        "current_state_contract": CURRENT_STATE_CONTRACT,
        "current_state_materialization_source": "executable_multistep_state_v1",
        "current_state_full_text_sha256": text_digest(current),
        "current_state_components_sha256": json_digest(components),
        "state_text_causal": causal,
        "causal_state_contract": CAUSAL_STATE_CONTRACT,
        "causal_prefix_events": prefix_events,
        "causal_prefix_event_count": len(prefix_events),
        "causal_prefix_sha256": json_digest(prefix_events),
        "causal_state_sha256": text_digest(causal),
        "zero_history_causal_equals_current": bool(not prefix_events and causal == current),
        "runtime_visible_catalog_id": str(catalog["inventory_catalog_id"]),
        "inventory_catalog_digest": str(catalog["inventory_catalog_digest"]),
        "inventory_pool_size": int(catalog["inventory_pool_size"]),
        "inventory_source": str(catalog["inventory_source"]),
        "inventory_available_before_decision": True,
        "inventory_protocol": "global_action_pool",
        "pair_support_only": bool(pair_support_only),
        "capabilities": {
            "ordered_next_tool": bool(step > 0 and not pair_support_only),
            "actual_execution_result": True,
            "causal_branch_pair": bool(pair_support_only),
        },
        "provenance": {
            "source_id": SOURCE_ID,
            "label_quality": "weak_bootstrap_executable_replay",
            "query_source_id": query_id,
            "executor_protocol": "sqlite_real_execution_v1",
            "pair_support_only": bool(pair_support_only),
        },
    }


def _execute_plan(
    executor: MultiStepPaperSearch,
    *,
    query_id: str,
    query: str,
    split: str,
    trajectory_id: str,
    expansion_bases: list[str],
    catalog: dict[str, Any],
    pair_support_only: bool,
) -> list[dict[str, Any]]:
    max_steps = len(expansion_bases) + 2
    state = MultiStepSearchState(
        query=query,
        filters=SearchFilters(),
        max_steps=max_steps,
    )
    rows: list[dict[str, Any]] = []
    requested = [SEARCH_PAPERS, *expansion_bases, STOP]
    for requested_action in requested:
        legal = executor.legal_actions(state)
        if requested_action in {SEARCH_PAPERS, STOP}:
            skill_id = requested_action
        else:
            skill_id = next(
                (
                    candidate
                    for candidate in legal
                    if candidate.startswith(requested_action + "@")
                ),
                "",
            )
        if not skill_id or skill_id not in legal:
            raise ValueError(
                f"trajectory action is not executable: {requested_action}; legal={legal[:20]}"
            )
        event = executor.execute_action(
            state,
            skill_id,
            legal_actions=legal,
            policy_metadata={"method": "weak_bootstrap_plan_v1"},
        )
        row = _row_from_event(
            query_id=query_id,
            query=query,
            trajectory_id=trajectory_id,
            split=split,
            event=event,
            prior_rows=rows,
            catalog=catalog,
            max_steps=max_steps,
            pair_support_only=pair_support_only,
        )
        rows.append(row)
    return rows


def _prefix_record(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    events = [
        {
            "skill_id": str(row["target_skill_id"]),
            "action_text": str(row["action_text"]),
            "actual_result_text": str(row["actual_result_text"]),
            "actual_result_executed": bool(row["actual_result_text"]),
        }
        for row in rows[: int(index)]
    ]
    event_digests = [json_digest(event) for event in events]
    return {
        "digest": json_digest(events),
        "event_digests": event_digests,
        "skill_counts": Counter(event["skill_id"] for event in events),
    }


def _branch_pair(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    pair_id: str,
) -> dict[str, Any]:
    index = 2
    prefix_a = _prefix_record(rows_a, index)
    prefix_b = _prefix_record(rows_b, index)
    divergence = next(
        i
        for i, values in enumerate(zip(prefix_a["event_digests"], prefix_b["event_digests"]))
        if values[0] != values[1]
    )
    return {
        "causal_pair_id": pair_id,
        "causal_pair_kind": "history_branch",
        "causal_cluster_id": pair_id,
        "source_id": SOURCE_ID,
        "causal_label_verified": True,
        "trainable_causal_label": True,
        "causal_verification": {
            "evidence_type": "verified_unordered_remaining_tool_branch_v1",
            "unordered_required_set_verified": True,
            "actual_swapped_event_results": True,
            "shared_prefix_event_count": 1,
            "verification_scope": "deterministic_executor_replay_not_human_annotation",
        },
        "history_a_digest": prefix_a["digest"],
        "history_b_digest": prefix_b["digest"],
        "first_history_divergence_index": divergence,
        "row_a": {
            "trajectory_id": rows_a[index]["trajectory_id"],
            "step_index": index,
            "target_skill_id": rows_a[index]["target_skill_id"],
        },
        "row_b": {
            "trajectory_id": rows_b[index]["trajectory_id"],
            "step_index": index,
            "target_skill_id": rows_b[index]["target_skill_id"],
        },
    }


def build(
    *,
    db_path: Path,
    query_source: Path,
    stage0_data_dir: Path,
    output_dir: Path,
    train_queries: int,
    dev_queries: int,
    train_pair_queries: int,
    dev_pair_queries: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = stage0_data_dir / "inventory_catalogs.jsonl"
    skills_path = stage0_data_dir / "skills.jsonl"
    catalog = _load_catalog(catalog_path)
    total = int(train_queries) + int(dev_queries) + int(train_pair_queries) + int(dev_pair_queries)
    queries = _query_pool(query_source, limit=total, buffer_factor=10)
    executor = MultiStepPaperSearch(
        HybridPaperSearch(PaperSearchIndex(db_path), recall_k=100),
        expansion_seed_k=12,
        candidate_limit=100,
    )
    ordinary = {"train": [], "dev": []}
    pair_support = {"train": [], "dev": []}
    pairs = {"train": [], "dev": []}
    cursor = 0
    ordinary_plans = (
        [EXPAND_AUTHORS, EXPAND_KEYWORDS],
        [EXPAND_KEYWORDS, EXPAND_SUBJECTS],
        [EXPAND_SUBJECTS, EXPAND_VENUE_YEAR],
        [EXPAND_VENUE_YEAR, EXPAND_AUTHORS],
    )
    skipped: Counter[str] = Counter()

    def next_query(split: str, accepted: int) -> dict[str, str]:
        if cursor >= len(queries):
            raise ValueError(
                "query pool exhausted before requested executable trajectories "
                f"were built (split={split}, accepted={accepted}, "
                f"pool_size={len(queries)}, skipped={sum(skipped.values())})"
            )
        return queries[cursor]

    for split, count in (("train", train_queries), ("dev", dev_queries)):
        accepted = 0
        attempts = 0
        while accepted < int(count):
            item = next_query(split, accepted)
            cursor += 1
            plan = ordinary_plans[attempts % len(ordinary_plans)]
            attempts += 1
            try:
                rows = _execute_plan(
                    executor,
                    query_id=item["query_id"],
                    query=item["query"],
                    split=split,
                    trajectory_id=f"ordinary/{split}/{accepted:05d}",
                    expansion_bases=list(plan),
                    catalog=catalog,
                    pair_support_only=False,
                )
            except ValueError as exc:
                skipped[f"ordinary:{split}:{str(exc).split(':', 1)[0]}"] += 1
                continue
            ordinary[split].extend(rows)
            accepted += 1
    for split, count in (("train", train_pair_queries), ("dev", dev_pair_queries)):
        accepted = 0
        while accepted < int(count):
            item = next_query(f"pair:{split}", accepted)
            cursor += 1
            try:
                rows_a = _execute_plan(
                    executor,
                    query_id=item["query_id"],
                    query=item["query"],
                    split=split,
                    trajectory_id=f"pair/{split}/{accepted:05d}/author_first",
                    expansion_bases=[EXPAND_AUTHORS, EXPAND_KEYWORDS],
                    catalog=catalog,
                    pair_support_only=True,
                )
                rows_b = _execute_plan(
                    executor,
                    query_id=item["query_id"],
                    query=item["query"],
                    split=split,
                    trajectory_id=f"pair/{split}/{accepted:05d}/keyword_first",
                    expansion_bases=[EXPAND_KEYWORDS, EXPAND_AUTHORS],
                    catalog=catalog,
                    pair_support_only=True,
                )
                pair = _branch_pair(
                    rows_a,
                    rows_b,
                    pair_id=f"branch/{split}/{accepted:05d}",
                )
            except (StopIteration, ValueError) as exc:
                skipped[f"pair:{split}:{type(exc).__name__}"] += 1
                continue
            pair_support[split].extend([*rows_a, *rows_b])
            pairs[split].append(pair)
            accepted += 1
    paths = {
        "training_skills": skills_path,
        "trajectory_rows": output_dir / "trajectory_train.jsonl",
        "trajectory_dev_rows": output_dir / "trajectory_dev.jsonl",
        "causal_pair_support_rows": output_dir / "causal_pair_support_train.jsonl",
        "causal_pair_support_dev_rows": output_dir / "causal_pair_support_dev.jsonl",
        "inventory_catalogs": catalog_path,
        "causal_branch_pairs": output_dir / "causal_branch_pairs_train.jsonl",
        "causal_branch_dev_pairs": output_dir / "causal_branch_pairs_dev.jsonl",
    }
    write_jsonl(paths["trajectory_rows"], ordinary["train"])
    write_jsonl(paths["trajectory_dev_rows"], ordinary["dev"])
    write_jsonl(paths["causal_pair_support_rows"], pair_support["train"])
    write_jsonl(paths["causal_pair_support_dev_rows"], pair_support["dev"])
    write_jsonl(paths["causal_branch_pairs"], pairs["train"])
    write_jsonl(paths["causal_branch_dev_pairs"], pairs["dev"])
    files = {key: contract_entry(path) for key, path in paths.items()}
    contract = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "task": "multi_step_retrieval_operation_routing",
        "label_quality": "weak_bootstrap_executable_replay",
        "stage_contract": "canonical_clstr_vnext_stage2",
        "execution_contract": {
            "backend": "sqlite_fts5_real_execution",
            "arguments_bound_from_current_results": True,
            "result_text_from_actual_executor": True,
            "human_trajectory_annotation_claimed": False,
            "causal_pair_evidence": "deterministic_swapped_order_replay",
        },
        "inventory": {
            "catalog_id": catalog["inventory_catalog_id"],
            "catalog_digest": catalog["inventory_catalog_digest"],
            "candidate_count": catalog["inventory_pool_size"],
        },
        "counts": {
            "ordinary_train_trajectories": int(train_queries),
            "ordinary_dev_trajectories": int(dev_queries),
            "ordinary_train_rows": len(ordinary["train"]),
            "ordinary_dev_rows": len(ordinary["dev"]),
            "pair_support_train_trajectories": int(train_pair_queries) * 2,
            "pair_support_dev_trajectories": int(dev_pair_queries) * 2,
            "pair_support_train_rows": len(pair_support["train"]),
            "pair_support_dev_rows": len(pair_support["dev"]),
            "causal_branch_train_pairs": len(pairs["train"]),
            "causal_branch_dev_pairs": len(pairs["dev"]),
            "skipped": dict(sorted(skipped.items())),
        },
        "source_files": {
            "query_source": contract_entry(query_source),
            "sqlite_db": contract_entry(db_path),
            "stage0_data_contract": contract_entry(stage0_data_dir / "data_contract.json"),
        },
        "files": files,
        "notes": [
            "All trajectory actions were executed by the same backend used for online inference.",
            "These labels are deterministic weak bootstrap, not observed human sessions.",
            "Causal branch labels verify swapped executable operation order, not user preference.",
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
    parser.add_argument("--db", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--stage0-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-queries", type=int, default=96)
    parser.add_argument("--dev-queries", type=int, default=48)
    parser.add_argument("--train-pair-queries", type=int, default=40)
    parser.add_argument("--dev-pair-queries", type=int, default=24)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                db_path=Path(args.db),
                query_source=Path(args.queries),
                stage0_data_dir=Path(args.stage0_data_dir),
                output_dir=Path(args.output_dir),
                train_queries=args.train_queries,
                dev_queries=args.dev_queries,
                train_pair_queries=args.train_pair_queries,
                dev_pair_queries=args.dev_pair_queries,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
