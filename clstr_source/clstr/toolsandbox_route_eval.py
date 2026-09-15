from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from clstr.full_base_train import UNIFIED_MEMORY_ROUTE_SCORER, _skill_id
from clstr.history_channel import audit_history_channel_rows, router_state_text
from clstr.logged_online_stage4_train import (
    attach_trajectory_prefix_online_memory_scores,
    evaluate_logged_online_stage4_rows,
)
from clstr.memory_candidate_recall import (
    candidate_recall_protocol_metadata,
    declared_candidate_pool_size,
    row_positive_skill_ids,
    source_rows_have_causal_sequence,
)
from clstr.memory_utility_gate_train import resolve_reliability_gate
from clstr.memory_utility_records import canonical_digest
from clstr.mt_ablation_eval import evaluate_mt_ablation_rows
from clstr.native_benchmark_checkpoint_adapter import (
    restore_native_benchmark_checkpoint_chain,
)
from clstr.stage0_skillrouter_baseline import _encode_skillrouter_texts, _skillrouter_skill_text
from clstr.stage4_act_train import STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE
from clstr.stage_checkpoint_init import build_clstr_model_from_stage0_checkpoint, load_head_checkpoint_into_model
from clstr.tau2_route_eval import (
    _mean,
    _mrr,
    _numeric_delta,
    _rank_position,
    _ranked_positive_fields,
    _recall_at,
    _stage0_prior_eval_from_ranked_rows,
    _write_json,
    _write_jsonl,
    rank_tau2_candidates_with_stage0_prior,
)
from clstr.tau2_skillrouter_eval import _apply_skillrouter_adapter, _rank_candidates
from clstr.toolbench_full_clstr_route_eval import strict_metric_contract, strict_stage4_metrics


@dataclass(frozen=True)
class ToolSandboxRouteCorpus:
    skills: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    report: dict[str, Any]


TOOLSANDBOX_MODEL_SKILL_POOL_MODES = frozenset(
    {"checkpoint_faithful", "local_table_rebuild"}
)
ROUTER_STATE_CONTRACT = "history_free_current_state_v1"


def toolsandbox_start_skill_id() -> str:
    return "toolsandbox/__start__"


def toolsandbox_tool_skill_id(tool_name: str) -> str:
    return f"toolsandbox/{_clean_tool_name(tool_name)}"


def _ordered_skill_ids(skill_id_to_idx: dict[str, int]) -> list[str]:
    return [
        skill_id
        for skill_id, _index in sorted(
            skill_id_to_idx.items(),
            key=lambda item: int(item[1]),
        )
    ]


def _load_toolsandbox_model_for_pool_mode(
    *,
    model_skill_pool_mode: str,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    training_skills_path: str | Path | None,
    benchmark_skills: list[dict[str, Any]],
    benchmark_skills_path: str | Path,
    model_cache_dir: str | Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, int], dict[str, Any]]:
    mode = str(model_skill_pool_mode or "").strip()
    if mode not in TOOLSANDBOX_MODEL_SKILL_POOL_MODES:
        raise ValueError(
            "model_skill_pool_mode must be checkpoint_faithful or local_table_rebuild"
        )
    if mode == "checkpoint_faithful":
        if training_skills_path is None:
            raise ValueError(
                "checkpoint_faithful ToolSandbox requires training_skills_path"
            )
        model, model_config, skill_id_to_idx, adapter_report = (
            restore_native_benchmark_checkpoint_chain(
                stage0_checkpoint_path=stage0_checkpoint_path,
                stage2_checkpoint_path=stage2_checkpoint_path,
                stage4_checkpoint_path=stage4_checkpoint_path,
                training_skills_path=training_skills_path,
                benchmark_skills=benchmark_skills,
                model_cache_dir=model_cache_dir,
                device=device,
                require_safe_memory_delta=True,
            )
        )
        return model, model_config, skill_id_to_idx, {
            "model_skill_pool_mode": mode,
            "eligible_for_checkpoint_selection": True,
            "final_model_skill_ids": _ordered_skill_ids(skill_id_to_idx),
            "routing_init": dict(adapter_report["stage0"]),
            "stage2_load": dict(adapter_report["stage2"]),
            "stage4_load": adapter_report.get("stage4"),
            "skill_pool_adapter": adapter_report,
        }

    if training_skills_path is not None:
        raise ValueError(
            "local_table_rebuild ToolSandbox must not receive training_skills_path"
        )
    skill_id_to_idx = {
        str(_skill_id(skill, idx)): idx for idx, skill in enumerate(benchmark_skills)
    }
    model, model_config, routing_report = build_clstr_model_from_stage0_checkpoint(
        checkpoint_path=Path(stage0_checkpoint_path),
        skills_path=Path(benchmark_skills_path),
        model_cache_dir=Path(model_cache_dir),
    )
    stage2_load_report = load_head_checkpoint_into_model(
        model,
        Path(stage2_checkpoint_path),
        partial_load_mode="stage2_checkpoint_compatible_state",
    )
    stage4_load_report = None
    if stage4_checkpoint_path is not None:
        stage4_load_report = load_head_checkpoint_into_model(
            model,
            Path(stage4_checkpoint_path),
            partial_load_mode="stage4_checkpoint_compatible_state",
        )
    return model, model_config, skill_id_to_idx, {
        "model_skill_pool_mode": mode,
        "eligible_for_checkpoint_selection": False,
        "final_model_skill_ids": _ordered_skill_ids(skill_id_to_idx),
        "routing_init": routing_report,
        "stage2_load": stage2_load_report,
        "stage4_load": stage4_load_report,
        "skill_pool_adapter": {},
    }


def build_toolsandbox_paired_protocol_manifest(
    *,
    checkpoint_report: dict[str, Any],
    local_rebuild_report: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_mode = str(
        (checkpoint_report.get("config") or {}).get("model_skill_pool_mode") or ""
    )
    local_mode = str(
        (local_rebuild_report.get("config") or {}).get("model_skill_pool_mode") or ""
    )
    if checkpoint_mode != "checkpoint_faithful" or local_mode != "local_table_rebuild":
        raise ValueError("ToolSandbox paired reports use the wrong pool modes")
    if checkpoint_report.get("eligible_for_checkpoint_selection") is not True:
        raise ValueError("checkpoint-faithful report is not selection eligible")
    if local_rebuild_report.get("eligible_for_checkpoint_selection") is not False:
        raise ValueError("local rebuild report must be ablation only")
    candidate_identity = str(
        checkpoint_report.get("candidate_protocol_identity_sha256") or ""
    )
    if (
        not candidate_identity
        or candidate_identity
        != str(local_rebuild_report.get("candidate_protocol_identity_sha256") or "")
    ):
        raise ValueError("ToolSandbox candidate protocol identity mismatch")
    replay_identity = str(
        checkpoint_report.get("causal_replay_identity_sha256") or ""
    )
    if (
        not replay_identity
        or replay_identity
        != str(local_rebuild_report.get("causal_replay_identity_sha256") or "")
    ):
        raise ValueError("ToolSandbox causal replay identity mismatch")
    checkpoint_mrr = float(
        (checkpoint_report.get("stage4_eval") or {}).get(
            "stage4_next_skill_mrr",
            0.0,
        )
    )
    local_mrr = float(
        (local_rebuild_report.get("stage4_eval") or {}).get(
            "stage4_next_skill_mrr",
            0.0,
        )
    )
    manifest = {
        "schema_version": "toolsandbox_paired_protocol_v1",
        "status": "ok",
        "checkpoint_selection_mode": "checkpoint_faithful",
        "local_rebuild_ablation_only": True,
        "candidate_protocol_identity_sha256": candidate_identity,
        "causal_replay_identity_sha256": replay_identity,
        "checkpoint_faithful_mrr": checkpoint_mrr,
        "local_table_rebuild_mrr": local_mrr,
        "delta_local_rebuild_minus_checkpoint_faithful_mrr": (
            local_mrr - checkpoint_mrr
        ),
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return manifest


def _toolsandbox_candidate_protocol_identity(rows: list[dict[str, Any]]) -> str:
    return canonical_digest(
        {
            "router_state_contract": ROUTER_STATE_CONTRACT,
            "rows": [
                {
                    "task_id": str(row.get("task_id") or ""),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "step_index": int(row.get("step_index") or 0),
                    "candidate_next_skill_ids": [
                        str(item)
                        for item in row.get("candidate_next_skill_ids") or []
                    ],
                }
                for row in rows
            ],
        }
    )


def _toolsandbox_causal_replay_identity(rows: list[dict[str, Any]]) -> str:
    normalized_rows = []
    for row in rows:
        normalized_prefix = []
        for step in row.get("replay_prefix") or []:
            if not isinstance(step, dict):
                continue
            normalized_prefix.append(
                {
                    key: step.get(key)
                    for key in (
                        "skill_id",
                        "action_text",
                        "observation_text",
                        "next_observation_text",
                        "trajectory_id",
                        "step_index",
                        "observation_source",
                        "replay_protocol",
                    )
                    if key in step
                }
            )
        normalized_rows.append(
            {
                "task_id": str(row.get("task_id") or ""),
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "step_index": int(row.get("step_index") or 0),
                "replay_prefix": normalized_prefix,
            }
        )
    return canonical_digest(normalized_rows)


def _clean_tool_name(tool_name: str) -> str:
    return str(tool_name or "").strip()


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_str_list(node: ast.AST | None) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    return [value for value in (_literal_str(elt) for elt in node.elts) if value]


def _literal_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _literal_int_pairs(node: ast.AST | None) -> list[tuple[int, int]] | None:
    if node is None:
        return None
    if not isinstance(node, ast.List):
        raise ValueError("ToolSandbox milestone_edge_list must be a literal list")
    output: list[tuple[int, int]] = []
    for item in node.elts:
        if not isinstance(item, (ast.Tuple, ast.List)) or len(item.elts) != 2:
            raise ValueError("ToolSandbox milestone edge must be a literal pair")
        source = _literal_int(item.elts[0])
        target = _literal_int(item.elts[1])
        if source is None or target is None:
            raise ValueError("ToolSandbox milestone edge indices must be integers")
        output.append((source, target))
    return output


def _dict_string_value(node: ast.Dict, key_name: str) -> str | None:
    for key, value in zip(node.keys, node.values):
        if _literal_str(key) == key_name:
            return _literal_str(value)
    return None


def _extract_messages(node: ast.AST | None) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    messages: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Dict):
            continue
        content = _dict_string_value(item, "content")
        if content:
            messages.append(content)
    return messages


def _literal_jsonish(node: ast.AST | None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key, value in zip(node.keys, node.values):
            key_value = _literal_jsonish(key)
            if isinstance(key_value, str):
                result[key_value] = _literal_jsonish(value)
        return result
    if isinstance(node, ast.List):
        return [_literal_jsonish(item) for item in node.elts]
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _tool_trace_from_json_dumps(node: ast.Call) -> dict[str, Any] | None:
    if not _call_name(node.func).endswith("json.dumps") and _call_name(node.func) != "json.dumps":
        return None
    if not node.args:
        return None
    payload = _literal_jsonish(node.args[0])
    if not isinstance(payload, dict):
        return None
    tool_name = _clean_tool_name(payload.get("tool_name"))
    if not tool_name:
        return None
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    return {"tool_name": tool_name, "arguments": arguments}


def _extract_required_tool_traces(node: ast.AST | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    traces: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            trace = _tool_trace_from_json_dumps(child)
            if trace is not None:
                traces.append(trace)
    return traces


def _extract_milestone_tool_traces(
    node: ast.AST | None,
) -> list[list[dict[str, Any]]]:
    if not isinstance(node, ast.List):
        return []
    output: list[list[dict[str, Any]]] = []
    for item in node.elts:
        if not isinstance(item, ast.Call) or not _call_name(item.func).endswith(
            "Milestone"
        ):
            raise ValueError("ToolSandbox milestones must be literal Milestone calls")
        output.append(_extract_required_tool_traces(item))
    return output


def _milestone_reachability(
    milestone_count: int,
    explicit_edges: list[tuple[int, int]] | None,
) -> tuple[list[list[bool]], list[tuple[int, int]], bool]:
    if explicit_edges is None:
        edges = [(index, index + 1) for index in range(max(0, milestone_count - 1))]
        explicit = False
    else:
        edges = list(explicit_edges)
        explicit = True
    reach = [[False for _ in range(milestone_count)] for _ in range(milestone_count)]
    for source, target in edges:
        if not (0 <= source < milestone_count and 0 <= target < milestone_count):
            raise ValueError("ToolSandbox milestone edge is out of range")
        if source == target:
            raise ValueError("ToolSandbox milestone DAG contains a self-loop")
        reach[source][target] = True
    for pivot in range(milestone_count):
        for source in range(milestone_count):
            if not reach[source][pivot]:
                continue
            for target in range(milestone_count):
                reach[source][target] = reach[source][target] or reach[pivot][target]
    if any(reach[index][index] for index in range(milestone_count)):
        raise ValueError("ToolSandbox milestone_edge_list is cyclic")
    return reach, edges, explicit


def _tool_route_variants(
    milestone_traces: list[list[dict[str, Any]]],
    explicit_edges: list[tuple[int, int]] | None,
    *,
    maximum_variants: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reach, edges, explicit = _milestone_reachability(
        len(milestone_traces),
        explicit_edges,
    )
    nodes: list[dict[str, Any]] = []
    for milestone_index, traces in enumerate(milestone_traces):
        for trace_index, trace in enumerate(traces):
            nodes.append(
                {
                    "node_index": len(nodes),
                    "milestone_index": milestone_index,
                    "trace_index": trace_index,
                    "trace": trace,
                }
            )
    if not nodes:
        return [], {
            "explicit_milestone_dag": explicit,
            "milestone_count": len(milestone_traces),
            "tool_event_count": 0,
            "route_variant_count": 0,
            "route_variants_truncated": False,
        }
    predecessors: dict[int, set[int]] = {int(node["node_index"]): set() for node in nodes}
    for left in nodes:
        for right in nodes:
            if reach[int(left["milestone_index"])][int(right["milestone_index"])]:
                predecessors[int(right["node_index"])].add(int(left["node_index"]))
    ordered_nodes = {
        int(node["node_index"]): node
        for node in sorted(
            nodes,
            key=lambda value: (
                int(value["milestone_index"]),
                int(value["trace_index"]),
                _clean_tool_name(value["trace"].get("tool_name")),
            ),
        )
    }
    limit = max(1, int(maximum_variants))
    variants: list[dict[str, Any]] = []
    truncated = False

    def visit(
        prefix: list[int],
        remaining: set[int],
        equivalent_names: list[list[str]],
    ) -> None:
        nonlocal truncated
        if len(variants) >= limit:
            truncated = True
            return
        if not remaining:
            variants.append(
                {
                    "required_tool_traces": [
                        dict(ordered_nodes[node_index]["trace"])
                        for node_index in prefix
                    ],
                    "equivalent_tool_names_by_step": [
                        list(names) for names in equivalent_names
                    ],
                    "tool_node_order": list(prefix),
                }
            )
            return
        completed = set(prefix)
        available = sorted(
            node_index
            for node_index in remaining
            if predecessors[node_index].issubset(completed)
        )
        if not available:
            raise ValueError("ToolSandbox tool-event partial order is cyclic")
        frontier_names = sorted(
            {
                _clean_tool_name(ordered_nodes[node_index]["trace"].get("tool_name"))
                for node_index in available
                if _clean_tool_name(
                    ordered_nodes[node_index]["trace"].get("tool_name")
                )
            }
        )
        for node_index in available:
            visit(
                [*prefix, node_index],
                remaining - {node_index},
                [*equivalent_names, frontier_names],
            )
            if len(variants) >= limit:
                truncated = bool(len(remaining) > 1 or node_index != available[-1])
                break

    visit([], set(ordered_nodes), [])
    return variants, {
        "explicit_milestone_dag": explicit,
        "milestone_count": len(milestone_traces),
        "milestone_edge_count": len(edges),
        "tool_event_count": len(nodes),
        "multi_tool_milestone_count": sum(len(traces) > 1 for traces in milestone_traces),
        "route_variant_count": len(variants),
        "route_variants_truncated": bool(truncated),
        "maximum_route_variants": limit,
        "protocol": "milestone_dag_topological_multi_positive_v1",
    }


def _scenario_records_from_file(
    path: Path,
    *,
    maximum_route_variants: int,
) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "ScenarioExtension":
            continue
        by_name = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        name = _literal_str(by_name.get("name"))
        if not name:
            continue
        milestone_traces = _extract_milestone_tool_traces(by_name.get("milestones"))
        route_variants, dag_report = _tool_route_variants(
            milestone_traces,
            _literal_int_pairs(by_name.get("milestone_edge_list")),
            maximum_variants=maximum_route_variants,
        )
        records.append(
            {
                "scenario_name": name,
                "scenario_file": path.name,
                "scenario_group": path.stem.replace("_scenarios", ""),
                "messages": _extract_messages(by_name.get("messages")),
                "tool_allow_list": _literal_str_list(by_name.get("tool_allow_list")),
                "required_tool_traces": [
                    trace for traces in milestone_traces for trace in traces
                ],
                "route_variants": route_variants,
                "dag_report": dag_report,
            }
        )
    return records


def _load_tool_docstrings(tools_root: str | Path | None) -> dict[str, str]:
    if tools_root is None:
        return {}
    root = Path(tools_root)
    if not root.exists():
        return {}
    docs: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("__"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docs[node.name] = doc.strip()
    return docs


def _skill_row(tool_name: str | None, docs: dict[str, str]) -> dict[str, Any]:
    if tool_name is None:
        description = "Initial state before any tool action."
        body = description
        return {
            "skill_id": toolsandbox_start_skill_id(),
            "name": "ToolSandbox.START",
            "description": description,
            "executor_desc": description,
            "body": body,
            "skill_md": body,
            "source_benchmark": "toolsandbox",
            "domain": "toolsandbox",
        }
    tool_name = _clean_tool_name(tool_name)
    doc = docs.get(tool_name) or f"Use tool `{tool_name}` when it is required by the current state."
    return {
        "skill_id": toolsandbox_tool_skill_id(tool_name),
        "name": tool_name,
        "description": doc,
        "executor_desc": doc,
        "body": f"Tool name: {tool_name}\n{doc}",
        "skill_md": f"Tool name: {tool_name}\n{doc}",
        "source_benchmark": "toolsandbox",
        "domain": "toolsandbox",
    }


def _state_text(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    include_history: bool = True,
) -> str:
    parts: list[str] = []
    messages = [str(item) for item in record.get("messages") or [] if str(item).strip()]
    if messages:
        parts.append("user_messages:\n" + "\n".join(messages))
    if include_history and history:
        parts.append(
            "history:\n"
            + "\n".join(
                f"{idx}. {trace['tool_name']}({trace.get('arguments') or {}})" for idx, trace in enumerate(history, start=1)
            )
        )
    return "\n".join(parts)


def _history_text(history: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{idx}. {trace['tool_name']}({trace.get('arguments') or {}})" for idx, trace in enumerate(history, start=1)
    )


def load_toolsandbox_route_corpus(
    *,
    scenarios_root: str | Path,
    tools_root: str | Path | None = None,
    max_scenarios: int | None = None,
    maximum_route_variants: int = 64,
) -> ToolSandboxRouteCorpus:
    scenarios_root = Path(scenarios_root)
    docs = _load_tool_docstrings(tools_root)
    raw_records: list[dict[str, Any]] = []
    for path in sorted(scenarios_root.glob("*_scenarios.py")):
        raw_records.extend(
            _scenario_records_from_file(
                path,
                maximum_route_variants=maximum_route_variants,
            )
        )
    skipped: Counter[str] = Counter()
    all_valid_records: list[dict[str, Any]] = []
    for record in raw_records:
        route_variants = [
            variant
            for variant in record.get("route_variants") or []
            if variant.get("required_tool_traces")
        ]
        if not route_variants:
            skipped["no_required_tool_trace"] += 1
            continue
        record = dict(record)
        record["route_variants"] = route_variants
        all_valid_records.append(record)

    valid_records = (
        all_valid_records[: max(0, int(max_scenarios))] if max_scenarios is not None else all_valid_records
    )
    tool_names: list[str] = []
    for record in valid_records:
        for tool in record.get("tool_allow_list") or []:
            if _clean_tool_name(tool):
                tool_names.append(_clean_tool_name(tool))
        for variant in record.get("route_variants") or []:
            for trace in variant.get("required_tool_traces") or []:
                tool_names.append(_clean_tool_name(trace["tool_name"]))

    seen: set[str] = set()
    unique_tool_names = [tool for tool in tool_names if not (tool in seen or seen.add(tool))]
    skills = [_skill_row(None, docs)] + [_skill_row(tool, docs) for tool in unique_tool_names]

    source_rows: list[dict[str, Any]] = []
    step_counts_by_group: Counter[str] = Counter()
    dag_counts: Counter[str] = Counter()
    for record in valid_records:
        allowed = [_clean_tool_name(tool) for tool in record.get("tool_allow_list") or [] if _clean_tool_name(tool)]
        candidate_tools = list(dict.fromkeys(allowed or unique_tool_names))
        for variant in record["route_variants"]:
            for trace in variant["required_tool_traces"]:
                tool = _clean_tool_name(trace["tool_name"])
                if tool not in candidate_tools:
                    candidate_tools.append(tool)
        candidate_ids = [toolsandbox_tool_skill_id(tool) for tool in candidate_tools]
        dag_report = record.get("dag_report") or {}
        dag_counts["explicit_dag_scenarios"] += int(
            bool(dag_report.get("explicit_milestone_dag"))
        )
        dag_counts["multi_tool_milestone_scenarios"] += int(
            int(dag_report.get("multi_tool_milestone_count") or 0) > 0
        )
        dag_counts["truncated_scenarios"] += int(
            bool(dag_report.get("route_variants_truncated"))
        )
        dag_counts["route_variants"] += len(record["route_variants"])
        for variant_index, variant in enumerate(record["route_variants"]):
            required = list(variant["required_tool_traces"])
            equivalents = list(variant["equivalent_tool_names_by_step"])
            trajectory_id = f"toolsandbox/{record['scenario_name']}"
            if len(record["route_variants"]) > 1:
                trajectory_id += f"::route{variant_index:03d}"
            for step_index, trace in enumerate(required):
                history = required[:step_index]
                previous = history[-1] if history else None
                next_tool = _clean_tool_name(trace["tool_name"])
                state_text_current = _state_text(
                    record,
                    history,
                    include_history=False,
                )
                state_text_full = _state_text(record, history, include_history=True)
                source_rows.append(
                    {
                    "benchmark": "toolsandbox",
                    "source_benchmark": "toolsandbox",
                    "split": "eval",
                    "domain": "toolsandbox",
                    "scenario_group": record["scenario_group"],
                    "task_id": f"{trajectory_id}::{step_index}",
                    "trajectory_id": trajectory_id,
                    "step_index": step_index,
                    "state_text": state_text_current,
                    "state_text_current": state_text_current,
                    "state_text_full": state_text_full,
                    "history_text": _history_text(history),
                    "action_text": f"previous_tool: {_clean_tool_name(previous['tool_name'])}" if previous else "previous_tool: START",
                    "target_action_text": (
                        f"tool: {next_tool}\narguments: "
                        + json.dumps(trace.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
                    ),
                    "split_semantic_text": "\n".join(
                        str(item) for item in record.get("messages") or [] if str(item).strip()
                    ),
                    "next_observation_text": f"oracle_next_tool_arguments: {trace.get('arguments') or {}}",
                    "observation_source": "oracle_next_tool_arguments",
                    "skill_id": toolsandbox_tool_skill_id(previous["tool_name"]) if previous else toolsandbox_start_skill_id(),
                    "next_skill_id": toolsandbox_tool_skill_id(next_tool),
                    "equivalent_next_skill_ids": [
                        toolsandbox_tool_skill_id(tool_name)
                        for tool_name in equivalents[step_index]
                    ],
                    "candidate_next_skill_ids": list(candidate_ids),
                    "loss_mask": {"routing": True, "L_trans_skill_ce": True},
                    "provenance": {
                        "source": "toolsandbox_source_milestone_trace",
                        "scenario_file": record["scenario_file"],
                        "scenario_group": record["scenario_group"],
                        "scenario_name": record["scenario_name"],
                        "route_variant_index": variant_index,
                        "route_variant_count": len(record["route_variants"]),
                        "route_protocol": "milestone_dag_topological_multi_positive_v1",
                    },
                    }
                )
                step_counts_by_group[record["scenario_group"]] += 1

    history_channel = audit_history_channel_rows(
        source_rows,
        require_explicit_current=True,
    )
    report = {
        "benchmark": "toolsandbox",
        "scenarios_root": str(scenarios_root),
        "tools_root": None if tools_root is None else str(tools_root),
        "raw_scenario_count": len(raw_records),
        "scenario_count": len(valid_records),
        "source_row_count": len(source_rows),
        "skill_count": len(skills),
        "unique_tool_count": len(unique_tool_names),
        "source_row_count_by_group": dict(sorted(step_counts_by_group.items())),
        "dag_route_supervision": {
            "protocol": "milestone_dag_topological_multi_positive_v1",
            "maximum_route_variants_per_scenario": int(maximum_route_variants),
            **dict(sorted(dag_counts.items())),
        },
        "skipped_reasons": dict(sorted(skipped.items())),
        "history_channel": history_channel,
        "causal_memory_observation_eligible": False,
        "causal_memory_observation_blocker": "source milestones omit actual tool-result observations",
    }
    return ToolSandboxRouteCorpus(skills=skills, source_rows=source_rows, report=report)


def _stage4_rows_from_ranked_toolsandbox(
    source_rows: list[dict[str, Any]],
    skill_id_to_idx: dict[str, int],
    *,
    candidate_count: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    skipped: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for row in source_rows:
        candidates = [str(item) for item in row.get("candidate_next_skill_ids") or [] if str(item) in skill_id_to_idx]
        if candidate_count is not None:
            candidates = candidates[: max(1, int(candidate_count))]
        current_skill_id = str(row.get("skill_id") or "")
        next_skill_id = str(row.get("next_skill_id") or "")
        if current_skill_id not in skill_id_to_idx:
            skipped["current_skill_not_in_pool"] += 1
            continue
        if not row_positive_skill_ids(row):
            skipped["next_skill_not_in_pool"] += 1
            continue
        positive_ids, positive_indices, positive_positions = _ranked_positive_fields(
            row,
            candidates,
            skill_id_to_idx,
        )
        if not positive_positions:
            skipped["next_positive_missing_from_candidates"] += 1
            continue
        positive_pos = positive_positions[0]
        prior_scores = list(row.get("candidate_next_prior_scores") or [])
        if len(prior_scores) != len(row.get("candidate_next_skill_ids") or []):
            prior_scores = [-float(idx) for idx in range(len(row.get("candidate_next_skill_ids") or []))]
        prior_scores = prior_scores[: len(candidates)]
        rows.append(
            {
                "task_id": row.get("task_id"),
                "trajectory_id": row.get("trajectory_id"),
                "step_index": row.get("step_index"),
                "benchmark": "toolsandbox",
                "source_benchmark": "toolsandbox",
                "state_text": router_state_text(row),
                "state_text_current": router_state_text(row),
                "state_text_full": str(row.get("state_text_full") or row.get("state_text") or ""),
                "history_text": str(row.get("history_text") or ""),
                "action_text": str(row.get("action_text") or ""),
                "next_observation_text": str(row.get("next_observation_text") or ""),
                "observation_source": str(row.get("observation_source") or ""),
                "skill_id": current_skill_id,
                "next_skill_id": next_skill_id,
                "equivalent_next_skill_ids": [
                    skill_id
                    for skill_id in row_positive_skill_ids(row)
                    if skill_id != next_skill_id
                ],
                "skill_idx": int(skill_id_to_idx[current_skill_id]),
                "positive_next_skill_ids": positive_ids,
                "positive_next_skill_indices": positive_indices,
                "positive_next_skill_positions": positive_positions,
                "positive_next_skill_idx": positive_indices[0],
                "positive_next_skill_position": int(positive_pos),
                "candidate_next_skill_ids": candidates,
                "candidate_pool_protocol": "benchmark_local",
                "candidate_next_skill_indices": [int(skill_id_to_idx[item]) for item in candidates],
                "candidate_next_prior_scores": [float(value) for value in prior_scores],
                "positive_injected": False,
                "provenance": {
                    "source": "toolsandbox_stage0_ranked",
                    "original_provenance": row.get("provenance") or {},
                    "candidate_count_requested": candidate_count,
                },
            }
        )
        candidate_counts.append(len(candidates))
    return rows, {
        "source_rows": len(source_rows),
        "stage4_rows": len(rows),
        "positive_injected_rows": 0,
        "candidate_source": "toolsandbox_stage0_ranked",
        "candidate_count_requested": candidate_count,
        "candidate_count_mean": _mean(candidate_counts),
        "skipped_reasons": dict(sorted(skipped.items())),
    }


def _attach_toolsandbox_causal_replay_prefixes(
    rows: list[dict[str, Any]],
    *,
    skill_id_to_idx: dict[str, int],
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay executed milestone tools without inventing tool-result observations."""
    max_steps = max(0, int(max_steps))
    prepared = [dict(row) for row in rows]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_index, row in enumerate(rows):
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if trajectory_id:
            grouped.setdefault(trajectory_id, []).append((row_index, row))

    rows_with_prefix = 0
    total_prefix_steps = 0
    for indexed_rows in grouped.values():
        ordered = sorted(
            indexed_rows,
            key=lambda item: (int(item[1].get("step_index") or 0), item[0]),
        )
        prior_events: list[dict[str, Any]] = []
        previous_step: int | None = None
        for _order_position, (row_index, row) in enumerate(ordered):
            step_index = int(row.get("step_index") or 0)
            if previous_step is None or step_index != previous_step + 1:
                prior_events = []
            if prior_events and max_steps > 0:
                copied = dict(row)
                copied["replay_prefix"] = [dict(step) for step in prior_events[-max_steps:]]
                prepared[row_index] = copied
                rows_with_prefix += 1
                total_prefix_steps += len(copied["replay_prefix"])

            executed_skill_id = str(row.get("next_skill_id") or "").strip()
            if executed_skill_id not in skill_id_to_idx:
                raise ValueError(
                    f"ToolSandbox causal replay skill is absent from model pool: {executed_skill_id}"
                )
            executed_tool = executed_skill_id.rsplit("/", 1)[-1]
            argument_context = str(row.get("next_observation_text") or "").strip()
            prior_events.append(
                {
                    "observation_text": router_state_text(row),
                    "action_text": (
                        f"executed_tool: {executed_tool}\n{argument_context}"
                        if argument_context
                        else f"executed_tool: {executed_tool}"
                    ),
                    "next_observation_text": "",
                    "skill_id": executed_skill_id,
                    "skill_idx": int(skill_id_to_idx[executed_skill_id]),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "step_index": step_index,
                    "observation_source": "action_only_no_tool_result",
                    "replay_protocol": "toolsandbox_oracle_action_only_v2",
                }
            )
            previous_step = step_index

    return prepared, {
        "mode": "action_only",
        "protocol": "toolsandbox_oracle_action_only_v2",
        "max_steps": max_steps,
        "row_count": len(rows),
        "trajectory_count": len(grouped),
        "rows_with_causal_replay_prefix": rows_with_prefix,
        "rows_with_action_history_replay_prefix": rows_with_prefix,
        "total_prefix_steps": total_prefix_steps,
        "observation_source": "action_only_no_tool_result",
        "actual_tool_result_observations": False,
        "eligible_claim": "action_history_memory_only",
    }


def _build_toolsandbox_report(
    *,
    output_dir: str | Path,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    skills_path: str | Path,
    source_eval_rows: int,
    retained_eval_rows: int,
    corpus_report: dict[str, Any],
    route_data_report: dict[str, Any],
    memory_report: dict[str, Any],
    stage0_prior_eval: dict[str, Any],
    base_eval: dict[str, Any],
    stage4_eval: dict[str, Any],
    config: dict[str, Any],
    stage0_prior_report: dict[str, Any],
    mt_ablation_eval: dict[str, Any] | None = None,
    model_load: dict[str, Any] | None = None,
    candidate_recall: dict[str, Any] | None = None,
    causal_replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strict_prior = strict_stage4_metrics(stage0_prior_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_base = strict_stage4_metrics(base_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    strict_stage4 = strict_stage4_metrics(stage4_eval, retained_rows=retained_eval_rows, source_rows=source_eval_rows)
    blockers: list[str] = []
    if source_eval_rows <= 0:
        blockers.append("no_source_eval_rows")
    if retained_eval_rows <= 0:
        blockers.append("no_retained_stage4_eval_rows")
    return {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "toolsandbox",
        "stage4_route": "source_milestone_required_tool_stage0_ranked_logged_online_memory",
        "route_scorer": str(config.get("route_scorer") or UNIFIED_MEMORY_ROUTE_SCORER),
        "output_dir": str(output_dir),
        "stage0_checkpoint_path": str(stage0_checkpoint_path),
        "stage2_checkpoint_path": str(stage2_checkpoint_path),
        "stage4_checkpoint_path": None if stage4_checkpoint_path is None else str(stage4_checkpoint_path),
        "skills_path": str(skills_path),
        "source_eval_rows": int(source_eval_rows),
        "retained_eval_rows": int(retained_eval_rows),
        "corpus_report": corpus_report,
        "stage0_prior_report": stage0_prior_report,
        "route_data_report": route_data_report,
        "memory_report": memory_report,
        "stage0_prior_eval": stage0_prior_eval,
        "base_eval": base_eval,
        "stage4_eval": stage4_eval,
        "mt_ablation_eval": mt_ablation_eval or {},
        "delta_base_vs_stage0_prior": _numeric_delta(base_eval, stage0_prior_eval),
        "delta_stage4_vs_base": _numeric_delta(stage4_eval, base_eval),
        "delta_stage4_vs_stage0_prior": _numeric_delta(stage4_eval, stage0_prior_eval),
        "strict": {"stage0_prior": strict_prior, "base": strict_base, "stage4": strict_stage4},
        "metric_contract": strict_metric_contract(),
        "strict_delta_stage4_vs_base": _numeric_delta(strict_stage4, strict_base),
        "candidate_recall": candidate_recall or {},
        "causal_replay": causal_replay or {},
        "config": config,
        "model_load": model_load or {},
        "paper_scope_note": (
            "ToolSandbox is evaluated here as a static oracle required-tool routing diagnostic extracted from "
            "scenario milestone source code. This is not official ToolSandbox interactive pass-rate."
        ),
    }


def run_toolsandbox_full_clstr_route_eval(
    *,
    scenarios_root: str | Path,
    tools_root: str | Path | None,
    stage0_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    stage4_checkpoint_path: str | Path | None,
    training_skills_path: str | Path | None = None,
    model_skill_pool_mode: str = "checkpoint_faithful",
    output_dir: str | Path,
    max_scenarios: int | None = None,
    max_eval_rows: int | None = None,
    candidate_count: int | None = None,
    batch_size: int = 8,
    stage0_candidate_batch_size: int = 16,
    online_memory_mode: str = "latest_exact",
    online_memory_weight: float = 1.0,
    online_memory_next_skill_bonus: float = 0.0,
    online_memory_exact_transition_bonus: float = 5.0,
    transition_residual_lambda: float = 0.5,
    transition_scoring_mode: str = STAGE0_RANK_PRIOR_TRANSITION_SCORING_MODE,
    route_scorer: str = UNIFIED_MEMORY_ROUTE_SCORER,
    reliability_mode: str = "dynamic",
    fixed_alpha: float = 1.0,
    memory_utility_gate_checkpoint_path: str | Path | None = None,
    expected_memory_utility_gate_checkpoint_sha256: str | None = None,
    expected_memory_utility_gate_audit_sha256: str | None = None,
    feature_update_count_cap: float = 1.0,
    feature_candidate_count_cap: float = 1.0,
    write_mt_ablation_report: bool = False,
    mt_ablation_auto_replay_prefix_max_steps: int = 3,
    include_mt_effect_diagnostics: bool = False,
    toolsandbox_replay_mode: str = "action_only",
) -> dict[str, Any]:
    requested_toolsandbox_replay_mode = str(toolsandbox_replay_mode).strip()
    if requested_toolsandbox_replay_mode not in {
        "legacy_auto",
        "action_only",
        "corrected_causal",
    }:
        raise ValueError(
            "toolsandbox_replay_mode must be legacy_auto, action_only, or corrected_causal"
        )
    toolsandbox_replay_mode = (
        "action_only"
        if requested_toolsandbox_replay_mode == "corrected_causal"
        else requested_toolsandbox_replay_mode
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    corpus = load_toolsandbox_route_corpus(
        scenarios_root=scenarios_root,
        tools_root=tools_root,
        max_scenarios=max_scenarios,
    )
    source_rows = corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    skills_path = output_dir / "toolsandbox_skill_pool.jsonl"
    source_rows_path = output_dir / "toolsandbox_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, skill_id_to_idx, model_pool_report = (
        _load_toolsandbox_model_for_pool_mode(
            model_skill_pool_mode=model_skill_pool_mode,
            stage0_checkpoint_path=stage0_checkpoint_path,
            stage2_checkpoint_path=stage2_checkpoint_path,
            stage4_checkpoint_path=stage4_checkpoint_path,
            training_skills_path=training_skills_path,
            benchmark_skills=corpus.skills,
            benchmark_skills_path=skills_path,
            model_cache_dir=output_dir / "model_cache",
            device=device,
        )
    )
    routing_report = dict(model_pool_report["routing_init"])
    stage2_load_report = dict(model_pool_report["stage2_load"])
    stage4_load_report = model_pool_report.get("stage4_load")
    skill_pool_adapter_report = dict(model_pool_report["skill_pool_adapter"])
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    memory_utility_gate, reliability_gate_report = resolve_reliability_gate(
        reliability_mode=reliability_mode,
        gate_checkpoint_path=memory_utility_gate_checkpoint_path,
        expected_gate_sha256=expected_memory_utility_gate_checkpoint_sha256,
        expected_audit_sha256=expected_memory_utility_gate_audit_sha256,
        device=device,
    )
    if reliability_mode == "learned":
        feature_update_count_cap = float(reliability_gate_report["feature_update_count_cap"])
        feature_candidate_count_cap = float(reliability_gate_report["feature_candidate_count_cap"])

    ranked_rows, stage0_prior_report = rank_tau2_candidates_with_stage0_prior(
        model,
        source_rows,
        skill_id_to_idx,
        batch_size=stage0_candidate_batch_size,
        device=device,
    )
    stage4_rows, route_data_report = _stage4_rows_from_ranked_toolsandbox(
        ranked_rows,
        skill_id_to_idx,
        candidate_count=candidate_count,
    )
    scored_rows, memory_report = attach_trajectory_prefix_online_memory_scores(
        stage4_rows,
        feedback_rows=stage4_rows,
        next_skill_bonus=online_memory_next_skill_bonus,
        exact_transition_bonus=online_memory_exact_transition_bonus,
        memory_mode=online_memory_mode,
    )
    if toolsandbox_replay_mode == "action_only":
        scored_rows, causal_replay_report = _attach_toolsandbox_causal_replay_prefixes(
            scored_rows,
            skill_id_to_idx=skill_id_to_idx,
            max_steps=mt_ablation_auto_replay_prefix_max_steps,
        )
    else:
        causal_replay_report = {
            "mode": "legacy_auto",
            "protocol": "generic_previous_row_v1",
            "max_steps": int(mt_ablation_auto_replay_prefix_max_steps),
        }
    candidate_protocol_identity_sha256 = _toolsandbox_candidate_protocol_identity(
        stage4_rows
    )
    causal_replay_identity_sha256 = _toolsandbox_causal_replay_identity(
        scored_rows
    )
    local_pool_size = declared_candidate_pool_size(ranked_rows)
    local_static_k = (
        local_pool_size
        if candidate_count is None
        else min(local_pool_size, max(0, int(candidate_count)))
    )
    candidate_eval_kwargs = {
        "candidate_recall_mode": "static_plus_dynamic_extra",
        "skill_id_to_idx": skill_id_to_idx,
        "static_k": local_static_k,
        "dynamic_extra_k": 64,
        "final_k": max(1, local_static_k),
    }
    stage0_prior_eval = _stage0_prior_eval_from_ranked_rows(stage4_rows)
    base_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=0.0,
        score_calibrator_enabled=False,
        route_scorer=route_scorer,
        reliability_mode="static",
        fixed_alpha=1.0,
        **candidate_eval_kwargs,
    )
    stage4_eval = evaluate_logged_online_stage4_rows(
        model,
        scored_rows,
        batch_size=batch_size,
        device=device,
        transition_residual_lambda=transition_residual_lambda,
        transition_scoring_mode=transition_scoring_mode,
        online_memory_weight=online_memory_weight,
        score_calibrator_enabled=True,
        route_scorer=route_scorer,
        reliability_mode=reliability_mode,
        fixed_alpha=fixed_alpha,
        memory_utility_gate=memory_utility_gate,
        feature_update_count_cap=feature_update_count_cap,
        feature_candidate_count_cap=feature_candidate_count_cap,
        **candidate_eval_kwargs,
    )
    mt_ablation_eval = None
    if write_mt_ablation_report:
        mt_ablation_eval = evaluate_mt_ablation_rows(
            model,
            scored_rows,
            source_rows=len(source_rows),
            retained_rows=len(stage4_rows),
            batch_size=batch_size,
            device=device,
            transition_residual_lambda=transition_residual_lambda,
            transition_scoring_mode=transition_scoring_mode,
            online_memory_weight=online_memory_weight,
            auto_replay_prefix_max_steps=mt_ablation_auto_replay_prefix_max_steps,
            route_scorer=route_scorer,
            include_pairwise_effect_diagnostics=include_mt_effect_diagnostics,
        )
    candidate_recall = candidate_recall_protocol_metadata(
        pool_protocol="benchmark_local",
        candidate_source=str(
            route_data_report.get("candidate_source") or "toolsandbox_stage0_ranked"
        ),
        legal_pool_size=local_pool_size,
        static_k=local_static_k,
        dynamic_extra_k=64,
        final_k=max(1, local_static_k),
        causal_sequential=source_rows_have_causal_sequence(source_rows),
    )
    report = _build_toolsandbox_report(
        output_dir=output_dir,
        stage0_checkpoint_path=stage0_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage4_checkpoint_path=stage4_checkpoint_path,
        skills_path=skills_path,
        source_eval_rows=len(source_rows),
        retained_eval_rows=len(stage4_rows),
        corpus_report=corpus.report,
        route_data_report=route_data_report,
        memory_report=memory_report,
        stage0_prior_eval=stage0_prior_eval,
        base_eval=base_eval,
        stage4_eval=stage4_eval,
        stage0_prior_report=stage0_prior_report,
        mt_ablation_eval=mt_ablation_eval,
        config={
            "max_scenarios": max_scenarios,
            "router_state_contract": ROUTER_STATE_CONTRACT,
            "max_eval_rows": max_eval_rows,
            "candidate_count": candidate_count,
            "batch_size": int(batch_size),
            "stage0_candidate_batch_size": int(stage0_candidate_batch_size),
            "online_memory_mode": online_memory_mode,
            "online_memory_weight": float(online_memory_weight),
            "online_memory_next_skill_bonus": float(online_memory_next_skill_bonus),
            "online_memory_exact_transition_bonus": float(online_memory_exact_transition_bonus),
            "transition_residual_lambda": float(transition_residual_lambda),
            "transition_scoring_mode": str(transition_scoring_mode),
            "route_scorer": str(route_scorer),
            "reliability_mode": reliability_mode,
            "fixed_alpha": float(fixed_alpha),
            "reliability_gate": reliability_gate_report,
            "write_mt_ablation_report": bool(write_mt_ablation_report),
            "mt_ablation_auto_replay_prefix_max_steps": int(mt_ablation_auto_replay_prefix_max_steps),
            "include_mt_effect_diagnostics": bool(include_mt_effect_diagnostics),
            "training_skills_path": None if training_skills_path is None else str(training_skills_path),
            "model_skill_pool_mode": str(model_skill_pool_mode),
            "toolsandbox_replay_mode": toolsandbox_replay_mode,
            "toolsandbox_replay_mode_requested": requested_toolsandbox_replay_mode,
            "corrected_causal_compatibility_alias_used": (
                requested_toolsandbox_replay_mode == "corrected_causal"
            ),
        },
        model_load={
            "routing_init": routing_report,
            "stage2_load": stage2_load_report,
            "stage4_load": stage4_load_report,
            "model_config": model_config,
            "skill_pool_adapter": skill_pool_adapter_report or {},
            "model_skill_pool": model_pool_report,
        },
        candidate_recall=candidate_recall,
        causal_replay=causal_replay_report,
    )
    report["eligible_for_checkpoint_selection"] = bool(
        model_pool_report["eligible_for_checkpoint_selection"]
    )
    report["candidate_protocol_identity_sha256"] = (
        candidate_protocol_identity_sha256
    )
    report["causal_replay_identity_sha256"] = causal_replay_identity_sha256
    _write_json(output_dir / "toolsandbox_full_clstr_route_eval_report.json", report)
    return report


def _toolsandbox_skillrouter_query_text(row: dict[str, Any]) -> str:
    state = str(row.get("state_text_full") or row.get("state_text") or "")
    return (
        "Instruct: Given a ToolSandbox user request, allowed tools, and tool-use history, retrieve the next "
        "required tool document.\nQuery:"
        f"{state[:1500]}"
    )


def run_toolsandbox_skillrouter_frozen_eval(
    *,
    scenarios_root: str | Path,
    tools_root: str | Path | None,
    output_dir: str | Path,
    model_name_or_path: str | Path = ".cache/hf_models/SkillRouter-Embedding-0.6B",
    max_scenarios: int | None = None,
    max_eval_rows: int | None = None,
    candidate_count: int | None = None,
    batch_size: int = 16,
    max_length: int = 2048,
    adapter_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_toolsandbox_route_corpus(
        scenarios_root=scenarios_root,
        tools_root=tools_root,
        max_scenarios=max_scenarios,
    )
    source_rows = corpus.source_rows[: max(0, int(max_eval_rows))] if max_eval_rows is not None else list(corpus.source_rows)
    skills_path = output_dir / "toolsandbox_skill_pool.jsonl"
    source_rows_path = output_dir / "toolsandbox_source_rows.jsonl"
    _write_jsonl(skills_path, corpus.skills)
    _write_jsonl(source_rows_path, source_rows)
    query_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_toolsandbox_skillrouter_query_text(row) for row in source_rows],
        batch_size=batch_size,
        max_length=max_length,
    )
    skill_embs = _encode_skillrouter_texts(
        model_name_or_path=str(model_name_or_path),
        texts=[_skillrouter_skill_text(skill) for skill in corpus.skills],
        batch_size=batch_size,
        max_length=max_length,
    )
    query_embs, skill_embs, adapter_report = _apply_skillrouter_adapter(
        query_embs=query_embs,
        skill_embs=skill_embs,
        adapter_checkpoint_path=adapter_checkpoint_path,
    )
    ranked_rows, ranking_report = _rank_candidates(
        rows=source_rows,
        skills=corpus.skills,
        query_embs=query_embs,
        skill_embs=skill_embs,
        candidate_count=candidate_count,
    )
    ranked_path = output_dir / "toolsandbox_skillrouter_ranked_rows.jsonl"
    _write_jsonl(ranked_path, ranked_rows)
    blockers: list[str] = []
    if not source_rows:
        blockers.append("no_source_eval_rows")
    if int(ranking_report.get("candidate_skill_missing_count") or 0) > 0:
        blockers.append("candidate_skill_missing")
    method = "skillrouter_finetuned_biencoder_adapter" if adapter_checkpoint_path is not None else "skillrouter_frozen_biencoder"
    report = {
        "status": "ok" if not blockers else "action_required",
        "blockers": blockers,
        "benchmark": "toolsandbox",
        "method": method,
        "output_dir": str(output_dir),
        "model_name_or_path": str(model_name_or_path),
        "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        "adapter_report": adapter_report,
        "skills_path": str(skills_path),
        "source_rows_path": str(source_rows_path),
        "ranked_rows_path": str(ranked_path),
        "source_eval_rows": len(source_rows),
        "retained_eval_rows": len(ranked_rows),
        "corpus_report": corpus.report,
        "ranking_report": ranking_report,
        "metrics": ranking_report.get("metrics", {}),
        "config": {
            "max_scenarios": max_scenarios,
            "max_eval_rows": max_eval_rows,
            "candidate_count": candidate_count,
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "adapter_checkpoint_path": None if adapter_checkpoint_path is None else str(adapter_checkpoint_path),
        },
        "paper_scope_note": (
            "This is a SkillRouter-style bi-encoder baseline on ToolSandbox source-derived required-tool routing rows. "
            "It is not official ToolSandbox interactive pass-rate."
        ),
    }
    report_name = (
        "toolsandbox_skillrouter_finetuned_eval_report.json"
        if adapter_checkpoint_path is not None
        else "toolsandbox_skillrouter_frozen_eval_report.json"
    )
    _write_json(output_dir / report_name, report)
    return report
