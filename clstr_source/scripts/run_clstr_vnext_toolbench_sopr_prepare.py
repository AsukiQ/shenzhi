#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clstr.stabletoolbench_clstr_queries import _query_id, _query_text
from clstr.stabletoolbench_toolenv import build_stabletoolbench_toolenv_from_queries
from clstr.vnext_online_selector import VNextOnlineSelector


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _toolbench_skill_to_api(skill: dict[str, Any]) -> dict[str, Any]:
    skill_id = str(skill.get("skill_id") or "")
    parts = skill_id.split("/")
    if len(parts) != 3 or parts[0] != "toolbench-g3":
        raise ValueError(f"not an executable ToolBench-G3 skill: {skill_id}")
    schema = skill.get("input_schema")
    if not isinstance(schema, dict):
        schema = {}
    api: dict[str, Any] = {
        "category_name": str(skill.get("environment") or "Unknown"),
        "tool_name": parts[1],
        "api_name": str(skill.get("name") or parts[2]),
        "api_description": str(
            skill.get("executor_desc") or skill.get("description") or parts[2]
        ),
        "required_parameters": list(schema.get("required_parameters") or []),
        "optional_parameters": list(schema.get("optional_parameters") or []),
        "method": schema.get("method") or "GET",
    }
    output_schema = skill.get("output_schema")
    if output_schema not in (None, {}, []):
        api["template_response"] = output_schema
    return api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CLSTR-routed StableToolBench queries/toolenv for official "
            "raw generation, answer conversion, and SoPR/pass-rate evaluation."
        )
    )
    parser.add_argument("--matched_release_selection_path", required=True)
    parser.add_argument("--stage2_checkpoint_path", default=None)
    parser.add_argument("--training_skills_path", default=None)
    parser.add_argument("--benchmark_skills_path", default=None)
    parser.add_argument("--original_query_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--coarse_k", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_path = Path(args.original_query_file).resolve()
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError("StableToolBench original query file must be a non-empty list")
    if args.max_queries is not None:
        if int(args.max_queries) <= 0:
            raise ValueError("max_queries must be positive")
        queries = queries[: int(args.max_queries)]
    selector = VNextOnlineSelector(
        matched_release_selection_path=args.matched_release_selection_path,
        stage2_checkpoint_path=args.stage2_checkpoint_path,
        training_skills_path=args.training_skills_path,
        benchmark_skills_path=args.benchmark_skills_path,
        device=args.device,
        coarse_k=args.coarse_k,
        dynamic_extra_k=args.dynamic_extra_k,
    )
    executable_skill_ids = [
        skill_id
        for skill_id in selector.skill_ids
        if skill_id.startswith("toolbench-g3/")
    ]
    if not executable_skill_ids:
        raise ValueError("CLSTR release contains no executable ToolBench-G3 APIs")
    routed_queries: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for index, raw_query in enumerate(queries):
        if not isinstance(raw_query, dict):
            raise ValueError(f"StableToolBench query {index} must be an object")
        query_id = _query_id(raw_query, index)
        query_text = _query_text(raw_query).strip()
        if not query_text:
            raise ValueError(f"StableToolBench query {query_id} has no text")
        session = selector.new_session()
        selected = session.select(
            f"goal: {query_text}",
            candidate_skill_ids=executable_skill_ids,
            top_k=min(int(args.top_k), len(executable_skill_ids)),
        )
        api_list = [_toolbench_skill_to_api(item["skill"]) for item in selected]
        if not api_list:
            raise ValueError(f"StableToolBench query {query_id} selected no APIs")
        routed = dict(raw_query)
        routed["api_list"] = api_list
        routed_queries.append(routed)
        selection_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "method": selector.method,
                "history_depth": 0,
                "selected": [
                    {
                        "rank": item["rank"],
                        "skill_id": item["skill_id"],
                        "name": str(item["skill"].get("name") or ""),
                        "score": item["score"],
                        "static_score": item["static_score"],
                        "dynamic_score": item["dynamic_score"],
                        "selector_probability": item["selector_probability"],
                        "mixture_probability": item["mixture_probability"],
                    }
                    for item in selected
                ],
            }
        )
    routed_query_path = output_dir / "routed_queries.json"
    selection_path = output_dir / "retrieval_selections.jsonl"
    toolenv_root = output_dir / "toolenv" / "tools"
    toolenv_manifest_path = output_dir / "toolenv_manifest.json"
    _write_json(routed_query_path, routed_queries)
    _write_jsonl(selection_path, selection_rows)
    toolenv_report = build_stabletoolbench_toolenv_from_queries(
        query_file=routed_query_path,
        output_root=toolenv_root,
        manifest_path=toolenv_manifest_path,
    )
    if toolenv_report.get("status") != "ok":
        raise ValueError(
            "StableToolBench routed toolenv preparation failed: "
            + ", ".join(toolenv_report.get("blockers") or [])
        )
    report = {
        "status": "ok",
        "metric_scope": (
            "CLSTR-assisted initial ToolBench API-set retrieval only; official "
            "SoPR/pass rate is produced after StableToolBench raw generation, "
            "answer conversion, and eval_pass_rate.py. The StableToolBench QA "
            "executor does not update recurrent CLSTR memory inside an episode."
        ),
        "method": selector.method,
        "evaluation_scope": "full" if args.max_queries is None else "smoke",
        "query_count": len(routed_queries),
        "executable_candidate_count": len(executable_skill_ids),
        "top_k": min(int(args.top_k), len(executable_skill_ids)),
        "coarse_k": min(int(args.coarse_k), len(executable_skill_ids)),
        "recurrent_updates_inside_stabletoolbench_executor": False,
        "routed_query_path": str(routed_query_path),
        "selection_path": str(selection_path),
        "toolenv_root": str(toolenv_root),
        "toolenv_manifest_path": str(toolenv_manifest_path),
        "checkpoint_binding": selector.checkpoint_binding,
    }
    _write_json(output_dir / "preparation_report.json", report)
    return report


def main() -> int:
    report = run_from_args(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
