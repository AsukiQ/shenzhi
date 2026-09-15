from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SKILL_STOPWORDS = {
    "api",
    "bench",
    "g3",
    "get",
    "tool",
    "toolbench",
    "toolbenchg3",
    "traject",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top1_skill_id(row: dict[str, Any]) -> str:
    top1 = str(row.get("top1_skill_id") or "")
    if top1:
        return top1
    top_ids = row.get("top_skill_ids")
    if isinstance(top_ids, list) and top_ids:
        return str(top_ids[0])
    return ""


def skill_provider(skill_id: str, *, depth: int = 2) -> str:
    parts = [part for part in str(skill_id or "").split("/") if part]
    if not parts:
        return ""
    return "/".join(parts[: max(1, depth)])


def skill_tokens(skill_id: str) -> set[str]:
    tokens = set(_TOKEN_RE.findall(str(skill_id or "").lower().replace("_", "-")))
    return {token for token in tokens if token not in _SKILL_STOPWORDS and len(token) > 1}


def token_jaccard(left: str, right: str) -> float:
    left_tokens = skill_tokens(left)
    right_tokens = skill_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def classify_transition_row(row: dict[str, Any]) -> dict[str, bool | float | str | None]:
    stage0_rank = _as_int(row.get("stage0_gold_rank"))
    stage2_rank = _as_int(row.get("stage2_gold_rank"))
    current_skill_id = str(row.get("skill_id") or "")
    gold_skill_id = str(row.get("next_skill_id") or "")
    top1_skill_id = _top1_skill_id(row)
    top1_similarity = token_jaccard(top1_skill_id, gold_skill_id)
    stage2_worse = stage0_rank is not None and stage2_rank is not None and stage2_rank > stage0_rank
    stage2_improved = stage0_rank is not None and stage2_rank is not None and stage2_rank < stage0_rank
    stage0_top5_stage2_miss = stage0_rank is not None and stage2_rank is not None and stage0_rank <= 5 < stage2_rank
    stage0_top20_stage2_miss = stage0_rank is not None and stage2_rank is not None and stage0_rank <= 20 < stage2_rank
    stage0_low_rank = stage0_rank is not None and stage0_rank > 20
    stage2_saved_low_rank = stage0_rank is not None and stage2_rank is not None and stage0_rank > 20 and stage2_rank <= 5
    return {
        "stage2_worse_than_stage0": stage2_worse,
        "stage2_improved_vs_stage0": stage2_improved,
        "stage0_top5_stage2_miss": stage0_top5_stage2_miss,
        "stage0_top20_stage2_miss": stage0_top20_stage2_miss,
        "stage0_low_rank": stage0_low_rank,
        "stage2_saved_low_rank": stage2_saved_low_rank,
        "top1_is_current_skill": bool(top1_skill_id and top1_skill_id == current_skill_id),
        "top1_equals_gold": bool(top1_skill_id and top1_skill_id == gold_skill_id),
        "top1_same_root_namespace_as_gold": bool(
            top1_skill_id and gold_skill_id and skill_provider(top1_skill_id, depth=1) == skill_provider(gold_skill_id, depth=1)
        ),
        "top1_same_provider_as_gold": bool(
            top1_skill_id and gold_skill_id and skill_provider(top1_skill_id) == skill_provider(gold_skill_id)
        ),
        "top1_probably_equivalent_to_gold": bool(
            top1_skill_id and gold_skill_id and top1_skill_id != gold_skill_id and top1_similarity >= 0.6
        ),
        "self_transition": bool(current_skill_id and current_skill_id == gold_skill_id),
        "top1_gold_token_jaccard": float(top1_similarity),
        "top1_provider": skill_provider(top1_skill_id),
        "gold_provider": skill_provider(gold_skill_id),
    }


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 1:
        return "01"
    if rank <= 5:
        return "02-05"
    if rank <= 20:
        return "06-20"
    if rank <= 100:
        return "021-100"
    return "101+"


def _step_bucket(step: int | None) -> str:
    if step is None:
        return "missing"
    if step <= 0:
        return "0"
    if step == 1:
        return "1"
    if step == 2:
        return "2"
    if step <= 5:
        return "3-5"
    return "6+"


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not rows:
        return {
            "row_count": 0,
            "stage0_recall@5": 0.0,
            "stage2_recall@5": 0.0,
            "stage0_top5_stage2_miss_count": 0,
            "stage0_low_rank_count": 0,
        }
    enriched = [(row, classify_transition_row(row)) for row in rows]

    def fraction(flag: str) -> float:
        return sum(1 for _row, flags in enriched if bool(flags.get(flag))) / count

    def count_flag(flag: str) -> int:
        return sum(1 for _row, flags in enriched if bool(flags.get(flag)))

    def rank_values(key: str) -> list[int]:
        values = [_as_int(row.get(key)) for row in rows]
        return [value for value in values if value is not None]

    stage0_ranks = rank_values("stage0_gold_rank")
    stage2_ranks = rank_values("stage2_gold_rank")
    provider_confusions = Counter(
        f"{flags.get('gold_provider')} -> {flags.get('top1_provider')}"
        for row, flags in enriched
        if not bool(flags.get("top1_equals_gold")) and flags.get("gold_provider") and flags.get("top1_provider")
    )
    return {
        "row_count": count,
        "stage0_recall@5": sum(1 for rank in stage0_ranks if rank <= 5) / count,
        "stage0_recall@20": sum(1 for rank in stage0_ranks if rank <= 20) / count,
        "stage2_recall@5": sum(1 for rank in stage2_ranks if rank <= 5) / count,
        "stage2_recall@20": sum(1 for rank in stage2_ranks if rank <= 20) / count,
        "mean_stage0_gold_rank": mean(stage0_ranks) if stage0_ranks else None,
        "mean_stage2_gold_rank": mean(stage2_ranks) if stage2_ranks else None,
        "stage2_worse_than_stage0_count": count_flag("stage2_worse_than_stage0"),
        "stage2_worse_than_stage0_fraction": fraction("stage2_worse_than_stage0"),
        "stage2_improved_vs_stage0_count": count_flag("stage2_improved_vs_stage0"),
        "stage2_improved_vs_stage0_fraction": fraction("stage2_improved_vs_stage0"),
        "stage0_top5_stage2_miss_count": count_flag("stage0_top5_stage2_miss"),
        "stage0_top5_stage2_miss_fraction": fraction("stage0_top5_stage2_miss"),
        "stage0_top20_stage2_miss_count": count_flag("stage0_top20_stage2_miss"),
        "stage0_top20_stage2_miss_fraction": fraction("stage0_top20_stage2_miss"),
        "stage0_low_rank_count": count_flag("stage0_low_rank"),
        "stage0_low_rank_fraction": fraction("stage0_low_rank"),
        "stage2_saved_low_rank_count": count_flag("stage2_saved_low_rank"),
        "stage2_saved_low_rank_fraction": fraction("stage2_saved_low_rank"),
        "top1_is_current_skill_count": count_flag("top1_is_current_skill"),
        "top1_is_current_skill_fraction": fraction("top1_is_current_skill"),
        "top1_same_root_namespace_as_gold_count": count_flag("top1_same_root_namespace_as_gold"),
        "top1_same_root_namespace_as_gold_fraction": fraction("top1_same_root_namespace_as_gold"),
        "top1_same_provider_as_gold_count": count_flag("top1_same_provider_as_gold"),
        "top1_same_provider_as_gold_fraction": fraction("top1_same_provider_as_gold"),
        "top1_probably_equivalent_to_gold_count": count_flag("top1_probably_equivalent_to_gold"),
        "top1_probably_equivalent_to_gold_fraction": fraction("top1_probably_equivalent_to_gold"),
        "self_transition_count": count_flag("self_transition"),
        "self_transition_fraction": fraction("self_transition"),
        "top_provider_confusions": dict(provider_confusions.most_common(20)),
    }


def _breakdown(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {key: _stats(group_rows) for key, group_rows in sorted(groups.items())}


def summarize_transition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered = list(rows)
    return {
        "overall": _stats(filtered),
        "by_benchmark": _breakdown(filtered, lambda row: row.get("benchmark") or "unknown"),
        "by_transition_type": _breakdown(
            filtered,
            lambda row: "self" if str(row.get("skill_id") or "") == str(row.get("next_skill_id") or "") else "switch",
        ),
        "by_benchmark_transition_type": _breakdown(
            filtered,
            lambda row: (
                f"{row.get('benchmark') or 'unknown'}::"
                f"{'self' if str(row.get('skill_id') or '') == str(row.get('next_skill_id') or '') else 'switch'}"
            ),
        ),
        "by_stage0_rank_bucket": _breakdown(filtered, lambda row: _rank_bucket(_as_int(row.get("stage0_gold_rank")))),
        "by_step_bucket": _breakdown(filtered, lambda row: _step_bucket(_as_int(row.get("step_index")))),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("benchmark") or ""),
        str(row.get("task_id") or ""),
        str(row.get("trajectory_id") or ""),
        str(row.get("step_index") if row.get("step_index") is not None else ""),
    )


def _context_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "goal_text",
        "task_text",
        "state_text",
        "history_text",
        "action_text",
        "next_action_text",
        "next_observation_text",
        "candidate_source",
        "source_quality",
        "provenance",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _load_contexts(
    trajectories_path: str | Path | None,
    rows: list[dict[str, Any]],
    benchmark_caps: dict[str, int] | None,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    if trajectories_path is None:
        return {}, {}
    path = Path(trajectories_path)
    if not path.exists():
        return {}, {}
    target_indices = {_as_int(row.get("row_index")) for row in rows}
    target_indices.discard(None)
    target_keys = {_row_key(row) for row in rows}
    by_index: dict[int, dict[str, Any]] = {}
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    retained_by_benchmark: Counter[str] = Counter()
    retained_index = 0
    benchmark_caps = benchmark_caps or {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark = str(row.get("benchmark") or "")
            cap = benchmark_caps.get(benchmark)
            if cap is not None and cap >= 0 and retained_by_benchmark[benchmark] >= cap:
                continue
            retained_by_benchmark[benchmark] += 1
            row_key = _row_key(row)
            if retained_index in target_indices:
                by_index[retained_index] = _context_subset(row)
            if row_key in target_keys:
                by_key[row_key] = _context_subset(row)
            retained_index += 1
            if len(by_index) >= len(target_indices) and len(by_key) >= len(target_keys):
                break
    return by_index, by_key


def _case_predicates() -> dict[str, Any]:
    return {
        "stage0_top5_stage2_miss": lambda row, flags: bool(flags["stage0_top5_stage2_miss"]),
        "stage2_worse_than_stage0": lambda row, flags: bool(flags["stage2_worse_than_stage0"]),
        "stage0_low_rank_miss": lambda row, flags: bool(flags["stage0_low_rank"]) and _as_int(row.get("stage2_gold_rank")) > 5,
        "stage2_saved_low_rank": lambda row, flags: bool(flags["stage2_saved_low_rank"]),
        "top1_current_skill_miss": lambda row, flags: bool(flags["top1_is_current_skill"])
        and not bool(flags["top1_equals_gold"]),
        "top1_probably_equivalent_miss": lambda row, flags: bool(flags["top1_probably_equivalent_to_gold"])
        and not bool(flags["top1_equals_gold"]),
    }


def _example_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    stage0_rank = _as_int(row.get("stage0_gold_rank")) or 0
    stage2_rank = _as_int(row.get("stage2_gold_rank")) or 0
    return (stage2_rank - stage0_rank, stage2_rank, stage0_rank)


def _example_payload(
    *,
    case: str,
    row: dict[str, Any],
    context_by_index: dict[int, dict[str, Any]],
    context_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    row_index = _as_int(row.get("row_index"))
    flags = classify_transition_row(row)
    context = context_by_index.get(row_index) if row_index is not None else None
    if context is None:
        context = context_by_key.get(_row_key(row), {})
    return {
        "case": case,
        "row_index": row.get("row_index"),
        "benchmark": row.get("benchmark"),
        "task_id": row.get("task_id"),
        "trajectory_id": row.get("trajectory_id"),
        "step_index": row.get("step_index"),
        "skill_id": row.get("skill_id"),
        "next_skill_id": row.get("next_skill_id"),
        "top1_skill_id": _top1_skill_id(row),
        "stage0_gold_rank": row.get("stage0_gold_rank"),
        "stage2_gold_rank": row.get("stage2_gold_rank"),
        "stage2_cross_entropy": row.get("stage2_cross_entropy"),
        "flags": flags,
        "action_text": row.get("action_text"),
        "next_action_text": row.get("next_action_text"),
        "top_skill_ids": row.get("top_skill_ids"),
        "top_scores": row.get("top_scores"),
        "context": context,
    }


def select_audit_examples(
    rows: list[dict[str, Any]],
    *,
    example_limit: int = 20,
    context_by_index: dict[int, dict[str, Any]] | None = None,
    context_by_key: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    context_by_index = context_by_index or {}
    context_by_key = context_by_key or {}
    predicates = _case_predicates()
    examples: list[dict[str, Any]] = []
    for benchmark in sorted({str(row.get("benchmark") or "unknown") for row in rows}):
        benchmark_rows = [row for row in rows if str(row.get("benchmark") or "unknown") == benchmark]
        for case, predicate in predicates.items():
            matched = [
                row
                for row in benchmark_rows
                if predicate(row, classify_transition_row(row))
            ]
            matched.sort(key=_example_sort_key, reverse=True)
            examples.extend(
                _example_payload(
                    case=case,
                    row=row,
                    context_by_index=context_by_index,
                    context_by_key=context_by_key,
                )
                for row in matched[: max(0, int(example_limit))]
            )
    return examples


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage2 v4 Pre-Audit",
        "",
        f"- status: `{report.get('status')}`",
        f"- row diagnostics: `{report.get('row_diagnostics_path')}`",
        f"- trajectories: `{report.get('trajectories_path')}`",
        f"- examples: `{report.get('examples_path')}`",
        "",
        "## By Benchmark",
        "",
        "| benchmark | rows | Stage0 r@5 | Stage2 r@5 | top5 pushed out | Stage0 low rank | Stage2 worse | top1=current | same root | same provider |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, metrics in (report.get("summary") or {}).get("by_benchmark", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(benchmark),
                    str(metrics.get("row_count")),
                    _format_metric(metrics.get("stage0_recall@5")),
                    _format_metric(metrics.get("stage2_recall@5")),
                    str(metrics.get("stage0_top5_stage2_miss_count")),
                    str(metrics.get("stage0_low_rank_count")),
                    _format_metric(metrics.get("stage2_worse_than_stage0_fraction")),
                    _format_metric(metrics.get("top1_is_current_skill_fraction")),
                    _format_metric(metrics.get("top1_same_root_namespace_as_gold_fraction")),
                    _format_metric(metrics.get("top1_same_provider_as_gold_fraction")),
                ]
            )
            + " |"
        )
    benchmark_transition = (report.get("summary") or {}).get("by_benchmark_transition_type") or {}
    if benchmark_transition:
        lines.extend(
            [
                "",
                "## By Benchmark x Transition Type",
                "",
                "| group | rows | Stage0 r@5 | Stage2 r@5 | top5 pushed out | Stage0 low rank | Stage2 worse | mean Stage0 rank | mean Stage2 rank |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, metrics in benchmark_transition.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(group),
                        str(metrics.get("row_count")),
                        _format_metric(metrics.get("stage0_recall@5")),
                        _format_metric(metrics.get("stage2_recall@5")),
                        str(metrics.get("stage0_top5_stage2_miss_count")),
                        str(metrics.get("stage0_low_rank_count")),
                        _format_metric(metrics.get("stage2_worse_than_stage0_fraction")),
                        _format_metric(metrics.get("mean_stage0_gold_rank")),
                        _format_metric(metrics.get("mean_stage2_gold_rank")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report.get("interpretation", []))
    lines.extend(["", "## Recommended Stage2 v4 Actions", ""])
    lines.extend(f"- {item}" for item in report.get("recommended_actions", []))
    return "\n".join(lines) + "\n"


def _interpret(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    by_benchmark = summary.get("by_benchmark", {})
    toolbench = by_benchmark.get("toolbench_g3", {})
    traject = by_benchmark.get("traject_bench", {})
    if toolbench:
        if float(toolbench.get("top1_same_root_namespace_as_gold_fraction") or 0.0) < 0.5:
            notes.append(
                "ToolBench-G3 top1 predictions are usually outside the ToolBench-G3 root namespace; Stage2 is ranking "
                "cross-dataset skills from the unified pool, so v4 needs an available-skill/inventory mask."
            )
        if float(toolbench.get("stage2_worse_than_stage0_fraction") or 0.0) > 0.5:
            notes.append(
                "ToolBench-G3 shows systematic reranker degradation: Stage2 worsens more than half of rows, "
                "so the first fix should target Stage2 labels/input/loss instead of retraining Stage0."
            )
        if float(toolbench.get("top1_is_current_skill_fraction") or 0.0) > 0.2:
            notes.append(
                "ToolBench-G3 has substantial top1=current-skill stickiness; transition input/loss should explicitly "
                "separate self-transition from next-skill switching."
            )
    if traject:
        if float(traject.get("stage0_low_rank_fraction") or 0.0) > 0.5:
            notes.append(
                "TrajectBench is dominated by low Stage0 gold ranks; Stage2 can help but Stage0 sequential retrieval "
                "or candidate construction likely needs a v4 pass after label audit."
            )
        if float(traject.get("stage2_improved_vs_stage0_fraction") or 0.0) > 0.5:
            notes.append(
                "TrajectBench Stage2 often improves rank, which suggests the transition signal is useful but top-M "
                "candidate quality/top5 calibration is insufficient."
            )
    return notes


def _recommended_actions(summary: dict[str, Any]) -> list[str]:
    return [
        "Do not launch another full Stage2 job before inspecting the example JSONL cases.",
        "Add a generic available-skill inventory mask before Stage2 reranking; this is not ToolBench-specific and prevents invalid cross-environment skills.",
        "For ToolBench-G3, add multi-positive next-skill labels and listwise loss over top-M candidates, then smoke-test no-inject eval.",
        "For ToolBench-G3, audit self-transition rows separately from switch rows; current-skill stickiness should not be rewarded on switch rows.",
        "For TrajectBench, decide Stage0 v4 only after checking whether low-rank examples are caused by query construction or label equivalence.",
        "Regenerate row diagnostics later with persisted Stage0 top candidate ids; the current artifact has Stage0 gold rank but not Stage0 top-k ids.",
    ]


def run_stage2_v4_pre_audit(
    *,
    row_diagnostics_path: str | Path,
    output_dir: str | Path,
    trajectories_path: str | Path | None = None,
    benchmarks: list[str] | tuple[str, ...] | set[str] | None = None,
    example_limit: int = 20,
    benchmark_caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(row_diagnostics_path)
    allowed = {str(item) for item in benchmarks} if benchmarks else set()
    if allowed:
        rows = [row for row in rows if str(row.get("benchmark") or "") in allowed]
    context_by_index, context_by_key = _load_contexts(trajectories_path, rows, benchmark_caps)
    summary = summarize_transition_rows(rows)
    examples = select_audit_examples(
        rows,
        example_limit=example_limit,
        context_by_index=context_by_index,
        context_by_key=context_by_key,
    )
    examples_path = output_dir / "stage2_v4_pre_audit_examples.jsonl"
    report_path = output_dir / "stage2_v4_pre_audit_report.json"
    markdown_path = output_dir / "stage2_v4_pre_audit_report.md"
    _write_jsonl(examples_path, examples)
    report = {
        "status": "ok" if rows else "action_required",
        "blockers": [] if rows else ["no_rows_after_filter"],
        "row_diagnostics_path": str(row_diagnostics_path),
        "trajectories_path": str(trajectories_path) if trajectories_path is not None else None,
        "output_dir": str(output_dir),
        "examples_path": str(examples_path),
        "config": {
            "benchmarks": sorted(allowed) if allowed else "all",
            "example_limit_per_case_per_benchmark": int(example_limit),
            "benchmark_caps": benchmark_caps or {},
        },
        "summary": summary,
        "interpretation": _interpret(summary),
        "recommended_actions": _recommended_actions(summary),
        "limitations": [
            "Existing row diagnostics do not persist Stage0 top candidate ids, only Stage0 gold rank.",
            "Probable equivalence uses skill-id token overlap only; it is a conservative heuristic, not a semantic proof.",
        ],
    }
    _write_json(report_path, report)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report
