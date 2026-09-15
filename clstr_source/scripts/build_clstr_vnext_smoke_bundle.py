#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MATCHED_ACTION_ONLY_SOURCES = frozenset(
    {
        "matched_tau2_action_only_v1",
        "matched_toolsandbox_action_only_v1",
    }
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _skill_id(row: dict[str, Any]) -> str:
    return str(row.get("skill_id") or row.get("canonical_skill_id") or "").strip()


def _artifact(manifest: dict[str, Any], name: str) -> Path:
    entry = (manifest.get("files") or {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"data manifest lacks artifact: {name}")
    path = Path(str(entry.get("path") or ""))
    expected = str(entry.get("sha256") or "")
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ValueError(f"data manifest artifact contract failed: {name}")
    return path


def _stable_key(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_ids(row: dict[str, Any], field: str) -> list[str]:
    return sorted({str(item) for item in row.get(field) or [] if str(item)})


def _source_id(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    persisted = provenance if isinstance(provenance, dict) else {}
    return str(
        persisted.get("source_id")
        or row.get("source")
        or row.get("benchmark")
        or ""
    )


def _benchmark(row: dict[str, Any]) -> str:
    return str(row.get("benchmark") or "").strip().lower()


def _source_round_robin(
    rows: list[Any],
    *,
    source_key,
    stable_key,
    limit: int,
    initial: list[Any] | None = None,
) -> list[Any]:
    selected = list(initial or [])
    selected_ids = {id(item) for item in selected}
    by_source: dict[str, deque[Any]] = {}
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in rows:
        if id(item) not in selected_ids:
            grouped[str(source_key(item))].append(item)
    for source, items in grouped.items():
        items.sort(key=stable_key)
        by_source[source] = deque(items)
    sources = sorted(by_source)
    while len(selected) < int(limit) and sources:
        next_sources: list[str] = []
        for source in sources:
            queue = by_source[source]
            if queue and len(selected) < int(limit):
                selected.append(queue.popleft())
            if queue:
                next_sources.append(source)
        sources = next_sources
    return selected


def _required_benchmark_examples(
    rows: list[Any],
    *,
    required_benchmarks: tuple[str, ...],
    row_for_benchmark,
    stable_key,
) -> list[Any]:
    if len(required_benchmarks) > len(rows):
        raise ValueError("required benchmark coverage exceeds eligible smoke rows")
    selected: list[Any] = []
    for benchmark in required_benchmarks:
        matches = [
            item for item in rows if _benchmark(row_for_benchmark(item)) == benchmark
        ]
        if not matches:
            raise ValueError(
                f"smoke bundle lacks required benchmark coverage: {benchmark}"
            )
        selected.append(min(matches, key=stable_key))
    return selected


def _select_stage0_rows(
    rows: list[dict[str, Any]],
    *,
    positive_field: str,
    limit: int,
    required_benchmarks: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("state_text_current") or "").strip()
        and str(row.get("state_text_causal") or "").strip()
        and str(row.get("runtime_visible_catalog_id") or "").strip()
        and _positive_ids(row, positive_field)
    ]
    stable_key = lambda row: _stable_key(
        {
            "source": _source_id(row),
            "query": row.get("state_text_causal"),
            "positives": _positive_ids(row, positive_field),
        }
    )
    required = _required_benchmark_examples(
        eligible,
        required_benchmarks=required_benchmarks,
        row_for_benchmark=lambda row: row,
        stable_key=stable_key,
    )
    selected = _source_round_robin(
        eligible,
        source_key=_source_id,
        stable_key=stable_key,
        limit=limit,
        initial=required,
    )
    if not selected:
        raise ValueError(f"smoke bundle has no Stage0 rows for {positive_field}")
    return selected


def _select_ordinary_trajectories(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    minimum_length: int,
    required_benchmarks: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_id") or row.get("task_id") or "")].append(row)
    eligible: list[list[dict[str, Any]]] = []
    for trajectory_rows in grouped.values():
        trajectory_rows.sort(key=lambda row: int(row.get("step_index") or 0))
        steps = [int(row.get("step_index") or 0) for row in trajectory_rows]
        ordered = [
            bool((row.get("capabilities") or {}).get("ordered_next_tool"))
            for row in trajectory_rows
        ]
        has_result = any(
            str(row.get("actual_result_text") or "").strip()
            for row in trajectory_rows
        )
        action_only = (
            _source_id(trajectory_rows[0]) in MATCHED_ACTION_ONLY_SOURCES
            and all(
                not str(row.get("actual_result_text") or "").strip()
                and not bool(row.get("actual_result_executed"))
                and not bool(
                    (row.get("capabilities") or {}).get("actual_execution_result")
                )
                for row in trajectory_rows
            )
        )
        if (
            len(trajectory_rows) < int(minimum_length)
            or not steps
            or steps[0] != 0
            or any(current != previous + 1 for previous, current in zip(steps, steps[1:]))
            or ordered[0]
            or not all(ordered[1:])
            or not (has_result or action_only)
        ):
            continue
        eligible.append(trajectory_rows)
    stable_key = lambda trajectory_rows: _stable_key(
        {
            "source": _source_id(trajectory_rows[0]),
            "trajectory_id": trajectory_rows[0].get("trajectory_id"),
        }
    )
    required = _required_benchmark_examples(
        eligible,
        required_benchmarks=required_benchmarks,
        row_for_benchmark=lambda trajectory_rows: trajectory_rows[0],
        stable_key=stable_key,
    )
    selected = _source_round_robin(
        eligible,
        source_key=lambda trajectory_rows: _source_id(trajectory_rows[0]),
        stable_key=stable_key,
        limit=limit,
        initial=required,
    )
    if not selected:
        raise ValueError("smoke bundle has no eligible ordered trajectories")
    return [row for trajectory_rows in selected for row in trajectory_rows]


def _select_pairs_and_support(
    pair_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = sorted(
        pair_rows,
        key=lambda row: (
            str(row.get("causal_cluster_id") or ""),
            str(row.get("causal_pair_id") or ""),
        ),
    )[: int(limit)]
    if not pairs:
        raise ValueError("smoke bundle has no verified causal branch pair")
    pair_ids = {str(row.get("causal_pair_id") or "") for row in pairs}
    support = [
        row
        for row in support_rows
        if str(row.get("causal_pair_id") or "") in pair_ids
    ]
    observed = {str(row.get("causal_pair_id") or "") for row in support}
    if observed != pair_ids:
        raise ValueError("smoke causal pair support is incomplete")
    return pairs, support


def build_smoke_bundle(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.data_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise ValueError("smoke bundle requires an approved vNext data release")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"smoke output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    artifact_names = (
        "training_skills",
        "inventory_catalogs",
        "retrieval_rows",
        "retrieval_dev_rows",
        "static_route_rows",
        "static_route_dev_rows",
        "trajectory_rows",
        "trajectory_dev_rows",
        "causal_branch_pairs",
        "causal_branch_dev_pairs",
        "causal_pair_support_rows",
        "causal_pair_support_dev_rows",
    )
    source_paths = {name: _artifact(manifest, name) for name in artifact_names}
    source_rows = {name: _read_jsonl(path) for name, path in source_paths.items()}
    required_benchmarks = tuple(
        sorted(
            {
                str(item).strip().lower()
                for item in (args.required_benchmarks or [])
                if str(item).strip()
            }
        )
    )

    train_pairs, train_support = _select_pairs_and_support(
        source_rows["causal_branch_pairs"],
        source_rows["causal_pair_support_rows"],
        limit=args.pair_limit,
    )
    dev_pairs, dev_support = _select_pairs_and_support(
        source_rows["causal_branch_dev_pairs"],
        source_rows["causal_pair_support_dev_rows"],
        limit=args.pair_limit,
    )
    train_trajectories = _select_ordinary_trajectories(
        source_rows["trajectory_rows"],
        limit=args.trajectory_limit,
        minimum_length=args.minimum_trajectory_length,
        required_benchmarks=required_benchmarks,
    )
    dev_trajectories = _select_ordinary_trajectories(
        source_rows["trajectory_dev_rows"],
        limit=args.trajectory_limit,
        minimum_length=args.minimum_trajectory_length,
        required_benchmarks=required_benchmarks,
    )
    selected_rows = {
        "retrieval_rows": _select_stage0_rows(
            source_rows["retrieval_rows"],
            positive_field="required_tool_set_skill_ids",
            limit=args.stage0_rows_per_kind,
        ),
        "retrieval_dev_rows": _select_stage0_rows(
            source_rows["retrieval_dev_rows"],
            positive_field="required_tool_set_skill_ids",
            limit=args.stage0_dev_rows_per_kind,
        ),
        "static_route_rows": _select_stage0_rows(
            source_rows["static_route_rows"],
            positive_field="current_state_route_set_skill_ids",
            limit=args.stage0_rows_per_kind,
            required_benchmarks=required_benchmarks,
        ),
        "static_route_dev_rows": _select_stage0_rows(
            source_rows["static_route_dev_rows"],
            positive_field="current_state_route_set_skill_ids",
            limit=args.stage0_dev_rows_per_kind,
            required_benchmarks=required_benchmarks,
        ),
        "trajectory_rows": train_trajectories,
        "trajectory_dev_rows": dev_trajectories,
        "causal_branch_pairs": train_pairs,
        "causal_branch_dev_pairs": dev_pairs,
        "causal_pair_support_rows": train_support,
        "causal_pair_support_dev_rows": dev_support,
    }

    catalogs = {
        str(row.get("inventory_catalog_id") or ""): row
        for row in source_rows["inventory_catalogs"]
    }
    required_skill_ids: set[str] = set()
    referenced_catalog_ids: set[str] = set()
    for name, rows in selected_rows.items():
        if name.startswith("causal_branch"):
            continue
        for row in rows:
            referenced_catalog_ids.add(str(row.get("runtime_visible_catalog_id") or ""))
            required_skill_ids.update(
                str(item)
                for item in (
                    row.get("required_tool_set_skill_ids")
                    or row.get("current_state_route_set_skill_ids")
                    or []
                )
                if str(item)
            )
            for key in ("target_skill_id", "skill_id"):
                if str(row.get(key) or ""):
                    required_skill_ids.add(str(row[key]))
            required_skill_ids.update(
                str(item) for item in row.get("equivalent_next_skill_ids") or [] if str(item)
            )
    for catalog_id in sorted(referenced_catalog_ids):
        catalog = catalogs.get(catalog_id)
        if catalog is None:
            raise ValueError(f"smoke row references a missing catalog: {catalog_id}")
        required_skill_ids.update(
            str(item)
            for item in (catalog.get("runtime_visible_skill_ids") or [])[: args.negatives_per_catalog]
            if str(item)
        )

    skills = source_rows["training_skills"]
    known_ids = {_skill_id(row) for row in skills}
    missing = sorted(required_skill_ids - known_ids)
    if missing:
        raise ValueError(f"smoke rows reference unknown training skills: {missing[:4]}")
    if len(required_skill_ids) > int(args.maximum_skills):
        raise ValueError(
            f"required smoke skills exceed maximum: {len(required_skill_ids)} > {args.maximum_skills}"
        )
    selected_skill_ids = set(required_skill_ids)
    for row in skills:
        if len(selected_skill_ids) >= int(args.maximum_skills):
            break
        selected_skill_ids.add(_skill_id(row))
    selected_skills = [row for row in skills if _skill_id(row) in selected_skill_ids]
    if len(selected_skills) != len(selected_skill_ids):
        raise ValueError("smoke skill selection is not one-to-one")

    output_rows = {
        "training_skills": selected_skills,
        "inventory_catalogs": [catalogs[key] for key in sorted(referenced_catalog_ids)],
        **selected_rows,
    }
    counts: dict[str, int] = {}
    files: dict[str, dict[str, Any]] = {}
    for name, rows in output_rows.items():
        path = output_dir / f"{name}.jsonl"
        counts[name] = _write_jsonl(path, rows)
        files[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }

    report = {
        "status": "ok",
        "schema_version": "clstr_vnext_smoke_bundle_v1",
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "counts": counts,
        "files": files,
        "settings": {
            "maximum_skills": int(args.maximum_skills),
            "stage0_rows_per_kind": int(args.stage0_rows_per_kind),
            "stage0_dev_rows_per_kind": int(args.stage0_dev_rows_per_kind),
            "trajectory_limit": int(args.trajectory_limit),
            "minimum_trajectory_length": int(args.minimum_trajectory_length),
            "pair_limit": int(args.pair_limit),
            "negatives_per_catalog": int(args.negatives_per_catalog),
            "required_benchmarks": list(required_benchmarks),
        },
        "selected_benchmark_counts": {
            name: dict(sorted(Counter(_benchmark(row) for row in rows).items()))
            for name, rows in selected_rows.items()
            if name
            in {
                "static_route_rows",
                "static_route_dev_rows",
                "trajectory_rows",
                "trajectory_dev_rows",
            }
        },
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic CLSTR vNext smoke bundle")
    parser.add_argument("--data_manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--maximum_skills", type=int, default=768)
    parser.add_argument("--stage0_rows_per_kind", type=int, default=32)
    parser.add_argument("--stage0_dev_rows_per_kind", type=int, default=32)
    parser.add_argument("--trajectory_limit", type=int, default=8)
    parser.add_argument("--minimum_trajectory_length", type=int, default=3)
    parser.add_argument("--pair_limit", type=int, default=4)
    parser.add_argument("--negatives_per_catalog", type=int, default=8)
    parser.add_argument("--required_benchmarks", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    report = build_smoke_bundle(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
