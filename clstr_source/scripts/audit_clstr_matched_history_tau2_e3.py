#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clstr.matched_history_online import (  # noqa: E402
    E3_FOUNDATION_SHA256,
    E3_SELECTED_CHECKPOINTS,
    E3_SKILLS_SHA256,
)


METHODS = ("static", "transformer", "lstr")
EXPECTED_DOMAIN_COUNTS = {"airline": 20, "retail": 40, "telecom": 40}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_binary_statistics(
    reference: list[bool],
    candidate: list[bool],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired outcomes must have equal nonzero length")
    differences = [float(right) - float(left) for left, right in zip(reference, candidate)]
    rng = random.Random(int(seed))
    draws: list[float] = []
    count = len(differences)
    for _ in range(int(bootstrap_draws)):
        draws.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    candidate_only = sum((not left) and right for left, right in zip(reference, candidate))
    reference_only = sum(left and (not right) for left, right in zip(reference, candidate))
    discordant = candidate_only + reference_only
    if discordant:
        tail = min(candidate_only, reference_only)
        probability = 2.0 * sum(
            math.comb(discordant, value) for value in range(tail + 1)
        ) / (2.0**discordant)
        mcnemar_p = min(1.0, probability)
    else:
        mcnemar_p = 1.0
    return {
        "task_count": count,
        "reference_success_count": sum(reference),
        "candidate_success_count": sum(candidate),
        "mean_success_difference": sum(differences) / count,
        "paired_task_bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "ci_low": _quantile(draws, 0.025),
            "ci_high": _quantile(draws, 0.975),
        },
        "mcnemar_exact": {
            "candidate_only_success": candidate_only,
            "reference_only_success": reference_only,
            "discordant_count": discordant,
            "two_sided_p": mcnemar_p,
        },
    }


def _audit_selection_log(path: Path, method: str) -> dict[str, Any]:
    rows = _read_jsonl(path)
    selections = [row for row in rows if row.get("record_type") == "selection"]
    updates = [row for row in rows if row.get("record_type") == "memory_update"]
    if not selections:
        raise ValueError(f"{method} selection log is empty")
    history_counts: Counter[str] = Counter()
    candidate_counts: Counter[int] = Counter()
    for row in selections:
        if str(row.get("method")) != f"matched_history_e3_{method}":
            raise ValueError(f"{method} selection log method differs")
        candidates = [str(value) for value in row.get("candidate_skill_ids") or []]
        selected = list(row.get("selected") or [])
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"{method} selection has an invalid candidate set")
        if len(selected) != min(8, len(candidates)):
            raise ValueError(f"{method} selection is not Top-8")
        candidate_counts[len(candidates)] += 1
        support_hashes = {str(item.get("static_support_sha256") or "") for item in selected}
        support_counts = {int(item.get("static_support_count") or 0) for item in selected}
        if len(support_hashes) != 1 or "" in support_hashes:
            raise ValueError(f"{method} selected items disagree on Static support hash")
        if support_counts != {len(candidates)}:
            raise ValueError(f"{method} Static support is not the full legal Tau2 tool set")
        selected_ids = [str(item.get("skill_id") or "") for item in selected]
        if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(candidates):
            raise ValueError(f"{method} selected tools are invalid")
        depth = int(row.get("history_depth") or 0)
        expected_expert = "static" if method == "static" or depth == 0 else "dynamic"
        for item in selected:
            if str(item.get("encoder_kind")) != method:
                raise ValueError(f"{method} logged encoder kind differs")
            if str(item.get("selected_expert")) != expected_expert:
                raise ValueError(f"{method} selected expert violates the E3 contract")
            if int(item.get("history_window_depth") or 0) != min(depth, 16):
                raise ValueError(f"{method} history horizon differs")
        history_counts["zero" if depth == 0 else "history"] += 1
    if method != "static" and history_counts["history"] <= 0:
        raise ValueError(f"{method} full run never exercised a history-bearing decision")
    for row in updates:
        if not str(row.get("event_action_text") or "").startswith("tool: "):
            raise ValueError(f"{method} update action is not E1-canonical")
        if not bool(row.get("prefix_reconstructed_at_next_selection")):
            raise ValueError(f"{method} update did not declare prefix reconstruction")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "row_count": len(rows),
        "selection_count": len(selections),
        "memory_update_count": len(updates),
        "zero_history_selection_count": history_counts["zero"],
        "history_selection_count": history_counts["history"],
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "all_static_supports_equal_full_legal_set": True,
    }


def _audit_one_report(path: Path, expected_method: str) -> tuple[dict[str, Any], dict[str, bool]]:
    report = _read_json(path)
    if report.get("schema_version") != "clstr_matched_history_tau2_e3_result_v1":
        raise ValueError(f"{expected_method} report has the wrong schema")
    if report.get("status") != "ok" or report.get("evaluation_scope") != "full":
        raise ValueError(f"{expected_method} report is not a successful full run")
    if str(report.get("controlled_method")) != expected_method:
        raise ValueError(f"{expected_method} report method differs")
    if any(
        int(report.get(key, -1)) != value
        for key, value in (
            ("simulation_count", 100),
            ("reward_count", 100),
            ("missing_reward_count", 0),
            ("infra_error_count", 0),
            ("top_k", 8),
            ("max_steps", 100),
            ("num_trials", 1),
        )
    ):
        raise ValueError(f"{expected_method} report denominator/protocol differs")
    if abs(float(report.get("executor_temperature") or 0.0)) > 1.0e-12:
        raise ValueError(f"{expected_method} executor temperature differs")
    domains = report.get("domains") or {}
    if {
        domain: int((domains.get(domain) or {}).get("reward_count") or 0)
        for domain in EXPECTED_DOMAIN_COUNTS
    } != EXPECTED_DOMAIN_COUNTS:
        raise ValueError(f"{expected_method} domain denominators differ")
    if not all(
        bool((domains.get(domain) or {}).get("official_test_identity_verified"))
        for domain in EXPECTED_DOMAIN_COUNTS
    ):
        raise ValueError(f"{expected_method} official identities were not verified")
    binding = report.get("checkpoint_binding") or {}
    if binding.get("foundation_checkpoint_sha256") != E3_FOUNDATION_SHA256:
        raise ValueError(f"{expected_method} foundation hash differs")
    if binding.get("skills_sha256") != E3_SKILLS_SHA256:
        raise ValueError(f"{expected_method} skill-table hash differs")
    if expected_method == "static":
        if binding.get("e1_checkpoint_sha256") is not None:
            raise ValueError("Static report unexpectedly loaded a learned E1 checkpoint")
    elif binding.get("e1_checkpoint_sha256") != E3_SELECTED_CHECKPOINTS[expected_method][
        "sha256"
    ]:
        raise ValueError(f"{expected_method} E1 checkpoint hash differs")
    outcomes_path = Path(str(report.get("task_outcomes_path") or "")).resolve()
    selection_path = Path(str(report.get("selection_log_path") or "")).resolve()
    if not outcomes_path.is_file() or not selection_path.is_file():
        raise ValueError(f"{expected_method} report is missing raw artifacts")
    if report.get("task_outcomes_sha256") != _sha256(outcomes_path):
        raise ValueError(f"{expected_method} task-outcome hash differs")
    outcomes = _read_jsonl(outcomes_path)
    if len(outcomes) != 100:
        raise ValueError(f"{expected_method} task-outcome count differs")
    by_key: dict[str, bool] = {}
    counts: Counter[str] = Counter()
    for row in outcomes:
        domain = str(row.get("domain") or "")
        task_id = str(row.get("task_id") or "")
        key = f"{domain}/{task_id}"
        if domain not in EXPECTED_DOMAIN_COUNTS or not task_id or key in by_key:
            raise ValueError(f"{expected_method} task outcomes have invalid identities")
        if row.get("reward") is None or bool(row.get("infra_error")):
            raise ValueError(f"{expected_method} task outcome is missing/infra-failed")
        by_key[key] = bool(row.get("success"))
        counts[domain] += 1
    if dict(counts) != EXPECTED_DOMAIN_COUNTS:
        raise ValueError(f"{expected_method} raw domain counts differ")
    if sum(by_key.values()) != int(report.get("success_count") or 0):
        raise ValueError(f"{expected_method} raw success count differs")
    selection_audit = _audit_selection_log(selection_path, expected_method)
    summary = {
        "report_path": str(path),
        "report_sha256": _sha256(path),
        "success_count": sum(by_key.values()),
        "success_rate": sum(by_key.values()) / 100.0,
        "outcomes_path": str(outcomes_path),
        "outcomes_sha256": _sha256(outcomes_path),
        "selection_log": selection_audit,
    }
    return {"report": report, "summary": summary}, by_key


def audit_e3(
    report_paths: dict[str, Path],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    audited: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, bool]] = {}
    for method in METHODS:
        audited[method], outcomes[method] = _audit_one_report(
            report_paths[method].resolve(), method
        )
    reference_keys = sorted(outcomes["static"])
    if any(sorted(outcomes[method]) != reference_keys for method in METHODS[1:]):
        raise ValueError("E3 task identities differ across arms")
    shared_fields = (
        "executor_model",
        "executor_temperature",
        "user_model",
        "matched_union_manifest_sha256",
        "top_k",
        "max_steps",
        "num_trials",
        "seed",
    )
    static_report = audited["static"]["report"]
    for method in METHODS[1:]:
        candidate_report = audited[method]["report"]
        for field in shared_fields:
            if candidate_report.get(field) != static_report.get(field):
                raise ValueError(f"E3 arms differ on shared field {field}")

    pairwise: dict[str, Any] = {}
    comparisons = (
        ("transformer_minus_static", "static", "transformer"),
        ("lstr_minus_static", "static", "lstr"),
        ("lstr_minus_transformer", "transformer", "lstr"),
    )
    for offset, (name, reference, candidate) in enumerate(comparisons):
        pairwise[name] = paired_binary_statistics(
            [outcomes[reference][key] for key in reference_keys],
            [outcomes[candidate][key] for key in reference_keys],
            bootstrap_draws=bootstrap_draws,
            seed=int(seed) + offset,
        )
    return {
        "schema_version": "clstr_matched_history_tau2_e3_audit_v1",
        "status": "ok",
        "methods": {method: audited[method]["summary"] for method in METHODS},
        "task_identity_count": len(reference_keys),
        "shared_protocol_fields": {
            field: static_report.get(field) for field in shared_fields
        },
        "pairwise": pairwise,
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "checks": {
            "official_20_40_40_identity": True,
            "reward_count_100_each": True,
            "missing_reward_count_zero_each": True,
            "infra_error_count_zero_each": True,
            "same_foundation_and_skill_table": True,
            "same_executor_user_budget": True,
            "static_support_is_full_legal_set_every_decision": True,
            "selector_and_candidate_extras_disabled": True,
            "paired_outcomes_complete": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly audit the official Tau2 E3 matrix.")
    parser.add_argument("--static_report_path", required=True)
    parser.add_argument("--transformer_report_path", required=True)
    parser.add_argument("--lstr_report_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_e3(
        {
            "static": Path(args.static_report_path),
            "transformer": Path(args.transformer_report_path),
            "lstr": Path(args.lstr_report_path),
        },
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    output = Path(args.output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
