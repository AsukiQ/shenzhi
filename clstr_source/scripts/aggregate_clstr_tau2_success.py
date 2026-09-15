#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _expected_domain_ids(manifest_path: Path) -> dict[str, list[str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["tau2_split_manifest"]["splits"]["test"]
    identities = {domain: [] for domain in ("airline", "retail", "telecom")}
    for record in records:
        domain, separator, task_id = str(record).partition("/")
        if not separator or domain not in identities or not task_id:
            raise ValueError(f"invalid Tau2 test identity: {record}")
        identities[domain].append(task_id)
    return {domain: sorted(task_ids) for domain, task_ids in identities.items()}


def _verified_raw_domain_row(
    *,
    domain: str,
    row: dict[str, Any],
    expected_task_ids: list[str],
) -> dict[str, Any]:
    results_path = Path(str(row.get("results_path") or "")).resolve()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    tasks = list(payload.get("tasks") or [])
    simulations = list(payload.get("simulations") or [])
    observed_task_ids = sorted(str(item.get("id")) for item in tasks)
    simulation_task_ids = sorted(str(item.get("task_id")) for item in simulations)
    if observed_task_ids != expected_task_ids:
        raise ValueError(f"Tau2 raw task identities differ: {domain}")
    if simulation_task_ids != expected_task_ids:
        raise ValueError(f"Tau2 raw simulation identities differ: {domain}")
    rewards = []
    for simulation in simulations:
        reward_info = simulation.get("reward_info")
        if not isinstance(reward_info, dict) or reward_info.get("reward") is None:
            raise ValueError(f"Tau2 raw simulation lacks a reward: {domain}")
        rewards.append(float(reward_info["reward"]))
    success_count = sum(reward >= 1.0 - 1e-12 for reward in rewards)
    if row.get("success_count") is not None and int(row["success_count"]) != success_count:
        raise ValueError(f"Tau2 report/raw success count differs: {domain}")
    return {
        **row,
        "task_count": len(tasks),
        "simulation_count": len(simulations),
        "reward_count": len(rewards),
        "success_count": success_count,
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate": success_count / len(rewards),
        "official_test_identity_verified": True,
        "identity_verification_source": "raw_official_results_json",
        "results_path": str(results_path),
        "results_sha256": _sha256(results_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_report", action="append", required=True)
    parser.add_argument("--matched_union_manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--verify_raw_results", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.matched_union_manifest_path).resolve()
    expected_ids = _expected_domain_ids(manifest_path)
    expected = {domain: len(task_ids) for domain, task_ids in expected_ids.items()}
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.input_report
    ]
    if any(report.get("status") != "ok" for report in reports):
        raise ValueError("all Tau2 domain reports must have status=ok")
    invariants = (
        "method",
        "executor_model",
        "user_model",
        "split_protocol",
        "matched_union_manifest_sha256",
    )
    for key in invariants:
        if len({json.dumps(report.get(key), sort_keys=True) for report in reports}) != 1:
            raise ValueError(f"Tau2 shard invariant differs: {key}")
    if reports[0]["matched_union_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Tau2 aggregate manifest digest differs")

    domains: dict[str, Any] = {}
    for report in reports:
        if report.get("evaluation_scope") != "full" and not args.verify_raw_results:
            raise ValueError("Tau2 parallel aggregation requires full domain reports")
        for domain, row in report["domains"].items():
            if domain in domains:
                raise ValueError(f"Tau2 domain appears in multiple shards: {domain}")
            if args.verify_raw_results:
                row = _verified_raw_domain_row(
                    domain=domain,
                    row=row,
                    expected_task_ids=expected_ids[domain],
                )
            if not row.get("official_test_identity_verified"):
                raise ValueError(f"Tau2 domain identity was not verified: {domain}")
            if int(row["task_count"]) != expected[domain]:
                raise ValueError(f"Tau2 domain count differs: {domain}")
            if int(row["reward_count"]) != int(row["simulation_count"]):
                raise ValueError(f"Tau2 domain contains infrastructure errors: {domain}")
            domains[domain] = row
    if set(domains) != set(expected):
        raise ValueError("Tau2 aggregate must cover airline, retail, and telecom")

    simulation_count = sum(int(row["simulation_count"]) for row in domains.values())
    success_count = sum(int(row["success_count"]) for row in domains.values())
    reward_sum = sum(
        float(row["mean_reward"]) * int(row["reward_count"])
        for row in domains.values()
    )
    if simulation_count != sum(expected.values()):
        raise ValueError("Tau2 aggregate denominator differs from official test split")
    report = {
        "status": "ok",
        "metric_scope": "official Tau2 test environment reward/task success",
        "method": reports[0]["method"],
        "executor_model": reports[0]["executor_model"],
        "user_model": reports[0]["user_model"],
        "split_protocol": reports[0]["split_protocol"],
        "evaluation_scope": "full_parallel_domains",
        "raw_result_identity_reverified": bool(args.verify_raw_results),
        "shard_count": len(reports),
        "domains": domains,
        "simulation_count": simulation_count,
        "success_count": success_count,
        "mean_reward": reward_sum / simulation_count,
        "success_rate": success_count / simulation_count,
        "matched_union_manifest_path": str(manifest_path),
        "matched_union_manifest_sha256": _sha256(manifest_path),
        "input_reports": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path).resolve()),
            }
            for path in args.input_report
        ],
        "checkpoint_binding": reports[0]["checkpoint_binding"],
        "oracle_or_private_task_fields_used_by_agent": False,
    }
    _write_json(Path(args.output_path).resolve(), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
