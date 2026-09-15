from __future__ import annotations

import json
from pathlib import Path

from clstr.memory_utility_records import canonical_digest
from clstr.qwen_clstr_multibench_submit import QWEN_CLSTR_MULTIBENCH_STAGE_ORDER


CHAIN_DIGEST = "1" * 64
CHAIN_MANIFEST_SHA = "2" * 64


def _write_registry(path: Path) -> dict:
    stage_order = list(QWEN_CLSTR_MULTIBENCH_STAGE_ORDER)
    payload = {
        "schema_version": 1,
        "status": "submitted",
        "input_fingerprint": "3" * 64,
        "plan_fingerprint": "4" * 64,
        "submission_identity": {
            "stage_order": stage_order,
            "checkpoint_chain_digest": CHAIN_DIGEST,
            "final_chain_manifest_sha256": CHAIN_MANIFEST_SHA,
            "scope": "full",
            "corpus_manifest_sha256_by_stage": {
                key: f"{index + 5:064x}" for index, key in enumerate(stage_order)
            },
        },
        "stage_order": stage_order,
        "stages": {
            key: {
                "status": "submitted",
                "job_id": str(1201 + index),
            }
            for index, key in enumerate(stage_order)
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def test_job_evidence_collects_exact_completed_slurm_allocations_and_self_hashes(tmp_path):
    from clstr.qwen_clstr_multibench_jobs import (
        collect_qwen_clstr_multibench_job_evidence,
    )

    registry_path = tmp_path / "full-submission.json"
    registry = _write_registry(registry_path)
    output_path = tmp_path / "full-job-evidence.json"
    job_ids = [registry["stages"][key]["job_id"] for key in registry["stage_order"]]
    captured = {}

    def fake_sacct(command):
        captured["command"] = command
        return "".join(f"{job_id}|COMPLETED|0:0\n" for job_id in job_ids)

    evidence = collect_qwen_clstr_multibench_job_evidence(
        registry_path=registry_path,
        output_path=output_path,
        sacct_runner=fake_sacct,
    )

    assert evidence["status"] == "ok"
    assert evidence["blockers"] == []
    assert captured["command"] == [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode",
    ]
    assert evidence["jobs"]["frozen/toolbench_g3"] == {
        "job_id": "1201",
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    payload = dict(evidence)
    recorded = payload.pop("evidence_sha256")
    assert canonical_digest(payload) == recorded
    assert json.loads(output_path.read_text(encoding="utf-8")) == evidence


def test_job_evidence_records_failed_and_missing_jobs_as_distinct_blockers(tmp_path):
    from clstr.qwen_clstr_multibench_jobs import (
        collect_qwen_clstr_multibench_job_evidence,
    )

    registry_path = tmp_path / "full-submission.json"
    registry = _write_registry(registry_path)
    output_path = tmp_path / "failed-job-evidence.json"
    stage_order = registry["stage_order"]
    job_ids = [registry["stages"][key]["job_id"] for key in stage_order]

    def fake_sacct(_command):
        rows = []
        for index, job_id in enumerate(job_ids[:-1]):
            if stage_order[index] == "native/tau2":
                rows.append(f"{job_id}|FAILED|1:0\n")
            else:
                rows.append(f"{job_id}|COMPLETED|0:0\n")
        return "".join(rows)

    evidence = collect_qwen_clstr_multibench_job_evidence(
        registry_path=registry_path,
        output_path=output_path,
        sacct_runner=fake_sacct,
    )

    assert evidence["status"] == "action_required"
    assert "native/tau2:job_not_completed:FAILED:1:0" in evidence["blockers"]
    assert (
        "closed_loop/alfworld_valid_unseen:missing_job_state"
        in evidence["blockers"]
    )


def test_job_evidence_cli_forwards_registry_output_and_sacct_runner(monkeypatch, tmp_path):
    import scripts.collect_qwen06_clstr_multibench_job_states as cli

    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "job-evidence.json"
    captured = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "evidence_sha256": "9" * 64}

    monkeypatch.setattr(cli, "collect_qwen_clstr_multibench_job_evidence", fake_collect)
    args = cli.build_parser().parse_args(
        [
            "--registry_path",
            str(registry_path),
            "--output_path",
            str(output_path),
        ]
    )

    evidence = cli.run_from_args(args)

    assert evidence == {"status": "ok", "evidence_sha256": "9" * 64}
    assert captured["registry_path"] == str(registry_path)
    assert captured["output_path"] == str(output_path)
    assert captured["sacct_runner"] is cli.run_sacct
