import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from clstr.memory_utility_records import canonical_digest, checkpoint_chain_digest
from clstr.qwen_clstr_final_chain import FINAL_CHAIN_SCHEMA_VERSION
from clstr.qwen_clstr_lineage import sha256_path


EXPECTED_STAGE_ORDER = [
    "frozen/toolbench_g3",
    "frozen/toolsandbox",
    "frozen/tau2",
    "frozen/alfworld_offline",
    "native/toolbench_g3",
    "native/toolsandbox",
    "native/tau2",
    "closed_loop/alfworld_valid_seen",
    "closed_loop/alfworld_valid_unseen",
]


def test_alfworld_launcher_pins_action_skills_through_manifest_content():
    text = Path("scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh").read_text(
        encoding="utf-8"
    )

    assert 'action_manifest.get("canonical_skills")' in text
    assert "skills_file_sha256" not in text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_final_chain(tmp_path: Path) -> Path:
    checkpoints = {}
    for role in ("stage0", "stage1", "stage2", "stage4"):
        path = tmp_path / "checkpoints" / f"{role}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}-checkpoint".encode("utf-8"))
        checkpoints[role] = path
    backbone = tmp_path / "models" / "Qwen3-Embedding-0.6B"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text("{}\n", encoding="utf-8")
    skills = tmp_path / "training" / "skill_pool.jsonl"
    skills.parent.mkdir(parents=True)
    skills.write_text('{"skill_id":"training/a"}\n', encoding="utf-8")
    stage4_selection = tmp_path / "stage4_selection.json"
    stage4_selection.write_text('{"status":"ok"}\n', encoding="utf-8")
    reliability = {
        "mode": "causal_gate",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "safe_memory_residual_bound": 2.0,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    payload = {
        "schema_version": FINAL_CHAIN_SCHEMA_VERSION,
        "status": "ok",
        "final_checkpoint_role": "stage4",
        "checkpoint_chain_digest": checkpoint_chain_digest(checkpoints),
        "checkpoints": {
            role: {**sha256_path(path), "stage": role}
            for role, path in checkpoints.items()
        },
        "backbone": {**sha256_path(backbone), "frozen": True},
        "skill_pool": sha256_path(skills),
        "stage4_selection": sha256_path(stage4_selection),
        "route_scorer": "unified_memory",
        "state_query": {"prompt_version": "clstr_causal_state_v1"},
        "reliability": reliability,
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    path = tmp_path / "final_chain.json"
    _write_json(path, payload)
    return path


def _write_corpus(root: Path, *, benchmark: str, split: str = "eval") -> dict[str, Path | str]:
    source_rows = root / "source_rows.jsonl"
    skills = root / "skills.jsonl"
    source_rows.parent.mkdir(parents=True, exist_ok=True)
    source_rows.write_text('{"row_id":"r0"}\n', encoding="utf-8")
    skills.write_text('{"skill_id":"s0"}\n', encoding="utf-8")
    payload = {
        "schema_version": 1,
        "benchmark": benchmark,
        "split": split,
        "source_row_count": 1,
        "skill_count": 1,
    }
    payload["manifest_sha256"] = canonical_digest(payload)
    manifest = root / "manifest.json"
    _write_json(manifest, payload)
    return {
        "manifest_path": manifest,
        "source_rows_path": source_rows,
        "skills_path": skills,
        "manifest_sha256": payload["manifest_sha256"],
    }


def _evaluation_config(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    project_root = tmp_path / "repo"
    (project_root / "scripts" / "sbatch").mkdir(parents=True)
    for filename in (
        "run_qwen06_clstr_frozen_route_eval.sh",
        "run_qwen06_clstr_native_route_eval.sh",
        "run_qwen06_clstr_alfworld_eval.sh",
        "run_alfworld_qwen_clstr_executor_gate.sh",
    ):
        (project_root / "scripts" / "sbatch" / filename).write_text(
            "#!/bin/bash\n",
            encoding="utf-8",
        )
    output_root = tmp_path / "evaluation"
    final_chain = _write_final_chain(tmp_path)
    offline = {
        "toolbench_g3": _write_corpus(tmp_path / "corpora" / "toolbench_g3", benchmark="toolbench_g3"),
        "toolsandbox": _write_corpus(tmp_path / "corpora" / "toolsandbox", benchmark="toolsandbox"),
        "tau2": _write_corpus(tmp_path / "corpora" / "tau2", benchmark="tau2"),
        "alfworld_offline": _write_corpus(
            tmp_path / "corpora" / "alfworld_offline",
            benchmark="alfworld",
            split="valid_seen+valid_unseen",
        ),
    }
    seen = _write_corpus(
        tmp_path / "corpora" / "alfworld_valid_seen",
        benchmark="alfworld",
        split="valid_seen",
    )
    unseen = _write_corpus(
        tmp_path / "corpora" / "alfworld_valid_unseen",
        benchmark="alfworld",
        split="valid_unseen",
    )
    native_root = tmp_path / "native"
    native_root.mkdir()
    toolbench_train = native_root / "toolbench_train.jsonl"
    toolbench_eval = native_root / "toolbench_eval.jsonl"
    toolbench_train.write_text("{}\n", encoding="utf-8")
    toolbench_eval.write_text("{}\n", encoding="utf-8")
    tau2_root = native_root / "tau2"
    scenarios_root = native_root / "toolsandbox_scenarios"
    tools_root = native_root / "toolsandbox_tools"
    official_repo = native_root / "alfworld_repo"
    alfworld_data = native_root / "alfworld_data"
    qwen_executor = native_root / "Qwen3-14B"
    for directory in (
        tau2_root,
        scenarios_root,
        tools_root,
        official_repo,
        alfworld_data,
        qwen_executor,
    ):
        directory.mkdir()
    config = {
        "project_root": project_root,
        "output_root": output_root,
        "final_chain_manifest_path": final_chain,
        "offline_corpora": {
            benchmark: {
                "manifest_path": spec["manifest_path"],
                "source_rows_path": spec["source_rows_path"],
                "skills_path": spec["skills_path"],
            }
            for benchmark, spec in offline.items()
        },
        "alfworld_manifest_by_split": {
            "valid_seen": seen["manifest_path"],
            "valid_unseen": unseen["manifest_path"],
        },
        "native_inputs": {
            "toolbench_g3": {
                "train_trajectories_path": toolbench_train,
                "eval_trajectories_path": toolbench_eval,
            },
            "tau2": {"data_root": tau2_root},
            "toolsandbox": {
                "scenarios_root": scenarios_root,
                "tools_root": tools_root,
            },
            "alfworld": {
                "official_repo": official_repo,
                "data_dir": alfworld_data,
                "qwen_model_name_or_path": qwen_executor,
            },
        },
    }
    manifest_shas = {
        **{key: str(value["manifest_sha256"]) for key, value in offline.items()},
        "alfworld_valid_seen": str(seen["manifest_sha256"]),
        "alfworld_valid_unseen": str(unseen["manifest_sha256"]),
    }
    return config, manifest_shas


def _write_accepted_smoke_gate(
    tmp_path: Path,
    config: dict,
    manifest_shas: dict[str, str],
) -> Path:
    final_chain = json.loads(
        Path(config["final_chain_manifest_path"]).read_text(encoding="utf-8")
    )
    corpus_by_stage = {
        "frozen/toolbench_g3": manifest_shas["toolbench_g3"],
        "frozen/toolsandbox": manifest_shas["toolsandbox"],
        "frozen/tau2": manifest_shas["tau2"],
        "frozen/alfworld_offline": manifest_shas["alfworld_offline"],
        "native/toolbench_g3": manifest_shas["toolbench_g3"],
        "native/toolsandbox": manifest_shas["toolsandbox"],
        "native/tau2": manifest_shas["tau2"],
        "closed_loop/alfworld_valid_seen": manifest_shas["alfworld_valid_seen"],
        "closed_loop/alfworld_valid_unseen": manifest_shas["alfworld_valid_unseen"],
    }
    payload = {
        "schema_version": "qwen06_clstr_multibench_report_v1",
        "status": "ok",
        "blockers": [],
        "scope": "smoke",
        "stage_order": EXPECTED_STAGE_ORDER,
        "checkpoint_chain_digest": final_chain["checkpoint_chain_digest"],
        "final_chain_manifest_sha256": final_chain["manifest_sha256"],
        "corpus_manifest_sha256_by_stage": corpus_by_stage,
        "execution": {
            "status": "ok",
            "blockers": [],
            "jobs": {
                stage: {
                    "job_id": str(1401 + index),
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                }
                for index, stage in enumerate(EXPECTED_STAGE_ORDER)
            },
        },
        "routing": {"frozen": {}, "native": {}},
        "task_success": {},
    }
    payload["report_sha256"] = canonical_digest(payload)
    path = tmp_path / "accepted-smoke-gate.json"
    _write_json(path, payload)
    return path


def test_build_smoke_stages_has_deterministic_order_and_pins_chain_and_corpora(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, manifest_shas = _evaluation_config(tmp_path)
    stages = build_qwen_clstr_multibench_stages(
        config,
        scope="smoke",
        partition="gpu_a800",
    )

    assert [stage.key for stage in stages] == EXPECTED_STAGE_ORDER
    final_chain = json.loads(Path(config["final_chain_manifest_path"]).read_text(encoding="utf-8"))
    for stage in stages:
        assert stage.exports["PROJECT_ROOT"] == str(Path(config["project_root"]))
        assert stage.exports["EVAL_SCOPE"] == "smoke"
        assert stage.exports["FINAL_CHAIN_MANIFEST_SHA256"] == final_chain["manifest_sha256"]
        assert stage.exports["CHECKPOINT_CHAIN_DIGEST"] == final_chain["checkpoint_chain_digest"]
        assert len(stage.exports["CORPUS_MANIFEST_SHA256"]) == 64
        assert stage.partition == "gpu_a800"
        assert stage.exports["RELIABILITY_MODE"] == "causal_gate"
        assert stage.exports["FIXED_ALPHA"] == "1.0"
        assert stage.exports["SAFE_MEMORY_RESIDUAL_BOUND"] == "2.0"
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_PATH"] == ""
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_SHA256"] == ""
        assert stage.exports["MEMORY_UTILITY_GATE_AUDIT_SHA256"] == ""
    assert stages[0].exports["CORPUS_MANIFEST_SHA256"] == manifest_shas["toolbench_g3"]
    assert stages[3].exports["CORPUS_MANIFEST_SHA256"] == manifest_shas["alfworld_offline"]
    assert stages[7].exports["CORPUS_MANIFEST_SHA256"] == manifest_shas["alfworld_valid_seen"]
    assert stages[8].exports["CORPUS_MANIFEST_SHA256"] == manifest_shas["alfworld_valid_unseen"]
    assert stages[7].exports["ALFWORLD_SKILLS_PATH"] == str(
        Path(config["offline_corpora"]["alfworld_offline"]["skills_path"])
    )
    assert stages[8].exports["ALFWORLD_SKILLS_PATH"] == str(
        Path(config["offline_corpora"]["alfworld_offline"]["skills_path"])
    )
    assert stages[7].exports["ALFWORLD_SKILLS_MANIFEST_SHA256"] == manifest_shas[
        "alfworld_offline"
    ]
    assert stages[8].exports["ALFWORLD_SKILLS_MANIFEST_SHA256"] == manifest_shas[
        "alfworld_offline"
    ]


def test_build_cmc_stages_maps_manifest_identity_to_runtime_mode_and_feature_caps(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, _manifest_shas = _evaluation_config(tmp_path)
    final_chain_path = Path(config["final_chain_manifest_path"])
    final_chain = json.loads(final_chain_path.read_text(encoding="utf-8"))
    gate_checkpoint = tmp_path / "memory_utility_gate.pt"
    gate_checkpoint.write_bytes(b"gate")
    gate_report = tmp_path / "memory_utility_gate_report.json"
    gate_report.write_text("{}\n", encoding="utf-8")
    reliability = {
        "mode": "cmc_candidate_gate",
        "fixed_alpha": None,
        "gate_checkpoint": sha256_path(gate_checkpoint),
        "gate_report": sha256_path(gate_report),
        "audit_manifest_sha256": "a" * 64,
        "selected_stage4_checkpoint_sha256": final_chain["checkpoints"]["stage4"][
            "sha256"
        ],
        "feature_update_count_cap": 4.0,
        "feature_candidate_count_cap": 64.0,
        "base_gate_source": "stage4_checkpoint",
        "deployed_gate_source": "direct_utility_overlay",
        "gate_output_semantics": "anchored_harm_suppression_alpha",
        "alpha_base": 0.85,
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    final_chain["reliability"] = reliability
    final_chain.pop("manifest_sha256")
    final_chain["manifest_sha256"] = canonical_digest(final_chain)
    _write_json(final_chain_path, final_chain)

    stages = build_qwen_clstr_multibench_stages(config, scope="smoke")

    assert stages
    for stage in stages:
        assert stage.exports["RELIABILITY_MODE"] == "cmc"
        assert stage.exports["FEATURE_UPDATE_COUNT_CAP"] == "4.0"
        assert stage.exports["FEATURE_CANDIDATE_COUNT_CAP"] == "64.0"
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_PATH"] == str(
            gate_checkpoint.resolve()
        )
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_SHA256"] == (
            sha256_path(gate_checkpoint)["sha256"]
        )
        assert stage.exports["MEMORY_UTILITY_GATE_AUDIT_SHA256"] == "a" * 64
    assert all(stage.exports.get("MAX_EVAL_ROWS") == "32" for stage in stages[:4])
    assert stages[7].exports["MAX_EPISODES"] == "5"
    assert stages[8].exports["MAX_EPISODES"] == "5"
    assert stages[7].exports["MEMORY_PROTOCOL"] == "stateful_post_action_v1"
    assert stages[8].exports["MEMORY_PROTOCOL"] == "stateful_post_action_v1"
    assert "REPLAY_PREFIX_MAX_STEPS" not in stages[7].exports
    assert "REPLAY_PREFIX_MAX_STEPS" not in stages[8].exports
    by_key = {stage.key: stage for stage in stages}
    assert by_key["frozen/tau2"].exports["BATCH_SIZE"] == "128"
    assert by_key["native/tau2"].exports["BATCH_SIZE"] == "64"
    assert by_key["native/tau2"].exports["STAGE0_CANDIDATE_BATCH_SIZE"] == "128"
    assert by_key["closed_loop/alfworld_valid_seen"].exports["BATCH_SIZE"] == "4"
    assert by_key["closed_loop/alfworld_valid_unseen"].exports["BATCH_SIZE"] == "4"
    assert by_key["frozen/toolbench_g3"].exports["BATCH_SIZE"] == "16"
    assert by_key["frozen/toolsandbox"].exports["BATCH_SIZE"] == "16"
    assert by_key["native/toolbench_g3"].exports["BATCH_SIZE"] == "8"
    assert by_key["native/toolsandbox"].exports["BATCH_SIZE"] == "8"
    assert by_key["native/toolbench_g3"].exports["STAGE0_CANDIDATE_BATCH_SIZE"] == "16"
    assert by_key["native/toolsandbox"].exports["STAGE0_CANDIDATE_BATCH_SIZE"] == "16"
    assert by_key["native/toolbench_g3"].exports["STATIC_K"] == "500"
    assert by_key["native/toolbench_g3"].exports["DYNAMIC_EXTRA_K"] == "64"
    assert by_key["native/toolbench_g3"].exports["CANDIDATE_COUNT"] == "100"
    assert by_key["native/toolsandbox"].exports["CANDIDATE_COUNT"] == "64"
    assert by_key["native/tau2"].exports["CANDIDATE_COUNT"] == "64"
    assert by_key["native/toolsandbox"].exports["MODEL_SKILL_POOL_MODE"] == (
        "checkpoint_faithful"
    )
    assert by_key["native/toolsandbox"].exports["TOOLSANDBOX_REPLAY_MODE"] == (
        "corrected_causal"
    )
    assert by_key["native/tau2"].exports["TASK_SPLIT"] == "base"
    for key in (
        "closed_loop/alfworld_valid_seen",
        "closed_loop/alfworld_valid_unseen",
    ):
        stage = by_key[key]
        assert stage.launcher.name == "run_alfworld_qwen_clstr_executor_gate.sh"
        assert stage.exports["QWEN_MODEL_NAME_OR_PATH"].endswith("Qwen3-14B")
        assert stage.exports["QWEN_WEIGHT"] == "1.0"
        assert stage.exports["CLSTR_WEIGHT"] == "0.25"
        assert stage.exports["CLSTR_PRIOR_MODE"] == (
            "unified_memory_concrete_action"
        )
        assert stage.exports["SCORING_METHOD"] == "generate"
        assert stage.exports["LOOP_GUARD"] == "1"
        assert stage.exports["RUN_QWEN_ONLY"] == "1"
        assert stage.exports["RUN_HYBRID"] == "1"
        assert stage.exports["RELIABILITY_MODE"] == "cmc"
        assert stage.exports["FEATURE_UPDATE_COUNT_CAP"] == "4.0"
        assert stage.exports["FEATURE_CANDIDATE_COUNT_CAP"] == "64.0"


def test_build_candidate_provenance_stages_preserves_runtime_mode_and_has_no_gate(
    tmp_path,
):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, _manifest_shas = _evaluation_config(tmp_path)
    final_chain_path = Path(config["final_chain_manifest_path"])
    final_chain = json.loads(final_chain_path.read_text(encoding="utf-8"))
    reliability = {
        "mode": "cmc_candidate_provenance",
        "fixed_alpha": None,
        "gate_checkpoint": None,
        "feature_update_count_cap": 16.0,
        "feature_candidate_count_cap": 256.0,
        "safe_memory_residual_bound": 2.0,
        "selected_stage4_checkpoint_sha256": final_chain["checkpoints"]["stage4"][
            "sha256"
        ],
        "base_reliability_source": "stage4_checkpoint",
        "deployed_reliability_source": "candidate_provenance_positive_residual",
    }
    reliability["reliability_sha256"] = canonical_digest(reliability)
    final_chain["reliability"] = reliability
    final_chain.pop("manifest_sha256")
    final_chain["manifest_sha256"] = canonical_digest(final_chain)
    _write_json(final_chain_path, final_chain)

    stages = build_qwen_clstr_multibench_stages(config, scope="smoke")

    assert stages
    for stage in stages:
        assert stage.exports["RELIABILITY_MODE"] == "cmc_candidate_provenance"
        assert stage.exports["FEATURE_UPDATE_COUNT_CAP"] == "16.0"
        assert stage.exports["FEATURE_CANDIDATE_COUNT_CAP"] == "256.0"
        assert stage.exports["SAFE_MEMORY_RESIDUAL_BOUND"] == "2.0"
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_PATH"] == ""
        assert stage.exports["MEMORY_UTILITY_GATE_CHECKPOINT_SHA256"] == ""
        assert stage.exports["MEMORY_UTILITY_GATE_AUDIT_SHA256"] == ""
        assert stage.exports["CHECKPOINT_CHAIN_DIGEST"] == final_chain[
            "checkpoint_chain_digest"
        ]


@pytest.mark.parametrize(
    ("module_name", "runner_name"),
    [
        (
            "scripts.run_toolbench_g3_full_clstr_route_eval",
            "run_toolbench_full_clstr_route_eval",
        ),
        ("scripts.run_toolsandbox_full_clstr_route_eval", "run_toolsandbox_full_clstr_route_eval"),
        ("scripts.run_tau2_full_clstr_route_eval", "run_tau2_full_clstr_route_eval"),
    ],
)
def test_native_route_cli_accepts_candidate_provenance_runtime_mode(
    monkeypatch,
    tmp_path,
    module_name,
    runner_name,
):
    cli = importlib.import_module(module_name)
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli, runner_name, fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module_name,
            "--output_dir",
            str(tmp_path / "out"),
            "--reliability_mode",
            "cmc_candidate_provenance",
        ],
    )

    assert cli.main() == 0
    assert captured["reliability_mode"] == "cmc_candidate_provenance"


def test_toolbench_accepts_candidate_provenance_as_safe_public_mode():
    from clstr.toolbench_full_clstr_route_eval import SAFE_PUBLIC_RELIABILITY_MODES

    assert "cmc_candidate_provenance" in SAFE_PUBLIC_RELIABILITY_MODES


def test_global_pool_accepts_candidate_provenance_runtime_mode(
    monkeypatch,
    tmp_path,
):
    import scripts.run_global_pool_clstr_route_eval as cli
    from clstr.global_pool_route_eval import SAFE_PUBLIC_RELIABILITY_MODES

    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli, "_build_corpus", lambda _args: object())
    monkeypatch.setattr(cli, "run_global_pool_clstr_route_eval", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_global_pool_clstr_route_eval.py",
            "--benchmark",
            "tau2",
            "--output_dir",
            str(tmp_path / "out"),
            "--reliability_mode",
            "cmc_candidate_provenance",
        ],
    )

    assert cli.main() == 0
    assert captured["reliability_mode"] == "cmc_candidate_provenance"
    assert "cmc_candidate_provenance" in SAFE_PUBLIC_RELIABILITY_MODES


def test_alfworld_cli_accepts_candidate_provenance_runtime_mode_before_env_validation(
    monkeypatch,
):
    import scripts.run_alfworld_clstr_eval as cli

    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_alfworld_clstr_eval.py",
            "eval",
            "--reliability_mode",
            "cmc_candidate_provenance",
        ],
    )

    with pytest.raises(ValueError, match="--data_dir or ALFWORLD_DATA is required"):
        cli.main()


def test_stage_builder_accepts_explicit_fallback_batch_profile(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, _manifest_shas = _evaluation_config(tmp_path)
    config["batch_profile"] = {
        "frozen_tau2": 16,
        "native_tau2": 8,
        "native_tau2_stage0": 16,
        "alfworld": 1,
    }

    stages = build_qwen_clstr_multibench_stages(config, scope="smoke")
    by_key = {stage.key: stage for stage in stages}

    assert by_key["frozen/tau2"].exports["BATCH_SIZE"] == "16"
    assert by_key["native/tau2"].exports["BATCH_SIZE"] == "8"
    assert by_key["native/tau2"].exports["STAGE0_CANDIDATE_BATCH_SIZE"] == "16"
    assert by_key["closed_loop/alfworld_valid_seen"].exports["BATCH_SIZE"] == "1"
    assert by_key["closed_loop/alfworld_valid_unseen"].exports["BATCH_SIZE"] == "1"


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "128"])
def test_stage_builder_rejects_invalid_batch_profile_values(tmp_path, invalid):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, _manifest_shas = _evaluation_config(tmp_path)
    config["batch_profile"] = {"frozen_tau2": invalid}

    with pytest.raises(ValueError, match="batch_profile"):
        build_qwen_clstr_multibench_stages(config, scope="smoke")


def test_stage_builder_rejects_a_missing_launcher(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, _manifest_shas = _evaluation_config(tmp_path)
    missing = (
        Path(config["project_root"])
        / "scripts/sbatch/run_qwen06_clstr_native_route_eval.sh"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="native route launcher"):
        build_qwen_clstr_multibench_stages(config, scope="smoke")


def test_full_stage_builder_requires_and_pins_one_accepted_smoke_gate(tmp_path):
    from clstr.qwen_clstr_multibench_submit import (
        build_qwen_clstr_multibench_stages,
        submit_qwen_clstr_multibench_stage_plan,
    )

    config, manifest_shas = _evaluation_config(tmp_path)

    with pytest.raises(ValueError, match="accepted smoke gate"):
        build_qwen_clstr_multibench_stages(config, scope="full")

    gate_path = _write_accepted_smoke_gate(tmp_path, config, manifest_shas)
    config["accepted_smoke_gate_path"] = gate_path
    stages = build_qwen_clstr_multibench_stages(config, scope="full")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    assert [stage.key for stage in stages] == EXPECTED_STAGE_ORDER
    for stage in stages:
        assert stage.exports["EVAL_SCOPE"] == "full"
        assert stage.exports["ACCEPTED_SMOKE_GATE_PATH"] == str(gate_path.resolve())
        assert stage.exports["ACCEPTED_SMOKE_GATE_SHA256"] == gate["report_sha256"]
    by_key = {stage.key: stage for stage in stages}
    assert by_key["closed_loop/alfworld_valid_seen"].exports["MAX_EPISODES"] == "0"
    assert by_key["closed_loop/alfworld_valid_unseen"].exports["MAX_EPISODES"] == "0"
    registry = submit_qwen_clstr_multibench_stage_plan(
        stages=stages,
        registry_path=tmp_path / "full-registry.json",
        submitter=lambda _command: "1300",
    )
    assert registry["submission_identity"]["accepted_smoke_gate_sha256"] == gate[
        "report_sha256"
    ]


def test_full_stage_builder_rejects_a_smoke_gate_from_another_chain(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, manifest_shas = _evaluation_config(tmp_path)
    gate_path = _write_accepted_smoke_gate(tmp_path, config, manifest_shas)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.pop("report_sha256")
    gate["checkpoint_chain_digest"] = "7" * 64
    gate["report_sha256"] = canonical_digest(gate)
    _write_json(gate_path, gate)
    config["accepted_smoke_gate_path"] = gate_path

    with pytest.raises(ValueError, match="checkpoint chain mismatch"):
        build_qwen_clstr_multibench_stages(config, scope="full")


def test_full_stage_builder_rejects_a_smoke_gate_without_terminal_job_evidence(tmp_path):
    from clstr.qwen_clstr_multibench_submit import build_qwen_clstr_multibench_stages

    config, manifest_shas = _evaluation_config(tmp_path)
    gate_path = _write_accepted_smoke_gate(tmp_path, config, manifest_shas)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.pop("report_sha256")
    gate.pop("execution")
    gate["report_sha256"] = canonical_digest(gate)
    _write_json(gate_path, gate)
    config["accepted_smoke_gate_path"] = gate_path

    with pytest.raises(ValueError, match="terminal job evidence"):
        build_qwen_clstr_multibench_stages(config, scope="full")


def test_accepted_smoke_gate_cli_validates_runtime_identity(tmp_path):
    config, manifest_shas = _evaluation_config(tmp_path)
    gate_path = _write_accepted_smoke_gate(tmp_path, config, manifest_shas)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    final_chain = json.loads(
        Path(config["final_chain_manifest_path"]).read_text(encoding="utf-8")
    )
    command = [
        sys.executable,
        "scripts/validate_qwen06_clstr_multibench_smoke_gate.py",
        "--smoke_gate_path",
        str(gate_path),
        "--expected_smoke_gate_sha256",
        gate["report_sha256"],
        "--expected_checkpoint_chain_digest",
        final_chain["checkpoint_chain_digest"],
        "--expected_final_chain_manifest_sha256",
        final_chain["manifest_sha256"],
    ]

    completed = subprocess.run(command, text=True, capture_output=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["report_sha256"] == gate["report_sha256"]

    command[command.index(gate["report_sha256"])] = "6" * 64
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode == 2
    assert "SHA-256 mismatch" in rejected.stderr


def test_submission_registry_supports_dry_run_exact_resume_and_plan_drift_rejection(tmp_path):
    from clstr.qwen_clstr_multibench_submit import (
        build_qwen_clstr_multibench_stages,
        submit_qwen_clstr_multibench_stage_plan,
    )

    config, _manifest_shas = _evaluation_config(tmp_path)
    stages = build_qwen_clstr_multibench_stages(config, scope="smoke")
    registry_path = tmp_path / "submission_registry.json"
    calls: list[list[str]] = []

    dry_run = submit_qwen_clstr_multibench_stage_plan(
        stages=stages,
        registry_path=registry_path,
        submitter=lambda command: calls.append(command) or "999",
        dry_run=True,
    )

    assert dry_run["status"] == "dry_run"
    assert dry_run["stage_order"] == EXPECTED_STAGE_ORDER
    assert len(dry_run["commands"]) == 9
    assert dry_run["commands"][0][0:2] == ["sbatch", "--parsable"]
    assert any(item.startswith("--export=ALL,") for item in dry_run["commands"][0])
    assert calls == []
    assert not registry_path.exists()

    submitted = submit_qwen_clstr_multibench_stage_plan(
        stages=stages,
        registry_path=registry_path,
        submitter=lambda command: calls.append(command) or str(1200 + len(calls)),
    )
    assert submitted["status"] == "submitted"
    assert submitted["stage_order"] == EXPECTED_STAGE_ORDER
    assert len(calls) == 9
    assert [submitted["stages"][key]["job_id"] for key in EXPECTED_STAGE_ORDER] == [
        str(job_id) for job_id in range(1201, 1210)
    ]

    resumed = submit_qwen_clstr_multibench_stage_plan(
        stages=stages,
        registry_path=registry_path,
        submitter=lambda command: calls.append(command) or "9999",
        resume=True,
    )
    assert resumed == submitted
    assert len(calls) == 9

    drifted = [
        replace(stages[0], exports={**stages[0].exports, "TOP_K": "20"}),
        *stages[1:],
    ]
    try:
        submit_qwen_clstr_multibench_stage_plan(
            stages=drifted,
            registry_path=registry_path,
            submitter=lambda _command: "9999",
            resume=True,
        )
    except ValueError as exc:
        assert "stage plan mismatch" in str(exc)
    else:
        raise AssertionError("resume must reject a drifted stage plan")


def test_multibench_cli_forwards_config_scope_registry_and_dry_run(monkeypatch, tmp_path):
    import scripts.submit_qwen06_clstr_multibench as cli

    config_path = tmp_path / "config.json"
    registry_path = tmp_path / "registry.json"
    _write_json(config_path, {"project_root": "/repo"})
    captured = {}
    fake_stages = [object()]

    def fake_build(config, *, scope, partition):
        captured["build"] = {
            "config": config,
            "scope": scope,
            "partition": partition,
        }
        return fake_stages

    def fake_submit(**kwargs):
        captured["submit"] = kwargs
        return {"status": "dry_run"}

    monkeypatch.setattr(cli, "build_qwen_clstr_multibench_stages", fake_build)
    monkeypatch.setattr(cli, "submit_qwen_clstr_multibench_stage_plan", fake_submit)
    args = cli.build_parser().parse_args(
        [
            "--evaluation_config_path",
            str(config_path),
            "--registry_path",
            str(registry_path),
            "--scope",
            "smoke",
            "--partition",
            "gpu_a800,gpu_h100",
            "--dry_run",
        ]
    )

    report = cli.run_from_args(args)

    assert report == {"status": "dry_run"}
    assert captured["build"] == {
        "config": {"project_root": "/repo"},
        "scope": "smoke",
        "partition": "gpu_a800,gpu_h100",
    }
    assert captured["submit"]["stages"] is fake_stages
    assert captured["submit"]["registry_path"] == str(registry_path)
    assert captured["submit"]["resume"] is False
    assert captured["submit"]["dry_run"] is True


def test_qwen_clstr_multibench_wrappers_restore_project_root_and_forward_identity_contract():
    root = Path(__file__).resolve().parents[1]
    wrappers = {
        "frozen": root / "scripts/sbatch/run_qwen06_clstr_frozen_route_eval.sh",
        "native": root / "scripts/sbatch/run_qwen06_clstr_native_route_eval.sh",
        "alfworld": root / "scripts/sbatch/run_qwen06_clstr_alfworld_eval.sh",
    }

    for source in wrappers.values():
        text = source.read_text(encoding="utf-8")
        assert "REQUESTED_PROJECT_ROOT=" in text
        assert 'source "${REQUESTED_PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in text
        assert 'export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"' in text
        assert 'cd "${PROJECT_ROOT}"' in text
        assert "FINAL_CHAIN_MANIFEST_PATH" in text
        assert "CHECKPOINT_CHAIN_DIGEST" in text
        assert "CORPUS_MANIFEST_SHA256" in text
        assert "STAGE0_CHECKPOINT_PATH" in text
        assert "STAGE2_CHECKPOINT_PATH" in text
        assert "STAGE4_CHECKPOINT_PATH" in text
        assert "ACCEPTED_SMOKE_GATE_PATH" in text
        assert "ACCEPTED_SMOKE_GATE_SHA256" in text
        assert "validate_qwen06_clstr_multibench_smoke_gate.py" in text

    frozen = wrappers["frozen"].read_text(encoding="utf-8")
    assert "scripts/run_frozen_clstr_route_eval.py" in frozen
    assert "--expected_benchmark_manifest_sha256" in frozen
    assert "--expected_final_chain_manifest_sha256" in frozen
    assert "--expected_checkpoint_chain_digest" in frozen

    native = wrappers["native"].read_text(encoding="utf-8")
    assert "run_toolbench_g3_full_clstr_route_eval.py" in native
    assert "run_toolsandbox_full_clstr_route_eval.py" in native
    assert "run_tau2_full_clstr_route_eval.py" in native
    assert "--training_skills_path" in native
    assert "STAGE0_CANDIDATE_BATCH_SIZE" in native
    assert "--reliability_mode" in native
    assert "--feature_update_count_cap" in native
    assert "--feature_candidate_count_cap" in native
    assert "--memory_utility_gate_checkpoint_path" in native
    assert "--model_skill_pool_mode" in native
    assert '"${MODEL_SKILL_POOL_MODE:-checkpoint_faithful}"' in native
    assert (
        'if [[ "${MODEL_SKILL_POOL_MODE:-checkpoint_faithful}" '
        '== "checkpoint_faithful" ]]'
    ) in native
    assert 'args+=(--training_skills_path "${TRAINING_SKILLS_PATH}")' in native
    assert "--toolsandbox_replay_mode" in native
    assert '"${TOOLSANDBOX_REPLAY_MODE:-corrected_causal}"' in native
    assert "--task_split" in native
    assert '"${TASK_SPLIT:-base}"' in native

    alfworld = wrappers["alfworld"].read_text(encoding="utf-8")
    assert "scripts/run_alfworld_clstr_eval.py" in alfworld
    assert "--skill_rows_path_override" in alfworld
    assert '"${TRAINING_SKILLS_PATH}"' in alfworld
    assert "--stage0_checkpoint_path" in alfworld
    assert "--benchmark_skill_rows_path" in alfworld
    assert '"${ALFWORLD_SKILLS_PATH}"' in alfworld
    assert "ALFWORLD_SKILLS_MANIFEST_PATH" in alfworld
    assert "ALFWORLD_SKILLS_MANIFEST_SHA256" in alfworld
    assert "--scorer_mode" in alfworld
    assert "unified_memory_admissible_action" in alfworld
    assert "--memory_protocol" in alfworld
    assert "stateful_post_action_v1" in alfworld
    assert "--reliability_mode" in alfworld
    assert "--feature_update_count_cap" in alfworld
    assert "--feature_candidate_count_cap" in alfworld
    assert "SAFE_MEMORY_RESIDUAL_BOUND" in alfworld
    assert "--safe_memory_residual_bound" in alfworld
    assert '"${SAFE_MEMORY_RESIDUAL_BOUND}"' in alfworld
    assert "--protocol_report_path" in alfworld
    assert '"${OUTPUT_DIR}/protocol_report.json"' in alfworld
    assert "--env_report_path" in alfworld
    assert '"${OUTPUT_DIR}/env_report.json"' in alfworld

    alfworld_cli = (root / "scripts/run_alfworld_clstr_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'eval_cmd.add_argument("--safe_memory_residual_bound"' in alfworld_cli
    assert 'eval_cmd.add_argument("--feature_update_count_cap"' in alfworld_cli
    assert 'eval_cmd.add_argument("--feature_candidate_count_cap"' in alfworld_cli
    assert "safe_memory_residual_bound=args.safe_memory_residual_bound" in alfworld_cli
    assert "feature_update_count_cap=args.feature_update_count_cap" in alfworld_cli
    assert "feature_candidate_count_cap=args.feature_candidate_count_cap" in alfworld_cli


def test_alfworld_executor_gate_pins_cmc_final_chain_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh"
    ).read_text(encoding="utf-8")
    cli = (root / "scripts/run_alfworld_qwen_clstr_executor_gate.py").read_text(
        encoding="utf-8"
    )

    for name in (
        "FINAL_CHAIN_MANIFEST_PATH",
        "FINAL_CHAIN_MANIFEST_SHA256",
        "CHECKPOINT_CHAIN_DIGEST",
        "STAGE0_CHECKPOINT_PATH",
        "STAGE2_CHECKPOINT_PATH",
        "STAGE4_CHECKPOINT_PATH",
        "TRAINING_SKILLS_PATH",
        "ALFWORLD_SKILLS_PATH",
        "CORPUS_MANIFEST_SHA256",
    ):
        assert name in launcher
    assert "QWEN_MODEL_NAME_OR_PATH=${QWEN_MODEL_NAME_OR_PATH:-models/Qwen3-14B}" in launcher
    assert "QWEN_WEIGHT=${QWEN_WEIGHT:-1.0}" in launcher
    assert "CLSTR_WEIGHT=${CLSTR_WEIGHT:-0.25}" in launcher
    assert "CLSTR_PRIOR_MODE=${CLSTR_PRIOR_MODE:-unified_memory_concrete_action}" in launcher
    assert "RELIABILITY_MODE=${RELIABILITY_MODE:-cmc}" in launcher
    assert "FEATURE_UPDATE_COUNT_CAP=${FEATURE_UPDATE_COUNT_CAP:-16.0}" in launcher
    assert "FEATURE_CANDIDATE_COUNT_CAP=${FEATURE_CANDIDATE_COUNT_CAP:-256.0}" in launcher
    assert "--stage0_checkpoint_path" in launcher
    assert "--training_skills_path" in launcher
    assert "--benchmark_skill_rows_path" in launcher
    assert "--reliability_mode" in launcher
    assert "--feature_update_count_cap" in launcher
    assert "--feature_candidate_count_cap" in launcher
    assert 'skills_payload.get("canonical_skills")' in launcher
    assert "ALFWorld benchmark skills do not match manifest" in launcher

    assert 'parser.add_argument("--clstr_weight", type=float, default=0.25)' in cli
    assert 'default="unified_memory_concrete_action"' in cli
    assert 'parser.add_argument("--stage0_checkpoint_path"' in cli
    assert 'parser.add_argument("--training_skills_path"' in cli
    assert 'parser.add_argument("--benchmark_skill_rows_path"' in cli
    assert 'parser.add_argument("--reliability_mode", default="cmc"' in cli
    assert "stage0_checkpoint_path=" in cli
    assert "benchmark_skill_rows_path=" in cli
    assert "reliability_mode=args.reliability_mode" in cli
