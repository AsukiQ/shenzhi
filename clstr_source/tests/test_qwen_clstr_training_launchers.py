from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


SBATCH_DIR = Path("scripts/sbatch")


def _launcher(name: str) -> str:
    return (SBATCH_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("launcher_name", "mode_env", "expected_error"),
    [
        (
            "run_qwen06_clstr_stage0_train.sh",
            {"FULL_RUN": "invalid"},
            "FULL_RUN must be 0 or 1",
        ),
        (
            "run_qwen06_clstr_stage0_audit.sh",
            {"TARGET_STEP": "invalid"},
            "TARGET_STEP must be one of",
        ),
    ],
)
def test_stage0_launchers_resolve_project_root_from_slurm_submit_dir(
    tmp_path: Path,
    launcher_name: str,
    mode_env: dict[str, str],
    expected_error: str,
):
    submit_root = tmp_path / "submit_root"
    fake_sbatch_dir = submit_root / "scripts" / "sbatch"
    fake_sbatch_dir.mkdir(parents=True)
    (fake_sbatch_dir / "_clstr_gpu_env.sh").write_text(
        'PYTHON_BIN="python"\nexport PYTHON_BIN\ncd "${PROJECT_ROOT}"\n',
        encoding="utf-8",
    )
    spool_dir = tmp_path / "var" / "spool" / "slurmd" / "job123"
    spool_dir.mkdir(parents=True)
    spooled_script = spool_dir / "slurm_script"
    spooled_script.write_text(_launcher(launcher_name), encoding="utf-8")
    env = os.environ.copy()
    env.pop("PROJECT_ROOT", None)
    env.update({"SLURM_SUBMIT_DIR": str(submit_root), **mode_env})

    completed = subprocess.run(
        ["bash", str(spooled_script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert "No such file or directory" not in completed.stderr


def test_gpu_environment_helper_honors_the_callers_project_root():
    text = _launcher("_clstr_gpu_env.sh")

    assert 'PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}' in text
    assert 'cd "${PROJECT_ROOT}"' in text


def test_stage0_train_launcher_locks_frozen_qwen_unified_memory_contract():
    text = _launcher("run_qwen06_clstr_stage0_train.sh")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --cpus-per-task" not in text
    assert "#SBATCH --mem=" not in text
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in text
    assert 'PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}' in text
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in text
    assert 'cd "${PROJECT_ROOT}"' in text
    assert "FULL_RUN=${FULL_RUN:-0}" in text
    assert "models/Qwen3-Embedding-0.6B" in text
    assert "clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2" in text
    assert "ROUTE_SCORER=unified_memory" in text
    assert "BELIEF_TOP_K=64" in text
    assert "STATE_QUERY_PROMPT_VERSION=clstr_causal_state_v1" in text
    assert "STATE_QUERY_MAX_CHARS=2000" in text
    assert "STATE_QUERY_TRUNCATION=head_tail_v1" in text
    assert "MINED_HARD_NEGATIVE_LOSS_WEIGHT=0.2" in text
    assert "MINED_HARD_NEGATIVE_MARGIN=0.1" in text
    assert "MINED_HARD_NEGATIVE_TOP_K=32" in text
    assert "EXPLICIT_NEGATIVE_LOSS_WEIGHT=0.0" in text
    assert "TRAIN_ENCODER_BACKBONE=0" in text
    assert "--train_encoder_backbone" not in text
    assert "TARGET_STEP" in text
    assert "PREVIOUS_GATE_PATH" in text
    assert "MAX_SKILLS=8192" in text
    assert "MAX_QUERIES=4096" in text
    assert "clstr_unified_retrieval_v2-step0.pt" in text
    assert "audit_qwen06_clstr_lineage.py" in text
    assert "audit_qwen06_clstr_stage0_gate.py" in text


def test_stage0_audit_launcher_uses_fixed_checkpoint_query_protocol_and_promotion_gate():
    text = _launcher("run_qwen06_clstr_stage0_audit.sh")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --cpus-per-task" not in text
    assert "#SBATCH --mem=" not in text
    assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in text
    assert 'PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}' in text
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in text
    assert 'cd "${PROJECT_ROOT}"' in text
    assert "audit_clstr_stage0_handoff_coverage.py" in text
    assert '--top_k_values "20,50,100,200,500"' in text
    assert "--query_modes checkpoint_state_query" in text
    assert "--max_rows 2048" in text
    assert "--batch_size 8" in text
    assert "--model_cache_dir" in text
    assert "audit_qwen06_clstr_stage0_gate.py" in text
    assert "promote" in text
    assert "BASELINE_REPORT" in text
    assert "PREVIOUS_REPORT" in text
    assert "promotion_gate.json" in text


def test_generic_stage1_launcher_forwards_checkpoint_state_query_mode():
    text = _launcher("run_clstr_unified_stage1_heads_init.sh")

    assert "STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}" in text
    assert '--stage0_handoff_query_mode "${STAGE0_HANDOFF_QUERY_MODE}"' in text


def test_generic_stage2_launcher_forwards_optional_target_total_steps():
    text = _launcher("run_clstr_unified_stage2_full_base_train.sh")

    assert 'if [[ -n "${TARGET_TOTAL_STEPS}" ]]; then' in text
    assert 'ARGS+=(--target_total_steps "${TARGET_TOTAL_STEPS}")' in text
    assert "ANCHORED_ROUTING_FOUNDATION=${ANCHORED_ROUTING_FOUNDATION:-1}" in text
    assert '--static_route_anchor_weight "${STATIC_ROUTE_ANCHOR_WEIGHT}"' in text
    assert '--static_route_anchor_max_regression "${STATIC_ROUTE_ANCHOR_MAX_REGRESSION}"' in text
    assert 'ARGS+=(--anchored_routing_foundation)' in text


def test_generic_stage4_launcher_forwards_explicit_method() -> None:
    text = _launcher("run_clstr_unified_stage4_act_train.sh")

    assert "STAGE4_METHOD=${STAGE4_METHOD:-stage4_safe_memory_v1}" in text
    assert '--stage4_method "${STAGE4_METHOD}"' in text


@pytest.mark.parametrize(
    "launcher_name",
    [
        "run_qwen06_clstr_stage1_train.sh",
        "run_qwen06_clstr_stage2_train.sh",
        "run_qwen06_clstr_stage4_train.sh",
    ],
)
def test_downstream_launchers_resolve_project_root_from_slurm_submit_dir(
    tmp_path: Path,
    launcher_name: str,
):
    submit_root = tmp_path / "submit_root"
    fake_sbatch_dir = submit_root / "scripts" / "sbatch"
    fake_sbatch_dir.mkdir(parents=True)
    (fake_sbatch_dir / "_clstr_gpu_env.sh").write_text(
        'PYTHON_BIN="python"\nexport PYTHON_BIN\ncd "${PROJECT_ROOT}"\n',
        encoding="utf-8",
    )
    spool_dir = tmp_path / "var" / "spool" / "slurmd" / "job456"
    spool_dir.mkdir(parents=True)
    spooled_script = spool_dir / "slurm_script"
    spooled_script.write_text(_launcher(launcher_name), encoding="utf-8")
    env = os.environ.copy()
    env.pop("PROJECT_ROOT", None)
    env.update({"SLURM_SUBMIT_DIR": str(submit_root), "FULL_RUN": "invalid"})

    completed = subprocess.run(
        ["bash", str(spooled_script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "FULL_RUN must be 0 or 1" in completed.stderr
    assert "No such file or directory" not in completed.stderr


def test_qwen_downstream_launchers_lock_common_safety_contract():
    for launcher_name in (
        "run_qwen06_clstr_stage1_train.sh",
        "run_qwen06_clstr_stage2_train.sh",
        "run_qwen06_clstr_stage4_train.sh",
    ):
        text = _launcher(launcher_name)
        assert "#SBATCH --gpus=1" in text
        assert "#SBATCH --cpus-per-task" not in text
        assert "#SBATCH --mem=" not in text
        assert 'SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)' in text
        assert 'PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}' in text
        assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in text
        assert 'cd "${PROJECT_ROOT}"' in text
        assert "FULL_RUN=${FULL_RUN:-0}" in text
        assert "SMOKE_MAX_STEPS=2" in text
        assert "SMOKE_MAX_ROWS=128" in text
        assert "1)\n    MAX_STEPS=${FULL_MAX_STEPS}\n    MAX_ROWS=\n" in text
        assert "clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2" in text
        assert "ROUTE_SCORER=unified_memory" in text
        assert "STAGE0_HANDOFF_QUERY_MODE=checkpoint_state_query" in text
        assert "audit_qwen06_clstr_lineage.py" in text
        assert "--expected_model_path" in text
        assert "--expected_skill_pool_path" in text
        assert "--expected_data_manifest_path" in text
        assert "TRAIN_ENCODER_BACKBONE" not in text
        assert "--train_encoder_backbone" not in text


def test_qwen_downstream_launchers_enable_only_gated_exact_schedule_handoff_cache():
    for launcher_name in (
        "run_qwen06_clstr_stage1_train.sh",
        "run_qwen06_clstr_stage2_train.sh",
        "run_qwen06_clstr_stage4_train.sh",
    ):
        text = _launcher(launcher_name)
        assert "export STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16" in text
        assert "export STAGE0_HANDOFF_CACHE_MODE=auto" in text
        assert 'export STAGE0_HANDOFF_CACHE_DIR="${RUN_ROOT}/shared_stage0_handoff_cache"' in text
        assert "export STAGE0_HANDOFF_CACHE_FORMAT=row_sharded_v1" in text
        assert "export STAGE0_HANDOFF_CACHE_SHARD_SIZE=2048" in text


def test_qwen_stage1_launcher_pins_selected_stage0_and_heads_recipe():
    text = _launcher("run_qwen06_clstr_stage1_train.sh")

    assert "FULL_MAX_STEPS=3000" in text
    assert "STAGE0_SELECTION_PATH" in text
    assert "STAGE0_TOP_M=500" in text
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=64" in text
    assert "POLICY_LOSS_WEIGHT=0.6" in text
    assert "TRANSITION_LOSS_WEIGHT=0.2" in text
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=0.6" in text
    assert "STOP_LOSS_WEIGHT" not in text
    assert "RETRIEVAL_LOSS_WEIGHT" not in text
    assert "TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT" not in text
    assert "SAMPLING_STRATEGY=balanced_random" in text
    assert "audit_clstr_stage1_heads_quality.py" in text
    assert "stage1_quality_gate.json" in text
    assert "clstr_stage1_heads-step3000.pt" in text
    assert "--expected_checkpoint_stage clstr_stage1_heads_init" in text


def test_qwen_stage2_launcher_pins_clean_full_pool_replay_recipe():
    text = _launcher("run_qwen06_clstr_stage2_train.sh")

    assert "FULL_MAX_STEPS=10000" in text
    assert 'TARGET_TOTAL_STEPS=${TARGET_TOTAL_STEPS:-${FULL_MAX_STEPS}}' in text
    assert "export TARGET_TOTAL_STEPS" in text
    assert 'EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-always}' in text
    assert 'EMBEDDING_CACHE_MODE=${EMBEDDING_CACHE_MODE:-auto}' in text
    assert 'if [[ "${FULL_RUN}" == "1" ]]' in text
    assert 'EMBEDDING_CACHE_MAX_ROWS=${EMBEDDING_CACHE_MAX_ROWS:-20000}' in text
    assert "export EMBEDDING_CACHE_MODE EMBEDDING_CACHE_MAX_ROWS" in text
    assert "full Stage2 requires EMBEDDING_CACHE_MODE=always" in text
    assert "export EMBEDDING_CACHE_MODE=auto" not in text
    assert "STAGE1_QUALITY_GATE_PATH" in text
    assert "STAGE1_LINEAGE_PATH" in text
    assert "STAGE0_TOP_M=500" in text
    assert "NEXT_SKILL_POOL_MODE=full_pool" in text
    assert "TRANSITION_INVENTORY_MASK_MODE=explicit_only" in text
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=64" in text
    assert "ANCHORED_ROUTING_FOUNDATION=1" in text
    assert "STATIC_ROUTE_ANCHOR_WEIGHT=0.1" in text
    assert "STATIC_ROUTE_ANCHOR_MAX_REGRESSION=0.005" in text
    assert '--static_route_anchor_max_regression "${STATIC_ROUTE_ANCHOR_MAX_REGRESSION}"' in text
    assert "COUNTERFACTUAL_UTILITY_LOSS_WEIGHT" not in text
    assert "COUNTERFACTUAL_GAIN_MARGIN" not in text
    assert "COUNTERFACTUAL_SAFETY_TOLERANCE" not in text
    assert "COUNTERFACTUAL_WARMUP_FRACTION" not in text
    assert "SAFE_MEMORY_RESIDUAL_BOUND" not in text
    assert "SAFE_LOCAL_CANDIDATE_SIZES" not in text
    assert "STOP_LOSS_WEIGHT" not in text
    assert "RETRIEVAL_LOSS_WEIGHT" not in text
    assert "Q_SUCCESS_LOSS_WEIGHT" not in text
    assert "POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT" not in text
    assert "TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT" not in text
    assert "AUTO_REPLAY_PREFIX_MAX_STEPS=3" in text
    assert "TRAINABLE_REPLAY_PREFIX=1" in text
    assert "audit_clstr_stage2_quality.py" in text
    assert "stage2_quality_gate.json" in text
    assert 'clstr_full_base-step${TARGET_TOTAL_STEPS}.pt' in text
    assert "--expected_checkpoint_stage clstr_full_base_component_complete" in text


def test_stage2_python_cli_exposes_target_total_steps():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_clstr_stage2_full_base_train.py",
            "--help",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "--target_total_steps" in completed.stdout


def test_canonical_stage_clis_hide_removed_loss_flags():
    stage1 = subprocess.run(
        [sys.executable, "scripts/run_clstr_stage1_heads_init.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    stage2 = subprocess.run(
        [sys.executable, "scripts/run_clstr_stage2_full_base_train.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert stage1.returncode == 0
    assert stage2.returncode == 0
    for removed in (
        "--transition_hard_negative_margin_loss_weight",
        "--stop_loss_weight",
        "--retrieval_loss_weight",
    ):
        assert removed not in stage1.stdout
        assert removed not in stage2.stdout
    for removed in (
        "--policy_hard_negative_margin_loss_weight",
        "--q_success_loss_weight",
        "--counterfactual_utility_loss_weight",
    ):
        assert removed not in stage2.stdout
def test_qwen_stage4_launcher_pins_causal_recipe_and_derived_lineage():
    text = _launcher("run_qwen06_clstr_stage4_train.sh")

    assert "FULL_MAX_STEPS=3000" in text
    assert (
        "STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-${RUN_ROOT}/"
        "stage2_anchored_full10000_v1}" in text
    )
    assert "OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/stage4_cmc_full}" in text
    assert "STAGE4_METHOD=counterfactual_memory_calibration_v1" in text
    assert "LEARNING_RATE=3.0e-5" in text
    assert "MINIMUM_LEARNING_RATE=3.0e-6" in text
    assert "LEARNING_RATE_WARMUP_FRACTION=0.05" in text
    assert "SMOKE_VALIDATION_INTERVAL_STEPS=1" in text
    assert "FULL_VALIDATION_INTERVAL_STEPS=400" in text
    assert "VALIDATION_INTERVAL_STEPS=${SMOKE_VALIDATION_INTERVAL_STEPS}" in text
    assert "VALIDATION_INTERVAL_STEPS=${FULL_VALIDATION_INTERVAL_STEPS}" in text
    assert "FULL_VALIDATION_ROWS_PER_BENCHMARK=256" in text
    assert "FULL_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=128" in text
    assert "FULL_GATE_ROWS_PER_BENCHMARK=1" in text
    assert "SMOKE_VALIDATION_ROWS_PER_BENCHMARK=4" in text
    assert "SMOKE_MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=1" in text
    assert "SMOKE_GATE_ROWS_PER_BENCHMARK=1" in text
    assert "SMOKE_BENCHMARK_CAPS=toolbench_g3=32:traject_bench=32:alfworld=32:webshop=32" in text
    assert "FULL_BENCHMARK_CAPS=toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1" in text
    assert 'BENCHMARK_CAPS=${BENCHMARK_CAPS:-${SMOKE_BENCHMARK_CAPS}}' in text
    assert 'BENCHMARK_CAPS=${BENCHMARK_CAPS:-${FULL_BENCHMARK_CAPS}}' in text
    assert "STAGE2_QUALITY_GATE_PATH" in text
    assert "STAGE2_LINEAGE_PATH" in text
    assert "STAGE0_TOP_M=500" in text
    assert "CANDIDATE_COUNT=64" in text
    assert "NEXT_SKILL_POOL_MODE=full_pool" in text
    assert "COUNTERFACTUAL_UTILITY_WEIGHT" not in text
    assert "COUNTERFACTUAL_GAIN_MARGIN" not in text
    assert "COUNTERFACTUAL_SAFETY_TOLERANCE" not in text
    assert "COUNTERFACTUAL_WARMUP_FRACTION" not in text
    assert "SAFE_MEMORY_RESIDUAL_BOUND" not in text
    assert "SAFE_LOCAL_CANDIDATE_SIZES" not in text
    assert "AUTO_REPLAY_PREFIX_MAX_STEPS=3" in text
    assert "TRAINABLE_REPLAY_PREFIX=1" not in text
    assert "TRAIN_TRANSITION=1" not in text
    assert "audit_clstr_stage4_quality.py" in text
    assert "stage4_quality_gate.json" in text
    assert "stage4_dynamic_selection.json" in text
    assert "audit_clstr_memory_utility_oracle.py" not in text
    assert "train_clstr_memory_utility_gate.py" not in text
    assert "finalize_clstr_stage4_selection.py" in text
    assert "stage4_selection.json" in text
    assert "clstr_stage4_act-step3000.pt" not in text
    assert "create-derived" in text
    assert "--stage4_selection_path" in text
    assert "--identity_parent_role stage2" in text
    assert "--expected_checkpoint_stage clstr_stage4_transition_conditioned_next_skill" in text
    assert "train_encoder_backbone" not in text
    assert "unfreeze" not in text.lower()
def test_alfworld_executor_cli_supports_imported_control_and_cmc_gate_overlay():
    text = Path("scripts/run_alfworld_qwen_clstr_executor_gate.py").read_text(
        encoding="utf-8"
    )

    assert "--qwen_control_import_manifest" in text
    assert "load_alfworld_qwen_control_import" in text
    assert "--memory_utility_gate_checkpoint_path" in text
    assert "--expected_memory_utility_gate_checkpoint_sha256" in text
    assert "--expected_memory_utility_gate_audit_sha256" in text
    assert "resolve_reliability_gate" in text
    assert "memory_utility_gate=memory_utility_gate" in text


def test_candidate_admission_stage4_launcher_uses_cached_qwen_schedule() -> None:
    text = _launcher("run_qwen06_clstr_candidate_admission_stage4.sh")

    assert "STAGE4_METHOD=candidate_admission_residual_v1" in text
    assert "SMOKE_MAX_STEPS=300" in text
    assert "FULL_MAX_STEPS=1200" in text
    assert "SMOKE_VALIDATION_INTERVAL_STEPS=100" in text
    assert "FULL_VALIDATION_INTERVAL_STEPS=300" in text
    assert "BATCH_SIZE=16" in text
    assert "LEARNING_RATE=3.0e-5" in text
    assert "MINIMUM_LEARNING_RATE=3.0e-6" in text
    assert "stage4_candidate_admission_smoke" in text
    assert "stage4_candidate_admission_full" in text
    assert "stage4_cmc_full/stage4_selection.json" in text
    assert "shared_stage0_handoff_cache" in text
    assert "row_sharded_v1" in text
    assert "run_qwen06_clstr_stage4_train.sh" in text
