from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_stage0_cli_exposes_and_forwards_acceleration_options():
    text = _read("scripts/run_clstr_stage0_biencoder_train.py")

    assert "--frozen_backbone_cache_mode" in text
    assert "--frozen_backbone_cache_batch_size" in text
    assert "--resume_skill_table_mode" in text
    assert "frozen_backbone_cache_mode=args.frozen_backbone_cache_mode" in text
    assert (
        "frozen_backbone_cache_batch_size=args.frozen_backbone_cache_batch_size"
        in text
    )
    assert "resume_skill_table_mode=args.resume_skill_table_mode" in text


def test_shared_stage0_launcher_forwards_acceleration_options():
    text = _read("scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh")

    assert "FROZEN_BACKBONE_CACHE_MODE=${FROZEN_BACKBONE_CACHE_MODE:-off}" in text
    assert (
        "FROZEN_BACKBONE_CACHE_BATCH_SIZE=${FROZEN_BACKBONE_CACHE_BATCH_SIZE:-128}"
        in text
    )
    assert "RESUME_SKILL_TABLE_MODE=${RESUME_SKILL_TABLE_MODE:-rebuild}" in text
    assert '--frozen_backbone_cache_mode "${FROZEN_BACKBONE_CACHE_MODE}"' in text
    assert (
        '--frozen_backbone_cache_batch_size "${FROZEN_BACKBONE_CACHE_BATCH_SIZE}"'
        in text
    )
    assert '--resume_skill_table_mode "${RESUME_SKILL_TABLE_MODE}"' in text


def test_qwen_stage0_enables_schedule_after_gpu_gate_and_verifies_resume_table():
    text = _read("scripts/sbatch/run_qwen06_clstr_stage0_train.sh")

    assert "export TRAIN_ENCODER_BACKBONE=0" in text
    assert (
        "export FROZEN_BACKBONE_CACHE_MODE="
        "${FROZEN_BACKBONE_CACHE_MODE:-schedule}" in text
    )
    assert (
        "export FROZEN_BACKBONE_CACHE_BATCH_SIZE="
        "${FROZEN_BACKBONE_CACHE_BATCH_SIZE:-128}" in text
    )
    assert "export RESUME_SKILL_TABLE_MODE=verified_checkpoint" in text
    assert "export RESUME_SKILL_TABLE_MODE=rebuild" in text
    assert "--unfreeze_backbone" not in text
