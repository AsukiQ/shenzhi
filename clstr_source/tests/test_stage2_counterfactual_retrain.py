import inspect
from pathlib import Path

import torch

from clstr.full_base_train import (
    _load_full_base_warm_start_model_state,
    canonical_stage_loss_weights,
    train_clstr_full_base_with_model,
)


def test_stage2_counterfactual_history_cli_and_sbatch_contract():
    weights = canonical_stage_loss_weights(
        "stage2", {"counterfactual_history": 1.0}
    )
    assert weights["counterfactual_history"] == 1.0

    cli = Path("scripts/run_clstr_stage2_full_base_train.py").read_text(
        encoding="utf-8"
    )
    assert "--counterfactual_history_loss_weight" in cli
    assert "--counterfactual_history_margin" in cli
    assert '"counterfactual_history": args.counterfactual_history_loss_weight' in cli

    sbatch = Path(
        "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh"
    ).read_text(encoding="utf-8")
    assert (
        "COUNTERFACTUAL_HISTORY_LOSS_WEIGHT=${COUNTERFACTUAL_HISTORY_LOSS_WEIGHT:-0.0}"
        in sbatch
    )
    assert "COUNTERFACTUAL_HISTORY_MARGIN=${COUNTERFACTUAL_HISTORY_MARGIN:-0.1}" in sbatch
    assert '--counterfactual_history_loss_weight "${COUNTERFACTUAL_HISTORY_LOSS_WEIGHT}"' in sbatch
    assert '--counterfactual_history_margin "${COUNTERFACTUAL_HISTORY_MARGIN}"' in sbatch
    assert "WARM_START_CHECKPOINT_PATH=${WARM_START_CHECKPOINT_PATH:-}" in sbatch
    assert '--warm_start_checkpoint_path "${WARM_START_CHECKPOINT_PATH}"' in sbatch
    assert '--minimum_learning_rate "${MINIMUM_LEARNING_RATE}"' in sbatch
    assert '--checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}"' in sbatch


def test_stage2_warm_start_loads_weights_without_optimizer(tmp_path):
    source = torch.nn.Linear(3, 2, bias=False)
    target = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        source.weight.fill_(0.75)
        target.weight.zero_()
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "stage": "clstr_full_base_component_complete",
            "step": 10000,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": {"must_not_load": True},
        },
        checkpoint,
    )

    report = _load_full_base_warm_start_model_state(target, checkpoint)

    assert torch.equal(target.weight, source.weight)
    assert report["loaded"] is True
    assert report["checkpoint_step"] == 10000
    assert len(report["checkpoint_sha256"]) == 64
    assert report["optimizer_loaded"] is False


def test_stage2_warm_start_runs_after_internal_stage0_reload():
    source = inspect.getsource(train_clstr_full_base_with_model)

    assert source.index("_load_routing_checkpoint_into_model(") < source.index(
        "_load_full_base_warm_start_model_state("
    )
    assert 'normalized_loss_weights["L_policy"] > 0.0' in source
    assert "policy_candidates_zero_weight" in source


def test_qwen_stage2_counterfactual_repair_launcher_contract():
    launcher = Path(
        "scripts/sbatch/run_qwen06_clstr_stage2_counterfactual_repair.sh"
    ).read_text(encoding="utf-8")

    assert "export MAX_STEPS=1200" in launcher
    assert "export TARGET_TOTAL_STEPS=1200" in launcher
    assert "stage2_anchored_full10000_v1/checkpoints/clstr_full_base-step10000.pt" in launcher
    assert "stage2_counterfactual_repair_smoke" in launcher
    assert "export BATCH_SIZE=16" in launcher
    assert "export LEARNING_RATE=3.0e-5" in launcher
    assert "export MINIMUM_LEARNING_RATE=3.0e-6" in launcher
    assert "export CHECKPOINT_INTERVAL_STEPS=300" in launcher
    assert "export EMBEDDING_CACHE_MODE=always" in launcher
    assert "export STAGE0_HANDOFF_SAMPLE_MULTIPLIER=1" in launcher
    assert "export POLICY_LOSS_WEIGHT=0.0" in launcher
    assert "export TRANSITION_SKILL_CE_LOSS_WEIGHT=1.0" in launcher
    assert "export COUNTERFACTUAL_HISTORY_LOSS_WEIGHT=1.0" in launcher
    assert "export ANCHORED_ROUTING_FOUNDATION=0" in launcher
    assert "export STATIC_ROUTE_ANCHOR_WEIGHT=0.0" in launcher
    assert "COUNTERFACTUAL_UTILITY" not in launcher
    assert "STOP_LOSS_WEIGHT" not in launcher
