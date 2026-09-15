import json
import inspect
from pathlib import Path


def test_stage12_cli_keeps_training_stack_imports_lazy():
    text = Path("scripts/run_clstr_stage12_consolidated_train.py").read_text(encoding="utf-8")
    prefix = text.split("def _load_training_stack", 1)[0]

    assert "from clstr.full_base_train import" not in prefix
    assert "from clstr.stage1_heads_quality_gate import" not in prefix
    assert "from clstr.stage2_quality_gate import" not in prefix
    assert "def _load_training_stack" in text


def test_stage12_consolidated_train_runs_warmup_gate_main_and_writes_report(tmp_path, monkeypatch):
    from scripts import run_clstr_stage12_consolidated_train as runner

    calls: list[tuple[str, dict]] = []
    routing_checkpoint = tmp_path / "stage0.pt"
    routing_checkpoint.write_bytes(b"stage0")

    def fake_stage1(**kwargs):
        calls.append(("stage1", kwargs))
        output_dir = Path(kwargs["output_dir"])
        checkpoint = output_dir / "checkpoints" / "clstr_stage1_heads-step7.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"stage1")
        return {
            "status": "ok",
            "stage": "clstr_stage1_heads_init",
            "checkpoint": str(checkpoint),
        }

    def fake_stage1_gate(**kwargs):
        calls.append(("stage1_gate", kwargs))
        Path(kwargs["output_path"]).write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
        return {"status": "ok", "blockers": []}

    def fake_stage2(**kwargs):
        calls.append(("stage2", kwargs))
        output_dir = Path(kwargs["output_dir"])
        checkpoint = output_dir / "checkpoints" / "clstr_full_base-step11.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"stage2")
        return {
            "status": "ok",
            "stage": "clstr_full_base_component_complete",
            "checkpoint": str(checkpoint),
        }

    def fake_stage2_gate(**kwargs):
        calls.append(("stage2_gate", kwargs))
        Path(kwargs["output_path"]).write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
        return {"status": "ok", "blockers": []}

    monkeypatch.setattr(runner, "run_clstr_stage1_heads_init", fake_stage1)
    monkeypatch.setattr(runner, "audit_stage1_heads_quality", fake_stage1_gate)
    monkeypatch.setattr(runner, "run_clstr_full_base_train", fake_stage2)
    monkeypatch.setattr(runner, "audit_stage2_full_base_quality", fake_stage2_gate)

    report = runner.run_stage12_consolidated_train(
        train_path=tmp_path / "trajectories.jsonl",
        skills_path=tmp_path / "skill_pool.jsonl",
        output_dir=tmp_path / "stage12",
        routing_checkpoint_path=routing_checkpoint,
        stage1_max_steps=7,
        stage2_max_steps=11,
        batch_size=3,
        learning_rate=2.0e-4,
        stage0_top_m=500,
        transition_inventory_mask_mode="explicit_only",
        transition_inventory_min_candidates=50,
        transition_loss_type="listwise_nll",
        transition_positive_mode="gold_plus_equivalent",
        transition_residual_lambda=0.25,
        transition_scoring_mode="stage0_rank_prior_plus_transition_residual",
        route_scorer="unified_memory",
        next_skill_pool_mode="full_pool",
        counterfactual_gain_margin=0.12,
        counterfactual_safety_tolerance=0.02,
        counterfactual_gain_weight=1.3,
        counterfactual_safety_weight=0.8,
        counterfactual_warmup_fraction=0.15,
        stage2_counterfactual_utility_loss_weight=0.07,
        gated_temporal_lambda_max=0.5,
        gated_temporal_kl_alpha=0.03,
        gated_temporal_rank_drop_beta=0.05,
        gated_temporal_context_top_k=64,
        freeze_gated_temporal_only=True,
        stage1_freeze_gated_temporal_only=True,
        stage2_freeze_gated_temporal_only=False,
        sampling_strategy="benchmark_transition_quota_random",
        stage0_handoff_cache_mode="refresh",
        stage0_handoff_cache_dir=tmp_path / "cache" / "handoff",
        auto_replay_prefix_max_steps=3,
        trainable_replay_prefix=True,
        stage2_gate_max_recall_at_5_drop_vs_stage0_prior=0.03,
        stage2_gate_max_mrr_drop_vs_stage0_prior=0.02,
        stage2_gate_max_worse_than_stage0_prior_fraction=0.4,
    )

    assert [name for name, _ in calls] == ["stage1", "stage1_gate", "stage2", "stage2_gate"]
    assert calls[0][1]["routing_checkpoint_path"] == routing_checkpoint
    assert calls[0][1]["output_dir"] == tmp_path / "stage12" / "stage1_heads_init"
    assert calls[0][1]["stage0_top_m"] == 500
    assert calls[0][1]["transition_inventory_mask_mode"] == "explicit_only"
    assert calls[0][1]["loss_weights"]["counterfactual_utility"] == 0.0
    assert calls[0][1]["gated_temporal_lambda_max"] == 0.5
    assert calls[0][1]["gated_temporal_kl_alpha"] == 0.03
    assert calls[0][1]["gated_temporal_rank_drop_beta"] == 0.05
    assert calls[0][1]["gated_temporal_context_top_k"] == 64
    assert calls[0][1]["route_scorer"] == "unified_memory"
    assert calls[0][1]["freeze_gated_temporal_only"] is True
    assert calls[0][1]["auto_replay_prefix_max_steps"] == 0
    assert calls[0][1]["trainable_replay_prefix"] is False
    assert calls[0][1]["stage0_handoff_cache_mode"] == "refresh"
    assert calls[0][1]["stage0_handoff_cache_dir"] == tmp_path / "cache" / "handoff"
    assert calls[1][1]["min_steps"] == 7
    assert calls[1][1]["expected_route_scorer"] == "unified_memory"
    assert calls[2][1]["routing_checkpoint_path"] == routing_checkpoint
    assert calls[2][1]["stage1_checkpoint_path"] == Path(calls[0][1]["output_dir"]) / "checkpoints" / "clstr_stage1_heads-step7.pt"
    assert calls[2][1]["output_dir"] == tmp_path / "stage12" / "stage2_full_base"
    assert calls[2][1]["stage0_top_m"] == 500
    assert calls[2][1]["transition_inventory_mask_mode"] == "explicit_only"
    assert calls[2][1]["next_skill_pool_mode"] == "full_pool"
    assert calls[2][1]["loss_weights"]["counterfactual_utility"] == 0.07
    assert calls[2][1]["counterfactual_gain_margin"] == 0.12
    assert calls[2][1]["counterfactual_safety_tolerance"] == 0.02
    assert calls[2][1]["counterfactual_gain_weight"] == 1.3
    assert calls[2][1]["counterfactual_safety_weight"] == 0.8
    assert calls[2][1]["counterfactual_warmup_fraction"] == 0.15
    assert calls[2][1]["gated_temporal_lambda_max"] == 0.5
    assert calls[2][1]["gated_temporal_kl_alpha"] == 0.03
    assert calls[2][1]["gated_temporal_rank_drop_beta"] == 0.05
    assert calls[2][1]["gated_temporal_context_top_k"] == 64
    assert calls[2][1]["route_scorer"] == "unified_memory"
    assert calls[2][1]["freeze_gated_temporal_only"] is False
    assert calls[2][1]["auto_replay_prefix_max_steps"] == 3
    assert calls[2][1]["trainable_replay_prefix"] is True
    assert calls[3][1]["expected_route_scorer"] == "unified_memory"
    assert calls[3][1]["expected_next_skill_pool_mode"] == "full_pool"
    assert calls[2][1]["stage0_handoff_cache_mode"] == "refresh"
    assert calls[2][1]["stage0_handoff_cache_dir"] == tmp_path / "cache" / "handoff"
    assert calls[3][1]["min_steps"] == 11
    assert calls[3][1]["max_stage2_recall_at_5_drop_vs_stage0_prior"] == 0.03
    assert calls[3][1]["max_stage2_mrr_drop_vs_stage0_prior"] == 0.02
    assert calls[3][1]["max_stage2_worse_than_stage0_prior_fraction"] == 0.4

    assert report["status"] == "ok"
    assert report["stage"] == "clstr_stage12_supervised_heads"
    assert report["checkpoint"].endswith("clstr_full_base-step11.pt")
    written = json.loads((tmp_path / "stage12" / "consolidated_train_report.json").read_text(encoding="utf-8"))
    assert written["stage1"]["gate"]["status"] == "ok"
    assert written["stage2"]["gate"]["status"] == "ok"
    assert written["config"]["stage0_handoff_cache_mode"] == "refresh"
    assert written["config"]["stage0_handoff_cache_dir"] == str(tmp_path / "cache" / "handoff")
    assert written["config"]["next_skill_pool_mode"] == "full_pool"
    assert written["config"]["stage2_counterfactual_utility_loss_weight"] == 0.07
    assert written["config"]["counterfactual_warmup_fraction"] == 0.15


def test_stage12_consolidated_defaults_to_full_pool_counterfactual_mainline():
    from scripts.run_clstr_stage12_consolidated_train import run_stage12_consolidated_train

    parameters = inspect.signature(run_stage12_consolidated_train).parameters

    assert parameters["route_scorer"].default == "unified_memory"
    assert parameters["next_skill_pool_mode"].default == "full_pool"
    assert parameters["transition_inventory_mask_mode"].default == "explicit_only"
    assert parameters["stage2_counterfactual_utility_loss_weight"].default == 0.05
    assert parameters["counterfactual_gain_margin"].default == 0.1
    assert parameters["counterfactual_safety_tolerance"].default == 0.01
    assert parameters["counterfactual_gain_weight"].default == 1.0
    assert parameters["counterfactual_safety_weight"].default == 1.0
    assert parameters["counterfactual_warmup_fraction"].default == 0.1


def test_stage12_consolidated_sbatch_uses_function_aug_v2_mainline_and_single_gpu():
    script = Path("scripts/sbatch/run_clstr_unified_stage12_consolidated_function_aug_v2.sh").read_text(
        encoding="utf-8"
    )

    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/skill_pool.jsonl" in script
    assert "outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_consolidated" in script
    assert "scripts/run_clstr_stage12_consolidated_train.py" in script
    assert "BATCH_SIZE=${BATCH_SIZE:-16}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-500}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-explicit_only}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-50}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "GATED_TEMPORAL_LAMBDA_MAX=${GATED_TEMPORAL_LAMBDA_MAX:-0.5}" in script
    assert "GATED_TEMPORAL_KL_ALPHA=${GATED_TEMPORAL_KL_ALPHA:-0.03}" in script
    assert "GATED_TEMPORAL_RANK_DROP_BETA=${GATED_TEMPORAL_RANK_DROP_BETA:-0.05}" in script
    assert "GATED_TEMPORAL_CONTEXT_TOP_K=${GATED_TEMPORAL_CONTEXT_TOP_K:-64}" in script
    assert "FREEZE_GATED_TEMPORAL_ONLY=${FREEZE_GATED_TEMPORAL_ONLY:-true}" in script
    assert "STAGE1_FREEZE_GATED_TEMPORAL_ONLY=${STAGE1_FREEZE_GATED_TEMPORAL_ONLY:-auto}" in script
    assert "STAGE2_FREEZE_GATED_TEMPORAL_ONLY=${STAGE2_FREEZE_GATED_TEMPORAL_ONLY:-auto}" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-benchmark_transition_quota_random}" in script
    assert "STAGE1_AUTO_REPLAY_PREFIX_MAX_STEPS=${STAGE1_AUTO_REPLAY_PREFIX_MAX_STEPS:-0}" in script
    assert "STAGE2_AUTO_REPLAY_PREFIX_MAX_STEPS=${STAGE2_AUTO_REPLAY_PREFIX_MAX_STEPS:-${AUTO_REPLAY_PREFIX_MAX_STEPS}}" in script
    assert "STAGE1_TRAINABLE_REPLAY_PREFIX=${STAGE1_TRAINABLE_REPLAY_PREFIX:-false}" in script
    assert "STAGE2_TRAINABLE_REPLAY_PREFIX=${STAGE2_TRAINABLE_REPLAY_PREFIX:-${TRAINABLE_REPLAY_PREFIX}}" in script
    assert "STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-auto}" in script
    assert "STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}" in script
    assert "STAGE1_POLICY_LOSS_WEIGHT=${STAGE1_POLICY_LOSS_WEIGHT:-0.6}" in script
    assert "STAGE1_TRANSITION_SKILL_CE_LOSS_WEIGHT=${STAGE1_TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.6}" in script
    assert "ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}" in script
    assert "NEXT_SKILL_POOL_MODE=${NEXT_SKILL_POOL_MODE:-full_pool}" in script
    assert "STAGE2_COUNTERFACTUAL_UTILITY_LOSS_WEIGHT=${STAGE2_COUNTERFACTUAL_UTILITY_LOSS_WEIGHT:-0.05}" in script
    assert "COUNTERFACTUAL_GAIN_MARGIN=${COUNTERFACTUAL_GAIN_MARGIN:-0.1}" in script
    assert "COUNTERFACTUAL_SAFETY_TOLERANCE=${COUNTERFACTUAL_SAFETY_TOLERANCE:-0.01}" in script
    assert "COUNTERFACTUAL_GAIN_WEIGHT=${COUNTERFACTUAL_GAIN_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_SAFETY_WEIGHT=${COUNTERFACTUAL_SAFETY_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_WARMUP_FRACTION=${COUNTERFACTUAL_WARMUP_FRACTION:-0.1}" in script
    assert "--next_skill_pool_mode" in script
    assert "--stage2_counterfactual_utility_loss_weight" in script
    assert "--counterfactual_gain_margin" in script
    assert "--counterfactual_safety_tolerance" in script
    assert "--counterfactual_gain_weight" in script
    assert "--counterfactual_safety_weight" in script
    assert "--counterfactual_warmup_fraction" in script
    assert "STAGE2_GATE_MAX_RECALL_AT_5_DROP_VS_STAGE0_PRIOR=${STAGE2_GATE_MAX_RECALL_AT_5_DROP_VS_STAGE0_PRIOR:-0.03}" in script
    assert "--stage2_gate_max_recall_at_5_drop_vs_stage0_prior" in script
    assert "--gated_temporal_lambda_max" in script
    assert "--gated_temporal_kl_alpha" in script
    assert "--gated_temporal_rank_drop_beta" in script
    assert "--gated_temporal_context_top_k" in script
    assert "--stage1_freeze_gated_temporal_only" in script
    assert "--stage2_freeze_gated_temporal_only" in script
    assert "--freeze_gated_temporal_only" in script
    assert "--stage0_handoff_cache_mode" in script
    assert "--stage0_handoff_cache_dir" in script
    assert "\nsbatch " not in script
    assert "\nsbatch\t" not in script
    assert "--gres" not in script
    assert "--mem" not in script
    assert "--cpus-per-task" not in script
