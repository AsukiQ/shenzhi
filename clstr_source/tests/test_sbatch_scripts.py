from pathlib import Path


CANONICAL_STAGE0_OUTPUT_DIR = "outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE"
CANONICAL_STAGE0_CHECKPOINT = (
    f"{CANONICAL_STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt"
)
CANONICAL_STAGE0_EVAL_METRICS = f"{CANONICAL_STAGE0_OUTPUT_DIR}/full_retrieval_eval/metrics.json"
V4_2_NOWEAK_DATA_ROOT = "data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak"
V4_2_STAGE0_OUTPUT_DIR = "outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE"
V4_2_STAGE0_CHECKPOINT = (
    f"{V4_2_STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt"
)
V4_2_STAGE0_EVAL_METRICS = f"{V4_2_STAGE0_OUTPUT_DIR}/full_retrieval_eval/metrics.json"
V4_2_STAGE1_OUTPUT_DIR = "outputs/clstr_unified_stage1_v4_2_nowweak_top350_inventory_listwise_heads_init"
V4_2_STAGE1_CHECKPOINT = f"{V4_2_STAGE1_OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt"
V4_2_STAGE2_OUTPUT_DIR = "outputs/clstr_unified_stage2_v4_2_nowweak_top350_inventory_listwise"
V4_2_PROGRESSIVE_DATA_ROOT = "data/clstr_unified_pretrain_v4_2_progressive_final"
V4_2_PROGRESSIVE_STAGE1_OUTPUT_DIR = (
    "outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"
)
V4_2_PROGRESSIVE_STAGE1_CHECKPOINT = (
    f"{V4_2_PROGRESSIVE_STAGE1_OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt"
)
V4_2_PROGRESSIVE_STAGE2_STAGE1_OUTPUT_DIR = (
    "outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init"
)
V4_2_PROGRESSIVE_STAGE2_STAGE1_CHECKPOINT = (
    f"{V4_2_PROGRESSIVE_STAGE2_STAGE1_OUTPUT_DIR}/checkpoints/clstr_stage1_heads-step3000.pt"
)
V4_2_PROGRESSIVE_STAGE2_OUTPUT_DIR = (
    "outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"
)
V4_2_PROGRESSIVE_STAGE2_CHECKPOINT = f"{V4_2_PROGRESSIVE_STAGE2_OUTPUT_DIR}/checkpoints/clstr_full_base-step10000.pt"
V4_2_PROGRESSIVE_STAGE2_RANKPRIOR_OUTPUT_DIR = (
    "outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025"
)
V4_2_PROGRESSIVE_STAGE4_OUTPUT_DIR = "outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act"
V4_2_PROGRESSIVE_LOGGED_ONLINE_STAGE4_OUTPUT_DIR = "outputs/logged_online_stage4_v4_2_progressive_smoke"
V4_2_PROGRESSIVE_STAGE0_HANDOFF_GATE = (
    "outputs/clstr_stage0_handoff_coverage_audit/"
    "v4_2_progressive_final_balanced3840/handoff_coverage_gate.json"
)
V4_1B_DATA_ROOT = "data/clstr_unified_pretrain_v4_1b"
V4_1B_STAGE0_CHECKPOINT = (
    "outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt"
)
V4_1B_STAGE1_CHECKPOINT = (
    "outputs/clstr_unified_stage1_v4_1b_top350_heads_init/checkpoints/clstr_stage1_heads-step3000.pt"
)
V4_1B_CONSERVATIVE_LISTWISE_STAGE2_CHECKPOINT = (
    "outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt"
)
V4_1B_CONSERVATIVE_LISTWISE_STAGE4_OUTPUT_DIR = (
    "outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act"
)
FUNCTION_AUG_V2_DATA_ROOT = "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2"
FUNCTION_AUG_V2_STAGE0_CHECKPOINT = (
    "outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue300_from4400/"
    "checkpoints/clstr_unified_retrieval_v2-step300.pt"
)
FUNCTION_AUG_V2_STAGE2_CHECKPOINT = (
    "outputs/clstr_unified_stage2_function_aug_v2_true_adapt_s0_top500_stage0prior50_"
    "rankprior_trainl025_calib_full/checkpoints/clstr_full_base-step10000.pt"
)
FUNCTION_AUG_V2_STAGE4_OUTPUT_DIR = (
    "outputs/clstr_unified_stage4_function_aug_v2_true_adapt_s0_top500_stage0prior50_"
    "rankprior_l050_joint_act"
)
FUNCTION_AUG_V2_TOOLBENCH_CLEAN_DATA_ROOT = (
    "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean"
)


def _assert_sources_shared_clstr_gpu_env(script: str) -> None:
    assert "PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}" in script
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert 'source "$(dirname "$0")/_clstr_gpu_env.sh"' not in script


def test_prior_gate_sweep_sbatch_uses_clean_stage1_checkpoint_and_gpu_env():
    script = Path("scripts/sbatch/run_clstr_prior_gate_sweep.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/sweep_clstr_prior_gate.py" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean/trajectories.jsonl" in script
    assert "outputs/clstr_unified_stage0_function_aug_v2_toolbench_clean_true_adapt_proj_full5000/checkpoints/clstr_unified_retrieval_v2-step5000.pt" in script
    assert "outputs/clstr_unified_stage12_function_aug_v2_toolbench_clean_quota_b16_full2_s1_3000_s2_10000/stage1_heads_init/checkpoints/clstr_stage1_heads-step3000.pt" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-stage0_topk_trajectory_prior}" in script
    assert "FORMULAS=${FORMULAS:-fixed_0_00,fixed_0_10,fixed_0_15,fixed_0_25,fixed_0_50,margin_gate,entropy_gate,margin_entropy_gate}" in script
    assert "--stage_checkpoint_path" in script
    assert "--transition_inventory_mask_mode" in script
    assert "--formulas" in script


def test_appworld_clstr_eval_sbatch_exposes_policy_ranking_mode():
    script = Path("scripts/sbatch/run_appworld_clstr_eval.sh").read_text(encoding="utf-8")

    assert "RANKING_MODE" in script
    assert "CANDIDATE_TOP_K" in script
    assert "POLICY_BLEND_ALPHA" in script
    assert "ALLOW_LEGACY_POLICY_SKILL_ROUTER" in script
    assert "--ranking_mode" in script
    assert "--candidate_top_k" in script
    assert "--policy_blend_alpha" in script
    assert "--allow_legacy_policy_skill_router" in script


def test_appworld_qwen_executor_sbatch_exposes_skill_context_mode():
    script = Path("scripts/sbatch/run_appworld_qwen_executor_eval.sh").read_text(encoding="utf-8")

    assert "SKILL_CONTEXT_MODE" in script
    assert "--skill_context_mode" in script


def test_appworld_official_executor_sbatch_normalizes_task_ids_for_sbatch_export():
    script = Path("scripts/sbatch/run_appworld_official_executor_eval.sh").read_text(encoding="utf-8")

    assert "TASK_IDS=${TASK_IDS:-}" in script
    assert 'TASK_IDS="${TASK_IDS//:/,}"' in script
    assert '--task_ids "${TASK_IDS}"' in script
    assert "HANDOFF_PROMPT_STYLE" in script
    assert "--handoff_prompt_style" in script
    assert "HANDOFF_VISIBLE_SKILL_LIMIT" in script
    assert "--handoff_visible_skill_limit" in script
    assert "COMPLETION_PRECHECK_MODE" in script
    assert "--completion_precheck_mode" in script
    assert "PREFLIGHT_MAX_REPAIRS" in script
    assert "--preflight_max_repairs" in script
    assert "CURRENT_ROUTE_ROLLOUTS_PATH" in script
    assert "--current_route_rollouts_path" in script
    assert "CLSTR_STATE_CONTEXT" in script
    assert "--clstr_state_context" in script


def test_appworld_current_route_online_stage4_sbatch_uses_official_executor_and_online_updates():
    script = Path("scripts/sbatch/run_appworld_current_route_online_stage4_train.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_appworld_current_route_online_stage4_train.py" in script
    assert "TASK_IDS=${TASK_IDS:-}" in script
    assert 'TASK_IDS="${TASK_IDS//:/,}"' in script
    assert "MAX_ONLINE_UPDATES=${MAX_ONLINE_UPDATES:-1}" in script
    assert "--max_online_updates" in script
    assert "ONLINE_UPDATE_EPOCHS=${ONLINE_UPDATE_EPOCHS:-1}" in script
    assert "--online_update_epochs" in script
    assert "REPLAY_SUCCESS_ROLLOUTS_PATH=${REPLAY_SUCCESS_ROLLOUTS_PATH:-}" in script
    assert "--replay_success_rollouts_path" in script
    assert "MAX_REPLAY_UPDATES=${MAX_REPLAY_UPDATES:-0}" in script
    assert "--max_replay_updates" in script
    assert "COMPLETION_PRECHECK_MODE=${COMPLETION_PRECHECK_MODE:-constraint_tokens}" in script
    assert "--completion_precheck_mode" in script
    assert "PREFLIGHT_MAX_REPAIRS=${PREFLIGHT_MAX_REPAIRS:-0}" in script
    assert "--preflight_max_repairs" in script
    assert "RANKING_MODE=${RANKING_MODE:-policy_transition_blend}" in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-true}" in script
    assert "STABILITY_KL_WEIGHT=${STABILITY_KL_WEIGHT:-0.1}" in script
    assert "LOSS_SCORE_MODE=${LOSS_SCORE_MODE:-policy_transition_blend}" in script
    assert "ENABLE_FAILURE_STAGE4_CORRECTION=${ENABLE_FAILURE_STAGE4_CORRECTION:-false}" in script
    assert "--enable_failure_stage4_correction" in script
    assert "LEARNED_COMPONENT_MIN_RANGE=${LEARNED_COMPONENT_MIN_RANGE:-0.1}" in script
    assert "LEARNED_COMPONENT_TRUST_TOP_K=${LEARNED_COMPONENT_TRUST_TOP_K:-80}" in script
    assert "TOP_K=${TOP_K:-20}" in script
    assert "HANDOFF_VISIBLE_SKILL_LIMIT=${HANDOFF_VISIBLE_SKILL_LIMIT:-5}" in script
    assert "HANDOFF_PROMPT_STYLE=${HANDOFF_PROMPT_STYLE:-legacy_hints}" in script
    assert "--stability_kl_weight" in script
    assert "--learned_component_min_range" in script
    assert "--learned_component_trust_top_k" in script
    assert "--handoff_visible_skill_limit" in script
    assert "CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt}" in script
    assert "APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-true}" in script


def test_appworld_current_route_preference_train_sbatch_defaults_to_aligned_stage4_loss():
    script = Path("scripts/sbatch/run_appworld_current_route_preference_train.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_appworld_current_route_preference_train.py" in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-true}" in script
    assert "LOSS_SCORE_MODE=${LOSS_SCORE_MODE:-policy_transition_blend}" in script
    assert "LEARNED_COMPONENT_MIN_RANGE=${LEARNED_COMPONENT_MIN_RANGE:-0.1}" in script
    assert "LEARNED_COMPONENT_TRUST_TOP_K=${LEARNED_COMPONENT_TRUST_TOP_K:-80}" in script
    assert "--loss_score_mode" in script
    assert "--learned_component_min_range" in script
    assert "--learned_component_trust_top_k" in script


def test_alfworld_qwen_clstr_executor_gate_sbatch_is_low_cost_and_uses_local_model():
    script = Path("scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "MAX_EPISODES=${MAX_EPISODES:-2}" in script
    assert "MAX_STEPS=${MAX_STEPS:-20}" in script
    assert "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}" in script
    assert "QWEN_SCORE_MODE=${QWEN_SCORE_MODE:-auto}" in script
    assert "LOOP_GUARD=${LOOP_GUARD:-0}" in script
    assert "QWEN_MODEL_NAME_OR_PATH=${QWEN_MODEL_NAME_OR_PATH:-models/Qwen3-14B}" in script
    assert "QWEN_CONTROL_IMPORT_MANIFEST=${QWEN_CONTROL_IMPORT_MANIFEST:-}" in script
    assert 'if [[ "${RUN_QWEN_ONLY}" = "0" ]]' in script
    assert "MEMORY_UTILITY_GATE_CHECKPOINT_PATH" in script
    assert "MEMORY_UTILITY_GATE_CHECKPOINT_SHA256" in script
    assert "MEMORY_UTILITY_GATE_AUDIT_SHA256" in script
    assert "scripts/run_alfworld_qwen_clstr_executor_gate.py" in script
    assert "--run_qwen_only" in script
    assert "--run_hybrid" in script
    assert "--qwen_score_mode" in script
    assert "--loop_guard" in script
    assert "--qwen_control_import_manifest" in script
    assert "--memory_utility_gate_checkpoint_path" in script
    assert "--expected_memory_utility_gate_checkpoint_sha256" in script
    assert "--expected_memory_utility_gate_audit_sha256" in script


def test_appworld_routing_diagnostics_sbatch_supports_executor_failure_topk():
    script = Path("scripts/sbatch/run_appworld_routing_diagnostics.sh").read_text(encoding="utf-8")

    assert 'COMMAND}" = "executor-failure-topk' in script
    assert "RUN_ARGS" in script
    assert "PREDICTION_ARGS" in script
    assert "--focus_method" in script
    assert "--reference_method" in script


def test_appworld_routing_comparison_sbatch_wraps_comparison_script():
    script = Path("scripts/sbatch/run_appworld_routing_comparison.sh").read_text(encoding="utf-8")

    assert "scripts/build_appworld_routing_comparison.py" in script
    assert "--reports" in script
    assert "--output_dir" in script


def test_appworld_multistep_executor_sbatch_exposes_controller_settings():
    script = Path("scripts/sbatch/run_appworld_multistep_executor_eval.sh").read_text(encoding="utf-8")

    assert ".deps/appworld_py310" in script
    assert "scripts/run_appworld_multistep_executor_eval.py" in script
    assert "MAX_STEPS" in script
    assert "--max_steps" in script
    assert "SKILL_CONTEXT_MODE" in script
    assert "--skill_context_mode" in script
    assert "CLSTR_MODEL_CONFIG" in script
    assert "--clstr_model_config" in script
    assert "CLSTR_CHECKPOINT_PATH" in script
    assert "--clstr_checkpoint_path" in script
    assert "RANKING_MODE" in script
    assert "--ranking_mode" in script
    assert "CANDIDATE_TOP_K" in script
    assert "--candidate_top_k" in script
    assert "CANDIDATE_SOURCE" in script
    assert "--candidate_source" in script
    assert "POLICY_BLEND_ALPHA" in script
    assert "--policy_blend_alpha" in script
    assert "ALLOW_LEGACY_POLICY_SKILL_ROUTER" in script
    assert "--allow_legacy_policy_skill_router" in script
    assert "CLSTR_ALPHA" in script
    assert "--clstr_alpha" in script
    assert "RECURRENT_BELIEF" in script
    assert "--no-recurrent_belief" in script
    assert "SKILLROUTER_MODEL_NAME_OR_PATH" in script
    assert "--skillrouter_model_name_or_path" in script
    assert "SKILLROUTER_CHECKPOINT_PATH" in script
    assert "--skillrouter_checkpoint_path" in script
    assert "SKILLROUTER_DEVICE" in script
    assert "--skillrouter_device" in script
    assert "APPWORLD_EXECUTOR_COMPATIBLE_ONLY" in script
    assert "--appworld_executor_compatible_only" in script


def test_appworld_current_route_sbatch_wrappers_pin_data_checkpoints_and_executor_filter():
    stage0 = Path("scripts/sbatch/run_clstr_appworld_current_stage0_biencoder_train.sh").read_text(encoding="utf-8")
    stage1 = Path("scripts/sbatch/run_clstr_appworld_current_stage1_heads_init.sh").read_text(encoding="utf-8")
    stage2 = Path("scripts/sbatch/run_clstr_appworld_current_stage2_full_base_train.sh").read_text(encoding="utf-8")
    stage4 = Path("scripts/sbatch/run_clstr_appworld_current_stage4_act_train.sh").read_text(encoding="utf-8")
    executor = Path("scripts/sbatch/run_appworld_current_route_multistep_executor_eval.sh").read_text(encoding="utf-8")

    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_appworld_current_route_v1}" in stage0
    assert "outputs/clstr_appworld_current_stage0_biencoder" in stage0
    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in stage0
    _assert_sources_shared_clstr_gpu_env(stage0)

    assert "TRAIN_PATH=${TRAIN_PATH:-data/clstr_appworld_current_route_v1/trajectories.jsonl}" in stage1
    assert "SKILLS_PATH=${SKILLS_PATH:-data/clstr_appworld_current_route_v1/skill_pool.jsonl}" in stage1
    assert "ROUTING_CHECKPOINT_PATH=${ROUTING_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step2000.pt}" in stage1
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-appworld}" in stage1
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-appworld=-1}" in stage1
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh" in stage1
    _assert_sources_shared_clstr_gpu_env(stage1)

    assert "STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage1_heads_init/checkpoints/clstr_stage1_heads-step2000.pt}" in stage2
    assert "APPWORLD_STAGE1_HANDOFF_MANIFEST=${APPWORLD_STAGE1_HANDOFF_MANIFEST:-outputs/clstr_appworld_current_stage1_heads_init/stage0_candidate_handoff.json}" in stage2
    assert "STAGE0_QUALITY_GATE_MODE=handoff_gate" in stage2
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-appworld}" in stage2
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in stage2
    _assert_sources_shared_clstr_gpu_env(stage2)

    assert "HEAD_CHECKPOINT_PATH=${HEAD_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage2_full_base/checkpoints/clstr_full_base-step5000.pt}" in stage4
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-appworld}" in stage4
    assert "scripts/sbatch/run_clstr_unified_stage4_act_train.sh" in stage4
    _assert_sources_shared_clstr_gpu_env(stage4)

    assert "METHOD=${METHOD:-clstr_multistep}" in executor
    assert "SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_current_route_v1/skill_pool.jsonl}" in executor
    assert "CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_appworld_current_stage4_act/checkpoints/clstr_stage4_act-step1000.pt}" in executor
    assert "APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}" in executor
    assert "scripts/sbatch/run_appworld_multistep_executor_eval.sh" in executor
    _assert_sources_shared_clstr_gpu_env(executor)


def test_appworld_mt_fusion_train_sbatch_uses_oracle_trajectories_and_checkpoint():
    script = Path("scripts/sbatch/run_appworld_mt_fusion_train.sh").read_text(encoding="utf-8")

    assert "scripts/run_appworld_mt_fusion_train.py" in script
    assert "TRAJECTORIES_PATH" in script
    assert "--trajectories_path" in script
    assert "BASE_CHECKPOINT_PATH" in script
    assert "--checkpoint_path" in script
    assert "MODEL_CONFIG" in script
    assert "--model_config" in script
    assert "CANDIDATE_TOP_K" in script
    assert "--candidate_top_k" in script
    assert "TRAJECTORY_BATCH_SIZE" in script
    assert "--trajectory_batch_size" in script


def test_appworld_skillx_leakage_audit_sbatch_runs_leakage_mode():
    script = Path("scripts/sbatch/run_appworld_skillx_leakage_audit.sh").read_text(encoding="utf-8")

    assert "scripts/audit_skillx_appworld.py" in script
    assert "--leakage_audit" in script
    assert "SKILL_POOL_PATH" in script
    assert "--skill_pool_path" in script
    assert "ROUTING_DIR" in script
    assert "--routing_dir" in script
    assert "OUTPUT_DIR" in script
    assert "--output_dir" in script


def test_unified_stage1_retrieval_warmup_sbatch_is_stage0_compatibility_wrapper():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh").read_text(encoding="utf-8")

    assert "deprecated" in script.lower()
    assert "Stage0" in script
    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    assert "scripts/run_skillret_retrieval_warmup.py" not in script
    assert V4_2_STAGE0_OUTPUT_DIR in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_2_nowweak_stage0_biencoder_sbatch_pins_current_mainline_data_and_outputs():
    script = Path("scripts/sbatch/run_clstr_unified_stage0_biencoder_train_v4_2_nowweak.sh").read_text(encoding="utf-8")

    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    assert f"DATA_ROOT=${{DATA_ROOT:-{V4_2_NOWEAK_DATA_ROOT}}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_STAGE0_OUTPUT_DIR}}}" in script
    assert "TOP_K=${TOP_K:-350}" in script
    assert "RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}" in script
    assert "EXPAND_ALIAS_POSITIVES=${EXPAND_ALIAS_POSITIVES:-1}" in script
    assert "TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}" in script
    assert "TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-0}" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_function_aug_v2_stage0_clean_calibration_sbatch_trains_w_scale_and_bias_only():
    script = Path("scripts/sbatch/run_clstr_unified_stage0_function_aug_v2_clean_calibration.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}" in script
    assert (
        "OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_function_aug_v2_clean_calibration_smoke}"
        in script
    )
    assert "TOP_K=${TOP_K:-350}" in script
    assert "MAX_STEPS=${MAX_STEPS:-1200}" in script
    assert "LEARNING_RATE=${LEARNING_RATE:-2.0e-5}" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-0}" in script
    assert "TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}" in script
    assert "TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}" in script
    assert "TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}" in script
    assert "INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-}" in script
    assert "INIT_CHECKPOINT_SKILLS_PATH=${INIT_CHECKPOINT_SKILLS_PATH:-}" in script
    assert "INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-outputs/" not in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_function_aug_v2_stage0_toolbench_clean_full_sbatch_runs_preflight_and_safe_acceleration():
    script = Path("scripts/sbatch/run_clstr_unified_stage0_function_aug_v2_toolbench_clean_full.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/audit_clean_training_preflight.py" in script
    assert f"DATA_ROOT=${{DATA_ROOT:-{FUNCTION_AUG_V2_TOOLBENCH_CLEAN_DATA_ROOT}}}" in script
    assert "MAX_STEPS=${MAX_STEPS:-5000}" in script
    assert "TOP_K=${TOP_K:-350}" in script
    assert "SKILL_TABLE_BATCH_SIZE=${SKILL_TABLE_BATCH_SIZE:-64}" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-1}" in script
    assert "TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}" in script
    assert "TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}" in script
    assert "TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}" in script
    assert "exec bash scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_function_aug_v2_stage0_true_adapt_proj_smoke_sbatch_trains_projection_and_calibration():
    script = Path("scripts/sbatch/run_clstr_unified_stage0_function_aug_v2_true_adapt_proj_smoke1200.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}" in script
    assert (
        "OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_smoke1200}"
        in script
    )
    assert "INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-}" in script
    assert "INIT_CHECKPOINT_SKILLS_PATH=${INIT_CHECKPOINT_SKILLS_PATH:-}" in script
    assert "MAX_STEPS=${MAX_STEPS:-1200}" in script
    assert "LEARNING_RATE=${LEARNING_RATE:-1.0e-5}" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-1}" in script
    assert "TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}" in script
    assert "TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-handoff_balanced}" in script
    assert "RETRIEVAL_LOSS_MODE=${RETRIEVAL_LOSS_MODE:-multi_positive_nll}" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_function_aug_v2_stage0_true_adapt_proj_continue_sbatch_resumes_smoke_checkpoint():
    script = Path("scripts/sbatch/run_clstr_unified_stage0_function_aug_v2_true_adapt_proj_continue3800.sh").read_text(
        encoding="utf-8"
    )

    assert "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}" in script
    assert (
        "OUTPUT_DIR=${OUTPUT_DIR:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_full5000}"
        in script
    )
    assert "MAX_STEPS=${MAX_STEPS:-3800}" in script
    assert (
        "INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_function_aug_v2_true_adapt_proj_smoke1200/checkpoints/clstr_unified_retrieval_v2-step1200.pt}"
        in script
    )
    assert (
        "INIT_CHECKPOINT_SKILLS_PATH=${INIT_CHECKPOINT_SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2/skill_pool.jsonl}"
        in script
    )
    assert "LEARNING_RATE=${LEARNING_RATE:-1.0e-5}" in script
    assert "TRAIN_SKILL_EMBEDDINGS=${TRAIN_SKILL_EMBEDDINGS:-0}" in script
    assert "TRAIN_SKILL_BIAS=${TRAIN_SKILL_BIAS:-1}" in script
    assert "TRAIN_ENCODER_PROJECTION=${TRAIN_ENCODER_PROJECTION:-1}" in script
    assert "TRAIN_SKILL_ADAPTER=${TRAIN_SKILL_ADAPTER:-1}" in script
    assert "TRAIN_RETRIEVAL_SCALE=${TRAIN_RETRIEVAL_SCALE:-1}" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_2_nowweak_stage0_baseline_and_eval_sbatch_share_same_pool_paths():
    baseline = Path("scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak.sh").read_text(encoding="utf-8")
    eval_script = Path("scripts/sbatch/run_clstr_stage0_full_retrieval_eval_v4_2_nowweak.sh").read_text(encoding="utf-8")

    assert f"DATA_ROOT=${{DATA_ROOT:-{V4_2_NOWEAK_DATA_ROOT}}}" in baseline
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak" in baseline
    assert "TOP_K=${TOP_K:-350}" in baseline
    assert "scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh" in baseline
    _assert_sources_shared_clstr_gpu_env(baseline)

    assert f"DATA_ROOT=${{DATA_ROOT:-{V4_2_NOWEAK_DATA_ROOT}}}" in eval_script
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/qrels.jsonl" in eval_script
    assert f"CHECKPOINT_PATH=${{CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in eval_script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_STAGE0_OUTPUT_DIR}/full_retrieval_eval}}" in eval_script
    assert "TOP_K=${TOP_K:-350}" in eval_script
    assert "RUN_NAME=${RUN_NAME:-clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE}" in eval_script
    assert "scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh" in eval_script
    _assert_sources_shared_clstr_gpu_env(eval_script)


def test_v4_2_nowweak_stage1_sbatch_uses_rebalanced_loss_and_top350_handoff():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh").read_text(encoding="utf-8")

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_2_NOWEAK_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_NOWEAK_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_STAGE1_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.4}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.6}" in script
    assert "STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.1}" in script
    assert "BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}" in script
    assert "RETRIEVAL_LOSS_WEIGHT=${RETRIEVAL_LOSS_WEIGHT:-0.0}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}" in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_2_nowweak_stage2_sbatch_uses_inventory_listwise_and_stage1_checkpoint():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh").read_text(encoding="utf-8")

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_2_NOWEAK_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_NOWEAK_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_STAGE2_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert f"STAGE0_OUTPUT_DIR=${{STAGE0_OUTPUT_DIR:-{V4_2_STAGE0_OUTPUT_DIR}}}" in script
    assert f"STAGE0_EVAL_METRICS_PATH=${{STAGE0_EVAL_METRICS_PATH:-{V4_2_STAGE0_EVAL_METRICS}}}" in script
    assert f"STAGE1_CHECKPOINT_PATH=${{STAGE1_CHECKPOINT_PATH:-{V4_2_STAGE1_CHECKPOINT}}}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.2}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-1.0}" in script
    assert "POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}" in script
    assert "Q_SUCCESS_LOSS_WEIGHT=${Q_SUCCESS_LOSS_WEIGHT:-0.0}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_main_gpu_training_sbatch_does_not_override_cluster_gpu_cpu_bundle():
    scripts = [
        "scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh",
        "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh",
        "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh",
        "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh",
        "scripts/sbatch/run_clstr_unified_stage4_act_train.sh",
    ]
    disallowed = (
        "#SBATCH --cpus-per-task",
        "#SBATCH -c",
        "--cpus-per-gpu",
        "--cores-per-socket",
        "--ntasks-per-socket",
        "--gpus-per-socket",
    )

    for script_path in scripts:
        script = Path(script_path).read_text(encoding="utf-8")
        for token in disallowed:
            assert token not in script, f"{script_path} must not set {token}"


def test_main_gpu_training_sbatch_uses_existing_default_conda_env_path():
    scripts = [
        "scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh",
        "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh",
        "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh",
        "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh",
        "scripts/sbatch/run_clstr_unified_stage4_act_train.sh",
    ]

    env_script = Path("scripts/sbatch/_clstr_gpu_env.sh").read_text(encoding="utf-8")
    assert "CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/scyb713/run/miniconda3/envs/xzf}" in env_script
    assert 'source activate "${CONDA_ENV_PATH}"' in env_script
    assert "source activate env" not in env_script
    assert "/data/home/scyb713/run/miniconda3/envs/xzf/bin/python" in env_script

    for script_path in scripts:
        script = Path(script_path).read_text(encoding="utf-8")
        _assert_sources_shared_clstr_gpu_env(script)
        assert "source activate env" not in script, script_path


def test_main_gpu_training_sbatch_sources_shared_clstr_environment():
    scripts = [
        "scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh",
        "scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh",
        "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh",
        "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh",
        "scripts/sbatch/run_clstr_unified_stage4_act_train.sh",
    ]

    env_script = Path("scripts/sbatch/_clstr_gpu_env.sh")
    assert env_script.is_file()
    env_text = env_script.read_text(encoding="utf-8")
    assert "CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/scyb713/run/miniconda3/envs/xzf}" in env_text
    assert 'source activate "${CONDA_ENV_PATH}"' in env_text
    assert "HF_ENDPOINT=https://hf-mirror.com" in env_text
    assert "PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}" in env_text
    assert 'cd "${PROJECT_ROOT}"' in env_text

    for script_path in scripts:
        script = Path(script_path).read_text(encoding="utf-8")
        _assert_sources_shared_clstr_gpu_env(script)


def test_stage0_handoff_coverage_audit_sbatch_supports_all_rows_cap():
    script = Path("scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh").read_text(encoding="utf-8")

    assert "all|ALL|none|NONE|null|NULL|0" in script
    assert 'MAX_ROWS=""' in script
    assert 'ARGS+=(--max_rows "${MAX_ROWS}")' in script


def test_stage0_traject_handoff_row_audit_sbatch_runs_low_cost_row_diagnostics():
    script = Path("scripts/sbatch/run_clstr_stage0_traject_handoff_row_audit.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=00:30:00" in script
    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/audit_clstr_stage0_handoff_rows.py" in script
    assert "BENCHMARK=${BENCHMARK:-traject_bench}" in script
    assert "MAX_ROWS=${MAX_ROWS:-960}" in script
    assert "TOP_K=${TOP_K:-10}" in script
    assert "--query_mode" in script
    assert "--model_cache_dir" in script
    assert 'tee "${OUTPUT_DIR}/audit_stdout.json"' in script


def test_unified_stage2_full_base_sbatch_defaults_to_action_aware_progressive_mainline():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh").read_text(encoding="utf-8")

    assert "scripts/run_clstr_stage2_full_base_train.py" in script
    assert "scripts/run_clstr_qwen3_full_base_train.py" not in script
    assert "ROUTING_INIT_MANIFEST" not in script
    assert "--routing_init_manifest" not in script
    assert "outputs/clstr_native_routing_init/manifest.json" not in script
    assert "TRAIN_PATH" in script
    assert f"{V4_2_PROGRESSIVE_DATA_ROOT}/trajectories.jsonl" in script
    assert "SKILLS_PATH" in script
    assert f"{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl" in script
    assert "QWEN_MODEL_PATH" not in script
    assert "models/Qwen3-8B" not in script
    assert "--local_files_only" not in script
    assert "ROUTING_CHECKPOINT_PATH" in script
    assert V4_2_STAGE0_CHECKPOINT in script
    assert "STAGE1_CHECKPOINT_PATH" in script
    assert V4_2_PROGRESSIVE_STAGE2_STAGE1_CHECKPOINT in script
    assert "[[ -s \"${STAGE1_CHECKPOINT_PATH}\" ]]" in script
    assert "Stage2 requires completed Stage1 heads checkpoint" in script
    assert "STAGE0_TOP_M" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "--stage0_top_m" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY" in script
    assert "--stage0_positive_missing_policy" in script
    assert "STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-4}" in script
    assert "--stage0_handoff_sample_multiplier" in script
    assert "STAGE0_CANDIDATE_ENCODE_BATCH_SIZE" in script
    assert "--stage0_candidate_encode_batch_size" in script
    assert "STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES" in script
    assert "--stage0_candidate_progress_interval_batches" in script
    assert "STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-auto}" in script
    assert "STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}" in script
    assert "--stage0_handoff_cache_mode" in script
    assert "--stage0_handoff_cache_dir" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "--transition_scoring_mode" in script
    assert "ALLOW_FULL_POOL_STAGE2_DEBUG" in script
    assert "--allow_full_pool_stage2_debug" in script
    assert "scripts/audit_clstr_stage0_quality.py" in script
    assert "stage0_quality_gate.json" in script
    assert "STAGE0_EVAL_METRICS_PATH" in script
    assert V4_2_STAGE0_EVAL_METRICS in script
    assert "--stage0_eval_metrics_path" in script
    assert "STAGE0_BASELINE_METRICS_PATH" in script
    assert "--baseline_metrics_path" in script
    assert "stdout.log" in script
    assert "--fail_on_action_required" in script
    assert "scripts/run_clstr_stage2_preflight.py" in script
    assert "--checkpoint_path" in script
    assert "--skills_path" in script
    assert "--model_dim" in script
    assert "EMBEDDING_CACHE_MODE" in script
    assert "EMBEDDING_CACHE_MAX_ROWS" in script
    assert "--embedding_cache_mode" in script
    assert "--embedding_cache_max_rows" in script
    assert "MAX_ROWS=${MAX_ROWS:-}" in script
    assert "--max_rows" in script
    assert "ALLOWED_BENCHMARKS" in script
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}" in script
    assert "--allowed_benchmarks" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1}" in script
    assert "--benchmark_caps" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}" in script
    assert "--sampling_strategy" in script
    assert "POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${POLICY_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}" in script
    assert "--policy_hard_negative_margin_loss_weight" in script
    assert "Q_SUCCESS_LOSS_WEIGHT=${Q_SUCCESS_LOSS_WEIGHT:-0.0}" in script
    assert "--q_success_loss_weight" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "--resume_checkpoint_path" in script
    assert "STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}" in script
    assert "STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}" in script
    assert "--stage0_handoff_cache_format" in script
    assert "--stage0_handoff_cache_shard_size" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-explicit_only}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}" in script
    assert "NEXT_SKILL_POOL_MODE=${NEXT_SKILL_POOL_MODE:-full_pool}" in script
    assert "COUNTERFACTUAL_UTILITY_LOSS_WEIGHT=${COUNTERFACTUAL_UTILITY_LOSS_WEIGHT:-0.05}" in script
    assert "COUNTERFACTUAL_GAIN_MARGIN=${COUNTERFACTUAL_GAIN_MARGIN:-0.1}" in script
    assert "COUNTERFACTUAL_SAFETY_TOLERANCE=${COUNTERFACTUAL_SAFETY_TOLERANCE:-0.01}" in script
    assert "COUNTERFACTUAL_GAIN_WEIGHT=${COUNTERFACTUAL_GAIN_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_SAFETY_WEIGHT=${COUNTERFACTUAL_SAFETY_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_WARMUP_FRACTION=${COUNTERFACTUAL_WARMUP_FRACTION:-0.1}" in script
    assert "--next_skill_pool_mode" in script
    assert "--counterfactual_utility_loss_weight" in script
    assert "--counterfactual_gain_margin" in script
    assert "--counterfactual_safety_tolerance" in script
    assert "--counterfactual_gain_weight" in script
    assert "--counterfactual_safety_weight" in script
    assert "--counterfactual_warmup_fraction" in script
    assert "AUTO_REPLAY_PREFIX_MAX_STEPS=${AUTO_REPLAY_PREFIX_MAX_STEPS:-auto}" in script
    assert "TRAINABLE_REPLAY_PREFIX=${TRAINABLE_REPLAY_PREFIX:-auto}" in script
    assert 'if [[ "${ROUTE_SCORER}" == "unified_memory" ]]' in script
    assert "AUTO_REPLAY_PREFIX_MAX_STEPS=3" in script
    assert "TRAINABLE_REPLAY_PREFIX=1" in script
    assert "--auto_replay_prefix_max_steps" in script
    assert "--trainable_replay_prefix" in script
    assert "--transition_inventory_min_candidates" in script
    assert "Stage 2 defaults to the current action-aware progressive-final mainline." in script
    assert "--routing_checkpoint_path" in script
    assert "--stage1_checkpoint_path" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_transition_lambda_sweep_sbatch_can_use_balanced_handoff_subset():
    script = Path("scripts/sbatch/run_clstr_transition_lambda_sweep.sh").read_text(encoding="utf-8")

    assert "STAGE0_HANDOFF_SAMPLE_MULTIPLIER=${STAGE0_HANDOFF_SAMPLE_MULTIPLIER:-}" in script
    assert "--stage0_handoff_sample_multiplier" in script


def test_submit_stage2_after_stage0_gate_is_deprecated_and_routes_to_stage1():
    script = Path("scripts/submit_clstr_stage2_after_stage0_gate.sh").read_text(encoding="utf-8")

    assert "deprecated" in script.lower()
    assert "scripts/submit_clstr_stage1_after_stage0_gate.sh" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" not in script
    assert "sbatch --gpus" not in script


def test_submit_stage1_after_stage0_gate_checks_stage0_gate_and_only_submits_stage1():
    script = Path("scripts/submit_clstr_stage1_after_stage0_gate.sh").read_text(encoding="utf-8")

    assert "MAX_ACTIVE_JOBS=${MAX_ACTIVE_JOBS:-4}" in script
    assert "bash scripts/guard_clstr_job_budget.sh" in script
    assert script.index("bash scripts/guard_clstr_job_budget.sh") < script.index("sbatch --gpus")
    assert "STAGE0_CHECKPOINT" in script
    assert "STAGE0_OUTPUT_DIR" in script
    assert V4_2_STAGE0_OUTPUT_DIR in script
    assert (
        "STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-"
        "${STAGE0_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt}"
    ) in script
    assert "[[ -s \"${STAGE0_CHECKPOINT}\" ]]" in script
    assert "scripts/audit_clstr_stage0_quality.py" in script
    assert "STAGE0_EVAL_METRICS_PATH" in script
    assert (
        "STAGE0_EVAL_METRICS_PATH=${STAGE0_EVAL_METRICS_PATH:-"
        "${STAGE0_OUTPUT_DIR}/full_retrieval_eval/metrics.json}"
    ) in script
    assert "--stage0_eval_metrics_path" in script
    assert "--fail_on_action_required" in script
    assert "scripts/run_clstr_stage2_preflight.py" in script
    assert "stage0_quality_gate.json" in script
    assert "stage1_preflight.json" in script
    assert "sbatch --gpus=\"${GPUS}\" -p \"${PARTITION}\"" in script
    assert V4_2_PROGRESSIVE_DATA_ROOT in script
    assert V4_2_PROGRESSIVE_STAGE1_OUTPUT_DIR in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh" in script
    assert V4_2_STAGE1_OUTPUT_DIR not in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh" not in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" not in script
    assert "run_clstr_unified_stage4_act_train.sh" not in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_unified_stage1_and_stage2_caps_support_colon_separated_sbatch_exports():
    for path in (
        "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh",
        "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh",
    ):
        script = Path(path).read_text(encoding="utf-8")

        assert 'BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"' in script
        assert "Use colon-separated BENCHMARK_CAPS when exporting through sbatch" in script


def test_stage0_full_retrieval_eval_sbatch_exports_same_pool_eval_metrics_for_gate():
    script = Path("scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=04:00:00" in script
    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/export_clstr_retrieval_run.py" in script
    assert "scripts/evaluate_retrieval_run.py" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final}" in script
    assert "QUERIES_PATH=${QUERIES_PATH:-${DATA_ROOT}/retrieval.jsonl}" in script
    assert "SKILLS_PATH=${SKILLS_PATH:-${DATA_ROOT}/skill_pool.jsonl}" in script
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/qrels.jsonl" in script
    assert f"{CANONICAL_STAGE0_OUTPUT_DIR}/full_retrieval_eval" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert 'rm -f "${OUTPUT_DIR}/metrics.json"' in script
    assert 'rm -f "${OUTPUT_DIR}/eval_stdout.json"' in script
    assert 'rm -f "${OUTPUT_DIR}/run.tsv"' in script
    assert 'rm -f "${OUTPUT_DIR}/predictions.jsonl"' in script
    assert 'rm -f "${OUTPUT_DIR}/progress.json"' in script
    assert "--run_format trec" in script
    assert "--k_values 20 50 100" in script
    assert "--disable_cross_encoder" in script
    assert "--query_text_format skillrouter" in script
    assert "--local_files_only" in script


def test_stage0_handoff_coverage_audit_sbatch_runs_on_compute_node_with_small_default():
    script = Path("scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=00:30:00" in script
    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/audit_clstr_stage0_handoff_coverage.py" in script
    assert "CHECKPOINT_PATH" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert "TRAIN_PATH" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl" in script
    assert "SKILLS_PATH" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl" in script
    assert "OUTPUT_PATH" in script
    assert "outputs/clstr_stage0_handoff_coverage_audit/v3_smoke/report.json" in script
    assert "MAX_ROWS=${MAX_ROWS:-128}" in script
    assert "--max_rows" in script
    assert "BATCH_SIZE=${BATCH_SIZE:-8}" in script
    assert "--batch_size" in script
    assert "TOP_K_VALUES=${TOP_K_VALUES:-20,50,100,200,500}" in script
    assert 'TOP_K_VALUES="${TOP_K_VALUES//:/,}"' in script
    assert "REQUIRE_MULTI_TOP_K=${REQUIRE_MULTI_TOP_K:-1}" in script
    assert "Use colon-separated TOP_K_VALUES when exporting through sbatch" in script
    assert "--top_k_values" in script
    assert "QUERY_MODES=${QUERY_MODES:-skillrouter_state}" in script
    assert 'QUERY_MODES="${QUERY_MODES//:/,}"' in script
    assert "--query_modes" in script
    assert "MODEL_CACHE_DIR" in script
    assert "--model_cache_dir" in script
    assert "MIN_GLOBAL_NEXT_RECALL_AT_500=${MIN_GLOBAL_NEXT_RECALL_AT_500:-0.90}" in script
    assert "MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500=${MIN_TOOLBENCH_G3_NEXT_RECALL_AT_500:-0.90}" in script
    assert "MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_500:-0.88}" in script
    assert "MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200=${MIN_TRAJECT_BENCH_NEXT_RECALL_AT_200:-0.75}" in script
    assert "handoff coverage gate blockers" in script
    assert "[[ -s \"${CHECKPOINT_PATH}\" ]]" in script
    assert "Stage0 handoff audit requires completed checkpoint" in script
    assert 'tee "${OUTPUT_DIR}/handoff_audit_stdout.json"' in script


def test_submit_stage2_after_stage1_gate_checks_stage1_gate_and_only_submits_stage2():
    script = Path("scripts/submit_clstr_stage2_after_stage1_gate.sh").read_text(encoding="utf-8")

    assert "MAX_ACTIVE_JOBS=${MAX_ACTIVE_JOBS:-4}" in script
    assert "bash scripts/guard_clstr_job_budget.sh" in script
    assert script.index("bash scripts/guard_clstr_job_budget.sh") < script.index("sbatch --gpus")
    assert "STAGE1_CHECKPOINT" in script
    assert "STAGE1_OUTPUT_DIR" in script
    assert V4_2_STAGE0_OUTPUT_DIR in script
    assert V4_2_PROGRESSIVE_STAGE2_STAGE1_OUTPUT_DIR in script
    assert V4_2_PROGRESSIVE_STAGE2_STAGE1_CHECKPOINT in script
    assert V4_2_PROGRESSIVE_STAGE2_RANKPRIOR_OUTPUT_DIR in script
    assert V4_2_PROGRESSIVE_DATA_ROOT in script
    assert V4_2_STAGE1_OUTPUT_DIR not in script
    assert V4_2_STAGE1_CHECKPOINT not in script
    assert "[[ -s \"${STAGE1_CHECKPOINT}\" ]]" in script
    assert "scripts/audit_clstr_stage1_heads_quality.py" in script
    assert "--fail_on_action_required" in script
    assert "stage1_quality_gate.json" in script
    assert "sbatch --gpus=\"${GPUS}\" -p \"${PARTITION}\"" in script
    assert "STAGE1_CHECKPOINT_PATH=\"${STAGE1_CHECKPOINT}\"" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_progressive_final.sh" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh" not in script
    assert "run_clstr_unified_stage4_act_train.sh" not in script


def test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_heads_init.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=06:00:00" in script
    assert "scripts/run_clstr_stage1_heads_init.py" in script
    assert "ROUTING_CHECKPOINT_PATH" in script
    assert V4_2_STAGE0_CHECKPOINT in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "--stage0_top_m" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY" in script
    assert "--stage0_positive_missing_policy" in script
    assert "MAX_STEPS=${MAX_STEPS:-3000}" in script
    assert "MAX_ROWS=${MAX_ROWS:-}" in script
    assert "--max_rows" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1,traject_bench=-1,alfworld=-1,webshop=-1}" in script
    assert "--benchmark_caps" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}" in script
    assert "--sampling_strategy" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN=${TRANSITION_HARD_NEGATIVE_MARGIN:-1.0}" in script
    assert "ROUTE_SCORER=${ROUTE_SCORER:-legacy_prior_residual}" in script
    assert "RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-}" in script
    assert "--transition_inventory_mask_mode" in script
    assert "--transition_inventory_min_candidates" in script
    assert "--transition_loss_type" in script
    assert "--transition_positive_mode" in script
    assert "--transition_scoring_mode" in script
    assert "--transition_hard_negative_margin_loss_weight" in script
    assert "--transition_hard_negative_margin" in script
    assert "--route_scorer" in script
    assert "--resume_checkpoint_path" in script
    assert "STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-auto}" in script
    assert "STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}" in script
    assert "STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}" in script
    assert "STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}" in script
    assert "--stage0_handoff_cache_mode" in script
    assert "--stage0_handoff_cache_dir" in script
    assert "--stage0_handoff_cache_format" in script
    assert "--stage0_handoff_cache_shard_size" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_PROGRESSIVE_STAGE1_OUTPUT_DIR}}}" in script
    assert "QWEN_MODEL_PATH" not in script
    assert "ROUTING_INIT_MANIFEST" not in script
    assert "scripts/run_clstr_stage2_full_base_train.py" not in script
    assert "run_clstr_unified_stage4_act_train.sh" not in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_2_progressive_final_stage1_sbatch_uses_single_final_recipe():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_PROGRESSIVE_STAGE1_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.7}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.2}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.5}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN=${TRANSITION_HARD_NEGATIVE_MARGIN:-1.0}" in script
    assert "BELIEF_LOSS_WEIGHT=${BELIEF_LOSS_WEIGHT:-0.1}" in script
    assert "STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.1}" in script
    assert "RETRIEVAL_LOSS_WEIGHT=${RETRIEVAL_LOSS_WEIGHT:-0.0}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-balanced_random}" in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_unified_stage2_sbatch_can_gate_on_stage0_handoff_report_instead_of_full_retrieval_eval():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh").read_text(encoding="utf-8")

    assert "STAGE0_QUALITY_GATE_MODE=${STAGE0_QUALITY_GATE_MODE:-strict}" in script
    assert "STAGE0_HANDOFF_GATE_PATH=${STAGE0_HANDOFF_GATE_PATH:-}" in script
    assert "handoff_gate" in script
    assert "stage0_handoff_gate.json" in script
    assert "--fail_on_action_required" in script


def test_v4_2_progressive_final_stage2_sbatch_uses_progressive_stage1_and_handoff_gate():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_progressive_final.sh").read_text(
        encoding="utf-8"
    )

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_PROGRESSIVE_STAGE2_RANKPRIOR_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert f"STAGE1_CHECKPOINT_PATH=${{STAGE1_CHECKPOINT_PATH:-{V4_2_PROGRESSIVE_STAGE2_STAGE1_CHECKPOINT}}}" in script
    assert f"STAGE0_HANDOFF_GATE_PATH=${{STAGE0_HANDOFF_GATE_PATH:-{V4_2_PROGRESSIVE_STAGE0_HANDOFF_GATE}}}" in script
    assert "STAGE0_QUALITY_GATE_MODE=${STAGE0_QUALITY_GATE_MODE:-handoff_gate}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-auto}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-64}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.2}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.3}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-1.0}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=${TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT:-0.0}" in script
    assert "TRANSITION_HARD_NEGATIVE_MARGIN=${TRANSITION_HARD_NEGATIVE_MARGIN:-1.0}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1}" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_conservative_stage2_sbatch_locks_legacy_ce_nomask_recipe():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative.sh").read_text(
        encoding="utf-8"
    )

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert f"STAGE1_CHECKPOINT_PATH=${{STAGE1_CHECKPOINT_PATH:-{V4_1B_STAGE1_CHECKPOINT}}}" in script
    assert "STAGE0_QUALITY_GATE_MODE=handoff_gate" in script
    assert "current_positive_coverage_below_0.98" in script
    assert "next_positive_coverage_below_0.98" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.05}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.2}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-cross_entropy}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-single}" in script
    assert "TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_aggressive_stage1_sbatch_locks_top50_listwise_recipe():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_aggressive_top50_listwise.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-stage0_topk_trajectory_prior}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-50}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_aggressive_stage2_sbatch_consumes_aggressive_stage1_checkpoint():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_aggressive_top50_listwise.sh").read_text(
        encoding="utf-8"
    )

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert "clstr_unified_stage1_v4_1b_aggressive_top50_listwise_trajectory_prior_heads_init/checkpoints/clstr_stage1_heads-step3000.pt" in script
    assert "STAGE0_QUALITY_GATE_MODE=handoff_gate" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-stage0_topk_trajectory_prior}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-50}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_conservative_listwise_stage1_keeps_nomask_top350_recipe():
    script = Path("scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_conservative_listwise.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert "clstr_unified_stage1_v4_1b_conservative_top350_listwise_nomask_heads_init" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}" in script
    assert "TRANSITION_LOSS_WEIGHT=${TRANSITION_LOSS_WEIGHT:-0.05}" in script
    assert "TRANSITION_SKILL_CE_LOSS_WEIGHT=${TRANSITION_SKILL_CE_LOSS_WEIGHT:-0.2}" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "scripts/sbatch/run_clstr_unified_stage1_heads_init.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_conservative_listwise_stage2_consumes_matching_stage1_checkpoint():
    script = Path("scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative_listwise.sh").read_text(
        encoding="utf-8"
    )

    assert f"TRAIN_PATH=${{TRAIN_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert "clstr_unified_stage1_v4_1b_conservative_top350_listwise_nomask_heads_init/checkpoints/clstr_stage1_heads-step3000.pt" in script
    assert "STAGE0_QUALITY_GATE_MODE=handoff_gate" in script
    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}" in script
    assert "TRANSITION_INVENTORY_MIN_CANDIDATES=${TRANSITION_INVENTORY_MIN_CANDIDATES:-0}" in script
    assert "TRANSITION_LOSS_TYPE=${TRANSITION_LOSS_TYPE:-listwise_nll}" in script
    assert "TRANSITION_POSITIVE_MODE=${TRANSITION_POSITIVE_MODE:-gold_plus_equivalent}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_skill_dedup_borderline_audit_sbatch_uses_unified_skill_pool_without_rebuilding_data():
    script = Path("scripts/sbatch/run_skill_dedup_borderline_audit.sh").read_text(encoding="utf-8")

    assert "scripts/audit_skill_dedup_borderline.py" in script
    assert "SKILL_POOL_PATH" in script
    assert "data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl" in script
    assert "OUTPUT_PATH" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.jsonl" in script
    assert "REPORT_PATH" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_report.json" in script
    assert "REVIEW_OUTPUT_PATH" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.reviewed.jsonl" in script
    assert "REVIEW_REPORT_PATH" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json" in script
    assert "--review_output_path" in script
    assert "--review_report_path" in script
    assert "MAX_CANDIDATES" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script
    assert "build_clstr_unified_pretrain.py" not in script


def test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints():
    script = Path("scripts/sbatch/run_clstr_unified_stage4_act_train.sh").read_text(encoding="utf-8")

    assert "scripts/run_clstr_stage4_act_train.py" in script
    assert "stdout.log" in script
    assert "scripts/run_clstr_qwen3_stage4_act_train.py" not in script
    assert "TRAJECTORIES_PATH" in script
    assert f"{FUNCTION_AUG_V2_TOOLBENCH_CLEAN_DATA_ROOT}/trajectories.jsonl" in script
    assert "SKILLS_PATH" in script
    assert f"{FUNCTION_AUG_V2_TOOLBENCH_CLEAN_DATA_ROOT}/skill_pool.jsonl" in script
    assert "ROUTING_CHECKPOINT_PATH" in script
    assert "outputs/clstr_unified_memory_stage0_full_b128_20260707_052137/checkpoints/clstr_unified_retrieval_v2-step5000.pt" in script
    assert "HEAD_CHECKPOINT_PATH" in script
    assert "outputs/clstr_unified_memory_stage12_full_mmargin_retry_20260707_080738/stage2_full_base/checkpoints/clstr_full_base-step10000.pt" in script
    assert "outputs/clstr_unified_memory_stage4_full_pool_counterfactual" in script
    assert "[[ -s \"${ROUTING_CHECKPOINT_PATH}\" ]]" in script
    assert "[[ -s \"${HEAD_CHECKPOINT_PATH}\" ]]" in script
    assert "Stage4 requires completed ACT init checkpoint" in script
    assert "CANDIDATE_COUNT" in script
    assert "--candidate_count" in script
    assert "STAGE0_TOP_M" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-500}" in script
    assert "--stage0_top_m" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY" in script
    assert "--stage0_positive_missing_policy" in script
    assert "STAGE0_HANDOFF_QUERY_MODE" in script
    assert "--stage0_handoff_query_mode" in script
    assert "STAGE0_INVENTORY_MIN_CANDIDATES" in script
    assert "--stage0_inventory_min_candidates" in script
    assert "STAGE0_HANDOFF_CACHE_MODE=${STAGE0_HANDOFF_CACHE_MODE:-off}" in script
    assert "STAGE0_HANDOFF_CACHE_DIR=${STAGE0_HANDOFF_CACHE_DIR:-outputs/cache/stage0_handoff}" in script
    assert "STAGE0_HANDOFF_CACHE_FORMAT=${STAGE0_HANDOFF_CACHE_FORMAT:-legacy_jsonl}" in script
    assert "STAGE0_HANDOFF_CACHE_SHARD_SIZE=${STAGE0_HANDOFF_CACHE_SHARD_SIZE:-2048}" in script
    assert "--stage0_handoff_cache_format" in script
    assert "--stage0_handoff_cache_shard_size" in script
    assert "LAMBDA_PREF" not in script
    assert "--lambda_pref" not in script
    assert "BETA_KL" not in script
    assert "--beta_kl" not in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "--transition_scoring_mode" in script
    assert "NEXT_SKILL_POOL_MODE=${NEXT_SKILL_POOL_MODE:-full_pool}" in script
    assert "COUNTERFACTUAL_UTILITY_WEIGHT=${COUNTERFACTUAL_UTILITY_WEIGHT:-0.05}" in script
    assert "COUNTERFACTUAL_GAIN_MARGIN=${COUNTERFACTUAL_GAIN_MARGIN:-0.1}" in script
    assert "COUNTERFACTUAL_SAFETY_TOLERANCE=${COUNTERFACTUAL_SAFETY_TOLERANCE:-0.01}" in script
    assert "COUNTERFACTUAL_GAIN_WEIGHT=${COUNTERFACTUAL_GAIN_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_SAFETY_WEIGHT=${COUNTERFACTUAL_SAFETY_WEIGHT:-1.0}" in script
    assert "COUNTERFACTUAL_WARMUP_FRACTION=${COUNTERFACTUAL_WARMUP_FRACTION:-0.05}" in script
    assert "LEARNING_RATE=${LEARNING_RATE:-3.0e-5}" in script
    assert "MINIMUM_LEARNING_RATE=${MINIMUM_LEARNING_RATE:-3.0e-6}" in script
    assert "LEARNING_RATE_WARMUP_FRACTION=${LEARNING_RATE_WARMUP_FRACTION:-0.05}" in script
    assert "VALIDATION_INTERVAL_STEPS=${VALIDATION_INTERVAL_STEPS:-400}" in script
    assert "VALIDATION_ROWS_PER_BENCHMARK=${VALIDATION_ROWS_PER_BENCHMARK:-256}" in script
    assert "MINIMUM_VALIDATION_ROWS_PER_BENCHMARK=${MINIMUM_VALIDATION_ROWS_PER_BENCHMARK:-128}" in script
    assert "GATE_ROWS_PER_BENCHMARK=${GATE_ROWS_PER_BENCHMARK:-512}" in script
    assert "--next_skill_pool_mode" in script
    assert "--counterfactual_utility_weight" in script
    assert "--counterfactual_gain_margin" in script
    assert "--counterfactual_safety_tolerance" in script
    assert "--counterfactual_gain_weight" in script
    assert "--counterfactual_safety_weight" in script
    assert "--counterfactual_warmup_fraction" in script
    assert "--minimum_learning_rate" in script
    assert "--learning_rate_warmup_fraction" in script
    assert "--validation_rows_per_benchmark" in script
    assert "--minimum_validation_rows_per_benchmark" in script
    assert "--gate_rows_per_benchmark" in script
    assert "--validation_interval_steps" in script
    assert "--resume_checkpoint_path" in script
    assert "PREFERENCE_MAX_ROWS" not in script
    assert "--preference_max_rows" not in script
    assert "PREFERENCE_BATCH_SIZE" not in script
    assert "--preference_batch_size" not in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}" in script
    assert "ALLOWED_BENCHMARKS" in script
    assert "toolbench_g3,traject_bench,alfworld,webshop" in script
    assert "--allowed_benchmarks" in script
    assert "--routing_checkpoint_path" in script
    assert "--head_checkpoint_path" in script
    assert "ROUTING_INIT_MANIFEST" not in script
    assert "--routing_init_manifest" not in script
    assert "outputs/clstr_native_routing_init/manifest.json" not in script
    assert "QWEN_MODEL_PATH" not in script
    assert "--qwen_model_path" not in script
    assert "--local_files_only" not in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_qwen06_handoff_acceleration_gate_is_gpu_only_audit():
    script_path = Path("scripts/sbatch/run_qwen06_stage0_handoff_acceleration_gate.sh")

    assert script_path.is_file()
    script = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in script
    assert "scripts/audit_qwen06_stage0_handoff_acceleration.py" in script
    assert "CANDIDATE_BATCH_SIZES=${CANDIDATE_BATCH_SIZES:-16,64,128}" in script
    assert "ROW_COUNT=${ROW_COUNT:-2048}" in script
    assert "TRANSFORMERS_OFFLINE" in script
    assert "stage0_selection.json" in script
    assert "run_clstr_stage1_heads_init.py" not in script
    assert "run_clstr_stage2_full_base_train.py" not in script


def test_candidate_recall_eval_launchers_expose_static_dynamic_and_final_budgets():
    global_cli = Path("scripts/run_global_pool_clstr_route_eval.py").read_text(encoding="utf-8")
    toolbench_cli = Path("scripts/run_toolbench_g3_full_clstr_route_eval.py").read_text(encoding="utf-8")
    global_sbatch = Path("scripts/sbatch/run_global_pool_clstr_route_eval.sh").read_text(encoding="utf-8")
    toolbench_sbatch = Path("scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh").read_text(encoding="utf-8")

    for cli in (global_cli, toolbench_cli):
        assert "--static_k" in cli
        assert "--dynamic_extra_k" in cli
        assert "--final_k" in cli
        assert "dynamic_extra_k=args.dynamic_extra_k" in cli

    for script in (global_sbatch, toolbench_sbatch):
        assert "STATIC_K=" in script
        assert "DYNAMIC_EXTRA_K=" in script
        assert "FINAL_K=" in script
        assert '--static_k "${STATIC_K}"' in script
        assert '--dynamic_extra_k "${DYNAMIC_EXTRA_K}"' in script
        assert '--final_k "${FINAL_K}"' in script

    assert "TRANSITION_INVENTORY_MASK_MODE=${TRANSITION_INVENTORY_MASK_MODE:-off}" in toolbench_sbatch


def test_memory_utility_route_record_launchers_are_explicit_and_optional():
    global_cli = Path("scripts/run_global_pool_clstr_route_eval.py").read_text(encoding="utf-8")
    toolbench_cli = Path("scripts/run_toolbench_g3_full_clstr_route_eval.py").read_text(encoding="utf-8")
    global_sbatch = Path("scripts/sbatch/run_global_pool_clstr_route_eval.sh").read_text(encoding="utf-8")
    toolbench_sbatch = Path("scripts/sbatch/run_toolbench_g3_full_clstr_route_eval.sh").read_text(encoding="utf-8")

    for cli in (global_cli, toolbench_cli):
        assert "--route_records_path" in cli
        assert "--route_record_manifest_path" in cli
        assert "--route_record_model_digest" in cli
        assert "route_records_path=args.route_records_path" in cli
        assert "route_record_manifest_path=args.route_record_manifest_path" in cli
        assert "route_record_model_digest=args.route_record_model_digest" in cli

    for script in (global_sbatch, toolbench_sbatch):
        assert "ROUTE_RECORDS_PATH=${ROUTE_RECORDS_PATH:-}" in script
        assert "ROUTE_RECORD_MANIFEST_PATH=${ROUTE_RECORD_MANIFEST_PATH:-}" in script
        assert "ROUTE_RECORD_MODEL_DIGEST=${ROUTE_RECORD_MODEL_DIGEST:-}" in script
        assert 'args+=(--route_records_path "${ROUTE_RECORDS_PATH}")' in script
        assert 'args+=(--route_record_manifest_path "${ROUTE_RECORD_MANIFEST_PATH}")' in script
        assert 'args+=(--route_record_model_digest "${ROUTE_RECORD_MODEL_DIGEST}")' in script


def test_unified_readiness_cli_exposes_memory_utility_evidence_gate():
    cli = Path("scripts/audit_clstr_unified_training_readiness.py").read_text(encoding="utf-8")

    assert "--memory_utility_gate_mode" in cli
    assert "--memory_utility_audit_report_path" in cli
    assert "--memory_utility_gate_checkpoint_path" in cli
    assert "memory_utility_gate_mode=args.memory_utility_gate_mode" in cli
    assert "memory_utility_audit_report_path=args.memory_utility_audit_report_path" in cli
    assert "memory_utility_gate_checkpoint_path=args.memory_utility_gate_checkpoint_path" in cli


def test_active_stage4_launchers_expose_no_legacy_preference_arguments():
    paths = [
        Path("scripts/run_clstr_stage4_act_train.py"),
        Path("scripts/sbatch/run_clstr_unified_stage4_act_train.sh"),
        Path("scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh"),
        Path("scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh"),
        Path("scripts/sbatch/run_clstr_unified_stage4_act_train_function_aug_v2.sh"),
        Path("scripts/sbatch/run_clstr_appworld_current_stage4_act_train.sh"),
    ]
    forbidden = (
        "preference_max_rows",
        "preference_batch_size",
        "lambda_pref",
        "beta_kl",
        "PREFERENCE_MAX_ROWS",
        "PREFERENCE_BATCH_SIZE",
        "LAMBDA_PREF",
        "BETA_KL",
    )

    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits.extend((str(path), token) for token in forbidden if token in text)

    assert hits == []


def test_stage4_cli_train_transition_help_distinguishes_legacy_and_unified_modes():
    script = Path("scripts/run_clstr_stage4_act_train.py").read_text(encoding="utf-8")

    assert 'default="outputs/clstr_unified_memory_stage4_full_pool_counterfactual"' in script
    assert "default=UNIFIED_MEMORY_ROUTE_SCORER" in script
    assert "legacy scoring mode" in script
    assert "Unified-memory mode always trains the causal transition/gate modules" in script
    assert "Default trains TransHead/action_proj only." not in script
    assert 'parser.add_argument("--next_skill_pool_mode"' in script
    assert 'parser.add_argument("--counterfactual_utility_weight"' in script
    assert 'parser.add_argument("--counterfactual_gain_margin"' in script
    assert 'parser.add_argument("--counterfactual_safety_tolerance"' in script
    assert 'parser.add_argument("--counterfactual_gain_weight"' in script
    assert 'parser.add_argument("--counterfactual_safety_weight"' in script
    assert 'parser.add_argument("--counterfactual_warmup_fraction"' in script
    assert 'parser.add_argument("--minimum_learning_rate"' in script
    assert 'parser.add_argument("--learning_rate_warmup_fraction"' in script
    assert 'parser.add_argument("--validation_fraction"' in script
    assert 'parser.add_argument("--validation_rows_per_benchmark"' in script
    assert '"--minimum_validation_rows_per_benchmark"' in script
    assert 'parser.add_argument("--gate_rows_per_benchmark"' in script
    assert 'parser.add_argument("--validation_interval_steps"' in script
    assert 'parser.add_argument("--resume_checkpoint_path"' in script


def test_v4_2_progressive_final_stage4_sbatch_starts_from_stage2_checkpoint_and_causal_act():
    script = Path("scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAJECTORIES_PATH=${{TRAJECTORIES_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_PROGRESSIVE_STAGE4_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert f"HEAD_CHECKPOINT_PATH=${{HEAD_CHECKPOINT_PATH:-{V4_2_PROGRESSIVE_STAGE2_CHECKPOINT}}}" in script
    assert "LAMBDA_PREF" not in script
    assert "BETA_KL" not in script
    assert "CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_INVENTORY_MIN_CANDIDATES=${STAGE0_INVENTORY_MIN_CANDIDATES:-64}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}" in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}" in script
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}" in script
    assert "scripts/sbatch/run_clstr_unified_stage4_act_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_function_aug_v2_stage4_sbatch_uses_calibrated_stage2_mainline():
    script = Path("scripts/sbatch/run_clstr_unified_stage4_act_train_function_aug_v2.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAJECTORIES_PATH=${{TRAJECTORIES_PATH:-{FUNCTION_AUG_V2_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{FUNCTION_AUG_V2_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{FUNCTION_AUG_V2_STAGE4_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{FUNCTION_AUG_V2_STAGE0_CHECKPOINT}}}" in script
    assert f"HEAD_CHECKPOINT_PATH=${{HEAD_CHECKPOINT_PATH:-{FUNCTION_AUG_V2_STAGE2_CHECKPOINT}}}" in script
    assert "TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.5}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-500}" in script
    assert "STAGE0_INVENTORY_MIN_CANDIDATES=${STAGE0_INVENTORY_MIN_CANDIDATES:-50}" in script
    assert "CANDIDATE_COUNT=${CANDIDATE_COUNT:-50}" in script
    assert "scripts/sbatch/run_clstr_unified_stage4_act_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_logged_online_stage4_sbatch_uses_frozen_retriever_handoff_and_tiny_defaults():
    script = Path("scripts/sbatch/run_logged_online_stage4_train.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=01:00:00" in script
    assert "scripts/run_logged_online_stage4_train.py" in script
    assert f"TRAJECTORIES_PATH=${{TRAJECTORIES_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_2_PROGRESSIVE_LOGGED_ONLINE_STAGE4_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert f"HEAD_CHECKPOINT_PATH=${{HEAD_CHECKPOINT_PATH:-{V4_2_PROGRESSIVE_STAGE2_CHECKPOINT}}}" in script
    assert "USE_STAGE0_HANDOFF=${USE_STAGE0_HANDOFF:-1}" in script
    assert "--use_stage0_handoff" in script
    assert "MAX_ROWS=${MAX_ROWS:-256}" in script
    assert "MAX_UPDATES=${MAX_UPDATES:-30}" in script
    assert "EVAL_ROWS=${EVAL_ROWS:-32}" in script
    assert "EVAL_SPLIT_MODE=${EVAL_SPLIT_MODE:-sequential_tail}" in script
    assert "TRAJECTORY_EVAL_STEPS=${TRAJECTORY_EVAL_STEPS:-1}" in script
    assert "--eval_split_mode" in script
    assert "--trajectory_eval_steps" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-stage0_rank_prior_plus_transition_residual}" in script
    assert "ROUTE_SCORER=${ROUTE_SCORER:-unified_memory}" in script
    assert "--route_scorer" in script
    assert "ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-0.0}" in script
    assert "ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.0}" in script
    assert "ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-5.0}" in script
    assert "ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-latest_exact}" in script
    assert "ONLINE_MEMORY_STATE_SIMILARITY_THRESHOLD=${ONLINE_MEMORY_STATE_SIMILARITY_THRESHOLD:-0.2}" in script
    assert "ONLINE_MEMORY_STATE_SIMILARITY_TEMPERATURE=${ONLINE_MEMORY_STATE_SIMILARITY_TEMPERATURE:-1.0}" in script
    assert "ONLINE_MEMORY_AUTO_GATE=${ONLINE_MEMORY_AUTO_GATE:-0}" in script
    assert "ONLINE_MEMORY_GATE_SCOPE=${ONLINE_MEMORY_GATE_SCOPE:-global}" in script
    assert "ONLINE_MEMORY_GATE_EVAL_ROWS=${ONLINE_MEMORY_GATE_EVAL_ROWS:-64}" in script
    assert "ONLINE_MEMORY_GATE_MIN_DELTA_MRR=${ONLINE_MEMORY_GATE_MIN_DELTA_MRR:-0.0}" in script
    assert "ONLINE_MEMORY_GATE_MIN_DELTA_RECALL5=${ONLINE_MEMORY_GATE_MIN_DELTA_RECALL5:-0.0}" in script
    assert "--online_memory_weight" in script
    assert "--online_memory_next_skill_bonus" in script
    assert "--online_memory_exact_transition_bonus" in script
    assert "--online_memory_mode" in script
    assert "--online_memory_state_similarity_threshold" in script
    assert "--online_memory_state_similarity_temperature" in script
    assert "--online_memory_gate_scope" in script
    assert "--online_memory_gate_eval_rows" in script
    assert "--online_memory_gate_min_delta_mrr" in script
    assert "--online_memory_gate_min_delta_recall5" in script
    assert "--online_memory_auto_gate" in script
    assert "STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=${STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES:-5}" in script
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3,traject_bench,alfworld,webshop}" in script
    assert 'BENCHMARK_CAPS=${BENCHMARK_CAPS:-""}' in script
    assert 'EVAL_BENCHMARK_CAPS=${EVAL_BENCHMARK_CAPS:-""}' in script
    assert 'BENCHMARK_CAPS_NORMALIZED=${BENCHMARK_CAPS//:/,}' in script
    assert 'EVAL_BENCHMARK_CAPS_NORMALIZED=${EVAL_BENCHMARK_CAPS//:/,}' in script
    assert "--benchmark_caps" in script
    assert "--eval_benchmark_caps" in script
    assert "SAVE_STAGE4_ROWS_PATH=${SAVE_STAGE4_ROWS_PATH:-}" in script
    assert "PREBUILT_STAGE4_ROWS_PATH=${PREBUILT_STAGE4_ROWS_PATH:-}" in script
    assert "REPLAY_PREFIX_SOURCE_ROWS_PATH=${REPLAY_PREFIX_SOURCE_ROWS_PATH:-}" in script
    assert "REPLAY_PREFIX_MAX_STEPS=${REPLAY_PREFIX_MAX_STEPS:-3}" in script
    assert "--save_stage4_rows_path" in script
    assert "--prebuilt_stage4_rows_path" in script
    assert "--replay_prefix_source_rows_path" in script
    assert "--replay_prefix_max_steps" in script
    assert "[[ -s \"${ROUTING_CHECKPOINT_PATH}\" ]]" in script
    assert "[[ -s \"${HEAD_CHECKPOINT_PATH}\" ]]" in script
    assert "--logged_steps_path" not in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_trajectbench_full_clstr_eval_sbatch_uses_eval_only_sanitized_route():
    script = Path("scripts/sbatch/run_trajectbench_full_clstr_eval.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_logged_online_stage4_train.py" in script
    assert "outputs/trajectbench_full_clstr_eval" in script
    assert (
        "TRAJECTORIES_PATH=${TRAJECTORIES_PATH:-.tmp/stage4_sanitized_trajectbench/"
        "trajectories_visible_global_full.jsonl}"
    ) in script
    assert "tool_inventory_skill_ids" not in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_2_PROGRESSIVE_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_2_STAGE0_CHECKPOINT}}}" in script
    assert f"HEAD_CHECKPOINT_PATH=${{HEAD_CHECKPOINT_PATH:-{V4_2_PROGRESSIVE_STAGE2_CHECKPOINT}}}" in script
    assert "USE_STAGE0_HANDOFF=${USE_STAGE0_HANDOFF:-1}" in script
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-traject_bench}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-traject_bench=-1}" in script
    assert "SAVE_STAGE4_ROWS_PATH=${SAVE_STAGE4_ROWS_PATH:-}" in script
    assert "PREBUILT_STAGE4_ROWS_PATH=${PREBUILT_STAGE4_ROWS_PATH:-}" in script
    assert "--save_stage4_rows_path" in script
    assert "--prebuilt_stage4_rows_path" in script
    assert "EVAL_BENCHMARK_CAPS=${EVAL_BENCHMARK_CAPS:-traject_bench=-1}" in script
    assert "MAX_UPDATES=${MAX_UPDATES:-0}" in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-0}" in script
    assert "TRAIN_SCORE_CALIBRATOR=${TRAIN_SCORE_CALIBRATOR:-0}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "EVAL_SPLIT_MODE=${EVAL_SPLIT_MODE:-trajectory_prefix}" in script
    assert "TRAJECTORY_EVAL_STEPS=${TRAJECTORY_EVAL_STEPS:-1}" in script
    assert "ONLINE_MEMORY_MODE=${ONLINE_MEMORY_MODE:-inventory_remaining}" in script
    assert "ONLINE_MEMORY_WEIGHT=${ONLINE_MEMORY_WEIGHT:-1.0}" in script
    assert "ONLINE_MEMORY_NEXT_SKILL_BONUS=${ONLINE_MEMORY_NEXT_SKILL_BONUS:-0.25}" in script
    assert "ONLINE_MEMORY_EXACT_TRANSITION_BONUS=${ONLINE_MEMORY_EXACT_TRANSITION_BONUS:-3.0}" in script
    assert "ONLINE_MEMORY_AUTO_GATE=${ONLINE_MEMORY_AUTO_GATE:-1}" in script
    assert "ONLINE_MEMORY_GATE_SCOPE=${ONLINE_MEMORY_GATE_SCOPE:-source_benchmark}" in script
    assert "tee \"${OUTPUT_DIR}/eval_stdout.json\"" in script
    assert "train_stdout.json" not in script
    assert "--logged_steps_path" not in script


def test_logged_online_stage4_candidate_diagnostics_sbatch_is_low_cost_toolbench_only():
    script = Path("scripts/sbatch/run_logged_online_stage4_candidate_diagnostics.sh").read_text(encoding="utf-8")

    assert "scripts/audit_logged_online_stage4_candidates.py" in script
    assert "ALLOWED_BENCHMARKS=${ALLOWED_BENCHMARKS:-toolbench_g3}" in script
    assert "MAX_ROWS=${MAX_ROWS:-1024}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "--allowed_benchmarks" in script
    assert "--max_rows" in script
    assert "--wrap" not in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_v4_1b_conservative_listwise_stage4_sbatch_starts_from_current_stage2_checkpoint():
    script = Path("scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=06:00:00" in script
    assert f"TRAJECTORIES_PATH=${{TRAJECTORIES_PATH:-{V4_1B_DATA_ROOT}/trajectories.jsonl}}" in script
    assert f"SKILLS_PATH=${{SKILLS_PATH:-{V4_1B_DATA_ROOT}/skill_pool.jsonl}}" in script
    assert f"OUTPUT_DIR=${{OUTPUT_DIR:-{V4_1B_CONSERVATIVE_LISTWISE_STAGE4_OUTPUT_DIR}}}" in script
    assert f"ROUTING_CHECKPOINT_PATH=${{ROUTING_CHECKPOINT_PATH:-{V4_1B_STAGE0_CHECKPOINT}}}" in script
    assert f"HEAD_CHECKPOINT_PATH=${{HEAD_CHECKPOINT_PATH:-{V4_1B_CONSERVATIVE_LISTWISE_STAGE2_CHECKPOINT}}}" in script
    assert "STAGE2_QUALITY_GATE_PATH=${STAGE2_QUALITY_GATE_PATH:-outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/stage2_quality_gate.json}" in script
    assert "CANDIDATE_COUNT=${CANDIDATE_COUNT:-64}" in script
    assert "STAGE0_TOP_M=${STAGE0_TOP_M:-350}" in script
    assert "STAGE0_POSITIVE_MISSING_POLICY=${STAGE0_POSITIVE_MISSING_POLICY:-skip}" in script
    assert "STAGE0_HANDOFF_QUERY_MODE=${STAGE0_HANDOFF_QUERY_MODE:-skillrouter_state}" in script
    assert "LAMBDA_PREF" not in script
    assert "BETA_KL" not in script
    assert "TRANSITION_RESIDUAL_LAMBDA=${TRANSITION_RESIDUAL_LAMBDA:-0.0}" in script
    assert "TRANSITION_SCORING_MODE=${TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "TRAIN_TRANSITION=${TRAIN_TRANSITION:-1}" in script
    assert "BENCHMARK_CAPS=${BENCHMARK_CAPS:-toolbench_g3=4000:traject_bench=4000:alfworld=-1:webshop=4000}" in script
    assert 'BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"' in script
    assert "EXPECTED_TRANSITION_SCORING_MODE=${EXPECTED_TRANSITION_SCORING_MODE:-v4_1b_action_observation}" in script
    assert "EXPECTED_TRANSITION_RESIDUAL_LAMBDA=${EXPECTED_TRANSITION_RESIDUAL_LAMBDA:-0.0}" in script
    assert "scripts/sbatch/run_clstr_unified_stage4_act_train.sh" in script
    assert "exec bash" in script
    _assert_sources_shared_clstr_gpu_env(script)


def test_toolret_eval_import_sbatch_requires_local_sources_and_uses_hf_mirror():
    script = Path("scripts/sbatch/run_import_toolret_eval.sh").read_text(encoding="utf-8")

    assert "scripts/import_toolret_eval.py" in script
    assert "QUERY_SOURCE is required on compute nodes" in script
    assert "TOOL_SOURCE is required on compute nodes" in script
    assert "ALLOW_HF_STREAMING" in script
    assert "data/toolret_eval" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface" in script
    assert "HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets" in script
    assert "envs/reasoning_trap/bin/python" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_traject_eval_import_sbatch_uses_local_public_data_and_repo_cache():
    script = Path("scripts/sbatch/run_import_traject_eval.sh").read_text(encoding="utf-8")

    assert "scripts/import_traject_eval.py" in script
    assert "scripts/audit_traject_eval_data.py" in script
    assert "PUBLIC_DATA" in script
    assert "SPLIT_PARTITION" in script
    assert "--split_partition" in script
    assert "../TRAJECT-Bench/public_data" in script
    assert "data/traject_eval_traject_split_test" in script
    assert "outputs/traject_eval_traject_split_test/traject_eval_audit.json" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface" in script
    assert "envs/reasoning_trap/bin/python" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_unified_readiness_sbatch_includes_traject_eval_gate():
    script = Path("scripts/sbatch/run_clstr_unified_readiness_audit.sh").read_text(encoding="utf-8")

    assert V4_2_PROGRESSIVE_DATA_ROOT in script
    assert "TRAJECT_EVAL_DIR" in script
    assert "--traject_eval_dir" in script
    assert "data/traject_eval_traject_split_test" in script
    assert "SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH" in script
    assert "--skill_dedup_borderline_review_report_path" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json" in script
    assert "STAGE0_EVAL_METRICS_PATH" in script
    assert "--stage0_eval_metrics_path" in script
    assert V4_2_STAGE0_EVAL_METRICS in script
    assert V4_2_STAGE0_CHECKPOINT in script
    assert V4_2_PROGRESSIVE_STAGE1_CHECKPOINT in script
    assert V4_2_PROGRESSIVE_STAGE2_CHECKPOINT in script
    assert "STAGE3" not in script
    assert "--stage3" not in script
    assert "clstr_unified_stage2_v3_traj_retrieval_full_base" not in script


def test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates():
    script = Path("scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh").read_text(encoding="utf-8")

    assert "SOURCE_ROOT" in script
    assert "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data" in script
    assert "EXPECTED_G3_ANSWER_FILES" in script
    assert "5000" in script
    assert "scripts/download_toolbench_data.py" in script
    assert "--verify_only" in script
    assert "--expected_answer_files" in script
    assert "scripts/import_toolbench_g3.py" in script
    assert "scripts/audit_toolbench_g3_data.py" in script
    assert "scripts/build_clstr_unified_pretrain.py" in script
    assert "data/clstr_unified_pretrain_v4_2_toolbench_g3_refresh" in script
    assert "SKILL_DEDUP_BORDERLINE_REVIEW_REPORT_PATH" in script
    assert "--skill_dedup_borderline_review_report_path" in script
    assert "outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json" in script
    assert "STAGE0_OUTPUT_DIR" in script
    assert "--stage0_output_dir" in script
    assert "STAGE0_EVAL_METRICS_PATH" in script
    assert "--stage0_eval_metrics_path" in script
    assert V4_2_STAGE0_EVAL_METRICS in script
    assert "STAGE0_BASELINE_METRICS_PATH" in script
    assert "--stage0_baseline_metrics_path" in script
    assert "outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json" in script
    assert V4_2_STAGE0_CHECKPOINT in script
    assert V4_2_PROGRESSIVE_STAGE1_CHECKPOINT in script
    assert V4_2_PROGRESSIVE_STAGE2_CHECKPOINT in script
    assert "STAGE3" not in script
    assert "--stage3" not in script
    assert "clstr_unified_stage2_v3_traj_retrieval_full_base" not in script
    assert "query_tool_fix" not in script
    assert "scripts/audit_clstr_unified_training_readiness.py" in script
    assert "--expected_toolbench_g3_answer_files" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_clstr_retrieval_export_sbatch_exports_and_optionally_evaluates_generic_run():
    script = Path("scripts/sbatch/run_export_clstr_retrieval_run.sh").read_text(encoding="utf-8")

    assert "scripts/export_clstr_retrieval_run.py" in script
    assert "scripts/evaluate_retrieval_run.py" in script
    assert "scripts/audit_toolret_eval_data.py" in script
    assert "QUERIES_PATH" in script
    assert "SKILLS_PATH" in script
    assert "QRELS_PATH" in script
    assert "RUN_PREFLIGHT_AUDIT" in script
    assert "TOOLRET_AUDIT_STATUS" in script
    assert "CHECKPOINT_PATH" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert "BASE_MODEL_NAME" in script
    assert "models/Qwen3-8B" in script
    assert "data/toolret_eval/queries.jsonl" in script
    assert "data/toolret_eval/skills.jsonl" in script
    assert "data/toolret_eval/qrels.jsonl" in script
    assert "outputs/toolret_eval/clstr_retrieval" in script
    assert "--local_files_only" in script
    assert "--disable_cross_encoder" in script
    assert "RUN_EVAL" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_traject_retrieval_export_sbatch_exports_and_evaluates_traject_run():
    script = Path("scripts/sbatch/run_export_traject_retrieval_run.sh").read_text(encoding="utf-8")

    assert "scripts/export_clstr_retrieval_run.py" in script
    assert "scripts/evaluate_retrieval_run.py" in script
    assert "scripts/evaluate_traject_sequence_proxy.py" in script
    assert "scripts/audit_traject_eval_data.py" in script
    assert "TRAJECT_DATA_DIR" in script
    assert "data/traject_eval_traject_split_test/queries.jsonl" in script
    assert "data/traject_eval_traject_split_test/skills.jsonl" in script
    assert "data/traject_eval_traject_split_test/qrels.jsonl" in script
    assert "outputs/traject_eval_traject_split_test/clstr_retrieval" in script
    assert "BENCHMARK=${BENCHMARK:-traject}" in script
    assert "--local_files_only" in script
    assert "--disable_cross_encoder" in script
    assert "RUN_PREFLIGHT_AUDIT" in script
    assert "RUN_SEQUENCE_PROXY_EVAL" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_appworld_qwen3_signal_smoke_is_low_cost_and_offline():
    script = Path("scripts/sbatch/run_appworld_qwen3_signal_smoke.sh").read_text(encoding="utf-8")

    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "scripts/run_appworld_multistep_executor_eval.py" in script
    assert "METHOD=${METHOD:-qwen_only}" in script
    assert "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}" in script
    assert "OUTPUT_DIR=${OUTPUT_DIR:-outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_smoke}" in script
    assert "MAX_TASKS=${MAX_TASKS:-2}" in script
    assert "MAX_STEPS=${MAX_STEPS:-3}" in script
    assert "MAX_INTERACTIONS=${MAX_INTERACTIONS:-10}" in script
    assert "--local_files_only" in script
    assert "UPDATES" not in script


def test_toolbench_g3_routing_eval_sbatch_prepares_exports_and_evaluates_run():
    script = Path("scripts/sbatch/run_toolbench_g3_routing_eval.sh").read_text(encoding="utf-8")

    assert "scripts/audit_toolbench_g3_data.py" in script
    assert "scripts/prepare_toolbench_g3_routing_eval.py" in script
    assert "scripts/export_clstr_retrieval_run.py" in script
    assert "scripts/evaluate_retrieval_run.py" in script
    assert "data/toolbench_g3/retrieval.jsonl" in script
    assert "data/toolbench_g3/skills.jsonl" in script
    assert "data/toolbench_g3_routing_eval/queries.jsonl" in script
    assert "data/toolbench_g3_routing_eval/qrels.jsonl" in script
    assert "outputs/toolbench_g3/clstr_routing_eval" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert "BENCHMARK=${BENCHMARK:-toolbench_g3_routing}" in script
    assert "--local_files_only" in script
    assert "--disable_cross_encoder" in script
    assert "RUN_PREFLIGHT_AUDIT" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_pass_rate_sbatch_audits_before_official_sopr():
    script = Path("scripts/sbatch/run_stabletoolbench_pass_rate.sh").read_text(encoding="utf-8")

    assert "scripts/audit_stabletoolbench_pass_rate.py" in script
    assert "--fail_on_action_required" in script
    assert "StableToolBench" in script
    assert "outputs/toolbench_g3/stabletoolbench_converted" in script
    assert "outputs/toolbench_g3/stabletoolbench_pass_rate" in script
    assert "CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}" in script
    assert "TEST_SET=${TEST_SET:-G3_instruction}" in script
    assert "API_POOL_FILE" in script
    assert "RUN_PASS_RATE" in script
    assert "toolbench/tooleval" in script
    assert "eval_pass_rate.py" in script
    assert "--converted_answer_path" in script
    assert "--reference_model" in script
    assert "--test_ids" in script
    assert "--evaluator" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_answer_conversion_sbatch_audits_raw_answers_before_convert():
    script = Path("scripts/sbatch/run_convert_stabletoolbench_answers.sh").read_text(encoding="utf-8")

    assert "scripts/convert_stabletoolbench_answers.py" in script
    assert "--fail_on_action_required" in script
    assert "RUN_CONVERT" in script
    assert "--run_convert" in script
    assert "RAW_ANSWER_PATH" in script
    assert "outputs/toolbench_g3/stabletoolbench_raw_answers" in script
    assert "CONVERTED_ANSWER_PATH" in script
    assert "outputs/toolbench_g3/stabletoolbench_converted" in script
    assert "CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}" in script
    assert "TEST_SET=${TEST_SET:-G3_instruction}" in script
    assert "METHOD=${METHOD:-CLSTR@1}" in script
    assert "StableToolBench" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_raw_generation_sbatch_audits_before_generation():
    script = Path("scripts/sbatch/run_stabletoolbench_raw_generation.sh").read_text(encoding="utf-8")

    assert "scripts/audit_stabletoolbench_raw_generation.py" in script
    assert "INPUT_QUERY_FILE" in script
    assert '--input_query_file "${INPUT_QUERY_FILE}"' in script
    assert "--fail_on_action_required" in script
    assert "RUN_GENERATION" in script
    assert "RUN_GENERATION=${RUN_GENERATION:-0}" in script
    assert "qa_pipeline_multithread.py" in script
    assert "RAW_ANSWER_PATH" in script
    assert "outputs/toolbench_g3/stabletoolbench_raw_answers" in script
    assert "TOOL_ROOT_DIR" in script
    assert "StableToolBench/toolenv/tools" in script
    assert "SERVICE_URL" in script
    assert "TOOLBENCH_KEY" in script
    assert "OPENAI_KEY" in script
    assert "CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3}" in script
    assert "TEST_SET=${TEST_SET:-G3_instruction}" in script
    assert "METHOD=${METHOD:-CLSTR@1}" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_local_raw_generation_sbatch_runs_server_and_generation_on_same_node():
    script = Path("scripts/sbatch/run_stabletoolbench_local_raw_generation.sh").read_text(encoding="utf-8")

    assert "scripts/audit_stabletoolbench_virtual_api.py" in script
    assert "scripts/audit_stabletoolbench_raw_generation.py" in script
    assert "INPUT_QUERY_FILE" in script
    assert '--input_query_file "${INPUT_QUERY_FILE}"' in script
    assert "RUN_GENERATION=${RUN_GENERATION:-0}" in script
    assert "SERVICE_URL=${SERVICE_URL:-http://127.0.0.1:${VIRTUAL_API_PORT}/virtual}" in script
    assert "VIRTUAL_API_PID" in script
    assert "trap cleanup EXIT" in script
    assert "main_mirrorapi.py" in script
    assert "qa_pipeline_multithread.py" in script
    assert "--fail_on_action_required" in script
    assert "curl" in script
    assert "OPENAI_KEY" in script
    assert "StableToolBench/toolenv/tools" in script
    assert "outputs/toolbench_g3/stabletoolbench_raw_answers" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_clstr_routed_local_raw_generation_sbatch_uses_derived_inputs():
    script = Path("scripts/sbatch/run_stabletoolbench_clstr_routed_local_raw_generation.sh").read_text(encoding="utf-8")

    assert "run_stabletoolbench_local_raw_generation.sh" in script
    assert "INPUT_QUERY_FILE" in script
    assert "outputs/toolbench_g3/stabletoolbench_clstr_topk/${TEST_SET}.json" in script
    assert "TOOL_ROOT_DIR" in script
    assert "outputs/toolbench_g3/stabletoolbench_clstr_topk_toolenv/tools" in script
    assert "CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3_clstr_topk}" in script
    assert "RAW_AUDIT_OUTPUT" in script
    assert "stabletoolbench_clstr_routed_raw_generation_readiness.json" in script
    assert "RUN_GENERATION=${RUN_GENERATION:-0}" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script
    assert "solvable_queries/test_instruction/${TEST_SET}.json" not in script
    assert "StableToolBench/toolenv/tools" not in script


def test_stabletoolbench_virtual_api_sbatch_audits_before_server_start():
    script = Path("scripts/sbatch/run_stabletoolbench_virtual_api_server.sh").read_text(encoding="utf-8")

    assert "scripts/audit_stabletoolbench_virtual_api.py" in script
    assert "--fail_on_action_required" in script
    assert "RUN_SERVER" in script
    assert "RUN_SERVER=${RUN_SERVER:-0}" in script
    assert "MODE=${MODE:-mirrorapi}" in script
    assert "StableToolBench" in script
    assert "outputs/toolbench_g3/stabletoolbench_virtual_api_readiness.json" in script
    assert "main_mirrorapi.py" in script
    assert "main_mirrorapi_cache.py" in script
    assert "main.py" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_toolenv_build_sbatch_uses_official_solvable_queries():
    script = Path("scripts/sbatch/run_build_stabletoolbench_toolenv.sh").read_text(encoding="utf-8")

    assert "scripts/build_stabletoolbench_toolenv.py" in script
    assert "StableToolBench" in script
    assert "solvable_queries/test_instruction" in script
    assert "G3_instruction" in script
    assert "toolenv/tools" in script
    assert "toolenv_${TEST_SET}_manifest.json" in script


def test_stabletoolbench_clstr_query_build_sbatch_uses_clstr_run_and_skills():
    script = Path("scripts/sbatch/run_build_stabletoolbench_clstr_queries.sh").read_text(encoding="utf-8")

    assert "scripts/build_stabletoolbench_clstr_queries.py" in script
    assert "RUN_PATH" in script
    assert "SKILLS_PATH" in script
    assert "STABLETOOLBENCH_ROOT" in script
    assert "solvable_queries/test_instruction/${TEST_SET}.json" in script
    assert "outputs/toolbench_g3/stabletoolbench_clstr_topk" in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script
    assert "--fail_on_action_required" in script
    assert "data_example" not in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_stabletoolbench_solvable_clstr_retrieval_sbatch_exports_queries_and_run():
    script = Path("scripts/sbatch/run_stabletoolbench_solvable_clstr_retrieval.sh").read_text(encoding="utf-8")

    assert "scripts/build_stabletoolbench_clstr_queries.py" in script
    assert "--export_retrieval_queries" in script
    assert "scripts/export_clstr_retrieval_run.py" in script
    assert "scripts/build_stabletoolbench_toolenv.py" in script
    assert "STABLETOOLBENCH_ROOT" in script
    assert "solvable_queries/test_instruction/${TEST_SET}.json" in script
    assert "data/stabletoolbench_g3_solvable_queries.jsonl" in script
    assert "data/toolbench_g3/skills.jsonl" in script
    assert "outputs/toolbench_g3/stabletoolbench_solvable_clstr_retrieval" in script
    assert "outputs/toolbench_g3/stabletoolbench_clstr_topk" in script
    assert "OUTPUT_QUERY_FILE" in script
    assert "TOOLENV_OUTPUT_ROOT" in script
    assert "--output_query_file" in script
    assert "--query_file \"${OUTPUT_QUERY_FILE}\"" in script
    assert "stabletoolbench_clstr_topk_toolenv" in script
    assert CANONICAL_STAGE0_CHECKPOINT in script
    assert "--local_files_only" in script
    assert "--disable_cross_encoder" in script
    assert "RUN_EVAL" not in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_eval_matrix_readiness_sbatch_audits_without_running_benchmarks():
    script = Path("scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh").read_text(encoding="utf-8")

    assert "scripts/audit_clstr_eval_matrix_readiness.py" in script
    assert "--fail_on_action_required" in script
    assert "STAGE4_CHECKPOINT" in script
    assert f"{V4_2_PROGRESSIVE_STAGE4_OUTPUT_DIR}/checkpoints/clstr_stage4_act-step2000.pt" in script
    assert "STAGE4_OUTPUT_DIR" in script
    assert f"STAGE4_OUTPUT_DIR=${{STAGE4_OUTPUT_DIR:-{V4_2_PROGRESSIVE_STAGE4_OUTPUT_DIR}}}" in script
    assert "--stage4_output_dir" in script
    assert "MIN_STAGE4_STEPS" in script
    assert "--min_stage4_steps" in script
    assert "TOOLRET_RUN_PATH" in script
    assert "TRAJECT_OFFICIAL_METRICS_PATH" in script
    assert "TOOLBENCH_G3_ROUTING_REPORT" in script
    assert "STABLETOOLBENCH_ROOT" in script
    assert "APPWORLD_COMBINATION_REPORT" in script
    assert "outputs/clstr_eval_matrix_readiness/eval_matrix_readiness.json" in script
    assert "RUN_PASS_RATE" not in script
    assert "RUN_GENERATION" not in script
    assert "run_clstr_qwen3_stage4_act_train.py" not in script
    assert "HF_ENDPOINT=https://hf-mirror.com" in script
    assert "cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr" in script


def test_appworld_current_route_preference_train_sbatch_uses_current_route_samples_and_checkpoint():
    script = Path("scripts/sbatch/run_appworld_current_route_preference_train.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_appworld_current_route_preference_train.py" in script
    assert "PREFERENCE_SAMPLES_PATH" in script
    assert "SKILL_POOL_PATH" in script
    assert "BASE_SKILL_POOL_PATH" in script
    assert "CLSTR_CHECKPOINT_PATH" in script
    assert "MAX_STEPS" in script
    assert "BATCH_SIZE" in script
    assert "--preference_samples_path" in script
    assert "--base_skill_pool_path" in script
    assert "--clstr_checkpoint_path" in script
    assert "current_route_preference" in script


def test_tau2_full_clstr_route_sbatch_can_load_stage4_checkpoint():
    script = Path("scripts/sbatch/run_tau2_full_clstr_route_eval.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "STAGE4_CHECKPOINT" in script
    assert "--stage4_checkpoint_path" in script
    assert "clstr_stage4_act-step2000.pt" in script


def test_unified_skillrouter_finetune_sbatch_defaults_to_function_aug_v2_data():
    script = Path("scripts/sbatch/run_unified_skillrouter_finetune.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_unified_skillrouter_finetune.py" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}" in script
    assert "OUTPUT_DIR=${OUTPUT_DIR:-outputs/unified_skillrouter_finetune/function_aug_v2_smoke}" in script
    assert "MAX_ROWS=${MAX_ROWS:-2048}" in script
    assert "MAX_SKILLS=${MAX_SKILLS:-8192}" in script
    assert "MAX_STEPS=${MAX_STEPS:-100}" in script
    assert "--data_root" in script
    assert "--max_steps" in script
    assert "--pooling" in script
    assert "--query_text_mode" in script
    assert "--checkpoint_every" in script


def test_bge_sr_embedding_train_sbatch_uses_bge_model_and_clean_data():
    script = Path("scripts/sbatch/run_bge_sr_embedding_train.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "scripts/run_unified_skillrouter_finetune.py" in script
    assert "ENCODER_MODEL=${ENCODER_MODEL:-${PROJECT_ROOT}/models/BAAI/bge-m3}" in script
    assert "DATA_ROOT=${DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2_toolbench_clean}" in script
    assert "TOKENIZER_PADDING_SIDE=${TOKENIZER_PADDING_SIDE:-auto}" in script
    assert "POOLING=${POOLING:-auto}" in script
    assert "QUERY_TEXT_MODE=${QUERY_TEXT_MODE:-auto}" in script
    assert "--checkpoint_every" in script
    assert "stdout.log" in script


def test_bge_route_eval_sbatch_can_apply_adapter_checkpoint():
    script = Path("scripts/sbatch/run_bge_route_eval.sh").read_text(encoding="utf-8")

    _assert_sources_shared_clstr_gpu_env(script)
    assert "ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH:-}" in script
    assert "--adapter_checkpoint_path" in script
    assert "[bge-route] adapter=" in script
def test_qwen06_cmc_direct_utility_recalibration_launcher_is_cpu_hidden():
    script = Path(
        "scripts/sbatch/run_qwen06_cmc_direct_utility_recalibration.sh"
    ).read_text(encoding="utf-8")

    assert "export CUDA_VISIBLE_DEVICES=" in script
    assert "run_qwen06_cmc_direct_utility_recalibration.py" in script
    assert "--temperature 0.5" in script
    assert "--max_steps 300" in script
    assert "--learning_rate 0.01" in script
