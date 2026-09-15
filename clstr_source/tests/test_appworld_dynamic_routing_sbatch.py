from pathlib import Path


def test_appworld_dynamic_routing_audit_sbatch_is_small_compute_node_smoke():
    script = Path("scripts/sbatch/run_appworld_dynamic_routing_audit_smoke.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH -p gpu_a800" in script
    assert "#SBATCH --gpus=1" in script
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "scripts/audit_appworld_dynamic_routing.py" in script
    assert "CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}" in script
    assert "BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}" in script
    assert "DYNAMIC_SKILL_POOL_PATH=${DYNAMIC_SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}" in script
    assert "RETRIEVAL_PATH=${RETRIEVAL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/retrieval.jsonl}" in script
    assert "MAX_PAIRS=${MAX_PAIRS:-32}" in script
    assert "SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-stride}" in script
    assert "QUERY_TEXT_FORMAT=${QUERY_TEXT_FORMAT:-auto}" in script
    assert "--max_pairs" in script
    assert "--sampling_strategy" in script
    assert "--batch_size" in script
    assert "--k_values" in script
    assert "--query_text_format" in script
    assert 'tee "${OUTPUT_DIR}/audit_stdout.json"' in script


def test_appworld_dynamic_controller_candidate_audit_sbatch_is_small_compute_node_smoke():
    cli = Path("scripts/audit_appworld_dynamic_controller_candidates.py").read_text(encoding="utf-8")
    script = Path("scripts/sbatch/run_appworld_dynamic_controller_candidate_audit_smoke.sh").read_text(encoding="utf-8")

    assert "audit_appworld_dynamic_controller_candidates" in cli
    assert "--candidate_top_k" in cli
    assert "--state_text_source" in cli
    assert "--appworld_executor_compatible_only" in cli
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH -p gpu_a800" in script
    assert "#SBATCH --gpus=1" in script
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "scripts/audit_appworld_dynamic_controller_candidates.py" in script
    assert "CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt}" in script
    assert "BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl}" in script
    assert "DYNAMIC_SKILL_POOL_PATH=${DYNAMIC_SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_2_nowweak_append/skill_pool.jsonl}" in script
    assert "RETRIEVAL_PATH=${RETRIEVAL_PATH:-outputs/appworld_current_route_online_stage4_smoke/stage0_online_correction_6171bbc_3_20260617_222301/stage0_retrieval_corrections.jsonl}" in script
    assert "TASKS_PATH=${TASKS_PATH:-data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl}" in script
    assert "APPWORLD_ROOT=${APPWORLD_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root}" in script
    assert "MAX_PAIRS=${MAX_PAIRS:-6}" in script
    assert "CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-20}" in script
    assert "STATE_TEXT_SOURCE=${STATE_TEXT_SOURCE:-retrieval_query}" in script
    assert "APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}" in script
    assert "--candidate_top_k" in script
    assert "--tasks_path" in script
    assert "--appworld_root" in script
    assert "--state_text_source" in script
    assert "--appworld_executor_compatible_only" in script
    assert 'tee "${OUTPUT_DIR}/audit_stdout.json"' in script


def test_appworld_dynamic_multistep_executor_sbatch_is_small_compute_node_smoke():
    script = Path("scripts/sbatch/run_appworld_dynamic_multistep_executor_smoke.sh").read_text(encoding="utf-8")

    assert "#SBATCH --time=00:45:00" in script
    assert 'source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"' in script
    assert "exec bash scripts/sbatch/run_appworld_multistep_executor_eval.sh" in script
    assert "METHOD=${METHOD:-clstr_multistep}" in script
    assert "SKILL_POOL_PATH=${SKILL_POOL_PATH:-data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl}" in script
    assert "BASE_SKILL_POOL_PATH=${BASE_SKILL_POOL_PATH:-data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl}" in script
    assert "CLSTR_CHECKPOINT_PATH=${CLSTR_CHECKPOINT_PATH:-outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt}" in script
    assert "APPWORLD_EXECUTOR_COMPATIBLE_ONLY=${APPWORLD_EXECUTOR_COMPATIBLE_ONLY:-1}" in script
    assert "MAX_TASKS=${MAX_TASKS:-3}" in script
    assert "MAX_STEPS=${MAX_STEPS:-3}" in script
