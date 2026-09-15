#!/bin/bash
#SBATCH --job-name=tb_g3_clstr_s2_eval
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

STAGE0_CHECKPOINT=${STAGE0_CHECKPOINT:-outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
TRAIN_PATH=${TRAIN_PATH:-outputs/toolbench_g3_official_skillrouter_comparison/trajectory_capped256_fullpool_data/eval_trajectories.jsonl}
SKILLS_PATH=${SKILLS_PATH:-data/clstr_unified_pretrain_v4_2_progressive_final/skill_pool.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_trajectory_capped256_fullpool}
TOP_M=${TOP_M:-350}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-13}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/evaluate_clstr_stage2_real_topm.py \
  --stage0_checkpoint_path "${STAGE0_CHECKPOINT}" \
  --stage2_checkpoint_path "${STAGE2_CHECKPOINT}" \
  --train_path "${TRAIN_PATH}" \
  --skills_path "${SKILLS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --top_m "${TOP_M}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --allowed_benchmarks toolbench_g3 \
  --benchmark_caps toolbench_g3=-1 \
  --stage0_handoff_query_mode skillrouter_state \
  --transition_inventory_mask_mode auto \
  --transition_loss_type listwise_nll \
  --transition_positive_mode gold_plus_equivalent \
  | tee "${OUTPUT_DIR}/stdout.json"
