#!/bin/bash
#SBATCH --job-name=webshop_clstr
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --exclude=d1n41a15g01
#SBATCH --output=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.tmp/slurm/%x-%j.out

set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.tmp/slurm"
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

REPO_PATH=${REPO_PATH:-third_party/WebShop}
SMOKE_REPORT_PATH=${SMOKE_REPORT_PATH:-outputs/webshop_eval/harness_smoke_report.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/webshop_eval/clstr_unified_memory_full3000_${SLURM_JOB_ID:-manual}}
ROUTING_INIT_MANIFEST=${ROUTING_INIT_MANIFEST:-outputs/clstr_native_routing_init/manifest.json}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_memory_stage12_full_mmargin_retry_20260707_080738/stage2_full_base/checkpoints/latest.pt}
STAGE4_CHECKPOINT_PATH=${STAGE4_CHECKPOINT_PATH:-outputs/clstr_unified_memory_stage4_full3000_from_stage2_step3022_fastbuild_20260707_152854/checkpoints/clstr_stage4_act-step3000.pt}
SKILL_ROWS_PATH_OVERRIDE=${SKILL_ROWS_PATH_OVERRIDE:-}
AUX_DATA_ROOT=${AUX_DATA_ROOT:-data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2}
MAX_EPISODES=${MAX_EPISODES:-100}
MAX_STEPS=${MAX_STEPS:-50}
REPLAY_PREFIX_MAX_STEPS=${REPLAY_PREFIX_MAX_STEPS:-6}
METHOD=${METHOD:-clstr_unified_memory_concrete_action}

mkdir -p "${OUTPUT_DIR}"

printf '[webshop-clstr] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[webshop-clstr] repo_path=%s smoke_report=%s\n' "${REPO_PATH}" "${SMOKE_REPORT_PATH}"
printf '[webshop-clstr] checkpoint=%s\n' "${CHECKPOINT_PATH}"
printf '[webshop-clstr] stage4_checkpoint=%s\n' "${STAGE4_CHECKPOINT_PATH}"
printf '[webshop-clstr] skill_rows=%s episodes=%s max_steps=%s\n' "${SKILL_ROWS_PATH_OVERRIDE}" "${MAX_EPISODES}" "${MAX_STEPS}"
printf '[webshop-clstr] method=%s replay_prefix_max_steps=%s\n' "${METHOD}" "${REPLAY_PREFIX_MAX_STEPS}"

"${PYTHON_BIN}" scripts/run_webshop_clstr_eval.py \
  --method "${METHOD}" \
  --output_dir "${OUTPUT_DIR}" \
  --smoke_report_path "${SMOKE_REPORT_PATH}" \
  --repo_path "${REPO_PATH}" \
  --routing_init_manifest "${ROUTING_INIT_MANIFEST}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --stage4_checkpoint_path "${STAGE4_CHECKPOINT_PATH}" \
  --skill_rows_path_override "${SKILL_ROWS_PATH_OVERRIDE}" \
  --aux_data_root "${AUX_DATA_ROOT}" \
  --max_episodes "${MAX_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --replay_prefix_max_steps "${REPLAY_PREFIX_MAX_STEPS}" \
  2>&1 | tee "${OUTPUT_DIR}/stdout.log"
