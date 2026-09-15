#!/bin/bash
#SBATCH --job-name=alf_clstr_main
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH --time=01:30:00
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

OFFICIAL_REPO=${OFFICIAL_REPO:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_repo}
DATA_DIR=${DATA_DIR:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/alfworld_data}
ROUTING_INIT_MANIFEST=${ROUTING_INIT_MANIFEST:-outputs/clstr_native_routing_init/manifest.json}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/alfworld_eval/clstr_mainline_v42_progressive_smoke_${SLURM_JOB_ID:-manual}}
SPLITS=${SPLITS:-valid_seen}
RUN_NAME=${RUN_NAME:-clstr_mainline_v42_progressive}
MAX_EPISODES=${MAX_EPISODES:-5}
MAX_STEPS=${MAX_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
CONTROLLER_MODE=${CONTROLLER_MODE:-policy_only}
INCLUDE_AVAILABLE_ACTIONS_IN_STATE=${INCLUDE_AVAILABLE_ACTIONS_IN_STATE:-0}
Q_SUCCESS_WEIGHT=${Q_SUCCESS_WEIGHT:-0.0}

mkdir -p "${OUTPUT_DIR}"
read -r -a SPLIT_ARGS <<< "${SPLITS}"
args=(
  "${PYTHON_BIN}" scripts/run_alfworld_clstr_eval.py eval
  --official_repo "${OFFICIAL_REPO}"
  --data_dir "${DATA_DIR}"
  --routing_init_manifest "${ROUTING_INIT_MANIFEST}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --splits "${SPLIT_ARGS[@]}"
  --run_name "${RUN_NAME}"
  --max_steps "${MAX_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --controller_mode "${CONTROLLER_MODE}"
  --q_success_weight "${Q_SUCCESS_WEIGHT}"
)
if [[ -n "${STAGE4_CHECKPOINT}" ]]; then
  args+=(--stage4_checkpoint_path "${STAGE4_CHECKPOINT}")
fi
if [[ "${INCLUDE_AVAILABLE_ACTIONS_IN_STATE}" == "1" || "${INCLUDE_AVAILABLE_ACTIONS_IN_STATE}" == "true" ]]; then
  args+=(--include_available_actions_in_state)
fi
if [[ -n "${MAX_EPISODES}" && "${MAX_EPISODES}" != "ALL" ]]; then
  args+=(--max_episodes "${MAX_EPISODES}")
fi

printf '[alfworld-clstr-mainline-eval] output_dir=%s\n' "${OUTPUT_DIR}"
printf '[alfworld-clstr-mainline-eval] splits=%s max_episodes=%s controller_mode=%s\n' "${SPLITS}" "${MAX_EPISODES:-ALL}" "${CONTROLLER_MODE}"
printf '[alfworld-clstr-mainline-eval] stage4_checkpoint=%s\n' "${STAGE4_CHECKPOINT:-none}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
