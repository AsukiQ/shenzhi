#!/bin/bash
#SBATCH --job-name=q06_clstr_frozen
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --exclude=d1n41a15g01

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${REQUESTED_PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

required=(
  EVAL_SCOPE
  FINAL_CHAIN_MANIFEST_PATH FINAL_CHAIN_MANIFEST_SHA256 CHECKPOINT_CHAIN_DIGEST
  STAGE0_CHECKPOINT_PATH STAGE2_CHECKPOINT_PATH STAGE4_CHECKPOINT_PATH
  CORPUS_MANIFEST_PATH CORPUS_MANIFEST_SHA256 BENCHMARK_MANIFEST_PATH
  SOURCE_ROWS_PATH FROZEN_SKILLS_PATH OUTPUT_DIR
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is empty: %s\n' "${name}" >&2
    exit 2
  fi
done
if [[ "${EVAL_SCOPE}" == "full" ]]; then
  for name in ACCEPTED_SMOKE_GATE_PATH ACCEPTED_SMOKE_GATE_SHA256; do
    if [[ -z "${!name:-}" ]]; then
      printf 'ERROR: full evaluation requires %s\n' "${name}" >&2
      exit 2
    fi
  done
  "${PYTHON_BIN}" scripts/validate_qwen06_clstr_multibench_smoke_gate.py \
    --smoke_gate_path "${ACCEPTED_SMOKE_GATE_PATH}" \
    --expected_smoke_gate_sha256 "${ACCEPTED_SMOKE_GATE_SHA256}" \
    --expected_checkpoint_chain_digest "${CHECKPOINT_CHAIN_DIGEST}" \
    --expected_final_chain_manifest_sha256 "${FINAL_CHAIN_MANIFEST_SHA256}"
elif [[ "${EVAL_SCOPE}" != "smoke" ]]; then
  printf 'ERROR: EVAL_SCOPE must be smoke or full; got %s\n' "${EVAL_SCOPE}" >&2
  exit 2
fi
if [[ "${CORPUS_MANIFEST_PATH}" != "${BENCHMARK_MANIFEST_PATH}" ]]; then
  printf 'ERROR: frozen corpus manifest path mismatch\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/.tmp/slurm"
args=(
  "${PYTHON_BIN}" scripts/run_frozen_clstr_route_eval.py
  --final_chain_manifest_path "${FINAL_CHAIN_MANIFEST_PATH}"
  --benchmark_manifest_path "${BENCHMARK_MANIFEST_PATH}"
  --source_rows_path "${SOURCE_ROWS_PATH}"
  --skills_path "${FROZEN_SKILLS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --expected_benchmark_manifest_sha256 "${CORPUS_MANIFEST_SHA256}"
  --expected_final_chain_manifest_sha256 "${FINAL_CHAIN_MANIFEST_SHA256}"
  --expected_checkpoint_chain_digest "${CHECKPOINT_CHAIN_DIGEST}"
  --batch_size "${BATCH_SIZE:-16}"
  --top_k "${TOP_K:-100}"
  --device cuda
)
if [[ -n "${MAX_EVAL_ROWS:-}" && "${MAX_EVAL_ROWS}" != "ALL" ]]; then
  args+=(--max_eval_rows "${MAX_EVAL_ROWS}")
fi

printf '[qwen06-clstr-frozen] benchmark=%s scope=%s output=%s\n' \
  "${BENCHMARK:-unknown}" "${EVAL_SCOPE:-unknown}" "${OUTPUT_DIR}"
printf '[qwen06-clstr-frozen] chain=%s corpus=%s\n' \
  "${CHECKPOINT_CHAIN_DIGEST}" "${CORPUS_MANIFEST_SHA256}"
"${args[@]}" 2>&1 | tee "${OUTPUT_DIR}/stdout.log"
