#!/bin/bash
#SBATCH --job-name=clstr_vnext_tests
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
export PROJECT_ROOT="${REQUESTED_PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/clstr_vnext_architecture_pytest_20260715}
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${PYTEST_TARGETS:-}" ]]; then
  read -r -a pytest_targets <<< "${PYTEST_TARGETS}"
else
  pytest_targets=(
    tests/test_history_channel.py
    tests/test_clean_training_preflight.py
    tests/test_protected_near_duplicate_filter.py
    tests/test_vnext_builder.py
    tests/test_vnext_core.py
    tests/test_vnext_unified_router.py
    tests/test_vnext_unified_router_train.py
    tests/test_vnext_data.py
    tests/test_vnext_candidates.py
    tests/test_vnext_losses.py
    tests/test_vnext_training.py
    tests/test_vnext_compressor_train.py
    tests/test_vnext_stage2_contract.py
    tests/test_stage2_matched_release.py
    tests/test_vnext_full_chain_finalizer.py
    tests/test_vnext_eval.py
    tests/test_toolbench_skillrouter_official.py
  )
fi

"${PYTHON_BIN}" -m pytest -q --maxfail=1 \
  "${pytest_targets[@]}" \
  | tee "${OUTPUT_DIR}/pytest.log"
