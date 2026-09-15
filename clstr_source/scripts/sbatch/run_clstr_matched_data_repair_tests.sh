#!/bin/bash
#SBATCH --job-name=clstr_data_repair
#SBATCH -p gpu_a800,gpu_h100,gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

SOURCE_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SOURCE_PROJECT_ROOT}}}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/clstr_matched_data_repair_tests}
TAU2_DATA_ROOT=${TAU2_DATA_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/tau2-bench/data/tau2}
TOOLSANDBOX_SCENARIOS_ROOT=${TOOLSANDBOX_SCENARIOS_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/scenarios}
TOOLSANDBOX_TOOLS_ROOT=${TOOLSANDBOX_TOOLS_ROOT:-/data/run01/scyb713/xzf/AAAI/autodl-tmp/ref_repos/ToolSandbox/tool_sandbox/tools}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m pytest -q --maxfail=1 \
  tests/test_memory_candidate_recall.py \
  tests/test_current_state_route_eval.py \
  tests/test_logged_online_stage4_train.py::test_build_logged_online_stage4_rows_converts_schema_to_act_rows \
  tests/test_logged_online_stage4_train.py::test_build_logged_online_stage4_rows_preserves_multiple_valid_next_skills \
  tests/test_tau2_route_eval.py \
  tests/test_tau2_successful_rollouts.py \
  tests/test_toolsandbox_route_eval.py \
  tests/test_matched_multibench_data.py \
  tests/test_vnext_stage2_contract.py \
  tests/test_clean_training_preflight.py \
  tests/test_vnext_eval.py::test_vnext_support_probe_is_exposed_by_cli_and_launcher \
  tests/test_vnext_eval.py::test_vnext_eval_uses_benchmark_local_pool_and_excludes_no_call \
  tests/test_vnext_eval.py::test_vnext_eval_multi_positive_rank_uses_best_legal_positive \
  tests/test_vnext_eval.py::test_vnext_eval_rejects_equivalent_positive_outside_legal_pool \
  tests/test_vnext_eval.py::test_vnext_eval_loads_only_digest_bound_matched_test_rows \
  tests/test_vnext_training.py::test_canonical_full_chain_uses_static_reranker_before_stage2 \
  tests/test_vnext_training.py::test_vnext_eval_launcher_requires_verified_toolbench_and_matched_heldout_rows \
  | tee "${OUTPUT_DIR}/pytest.log"

"${PYTHON_BIN}" scripts/audit_tau2_successful_rollouts.py \
  --tau2_data_root "${TAU2_DATA_ROOT}" \
  --task_split train \
  --output_path "${OUTPUT_DIR}/tau2_train_report.json" \
  > "${OUTPUT_DIR}/tau2_train_stdout.json"

"${PYTHON_BIN}" scripts/audit_tau2_successful_rollouts.py \
  --tau2_data_root "${TAU2_DATA_ROOT}" \
  --task_split test \
  --output_path "${OUTPUT_DIR}/tau2_test_report.json" \
  > "${OUTPUT_DIR}/tau2_test_stdout.json"

"${PYTHON_BIN}" scripts/audit_toolsandbox_dag_routes.py \
  --scenarios_root "${TOOLSANDBOX_SCENARIOS_ROOT}" \
  --tools_root "${TOOLSANDBOX_TOOLS_ROOT}" \
  --output_path "${OUTPUT_DIR}/toolsandbox_dag_report.json" \
  > "${OUTPUT_DIR}/toolsandbox_dag_stdout.json"
