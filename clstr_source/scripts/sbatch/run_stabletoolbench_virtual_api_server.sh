#!/bin/bash
set -euo pipefail

module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || echo "WARN: cuda/12.1 module not available; continuing with Python environment CUDA runtime"

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

PYTHON_BIN=${PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/xzf/bin/python}
STABLETOOLBENCH_ROOT=${STABLETOOLBENCH_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench}
MODE=${MODE:-mirrorapi}
AUTODL_TMP_ROOT=${AUTODL_TMP_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-outputs/toolbench_g3/stabletoolbench_virtual_api_readiness.json}
RUN_SERVER=${RUN_SERVER:-0}

"${PYTHON_BIN}" scripts/audit_stabletoolbench_virtual_api.py \
  --stabletoolbench_root "${STABLETOOLBENCH_ROOT}" \
  --mode "${MODE}" \
  --autodl_tmp_root "${AUTODL_TMP_ROOT}" \
  --output_path "${AUDIT_OUTPUT}" \
  --fail_on_action_required | tee "${AUDIT_OUTPUT}.stdout.json"

if [[ "${RUN_SERVER}" != "1" ]]; then
  echo "RUN_SERVER=${RUN_SERVER}; audited readiness only."
  exit 0
fi

case "${MODE}" in
  mirrorapi)
    SERVER_SCRIPT=main_mirrorapi.py
    ;;
  mirrorapi_cache)
    SERVER_SCRIPT=main_mirrorapi_cache.py
    ;;
  gpt_cache)
    SERVER_SCRIPT=main.py
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

cd "${STABLETOOLBENCH_ROOT}/server"
exec "${PYTHON_BIN}" "${SERVER_SCRIPT}"
