#!/usr/bin/env bash

# Shared same-allocation Qwen3-14B OpenAI-compatible executor lifecycle and
# compute-node provider proxy tunnel.
# Source this file from a Slurm launcher, then call
# start_clstr_qwen14_executor and stop_clstr_qwen14_executor.

CLSTR_QWEN14_EXECUTOR_PID=""
CLSTR_PROVIDER_PROXY_TUNNEL_PID=""

load_clstr_provider_credentials() {
  local credential_file=${1:-${CLSTR_PROVIDER_CREDENTIAL_FILE:-}}
  if [[ -z "${credential_file}" ]]; then
    return 0
  fi
  if [[ ! -r "${credential_file}" ]]; then
    echo "provider credential file is not readable: ${credential_file}" >&2
    return 2
  fi
  local python_bin=${PYTHON_BIN:-python3}
  local -a values=()
  mapfile -d '' -t values < <(
    "${python_bin}" - "${credential_file}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(payload, list):
    payload = payload[0] if payload else {}
if not isinstance(payload, dict):
    raise SystemExit(2)
key = str(payload.get("api_key") or "")
base = str(payload.get("api_base") or payload.get("base_url") or "")
if not key or not base:
    raise SystemExit(2)
sys.stdout.write(key)
sys.stdout.write("\0")
sys.stdout.write(base)
sys.stdout.write("\0")
PY
  )
  if [[ ${#values[@]} -ne 2 || -z "${values[0]}" || -z "${values[1]}" ]]; then
    echo "provider credential file does not contain one usable key/base pair" >&2
    return 2
  fi
  OPENAI_API_KEY=${OPENAI_API_KEY:-${values[0]}}
  OPENAI_BASE_URL=${OPENAI_BASE_URL:-${values[1]}}
  export OPENAI_API_KEY OPENAI_BASE_URL
}

start_clstr_qwen14_executor() {
  local log_path=${1:?executor log path is required}
  local host=${EXECUTOR_HOST:-127.0.0.1}
  local port=${EXECUTOR_PORT:-8000}
  local model_path=${QWEN14_MODEL_PATH:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstrs/clstr/models/Qwen3-14B}
  local python_bin=${SGLANG_PYTHON_BIN:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python}
  local cuda_home=${EXECUTOR_CUDA_HOME:-/data/home/scyb713/run/miniconda3/envs/reasoning_trap/lib/python3.11/site-packages/nvidia/cu13}
  local model_name=${EXECUTOR_MODEL:-qwen3-14b}
  local base_url="http://${host}:${port}/v1"

  if curl -fsS "${base_url}/models" >/dev/null 2>&1; then
    EXECUTOR_BASE_URL=${EXECUTOR_BASE_URL:-${base_url}}
    EXECUTOR_MODEL=${model_name}
    EXECUTOR_API_KEY=${EXECUTOR_API_KEY:-EMPTY}
    export EXECUTOR_BASE_URL EXECUTOR_MODEL EXECUTOR_API_KEY
    return 0
  fi
  if [[ ! -x "${python_bin}" ]]; then
    echo "missing SGLang Python: ${python_bin}" >&2
    return 2
  fi
  export PATH="$(dirname "${python_bin}"):${PATH}"
  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "missing Qwen3-14B model: ${model_path}" >&2
    return 2
  fi
  if [[ -x "${cuda_home}/bin/nvcc" ]]; then
    export CUDA_HOME="${cuda_home}"
    export PATH="${cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_home}/lib:${LD_LIBRARY_PATH:-}"
  fi
  mkdir -p "$(dirname "${log_path}")"
  "${python_bin}" -m sglang.launch_server \
    --model-path "${model_path}" \
    --served-model-name "${model_name}" \
    --host "${host}" \
    --port "${port}" \
    --tp-size 1 \
    --context-length "${EXECUTOR_CONTEXT_LENGTH:-16384}" \
    --mem-fraction-static "${EXECUTOR_MEM_FRACTION_STATIC:-0.68}" \
    --cuda-graph-max-bs "${EXECUTOR_CUDA_GRAPH_MAX_BS:-16}" \
    --random-seed "${EXECUTOR_RANDOM_SEED:-0}" \
    --attention-backend "${EXECUTOR_ATTENTION_BACKEND:-triton}" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen \
    --trust-remote-code \
    >"${log_path}" 2>&1 &
  CLSTR_QWEN14_EXECUTOR_PID=$!

  local attempt
  for attempt in $(seq 1 "${EXECUTOR_READY_ATTEMPTS:-180}"); do
    if ! kill -0 "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null; then
      tail -120 "${log_path}" >&2 || true
      echo "Qwen3-14B executor exited before readiness" >&2
      return 2
    fi
    if curl -fsS "${base_url}/models" >/dev/null 2>&1; then
      EXECUTOR_BASE_URL=${base_url}
      EXECUTOR_MODEL=${model_name}
      EXECUTOR_API_KEY=${EXECUTOR_API_KEY:-EMPTY}
      export EXECUTOR_BASE_URL EXECUTOR_MODEL EXECUTOR_API_KEY
      return 0
    fi
    sleep 5
  done
  tail -120 "${log_path}" >&2 || true
  echo "Qwen3-14B executor readiness timed out" >&2
  return 2
}

stop_clstr_qwen14_executor() {
  if [[ -n "${CLSTR_QWEN14_EXECUTOR_PID}" ]] && kill -0 "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null; then
    kill "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null || true
    local attempt
    for attempt in $(seq 1 "${EXECUTOR_STOP_GRACE_ATTEMPTS:-40}"); do
      if ! kill -0 "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null; then
        wait "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null || true
        CLSTR_QWEN14_EXECUTOR_PID=""
        return 0
      fi
      sleep "${EXECUTOR_STOP_GRACE_SECONDS:-0.25}"
    done
    echo "Qwen3-14B executor did not exit after SIGTERM; sending SIGKILL" >&2
    kill -KILL "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null || true
    wait "${CLSTR_QWEN14_EXECUTOR_PID}" 2>/dev/null || true
  fi
  CLSTR_QWEN14_EXECUTOR_PID=""
}

start_clstr_provider_proxy_tunnel() {
  local upstream_proxy=${CLSTR_UPSTREAM_PROXY_URL:-${CODEX_NET_PROXY_URL:-${HTTPS_PROXY:-${https_proxy:-}}}}
  if [[ ! "${upstream_proxy}" =~ ^https?://(127\.0\.0\.1|localhost):([0-9]+)/?$ ]]; then
    return 0
  fi
  local remote_port=${BASH_REMATCH[2]}
  local jump_host=${CLSTR_PROXY_JUMP_HOST:-}
  if [[ -z "${jump_host}" ]]; then
    echo "loopback provider proxy requires CLSTR_PROXY_JUMP_HOST under Slurm" >&2
    return 2
  fi
  local job_number=${SLURM_JOB_ID:-$$}
  local local_port=$((20000 + 10#${job_number} % 20000))
  local tunnel_attempt readiness_attempt
  local tunnel_ready=0
  for tunnel_attempt in $(seq 1 "${PROVIDER_TUNNEL_START_ATTEMPTS:-3}"); do
    env -u LD_LIBRARY_PATH /usr/bin/ssh -N -T \
      -o BatchMode=yes \
      -o ConnectTimeout=10 \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o StrictHostKeyChecking=accept-new \
      -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
      "${jump_host}" &
    CLSTR_PROVIDER_PROXY_TUNNEL_PID=$!
    for readiness_attempt in $(seq 1 40); do
      if ! kill -0 "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null; then
        wait "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null || true
        CLSTR_PROVIDER_PROXY_TUNNEL_PID=""
        break
      fi
      if (exec 3<>"/dev/tcp/127.0.0.1/${local_port}") 2>/dev/null; then
        exec 3>&-
        tunnel_ready=1
        break
      fi
      sleep 0.25
    done
    if [[ "${tunnel_ready}" == "1" ]]; then
      break
    fi
    if [[ -n "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" ]]; then
      kill "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null || true
      wait "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null || true
      CLSTR_PROVIDER_PROXY_TUNNEL_PID=""
    fi
    echo "provider proxy tunnel start ${tunnel_attempt} failed; retrying" >&2
    sleep 1
  done
  if [[ "${tunnel_ready}" != "1" ]]; then
    echo "provider proxy tunnel exited or timed out before readiness" >&2
    return 2
  fi
  local local_proxy="http://127.0.0.1:${local_port}"
  export HTTP_PROXY="${local_proxy}"
  export HTTPS_PROXY="${local_proxy}"
  export http_proxy="${local_proxy}"
  export https_proxy="${local_proxy}"
  export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,::1}
  export no_proxy=${no_proxy:-localhost,127.0.0.1,::1}
}

stop_clstr_provider_proxy_tunnel() {
  if [[ -n "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" ]] && kill -0 "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null; then
    kill "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null || true
    wait "${CLSTR_PROVIDER_PROXY_TUNNEL_PID}" 2>/dev/null || true
  fi
}
