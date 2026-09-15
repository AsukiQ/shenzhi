#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/aux_trajectories}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

if [[ "${USE_AUTODL_TURBO:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  trap 'unset http_proxy https_proxy' EXIT
fi

mkdir -p "${OUTPUT_ROOT}"

hf download moroqq/sft_alfworld_trajectory_dataset_v5_cleaned \
  --repo-type dataset \
  --local-dir "${OUTPUT_ROOT}/sft_alfworld_trajectory_dataset_v5_cleaned"

hf download uyffg/auto-dreamer \
  --repo-type dataset \
  --local-dir "${OUTPUT_ROOT}/auto-dreamer"

hf download agent-eto/eto-sft-trajectory \
  --repo-type dataset \
  --local-dir "${OUTPUT_ROOT}/eto-sft-trajectory"
