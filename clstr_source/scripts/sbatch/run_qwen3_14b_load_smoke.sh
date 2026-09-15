#!/bin/bash
#SBATCH --job-name=qwen3_14b_load
#SBATCH --output=slurm-%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_a800
#SBATCH --time=00:30:00

set -euo pipefail
module load miniforge3/25.11.0-1
module load cuda/12.1 2>/dev/null || true

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python - <<'PY'
from clstr.qwen_backend import QwenBackendConfig, QwenGenerationBackend

backend = QwenGenerationBackend(
    QwenBackendConfig(
        model_name_or_path="models/Qwen3-14B",
        local_files_only=True,
        torch_dtype="bfloat16",
        max_new_tokens=16,
    )
)
print(backend.metadata())
print(backend.generate_text("You are a test runner.", "Reply with OK only."))
PY
