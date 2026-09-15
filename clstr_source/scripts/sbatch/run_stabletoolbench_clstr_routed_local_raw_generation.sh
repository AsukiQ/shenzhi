#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/hf_datasets
export XDG_CACHE_HOME=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache
export IPYTHONDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

TEST_SET=${TEST_SET:-G3_instruction}
INPUT_QUERY_FILE=${INPUT_QUERY_FILE:-outputs/toolbench_g3/stabletoolbench_clstr_topk/${TEST_SET}.json}
TOOL_ROOT_DIR=${TOOL_ROOT_DIR:-outputs/toolbench_g3/stabletoolbench_clstr_topk_toolenv/tools}
CANDIDATE_MODEL=${CANDIDATE_MODEL:-clstr_toolbench_g3_clstr_topk}
RAW_AUDIT_OUTPUT=${RAW_AUDIT_OUTPUT:-outputs/toolbench_g3/stabletoolbench_clstr_routed_raw_generation_readiness.json}
RUN_GENERATION=${RUN_GENERATION:-0}

export TEST_SET
export INPUT_QUERY_FILE
export TOOL_ROOT_DIR
export CANDIDATE_MODEL
export RAW_AUDIT_OUTPUT
export RUN_GENERATION

bash scripts/sbatch/run_stabletoolbench_local_raw_generation.sh
