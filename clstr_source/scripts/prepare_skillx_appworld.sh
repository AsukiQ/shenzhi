#!/bin/bash
set -euo pipefail

# Run this on the login node, not on a compute node.
# It writes only under /data/home/scyb713/run/xzf/AAAI/autodl-tmp.

ROOT="/data/home/scyb713/run/xzf/AAAI/autodl-tmp"
TARGET="${ROOT}/SkillX"
REPO_URL="${SKILLX_REPO_URL:-https://ghfast.top/https://github.com/zjunlp/SkillX.git}"

if [ -e "${TARGET}" ]; then
  echo "SkillX target already exists: ${TARGET}"
  exit 0
fi

cd "${ROOT}"
git clone --depth 1 "${REPO_URL}" "${TARGET}"
echo "SkillX cloned to ${TARGET}"
