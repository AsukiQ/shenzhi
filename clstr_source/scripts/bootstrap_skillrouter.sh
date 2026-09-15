#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${SKILLROUTER_REPO_URL:-https://github.com/zhengyanzhao1997/SkillRouter.git}"
TARGET_DIR="${SKILLROUTER_TARGET_DIR:-/root/autodl-tmp/skillrouter}"

if [ -d "${TARGET_DIR}/.git" ]; then
  git -C "${TARGET_DIR}" fetch --all --tags
  git -C "${TARGET_DIR}" pull --ff-only
elif [ ! -e "${TARGET_DIR}" ]; then
  git clone "${REPO_URL}" "${TARGET_DIR}"
elif [ -d "${TARGET_DIR}" ] && [ -z "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  rmdir "${TARGET_DIR}"
  git clone "${REPO_URL}" "${TARGET_DIR}"
else
  echo "Error: target directory exists but is not a git repository: ${TARGET_DIR}" >&2
  exit 1
fi

git -C "${TARGET_DIR}" rev-parse --short HEAD
