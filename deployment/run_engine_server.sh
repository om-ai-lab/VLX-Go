#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[vlxgo] Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

# The model includes the language backbone and tokenizer; all default asset paths are relative to the repository.
export HF_MODEL_DIR="${HF_MODEL_DIR:-${HF_MODEL_ID:-${REPO_ROOT}/resources/vlxgo}}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${REPO_ROOT}/resources/dinov3-vits16-pretrain-lvd1689m}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-${REPO_ROOT}/resources/siglip-so400m-patch14-384}"
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
export VLXGO_ATTN_IMPL="${VLXGO_ATTN_IMPL:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m deployment.engine_server "$@"
