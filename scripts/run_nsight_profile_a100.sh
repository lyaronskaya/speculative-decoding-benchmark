#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Virtualenv is missing. Run scripts/setup_env_a100.sh first." >&2
  exit 1
fi

source .venv/bin/activate

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found in PATH. Install Nsight Compute CLI on the target host first." >&2
  exit 1
fi

if ! python -c "import deepspec" >/dev/null 2>&1; then
  echo "deepspec is not importable yet; running bootstrap_deepspec.sh" >&2
  bash scripts/bootstrap_deepspec.sh
fi

export HF_HOME="${HF_HOME:-${ROOT_DIR}/models/hf-home}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${ROOT_DIR}/models/hf-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

GPU_INDEX="${GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"

exec python -m speculative_decoding.nsight_profile \
  --gpu-id 0 \
  "$@"
