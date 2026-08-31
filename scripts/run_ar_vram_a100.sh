#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Virtualenv is missing. Run scripts/setup_env_a100.sh first." >&2
  exit 1
fi

source .venv/bin/activate
export HF_HOME="${HF_HOME:-${ROOT_DIR}/models/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${ROOT_DIR}/models/hf-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX:-0}}"

exec python ar_vram.py "$@"
