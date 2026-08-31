#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Virtualenv is missing. Run scripts/setup_env_a100.sh first." >&2
  exit 1
fi

source .venv/bin/activate

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

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" results

HAS_GPU_IDS_FLAG=0
for arg in "$@"; do
  if [[ "${arg}" == "--gpu-ids" ]]; then
    HAS_GPU_IDS_FLAG=1
    break
  fi
done

GPU_ARGS=()
if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -r -a GPU_ID_ITEMS <<< "${GPU_IDS}"
  if [[ ${HAS_GPU_IDS_FLAG} -eq 0 ]]; then
    GPU_ARGS+=(--gpu-ids)
    for ((idx = 0; idx < ${#GPU_ID_ITEMS[@]}; idx++)); do
      GPU_ARGS+=("${idx}")
    done
  fi
else
  GPU_INDEX="${GPU_INDEX:-0}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDEX}}"
  if [[ ${HAS_GPU_IDS_FLAG} -eq 0 ]]; then
    GPU_ARGS+=(--gpu-ids 0)
  fi
fi

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
PROFILE_PHASES="${PROFILE_PHASES:-1}"
PROFILE_ARGS=()
if [[ "${PROFILE_PHASES}" != "0" ]]; then
  PROFILE_ARGS+=(--enable-phase-profiling)
fi

exec python benchmark.py \
  --attn-implementation "${ATTN_IMPL}" \
  "${PROFILE_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "$@"
