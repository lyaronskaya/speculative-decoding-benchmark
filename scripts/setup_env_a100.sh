#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
BACKEND="${1:-none}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script expects a CUDA Linux host such as an A100 machine." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
bash scripts/bootstrap_local_package.sh

case "${BACKEND}" in
  none)
    ;;
  vllm)
    python -m pip install "vllm==0.24.0"
    ;;
  sglang)
    python -m pip install "sglang==0.5.3.post2"
    ;;
  *)
    echo "Unknown backend '${BACKEND}'. Use one of: none, vllm, sglang." >&2
    exit 1
    ;;
esac

bash scripts/bootstrap_deepspec.sh
python -m speculative_decoding.verify_env

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_0", torch.cuda.get_device_name(0))
PY

cat <<'EOF'
A100 environment is ready.

Recommended next steps:
  source .venv/bin/activate
  python -m speculative_decoding.prepare_eval_data
  python -m speculative_decoding.download_models
  bash scripts/run_benchmark_a100.sh --max-samples-per-dataset 10
EOF
