#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND="${1:-none}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"
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

cat <<'EOF'
Environment is ready.

Next steps:
  source .venv/bin/activate
  python -m speculative_decoding.prepare_eval_data
  python -m speculative_decoding.download_models
  python benchmark.py --max-samples-per-dataset 10
EOF
