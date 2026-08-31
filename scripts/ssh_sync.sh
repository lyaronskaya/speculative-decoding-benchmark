#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/ssh_sync.sh <user@host> [remote_dir]" >&2
  echo "Examples:" >&2
  echo "  bash scripts/ssh_sync.sh ubuntu@host" >&2
  echo "  bash scripts/ssh_sync.sh ubuntu@host speculative_decoding" >&2
  echo "  bash scripts/ssh_sync.sh ubuntu@host /home/ubuntu/speculative_decoding" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="$1"
REMOTE_DIR="${2:-speculative_decoding}"

if [[ "${REMOTE_DIR}" == *".."* ]]; then
  echo "Refusing remote_dir containing '..': ${REMOTE_DIR}" >&2
  echo "Use a path inside the remote home, for example 'speculative_decoding' or '/home/ubuntu/speculative_decoding'." >&2
  exit 1
fi

rsync -avz --delete \
  --exclude '.venv/' \
  --exclude '.deps/' \
  --exclude 'models/' \
  --exclude 'results/' \
  --exclude '__pycache__/' \
  "${ROOT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
