#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="${ROOT_DIR}/.deps"
DEEPSPEC_HOME="${DEPS_DIR}/DeepSpec"
VENV_DIR="${ROOT_DIR}/.venv"
DEEPSPEC_REF="${DEEPSPEC_REF:-main}"

mkdir -p "${DEPS_DIR}"

if [[ ! -d "${DEEPSPEC_HOME}/.git" ]]; then
  git clone --depth 1 --branch "${DEEPSPEC_REF}" https://github.com/deepseek-ai/DeepSpec "${DEEPSPEC_HOME}"
else
  git -C "${DEEPSPEC_HOME}" fetch origin "${DEEPSPEC_REF}" --depth 1
  git -C "${DEEPSPEC_HOME}" checkout "${DEEPSPEC_REF}"
  git -C "${DEEPSPEC_HOME}" pull --ff-only origin "${DEEPSPEC_REF}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtualenv not found at ${VENV_DIR}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import site

root = Path.cwd()
site_packages = [Path(path) for path in site.getsitepackages() if "site-packages" in path]
if not site_packages:
    raise SystemExit("Could not locate site-packages for the active virtualenv.")

pth_path = site_packages[0] / "deepspec_local.pth"
pth_path.write_text(str((root / ".deps" / "DeepSpec").resolve()) + "\n", encoding="utf-8")
print(f"Wrote {pth_path}")
PY

"${VENV_DIR}/bin/python" -c "import deepspec; print(deepspec.__file__)"

echo "DeepSpec is available from ${DEEPSPEC_HOME}"
