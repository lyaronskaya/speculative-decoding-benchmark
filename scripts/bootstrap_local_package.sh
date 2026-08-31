#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtualenv not found at ${VENV_PYTHON}. Run setup first." >&2
  exit 1
fi

"${VENV_PYTHON}" - <<'PY'
from pathlib import Path
import site

root = Path.cwd()
src_dir = (root / "src").resolve()
site_packages = [Path(path) for path in site.getsitepackages() if "site-packages" in path]
if not site_packages:
    raise SystemExit("Could not locate site-packages for the active virtualenv.")

pth_path = site_packages[0] / "speculative_decoding_local.pth"
pth_path.write_text(str(src_dir) + "\n", encoding="utf-8")
print(f"Wrote {pth_path}")
PY

"${VENV_PYTHON}" -c "import speculative_decoding; print(speculative_decoding.__version__)"
