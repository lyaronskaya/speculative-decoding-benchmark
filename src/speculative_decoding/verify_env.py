from __future__ import annotations

import importlib
import json
import sys

REQUIRED_MODULES = [
    "speculative_decoding",
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "datasets",
    "numpy",
    "matplotlib",
    "tqdm",
    "sentencepiece",
    "safetensors",
    "prettytable",
]


def main() -> int:
    results = []
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            results.append(
                {
                    "module": module_name,
                    "ok": True,
                    "version": getattr(module, "__version__", "unknown"),
                }
            )
        except Exception as exc:
            failures.append(module_name)
            results.append({"module": module_name, "ok": False, "error": repr(exc)})

    print(json.dumps({"python": sys.executable, "results": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
