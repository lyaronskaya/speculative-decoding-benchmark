from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

DEFAULT_DEEPSPEC_HOME = Path(".deps/DeepSpec")


def ensure_deepspec_importable(deepspec_home: str | os.PathLike[str] | None = None) -> None:
    try:
        importlib.import_module("deepspec")
        return
    except ModuleNotFoundError:
        pass

    candidate = Path(
        deepspec_home
        or os.environ.get("DEEPSPEC_HOME")
        or DEFAULT_DEEPSPEC_HOME
    ).expanduser()
    package_root = candidate.resolve()
    if not (package_root / "deepspec").exists():
        raise ModuleNotFoundError(
            "deepspec package is not importable. Run scripts/bootstrap_deepspec.sh "
            "or set DEEPSPEC_HOME to a DeepSpec checkout."
        )

    sys.path.insert(0, str(package_root))
    importlib.import_module("deepspec")


def require_cuda_device(requested_device: str | None = None) -> str:
    import torch

    if requested_device and requested_device != "auto":
        if not requested_device.startswith("cuda"):
            raise ValueError(
                "DeepSpec benchmark currently supports CUDA execution only. "
                f"Received device={requested_device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        if requested_device != "cuda":
            _, _, index = requested_device.partition(":")
            if not index.isdigit():
                raise ValueError(f"Invalid CUDA device string: {requested_device!r}.")
            device_index = int(index)
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {requested_device!r} is unavailable. "
                    f"Visible device count: {torch.cuda.device_count()}."
                )
        return requested_device

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the DeepSpec benchmark. "
            "This machine does not currently expose a CUDA device."
        )
    return "cuda:0"
