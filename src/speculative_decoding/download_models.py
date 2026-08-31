from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS = {
    "target": "Qwen/Qwen3-4B",
    "eagle3": "deepseek-ai/eagle3_qwen3_4b_ttt7",
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download target and draft checkpoints.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=sorted(DEFAULT_MODELS),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models/hf-cache"))
    parser.add_argument("--local-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face snapshot allow-patterns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for alias in args.models:
        repo_id = DEFAULT_MODELS[alias]
        target_dir = args.local_dir / alias
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(args.cache_dir),
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_patterns,
            resume_download=True,
        )
        manifest.append({"alias": alias, "repo_id": repo_id, "local_path": str(path)})

    manifest_path = args.local_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "models": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
