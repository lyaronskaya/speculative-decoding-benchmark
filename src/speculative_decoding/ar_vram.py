"""Measure autoregressive Qwen3 VRAM without loading a draft model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def encode_prompt(tokenizer: AutoTokenizer, turns: list[dict[str, str]]) -> torch.Tensor:
    # Render text first, then tokenize explicitly for compatibility with older Qwen tokenizers.
    try:
        rendered_prompt = tokenizer.apply_chat_template(
            turns,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
    except TypeError:
        rendered_prompt = tokenizer.apply_chat_template(
            turns,
            add_generation_prompt=True,
            tokenize=False,
        )
    if not isinstance(rendered_prompt, str):
        raise TypeError(
            "Qwen tokenizer chat template returned a non-string prompt: "
            f"{type(rendered_prompt).__name__}"
        )
    input_ids = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(dtype=torch.long)


def measure_sample(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    input_ids = encode_prompt(tokenizer, sample["turns"]).to(device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items() if value is not None
    }

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generation_kwargs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    generated_ids = output_ids[0, input_ids.shape[-1] :]
    output_token_count = int(generated_ids.numel())
    allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024**2))
    return {
        "dataset": sample.get("dataset"),
        "sample_id": sample.get("sample_id"),
        "prompt": sample["turns"][-1]["content"],
        "output_token_count": output_token_count,
        "total_time_s": elapsed,
        "time_per_output_token_s": elapsed / max(output_token_count, 1),
        "peak_vram_mb": allocated_mb,
        "peak_vram_gb": allocated_mb / 1024.0,
        "peak_vram_reserved_mb": reserved_mb,
        "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    return [
        {
            "dataset": dataset,
            "samples": len(dataset_rows),
            "mean_peak_vram_mb": statistics.fmean(
                row["peak_vram_mb"] for row in dataset_rows
            ),
            "mean_peak_vram_gb": statistics.fmean(
                row["peak_vram_gb"] for row in dataset_rows
            ),
            "max_peak_vram_gb": max(row["peak_vram_gb"] for row in dataset_rows),
            "mean_peak_vram_reserved_mb": statistics.fmean(
                row["peak_vram_reserved_mb"] for row in dataset_rows
            ),
            "mean_total_time_s": statistics.fmean(
                row["total_time_s"] for row in dataset_rows
            ),
            "mean_time_per_output_token_s": statistics.fmean(
                row["time_per_output_token_s"] for row in dataset_rows
            ),
        }
        for dataset, dataset_rows in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure pure autoregressive target-model VRAM without a draft model."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eval_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ar-vram"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math500", "aime25"],
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM measurement.")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    all_rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        dataset_path = args.dataset_root / f"{dataset_name}.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. Run "
                "python -m speculative_decoding.prepare_eval_data first."
            )
        samples = read_jsonl(dataset_path, args.max_samples_per_dataset)
        for sample in tqdm(samples, desc=f"ar:{dataset_name}"):
            all_rows.append(
                measure_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            )

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    summary_rows = summarize(all_rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "summary.csv", summary_rows)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "target_model": args.target_model,
                "datasets": args.datasets,
                "max_samples_per_dataset": args.max_samples_per_dataset,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "attn_implementation": args.attn_implementation,
                "gpu_id": args.gpu_id,
                "draft_model_loaded": False,
                "vram_metric": "torch.cuda.max_memory_allocated",
                "run_dir": str(run_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
