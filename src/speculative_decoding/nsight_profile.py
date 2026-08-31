from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from speculative_decoding.benchmark import DEFAULT_DRAFT_LENGTHS

NCU_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark.py under Nsight Compute and aggregate metrics.")
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--benchmark-entry", default="benchmark.py")
    parser.add_argument("--target-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--eagle3-checkpoint", default="deepseek-ai/eagle3_qwen3_4b_ttt7")
    parser.add_argument("--dflash-checkpoint", default="deepseek-ai/dflash_qwen3_4b_block7")
    parser.add_argument("--dspark-checkpoint", default="deepseek-ai/dspark_qwen3_4b_block7")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["eagle3", "dflash", "dspark"],
        choices=["eagle3", "dflash", "dspark"],
    )
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "aime25"])
    parser.add_argument("--draft-lengths", nargs="+", type=int, default=DEFAULT_DRAFT_LENGTHS)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eval_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-samples-per-dataset", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--experimental-unsafe-k-over-capability", action="store_true")
    return parser.parse_args()


def find_single_run_dir(parent_dir: Path) -> Path:
    run_dirs = [path for path in parent_dir.iterdir() if path.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(f"Expected exactly one benchmark run dir in {parent_dir}, found {len(run_dirs)}.")
    return run_dirs[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def parse_ncu_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    header = None
    start_index = None
    for index, row in enumerate(rows):
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            start_index = index + 1
            break

    if header is None or start_index is None:
        raise RuntimeError(f"Could not find Nsight Compute CSV header in {path}.")

    records: list[dict[str, str]] = []
    for row in rows[start_index:]:
        if not row or len(row) < len(header):
            continue
        record = dict(zip(header, row))
        metric_name = record.get("Metric Name", "")
        if not metric_name:
            continue
        records.append(record)
    return records


def to_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.lower() == "n/a":
        return None
    stripped = stripped.replace(",", "")
    try:
        return float(stripped)
    except ValueError:
        return None


def aggregate_ncu_records(records: list[dict[str, str]]) -> dict[str, float | None]:
    by_metric: dict[str, list[float]] = defaultdict(list)
    for record in records:
        metric_name = record.get("Metric Name")
        metric_value = to_float(record.get("Metric Value", ""))
        if metric_name is None or metric_value is None:
            continue
        by_metric[metric_name].append(metric_value)

    def sum_metric(name: str) -> float:
        return float(sum(by_metric.get(name, [])))

    def mean_metric(name: str) -> float | None:
        values = by_metric.get(name, [])
        if not values:
            return None
        return float(statistics.fmean(values))

    return {
        "dram_bytes_read": sum_metric("dram__bytes_read.sum"),
        "dram_bytes_write": sum_metric("dram__bytes_write.sum"),
        "dram_throughput_pct": mean_metric("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_throughput_pct": mean_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "tensor_pipe_pct": mean_metric("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    }


def build_nvtx_filter(dataset_name: str, algorithm: str, k: int) -> str:
    dataset = re.escape(dataset_name)
    algo = re.escape(algorithm)
    return f"regex:speculative__dataset={dataset}__algorithm={algo}__sample=.*__k={k}/"


def make_benchmark_command(args: argparse.Namespace, algorithm: str, dataset_name: str, k: int, output_dir: Path) -> list[str]:
    command = [
        args.python_bin,
        args.benchmark_entry,
        "--algorithms",
        algorithm,
        "--datasets",
        dataset_name,
        "--draft-lengths",
        str(k),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(output_dir),
        "--max-samples-per-dataset",
        str(args.max_samples_per_dataset),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--attn-implementation",
        args.attn_implementation,
        "--seed",
        str(args.seed),
        "--gpu-ids",
        "0",
        "--enable-nvtx-profile",
        "--target-model",
        args.target_model,
        "--eagle3-checkpoint",
        args.eagle3_checkpoint,
        "--dflash-checkpoint",
        args.dflash_checkpoint,
        "--dspark-checkpoint",
        args.dspark_checkpoint,
    ]
    if args.experimental_unsafe_k_over_capability:
        command.append("--experimental-unsafe-k-over-capability")
    return command


def run_ncu_profile(
    args: argparse.Namespace,
    algorithm: str,
    dataset_name: str,
    k: int,
    result_dir: Path,
) -> tuple[Path, Path]:
    benchmark_parent = result_dir / "benchmark_parent"
    benchmark_parent.mkdir(parents=True, exist_ok=True)
    raw_csv_path = result_dir / "ncu_raw.csv"
    report_path = result_dir / "report"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    benchmark_command = make_benchmark_command(args, algorithm, dataset_name, k, benchmark_parent)
    ncu_command = [
        args.ncu_bin,
        "--nvtx",
        "--nvtx-include",
        build_nvtx_filter(dataset_name, algorithm, k),
        "--target-processes",
        "application-only",
        "--replay-mode",
        "kernel",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(NCU_METRICS),
        "-o",
        str(report_path),
        *benchmark_command,
    ]

    with raw_csv_path.open("w", encoding="utf-8") as stdout_handle:
        completed = subprocess.run(
            ncu_command,
            cwd=Path.cwd(),
            env=env,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nsight Compute profiling failed for {algorithm}/{dataset_name}/k={k}:\n{completed.stderr}"
        )

    run_dir = find_single_run_dir(benchmark_parent)
    return raw_csv_path, run_dir


def summarize_profile_run(
    args: argparse.Namespace,
    algorithm: str,
    dataset_name: str,
    k: int,
    raw_csv_path: Path,
    benchmark_run_dir: Path,
) -> dict[str, Any]:
    samples = read_jsonl(benchmark_run_dir / "samples.jsonl")
    filtered = [
        row
        for row in samples
        if row["dataset"] == dataset_name and row["algorithm"] == algorithm and int(row["draft_length"]) == k
    ]
    if not filtered:
        raise RuntimeError(f"No speculative samples found for {algorithm}/{dataset_name}/k={k}.")

    ncu_records = parse_ncu_csv(raw_csv_path)
    ncu_summary = aggregate_ncu_records(ncu_records)

    total_time_s = float(sum(row["total_time_s"] for row in filtered))
    total_target_linear_flops = float(sum(row.get("target_linear_flops", 0) for row in filtered))
    total_bytes = float(ncu_summary["dram_bytes_read"] + ncu_summary["dram_bytes_write"])
    arithmetic_intensity = total_target_linear_flops / total_bytes if total_bytes > 0 else math.inf
    tensor_tflops = total_target_linear_flops / total_time_s / 1e12 if total_time_s > 0 else math.inf
    bandwidth_gbps = total_bytes / total_time_s / 1e9 if total_time_s > 0 else math.inf

    return {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "draft_length": k,
        "samples_profiled": len(filtered),
        "total_time_s": total_time_s,
        "target_linear_flops": total_target_linear_flops,
        "dram_bytes_read": ncu_summary["dram_bytes_read"],
        "dram_bytes_write": ncu_summary["dram_bytes_write"],
        "dram_bytes_total": total_bytes,
        "bytes_read_vram": ncu_summary["dram_bytes_read"],
        "bytes_written_vram": ncu_summary["dram_bytes_write"],
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "memory_bandwidth_gbps": bandwidth_gbps,
        "dram_throughput_pct": ncu_summary["dram_throughput_pct"],
        "sm_throughput_pct": ncu_summary["sm_throughput_pct"],
        "tensor_pipe_pct": ncu_summary["tensor_pipe_pct"],
        "gpu__dram_throughput_pct_of_peak_hbm": ncu_summary["dram_throughput_pct"],
        "sm__throughput_pct_of_peak": ncu_summary["sm_throughput_pct"],
        "sm__pipe_tensor_cycles_active_pct_of_peak": ncu_summary["tensor_pipe_pct"],
        "tensor_tflops": tensor_tflops,
        "benchmark_run_dir": str(benchmark_run_dir),
        "ncu_raw_csv": str(raw_csv_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_profile_plots(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["algorithm"])].append(row)

    for (dataset_name, algorithm), group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: row["draft_length"])

        xs_k = [row["draft_length"] for row in ordered]
        xs_ai = [row["arithmetic_intensity_flops_per_byte"] for row in ordered]
        ys_bw = [row["memory_bandwidth_gbps"] for row in ordered]
        ys_dram_pct = [row["dram_throughput_pct"] or 0.0 for row in ordered]
        ys_tensor_util = [row["tensor_pipe_pct"] or 0.0 for row in ordered]
        ys_sm_util = [row["sm_throughput_pct"] or 0.0 for row in ordered]
        ys_tflops = [row["tensor_tflops"] for row in ordered]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(xs_k, ys_bw, marker="o", color="#1f77b4", label="Bandwidth (GB/s)")
        ax1.plot(xs_k, ys_dram_pct, marker="s", linestyle="--", color="#4c9ed9", label="DRAM % peak")
        ax2.plot(xs_k, ys_tensor_util, marker="^", color="#d62728", label="Tensor pipe %")
        ax2.plot(xs_k, ys_sm_util, marker="x", linestyle="--", color="#ff7f0e", label="SM throughput %")
        ax1.set_xlabel("Draft Length (K)")
        ax1.set_ylabel("Memory Bandwidth")
        ax2.set_ylabel("Compute Utilization (%)")
        ax1.set_title(f"{dataset_name} / {algorithm}: K vs Bandwidth + Compute")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{dataset_name}_{algorithm}_k_vs_bandwidth_compute.png", dpi=160)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(xs_ai, ys_bw, marker="o", color="#1f77b4", label="Bandwidth (GB/s)")
        ax1.plot(xs_ai, ys_dram_pct, marker="s", linestyle="--", color="#4c9ed9", label="DRAM % peak")
        ax2.plot(xs_ai, ys_tflops, marker="^", color="#2ca02c", label="Tensor TFLOPS")
        ax2.plot(xs_ai, ys_tensor_util, marker="x", linestyle="--", color="#d62728", label="Tensor pipe %")
        ax1.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
        ax1.set_ylabel("Memory Bandwidth")
        ax2.set_ylabel("Tensor Core TFLOPS / Utilization (%)")
        ax1.set_title(f"{dataset_name} / {algorithm}: AI vs Bandwidth + Tensor")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{dataset_name}_{algorithm}_ai_vs_bandwidth_tensor.png", dpi=160)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"nsight-{timestamp}"
    plots_dir = run_dir / "plots"
    raw_dir = run_dir / "raw"
    plots_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for algorithm in args.algorithms:
        for dataset_name in args.datasets:
            for k in args.draft_lengths:
                profile_dir = raw_dir / dataset_name / algorithm / f"k_{k}"
                profile_dir.mkdir(parents=True, exist_ok=True)
                try:
                    raw_csv_path, benchmark_run_dir = run_ncu_profile(
                        args=args,
                        algorithm=algorithm,
                        dataset_name=dataset_name,
                        k=k,
                        result_dir=profile_dir,
                    )
                    summary = summarize_profile_run(
                        args=args,
                        algorithm=algorithm,
                        dataset_name=dataset_name,
                        k=k,
                        raw_csv_path=raw_csv_path,
                        benchmark_run_dir=benchmark_run_dir,
                    )
                    summaries.append(summary)
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "algorithm": algorithm,
                            "draft_length": k,
                            "error": str(exc),
                        }
                    )

    summary_json = run_dir / "nsight_summary.json"
    summary_csv = run_dir / "nsight_summary.csv"
    failures_json = run_dir / "nsight_failures.json"
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_csv(summary_csv, summaries)
    failures_json.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    create_profile_plots(summaries, plots_dir)

    metadata = {
        "run_dir": str(run_dir),
        "draft_lengths": args.draft_lengths,
        "algorithms": args.algorithms,
        "datasets": args.datasets,
        "metrics": NCU_METRICS,
        "gpu_id": args.gpu_id,
        "max_samples_per_dataset": args.max_samples_per_dataset,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "profiles": len(summaries), "failures": len(failures)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
