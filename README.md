# Speculative Decoding Benchmark

This repository contains the evaluation framework and experimental analysis comparing three draft model architectures—**Eagle3**, **DFlash**, and **DSpark**—paired with a **Qwen3-4B** target backbone.

Evaluations were conducted across three mathematical reasoning benchmarks of varying complexity: **GSM8K**, **MATH-500**, and **AIME25**.



## Implementation

The three speculative decoding methods are implemented with [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec).

| Method | Hugging Face checkpoint |
| --- | --- |
| Eagle3 | [`deepseek-ai/eagle3_qwen3_4b_ttt7`](https://huggingface.co/deepseek-ai/eagle3_qwen3_4b_ttt7) |
| DFlash | [`deepseek-ai/dflash_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7) |
| DSpark | [`deepseek-ai/dspark_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) |

The `python -m speculative_decoding.download_models` command downloads these checkpoints by default.


##  Evaluated Metrics

1. **Accepted Draft Length ($L_{\text{accepted}}$):** Average number of draft tokens accepted per target model verification step.
2. **Speedup vs. AR:** Net speedup ratio relative to pure autoregressive target execution.
3. **Peak VRAM (MB):** Peak GPU memory usage during inference.


### Prerequisites
- Linux host with an NVIDIA GPU and CUDA driver
- Python >= 3.10 or later
- Sufficient GPU memory for the Qwen3-4B target model and a draft model

### Quick Start
```bash

git clone https://github.com/lyaronskaya/speculative-decoding-benchmark.git
cd speculative-decoding-benchmark
bash scripts/setup_env_a100.sh
source .venv/bin/activate
python -m pip install -e . --no-build-isolation
bash scripts/bootstrap_local_package.sh
python -m pip install -r requirements.txt
python -m speculative_decoding.verify_env
python -m speculative_decoding.prepare_eval_data
python -m speculative_decoding.download_models
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_benchmark_a100.sh --max-samples-per-dataset 100
```

Benchmark results are written to the `results/` directory.
