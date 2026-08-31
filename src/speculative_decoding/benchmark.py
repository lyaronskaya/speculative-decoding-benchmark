from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import statistics
import time
import traceback
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from speculative_decoding.deepspec_runtime import ensure_deepspec_importable, require_cuda_device
from speculative_decoding.profiling import CudaPhaseProfiler, LinearFlopCounter, nvtx_range

DEFAULT_TARGET_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DRAFT_MODELS = {
    "eagle3": "deepseek-ai/eagle3_qwen3_4b_ttt7",
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
DEFAULT_DRAFT_LENGTHS = [1, 2, 3, 4, 6, 7]


@dataclass
class RunConfig:
    target_model: str
    draft_model: str
    algorithm: str
    temperature: float
    max_new_tokens: int
    confidence_threshold: float
    attn_implementation: str
    seed: int
    deepspec_home: str | None
    device: str = "cuda:0"
    allow_unsafe_k_over_capability: bool = False
    enable_nvtx_profile: bool = False
    enable_phase_profiling: bool = False


class BenchmarkRunner:
    def __init__(self, cfg: RunConfig) -> None:
        ensure_deepspec_importable(cfg.deepspec_home)
        from deepspec.data.parser import encode_chat_messages
        from deepspec.eval.base_evaluator import (
            DraftProposal,
            assert_no_final_target_layer,
            generate_decoding_sample,
            has_stop_token,
            resolve_stop_token_ids,
        )
        from deepspec.eval.dspark.draft_ops import (
            build_dspark_proposal,
            forward_dspark_draft_block,
        )
        from deepspec.modeling.dspark.common import extract_context_feature
        from deepspec.utils.sampling import logits_to_probs, sample_tokens
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        Qwen3DSparkModel = self._import_first_available(
            [
                ("deepspec.modeling.dspark.qwen3.modeling", "Qwen3DSparkModel"),
                ("deepspec.modeling.dspark.qwen3", "Qwen3DSparkModel"),
                ("deepspec.modeling.dspark.modeling_qwen3_dspark", "Qwen3DSparkModel"),
            ]
        )
        Qwen3Eagle3Model = self._import_first_available(
            [
                ("deepspec.modeling.eagle3.qwen3.modeling", "Qwen3Eagle3Model"),
                ("deepspec.modeling.eagle3.qwen3", "Qwen3Eagle3Model"),
                ("deepspec.modeling.eagle3.modeling_qwen3_eagle3", "Qwen3Eagle3Model"),
            ]
        )
        extract_eagle3_context_feature = self._import_first_available(
            [
                ("deepspec.modeling.eagle3", "extract_eagle3_context_feature"),
                ("deepspec.modeling.eagle3.common", "extract_eagle3_context_feature"),
            ]
        )

        self.cfg = cfg
        self.encode_chat_messages = encode_chat_messages
        self.DraftProposal = DraftProposal
        self.assert_no_final_target_layer = assert_no_final_target_layer
        self.generate_decoding_sample = generate_decoding_sample
        self.has_stop_token = has_stop_token
        self.resolve_stop_token_ids = resolve_stop_token_ids
        self.build_dspark_proposal = build_dspark_proposal
        self.forward_dspark_draft_block = forward_dspark_draft_block
        self.extract_context_feature = extract_context_feature
        self.extract_eagle3_context_feature = extract_eagle3_context_feature
        self.logits_to_probs = logits_to_probs
        self.sample_tokens = sample_tokens
        self.DynamicCache = DynamicCache
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.Qwen3DSparkModel = Qwen3DSparkModel
        self.Qwen3Eagle3Model = Qwen3Eagle3Model

        self.device = require_cuda_device(cfg.device)
        self.phase_profiler = CudaPhaseProfiler(
            enabled=cfg.enable_phase_profiling,
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model,
            attn_implementation=cfg.attn_implementation,
            dtype=torch.bfloat16,
        ).to(self.device)
        self.target_model.eval()
        self.target_flop_counter = LinearFlopCounter(self.target_model)

        if cfg.algorithm == "eagle3":
            self.draft_model = Qwen3Eagle3Model.from_pretrained(
                cfg.draft_model,
                attn_implementation=cfg.attn_implementation,
                dtype=torch.bfloat16,
            ).to(self.device)
            self.draft_model.target_layer_ids = [int(x) for x in self.draft_model.target_layer_ids]
            self.assert_no_final_target_layer(self.target_model, self.draft_model.target_layer_ids)
            self.capability = int(self.draft_model.ttt_length)
        else:
            self.draft_model = Qwen3DSparkModel.from_pretrained(
                cfg.draft_model,
                attn_implementation=cfg.attn_implementation,
                dtype=torch.bfloat16,
            ).to(self.device)
            self.assert_no_final_target_layer(self.target_model, self.draft_model.target_layer_ids)
            self.capability = int(self.draft_model.block_size)
        self.draft_model.eval()
        self.stop_token_ids = self._resolve_stop_token_ids_compat()

    def resolve_effective_k(self, requested_k: int) -> int:
        if requested_k <= self.capability:
            return requested_k
        if self.cfg.allow_unsafe_k_over_capability and self.cfg.algorithm == "eagle3":
            return requested_k
        raise ValueError(
            f"{self.cfg.algorithm} checkpoint supports at most {self.capability} draft tokens, "
            f"got {requested_k}. Enable --experimental-unsafe-k-over-capability only for eagle3 "
            "if you want to try values above the checkpoint capability."
        )

    @staticmethod
    def _import_first_available(candidates: list[tuple[str, str]]):
        errors: list[str] = []
        for module_name, symbol_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, symbol_name)
            except Exception as exc:
                errors.append(f"{module_name}.{symbol_name}: {exc!r}")

        joined = "\n".join(errors)
        raise ModuleNotFoundError(
            "Could not resolve a compatible DeepSpec symbol. Tried:\n"
            f"{joined}"
        )

    def encode_prompt(self, turns: list[dict[str, str]]) -> torch.Tensor:
        input_ids = self.encode_chat_messages(
            self.tokenizer,
            turns,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return input_ids.to(device=self.device, dtype=torch.long)

    def _sync(self) -> None:
        torch.cuda.synchronize()

    def _resolve_stop_token_ids_compat(self) -> list[int] | None:
        param_names = list(inspect.signature(self.resolve_stop_token_ids).parameters)
        if param_names and "target" in param_names[0]:
            return self.resolve_stop_token_ids(self.target_model, self.tokenizer)
        return self.resolve_stop_token_ids(self.tokenizer, self.cfg.target_model)

    def _init_eagle3_context(self, initial_output, output_ids: torch.Tensor, position_ids: torch.Tensor, num_input_tokens: int):
        target_hidden = self.extract_eagle3_context_feature(
            initial_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        shifted_prompt_ids = torch.cat(
            [
                output_ids[:, 1:num_input_tokens],
                output_ids[:, num_input_tokens : num_input_tokens + 1],
            ],
            dim=1,
        )
        draft_cache = self.DynamicCache()
        draft_hidden = self.draft_model.extend_draft_cache(
            hidden_states=target_hidden,
            input_ids=shifted_prompt_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=draft_cache,
        )
        return SimpleNamespace(
            draft_cache=draft_cache,
            draft_hidden=draft_hidden,
            position_ids=position_ids,
            current_pos=num_input_tokens,
            cache_len_before=0,
        )

    def _propose_eagle3(self, context: SimpleNamespace, output_ids: torch.Tensor, start: int, stop_token_ids: list[int] | None, k: int):
        context.cache_len_before = context.draft_cache.get_seq_length()
        candidate_ids = [output_ids[:, start : start + 1]]
        draft_logits_list = []
        proposal_hidden = context.draft_hidden
        next_position = start
        for _ in range(k):
            draft_logits = self.draft_model.compute_logits(proposal_hidden)
            draft_logits_list.append(draft_logits)
            next_token = self.sample_tokens(
                draft_logits,
                temperature=float(self.cfg.temperature),
            )
            candidate_ids.append(next_token[:, -1:])
            if self.has_stop_token(next_token, stop_token_ids):
                break
            proposal_hidden = self.draft_model(
                hidden_states=proposal_hidden,
                input_ids=next_token[:, -1:],
                position_ids=context.position_ids[:, next_position : next_position + 1],
                past_key_values=context.draft_cache,
                use_cache=True,
            )
            next_position += 1
        draft_logits = torch.cat(draft_logits_list, dim=1)
        return self.DraftProposal(
            draft_token_count=draft_logits.shape[1],
            verify_input_ids=torch.cat(candidate_ids, dim=1),
            draft_probs=self.logits_to_probs(draft_logits, float(self.cfg.temperature)),
        )

    def _update_eagle3(self, context: SimpleNamespace, verification) -> None:
        assert verification.committed_tokens is not None
        committed_length = int(verification.committed_tokens.shape[1])
        context.draft_cache.crop(int(context.cache_len_before))
        committed_hidden = self.extract_eagle3_context_feature(
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )[:, :committed_length, :]
        context.draft_hidden = self.draft_model.extend_draft_cache(
            hidden_states=committed_hidden,
            input_ids=verification.committed_tokens,
            position_ids=context.position_ids[
                :,
                context.current_pos : context.current_pos + committed_length,
            ],
            past_key_values=context.draft_cache,
        )
        context.current_pos += committed_length

    def _init_dspark_context(self, initial_output):
        return SimpleNamespace(
            past_key_values_draft=self.DynamicCache(),
            target_hidden_states=self.extract_context_feature(
                initial_output.hidden_states,
                self.draft_model.target_layer_ids,
            ),
        )

    def _propose_dspark(self, context: SimpleNamespace, output_ids: torch.Tensor, position_ids: torch.Tensor, start: int, k: int):
        model = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), k),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = self.forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=k,
        )
        return self.build_dspark_proposal(
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=k,
            temperature=float(self.cfg.temperature),
            confidence_threshold=float(self.cfg.confidence_threshold),
        )

    def _update_dspark(self, context: SimpleNamespace, verification) -> None:
        verified_target_hidden = self.extract_context_feature(
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        context.target_hidden_states = verified_target_hidden[
            :,
            : verification.accepted_draft_tokens + 1,
            :,
        ]

    def speculative_generate(
        self,
        turns: list[dict[str, str]],
        k: int,
        nvtx_label: str | None = None,
    ) -> dict[str, Any]:
        k = self.resolve_effective_k(k)

        prompt_ids = self.encode_prompt(turns)
        self.phase_profiler.reset()
        target_forward_call_index = 0
        original_forward = self.target_model.forward

        def profiled_target_forward(*args, **kwargs):
            nonlocal target_forward_call_index
            phase = "prefill" if target_forward_call_index == 0 else "verify"
            target_forward_call_index += 1
            with self.phase_profiler.measure(phase):
                return original_forward(*args, **kwargs)

        self.target_model.forward = profiled_target_forward
        if self.cfg.algorithm == "eagle3":
            def init_context(*, initial_output, output_ids, position_ids, num_input_tokens):
                return self._init_eagle3_context(initial_output, output_ids, position_ids, num_input_tokens)
            def propose(*, context, output_ids, position_ids, start, stop_token_ids=None):
                with self.phase_profiler.measure("draft"):
                    return self._propose_eagle3(context, output_ids, start, stop_token_ids, k)
            update = self._update_eagle3
        else:
            def init_context(*, initial_output, **kwargs):
                return self._init_dspark_context(initial_output)
            def propose(*, context, output_ids, position_ids, start, stop_token_ids=None):
                with self.phase_profiler.measure("draft"):
                    return self._propose_dspark(context, output_ids, position_ids, start, k)
            update = self._update_dspark

        self._sync()
        started = time.perf_counter()
        self.target_flop_counter.reset()
        try:
            with nvtx_range(nvtx_label if self.cfg.enable_nvtx_profile else None):
                with torch.inference_mode():
                    response = self.generate_decoding_sample(
                        target_model=self.target_model,
                        input_ids=prompt_ids,
                        max_proposal_tokens=k,
                        propose=propose,
                        init_context=init_context,
                        update=update,
                        temperature=self.cfg.temperature,
                        max_new_tokens=self.cfg.max_new_tokens,
                        stop_token_ids=self.stop_token_ids,
                    )
        finally:
            self.target_model.forward = original_forward
        self._sync()
        elapsed = time.perf_counter() - started

        output_ids = response.output_ids[0, response.num_input_tokens :].tolist()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        accepted = [int(value) for value in response.accepted_draft_lengths]
        acceptance = [int(value) for value in response.acceptance_lengths]
        proposals = [int(value) for value in response.proposal_lengths]
        return {
            "output_text": output_text,
            "output_token_count": len(output_ids),
            "total_time_s": elapsed,
            "time_per_output_token_s": elapsed / max(len(output_ids), 1),
            "accepted_draft_lengths": accepted,
            "acceptance_lengths": acceptance,
            "proposal_lengths": proposals,
            "mean_accepted_draft_length": statistics.fmean(accepted) if accepted else 0.0,
            "mean_acceptance_length": statistics.fmean(acceptance) if acceptance else 0.0,
            "steps": len(proposals),
            "target_linear_flops": self.target_flop_counter.total_flops,
            **self.phase_profiler.export(),
        }

    def autoregressive_generate(
        self,
        turns: list[dict[str, str]],
        nvtx_label: str | None = None,
    ) -> dict[str, Any]:
        prompt_ids = self.encode_prompt(turns)
        self.phase_profiler.reset()
        generation_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.temperature > 0,
            "temperature": self.cfg.temperature if self.cfg.temperature > 0 else None,
            "eos_token_id": self.stop_token_ids,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

        self._sync()
        started = time.perf_counter()
        self.target_flop_counter.reset()
        with self.phase_profiler.measure("verify"):
            with nvtx_range(nvtx_label if self.cfg.enable_nvtx_profile else None):
                with torch.inference_mode():
                    output_ids = self.target_model.generate(prompt_ids, **generation_kwargs)
        self._sync()
        elapsed = time.perf_counter() - started

        generated_ids = output_ids[0, prompt_ids.shape[-1]:].tolist()
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "output_text": output_text,
            "output_token_count": len(generated_ids),
            "total_time_s": elapsed,
            "time_per_output_token_s": elapsed / max(len(generated_ids), 1),
            "target_linear_flops": self.target_flop_counter.total_flops,
            **self.phase_profiler.export(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DeepSpec benchmark runner.")
    parser.add_argument("--deepspec-home", default=None)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--eagle3-checkpoint", default=DEFAULT_DRAFT_MODELS["eagle3"])
    parser.add_argument("--dflash-checkpoint", default=DEFAULT_DRAFT_MODELS["dflash"])
    parser.add_argument("--dspark-checkpoint", default=DEFAULT_DRAFT_MODELS["dspark"])
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["eagle3", "dflash", "dspark"],
        choices=["eagle3", "dflash", "dspark"],
    )
    parser.add_argument(
        "--draft-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_DRAFT_LENGTHS,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math500", "aime25"],
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/eval_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--enable-phase-profiling",
        action="store_true",
        help="Profile draft/verify/prefill time and peak VRAM with CUDA events.",
    )
    parser.add_argument(
        "--enable-nvtx-profile",
        action="store_true",
        help="Emit NVTX ranges around baseline and speculative generation for Nsight profiling.",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Logical CUDA device ids to shard samples across, for example --gpu-ids 0 1 2 3.",
    )
    parser.add_argument(
        "--experimental-unsafe-k-over-capability",
        action="store_true",
        help=(
            "Experimental: allow requested K above checkpoint capability for eagle3. "
            "This does not unlock true K>capability for block-based dflash/dspark checkpoints."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_capability(algorithm: str, draft_model: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(draft_model)
    if algorithm == "eagle3":
        value = getattr(config, "ttt_length", None)
    else:
        value = getattr(config, "block_size", None)
    if value is None:
        raise ValueError(
            f"Could not resolve capability for algorithm={algorithm!r} from draft model {draft_model!r}."
        )
    return int(value)


def make_run_config(args: argparse.Namespace, algorithm: str, draft_model: str, device: str) -> RunConfig:
    return RunConfig(
        target_model=args.target_model,
        draft_model=draft_model,
        algorithm=algorithm,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        confidence_threshold=args.confidence_threshold,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        deepspec_home=args.deepspec_home,
        device=device,
        allow_unsafe_k_over_capability=args.experimental_unsafe_k_over_capability,
        enable_nvtx_profile=args.enable_nvtx_profile,
        enable_phase_profiling=args.enable_phase_profiling,
    )


def sanitize_nvtx_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def make_nvtx_label(
    kind: str,
    dataset_name: str,
    algorithm: str,
    sample_id: Any,
    requested_k: int | None = None,
) -> str:
    parts = [
        kind,
        f"dataset={sanitize_nvtx_component(dataset_name)}",
        f"algorithm={sanitize_nvtx_component(algorithm)}",
        f"sample={sanitize_nvtx_component(str(sample_id))}",
    ]
    if requested_k is not None:
        parts.append(f"k={requested_k}")
    return "__".join(parts)


def resolve_requested_k_plan(
    algorithm: str,
    capability: int,
    requested_ks: list[int],
    allow_unsafe_k_over_capability: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runnable = []
    skipped = []
    for requested_k in requested_ks:
        if requested_k <= capability:
            runnable.append(
                {
                    "requested_k": requested_k,
                    "effective_k": requested_k,
                    "unsafe_over_capability": False,
                }
            )
            continue

        if allow_unsafe_k_over_capability and algorithm == "eagle3":
            runnable.append(
                {
                    "requested_k": requested_k,
                    "effective_k": requested_k,
                    "unsafe_over_capability": True,
                }
            )
            continue

        reason = f"checkpoint capability is {capability}"
        if allow_unsafe_k_over_capability and algorithm in {"dflash", "dspark"}:
            reason = (
                f"checkpoint capability is {capability}; experimental override is unsupported "
                f"for block-based {algorithm}"
            )
        skipped.append(
            {
                "algorithm": algorithm,
                "draft_length": requested_k,
                "reason": reason,
            }
        )

    return runnable, skipped


def evaluate_sample(
    runner: BenchmarkRunner,
    sample: dict[str, Any],
    dataset_name: str,
    algorithm: str,
    k_plan: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    prompt_seed = seed + int(sample.get("sample_id", 0))
    seed_everything(prompt_seed)
    baseline = runner.autoregressive_generate(
        sample["turns"],
        nvtx_label=make_nvtx_label(
            kind="baseline",
            dataset_name=dataset_name,
            algorithm=algorithm,
            sample_id=sample.get("sample_id"),
        ),
    )

    records = []
    for plan in k_plan:
        requested_k = int(plan["requested_k"])
        effective_k = int(plan["effective_k"])
        seed_everything(prompt_seed)
        speculative = runner.speculative_generate(
            sample["turns"],
            effective_k,
            nvtx_label=make_nvtx_label(
                kind="speculative",
                dataset_name=dataset_name,
                algorithm=algorithm,
                sample_id=sample.get("sample_id"),
                requested_k=requested_k,
            ),
        )
        records.append(
            {
                "dataset": dataset_name,
                "sample_id": sample.get("sample_id"),
                "algorithm": algorithm,
                "draft_length": requested_k,
                "draft_length_effective": effective_k,
                "unsafe_k_over_capability": bool(plan["unsafe_over_capability"]),
                "prompt": sample["turns"][-1]["content"],
                "ground_truth": sample.get("ground_truth"),
                "baseline_output_text": baseline["output_text"],
                "baseline_total_time_s": baseline["total_time_s"],
                "baseline_time_per_output_token_s": baseline["time_per_output_token_s"],
                "baseline_output_token_count": baseline["output_token_count"],
                "baseline_target_linear_flops": baseline["target_linear_flops"],
                "baseline_verify_time_ms": baseline["verify_time_ms"],
                "baseline_peak_vram_mb": baseline["peak_vram_mb"],
                "speedup_vs_autoregressive": (
                    baseline["total_time_s"] / speculative["total_time_s"]
                    if speculative["total_time_s"] > 0
                    else math.inf
                ),
                **speculative,
            }
        )
    return records


def shard_samples(samples: list[dict[str, Any]], worker_count: int) -> list[list[dict[str, Any]]]:
    shards = [[] for _ in range(worker_count)]
    for index, sample in enumerate(samples):
        shards[index % worker_count].append(sample)
    return [shard for shard in shards if shard]


def worker_run_sample_shard(
    gpu_id: int,
    cfg: RunConfig,
    algorithm: str,
    dataset_name: str,
    k_plan: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    result_queue: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(1)
        runner = BenchmarkRunner(cfg)
        for sample in samples:
            records = evaluate_sample(
                runner=runner,
                sample=sample,
                dataset_name=dataset_name,
                algorithm=algorithm,
                k_plan=k_plan,
                seed=cfg.seed,
            )
            result_queue.put(
                {
                    "type": "sample",
                    "gpu_id": gpu_id,
                    "sample_id": sample.get("sample_id"),
                    "records": records,
                }
            )
        result_queue.put({"type": "done", "gpu_id": gpu_id})
    except Exception:
        result_queue.put(
            {
                "type": "error",
                "gpu_id": gpu_id,
                "error": traceback.format_exc(),
            }
        )


def run_single_gpu_samples(
    args: argparse.Namespace,
    algorithm: str,
    dataset_name: str,
    draft_model: str,
    k_plan: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runner = BenchmarkRunner(make_run_config(args, algorithm, draft_model, "cuda:0"))
    records: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"{algorithm}:{dataset_name}", leave=False):
        records.extend(
            evaluate_sample(
                runner=runner,
                sample=sample,
                dataset_name=dataset_name,
                algorithm=algorithm,
                k_plan=k_plan,
                seed=args.seed,
            )
        )
    return records


def run_multi_gpu_samples(
    args: argparse.Namespace,
    algorithm: str,
    dataset_name: str,
    draft_model: str,
    k_plan: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    gpu_ids: list[int],
) -> list[dict[str, Any]]:
    worker_count = min(len(gpu_ids), len(samples))
    shards = shard_samples(samples, worker_count)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for shard, gpu_id in zip(shards, gpu_ids[:worker_count]):
        cfg = make_run_config(args, algorithm, draft_model, f"cuda:{gpu_id}")
        process = ctx.Process(
            target=worker_run_sample_shard,
            args=(gpu_id, cfg, algorithm, dataset_name, k_plan, shard, result_queue),
        )
        process.start()
        processes.append(process)

    records: list[dict[str, Any]] = []
    completed_workers = 0
    progress = tqdm(total=len(samples), desc=f"{algorithm}:{dataset_name}", leave=False)
    worker_errors: list[str] = []
    try:
        while completed_workers < len(processes):
            message = result_queue.get()
            if message["type"] == "sample":
                records.extend(message["records"])
                progress.update(1)
            elif message["type"] == "done":
                completed_workers += 1
            elif message["type"] == "error":
                worker_errors.append(message["error"])
                completed_workers += 1
    finally:
        progress.close()
        for process in processes:
            process.join()

    if worker_errors:
        raise RuntimeError(
            "One or more multi-GPU workers failed:\n" + "\n".join(worker_errors)
        )
    return records


def run_dataset_samples(
    args: argparse.Namespace,
    algorithm: str,
    dataset_name: str,
    draft_model: str,
    k_plan: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    gpu_ids: list[int],
) -> list[dict[str, Any]]:
    if not samples or not k_plan:
        return []
    if len(gpu_ids) <= 1 or len(samples) <= 1:
        return run_single_gpu_samples(args, algorithm, dataset_name, draft_model, k_plan, samples)
    return run_multi_gpu_samples(
        args=args,
        algorithm=algorithm,
        dataset_name=dataset_name,
        draft_model=draft_model,
        k_plan=k_plan,
        samples=samples,
        gpu_ids=gpu_ids,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["dataset"], record["algorithm"], record["draft_length"])
        groups.setdefault(key, []).append(record)

    summary_rows = []
    for (dataset, algorithm, draft_length), rows in sorted(groups.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "draft_length": draft_length,
                "samples": len(rows),
                "mean_accepted_draft_length": statistics.fmean(
                    row["mean_accepted_draft_length"] for row in rows
                ),
                "mean_acceptance_length": statistics.fmean(
                    row["mean_acceptance_length"] for row in rows
                ),
                "mean_total_time_s": statistics.fmean(row["total_time_s"] for row in rows),
                "mean_time_per_output_token_s": statistics.fmean(
                    row["time_per_output_token_s"] for row in rows
                ),
                "mean_speedup_vs_autoregressive": statistics.fmean(
                    row["speedup_vs_autoregressive"] for row in rows
                ),
                "mean_output_token_count": statistics.fmean(
                    row["output_token_count"] for row in rows
                ),
                "mean_draft_time_ms": statistics.fmean(row["draft_time_ms"] for row in rows),
                "mean_verify_time_ms": statistics.fmean(row["verify_time_ms"] for row in rows),
                "mean_prefill_time_ms": statistics.fmean(row["prefill_time_ms"] for row in rows),
                "mean_profiled_phase_time_ms": statistics.fmean(
                    row["profiled_phase_time_ms"] for row in rows
                ),
                "mean_draft_ratio": statistics.fmean(row["draft_ratio"] for row in rows),
                "mean_peak_vram_mb": statistics.fmean(row["peak_vram_mb"] for row in rows),
                "mean_baseline_verify_time_ms": statistics.fmean(
                    row["baseline_verify_time_ms"] for row in rows
                ),
                "mean_baseline_peak_vram_mb": statistics.fmean(
                    row["baseline_peak_vram_mb"] for row in rows
                ),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_plots(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not summary_rows:
        return

    datasets = sorted({row["dataset"] for row in summary_rows})
    metrics = [
        ("mean_accepted_draft_length", "Accepted Draft Length", "accepted_length.png"),
        ("mean_time_per_output_token_s", "Time Per Output Token (s)", "time_per_token.png"),
        ("mean_speedup_vs_autoregressive", "Speedup vs AR", "speedup.png"),
        ("mean_draft_time_ms", "Draft Time (ms)", "draft_time.png"),
        ("mean_verify_time_ms", "Verify Time (ms)", "verify_time.png"),
        ("mean_draft_ratio", "Draft Time Ratio", "draft_ratio.png"),
        ("mean_peak_vram_mb", "Peak VRAM (MB)", "peak_vram.png"),
    ]

    for dataset in datasets:
        dataset_rows = [row for row in summary_rows if row["dataset"] == dataset]
        algorithms = sorted({row["algorithm"] for row in dataset_rows})
        for metric_key, metric_label, filename in metrics:
            plt.figure(figsize=(8, 5))
            for algorithm in algorithms:
                algo_rows = sorted(
                    [row for row in dataset_rows if row["algorithm"] == algorithm],
                    key=lambda row: row["draft_length"],
                )
                xs = [row["draft_length"] for row in algo_rows]
                ys = [row[metric_key] for row in algo_rows]
                plt.plot(xs, ys, marker="o", label=algorithm)
            plt.title(f"{dataset}: {metric_label}")
            plt.xlabel("Draft length (K)")
            plt.ylabel(metric_label)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset}_{filename}", dpi=160)
            plt.close()


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / timestamp
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    draft_models = {
        "eagle3": args.eagle3_checkpoint,
        "dflash": args.dflash_checkpoint,
        "dspark": args.dspark_checkpoint,
    }
    gpu_ids = args.gpu_ids or [0]

    all_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for algorithm in args.algorithms:
        capability = resolve_capability(algorithm, draft_models[algorithm])
        k_plan, rejected = resolve_requested_k_plan(
            algorithm=algorithm,
            capability=capability,
            requested_ks=args.draft_lengths,
            allow_unsafe_k_over_capability=args.experimental_unsafe_k_over_capability,
        )
        skipped.extend(rejected)

        for dataset_name in args.datasets:
            dataset_path = args.dataset_root / f"{dataset_name}.jsonl"
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {dataset_path}. Run "
                    "python -m speculative_decoding.prepare_eval_data first."
                )
            samples = read_jsonl(dataset_path, limit=args.max_samples_per_dataset)
            all_records.extend(
                run_dataset_samples(
                    args=args,
                    algorithm=algorithm,
                    dataset_name=dataset_name,
                    draft_model=draft_models[algorithm],
                    k_plan=k_plan,
                    samples=samples,
                    gpu_ids=gpu_ids,
                )
            )

    sample_path = run_dir / "samples.jsonl"
    write_jsonl(sample_path, all_records)

    summary_rows = summarize_records(all_records)
    summary_json_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    write_csv(summary_csv_path, summary_rows)
    create_plots(summary_rows, plots_dir)

    metadata = {
        "target_model": args.target_model,
        "draft_models": draft_models,
        "draft_lengths_requested": args.draft_lengths,
        "skipped": skipped,
        "datasets": args.datasets,
        "dataset_root": str(args.dataset_root),
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "gpu_ids": gpu_ids,
        "enable_phase_profiling": args.enable_phase_profiling,
        "experimental_unsafe_k_over_capability": args.experimental_unsafe_k_over_capability,
        "output_dir": str(run_dir),
    }
    metadata_path = run_dir / "run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "summary_rows": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
