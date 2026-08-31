from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class LinearFlopCounter:
    model: torch.nn.Module

    def __post_init__(self) -> None:
        self.total_flops = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module: torch.nn.Linear, inputs, output) -> None:
        if not inputs:
            return
        input_tensor = inputs[0]
        if not isinstance(input_tensor, torch.Tensor):
            return
        if input_tensor.numel() == 0:
            return
        output_features = module.out_features
        input_features = module.in_features
        batch_items = input_tensor.numel() // input_features
        self.total_flops += int(2 * batch_items * input_features * output_features)

    def reset(self) -> None:
        self.total_flops = 0

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def nvtx_range(label: str | None) -> Iterator[None]:
    if not label:
        yield
        return

    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(label)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


@dataclass
class CudaPhaseProfiler:
    enabled: bool
    device: str
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    peak_vram_mb: float = 0.0
    draft_calls: int = 0
    verify_calls: int = 0
    prefill_calls: int = 0

    def reset(self) -> None:
        self.draft_time_ms = 0.0
        self.verify_time_ms = 0.0
        self.prefill_time_ms = 0.0
        self.peak_vram_mb = 0.0
        self.draft_calls = 0
        self.verify_calls = 0
        self.prefill_calls = 0
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self.enabled or not torch.cuda.is_available():
            yield
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
            if phase == "draft":
                self.draft_time_ms += elapsed_ms
                self.draft_calls += 1
            elif phase == "verify":
                self.verify_time_ms += elapsed_ms
                self.verify_calls += 1
            elif phase == "prefill":
                self.prefill_time_ms += elapsed_ms
                self.prefill_calls += 1
            self.peak_vram_mb = max(
                self.peak_vram_mb,
                float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            )

    def export(self) -> dict[str, float | int]:
        profiled_total_ms = self.draft_time_ms + self.verify_time_ms
        draft_ratio = self.draft_time_ms / profiled_total_ms if profiled_total_ms > 0 else 0.0
        return {
            "draft_time_ms": self.draft_time_ms,
            "verify_time_ms": self.verify_time_ms,
            "prefill_time_ms": self.prefill_time_ms,
            "profiled_phase_time_ms": profiled_total_ms,
            "draft_ratio": draft_ratio,
            "peak_vram_mb": self.peak_vram_mb,
            "draft_call_count": self.draft_calls,
            "verify_call_count": self.verify_calls,
            "prefill_call_count": self.prefill_calls,
        }
