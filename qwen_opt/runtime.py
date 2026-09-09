from __future__ import annotations

import contextlib
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from . import kernels
from .errors import UnsupportedPass
from .passes import PassReport, apply_passes, patch_static_cache
from .variants import Variant


@dataclass
class Timing:
    setup_ms: float
    prefill_ms: float
    decode_ms: float
    total_ms: float
    host_overhead_ms: float
    tokens_per_second: float
    peak_vram_mib: float


@dataclass
class GenerationResult:
    timing: Timing
    token_ids: list[int]


def environment(model_name: str) -> dict[str, object]:
    import transformers

    return {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
    }


def load_model_and_tokenizer(model_name: str, attention: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise UnsupportedPass("Qwen3-4B performance tests require CUDA")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation=attention,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def exact_length_prompt(tokenizer, text: str, context_length: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("prompt produced no tokens")
    repeated = (ids * ((context_length + len(ids) - 1) // len(ids)))[:context_length]
    return torch.tensor([repeated], dtype=torch.long, device="cuda")


def make_static_cache(model, max_length: int):
    from transformers import StaticCache

    kwargs = {
        "config": model.config,
        "max_cache_len": max_length,
        "device": model.device,
        "dtype": model.dtype,
    }
    try:
        return StaticCache(**kwargs)
    except TypeError:
        # Compatibility with releases where max_batch_size was still explicit.
        kwargs["max_batch_size"] = 1
        return StaticCache(**kwargs)


def make_dynamic_cache(model):
    from transformers import DynamicCache

    try:
        return DynamicCache(config=model.config)
    except TypeError:
        return DynamicCache()


def _mark_compile_step() -> None:
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


class DecodeCallable:
    def __init__(self, model, variant: Variant):
        self.model = model
        self.variant = variant
        self.argmax = (
            kernels.LMHeadArgmax(model.lm_head.weight)
            if "fused_lm_head_argmax" in variant.passes
            else None
        )
        if self.argmax is None:
            self.forward = model
        else:
            self.forward = model.model
        if "compile" in variant.passes:
            self.forward = torch.compile(
                self.forward,
                backend="inductor",
                fullgraph=False,
                options={"triton.cudagraphs": False},
            )

    def __call__(self, input_ids, cache, cache_position) -> torch.Tensor:
        if "compile" in self.variant.passes:
            _mark_compile_step()
        common = dict(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            return_dict=True,
        )
        if self.argmax is None:
            output = self.forward(**common, logits_to_keep=1)
            return output.logits[:, -1, :].argmax(dim=-1)
        output = self.forward(**common)
        return self.argmax(output.last_hidden_state[:, -1, :])


class ManualGraph:
    """One-token graph with model-owned mutable state isolated in static buffers."""

    def __init__(
        self,
        decode: DecodeCallable,
        cache,
        first_token: torch.Tensor,
        context_length: int,
        max_new_tokens: int,
    ):
        if not hasattr(torch.cuda, "CUDAGraph"):
            raise UnsupportedPass("this Torch build has no CUDA Graph support")
        self.decode = decode
        self.cache = cache
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.input = torch.empty((1, 1), dtype=torch.long, device="cuda")
        self.position = torch.empty((1,), dtype=torch.long, device="cuda")
        self.generated = torch.empty(max_new_tokens, dtype=torch.long, device="cuda")
        self.input.copy_(first_token.view(1, 1))
        self.position.fill_(context_length)
        self.generated[0].copy_(first_token[0])

        # Warm kernels and allocator on an independent cache so capture begins from
        # the exact post-prefill state of `cache`.
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                token = self.decode(self.input, self.cache, self.position)
                write_index = self.position - self.context_length + 1
                self.generated.index_copy_(0, write_index, token)
                self.input.copy_(token.view(1, 1))
                self.position.add_(1)
        except RuntimeError as exc:
            raise UnsupportedPass(f"manual CUDA Graph capture failed: {exc}") from exc

    def reset(self, first_token: torch.Tensor) -> None:
        self.input.copy_(first_token.view(1, 1))
        self.position.fill_(self.context_length)
        self.generated[0].copy_(first_token[0])

    def run_decode(self) -> torch.Tensor:
        for _ in range(self.max_new_tokens - 1):
            self.graph.replay()
        return self.generated


class Generator:
    def __init__(
        self,
        model,
        variant: Variant,
        context_length: int,
        max_new_tokens: int,
    ):
        self.model = model
        self.variant = variant
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.max_length = context_length + max_new_tokens
        self.decode = DecodeCallable(model, variant)
        self.runner_buffers = "runner_buffers" in variant.passes
        self.static_cache = None
        if variant.cache == "static":
            self.static_cache = make_static_cache(model, self.max_length)
            # Initialize cache storage outside timed regions. reset() returns it to
            # an empty logical state without changing storage addresses.
            dummy = torch.zeros((1, 1), dtype=torch.long, device="cuda")
            with torch.no_grad():
                self.decode(dummy, self.static_cache, torch.zeros(1, dtype=torch.long, device="cuda"))
            self.static_cache.reset()
            if "fused_kv_write" in variant.passes:
                patched = patch_static_cache(self.static_cache)
                if patched != len(model.model.layers):
                    raise UnsupportedPass(
                        f"patched {patched} cache layers; expected {len(model.model.layers)}"
                    )
        self.input_buffer = torch.empty((1, 1), dtype=torch.long, device="cuda")
        self.position_buffer = torch.empty((1,), dtype=torch.long, device="cuda")
        self.generated_buffer = torch.empty(
            max_new_tokens, dtype=torch.long, device="cuda"
        )
        self.manual_graph: ManualGraph | None = None

    def _cache_for_run(self):
        if self.variant.cache == "dynamic":
            return make_dynamic_cache(self.model)
        self.static_cache.reset()
        return self.static_cache

    def _prefill(self, prompt: torch.Tensor, cache) -> torch.Tensor:
        position = torch.arange(prompt.shape[1], device="cuda", dtype=torch.long)
        return self.decode(prompt, cache, position)

    def _eager_decode(self, first_token: torch.Tensor, cache) -> torch.Tensor:
        self.generated_buffer[0].copy_(first_token[0])
        current = first_token
        for step in range(1, self.max_new_tokens):
            if self.runner_buffers:
                self.input_buffer.copy_(current.view(1, 1))
                self.position_buffer.fill_(self.context_length + step - 1)
                input_ids = self.input_buffer
                position = self.position_buffer
            else:
                input_ids = current.view(1, 1)
                position = torch.tensor(
                    [self.context_length + step - 1], device="cuda", dtype=torch.long
                )
            current = self.decode(input_ids, cache, position)
            self.generated_buffer[step].copy_(current[0])
        return self.generated_buffer

    def warmup(self, prompt: torch.Tensor, count: int) -> None:
        with torch.inference_mode():
            for _ in range(count):
                cache = self._cache_for_run()
                first = self._prefill(prompt, cache)
                if "manual_cudagraph" in self.variant.passes:
                    # Graph capture is the warmup for this fixed shape. Its cache
                    # state is discarded before measured repetitions.
                    if self.manual_graph is None:
                        # Run one decode before capture so Triton/SDPA kernels and
                        # allocator paths are initialized outside capture.
                        self.decode(
                            first.view(1, 1),
                            cache,
                            torch.tensor(
                                [self.context_length], device="cuda", dtype=torch.long
                            ),
                        )
                        cache = self._cache_for_run()
                        first = self._prefill(prompt, cache)
                        self.manual_graph = ManualGraph(
                            self.decode,
                            cache,
                            first,
                            self.context_length,
                            self.max_new_tokens,
                        )
                else:
                    self._eager_decode(first, cache)
            torch.cuda.synchronize()

    def run_once(self, prompt: torch.Tensor) -> GenerationResult:
        torch.cuda.reset_peak_memory_stats()
        start_setup = torch.cuda.Event(enable_timing=True)
        end_setup = torch.cuda.Event(enable_timing=True)
        start_prefill = torch.cuda.Event(enable_timing=True)
        end_prefill = torch.cuda.Event(enable_timing=True)
        end_decode = torch.cuda.Event(enable_timing=True)
        # The headline latency is synchronized end-to-end wall time. CUDA events
        # separately attribute device work to prefill and decode without inserting
        # an artificial synchronization barrier between them.
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        with torch.inference_mode():
            start_setup.record()
            cache = self._cache_for_run()
            end_setup.record()
            start_prefill.record()
            first = self._prefill(prompt, cache)
            end_prefill.record()
            if "manual_cudagraph" in self.variant.passes:
                if self.manual_graph is None:
                    raise RuntimeError("manual graph was not captured during warmup")
                self.manual_graph.reset(first)
                generated = self.manual_graph.run_decode()
            else:
                generated = self._eager_decode(first, cache)
            end_decode.record()
            torch.cuda.synchronize()
            token_ids = generated.cpu().tolist()
        total_ms = (time.perf_counter() - wall_start) * 1000.0
        setup_ms = start_setup.elapsed_time(end_setup)
        prefill_ms = start_prefill.elapsed_time(end_prefill)
        decode_ms = end_prefill.elapsed_time(end_decode)
        timing = Timing(
            setup_ms=setup_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=total_ms,
            host_overhead_ms=max(0.0, total_ms - setup_ms - prefill_ms - decode_ms),
            tokens_per_second=self.max_new_tokens * 1000.0 / total_ms,
            peak_vram_mib=torch.cuda.max_memory_allocated() / (1024**2),
        )
        return GenerationResult(timing=timing, token_ids=token_ids)


def benchmark(
    model,
    variant: Variant,
    prompt: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
    warmup: int,
    repetitions: int,
    profile_kernels: bool = False,
) -> tuple[dict[str, object], PassReport]:
    pass_report = apply_passes(model, variant.passes)
    generator = Generator(model, variant, context_length, max_new_tokens)
    generator.warmup(prompt, warmup)
    results = [generator.run_once(prompt) for _ in range(repetitions)]
    token_ids = results[0].token_ids
    deterministic = all(result.token_ids == token_ids for result in results)
    if not deterministic:
        raise RuntimeError("greedy token IDs changed between repetitions")
    timings = [asdict(result.timing) for result in results]

    def median(field: str) -> float:
        return statistics.median(timing[field] for timing in timings)

    median_total = median("total_ms")
    summary = {
        "status": "ok",
        "variant": variant.name,
        "cache": variant.cache,
        "passes": list(variant.passes),
        "context_length": context_length,
        "max_new_tokens": max_new_tokens,
        "warmup": warmup,
        "repetitions": repetitions,
        "median_prefill_ms": median("prefill_ms"),
        "median_setup_ms": median("setup_ms"),
        "median_decode_ms": median("decode_ms"),
        "median_total_ms": median_total,
        "median_host_overhead_ms": median("host_overhead_ms"),
        "median_tokens_per_second": max_new_tokens * 1000.0 / median_total,
        "peak_vram_mib": max(timing["peak_vram_mib"] for timing in timings),
        "all_repetitions_same_tokens": deterministic,
        "token_ids": token_ids,
        "raw_timings": timings,
        "pass_report": asdict(pass_report),
    }
    if profile_kernels:
        from collections import Counter
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
            ) as profiler:
                generator.run_once(prompt)
            events = profiler.events()
            cuda_events = [
                event
                for event in events
                if str(getattr(event, "device_type", "")).lower().endswith("cuda")
            ]
            names = Counter(event.name for event in cuda_events)
            summary["profiler"] = {
                "status": "ok",
                "cuda_activity_events": len(cuda_events),
                "cpu_operator_events": len(events) - len(cuda_events),
                "top_cuda_activities": names.most_common(25),
                "note": "CUDA activity count includes kernels and memory operations; use Nsight for launch-level audit.",
            }
        except Exception as exc:
            summary["profiler"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "note": "Timing and exactness remain valid; use profile_nsys.sh for launch-level audit.",
            }
    return summary, pass_report
