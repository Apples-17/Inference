#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm

from qwen_opt.benchmarking import (
    WorkItem,
    build_work_items,
    distribution,
    file_hash,
    parse_int_list,
    summarize_repetitions,
    summarize_rows,
    token_hash,
)
from qwen_opt.prompts import DEFAULT_PROMPTS_FILE, load_prompt_cases


def exact_ids(tokenizer, text: str, context_length: int) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("prompt produced no tokens")
    return (ids * ((context_length + len(ids) - 1) // len(ids)))[:context_length]


def sampling_params(output_length: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=output_length,
        min_tokens=output_length,
        ignore_eos=True,
        detokenize=False,
    )


def timed_generate(engine, requests: list[dict], params: SamplingParams):
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(requests, params, use_tqdm=False)
    torch.cuda.synchronize()
    return outputs, (time.perf_counter() - start) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark vLLM on the frozen corpus.")
    parser.add_argument("--benchmark-name", default="controlled_100")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--prompt-mode", choices=("natural", "exact_length"), default="exact_length")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mixed-output-lengths", default="64")
    parser.add_argument("--fixed-context-lengths", default="")
    parser.add_argument("--fixed-context-output-length", type=int, default=64)
    parser.add_argument("--fixed-context-prompts", type=int, default=0)
    parser.add_argument("--stress-context-lengths", default="")
    parser.add_argument("--stress-output-length", type=int, default=256)
    parser.add_argument("--stress-prompts", type=int, default=0)
    parser.add_argument("--concurrency-levels", default="")
    parser.add_argument("--serving-output-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_prompt_cases(args.prompts_file)
    if args.limit < 1 or args.limit > len(cases):
        raise ValueError(f"limit must be between 1 and {len(cases)}")
    cases = cases[: args.limit]
    mixed_outputs = parse_int_list(args.mixed_output_lengths)
    fixed_contexts = parse_int_list(args.fixed_context_lengths) if args.fixed_context_lengths else []
    stress_contexts = parse_int_list(args.stress_context_lengths) if args.stress_context_lengths else []
    concurrency_levels = parse_int_list(args.concurrency_levels) if args.concurrency_levels else []
    items = build_work_items(
        cases,
        mixed_outputs,
        fixed_contexts,
        args.fixed_context_output_length,
        args.fixed_context_prompts,
        stress_contexts,
        args.stress_output_length,
        args.stress_prompts,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.prompt_mode == "natural":
        items = [
            replace(
                item,
                context_length=len(tokenizer.encode(item.prompt, add_special_tokens=False)),
                bucket="natural",
            )
            for item in items
        ]

    def prompt_ids(text: str, context_length: int) -> list[int]:
        if args.prompt_mode == "exact_length":
            return exact_ids(tokenizer, text, context_length)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("prompt produced no tokens")
        return ids

    max_model_len = max(item.context_length + item.output_length for item in items)
    if concurrency_levels:
        max_model_len = max(
            max_model_len,
            max(case.context_length for case in cases) + args.serving_output_length,
        )
    max_num_seqs = max(concurrency_levels, default=1)
    engine = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    grouped: dict[tuple[int, int], list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[(item.context_length, item.output_length)].append(item)
    rows = []
    for group_number, ((context_length, output_length), group) in enumerate(sorted(grouped.items()), 1):
        print(
            f"[{group_number}/{len(grouped)}] vLLM: "
            f"context={context_length}, output={output_length}, prompts={len(group)}",
            flush=True,
        )
        params = sampling_params(output_length)
        warm_ids = prompt_ids(group[0].prompt, context_length)
        warm_request = [{"prompt_token_ids": warm_ids}]
        for _ in range(args.warmup):
            timed_generate(engine, warm_request, params)
        for item in group:
            encoded_prompt_ids = prompt_ids(item.prompt, context_length)
            request = [{"prompt_token_ids": encoded_prompt_ids}]
            timings = []
            generated_sequences = []
            for _ in range(args.repetitions):
                outputs, total_ms = timed_generate(engine, request, params)
                tokens = list(outputs[0].outputs[0].token_ids)
                generated_sequences.append(tokens)
                timings.append(
                    {
                        "total_ms": total_ms,
                        "tokens_per_second": output_length * 1000.0 / total_ms,
                    }
                )
            generated_ids = generated_sequences[0]
            repeat_hashes = [token_hash(tokens) for tokens in generated_sequences]
            rows.append(
                {
                    "id": item.id,
                    "prompt": item.prompt,
                    "workload": item.workload,
                    "bucket": item.bucket,
                    "context_length": context_length,
                    "output_length": output_length,
                    "prompt_token_count": len(encoded_prompt_ids),
                    "prompt_token_ids_sha256": token_hash(encoded_prompt_ids),
                    "token_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "generated_token_ids_sha256": token_hash(generated_ids),
                    "stable_across_repetitions": len(set(repeat_hashes)) == 1,
                    "unique_output_count": len(set(repeat_hashes)),
                    "repetition_output_sha256": repeat_hashes,
                    "raw_timings": timings,
                    "summary": summarize_repetitions(timings),
                }
            )

    serving = []
    serving_cases = [
        (case, prompt_ids(case.prompt, case.context_length)) for case in cases
    ]
    serving_params = sampling_params(args.serving_output_length)
    for concurrency in concurrency_levels:
        selected = [serving_cases[index % len(serving_cases)] for index in range(concurrency)]
        requests = [{"prompt_token_ids": ids} for _, ids in selected]
        for _ in range(args.warmup):
            timed_generate(engine, requests, serving_params)
        batch_times = []
        request_latencies = []
        time_to_first_token = []
        for _ in range(args.repetitions):
            outputs, batch_ms = timed_generate(engine, requests, serving_params)
            batch_times.append(batch_ms)
            for output in outputs:
                metrics = getattr(output, "metrics", None)
                arrival = getattr(metrics, "arrival_time", None)
                finished = getattr(metrics, "finished_time", None)
                first = getattr(metrics, "first_token_time", None)
                if arrival is not None and finished is not None:
                    request_latencies.append((finished - arrival) * 1000.0)
                if arrival is not None and first is not None:
                    time_to_first_token.append((first - arrival) * 1000.0)
        median_batch_ms = statistics.median(batch_times)
        row = {
            "concurrency": concurrency,
            "output_length": args.serving_output_length,
            "batch_wall_ms": distribution(batch_times),
            "aggregate_tokens_per_second": (
                concurrency * args.serving_output_length * 1000.0 / median_batch_ms
            ),
        }
        if request_latencies:
            row["request_latency_ms"] = distribution(request_latencies)
        if time_to_first_token:
            row["time_to_first_token_ms"] = distribution(time_to_first_token)
        serving.append(row)

    payload = {
        "schema_version": 3,
        "benchmark_name": args.benchmark_name,
        "engine": "vllm",
        "variant": "vllm",
        "model": args.model,
        "precision": "float16",
        "batch_size": 1,
        "decoding": "greedy_temperature_0",
        "ignore_eos": True,
        "warmup_per_shape": args.warmup,
        "timed_repetitions_per_prompt": args.repetitions,
        "timing": {
            "headline": "CUDA-synchronized wall clock around LLM.generate",
            "prefill_decode": "not reported by the stable offline vLLM API",
        },
        "prompts_file": str(args.prompts_file),
        "prompt_mode": args.prompt_mode,
        "prompts_file_sha256": file_hash(args.prompts_file),
        "environment": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
        },
        "conditions": summarize_rows(rows),
        "serving": serving,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
