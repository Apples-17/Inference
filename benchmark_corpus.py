#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from qwen_opt.benchmarking import (
    WorkItem,
    build_work_items,
    file_hash,
    parse_int_list,
    summarize_repetitions,
    summarize_rows,
    token_hash,
)
from qwen_opt.passes import apply_passes
from qwen_opt.prompts import DEFAULT_PROMPTS_FILE, load_prompt_cases
from qwen_opt.runtime import Generator, environment, exact_length_prompt, load_model_and_tokenizer
from qwen_opt.variants import get_variant


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one HF/custom engine on the frozen corpus.")
    parser.add_argument("--variant", required=True)
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.limit < 1 or args.warmup < 0 or args.repetitions < 1:
        raise ValueError("limit/repetitions must be positive and warmup must be non-negative")
    cases = load_prompt_cases(args.prompts_file)
    if args.limit > len(cases):
        raise ValueError(f"requested {args.limit} prompts, corpus contains {len(cases)}")
    cases = cases[: args.limit]
    mixed_outputs = parse_int_list(args.mixed_output_lengths)
    fixed_contexts = parse_int_list(args.fixed_context_lengths) if args.fixed_context_lengths else []
    stress_contexts = parse_int_list(args.stress_context_lengths) if args.stress_context_lengths else []
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
    variant = get_variant(args.variant)
    model, tokenizer = load_model_and_tokenizer(args.model, args.attention)
    if args.prompt_mode == "natural":
        items = [
            replace(
                item,
                context_length=len(tokenizer.encode(item.prompt, add_special_tokens=False)),
                bucket="natural",
            )
            for item in items
        ]
    pass_report = apply_passes(model, variant.passes)

    def encode_prompt(text: str, context_length: int) -> torch.Tensor:
        if args.prompt_mode == "exact_length":
            return exact_length_prompt(tokenizer, text, context_length)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("prompt produced no tokens")
        return torch.tensor([ids], dtype=torch.long, device="cuda")

    grouped: dict[tuple[int, int], list[WorkItem]] = defaultdict(list)
    for item in items:
        grouped[(item.context_length, item.output_length)].append(item)
    rows = []
    for group_number, ((context_length, output_length), group) in enumerate(sorted(grouped.items()), 1):
        print(
            f"[{group_number}/{len(grouped)}] {variant.name}: "
            f"context={context_length}, output={output_length}, prompts={len(group)}",
            flush=True,
        )
        generator = Generator(model, variant, context_length, output_length)
        warm_prompt = encode_prompt(group[0].prompt, context_length)
        generator.warmup(warm_prompt, args.warmup)
        for item in group:
            prompt = encode_prompt(item.prompt, context_length)
            prompt_ids = prompt[0].cpu().tolist()
            results = [generator.run_once(prompt) for _ in range(args.repetitions)]
            generated_ids = results[0].token_ids
            if any(result.token_ids != generated_ids for result in results[1:]):
                raise RuntimeError(f"greedy output changed between repetitions for {item.id}")
            raw_timings = [asdict(result.timing) for result in results]
            rows.append(
                {
                    "id": item.id,
                    "prompt": item.prompt,
                    "workload": item.workload,
                    "bucket": item.bucket,
                    "context_length": context_length,
                    "output_length": output_length,
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": token_hash(prompt_ids),
                    "token_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "generated_token_ids_sha256": token_hash(generated_ids),
                    "raw_timings": raw_timings,
                    "summary": summarize_repetitions(raw_timings),
                }
            )
        del generator
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 3,
        "benchmark_name": args.benchmark_name,
        "engine": "hf_custom",
        "variant": variant.name,
        "model": args.model,
        "precision": "float16",
        "batch_size": 1,
        "decoding": "greedy_argmax",
        "ignore_eos": True,
        "warmup_per_shape": args.warmup,
        "timed_repetitions_per_prompt": args.repetitions,
        "timing": {
            "headline": "CUDA-synchronized wall clock",
            "device_phases": "CUDA events for cache setup, prefill, and decode; no synchronization barrier inserted between phases",
            "host_overhead": "wall total minus setup/prefill/decode CUDA event time",
        },
        "prompts_file": str(args.prompts_file),
        "prompt_mode": args.prompt_mode,
        "prompts_file_sha256": file_hash(args.prompts_file),
        "environment": environment(args.model),
        "pass_report": asdict(pass_report),
        "conditions": summarize_rows(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
