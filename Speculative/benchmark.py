# compare output + latency + tokens/sec + acceptance rate

# benchmark.py

import time
import torch

from speculative import (
    tokenizer,
    target_model,
    speculative_generate,
    DEVICE,
    GAMMA,
)


# CONFIG
PROMPT = "Explain why the sky appears blue during the day."
MAX_NEW_TOKENS = 100
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


# BASELINE GREEDY DECODING
# We use the SAME target model already loaded by speculative.py.
# This avoids loading Qwen3-4B twice. (Coz otherwise if we import them from baseline as well, then model will load twice... which is dangerous for VRAM)

@torch.inference_mode()
def baseline_generate(prompt, max_new_tokens=100):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        outputs= target_model(
            input_ids=generated_ids,
            use_cache=False,
        )

        next_logits =outputs.logits[:, -1, :]

        next_token=torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=-1,
        )

        # EOS handling
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_ids


# TIMING FUNCTION
def benchmark_function(
    function,
    name,
    prompt,
    max_new_tokens,
    runs,
    **kwargs,
):

    times = []
    generated_token_counts = []
    for run in range(runs):
        torch.cuda.synchronize()
        start =time.perf_counter()
        output_ids =function(
            prompt,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        # GPU kernels are asynchronous.
        # We MUST synchronize before stopping the timer.
        torch.cuda.synchronize()

        end = time.perf_counter()
        elapsed = end - start

        # Count prompt tokens
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids

        num_generated = (
            output_ids.shape[1]
            - prompt_ids.shape[1]
        )

        times.append(elapsed)
        generated_token_counts.append(num_generated)

        print(
            f"{name} | "
            f"Run {run + 1}/{runs} | "
            f"{elapsed:.4f} sec | "
            f"{num_generated} tokens"
        )

    average_time = sum(times) / len(times)

    average_tokens = (
        sum(generated_token_counts)
        / len(generated_token_counts)
    )

    latency_per_token = (
        average_time / average_tokens
    )

    tokens_per_second = (
        average_tokens / average_time
    )

    return {
        "name": name,
        "average_time": average_time,
        "average_tokens": average_tokens,
        "latency_per_token": latency_per_token,
        "tokens_per_second": tokens_per_second,
        "output_ids": output_ids,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    print("SPECULATIVE DECODING BENCHMARK")
    print("Prompt:")
    print(PROMPT)

    print("Max new tokens:", MAX_NEW_TOKENS)
    print("Gamma:", GAMMA)

    # WARMUP
    print("WARMUP")

    for i in range(WARMUP_RUNS):
        print(f"Warmup {i + 1}/{WARMUP_RUNS}")
        _ = baseline_generate(
            PROMPT,
            max_new_tokens=10,
        )

        _ = speculative_generate(
            PROMPT,
            max_new_tokens=10,
            gamma=GAMMA,
        )

    torch.cuda.synchronize()

    # CORRECTNESS CHECK
    print("CORRECTNESS CHECK")

    baseline_output = baseline_generate(
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    speculative_output = speculative_generate(
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        gamma=GAMMA,
    )

    exact_match = torch.equal(
        baseline_output,
        speculative_output,
    )

    print("Exact token match:", exact_match)

    if not exact_match:
        print(
            "WARNING: Speculative output does not "
            "match baseline."
        )
        print("Do NOT trust benchmark results yet.")

    else:
        print("Outputs match exactly.")

    # BASELINE BENCHMARK
    print("BASELINE")
    baseline_stats = benchmark_function(
        baseline_generate,
        name="Baseline",
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        runs=BENCHMARK_RUNS,
    )

    # SPECULATIVE BENCHMARK
    print("SPECULATIVE")
    speculative_stats = benchmark_function(
        speculative_generate,
        name="Speculative",
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        runs=BENCHMARK_RUNS,
        gamma=GAMMA,
    )

    # FINAL RESULTS
    baseline_tps = baseline_stats[
        "tokens_per_second"
    ]

    speculative_tps = speculative_stats[
        "tokens_per_second"
    ]

    speedup = (
        speculative_tps / baseline_tps
    )

    print("FINAL RESULTS")
    print("Baseline:")
    print(
        f"Average time       : "
        f"{baseline_stats['average_time']:.4f} sec"
    )
    print(
        f"Latency/token      : "
        f"{baseline_stats['latency_per_token'] * 1000:.3f} ms"
    )
    print(
        f"Tokens/sec         : "
        f"{baseline_tps:.2f}"
    )

    print("Speculative:")
    print(
        f"Average time       : "
        f"{speculative_stats['average_time']:.4f} sec"
    )
    print(
        f"Latency/token      : "
        f"{speculative_stats['latency_per_token'] * 1000:.3f} ms"
    )
    print(
        f"Tokens/sec         : "
        f"{speculative_tps:.2f}"
    )
    print(
        f"SPEEDUP            : "
        f"{speedup:.3f}x"
    )

    if speedup > 1:
        print("Speculative decoding is faster.")
    else:
        print("Speculative decoding is currently slower.")


    # OUTPUT

    print("Generated output:")

    print(
        tokenizer.decode(
            baseline_stats["output_ids"][0],
            skip_special_tokens=True,
        )
    )