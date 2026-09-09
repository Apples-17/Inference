# Lossless Qwen3-4B inference on a Tesla T4

This repository explores a simple question: how much can the single-request
decode path of `Qwen/Qwen3-4B` be accelerated without changing the tokens it
generates?

The implementation tests a set of execution-level optimizations, rejects any
candidate that changes the frozen reference output, and benchmarks the fastest
surviving candidate against vLLM. Model weights, architecture dimensions,
precision, and greedy decoding are unchanged.

## Result

The included run was measured on a Tesla T4 using all 80 first-turn MT-Bench
prompts and 256 generated tokens per prompt.

| Metric | Custom decode path | vLLM |
|---|---:|---:|
| Median prompt latency | 10.993 s | 11.892 s |
| Fastest prompt latency | 10.930 s | 11.027 s |
| Slowest prompt latency | 11.652 s | 16.531 s |
| Effective throughput | 23.166 tok/s | 20.835 tok/s |

| Speedup statistic | Result |
|---|---:|
| Aggregate | **1.112x** |
| Lowest paired | 1.009x |
| Median paired | 1.082x |
| Best paired | 1.419x |
| Exact outputs vs frozen reference | **80/80** |

The selected implementation was `p10_manual_cudagraph`. `p9_compile` was
faster during the eight-prompt screen, but it matched only 63 of 80 reference
outputs during full validation, so it was rejected automatically.

The complete tables are available in
[`results/final_mt_bench/final_report.md`](results/final_mt_bench/final_report.md).

## What “lossless” means here

Lossless means exact generated-token equality with a frozen output produced by
the unmodified Transformers model using a static KV cache. The check also
requires matching model, precision, prompt mode, and prompt-file hash.

vLLM is used as the performance baseline, not as the correctness oracle. In the
included comparison, 61/80 outputs were accepted as stable direct matches
between the custom model and vLLM, while the custom model matched the frozen
Transformers reference on 80/80. This distinction matters because small
implementation-level floating point differences can change a close greedy
argmax decision.

See [`docs/lossless-validation.md`](docs/lossless-validation.md) for the gate
design and the full-selection result.

## Benchmark configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| GPU | Tesla T4 |
| Precision | FP16 |
| Batch size | 1 |
| Decoding | Greedy argmax, EOS ignored |
| Prompt set | 80 first-turn MT-Bench prompts |
| Generated tokens | 256 per prompt |
| Warmup | 3 runs per shape |
| Timed repetitions | 5 per prompt |
| Timing | CUDA-synchronized end-to-end wall time |

The aggregate speedup is calculated as:

```text
sum(vLLM per-prompt median latency)
-----------------------------------
sum(custom per-prompt median latency)
```

## Quick start

The code was validated with Python 3.12, PyTorch 2.8.0, Transformers 4.57.6,
CUDA 12.8, and vLLM 0.10.2.

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-vllm.txt
python run_final_mt_bench.py
```

If vLLM's dependency resolver conflicts with the pinned PyTorch version, run
the vLLM benchmark in a separate environment. The two implementations exchange
results through JSON files.

The runner resumes from compatible completed files. To discard saved runs and
benchmark everything again:

```bash
python run_final_mt_bench.py --fresh
```

## What the runner does

`run_final_mt_bench.py` executes the complete workflow:

1. Screen every optimization pass on the same eight-prompt MT-Bench subset.
2. Save the raw result and lossless status for each pass.
3. Run one shared vLLM screen and calculate the per-pass speedup.
4. Rank only candidates that are exact on the screening subset.
5. Validate ranked candidates against all 80 frozen reference outputs.
6. Benchmark only the first full-validation winner and vLLM using the final
   warmup and repetition settings.
7. Write machine-readable JSON, CSV summaries, and Markdown tables.

No Transformers baseline timing is included in the final performance table;
the frozen Transformers output is used only for correctness.

## Optimization passes

| ID | Variant | Change |
|---|---|---|
| P1 | `lossless_safe` | Reuse runner-owned token, position, and result buffers |
| P2a | `p2_packed_qkv` | Pack Q, K, and V projections |
| P2b | `p2_packed_gate_up` | Pack gate and up projections |
| P3 | `p3_fused_rmsnorm_residual` | Fuse RMSNorm and residual work |
| P4 | `p4_fused_qk_rope` | Fuse Q/K normalization and RoPE |
| P5 | `p5_fused_kv_write` | Fuse static-cache K/V writes |
| P6 | `p6_fused_gqa_attention` | Use native grouped-query attention |
| P7 | `p7_fused_swiglu` | Fuse SiLU and elementwise multiplication |
| P8 | `p8_fused_lm_head_argmax` | Fuse LM-head projection and argmax |
| P9 | `p9_compile` | Compile the decode step with TorchInductor |
| P10 | `p10_manual_cudagraph` | Replay one-token decode with a manual CUDA graph |

The passes are screened independently on top of the shared static-cache runner.
Compilation and manual CUDA graph capture are intentionally separate because
they compete for graph ownership.

## Output layout

```text
results/final_mt_bench/
├── pass_sweep/
│   ├── raw/                 # one JSON result per pass
│   ├── summary.csv
│   └── summary.json
├── full_gate/               # full-corpus candidate validation
├── final/
│   ├── p10_manual_cudagraph.json
│   ├── vllm.json
│   ├── lossless_gate.json
│   ├── comparison.json
│   └── comparison.md
├── selected_variant.txt
├── final_report.json
└── final_report.md
```

## Repository layout

| Path | Purpose |
|---|---|
| `qwen_opt/` | Optimization passes, Triton kernels, runtime, and variant registry |
| `run_final_mt_bench.py` | End-to-end screening, selection, validation, and final benchmark |
| `benchmark_corpus.py` | Custom decode-path benchmark |
| `benchmark_vllm_corpus.py` | Matching vLLM benchmark |
| `check_lossless.py` | Exact-token correctness gate |
| `compare_benchmarks.py` | Two-engine comparison and report generation |
| `data/` | Frozen MT-Bench prompts and reference outputs |
| `tests/` | Registry, kernel, cache, benchmark, and correctness tests |
| `results/` | Included Tesla T4 run |

## Tests

CPU-safe tests can be run with:

```bash
pytest -q
```

CUDA and Triton kernel tests require a compatible NVIDIA GPU and are skipped or
reported as unsupported when their hardware requirements are not available.

## Scope and limitations

- These results apply to a Tesla T4, FP16, batch size 1, and fixed-length greedy
  generation. They should not be generalized to serving throughput or other
  GPUs without rerunning the benchmark.
- MT-Bench is used as a prompt corpus. This repository does not report an
  MT-Bench judge score or make a model-quality claim.
- Ignoring EOS makes every engine perform the same amount of decode work, but it
  does not represent every production workload.
- The short pass screen is a selection tool. Only the final 80-prompt run is
  used for the reported speedup.
