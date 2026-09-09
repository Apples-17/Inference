# Final MT-Bench inference report

Selected pass: **`p10_manual_cudagraph`**. Lossless: **80/80 PASS**. Aggregate speedup vs vLLM: **1.112x**.

## Pass screening

Each pass used the same MT-Bench subset and one shared vLLM screen. Only prefix-lossless passes were eligible for full-corpus validation.

| Pass | Status | Exact | Median ms | tok/s | Speedup vs vLLM |
|---|---|---:|---:|---:|---:|
| P1 runner buffers | ok | 8/8 | 3677.029 | 18.009 | 0.777x |
| P2a packed QKV | ok | 7/8 | 2720.969 | 22.476 | 0.970x |
| P2b packed gate/up | ok | 8/8 | 2927.591 | 20.509 | 0.885x |
| P3 fused RMSNorm/residual | ok | 8/8 | 3422.492 | 19.465 | 0.840x |
| P4 fused QK-Norm/RoPE | ok | 7/8 | 2844.673 | 22.990 | 0.992x |
| P5 fused KV write | ok | 8/8 | 3879.667 | 17.506 | 0.755x |
| P6 native GQA attention | ok | 8/8 | 2981.982 | 21.311 | 0.919x |
| P7 fused SwiGLU | ok | 7/8 | 4015.525 | 16.242 | 0.701x |
| P8 fused LM-head argmax | ok | 7/8 | 2841.920 | 22.123 | 0.954x |
| P9 torch.compile | ok | 8/8 | 2280.143 | 28.025 | 1.209x |
| P10 manual CUDA graph | ok | 8/8 | 2617.672 | 24.401 | 1.053x |

## Final engine comparison

| Metric | Optimized model (`p10_manual_cudagraph`) | vLLM |
|---|---:|---:|
| Median prompt latency | 10992.718 ms | 11891.591 ms |
| Fastest prompt latency | 10930.303 ms | 11027.299 ms |
| Slowest prompt latency | 11651.916 ms | 16530.931 ms |
| Summed per-prompt median latency | 884054.618 ms | 982939.395 ms |
| Effective throughput | 23.166 tok/s | 20.835 tok/s |

## Speedup vs vLLM

| Aggregate | Lowest paired | Median paired | Best paired |
|---:|---:|---:|---:|
| 1.112x | 1.009x | 1.082x | 1.419x |

Aggregate speedup is the sum of vLLM per-prompt median latencies divided by the corresponding optimized-model sum.
