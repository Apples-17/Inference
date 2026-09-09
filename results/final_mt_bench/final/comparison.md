# Optimized model vs vLLM

Benchmark: `mt_bench`

Observed speedup: **1.112x**. Lossless reference parity: **80/80**. Validated lossless speedup: **1.112x**.

| Result | Optimized model | vLLM |
|---|---:|---:|
| Median prompt latency | 10992.718 ms | 11891.591 ms |
| Fastest prompt latency | 10930.303 ms | 11027.299 ms |
| Slowest prompt latency | 11651.916 ms | 16530.931 ms |
| Summed per-prompt median latency | 884054.618 ms | 982939.395 ms |
| Effective throughput | 23.166 tok/s | 20.835 tok/s |
| Generated tokens | 20480 | 20480 |

Direct optimized/vLLM token match is 61/80; this is diagnostic because vLLM is the latency baseline, not the correctness oracle.

Observed speedup is credited only when the optimized output exactly matches the stable unoptimized-model token reference and both timed engines use matching methodology.

## Speedup distribution

| Aggregate | Lowest paired | Median paired | Best paired |
|---:|---:|---:|---:|
| 1.112x | 1.009x | 1.082x | 1.419x |

## By workload

| Workload | Context | Output | Exact outputs | Optimized ms | vLLM ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| mixed_context | mixed | 256 | 61/80 | 884054.618 | 982939.395 | 1.112x |

## Optimized model time breakdown

| Phase | Summed median ms | Share |
|---|---:|---:|
| setup | 40.938 | 0.00% |
| prefill | 5344.790 | 0.60% |
| decode | 878641.883 | 99.39% |
| host overhead | 12.069 | 0.00% |
