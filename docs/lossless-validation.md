# Lossless validation

The benchmark separates correctness from performance. A candidate is allowed
into the final comparison only after it reproduces the frozen Transformers
reference token for token.

## Why vLLM is not the token oracle

vLLM is the latency baseline, but different kernels and reduction orders can
produce slightly different FP16 values. When two vocabulary logits are close,
that small numerical difference can change the greedy argmax token and all
subsequent tokens.

The included comparison accepted 61/80 outputs as stable direct matches between
the optimized model and vLLM. Both engines are still evaluated fairly for
latency because they use the same model checkpoint, precision, prompts, batch
size, output length, warmup count, and repetition count. Correctness is
established separately against the frozen unmodified Transformers output.

## Two-stage gate

We uses two correctness checks:

1. **Screening gate:** each pass generates 64 tokens for the same eight
   MT-Bench prompts. A single mismatch removes that pass from consideration.
2. **Full gate:** surviving candidates are ranked by screening latency and then
   checked on all 80 prompts with 256 generated tokens. The first 80/80
   candidate becomes the final benchmark candidate.

The exact check includes:

- prompt token hash;
- generated token IDs;
- model name;
- precision;
- prompt mode; and
- prompt-file hash.

If the selected candidate fails the gate again during the final timed run, the
runner stops before reporting a speedup.

## Included selection result

| Candidate | Screening result | Full result | Decision |
|---|---:|---:|---|
| `p9_compile` | 8/8 | 63/80 | Rejected |
| `p10_manual_cudagraph` | 8/8 | **80/80** | Selected |

This is why the repository reports the manual CUDA graph result even though
TorchInductor was faster during the short screen.
