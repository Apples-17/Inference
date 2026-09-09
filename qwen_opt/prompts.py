from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


TOPICS = (
    "memory bandwidth in autoregressive decoding",
    "grouped-query attention",
    "rotary position embeddings",
    "static key-value caches",
    "kernel launch overhead",
    "RMS normalization",
    "SwiGLU feed-forward networks",
    "CUDA graph replay",
    "greedy decoding correctness",
    "roofline performance analysis",
    "prefill versus decode workloads",
    "tensor-core matrix multiplication",
    "GPU memory coalescing",
    "attention masking",
    "weight packing",
    "operator fusion",
    "language-model output heads",
    "benchmark reproducibility",
    "latency percentiles",
    "numerical equivalence in FP16",
)

TASKS = (
    "Explain {topic} in three precise sentences.",
    "Give two benefits and one limitation of {topic}.",
    "Write a compact technical checklist for {topic}.",
    "Contrast a naive and optimized implementation of {topic}.",
    "State the key invariant needed to optimize {topic} losslessly.",
)


def benchmark_prompts() -> list[str]:
    prompts = [template.format(topic=topic) for topic in TOPICS for template in TASKS]
    assert len(prompts) == 100
    return prompts


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str
    context_length: int
    bucket: str


DEFAULT_PROMPTS_FILE = Path(__file__).resolve().parents[1] / "data" / "prompts_100.jsonl"


def load_prompt_cases(path: Path = DEFAULT_PROMPTS_FILE) -> list[PromptCase]:
    """Load the frozen benchmark corpus and reject malformed/duplicate rows."""
    rows: list[PromptCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                row = PromptCase(
                    id=str(raw["id"]),
                    prompt=str(raw["prompt"]),
                    context_length=int(raw["context_length"]),
                    bucket=str(raw["bucket"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prompt row {line_number}: {exc}") from exc
            if not row.prompt or row.context_length < 1:
                raise ValueError(f"invalid prompt row {line_number}")
            rows.append(row)
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    if not rows:
        raise ValueError("prompt corpus is empty")
    return rows
