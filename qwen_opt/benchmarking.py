from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .prompts import PromptCase


@dataclass(frozen=True)
class WorkItem:
    id: str
    prompt: str
    context_length: int
    output_length: int
    bucket: str
    workload: str


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def build_work_items(
    cases: list[PromptCase],
    mixed_output_lengths: Iterable[int],
    fixed_context_lengths: Iterable[int] = (),
    fixed_context_output_length: int = 64,
    fixed_context_prompts: int = 0,
    stress_context_lengths: Iterable[int] = (),
    stress_output_length: int = 256,
    stress_prompts: int = 0,
) -> list[WorkItem]:
    items = [
        WorkItem(
            id=case.id,
            prompt=case.prompt,
            context_length=case.context_length,
            output_length=output_length,
            bucket=case.bucket,
            workload="mixed_context",
        )
        for output_length in mixed_output_lengths
        for case in cases
    ]
    selected = cases[:fixed_context_prompts]
    items.extend(
        WorkItem(
            id=case.id,
            prompt=case.prompt,
            context_length=context_length,
            output_length=fixed_context_output_length,
            bucket=str(context_length),
            workload="context_scaling",
        )
        for context_length in fixed_context_lengths
        for case in selected
    )
    stress_selected = cases[:stress_prompts]
    items.extend(
        WorkItem(
            id=case.id,
            prompt=case.prompt,
            context_length=context_length,
            output_length=stress_output_length,
            bucket=str(context_length),
            workload="stress",
        )
        for context_length in stress_context_lengths
        for case in stress_selected
    )
    return items


def token_hash(token_ids: Iterable[int]) -> str:
    encoded = ",".join(map(str, token_ids)).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        raise ValueError("cannot summarize an empty list")
    return {
        "count": len(data),
        "median": statistics.median(data),
        "p25": percentile(data, 0.25),
        "p75": percentile(data, 0.75),
        "mean": statistics.mean(data),
        "stddev": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
    }


def summarize_repetitions(timings: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    if not timings:
        raise ValueError("cannot summarize zero repetitions")
    fields = sorted(set.intersection(*(set(timing) for timing in timings)))
    return {
        field: distribution(float(timing[field]) for timing in timings)
        for field in fields
        if all(timing[field] is not None for timing in timings)
    }


def _condition_key(row: dict) -> tuple[str, str, int]:
    if row["workload"] == "mixed_context":
        return row["workload"], "mixed", int(row["output_length"])
    return row["workload"], str(row["context_length"]), int(row["output_length"])


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[_condition_key(row)].append(row)
    conditions = []
    for (workload, context, output_length), group in sorted(groups.items()):
        total_values = [float(row["summary"]["total_ms"]["median"]) for row in group]
        total_tokens = len(group) * output_length
        total_median_time = sum(total_values)
        summary: dict[str, object] = {
            "workload": workload,
            "context_length": context,
            "output_length": output_length,
            "prompt_count": len(group),
            "total_ms": distribution(total_values),
            "aggregate_tokens_per_second": total_tokens * 1000.0 / total_median_time,
        }
        for field in ("setup_ms", "prefill_ms", "decode_ms", "host_overhead_ms"):
            values = [
                float(row["summary"][field]["median"])
                for row in group
                if field in row["summary"]
            ]
            if values:
                summary[field] = distribution(values)
        if workload == "mixed_context":
            bucket_groups: dict[str, list[float]] = defaultdict(list)
            for row in group:
                bucket_groups[row["bucket"]].append(
                    float(row["summary"]["total_ms"]["median"])
                )
            summary["context_buckets"] = {
                bucket: distribution(values)
                for bucket, values in sorted(bucket_groups.items())
            }
        conditions.append(summary)
    return conditions
