#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from check_lossless import compare as compare_lossless
from qwen_opt.benchmarking import distribution


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict) -> tuple[str, str, int, int]:
    return (
        row["workload"],
        row["id"],
        int(row["context_length"]),
        int(row["output_length"]),
    )


def condition_key(row: dict) -> tuple[str, str, int]:
    context = "mixed" if row["workload"] == "mixed_context" else str(row["context_length"])
    return row["workload"], context, int(row["output_length"])


def methodology_comparison(vllm: dict, candidate: dict) -> dict:
    fields = (
        "model",
        "precision",
        "batch_size",
        "ignore_eos",
        "warmup_per_shape",
        "timed_repetitions_per_prompt",
        "prompts_file_sha256",
        "prompt_mode",
    )
    result = {
        field: {
            "vllm": vllm.get(field),
            "optimized": candidate.get(field),
            "match": vllm.get(field) is not None and vllm.get(field) == candidate.get(field),
        }
        for field in fields
    }
    vllm_gpu = vllm.get("environment", {}).get("gpu")
    optimized_gpu = candidate.get("environment", {}).get("gpu")
    result["gpu"] = {
        "vllm": vllm_gpu,
        "optimized": optimized_gpu,
        "match": vllm_gpu is not None and vllm_gpu == optimized_gpu,
    }
    return result


def optimized_phase_breakdown(candidate: dict) -> dict | None:
    fields = ("setup_ms", "prefill_ms", "decode_ms", "host_overhead_ms")
    totals = {field: 0.0 for field in fields}
    for row in candidate.get("rows", []):
        summary = row.get("summary", {})
        if any(field not in summary for field in fields):
            return None
        for field in fields:
            totals[field] += float(summary[field]["median"])
    total = sum(totals.values())
    return {
        field: {
            "summed_per_prompt_median_ms": value,
            "share": value / total if total else None,
        }
        for field, value in totals.items()
    }


def condition_comparisons(vllm: dict, candidate: dict) -> list[dict]:
    grouped: dict[tuple[str, str, int], dict[str, list[dict]]] = defaultdict(
        lambda: {"vllm": [], "optimized": []}
    )
    for row in vllm["rows"]:
        grouped[condition_key(row)]["vllm"].append(row)
    for row in candidate["rows"]:
        grouped[condition_key(row)]["optimized"].append(row)

    output = []
    for key, engines in sorted(grouped.items()):
        vllm_rows = {row_key(row): row for row in engines["vllm"]}
        optimized_rows = {row_key(row): row for row in engines["optimized"]}
        common = sorted(set(vllm_rows) & set(optimized_rows))
        vllm_ms = sum(float(vllm_rows[item]["summary"]["total_ms"]["median"]) for item in common)
        optimized_ms = sum(
            float(optimized_rows[item]["summary"]["total_ms"]["median"]) for item in common
        )
        exact = sum(
            vllm_rows[item]["prompt_token_ids_sha256"]
            == optimized_rows[item]["prompt_token_ids_sha256"]
            and vllm_rows[item]["token_ids"] == optimized_rows[item]["token_ids"]
            and vllm_rows[item].get("stable_across_repetitions", True)
            for item in common
        )
        token_count = sum(len(vllm_rows[item]["token_ids"]) for item in common)
        output.append(
            {
                "workload": key[0],
                "context_length": key[1],
                "output_length": key[2],
                "vllm_rows": len(vllm_rows),
                "optimized_rows": len(optimized_rows),
                "compared_outputs": len(common),
                "exact_outputs": exact,
                "vllm_summed_per_prompt_median_ms": vllm_ms,
                "optimized_summed_per_prompt_median_ms": optimized_ms,
                "vllm_tokens_per_second": token_count * 1000.0 / vllm_ms if vllm_ms else None,
                "optimized_tokens_per_second": (
                    token_count * 1000.0 / optimized_ms if optimized_ms else None
                ),
                "speedup_vs_vllm": vllm_ms / optimized_ms if optimized_ms else None,
            }
        )
    return output


def direct_vllm_comparison(vllm: dict, candidate: dict) -> dict:
    """Compare only the optimized implementation and vLLM, prompt by prompt."""
    expected = {row_key(row): row for row in vllm["rows"]}
    actual = {row_key(row): row for row in candidate["rows"]}
    keys = sorted(set(expected) & set(actual))
    details = []
    vllm_total_ms = 0.0
    candidate_total_ms = 0.0
    vllm_tokens = 0
    candidate_tokens = 0
    accepted_outputs = 0
    accepted_tokens = 0
    identical_positions = 0
    paired_speedups = []
    vllm_latencies = []
    optimized_latencies = []
    stable_vllm_outputs = 0

    for key in keys:
        baseline = expected[key]
        optimized = actual[key]
        baseline_ids = baseline["token_ids"]
        optimized_ids = optimized["token_ids"]
        stable = baseline.get("stable_across_repetitions", True)
        stable_vllm_outputs += int(stable)
        exact = (
            stable
            and baseline["prompt_token_ids_sha256"] == optimized["prompt_token_ids_sha256"]
            and baseline_ids == optimized_ids
        )
        baseline_ms = float(baseline["summary"]["total_ms"]["median"])
        optimized_ms = float(optimized["summary"]["total_ms"]["median"])
        speedup = baseline_ms / optimized_ms
        baseline_count = len(baseline_ids)
        optimized_count = len(optimized_ids)
        matching_positions = sum(left == right for left, right in zip(baseline_ids, optimized_ids))
        vllm_total_ms += baseline_ms
        candidate_total_ms += optimized_ms
        vllm_tokens += baseline_count
        candidate_tokens += optimized_count
        accepted_outputs += int(exact)
        accepted_tokens += baseline_count if exact else 0
        identical_positions += matching_positions
        paired_speedups.append(speedup)
        vllm_latencies.append(baseline_ms)
        optimized_latencies.append(optimized_ms)
        details.append(
            {
                "id": baseline["id"],
                "workload": baseline["workload"],
                "context_length": baseline["context_length"],
                "requested_output_tokens": baseline["output_length"],
                "vllm_stable_across_repetitions": stable,
                "accepted": exact,
                "vllm_generated_tokens": baseline_count,
                "optimized_generated_tokens": optimized_count,
                "accepted_tokens": baseline_count if exact else 0,
                "identical_token_positions": matching_positions,
                "vllm_output_sha256": baseline["generated_token_ids_sha256"],
                "optimized_output_sha256": optimized["generated_token_ids_sha256"],
                "vllm_median_ms": baseline_ms,
                "optimized_median_ms": optimized_ms,
                "speedup_vs_vllm": speedup,
            }
        )

    methodology = methodology_comparison(vllm, candidate)
    methodology_ok = all(item["match"] for item in methodology.values())
    expected_count = len(expected)
    all_outputs_accepted = (
        expected_count > 0
        and len(keys) == expected_count == len(actual)
        and accepted_outputs == expected_count
        and methodology_ok
    )
    raw_speedup = vllm_total_ms / candidate_total_ms if candidate_total_ms else None
    return {
        "definition": "speedup_vs_vllm = vLLM summed per-prompt median latency / optimized summed per-prompt median latency",
        "candidate": candidate.get("variant", "optimized"),
        "baseline": "vllm",
        "expected_outputs": expected_count,
        "optimized_outputs": len(actual),
        "compared_outputs": len(keys),
        "missing_optimized_outputs": [list(key) for key in sorted(set(expected) - set(actual))],
        "extra_optimized_outputs": [list(key) for key in sorted(set(actual) - set(expected))],
        "accepted_outputs": accepted_outputs,
        "rejected_outputs": expected_count - accepted_outputs,
        "output_acceptance_rate": accepted_outputs / expected_count if expected_count else 0.0,
        "stable_vllm_outputs": stable_vllm_outputs,
        "vllm_generated_tokens": vllm_tokens,
        "optimized_generated_tokens": candidate_tokens,
        "accepted_generated_tokens": accepted_tokens,
        "accepted_generated_token_rate": accepted_tokens / vllm_tokens if vllm_tokens else 0.0,
        "identical_token_positions": identical_positions,
        "identical_token_position_rate": identical_positions / vllm_tokens if vllm_tokens else 0.0,
        "methodology": methodology,
        "methodology_match": methodology_ok,
        "all_outputs_accepted": all_outputs_accepted,
        "vllm_total_median_ms": vllm_total_ms,
        "optimized_total_median_ms": candidate_total_ms,
        "vllm_tokens_per_second": vllm_tokens * 1000.0 / vllm_total_ms if vllm_total_ms else None,
        "optimized_tokens_per_second": (
            candidate_tokens * 1000.0 / candidate_total_ms if candidate_total_ms else None
        ),
        "speedup_vs_vllm": raw_speedup,
        "credited_speedup_vs_vllm": raw_speedup if all_outputs_accepted else None,
        "paired_prompt_speedup": distribution(paired_speedups) if paired_speedups else None,
        "vllm_prompt_latency_ms": distribution(vllm_latencies) if vllm_latencies else None,
        "optimized_prompt_latency_ms": (
            distribution(optimized_latencies) if optimized_latencies else None
        ),
        "optimized_phase_breakdown": optimized_phase_breakdown(candidate),
        "per_output": details,
    }


def markdown(report: dict) -> str:
    direct = report["optimized_vs_vllm"]
    gate = report["lossless_gate"]
    observed = direct["speedup_vs_vllm"]
    credited = direct["credited_speedup_vs_vllm"]
    observed_text = f"{observed:.3f}x" if observed is not None else "unavailable"
    credited_text = f"{credited:.3f}x" if credited is not None else "not credited"
    lines = [
        "# Optimized model vs vLLM",
        "",
        f"Benchmark: `{report['benchmark_name']}`",
        "",
        f"Observed speedup: **{observed_text}**. Lossless reference parity: "
        f"**{gate['exact_outputs']}/{gate['expected_outputs']}**. "
        f"Validated lossless speedup: "
        f"**{credited_text}**.",
        "",
        "| Result | Optimized model | vLLM |",
        "|---|---:|---:|",
        f"| Median prompt latency | {direct['optimized_prompt_latency_ms']['median']:.3f} ms | {direct['vllm_prompt_latency_ms']['median']:.3f} ms |",
        f"| Fastest prompt latency | {direct['optimized_prompt_latency_ms']['min']:.3f} ms | {direct['vllm_prompt_latency_ms']['min']:.3f} ms |",
        f"| Slowest prompt latency | {direct['optimized_prompt_latency_ms']['max']:.3f} ms | {direct['vllm_prompt_latency_ms']['max']:.3f} ms |",
        f"| Summed per-prompt median latency | {direct['optimized_total_median_ms']:.3f} ms | {direct['vllm_total_median_ms']:.3f} ms |",
        f"| Effective throughput | {direct['optimized_tokens_per_second']:.3f} tok/s | {direct['vllm_tokens_per_second']:.3f} tok/s |",
        f"| Generated tokens | {direct['optimized_generated_tokens']} | {direct['vllm_generated_tokens']} |",
        "",
        f"Direct optimized/vLLM token match is {direct['accepted_outputs']}/{direct['expected_outputs']}; "
        "this is diagnostic because vLLM is the latency baseline, not the correctness oracle.",
        "",
        "Observed speedup is credited only when the optimized output exactly matches the stable unoptimized-model token reference and both timed engines use matching methodology.",
        "",
        "## Speedup distribution",
        "",
        "| Aggregate | Lowest paired | Median paired | Best paired |",
        "|---:|---:|---:|---:|",
        f"| {direct['speedup_vs_vllm']:.3f}x | {direct['paired_prompt_speedup']['min']:.3f}x | {direct['paired_prompt_speedup']['median']:.3f}x | {direct['paired_prompt_speedup']['max']:.3f}x |",
        "",
        "## By workload",
        "",
        "| Workload | Context | Output | Exact outputs | Optimized ms | vLLM ms | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["conditions"]:
        speedup = row["speedup_vs_vllm"]
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "—"
        lines.append(
            f"| {row['workload']} | {row['context_length']} | {row['output_length']} | "
            f"{row['exact_outputs']}/{row['compared_outputs']} | "
            f"{row['optimized_summed_per_prompt_median_ms']:.3f} | "
            f"{row['vllm_summed_per_prompt_median_ms']:.3f} | {speedup_text} |"
        )
    phases = direct.get("optimized_phase_breakdown")
    if phases:
        lines.extend(
            [
                "",
                "## Optimized model time breakdown",
                "",
                "| Phase | Summed median ms | Share |",
                "|---|---:|---:|",
            ]
        )
        for field in ("setup_ms", "prefill_ms", "decode_ms", "host_overhead_ms"):
            row = phases[field]
            lines.append(
                f"| {field.removesuffix('_ms').replace('_', ' ')} | "
                f"{row['summed_per_prompt_median_ms']:.3f} | {row['share']:.2%} |"
            )
    if not direct["methodology_match"]:
        lines.extend(["", "## Methodology mismatch", ""])
        for field, values in direct["methodology"].items():
            if not values["match"]:
                lines.append(
                    f"- `{field}`: optimized={values['optimized']!r}, vLLM={values['vllm']!r}"
                )
    if not gate["lossless"]:
        lines.extend(["", "## Lossless gate failed", ""])
        for row in gate["first_divergences"][:10]:
            lines.append(
                f"- `{row['key'][1]}` first differs at generated token "
                f"{row['first_different_token_index']}."
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the optimized model directly with vLLM.")
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--benchmark-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    optimized = load(args.optimized)
    vllm = load(args.vllm)
    reference = load(args.reference)
    direct = direct_vllm_comparison(vllm, optimized)
    gate = compare_lossless(reference, optimized)
    vllm_gate = compare_lossless(reference, vllm)
    direct["exact_match_to_vllm"] = direct.pop("all_outputs_accepted")
    direct["all_outputs_accepted"] = gate["lossless"]
    direct["credited_speedup_vs_vllm"] = (
        direct["speedup_vs_vllm"]
        if gate["lossless"] and direct["methodology_match"]
        else None
    )
    report = {
        "schema_version": 4,
        "benchmark_name": args.benchmark_name or optimized.get("benchmark_name", "unspecified"),
        "engines": [optimized.get("variant", "optimized"), "vllm"],
        "metric": "CUDA-synchronized end-to-end wall latency",
        "correctness_oracle": reference.get("reference_engine", "hf_static"),
        "lossless_gate": gate,
        "vllm_reference_gate": vllm_gate,
        "optimized_vs_vllm": direct,
        "conditions": condition_comparisons(vllm, optimized),
        "serving": vllm.get("serving", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
