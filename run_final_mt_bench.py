#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from check_lossless import compare as compare_lossless
from qwen_opt.benchmarking import distribution, file_hash


ROOT = Path(__file__).resolve().parent

PASS_VARIANTS = (
    ("P1 runner buffers", "lossless_safe"),
    ("P2a packed QKV", "p2_packed_qkv"),
    ("P2b packed gate/up", "p2_packed_gate_up"),
    ("P3 fused RMSNorm/residual", "p3_fused_rmsnorm_residual"),
    ("P4 fused QK-Norm/RoPE", "p4_fused_qk_rope"),
    ("P5 fused KV write", "p5_fused_kv_write"),
    ("P6 native GQA attention", "p6_fused_gqa_attention"),
    ("P7 fused SwiGLU", "p7_fused_swiglu"),
    ("P8 fused LM-head argmax", "p8_fused_lm_head_argmax"),
    ("P9 torch.compile", "p9_compile"),
    ("P10 manual CUDA graph", "p10_manual_cudagraph"),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def completed(path: Path, expected: dict[str, object]) -> bool:
    """Return true only when a saved engine result matches this exact run."""
    if not path.exists():
        return False
    try:
        payload = load(path)
    except (OSError, json.JSONDecodeError):
        return False
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    actual_shapes = Counter(
        (row.get("workload"), int(row.get("output_length", -1))) for row in rows
    )
    return (
        len(rows) == expected["row_count"]
        and actual_shapes == expected["row_shape_counts"]
        and all(
            payload.get(key) == value
            for key, value in expected.items()
            if key not in {"row_count", "row_shape_counts"}
        )
    )


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def expected_run(
    *,
    benchmark_name: str,
    model: str,
    prompt_mode: str,
    prompts_file: Path,
    limit: int,
    output_length: int,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    return {
        "benchmark_name": benchmark_name,
        "model": model,
        "prompt_mode": prompt_mode,
        "prompts_file_sha256": file_hash(prompts_file),
        "warmup_per_shape": warmup,
        "timed_repetitions_per_prompt": repetitions,
        "row_count": limit,
        "row_shape_counts": Counter({("mixed_context", output_length): limit}),
    }


def custom_command(
    *,
    variant: str,
    model: str,
    prompts_file: Path,
    limit: int,
    output_length: int,
    warmup: int,
    repetitions: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "benchmark_corpus.py"),
        "--variant", variant,
        "--benchmark-name", "mt_bench",
        "--model", model,
        "--prompts-file", str(prompts_file),
        "--prompt-mode", "natural",
        "--limit", str(limit),
        "--mixed-output-lengths", str(output_length),
        "--warmup", str(warmup),
        "--repetitions", str(repetitions),
        "--output", str(output),
    ]


def vllm_command(
    *,
    model: str,
    prompts_file: Path,
    limit: int,
    output_length: int,
    warmup: int,
    repetitions: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "benchmark_vllm_corpus.py"),
        "--benchmark-name", "mt_bench",
        "--model", model,
        "--prompts-file", str(prompts_file),
        "--prompt-mode", "natural",
        "--limit", str(limit),
        "--mixed-output-lengths", str(output_length),
        "--warmup", str(warmup),
        "--repetitions", str(repetitions),
        "--output", str(output),
    ]


def latency_summary(payload: dict) -> dict:
    values = [float(row["summary"]["total_ms"]["median"]) for row in payload["rows"]]
    generated_tokens = sum(len(row["token_ids"]) for row in payload["rows"])
    total = sum(values)
    return {
        "prompt_latency_ms": distribution(values),
        "summed_per_prompt_median_ms": total,
        "tokens_per_second": generated_tokens * 1000.0 / total,
    }


def prefix_lossless(reference: dict, candidate: dict, output_length: int) -> dict:
    expected = {row["id"]: row for row in reference["rows"]}
    exact = 0
    mismatches = []
    for row in candidate.get("rows", []):
        oracle = expected.get(row["id"])
        matched = bool(
            oracle
            and oracle["prompt_token_ids_sha256"] == row["prompt_token_ids_sha256"]
            and oracle["token_ids"][:output_length] == row["token_ids"]
        )
        exact += int(matched)
        if not matched:
            mismatches.append(row["id"])
    count = len(candidate.get("rows", []))
    return {
        "expected_outputs": count,
        "exact_outputs": exact,
        "mismatched_ids": mismatches,
        "lossless": count > 0 and exact == count,
    }


def screen_speedup(vllm: dict, candidate: dict) -> dict:
    baseline = {row["id"]: row for row in vllm["rows"]}
    optimized = {row["id"]: row for row in candidate["rows"]}
    ids = sorted(set(baseline) & set(optimized))
    baseline_ms = [float(baseline[item]["summary"]["total_ms"]["median"]) for item in ids]
    optimized_ms = [float(optimized[item]["summary"]["total_ms"]["median"]) for item in ids]
    paired = [left / right for left, right in zip(baseline_ms, optimized_ms)]
    return {
        "compared_prompts": len(ids),
        "aggregate": sum(baseline_ms) / sum(optimized_ms),
        "paired": distribution(paired),
    }


def format_value(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.3f}{suffix}"


def final_markdown(summary: dict) -> str:
    final = summary["final"]
    optimized = final["optimized_model"]
    vllm = final["vllm"]
    speedup = final["speedup_vs_vllm"]
    gate = final["lossless"]
    lines = [
        "# Final MT-Bench inference report",
        "",
        f"Selected pass: **`{summary['selected_variant']}`**. "
        f"Lossless: **{gate['exact_outputs']}/{gate['expected_outputs']} PASS**. "
        f"Aggregate speedup vs vLLM: **{speedup['aggregate']:.3f}x**.",
        "",
        "## Pass screening",
        "",
        "Each pass used the same MT-Bench subset and one shared vLLM screen. "
        "Only prefix-lossless passes were eligible for full-corpus validation.",
        "",
        "| Pass | Status | Exact | Median ms | tok/s | Speedup vs vLLM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["pass_sweep"]:
        lines.append(
            f"| {row['label']} | {row['status']} | "
            f"{row.get('exact_outputs', 0)}/{row.get('expected_outputs', 0)} | "
            f"{format_value(row.get('median_prompt_ms'))} | "
            f"{format_value(row.get('tokens_per_second'))} | "
            f"{format_value(row.get('screen_speedup_vs_vllm'), 'x')} |"
        )
    lines.extend(
        [
            "",
            "## Final engine comparison",
            "",
            f"| Metric | Optimized model (`{summary['selected_variant']}`) | vLLM |",
            "|---|---:|---:|",
            f"| Median prompt latency | {optimized['prompt_latency_ms']['median']:.3f} ms | {vllm['prompt_latency_ms']['median']:.3f} ms |",
            f"| Fastest prompt latency | {optimized['prompt_latency_ms']['min']:.3f} ms | {vllm['prompt_latency_ms']['min']:.3f} ms |",
            f"| Slowest prompt latency | {optimized['prompt_latency_ms']['max']:.3f} ms | {vllm['prompt_latency_ms']['max']:.3f} ms |",
            f"| Summed per-prompt median latency | {optimized['summed_per_prompt_median_ms']:.3f} ms | {vllm['summed_per_prompt_median_ms']:.3f} ms |",
            f"| Effective throughput | {optimized['tokens_per_second']:.3f} tok/s | {vllm['tokens_per_second']:.3f} tok/s |",
            "",
            "## Speedup vs vLLM",
            "",
            "| Aggregate | Lowest paired | Median paired | Best paired |",
            "|---:|---:|---:|---:|",
            f"| {speedup['aggregate']:.3f}x | {speedup['paired']['min']:.3f}x | {speedup['paired']['median']:.3f}x | {speedup['paired']['max']:.3f}x |",
            "",
            "Aggregate speedup is the sum of vLLM per-prompt median latencies divided by the corresponding optimized-model sum.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep every optimization pass, select the fastest lossless pass, and run final MT-Bench vs vLLM."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--prompts-file", type=Path, default=Path("data/mt_bench.jsonl"))
    parser.add_argument("--reference", type=Path, default=Path("data/mt_bench_reference.json"))
    parser.add_argument("--selection-prompts", type=int, default=8)
    parser.add_argument("--selection-output-length", type=int, default=64)
    parser.add_argument("--selection-warmup", type=int, default=1)
    parser.add_argument("--final-prompts", type=int, default=80)
    parser.add_argument("--final-output-length", type=int, default=256)
    parser.add_argument("--final-warmup", type=int, default=3)
    parser.add_argument("--final-repetitions", type=int, default=5)
    parser.add_argument(
        "--passes",
        default=",".join(variant for _, variant in PASS_VARIANTS),
        help="Comma-separated pass variants to screen.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_mt_bench"))
    parser.add_argument("--fresh", action="store_true", help="Ignore compatible saved runs.")
    args = parser.parse_args()

    prompts = (ROOT / args.prompts_file).resolve()
    reference_path = (ROOT / args.reference).resolve()
    output_root = (ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference = load(reference_path)
    if reference.get("benchmark_name") != "mt_bench":
        raise ValueError("the correctness reference must be for mt_bench")
    if args.final_prompts != len(reference["rows"]):
        raise ValueError(
            f"final-prompts must equal the {len(reference['rows'])}-row correctness oracle"
        )

    requested = [value.strip() for value in args.passes.split(",") if value.strip()]
    known = {variant: label for label, variant in PASS_VARIANTS}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"unknown pass variants: {unknown}")

    screen_expected = expected_run(
        benchmark_name="mt_bench",
        model=args.model,
        prompt_mode="natural",
        prompts_file=prompts,
        limit=args.selection_prompts,
        output_length=args.selection_output_length,
        warmup=args.selection_warmup,
        repetitions=1,
    )
    sweep_rows = []
    raw_dir = output_root / "pass_sweep" / "raw"
    log_dir = output_root / "logs"
    print(f"\n=== PASS SWEEP: {len(requested)} optimization passes ===", flush=True)
    for index, variant in enumerate(requested, 1):
        print(f"\n[{index}/{len(requested)}] {known[variant]} ({variant})", flush=True)
        path = raw_dir / f"{variant}.json"
        status = "ok"
        if not args.fresh and completed(path, screen_expected):
            print(f"RESUME: keeping compatible {path}", flush=True)
        else:
            code = run_logged(
                custom_command(
                    variant=variant,
                    model=args.model,
                    prompts_file=prompts,
                    limit=args.selection_prompts,
                    output_length=args.selection_output_length,
                    warmup=args.selection_warmup,
                    repetitions=1,
                    output=path,
                ),
                log_dir / f"screen_{variant}.log",
            )
            if code != 0 or not path.exists():
                status = "failed"
        row = {"label": known[variant], "variant": variant, "status": status}
        if status == "ok":
            payload = load(path)
            gate = prefix_lossless(reference, payload, args.selection_output_length)
            timing = latency_summary(payload)
            row.update(
                expected_outputs=gate["expected_outputs"],
                exact_outputs=gate["exact_outputs"],
                lossless=gate["lossless"],
                median_prompt_ms=timing["prompt_latency_ms"]["median"],
                fastest_prompt_ms=timing["prompt_latency_ms"]["min"],
                slowest_prompt_ms=timing["prompt_latency_ms"]["max"],
                tokens_per_second=timing["tokens_per_second"],
                raw_result=str(path),
            )
        sweep_rows.append(row)

    print("\n=== SHARED vLLM SCREEN ===", flush=True)
    screen_vllm_path = output_root / "pass_sweep" / "vllm.json"
    if not args.fresh and completed(screen_vllm_path, screen_expected):
        print(f"RESUME: keeping compatible {screen_vllm_path}", flush=True)
    else:
        code = run_logged(
            vllm_command(
                model=args.model,
                prompts_file=prompts,
                limit=args.selection_prompts,
                output_length=args.selection_output_length,
                warmup=args.selection_warmup,
                repetitions=1,
                output=screen_vllm_path,
            ),
            log_dir / "screen_vllm.log",
        )
        if code != 0:
            raise RuntimeError("vLLM screening run failed; see its log")
    screen_vllm = load(screen_vllm_path)
    for row in sweep_rows:
        if row["status"] != "ok":
            continue
        result = screen_speedup(screen_vllm, load(Path(row["raw_result"])))
        row["screen_speedup_vs_vllm"] = result["aggregate"]
        row["screen_paired_speedup"] = result["paired"]

    eligible = sorted(
        (row for row in sweep_rows if row.get("lossless")),
        key=lambda row: row["median_prompt_ms"],
    )
    write_json(output_root / "pass_sweep" / "summary.json", {"passes": sweep_rows})
    with (output_root / "pass_sweep" / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "label",
            "variant",
            "status",
            "exact_outputs",
            "expected_outputs",
            "median_prompt_ms",
            "tokens_per_second",
            "screen_speedup_vs_vllm",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_rows)
    if not eligible:
        raise RuntimeError("no pass was lossless on the MT-Bench screening set")

    print("\n=== FULL 80-PROMPT LOSSLESS SELECTION ===", flush=True)
    validation_expected = expected_run(
        benchmark_name="mt_bench",
        model=args.model,
        prompt_mode="natural",
        prompts_file=prompts,
        limit=args.final_prompts,
        output_length=args.final_output_length,
        warmup=1,
        repetitions=1,
    )
    validation_rows = []
    selected = None
    for row in eligible:
        variant = row["variant"]
        validation_path = output_root / "full_gate" / f"{variant}.json"
        existing_lossless_check = ROOT / "results" / "lossless_check" / f"{variant}.json"
        if (
            not args.fresh
            and completed(existing_lossless_check, validation_expected)
        ):
            validation_path = existing_lossless_check
            print(f"RESUME: using prior full gate candidate {validation_path}", flush=True)
        elif not args.fresh and completed(validation_path, validation_expected):
            print(f"RESUME: keeping compatible {validation_path}", flush=True)
        else:
            code = run_logged(
                custom_command(
                    variant=variant,
                    model=args.model,
                    prompts_file=prompts,
                    limit=args.final_prompts,
                    output_length=args.final_output_length,
                    warmup=1,
                    repetitions=1,
                    output=validation_path,
                ),
                log_dir / f"full_gate_{variant}.log",
            )
            if code != 0:
                validation_rows.append({"variant": variant, "status": "failed"})
                continue
        gate = compare_lossless(reference, load(validation_path))
        gate_path = output_root / "full_gate" / f"{variant}_gate.json"
        write_json(gate_path, gate)
        validation_rows.append(
            {
                "variant": variant,
                "status": "pass" if gate["lossless"] else "rejected",
                "exact_outputs": gate["exact_outputs"],
                "expected_outputs": gate["expected_outputs"],
                "gate_result": str(gate_path),
            }
        )
        print(
            f"{variant}: {gate['exact_outputs']}/{gate['expected_outputs']} exact",
            flush=True,
        )
        if gate["lossless"]:
            selected = variant
            break
    write_json(output_root / "full_gate" / "summary.json", {"candidates": validation_rows})
    if selected is None:
        raise RuntimeError("no screened pass survived the full 80-prompt lossless gate")
    (output_root / "selected_variant.txt").write_text(selected + "\n", encoding="utf-8")

    print(f"\n=== FINAL MT-BENCH: {selected} vs vLLM ===", flush=True)
    final_expected = expected_run(
        benchmark_name="mt_bench",
        model=args.model,
        prompt_mode="natural",
        prompts_file=prompts,
        limit=args.final_prompts,
        output_length=args.final_output_length,
        warmup=args.final_warmup,
        repetitions=args.final_repetitions,
    )
    final_dir = output_root / "final"
    optimized_path = final_dir / f"{selected}.json"
    if not args.fresh and completed(optimized_path, final_expected):
        print(f"RESUME: keeping compatible {optimized_path}", flush=True)
    else:
        code = run_logged(
            custom_command(
                variant=selected,
                model=args.model,
                prompts_file=prompts,
                limit=args.final_prompts,
                output_length=args.final_output_length,
                warmup=args.final_warmup,
                repetitions=args.final_repetitions,
                output=optimized_path,
            ),
            log_dir / f"final_{selected}.log",
        )
        if code != 0:
            raise RuntimeError("final optimized-model run failed")
    final_gate = compare_lossless(reference, load(optimized_path))
    write_json(final_dir / "lossless_gate.json", final_gate)
    if not final_gate["lossless"]:
        raise RuntimeError(
            f"final run failed losslessness: {final_gate['exact_outputs']}/{final_gate['expected_outputs']}"
        )

    vllm_path = final_dir / "vllm.json"
    if not args.fresh and completed(vllm_path, final_expected):
        print(f"RESUME: keeping compatible {vllm_path}", flush=True)
    else:
        code = run_logged(
            vllm_command(
                model=args.model,
                prompts_file=prompts,
                limit=args.final_prompts,
                output_length=args.final_output_length,
                warmup=args.final_warmup,
                repetitions=args.final_repetitions,
                output=vllm_path,
            ),
            log_dir / "final_vllm.log",
        )
        if code != 0:
            raise RuntimeError("final vLLM run failed")

    comparison_path = final_dir / "comparison.json"
    comparison_md = final_dir / "comparison.md"
    compare_code = run_logged(
        [
            sys.executable,
            str(ROOT / "compare_benchmarks.py"),
            "--optimized", str(optimized_path),
            "--vllm", str(vllm_path),
            "--reference", str(reference_path),
            "--benchmark-name", "mt_bench",
            "--output", str(comparison_path),
            "--markdown", str(comparison_md),
        ],
        log_dir / "final_compare.log",
    )
    if compare_code != 0:
        raise RuntimeError("final comparison failed")
    comparison = load(comparison_path)
    direct = comparison["optimized_vs_vllm"]
    summary = {
        "schema_version": 1,
        "benchmark": {
            "name": "mt_bench",
            "model": args.model,
            "prompts": args.final_prompts,
            "output_tokens_per_prompt": args.final_output_length,
            "warmup": args.final_warmup,
            "repetitions": args.final_repetitions,
            "batch_size": 1,
            "precision": "float16",
            "decoding": "greedy",
        },
        "selected_variant": selected,
        "selection_rule": "lowest screening median among passes exact on the screen, followed by full 80-prompt exact validation",
        "pass_sweep": sweep_rows,
        "full_validation": validation_rows,
        "final": {
            "lossless": final_gate,
            "optimized_model": latency_summary(load(optimized_path)),
            "vllm": latency_summary(load(vllm_path)),
            "speedup_vs_vllm": {
                "definition": direct["definition"],
                "aggregate": direct["speedup_vs_vllm"],
                "credited": direct["credited_speedup_vs_vllm"],
                "paired": direct["paired_prompt_speedup"],
            },
        },
        "files": {
            "optimized": str(optimized_path),
            "vllm": str(vllm_path),
            "comparison": str(comparison_path),
        },
    }
    report_json = output_root / "final_report.json"
    report_md = output_root / "final_report.md"
    write_json(report_json, summary)
    report_md.write_text(final_markdown(summary), encoding="utf-8")
    print("\n" + final_markdown(summary), flush=True)
    print(f"Final JSON: {report_json}")
    print(f"Final table: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
