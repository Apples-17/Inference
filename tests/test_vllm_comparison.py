from compare_benchmarks import direct_vllm_comparison


def engine(variant: str, tokens: list[int], milliseconds: float) -> dict:
    return {
        "variant": variant,
        "model": "Qwen/Qwen3-4B",
        "precision": "float16",
        "batch_size": 1,
        "ignore_eos": True,
        "warmup_per_shape": 3,
        "timed_repetitions_per_prompt": 5,
        "prompts_file_sha256": "corpus-hash",
        "prompt_mode": "natural",
        "environment": {"gpu": "Tesla T4"},
        "rows": [
            {
                "id": "p0",
                "workload": "mixed_context",
                "context_length": 32,
                "output_length": len(tokens),
                "prompt_token_ids_sha256": "prompt-hash",
                "token_ids": tokens,
                "generated_token_ids_sha256": "output-hash",
                "summary": {"total_ms": {"median": milliseconds}},
            }
        ],
    }


def test_direct_vllm_comparison_reports_acceptance_and_speedup():
    result = direct_vllm_comparison(
        engine("vllm", [10, 11, 12, 13], 12.0),
        engine("optimized", [10, 11, 12, 13], 8.0),
    )
    assert result["accepted_outputs"] == 1
    assert result["accepted_generated_tokens"] == 4
    assert result["output_acceptance_rate"] == 1.0
    assert result["speedup_vs_vllm"] == 1.5
    assert result["credited_speedup_vs_vllm"] == 1.5
    assert result["per_output"][0]["accepted"] is True


def test_mismatch_is_visible_but_speedup_is_not_credited():
    result = direct_vllm_comparison(
        engine("vllm", [10, 11, 12, 13], 12.0),
        engine("optimized", [10, 99, 12, 13], 8.0),
    )
    assert result["accepted_outputs"] == 0
    assert result["identical_token_positions"] == 3
    assert result["speedup_vs_vllm"] == 1.5
    assert result["credited_speedup_vs_vllm"] is None


def test_unstable_vllm_output_is_not_accepted():
    baseline = engine("vllm", [10, 11], 12.0)
    baseline["rows"][0]["stable_across_repetitions"] = False
    result = direct_vllm_comparison(
        baseline,
        engine("optimized", [10, 11], 8.0),
    )
    assert result["stable_vllm_outputs"] == 0
    assert result["accepted_outputs"] == 0
    assert result["credited_speedup_vs_vllm"] is None
