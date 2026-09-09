from check_lossless import compare


def payload(variant: str, tokens: list[int]) -> dict:
    common = {
        "benchmark_name": "test",
        "model": "Qwen/Qwen3-4B",
        "precision": "float16",
        "prompt_mode": "natural",
        "prompts_file_sha256": "corpus",
    }
    row = {
        "id": "p0",
        "workload": "mixed_context",
        "context_length": 4,
        "output_length": len(tokens),
        "prompt_token_ids_sha256": "prompt",
        "token_ids": tokens,
    }
    if variant == "reference":
        return {**common, "reference_engine": "hf_static", "rows": [row]}
    return {**common, "variant": variant, "rows": [row]}


def test_exact_candidate_passes():
    result = compare(payload("reference", [1, 2, 3]), payload("lossless_safe", [1, 2, 3]))
    assert result["lossless"] is True
    assert result["exact_outputs"] == 1


def test_changed_token_fails_with_first_divergence():
    result = compare(payload("reference", [1, 2, 3]), payload("candidate", [1, 9, 3]))
    assert result["lossless"] is False
    assert result["first_divergences"][0]["first_different_token_index"] == 1


def test_missing_output_fails():
    candidate = payload("candidate", [1, 2, 3])
    candidate["rows"] = []
    assert compare(payload("reference", [1, 2, 3]), candidate)["lossless"] is False
