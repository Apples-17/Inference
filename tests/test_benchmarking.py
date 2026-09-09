from collections import Counter

import pytest

from qwen_opt.benchmarking import (
    build_work_items,
    distribution,
    parse_int_list,
    summarize_repetitions,
)
from qwen_opt.prompts import load_prompt_cases


def test_frozen_corpus_has_intended_buckets():
    cases = load_prompt_cases()
    assert len(cases) == 100
    assert Counter(case.bucket for case in cases) == {
        "32-64": 25,
        "128": 25,
        "256": 25,
        "512": 25,
    }
    assert {case.context_length for case in cases[:25]} == {32, 40, 48, 56, 64}


def test_distribution_uses_interpolated_quartiles():
    stats = distribution([1.0, 2.0, 3.0, 4.0])
    assert stats["median"] == 2.5
    assert stats["p25"] == 1.75
    assert stats["p75"] == 3.25


def test_repetition_summary_uses_each_timing_row():
    summary = summarize_repetitions(
        [
            {"total_ms": 10.0, "decode_ms": 8.0},
            {"total_ms": 12.0, "decode_ms": 9.0},
            {"total_ms": 11.0, "decode_ms": 8.5},
        ]
    )
    assert summary["total_ms"]["median"] == 11.0
    assert summary["decode_ms"]["median"] == 8.5


def test_repetition_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="zero repetitions"):
        summarize_repetitions([])


def test_work_item_matrix_is_explicit():
    cases = load_prompt_cases()[:2]
    items = build_work_items(cases, [16, 64], [128, 512], 64, 1)
    assert len(items) == 6
    assert [item.output_length for item in items[:4]] == [16, 16, 64, 64]
    assert [item.context_length for item in items[-2:]] == [128, 512]


def test_parse_int_list_rejects_invalid_values():
    assert parse_int_list("16, 32,64") == [16, 32, 64]
    with pytest.raises(ValueError):
        parse_int_list("")
    with pytest.raises(ValueError):
        parse_int_list("64,0")
