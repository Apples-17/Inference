#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_key(row: dict) -> tuple[str, str, int, int]:
    return (
        row["workload"],
        row["id"],
        int(row["context_length"]),
        int(row["output_length"]),
    )


def compare(reference: dict, candidate: dict) -> dict:
    metadata_fields = (
        "benchmark_name",
        "model",
        "precision",
        "prompt_mode",
        "prompts_file_sha256",
    )
    metadata = {
        field: {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
            "match": reference.get(field) is not None
            and reference.get(field) == candidate.get(field),
        }
        for field in metadata_fields
    }
    expected = {row_key(row): row for row in reference.get("rows", [])}
    actual = {row_key(row): row for row in candidate.get("rows", [])}
    common = sorted(set(expected) & set(actual))
    exact = []
    divergences = []
    for key in common:
        reference_row = expected[key]
        candidate_row = actual[key]
        prompt_match = (
            reference_row["prompt_token_ids_sha256"]
            == candidate_row["prompt_token_ids_sha256"]
        )
        left = reference_row["token_ids"]
        right = candidate_row["token_ids"]
        token_match = left == right
        exact.append(prompt_match and token_match)
        if not prompt_match or not token_match:
            first = next(
                (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                min(len(left), len(right)),
            )
            divergences.append(
                {
                    "key": list(key),
                    "prompt_match": prompt_match,
                    "first_different_token_index": first,
                    "reference_token_id": left[first] if first < len(left) else None,
                    "candidate_token_id": right[first] if first < len(right) else None,
                }
            )
    passed = (
        bool(expected)
        and all(item["match"] for item in metadata.values())
        and set(expected) == set(actual)
        and all(exact)
    )
    return {
        "candidate": candidate.get("variant", "optimized"),
        "reference": reference.get("reference_engine", "hf_static"),
        "expected_outputs": len(expected),
        "candidate_outputs": len(actual),
        "compared_outputs": len(common),
        "exact_outputs": sum(exact),
        "mismatched_outputs": len(expected) - sum(exact),
        "missing_outputs": [list(key) for key in sorted(set(expected) - set(actual))],
        "extra_outputs": [list(key) for key in sorted(set(actual) - set(expected))],
        "metadata": metadata,
        "first_divergences": divergences[:20],
        "lossless": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Require exact optimized-model token parity.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(reference, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    state = "PASS" if result["lossless"] else "FAIL"
    print(
        f"LOSSLESS {state}: {result['exact_outputs']}/{result['expected_outputs']} exact outputs"
    )
    return 0 if result["lossless"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
