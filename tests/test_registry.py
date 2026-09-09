from qwen_opt.variants import PASS_ORDER, VARIANTS, variant_names


def test_pass_names_are_unique():
    assert len(PASS_ORDER) == len(set(PASS_ORDER))


def test_variant_names_are_unique():
    names = [variant.name for variant in VARIANTS]
    assert len(names) == len(set(names))


def test_full_suite_contains_both_combined_runners():
    names = variant_names("full")
    assert "combined_compiled" in names
    assert "combined_manual_graph" in names
