from __future__ import annotations

from dataclasses import dataclass


PASS_ORDER = (
    "runner_buffers",
    "packed_qkv",
    "packed_gate_up",
    "fused_rmsnorm_residual",
    "fused_qk_rope",
    "fused_kv_write",
    "fused_gqa_attention",
    "fused_swiglu",
    "fused_lm_head_argmax",
    "compile",
    "manual_cudagraph",
)


@dataclass(frozen=True)
class Variant:
    name: str
    passes: tuple[str, ...]
    cache: str = "static"
    note: str = ""

    def validate(self) -> None:
        unknown = set(self.passes) - set(PASS_ORDER)
        if unknown:
            raise ValueError(f"unknown passes in {self.name}: {sorted(unknown)}")
        if "compile" in self.passes and "manual_cudagraph" in self.passes:
            raise ValueError(
                f"{self.name}: compile and manual_cudagraph are competing capture owners"
            )
        if self.cache not in {"dynamic", "static"}:
            raise ValueError(f"{self.name}: invalid cache {self.cache!r}")


# Independent rows share the same static-cache runner so the measured delta is the
# named execution pass. Dynamic and static HF rows remain as explicit baselines.
INDEPENDENT = (
    Variant("p1_runner_buffers", ("runner_buffers",)),
    Variant("p2_packed_qkv", ("runner_buffers", "packed_qkv")),
    Variant("p2_packed_gate_up", ("runner_buffers", "packed_gate_up")),
    Variant(
        "p3_fused_rmsnorm_residual",
        ("runner_buffers", "fused_rmsnorm_residual"),
    ),
    Variant("p4_fused_qk_rope", ("runner_buffers", "fused_qk_rope")),
    Variant("p5_fused_kv_write", ("runner_buffers", "fused_kv_write")),
    Variant(
        "p6_fused_gqa_attention", ("runner_buffers", "fused_gqa_attention")
    ),
    Variant("p7_fused_swiglu", ("runner_buffers", "fused_swiglu")),
    Variant(
        "p8_fused_lm_head_argmax", ("runner_buffers", "fused_lm_head_argmax")
    ),
    Variant("p9_compile", ("runner_buffers", "compile")),
    Variant(
        "p10_manual_cudagraph", ("runner_buffers", "manual_cudagraph")
    ),
)

CUMULATIVE = tuple(
    Variant(f"cumulative_p{i}", PASS_ORDER[:i]) for i in range(1, 10)
) + (
    Variant("combined_compiled", PASS_ORDER[:10]),
    # Manual graphs are measured against the same fused model without Inductor's
    # graph ownership. The report picks the faster *exact* combined candidate.
    Variant("combined_manual_graph", PASS_ORDER[:9] + ("manual_cudagraph",)),
)

VARIANTS = (
    Variant("hf_dynamic", (), cache="dynamic", note="untouched Transformers"),
    Variant("hf_static", (), note="untouched Transformers with StaticCache"),
    Variant("hf_compiled_static", ("compile",), note="Transformers+Inductor baseline"),
    Variant(
        "lossless_safe",
        ("runner_buffers",),
        note="Reuses runner storage without changing model arithmetic",
    ),
) + INDEPENDENT + CUMULATIVE

for _variant in VARIANTS:
    _variant.validate()


def get_variant(name: str) -> Variant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(f"unknown variant {name!r}")


def variant_names(suite: str) -> list[str]:
    if suite == "smoke":
        return ["hf_static", "p1_runner_buffers", "combined_compiled"]
    if suite == "independent":
        return ["hf_dynamic", "hf_static", "hf_compiled_static"] + [
            v.name for v in INDEPENDENT
        ]
    if suite == "cumulative":
        return ["hf_compiled_static"] + [v.name for v in CUMULATIVE]
    if suite == "full":
        return [v.name for v in VARIANTS]
    raise ValueError(f"unknown suite {suite!r}")
