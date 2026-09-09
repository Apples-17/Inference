import pytest
import torch

from qwen_opt import kernels


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or kernels.triton is None,
    reason="CUDA+Triton required",
)


def test_rmsnorm_matches_reference():
    torch.manual_seed(7)
    x = torch.randn(9, 2560, device="cuda", dtype=torch.float16)
    weight = torch.randn(2560, device="cuda", dtype=torch.float16)
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = expected.half() * weight
    actual = kernels.rmsnorm(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_add_rmsnorm_matches_reference():
    torch.manual_seed(8)
    x = torch.randn(5, 2560, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(2560, device="cuda", dtype=torch.float16)
    summed, actual = kernels.add_rmsnorm(x, residual, weight, 1e-6)
    expected_sum = (x.float() + residual.float()).half()
    expected = expected_sum.float() * torch.rsqrt(
        expected_sum.float().pow(2).mean(-1, keepdim=True) + 1e-6
    )
    expected = expected.half() * weight
    torch.testing.assert_close(summed, expected_sum, rtol=0, atol=2e-3)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_silu_mul_matches_reference():
    torch.manual_seed(9)
    gate = torch.randn(3, 9728, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    actual = kernels.silu_mul(gate, up)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_kv_write_changes_only_requested_position():
    key = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cache_k = torch.zeros(1, 8, 64, 128, device="cuda", dtype=torch.float16)
    cache_v = torch.zeros_like(cache_k)
    position = torch.tensor([17], device="cuda")
    kernels.kv_write(key, value, cache_k, cache_v, position)
    torch.testing.assert_close(cache_k[:, :, 17:18], key)
    torch.testing.assert_close(cache_v[:, :, 17:18], value)
    assert torch.count_nonzero(cache_k[:, :, :17]) == 0


def test_qk_norm_rope_matches_reference():
    torch.manual_seed(11)
    q = torch.randn(1, 3, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 3, 2, 128, device="cuda", dtype=torch.float16)
    qw = torch.randn(128, device="cuda", dtype=torch.float16)
    kw = torch.randn(128, device="cuda", dtype=torch.float16)
    angle = torch.randn(1, 3, 128, device="cuda", dtype=torch.float32)
    cos, sin = angle.cos().half(), angle.sin().half()

    def reference(x, weight):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        normalized = normalized.half() * weight
        first, second = normalized.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return normalized * cos.unsqueeze(2) + rotated * sin.unsqueeze(2)

    actual_q, actual_k = kernels.qk_norm_rope(q, k, qw, kw, cos, sin, 1e-6)
    torch.testing.assert_close(actual_q, reference(q, qw), rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(actual_k, reference(k, kw), rtol=4e-3, atol=4e-3)


def test_lm_head_argmax_matches_linear():
    torch.manual_seed(10)
    weight = torch.randn(4097, 256, device="cuda", dtype=torch.float16)
    hidden = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.linear(hidden, weight).argmax(-1)
    actual = kernels.LMHeadArgmax(weight)(hidden)
    assert actual.item() == expected.item()
