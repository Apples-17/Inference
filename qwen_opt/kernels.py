from __future__ import annotations

import math

import torch

from .errors import UnsupportedPass

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on the T4 host
    triton = None
    tl = None


def require_triton() -> None:
    if triton is None:
        raise UnsupportedPass("Triton is required for fused kernel passes")
    if not torch.cuda.is_available():
        raise UnsupportedPass("fused kernel passes require CUDA")


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(x, weight, out, n_cols: tl.constexpr, eps: tl.constexpr,
                        block: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < n_cols
        values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + eps)
        # Qwen3 casts the normalized value back to FP16 before multiplying the
        # learned weight. Preserve that rounding point.
        normalized = (values * inv_rms).to(tl.float16)
        weights = tl.load(weight + cols, mask=mask, other=0.0)
        tl.store(out + row * n_cols + cols, normalized * weights, mask=mask)

    @triton.jit
    def _add_rmsnorm_kernel(x, residual, weight, summed, normalized,
                            n_cols: tl.constexpr, eps: tl.constexpr,
                            block: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < n_cols
        x_value = tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        residual_value = tl.load(
            residual + row * n_cols + cols, mask=mask, other=0.0
        ).to(tl.float32)
        # Match the standalone FP16 residual add's rounding before RMSNorm.
        value = (x_value + residual_value).to(tl.float16)
        tl.store(summed + row * n_cols + cols, value, mask=mask)
        value_fp32 = value.to(tl.float32)
        variance = tl.sum(value_fp32 * value_fp32, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + eps)
        norm_value = (value_fp32 * inv_rms).to(tl.float16)
        weights = tl.load(weight + cols, mask=mask, other=0.0)
        tl.store(normalized + row * n_cols + cols, norm_value * weights, mask=mask)

    @triton.jit
    def _silu_mul_kernel(gate, up, out, n_elements: tl.constexpr,
                         block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n_elements
        gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
        up_value = tl.load(up + offsets, mask=mask).to(tl.float32)
        activated = gate_value * tl.sigmoid(gate_value)
        tl.store(out + offsets, activated * up_value, mask=mask)

    @triton.jit
    def _qk_norm_rope_kernel(
        q, k, q_weight, k_weight, cos, sin, q_out, k_out,
        q_rows: tl.constexpr, k_rows: tl.constexpr,
        q_heads: tl.constexpr, kv_heads: tl.constexpr,
        seq_len: tl.constexpr, head_dim: tl.constexpr,
        eps: tl.constexpr, block: tl.constexpr,
    ):
        row = tl.program_id(0)
        dims = tl.arange(0, block)
        valid_dim = dims < head_dim
        is_q = row < q_rows
        source_row = tl.where(is_q, row, row - q_rows)
        heads = tl.where(is_q, q_heads, kv_heads)
        token = (source_row // heads) % seq_len
        batch = source_row // (heads * seq_len)
        source = tl.where(is_q, q, k)
        weight = tl.where(is_q, q_weight, k_weight)
        output = tl.where(is_q, q_out, k_out)
        values = tl.load(
            source + source_row * head_dim + dims, mask=valid_dim, other=0.0
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + eps)
        weights = tl.load(weight + dims, mask=valid_dim, other=0.0)
        normalized = (values * inv_rms).to(tl.float16) * weights
        half = head_dim // 2
        partner_dims = tl.where(dims < half, dims + half, dims - half)
        partner_values = tl.load(
            source + source_row * head_dim + partner_dims,
            mask=valid_dim,
            other=0.0,
        ).to(tl.float32)
        partner_weights = tl.load(weight + partner_dims, mask=valid_dim, other=0.0)
        rotated = (partner_values * inv_rms).to(tl.float16) * partner_weights
        rotated = tl.where(dims < half, -rotated, rotated)
        rope_row = (batch * seq_len + token) * head_dim
        cos_value = tl.load(cos + rope_row + dims, mask=valid_dim).to(tl.float32)
        sin_value = tl.load(sin + rope_row + dims, mask=valid_dim).to(tl.float32)
        tl.store(
            output + source_row * head_dim + dims,
            normalized * cos_value + rotated * sin_value,
            mask=valid_dim,
        )

    @triton.jit
    def _kv_write_kernel(k, v, cache_k, cache_v, position,
                         kv_heads: tl.constexpr, head_dim: tl.constexpr,
                         cache_len: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        n_elements = kv_heads * head_dim
        mask = offsets < n_elements
        head = offsets // head_dim
        dim = offsets % head_dim
        position_value = tl.load(position)
        destination = head * cache_len * head_dim + position_value * head_dim + dim
        tl.store(cache_k + destination, tl.load(k + offsets, mask=mask), mask=mask)
        tl.store(cache_v + destination, tl.load(v + offsets, mask=mask), mask=mask)

    @triton.jit
    def _lm_head_partial_kernel(
        hidden, weight, partial_values, partial_ids,
        hidden_size: tl.constexpr, vocab_size: tl.constexpr,
        block_vocab: tl.constexpr, block_k: tl.constexpr,
    ):
        program = tl.program_id(0)
        rows = program * block_vocab + tl.arange(0, block_vocab)
        row_mask = rows < vocab_size
        accumulator = tl.zeros((block_vocab,), dtype=tl.float32)
        for start in range(0, hidden_size, block_k):
            cols = start + tl.arange(0, block_k)
            col_mask = cols < hidden_size
            x = tl.load(hidden + cols, mask=col_mask, other=0.0).to(tl.float32)
            w = tl.load(
                weight + rows[:, None] * hidden_size + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(w * x[None, :], axis=1)
        accumulator = tl.where(row_mask, accumulator, -float("inf"))
        local_index = tl.argmax(accumulator, axis=0)
        tl.store(partial_values + program, tl.max(accumulator, axis=0))
        tl.store(partial_ids + program, program * block_vocab + local_index)

    @triton.jit
    def _lm_head_reduce_kernel(values, ids, output, n_partials: tl.constexpr,
                               block: tl.constexpr):
        offsets = tl.arange(0, block)
        mask = offsets < n_partials
        candidates = tl.load(values + offsets, mask=mask, other=-float("inf"))
        best = tl.argmax(candidates, axis=0)
        tl.store(output, tl.load(ids + best))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    require_triton()
    if not x.is_cuda or not x.is_contiguous():
        raise UnsupportedPass("Triton RMSNorm requires a contiguous CUDA tensor")
    n_cols = x.shape[-1]
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise UnsupportedPass(f"RMSNorm width {n_cols} is unsupported")
    out = torch.empty_like(x)
    _rmsnorm_kernel[(x.numel() // n_cols,)](
        x, weight, out, n_cols=n_cols, eps=eps, block=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return out


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    require_triton()
    if not x.is_cuda or not x.is_contiguous() or not residual.is_contiguous():
        raise UnsupportedPass("add+RMSNorm requires contiguous CUDA tensors")
    n_cols = x.shape[-1]
    block = triton.next_power_of_2(n_cols)
    summed = torch.empty_like(x)
    normalized = torch.empty_like(x)
    _add_rmsnorm_kernel[(x.numel() // n_cols,)](
        x, residual, weight, summed, normalized,
        n_cols=n_cols, eps=eps, block=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return summed, normalized


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    require_triton()
    if not gate.is_cuda or not gate.is_contiguous() or not up.is_contiguous():
        raise UnsupportedPass("fused SwiGLU requires contiguous CUDA tensors")
    out = torch.empty_like(gate)
    block = 256
    _silu_mul_kernel[(triton.cdiv(gate.numel(), block),)](
        gate, up, out, n_elements=gate.numel(), block=block
    )
    return out


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head Q/K RMSNorm and half-rotation RoPE.

    Inputs are [batch, sequence, heads, head_dim]. Qwen3's rotary helper uses
    rotate_half, so this preserves the model's learned operation exactly.
    """
    require_triton()
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    if k.shape[:2] != (batch, seq_len) or k.shape[3] != head_dim:
        raise UnsupportedPass("Q/K shapes are incompatible")
    if head_dim % 2:
        raise UnsupportedPass("RoPE head dimension must be even")
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = batch * seq_len * q_heads
    k_rows = batch * seq_len * kv_heads
    block = triton.next_power_of_2(head_dim)
    _qk_norm_rope_kernel[(q_rows + k_rows,)](
        q, k, q_weight, k_weight, cos, sin, q_out, k_out,
        q_rows=q_rows, k_rows=k_rows, q_heads=q_heads, kv_heads=kv_heads,
        seq_len=seq_len, head_dim=head_dim, eps=eps, block=block,
        num_warps=4,
    )
    return q_out, k_out


def kv_write(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_key: torch.Tensor,
    cache_value: torch.Tensor,
    position: torch.Tensor,
) -> None:
    require_triton()
    if key.shape[0] != 1 or key.shape[2] != 1:
        raise UnsupportedPass("fused KV write is specialized for batch=1, query=1")
    kv_heads, head_dim = key.shape[1], key.shape[3]
    if cache_key.shape[0] != 1 or cache_key.shape[1] != kv_heads:
        raise UnsupportedPass("cache shape does not match KV heads")
    if position.numel() != 1:
        raise UnsupportedPass("fused KV write requires one cache position")
    n_elements = kv_heads * head_dim
    block = triton.next_power_of_2(n_elements)
    _kv_write_kernel[(1,)](
        key, value, cache_key, cache_value, position,
        kv_heads=kv_heads, head_dim=head_dim, cache_len=cache_key.shape[2],
        block=block, num_warps=8,
    )


class LMHeadArgmax:
    """No-logits-materialization full-vocabulary FP16 projection + argmax."""

    def __init__(self, weight: torch.Tensor):
        require_triton()
        if not weight.is_cuda or weight.ndim != 2:
            raise UnsupportedPass("LM-head weight must be a CUDA matrix")
        self.weight = weight
        self.vocab_size, self.hidden_size = weight.shape
        self.block_vocab = 32
        self.n_partials = math.ceil(self.vocab_size / self.block_vocab)
        self.partial_values = torch.empty(
            self.n_partials, device=weight.device, dtype=torch.float32
        )
        self.partial_ids = torch.empty(
            self.n_partials, device=weight.device, dtype=torch.int32
        )
        self.output = torch.empty(1, device=weight.device, dtype=torch.int32)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.numel() != self.hidden_size:
            raise UnsupportedPass("fused LM-head argmax supports one hidden row")
        hidden = hidden.contiguous().view(-1)
        _lm_head_partial_kernel[(self.n_partials,)](
            hidden, self.weight, self.partial_values, self.partial_ids,
            hidden_size=self.hidden_size, vocab_size=self.vocab_size,
            block_vocab=self.block_vocab, block_k=64, num_warps=8,
        )
        reduce_block = triton.next_power_of_2(self.n_partials)
        _lm_head_reduce_kernel[(1,)](
            self.partial_values, self.partial_ids, self.output,
            n_partials=self.n_partials, block=reduce_block, num_warps=8,
        )
        return self.output.to(torch.long)
