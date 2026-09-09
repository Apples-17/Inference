from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .errors import UnsupportedPass


@dataclass
class PassReport:
    applied: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


class TritonRMSNorm(nn.Module):
    def __init__(self, weight: nn.Parameter, eps: float):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return kernels.rmsnorm(hidden_states.contiguous(), self.weight, self.variance_epsilon)


def _eps(module: nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        value = getattr(module, name, None)
        if value is not None:
            return float(value)
    raise UnsupportedPass(f"cannot find RMSNorm epsilon on {type(module).__name__}")


def _make_packed_linear(parts: list[nn.Linear]) -> nn.Linear:
    first = parts[0]
    if any(part.in_features != first.in_features for part in parts):
        raise UnsupportedPass("packed projections have different input widths")
    with_bias = any(part.bias is not None for part in parts)
    if with_bias and not all(part.bias is not None for part in parts):
        raise UnsupportedPass("cannot pack a mixture of biased and bias-free projections")
    out_features = sum(part.out_features for part in parts)
    packed = nn.Linear(
        first.in_features,
        out_features,
        bias=with_bias,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    cursor = 0
    with torch.no_grad():
        for part in parts:
            width = part.out_features
            packed.weight[cursor : cursor + width].copy_(part.weight)
            if with_bias:
                packed.bias[cursor : cursor + width].copy_(part.bias)
            cursor += width
    packed.requires_grad_(False)
    return packed


def _install_packed_qkv(model: nn.Module) -> int:
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, "qkv_proj"):
            continue
        q_width = attention.q_proj.out_features
        k_width = attention.k_proj.out_features
        v_width = attention.v_proj.out_features
        attention.qkv_proj = _make_packed_linear(
            [attention.q_proj, attention.k_proj, attention.v_proj]
        )
        attention._qwen_opt_qkv_widths = (q_width, k_width, v_width)
        del attention.q_proj
        del attention.k_proj
        del attention.v_proj
        count += 1
    return count


def _install_packed_gate_up(model: nn.Module) -> int:
    count = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate_up_proj"):
            continue
        gate_width = mlp.gate_proj.out_features
        up_width = mlp.up_proj.out_features
        mlp.gate_up_proj = _make_packed_linear([mlp.gate_proj, mlp.up_proj])
        mlp._qwen_opt_gate_up_widths = (gate_width, up_width)
        del mlp.gate_proj
        del mlp.up_proj
        count += 1
    return count


def _mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(self, "gate_up_proj"):
        projected = self.gate_up_proj(x)
        gate, up = projected.split(self._qwen_opt_gate_up_widths, dim=-1)
    else:
        gate, up = self.gate_proj(x), self.up_proj(x)
    if getattr(self, "_qwen_opt_fused_swiglu", False):
        activated = kernels.silu_mul(gate.contiguous(), up.contiguous())
    else:
        activated = self.act_fn(gate) * up
    return self.down_proj(activated)


def _install_mlp_forward(model: nn.Module, fused_swiglu: bool) -> int:
    count = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        mlp._qwen_opt_fused_swiglu = fused_swiglu
        mlp.forward = types.MethodType(_mlp_forward, mlp)
        count += 1
    return count


def _attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    if hasattr(self, "qkv_proj"):
        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(
            self._qwen_opt_qkv_widths, dim=-1
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    query_states = query_states.view(hidden_shape)
    key_states = key_states.view(hidden_shape)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    if getattr(self, "_qwen_opt_fused_qk_rope", False):
        query_states, key_states = kernels.qk_norm_rope(
            query_states,
            key_states,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            _eps(self.q_norm),
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
    else:
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    dropout = 0.0 if not self.training else self.attention_dropout
    if getattr(self, "_qwen_opt_fused_gqa_attention", False):
        if not hasattr(F.scaled_dot_product_attention, "__call__"):
            raise UnsupportedPass("installed Torch lacks SDPA")
        try:
            attention_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout,
                scale=self.scaling,
                enable_gqa=True,
            )
        except (RuntimeError, TypeError) as exc:
            raise UnsupportedPass(f"native GQA SDPA is unavailable: {exc}") from exc
    else:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=self.scaling,
        )
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.view(*input_shape, -1)
    return self.o_proj(attention_output), None


def _install_attention_forward(
    model: nn.Module, *, fused_qk_rope: bool, fused_gqa_attention: bool
) -> int:
    count = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        attention._qwen_opt_fused_qk_rope = fused_qk_rope
        attention._qwen_opt_fused_gqa_attention = fused_gqa_attention
        attention.forward = types.MethodType(_attention_forward, attention)
        count += 1
    return count


def _decoder_layer_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states, mlp_input = kernels.add_rmsnorm(
        hidden_states.contiguous(),
        residual.contiguous(),
        self.post_attention_layernorm.weight,
        _eps(self.post_attention_layernorm),
    )
    return hidden_states + self.mlp(mlp_input)


def _install_fused_norms(model: nn.Module) -> tuple[int, int]:
    replaced = 0
    layers = 0
    norm_slots: list[tuple[nn.Module, str]] = [(model.model, "norm")]
    for layer in model.model.layers:
        norm_slots.extend(
            [
                (layer, "input_layernorm"),
                (layer, "post_attention_layernorm"),
                (layer.self_attn, "q_norm"),
                (layer.self_attn, "k_norm"),
            ]
        )
    for parent, name in norm_slots:
        old = getattr(parent, name)
        if not isinstance(old, TritonRMSNorm):
            setattr(parent, name, TritonRMSNorm(old.weight, _eps(old)))
            replaced += 1
    for layer in model.model.layers:
        layer.forward = types.MethodType(_decoder_layer_forward, layer)
        layers += 1
    return replaced, layers


def patch_static_cache(cache) -> int:
    """Replace the two index_copy launches per layer with one B1/Q1 kernel."""
    patched = 0
    for layer in cache.layers:
        if hasattr(layer, "_qwen_opt_original_update"):
            continue
        original = layer.update

        def update(self, key_states, value_states, cache_kwargs=None, _original=original):
            kwargs = cache_kwargs or {}
            position = kwargs.get("cache_position")
            initialized = bool(getattr(self, "is_initialized", False))
            if (
                not initialized
                or position is None
                or key_states.shape[0] != 1
                or key_states.shape[2] != 1
            ):
                return _original(key_states, value_states, kwargs)
            kernels.kv_write(
                key_states.contiguous(),
                value_states.contiguous(),
                self.keys,
                self.values,
                position,
            )
            # Transformers cache internals changed across releases. Older
            # StaticLayer implementations tracked this counter explicitly;
            # 4.57.x derives sequence length from the written cache and does not
            # expose it. Preserve the old counter when present, but never create
            # a private state variable that the installed implementation does not
            # understand.
            if hasattr(self, "cumulative_length"):
                self.cumulative_length += key_states.shape[2]
            return self.keys, self.values

        layer._qwen_opt_original_update = original
        layer.update = types.MethodType(update, layer)
        patched += 1
    return patched


def apply_passes(model: nn.Module, pass_names: Iterable[str]) -> PassReport:
    names = set(pass_names)
    kernels_needed = names & {
        "fused_rmsnorm_residual",
        "fused_qk_rope",
        "fused_kv_write",
        "fused_swiglu",
        "fused_lm_head_argmax",
    }
    if kernels_needed:
        kernels.require_triton()
    report = PassReport()

    if "packed_qkv" in names:
        report.details["packed_qkv_layers"] = _install_packed_qkv(model)
        report.applied.append("packed_qkv")
    if "packed_gate_up" in names:
        report.details["packed_gate_up_layers"] = _install_packed_gate_up(model)
        report.applied.append("packed_gate_up")
    if "fused_swiglu" in names or "packed_gate_up" in names:
        report.details["patched_mlp_layers"] = _install_mlp_forward(
            model, "fused_swiglu" in names
        )
        if "fused_swiglu" in names:
            report.applied.append("fused_swiglu")
    if "fused_rmsnorm_residual" in names:
        norms, layers = _install_fused_norms(model)
        report.details.update(fused_norm_modules=norms, fused_residual_layers=layers)
        report.applied.append("fused_rmsnorm_residual")
    if names & {"packed_qkv", "fused_qk_rope", "fused_gqa_attention"}:
        report.details["patched_attention_layers"] = _install_attention_forward(
            model,
            fused_qk_rope="fused_qk_rope" in names,
            fused_gqa_attention="fused_gqa_attention" in names,
        )
        if "fused_qk_rope" in names:
            report.applied.append("fused_qk_rope")
        if "fused_gqa_attention" in names:
            report.applied.append("fused_gqa_attention")
    if "fused_kv_write" in names:
        report.applied.append("fused_kv_write")
    if "fused_lm_head_argmax" in names:
        report.applied.append("fused_lm_head_argmax")
    if "runner_buffers" in names:
        report.applied.append("runner_buffers")
    if "compile" in names:
        report.applied.append("compile")
    if "manual_cudagraph" in names:
        report.applied.append("manual_cudagraph")
    return report
