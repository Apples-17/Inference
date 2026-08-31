from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .features import TargetFeatureExtractor


@dataclass
class TTTBatch:
    input_ids: torch.Tensor


def _neg_inf(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError(f"attention-mask dtype must be floating point, got {dtype}")
    return torch.finfo(dtype).min


def _causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask of shape (L, L): position i attends to j <= i."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    mask = torch.full(
        (length, length),
        _neg_inf(dtype),
        device=device,
        dtype=dtype,
    )
    return torch.triu(mask, diagonal=1)


def _ttt_full_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    neg = _neg_inf(dtype)
    full = torch.full(
        (2 * length, 2 * length),
        neg,
        device=device,
        dtype=dtype,
    )

    # Original context -> original context: ordinary causal dependency.
    full[:length, :length] = _causal_mask(length, device, dtype)

    # Simulated node i -> original context positions <= i.
    ctx = _causal_mask(length, device, dtype)
    full[length:, :length] = ctx

    # Simulated node i -> only its own simulated key i.
    idx = torch.arange(length, device=device)
    full[length + idx, length + idx] = 0.0

    return full


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected logits (B,T,V), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"expected labels (B,T), got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logit/label sequence mismatch: logits={tuple(logits.shape)}, "
            f"labels={tuple(labels.shape)}"
        )

    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )


def _extract_target_features(target_model, input_ids: torch.Tensor, cfg):
    extractor = TargetFeatureExtractor(target_model, cfg)
    try:
        with torch.no_grad():
            target_model(input_ids=input_ids, use_cache=False)
            return extractor.pop()
    finally:
        extractor.remove()


def _unique_trainable_parameters(draft_model):
    params = []
    seen = set()
    for p in draft_model.trainable_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


def train_step(
    target_model,
    draft_model,
    fusion,
    batch: TTTBatch,
    cfg,
    optimizer,
) -> float:
    input_ids = batch.input_ids
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (B,N), got {tuple(input_ids.shape)}")
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    B, N = input_ids.shape
    requested_steps = max(1, int(getattr(cfg, "ttt_steps", 1)))

    # Need one advanced token for the input and one next-token label even for
    # the native step.  Reduce the number of TTT rounds for very short inputs.
    n_steps = min(requested_steps, N - 2)
    if n_steps < 1:
        raise ValueError(
            f"sequence is too short for EAGLE training: got N={N}; need at least 3 tokens"
        )

    # Keeping a fixed T across rounds lets all branches be processed in parallel.
    T = N - n_steps - 1
    if T <= 0:
        raise ValueError(f"invalid effective training length T={T} for N={N}, steps={n_steps}")

    device = input_ids.device

    # 1) Frozen target-model features for the ORIGINAL context only.
    base_tokens = input_ids[:, :T]
    feats = _extract_target_features(target_model, base_tokens, cfg)
    g = fusion(feats.low, feats.mid, feats.high)  # (B,T,H), trainable fusion

    if g.shape[:2] != (B, T):
        raise ValueError(
            f"fused target features must have shape (B,T,H); got {tuple(g.shape)}, "
            f"expected first dims {(B, T)}"
        )

    dtype = g.dtype
    base_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # EAGLE alignment: g_i is paired with the sampled token t_{i+1}.
    advanced_tokens = input_ids[:, 1 : T + 1]
    context_input = draft_model.build_input(g, advanced_tokens)

    # 2) Native training round.
    native_mask = _causal_mask(T, device, dtype).view(1, 1, T, T)
    a_prev, _ = draft_model.forward_layer(
        context_input,
        base_pos,
        native_mask,
        use_cache=False,
    )

    logits = draft_model.logits_from_a(a_prev)
    labels = input_ids[:, 2 : T + 2]
    losses = [_cross_entropy(logits, labels)]

    # Token produced by the previous draft round.  Greedy self-feeding matches
    # the project's temperature=0 training/inference path.
    with torch.no_grad():
        pred_tok = logits.argmax(dim=-1)

    # 3) Training-time-test rounds.
    ttt_mask = _ttt_full_mask(T, device, dtype).view(1, 1, 2 * T, 2 * T)

    for step in range(2, n_steps + 1):
        simulated_input = draft_model.build_input(a_prev, pred_tok)
        full_input = torch.cat([context_input, simulated_input], dim=1)

        sim_pos = base_pos + (step - 1)
        full_pos = torch.cat([base_pos, sim_pos], dim=1)

        full_out, _ = draft_model.forward_layer(
            full_input,
            full_pos,
            ttt_mask,
            use_cache=False,
        )
        a_prev = full_out[:, T:, :]

        logits = draft_model.logits_from_a(a_prev)
        labels = input_ids[:, step + 1 : step + 1 + T]
        losses.append(_cross_entropy(logits, labels))

        with torch.no_grad():
            pred_tok = logits.argmax(dim=-1)

    total_loss = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    params = _unique_trainable_parameters(draft_model)
    grad_clip: Optional[float] = getattr(cfg, "grad_clip", None)
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)

    optimizer.step()
    return float(total_loss.detach().item())
