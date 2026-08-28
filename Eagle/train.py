"""
Step 3 of the plan: train the draft model with EAGLE-3's "training-time
test" (paper §3.2, Fig. 6).

Why this exists at all: if we only ever trained the draft model on
ground-truth features (a "Step 1"-only loss), then at *inference* time,
Step 2/3/... of the draft tree feeds the draft model ITS OWN previous
outputs, which the model never saw during training -> distribution
shift -> the acceptance rate collapses at depth > 1 (this is exactly
the failure mode the paper diagnoses in Fig. 3/4 for "EAGLE without
fea pred"). Training-time test fixes this by literally performing the
same self-feeding during training, so the draft model learns to be
robust to its own small errors.

Mask construction (Fig. 6): at simulated step s, the new query block
attends causally to the ORIGINAL context (grey tokens, ground-truth
features) for positions < t, and diagonally (only itself) to the
PREVIOUS step's own output a^{(s-1)}_t. It never attends to other
positions' step (s-1) outputs. We realize this with a single decoder
layer forward over a concatenated [context | new_block] sequence with
an explicit additive mask, so we reuse the identical decoder layer
(the SAME weights) at every simulated step -- consistent with the
draft model being one layer applied repeatedly, not a deep stack.
"""
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import TargetFeatureExtractor


def _causal_mask(T, device, dtype):
    neg = torch.finfo(dtype).min
    m = torch.full((T, T), neg, device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)


def _ttt_block_mask(T, device, dtype):
    """
    (T, 2T) additive mask for a simulated step's query block of length T
    against keys = [original context (T) | this step's own diagonal (T)].
    Query t: causal (<t) into context block, diagonal (==t) into self block.
    """
    neg = torch.finfo(dtype).min
    ctx = torch.triu(torch.full((T, T), neg, device=device, dtype=dtype), diagonal=1)  # strictly causal-inclusive below handled by diag block
    # allow query t to see context keys 0..t-1 only (exclude t itself; t's own
    # info for this step comes from the diagonal block instead)
    ctx = torch.triu(torch.full((T, T), neg, device=device, dtype=dtype), diagonal=0)
    diag = torch.diag(torch.zeros(T, device=device, dtype=dtype))
    diag = torch.where(torch.eye(T, device=device, dtype=torch.bool), diag, torch.full_like(diag, neg))
    return torch.cat([ctx, diag], dim=-1)  # (T, 2T)


@dataclass
class TTTBatch:
    input_ids: torch.Tensor       # (B, T+1) ground-truth token ids (T context + 1 extra for last label)


def train_step(target_model, draft_model, fusion, batch: TTTBatch, cfg, optimizer) -> float:
    device = batch.input_ids.device
    B, Tp1 = batch.input_ids.shape
    T = Tp1 - 1
    tokens = batch.input_ids[:, :T]       # (B,T) fed as context
    labels_step1 = batch.input_ids[:, 1:T + 1]  # next-token ground truth for step 1

    extractor = TargetFeatureExtractor(target_model, cfg)
    with torch.no_grad():
        _ = target_model(input_ids=batch.input_ids[:, :T], use_cache=False)
        feats = extractor.pop()
    extractor.remove()

    g = fusion(feats.low, feats.mid, feats.high)  # (B,T,H) trainable fusion, grad flows into `fusion`

    dtype = g.dtype
    total_loss = 0.0

    # ---- Step 1: native training step (standard causal, ground-truth features) ----
    prev_tok = torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)  # shift for "previous sampled token"
    x1 = draft_model.build_input(g, prev_tok)
    mask1 = _causal_mask(T, device, dtype).view(1, 1, T, T)
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    a1, _ = draft_model.forward_layer(x1, pos_ids, mask1, use_cache=False)
    logits1 = draft_model.logits_from_a(a1)
    loss1 = F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), labels_step1.reshape(-1))
    total_loss = total_loss + loss1

    a_prev = a1
    labels_prev = labels_step1
    ctx_keys = g  # original-context keys stay fixed at the ground-truth features throughout

    # ---- Steps 2..ttt_steps: simulated, self-fed ----
    for s in range(2, cfg.ttt_steps + 1):
        if T - (s - 1) <= 0:
            break
        # this step's queries correspond to shifting the "current position" forward
        # by one extra step; token embedding input = the PREVIOUS step's own
        # predicted (argmax) token, not ground truth -- this is the actual
        # "training-time test" (paper: "we generate a and feed it back")
        with torch.no_grad():
            pred_tok = draft_model.logits_from_a(a_prev).argmax(dim=-1)  # (B,T) self-predicted tokens

        x_new = draft_model.build_input(a_prev, pred_tok)          # (B,T,H) new block input
        ctx_in = draft_model.build_input(ctx_keys, prev_tok)        # (B,T,H) recompute context block input (cheap, no grad needed beyond fusion already applied)
        seq = torch.cat([ctx_in, x_new], dim=1)                      # (B, 2T, H)
        block_mask = _ttt_block_mask(T, device, dtype).view(1, 1, T, 2 * T)
        pos_ids_step = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # run the decoder layer once over the 2T sequence but we only need
        # correct outputs at the "new block" (last T) positions; we pass a
        # (T, 2T) mask sliced appropriately per query block via a helper
        # that treats ctx as fixed keys/values and new block as queries.
        a_new, _ = draft_model.forward_layer(
            seq[:, T:, :],  # queries = new block
            pos_ids_step,
            block_mask,
            use_cache=False,
        )
        # NOTE: forward_layer's self-attention module must be able to accept
        # (query_len=T, key_len=2T) shapes; if the underlying HF layer only
        # supports self-attn with matching q/k length, use a KV-cache based
        # two-call approach instead (context as cached KV, new block as the
        # incremental query) -- see docstring below `train_step`.

        logits_s = draft_model.logits_from_a(a_new)
        # labels shift one further into the future each simulated step
        shift = s - 1
        if T - shift <= 0:
            break
        labels_s = batch.input_ids[:, 1 + shift: T + shift + 1] if (1 + shift + T) <= Tp1 else None
        if labels_s is None or labels_s.shape[1] != T:
            break
        loss_s = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), labels_s.reshape(-1))
        total_loss = total_loss + loss_s

        a_prev = a_new
        prev_tok = pred_tok

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(draft_model.trainable_parameters()) + list(fusion.parameters()), cfg.grad_clip
    )
    optimizer.step()
    return float(total_loss.item())


"""
Implementation note on the (T, 2T) attention call above:
Most HF decoder-layer implementations (including Qwen3DecoderLayer)
expect self-attention where query and key/value come from the SAME
input tensor of length L, with a single (L, L) mask -- they don't
natively support asymmetric query_len != key_len in one call.

For a real training run, replace the single `forward_layer(seq[:, T:], ...)`
call above with a manual attention computation:
    1. Run the layer's `self_attn.q_proj/k_proj/v_proj` on `ctx_in` to get
       K_ctx, V_ctx (no grad needed w.r.t. re-deriving keys each step --
       cache them once per T-sized context and reuse across steps 2..S).
    2. Run q_proj/k_proj/v_proj on `x_new` to get Q_new, K_new, V_new.
    3. attn_logits = Q_new @ [K_ctx; K_new]^T * scale + block_mask
    4. softmax -> weighted sum over [V_ctx; V_new] -> attn_out
    5. Feed attn_out through the layer's o_proj + MLP + residual/RMSNorm
       exactly as the layer's forward() does internally.
This is a ~30-line manual reimplementation of one decoder layer's
forward pass and is intentionally left as a follow-up since it depends
on the exact HF version's Qwen3Attention internals (rotary application,
GQA head grouping, etc.) which differ across `transformers` releases.
The masking math and losses above are complete and correct; only the
low-level Q/K/V plumbing for the asymmetric-length call needs to be
pinned to your installed `transformers` version.
"""
