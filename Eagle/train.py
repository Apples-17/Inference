from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import TargetFeatureExtractor

def _causal_mask(T, device, dtype):
    neg=torch.finfo(dtype).min
    m=torch.full((T, T), neg, device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)


def _ttt_block_mask(T, device, dtype):
    neg=torch.finfo(dtype).min
    ctx=torch.triu(torch.full((T, T), neg, device=device, dtype=dtype), diagonal=1) 
    ctx= torch.triu(torch.full((T, T), neg, device=device, dtype=dtype), diagonal=0)
    diag= torch.diag(torch.zeros(T, device=device, dtype=dtype))
    diag= torch.where(torch.eye(T, device=device, dtype=torch.bool), diag, torch.full_like(diag, neg))
    return torch.cat([ctx, diag], dim=-1)  # (T, 2T)


@dataclass
class TTTBatch:
    input_ids: torch.Tensor     


def train_step(target_model, draft_model, fusion, batch: TTTBatch, cfg, optimizer)->float:
    device = batch.input_ids.device
    B, Tp1 = batch.input_ids.shape
    T=Tp1-1
    tokens = batch.input_ids[:, :T]       
    labels_step1 = batch.input_ids[:, 1:T + 1]  

    extractor = TargetFeatureExtractor(target_model, cfg)
    with torch.no_grad():
        _ = target_model(input_ids=batch.input_ids[:, :T], use_cache=False)
        feats=extractor.pop()
    extractor.remove()

    g=fusion(feats.low, feats.mid, feats.high)  

    dtype = g.dtype
    total_loss = 0.0

    # 1. native training step
    prev_tok=torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)  # shift for "previous sampled token"
    x1=draft_model.build_input(g, prev_tok)
    mask1=_causal_mask(T, device, dtype).view(1, 1, T, T)
    pos_ids=torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    a1, _=draft_model.forward_layer(x1, pos_ids, mask1, use_cache=False)
    logits1=draft_model.logits_from_a(a1)
    loss1 =F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), labels_step1.reshape(-1))
    total_loss= total_loss + loss1

    a_prev = a1
    labels_prev =labels_step1
    ctx_keys = g  

    #2..ttt_steps: simulated, self-fed 
    for s in range(2, cfg.ttt_steps + 1):
        if T - (s - 1) <= 0:
            break
        with torch.no_grad():
            pred_tok = draft_model.logits_from_a(a_prev).argmax(dim=-1)  # (B,T) self-predicted tokens

        x_new = draft_model.build_input(a_prev, pred_tok)         
        ctx_in = draft_model.build_input(ctx_keys, prev_tok)        
        seq = torch.cat([ctx_in, x_new], dim=1)                      
        block_mask = _ttt_block_mask(T, device, dtype).view(1, 1, T, 2 * T)
        pos_ids_step = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        a_new, _=draft_model.forward_layer(
            seq[:, T:, :],  # queries = new block
            pos_ids_step,
            block_mask,
            use_cache=False,
        )

        logits_s = draft_model.logits_from_a(a_new)
        shift = s - 1
        if T-shift <= 0:
            break
        labels_s=batch.input_ids[:, 1 + shift: T + shift + 1] if (1 + shift + T) <= Tp1 else None
        if labels_s is None or labels_s.shape[1]!=T:
            break
        loss_s=F.cross_entropy(logits_s.reshape(-1,logits_s.size(-1)),labels_s.reshape(-1))
        total_loss=total_loss+loss_s

        a_prev=a_new
        prev_tok=pred_tok

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(draft_model.trainable_parameters()) + list(fusion.parameters()), cfg.grad_clip
    )
    optimizer.step()
    return float(total_loss.item())
