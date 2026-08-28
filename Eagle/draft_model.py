import torch
import torch.nn as nn
import copy

class EagleDraftModel(nn.Module):
    def __init__(self, target_model, cfg):
        super().__init__()
        base=getattr(target_model, "model", target_model)
        layer_cls=type(base.layers[0])
        target_config=copy.deepcopy(base.config)

        hidden=target_config.hidden_size
        self.hidden_size=hidden

        self.embed_tokens=base.embed_tokens          
        self.lm_head=getattr(target_model, "lm_head")  
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        self.rotary_emb = getattr(base, "rotary_emb", None)

        #trained EAGLE components
        from .features import FeatureFusion
        self.fusion=FeatureFusion(hidden)            
        self.input_fc=nn.Linear(2 * hidden, hidden, bias=False)  
        self.decoder_layer=layer_cls(target_config, layer_idx=0)

        self.config=target_config

    def build_input(self, feat_or_a: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        embeds=self.embed_tokens(token_ids)
        x=torch.cat([feat_or_a, embeds], dim=-1)
        return self.input_fc(x)

    def forward_layer(self, hidden_states, position_ids, attention_mask, position_embeddings=None,
                       past_key_value=None, use_cache=False):
        if position_embeddings is None and self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        out = self.decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        a = out[0] if isinstance(out, tuple) else out
        new_cache =out[-1] if (isinstance(out, tuple) and use_cache) else None
        return a, new_cache

    @torch.no_grad()
    def logits_from_a(self, a: torch.Tensor) -> torch.Tensor:
        return self.lm_head(a)

    def trainable_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith("embed_tokens") or n.startswith("lm_head"):
                continue
            yield p
