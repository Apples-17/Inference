from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class CapturedFeatures:
    low:torch.Tensor    # (B, T, H)
    mid:torch.Tensor    # (B, T, H)
    high:torch.Tensor   # (B, T, H)

class TargetFeatureExtractor:
    def __init__(self, target_model: nn.Module, cfg):
        self.model=target_model
        layers=self._get_decoder_layers(target_model)
        n=len(layers)

        self.low_idx=max(0, int(round(cfg.low_layer_frac*(n - 1))))
        self.mid_idx=max(0, int(round(cfg.mid_layer_frac*(n - 1))))
        self.high_idx=n - 1

        self._buf = {}
        self._handles = []
        self._handles.append(layers[self.low_idx].register_forward_hook(self._make_hook("low")))
        self._handles.append(layers[self.mid_idx].register_forward_hook(self._make_hook("mid")))
        self._handles.append(layers[self.high_idx].register_forward_hook(self._make_hook("high")))

    @staticmethod
    def _get_decoder_layers(model: nn.Module):
        base=getattr(model, "model", model)
        if not hasattr(base, "layers"):
            raise AttributeError(
                "Could not find `.model.layers` on the target model; "
                "check the transformers version's Qwen3 module layout."
            )
        return base.layers

    def _make_hook(self, name):
        def hook(module, inputs, output):
            hs=output[0] if isinstance(output, tuple) else output
            self._buf[name]=hs.detach()
        return hook

    def pop(self)->CapturedFeatures:
        assert all(k in self._buf for k in ("low", "mid", "high")), \
            "No forward pass has been run yet, or hooks did not fire."
        feats=CapturedFeatures(low=self._buf["low"], mid=self._buf["mid"], high=self._buf["high"])
        self._buf={}
        return feats

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles =[]


class FeatureFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(3*hidden_size, hidden_size, bias=False)

    def forward(self, low:torch.Tensor, mid:torch.Tensor, high:torch.Tensor)->torch.Tensor:
        x=torch.cat([low, mid, high], dim=-1)
        return self.fc(x)
