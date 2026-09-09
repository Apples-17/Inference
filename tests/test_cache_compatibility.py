import torch

from qwen_opt import kernels
from qwen_opt.passes import patch_static_cache


class FakeStaticLayer:
    def __init__(self, *, legacy_counter: bool):
        self.is_initialized = True
        self.keys = torch.zeros(1, 2, 8, 4)
        self.values = torch.zeros_like(self.keys)
        if legacy_counter:
            self.cumulative_length = 3

    def update(self, key_states, value_states, cache_kwargs=None):
        raise AssertionError("compatible B=1/Q=1 update should take the fused path")


class FakeCache:
    def __init__(self, layer):
        self.layers = [layer]


def fake_kv_write(key, value, cache_key, cache_value, position):
    cache_key.index_copy_(2, position, key)
    cache_value.index_copy_(2, position, value)


def test_cache_patch_supports_transformers_457_layer_without_counter(monkeypatch):
    layer = FakeStaticLayer(legacy_counter=False)
    monkeypatch.setattr(kernels, "kv_write", fake_kv_write)
    assert patch_static_cache(FakeCache(layer)) == 1
    key = torch.ones(1, 2, 1, 4)
    value = torch.full_like(key, 2.0)
    returned_key, returned_value = layer.update(
        key, value, {"cache_position": torch.tensor([4])}
    )
    torch.testing.assert_close(returned_key[:, :, 4:5], key)
    torch.testing.assert_close(returned_value[:, :, 4:5], value)
    assert not hasattr(layer, "cumulative_length")


def test_cache_patch_preserves_legacy_counter(monkeypatch):
    layer = FakeStaticLayer(legacy_counter=True)
    monkeypatch.setattr(kernels, "kv_write", fake_kv_write)
    patch_static_cache(FakeCache(layer))
    key = torch.ones(1, 2, 1, 4)
    layer.update(key, key, {"cache_position": torch.tensor([4])})
    assert layer.cumulative_length == 4
