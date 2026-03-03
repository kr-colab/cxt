"""Tests for cxt.train — training components (IEAdapter, FrozenDecoderWithAdapter, LitDecoder)."""

import torch
import pytest

from cxt.config import ModelConfig
from cxt.model import TokenFreeDecoder

try:
    from cxt.train import IEAdapter, FrozenDecoderWithAdapter, _set_trainable, _select_backbone_params
    HAS_LIGHTNING = True
except (ImportError, RuntimeError, AttributeError):
    HAS_LIGHTNING = False

pytestmark = pytest.mark.skipif(not HAS_LIGHTNING, reason="lightning not available")


class TestIEAdapter:
    def test_forward_shape(self):
        adapter = IEAdapter(ie_in=10, ie_out=50, bottleneck=32, dropout=0.0)
        x = torch.randn(4, 2, 4, 500, 10)
        out = adapter(x)
        assert out.shape == (4, 2, 4, 500, 50)

    def test_gradient_flows(self):
        adapter = IEAdapter(ie_in=5, ie_out=20, bottleneck=16, dropout=0.0)
        x = torch.randn(1, 2, 4, 10, 5, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None


class TestFrozenDecoderWithAdapter:
    @pytest.fixture
    def small_model(self):
        cfg = ModelConfig(
            n_layer=2, n_embd=80, n_head=2,
            output_dim=326, num_samples=10,
            sample_scale_embd=2, combined_dim=101,
            window_size=200, bias=False, dropout=0.0,
        )
        return TokenFreeDecoder(cfg)

    def test_construction(self, small_model):
        wrapped = FrozenDecoderWithAdapter(
            small_model, ie_in=5, ie_out=10,
            adapter_bottleneck=16, adapter_dropout=0.0,
        )
        assert wrapped.adapter is not None
        assert wrapped.backbone is not None

    def test_backbone_mostly_frozen(self, small_model):
        wrapped = FrozenDecoderWithAdapter(
            small_model, ie_in=5, ie_out=10,
            adapter_bottleneck=16, adapter_dropout=0.0,
            unfreeze="ln_lastN", last_n=1,
        )
        frozen_count = sum(1 for p in wrapped.backbone.parameters() if not p.requires_grad)
        total_count = sum(1 for p in wrapped.backbone.parameters())
        assert frozen_count > total_count * 0.5

    def test_forward(self, small_model):
        cfg = small_model.config
        wrapped = FrozenDecoderWithAdapter(
            small_model, ie_in=5, ie_out=cfg.num_samples,
            adapter_bottleneck=16, adapter_dropout=0.0,
        )
        B = 2
        NW = cfg.combined_dim - 51
        x = torch.randn(B, 2, 4, NW, 5)
        y = torch.randint(0, 326, (B, 51)).long()
        from cxt.translate import generate_causal_mask
        mask = generate_causal_mask(cfg.combined_dim).repeat(B, 1, 1, 1)
        logits, loss = wrapped(x, y, mask)
        assert loss.ndim == 0
        assert logits.shape == (B, 51, cfg.output_dim)


class TestSetTrainable:
    def test_freeze(self):
        layer = torch.nn.Linear(10, 10)
        _set_trainable(layer, False)
        assert not any(p.requires_grad for p in layer.parameters())

    def test_unfreeze(self):
        layer = torch.nn.Linear(10, 10)
        _set_trainable(layer, False)
        _set_trainable(layer, True)
        assert all(p.requires_grad for p in layer.parameters())


class TestSelectBackboneParams:
    def test_none_strategy(self):
        cfg = ModelConfig(n_layer=2, n_embd=64, n_head=2,
                          output_dim=326, num_samples=10,
                          sample_scale_embd=2, combined_dim=101)
        model = TokenFreeDecoder(cfg)
        params = _select_backbone_params(model, strategy="none")
        assert len(params) == 0

    def test_ln_lastN_strategy(self):
        cfg = ModelConfig(n_layer=2, n_embd=64, n_head=2,
                          output_dim=326, num_samples=10,
                          sample_scale_embd=2, combined_dim=101)
        model = TokenFreeDecoder(cfg)
        params = _select_backbone_params(model, strategy="ln_lastN", last_n=1)
        assert len(params) > 0
