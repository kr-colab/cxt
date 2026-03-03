"""Tests for cxt.model and cxt.modules using the new ModelConfig API."""

import torch
import pytest
import numpy as np

from cxt.config import ModelConfig, PRESETS
from cxt.model import TokenFreeDecoder
from cxt.modules import (
    MutationsToLatentSpace,
    MLP,
    LayerNorm,
    CausalSelfAttention,
    Block,
)
from cxt.translate import generate_causal_mask


@pytest.fixture
def narrow_config():
    return PRESETS["narrow"]


@pytest.fixture
def small_config():
    """Minimal config for fast tests.
    n_embd must equal 4 * sample_scale_embd * num_samples (from MutationsToLatentSpace).
    """
    return ModelConfig(
        n_layer=2, n_embd=80, n_head=2,
        output_dim=326, num_samples=10,
        sample_scale_embd=2, combined_dim=101,
        window_size=200, bias=False, dropout=0.0,
    )


class TestMutationsToLatentSpace:
    def test_forward_shape(self):
        cfg = ModelConfig(num_samples=10, sample_scale_embd=2, dropout=0.0)
        model = MutationsToLatentSpace(cfg)
        B, C, M, W, S = 2, 2, 4, 50, 10
        x = torch.randn(B, C, M, W, S)
        out = model(x)
        expected_embd = S * cfg.sample_scale_embd * 4  # 10*2*4=80
        assert out.shape == (B, W, expected_embd)

    def test_gradient_flows(self):
        cfg = ModelConfig(num_samples=10, sample_scale_embd=2, dropout=0.0,
                          mask_singletons=False)
        model = MutationsToLatentSpace(cfg)
        x = torch.randn(1, 2, 4, 10, 10, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


class TestMLP:
    def test_forward_shape(self):
        cfg = ModelConfig(n_embd=80, bias=False, dropout=0.0)
        model = MLP(cfg)
        x = torch.randn(4, 80)
        assert model(x).shape == (4, 80)


class TestLayerNorm:
    def test_forward(self):
        ln = LayerNorm(80, use_bias=True)
        x = torch.randn(4, 80)
        out = ln(x)
        assert out.shape == (4, 80)
        # should be roughly normalized
        np.testing.assert_allclose(out.mean(dim=-1).detach().numpy(), 0.0, atol=0.1)


class TestCausalSelfAttention:
    def test_without_kv_cache(self):
        cfg = ModelConfig(n_embd=80, n_head=2, combined_dim=50,
                          use_kv_cache=False, bias=False, dropout=0.0)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        assert attn.cache_k is None
        assert attn.cache_v is None
        x = torch.randn(1, 50, 80)
        mask = generate_causal_mask(50)
        out = attn(x, mask)
        assert out.shape == (1, 50, 80)

    def test_with_kv_cache(self):
        cfg = ModelConfig(n_embd=64, n_head=2, combined_dim=50,
                          use_kv_cache=True, batch_size=1, bias=False, dropout=0.0)
        attn = CausalSelfAttention(cfg, layer_idx=0)
        assert attn.cache_k is not None
        assert attn.cache_v is not None
        # KV cache uses config.combined_dim (not the smaller value)
        assert attn.cache_k.shape[0] == 1
        assert attn.cache_k.shape[1] == 2  # n_head
        assert attn.cache_k.shape[3] == 32  # head_size


class TestBlock:
    def test_forward(self):
        cfg = ModelConfig(n_embd=80, n_head=2, combined_dim=50,
                          use_kv_cache=False, bias=False, dropout=0.0)
        block = Block(cfg, layer_idx=0)
        x = torch.randn(1, 50, 80)
        mask = generate_causal_mask(50)
        out = block(x, mask)
        assert out.shape == (1, 50, 80)


class TestTokenFreeDecoder:
    def test_construction_from_preset(self, narrow_config):
        model = TokenFreeDecoder(narrow_config)
        assert len(model.transformer.h) == 6

    def test_construction_from_small_config(self, small_config):
        model = TokenFreeDecoder(small_config)
        assert len(model.transformer.h) == 2

    def test_forward_no_loss(self, small_config):
        model = TokenFreeDecoder(small_config)
        B = 2
        # combined_dim=101 => src=50 windows, tgt=51 tokens
        # MutationsToLatentSpace expects (B, 2, 4, n_windows, num_samples)
        x = torch.randn(B, 2, 4, 50, small_config.num_samples)
        y = torch.randint(0, 326, (B, 51)).long()
        mask = generate_causal_mask(small_config.combined_dim)
        mask = mask.repeat(B, 1, 1, 1)
        out = model(x, y, mask, calculate_loss=False)
        assert out.shape == (B, 51, small_config.output_dim)

    def test_forward_with_loss(self, small_config):
        model = TokenFreeDecoder(small_config)
        B = 2
        x = torch.randn(B, 2, 4, 50, small_config.num_samples)
        y = torch.randint(0, 326, (B, 51)).long()
        mask = generate_causal_mask(small_config.combined_dim)
        mask = mask.repeat(B, 1, 1, 1)
        logits, loss = model(x, y, mask, calculate_loss=True)
        assert logits.shape == (B, 51, small_config.output_dim)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_enable_kv_cache(self, small_config):
        model = TokenFreeDecoder(small_config)
        for block in model.transformer.h:
            assert block.attn.cache_k is None
        model.enable_kv_cache(batch_size=2)
        for block in model.transformer.h:
            assert block.attn.cache_k is not None
            assert block.attn.cache_k.shape[0] == 2

    def test_clear_cache(self, small_config):
        model = TokenFreeDecoder(small_config)
        model.enable_kv_cache(batch_size=1)
        for block in model.transformer.h:
            block.attn.cache_k.fill_(1.0)
        model.clear_cache()
        for block in model.transformer.h:
            assert block.attn.cache_k.abs().sum().item() == 0.0

    def test_parameter_count(self, small_config):
        model = TokenFreeDecoder(small_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_eval_mode(self, small_config):
        model = TokenFreeDecoder(small_config)
        model.eval()
        assert not model.training


class TestGenerateCausalMask:
    def test_shape(self):
        mask = generate_causal_mask(100)
        assert mask.shape == (1, 1, 100, 100)

    def test_lower_triangular(self):
        mask = generate_causal_mask(10)
        m = mask[0, 0].numpy()
        for i in range(10):
            for j in range(10):
                if j <= i:
                    assert m[i, j] == True
                else:
                    assert m[i, j] == False

    def test_full_attention_block(self):
        mask = generate_causal_mask(10, full_attention_n=5)
        m = mask[0, 0].numpy()
        for i in range(5):
            for j in range(5):
                assert m[i, j] == True

    def test_caching(self):
        m1 = generate_causal_mask(50)
        m2 = generate_causal_mask(50)
        assert m1 is m2  # same object from cache
