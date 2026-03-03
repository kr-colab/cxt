"""Tests for cxt.translate — inference pipeline."""

import numpy as np
import torch
import pytest

from cxt.config import ModelConfig
from cxt.model import TokenFreeDecoder
from cxt.translate import (
    generate_causal_mask,
    to_log_times,
    TIMES,
    GRID_SIZE,
    generate,
    translate_from_ts,
    translate,
)
from cxt.simulate import simulate_parameterized_tree_sequence


@pytest.fixture
def tiny_model():
    """Minimal model for fast integration tests.
    n_embd = 4 * sample_scale_embd * num_samples = 4*2*50 = 400.
    """
    cfg = ModelConfig(
        n_layer=2, n_embd=400, n_head=4,
        output_dim=326, num_samples=50,
        sample_scale_embd=2, combined_dim=1001,
        window_size=2000, bias=False, dropout=0.0,
        use_kv_cache=True, batch_size=1,
    )
    model = TokenFreeDecoder(cfg)
    model.enable_kv_cache(batch_size=1)
    model.eval()
    return model


@pytest.fixture
def small_ts():
    return simulate_parameterized_tree_sequence(seed=42, samples=25,
                                                 sequence_length=1e6)


class TestToLogTimes:
    def test_basic_shape(self):
        B = 3
        rng = np.random.default_rng(42)
        yhat = torch.randint(2, 326, (B, 501))
        result = to_log_times(yhat)
        assert result.shape == (B, 500)

    def test_rep_mode_shape(self):
        # rep_mode expects 3D: (n_reps, B, seq_len)
        n_reps, B, seq_len = 5, 3, 501
        yhat = torch.randint(2, 326, (n_reps, B, seq_len))
        result = to_log_times(yhat, rep_mode=True)
        assert result.shape == (B, n_reps, seq_len - 1)

    def test_values_in_times_range(self):
        yhat = torch.randint(2, 326, (1, 501))
        result = to_log_times(yhat)
        assert np.all(result >= TIMES[0])
        assert np.all(result <= TIMES[-1])


class TestGenerate:
    def test_output_shape(self, tiny_model):
        src = torch.randn(1, 2, 4, 500, 50)
        out = generate(
            tiny_model, src, B=1, device="cpu",
            top_k=10, base_seed=42,
            cache_matching=False, progress=False, decode_bar=False,
        )
        assert out.shape[0] == 1
        assert out.shape[1] == 501

    def test_deterministic_with_seed(self, tiny_model):
        src = torch.randn(1, 2, 4, 500, 50)
        o1 = generate(tiny_model, src, B=1, device="cpu", top_k=10,
                       base_seed=42, cache_matching=False, progress=False)
        tiny_model.clear_cache()
        o2 = generate(tiny_model, src, B=1, device="cpu", top_k=10,
                       base_seed=42, cache_matching=False, progress=False)
        torch.testing.assert_close(o1, o2)


class TestTranslateFromTs:
    def test_basic_integration(self, tiny_model, small_ts):
        """End-to-end test: ts -> translate -> (tmrca, index_map)."""
        tmrca, index_map = translate_from_ts(
            small_ts, tiny_model,
            blocks=[(0, 1_000_000)],
            pivot_pairs=[(0, 1)],
            devices=["cpu"],
            B=1, B_per_device=1,
            n_reps=1, base_seed=42,
            top_k=10, cache_matching=True,
            progress=False, decode_bar=False,
            build_workers=1,
        )
        assert tmrca.ndim >= 2
        assert index_map.ndim == 2


class TestTranslateDispatch:
    def test_auto_detect_ts(self, tiny_model, small_ts):
        tmrca, index_map = translate(
            small_ts, tiny_model,
            blocks=[(0, 1_000_000)],
            pivot_pairs=[(0, 1)],
            devices=["cpu"],
            B=1, B_per_device=1,
            n_reps=1, base_seed=42,
            top_k=10, cache_matching=True,
            progress=False, decode_bar=False,
            build_workers=1,
        )
        assert tmrca.ndim >= 2

    def test_explicit_data_type(self, tiny_model, small_ts):
        tmrca, index_map = translate(
            small_ts, tiny_model,
            data_type="ts",
            blocks=[(0, 1_000_000)],
            pivot_pairs=[(0, 1)],
            devices=["cpu"],
            B=1, B_per_device=1,
            n_reps=1, base_seed=42,
            top_k=10, cache_matching=True,
            progress=False, decode_bar=False,
            build_workers=1,
        )
        assert tmrca.ndim >= 2

    def test_invalid_data_type_raises(self, tiny_model):
        with pytest.raises((ValueError, TypeError)):
            translate(
                "not_a_valid_input", tiny_model,
                data_type="invalid",
                blocks=[(0, 100000)],
                pivot_pairs=[(0, 1)],
            )
