"""Tests that every registered model loads and runs a quick inference pass.

These tests download checkpoints on first run (~200 MB total), then cache
them under ``~/.cache/cxt/checkpoints/``.
"""

import numpy as np
import torch
import pytest

import cxt
from cxt.checkpoint import CHECKPOINT_REGISTRY, load_model
from cxt.config import PRESETS
from cxt.translate import generate_causal_mask, generate, to_log_times
from cxt.simulate import simulate_parameterized_tree_sequence
from cxt.sfs import build_src, basic_filtering


ALL_MODEL_TYPES = sorted(CHECKPOINT_REGISTRY.keys())
BASE_MODELS = [k for k in ALL_MODEL_TYPES if "adapter" not in k]
ADAPTER_MODELS = [k for k in ALL_MODEL_TYPES if "adapter" in k]

try:
    load_model("broad+adapter", device="cpu")
    HAS_LIGHTNING = True
except (ImportError, RuntimeError, AttributeError):
    HAS_LIGHTNING = False


@pytest.fixture(scope="module")
def small_ts():
    """A small tree sequence for quick integration tests."""
    return simulate_parameterized_tree_sequence(seed=42, samples=25, sequence_length=1e6)


@pytest.fixture(scope="module")
def small_ts_src(small_ts):
    """Pre-built source tensor (step_size=2000, always 500 windows)."""
    gm = small_ts.genotype_matrix().T
    positions = small_ts.tables.sites.position
    gm_f, pos_f = basic_filtering(gm, positions, num_samples=gm.shape[0])
    src = build_src(pos_f, gm_f, pivot_id_A=0, pivot_id_B=1,
                    sequence_length=1e6, step_size=2000)
    return torch.tensor(src, dtype=torch.float32).unsqueeze(0)


class TestLoadAllModels:
    @pytest.mark.parametrize("model_type", BASE_MODELS)
    def test_load_base_model(self, model_type):
        model = load_model(model_type, device="cpu")
        assert model is not None
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        print(f"  {model_type}: {n_params:,} parameters")

    @pytest.mark.skipif(not HAS_LIGHTNING, reason="lightning/torchvision not available")
    @pytest.mark.parametrize("model_type", ADAPTER_MODELS)
    def test_load_adapter_model(self, model_type):
        model = load_model(model_type, device="cpu")
        assert model is not None
        assert hasattr(model, "backbone")
        assert hasattr(model, "adapter")
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        print(f"  {model_type}: {n_params:,} parameters")


class TestBaseModelInference:
    @pytest.mark.parametrize("model_type", BASE_MODELS)
    def test_single_token_generation(self, model_type, small_ts_src):
        """Load model, feed source through encoder, generate one token."""
        model = load_model(model_type, device="cpu")

        attn_mask = generate_causal_mask(1001, full_attention_n=501, device="cpu")

        with torch.inference_mode():
            model.clear_cache()
            logits = model(small_ts_src, None, attn_mask, calculate_loss=False,
                           use_cache=True, position=0)
            assert logits.shape == (1, 500, 326)

            next_input = torch.ones(1, 1, dtype=torch.long)
            logits_step = model(small_ts_src, next_input, attn_mask,
                                calculate_loss=False, use_cache=True, position=500)
            assert logits_step.shape == (1, 1, 326)

    @pytest.mark.parametrize("model_type", BASE_MODELS)
    def test_full_autoregressive_generation(self, model_type, small_ts_src):
        """Run full 500-token autoregressive decode and convert to log-times."""
        model = load_model(model_type, device="cpu")

        tokens = generate(
            model, small_ts_src, B=1, device="cpu",
            top_k=50, base_seed=42,
            cache_matching=True, progress=False, decode_bar=False,
        )
        assert tokens.shape == (1, 501)
        assert tokens[:, 0].item() == 1  # start token

        log_times = to_log_times(tokens)
        assert log_times.shape == (1, 500)
        assert np.all(np.isfinite(log_times))
        assert np.all(log_times >= 3.0)
        assert np.all(log_times <= 17.0)


class TestAdapterModelInference:
    @pytest.fixture(scope="class")
    def adapter_src(self, small_ts):
        """Source tensor with 10 samples (ie_in=10 for adapter models)."""
        ts_small = simulate_parameterized_tree_sequence(seed=42, samples=5, sequence_length=1e6)
        gm = ts_small.genotype_matrix().T
        positions = ts_small.tables.sites.position
        gm_f, pos_f = basic_filtering(gm, positions, num_samples=gm.shape[0])
        src = build_src(pos_f, gm_f, pivot_id_A=0, pivot_id_B=1,
                        sequence_length=1e6, step_size=2000)
        return torch.tensor(src, dtype=torch.float32).unsqueeze(0)

    @pytest.mark.skipif(not HAS_LIGHTNING, reason="lightning/torchvision not available")
    @pytest.mark.parametrize("model_type", ADAPTER_MODELS)
    def test_adapter_full_generation(self, model_type, adapter_src):
        """Full autoregressive decode through adapter + backbone."""
        model = load_model(model_type, device="cpu")
        backbone = model.backbone
        adapter = model.adapter

        tokens = generate(
            backbone, adapter_src, B=1, device="cpu",
            top_k=50, base_seed=42,
            cache_matching=True, progress=False, decode_bar=False,
            adapter=adapter,
        )
        assert tokens.shape == (1, 501)

        log_times = to_log_times(tokens)
        assert log_times.shape == (1, 500)
        assert np.all(np.isfinite(log_times))


class TestEndToEndTranslate:
    @pytest.mark.parametrize("model_type", BASE_MODELS)
    def test_translate_from_ts(self, model_type, small_ts):
        """Full cxt.translate pipeline for every base model."""
        model = load_model(model_type, device="cpu")
        tmrca, index_map = cxt.translate(
            small_ts, model,
            blocks=[(0, 1_000_000)],
            pivot_pairs=[(0, 1)],
            devices=["cpu"],
            B=1, B_per_device=1,
            n_reps=1, base_seed=42,
            top_k=50, cache_matching=True,
            progress=False, decode_bar=False,
            build_workers=1,
        )
        assert tmrca.ndim >= 2
        assert index_map.ndim == 2
        assert np.all(np.isfinite(tmrca))


class TestModelDeterminism:
    def test_same_seed_same_output(self, small_ts_src):
        """Same model + same seed => identical output."""
        model = load_model("broad", device="cpu")

        t1 = generate(model, small_ts_src, B=1, device="cpu", top_k=50,
                       base_seed=999, cache_matching=True, progress=False)
        t2 = generate(model, small_ts_src, B=1, device="cpu", top_k=50,
                       base_seed=999, cache_matching=True, progress=False)
        torch.testing.assert_close(t1, t2)

    def test_different_seed_different_output(self, small_ts_src):
        """Different seeds should (almost certainly) give different outputs."""
        model = load_model("broad", device="cpu")

        t1 = generate(model, small_ts_src, B=1, device="cpu", top_k=50,
                       base_seed=1, cache_matching=True, progress=False)
        t2 = generate(model, small_ts_src, B=1, device="cpu", top_k=50,
                       base_seed=9999, cache_matching=True, progress=False)
        assert not torch.equal(t1, t2)
