"""Tests for cxt public API surface (cxt.__init__)."""

import pytest


class TestPublicAPI:
    def test_import_cxt(self):
        import cxt
        assert hasattr(cxt, "load_model")
        assert hasattr(cxt, "translate")
        assert hasattr(cxt, "ModelConfig")
        assert hasattr(cxt, "PRESETS")
        assert hasattr(cxt, "AdapterConfig")
        assert hasattr(cxt, "TrainingConfig")

    def test_all_exports(self):
        import cxt
        expected = {
            "ModelConfig", "PRESETS", "AdapterConfig", "TrainingConfig",
            "load_model", "translate",
        }
        assert expected.issubset(set(cxt.__all__))

    def test_config_accessible(self):
        from cxt import ModelConfig, PRESETS
        cfg = ModelConfig()
        assert cfg.n_layer == 10
        assert "broad" in PRESETS

    def test_submodule_imports(self):
        from cxt.sfs import calculate_window_sfs, build_src
        from cxt.correction import diversity_bias_correction
        from cxt.simulate import simulate_parameterized_tree_sequence
        from cxt.translate import generate, generate_causal_mask
        from cxt.dataset import PairDataset, discretize
        from cxt.preprocess import interpolate_tmrcas
        from cxt.checkpoint import CHECKPOINT_REGISTRY
        from cxt.utils import TIMES, coalescence_rates, xor, xnor

    def test_no_stale_imports(self):
        """Verify deleted modules are not importable."""
        # cxt.api is a leftover directory (not a module), skip it
        with pytest.raises(ImportError):
            import cxt.api2
        with pytest.raises(ImportError):
            import cxt.inference
        with pytest.raises(ImportError):
            import cxt.dataset2
        with pytest.raises(ImportError):
            import cxt.plotting
        with pytest.raises(ImportError):
            import cxt.simulation
        with pytest.raises(ImportError):
            import cxt.simulation_ts_only
        with pytest.raises(ImportError):
            import cxt.simulation_parameters
