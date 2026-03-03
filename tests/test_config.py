"""Tests for cxt.config — ModelConfig, PRESETS, AdapterConfig, TrainingConfig."""

import pytest
from dataclasses import fields

from cxt.config import (
    ModelConfig,
    PRESETS,
    AdapterConfig,
    TrainingConfig,
    NarrowModelConfig,
    BroadModelConfig,
    TokenFreeDecoderConfig,
)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_layer == 10
        assert cfg.n_embd == 400
        assert cfg.n_head == 4
        assert cfg.output_dim == 326
        assert cfg.num_samples == 50
        assert cfg.mask_singletons is True
        assert cfg.use_kv_cache is False
        assert cfg.bias is False
        assert cfg.dropout == 0.1
        assert cfg.device == "cpu"
        assert cfg.batch_size == 1

    def test_for_inference(self):
        cfg = ModelConfig()
        inf = cfg.for_inference(batch_size=4, device="cuda:0")
        assert inf.use_kv_cache is True
        assert inf.batch_size == 4
        assert inf.device == "cuda:0"
        assert cfg.use_kv_cache is False  # original unchanged

    def test_for_training(self):
        cfg = ModelConfig()
        tr = cfg.for_training(batch_size=64, device="cuda")
        assert tr.use_kv_cache is False
        assert tr.batch_size == 64
        assert tr.device == "cuda"

    def test_custom_values(self):
        cfg = ModelConfig(n_layer=6, window_size=200, mask_singletons=False)
        assert cfg.n_layer == 6
        assert cfg.window_size == 200
        assert cfg.mask_singletons is False


class TestPresets:
    def test_all_presets_exist(self):
        expected = {"narrow", "broad", "broad_w200", "residual", "w200_wmissing"}
        assert set(PRESETS.keys()) == expected

    def test_narrow(self):
        assert PRESETS["narrow"].n_layer == 6
        assert PRESETS["narrow"].window_size == 2000

    def test_broad(self):
        assert PRESETS["broad"].n_layer == 10
        assert PRESETS["broad"].window_size == 2000

    def test_broad_w200(self):
        assert PRESETS["broad_w200"].window_size == 200
        assert PRESETS["broad_w200"].n_layer == 10

    def test_w200_wmissing(self):
        cfg = PRESETS["w200_wmissing"]
        assert cfg.window_size == 200
        assert cfg.mask_singletons is False

    def test_presets_are_independent_copies(self):
        a = PRESETS["broad"]
        b = PRESETS["narrow"]
        assert a is not b
        assert a.n_layer != b.n_layer


class TestAdapterConfig:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.ie_in == 10
        assert cfg.ie_out == 50
        assert cfg.bottleneck == 32
        assert cfg.use_se is True


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.max_lr == 3e-4
        assert cfg.batch_size == 128
        assert cfg.betas == (0.9, 0.95)


class TestBackwardCompatibility:
    def test_aliases_are_model_config(self):
        assert NarrowModelConfig is ModelConfig
        assert BroadModelConfig is ModelConfig
        assert TokenFreeDecoderConfig is ModelConfig

    def test_aliases_construct_same_object(self):
        a = NarrowModelConfig(n_layer=6)
        b = ModelConfig(n_layer=6)
        assert a == b
