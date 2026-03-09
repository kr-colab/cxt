"""Tests for cxt.checkpoint — checkpoint registry and loading logic."""

import pytest

from cxt.checkpoint import (
    CHECKPOINT_REGISTRY,
    LEGACY_REGISTRY,
    GITHUB_BASE,
    LEGACY_GITHUB_BASE,
    _cache_dir,
)
from cxt.config import PRESETS, AdapterConfig


class TestCheckpointRegistry:
    def test_all_non_legacy_presets_have_entries(self):
        for name in PRESETS:
            if name == "residual":
                assert name in LEGACY_REGISTRY
                continue
            assert name in CHECKPOINT_REGISTRY, f"Missing registry entry for preset {name!r}"

    def test_all_entries_have_filename(self):
        for name, info in CHECKPOINT_REGISTRY.items():
            assert "filename" in info, f"No filename for {name!r}"
            assert info["filename"].endswith(".ckpt")

    def test_adapter_entries_have_base_preset(self):
        for name, info in CHECKPOINT_REGISTRY.items():
            if "adapter" in info:
                assert "base_preset" in info, f"Adapter {name!r} missing base_preset"
                assert info["base_preset"] in PRESETS

    def test_adapter_config_type(self):
        for name, info in CHECKPOINT_REGISTRY.items():
            if "adapter" in info:
                assert isinstance(info["adapter"], AdapterConfig)

    def test_known_models(self):
        expected = {
            "broad", "broad+adapter", "narrow", "broad_w200",
            "w200_wmissing", "w200_wmissing_adapter",
        }
        assert set(CHECKPOINT_REGISTRY.keys()) == expected

    def test_legacy_includes_residual(self):
        assert "residual" in LEGACY_REGISTRY

    def test_legacy_is_superset(self):
        for name in CHECKPOINT_REGISTRY:
            assert name in LEGACY_REGISTRY

    def test_github_base_url(self):
        assert GITHUB_BASE.startswith("https://")
        assert "cxt" in GITHUB_BASE

    def test_legacy_github_base_url(self):
        assert LEGACY_GITHUB_BASE.startswith("https://")
        assert "legacy" in LEGACY_GITHUB_BASE


class TestCacheDir:
    def test_returns_path(self):
        d = _cache_dir()
        assert d.exists()
        assert "cxt" in str(d)
