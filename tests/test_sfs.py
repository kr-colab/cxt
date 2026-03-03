"""Tests for cxt.sfs — SFS computation and source building."""

import numpy as np
import pytest

from cxt.sfs import calculate_window_sfs, basic_filtering, build_src, W_MULTIPLIERS


class TestCalculateWindowSfs:
    def test_empty_positions(self):
        pos = np.array([], dtype=float)
        freq = np.array([], dtype=int)
        sfs = calculate_window_sfs(pos, freq, window_size=100, sequence_length=1000,
                                   num_samples=10, step_size=100)
        assert sfs.shape == (10, 10)
        assert sfs.sum() == 0

    def test_single_snp(self):
        pos = np.array([50.0])
        freq = np.array([3])
        sfs = calculate_window_sfs(pos, freq, window_size=100, sequence_length=100,
                                   num_samples=10, step_size=100)
        assert sfs.shape == (1, 10)
        assert sfs[0, 3] == 1
        assert sfs.sum() == 1

    def test_multiple_windows(self):
        pos = np.array([50.0, 150.0, 250.0])
        freq = np.array([1, 2, 3])
        sfs = calculate_window_sfs(pos, freq, window_size=100, sequence_length=300,
                                   num_samples=10, step_size=100)
        assert sfs.shape == (3, 10)
        assert sfs[0, 1] == 1
        assert sfs[1, 2] == 1
        assert sfs[2, 3] == 1

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        pos = np.sort(rng.uniform(0, 1e6, 500))
        freq = rng.integers(1, 50, 500)
        sfs = calculate_window_sfs(pos, freq, window_size=2000, sequence_length=1e6,
                                   num_samples=50, step_size=2000)
        assert sfs.shape == (500, 50)


class TestBasicFiltering:
    def test_removes_invariant_sites(self):
        gm = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],  # all zeros
        ]).T  # shape: (4 samples, 3 sites)
        pos = np.array([100, 200, 300], dtype=float)
        gm_f, pos_f = basic_filtering(gm, pos, num_samples=4)
        assert len(pos_f) <= len(pos)
        assert gm_f.shape[1] == len(pos_f)

    def test_removes_multiallelic(self):
        gm = np.array([
            [0, 2, 1],
            [0, 0, 1],
            [1, 0, 0],
        ]).T  # (3, 3)
        pos = np.array([10, 20, 30], dtype=float)
        gm_f, pos_f = basic_filtering(gm, pos, num_samples=3)
        assert not np.any(gm_f >= 2)


class TestBuildSrc:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        n_sites = 200
        n_samples = 10
        pos = np.sort(rng.uniform(0, 1e6, n_sites))
        gm = rng.integers(0, 2, (n_samples, n_sites)).astype(np.int8)
        src = build_src(pos, gm, pivot_id_A=0, pivot_id_B=1,
                        sequence_length=1e6, step_size=2000)
        n_steps = int(np.ceil(1e6 / 2000))
        assert src.shape[0] == 2  # xor/xnor channels
        assert src.shape[1] == len(W_MULTIPLIERS)
        assert src.shape[2] == n_steps
        assert src.shape[3] == n_samples
