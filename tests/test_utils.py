"""Tests for cxt.utils — utility functions and constants."""

import numpy as np
import pytest

from cxt.utils import (
    TIMES,
    GRID_SIZE,
    xor,
    xnor,
    mse,
    discretize,
    coalescence_rates,
    population_time,
    post_process,
    accumulating_mses,
)


class TestConstants:
    def test_grid_size(self):
        assert GRID_SIZE == 324

    def test_times_shape(self):
        assert TIMES.shape == (GRID_SIZE,)
        assert TIMES[0] == 3
        assert TIMES[-1] == 17

    def test_times_monotonic(self):
        assert np.all(np.diff(TIMES) > 0)



class TestXorXnor:
    def test_xor_basic(self):
        a = np.array([0, 1, 0, 1])
        b = np.array([0, 0, 1, 1])
        result = xor(a, b)
        np.testing.assert_array_equal(result, [0, 1, 1, 0])

    def test_xnor_basic(self):
        a = np.array([0, 1, 0, 1])
        b = np.array([0, 0, 1, 1])
        result = xnor(a, b)
        np.testing.assert_array_equal(result, [1, 0, 0, 1])

    def test_xor_xnor_complement(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, 100)
        b = rng.integers(0, 2, 100)
        np.testing.assert_array_equal(xor(a, b) + xnor(a, b), np.ones(100))


class TestMse:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mse(a, a) == 0.0

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert mse(a, b) == 1.0


class TestDiscretize:
    def test_basic(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        seq = np.array([0.5, 1.5, 2.5])
        idx = discretize(seq, grid)
        assert idx.shape == seq.shape
        assert np.all(idx >= 0)
        assert np.all(idx < len(grid))

    def test_exact_values(self):
        grid = np.array([0.0, 1.0, 2.0])
        seq = np.array([0.0, 1.0, 2.0])
        idx = discretize(seq, grid)
        assert idx.shape == seq.shape
        assert np.all(idx >= 0)
        assert np.all(idx < len(grid))

    def test_clipping(self):
        grid = np.array([1.0, 2.0, 3.0])
        seq = np.array([-10.0, 100.0])
        idx = discretize(seq, grid)
        assert idx[0] == 0
        assert idx[1] == len(grid) - 1


class TestCoalescenceRates:
    def test_output_shape(self):
        times = np.array([100, 500, 1000, 5000, 10000, 50000], dtype=float)
        windows = np.array([0, 1000, 10000, 100000], dtype=float)
        rates = coalescence_rates(times, windows)
        assert rates.shape == (len(windows) - 1,)

    def test_all_same_time(self):
        times = np.full(100, 5000.0)
        windows = np.array([0, 1000, 10000, 100000], dtype=float)
        rates = coalescence_rates(times, windows)
        assert rates.shape == (len(windows) - 1,)
        # NaN can occur in empty windows; just ensure output has right shape

    def test_positive_rates(self):
        rng = np.random.default_rng(42)
        times = rng.exponential(10000, 500)
        windows = np.logspace(2, 5, 10)
        windows[0] = 0
        rates = coalescence_rates(times, windows)
        assert np.all(rates >= 0)


class TestPopulationTime:
    def test_output_shape(self):
        pt = population_time()
        assert pt.shape == (40,)

    def test_custom_params(self):
        pt = population_time(num_time_windows=20)
        assert pt.shape == (20,)

    def test_monotonic(self):
        pt = population_time()
        assert np.all(np.diff(pt) > 0)


class TestAccumulatingMses:
    def test_single_rep(self):
        yhats = [np.array([1.0, 2.0, 3.0])]
        ytrues = [np.array([1.1, 2.1, 3.1])]
        result = accumulating_mses(yhats, ytrues)
        assert len(result) == 1

    def test_decreasing_mse(self):
        rng = np.random.default_rng(42)
        true = rng.normal(0, 1, 200)
        yhats = [true + rng.normal(0, 0.1, 200) for _ in range(20)]
        ytrues = [true] * 20
        mses = accumulating_mses(yhats, ytrues)
        assert len(mses) == 20
        # averaging more reps should generally decrease or maintain MSE
        assert mses[-1] <= mses[0] + 0.1
