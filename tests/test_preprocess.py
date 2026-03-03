"""Tests for cxt.preprocess — feature extraction and windowing utilities."""

import numpy as np
import pytest

from cxt.preprocess import (
    interpolate_tmrcas,
    interpolate_tmrca_per_window_spanavg,
    bitmask_to_intervals,
    missingness_by_window_scales,
    choose_pairs,
    process_X,
    process_y,
)
from cxt.simulate import simulate_parameterized_tree_sequence


@pytest.fixture
def small_ts():
    return simulate_parameterized_tree_sequence(seed=42, samples=5,
                                                 sequence_length=1e5)


class TestInterpolateTmrcas:
    def test_output_shape(self, small_ts):
        y = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=100000)
        assert y.shape == (50,)  # 100000 / 2000

    def test_positive_tmrcas(self, small_ts):
        y = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=100000)
        assert np.all(y > 0)

    def test_different_pairs(self, small_ts):
        y1 = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=100000,
                                sample_a=0, sample_b=1)
        y2 = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=100000,
                                sample_a=2, sample_b=3)
        assert not np.allclose(y1, y2)


class TestInterpolateTmrcaPerWindowSpanAvg:
    def test_constant_signal(self):
        lefts = np.array([0, 500, 1000], dtype=np.int64)
        rights = np.array([500, 1000, 1500], dtype=np.int64)
        values = np.array([100.0, 100.0, 100.0])
        result = interpolate_tmrca_per_window_spanavg(
            lefts, rights, values,
            interval_start=0, interval_end=1500, interval_size=500,
        )
        np.testing.assert_allclose(result, [100.0, 100.0, 100.0])

    def test_step_signal(self):
        lefts = np.array([0, 1000], dtype=np.int64)
        rights = np.array([1000, 2000], dtype=np.int64)
        values = np.array([10.0, 20.0])
        result = interpolate_tmrca_per_window_spanavg(
            lefts, rights, values,
            interval_start=0, interval_end=2000, interval_size=1000,
        )
        np.testing.assert_allclose(result, [10.0, 20.0])

    def test_partial_overlap(self):
        lefts = np.array([0, 500], dtype=np.int64)
        rights = np.array([500, 1000], dtype=np.int64)
        values = np.array([10.0, 30.0])
        result = interpolate_tmrca_per_window_spanavg(
            lefts, rights, values,
            interval_start=0, interval_end=1000, interval_size=1000,
        )
        np.testing.assert_allclose(result, [20.0])  # (500*10 + 500*30)/1000


class TestBitmaskToIntervals:
    def test_empty(self):
        bm = np.array([], dtype=bool)
        result = bitmask_to_intervals(bm)
        assert result.shape == (0, 2)

    def test_all_true(self):
        bm = np.ones(100, dtype=bool)
        result = bitmask_to_intervals(bm)
        assert result.shape == (1, 2)
        assert result[0, 0] == 0
        assert result[0, 1] == 100

    def test_all_false(self):
        bm = np.zeros(100, dtype=bool)
        result = bitmask_to_intervals(bm)
        assert result.shape[0] == 0

    def test_alternating(self):
        bm = np.array([True, True, False, False, True, True, True, False])
        result = bitmask_to_intervals(bm)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result[0], [0, 2])
        np.testing.assert_array_equal(result[1], [4, 7])


class TestMissingnessByWindowScales:
    def test_no_missing(self):
        mask = np.zeros(1000, dtype=int)
        result = missingness_by_window_scales(
            mask, base_window=100, step_size=100,
            multipliers=np.array([1, 2]), sequence_length=1000,
        )
        np.testing.assert_allclose(result, 0.0)

    def test_all_missing(self):
        mask = np.ones(1000, dtype=int)
        result = missingness_by_window_scales(
            mask, base_window=100, step_size=100,
            multipliers=np.array([1]), sequence_length=1000,
        )
        np.testing.assert_allclose(result, 1.0)

    def test_shape(self):
        mask = np.zeros(10000, dtype=int)
        mults = np.array([2, 8, 32, 64])
        result = missingness_by_window_scales(
            mask, base_window=200, step_size=200,
            multipliers=mults, sequence_length=10000,
        )
        n_steps = int(np.ceil(10000 / 200))
        assert result.shape == (len(mults), n_steps)


class TestChoosePairs:
    def test_correct_count(self, small_ts):
        pairs = choose_pairs(small_ts, num_pairs=5, seed=42)
        assert pairs.shape == (5, 2)

    def test_unique_pairs(self, small_ts):
        pairs = choose_pairs(small_ts, num_pairs=10, seed=42)
        pair_set = set(map(tuple, pairs))
        assert len(pair_set) == 10

    def test_deterministic(self, small_ts):
        p1 = choose_pairs(small_ts, num_pairs=5, seed=42)
        p2 = choose_pairs(small_ts, num_pairs=5, seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestProcessXY:
    def test_process_x_shape(self, small_ts):
        pairs = [(0, 1), (2, 3)]
        X = process_X(small_ts, pairs, window_size=2000,
                      sequence_length=100000)
        n_steps = int(np.ceil(100000 / 2000))
        assert X.shape[0] == 2  # num_pairs
        assert X.shape[1] == 2  # channels (xor/xnor)
        assert X.shape[2] == 4  # multipliers
        assert X.shape[3] == n_steps
        assert X.shape[4] == small_ts.num_samples

    def test_process_y_shape(self, small_ts):
        pairs = [(0, 1), (2, 3)]
        y = process_y(small_ts, pairs, window_size=2000,
                      sequence_length=100000)
        n_steps = int(100000 / 2000)
        assert y.shape == (2, n_steps)

    def test_process_y_log_transform(self, small_ts):
        pairs = [(0, 1)]
        y = process_y(small_ts, pairs, window_size=2000,
                      sequence_length=100000, transform=np.log)
        assert np.all(np.isfinite(y))
