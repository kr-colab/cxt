"""Tests for cxt.dataset — discretization and grid constants."""

import numpy as np
import pytest

from cxt.dataset import (
    TIMES,
    GRID_SIZE,
    discretize,
)


class TestDatasetConstants:
    def test_grid_size(self):
        assert GRID_SIZE == 324

    def test_times_range(self):
        assert TIMES[0] == 3.0
        assert TIMES[-1] == 17.0
        assert len(TIMES) == GRID_SIZE



class TestDiscretize:
    def test_returns_indices(self):
        seq = np.array([5.0, 10.0, 15.0])
        idx = discretize(seq, TIMES)
        assert idx.dtype in (np.intp, np.int64, np.int32)
        assert np.all(idx >= 0)
        assert np.all(idx < len(TIMES))

    def test_monotonic_input_monotonic_output(self):
        seq = np.linspace(3, 17, 50)
        idx = discretize(seq, TIMES)
        assert np.all(np.diff(idx) >= 0)

    def test_clamps_to_grid(self):
        seq = np.array([0.0, 100.0])
        idx = discretize(seq, TIMES)
        assert idx[0] == 0
        assert idx[1] == len(TIMES) - 1


