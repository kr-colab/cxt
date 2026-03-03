"""Tests for cxt.simulate — simulation and feature extraction."""

import numpy as np
import pytest
import msprime

from cxt.simulate import (
    simulate_parameterized_tree_sequence,
    create_sawtooth_demography,
    sample_population_size,
    sample_demography,
    ts2X,
    interpolate_tmrcas,
)


class TestSimulateParameterizedTreeSequence:
    def test_basic_constant(self):
        ts = simulate_parameterized_tree_sequence(seed=42)
        assert ts.num_samples == 50  # 25 diploid
        assert ts.num_sites > 0
        assert ts.sequence_length == 1e6

    def test_custom_sample_size(self):
        ts = simulate_parameterized_tree_sequence(seed=42, samples=10)
        assert ts.num_samples == 20  # diploid

    def test_with_sawtooth_demography(self):
        dem = create_sawtooth_demography(Ne=2e4, magnitude=3)
        ts = simulate_parameterized_tree_sequence(seed=42, demography=dem)
        assert ts.num_samples == 50
        assert ts.num_sites > 0

    def test_reproducible(self):
        ts1 = simulate_parameterized_tree_sequence(seed=123)
        ts2 = simulate_parameterized_tree_sequence(seed=123)
        assert ts1.num_sites == ts2.num_sites
        np.testing.assert_array_equal(
            ts1.genotype_matrix(), ts2.genotype_matrix()
        )


class TestCreateSawtoothDemography:
    def test_returns_demography(self):
        dem = create_sawtooth_demography()
        assert isinstance(dem, msprime.Demography)

    def test_custom_ne(self):
        dem = create_sawtooth_demography(Ne=1e5)
        assert isinstance(dem, msprime.Demography)

    def test_simulatable(self):
        dem = create_sawtooth_demography(Ne=2e4, magnitude=4)
        ts = msprime.sim_ancestry(
            samples=5, demography=dem,
            sequence_length=1e5, recombination_rate=1e-8,
            random_seed=42,
        )
        assert ts.num_trees > 0


class TestSamplePopulationSize:
    def test_returns_list(self):
        sizes = sample_population_size(seed=42)
        assert isinstance(sizes, (list, np.ndarray))
        assert len(sizes) == 20

    def test_within_range(self):
        sizes = sample_population_size(n_min=100, n_max=10000, seed=42)
        for s in sizes:
            assert 100 <= s <= 10000


class TestSampleDemography:
    def test_returns_demography(self):
        dem = sample_demography(seed=42)
        assert isinstance(dem, msprime.Demography)


class TestTs2X:
    @pytest.fixture
    def small_ts(self):
        return simulate_parameterized_tree_sequence(seed=42, samples=5,
                                                     sequence_length=1e6)

    def test_output_shape(self, small_ts):
        X = ts2X(small_ts, window_size=4000, step_size=2000, pivot_A=0, pivot_B=1)
        assert X.ndim == 3
        # ts2X hardcodes 1e6 for n_windows; n_samples from genotype_matrix
        n_steps = int(np.ceil(1e6 / 2000))
        assert X.shape[0] == 4  # w_multipliers
        assert X.shape[1] == n_steps
        assert X.shape[2] == small_ts.num_samples

    def test_nonnegative(self, small_ts):
        X = ts2X(small_ts, window_size=4000, step_size=2000)
        assert np.all(X >= 0)


class TestInterpolateTmrcas:
    @pytest.fixture
    def small_ts(self):
        return simulate_parameterized_tree_sequence(seed=42, samples=5,
                                                     sequence_length=1e5)

    def test_output_shape(self, small_ts):
        tmrcas = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=1e5)
        n_windows = int(1e5 / 2000)
        assert len(tmrcas) == n_windows

    def test_positive_values(self, small_ts):
        tmrcas = interpolate_tmrcas(small_ts, window_size=2000, sequence_length=1e5)
        assert np.all(np.array(tmrcas) > 0)
