"""Tests for cxt.correction — bias correction methods."""

import numpy as np
import pytest
import msprime

from cxt.correction import (
    _mutation_count_from_gm,
    diversity_bias_correction,
    stochastic_diversity_bias_correction,
    stochastic_diversity_bias_correction_v2,
)


@pytest.fixture
def ts_and_predictions():
    """Small tree sequence with fake predictions for testing."""
    ts = msprime.sim_ancestry(
        samples=5, sequence_length=1e5,
        recombination_rate=1e-8, population_size=1e4,
        random_seed=42,
    )
    ts = msprime.mutate(ts, rate=1e-8, random_seed=42)
    n_pairs = 3
    n_windows = 50
    n_reps = 5
    rng = np.random.default_rng(42)
    predictions = rng.normal(8, 1, (n_reps, n_pairs, n_windows))
    pivot_pairs = np.array([(0, 1), (2, 3), (4, 5)])
    return ts, predictions, pivot_pairs


class TestMutationCountFromGm:
    def test_basic(self):
        gm = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1],
        ])
        pairs = np.array([(0, 1), (2, 3)])
        counts = _mutation_count_from_gm(gm, pairs)
        assert counts.shape == (2,)
        assert np.all(counts >= 0)

    def test_identical_samples_zero_count(self):
        gm = np.array([
            [0, 1, 0],
            [0, 1, 0],
        ])
        pairs = np.array([(0, 1)])
        counts = _mutation_count_from_gm(gm, pairs)
        assert counts[0] == 0


class TestDiversityBiasCorrection:
    def test_output_shape(self, ts_and_predictions):
        ts, predictions, pivot_pairs = ts_and_predictions
        corrected = diversity_bias_correction(
            tree_sequence=ts,
            mutation_rate=1e-8,
            predictions=predictions,
            pivot_pairs=pivot_pairs,
        )
        assert corrected.shape == predictions.shape

    def test_with_intercept(self, ts_and_predictions):
        ts, predictions, pivot_pairs = ts_and_predictions
        corrected, intercept = diversity_bias_correction(
            tree_sequence=ts,
            mutation_rate=1e-8,
            predictions=predictions,
            pivot_pairs=pivot_pairs,
            return_intercept=True,
        )
        assert corrected.shape == predictions.shape
        # intercept is (1, n_pairs, 1), broadcastable against predictions
        assert intercept.ndim == 3


class TestStochasticDiversityBiasCorrection:
    def test_output_shape(self, ts_and_predictions):
        ts, predictions, pivot_pairs = ts_and_predictions
        corrected = stochastic_diversity_bias_correction(
            tree_sequence=ts,
            mutation_rate=1e-8,
            predictions=predictions,
            pivot_pairs=pivot_pairs,
            rng=np.random.default_rng(42),
        )
        assert corrected.shape == predictions.shape

    def test_deterministic_with_seed(self, ts_and_predictions):
        ts, predictions, pivot_pairs = ts_and_predictions
        c1 = stochastic_diversity_bias_correction(
            tree_sequence=ts, mutation_rate=1e-8,
            predictions=predictions, pivot_pairs=pivot_pairs,
            rng=np.random.default_rng(42),
        )
        c2 = stochastic_diversity_bias_correction(
            tree_sequence=ts, mutation_rate=1e-8,
            predictions=predictions, pivot_pairs=pivot_pairs,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_allclose(c1, c2)


class TestStochasticDiversityBiasCorrectionV2:
    def test_from_genotype_matrix(self, ts_and_predictions):
        ts, predictions, pivot_pairs = ts_and_predictions
        gm = ts.genotype_matrix().T
        corrected = stochastic_diversity_bias_correction_v2(
            genotype_matrix=gm,
            mutation_rate=1e-8,
            predictions=predictions,
            pivot_pairs=pivot_pairs,
            rng=np.random.default_rng(42),
            sequence_length=1e5,
            window_size=200,
        )
        assert corrected.shape == predictions.shape
