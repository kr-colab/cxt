"""Diversity-based bias correction for cxt TMRCA predictions.

Provides both tree-sequence-based and genotype-matrix-based
stochastic correction, plus a deterministic variant.
"""

from __future__ import annotations

import numpy as np
import tskit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mutation_count_from_gm(
    gm: np.ndarray,
    pivot_pairs: np.ndarray,
) -> np.ndarray:
    """Count XOR-different sites per pivot pair from a genotype matrix."""
    counts = np.zeros(len(pivot_pairs))
    for i, (a, b) in enumerate(pivot_pairs):
        pair_gm = gm[[a, b]]
        mask = pair_gm.sum(0) >= 1
        counts[i] = ((pair_gm[0, mask] ^ pair_gm[1, mask]) >= 1).sum()
    return counts


# ---------------------------------------------------------------------------
# Deterministic correction (ratio of expected / observed diversity)
# ---------------------------------------------------------------------------

def diversity_bias_correction(
    tree_sequence: tskit.TreeSequence,
    mutation_rate: float,
    predictions: np.ndarray,
    pivot_pairs: np.ndarray,
    zero_offset: float = 0.0,
    return_intercept: bool = False,
):
    """Additive log-space correction so E[diversity] matches observed.

    Parameters
    ----------
    predictions : array (replicates, pairs, windows) in log-TMRCA.
    """
    assert predictions.ndim == 3
    obs_div = tree_sequence.trim().diversity(sample_sets=pivot_pairs)
    fit_div = 2 * np.exp(predictions.mean(axis=0)).mean(axis=-1) * mutation_rate
    obs_div[obs_div == 0] = (
        obs_div[obs_div > 0].min()
        if zero_offset == 0
        else zero_offset / tree_sequence.trim().sequence_length
    )
    corrected = predictions + (np.log(obs_div) - np.log(fit_div))[None, :, None]
    if not return_intercept:
        return corrected
    intercept = np.log(obs_div / mutation_rate / 2)[None, :, None]
    return corrected, intercept


def diversity_bias_correction_by_rep(
    tree_sequence: tskit.TreeSequence,
    mutation_rate: float,
    predictions: np.ndarray,
    pivot_pairs: np.ndarray,
    zero_offset: float = 0.0,
    return_intercept: bool = False,
):
    """Per-replicate version of ``diversity_bias_correction``."""
    assert predictions.ndim == 3
    obs_div = tree_sequence.trim().diversity(sample_sets=pivot_pairs)
    obs_div[obs_div == 0] = (
        obs_div[obs_div > 0].min()
        if zero_offset == 0
        else zero_offset / tree_sequence.trim().sequence_length
    )
    corrected = []
    for rep in predictions:
        fit_div = 2 * np.exp(rep).mean(axis=-1) * mutation_rate
        corrected.append(rep + (np.log(obs_div) - np.log(fit_div))[:, None])
    corrected = np.stack(corrected)
    if not return_intercept:
        return corrected
    intercept = np.log(obs_div / mutation_rate / 2)[None, :, None]
    return corrected, intercept


# ---------------------------------------------------------------------------
# Stochastic correction (Gamma posterior on scaling factor)
# ---------------------------------------------------------------------------

def stochastic_diversity_bias_correction(
    tree_sequence: tskit.TreeSequence,
    mutation_rate: float,
    predictions: np.ndarray,
    pivot_pairs: np.ndarray,
    return_intercept: bool = False,
    rng: np.random.Generator | None = None,
):
    r"""Stochastic correction via Gamma posterior on per-pair scaling.

    Under the model, mutation_count ~ Poisson(2 * c * mu * sum_i TMRCA_i * w_i),
    with posterior c ~ Gamma(mutation_count + 1, rate).
    """
    assert predictions.ndim == 3
    if rng is None:
        rng = np.random.default_rng()
    mutation_count = tree_sequence.trim().diversity(
        sample_sets=pivot_pairs, span_normalise=False
    )
    seq_len = tree_sequence.trim().sequence_length
    corrected, intercept = [], []
    for log_tmrca in predictions:
        rate = 2 * np.exp(log_tmrca).mean(axis=-1) * mutation_rate * seq_len
        c = rng.gamma(shape=mutation_count + 1, scale=1 / rate)
        corrected.append(log_tmrca + np.log(c)[:, None])
        intercept.append(np.log(np.exp(log_tmrca).mean(axis=-1) * c)[:, None])
    corrected = np.stack(corrected)
    if not return_intercept:
        return corrected
    return corrected, np.stack(intercept)


def stochastic_diversity_bias_correction_v2(
    genotype_matrix: np.ndarray,
    mutation_rate: float,
    predictions: np.ndarray,
    pivot_pairs: np.ndarray,
    return_intercept: bool = False,
    rng: np.random.Generator | None = None,
    sequence_length: float = 1e6,
    window_size: int = 200,
    availability_mask: np.ndarray | None = None,
    mask_missingness=None,
):
    """Genotype-matrix variant (no tree sequence needed)."""
    assert predictions.ndim == 3
    if rng is None:
        rng = np.random.default_rng()

    mutation_count = _mutation_count_from_gm(genotype_matrix, pivot_pairs)

    if mask_missingness is not None:
        avail = 1 - np.asarray(mask_missingness)
        available_bp = avail * window_size
    else:
        available_bp = np.ones(predictions.shape[-1]) * window_size

    corrected, intercept = [], []
    for log_tmrca in predictions:
        rate = 2 * mutation_rate * (np.exp(log_tmrca) @ available_bp)
        c = rng.gamma(shape=mutation_count + 1, scale=1 / rate)
        corrected.append(log_tmrca + np.log(c)[:, None])
        intercept.append(np.log(np.exp(log_tmrca).mean(axis=-1) * c)[:, None])
    corrected = np.stack(corrected)
    if not return_intercept:
        return corrected
    return corrected, np.stack(intercept)
