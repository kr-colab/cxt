"""Site-frequency spectrum computation and source building.

Single canonical implementation consolidating duplicates from
``utils.py`` and ``api2.py``.
"""

from __future__ import annotations

import numpy as np

W_MULTIPLIERS = (2, 8, 32, 64)


def calculate_window_sfs(
    positions: np.ndarray,
    pivot_frequencies: np.ndarray,
    window_size: int = 2000,
    sequence_length: float = 1e6,
    num_samples: int = 50,
    step_size: int = 2000,
    availability_mask: np.ndarray | None = None,
    use_interpolation: bool = False,
) -> np.ndarray:
    """Bin site frequencies into genomic windows.

    Parameters
    ----------
    positions : array of site positions (bp).
    pivot_frequencies : integer allele frequencies at each site.
    window_size : width of each SFS window (bp).
    step_size : stride between window starts (bp).
    sequence_length : total length of the region (bp).
    num_samples : number of haploid samples.
    availability_mask : per-step availability fraction [0, 1].
        If provided, counts are scaled by ``window_size / available_bp``.
    use_interpolation : if True and availability_mask makes a window fully
        inaccessible, linearly interpolate from neighboring windows.

    Returns
    -------
    sfs : ndarray of shape ``(n_windows, num_samples)``
    """
    n_windows = int(np.ceil(sequence_length / step_size))
    window_starts = np.arange(n_windows) * step_size
    window_ends = np.minimum(window_starts + window_size, sequence_length)
    site_in_window = (
        (positions[:, np.newaxis] >= window_starts)
        & (positions[:, np.newaxis] < window_ends)
    )

    if availability_mask is None:
        sfs = np.zeros((n_windows, num_samples), dtype=int)
        for i in range(n_windows):
            wf = pivot_frequencies[site_in_window[:, i]]
            if wf.size:
                sfs[i] = np.bincount(wf, minlength=num_samples)
        return sfs

    # Scaled path when some positions are inaccessible
    availability_mask = np.asarray(availability_mask, dtype=float)
    step_starts = np.arange(n_windows, dtype=np.int64) * step_size
    step_ends = np.minimum(step_starts + step_size, int(sequence_length)).astype(np.int64)

    sfs = np.zeros((n_windows, num_samples), dtype=float)
    for i in range(n_windows):
        ws, we = int(window_starts[i]), int(window_ends[i])
        if we <= ws:
            continue
        overlaps = np.clip(
            np.minimum(we, step_ends) - np.maximum(ws, step_starts), 0, None
        ).astype(float)
        available_bp = float(np.dot(overlaps, availability_mask))

        wf = pivot_frequencies[site_in_window[:, i]]
        counts = (
            np.bincount(wf, minlength=num_samples).astype(float)
            if wf.size
            else np.zeros(num_samples, dtype=float)
        )
        if available_bp > 0:
            sfs[i] = counts * ((we - ws) / available_bp)
        elif use_interpolation:
            sfs[i] = np.nan
        # else: zeros (default)

    if use_interpolation and np.isnan(sfs).any():
        for k in range(num_samples):
            y = sfs[:, k]
            mask = np.isnan(y)
            if mask.any() and (~mask).any():
                y[mask] = np.interp(
                    np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask]
                )
            elif mask.all():
                y[:] = 0.0

    return sfs


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def basic_filtering(
    gm: np.ndarray, positions: np.ndarray, num_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-biallelic and fixed sites from a genotype matrix.

    Parameters
    ----------
    gm : ndarray of shape ``(n_samples, n_sites)``
        Haploid genotype matrix (0/1 entries expected for biallelic sites).
    positions : ndarray of shape ``(n_sites,)``
        Genomic positions corresponding to each column of *gm*.
    num_samples : int or None
        Total haploid sample count.  If None, inferred from ``gm.shape[0]``.

    Returns
    -------
    gm_filtered : ndarray
        Filtered genotype matrix.
    positions_filtered : ndarray
        Matching filtered positions.
    """
    if num_samples is None:
        num_samples = gm.shape[0]
    non_bial = np.any(gm >= 2, axis=0)
    freq = gm.sum(0)
    fixed = (freq == 0) | (freq >= num_samples)
    keep = ~(non_bial | fixed)
    return gm[:, keep], positions[keep]


# ---------------------------------------------------------------------------
# Source building (one pivot pair)
# ---------------------------------------------------------------------------

def build_src(
    block_positions: np.ndarray,
    block_gm: np.ndarray,
    pivot_id_A: int,
    pivot_id_B: int,
    sequence_length: float = 1e6,
    step_size: int = 2000,
    availability_mask: np.ndarray | None = None,
    use_interpolation: bool = False,
    missingness_bitmask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the log1p-transformed SFS source tensor for one pivot pair.

    Returns
    -------
    X : ndarray of shape ``(2, len(W_MULTIPLIERS), n_windows, num_samples)``
        in float16, already log1p-transformed.
    """
    num_samples, _num_sites = block_gm.shape

    xor_mask = (block_gm[pivot_id_A] ^ block_gm[pivot_id_B]).astype(bool)
    xnor_mask = ~xor_mask
    freqs = block_gm.sum(0).astype(np.int32)

    pos_xor, freq_xor = block_positions[xor_mask], freqs[xor_mask]
    pos_xnor, freq_xnor = block_positions[xnor_mask], freqs[xnor_mask]

    def _sfs(pos, f, w_mult):
        if pos.size == 0:
            return np.zeros(
                (int(np.ceil(sequence_length / step_size)), num_samples),
                dtype=np.int32,
            )
        return calculate_window_sfs(
            positions=pos.astype(np.float32),
            pivot_frequencies=f.astype(np.int32),
            window_size=step_size * w_mult,
            sequence_length=sequence_length,
            num_samples=num_samples,
            step_size=step_size,
            availability_mask=availability_mask,
            use_interpolation=use_interpolation,
        )

    n_w = int(np.ceil(sequence_length / step_size))
    Xs_xor = np.zeros((len(W_MULTIPLIERS), n_w, num_samples), dtype=np.int32)
    Xs_xnor = np.zeros_like(Xs_xor)

    for i, w in enumerate(W_MULTIPLIERS):
        Xs_xor[i] = _sfs(pos_xor, freq_xor, w)
        Xs_xnor[i] = _sfs(pos_xnor, freq_xnor, w)

    X = np.stack([Xs_xor, Xs_xnor], axis=0).astype(np.float16)

    if missingness_bitmask is not None:
        from cxt.preprocess import missingness_by_window_scales

        missing_by_mult = missingness_by_window_scales(
            missing_mask=missingness_bitmask.astype(int),
            base_window=step_size,
            step_size=step_size,
            multipliers=W_MULTIPLIERS,
            sequence_length=sequence_length,
        )
        X[0, :, :, 0] = np.exp(missing_by_mult)
        X[1, :, :, 0] = np.exp(1 - missing_by_mult)

    return np.log1p(X)
