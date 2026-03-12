"""
Figure 3 (PNAS): Marginal coalescence distributions for stdpopsim v0.2 species.

Loads cached TMRCA data and metadata from fig3/, then plots true (black) vs
predicted (dodgerblue) KDE for each species/demography. Produces two figures:
  - figure3       (species without genetic maps)
  - figure3_map   (species with genetic maps)

Matches the plotting logic of the original fig3_stdpopsim_v2_coalescence.py.
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from pnas_defaults import (
    apply_pnas_style, savefig,
    resolve_cache, resolve_cache_dir,
    DOUBLE_COL, TIMES,
)

_LN10 = np.log(10.0)
COLS = 4
ROW_HEIGHT = 1.1


def discretize(sequence):
    indices = np.searchsorted(TIMES, sequence, side="right") - 1
    return np.clip(indices, 0, len(TIMES) - 1)


def kde_pdf(samples, grid, bw_method=None):
    if np.all(samples == samples[0]):
        loc = float(samples[0])
        spread = 1e-6 + 0.01 * (np.ptp(grid) or 1.0)
        pdf = np.exp(-0.5 * ((grid - loc) / spread) ** 2)
        pdf /= np.trapezoid(pdf, grid)
        return pdf
    kde = gaussian_kde(samples, bw_method=bw_method)
    pdf = kde(grid)
    area = np.trapezoid(pdf, grid)
    return pdf / area if area > 0 else pdf


def kl_divergence(p_grid, q_grid, x_grid, eps=1e-12):
    p, q = p_grid + eps, q_grid + eps
    p /= np.trapezoid(p, x_grid)
    q /= np.trapezoid(q, x_grid)
    return float(np.trapezoid(p * (np.log(p) - np.log(q)), x_grid))


def robust_grid(a, b, n=512, q_lo=0.005, q_hi=0.995):
    lo = np.nanmin([np.quantile(a, q_lo), np.quantile(b, q_lo)])
    hi = np.nanmax([np.quantile(a, q_hi), np.quantile(b, q_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = min(np.nanmin(a), np.nanmin(b))
        hi = max(np.nanmax(a), np.nanmax(b))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, n)


def set_loge_power10_ticks(ax, xmin, xmax, step=2):
    """Place ticks at 10^k in ln-space, every `step` powers to avoid clutter."""
    kmin = int(np.ceil(xmin / _LN10))
    kmax = int(np.floor(xmax / _LN10))
    ticks = [k * _LN10 for k in range(kmin, kmax + 1, step)]
    if ticks:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: r"$10^{%d}$" % round(np.log10(np.e) * x))
        )


_ID_ABBREVS = {
    "OutOfAfricaExtendedNeandertalAdmixturePulse": "OOAExtNdrtlAdmPulse",
    "OutOfAfricaArchaicAdmixture": "OOAArchaicAdmixt",
    "AmericanAdmixture": "AmerAdmixture",
    "PiecewiseConstant": "PiecewiseConst",
    "HolsteinFriesian": "HolsteinFries",
    "EarlyWolfAdmixture": "EarlyWolfAdmixt",
    "BottleneckMigration": "BtlnkMigration",
}


def _shorten_id(model_id, max_len=35):
    s = model_id
    for long, short in _ID_ABBREVS.items():
        s = s.replace(long, short)
    if len(s) > max_len:
        s = s[:max_len - 1] + "\u2026"
    return s


def _metadata_name(m):
    return (
        f"{m['species_name']} {m['demography']} with map {m.get('genetic_map')}"
        .replace(" ", "_").replace("/", "_") + ".trees"
    )


def plot_kde_grid(species_list, metadata_all, cache_dir, save_name, output_dir):
    valid = []
    for idx in species_list:
        m = metadata_all[idx]
        fname = _metadata_name(m).replace(".trees", "_tmrca.npz")
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            d = np.load(path)
            valid.append((idx, d["yhats"], d["ytrues"]))
        else:
            print(f"  skipping {m['species_name']} — cache not found: {path}")

    if not valid:
        print(f"  no data for {save_name}, skipping")
        return

    n = len(valid)
    cols = min(n, COLS)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(DOUBLE_COL, ROW_HEIGHT * rows),
        squeeze=False,
    )

    for k, (orig_idx, yhat_raw, ytrue_raw) in enumerate(valid):
        yhat = np.asarray(yhat_raw.mean(0)).flatten()
        ytrue = np.asarray(ytrue_raw).flatten()
        mask = np.isfinite(yhat) & np.isfinite(ytrue)
        yhat, ytrue = yhat[mask], ytrue[mask]
        ytrue = TIMES[discretize(ytrue)]

        mse_val = float(np.mean((yhat - ytrue) ** 2))
        x = robust_grid(yhat, ytrue, n=512)
        p_true = kde_pdf(ytrue, x)
        p_pred = kde_pdf(yhat, x)
        kl_val = np.log10(kl_divergence(p_true, p_pred, x))

        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.plot(x, p_true, color="black", label="True KDE")
        ax.plot(x, p_pred, color="dodgerblue", label="Pred KDE")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 16.2)
        ax.grid(alpha=0.3)
        set_loge_power10_ticks(ax, 0, 16.2, step=2)

        m = metadata_all[orig_idx]
        short_id = _shorten_id(m.get("id", ""))
        ax.set_title(
            f"{m['species_name']}\n{short_id}",
            loc="left", fontsize=9,
        )
        if k == 0:
            ax.text(
                0.72, 0.97,
                f"MSE={mse_val:.3g}\nlog\u2081\u2080KL={kl_val:.2f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            )
        else:
            ax.text(
                0.05, 0.97,
                f"MSE={mse_val:.3g}\nlog\u2081\u2080KL={kl_val:.2f}",
                ha="left", va="top", transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            )
        if k == 0:
            ax.legend(loc="lower left", fontsize=7)
        if c == 0:
            ax.set_ylabel("Density")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if r == rows - 1:
            ax.set_xlabel("TMRCA (gen.)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, save_name, output_dir=output_dir)
    print(f"  saved {save_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    apply_pnas_style()

    cache_dir = resolve_cache_dir("main/cache/fig3")
    meta_path = resolve_cache("main/cache/fig3/stdpopsim_metadata.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "rb") as f:
        metadata_all = pickle.load(f)

    no_map_idx, with_map_idx = [], []
    for i, m in enumerate(metadata_all):
        if m.get("genetic_map") is None:
            no_map_idx.append(i)
        else:
            with_map_idx.append(i)

    output_dir = args.output_dir
    plot_kde_grid(no_map_idx, metadata_all, cache_dir, "figure3", output_dir)
    plot_kde_grid(with_map_idx, metadata_all, cache_dir, "figure3_map", output_dir)


if __name__ == "__main__":
    main()
