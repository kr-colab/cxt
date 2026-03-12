"""
Figure 4 (PNAS): OOD evaluation on stdpopsim v0.3 species.

Loads cached TMRCA data for cxt (and optionally Singer) from fig4/ and
singer/ caches, plots marginal coalescence KDEs. Matches original
fig4_stdpopsim_v3_ood.py logic.
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
ROW_HEIGHT = 1.0


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
    kmin = int(np.ceil(xmin / _LN10))
    kmax = int(np.floor(xmax / _LN10))
    ticks = [k * _LN10 for k in range(kmin, kmax + 1, step)]
    if ticks:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: r"$10^{%d}$" % round(np.log10(np.e) * x))
        )


def _metadata_name(m):
    return (
        f"{m['species_name']} {m['demography']} with map {m.get('genetic_map')}"
        .replace(" ", "_").replace("/", "_") + ".trees"
    )


def _load_singer_v3(cache_dir, metadata_all):
    """Return dict mapping species_name -> flat singer predictions, or None."""
    for fname in ["singer_v3_tmrcas.npz", "singer_stdpopsim_v3.npz"]:
        p = resolve_cache(f"main/cache/singer/{fname}")
        if os.path.exists(p):
            try:
                data = np.load(p, allow_pickle=True)
                result = {}
                for m in metadata_all:
                    key = m["species_name"]
                    if key in data:
                        result[key] = np.asarray(data[key])
                return result if result else None
            except Exception:
                continue

    singer_by_species = {}
    for m in metadata_all:
        species = m["species_name"]
        p = resolve_cache(f"main/cache/singer/true_tmrcas_{species}.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            if "singer_yhats" in d:
                singer_by_species[species] = np.asarray(d["singer_yhats"])
    return singer_by_species if singer_by_species else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    apply_pnas_style()

    cache_dir = resolve_cache_dir("main/cache/fig4")
    meta_path = resolve_cache("main/cache/fig4/stdpopsim_v3_metadata.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "rb") as f:
        metadata_all = pickle.load(f)

    singer_by_species = _load_singer_v3(cache_dir, metadata_all)

    valid = []
    for i, m in enumerate(metadata_all):
        fname = _metadata_name(m).replace(".trees", "_tmrca.npz")
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            d = np.load(path)
            valid.append((i, d["yhats"], d["ytrues"]))
        else:
            print(f"  skipping {m['species_name']} — cache not found: {path}")

    if not valid:
        raise RuntimeError("No cached tmrca data found in " + cache_dir)

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
        ax.plot(x, p_pred, color="dodgerblue", label="cxt")

        species_name = metadata_all[orig_idx]["species_name"]
        if singer_by_species and species_name in singer_by_species:
            singer_flat = np.asarray(singer_by_species[species_name]).flatten()
            singer_mask = np.isfinite(singer_flat)
            singer_flat = singer_flat[singer_mask]
            if singer_flat.size > 10:
                p_singer = kde_pdf(singer_flat, x)
                ax.plot(x, p_singer, color="darkblue", lw=0.6, ls="--",
                        label="Singer")

        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 16.2)
        ax.grid(alpha=0.3)
        set_loge_power10_ticks(ax, 0, 16.2, step=2)

        m = metadata_all[orig_idx]
        ax.set_title(
            f"{m['species_name']}\n{m.get('id', '')}",
            loc="left", fontsize=7,
        )
        if k == 0:
            ax.text(
                0.72, 0.97,
                f"MSE={mse_val:.3g}\nlog\u2081\u2080KL={kl_val:.2f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=5,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            )
        else:
            ax.text(
                0.05, 0.97,
                f"MSE={mse_val:.3g}\nlog\u2081\u2080KL={kl_val:.2f}",
                ha="left", va="top", transform=ax.transAxes, fontsize=5,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
            )
        if k == 0:
            ax.legend(loc="lower left", fontsize=5)
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
    savefig(fig, "figure4", output_dir=args.output_dir)
    print("  saved figure4")


if __name__ == "__main__":
    main()
