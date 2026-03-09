"""
Figure S9: Mosquito Rdl region comparison (cxt vs Singer+Polegon vs SMC++).

Loads pre-computed TMRCA data for each method and population from
existing caches, then assembles a multi-panel comparison figure.

Faithful reproduction of:
  - revision/figure_mosquito/experiment_integrated_missing.ipynb (cxt panels)
  - revision/figure_mosquito/experiment_singer.ipynb (Singer panels)
  - revision/figure_mosquito/experiment_smc++.ipynb (SMC++ panels)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from figures.paths import AG1000G_ACCESSIBILITY


POPULATIONS = ["BurkinaFaso", "Mali", "Cameroon", "Ghana", "Uganda"]
POP_DISPLAY = {
    "BurkinaFaso": "Burkina Faso",
    "Mali": "Mali",
    "Cameroon": "Cameroon",
    "Ghana": "Ghana",
    "Uganda": "Uganda",
}
METHODS = ["cxt", "Singer+Polegon", "SMC++"]

REGION_START = 25_100_000
REGION_END = 25_600_000
STEP_BP = 200

RDL_START = 25_363_652
RDL_END = 25_434_556

FIG7_CACHE = "figures/output/main/cache/fig7"
REVISION_CACHE = "revision/figure_mosquito/cache"

POP_MAP_CXT = {
    "BurkinaFaso": "burkinafaso",
    "Mali": "mali",
    "Cameroon": "cameroon",
    "Ghana": "ghana",
    "Uganda": "uganda",
}
POP_MAP_SINGER = {
    "BurkinaFaso": ("tmrca_singer_bf.npz", "tmrca_bf"),
    "Mali": ("tmrca_singer_mali.npz", "tmrca_mali"),
    "Cameroon": ("tmrca_singer_cameroon.npz", "tmrca_cameroon"),
    "Ghana": ("tmrca_singer_ghana.npz", "tmrca_ghana"),
    "Uganda": ("tmrca_singer_uganda.npz", "tmrca_uganda"),
}
POP_MAP_SMCPP = {
    "BurkinaFaso": "smcpp_burkina_faso.npz",
    "Mali": "smcpp_mali.npz",
    "Cameroon": "smcpp_cameroon.npz",
    "Ghana": "smcpp_ghana.npz",
    "Uganda": "smcpp_uganda.npz",
}

METHOD_STYLES = {
    "cxt": {
        "rep_color": "dodgerblue",
        "mean_color": "dodgerblue",
        "band_color": "deepskyblue",
        "edge_color": "dodgerblue",
    },
    "Singer+Polegon": {
        "rep_color": "#4B4E91",
        "mean_color": "darkblue",
        "band_color": "#B8B9D1",
        "edge_color": "#4B4E91",
    },
    "SMC++": {
        "rep_color": "lightseagreen",
        "mean_color": "teal",
        "band_color": "paleturquoise",
        "edge_color": "lightseagreen",
    },
}

fmt_mb = FuncFormatter(lambda v, _: f"{v/1e6:.1f}")


# ── Data loaders ──────────────────────────────────────────────

def _region_slice(data_2d, step_bp):
    """Extract the RDL region from a (n_reps, n_bins) array. Return (x, slice)."""
    bin_start = REGION_START // step_bp
    bin_end = REGION_END // step_bp
    n_bins = data_2d.shape[1]
    if bin_end > n_bins:
        bin_end = n_bins
    x = np.arange(bin_start, bin_end) * step_bp
    return x, data_2d[:, bin_start:bin_end]


def load_cxt(pop):
    """Returns (x_bp, tmrca_reps) where tmrca_reps is (n_reps, n_bins) in LINEAR space."""
    tag = POP_MAP_CXT[pop]
    path = os.path.join(FIG7_CACHE, f"genome_{tag}.npz")
    if not os.path.exists(path):
        return None, None
    genome = np.load(path)["genome"]  # already LINEAR (generations)
    return _region_slice(genome, STEP_BP)


def load_singer(pop):
    """Returns (x_bp, tmrca_reps) where tmrca_reps is (n_reps, n_bins) in LINEAR space."""
    fname, key = POP_MAP_SINGER[pop]
    path = os.path.join(REVISION_CACHE, fname)
    if not os.path.exists(path):
        return None, None
    data = np.load(path)[key]  # shape (n_bins_total, n_reps), LOG space
    genome = np.exp(data.T)    # -> (n_reps, n_bins_total), LINEAR
    genome = np.where(np.isfinite(genome) & (genome > 0), genome, np.nan)
    return _region_slice(genome, STEP_BP)


def load_smcpp(pop):
    """Returns (x_bp, tmrca_reps) where tmrca_reps is (n_reps, n_bins) in LINEAR space."""
    fname = POP_MAP_SMCPP[pop]
    path = os.path.join(REVISION_CACHE, fname)
    if not os.path.exists(path):
        return None, None
    yhats = np.load(path, allow_pickle=True)["yhats"]  # (n_reps, n_bins_total), LOG
    genome = np.exp(yhats)  # -> LINEAR
    genome = np.where(np.isfinite(genome) & (genome > 0), genome, np.nan)
    return _region_slice(genome, STEP_BP)


LOADERS = {
    "cxt": load_cxt,
    "Singer+Polegon": load_singer,
    "SMC++": load_smcpp,
}


# ── Missingness helpers ───────────────────────────────────────

def moving_average(frac, smooth_bp=5_000, step_bp=200):
    k = max(1, int(round(smooth_bp / step_bp)))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(frac.astype(np.float32), kernel, mode="same")


def load_missingness(mask_path, step_bp):
    """Load accessibility mask and compute per-bin missing fraction for the RDL region."""
    if not os.path.exists(mask_path):
        return None
    bitmask = np.load(mask_path)["access_2L"]
    unaccessible = ~bitmask

    region = np.asarray(unaccessible[REGION_START:REGION_END], dtype=np.bool_)
    n_bins = (REGION_END - REGION_START) // step_bp
    pad = n_bins * step_bp - region.size
    if pad > 0:
        region = np.pad(region, (0, pad), constant_values=False)
    miss_frac = region[:n_bins * step_bp].reshape(n_bins, step_bp).mean(axis=1).astype(np.float32)
    return miss_frac


def draw_missing_track(ax, x, missing_frac, smooth_bp=5_000, step_bp=200,
                       height=0.16, pad_frac=0.26, xlabel=None):
    """Draw compact missingness track below main axis."""
    y_raw = np.clip(missing_frac, 0, 1)
    y_smooth = np.clip(moving_average(y_raw, smooth_bp, step_bp), 0, 1)

    tr = ax.inset_axes([0.0, -pad_frac, 1.0, height], transform=ax.transAxes)
    tr.fill_between(x, 0, y_smooth, alpha=0.35, color="lightsteelblue", zorder=1)
    tr.plot(x, y_smooth, lw=1.1, color="steelblue", alpha=0.95, zorder=2)
    tr.plot(x, y_raw, lw=0.5, color="steelblue", alpha=0.30, zorder=2)

    tr.set_xlim(x.min(), x.max())
    tr.set_ylim(0, 1)
    tr.xaxis.set_major_formatter(fmt_mb)
    for s in ("top", "right", "left"):
        tr.spines[s].set_visible(False)
    tr.spines["bottom"].set_linewidth(0.5)
    tr.spines["bottom"].set_alpha(0.5)
    tr.set_yticks([])
    tr.set_facecolor("none")
    tr.grid(False)

    if xlabel is None:
        tr.tick_params(axis="x", which="both", labelbottom=False)
    else:
        tr.set_xlabel(xlabel, labelpad=10, fontsize=9)

    for line in tr.lines:
        line.set_clip_on(False)
    for coll in tr.collections:
        coll.set_clip_on(False)


# ── Panel plotting ────────────────────────────────────────────

def plot_panel(ax, x, tmrca_reps, method, show_rdl=True,
               missing_frac=None, xlabel=None, show_ylabel=False):
    """
    Draw a single panel with replicate curves, mean ± SD band, and RDL annotation.

    tmrca_reps: (n_reps, n_bins) in LINEAR space (generations).
    x:          (n_bins,) genomic bp coordinates.
    """
    style = METHOD_STYLES[method]

    mean_t = np.nanmean(tmrca_reps, axis=0)
    std_t = np.nanstd(tmrca_reps, axis=0)
    n = tmrca_reps.shape[0]

    for i in range(n):
        alpha = 0.18 + 0.30 * (i / max(1, n - 1))
        lw = 0.55 + 0.35 * (i % 5 == 0)
        ax.plot(x, tmrca_reps[i], lw=lw, color=style["rep_color"],
                alpha=alpha, zorder=0)

    eps = 1e-12
    lower = np.clip(mean_t - std_t, eps, None)
    upper = np.clip(mean_t + std_t, eps, None)

    ax.fill_between(x, lower, upper, color=style["band_color"],
                    alpha=0.30, zorder=1)
    ax.plot(x, lower, ls="--", lw=0.8, color=style["edge_color"],
            alpha=0.85, zorder=2)
    ax.plot(x, upper, ls="--", lw=0.8, color=style["edge_color"],
            alpha=0.85, zorder=2)
    ax.plot(x, mean_t, lw=1.2, color=style["mean_color"], zorder=3)

    if show_rdl:
        ax.axvspan(RDL_START, RDL_END, color="crimson", alpha=0.10, zorder=0)
        mid = (RDL_START + RDL_END) / 2
        valid = mean_t[np.isfinite(mean_t) & (mean_t > 0)]
        if valid.size > 0:
            y_arrow = np.exp((np.log(valid.min()) + np.log(valid.max())) / 3)
        else:
            y_arrow = 1e3
        ax.annotate(
            "", xy=(RDL_END, y_arrow), xytext=(RDL_START, y_arrow),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"), zorder=6,
        )
        ax.text(mid, y_arrow * 1.05, "RDL", fontsize=8, ha="center",
                va="bottom", zorder=6)

    ax.set_yscale("log")
    ax.set_xlim(REGION_START, REGION_END)
    ax.set_ylim(0.5e2, 5e6)
    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.xaxis.set_major_formatter(fmt_mb)
    ax.tick_params(axis="x", which="both", labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("TMRCA (gen.)", fontsize=9)

    if missing_frac is not None:
        draw_missing_track(ax, x, missing_frac, xlabel=xlabel)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    missing_frac = load_missingness(AG1000G_ACCESSIBILITY, STEP_BP)

    n_rows = len(POPULATIONS)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 2.5 * n_rows),
        sharex=True, sharey=True,
    )

    for i, pop in enumerate(POPULATIONS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            loader = LOADERS[method]
            x, tmrca_reps = loader(pop)

            if x is not None:
                is_bottom = (i == n_rows - 1)
                xlabel = "Position on chr2L (Mb)" if is_bottom else None
                miss = missing_frac if is_bottom else None

                plot_panel(
                    ax, x, tmrca_reps, method,
                    missing_frac=miss,
                    xlabel=xlabel,
                    show_ylabel=(j == 0),
                )
            else:
                ax.text(0.5, 0.5, "No data", ha="center",
                        va="center", transform=ax.transAxes)

            if i == 0:
                ax.set_title(method, fontsize=12)
            if j == 0:
                ax.annotate(
                    POP_DISPLAY[pop], xy=(0, 0.5),
                    xytext=(-0.35, 0.5), textcoords="axes fraction",
                    xycoords="axes fraction",
                    fontsize=10, fontweight="bold", va="center", ha="right",
                    rotation=0,
                )

    custom_minor = np.array([
        0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
        100, 200, 500, 1000, 2000, 5000,
        1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6,
    ])
    ymin, ymax = axes[0, 0].get_ylim()
    ticks = custom_minor[(custom_minor >= ymin) & (custom_minor <= ymax)]
    for row in axes:
        for ax in row:
            ax.set_yticks(ticks, minor=True)
            ax.grid(True, which="minor", linestyle="--", alpha=0.20, linewidth=0.5)

    fig.subplots_adjust(
        left=0.08, right=0.995, top=0.94,
        bottom=0.14, hspace=0.30, wspace=0.08,
    )

    out = os.path.join(args.output_dir, "figS9_mosquito_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
