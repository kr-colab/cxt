"""
Figure 7 (PNAS): A. gambiae RDL region + Uganda genome-wide chr2L.

Double-column figure.  Top row: five populations side-by-side showing the
RDL insecticide-resistance region (25.1–25.6 Mb on chr2L).  Bottom row:
Uganda genome-wide chr2L with In(2L)a inversion annotation.  Missingness
tracks beneath each panel.

Targets ~0.25 of a PNAS page height.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT,
)

STEP_BP = 200
REGION_START = 25_100_000
REGION_END = 25_600_000
RDL_START = 25_363_652
RDL_END = 25_434_556

INV_START = 20_524_058
INV_END = 42_165_532

POP_ORDER = ["Mali", "BurkinaFaso", "Cameroon", "Ghana", "Uganda"]
POPULATIONS = {
    "BurkinaFaso": "Burkina Faso",
    "Mali": "Mali",
    "Cameroon": "Cameroon",
    "Ghana": "Ghana",
    "Uganda": "Uganda",
}

AG1000G_ACCESSIBILITY = (
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/"
    "agp3.is_accessible.txt.npz"
)

fmt_mb = FuncFormatter(lambda v, _: f"{v / 1e6:.1f} Mb")
fmt_mb_plain = FuncFormatter(lambda v, _: f"{v / 1e6:.0f} Mb")


def _as_2d(x):
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _extract_rdl_slice(genome, step_bp):
    start_block = REGION_START // step_bp
    n_bins_region = (REGION_END - REGION_START) // step_bp
    end_block = start_block + n_bins_region
    n_bins_chr = genome.shape[1]
    if end_block > n_bins_chr:
        end_block = n_bins_chr
        start_block = end_block - n_bins_region
    return genome[:, start_block:end_block]


def moving_average(frac, smooth_bp=5_000, step_bp=200):
    k = max(1, int(round(smooth_bp / step_bp)))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(frac.astype(np.float32), kernel, mode="same")


def missing_track_region(unaccessible_bitmask, start, end, step_bp=200):
    region = np.asarray(unaccessible_bitmask[start:end], dtype=np.bool_)
    n_bins = (end - start) // step_bp
    pad = n_bins * step_bp - region.size
    if pad > 0:
        region = np.pad(region, (0, pad), constant_values=False)
    return (
        region[: n_bins * step_bp]
        .reshape(n_bins, step_bp)
        .mean(axis=1)
        .astype(np.float32)
    )


def _draw_missing_track(ax, x, missing_frac, smooth_bp=5_000, step_bp=200,
                        height=0.14, pad=0.22, xlabel=None):
    y_raw = np.clip(missing_frac, 0, 1)
    y_smooth = np.clip(moving_average(y_raw, smooth_bp, step_bp), 0, 1)

    tr = ax.inset_axes([0.0, -pad, 1.0, height], transform=ax.transAxes)
    tr.fill_between(x, 0, y_smooth, alpha=0.35, color="lightsteelblue", zorder=1)
    tr.plot(x, y_smooth, lw=0.4, color="steelblue", alpha=0.95, zorder=2)
    tr.plot(x, y_raw, lw=0.2, color="steelblue", alpha=0.30, zorder=2)

    tr.set_xlim(x.min(), x.max())
    tr.set_ylim(0, 1)
    tr.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}"))
    for s in ("top", "right", "left"):
        tr.spines[s].set_visible(False)
    tr.spines["bottom"].set_linewidth(0.4)
    tr.spines["bottom"].set_alpha(0.5)
    tr.set_yticks([])
    tr.set_facecolor("none")
    tr.grid(False)

    if xlabel is None:
        tr.tick_params(axis="x", which="both", labelbottom=False)
    else:
        tr.tick_params(axis="x", labelsize=5)
        tr.set_xlabel(xlabel, fontsize=5.5)

    for line in tr.lines:
        line.set_clip_on(False)
    for coll in tr.collections:
        coll.set_clip_on(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_pnas_style()

    genome_data = {}
    for pop_key in POP_ORDER:
        path = resolve_cache(f"main/cache/fig7/genome_{pop_key.lower()}.npz")
        if os.path.exists(path):
            print(f"Loading {pop_key}: {path}")
            genome_data[pop_key] = np.load(path)["genome"]
        else:
            print(f"Cache not found: {path}")

    if not genome_data:
        print("No genome caches found.")
        return

    unaccessible = None
    if os.path.exists(AG1000G_ACCESSIBILITY):
        unaccessible = ~np.load(AG1000G_ACCESSIBILITY)["access_2L"]

    # --- RDL region data ---
    n_bins_region = (REGION_END - REGION_START) // STEP_BP
    rdl_x = np.arange(n_bins_region) * STEP_BP + REGION_START
    rdl_missing = None
    if unaccessible is not None:
        rdl_missing = missing_track_region(unaccessible, REGION_START, REGION_END, STEP_BP)

    available_pops = [p for p in POP_ORDER if p in genome_data]
    n_pops = len(available_pops)

    # --- Genome-wide Uganda data ---
    uganda_genome = _as_2d(genome_data.get("Uganda", np.array([[]])))
    n_bins_chr = uganda_genome.shape[1]
    chr_x = np.arange(n_bins_chr) * STEP_BP
    chr_len = n_bins_chr * STEP_BP
    chr_missing = None
    if unaccessible is not None:
        chr_missing = missing_track_region(unaccessible, 0, chr_len, STEP_BP)
        if chr_missing.shape[0] > n_bins_chr:
            chr_missing = chr_missing[:n_bins_chr]

    # --- Figure layout ---
    fig = plt.figure(figsize=(DOUBLE_COL, 2.65))
    gs = GridSpec(
        2, n_pops, figure=fig,
        height_ratios=[1.1, 1.0],
        hspace=0.75, wspace=0.12,
    )

    # ── Top row: RDL zoom per population ──────────────────────────────────
    rdl_axes = []
    for i, pop_key in enumerate(available_pops):
        ax = fig.add_subplot(gs[0, i])
        rdl_axes.append(ax)
        genome = _as_2d(genome_data[pop_key])
        tmrca_slice = _extract_rdl_slice(genome, STEP_BP)
        name = POPULATIONS[pop_key]
        n = tmrca_slice.shape[0]

        for j in range(n):
            alpha = 0.25 + 0.35 * (j / max(1, n - 1)) if n > 1 else 0.7
            ax.plot(rdl_x, tmrca_slice[j], lw=0.35, color="steelblue",
                    alpha=alpha, zorder=0)

        mean_t = np.nanmean(tmrca_slice, axis=0)
        ax.plot(rdl_x, mean_t, lw=0.7, color="dodgerblue", zorder=3)

        ax.axvspan(RDL_START, RDL_END, color="crimson", alpha=0.10, zorder=0)
        mid = (RDL_START + RDL_END) / 2
        valid = mean_t[np.isfinite(mean_t) & (mean_t > 0)]
        if valid.size:
            y_arrow = np.exp(
                (np.log(valid.min()) + np.log(valid.max())) / 3
            )
            ax.annotate(
                "", xy=(RDL_END, y_arrow), xytext=(RDL_START, y_arrow),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"),
                zorder=6,
            )
            ax.text(mid, y_arrow * 1.08, "RDL", fontsize=4.5,
                    ha="center", va="bottom", zorder=6)

        ax.set_yscale("log")
        ax.set_xlim(REGION_START, REGION_END)
        ax.set_ylim(0.5e2, 5e6)
        ax.set_title(name, loc="left", fontsize=6, pad=2)
        ax.xaxis.set_major_formatter(fmt_mb)
        ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.tick_params(axis="both", labelsize=5, pad=1)

        if i == 0:
            ax.set_ylabel("TMRCA\n(generations)", fontsize=5.5)
        else:
            ax.tick_params(labelleft=False)

        if rdl_missing is not None:
            is_mid = (i == n_pops // 2)
            _draw_missing_track(
                ax, rdl_x, rdl_missing,
                smooth_bp=5_000, step_bp=STEP_BP, height=0.12, pad=0.20,
                xlabel="Position on 2L (Mb)" if is_mid else " ",
            )

    # ── Bottom: Uganda genome-wide ────────────────────────────────────────
    ax_gw = fig.add_subplot(gs[1, :])
    n_ug = uganda_genome.shape[0]

    for j in range(n_ug):
        alpha = 0.08 + 0.15 * (j / max(1, n_ug - 1)) if n_ug > 1 else 0.3
        ax_gw.plot(chr_x, uganda_genome[j], lw=0.2, color="silver",
                   alpha=alpha, zorder=0)

    mean_ug = np.nanmean(uganda_genome, axis=0)
    ax_gw.plot(chr_x, mean_ug, lw=0.5, color="dodgerblue", zorder=3)

    ax_gw.axvspan(INV_START, INV_END, color="cornflowerblue", alpha=0.08, zorder=0)
    inv_mid = (INV_START + INV_END) / 2
    ax_gw.annotate(
        "", xy=(INV_END, 1.5e4), xytext=(INV_START, 1.5e4),
        arrowprops=dict(arrowstyle="<->", lw=0.6, color="black"),
        zorder=6,
    )
    ax_gw.text(inv_mid, 2.0e4, "2La inversion", fontsize=5,
               ha="center", va="bottom", style="italic", zorder=6)

    ax_gw.set_yscale("log")
    ax_gw.set_xlim(0, chr_len)
    ax_gw.set_ylim(0.5e2, 5e6)
    ax_gw.set_title("Uganda", loc="left", fontsize=6, pad=2)
    ax_gw.set_ylabel("TMRCA\n(generations)", fontsize=5.5)
    ax_gw.xaxis.set_major_formatter(fmt_mb_plain)
    ax_gw.tick_params(axis="x", which="both", labelbottom=False)
    ax_gw.tick_params(axis="both", labelsize=5, pad=1)

    if chr_missing is not None:
        _draw_missing_track(
            ax_gw, chr_x, chr_missing,
            smooth_bp=50_000, step_bp=STEP_BP, height=0.12, pad=0.18,
            xlabel="Position on 2L (Mb)",
        )

    savefig(fig, "figure7", output_dir=args.output_dir)
    print("Done: figure7")


if __name__ == "__main__":
    main()
