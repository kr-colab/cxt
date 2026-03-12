"""
Figure 6 (PNAS): Human 1000 Genomes TMRCA landscapes for chr2 and chr6.

Produces two figures:
  1. Chromosome-wide TMRCA overview (chr2, chr6) with LCT/HLA highlights.
  2. Zoomed LCT locus and HLA region with individual traces and gene annotations.

Matches the plotting logic of the original fig6_human_1kg.py.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT,
)

GENERATION_TIME = 28
BIN_BP = 2000

LCT_START, LCT_END = 135_300_000, 136_400_000
HLA_START, HLA_END = 29_600_000, 33_300_000

HLA_GENES = [
    ("HLA-V", 29759130, 29765588), ("HLA-H", 29855342, 29858857),
    ("HLA-F-AS1", 29675299, 29796273), ("HLA-DRB6", 32519632, 32526623),
    ("HLA-DPA1", 33032346, 33041426), ("HLA-L", 30227330, 30234728),
    ("HLA-DOB", 32780540, 32784779), ("HLA-DQB2", 32723875, 32731309),
    ("HLA-DQB1", 32627244, 32634434), ("HLA-DQB1-AS1", 32627657, 32628506),
    ("HLA-DPB1", 33043767, 33057473), ("HLA-DMB", 32902413, 32908805),
    ("HLA-B", 31321652, 31324956), ("HLA-DRA", 32407664, 32412823),
    ("HLA-A", 29910309, 29913647), ("HLA-E", 30457286, 30461971),
    ("HLA-C", 31236526, 31239869), ("HLA-DRB5", 32485130, 32498064),
    ("HLA-DQA2", 32709168, 32714975), ("HLA-DMA", 32916395, 32920874),
    ("HLA-G", 29795602, 29798798), ("HLA-DRB1", 32546552, 32557625),
    ("HLA-DQA1", 32605183, 32611461), ("HLA-F", 29691211, 29695073),
    ("HLA-DOA", 32971959, 32977368),
]
LCT_GENES = [
    ("MAP3K19", 134964490, 135047447), ("RAB3GAP1", 135052291, 135176396),
    ("ZRANB3", 135196968, 135531218), ("R3HDM1", 135531483, 135725269),
    ("UBXN4", 135741854, 135785056), ("LCT", 135787849, 135837184),
    ("MCM6", 135839625, 135876443), ("DARS1", 135905880, 135985684),
    ("CXCR4", 136114348, 136118149),
]


def _load_genome(arm_name):
    """Load a genome arm, trying genome_* then gbr_* naming conventions."""
    for prefix in ["genome_", "gbr_"]:
        path = resolve_cache(f"main/cache/fig6/{prefix}{arm_name}.npz")
        if os.path.exists(path):
            return np.load(path)["genome"]
    return None


def _stack_gene_labels(genes, min_sep_mb=0.6):
    sorted_genes = sorted(genes, key=lambda g: (g[1] + g[2]) / 2)
    levels, coords = [], []
    for name, start, end in sorted_genes:
        mid = (start + end) / 2 / 1e6
        lvl = 0
        while lvl < len(levels) and any(
                abs(mid - m) < min_sep_mb for m in levels[lvl]):
            lvl += 1
        if lvl == len(levels):
            levels.append([])
        levels[lvl].append(mid)
        coords.append((mid, lvl, name, start / 1e6, end / 1e6))
    return coords


def _plot_zoom(ax, genome_log, chr_label, zoom_lo_mb, zoom_hi_mb,
               genes, base_y=3e4, spacing=1.8):
    R, W = genome_log.shape
    start_idx = max(0, int(np.floor((zoom_lo_mb * 1e6) / BIN_BP)))
    end_idx = min(W, int(np.ceil((zoom_hi_mb * 1e6) / BIN_BP)))
    idx = np.arange(start_idx, end_idx)
    x_mb = (idx * BIN_BP) / 1e6
    Y = np.exp(genome_log[:, start_idx:end_idx].astype(float)) * GENERATION_TIME
    Y[~np.isfinite(Y)] = np.nan

    for r in range(min(R, 10)):
        ax.plot(x_mb, Y[r], lw=0.6, color="dodgerblue", alpha=0.25, zorder=1)

    q25 = np.nanpercentile(Y, 25, axis=0)
    q75 = np.nanpercentile(Y, 75, axis=0)
    mu = np.nanmean(Y, axis=0)
    med = np.nanmedian(Y, axis=0)

    ax.fill_between(x_mb, q25, q75, alpha=0.18, color="dodgerblue", zorder=2,
                    label="IQR (25\u201375%)")
    ax.plot(x_mb, mu, lw=2.0, color="dodgerblue", zorder=3, label="Mean")
    ax.plot(x_mb, med, lw=1.6, color="dodgerblue", ls="--", zorder=3,
            label="Median")

    coords = _stack_gene_labels(genes, min_sep_mb=0.6)
    for mid, lvl, name, g_lo, g_hi in coords:
        if g_hi < zoom_lo_mb or g_lo > zoom_hi_mb:
            continue
        y = base_y * (spacing ** lvl)
        ax.hlines(y, max(g_lo, zoom_lo_mb), min(g_hi, zoom_hi_mb),
                  colors="crimson", lw=1.6, alpha=0.95, zorder=5)
        ax.text(mid, y, name, ha="center", va="bottom", fontsize=5,
                color="crimson", zorder=6)

    ax.set_title(chr_label, loc="left")
    ax.set_xlim(zoom_lo_mb, zoom_hi_mb)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both", ls="--")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_pnas_style()

    genome_chr2p = _load_genome("chr2p")
    genome_chr2q = _load_genome("chr2q")
    genome_chr6p = _load_genome("chr6p")
    genome_chr6q = _load_genome("chr6q")

    genome_chr2, genome_chr6 = None, None
    if genome_chr2p is not None and genome_chr2q is not None:
        genome_chr2 = genome_chr2p + genome_chr2q
    elif genome_chr2p is not None:
        genome_chr2 = genome_chr2p
    elif genome_chr2q is not None:
        genome_chr2 = genome_chr2q
    if genome_chr6p is not None and genome_chr6q is not None:
        genome_chr6 = genome_chr6p + genome_chr6q
    elif genome_chr6p is not None:
        genome_chr6 = genome_chr6p
    elif genome_chr6q is not None:
        genome_chr6 = genome_chr6q

    if genome_chr2 is None and genome_chr6 is None:
        print("No genome caches found for fig6.")
        return

    # --- Figure 1: chromosome-wide overview ---
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 1.8),
                             sharex=False, sharey=True)

    for col, (genome, label, reg_start, reg_end, reg_label) in enumerate([
        (genome_chr2, "Chromosome 2", LCT_START, LCT_END, "LCT"),
        (genome_chr6, "Chromosome 6", HLA_START, HLA_END, "HLA"),
    ]):
        ax = axes[col]
        if genome is not None:
            x = np.arange(genome.shape[1]) * BIN_BP
            y = np.exp(genome.mean(0)) * GENERATION_TIME
            ax.plot(x, y, linewidth=0.8, color='dodgerblue', alpha=0.9)
            ax.set_yscale('log')
            ax.set_ylim(1e3, 1e7)
            ax.axvspan(reg_start, reg_end, color='crimson', alpha=0.3)
            ax.text(reg_start, 3e6, reg_label, color='crimson',
                    fontsize=8, va='bottom')
            ax.grid(alpha=0.3, which='both', linestyle='--')
        else:
            ax.text(0.5, 0.5, f"{label}: cache missing", ha="center",
                    va="center", fontsize=7, alpha=0.5,
                    transform=ax.transAxes)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Position (bp)")

    axes[0].set_ylabel("Mean TMRCA (years)")
    plt.tight_layout()
    savefig(fig, "figure6", output_dir=args.output_dir)
    print("Done: figure6")

    # --- Figure 2: zoomed LCT and HLA ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.25), sharey=True)

    if genome_chr2 is not None:
        _plot_zoom(axes2[0], genome_chr2, "Chr2 \u2013 LCT locus",
                   133.0, 138.0, LCT_GENES, base_y=3e4, spacing=1.9)
    else:
        axes2[0].text(0.5, 0.5, "chr2 cache missing", ha="center",
                      va="center", fontsize=7, alpha=0.5,
                      transform=axes2[0].transAxes)
        axes2[0].set_title("Chr2 \u2013 LCT locus", loc="left")

    if genome_chr6 is not None:
        _plot_zoom(axes2[1], genome_chr6, "Chr6 \u2013 HLA region",
                   29.0, 34.0, HLA_GENES, base_y=3e4, spacing=1.9)
        axes2[1].legend(loc="lower right", ncol=3)
    else:
        axes2[1].text(0.5, 0.5, "chr6 cache missing", ha="center",
                      va="center", fontsize=7, alpha=0.5,
                      transform=axes2[1].transAxes)
        axes2[1].set_title("Chr6 \u2013 HLA region", loc="left")

    axes2[0].set_ylabel("TMRCA (years)")
    axes2[0].set_xlabel("Genomic position (Mb)")
    axes2[1].set_xlabel("Genomic position (Mb)")
    axes2[0].set_ylim(1e3, 3e8)
    plt.tight_layout()
    savefig(fig2, "figure6_zoom", output_dir=args.output_dir)
    print("Done: figure6_zoom")


if __name__ == "__main__":
    main()
