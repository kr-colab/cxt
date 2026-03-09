"""
Figure 6: Inference of coalescent times from GBR individuals of the 1000 Genomes Project.

Produces chromosome-wide TMRCA landscapes for chr2 and chr6, with highlighted
LCT locus and HLA region.

Faithful reproduction of revision/figure6/experiment.ipynb.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tszip

import cxt
from cxt.correction import stochastic_diversity_bias_correction
from figures.paths import HG1KG_TSZ_DIR


GENERATION_TIME = 28
BIN_BP = 2000
MUTATION_RATE = 1.29e-8

# Module-level globals for ProcessPoolExecutor workers
_TMRCA = None
_INDEX_MAP = None
_TS = None
_BLOCKS = None
_AVAILABILITY = None
_PIVOT_PAIRS = None


def get_missingness_grid(chrom, mask_dir, window_size=1e6):
    """Compute per-1Mb-window availability from 1KG mask files."""
    mask = pd.read_pickle(Path(mask_dir) / f'{chrom}.mask.pkl')
    pos = mask['position']
    bcgm_mask = mask['bcgm_masked'].astype(int)

    edges = np.arange(0, pos.max() + window_size, window_size)
    idx = np.digitize(pos, edges) - 1
    totals = np.bincount(idx, minlength=len(edges) - 1)
    miss = np.bincount(idx, weights=bcgm_mask, minlength=len(edges) - 1)

    frac_missing = np.divide(
        miss, totals, out=np.zeros_like(miss, float), where=totals > 0
    )
    return frac_missing


def load_or_infer_arm(ts_loader, cache_name, model, pivot_pairs, devices, cache_dir):
    """Load cached TMRCA or run cxt inference for one chromosome arm."""
    cache_path = os.path.join(cache_dir, f"{cache_name}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["tmrca"], d["index_map"], ts_loader()

    ts = ts_loader()
    num_blocks = int(ts.sequence_length // 1e6)
    blocks = [(int(i), int(i + 1e6))
              for i in np.linspace(0, num_blocks * 1e6 - 1e6, num_blocks)]

    tmrca, index_map = cxt.translate(
        ts, model, pivot_pairs=pivot_pairs,
        blocks=blocks, devices=devices,
        B_per_device=128, B=128,
        build_workers=32, mutation_rate=None,
    )
    np.savez_compressed(cache_path, tmrca=tmrca, index_map=index_map)
    return tmrca, index_map, ts


def _one_block(i):
    """Bias-correct one block (uses module-level globals, fork-safe)."""
    sample_index = (_INDEX_MAP[:, 0] == i)
    tmrca_block = _TMRCA[:, sample_index]
    block = _BLOCKS[i]
    rng = np.random.default_rng(20_000_001 + i)
    try:
        ts_block = _TS.keep_intervals([block], simplify=True)
        avail = _AVAILABILITY[i] / 100 if _AVAILABILITY[i] > 0 else 1.0
        out = stochastic_diversity_bias_correction(
            tree_sequence=ts_block,
            mutation_rate=MUTATION_RATE / avail,
            predictions=tmrca_block,
            pivot_pairs=np.array(_PIVOT_PAIRS),
            rng=rng,
        )
        return out.mean(0)
    except Exception:
        return np.zeros([25, 500])


def correct_arm(tmrca, index_map, ts, availability,
                pivot_pairs, cache_dir, name, workers=64):
    """Apply stochastic diversity bias correction per 1-Mb block."""
    global _TMRCA, _INDEX_MAP, _TS, _BLOCKS, _AVAILABILITY, _PIVOT_PAIRS

    cache_path = os.path.join(cache_dir, f"genome_{name}.npz")
    if os.path.exists(cache_path):
        return np.load(cache_path)["genome"]

    num_blocks = int(ts.sequence_length // 1e6)
    blocks = [(int(i), int(i + 1e6))
              for i in np.linspace(0, num_blocks * 1e6 - 1e6, num_blocks)]

    _TMRCA = tmrca
    _INDEX_MAP = index_map
    _TS = ts
    _BLOCKS = blocks
    _AVAILABILITY = availability
    _PIVOT_PAIRS = pivot_pairs

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one_block, range(num_blocks)))

    genome = np.stack(results, axis=0).transpose(1, 0, 2).reshape(25, -1)
    np.savez_compressed(cache_path, genome=genome)
    return genome


def make_genome_nocorrection(tmrca, index_map, n_pairs):
    """Assemble genome without correction: group by pair, mean over reps."""
    n_blocks = index_map[:, 0].max() + 1
    tmrca_mean = tmrca.mean(0)  # (N, 500)
    tmrca_3d = tmrca_mean.reshape(n_blocks, n_pairs, 500)
    return tmrca_3d.transpose(1, 0, 2).reshape(n_pairs, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig6")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--tsz-dir", default=HG1KG_TSZ_DIR)
    parser.add_argument("--no-correction", action="store_true",
                        help="Skip bias correction (faster, for testing)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = cxt.load_model("broad", device="cpu")
    pivot_pairs = [(i, i + 1) for i in range(0, 50, 2)]

    # --- Load missingness masks ---
    chr2_available = (1 - get_missingness_grid("chr2", args.tsz_dir)) * 100
    chr6_available = (1 - get_missingness_grid("chr6", args.tsz_dir)) * 100

    # --- Infer or load each arm ---
    arms = {}
    for arm_name in ["chr6p", "chr6q", "chr2p", "chr2q"]:
        def _load(name=arm_name):
            tsz_path = os.path.join(args.cache_dir, f"gbr_{name}.tsz")
            if not os.path.exists(tsz_path):
                src = os.path.join(args.tsz_dir, f"{name}.tsz")
                ts = tszip.load(src)
                ts = ts.simplify(samples=np.arange(50))
                tszip.compress(ts, tsz_path)
                return ts
            return tszip.load(tsz_path)

        tmrca, index_map, ts = load_or_infer_arm(
            _load, f"gbr_{arm_name}", model, pivot_pairs,
            args.devices, args.cache_dir,
        )
        arms[arm_name] = {"tmrca": tmrca, "index_map": index_map, "ts": ts}

    # --- Bias correction per arm ---
    if args.no_correction:
        genome_chr2p = make_genome_nocorrection(
            arms["chr2p"]["tmrca"], arms["chr2p"]["index_map"], 25)
        genome_chr2q = make_genome_nocorrection(
            arms["chr2q"]["tmrca"], arms["chr2q"]["index_map"], 25)
        genome_chr6p = make_genome_nocorrection(
            arms["chr6p"]["tmrca"], arms["chr6p"]["index_map"], 25)
        genome_chr6q = make_genome_nocorrection(
            arms["chr6q"]["tmrca"], arms["chr6q"]["index_map"], 25)
    else:
        genome_chr2p = correct_arm(
            arms["chr2p"]["tmrca"], arms["chr2p"]["index_map"],
            arms["chr2p"]["ts"], chr2_available, pivot_pairs,
            args.cache_dir, "chr2p",
        )
        genome_chr2q = correct_arm(
            arms["chr2q"]["tmrca"], arms["chr2q"]["index_map"],
            arms["chr2q"]["ts"], chr2_available, pivot_pairs,
            args.cache_dir, "chr2q",
        )
        genome_chr6p = correct_arm(
            arms["chr6p"]["tmrca"], arms["chr6p"]["index_map"],
            arms["chr6p"]["ts"], chr6_available, pivot_pairs,
            args.cache_dir, "chr6p",
        )
        genome_chr6q = correct_arm(
            arms["chr6q"]["tmrca"], arms["chr6q"]["index_map"],
            arms["chr6q"]["ts"], chr6_available, pivot_pairs,
            args.cache_dir, "chr6q",
        )

    genome_chr2 = genome_chr2p + genome_chr2q
    genome_chr6 = genome_chr6p + genome_chr6q

    lct_start, lct_end = 135_300_000, 136_400_000
    hla_start, hla_end = 29_600_000, 33_300_000

    # ---- Panel 1: chromosome-wide with LCT/HLA highlighted ----
    x_chr2 = np.arange(genome_chr2.shape[1]) * BIN_BP
    x_chr6 = np.arange(genome_chr6.shape[1]) * BIN_BP

    datasets = [
        (x_chr2, np.exp(genome_chr2.mean(0)) * GENERATION_TIME, 'Chromosome 2'),
        (x_chr6, np.exp(genome_chr6.mean(0)) * GENERATION_TIME, 'Chromosome 6'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=False, sharey=True)
    for ax, (x, y, label) in zip(axes, datasets):
        ax.plot(x, y, linewidth=0.8, color='dodgerblue', alpha=0.9)
        ax.set_yscale('log')
        ax.set_title(label, loc='left', fontsize=12)
        ax.grid(alpha=0.3, which='both', linestyle='--')

    axes[0].axvspan(lct_start, lct_end, color='crimson', alpha=0.3, label='LCT')
    axes[1].axvspan(hla_start, hla_end, color='crimson', alpha=0.3, label='HLA')
    axes[0].text(lct_start, 3e6, "LCT", color='crimson', fontsize=10, va='bottom')
    axes[1].text(hla_start, 3e6, "HLA", color='crimson', fontsize=10, va='bottom')

    fig.text(0.5, 0.03, 'Position (bp)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Mean TMRCA (years)', va='center',
             rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.ylim(1e3, 1e7)

    out = os.path.join(args.output_dir, "figure6_human_1kg.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    # ---- Panel 2: zoomed LCT and HLA regions ----
    hla_genes = [
        ("HLA-V",29759130,29765588), ("HLA-H",29855342,29858857),
        ("HLA-F-AS1",29675299,29796273), ("HLA-DRB6",32519632,32526623),
        ("HLA-DPA1",33032346,33041426), ("HLA-L",30227330,30234728),
        ("HLA-DOB",32780540,32784779), ("HLA-DQB2",32723875,32731309),
        ("HLA-DQB1",32627244,32634434), ("HLA-DQB1-AS1",32627657,32628506),
        ("HLA-DPB1",33043767,33057473), ("HLA-DMB",32902413,32908805),
        ("HLA-B",31321652,31324956), ("HLA-DRA",32407664,32412823),
        ("HLA-A",29910309,29913647), ("HLA-E",30457286,30461971),
        ("HLA-C",31236526,31239869), ("HLA-DRB5",32485130,32498064),
        ("HLA-DQA2",32709168,32714975), ("HLA-DMA",32916395,32920874),
        ("HLA-G",29795602,29798798), ("HLA-DRB1",32546552,32557625),
        ("HLA-DQA1",32605183,32611461), ("HLA-F",29691211,29695073),
        ("HLA-DOA",32971959,32977368),
    ]
    lct_genes = [
        ("MAP3K19",134964490,135047447), ("RAB3GAP1",135052291,135176396),
        ("ZRANB3",135196968,135531218), ("R3HDM1",135531483,135725269),
        ("UBXN4",135741854,135785056), ("LCT",135787849,135837184),
        ("MCM6",135839625,135876443), ("DARS1",135905880,135985684),
        ("CXCR4",136114348,136118149),
    ]

    def _stack_gene_labels(genes, min_sep_mb=0.6):
        sorted_genes = sorted(genes, key=lambda g: (g[1] + g[2]) / 2)
        levels, coords = [], []
        for name, start, end in sorted_genes:
            mid = (start + end) / 2 / 1e6
            lvl = 0
            while lvl < len(levels) and any(abs(mid - m) < min_sep_mb for m in levels[lvl]):
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
            ax.text(mid, y, name, ha="center", va="bottom", fontsize=7,
                    color="crimson", zorder=6)

        ax.set_title(chr_label, loc="left", fontsize=12)
        ax.set_xlim(zoom_lo_mb, zoom_hi_mb)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both", ls="--")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    _plot_zoom(axes2[0], genome_chr2, "Chr2 LCT locus",
               133.0, 138.0, lct_genes, base_y=3e4, spacing=1.9)
    _plot_zoom(axes2[1], genome_chr6, "Chr6 HLA region",
               29.0, 34.0, hla_genes, base_y=3e4, spacing=1.9)

    fig2.text(0.5, 0.03, "Genomic position (Mb)", ha="center", fontsize=12)
    fig2.text(0.04, 0.5, "TMRCA (years)", va="center",
              rotation="vertical", fontsize=12)
    axes2[1].legend(loc="lower right", fontsize=9, ncol=3)
    plt.tight_layout(rect=[0.06, 0.05, 1, 0.98])
    plt.ylim(1e3, 3e8)

    out2 = os.path.join(args.output_dir, "figure6_human_1kg_zoom.png")
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
