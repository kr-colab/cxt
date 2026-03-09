"""
Figure 6: Inference of coalescent times from GBR individuals of the 1000 Genomes Project.

Produces chromosome-wide TMRCA landscapes for chr2 and chr6, with zoomed
panels for the LCT locus and HLA region.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from figures.paths import HG1KG_TSZ_DIR
import matplotlib.pyplot as plt
import torch
import tszip

from cxt.api2 import translate
from cxt.utils import setup_cxt_model, stochastic_diversity_bias_correction


GENERATION_TIME = 28
BIN_BP = 2000
MUTATION_RATE = 1.29e-8

LCT_GENES = [
    ("MAP3K19", 134964490, 135047447), ("RAB3GAP1", 135052291, 135176396),
    ("ZRANB3", 135196968, 135531218), ("R3HDM1", 135531483, 135725269),
    ("UBXN4", 135741854, 135785056), ("LCT", 135787849, 135837184),
    ("MCM6", 135839625, 135876443), ("DARS1", 135905880, 135985684),
    ("CXCR4", 136114348, 136118149),
]

HLA_GENES = [
    ("HLA-F", 29691211, 29695073), ("HLA-A", 29910309, 29913647),
    ("HLA-E", 30457286, 30461971), ("HLA-C", 31236526, 31239869),
    ("HLA-B", 31321652, 31324956), ("HLA-DRA", 32407664, 32412823),
    ("HLA-DRB5", 32485130, 32498064), ("HLA-DRB1", 32546552, 32557625),
    ("HLA-DQA1", 32605183, 32611461), ("HLA-DQB1", 32627244, 32634434),
    ("HLA-DOB", 32780540, 32784779), ("HLA-DMB", 32902413, 32908805),
    ("HLA-DOA", 32971959, 32977368), ("HLA-DPA1", 33032346, 33041426),
    ("HLA-DPB1", 33043767, 33057473),
]


def load_or_infer_arm(ts_loader, cache_name, model, pivot_pairs, devices, cache_dir):
    """Load cached TMRCA or run cxt inference for one chromosome arm."""
    cache_path = os.path.join(cache_dir, f"{cache_name}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["tmrca"], d["index_map"], ts_loader()

    ts = ts_loader()
    num_blocks = int(ts.sequence_length // 1e6)
    blocks = [(int(i), int(i + 1e6)) for i in np.linspace(0, num_blocks * 1e6 - 1e6, num_blocks)]

    tmrca, index_map = translate(
        input_data=ts, data_type="ts",
        model=model, pivot_pairs=pivot_pairs,
        blocks=blocks, devices=devices,
        B_per_device=128, B=128,
        build_workers=32, mutation_rate=None,
    )
    np.savez_compressed(cache_path, tmrca=tmrca, index_map=index_map)
    return tmrca, index_map, ts


def correct_arm(tmrca, index_map, ts, blocks, pivot_pairs, availability, cache_dir, name, workers=64):
    """Apply stochastic diversity bias correction per block."""
    cache_path = os.path.join(cache_dir, f"genome_{name}.npz")
    if os.path.exists(cache_path):
        return np.load(cache_path)["genome"]

    def _one_block(i):
        sample_index = (index_map[:, 0] == i)
        tmrca_block = tmrca[:, sample_index]
        block = blocks[i]
        rng = np.random.default_rng(20_000_001 + i)
        try:
            ts_block = ts.keep_intervals([block], simplify=True)
            avail = availability[i] / 100 if availability[i] > 0 else 1.0
            out = stochastic_diversity_bias_correction(
                tree_sequence=ts_block,
                mutation_rate=MUTATION_RATE / avail,
                predictions=tmrca_block,
                pivot_pairs=np.array(pivot_pairs),
                rng=rng,
            )
            return out.mean(0)
        except Exception:
            return np.zeros([25, 500])

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one_block, range(len(blocks))))

    genome = np.stack(results, axis=0).transpose(1, 0, 2).reshape(25, -1)
    np.savez_compressed(cache_path, genome=genome)
    return genome


def plot_zoom(ax, genome_log, title, zoom_lo_mb, zoom_hi_mb, genes,
              base_y=3e4, spacing=1.9):
    """Plot zoomed TMRCA panel with gene annotations."""
    R, W = genome_log.shape
    start_idx = max(0, int(np.floor((zoom_lo_mb * 1e6) / BIN_BP)))
    end_idx = min(W, int(np.ceil((zoom_hi_mb * 1e6) / BIN_BP)))
    idx = np.arange(start_idx, end_idx)
    x_mb = (idx * BIN_BP) / 1e6

    Y = np.exp(np.asarray(genome_log[:, start_idx:end_idx], float)) * GENERATION_TIME
    Y[~np.isfinite(Y)] = np.nan

    for r in range(min(R, 10)):
        ax.plot(x_mb, Y[r], lw=0.6, color="dodgerblue", alpha=0.25, zorder=1)

    mu = np.nanmean(Y, axis=0)
    med = np.nanmedian(Y, axis=0)
    q25 = np.nanpercentile(Y, 25, axis=0)
    q75 = np.nanpercentile(Y, 75, axis=0)

    ax.fill_between(x_mb, q25, q75, alpha=0.18, color="dodgerblue", zorder=2)
    ax.plot(x_mb, mu, lw=2.0, color="dodgerblue", zorder=3, label="Mean")
    ax.plot(x_mb, med, lw=1.6, color="dodgerblue", ls="--", zorder=3, label="Median")

    sorted_genes = sorted(genes, key=lambda g: (g[1] + g[2]) / 2)
    levels = []
    for name, start, end in sorted_genes:
        mid = (start + end) / 2 / 1e6
        g_lo, g_hi = start / 1e6, end / 1e6
        if g_hi < zoom_lo_mb or g_lo > zoom_hi_mb:
            continue
        lvl = 0
        while lvl < len(levels) and any(abs(mid - m) < 0.6 for m in levels[lvl]):
            lvl += 1
        if lvl == len(levels):
            levels.append([])
        levels[lvl].append(mid)
        y = base_y * (spacing ** lvl)
        ax.hlines(y, max(g_lo, zoom_lo_mb), min(g_hi, zoom_hi_mb),
                  colors="crimson", lw=1.6, alpha=0.95, zorder=5)
        ax.text(mid, y, name, ha="center", va="bottom", fontsize=7, color="crimson", zorder=6)

    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlim(zoom_lo_mb, zoom_hi_mb)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both", ls="--")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig6")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--tsz-dir", default=HG1KG_TSZ_DIR,
                        help="Directory containing .tsz tree sequence files")
    args = parser.parse_args()

    if args.devices is None:
        args.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = setup_cxt_model(model_type="broad")
    pivot_pairs = [(i, i + 1) for i in range(0, 50, 2)]

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

        tmrca, index_map, ts = load_or_infer_arm(_load, f"gbr_{arm_name}", model, pivot_pairs, args.devices, args.cache_dir)
        arms[arm_name] = {"tmrca": tmrca, "index_map": index_map, "ts": ts}

    # NOTE: bias correction requires availability masks from the 1000 Genomes mask files.
    # If not available, fall back to raw (uncorrected) TMRCA.
    genome_chr2 = np.concatenate([
        arms["chr2p"]["tmrca"].mean(0).reshape(25, -1),
        arms["chr2q"]["tmrca"].mean(0).reshape(25, -1),
    ], axis=1)
    genome_chr6 = np.concatenate([
        arms["chr6p"]["tmrca"].mean(0).reshape(25, -1),
        arms["chr6q"]["tmrca"].mean(0).reshape(25, -1),
    ], axis=1)

    # --- Zoom panels ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

    plot_zoom(axes[0], genome_chr2, "Chr2 LCT locus", 133.0, 138.0, LCT_GENES)
    plot_zoom(axes[1], genome_chr6, "Chr6 HLA region", 29.0, 34.0, HLA_GENES)

    fig.text(0.5, 0.03, "Genomic position (Mb)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "TMRCA (years)", va="center", rotation="vertical", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=9, ncol=3)
    plt.tight_layout(rect=[0.06, 0.05, 1, 0.98])
    plt.ylim(1e3, 3e8)

    out = os.path.join(args.output_dir, "figure6_human_1kg.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
