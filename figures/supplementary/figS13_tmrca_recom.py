"""
Figure S13: TMRCA vs recombination breakpoints for chromosomes 2 and 6.

Loads corrected genome TMRCA caches produced by fig6_human_1kg and plots
average TMRCA per 100 kb bin against inferred recombination breakpoint counts.

Replicates the analysis from scratch/revision/figure6/experiment.ipynb
using run_fresh caches.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


GENERATION_TIME = 28
WINDOW_SIZE = 2_000      # 2 kb
BIN_SIZE_BP = 100_000    # 100 kb


def recombinations_per_100kb(tmrcas, window_size=WINDOW_SIZE,
                             bin_size=BIN_SIZE_BP, threshold=10):
    n_ind, n_win = tmrcas.shape
    windows_per_bin = bin_size // window_size
    n_bins = n_win // windows_per_bin
    trimmed = tmrcas[:, :n_bins * windows_per_bin].reshape(n_ind, n_bins, windows_per_bin)
    diffs = np.abs(np.diff(trimmed, axis=2))
    recombs = (diffs > threshold).sum(axis=2)
    return recombs


def mean_tmrca_per_bin(tmrcas, window_size=WINDOW_SIZE, bin_size=BIN_SIZE_BP):
    n_ind, n_win = tmrcas.shape
    windows_per_bin = bin_size // window_size
    n_bins = n_win // windows_per_bin
    binned = tmrcas[:, :n_bins * windows_per_bin].reshape(n_ind, n_bins, windows_per_bin)
    return binned.mean(axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig6")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    genome_chr2p = np.load(os.path.join(args.cache_dir, "genome_chr2p.npz"))["genome"]
    genome_chr2q = np.load(os.path.join(args.cache_dir, "genome_chr2q.npz"))["genome"]
    genome_chr6p = np.load(os.path.join(args.cache_dir, "genome_chr6p.npz"))["genome"]
    genome_chr6q = np.load(os.path.join(args.cache_dir, "genome_chr6q.npz"))["genome"]

    genome_chr2 = genome_chr2p + genome_chr2q
    genome_chr6 = genome_chr6p + genome_chr6q

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (genome, label) in zip(axes, [
        (genome_chr2, "Chromosome 2"),
        (genome_chr6, "Chromosome 6"),
    ]):
        tmrcas = np.exp(genome) * GENERATION_TIME
        recomb = recombinations_per_100kb(tmrcas)
        mean_tmrca = mean_tmrca_per_bin(tmrcas)

        sns.scatterplot(x=recomb.flatten(), y=mean_tmrca.flatten(),
                        color="dodgerblue", alpha=0.4, s=10, ax=ax,
                        edgecolor="none")
        ax.set_yscale("log")
        ax.set_ylim(1000, 2e7)
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_xlabel("Recombination breakpoints\nper 100 kb")
        ax.grid(alpha=0.3, which="both", ls="--")

    axes[0].set_ylabel("Average TMRCA per 100 kb (years)")
    plt.tight_layout()

    out = os.path.join(args.output_dir, "tmrca_recom.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
