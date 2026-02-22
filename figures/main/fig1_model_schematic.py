"""
Figure 1: Model schematic and batch inference demonstration.

Infers pairwise coalescence times for all 1225 pairs of 50 haplotypes
under a constant population size scenario and visualizes selected
TMRCA trajectories with uncertainty bands.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import setup_cxt_model, simulate_parameterized_tree_sequence, TIMES


def build_yhats_ytrues(ts, pivot_ids, yhat_tmrca, max_workers=None):
    yhat = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ytrues = list(ex.map(
            lambda args: interpolate_tmrcas(args[0], 2000, 1e6, args[1], args[2]),
            [(ts, a, b) for a, b in pivot_ids],
        ))
    return yhat, ytrues


def discretize(sequence, population_time):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    indices = np.clip(indices, 0, len(population_time) - 1)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig1")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = setup_cxt_model(model_type="narrow")

    num_samples = 50
    pivot_pairs = [(i, j) for i in range(num_samples) for j in range(i + 1, num_samples)]
    blocks = [(0, 1e6)]

    ts = simulate_parameterized_tree_sequence(seed=103370001)

    cache_path = os.path.join(args.cache_dir, "constant_cxt.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        yhats, ytrues = data["yhats"], data["ytrues"]
    else:
        yhat_tmrca, _ = translate(
            input_data=ts, data_type="ts",
            model=model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=args.batch_size, B=args.batch_size,
            build_workers=8, mutation_rate=1.29e-08,
        )
        yhats, ytrues = build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, max_workers=24)
        yhats, ytrues = np.array([yhats]), np.array([ytrues])
        np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)

    ytrues = np.exp(TIMES[discretize(np.log(ytrues), TIMES)])
    ytrues = np.tile(ytrues, (15, 1, 1))
    yhats = np.log(yhats[0])
    ytrues = np.log(ytrues)

    # --- Plot ---
    x_values = np.arange(0, 1_000_000, 2000)
    fontsize = 14
    plt.rcParams.update({"font.size": fontsize})

    def calculate_mean_std(data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    yhat_mean1, yhat_std1 = calculate_mean_std(yhats[:, :2, :])
    ytrue_mean1 = np.mean(ytrues[:, :2, :], axis=0)
    yhat_mean2, yhat_std2 = calculate_mean_std(yhats[:, -2:, :])
    ytrue_mean2 = np.mean(ytrues[:, -2:, :], axis=0)

    fig, axs = plt.subplots(
        5, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": [1, 1, 0.5, 1, 1], "hspace": 0.5},
    )

    def millions(x, pos):
        return "0" if x == 0 else f"{x / 1e6:.1f}x10\u2076"

    panels = [
        (axs[0], yhat_mean1[0], yhat_std1[0], ytrue_mean1[0], "[1/1225]"),
        (axs[1], yhat_mean1[1], yhat_std1[1], ytrue_mean1[1], "[2/1225]"),
        (axs[3], yhat_mean2[0], yhat_std2[0], ytrue_mean2[0], "[1224/1225]"),
        (axs[4], yhat_mean2[1], yhat_std2[1], ytrue_mean2[1], "[1225/1225]"),
    ]

    for ax, mean, std, true, title in panels:
        ax.plot(x_values, np.exp(mean), color="#4682B4")
        ax.fill_between(x_values, np.exp(mean - std), np.exp(mean + std), color="#ADDFFF", alpha=1)
        ax.plot(x_values, np.exp(mean - std), ls="-", lw=0.2, color="black")
        ax.plot(x_values, np.exp(mean + std), ls="-", lw=0.2, color="black")
        ax.plot(x_values, np.exp(true), color="black", lw=0.5, drawstyle="steps-mid")
        ax.set_title(title, fontsize=fontsize, loc="left")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelbottom=False)

    axs[2].plot([])
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].set_title("[3-1223]", fontsize=fontsize, loc="left")
    axs[2].text(0.5, 0.5, "[3-1223]", ha="center", va="center", fontsize=12, alpha=0.6)
    for spine in axs[2].spines.values():
        spine.set_linestyle("--")
    axs[2].grid(False)
    axs[2].tick_params(labelbottom=False)

    axs[4].xaxis.set_major_formatter(FuncFormatter(millions))
    axs[4].tick_params(labelbottom=True)
    axs[4].set_xlabel("Sequence [bp]", fontsize=fontsize)

    fig.text(0.04, 0.5, "Time [generations]", va="center", rotation="vertical", fontsize=fontsize)
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    out = os.path.join(args.output_dir, "figure1.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
