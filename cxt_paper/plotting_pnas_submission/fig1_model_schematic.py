"""
Figure 1 (PNAS): Two-column figure.

Left: conceptual schematic (supplied as PNG).
Right: five stacked TMRCA trajectory plots showing all 1225 pairwise
inferences from a constant-size simulation.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT, TIMES,
)

SCHEMATIC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "cxt_paper", "figures", "schematic.png"
)


def discretize(sequence, population_time):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    return np.clip(indices, 0, len(population_time) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--schematic", default=SCHEMATIC_PATH)
    args = parser.parse_args()

    apply_pnas_style()

    cache_path = resolve_cache("main/cache/fig1/constant_cxt.npz")
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        return

    data = np.load(cache_path)
    yhats, ytrues = data["yhats"], data["ytrues"]

    ytrues = np.exp(TIMES[discretize(np.log(ytrues), TIMES)])
    ytrues = np.tile(ytrues, (15, 1, 1))
    yhats = np.log(yhats[0])
    ytrues = np.log(ytrues)

    x_values = np.arange(0, 1_000_000, 2000)

    def calc_mean_std(d):
        return np.mean(d, axis=0), np.std(d, axis=0)

    yhat_mean1, yhat_std1 = calc_mean_std(yhats[:, :2, :])
    ytrue_mean1 = np.mean(ytrues[:, :2, :], axis=0)
    yhat_mean2, yhat_std2 = calc_mean_std(yhats[:, -2:, :])
    ytrue_mean2 = np.mean(ytrues[:, -2:, :], axis=0)

    fig_height = 2.85
    fig = plt.figure(figsize=(DOUBLE_COL, fig_height))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.14)

    # -- Left: schematic image, stretched to match right panel height --
    ax_left = fig.add_subplot(gs[0, 0])
    if os.path.exists(args.schematic):
        img = mpimg.imread(args.schematic)
        ax_left.imshow(img, aspect="auto")
    ax_left.axis("off")

    # -- Right: 5 stacked TMRCA panels --
    gs_right = gs[0, 1].subgridspec(5, 1, height_ratios=[1, 1, 0.4, 1, 1],
                                     hspace=0.42)

    axs = [fig.add_subplot(gs_right[i]) for i in range(5)]

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
        ax.fill_between(x_values, np.exp(mean - std), np.exp(mean + std),
                        color="#ADDFFF", alpha=1)
        ax.plot(x_values, np.exp(mean - std), ls="-", lw=0.2, color="black")
        ax.plot(x_values, np.exp(mean + std), ls="-", lw=0.2, color="black")
        ax.plot(x_values, np.exp(true), color="black", lw=0.5,
                drawstyle="steps-mid")
        ax.set_title(title, loc="left", fontsize=5, pad=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelbottom=False, labelsize=5)

    axs[2].plot([])
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].text(0.5, 0.5, "[3-1223]", ha="center", va="center",
                fontsize=6, alpha=0.6)
    for spine in axs[2].spines.values():
        spine.set_linestyle("--")
    axs[2].grid(False)
    axs[2].tick_params(labelbottom=False, labelleft=False, labelsize=5)

    axs[2].set_ylabel("Time [gen.]", fontsize=6, labelpad=16)

    axs[4].xaxis.set_major_formatter(FuncFormatter(millions))
    axs[4].tick_params(labelbottom=True, labelsize=5)
    axs[4].set_xlabel("Sequence [bp]", fontsize=6)

    savefig(fig, "figure1", output_dir=args.output_dir)
    print("Done: figure1")


if __name__ == "__main__":
    main()
