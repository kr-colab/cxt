"""
Figure S9: Mosquito Rdl region comparison panel (cxt vs Singer vs SMC++).

Loads pre-computed TMRCA caches for each method and population, then
assembles a multi-panel comparison figure.

NOTE: Requires pre-computed results from:
  - cxt inference (fig7 cache)
  - Singer+Polegon (experiment_singer.ipynb cache)
  - SMC++ (experiment_smc++.ipynb cache)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


POPULATIONS = ["BurkinaFaso", "Mali", "Cameroon", "Ghana", "Uganda"]
METHODS = ["cxt", "Singer+Polegon", "SMC++"]
METHOD_COLORS = {"cxt": "dodgerblue", "Singer+Polegon": "darkblue", "SMC++": "lightseagreen"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/supplementary/cache/figS9")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(POPULATIONS), len(METHODS),
                             figsize=(5 * len(METHODS), 3 * len(POPULATIONS)),
                             sharex=True, sharey="row")

    for i, pop in enumerate(POPULATIONS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            cache_path = os.path.join(args.cache_dir, f"{pop}_{method.replace('+', '_').replace(' ', '_')}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                x = data.get("x_bp", np.arange(data["tmrca"].shape[-1]) * 200)
                tmrca = data["tmrca"]
                ax.plot(x / 1e6, np.exp(tmrca.mean(0)) if tmrca.ndim > 1 else np.exp(tmrca),
                        lw=0.8, color=METHOD_COLORS[method])
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(method, fontsize=12)
            if j == 0:
                ax.set_ylabel(pop, fontsize=10)
            if i == len(POPULATIONS) - 1:
                ax.set_xlabel("Position on chr2L (Mb)")

    plt.tight_layout()
    out = os.path.join(args.output_dir, "figS9_mosquito_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
