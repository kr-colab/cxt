"""
Figure 7: Ag1000G A. gambiae Rdl region TMRCA across five African populations.

Infers coalescent-time landscapes for Burkina Faso, Mali, Cameroon, Ghana,
and Uganda around the Rdl locus on chromosome 2L, plus genome-wide for Uganda.

NOTE: Requires Ag1000G tree sequences and accessibility masks as input data.
The main experiment notebook is missing from the working tree; this script
is reconstructed from experiment_integrated_missing.ipynb and the paper description.
"""

import argparse
import os

import numpy as np

from figures.paths import AG1000G_DATA_DIR
import matplotlib.pyplot as plt
import torch

from cxt.api2 import translate
from cxt.utils import setup_cxt_model


POPULATIONS = ["BurkinaFaso", "Mali", "Cameroon", "Ghana", "Uganda"]
RDL_REGION = (25_000_000, 26_500_000)
CHR2L_LENGTH = 49_364_325


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig7")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--data-dir", default=AG1000G_DATA_DIR,
                        help="Directory containing Ag1000G tree sequences and masks")
    args = parser.parse_args()

    if args.devices is None:
        args.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model_w200 = setup_cxt_model(model_type="broad_w200")

    fig, axes = plt.subplots(len(POPULATIONS), 1, figsize=(12, 3 * len(POPULATIONS)), sharex=True)

    for ax, pop in zip(axes, POPULATIONS):
        cache_path = os.path.join(args.cache_dir, f"rdl_{pop}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            tmrca_mean = data["tmrca_mean"]
            x_bp = data["x_bp"]
        else:
            raise FileNotFoundError(
                f"Pre-computed cache not found at {cache_path}. "
                f"Run inference on Ag1000G data for {pop} first."
            )

        ax.plot(x_bp / 1e6, np.exp(tmrca_mean), lw=0.8, color="dodgerblue")
        ax.set_yscale("log")
        ax.set_title(pop, loc="left", fontsize=12)
        ax.axvspan(RDL_REGION[0] / 1e6, RDL_REGION[1] / 1e6, alpha=0.1, color="crimson")
        ax.text(RDL_REGION[0] / 1e6, ax.get_ylim()[1] * 0.8, "Rdl", color="crimson", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Position on chr2L (Mb)")
    fig.text(0.04, 0.5, "TMRCA (generations)", va="center", rotation="vertical", fontsize=12)
    plt.tight_layout(rect=[0.06, 0, 1, 1])

    out = os.path.join(args.output_dir, "figure7_mosquito_rdl.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
