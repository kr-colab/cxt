"""
Figure S11: Interpolation and extrapolation benchmark across mutation/recombination rates.

Evaluates cxt-broad on a grid of mutation and recombination rates with
fixed Ne=20,000, computing MSE and KL divergence at each grid point.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import msprime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES, setup_cxt_model


BASE_MU = 1.29e-8
BASE_R = 1.29e-8
NE = 20_000
SEQ_LEN = 1_000_000
WINDOW_BP = 2000
GRID_SIZE = 7


def discretize(vals):
    indices = np.searchsorted(TIMES, np.log(vals), side="right") - 1
    return np.exp(TIMES[np.clip(indices, 0, len(TIMES) - 1)])


def simulate(mu, r, seed=42):
    ts = msprime.sim_ancestry(
        samples=25, sequence_length=SEQ_LEN,
        recombination_rate=r, population_size=NE, random_seed=seed,
    )
    return msprime.mutate(ts, rate=mu, random_seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/supplementary/cache/figS11")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = setup_cxt_model(model_type="broad")

    mu_factors = np.logspace(-1, 1, args.grid_size)
    r_factors = np.logspace(-1, 1, args.grid_size)

    mse_grid = np.full((args.grid_size, args.grid_size), np.nan)
    pivot_pairs = [(0, 1)]
    blocks = [(0, SEQ_LEN)]

    for i, mf in enumerate(mu_factors):
        for j, rf in enumerate(r_factors):
            mu = BASE_MU * mf
            r = BASE_R * rf
            cache_name = f"grid_{i}_{j}.npz"
            cache_path = os.path.join(args.cache_dir, cache_name)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                yhats, ytrues = data["yhats"], data["ytrues"]
            else:
                ts = simulate(mu, r, seed=42 + i * args.grid_size + j)
                yhat_tmrca, _ = translate(
                    input_data=ts, data_type="ts",
                    model=model, pivot_pairs=pivot_pairs,
                    blocks=blocks, devices=args.devices,
                    B_per_device=args.batch_size, B=args.batch_size,
                    build_workers=8, mutation_rate=mu,
                )
                yhats = np.exp(yhat_tmrca).mean(0).flatten()
                ytrues = np.array(interpolate_tmrcas(ts, WINDOW_BP, SEQ_LEN, 0, 1))
                np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)

            ytrues_d = discretize(ytrues)
            log_yhats = np.log(np.clip(yhats, 1e-12, None))
            log_ytrues = np.log(np.clip(ytrues_d, 1e-12, None))
            mask = np.isfinite(log_yhats) & np.isfinite(log_ytrues)
            mse_grid[i, j] = np.mean((log_yhats[mask] - log_ytrues[mask]) ** 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mse_grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(args.grid_size))
    ax.set_xticklabels([f"{f:.2f}" for f in r_factors], rotation=45)
    ax.set_yticks(np.arange(args.grid_size))
    ax.set_yticklabels([f"{f:.2f}" for f in mu_factors])
    ax.set_xlabel("Recombination rate factor")
    ax.set_ylabel("Mutation rate factor")
    ax.set_title("MSE (ln-space) across mutation/recombination grid", loc="left")
    plt.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()

    out = os.path.join(args.output_dir, "figS11_interpolation_grid.png")
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
