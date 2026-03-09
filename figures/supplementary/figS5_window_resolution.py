"""
Figure S5: Effect of window resolution on TMRCA prediction accuracy.

Compares cxt-broad at w=2000 bp, w=200 bp, and w=20 bp, plus the
residual model at w=2000 bp. Uses a constant population size scenario
with high Ne (1.5M) and low mutation rate (1e-9).

Faithful reproduction of revision/figure_resolution/experiment.ipynb.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import msprime
import numpy as np

import cxt
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES
from figures.utils import plot_tmrca_scatter


MUTATION_RATE = 1e-9
NUM_SAMPLES = 50


def _interp_worker_s5(args, window_bp, seq_len):
    ts, a, b = args
    return interpolate_tmrcas(ts, window_bp, seq_len, a, b)


def discretize_ytrues(ytrues):
    indices = np.searchsorted(TIMES, np.log(ytrues), side="right") - 1
    indices = np.clip(indices, 0, len(TIMES) - 1)
    return np.exp(TIMES[indices])


def run_window_size(ts, model, window_bp, seq_len, pivot_pairs,
                    devices, B, cache_dir, cache_name, mu):
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["yhats"], data["ytrues"]

    blocks = [(0, seq_len)]
    kwargs = dict(
        pivot_pairs=pivot_pairs,
        blocks=blocks, devices=devices,
        B_per_device=B, B=B,
        build_workers=8, mutation_rate=mu,
    )

    yhat_tmrca, _ = cxt.translate(ts, model, **kwargs)

    yhat_means = np.exp(yhat_tmrca)
    _worker = partial(_interp_worker_s5,
                      window_bp=window_bp, seq_len=seq_len)
    with ProcessPoolExecutor(max_workers=24) as ex:
        ytrues = list(ex.map(
            _worker, [(ts, a, b) for a, b in pivot_pairs]))

    n_pairs = len(pivot_pairs)
    yhats = np.array(
        [yhat_means.mean(0)[i] for i in range(n_pairs)]
    ).flatten()
    ytrues = np.array(ytrues).flatten()
    np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)
    return yhats, ytrues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",
                        default="figures/output/supplementary")
    parser.add_argument("--cache-dir",
                        default="figures/output/supplementary/cache/figS5")
    parser.add_argument("--devices", nargs="+",
                        default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ts = msprime.sim_ancestry(
        samples=NUM_SAMPLES // 2,
        sequence_length=1e6,
        recombination_rate=1e-9,
        population_size=1.5e6,
        random_seed=1,
    )
    ts = msprime.mutate(ts, rate=MUTATION_RATE, random_seed=1)

    model_broad = cxt.load_model("broad", device="cpu")
    pivot_pairs = [(i, j)
                   for i in range(NUM_SAMPLES)
                   for j in range(i + 1, NUM_SAMPLES)]

    # (window_bp, seq_len, cache_name, tool_label, out_name, do_discretize)
    configs = [
        (2000, 1e6, "constant_w2000_cxt.npz",
         r"$\mathbf{cxt}$ w=2000 bp", "cxt_w2000.png", False),
        (200, 0.1e6, "constant_w200_cxt.npz",
         r"$\mathbf{cxt}$ w=200 bp", "cxt_w200.png", True),
        (20, 0.01e6, "constant_w20_cxt.npz",
         r"$\mathbf{cxt}$ w=20 bp", "cxt_w20.png", True),
    ]

    for window_bp, seq_len, cache_name, tool_label, out_name, do_disc in configs:
        yhats, ytrues = run_window_size(
            ts, model_broad, window_bp, seq_len, pivot_pairs,
            args.devices, args.batch_size,
            args.cache_dir, cache_name, MUTATION_RATE,
        )
        if do_disc:
            ytrues = discretize_ytrues(ytrues)
        plot_tmrca_scatter(
            yhats, ytrues,
            os.path.join(args.output_dir, out_name),
            tool=tool_label,
        )

    model_res = cxt.load_model("residual", device="cpu")
    yhats, ytrues = run_window_size(
        ts, model_res, 2000, 1e6, pivot_pairs,
        args.devices, args.batch_size,
        args.cache_dir, "constant_w2000_cxt_residual.npz",
        MUTATION_RATE,
    )
    ytrues = discretize_ytrues(ytrues)
    plot_tmrca_scatter(
        yhats, ytrues,
        os.path.join(args.output_dir, "cxt_w2000_residual.png"),
        tool=r"$\mathbf{cxt}$-residual w=2000 bp",
    )

    print("Figure S5 panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
