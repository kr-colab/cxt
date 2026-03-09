"""
Figure S5: Effect of window resolution on TMRCA prediction accuracy.

Compares cxt-broad at w=2000 bp, w=200 bp, and w=20 bp, plus the
residual model at w=2000 bp. Uses a constant population size scenario
with high Ne (1.5M) and low mutation rate (1e-9).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import msprime
import numpy as np

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES, setup_cxt_model
from figures.utils import plot_tmrca_scatter


MUTATION_RATE = 1e-9


def discretize(vals):
    indices = np.searchsorted(TIMES, np.log(vals), side="right") - 1
    return np.exp(TIMES[np.clip(indices, 0, len(TIMES) - 1)])


def run_window_size(ts, model, window_bp, seq_len, pivot_pairs, devices, B, cache_dir, cache_name, mu, residual=False):
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["yhats"], data["ytrues"]

    blocks = [(0, seq_len)]
    kwargs = dict(
        input_data=ts, data_type="ts",
        model=model, pivot_pairs=pivot_pairs,
        blocks=blocks, devices=devices,
        B_per_device=B, B=B,
        build_workers=8, mutation_rate=mu,
    )
    if residual:
        kwargs["residual_model"] = True

    yhat_tmrca, _ = translate(**kwargs)

    yhat_means = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=24) as ex:
        ytrues = list(ex.map(
            lambda args: interpolate_tmrcas(args[0], window_bp, seq_len, args[1], args[2]),
            [(ts, a, b) for a, b in pivot_pairs],
        ))
    yhats = np.array([yhat_means.mean(0)[i] for i in range(len(pivot_pairs))]).flatten()
    ytrues = np.array(ytrues).flatten()
    np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)
    return yhats, ytrues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/supplementary/cache/figS5")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ts = msprime.sim_ancestry(
        samples=25, sequence_length=1e6, recombination_rate=1e-9,
        population_size=1.5e6, random_seed=1,
    )
    ts = msprime.mutate(ts, rate=MUTATION_RATE, random_seed=1)

    model_broad = setup_cxt_model(model_type="broad")
    pivot_pairs = [(0, 1)]

    configs = [
        (2000, 1e6, "constant_w2000_cxt.npz", r"$\mathbf{cxt}$ w=2000 bp", "cxt_w2000.png", False),
        (200, 0.1e6, "constant_w200_cxt.npz", r"$\mathbf{cxt}$ w=200 bp", "cxt_w200.png", False),
        (20, 0.01e6, "constant_w20_cxt.npz", r"$\mathbf{cxt}$ w=20 bp", "cxt_w20.png", False),
    ]

    for window_bp, seq_len, cache_name, tool_label, out_name, residual in configs:
        model = model_broad
        yhats, ytrues = run_window_size(
            ts, model, window_bp, seq_len, pivot_pairs,
            args.devices, args.batch_size, args.cache_dir, cache_name,
            MUTATION_RATE, residual=residual,
        )
        ytrues_d = discretize(ytrues)
        plot_tmrca_scatter(yhats, ytrues_d, os.path.join(args.output_dir, out_name), tool=tool_label)

    # Residual model at w=2000
    model_res = setup_cxt_model(model_type="residual")
    yhats, ytrues = run_window_size(
        ts, model_res, 2000, 1e6, pivot_pairs,
        args.devices, args.batch_size, args.cache_dir, "constant_w2000_cxt_residual.npz",
        MUTATION_RATE, residual=True,
    )
    ytrues_d = discretize(ytrues)
    plot_tmrca_scatter(yhats, ytrues_d,
                       os.path.join(args.output_dir, "cxt_w2000_residual.png"),
                       tool=r"$\mathbf{cxt}$-residual w=2000 bp")

    print("Figure S5 panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
