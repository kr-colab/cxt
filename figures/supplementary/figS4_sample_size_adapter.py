"""
Figure S4: Sample-size adapter evaluation (n=5 diploids from cxt-broad trained at n=25).

Shows TMRCA scatter plots for constant and sawtooth demographies using the
adapter-fine-tuned model at reduced sample size.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import (
    TIMES,
    create_sawtooth_demogaphy_object,
    setup_cxt_model,
    simulate_parameterized_tree_sequence,
)
from figures.utils import plot_tmrca_scatter


NUM_SAMPLES = 10
BIN_BP = 2000
SEQ_LEN = 1_000_000


def build_yhats_ytrues(ts, pivot_ids, yhat_tmrca, max_workers=None):
    yhat_means = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ytrues = list(ex.map(
            lambda args: interpolate_tmrcas(args[0], BIN_BP, SEQ_LEN, args[1], args[2]),
            [(ts, a, b) for a, b in pivot_ids],
        ))
    return yhat_means, ytrues


def discretize(vals):
    indices = np.searchsorted(TIMES, np.log(vals), side="right") - 1
    return np.exp(TIMES[np.clip(indices, 0, len(TIMES) - 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/supplementary/cache/figS4")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pivot_pairs = [(i, j) for i in range(NUM_SAMPLES) for j in range(i + 1, NUM_SAMPLES)]
    blocks = [(0, SEQ_LEN)]
    mu = 1.29e-8

    configs = [
        ("constant", None, r"$\mathbf{cxt}$-adapter: Constant Ne"),
        ("sawtooth", create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3),
         r"$\mathbf{cxt}$-adapter: Sawtooth Ne"),
    ]

    for model_type in ["broad", "broad+adapter"]:
        model = setup_cxt_model(model_type=model_type)
        for name, demography, title in configs:
            sim_fn = partial(simulate_parameterized_tree_sequence, demography=demography) if demography else simulate_parameterized_tree_sequence
            ts = sim_fn(seed=103370001, num_samples=5)

            cache_name = f"{name}_{model_type.replace('+', '_')}.npz"
            cache_path = os.path.join(args.cache_dir, cache_name)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                yhats, ytrues = data["yhats"], data["ytrues"]
            else:
                yhat_tmrca, _ = translate(
                    input_data=ts, data_type="ts",
                    model=model, pivot_pairs=pivot_pairs,
                    blocks=blocks, devices=args.devices,
                    B_per_device=args.batch_size, B=args.batch_size,
                    build_workers=8, mutation_rate=mu,
                )
                yhats, ytrues = build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, max_workers=24)
                yhats, ytrues = np.array(yhats), np.array(ytrues)
                np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)

            ytrues_d = discretize(ytrues)
            out_name = f"cxt_{model_type.replace('+', '_')}_{name}.png"
            plot_tmrca_scatter(
                yhats.mean(0) if yhats.ndim > 1 else yhats, ytrues_d,
                os.path.join(args.output_dir, out_name), tool=title,
            )

    print("Figure S4 panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
