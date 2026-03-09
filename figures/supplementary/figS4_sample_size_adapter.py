"""
Figure S4: Sample-size adapter evaluation (n=5 diploids from cxt-broad trained at n=25).

Shows TMRCA scatter plots for constant, sawtooth, and island demographies
using both the base broad model (n=25) and the adapter-fine-tuned model (n=5).

Faithful reproduction of revision/figure_sample_size_5/experiment.ipynb.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import msprime
import numpy as np

import cxt
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import create_sawtooth_demography, simulate_parameterized_tree_sequence
from cxt.utils import TIMES
from figures.utils import plot_tmrca_scatter


BIN_BP = 2000
SEQ_LEN = 1_000_000


def _interp_worker(args):
    ts, a, b = args
    return interpolate_tmrcas(ts, BIN_BP, SEQ_LEN, a, b)


def build_yhats_ytrues(ts, pivot_ids, yhat_tmrca, max_workers=None):
    yhat_means = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ytrues = list(ex.map(_interp_worker, [(ts, a, b) for a, b in pivot_ids]))
    return yhat_means, ytrues


def discretize(vals):
    indices = np.searchsorted(TIMES, np.log(vals), side="right") - 1
    return np.exp(TIMES[np.clip(indices, 0, len(TIMES) - 1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir",
                        default="figures/output/supplementary/cache/figS4")
    parser.add_argument("--devices", nargs="+",
                        default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    blocks = [(0, SEQ_LEN)]
    mu = 1.29e-8
    seed = 103370001

    island_dem = msprime.Demography.island_model(
        [10000, 5000, 5000], migration_rate=0.1)

    configs = [
        ("constant", None, None, 25),
        ("sawtooth", create_sawtooth_demography(Ne=20e3, magnitude=3), None, 25),
        ("island", None, island_dem, {0: 15, 1: 5, 2: 5}),
    ]

    broad_model = cxt.load_model("broad", device="cpu")
    adapter_model = cxt.load_model("broad+adapter", device="cpu")

    model_specs = [
        ("broad", broad_model, False, 50, r"$\mathbf{cxt}$-broad"),
        ("broad+adapter", adapter_model, True, 10, r"$\mathbf{cxt}$-adapter"),
    ]

    for model_type, model, is_adapter, n_haploid, tool_prefix in model_specs:
        backbone = model.backbone if is_adapter else model
        adapter = model.adapter if is_adapter else None

        pivot_pairs = [(i, j) for i in range(n_haploid)
                       for j in range(i + 1, n_haploid)]

        for demo_name, demography, island_demography, samples_arg in configs:
            title = f"{tool_prefix}: {demo_name.capitalize()} Ne"
            cache_name = f"{demo_name}_{model_type.replace('+', '_')}.npz"
            cache_path = os.path.join(args.cache_dir, cache_name)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                yhats, ytrues = data["yhats"], data["ytrues"]
            else:
                if island_demography is not None:
                    ts_full = simulate_parameterized_tree_sequence(
                        seed=seed, island_demography=island_demography,
                        samples=samples_arg)
                elif demography is not None:
                    ts_full = simulate_parameterized_tree_sequence(
                        seed=seed, demography=demography, samples=25)
                else:
                    ts_full = simulate_parameterized_tree_sequence(
                        seed=seed, samples=25)

                if is_adapter:
                    ts_input = ts_full.simplify(samples=range(n_haploid))
                else:
                    ts_input = ts_full

                yhat_tmrca, _ = cxt.translate(
                    ts_input, backbone, pivot_pairs=pivot_pairs,
                    blocks=blocks, devices=args.devices,
                    B_per_device=args.batch_size, B=args.batch_size,
                    build_workers=8, mutation_rate=mu,
                    adapter=adapter,
                )
                yhats, ytrues = build_yhats_ytrues(
                    ts_full, pivot_pairs, yhat_tmrca, max_workers=24)
                yhats, ytrues = np.array(yhats), np.array(ytrues)
                np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)

            ytrues_d = discretize(ytrues)
            out_name = f"cxt_{model_type.replace('+', '_')}_{demo_name}.png"
            plot_tmrca_scatter(
                yhats.mean(0) if yhats.ndim > 1 else yhats, ytrues_d,
                os.path.join(args.output_dir, out_name), tool=title,
            )

    print("Figure S4 panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
