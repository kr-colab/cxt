"""
Figure 2: True vs predicted coalescence times for cxt, Singer, and SMC++
across constant and sawtooth demographic scenarios.
"""

import argparse
import json
import os
import pickle
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
from figures.utils import analyze_ts_with_smcpp, plot_tmrca_scatter


NUM_SAMPLES = 50
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


def discretize(sequence):
    indices = np.searchsorted(TIMES, np.log(sequence), side="right") - 1
    indices = np.clip(indices, 0, len(TIMES) - 1)
    return np.exp(TIMES[indices])


def run_cxt(ts, model, pivot_pairs, devices, B, cache_dir, cache_name, mu=1.29e-8):
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["yhats"], data["ytrues"]

    yhat_tmrca, _ = translate(
        input_data=ts, data_type="ts",
        model=model, pivot_pairs=pivot_pairs,
        blocks=[(0, SEQ_LEN)], devices=devices,
        B_per_device=B, B=B,
        build_workers=8, mutation_rate=mu,
    )
    yhats, ytrues = build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, max_workers=24)
    yhats, ytrues = np.array(yhats), np.array(ytrues)
    np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)
    return yhats, ytrues


def run_smcpp(ts, pairs, mu, cache_dir, cache_name, sif_path=None):
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = [
        analyze_ts_with_smcpp(ts, pair=p, mu=mu, tmp_dir="tmp", sif_path=sif_path)
        for p in pairs
    ]
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results


def smcpp_to_scatter(results, ts, pairs):
    bins = np.arange(0, SEQ_LEN + BIN_BP, BIN_BP)
    x_grid = 0.5 * (bins[:-1] + bins[1:])
    smcpp_all, true_all = [], []

    for i, (a, b) in enumerate(pairs):
        r = results[i]
        hs, gamma, N0 = r["hidden_states"], r["gamma"], float(r["N0"])
        sites = r["site_midpoints"]
        smcpp_tmrca = np.log(hs @ gamma * 2 * N0)
        windowed = np.array([
            np.mean(smcpp_tmrca[(sites >= x0) & (sites < x1)])
            for x0, x1 in zip(bins[:-1], bins[1:])
        ])
        smcpp_all.append(windowed)
        true_tmrca = interpolate_tmrcas(ts, BIN_BP, SEQ_LEN, a, b)
        true_all.append(np.log(true_tmrca))

    return np.exp(np.array(smcpp_all).flatten()), np.exp(np.array(true_all).flatten())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig2")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sif-path", default=None)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pivot_pairs = [(i, j) for i in range(NUM_SAMPLES) for j in range(i + 1, NUM_SAMPLES)]
    smcpp_pairs = [(i, i + 1) for i in range(0, 25)]
    mu = 1.29e-8

    # --- Constant demography ---
    ts_const = simulate_parameterized_tree_sequence(seed=103370001)

    model_narrow = setup_cxt_model(model_type="narrow")
    yhats, ytrues = run_cxt(ts_const, model_narrow, pivot_pairs, args.devices,
                            args.batch_size, args.cache_dir, "constant_cxt.npz", mu)
    ytrues_d = discretize(ytrues)
    plot_tmrca_scatter(yhats.mean(0), ytrues_d,
                       os.path.join(args.output_dir, "cxt_constant.png"),
                       tool=r"$\mathbf{cxt}$-narrow: Constant Ne")

    # --- Sawtooth demography ---
    simulate_sawtooth = partial(
        simulate_parameterized_tree_sequence,
        demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3),
    )
    ts_saw = simulate_sawtooth(seed=103370001)

    yhats_saw, ytrues_saw = run_cxt(ts_saw, model_narrow, pivot_pairs, args.devices,
                                    args.batch_size, args.cache_dir, "sawtooth_narrow_cxt.npz", mu)
    ytrues_saw_d = discretize(ytrues_saw)
    plot_tmrca_scatter(yhats_saw.mean(0), ytrues_saw_d,
                       os.path.join(args.output_dir, "cxt_sawtooth.png"),
                       tool=r"$\mathbf{cxt}$-narrow: Sawtooth Ne")

    model_broad = setup_cxt_model(model_type="broad")
    yhats_broad, ytrues_broad = run_cxt(ts_saw, model_broad, pivot_pairs, args.devices,
                                        args.batch_size, args.cache_dir, "sawtooth_broad_cxt.npz", mu)
    ytrues_broad_d = discretize(ytrues_broad)
    plot_tmrca_scatter(yhats_broad.mean(0), ytrues_broad_d,
                       os.path.join(args.output_dir, "cxt_sawtooth_broad.png"),
                       tool=r"$\mathbf{cxt}$-broad: Sawtooth Ne")

    # --- SMC++ ---
    results_const = run_smcpp(ts_const, smcpp_pairs, mu, args.cache_dir,
                              "smcpp_constant.pkl", sif_path=args.sif_path)
    yh_smcpp, yt_smcpp = smcpp_to_scatter(results_const, ts_const, smcpp_pairs)
    plot_tmrca_scatter(yh_smcpp, yt_smcpp,
                       os.path.join(args.output_dir, "smcpp_constant.png"),
                       tool=r"$\mathbf{smc++}$: Constant Ne")

    results_saw = run_smcpp(ts_saw, smcpp_pairs, mu, args.cache_dir,
                            "smcpp_sawtooth.pkl", sif_path=args.sif_path)
    yh_smcpp_saw, yt_smcpp_saw = smcpp_to_scatter(results_saw, ts_saw, smcpp_pairs)
    plot_tmrca_scatter(yh_smcpp_saw, yt_smcpp_saw,
                       os.path.join(args.output_dir, "smcpp_sawtooth.png"),
                       tool=r"$\mathbf{smc++}$: Sawtooth Ne")

    print("Figure 2 panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
