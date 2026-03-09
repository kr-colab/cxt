"""
Figure S10: Cross-coalescence rate under OutOfAfrica_2T12 two-population model.

Computes cross-population (AFR|EUR), within-AFR, and within-EUR coalescence
rates from cxt predictions on a 10 Mb simulation of the OutOfAfrica 2T12 model.
"""

import argparse
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import stdpopsim
import tskit

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import coalescence_rates, setup_cxt_model


NUM_PAIRS = 25
NUM_TIME_WINDOWS = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/supplementary/cache/figS10")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    species = stdpopsim.get_species("HomSap")
    demogr = species.get_demographic_model("OutOfAfrica_2T12")
    contig = species.get_contig("chr1", right=10e6)
    engine = stdpopsim.get_engine("msprime")

    ts_path = os.path.join(args.cache_dir, "ooa_2t12.ts")
    if os.path.exists(ts_path):
        ts = tskit.load(ts_path)
    else:
        ts = engine.simulate(
            contig=contig, samples={"AFR": 12, "EUR": 13},
            demographic_model=demogr, seed=10_000_001,
        ).trim()
        ts.dump(ts_path)

    all_pairs = list(combinations(range(NUM_PAIRS * 2), 2))
    idx_afr_eur = [i for i, (a, b) in enumerate(all_pairs) if (a < 24) != (b < 24)]
    idx_afr = [i for i, (a, b) in enumerate(all_pairs) if a < 24 and b < 24]
    idx_eur = [i for i, (a, b) in enumerate(all_pairs) if a >= 24 and b >= 24]

    pivot_pairs = list(combinations(range(50), 2))
    blocks = [(int(i), int(i + 1e6)) for i in np.linspace(0, 9e6, 10)]

    model = setup_cxt_model(model_type="broad")

    cache_path = os.path.join(args.cache_dir, "tmrca.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        tmrca, index_map = data["tmrca"], data["index_map"]
    else:
        tmrca, index_map = translate(
            input_data=ts, data_type="ts",
            model=model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=512, build_workers=36,
            mutation_rate=1.29e-8,
        )
        np.savez_compressed(cache_path, tmrca=tmrca, index_map=index_map)

    ytrues = []
    for a, b in pivot_pairs:
        ytrues.append(interpolate_tmrcas(ts, 2000, 10e6, a, b))
    ytrues = np.array(ytrues)

    time_windows = np.logspace(2, np.log10(np.quantile(ts.nodes_time, 0.95)), NUM_TIME_WINDOWS)
    time_windows[0] = 0.0
    fine_time_grid = np.logspace(2, np.floor(np.log10(ts.max_time)), 1000)

    configs = [
        ("AFR|EUR", idx_afr_eur, {"AFR": 1, "EUR": 1}),
        ("AFR|AFR", idx_afr, {"AFR": 2}),
        ("EUR|EUR", idx_eur, {"EUR": 2}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (label, indices, lineages) in zip(axes, configs):
        coalrate_ck, _ = demogr.model.debug().coalescence_rate_trajectory(
            lineages=lineages, steps=fine_time_grid,
        )

        indices_10mb = []
        for b in range(len(blocks)):
            indices_10mb += (np.array(indices) + b * 1225).tolist()

        yhat = np.exp(tmrca[:, indices_10mb])
        yhat_coalrates = np.array([
            coalescence_rates(rep.flatten(), time_windows) for rep in yhat
        ])

        ytrue_coalrate = coalescence_rates(ytrues[indices].flatten(), time_windows)

        ax.plot(fine_time_grid, coalrate_ck, "-", color="black", label="Expectation")
        ax.step(time_windows[:-1], ytrue_coalrate, where="post", color="firebrick", label="Inference limit")
        ax.step(time_windows[:-1], np.mean(yhat_coalrates, axis=0), where="post",
                color="dodgerblue", label="cxt")
        ax.fill_between(
            time_windows[:-1],
            np.mean(yhat_coalrates, 0) - 1.96 * np.std(yhat_coalrates, 0) / np.sqrt(len(yhat_coalrates)),
            np.mean(yhat_coalrates, 0) + 1.96 * np.std(yhat_coalrates, 0) / np.sqrt(len(yhat_coalrates)),
            color="dodgerblue", alpha=0.2,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(10e-7, 5e-5)
        ax.set_xlabel("Time (Generations)")
        ax.set_title(f"OutOfAfrica_2T12 (Pivots {label})", loc="left")
        ax.grid(True)

    axes[0].set_ylabel("IICR / 2")
    axes[-1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out = os.path.join(args.output_dir, "figS10_cross_coalescence.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
